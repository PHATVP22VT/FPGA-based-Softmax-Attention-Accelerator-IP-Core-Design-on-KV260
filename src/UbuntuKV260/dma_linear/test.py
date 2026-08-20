#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script Linear/MatMul accelerator tren KV260 (Khong co Softmax).
Kich thuoc: 64x64 nhan 64x64.
"""

import os, sys, struct, time, ctypes, ctypes.util

# ---------- devmem-style 32-bit MMIO via /dev/mem ----------
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype  = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                      ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

MAP_SIZE  = 0x10000
PROT_RW   = 0x3   # PROT_READ | PROT_WRITE
MAP_SHARED = 0x01

def phys_map(phys_addr, size=MAP_SIZE):
    """Map a physical address via /dev/mem, return (fd, virt_base)."""
    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    virt = libc.mmap(None, size, PROT_RW, MAP_SHARED, fd, phys_addr)
    if virt == ctypes.c_void_p(-1).value:
        raise OSError(f"mmap failed for 0x{phys_addr:08x}")
    return fd, virt

def reg_wr(base, offset, val):
    """32-bit MMIO write."""
    ptr = ctypes.cast(base + offset, ctypes.POINTER(ctypes.c_uint32))
    ptr[0] = val

def reg_rd(base, offset):
    """32-bit MMIO read."""
    ptr = ctypes.cast(base + offset, ctypes.POINTER(ctypes.c_uint32))
    return ptr[0]

# ---------- Addresses from Vivado block design ----------
SRC_BASE_PHYS = 0xA0000000
DST_BASE_PHYS = 0xA0010000
DMA_BASE_PHYS = 0xA0020000
LIN_BASE_PHYS = 0xA0030000
# SM_BASE_PHYS da bi xoa

# ---------- DMA register offsets ----------
MM2S_DMACR  = 0x00;  MM2S_DMASR  = 0x04
MM2S_SA     = 0x18;  MM2S_LENGTH = 0x28
S2MM_DMACR  = 0x30;  S2MM_DMASR  = 0x34
S2MM_DA     = 0x48;  S2MM_LENGTH = 0x58

DMA_CR_RS      = 1 << 0
DMA_CR_RESET   = 1 << 2
DMA_SR_HALTED  = 1 << 0
DMA_SR_IDLE    = 1 << 1
DMA_SR_IOC_IRQ = 1 << 12
DMA_SR_ERR_IRQ = 1 << 14
DMA_CR_IOC_EN  = 1 << 12

# ---------- HW Fixed Parameters (64x64) ----------
HW_SEQ_LEN   = 64
HW_D_HEAD    = 64
HW_D_MODEL   = 64
HW_WORDS_K   = HW_D_HEAD * HW_D_MODEL       # 4096
HW_BYTES_K   = HW_WORDS_K * 4               # 16384
HW_WORDS_Q   = HW_SEQ_LEN * HW_D_MODEL      # 4096
HW_BYTES_Q   = HW_WORDS_Q * 4               # 16384
HW_WORDS_OUT = HW_SEQ_LEN * HW_D_HEAD       # 4096
HW_BYTES_OUT = HW_WORDS_OUT * 4             # 16384

K_BASE_PHYS  = SRC_BASE_PHYS
Q_BASE_PHYS  = SRC_BASE_PHYS + HW_BYTES_K


# ==========================================================================
# Helper functions
# ==========================================================================
def wait_done(dma, sr_off, name, timeout=5_000_000):
    for _ in range(timeout):
        sr = reg_rd(dma, sr_off)
        if sr & DMA_SR_ERR_IRQ:
            print(f"[FAIL] {name} DMA error, DMASR=0x{sr:08x}")
            return False
        if sr & DMA_SR_IOC_IRQ:
            return True
    print(f"[FAIL] {name} DMA timeout, DMASR=0x{reg_rd(dma, sr_off):08x}")
    return False


def dma_reset(dma):
    """Full DMA reset. Returns True on success."""
    reg_wr(dma, MM2S_DMACR, DMA_CR_RESET)
    reg_wr(dma, S2MM_DMACR, DMA_CR_RESET)
    for _ in range(500_000):
        if (reg_rd(dma, MM2S_DMACR) & DMA_CR_RESET) == 0 and \
           (reg_rd(dma, S2MM_DMACR) & DMA_CR_RESET) == 0:
            return True
    return False


def extract_q_tile(q_words, total_seq_len, tile_idx):
    """Extract one tile of 64 Q rows."""
    row_start = tile_idx * HW_SEQ_LEN
    tile = []
    for r in range(HW_SEQ_LEN):
        global_row = row_start + r
        for c in range(HW_D_MODEL):
            if global_row < total_seq_len:
                tile.append(q_words[global_row * HW_D_MODEL + c])
            else:
                tile.append(0)  # zero-pad
    return tile


def run_single_tile(src, dst, dma, lin, k_words, q_tile_words):
    """Run one HW pass: Q_tile[64x64] x K[64x64]^T -> Linear -> Output[64x64]."""
    # -- Write K + Q_tile to src BRAM --
    for i in range(HW_WORDS_K):
        reg_wr(src, i * 4, k_words[i])
    for i in range(HW_WORDS_Q):
        reg_wr(src, HW_BYTES_K + i * 4, q_tile_words[i])

    # -- DMA reset --
    if not dma_reset(dma):
        print("[FAIL] DMA reset timeout")
        return None

    # ===== PHASE 1: DMA K -> Linear =====
    reg_wr(dma, MM2S_DMACR, DMA_CR_RS | DMA_CR_IOC_EN)
    reg_wr(dma, MM2S_SA, K_BASE_PHYS)
    reg_wr(lin, 0x00, 1)                        # start linear IP
    reg_wr(dma, MM2S_LENGTH, HW_BYTES_K)        # triggers K transfer

    if not wait_done(dma, MM2S_DMASR, "MM2S-K"):
        return None
    reg_wr(dma, MM2S_DMASR, DMA_SR_IOC_IRQ)    # clear IOC (W1C)

    # ===== PHASE 2: DMA Q -> Linear -> S2MM =====
    # Arm S2MM
    reg_wr(dma, S2MM_DMACR, DMA_CR_RS | DMA_CR_IOC_EN)
    reg_wr(dma, S2MM_DA, DST_BASE_PHYS)
    reg_wr(dma, S2MM_LENGTH, HW_BYTES_OUT)

    # Restart MM2S for Q
    reg_wr(dma, MM2S_DMACR, DMA_CR_RS | DMA_CR_IOC_EN)
    reg_wr(dma, MM2S_SA, Q_BASE_PHYS)
    reg_wr(dma, MM2S_LENGTH, HW_BYTES_Q)        # triggers Q transfer

    if not wait_done(dma, MM2S_DMASR, "MM2S-Q"):
        return None
    if not wait_done(dma, S2MM_DMASR, "S2MM"):
        return None

    # -- Read output from dst BRAM --
    output = []
    for i in range(HW_WORDS_OUT):
        output.append(reg_rd(dst, i * 4))
    return output


# ==========================================================================
# Main
# ==========================================================================
def main():
    # -- Load test vector files (Su dung golden_score thay vi golden_softmax) --
    for f in ("k_data.bin", "q_data.bin", "golden_score.bin"):
        if not os.path.exists(f):
            print(f"[ERROR] Missing file: {f}")
            sys.exit(1)

    k_raw      = open("k_data.bin", "rb").read()
    q_raw      = open("q_data.bin", "rb").read()
    golden_raw = open("golden_score.bin", "rb").read()

    # -- Detect matrix dimensions --
    num_k_words      = len(k_raw) // 4
    num_q_words      = len(q_raw) // 4
    num_golden_words = len(golden_raw) // 4

    if num_k_words != HW_WORDS_K:
        print(f"[ERROR] k_data.bin has {num_k_words} words, expected {HW_WORDS_K}")
        sys.exit(1)

    total_seq_len = num_q_words // HW_D_MODEL
    if num_q_words != total_seq_len * HW_D_MODEL:
        print(f"[ERROR] q_data.bin size is invalid")
        sys.exit(1)

    expected_golden = total_seq_len * HW_D_HEAD
    if num_golden_words != expected_golden:
        print(f"[ERROR] golden_score.bin has {num_golden_words} words, expected {expected_golden}")
        sys.exit(1)

    num_tiles = (total_seq_len + HW_SEQ_LEN - 1) // HW_SEQ_LEN

    print("=" * 65)
    print("  IP LINEAR 64x64 BARE-METAL TEST ON UBUNTU")
    print("=" * 65)
    print(f"  HW tile size: Q[{HW_SEQ_LEN}x{HW_D_MODEL}] x K[{HW_D_HEAD}x{HW_D_MODEL}]")
    print("=" * 65)

    k_words      = struct.unpack(f'<{num_k_words}I', k_raw)
    q_words      = struct.unpack(f'<{num_q_words}I', q_raw)
    golden_words = struct.unpack(f'<{num_golden_words}I', golden_raw)

    print("\nMapping /dev/mem for SRC/DST/DMA/LINEAR...")
    _, src = phys_map(SRC_BASE_PHYS)
    _, dst = phys_map(DST_BASE_PHYS)
    _, dma = phys_map(DMA_BASE_PHYS)
    _, lin = phys_map(LIN_BASE_PHYS)

    all_hw_output = []
    total_hw_ns   = 0

    for tile_idx in range(num_tiles):
        q_tile = extract_q_tile(q_words, total_seq_len, tile_idx)
        
        t_start = time.time_ns()
        output  = run_single_tile(src, dst, dma, lin, k_words, q_tile)
        t_end   = time.time_ns()

        if output is None:
            print(f"[FAIL] Tile HW execution failed!")
            sys.exit(1)

        elapsed_ns = t_end - t_start
        total_hw_ns += elapsed_ns
        
        row_start   = tile_idx * HW_SEQ_LEN
        row_end     = min(row_start + HW_SEQ_LEN, total_seq_len)
        actual_rows = row_end - row_start
        
        for r in range(actual_rows):
            for c in range(HW_D_HEAD):
                all_hw_output.append(output[r * HW_D_HEAD + c])

    print(f"\n{'='*65}")
    print(f"  RESULTS — Total HW time: {total_hw_ns:,} ns  ({total_hw_ns / 1000:.1f} us)")
    print(f"{'='*65}")

    fail_cnt = 0
    with open("result_linear.txt", "w") as f:
        f.write(f"{'Index':<7} | {'Golden Score':<15} | {'RTL Data':<15} | {'Mismatch':<15} | {'Status'}\n")
        f.write("-" * 75 + "\n")

        for i in range(num_golden_words):
            g = all_hw_output[i]
            e = golden_words[i]
            status   = "PASS" if g == e else "FAIL"
            mismatch = "0" if g == e else f"0x{(g ^ e):08x}"

            if g != e:
                fail_cnt += 1
                if fail_cnt <= 20:
                    row = i // HW_D_HEAD
                    col = i % HW_D_HEAD
                    print(f"  FAIL [{i}] row={row} col={col} exp=0x{e:08x} got=0x{g:08x}")

            f.write(f"{i:<7} | 0x{e:08x}      | 0x{g:08x}      | {mismatch:<15} | {status}\n")

    print()
    if fail_cnt == 0:
        print(f"  *** PASS: all {num_golden_words} values match golden score ***")
    else:
        print(f"  *** FAIL: {fail_cnt}/{num_golden_words} values mismatch ***")

    print(f"\n  Results saved to: result_linear.txt")
    sys.exit(0 if fail_cnt == 0 else 1)

if __name__ == "__main__":
    main()
