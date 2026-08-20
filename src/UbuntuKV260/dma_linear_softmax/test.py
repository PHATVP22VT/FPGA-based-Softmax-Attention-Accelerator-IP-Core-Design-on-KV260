#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script attn accelerator tren KV260 - Software Tiling support.

Supports Q matrices with more rows than HW SEQ_LEN (16) by automatically
tiling Q rows. K matrix must be exactly D_HEAD(16) x D_MODEL(64).
D_MODEL must remain 64 (HW fixed parameter).

Tiling constraint:
  - Chiều M (Q rows): tiling OK, mỗi tile xử lý 16 hàng độc lập
  - Chiều N (K rows): KHÔNG tiling được (softmax cần full row)
  - Chiều D (D_MODEL): KHÔNG tiling được (partial dot product)

Su dung ctypes de dam bao 32-bit atomic MMIO access (giong Xil_Out32/Xil_In32).
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
SM_BASE_PHYS  = 0xA0040000

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

# ---------- HW Fixed Parameters ----------
HW_SEQ_LEN   = 64
HW_D_HEAD    = 64
HW_D_MODEL   = 64
HW_WORDS_K   = HW_D_HEAD * HW_D_MODEL       # 1024
HW_BYTES_K   = HW_WORDS_K * 4                # 4096
HW_WORDS_Q   = HW_SEQ_LEN * HW_D_MODEL      # 1024
HW_BYTES_Q   = HW_WORDS_Q * 4                # 4096
HW_WORDS_OUT = HW_SEQ_LEN * HW_D_HEAD        # 256
HW_BYTES_OUT = HW_WORDS_OUT * 4              # 1024

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
    """
    Extract one tile of 16 Q rows from the full Q word array.
    Zero-pads if the last tile has fewer than 16 rows.
    Returns list of HW_WORDS_Q (1024) words.
    """
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


def run_single_tile(src, dst, dma, lin, sm, k_words, q_tile_words):
    """
    Run one HW pass: Q_tile[16x64] x K[16x64]^T -> Linear -> Softmax -> Output[16x16].
    Returns list of HW_WORDS_OUT (256) output words, or None on failure.
    """
    # -- Write K + Q_tile to src BRAM (word-by-word MMIO) --
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

    # ===== PHASE 2: Softmax + DMA Q -> S2MM =====
    reg_wr(sm, 0x00, 1)                          # start softmax IP

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
        
    # Read Exact Hardware Cycle Counters
    linear_cycles = reg_rd(lin, 0x08)
    softmax_cycles = reg_rd(sm, 0x08)
    
    return output, linear_cycles, softmax_cycles


# ==========================================================================
# Main
# ==========================================================================
def main():
    # -- Load test vector files --
    for f in ("k_data.bin", "q_data.bin", "golden_softmax.bin"):
        if not os.path.exists(f):
            print(f"[ERROR] Missing file: {f}")
            sys.exit(1)

    k_raw      = open("k_data.bin", "rb").read()
    q_raw      = open("q_data.bin", "rb").read()
    golden_raw = open("golden_softmax.bin", "rb").read()

    # -- Detect matrix dimensions from file sizes --
    num_k_words      = len(k_raw) // 4
    num_q_words      = len(q_raw) // 4
    num_golden_words = len(golden_raw) // 4

    # K must be exactly HW_D_HEAD x HW_D_MODEL = 16 x 64 = 1024 words
    if num_k_words != HW_WORDS_K:
        print(f"[ERROR] k_data.bin has {num_k_words} words, "
              f"expected {HW_WORDS_K} (D_HEAD={HW_D_HEAD} x D_MODEL={HW_D_MODEL})")
        sys.exit(1)

    # Detect total SEQ_LEN (number of Q rows) from Q file size
    total_seq_len = num_q_words // HW_D_MODEL
    if num_q_words != total_seq_len * HW_D_MODEL:
        print(f"[ERROR] q_data.bin has {num_q_words} words, "
              f"not divisible by D_MODEL={HW_D_MODEL}")
        sys.exit(1)

    expected_golden = total_seq_len * HW_D_HEAD
    if num_golden_words != expected_golden:
        print(f"[ERROR] golden_softmax.bin has {num_golden_words} words, "
              f"expected {expected_golden} (SEQ_LEN={total_seq_len} x D_HEAD={HW_D_HEAD})")
        sys.exit(1)

    # Calculate number of tiles
    num_tiles = (total_seq_len + HW_SEQ_LEN - 1) // HW_SEQ_LEN

    print("=" * 65)
    print("  ATTENTION ACCELERATOR TEST — KV260 Software Tiling")
    print("=" * 65)
    print(f"  Q  : [{total_seq_len} x {HW_D_MODEL}]  ({num_q_words} words, {len(q_raw)} bytes)")
    print(f"  K  : [{HW_D_HEAD} x {HW_D_MODEL}]  ({num_k_words} words, {len(k_raw)} bytes)")
    print(f"  Out: [{total_seq_len} x {HW_D_HEAD}]  ({num_golden_words} words, {len(golden_raw)} bytes)")
    print(f"  HW tile size: Q[{HW_SEQ_LEN}x{HW_D_MODEL}] x K[{HW_D_HEAD}x{HW_D_MODEL}]")
    print(f"  Number of tiles: {num_tiles}")
    if total_seq_len % HW_SEQ_LEN != 0:
        last_tile_rows = total_seq_len % HW_SEQ_LEN
        print(f"  Last tile: {last_tile_rows} real rows + "
              f"{HW_SEQ_LEN - last_tile_rows} zero-padded rows")
    print("=" * 65)

    # -- Unpack binary data --
    k_words      = struct.unpack(f'<{num_k_words}I', k_raw)
    q_words      = struct.unpack(f'<{num_q_words}I', q_raw)
    golden_words = struct.unpack(f'<{num_golden_words}I', golden_raw)

    # -- Map all peripherals via /dev/mem --
    print("\nMapping /dev/mem for SRC/DST/DMA/LINEAR/SOFTMAX...")
    _, src = phys_map(SRC_BASE_PHYS)
    _, dst = phys_map(DST_BASE_PHYS)
    _, dma = phys_map(DMA_BASE_PHYS)
    _, lin = phys_map(LIN_BASE_PHYS)
    _, sm  = phys_map(SM_BASE_PHYS)

    # -- Verify BRAM access with first K tile --
    print("Writing K to BRAM and verifying readback...")
    for i in range(HW_WORDS_K):
        reg_wr(src, i * 4, k_words[i])
    ok = True
    for i in range(8):
        got = reg_rd(src, i * 4)
        exp = k_words[i]
        if got != exp:
            print(f"  [{i}] wrote=0x{exp:08x} read=0x{got:08x} MISMATCH")
            ok = False
    if not ok:
        print("[FAIL] BRAM readback failed — aborting")
        sys.exit(1)
    print("[OK] BRAM readback verified")

    # -- Run all tiles --
    all_hw_output = []
    total_hw_ns   = 0
    total_exact_linear_ns = 0
    total_exact_softmax_ns = 0

    for tile_idx in range(num_tiles):
        row_start   = tile_idx * HW_SEQ_LEN
        row_end     = min(row_start + HW_SEQ_LEN, total_seq_len)
        actual_rows = row_end - row_start

        print(f"\n{'='*50}")
        print(f"  TILE {tile_idx + 1}/{num_tiles}: "
              f"Q rows [{row_start} : {row_end}]  ({actual_rows} rows"
              f"{', +' + str(HW_SEQ_LEN - actual_rows) + ' padded' if actual_rows < HW_SEQ_LEN else ''})")
        print(f"{'='*50}")

        # Extract Q tile with zero-padding
        q_tile = extract_q_tile(q_words, total_seq_len, tile_idx)

        # Run HW
        t_start = time.time_ns()
        res  = run_single_tile(src, dst, dma, lin, sm, k_words, q_tile)
        t_end   = time.time_ns()

        if res is None:
            print(f"[FAIL] Tile {tile_idx + 1} HW execution failed!")
            sys.exit(1)
            
        output, l_cycles, s_cycles = res

        elapsed_ns = t_end - t_start
        total_hw_ns += elapsed_ns
        
        # Assume 150MHz clock = 1000 / 150 ns (~6.667ns) per cycle
        exact_lin_ns = int(l_cycles * (1000 / 150))
        exact_sm_ns  = int(s_cycles * (1000 / 150))
        total_exact_linear_ns += exact_lin_ns
        total_exact_softmax_ns += exact_sm_ns
        
        print(f"  Tile python time : {elapsed_ns:,} ns  ({elapsed_ns / 1000:.1f} us)")
        print(f"  Tile EXACT LINEAR: {l_cycles:,} cycles = {exact_lin_ns:,} ns")
        print(f"  Tile EXACT S.MAX : {s_cycles:,} cycles = {exact_sm_ns:,} ns")
        print(f"  LINEAR status=0x{reg_rd(lin, 0x04):08x}  "
              f"SOFTMAX status=0x{reg_rd(sm, 0x04):08x}")

        # Keep only valid rows (discard padded rows' output)
        for r in range(actual_rows):
            for c in range(HW_D_HEAD):
                all_hw_output.append(output[r * HW_D_HEAD + c])

    # -- Compare output with golden --
    print(f"\n{'='*75}")
    print(f"  RESULTS — Software Timer Total: {total_hw_ns:,} ns  ({total_hw_ns / 1000:.1f} us)")
    print(f"  RESULTS — EXACT HARDWARE Time : {(total_exact_linear_ns + total_exact_softmax_ns):,} ns  ({(total_exact_linear_ns + total_exact_softmax_ns) / 1000:.1f} us)")
    print(f"{'='*75}")

    assert len(all_hw_output) == num_golden_words, \
        f"Output count mismatch: got {len(all_hw_output)}, expected {num_golden_words}"

    fail_cnt = 0
    with open("result.txt", "w") as f:
        f.write(f"{'Index':<7} | {'Golden Data':<15} | {'RTL Data':<15} | "
                f"{'Mismatch':<15} | {'Status'}\n")
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
                    print(f"  FAIL [{i}] row={row} col={col}  "
                          f"exp=0x{e:08x}  got=0x{g:08x}")

            f.write(f"{i:<7} | 0x{e:08x}      | 0x{g:08x}      | "
                    f"{mismatch:<15} | {status}\n")

    print()
    if fail_cnt == 0:
        print(f"  *** PASS: all {num_golden_words} values match golden softmax ***")
    else:
        print(f"  *** FAIL: {fail_cnt}/{num_golden_words} values mismatch ***")

    print(f"\n  Results saved to: result.txt")
    print(f"  Total Python time: {total_hw_ns:,} ns  ({total_hw_ns / 1000:.1f} us)")
    print(f"  Total Exact HW time: {(total_exact_linear_ns + total_exact_softmax_ns):,} ns  ({(total_exact_linear_ns + total_exact_softmax_ns) / 1000:.1f} us)")
    if num_tiles > 1:
        avg_ns = total_hw_ns // num_tiles
        print(f"  Avg per tile:     {avg_ns:,} ns  ({avg_ns / 1000:.1f} us)")

    sys.exit(0 if fail_cnt == 0 else 1)


if __name__ == "__main__":
    main()
