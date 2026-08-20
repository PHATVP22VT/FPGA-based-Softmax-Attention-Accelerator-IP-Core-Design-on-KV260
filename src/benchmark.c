#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

//==============================================================================
// PARAMETERS (Matching golden_model.py)
//==============================================================================
#define SEQ_LEN       64
#define D_HEAD        64
#define D_MODEL       64
#define DATA_WIDTH    16
#define FRAC_BITS     8
#define SQRT_SHIFT    3

#define ROM_DEPTH     2048
#define RECIP_ADDR_W  12
#define RECIP_OUT_W   19
#define RECIP_ROM_DEPTH (1 << RECIP_ADDR_W)

//==============================================================================
// GLOBAL LUTs & ARRAYS
//==============================================================================
int32_t exp_lut[ROM_DEPTH];
int32_t recip_lut[RECIP_ROM_DEPTH];

int32_t *Q_data, *K_data, *golden_softmax;
int64_t score_int[SEQ_LEN][D_HEAD];
int64_t exp_Z[SEQ_LEN][D_HEAD];
int64_t weights_q15[SEQ_LEN][D_HEAD];

//==============================================================================
// HARDWARE REGISTERS & MACROS
//==============================================================================
#define SRC_BASE_PHYS 0xA0000000
#define DST_BASE_PHYS 0xA0010000
#define DMA_BASE_PHYS 0xA0020000
#define LIN_BASE_PHYS 0xA0030000
#define SM_BASE_PHYS  0xA0040000
#define MAP_SIZE      0x10000

#define MM2S_DMACR  0x00
#define MM2S_DMASR  0x04
#define MM2S_SA     0x18
#define MM2S_LENGTH 0x28
#define S2MM_DMACR  0x30
#define S2MM_DMASR  0x34
#define S2MM_DA     0x48
#define S2MM_LENGTH 0x58

#define DMA_CR_RS      (1 << 0)
#define DMA_CR_RESET   (1 << 2)
#define DMA_SR_HALTED  (1 << 0)
#define DMA_SR_IDLE    (1 << 1)
#define DMA_SR_IOC_IRQ (1 << 12)
#define DMA_SR_ERR_IRQ (1 << 14)
#define DMA_CR_IOC_EN  (1 << 12)

#define HW_SEQ_LEN   64
#define HW_WORDS_K   (D_HEAD * D_MODEL)
#define HW_BYTES_K   (HW_WORDS_K * 4)
#define HW_WORDS_Q   (HW_SEQ_LEN * D_MODEL)
#define HW_BYTES_Q   (HW_WORDS_Q * 4)
#define HW_WORDS_OUT (HW_SEQ_LEN * D_HEAD)
#define HW_BYTES_OUT (HW_WORDS_OUT * 4)

static inline void reg_wr(volatile uint32_t *base, uint32_t offset, uint32_t val) {
    base[offset >> 2] = val;
}

static inline uint32_t reg_rd(volatile uint32_t *base, uint32_t offset) {
    return base[offset >> 2];
}

volatile uint32_t *src_map, *dst_map, *dma_map, *lin_map, *sm_map;
int mem_fd;

volatile uint32_t* phys_map(uint32_t phys_addr) {
    volatile uint32_t* virt = (volatile uint32_t*)mmap(NULL, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, phys_addr);
    if (virt == MAP_FAILED) {
        printf("[ERROR] mmap failed for 0x%08X\n", phys_addr);
        exit(1);
    }
    return virt;
}

//==============================================================================
// INITIALIZATION
//==============================================================================
void init_exp_lut() {
    for (int i = 0; i < ROM_DEPTH; i++) {
        double x = -i / (double)(1 << FRAC_BITS);
        double val = exp(x);
        int32_t q_val = (int32_t)round(val * (1 << FRAC_BITS));
        if (q_val == 0 && val > 0) q_val = 1;
        exp_lut[i] = q_val;
    }
}

void init_recip_lut() {
    int n = 1 << RECIP_ADDR_W;
    for (int i = 0; i < n; i++) {
        double mantissa = 1.0 + (double)i / n;
        double recip = 1.0 / mantissa;
        int32_t q_val = (int32_t)round(recip * (1 << RECIP_OUT_W));
        if (q_val >= (1 << RECIP_OUT_W)) q_val = (1 << RECIP_OUT_W) - 1;
        recip_lut[i] = q_val;
    }
}

int32_t* read_bin(const char* filepath, size_t elements) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        printf("[ERROR] Cannot open %s\n", filepath);
        exit(1);
    }
    int32_t* data = (int32_t*)malloc(elements * sizeof(int32_t));
    size_t read_cnt = fread(data, sizeof(int32_t), elements, f);
    if (read_cnt != elements) {
        printf("[WARN] %s: Read %zu elements, expected %zu\n", filepath, read_cnt, elements);
    }
    fclose(f);
    printf("[OK] Loaded %s\n", filepath);
    return data;
}

//==============================================================================
// CORE FUNCTIONS (Bit-true equivalent)
//==============================================================================
int64_t round_half_to_even_shift(int64_t acc, int shift) {
    int64_t half = 1LL << (shift - 1);
    int64_t low_mask = (1LL << shift) - 1;
    int64_t remainder = acc & low_mask;
    int64_t quotient = acc >> shift;
    int at_half = (remainder == half);
    int above_half = (remainder > half);
    int lsb = quotient & 1;
    int round_up = above_half || (at_half && lsb != 0);
    return quotient + round_up;
}

void compute_attention_score(int32_t* Q, int32_t* K) {
    int64_t max_val = (1LL << (DATA_WIDTH - 1)) - 1;
    int64_t min_val = -(1LL << (DATA_WIDTH - 1));

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < D_HEAD; j++) {
            int64_t mac_sum = 0;
            for (int k = 0; k < D_MODEL; k++) {
                // Must cast to int16_t first to properly sign-extend negative 16-bit fixed point values!
                int64_t q_val = (int16_t)Q[i * D_MODEL + k];
                int64_t k_val = (int16_t)K[j * D_MODEL + k];
                mac_sum += q_val * k_val;
            }
            
            // Phase 1: pe_unit truncation
            int64_t pe_out = round_half_to_even_shift(mac_sum, FRAC_BITS);
            if (pe_out > max_val) pe_out = max_val;
            if (pe_out < min_val) pe_out = min_val;
            
            // Phase 2: SQRT_SHIFT arithmetic shift right
            int64_t score = pe_out >> SQRT_SHIFT;
            if (score > max_val) score = max_val;
            if (score < min_val) score = min_val;
            
            score_int[i][j] = score;
        }
    }
}

int64_t recip_divide_bit_true(int64_t dividend, int64_t divisor, int sum_width, int out_w, int exp_width) {
    if (divisor == 0) return 0;
    
    // GCC Built-in priority encoder for 64-bit uint: count leading zeros
    // (63 - clz) gives MSB position
    int msb_pos = 63 - __builtin_clzll((uint64_t)divisor);
    if (msb_pos >= sum_width) msb_pos = sum_width - 1;

    int64_t recip_addr;
    if (msb_pos >= RECIP_ADDR_W) {
        recip_addr = (divisor >> (msb_pos - RECIP_ADDR_W)) & ((1 << RECIP_ADDR_W) - 1);
    } else {
        recip_addr = (divisor << (RECIP_ADDR_W - msb_pos)) & ((1 << RECIP_ADDR_W) - 1);
    }
    
    int64_t recip_rom_data = recip_lut[recip_addr];
    int64_t prod = dividend * recip_rom_data;
    int shift_total = out_w + msb_pos;
    int64_t shifted = prod >> shift_total;
    
    int64_t max_q15 = (1LL << exp_width) - 1;
    if (shifted > max_q15) shifted = max_q15;
    
    return shifted;
}

void compute_softmax_golden() {
    int sum_width = ceil(log2(D_HEAD)) + DATA_WIDTH;
    int64_t sum_mask = (1LL << sum_width) - 1;
    int64_t dividend_width = DATA_WIDTH + 15;
    int64_t dividend_mask = (1LL << dividend_width) - 1;

    for (int i = 0; i < SEQ_LEN; i++) {
        // Find Max
        int64_t max_score = score_int[i][0];
        for (int j = 1; j < D_HEAD; j++) {
            if (score_int[i][j] > max_score) max_score = score_int[i][j];
        }
        
        // Subtract Max, Look up EXP, Sum
        int64_t sum_latched = 0;
        for (int j = 0; j < D_HEAD; j++) {
            int64_t z_val = score_int[i][j] - max_score;
            int addr = (-z_val) & 0x7FF;
            int64_t e_val = (addr < ROM_DEPTH) ? exp_lut[addr] : 0;
            exp_Z[i][j] = e_val;
            sum_latched += e_val;
        }
        sum_latched &= sum_mask;

        // Reciprocal Division
        for (int j = 0; j < D_HEAD; j++) {
            int64_t div_dividend = (exp_Z[i][j] << 15) & dividend_mask;
            weights_q15[i][j] = recip_divide_bit_true(div_dividend, sum_latched, sum_width, RECIP_OUT_W, DATA_WIDTH);
        }
    }
}

//==============================================================================
// HARDWARE EXECUTION
//==============================================================================
int wait_done(volatile uint32_t* dma, uint32_t sr_off, const char* name) {
    for (int i = 0; i < 5000000; i++) {
        uint32_t sr = reg_rd(dma, sr_off);
        if (sr & DMA_SR_ERR_IRQ) {
            printf("[FAIL] %s DMA error, DMASR=0x%08X\n", name, sr);
            return 0;
        }
        if (sr & DMA_SR_IOC_IRQ) {
            return 1;
        }
    }
    printf("[FAIL] %s DMA timeout, DMASR=0x%08X\n", name, reg_rd(dma, sr_off));
    return 0;
}

int dma_reset(volatile uint32_t* dma) {
    reg_wr(dma, MM2S_DMACR, DMA_CR_RESET);
    reg_wr(dma, S2MM_DMACR, DMA_CR_RESET);
    for (int i = 0; i < 500000; i++) {
        if (!(reg_rd(dma, MM2S_DMACR) & DMA_CR_RESET) && !(reg_rd(dma, S2MM_DMACR) & DMA_CR_RESET)) {
            return 1;
        }
    }
    return 0;
}

//==============================================================================
// MAIN
//==============================================================================
int main() {
    mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        printf("[ERROR] Cannot open /dev/mem. Please run as root (sudo).\n");
        return 1;
    }

    printf("=========================================================\n");
    printf("  CPU BENCHMARK: Q[%dx%d] * K[%dx%d]^T + SOFTMAX \n", SEQ_LEN, D_MODEL, D_HEAD, D_MODEL);
    printf("=========================================================\n");

    // 1. Initialize LUTs
    printf("[INFO] Generating ROM LUTs...\n");
    init_exp_lut();
    init_recip_lut();

    // 2. Read Binary Files
    printf("[INFO] Loading binary files...\n");
    size_t q_elements = SEQ_LEN * D_MODEL;
    size_t k_elements = D_HEAD * D_MODEL;
    size_t s_elements = SEQ_LEN * D_HEAD;
    
    Q_data = read_bin("q_data.bin", q_elements);
    K_data = read_bin("k_data.bin", k_elements);
    golden_softmax = read_bin("golden_softmax.bin", s_elements);

    // 3. Timing Execution
    printf("\n[INFO] Starting execution and timer...\n");
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Core Execution
    compute_attention_score(Q_data, K_data);
    compute_softmax_golden();
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    // 4. Verification
    printf("[INFO] Verifying with golden_softmax.bin...\n");
    int error_count = 0;
    for(int i=0; i<SEQ_LEN; i++) {
        for(int j=0; j<D_HEAD; j++) {
            int64_t computed = weights_q15[i][j];
            int32_t expected = golden_softmax[i * D_HEAD + j] & 0xFFFF; // Match format
            if(computed != expected) {
                if(error_count < 5) {
                    printf("  -> Mismatch at [%d][%d]: Expected %d, Got %lld\n", i, j, expected, computed);
                }
                error_count++;
            }
        }
    }
    if (error_count == 0) {
        printf("[OK] All Softmax outputs match identically!\n");
    } else {
        printf("[WARN] Total %d mismatches found.\n", error_count);
    }

    // 5. Compute Latency and Throughput
    double elapsed_us = (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3;
    
    // Operations: 2 * M * N * K
    double ops = 2.0 * SEQ_LEN * D_HEAD * D_MODEL;
    double throughput_gops = (ops / elapsed_us) / 1000.0;
    
    // Performance will be printed at the end in a unified table.
    
    //==========================================================================
    // HARDWARE ACCELERATOR RUN
    //==========================================================================
    printf("\n[INFO] Mapping /dev/mem for HW Accelerator...\n");
    src_map = phys_map(SRC_BASE_PHYS);
    dst_map = phys_map(DST_BASE_PHYS);
    dma_map = phys_map(DMA_BASE_PHYS);
    lin_map = phys_map(LIN_BASE_PHYS);
    sm_map  = phys_map(SM_BASE_PHYS);

    printf("[INFO] Writing K to BRAM (One-time)...\n");
    for (int i = 0; i < HW_WORDS_K; i++) {
        reg_wr(src_map, i * 4, K_data[i]);
    }

    int num_tiles = (SEQ_LEN + HW_SEQ_LEN - 1) / HW_SEQ_LEN;
    printf("[INFO] Starting HW execution with %d tile(s)...\n", num_tiles);
    
    struct timespec hw_start, hw_end;
    double hw_elapsed_us = 0;
    uint32_t linear_cycles = 0;
    uint32_t softmax_cycles = 0;
    int32_t* hw_output = (int32_t*)malloc(SEQ_LEN * D_HEAD * sizeof(int32_t));

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int row_start = tile_idx * HW_SEQ_LEN;
        int row_end = row_start + HW_SEQ_LEN;
        if (row_end > SEQ_LEN) row_end = SEQ_LEN;
        int actual_rows = row_end - row_start;

        // Load Q tile into BRAM (zero-padded)
        for (int r = 0; r < HW_SEQ_LEN; r++) {
            for (int c = 0; c < D_MODEL; c++) {
                int32_t val = 0;
                if (r < actual_rows) {
                    val = Q_data[(row_start + r) * D_MODEL + c];
                }
                reg_wr(src_map, HW_BYTES_K + (r * D_MODEL + c) * 4, val);
            }
        }

        if (!dma_reset(dma_map)) {
            printf("[FAIL] DMA reset failed on tile %d!\n", tile_idx);
            return 1;
        }

        clock_gettime(CLOCK_MONOTONIC, &hw_start);

        // PHASE 1: DMA K -> Linear
        reg_wr(dma_map, MM2S_DMACR, DMA_CR_RS | DMA_CR_IOC_EN);
        reg_wr(dma_map, MM2S_SA, SRC_BASE_PHYS);
        reg_wr(lin_map, 0x00, 1);
        reg_wr(dma_map, MM2S_LENGTH, HW_BYTES_K);

        if (!wait_done(dma_map, MM2S_DMASR, "MM2S-K")) return 1;
        reg_wr(dma_map, MM2S_DMASR, DMA_SR_IOC_IRQ); // clear IOC

        // PHASE 2: Softmax + DMA Q -> S2MM
        reg_wr(sm_map, 0x00, 1);

        // Arm S2MM
        reg_wr(dma_map, S2MM_DMACR, DMA_CR_RS | DMA_CR_IOC_EN);
        reg_wr(dma_map, S2MM_DA, DST_BASE_PHYS);
        reg_wr(dma_map, S2MM_LENGTH, HW_BYTES_OUT);

        // Restart MM2S for Q
        reg_wr(dma_map, MM2S_DMACR, DMA_CR_RS | DMA_CR_IOC_EN);
        reg_wr(dma_map, MM2S_SA, SRC_BASE_PHYS + HW_BYTES_K);
        reg_wr(dma_map, MM2S_LENGTH, HW_BYTES_Q);

        if (!wait_done(dma_map, MM2S_DMASR, "MM2S-Q")) return 1;
        if (!wait_done(dma_map, S2MM_DMASR, "S2MM")) return 1;

        clock_gettime(CLOCK_MONOTONIC, &hw_end);

        // Accumulate Linux Overhead Timing
        hw_elapsed_us += (hw_end.tv_sec - hw_start.tv_sec) * 1e6 + (hw_end.tv_nsec - hw_start.tv_nsec) / 1e3;
        
        // Accumulate Hardware Cycles
        linear_cycles += reg_rd(lin_map, 0x08);
        softmax_cycles += reg_rd(sm_map, 0x08);

        // Read outputs for this tile (discarding zero-padded output rows)
        for (int r = 0; r < actual_rows; r++) {
            for (int c = 0; c < D_HEAD; c++) {
                hw_output[(row_start + r) * D_HEAD + c] = reg_rd(dst_map, (r * D_HEAD + c) * 4);
            }
        }
    }
    
    // Assume 150MHz clock for exact HW time (6.6667ns per cycle)
    double exact_lin_us = linear_cycles * (1000.0 / 150.0) / 1000.0;
    double exact_sm_us  = softmax_cycles * (1000.0 / 150.0) / 1000.0;
    double exact_hw_us = exact_lin_us + exact_sm_us;

    // Verify HW output
    int hw_error_count = 0;
    for(int i=0; i<SEQ_LEN; i++) {
        for(int j=0; j<D_HEAD; j++) {
            int32_t computed = hw_output[i * D_HEAD + j];
            int32_t expected = golden_softmax[i * D_HEAD + j] & 0xFFFF;
            if(computed != expected) {
                if(hw_error_count < 5) {
                    printf("  -> HW Mismatch at [%d][%d]: Expected %d, Got %d\n", i, j, expected, computed);
                }
                hw_error_count++;
            }
        }
    }
    if (hw_error_count == 0) {
        printf("[OK] Hardware outputs match perfectly!\n");
    }

    double hw_throughput_gops = (ops / exact_hw_us) / 1000.0;
    double speedup = elapsed_us / exact_hw_us;

    printf("\n========================================================================\n");
    printf("                      FINAL PERFORMANCE BENCHMARK                       \n");
    printf("========================================================================\n");
    printf(" Metric                |  CPU (Cortex-A53)  |  FPGA (Hardware)   | Unit \n");
    printf("------------------------------------------------------------------------\n");
    printf(" Execution Time        | %18.2f | %18.2f | us   \n", elapsed_us, exact_hw_us);
    printf(" Total Operations      | %18.0f | %18.0f | OPs  \n", ops, ops);
    printf(" Throughput            | %18.4f | %18.4f | GOP/s\n", throughput_gops, hw_throughput_gops);
    printf("------------------------------------------------------------------------\n");
    printf(" SPEEDUP (FPGA / CPU)  |                    | %17.2fx |      \n", speedup);
    printf("========================================================================\n");
    printf(" * Note: FPGA time based on exact %u (Linear) + %u (Softmax) cycles @ 150MHz.\n\n", linear_cycles, softmax_cycles);

    free(Q_data);
    free(K_data);
    free(golden_softmax);
    free(hw_output);
    
    munmap((void*)src_map, MAP_SIZE);
    munmap((void*)dst_map, MAP_SIZE);
    munmap((void*)dma_map, MAP_SIZE);
    munmap((void*)lin_map, MAP_SIZE);
    munmap((void*)sm_map, MAP_SIZE);
    close(mem_fd);
    
    return 0;
}
