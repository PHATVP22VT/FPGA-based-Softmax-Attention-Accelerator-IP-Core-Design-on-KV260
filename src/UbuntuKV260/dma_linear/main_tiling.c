#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <time.h>

// --- Thay the cho <sys/mmap.h> bi thieu ---
#define PROT_READ  0x1
#define PROT_WRITE 0x2
#define MAP_SHARED 0x01
#define MAP_FAILED ((void *)-1)
extern void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
// ------------------------------------------

#define MAP_SIZE      0x10000

// Addresses
#define SRC_BASE_PHYS 0xA0000000
#define DST_BASE_PHYS 0xA0010000
#define DMA_BASE_PHYS 0xA0020000
#define LIN_BASE_PHYS 0xA0030000

// DMA Offsets
#define MM2S_DMACR    0x00
#define MM2S_DMASR    0x04
#define MM2S_SA       0x18
#define MM2S_LENGTH   0x28
#define S2MM_DMACR    0x30
#define S2MM_DMASR    0x34
#define S2MM_DA       0x48
#define S2MM_LENGTH   0x58

#define DMA_CR_RS       (1 << 0)
#define DMA_CR_RESET    (1 << 2)
#define DMA_SR_ERR_IRQ  (1 << 14)
#define DMA_SR_IOC_IRQ  (1 << 12)
#define DMA_CR_IOC_EN   (1 << 12)

// HW Parameters (Fixed 64x64 block)
#define HW_SEQ_LEN    64
#define HW_D_HEAD     64
#define HW_D_MODEL    64
#define HW_WORDS_K    (HW_D_HEAD * HW_D_MODEL)
#define HW_BYTES_K    (HW_WORDS_K * 4)
#define HW_WORDS_Q    (HW_SEQ_LEN * HW_D_MODEL)
#define HW_BYTES_Q    (HW_WORDS_Q * 4)
#define HW_WORDS_OUT  (HW_SEQ_LEN * HW_D_HEAD)
#define HW_BYTES_OUT  (HW_WORDS_OUT * 4)

#define K_BASE_PHYS   SRC_BASE_PHYS
#define Q_BASE_PHYS   (SRC_BASE_PHYS + HW_BYTES_K)

// Helper: MMIO Write
static inline void reg_wr(volatile void *base, uint32_t offset, uint32_t val) {
    *(volatile uint32_t *)((uint8_t *)base + offset) = val;
}

// Helper: MMIO Read
static inline uint32_t reg_rd(volatile void *base, uint32_t offset) {
    return *(volatile uint32_t *)((uint8_t *)base + offset);
}

// Helper: Map Physical Memory
void* phys_map(int fd, off_t phys_addr) {
    void *virt = mmap(NULL, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, phys_addr);
    if (virt == MAP_FAILED) {
        printf("[ERROR] mmap failed for 0x%lx\n", phys_addr);
        exit(1);
    }
    return virt;
}

int wait_done(volatile void *dma, uint32_t sr_off, const char *name) {
    // Bước 1: Chống Race Condition. Đợi một chút để lệnh chạy kịp truyền tới DMA
    // Vòng lặp này tương đương khoảng vài chục micro giây.
    for (volatile int i = 0; i < 50000; i++) {
        uint32_t sr = reg_rd(dma, sr_off);
        if (!(sr & (1 << 1))) break; // Vừa thấy DMA bận (thoát IDLE cũ) là chuyển sang bước 2 ngay
    }
    
    // Bước 2: Chờ DMA rảnh rỗi trở lại (IDLE mới = Truyền hoàn tất 100%)
    for (volatile int i = 0; i < 5000000; i++) {
        uint32_t sr = reg_rd(dma, sr_off);
        if (sr & DMA_SR_ERR_IRQ) {
            printf("[FAIL] %s DMA error, DMASR=0x%08x\n", name, sr);
            return 0;
        }
        if (sr & (1 << 1)) return 1; // Truyền xong!
    }
    printf("[FAIL] %s DMA timeout, DMASR=0x%08x\n", name, reg_rd(dma, sr_off));
    return 0;
}

int dma_reset(volatile void *dma) {
    reg_wr(dma, MM2S_DMACR, DMA_CR_RESET);
    reg_wr(dma, S2MM_DMACR, DMA_CR_RESET);
    for (int i = 0; i < 500000; i++) {
        if (!(reg_rd(dma, MM2S_DMACR) & DMA_CR_RESET) &&
            !(reg_rd(dma, S2MM_DMACR) & DMA_CR_RESET)) {
            return 1;
        }
    }
    return 0;
}

// Read binary file to allocated memory
uint32_t* read_bin_file(const char *filename, int *out_words) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        printf("[ERROR] Cannot open %s\n", filename);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *data = (uint32_t *)malloc(size);
    if(fread(data, 1, size, f) != size) {
        printf("[ERROR] Failed to read %s\n", filename);
        exit(1);
    }
    fclose(f);
    *out_words = size / 4;
    return data;
}

// Time diff in nanoseconds
long long get_ns(struct timespec t1, struct timespec t2) {
    return (long long)(t2.tv_sec - t1.tv_sec) * 1000000000LL + (t2.tv_nsec - t1.tv_nsec);
}

int main() {
    printf("=================================================================\n");
    printf("  IP LINEAR BARE-METAL TILING TEST ON UBUNTU (C VERSION)\n");
    printf("=================================================================\n");

    int num_k_words, num_q_words, num_golden_words;
    uint32_t *k_words = read_bin_file("k_data.bin", &num_k_words);
    uint32_t *q_words = read_bin_file("q_data.bin", &num_q_words);
    uint32_t *golden_words = read_bin_file("golden_score.bin", &num_golden_words);

    int total_seq_len = num_q_words / HW_D_MODEL;
    int total_d_head  = num_k_words / HW_D_MODEL;

    if (num_q_words != total_seq_len * HW_D_MODEL) {
        printf("[ERROR] q_data.bin size invalid\n"); return 1;
    }
    if (num_k_words != total_d_head * HW_D_MODEL) {
        printf("[ERROR] k_data.bin size invalid\n"); return 1;
    }

    int expected_golden = total_seq_len * total_d_head;
    if (num_golden_words != expected_golden) {
        printf("[ERROR] golden_score.bin has %d words, expected %d\n", num_golden_words, expected_golden);
        return 1;
    }

    int num_tiles_q = (total_seq_len + HW_SEQ_LEN - 1) / HW_SEQ_LEN;
    int num_tiles_k = (total_d_head + HW_D_HEAD - 1) / HW_D_HEAD;

    printf("  Target Matrix  : Q[%dx%d] x K[%dx%d]\n", total_seq_len, HW_D_MODEL, total_d_head, HW_D_MODEL);
    printf("  Hardware Block : 64x64\n");
    printf("  Total Tiles    : %d (Rows) x %d (Cols) = %d passes\n", num_tiles_q, num_tiles_k, num_tiles_q * num_tiles_k);
    printf("=================================================================\n");

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        printf("[ERROR] Cannot open /dev/mem. Are you root?\n"); return 1;
    }

    volatile void *src = phys_map(fd, SRC_BASE_PHYS);
    volatile void *dst = phys_map(fd, DST_BASE_PHYS);
    volatile void *dma = phys_map(fd, DMA_BASE_PHYS);
    volatile void *lin = phys_map(fd, LIN_BASE_PHYS);

    uint32_t *final_output = (uint32_t *)calloc(expected_golden, sizeof(uint32_t));
    long long total_hw_ns = 0;
    struct timespec t_start, t_end;

    uint32_t q_tile[HW_WORDS_Q];
    uint32_t k_tile[HW_WORDS_K];
    uint32_t out_tile[HW_WORDS_OUT];

    for (int tr = 0; tr < num_tiles_q; tr++) {
        // Extract Q tile
        int q_row_start = tr * HW_SEQ_LEN;
        for (int r = 0; r < HW_SEQ_LEN; r++) {
            int global_row = q_row_start + r;
            for (int c = 0; c < HW_D_MODEL; c++) {
                if (global_row < total_seq_len)
                    q_tile[r * HW_D_MODEL + c] = q_words[global_row * HW_D_MODEL + c];
                else
                    q_tile[r * HW_D_MODEL + c] = 0; // Zero padding
            }
        }

        for (int tc = 0; tc < num_tiles_k; tc++) {
            // Extract K tile
            int k_row_start = tc * HW_D_HEAD;
            for (int r = 0; r < HW_D_HEAD; r++) {
                int global_row = k_row_start + r;
                for (int c = 0; c < HW_D_MODEL; c++) {
                    if (global_row < total_d_head)
                        k_tile[r * HW_D_MODEL + c] = k_words[global_row * HW_D_MODEL + c];
                    else
                        k_tile[r * HW_D_MODEL + c] = 0; // Zero padding
                }
            }

            // --- HW RUN ---
            clock_gettime(CLOCK_MONOTONIC, &t_start);

            // Write to SRC BRAM
            for (int i = 0; i < HW_WORDS_K; i++) reg_wr(src, i * 4, k_tile[i]);
            for (int i = 0; i < HW_WORDS_Q; i++) reg_wr(src, HW_BYTES_K + i * 4, q_tile[i]);

            // [RẤT QUAN TRỌNG] Data Memory Barrier: Ép CPU ARM phải ghi dữ liệu từ L1/L2 Cache xuống hẳn BRAM trước khi kích hoạt DMA
            __sync_synchronize();

            if (!dma_reset(dma)) return 1;

            // [SỬA LỖI DEADLOCK]: Bật S2MM (Đầu nhận) hứng sẵn ở BRAM_DST TRƯỚC KHI cấp K vào IP
            reg_wr(dma, S2MM_DMACR, DMA_CR_RS);
            reg_wr(dma, S2MM_DA, DST_BASE_PHYS);
            reg_wr(dma, S2MM_LENGTH, HW_BYTES_OUT);

            // Phase 1: DMA K
            reg_wr(dma, MM2S_DMACR, DMA_CR_RS);
            reg_wr(dma, MM2S_SA, K_BASE_PHYS);
            reg_wr(lin, 0x00, 1);
            reg_wr(dma, MM2S_LENGTH, HW_BYTES_K);

            if (!wait_done(dma, MM2S_DMASR, "MM2S-K")) return 1;
            reg_wr(dma, MM2S_DMASR, DMA_SR_IOC_IRQ);

            // Phase 2: DMA Q -> LIN
            reg_wr(dma, MM2S_DMACR, DMA_CR_RS);
            reg_wr(dma, MM2S_SA, Q_BASE_PHYS);
            reg_wr(dma, MM2S_LENGTH, HW_BYTES_Q);

            if (!wait_done(dma, MM2S_DMASR, "MM2S-Q")) return 1;
            if (!wait_done(dma, S2MM_DMASR, "S2MM")) return 1;

            // Read DST BRAM
            for (int i = 0; i < HW_WORDS_OUT; i++) out_tile[i] = reg_rd(dst, i * 4);

            clock_gettime(CLOCK_MONOTONIC, &t_end);
            total_hw_ns += get_ns(t_start, t_end);
            // --- END HW RUN ---

            // Crop and place into final_output
            int row_start = tr * HW_SEQ_LEN;
            int row_end   = (row_start + HW_SEQ_LEN < total_seq_len) ? (row_start + HW_SEQ_LEN) : total_seq_len;
            int col_start = tc * HW_D_HEAD;
            int col_end   = (col_start + HW_D_HEAD < total_d_head) ? (col_start + HW_D_HEAD) : total_d_head;
            
            int actual_rows = row_end - row_start;
            int actual_cols = col_end - col_start;

            for (int r = 0; r < actual_rows; r++) {
                for (int c = 0; c < actual_cols; c++) {
                    int global_idx = (row_start + r) * total_d_head + (col_start + c);
                    int tile_idx   = r * HW_D_HEAD + c;
                    final_output[global_idx] = out_tile[tile_idx];
                }
            }
        }
    }

    printf("\n=================================================================\n");
    printf("  RESULTS - Total HW time: %lld ns  (%.2f ms)\n", total_hw_ns, (float)total_hw_ns / 1000000.0f);
    printf("=================================================================\n");

    int fail_cnt = 0;
    FILE *fout = fopen("result_linear_tiling_c.txt", "w");
    fprintf(fout, "Index   | Golden Score    | RTL Data        | Mismatch        | Status\n");
    fprintf(fout, "---------------------------------------------------------------------------\n");

    for (int i = 0; i < num_golden_words; i++) {
        uint32_t g = final_output[i];
        uint32_t e = golden_words[i];
        const char *status = (g == e) ? "PASS" : "FAIL";
        uint32_t mismatch = (g == e) ? 0 : (g ^ e);

        if (g != e) {
            fail_cnt++;
            if (fail_cnt <= 20) {
                int row = i / total_d_head;
                int col = i % total_d_head;
                printf("  FAIL [%d] row=%d col=%d exp=0x%08x got=0x%08x\n", i, row, col, e, g);
            }
        }
        fprintf(fout, "%-7d | 0x%08x      | 0x%08x      | 0x%-13x | %s\n", i, e, g, mismatch, status);
    }
    fclose(fout);

    if (fail_cnt == 0) {
        printf("  *** PASS: all %d values match golden score ***\n", num_golden_words);
    } else {
        printf("  *** FAIL: %d/%d values mismatch ***\n", fail_cnt, num_golden_words);
    }

    printf("\n  Results saved to: result_linear_tiling_c.txt\n\n");

    free(k_words);
    free(q_words);
    free(golden_words);
    free(final_output);

    return (fail_cnt == 0) ? 0 : 1;
}
