# Kế hoạch Benchmark bằng mã nguồn C trên Ubuntu Linux (KV260)

## Mục tiêu
Triển khai một bài test hiệu năng (benchmark) bằng ngôn ngữ C đảm bảo độ chính xác tới từng bit (bit-true) chạy trên Ubuntu Linux. Mục đích là để đánh giá thời gian thực thi và thông lượng (Speed-up) của cơ chế Attention: tính $Q[64 \times 64] \times K[64 \times 64]^T$ + Softmax trên vi xử lý CPU (Cortex-A53) của board KV260. Thuật toán sẽ tuân thủ nghiêm ngặt mô hình chuẩn (golden model) Python (`golden_model.py`), sử dụng hoàn toàn số nguyên dấu phẩy tĩnh (fixed-point integer arithmetic).

## Phương án triển khai

### 1. Kỹ thuật đo lường thời gian (Timing)
Sử dụng hàm POSIX tiêu chuẩn `clock_gettime(CLOCK_MONOTONIC, &ts)` để đo lường thời gian chạy phần mềm trên Ubuntu Linux. Hàm này cung cấp độ phân giải tới mức nano-giây, đủ độ chính xác để tính toán `GOP/s`.

```c
struct timespec start, end;
clock_gettime(CLOCK_MONOTONIC, &start);
// [Vùng thực thi phép nhân QxK^T + Softmax]
clock_gettime(CLOCK_MONOTONIC, &end);
double elapsed_us = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_nsec - start.tv_nsec) / 1000.0;
```

### 2. Dịch thuật mô hình Python Golden Model sang C
- **Kiểu dữ liệu (Data Types)**: Sử dụng `int16_t` cho mảng dữ liệu đầu vào và `int64_t` cho bộ tích lũy nhân cộng (MAC), đảm bảo độ chính xác fixed-point y hệt như mô hình Python.
- **Kích thước ma trận**: $M = 64, N = 64, K = 64$ (`SEQ_LEN=64`, `D_MODEL=64`, `D_HEAD=64`).
- **Khởi tạo Bảng tra cứu (LUT)**: Mã nguồn C sẽ tự động sinh (generate) ra các bảng `exp_lut` và `recip_lut` ở đầu chương trình (giống như `generate_exp_lut` và `generate_recip_lut`). **Lưu ý:** Quá trình khởi tạo LUT này sẽ *không* bị tính vào thời gian đo đếm, vì trên phần cứng các ROM này đã được load sẵn.
- **Thực thi các thuật toán lõi**:
  - **`compute_attention_score()`**: Dùng 3 vòng lặp lồng nhau (`64 x 64 x 64`) để tính toán tích lũy MAC trên biến `int64_t`, sau đó áp dụng chính xác thuật toán `round_half_to_even_shift(mac_sum, 8)` và dịch phải (cắt bit) `>> 3` (`SQRT_SHIFT`).
  - **`compute_softmax_golden()`**:
    1. **Tìm Max & Tính EXP**: Duyệt qua từng hàng để tìm Max, tính `(-z) & 0x7FF`, truy xuất từ `exp_lut`, và cộng tích lũy vào `sum_latched`.
    2. **Phép chia (Division)**: Sử dụng hàm dựng sẵn của trình biên dịch GCC `__builtin_clzll()` hoặc `__builtin_clz()` để tìm vị trí bit cao nhất (MSB) của mẫu số (divisor), từ đó tính ra `recip_addr`, truy xuất từ bảng `recip_lut`, thực hiện phép nhân và dịch bit theo đúng số học của mạch chia.

### 3. Tính toán Thông lượng (Throughput)
- Chương trình sẽ in ra tổng độ trễ thực thi (Latency) tính bằng micro-giây ($\mu s$).
- Tính thông lượng (GOP/s) dựa trên công thức thảo luận trước đó:
  $Operations = 2 \times 64 \times 64 \times 64 = 524,288 \text{ OPs}$
  $GOP/s = \frac{524,288}{\text{Latency (us)} \times 1000}$

## Các Câu Hỏi Mở Cần Xác Nhận

> [!WARNING]
> Xin bạn hãy xem qua và phản hồi lại 2 câu hỏi sau trước khi tôi bắt đầu viết code:
> 1. Bạn muốn code C tự phát sinh ngẫu nhiên (random generate) 2 ma trận $Q$ và $K$ ngay khi chạy, hay bạn muốn tôi viết thêm 1 hàm để chương trình C đi đọc trực tiếp file nhị phân (`q_data.bin`, `k_data.bin`) mà Python script đã xuất ra? (Việc sinh số ngẫu nhiên thẳng trong C sẽ giúp code gọn nhẹ hơn nhiều).
> 2. Bạn muốn xuất file `benchmark.c` ra thư mục/đường dẫn cụ thể nào trên máy tính?
