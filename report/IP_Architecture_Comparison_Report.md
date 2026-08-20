# Báo Cáo Phân Tích Kiến Trúc
## So Sánh Phiên Bản IP Attention Accelerator: v1.0 (Baseline) vs v5.0 (Optimized)

> **Ngày lập:** 09/08/2026  
> **Dự án:** Thiết kế IP tăng tốc phần cứng cho cơ chế Attention trên nền tảng FPGA Xilinx Kria KV260  
> **Tác giả:** Nhóm thiết kế RTL  
> **Phạm vi:** So sánh chi tiết kiến trúc giữa hai phiên bản IP Linear và IP Softmax dựa trên 5 tiêu chí kỹ thuật

---

## Mục Lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Phạm vi so sánh](#2-phạm-vi-so-sánh)
3. [Phân tích chi tiết](#3-phân-tích-chi-tiết)
   - 3.1 [Port Interface](#31-port-interface)
   - 3.2 [AXI-Lite Register Map](#32-axi-lite-register-map)
   - 3.3 [FSM & Control Plane](#33-fsm--control-plane)
   - 3.4 [Datapath & Pipeline](#34-datapath--pipeline)
   - 3.5 [Softmax Division Method](#35-softmax-division-method)
4. [Tổng kết các thay đổi](#4-tổng-kết-các-thay-đổi)

---

## 1. Giới Thiệu

Tài liệu này trình bày kết quả phân tích và so sánh kiến trúc giữa hai phiên bản của hệ thống IP tăng tốc phần cứng (Hardware Accelerator) cho cơ chế **Scaled Dot-Product Attention**, bao gồm hai khối IP chính:

- **IP Linear** (`ip_axi_linear`): Thực hiện phép nhân ma trận $S = Q \times K^T$ với scaling $\frac{1}{\sqrt{d_k}}$.
- **IP Softmax** (`ip_axi_softmax`): Thực hiện hàm kích hoạt $\text{Softmax}(S)$ theo hàng, sử dụng số học Fixed-Point.

Hai phiên bản được so sánh:

| Thuộc tính | Phiên bản Baseline (v1.0) | Phiên bản Optimized (v5.0) |
|:---|:---|:---|
| **Nguồn** | Repository Git (clone) | Vivado IP Repository (đang triển khai) |
| **Đường dẫn gốc** | `..\GitProjects\IP-LINEAR-SOFTMAX-DMA\src\ip\` | `..\Xilinx_projects\ip_repo\` |
| **IP Linear** | `linear\` | `ip_axi_linear_5_0\` |
| **IP Softmax** | `softmax\` | `ip_axi_softmax_1_0\` |

Mục tiêu chính của báo cáo là xác định, phân loại, và đánh giá tất cả các thay đổi về mặt kiến trúc giữa hai phiên bản nhằm phục vụ cho việc truy vết thiết kế (design traceability) và tài liệu hóa hệ thống.

---

## 2. Phạm Vi So Sánh

Việc so sánh được thực hiện dựa trên **5 tiêu chí kỹ thuật** (được lựa chọn bởi nhóm thiết kế):

| # | Tiêu chí | Mô tả |
|:---:|:---|:---|
| 1 | **Port Interface** | Giao diện cổng vào/ra của Top-level Wrapper |
| 2 | **AXI-Lite Register Map** | Bản đồ thanh ghi điều khiển/trạng thái (Control/Status) |
| 3 | **FSM & Control Plane** | Máy trạng thái hữu hạn và luồng điều khiển |
| 4 | **Datapath & Pipeline** | Đường dẫn dữ liệu: MAC, Rounding, Buffering |
| 5 | **Softmax Division Method** | Phương pháp thực hiện phép chia trong khối Softmax |

Các file RTL được phân tích bao gồm:

| Thành phần | File | Vai trò |
|:---|:---|:---|
| Top Wrapper | `ip_axi_linear.v`, `ip_axi_softmax.v` | Vỏ bọc tích hợp AXI |
| AXI-Lite Slave | `*_S00_AXI.v` | Giao tiếp thanh ghi với SoC |
| Core Logic | `linear.sv`, `softmax.sv` | Lõi tính toán chính |

---

## 3. Phân Tích Chi Tiết

### 3.1 Port Interface

#### 3.1.1 IP Linear — Top Wrapper (`ip_axi_linear.v`)

Cả hai phiên bản đều giữ nguyên cấu trúc giao diện ngoài gồm:
- 1 cổng AXI4-Lite Slave (điều khiển/trạng thái)
- 1 cổng AXI4-Stream Slave (nhận dữ liệu $K$, $Q$ từ DMA)
- 1 cổng AXI4-Stream Master (xuất Attention Score $S$ ra DMA)

Điểm khác biệt nằm ở **tín hiệu nội bộ** (internal wires) kết nối giữa AXI-Lite Slave và Core:

| Tín hiệu nội bộ | Baseline (v1.0) | Optimized (v5.0) | Thay đổi |
|:---|:---:|:---:|:---|
| `start_attn_score` | ✅ | ✅ | Không đổi |
| `attn_score_done` | ✅ | ✅ | Không đổi |
| `busy` | ✅ | ✅ | Không đổi |
| `linear_cycles` `[31:0]` | ❌ | ✅ | **Thêm mới** — Kết nối từ `u_linear.o_linear_cycles` tới `S00_AXI_inst.i_linear_cycles` |

#### 3.1.2 IP Softmax — Top Wrapper (`ip_axi_softmax.v`)

Tương tự IP Linear, giao diện ngoài không thay đổi. Thay đổi nội bộ:

| Tín hiệu nội bộ | Baseline (v1.0) | Optimized (v5.0) | Thay đổi |
|:---|:---:|:---:|:---|
| `start_softmax` | ✅ | ✅ | Không đổi |
| `softmax_done` | ✅ | ✅ | Không đổi |
| `busy` | ✅ | ✅ | Không đổi |
| `softmax_cycles` `[31:0]` | ❌ | ✅ | **Thêm mới** — Kết nối từ `u_softmax.o_softmax_cycles` tới `S00_AXI_inst.i_softmax_cycles` |

#### 3.1.3 Core Logic — Cổng đầu ra phi tiêu chuẩn

| Cổng | Module | Baseline | Optimized | Ghi chú |
|:---|:---|:---:|:---:|:---|
| `o_linear_cycles [31:0]` | `linear.sv` | ❌ | ✅ | Xuất giá trị bộ đếm chu kỳ phần cứng |
| `o_softmax_cycles [31:0]` | `softmax.sv` | ❌ | ✅ | Xuất giá trị bộ đếm chu kỳ phần cứng |

> **Nhận xét:** Giao diện bên ngoài (External Interface) của cả hai IP hoàn toàn tương thích ngược (backward-compatible). Mọi thay đổi đều nằm ở lớp nội bộ, không ảnh hưởng tới Block Design hiện có.

---

### 3.2 AXI-Lite Register Map

#### 3.2.1 IP Linear Slave (`ip_axi_linear_slave_lite_v5_0_S00_AXI.v`)

| Thanh ghi | Offset | Baseline (v1.0) | Optimized (v5.0) |
|:---|:---:|:---|:---|
| `slv_reg0` | `0x00` | **Control:** bit[0] = `o_start_attn_score` (auto-clear khi `busy_rising`) | Không đổi |
| `slv_reg1` | `0x04` | **Status (Read-Only View):** `{30'b0, i_busy, latched_done}` | Không đổi |
| `slv_reg2` | `0x08` | **Scratchpad R/W:** Trả về `slv_reg2` (thanh ghi đa dụng) | **⚡ HW Cycle Counter (Read-Only View):** Trả về `i_linear_cycles` |
| `slv_reg3` | `0x0C` | **Scratchpad R/W:** Trả về `slv_reg3` | Không đổi |

#### 3.2.2 IP Softmax Slave (`ip_axi_softmax_slave_lite_v1_0_S00_AXI.v`)

| Thanh ghi | Offset | Baseline (v1.0) | Optimized (v5.0) |
|:---|:---:|:---|:---|
| `slv_reg0` | `0x00` | **Control:** bit[0] = `o_start_softmax` (auto-clear khi `busy_rising`) | Không đổi |
| `slv_reg1` | `0x04` | **Status (Read-Only View):** `{30'b0, i_busy, latched_done}` | Không đổi |
| `slv_reg2` | `0x08` | **Scratchpad R/W:** Trả về `slv_reg2` (thanh ghi đa dụng) | **⚡ HW Cycle Counter (Read-Only View):** Trả về `i_softmax_cycles` |
| `slv_reg3` | `0x0C` | **Scratchpad R/W:** Trả về `slv_reg3` | Không đổi |

#### 3.2.3 Cơ chế hoạt động của Hardware Cycle Counter (Phiên bản Optimized)

Bộ đếm chu kỳ phần cứng hoạt động theo nguyên tắc sau (áp dụng cho cả Linear và Softmax):

1. **Reset:** Thanh ghi đếm bị xóa về `0` khi có tín hiệu Reset toàn cục (`irst_n = 0`) hoặc khi nhận xung kích hoạt mới (`i_start = 1`).
2. **Đếm:** Tự động tăng thêm `+1` ở mỗi cạnh lên của Clock trong suốt khoảng thời gian `o_busy = 1`.
3. **Đọc:** Giá trị đếm được ánh xạ trực tiếp lên thanh ghi `slv_reg2` (Offset `0x08`), cho phép SoC đọc chính xác số chu kỳ xung nhịp mà IP đã sử dụng để hoàn thành tính toán.

> **Nhận xét:** Thay đổi này cho phép đo đạc hiệu năng chính xác tới từng chu kỳ clock ở mức phần cứng, loại bỏ hoàn toàn sai số do hệ điều hành (OS jitter) khi đo bằng phần mềm.

---

### 3.3 FSM & Control Plane

#### 3.3.1 IP Linear — Máy Trạng Thái (`linear.sv`)

**Phiên bản Baseline (v1.0) — 5 trạng thái:**

```
ST_IDLE (000) → ST_LOAD_K (001) → ST_PRELOAD_MAC (010) → ST_COMPUTE (011) → ST_DONE (100)
```

**Phiên bản Optimized (v5.0) — 4 trạng thái:**

```
ST_IDLE (000) → ST_LOAD_K (001) → ST_COMPUTE (011) → ST_DONE (100)
```

Bảng so sánh chi tiết:

| Trạng thái | Encoding | Baseline | Optimized | Phân tích |
|:---|:---:|:---:|:---:|:---|
| `ST_IDLE` | `3'b000` | ✅ | ✅ | Chờ `i_start_attn_score`. Không thay đổi. |
| `ST_LOAD_K` | `3'b001` | Đọc $K$ từ AXI-Stream → Ghi vào **BRAM** (`k_ram`) | Đọc $K$ từ AXI-Stream → Nạp **trực tiếp vào PE** | **⚡ Thay đổi chức năng.** Bỏ bước trung gian qua BRAM. |
| `ST_PRELOAD_MAC` | `3'b010` | Quét BRAM `k_ram` → Nạp vào thanh ghi PE (pipeline 2 cycle bù latency BRAM) | ❌ **Bị loại bỏ** | **⚡ Xóa trạng thái.** Chức năng được gộp vào `ST_LOAD_K`. |
| `ST_COMPUTE` | `3'b011` | Nhận $Q$ qua AXI-Stream, thực hiện MAC song song, xuất kết quả | Không thay đổi | — |
| `ST_DONE` | `3'b100` | Phát xung `o_attn_score_done`, quay về `ST_IDLE` | Không thay đổi | — |

#### 3.3.2 IP Softmax — Máy Trạng Thái (`softmax.sv`)

**Phiên bản Baseline (v1.0) — 8 trạng thái:**

```
ST_IDLE → ST_LOAD_ROW → ST_FIND_MAX → ST_EXP_SUM → ST_DIV_ISSUE → ST_DIV_DRAIN → ST_SERIALIZE → ST_DONE
```

**Phiên bản Optimized (v5.0) — 7 trạng thái:**

```
ST_IDLE → ST_COMPUTE_WAIT → ST_FIND_MAX → ST_EXP_SUM → ST_RECIP_PREP → ST_DIV_ISSUE → ST_DONE
```

Bảng so sánh chi tiết:

| Encoding | Baseline (v1.0) | Optimized (v5.0) | Phân tích |
|:---:|:---|:---|:---|
| `4'd0` | `ST_IDLE` | `ST_IDLE` | Không đổi |
| `4'd1` | `ST_LOAD_ROW` — Đọc 1 row ($D\_HEAD$ phần tử) từ AXI-Stream vào `s_row_buf` | `ST_COMPUTE_WAIT` — Chờ dữ liệu sẵn sàng trong Ping-Pong buffer | **⚡ Đổi tên và cơ chế.** Dữ liệu được nạp vào buffer theo cơ chế Ping-Pong thay vì đọc tuần tự. |
| `4'd2` | `ST_FIND_MAX` | `ST_FIND_MAX` | Không đổi |
| `4'd3` | `ST_EXP_SUM` | `ST_EXP_SUM` | Không đổi |
| `4'd4` | `ST_DIV_ISSUE` — Đẩy từng cặp (dividend, divisor) vào module `reciprocal_divider` | `ST_RECIP_PREP` — Tính Priority Encoder, tra `recip_rom`, chuẩn bị hệ số nhân/dịch | **⚡ Đổi tên và cơ chế.** Tách riêng bước chuẩn bị (1 lần/row) khỏi bước tính toán. |
| `4'd5` | `ST_DIV_DRAIN` — Chờ pipeline `reciprocal_divider` trả hết kết quả | `ST_DIV_ISSUE` — Thực hiện phép nhân song song 8 đường và xuất kết quả | **⚡ Đổi tên và cơ chế.** Throughput tăng ×8 nhờ kiến trúc song song. |
| `4'd6` | `ST_SERIALIZE` — Xuất kết quả row ra AXI-Stream Master | ❌ **Bị loại bỏ** | **⚡ Gộp vào trạng thái khác.** |
| `4'd7` | `ST_DONE` | `ST_DONE` | Không đổi |

---

### 3.4 Datapath & Pipeline

#### 3.4.1 IP Linear — Đường Dẫn Dữ Liệu

| Thành phần | Baseline (v1.0) | Optimized (v5.0) |
|:---|:---|:---|
| **Lưu trữ ma trận K** | BRAM nội bộ `k_ram` (instance `u_k_ram`), kích thước `K_DEPTH = D_HEAD × D_MODEL` | Trực tiếp vào thanh ghi PE (Direct-to-PE Preload) |
| **Nạp K vào PE** | 2 bước tuần tự: AXI-Stream → BRAM → PE (pipeline 2 cycle bù latency BRAM) | 1 bước duy nhất: AXI-Stream → PE (qua `matmul_preload_en/j/k/tile_sel`) |
| **Localparam `K_DEPTH`** | `D_HEAD * D_MODEL` (kích thước bộ nhớ BRAM) | Không còn tồn tại (không cần BRAM cho K) |
| **Q Row Buffer** | Ping-Pong double-buffering (`q_row_buf_0`, `q_row_buf_1`) | Không thay đổi |
| **MAC Engine** | Module `matmul_ip` (`u_matmul_ip`), $N\_PE$ PE song song, xóa accumulator bằng `matmul_acc_clear` | Không thay đổi |
| **Result Buffer** | Ping-Pong double-buffering (`result_buffer_0`, `result_buffer_1`) | Không thay đổi |
| **PE Masking** | Điều kiện `(col_base + p) < D_HEAD` ngăn ghi kết quả rác khi $D\_HEAD \mod N\_PE \neq 0$ | Không thay đổi |
| **Output Scaling** | Arithmetic right shift: `$signed(val) >>> SQRT_SHIFT` | Không thay đổi |
| **Rounding (pe_unit)** | Round-half-to-even (Banker's Rounding) | Không thay đổi |

> **Nhận xét:** Thay đổi lớn nhất ở IP Linear là **loại bỏ hoàn toàn BRAM trung gian** cho ma trận $K$, chuyển sang nạp trực tiếp vào thanh ghi PE. Điều này mang lại hai lợi ích: (1) Giảm latency khởi tạo do bỏ trạng thái `ST_PRELOAD_MAC`; (2) Tiết kiệm tài nguyên BRAM trên FPGA.

#### 3.4.2 IP Softmax — Đường Dẫn Dữ Liệu

| Thành phần | Baseline (v1.0) | Optimized (v5.0) |
|:---|:---|:---|
| **S Row Buffer (đầu vào)** | Buffer đơn `s_row_buf` | Ping-Pong `s_row_buf_0`, `s_row_buf_1` |
| **Find Max** | Quét tuần tự `D_HEAD` phần tử, so sánh signed | Không thay đổi |
| **Exp Lookup** | Tra ROM `exp_rom`: `addr = (-z_val) & 0x7FF` | Không thay đổi |
| **Sum Accumulation** | `sum_acc += exp_rom_data`, chốt vào `sum_latched`, mask `SUM_WIDTH` bit | Không thay đổi |
| **Dividend Formation** | `exp_row_buf[i] << 15` (chuyển sang định dạng Q1.15) | Không thay đổi |
| **Output Row Buffer** | Buffer đơn `out_row_buf` | Ping-Pong `out_row_buf_0`, `out_row_buf_1` |
| **Saturation** | Clamp kết quả chia về `MAX_Q15 = {EXP_WIDTH{1'b1}}` | Không thay đổi |

> **Nhận xét:** Phiên bản Optimized nâng cấp cả buffer đầu vào và đầu ra lên cơ chế **Ping-Pong double-buffering**, cho phép overlap giữa việc nạp row mới và xử lý row hiện tại, tăng thông lượng tổng thể.

---

### 3.5 Softmax Division Method

Đây là thay đổi kiến trúc lớn nhất trong toàn bộ hệ thống. Phương pháp chia được tái thiết kế hoàn toàn về mặt cấu trúc, trong khi vẫn giữ nguyên thuật toán cốt lõi (bit-true).

#### 3.5.1 So sánh kiến trúc tổng quan

| Tiêu chí | Baseline (v1.0) | Optimized (v5.0) |
|:---|:---|:---|
| **Phương pháp** | Reciprocal-LUT Multiply + Shift | Reciprocal-LUT Multiply + Shift (giống hệt) |
| **Cấu trúc module** | Module con riêng biệt `reciprocal_divider` | **Inline** trực tiếp trong `softmax.sv` |
| **Mức độ song song** | Tuần tự — 1 phép chia/chu kỳ qua pipeline | **8 đường song song** (`NUM_WAYS = 8`) |
| **Xilinx div_gen IP** | Không sử dụng | Không sử dụng |
| **Sub-module instances** | `exp_rom` + `reciprocal_divider` (chứa `recip_rom` bên trong) | `exp_rom` + `recip_rom` (cả hai trực tiếp, không qua module trung gian) |

#### 3.5.2 Luồng xử lý phép chia

**Baseline (v1.0) — Pipeline tuần tự:**

```
[exp_row_buf] ──→ reciprocal_divider (pipeline, 1 element/cycle)
                       │
                       ├─ Stage 0: Priority Encoder → msb_pos
                       ├─ Stage 1: Lookup recip_rom → recip_data
                       └─ Stage 2: Multiply + Shift → Q1.15 result
                                    │
                              [out_row_buf]
```

- Trạng thái FSM: `ST_DIV_ISSUE` (đẩy vào pipeline) → `ST_DIV_DRAIN` (chờ kết quả).
- Throughput: **1 phần tử / chu kỳ**.

**Optimized (v5.0) — Song song 8 đường:**

```
[exp_row_buf] ──→ 8x Parallel Multipliers (inline, 8 elements/cycle)
                       │
                       ├─ Precompute (1 lần/row): msb_pos_c → recip_addr_c → recip_rom → precomp_recip, precomp_shift
                       └─ Issue (mỗi cycle): div_dividend[0..7] × precomp_recip → >> precomp_shift → clamp → Q1.15
                                    │
                              [out_row_buf]
```

- Trạng thái FSM: `ST_RECIP_PREP` (chuẩn bị hệ số, 1 lần/row) → `ST_DIV_ISSUE` (nhân song song 8 phần tử/cycle).
- Throughput: **8 phần tử / chu kỳ**.

#### 3.5.3 Thuật toán cốt lõi (giữ nguyên bit-true)

Cả hai phiên bản đều thực hiện cùng một chuỗi phép tính:

$$\text{result} = \text{clamp}\left(\frac{\text{dividend} \times \text{recip\_lut}[\text{addr}]}{2^{\text{OUT\_W} + \text{msb\_pos}}}\right)$$

Trong đó:
- `msb_pos` = vị trí bit 1 cao nhất (MSB) của `sum_latched` (Priority Encoder)
- `addr` = `ADDR_W` bit mantissa ngay dưới MSB
- `recip_lut[addr]` = giá trị nghịch đảo tra từ ROM ($Q0.\text{OUT\_W}$ unsigned)
- Clamp về phạm vi $[0, 2^{\text{EXP\_WIDTH}} - 1]$

> **Nhận xét:** Phiên bản Optimized khai thác một tính chất quan trọng: trong hàm Softmax, **tất cả phần tử trên cùng một hàng chia cho cùng một mẫu số** (`sum_latched`). Do đó, phép tra ROM và tính `msb_pos` chỉ cần thực hiện **1 lần duy nhất** cho mỗi row (ở `ST_RECIP_PREP`), sau đó tái sử dụng kết quả cho toàn bộ $D\_HEAD$ phần tử trong row — cho phép song song hóa phép nhân lên 8 đường mà không tăng số lượng ROM.

---

## 4. Tổng Kết Các Thay Đổi

### 4.1 Bảng tóm tắt

| # | Thay đổi | IP | Loại | Tác động |
|:---:|:---|:---:|:---:|:---|
| 1 | Thêm Hardware Cycle Counter (`o_linear_cycles`, `o_softmax_cycles`) | Cả hai | Thêm mới | Cho phép đo đạc hiệu năng chính xác tới từng chu kỳ clock |
| 2 | Ánh xạ `slv_reg2` thành Cycle Counter Read-Only | Cả hai | Sửa đổi | SoC đọc trực tiếp số cycle qua AXI-Lite offset `0x08` |
| 3 | Loại bỏ trạng thái `ST_PRELOAD_MAC` | Linear | Tối ưu | Giảm latency khởi tạo, tiết kiệm BRAM |
| 4 | Nạp K trực tiếp vào PE (Direct-to-PE Preload) | Linear | Tái cấu trúc | Loại bỏ BRAM trung gian `k_ram` |
| 5 | Tái cấu trúc FSM Softmax (8 → 7 trạng thái) | Softmax | Tái cấu trúc | Gộp `ST_SERIALIZE`, tách `ST_RECIP_PREP` |
| 6 | Inline 8-way Parallel Division | Softmax | Tối ưu | Tăng throughput phép chia ×8 |
| 7 | Nâng cấp Ping-Pong Buffer cho Softmax | Softmax | Tối ưu | Overlap nạp/xử lý row, tăng thông lượng pipeline |

### 4.2 Tính tương thích

- **Tương thích ngược về giao diện ngoài:** ✅ — Tất cả các cổng AXI-Lite và AXI-Stream ở Top-level đều không thay đổi. Block Design hiện có không cần chỉnh sửa kết nối.
- **Tương thích ngược về thanh ghi:** ⚠️ — `slv_reg2` (Offset `0x08`) thay đổi chức năng từ Scratchpad sang Cycle Counter. Phần mềm đọc thanh ghi này cần cập nhật cách diễn giải giá trị.
- **Tương thích bit-true:** ✅ — Kết quả tính toán (Attention Score và Softmax Weights Q1.15) giữ nguyên bit-true giữa hai phiên bản. Mọi thay đổi chỉ ảnh hưởng tới cấu trúc và hiệu năng, không ảnh hưởng tới tính đúng đắn chức năng.

---

*Hết báo cáo.*
