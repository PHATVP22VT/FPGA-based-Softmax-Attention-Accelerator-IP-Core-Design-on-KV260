`timescale 1ns / 1ps
//==============================================================================
// matmul_ip.sv — Parallel MAC array
//
// Computes C = A · Bᵀ
//   A : [N_ROWS × D_MODEL]  — broadcast element-by-element from caller
//   B : [N_COLS × D_MODEL]  — preloaded into PE weight registers
//   C : [N_ROWS × N_COLS]   — N_PE columns output per result_valid pulse
//
// Tiling (when N_PE < N_COLS):
//   Caller manages tile loop:
//     for tile = 0..N_TILES-1:
//       assert i_col_base = tile * N_PE
//       preload B[tile*N_PE .. (tile+1)*N_PE-1][:] into PE[0..N_PE-1]
//       for i = 0..N_ROWS-1:
//         drive i_data_valid=1, i_a_data=A[i][k], k=0..D_MODEL-1
//         capture o_result on o_result_valid pulse
//
// When N_PE = N_COLS (current config): N_TILES=1, set i_col_base=0.
//==============================================================================

//------------------------------------------------------------------------------
// pe_unit — Processing Element
//
// Stores one row of B (B[j][0..D_MODEL-1]) in weight[] registers.
// MAC: acc += broadcast_x * weight[k_index]
// Output: round-to-nearest-even truncation to DATA_WIDTH bits
//------------------------------------------------------------------------------
module pe_unit #(
    parameter int D_MODEL    = 64,
    parameter int DATA_WIDTH = 16,
    parameter int N_TILES    = 1
)(
    input  logic                          i_clk,
    input  logic                          i_reset_n,

    // Preload interface
    input  logic                          i_preload_en,
    input  logic [$clog2(D_MODEL)-1:0]   i_preload_k,
    input  logic signed [DATA_WIDTH-1:0] i_preload_data,
    input  logic [$clog2(N_TILES>1?N_TILES:2)-1:0] i_preload_tile_sel,

    // Compute interface
    input  logic                          i_compute_en,
    input  logic                          i_acc_clear,
    input  logic [$clog2(D_MODEL)-1:0]   i_k_index,
    input  logic signed [DATA_WIDTH-1:0] i_broadcast_x,
    input  logic [$clog2(N_TILES>1?N_TILES:2)-1:0] i_tile_sel,

    output logic signed [DATA_WIDTH-1:0] o_result
);
    localparam int ACC_WIDTH = DATA_WIDTH * 2 + $clog2(D_MODEL);
    localparam int FRAC_BITS = DATA_WIDTH / 2;

    // weight[tile_idx][k] — N_TILES banks, selected by i_tile_sel (compute)
    // / i_preload_tile_sel (preload)
    logic signed [DATA_WIDTH-1:0] weight [0:N_TILES-1][0:D_MODEL-1];
    logic signed [ACC_WIDTH-1:0]  acc;
    logic signed [DATA_WIDTH-1:0] weight_mux;

    // Pipeline Stage 1 registers (M_REG equivalent in DSP48E2)
    // Captures multiply result for 1 cycle before feeding into accumulator.
    // This cuts the critical path: [MUX→MULTIPLY] | [ADD→ACC] instead of
    // [MUX→MULTIPLY→ADD→ACC] in a single cycle.
    logic signed [2*DATA_WIDTH-1:0] mult_reg;
    logic                            mult_valid;
    logic                            mult_acc_clear;

    // Weight preload
    always_ff @(posedge i_clk or negedge i_reset_n) begin
        if (!i_reset_n) begin
            // weights uninitialised on reset — valid after preload phase
        end else if (i_preload_en) begin
            weight[i_preload_tile_sel][i_preload_k] <= i_preload_data;
        end
    end

    // Bank-select mux — combinational, 1 stage ahead of MAC input
    assign weight_mux = weight[i_tile_sel][i_k_index];

    // Pipeline Stage 1: Multiply (M_REG)
    // Captures: product, valid flag, and acc_clear flag.
    // These delayed signals drive Stage 2 on the NEXT cycle.
    always_ff @(posedge i_clk or negedge i_reset_n) begin
        if (!i_reset_n) begin
            mult_reg       <= '0;
            mult_valid     <= 1'b0;
            mult_acc_clear <= 1'b0;
        end else begin
            mult_reg       <= $signed(i_broadcast_x) * $signed(weight_mux);
            mult_valid     <= i_compute_en;
            mult_acc_clear <= i_acc_clear;
        end
    end

    // Pipeline Stage 2: Accumulate (P_REG)
    // Uses multiply result from Stage 1 (1 cycle old).
    // mult_acc_clear is the DELAYED version of i_acc_clear, so it
    // correctly clears acc on the first beat of each new tile/row.
    always_ff @(posedge i_clk or negedge i_reset_n) begin
        if (!i_reset_n) begin
            acc <= '0;
        end else if (mult_valid) begin
            if (mult_acc_clear)
                acc <= ACC_WIDTH'(mult_reg);
            else
                acc <= acc + ACC_WIDTH'(mult_reg);
        end
    end

    // Round-to-nearest-even output truncation
    logic round_up;
    assign round_up = acc[FRAC_BITS-1] & (|acc[FRAC_BITS-2:0] | acc[FRAC_BITS]);
    assign o_result = acc[FRAC_BITS + DATA_WIDTH - 1 : FRAC_BITS]
                    + DATA_WIDTH'({(DATA_WIDTH-1)'(0), round_up});

endmodule


//==============================================================================
// matmul_ip — Top-level parallel PE array
//==============================================================================
module matmul_ip #(
    parameter int N_COLS     = 64,
    parameter int D_MODEL    = 64,
    parameter int N_PE       = 64,
    parameter int DATA_WIDTH = 16,
    parameter int N_TILES    = 1
)(
    input  logic i_clk,
    input  logic i_reset_n,

    // Preload: caller drives B[j_local][k] into PE[j_local], bank i_preload_tile_sel
    input  logic                                        i_preload_en,
    input  logic [$clog2(N_PE>1?N_PE:2)-1:0]          i_preload_j,
    input  logic [$clog2(D_MODEL)-1:0]                 i_preload_k,
    input  logic signed [DATA_WIDTH-1:0]               i_preload_data,
    input  logic [$clog2(N_TILES>1?N_TILES:2)-1:0]    i_preload_tile_sel,

    // Compute: caller broadcasts A[i][k]
    input  logic                                        i_data_valid,
    input  logic                                        i_acc_clear,
    input  logic [$clog2(D_MODEL)-1:0]                 i_k_index,
    input  logic signed [DATA_WIDTH-1:0]               i_a_data,
    input  logic [$clog2(N_TILES>1?N_TILES:2)-1:0]    i_tile_sel,

    // Tile column offset (set 0 when N_PE = N_COLS)
    input  logic [$clog2(N_COLS>1?N_COLS:2)-1:0]      i_col_base,

    // Output
    output logic                                        o_result_valid,
    output logic signed [N_PE-1:0][DATA_WIDTH-1:0]     o_result,
    output logic [$clog2(N_COLS>1?N_COLS:2)-1:0]       o_result_col_base
);

    localparam int J_W = (N_PE   > 1) ? $clog2(N_PE)   : 1;
    localparam int C_W = (N_COLS > 1) ? $clog2(N_COLS) : 1;

    // synthesis translate_off
    initial begin
        assert (N_PE >= 1 && N_PE <= N_COLS) else
            $fatal(1, "[matmul_ip] N_PE=%0d must be in [1, N_COLS=%0d]", N_PE, N_COLS);
        assert (D_MODEL >= 2) else
            $fatal(1, "[matmul_ip] D_MODEL=%0d must be >= 2", D_MODEL);
        if (N_COLS % N_PE != 0)
            $warning("[matmul_ip] N_COLS=%0d not divisible by N_PE=%0d — last %0d PEs disabled",
                     N_COLS, N_PE, N_PE - (N_COLS % N_PE));
    end
    // synthesis translate_on

    //--------------------------------------------------------------------------
    // PE array
    //--------------------------------------------------------------------------
    genvar gp;
    generate
        for (gp = 0; gp < N_PE; gp++) begin : gen_pe
            logic pe_preload_en;
            assign pe_preload_en = i_preload_en & (i_preload_j == J_W'(gp));

            pe_unit #(
                .D_MODEL   (D_MODEL),
                .DATA_WIDTH(DATA_WIDTH),
                .N_TILES   (N_TILES)
            ) u_pe (
                .i_clk              (i_clk),
                .i_reset_n          (i_reset_n),
                .i_preload_en       (pe_preload_en),
                .i_preload_k        (i_preload_k),
                .i_preload_data     (i_preload_data),
                .i_preload_tile_sel (i_preload_tile_sel),
                .i_compute_en       (i_data_valid),
                .i_acc_clear        (i_acc_clear),
                .i_k_index          (i_k_index),
                .i_broadcast_x      (i_a_data),
                .i_tile_sel         (i_tile_sel),
                .o_result           (o_result[gp])
            );
        end
    endgenerate

    //--------------------------------------------------------------------------
    // o_result_valid: pulse 1 cycle after the LAST column (i_k_index ==
    // D_MODEL-1) of a row has been accumulated.
    //
    // NOTE — this replaces the previous falling-edge detector on
    // i_data_valid ("valid_d1 & ~i_data_valid"). That detector only fires
    // once i_data_valid drops to 0, which under the documented dataflow
    // ("Q stream lien tuc, khong stall" — context.md sec 3.4) never happens
    // BETWEEN rows, only after the very last row of the very last Q
    // transfer. Net effect: with back-to-back rows and no inter-row gap,
    // the previous logic produced exactly ONE result row for the entire
    // SEQ_LEN x D_HEAD output instead of one row every D_MODEL cycles —
    // the consumer (serializer / M_AXIS) would then permanently wait for
    // S_DEPTH-1 more rows that are never produced.
    //
    // Row-boundary detection via i_k_index is correct for both continuous
    // and gapped streaming, matches the always_ff accumulate timing in
    // pe_unit exactly (acc holds the final sum for the row starting the
    // cycle AFTER i_k_index==D_MODEL-1 fires), and needs no protocol-level
    // "gap between rows" assumption.
    //--------------------------------------------------------------------------
    localparam int K_W = $clog2(D_MODEL);

    logic row_last_beat;
    logic row_last_beat_d1;
    logic row_last_beat_d2;
    assign row_last_beat = i_data_valid & (i_k_index == K_W'(D_MODEL - 1));

    // 2-stage delay chain: +1 cycle for original pe_unit latency,
    // +1 cycle for the new multiply pipeline stage (M_REG).
    // Total: result_valid fires 2 cycles after the last MAC beat,
    // which is exactly when acc holds the final accumulated sum
    // AND before the next tile's acc_clear overwrites it.
    always_ff @(posedge i_clk or negedge i_reset_n) begin
        if (!i_reset_n) begin
            row_last_beat_d1 <= 1'b0;
            row_last_beat_d2 <= 1'b0;
        end else begin
            row_last_beat_d1 <= row_last_beat;
            row_last_beat_d2 <= row_last_beat_d1;
        end
    end
    assign o_result_valid = row_last_beat_d2;

    //--------------------------------------------------------------------------
    // o_result_col_base: 2-cycle delay to align with o_result_valid
    //--------------------------------------------------------------------------
    logic [C_W-1:0] col_base_d1;
    logic [C_W-1:0] col_base_d2;
    always_ff @(posedge i_clk or negedge i_reset_n) begin
        if (!i_reset_n) begin
            col_base_d1 <= '0;
            col_base_d2 <= '0;
        end else begin
            col_base_d1 <= i_col_base;
            col_base_d2 <= col_base_d1;
        end
    end
    assign o_result_col_base = col_base_d2;

endmodule