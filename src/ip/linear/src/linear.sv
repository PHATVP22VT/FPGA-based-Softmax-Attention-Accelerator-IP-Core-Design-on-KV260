`timescale 1ns / 1ps
//==============================================================================
// linear.sv — Core Datapath & Control FSM
//
// Computes S = Q × Kᵀ  (Scaled Dot-Product Attention Score)
//   K : [D_HEAD × D_MODEL]  — streamed directly into PE weight registers
//                              (preload-on-the-fly, no intermediate BRAM)
//   Q : [SEQ_LEN × D_MODEL] — streamed in via AXI-Stream
//   S : [SEQ_LEN × D_HEAD]  — serialized out via AXI-Stream
//
// Key design decisions:
//   - Weight-Stationary: K loaded once per inference call
//   - Preload-on-the-fly: K data from AXI-Stream goes directly into PE
//     weight registers during ST_LOAD_K, eliminating k_ram BRAM
//   - Q Ping-Pong Buffer: Two SEPARATE 1D Q row buffers allow the Loader
//     and Compute to run simultaneously on different banks. Uses explicit
//     q_bank_full[1:0] handshake flags (no dynamic-index writes).
//   - Result Ping-Pong with result_pending handshake: Compute writes into
//     one result bank while Serializer reads from the other. Uses
//     result_pending[1:0] flags to enable true overlap (unlike the
//     !serialize_active guard which serializes Compute and Serialize).
//   - All arrays are 1D (Vivado-safe). All bank selection uses explicit
//     if-else with literal indices. Intermediate wires (current_q_val,
//     current_result_val) are used at module ports to avoid 2D array
//     expressions that crash Vivado's xelab elaborator.
//   - tdest removed (Option C): FSM alone distinguishes K vs Q phase
//==============================================================================

module linear #(
    parameter int D_MODEL    = 64,
    parameter int SEQ_LEN    = 64,
    parameter int DATA_WIDTH = 16,
    parameter int N_PE       = 64,
    parameter int D_HEAD     = 64
)(
    input  logic        iclk,
    input  logic        irst_n,

    // Control from AXI-Lite slave
    input  logic        i_start_attn_score,
    output logic        o_attn_score_done,
    output logic        o_busy,
    output logic [31:0] o_linear_cycles, // Hardware cycle counter


    // AXI-Stream Slave (DMA → IP): K first, then Q
    input  logic [31:0] i_s_axis_tdata,
    input  logic        i_s_axis_tvalid,
    input  logic        i_s_axis_tlast,
    output logic        o_s_axis_tready,

    // AXI-Stream Master (IP → DMA): attention scores
    output logic [31:0] o_m_axis_tdata,
    output logic        o_m_axis_tvalid,
    output logic        o_m_axis_tlast,
    input  logic        i_m_axis_tready
);

    //--------------------------------------------------------------------------
    // Local parameters
    //--------------------------------------------------------------------------
    localparam int SQRT_SHIFT = $clog2(D_MODEL) / 2;   // approx divide by sqrt(D_MODEL)

    // N_TILES = ceil(D_HEAD / N_PE), elaboration-time integer division
    localparam int N_TILES = (D_HEAD + N_PE - 1) / N_PE;

    // Counter widths — guard against clog2(1)=0
    localparam int J_W = (D_HEAD  > 1) ? $clog2(D_HEAD)  : 1;
    localparam int K_W = (D_MODEL > 1) ? $clog2(D_MODEL) : 1;
    localparam int P_W = (N_PE    > 1) ? $clog2(N_PE)    : 1;
    localparam int S_W = (SEQ_LEN > 1) ? $clog2(SEQ_LEN) : 1;
    localparam int T_W = (N_TILES > 1) ? $clog2(N_TILES) : 1;

    //--------------------------------------------------------------------------
    // Synthesis-time validity assertions
    //--------------------------------------------------------------------------
    // synthesis translate_off
    initial begin
        assert (N_PE <= D_HEAD)
            else $fatal(1, "[linear] N_PE=%0d must be <= D_HEAD=%0d", N_PE, D_HEAD);
        assert (D_MODEL >= D_HEAD)
            else $fatal(1, "[linear] D_MODEL=%0d >= D_HEAD=%0d required for ping-pong safety", D_MODEL, D_HEAD);
        assert (N_PE * N_TILES >= D_HEAD)
            else $fatal(1, "[linear] N_TILES calculation error: N_PE=%0d * N_TILES=%0d < D_HEAD=%0d", N_PE, N_TILES, D_HEAD);
    end
    // synthesis translate_on

    //--------------------------------------------------------------------------
    // FSM — 4 states
    //--------------------------------------------------------------------------
    typedef enum logic [2:0] {
        ST_IDLE    = 3'b000,
        ST_LOAD_K  = 3'b001,
        ST_COMPUTE = 3'b011,
        ST_DONE    = 3'b100
    } state_t;
    state_t state;

    //--------------------------------------------------------------------------
    // Internal signals
    //--------------------------------------------------------------------------

    // K load — direct-to-PE preload counters
    logic            k_load_done;
    logic            load_k_beat_valid;
    logic [P_W-1:0]  load_row_j;
    logic [K_W-1:0]  load_col_k;
    logic [T_W-1:0]  load_tile_idx;

    // matmul_ip control
    logic            matmul_preload_en;
    logic [P_W-1:0]  matmul_preload_j;
    logic [K_W-1:0]  matmul_preload_k;
    logic [T_W-1:0]  matmul_preload_tile_sel;
    logic            matmul_data_valid;
    logic            matmul_acc_clear;
    logic [K_W-1:0]  matmul_k_index;
    logic [T_W-1:0]  matmul_tile_sel;

    // matmul_ip output
    logic                                   matmul_result_valid;
    logic signed [N_PE-1:0][DATA_WIDTH-1:0] matmul_result;
    logic [J_W-1:0]                         matmul_result_col_base;

    //----------------------------------------------------------------------
    // Q Ping-Pong Buffer — TWO SEPARATE 1D ARRAYS
    //
    // Vivado's xelab crashes on 2D unpacked arrays with dual dynamic
    // indices at module port connections. Using two independent 1D arrays
    // plus an explicit MUX wire (current_q_val) eliminates this entirely.
    //
    //   q_wr_bank / q_rd_bank  : select which bank each process uses
    //   q_bank_full[0:1]       : handshake between Loader and Compute
    //     - Loader SETS   q_bank_full[wr_bank] when row fully loaded
    //     - Compute CLEARS q_bank_full[rd_bank] when tile loop finishes
    //   q_all_loaded           : set when DMA sends tlast on Q data
    //----------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] q_row_buf_0 [0:D_MODEL-1];
    logic signed [DATA_WIDTH-1:0] q_row_buf_1 [0:D_MODEL-1];
    logic            q_wr_bank;
    logic            q_rd_bank;
    logic [1:0]      q_bank_full;
    logic            q_all_loaded;
    logic [K_W-1:0]  q_col_k;

    // Intermediate wire — explicit MUX, safe at module port
    logic signed [DATA_WIDTH-1:0] current_q_val;

    // Compute control — replaces old compute_phase
    logic            q_compute_active;
    logic [T_W-1:0]  tile_idx;
    logic [K_W-1:0]  tile_k_cnt;
    logic [T_W-1:0]  tile_result_cnt;

    //----------------------------------------------------------------------
    // Result Ping-Pong — TWO SEPARATE 1D ARRAYS
    //
    // result_pending[1:0] handshake enables TRUE overlap between Compute
    // and Serializer (unlike !serialize_active guard which forces them
    // to run sequentially, wasting the Q Ping-Pong speedup).
    //
    //   result_pending[X] : SET by row_complete_pulse (Compute filled bank X)
    //                       CLEAR by ser_finish_pulse (Serializer drained bank X)
    //   buf_write_sel     : Compute writes here; flips on row_complete_pulse
    //   ser_rd_bank       : latched when Serializer starts, stable throughout
    //----------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] result_buffer_0 [0:D_HEAD-1];
    logic signed [DATA_WIDTH-1:0] result_buffer_1 [0:D_HEAD-1];
    logic        buf_write_sel;
    logic [1:0]  result_pending;

    // Intermediate wire for M_AXIS output
    logic signed [DATA_WIDTH-1:0] current_result_val;

    // Serializer
    logic            serialize_active;
    logic            ser_rd_bank;       // latched at start, stable during output
    logic [J_W-1:0]  ser_col_j;
    logic [S_W-1:0]  ser_row_i;

    // Done detection
    logic last_row_last_col;
    logic compute_done;

    //----------------------------------------------------------------------
    // Combinational helpers (all explicit ternary, no dynamic-index writes)
    //----------------------------------------------------------------------

    // Q bank MUX — safe at module port (simple wire)
    assign current_q_val = (q_rd_bank == 1'b0) ? q_row_buf_0[tile_k_cnt]
                                                : q_row_buf_1[tile_k_cnt];

    // Result bank MUX — safe at output path (simple wire)
    assign current_result_val = (ser_rd_bank == 1'b0) ? result_buffer_0[ser_col_j]
                                                       : result_buffer_1[ser_col_j];

    // Q transfer (Loader accepted a beat from AXI-Stream)
    logic q_transfer;
    assign q_transfer = (state == ST_COMPUTE) & i_s_axis_tvalid & o_s_axis_tready;

    // Q row loaded pulse (last beat of a Q row)
    logic q_row_loaded;
    assign q_row_loaded = q_transfer & (i_s_axis_tlast | (q_col_k == K_W'(D_MODEL - 1)));

    // Compute row done pulse (last MAC cycle of last tile for one Q row)
    logic compute_row_done;
    assign compute_row_done = q_compute_active
                            & (tile_k_cnt == K_W'(D_MODEL - 1))
                            & (tile_idx == T_W'(N_TILES - 1));

    // Tile-loop MAC enable
    logic tile_mac_en;
    assign tile_mac_en = (state == ST_COMPUTE) & q_compute_active;

    // row_complete_pulse — from matmul pipeline, marks result buffer full
    logic row_complete_pulse;
    assign row_complete_pulse = matmul_result_valid
                              & (tile_result_cnt == T_W'(N_TILES - 1));

    // Serializer finish pulse — last column of current row accepted
    logic ser_finish_pulse;
    assign ser_finish_pulse = serialize_active & i_m_axis_tready
                            & (ser_col_j == J_W'(D_HEAD - 1));

    // Compute start guards (explicit ternary reads, no dynamic-index risk)
    logic q_rd_ready;
    logic write_buf_avail;
    logic compute_wr_bank; // Tracks which bank the NEXT row will write to
    assign q_rd_ready     = (q_rd_bank == 1'b0)     ? q_bank_full[0]     : q_bank_full[1];
    assign write_buf_avail = (compute_wr_bank == 1'b0) ? ~result_pending[0] : ~result_pending[1];

    // S_AXIS tready (explicit ternary for q_wr_bank_full)
    logic q_wr_bank_full;
    assign q_wr_bank_full = (q_wr_bank == 1'b0) ? q_bank_full[0] : q_bank_full[1];
    assign o_s_axis_tready = (state == ST_LOAD_K)  ? 1'b1 :
                             (state == ST_COMPUTE)  ? (~q_wr_bank_full & ~q_all_loaded) :
                                                       1'b0;

    // K load
    assign load_k_beat_valid = i_s_axis_tvalid & o_s_axis_tready & (state == ST_LOAD_K);
    assign k_load_done       = load_k_beat_valid & i_s_axis_tlast;

    // matmul_ip preload interface
    assign matmul_preload_en       = load_k_beat_valid;
    assign matmul_preload_j        = load_row_j;
    assign matmul_preload_k        = load_col_k;
    assign matmul_preload_tile_sel = load_tile_idx;

    // matmul_ip compute interface
    assign matmul_data_valid = tile_mac_en;
    assign matmul_k_index    = tile_k_cnt;
    assign matmul_acc_clear  = tile_mac_en & (tile_k_cnt == '0);
    assign matmul_tile_sel   = tile_idx;

    // Status outputs
    assign o_busy            = (state != ST_IDLE);
    assign o_attn_score_done = (state == ST_DONE);

    // M_AXIS output
    assign o_m_axis_tvalid = serialize_active;
    assign o_m_axis_tlast  = serialize_active
                           & (ser_col_j == J_W'(D_HEAD - 1))
                           & (ser_row_i == S_W'(SEQ_LEN - 1));
    assign o_m_axis_tdata  = 32'(($signed(current_result_val)) >>> SQRT_SHIFT);

    // ST_COMPUTE exit condition
    assign last_row_last_col = o_m_axis_tvalid
                             & o_m_axis_tlast
                             & i_m_axis_tready
                             & (ser_row_i == S_W'(SEQ_LEN - 1));

    //--------------------------------------------------------------------------
    // matmul_ip instantiation
    //--------------------------------------------------------------------------
    matmul_ip #(
        .N_COLS    (D_HEAD),
        .D_MODEL   (D_MODEL),
        .N_PE      (N_PE),
        .DATA_WIDTH(DATA_WIDTH),
        .N_TILES   (N_TILES)
    ) u_matmul_ip (
        .i_clk               (iclk),
        .i_reset_n           (irst_n),

        .i_preload_en        (matmul_preload_en),
        .i_preload_j         (matmul_preload_j),
        .i_preload_k         (matmul_preload_k),
        .i_preload_data      (i_s_axis_tdata[DATA_WIDTH-1:0]),
        .i_preload_tile_sel  (matmul_preload_tile_sel),

        .i_data_valid        (matmul_data_valid),
        .i_acc_clear         (matmul_acc_clear),
        .i_k_index           (matmul_k_index),
        .i_a_data            (current_q_val),       // SAFE: intermediate wire, no 2D at port
        .i_tile_sel          (matmul_tile_sel),
        .i_col_base          (J_W'(32'(tile_idx) * 32'(N_PE))),

        .o_result_valid      (matmul_result_valid),
        .o_result            (matmul_result),
        .o_result_col_base   (matmul_result_col_base)
    );

    //==========================================================================
    // FSM — state transitions only (unchanged)
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            state <= ST_IDLE;
        end else begin
            case (state)
                ST_IDLE:
                    if (i_start_attn_score)
                        state <= ST_LOAD_K;

                ST_LOAD_K:
                    if (k_load_done)
                        state <= ST_COMPUTE;

                ST_COMPUTE:
                    if (compute_done)
                        state <= ST_DONE;

                ST_DONE:
                    state <= ST_IDLE;

                default:
                    state <= ST_IDLE;
            endcase
        end
    end

    //==========================================================================
    // Direct-to-PE preload counters (ST_LOAD_K) — unchanged
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            load_row_j    <= '0;
            load_col_k    <= '0;
            load_tile_idx <= '0;
        end else begin
            if (state == ST_IDLE && i_start_attn_score) begin
                load_row_j    <= '0;
                load_col_k    <= '0;
                load_tile_idx <= '0;
            end else if (load_k_beat_valid) begin
                if (load_col_k == K_W'(D_MODEL - 1)) begin
                    load_col_k <= '0;
                    if (load_row_j == P_W'(N_PE - 1)) begin
                        load_row_j    <= '0;
                        load_tile_idx <= load_tile_idx + 1;
                    end else begin
                        load_row_j <= load_row_j + 1;
                    end
                end else begin
                    load_col_k <= load_col_k + 1;
                end
            end
        end
    end

    //==========================================================================
    // Q Ping-Pong Loader (ST_COMPUTE)
    //
    // Accepts Q beats from AXI-Stream and writes them into the correct bank
    // using EXPLICIT if-else (no dynamic array index). Runs INDEPENDENTLY
    // of the Compute process.
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            q_col_k      <= '0;
            q_wr_bank    <= 1'b0;
            q_all_loaded <= 1'b0;
        end else if (state != ST_COMPUTE) begin
            q_col_k      <= '0;
            q_wr_bank    <= 1'b0;
            q_all_loaded <= 1'b0;
        end else if (q_transfer) begin
            // Write into correct bank — EXPLICIT, no dynamic index
            if (q_wr_bank == 1'b0)
                q_row_buf_0[q_col_k] <= $signed(i_s_axis_tdata[DATA_WIDTH-1:0]);
            else
                q_row_buf_1[q_col_k] <= $signed(i_s_axis_tdata[DATA_WIDTH-1:0]);

            if (i_s_axis_tlast || (q_col_k == K_W'(D_MODEL - 1))) begin
                q_col_k   <= '0;
                q_wr_bank <= ~q_wr_bank;
                if (i_s_axis_tlast)
                    q_all_loaded <= 1'b1;
            end else begin
                q_col_k <= q_col_k + 1;
            end
        end
    end

    //==========================================================================
    // Q Bank Full Flags — explicit per-bank (no dynamic-index write)
    //
    // SET by Loader:  q_bank_full[X] = 1 when q_row_loaded & q_wr_bank==X
    // CLEAR by Compute: q_bank_full[X] = 0 when compute_row_done & q_rd_bank==X
    //
    // After priming, q_wr_bank and q_rd_bank always point to different banks,
    // so SET and CLEAR never target the same bank simultaneously.
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            q_bank_full <= 2'b00;
        end else if (state != ST_COMPUTE) begin
            q_bank_full <= 2'b00;
        end else begin
            // Bank 0
            if (q_row_loaded && q_wr_bank == 1'b0)
                q_bank_full[0] <= 1'b1;
            else if (compute_row_done && q_rd_bank == 1'b0)
                q_bank_full[0] <= 1'b0;
            // Bank 1
            if (q_row_loaded && q_wr_bank == 1'b1)
                q_bank_full[1] <= 1'b1;
            else if (compute_row_done && q_rd_bank == 1'b1)
                q_bank_full[1] <= 1'b0;
        end
    end

    //==========================================================================
    // Compute Process — tile-loop with Ping-Pong read bank
    //
    // Starts when BOTH conditions are met:
    //   1. q_rd_ready     : Q data available in read bank
    //   2. write_buf_avail: result write buffer not pending serialization
    //
    // The write_buf_avail guard (using result_pending) is the key difference
    // from the failed !serialize_active approach. It allows Compute to run
    // in parallel with Serializer as long as they use different result banks,
    // achieving true 2x throughput improvement.
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            q_compute_active <= 1'b0;
            q_rd_bank        <= 1'b0;
            compute_wr_bank  <= 1'b0;
            tile_idx         <= '0;
            tile_k_cnt       <= '0;
        end else if (state != ST_COMPUTE) begin
            q_compute_active <= 1'b0;
            q_rd_bank        <= 1'b0;
            compute_wr_bank  <= 1'b0;
            tile_idx         <= '0;
            tile_k_cnt       <= '0;
        end else begin
            if (!q_compute_active) begin
                // Start when Q bank ready AND result write buffer available
                if (q_rd_ready && write_buf_avail) begin
                    q_compute_active <= 1'b1;
                    tile_idx         <= '0;
                    tile_k_cnt       <= '0;
                end
            end else begin
                if (tile_k_cnt == K_W'(D_MODEL - 1)) begin
                    tile_k_cnt <= '0;
                    if (tile_idx == T_W'(N_TILES - 1)) begin
                        // All tiles for this row done
                        q_compute_active <= 1'b0;
                        q_rd_bank        <= ~q_rd_bank;
                        compute_wr_bank  <= ~compute_wr_bank;
                        tile_idx         <= '0;
                    end else begin
                        tile_idx <= tile_idx + 1;
                    end
                end else begin
                    tile_k_cnt <= tile_k_cnt + 1;
                end
            end
        end
    end

    //==========================================================================
    // tile_result_cnt — counts matmul_result_valid pulses per row (unchanged)
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            tile_result_cnt <= '0;
        end else begin
            if (matmul_result_valid) begin
                tile_result_cnt <= (tile_result_cnt == T_W'(N_TILES - 1))
                                  ? '0
                                  : tile_result_cnt + 1;
            end
        end
    end

    //==========================================================================
    // Result Buffer Write — explicit per-bank (no dynamic-index write)
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            buf_write_sel <= 1'b0;
            for (int i = 0; i < D_HEAD; i++) begin
                result_buffer_0[i] <= '0;
                result_buffer_1[i] <= '0;
            end
        end else if (state != ST_COMPUTE) begin
            buf_write_sel <= 1'b0;
        end else begin
            if (matmul_result_valid) begin
                for (int p = 0; p < N_PE; p++) begin
                    if ((32'(matmul_result_col_base) + p) < D_HEAD) begin
                        if (buf_write_sel == 1'b0)
                            result_buffer_0[32'(matmul_result_col_base) + p] <= matmul_result[p];
                        else
                            result_buffer_1[32'(matmul_result_col_base) + p] <= matmul_result[p];
                    end
                end
                if (row_complete_pulse)
                    buf_write_sel <= ~buf_write_sel;
            end
        end
    end

    //==========================================================================
    // Result Pending Flags — explicit per-bank (no dynamic-index write)
    //
    // SET by Compute:    result_pending[X] = 1 on row_complete_pulse
    //                    when buf_write_sel == X
    // CLEAR by Serializer: result_pending[X] = 0 on ser_finish_pulse
    //                    when ser_rd_bank == X
    //
    // Mutual exclusion: Compute writes to buf_write_sel while Serializer
    // reads from ser_rd_bank. These are always different banks (guaranteed
    // by the write_buf_avail guard), so SET and CLEAR never target the
    // same bank simultaneously.
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            result_pending <= 2'b00;
        end else if (state != ST_COMPUTE) begin
            result_pending <= 2'b00;
        end else begin
            // Bank 0
            if (row_complete_pulse && buf_write_sel == 1'b0)
                result_pending[0] <= 1'b1;
            else if (ser_finish_pulse && ser_rd_bank == 1'b0)
                result_pending[0] <= 1'b0;
            // Bank 1
            if (row_complete_pulse && buf_write_sel == 1'b1)
                result_pending[1] <= 1'b1;
            else if (ser_finish_pulse && ser_rd_bank == 1'b1)
                result_pending[1] <= 1'b0;
        end
    end

    //==========================================================================
    // Serializer — reads from latched ser_rd_bank (stable during output)
    //
    // When idle: checks result_pending to find next row to serialize.
    // ser_rd_bank is latched at start and never changes while active,
    // ensuring current_result_val is stable even if buf_write_sel flips.
    //
    // Backpressure: if i_m_axis_tready=0, counters freeze (valid stays
    // high, data stable) — AXI-Stream compliant.
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            serialize_active <= 1'b0;
            ser_rd_bank      <= 1'b0;
            ser_col_j        <= '0;
            ser_row_i        <= '0;
        end else begin
            // Clear row counter on return to IDLE
            if (state == ST_DONE)
                ser_row_i <= '0;

            if (!serialize_active) begin
                // Pick up pending result — priority order matches buffer
                // fill sequence (alternating 0, 1, 0, 1, ...)
                if (result_pending[0]) begin
                    ser_rd_bank      <= 1'b0;
                    serialize_active <= 1'b1;
                    ser_col_j        <= '0;
                end else if (result_pending[1]) begin
                    ser_rd_bank      <= 1'b1;
                    serialize_active <= 1'b1;
                    ser_col_j        <= '0;
                end
            end else if (i_m_axis_tready) begin
                if (ser_col_j == J_W'(D_HEAD - 1)) begin
                    // Last column of this row sent
                    ser_col_j        <= '0;
                    serialize_active <= 1'b0;
                    if (ser_row_i == S_W'(SEQ_LEN - 1))
                        ser_row_i <= '0;
                    else
                        ser_row_i <= ser_row_i + 1;
                end else begin
                    ser_col_j <= ser_col_j + 1;
                end
            end
        end
    end

    //==========================================================================
    // compute_done — registered for clean FSM transition
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n)
            compute_done <= 1'b0;
        else
            compute_done <= last_row_last_col;
    end

    //==========================================================================
    // Hardware Cycle Counter (Synthesizable)
    //==========================================================================
    logic [31:0] linear_cycles;
    assign o_linear_cycles = linear_cycles;
    
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            linear_cycles <= 0;
        end else begin
            if (i_start_attn_score) begin
                linear_cycles <= 0;
            end else if (o_busy) begin
                linear_cycles <= linear_cycles + 1;
            end

            // synthesis translate_off
            if (o_attn_score_done) begin
                $display("[%0t] [LINEAR IP] Inference completed! Total execution time: %0d clock cycles", $time, linear_cycles);
            end
            // synthesis translate_on
        end
    end

endmodule
