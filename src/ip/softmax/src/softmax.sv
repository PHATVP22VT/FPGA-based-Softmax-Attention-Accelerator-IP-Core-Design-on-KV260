`timescale 1ns / 1ps
//==============================================================================
// softmax.sv - Core Datapath & Control FSM (Ping-Pong Architecture)
//==============================================================================

module softmax #(
    parameter int D_HEAD          = 64,
    parameter int SEQ_LEN         = 64,
    parameter int DATA_WIDTH      = 16,
    parameter int EXP_WIDTH       = 16,
    parameter int RECIP_ADDR_W    = 12,
    parameter int RECIP_OUT_W     = 19
)(
    input  logic        iclk,
    input  logic        irst_n,

    input  logic        i_start_softmax,
    output logic        o_softmax_done,
    output logic        o_busy,
    output logic [31:0] o_softmax_cycles,

    input  logic [31:0] i_s_axis_tdata,
    input  logic        i_s_axis_tvalid,
    input  logic        i_s_axis_tlast,
    output logic        o_s_axis_tready,

    output logic [31:0] o_m_axis_tdata,
    output logic        o_m_axis_tvalid,
    output logic        o_m_axis_tlast,
    input  logic        i_m_axis_tready
);

    localparam int SUM_WIDTH = $clog2(D_HEAD > 1 ? D_HEAD : 2) + EXP_WIDTH;
    localparam int DIVIDEND_WIDTH = EXP_WIDTH + 15;
    localparam int J_W = (D_HEAD  > 1) ? $clog2(D_HEAD)  : 1;
    localparam int S_W = (SEQ_LEN > 1) ? $clog2(SEQ_LEN) : 1;
    localparam int MSB_W_RECIP = (SUM_WIDTH > 1) ? $clog2(SUM_WIDTH) : 1;
    localparam int PROD_WIDTH = DIVIDEND_WIDTH + RECIP_OUT_W;
    localparam int SHIFT_W_RECIP = $clog2(PROD_WIDTH + 1) + 1;
    localparam logic [EXP_WIDTH-1:0] MAX_Q15 = {EXP_WIDTH{1'b1}};

    // synthesis translate_off
    initial begin
        assert (D_HEAD >= 1) else $fatal(1, "[softmax] D_HEAD=%0d must be >= 1", D_HEAD);
        assert (SEQ_LEN >= 1) else $fatal(1, "[softmax] SEQ_LEN=%0d must be >= 1", SEQ_LEN);
        assert (RECIP_OUT_W > EXP_WIDTH) else $fatal(1, "[softmax] RECIP_OUT_W=%0d must be > EXP_WIDTH=%0d", RECIP_OUT_W, EXP_WIDTH);
    end
    // synthesis translate_on

    typedef enum logic [3:0] {
        ST_IDLE         = 4'd0,
        ST_COMPUTE_WAIT = 4'd1,
        ST_FIND_MAX     = 4'd2,
        ST_EXP_SUM      = 4'd3,
        ST_RECIP_PREP   = 4'd4,
        ST_DIV_ISSUE    = 4'd5,
        ST_DONE         = 4'd7
    } state_t;
    state_t state;

    // Ping-Pong Buffers
    logic signed [DATA_WIDTH-1:0] s_row_buf_0 [0:D_HEAD-1];
    logic signed [DATA_WIDTH-1:0] s_row_buf_1 [0:D_HEAD-1];
    logic        [EXP_WIDTH-1:0]  exp_row_buf [0:D_HEAD-1];
    logic        [EXP_WIDTH-1:0]  out_row_buf_0 [0:D_HEAD-1];
    logic        [EXP_WIDTH-1:0]  out_row_buf_1 [0:D_HEAD-1];

    // Bank Control
    logic [1:0] s_bank_full;
    logic [1:0] result_pending;
    
    logic s_wr_bank;
    logic s_rd_bank;
    logic compute_out_wr_bank;
    logic ser_rd_bank;

    // Loader Process
    logic [J_W-1:0] s_col_k;
    logic           s_all_loaded;
    logic           s_transfer;
    logic           s_row_loaded;
    logic           s_wr_bank_full;

    // Compute Process
    logic           compute_row_done;
    logic           s_rd_ready;
    logic           out_buf_avail;

    // FIND_MAX
    logic signed [DATA_WIDTH-1:0] max_val;
    logic [J_W-1:0]               findmax_idx;
    logic                         findmax_done;

    // EXP_SUM
    logic [J_W-1:0]        scan_idx;
    logic                  scan_active;
    logic [J_W-1:0]        rom_idx_q;
    logic                  rom_valid_q;
    logic signed [DATA_WIDTH-1:0] z_val;
    logic [10:0]           exp_rom_addr;
    logic [EXP_WIDTH-1:0]  exp_rom_data;
    logic [SUM_WIDTH-1:0]  sum_acc;
    logic [SUM_WIDTH-1:0]  sum_latched;
    logic                  expsum_done;

    // RECIP_PREP
    logic [MSB_W_RECIP-1:0]  msb_pos_c;
    logic                    msb_found_c;
    logic [RECIP_ADDR_W-1:0] recip_addr_c;
    logic [RECIP_OUT_W-1:0]  recip_rom_data;
    logic                    recip_prep_cnt;
    logic                    recip_prep_done;
    logic [RECIP_OUT_W-1:0]  precomp_recip;
    logic [SHIFT_W_RECIP-1:0]precomp_shift;

    // DIV_ISSUE
    localparam int NUM_WAYS = 8;
    logic [J_W-1:0]            div_in_idx;
    logic [DIVIDEND_WIDTH-1:0] div_dividend [NUM_WAYS];
    logic [PROD_WIDTH-1:0]     div_prod [NUM_WAYS];
    logic [PROD_WIDTH-1:0]     div_prod_reg [NUM_WAYS];
    logic [J_W-1:0]            div_wr_idx;
    logic                      div_wr_active;
    logic                      div_issue_active;
    logic [PROD_WIDTH-1:0]     div_shifted [NUM_WAYS];
    logic [EXP_WIDTH-1:0]      div_result [NUM_WAYS];
    logic                      div_issue_done;

    // Serializer
    logic [J_W-1:0] ser_col_j;
    logic [S_W-1:0] ser_row_i;
    logic           serialize_active;
    logic           ser_finish_pulse;

    // Done detection
    logic last_row_last_col;
    logic compute_done;

    // Multiplexers
    logic [J_W-1:0] s_rd_idx;
    logic signed [DATA_WIDTH-1:0] current_s_val;
    logic [EXP_WIDTH-1:0] current_out_val;

    assign s_rd_idx = (state == ST_FIND_MAX) ? findmax_idx : scan_idx;
    assign current_s_val = (s_rd_bank == 1'b0) ? s_row_buf_0[s_rd_idx] : s_row_buf_1[s_rd_idx];
    assign current_out_val = (ser_rd_bank == 1'b0) ? out_row_buf_0[ser_col_j] : out_row_buf_1[ser_col_j];

    assign s_rd_ready    = (s_rd_bank == 1'b0) ? s_bank_full[0] : s_bank_full[1];
    assign out_buf_avail = (compute_out_wr_bank == 1'b0) ? ~result_pending[0] : ~result_pending[1];

    assign s_wr_bank_full = (s_wr_bank == 1'b0) ? s_bank_full[0] : s_bank_full[1];
    assign o_s_axis_tready = (state != ST_IDLE && state != ST_DONE) ? (~s_wr_bank_full & ~s_all_loaded) : 1'b0;

    assign s_transfer = (state != ST_IDLE && state != ST_DONE) & i_s_axis_tvalid & o_s_axis_tready;
    assign s_row_loaded = s_transfer & (i_s_axis_tlast | (s_col_k == J_W'(D_HEAD - 1)));
    assign compute_row_done = (state == ST_DIV_ISSUE && div_issue_done);

    // exp_rom
    always_comb begin
        if (z_val <= 0)
            exp_rom_addr = 11'(32'(-32'(z_val)) & 32'h7FF);
        else
            exp_rom_addr = 11'h000;
    end
    assign z_val = scan_active ? (current_s_val - max_val) : '0;

    exp_rom u_exp_rom (
        .clka  (iclk),
        .ena   (1'b1),
        .addra (exp_rom_addr),
        .douta (exp_rom_data)
    );

    // recip_rom
    always_comb begin
        msb_pos_c   = '0;
        msb_found_c = 1'b0;
        for (int b = SUM_WIDTH-1; b >= 0; b--) begin
            if (!msb_found_c && sum_latched[b]) begin
                msb_pos_c   = MSB_W_RECIP'(b);
                msb_found_c = 1'b1;
            end
        end
    end
    always_comb begin
        if (int'(msb_pos_c) >= RECIP_ADDR_W)
            recip_addr_c = sum_latched[int'(msb_pos_c)-1 -: RECIP_ADDR_W];
        else
            recip_addr_c = RECIP_ADDR_W'(sum_latched) << (RECIP_ADDR_W - int'(msb_pos_c));
    end
    recip_rom u_recip_rom (
        .clka  (iclk),
        .ena   (1'b1),
        .addra (recip_addr_c),
        .douta (recip_rom_data)
    );

    //==========================================================================
    // Loader Process
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            s_col_k      <= '0;
            s_wr_bank    <= 1'b0;
            s_all_loaded <= 1'b0;
        end else if (state == ST_IDLE || state == ST_DONE) begin
            s_col_k      <= '0;
            s_wr_bank    <= 1'b0;
            s_all_loaded <= 1'b0;
        end else if (s_transfer) begin
            if (s_wr_bank == 1'b0)
                s_row_buf_0[s_col_k] <= $signed(i_s_axis_tdata[DATA_WIDTH-1:0]);
            else
                s_row_buf_1[s_col_k] <= $signed(i_s_axis_tdata[DATA_WIDTH-1:0]);

            if (i_s_axis_tlast || (s_col_k == J_W'(D_HEAD - 1))) begin
                s_col_k   <= '0;
                s_wr_bank <= ~s_wr_bank;
                if (i_s_axis_tlast)
                    s_all_loaded <= 1'b1;
            end else begin
                s_col_k <= s_col_k + 1;
            end
        end
    end

    //==========================================================================
    // Bank Handshake (s_bank_full, result_pending)
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            s_bank_full <= 2'b00;
        end else if (state == ST_IDLE || state == ST_DONE) begin
            s_bank_full <= 2'b00;
        end else begin
            if (s_row_loaded && s_wr_bank == 1'b0) s_bank_full[0] <= 1'b1;
            else if (compute_row_done && s_rd_bank == 1'b0) s_bank_full[0] <= 1'b0;

            if (s_row_loaded && s_wr_bank == 1'b1) s_bank_full[1] <= 1'b1;
            else if (compute_row_done && s_rd_bank == 1'b1) s_bank_full[1] <= 1'b0;
        end
    end

    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            result_pending <= 2'b00;
        end else if (state == ST_IDLE || state == ST_DONE) begin
            result_pending <= 2'b00;
        end else begin
            if (compute_row_done && compute_out_wr_bank == 1'b0) result_pending[0] <= 1'b1;
            else if (ser_finish_pulse && ser_rd_bank == 1'b0) result_pending[0] <= 1'b0;

            if (compute_row_done && compute_out_wr_bank == 1'b1) result_pending[1] <= 1'b1;
            else if (ser_finish_pulse && ser_rd_bank == 1'b1) result_pending[1] <= 1'b0;
        end
    end

    //==========================================================================
    // Compute Pointers
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            s_rd_bank           <= 1'b0;
            compute_out_wr_bank <= 1'b0;
        end else if (state == ST_IDLE || state == ST_DONE) begin
            s_rd_bank           <= 1'b0;
            compute_out_wr_bank <= 1'b0;
        end else if (compute_row_done) begin
            s_rd_bank           <= ~s_rd_bank;
            compute_out_wr_bank <= ~compute_out_wr_bank;
        end
    end

    //==========================================================================
    // Main FSM
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            state <= ST_IDLE;
        end else begin
            case (state)
                ST_IDLE:
                    if (i_start_softmax)
                        state <= ST_COMPUTE_WAIT;

                ST_COMPUTE_WAIT:
                    if (compute_done)
                        state <= ST_DONE;
                    else if (s_rd_ready && out_buf_avail)
                        state <= ST_FIND_MAX;

                ST_FIND_MAX:
                    if (findmax_done)
                        state <= ST_EXP_SUM;

                ST_EXP_SUM:
                    if (expsum_done)
                        state <= ST_RECIP_PREP;

                ST_RECIP_PREP:
                    if (recip_prep_done)
                        state <= ST_DIV_ISSUE;

                ST_DIV_ISSUE:
                    if (div_issue_done)
                        state <= ST_COMPUTE_WAIT;

                ST_DONE:
                    state <= ST_IDLE;

                default:
                    state <= ST_IDLE;
            endcase
        end
    end

    //==========================================================================
    // Compute Stages
    //==========================================================================
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            max_val      <= {1'b1, {(DATA_WIDTH-1){1'b0}}};
            findmax_idx  <= '0;
            findmax_done <= 1'b0;
        end else begin
            findmax_done <= 1'b0;
            if (state == ST_COMPUTE_WAIT && s_rd_ready && out_buf_avail) begin
                max_val     <= {1'b1, {(DATA_WIDTH-1){1'b0}}};
                findmax_idx <= '0;
            end else if (state == ST_FIND_MAX) begin
                if (current_s_val > max_val)
                    max_val <= current_s_val;

                if (findmax_idx == J_W'(D_HEAD - 1)) begin
                    findmax_idx  <= '0;
                    findmax_done <= 1'b1;
                end else begin
                    findmax_idx <= findmax_idx + 1;
                end
            end
        end
    end

    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            scan_idx    <= '0;
            scan_active <= 1'b0;
            rom_idx_q   <= '0;
            rom_valid_q <= 1'b0;
            sum_acc     <= '0;
            sum_latched <= '0;
            expsum_done <= 1'b0;
        end else begin
            expsum_done <= 1'b0;
            if (state == ST_FIND_MAX && findmax_done) begin
                scan_idx    <= '0;
                scan_active <= 1'b1;
                rom_valid_q <= 1'b0;
                sum_acc     <= '0;
            end else if (state == ST_EXP_SUM) begin
                if (scan_active) begin
                    if (scan_idx == J_W'(D_HEAD - 1)) begin
                        scan_active <= 1'b0;
                    end else begin
                        scan_idx <= scan_idx + 1;
                    end
                end

                rom_idx_q   <= scan_idx;
                rom_valid_q <= scan_active;

                if (rom_valid_q) begin
                    exp_row_buf[rom_idx_q] <= exp_rom_data;
                    sum_acc <= sum_acc + SUM_WIDTH'(exp_rom_data);

                    if (rom_idx_q == J_W'(D_HEAD - 1)) begin
                        sum_latched <= sum_acc + SUM_WIDTH'(exp_rom_data);
                        expsum_done <= 1'b1;
                    end
                end
            end
        end
    end

    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            recip_prep_cnt  <= 1'b0;
            recip_prep_done <= 1'b0;
            precomp_recip   <= '0;
            precomp_shift   <= '0;
        end else begin
            recip_prep_done <= 1'b0;
            if (state == ST_EXP_SUM && expsum_done) begin
                recip_prep_cnt <= 1'b0;
            end else if (state == ST_RECIP_PREP) begin
                if (!recip_prep_cnt) begin
                    recip_prep_cnt <= 1'b1;
                end else begin
                    precomp_recip   <= recip_rom_data;
                    precomp_shift   <= SHIFT_W_RECIP'(RECIP_OUT_W) + SHIFT_W_RECIP'(msb_pos_c);
                    recip_prep_done <= 1'b1;
                end
            end
        end
    end

    genvar w;
    generate
        for (w = 0; w < NUM_WAYS; w++) begin : gen_div_ways
            assign div_dividend[w] = {exp_row_buf[div_in_idx + J_W'(w)], 15'd0};
            assign div_prod[w]     = div_dividend[w] * precomp_recip;
            assign div_shifted[w]  = div_prod_reg[w] >> precomp_shift;
            assign div_result[w]   = (div_shifted[w] > PROD_WIDTH'(MAX_Q15)) ? MAX_Q15 : EXP_WIDTH'(div_shifted[w]);
        end
    endgenerate

    assign div_issue_done = (state == ST_DIV_ISSUE) && div_wr_active && (div_wr_idx == J_W'(D_HEAD - NUM_WAYS));

    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            div_in_idx       <= '0;
            div_wr_idx       <= '0;
            div_issue_active <= 1'b0;
            div_wr_active    <= 1'b0;
            for (int i = 0; i < NUM_WAYS; i++) div_prod_reg[i] <= '0;
        end else begin
            if (state == ST_RECIP_PREP && recip_prep_done) begin
                div_in_idx       <= '0;
                div_issue_active <= 1'b1;
                div_wr_active    <= 1'b0;
            end else if (state == ST_DIV_ISSUE) begin
                if (div_issue_active) begin
                    for (int i = 0; i < NUM_WAYS; i++) begin
                        div_prod_reg[i] <= div_prod[i];
                    end
                    div_wr_idx    <= div_in_idx;
                    div_wr_active <= 1'b1;

                    if (div_in_idx == J_W'(D_HEAD - NUM_WAYS)) begin
                        div_issue_active <= 1'b0;
                    end else begin
                        div_in_idx <= div_in_idx + J_W'(NUM_WAYS);
                    end
                end else begin
                    div_wr_active <= 1'b0;
                end

                if (div_wr_active) begin
                    for (int i = 0; i < NUM_WAYS; i++) begin
                        if (int'(div_wr_idx) + i < D_HEAD) begin
                            if (compute_out_wr_bank == 1'b0)
                                out_row_buf_0[div_wr_idx + J_W'(i)] <= div_result[i];
                            else
                                out_row_buf_1[div_wr_idx + J_W'(i)] <= div_result[i];
                        end
                    end
                end
            end
        end
    end

    //==========================================================================
    // Serializer Process
    //==========================================================================
    assign ser_finish_pulse = serialize_active & i_m_axis_tready & (ser_col_j == J_W'(D_HEAD - 1));

    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            serialize_active <= 1'b0;
            ser_rd_bank      <= 1'b0;
            ser_col_j        <= '0;
            ser_row_i        <= '0;
        end else begin
            if (state == ST_DONE || state == ST_IDLE) begin
                ser_row_i <= '0;
                serialize_active <= 1'b0;
            end else if (!serialize_active) begin
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

    assign o_busy         = (state != ST_IDLE);
    assign o_softmax_done = (state == ST_DONE);

    assign o_m_axis_tvalid = serialize_active;
    assign o_m_axis_tlast  = serialize_active
                           & (ser_col_j == J_W'(D_HEAD - 1))
                           & (ser_row_i == S_W'(SEQ_LEN - 1));
    assign o_m_axis_tdata  = 32'(current_out_val);

    assign last_row_last_col = o_m_axis_tvalid
                             & o_m_axis_tlast
                             & i_m_axis_tready
                             & (ser_row_i == S_W'(SEQ_LEN - 1));

    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n)
            compute_done <= 1'b0;
        else
            compute_done <= last_row_last_col;
    end

    //==========================================================================
    // Hardware Cycle Counter (Synthesizable)
    //==========================================================================
    logic [31:0] softmax_cycles;
    assign o_softmax_cycles = softmax_cycles;
    
    always_ff @(posedge iclk or negedge irst_n) begin
        if (!irst_n) begin
            softmax_cycles <= 0;
        end else begin
            if (i_start_softmax) begin
                softmax_cycles <= 0;
            end else if (o_busy) begin
                softmax_cycles <= softmax_cycles + 1;
            end

            // synthesis translate_off
            if (o_softmax_done) begin
                $display("[%0t] [SOFTMAX IP] Inference completed! Total execution time: %0d clock cycles", $time, softmax_cycles);
            end
            // synthesis translate_on
        end
    end

endmodule
