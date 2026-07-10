//============================================================================
// NEXUS-OMEGA Stage 20: Hardware Risk Guardian
// 
// Synthesizable SystemVerilog RTL for nanosecond-level risk monitoring.
// Monitors position limits, margin requirements, and sequence gaps in real-time.
// 
// CRITICAL CONSTRAINTS:
// - NO floating-point math (all fixed-point)
// - NO unsynthesizable constructs
// - Updates on every clock edge (< 3ns latency)
// - Triggers kill-switch immediately on breach
// 
// Monitored Metrics:
// - Gross notional exposure per symbol and total
// - Net delta exposure
// - Margin utilization percentage
// - Sequence ID gaps from exchange feed
// - Order rate limiting
//============================================================================

`timescale 1ns / 1ps
`default_nettype wire

module hardware_risk_guardian #(
    // Fixed-point parameters
    parameter PRICE_WIDTH = 32,
    parameter INTEGER_WIDTH = 16,
    parameter FRACTIONAL_WIDTH = 16,
    
    // Position tracking
    parameter QUANTITY_WIDTH = 32,
    parameter SYMBOL_ID_WIDTH = 16,
    parameter NUM_SYMBOLS = 256,
    
    // Risk limits (configured by host via registers)
    parameter MAX_GROSS_NOTIONAL_WIDTH = 64,
    parameter MAX_NET_DELTA_WIDTH = 32,
    parameter MARGIN_LIMIT_WIDTH = 32,
    
    // Rate limiting
    parameter ORDER_RATE_WINDOW = 1000,  // ms
    parameter MAX_ORDERS_PER_WINDOW = 10000
) (
    // Global clock and reset
    input wire clk,
    input wire rst_n,              // Active-low async reset
    
    // Configuration registers (from host via PCIe)
    input wire                      cfg_valid,
    input wire [MAX_GROSS_NOTIONAL_WIDTH-1:0] cfg_max_gross_notional,
    input wire [MAX_NET_DELTA_WIDTH-1:0] cfg_max_net_delta,
    input wire [MARGIN_LIMIT_WIDTH-1:0] cfg_margin_limit_pct,  // e.g., 80 = 80%
    input wire [31:0]               cfg_account_equity,
    
    // Real-time market data
    input wire                      md_valid,
    input wire [SYMBOL_ID_WIDTH-1:0] md_symbol_id,
    input wire [PRICE_WIDTH-1:0]    md_price,
    input wire [1:0]                md_type,       // 0=trade, 1=quote
    
    // Position updates (from matching engine)
    input wire                      pos_update_valid,
    input wire [SYMBOL_ID_WIDTH-1:0] pos_symbol_id,
    input wire [QUANTITY_WIDTH-1:0] pos_delta,     // Signed quantity change
    input wire [PRICE_WIDTH-1:0]    pos_price,
    input wire [1:0]                pos_side,      // 0=buy, 1=sell
    
    // Sequence monitoring
    input wire                      seq_valid,
    input wire [31:0]               seq_current_id,
    input wire [31:0]               seq_expected_id,
    
    // Order rate monitoring
    input wire                      order_submit,
    
    // Risk breach outputs
    output wire                     gross_notional_breach,
    output wire                     net_delta_breach,
    output wire                     margin_breach,
    output wire                     sequence_gap_breach,
    output wire                     rate_limit_breach,
    output wire                     any_breach,
    
    // Current metrics (for telemetry)
    output wire [MAX_GROSS_NOTIONAL_WIDTH-1:0] current_gross_notional,
    output wire [MAX_NET_DELTA_WIDTH-1:0] current_net_delta,
    output wire [MARGIN_LIMIT_WIDTH-1:0] current_margin_pct,
    output wire [31:0]               orders_in_window,
    
    // Kill-switch trigger
    output wire                     kill_switch_trigger
);

    //========================================================================
    // Internal Signals
    //========================================================================
    
    // Per-symbol position storage
    logic signed [QUANTITY_WIDTH-1:0] symbol_positions [0:NUM_SYMBOLS-1];
    logic signed [MAX_GROSS_NOTIONAL_WIDTH-1:0] symbol_notionals [0:NUM_SYMBOLS-1];
    
    // Aggregate metrics
    logic signed [MAX_GROSS_NOTIONAL_WIDTH-1:0] total_gross_notional;
    logic signed [MAX_NET_DELTA_WIDTH-1:0] total_net_delta;
    logic [MARGIN_LIMIT_WIDTH-1:0] margin_utilization;
    
    // Sequence tracking
    logic [31:0] last_seq_id;
    logic seq_gap_flag;
    
    // Rate limiting
    logic [31:0] order_counter;
    logic [31:0] window_start_time;
    logic [31:0] current_time_ms;
    logic rate_exceeded;
    
    // Breach flags (registered for stability)
    logic gross_breach_reg;
    logic delta_breach_reg;
    logic margin_breach_reg;
    logic seq_breach_reg;
    logic rate_breach_reg;
    
    //========================================================================
    // Position Tracking (Per-Symbol)
    //========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (integer i = 0; i < NUM_SYMBOLS; i++) begin
                symbol_positions[i] <= '0;
                symbol_notionals[i] <= '0;
            end
        end else if (pos_update_valid) begin
            // Update position for this symbol
            integer idx;
            idx = pos_symbol_id;
            
            if (idx < NUM_SYMBOLS) begin
                // Update signed position
                if (pos_side == 1'b0) begin
                    symbol_positions[idx] <= symbol_positions[idx] + pos_delta;
                end else begin
                    symbol_positions[idx] <= symbol_positions[idx] - pos_delta;
                end
                
                // Update notional (position * price)
                // Simplified: Using lower bits for multiplication
                symbol_notionals[idx] <= symbol_positions[idx] * pos_price[INTEGER_WIDTH-1:0];
            end
        end
    end
    
    //========================================================================
    // Gross Notional Calculation
    //========================================================================
    
    always_comb begin
        total_gross_notional = '0;
        
        for (integer i = 0; i < NUM_SYMBOLS; i++) begin
            // Add absolute value of each symbol's notional
            if (symbol_notionals[i] >= 0) begin
                total_gross_notional = total_gross_notional + symbol_notionals[i];
            end else begin
                total_gross_notional = total_gross_notional - symbol_notionals[i];
            end
        end
    end
    
    assign current_gross_notional = total_gross_notional;
    
    // Gross notional breach detection
    assign gross_notional_breach = gross_breach_reg;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gross_breach_reg <= 1'b0;
        end else begin
            gross_breach_reg <= (total_gross_notional > cfg_max_gross_notional);
        end
    end
    
    //========================================================================
    // Net Delta Calculation
    //========================================================================
    
    always_comb begin
        total_net_delta = '0;
        
        for (integer i = 0; i < NUM_SYMBOLS; i++) begin
            total_net_delta = total_net_delta + symbol_positions[i];
        end
    end
    
    assign current_net_delta = total_net_delta;
    assign net_delta_breach = delta_breach_reg;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            delta_breach_reg <= 1'b0;
        end else begin
            // Check absolute value of net delta
            if (total_net_delta >= 0) begin
                delta_breach_reg <= (total_net_delta > cfg_max_net_delta);
            end else begin
                delta_breach_reg <= (-total_net_delta > cfg_max_net_delta);
            end
        end
    end
    
    //========================================================================
    // Margin Utilization
    //========================================================================
    
    // Margin % = (Gross Notional / Equity) * 100
    // Simplified division using shift-based approximation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            margin_utilization <= '0;
            margin_breach_reg <= 1'b0;
        end else if (cfg_account_equity > '0) begin
            // Approximate: (notional << 7) / equity = percentage * 128
            margin_utilization <= (total_gross_notional << 7) / cfg_account_equity;
            margin_breach_reg <= (margin_utilization > cfg_margin_limit_pct);
        end
    end
    
    assign current_margin_pct = margin_utilization;
    assign margin_breach = margin_breach_reg;
    
    //========================================================================
    // Sequence Gap Detection
    //========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            last_seq_id <= '0;
            seq_gap_flag <= 1'b0;
            seq_breach_reg <= 1'b0;
        end else if (seq_valid) begin
            last_seq_id <= seq_current_id;
            
            if (seq_current_id != seq_expected_id) begin
                seq_gap_flag <= 1'b1;
                seq_breach_reg <= 1'b1;
            end else begin
                seq_gap_flag <= 1'b0;
                seq_breach_reg <= 1'b0;
            end
        end
    end
    
    assign sequence_gap_breach = seq_breach_reg;
    
    //========================================================================
    // Order Rate Limiting
    //========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            order_counter <= '0;
            rate_breach_reg <= 1'b0;
            window_start_time <= '0;
        end else begin
            // Simple sliding window counter
            if (order_submit) begin
                if (order_counter >= MAX_ORDERS_PER_WINDOW) begin
                    rate_exceeded <= 1'b1;
                    rate_breach_reg <= 1'b1;
                end else begin
                    order_counter <= order_counter + 1;
                end
            end
            
            // Reset counter after window expires (simplified timing)
            if (current_time_ms - window_start_time > ORDER_RATE_WINDOW) begin
                order_counter <= '0;
                window_start_time <= current_time_ms;
                rate_breach_reg <= 1'b0;
            end
        end
    end
    
    assign orders_in_window = order_counter;
    assign rate_limit_breach = rate_breach_reg;
    
    //========================================================================
    // Any Breach Detection & Kill-Switch Trigger
    //========================================================================
    
    assign any_breach = gross_notional_breach || 
                        net_delta_breach || 
                        margin_breach || 
                        sequence_gap_breach || 
                        rate_limit_breach;
    
    // Kill-switch triggers on ANY breach
    // This is registered to prevent glitches
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kill_switch_trigger <= 1'b0;
        end else begin
            kill_switch_trigger <= any_breach;
        end
    end
    
endmodule
