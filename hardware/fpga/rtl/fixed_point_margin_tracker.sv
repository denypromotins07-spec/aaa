//============================================================================
// NEXUS-OMEGA Stage 20: Fixed-Point Margin Tracker
// 
// Synthesizable SystemVerilog RTL for real-time margin calculation.
// Tracks initial margin, maintenance margin, and variation margin
// using fixed-point arithmetic only.
// 
// CRITICAL CONSTRAINTS:
// - NO floating-point math
// - NO unsynthesizable constructs
// - Updates on every clock cycle
// - Supports multiple asset classes with different margin rates
//============================================================================

`timescale 1ns / 1ps
`default_nettype wire

module fixed_point_margin_tracker #(
    // Fixed-point parameters
    parameter PRICE_WIDTH = 32,
    parameter INTEGER_WIDTH = 16,
    parameter FRACTIONAL_WIDTH = 16,
    
    // Quantity and margin parameters
    parameter QUANTITY_WIDTH = 32,
    parameter MARGIN_RATE_WIDTH = 16,     // Q8.8 format for margin rates
    parameter MARGIN_VALUE_WIDTH = 64,    // Accumulated margin values
    
    // Number of supported symbols
    parameter NUM_SYMBOLS = 256,
    
    // Asset class encoding
    parameter ASSET_EQUITY = 2'd0,
    parameter ASSET_FUTURE = 2'd1,
    parameter ASSET_OPTION = 2'd2,
    parameter ASSET_CRYPTO = 2'd3
) (
    // Global clock and reset
    input wire clk,
    input wire rst_n,              // Active-low async reset
    
    // Configuration: Per-symbol margin rates (from host)
    input wire                      cfg_valid,
    input wire [15:0]               cfg_symbol_id,
    input wire [MARGIN_RATE_WIDTH-1:0] cfg_initial_margin_rate,   // e.g., 0x1000 = 100%
    input wire [MARGIN_RATE_WIDTH-1:0] cfg_maintenance_margin_rate,
    input wire [1:0]                cfg_asset_class,
    
    // Position updates
    input wire                      pos_update_valid,
    input wire [15:0]               pos_symbol_id,
    input wire signed [QUANTITY_WIDTH-1:0] pos_quantity,
    input wire [PRICE_WIDTH-1:0]    pos_entry_price,
    input wire [PRICE_WIDTH-1:0]    pos_current_price,
    input wire [1:0]                pos_asset_class,
    
    // Mark-to-market price updates
    input wire                      mtm_valid,
    input wire [15:0]               mtm_symbol_id,
    input wire [PRICE_WIDTH-1:0]    mtm_price,
    
    // Output: Total margin requirements
    output wire [MARGIN_VALUE_WIDTH-1:0] total_initial_margin,
    output wire [MARGIN_VALUE_WIDTH-1:0] total_maintenance_margin,
    output wire [MARGIN_VALUE_WIDTH-1:0] total_variation_margin,
    output wire [MARGIN_VALUE_WIDTH-1:0] total_excess_margin,
    
    // Per-symbol breakdown (for telemetry)
    output wire [MARGIN_VALUE_WIDTH-1:0] symbol_initial_margin,
    output wire [MARGIN_VALUE_WIDTH-1:0] symbol_maintenance_margin,
    output wire [MARGIN_VALUE_WIDTH-1:0] symbol_unrealized_pnl,
    
    // Margin breach detection
    input wire [MARGIN_VALUE_WIDTH-1:0] account_equity,
    output wire                     margin_call,
    output wire                     liquidation_warning
);

    //========================================================================
    // Internal Types and Signals
    //========================================================================
    
    // Per-symbol position storage
    logic signed [QUANTITY_WIDTH-1:0] positions [0:NUM_SYMBOLS-1];
    logic [PRICE_WIDTH-1:0] entry_prices [0:NUM_SYMBOLS-1];
    logic [PRICE_WIDTH-1:0] current_prices [0:NUM_SYMBOLS-1];
    logic [MARGIN_RATE_WIDTH-1:0] initial_rates [0:NUM_SYMBOLS-1];
    logic [MARGIN_RATE_WIDTH-1:0] maintenance_rates [0:NUM_SYMBOLS-1];
    logic [1:0] asset_classes [0:NUM_SYMBOLS-1];
    logic position_valid [0:NUM_SYMBOLS-1];
    
    // Margin calculations
    logic [MARGIN_VALUE_WIDTH-1:0] symbol_init_margin [0:NUM_SYMBOLS-1];
    logic [MARGIN_VALUE_WIDTH-1:0] symbol maint_margin [0:NUM_SYMBOLS-1];
    logic signed [MARGIN_VALUE_WIDTH-1:0] symbol_var_margin [0:NUM_SYMBOLS-1];
    
    // Aggregate totals
    logic [MARGIN_VALUE_WIDTH-1:0] agg_initial_margin;
    logic [MARGIN_VALUE_WIDTH-1:0] agg_maintenance_margin;
    logic signed [MARGIN_VALUE_WIDTH-1:0] agg_variation_margin;
    
    // Temporary calculation signals
    logic [MARGIN_VALUE_WIDTH+31:0] temp_mult;  // Extra bits for multiplication
    logic signed [PRICE_WIDTH:0] price_diff;
    
    //========================================================================
    // Configuration Register Write
    //========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (integer i = 0; i < NUM_SYMBOLS; i++) begin
                initial_rates[i] <= '0;
                maintenance_rates[i] <= '0;
                asset_classes[i] <= '0;
            end
        end else if (cfg_valid) begin
            initial_rates[cfg_symbol_id] <= cfg_initial_margin_rate;
            maintenance_rates[cfg_symbol_id] <= cfg_maintenance_margin_rate;
            asset_classes[cfg_symbol_id] <= cfg_asset_class;
        end
    end
    
    //========================================================================
    // Position Tracking
    //========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (integer i = 0; i < NUM_SYMBOLS; i++) begin
                positions[i] <= '0;
                entry_prices[i] <= '0;
                current_prices[i] <= '0;
                position_valid[i] <= 1'b0;
            end
        end else if (pos_update_valid) begin
            integer idx;
            idx = pos_symbol_id;
            
            if (pos_quantity != '0) begin
                // New or modified position
                positions[idx] <= pos_quantity;
                entry_prices[idx] <= pos_entry_price;
                position_valid[idx] <= 1'b1;
            end else begin
                // Position closed
                position_valid[idx] <= 1'b0;
            end
            
            asset_classes[idx] <= pos_asset_class;
        end
    end
    
    //========================================================================
    // Mark-to-Market Price Updates
    //========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (integer i = 0; i < NUM_SYMBOLS; i++) begin
                current_prices[i] <= '0;
            end
        end else if (mtm_valid) begin
            current_prices[mtm_symbol_id] <= mtm_price;
        end
    end
    
    //========================================================================
    // Per-Symbol Margin Calculation
    //========================================================================
    
    genvar g;
    generate
        for (g = 0; g < NUM_SYMBOLS; g++) begin : symbol_margin_gen
            
            // Initial Margin = |Position| * Entry_Price * Initial_Rate
            always_comb begin
                if (position_valid[g]) begin
                    // Absolute value of position
                    logic [QUANTITY_WIDTH-1:0] abs_qty;
                    abs_qty = positions[g] >= 0 ? positions[g] : -positions[g];
                    
                    // Multiply: qty * price (result is Q16.32)
                    temp_mult = abs_qty * entry_prices[g];
                    
                    // Apply margin rate (Q8.8 format)
                    symbol_init_margin[g] = (temp_mult * initial_rates[g]) >> 8;
                end else begin
                    symbol_init_margin[g] = '0;
                end
            end
            
            // Maintenance Margin = |Position| * Current_Price * Maint_Rate
            always_comb begin
                if (position_valid[g]) begin
                    logic [QUANTITY_WIDTH-1:0] abs_qty;
                    abs_qty = positions[g] >= 0 ? positions[g] : -positions[g];
                    
                    temp_mult = abs_qty * current_prices[g];
                    symbol_maint_margin[g] = (temp_mult * maintenance_rates[g]) >> 8;
                end else begin
                    symbol_maint_margin[g] = '0;
                end
            end
            
            // Variation Margin (Unrealized PnL) = Position * (Current_Price - Entry_Price)
            always_comb begin
                if (position_valid[g]) begin
                    price_diff = current_prices[g] - entry_prices[g];
                    symbol_var_margin[g] = positions[g] * price_diff;
                end else begin
                    symbol_var_margin[g] = '0;
                end
            end
            
        end
    endgenerate
    
    //========================================================================
    // Aggregate Margin Totals
    //========================================================================
    
    always_comb begin
        agg_initial_margin = '0;
        agg_maintenance_margin = '0;
        agg_variation_margin = '0;
        
        for (integer i = 0; i < NUM_SYMBOLS; i++) begin
            agg_initial_margin = agg_initial_margin + symbol_init_margin[i];
            agg_maintenance_margin = agg_maintenance_margin + symbol_maint_margin[i];
            agg_variation_margin = agg_variation_margin + symbol_var_margin[i];
        end
    end
    
    assign total_initial_margin = agg_initial_margin;
    assign total_maintenance_margin = agg_maintenance_margin;
    assign total_variation_margin = agg_variation_margin;
    
    // Excess Margin = Equity - (Initial Margin + Negative Variation)
    assign total_excess_margin = account_equity - agg_initial_margin - 
                                  (agg_variation_margin < 0 ? -agg_variation_margin : '0);
    
    //========================================================================
    // Symbol-Level Outputs (for first symbol as example)
    //========================================================================
    
    assign symbol_initial_margin = symbol_init_margin[0];
    assign symbol_maintenance_margin = symbol_maint_margin[0];
    assign symbol_unrealized_pnl = symbol_var_margin[0];
    
    //========================================================================
    // Margin Breach Detection
    //========================================================================
    
    // Margin Call: Excess margin below maintenance requirement
    assign margin_call = (account_equity < agg_maintenance_margin);
    
    // Liquidation Warning: Excess margin below 50% of initial requirement
    assign liquidation_warning = (account_equity < (agg_initial_margin >> 1));
    
endmodule
