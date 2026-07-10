//============================================================================
// NEXUS-OMEGA Stage 20: Fixed-Point Hardware Matching Engine
// 
// Synthesizable SystemVerilog RTL for hardware order matching.
// Implements price-time priority matching using fixed-point arithmetic.
// 
// CRITICAL CONSTRAINTS:
// - NO floating-point math (all fixed-point)
// - NO dynamic memory allocation
// - NO unsynthesizable constructs
// - Single-cycle match detection for best bid/ask crossing
// 
// Matching Logic:
// - Incoming buy order matches against lowest ask if bid_price >= ask_price
// - Incoming sell order matches against highest bid if sell_price <= bid_price
// - Partial fills supported, remaining quantity re-queued
//============================================================================

`timescale 1ns / 1ps
`default_nettype wire

module fixed_point_matcher #(
    // Fixed-point parameters
    parameter PRICE_WIDTH = 32,
    parameter INTEGER_WIDTH = 16,
    parameter FRACTIONAL_WIDTH = 16,
    
    // Quantity parameters
    parameter QUANTITY_WIDTH = 32,
    
    // Order ID width
    parameter ORDER_ID_WIDTH = 64,
    
    // Trade event output FIFO depth
    parameter TRADE_FIFO_DEPTH = 64
) (
    // Global clock and reset
    input wire clk,
    input wire rst_n,              // Active-low async reset
    
    // Input: New order from LOB
    input wire                      order_valid,
    input wire [1:0]                order_side,      // 0=BID(buy), 1=ASK(sell)
    input wire [PRICE_WIDTH-1:0]    order_price,
    input wire [QUANTITY_WIDTH-1:0] order_quantity,
    input wire [ORDER_ID_WIDTH-1:0] order_id,
    input wire                      order_is_new,    // 1=new order, 0=resting order
    
    // Input: Best bid from LOB
    input wire [PRICE_WIDTH-1:0]    best_bid_price,
    input wire [QUANTITY_WIDTH-1:0] best_bid_qty,
    input wire [ORDER_ID_WIDTH-1:0] best_bid_order_id,
    input wire                      best_bid_valid,
    
    // Input: Best ask from LOB
    input wire [PRICE_WIDTH-1:0]    best_ask_price,
    input wire [QUANTITY_WIDTH-1:0] best_ask_qty,
    input wire [ORDER_ID_WIDTH-1:0] best_ask_order_id,
    input wire                      best_ask_valid,
    
    // Output: Match detected
    output wire                     match_detected,
    output wire [1:0]               match_aggressor_side,
    output wire [PRICE_WIDTH-1:0]   match_price,
    output wire [QUANTITY_WIDTH-1:0] match_quantity,
    output wire [ORDER_ID_WIDTH-1:0] match_taker_order_id,
    output wire [ORDER_ID_WIDTH-1:0] match_maker_order_id,
    
    // Output: Updated order (for partial fills)
    output wire                     update_valid,
    output wire [ORDER_ID_WIDTH-1:0] update_order_id,
    output wire [QUANTITY_WIDTH-1:0] update_remaining_qty,
    
    // Output: Trade event to host
    output wire                     trade_event_valid,
    output wire [7:0]               trade_event_data,
    output wire                     trade_event_last,
    input wire                      trade_event_ready,
    
    // Status
    output wire                     matcher_busy,
    output wire                     matcher_ready,
    output wire [31:0]              total_matches,
    output wire [31:0]              total_volume
);

    //========================================================================
    // Internal Signals
    //========================================================================
    
    // Match condition flags
    logic buy_can_match;
    logic sell_can_match;
    logic price_crosses;
    logic quantity_available;
    
    // Match calculation
    logic [QUANTITY_WIDTH:0] match_qty_calc;
    logic [QUANTITY_WIDTH-1:0] aggressor_remaining;
    logic [QUANTITY_WIDTH-1:0] maker_remaining;
    logic full_fill;
    logic partial_fill;
    
    // Pipeline registers
    logic match_pipe_valid;
    logic [1:0] match_pipe_side;
    logic [PRICE_WIDTH-1:0] match_pipe_price;
    logic [QUANTITY_WIDTH-1:0] match_pipe_qty;
    logic [ORDER_ID_WIDTH-1:0] match_pipe_taker;
    logic [ORDER_ID_WIDTH-1:0] match_pipe_maker;
    
    // Statistics counters
    logic [31:0] match_count_reg;
    logic [31:0] volume_count_reg;
    logic [QUANTITY_WIDTH:0] temp_volume;
    
    // Trade event serialization
    typedef enum logic [2:0] {
        EVT_IDLE = 3'd0,
        EVT_HEADER = 3'd1,
        EVT_TAKER_ID = 3'd2,
        EVT_MAKER_ID = 3'd3,
        EVT_PRICE = 3'd4,
        EVT_QTY = 3'd5,
        EVT_FOOTER = 3'd6
    } trade_state_t;
    
    trade_state_t trade_state;
    logic [7:0] trade_shift_reg;
    logic [3:0] byte_counter;
    
    //========================================================================
    // Match Condition Detection (Combinational)
    //========================================================================
    
    // Buy order can match if there's a valid ask and bid_price >= ask_price
    assign buy_can_match = order_valid && order_side == 1'b0 && best_ask_valid;
    assign sell_can_match = order_valid && order_side == 1'b1 && best_bid_valid;
    
    // Price crossing detection (fixed-point comparison)
    assign price_crosses = (order_side == 1'b0) ? 
                           (order_price >= best_ask_price) :  // Buy crosses ask
                           (order_price <= best_bid_price);   // Sell crosses bid
    
    // Quantity availability check
    assign quantity_available = (order_side == 1'b0) ? 
                                (best_ask_qty > '0) : 
                                (best_bid_qty > '0);
    
    // Final match signal
    assign match_detected = match_pipe_valid;
    assign match_aggressor_side = match_pipe_side;
    assign match_price = match_pipe_price;
    assign match_quantity = match_pipe_qty;
    assign match_taker_order_id = match_pipe_taker;
    assign match_maker_order_id = match_pipe_maker;
    
    //========================================================================
    // Match Quantity Calculation
    //========================================================================
    
    always_comb begin
        // Default values
        match_qty_calc = '0;
        aggressor_remaining = '0;
        maker_remaining = '0;
        full_fill = 1'b0;
        partial_fill = 1'b0;
        
        if (order_valid && price_crosses && quantity_available) begin
            if (order_side == 1'b0) begin
                // Buy order hitting ask
                if (order_quantity >= best_ask_qty) begin
                    // Full fill of ask
                    match_qty_calc = best_ask_qty;
                    aggressor_remaining = order_quantity - best_ask_qty;
                    maker_remaining = '0;
                    full_fill = 1'b1;
                end else begin
                    // Partial fill
                    match_qty_calc = order_quantity;
                    aggressor_remaining = '0;
                    maker_remaining = best_ask_qty - order_quantity;
                    partial_fill = 1'b1;
                end
            end else begin
                // Sell order hitting bid
                if (order_quantity >= best_bid_qty) begin
                    match_qty_calc = best_bid_qty;
                    aggressor_remaining = order_quantity - best_bid_qty;
                    maker_remaining = '0;
                    full_fill = 1'b1;
                end else begin
                    match_qty_calc = order_quantity;
                    aggressor_remaining = '0;
                    maker_remaining = best_bid_qty - order_quantity;
                    partial_fill = 1'b1;
                end
            end
        end
    end
    
    //========================================================================
    // Match Pipeline Register (Single-Cycle Match Detection)
    //========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            match_pipe_valid <= 1'b0;
            match_pipe_side <= '0;
            match_pipe_price <= '0;
            match_pipe_qty <= '0;
            match_pipe_taker <= '0;
            match_pipe_maker <= '0;
        end else begin
            if (order_valid && price_crosses && quantity_available) begin
                match_pipe_valid <= 1'b1;
                match_pipe_side <= order_side;
                
                // Match price is the resting order's price (price-time priority)
                match_pipe_price <= (order_side == 1'b0) ? best_ask_price : best_bid_price;
                match_pipe_qty <= match_qty_calc[QUANTITY_WIDTH-1:0];
                match_pipe_taker <= order_id;
                match_pipe_maker <= (order_side == 1'b0) ? best_ask_order_id : best_bid_order_id;
            end else begin
                match_pipe_valid <= 1'b0;
            end
        end
    end
    
    //========================================================================
    // Update Signal for Partial Fills
    //========================================================================
    
    assign update_valid = partial_fill;
    assign update_order_id = (order_side == 1'b0) ? best_ask_order_id : best_bid_order_id;
    assign update_remaining_qty = maker_remaining;
    
    //========================================================================
    // Statistics Counters
    //========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            match_count_reg <= '0;
            volume_count_reg <= '0;
        end else if (match_pipe_valid) begin
            match_count_reg <= match_count_reg + 1;
            
            temp_volume = volume_count_reg + match_qty_calc;
            volume_count_reg <= temp_volume[31:0];
        end
    end
    
    assign total_matches = match_count_reg;
    assign total_volume = volume_count_reg;
    
    //========================================================================
    // Trade Event Serialization (AXI4-Stream Compatible)
    //========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            trade_state <= EVT_IDLE;
            trade_shift_reg <= '0;
            byte_counter <= '0;
            trade_event_valid <= 1'b0;
            trade_event_last <= 1'b0;
            trade_event_data <= '0;
        end else begin
            trade_event_valid <= 1'b0;
            trade_event_last <= 1'b0;
            
            case (trade_state)
                EVT_IDLE: begin
                    if (match_pipe_valid && trade_event_ready) begin
                        trade_state <= EVT_HEADER;
                        trade_shift_reg <= 8'hAA;  // Magic header
                        byte_counter <= '0;
                    end
                end
                
                EVT_HEADER: begin
                    if (trade_event_ready) begin
                        trade_event_valid <= 1'b1;
                        trade_event_data <= trade_shift_reg;
                        trade_state <= EVT_TAKER_ID;
                        byte_counter <= '0;
                    end
                end
                
                EVT_TAKER_ID: begin
                    if (trade_event_ready) begin
                        trade_event_valid <= 1'b1;
                        // Shift out taker order ID (8 bytes)
                        trade_event_data <= match_pipe_taker[byte_counter * 8 +: 8];
                        byte_counter <= byte_counter + 1;
                        
                        if (byte_counter == 7) begin
                            trade_state <= EVT_MAKER_ID;
                            byte_counter <= '0;
                        end
                    end
                end
                
                EVT_MAKER_ID: begin
                    if (trade_event_ready) begin
                        trade_event_valid <= 1'b1;
                        trade_event_data <= match_pipe_maker[byte_counter * 8 +: 8];
                        byte_counter <= byte_counter + 1;
                        
                        if (byte_counter == 7) begin
                            trade_state <= EVT_PRICE;
                            byte_counter <= '0;
                        end
                    end
                end
                
                EVT_PRICE: begin
                    if (trade_event_ready) begin
                        trade_event_valid <= 1'b1;
                        trade_event_data <= match_pipe_price[byte_counter * 8 +: 8];
                        byte_counter <= byte_counter + 1;
                        
                        if (byte_counter == 3) begin
                            trade_state <= EVT_QTY;
                            byte_counter <= '0;
                        end
                    end
                end
                
                EVT_QTY: begin
                    if (trade_event_ready) begin
                        trade_event_valid <= 1'b1;
                        trade_event_data <= match_pipe_qty[byte_counter * 8 +: 8];
                        byte_counter <= byte_counter + 1;
                        
                        if (byte_counter == 3) begin
                            trade_state <= EVT_FOOTER;
                            byte_counter <= '0;
                        end
                    end
                end
                
                EVT_FOOTER: begin
                    if (trade_event_ready) begin
                        trade_event_valid <= 1'b1;
                        trade_event_data <= 8'h55;  // Magic footer
                        trade_event_last <= 1'b1;
                        trade_state <= EVT_IDLE;
                    end
                end
                
                default: trade_state <= EVT_IDLE;
            endcase
        end
    end
    
    //========================================================================
    // Status Outputs
    //========================================================================
    
    assign matcher_busy = (trade_state != EVT_IDLE);
    assign matcher_ready = !matcher_busy;
    
endmodule
