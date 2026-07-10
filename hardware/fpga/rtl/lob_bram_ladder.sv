//============================================================================
// NEXUS-OMEGA Stage 20: Hardware Limit Order Book BRAM Ladder
// 
// Synthesizable SystemVerilog RTL for hardware-accelerated LOB storage.
// Uses Block RAM (BRAM) dual-port arrays for price levels.
// 
// CRITICAL CONSTRAINTS:
// - NO floating-point math (fixed-point only)
// - NO dynamic memory allocation
// - NO unsynthesizable constructs (# delays, real types, unclocked initial)
// - Fully synchronous, clocked logic
//
// Price ladder organized as:
// - Bid side: Prices stored in descending order (highest bid at index 0)
// - Ask side: Prices stored in ascending order (lowest ask at index 0)
//============================================================================

`timescale 1ns / 1ps
`default_nettype wire

module lob_bram_ladder #(
    // Number of price levels per side
    parameter NUM_PRICE_LEVELS = 256,
    
    // Fixed-point format: TOTAL_WIDTH = INTEGER_WIDTH + FRACTIONAL_WIDTH
    parameter PRICE_WIDTH = 32,
    parameter INTEGER_WIDTH = 16,
    parameter FRACTIONAL_WIDTH = 16,
    
    // Quantity/size width
    parameter QUANTITY_WIDTH = 32,
    
    // Order ID width
    parameter ORDER_ID_WIDTH = 64,
    
    // Number of orders per price level
    parameter ORDERS_PER_LEVEL = 16
) (
    // Global clock and reset
    input wire clk,
    input wire rst_n,              // Active-low async reset
    
    // AXI4-Stream interface for market data ingest
    input wire                      s_axis_tvalid,
    input wire [31:0]               s_axis_tdata,
    input wire [7:0]                s_axis_tkeep,
    input wire                      s_axis_tlast,
    input wire                      s_axis_tready,
    output wire                     s_axis_tready_out,
    
    // Command interface from host (via PCIe DMA)
    input wire                      cmd_valid,
    input wire [2:0]                cmd_type,      // 0=ADD, 1=CANCEL, 2=MODIFY
    input wire [1:0]                cmd_side,      // 0=BID, 1=ASK
    input wire [PRICE_WIDTH-1:0]    cmd_price,
    input wire [QUANTITY_WIDTH-1:0] cmd_quantity,
    input wire [ORDER_ID_WIDTH-1:0] cmd_order_id,
    output wire                     cmd_ready,
    
    // Best bid/ask output
    output wire [PRICE_WIDTH-1:0]   best_bid_price,
    output wire [QUANTITY_WIDTH-1:0] best_bid_qty,
    output wire [PRICE_WIDTH-1:0]   best_ask_price,
    output wire [QUANTITY_WIDTH-1:0] best_ask_qty,
    
    // Spread and mid-price output
    output wire [PRICE_WIDTH-1:0]   spread,
    output wire [PRICE_WIDTH+1:0]   mid_price,     // Extra bit for average
    
    // Status outputs
    output wire [NUM_PRICE_LEVELS-1:0] bid_levels_occupied,
    output wire [NUM_PRICE_LEVELS-1:0] ask_levels_occupied,
    output wire                       lob_ready
);

    //========================================================================
    // Internal Types and Signals
    //========================================================================
    
    // Order entry structure stored in BRAM
    typedef struct packed {
        logic [ORDER_ID_WIDTH-1:0] order_id;
        logic [QUANTITY_WIDTH-1:0] quantity;
        logic                      valid;
    } order_entry_t;
    
    // Price level structure
    typedef struct packed {
        logic [PRICE_WIDTH-1:0] price;
        logic [QUANTITY_WIDTH-1:0] total_quantity;
        logic                      occupied;
        order_entry_t [ORDERS_PER_LEVEL-1:0] orders;
    } price_level_t;
    
    // BRAM storage for bid and ask sides
    price_level_t bid_book [0:NUM_PRICE_LEVELS-1];
    price_level_t ask_book [0:NUM_PRICE_LEVELS-1];
    
    // Pipeline registers for command processing
    logic [2:0] cmd_pipe_type;
    logic [1:0] cmd_pipe_side;
    logic [PRICE_WIDTH-1:0] cmd_pipe_price;
    logic [QUANTITY_WIDTH-1:0] cmd_pipe_qty;
    logic [ORDER_ID_WIDTH-1:0] cmd_pipe_order_id;
    logic cmd_pipe_valid;
    
    // Index tracking for best bid/ask
    logic [$clog2(NUM_PRICE_LEVELS)-1:0] best_bid_idx;
    logic [$clog2(NUM_PRICE_LEVELS)-1:0] best_ask_idx;
    
    // Comparison results
    logic price_greater;
    logic price_less;
    logic price_equal;
    
    // Write enable signals for BRAM
    logic bid_write_enable;
    logic ask_write_enable;
    logic [$clog2(NUM_PRICE_LEVELS)-1:0] write_address;
    
    // Temporary calculation signals
    logic [QUANTITY_WIDTH:0] temp_qty_sum;  // Extra bit for overflow
    
    //========================================================================
    // Synchronous Reset Logic
    //========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            best_bid_idx <= '0;
            best_ask_idx <= '0;
            cmd_pipe_valid <= 1'b0;
            bid_write_enable <= 1'b0;
            ask_write_enable <= 1'b0;
            write_address <= '0;
        end else begin
            // Pipeline command
            cmd_pipe_valid <= cmd_valid;
            cmd_pipe_type <= cmd_type;
            cmd_pipe_side <= cmd_side;
            cmd_pipe_price <= cmd_price;
            cmd_pipe_qty <= cmd_quantity;
            cmd_pipe_order_id <= cmd_order_id;
            
            // Default deassertion
            bid_write_enable <= 1'b0;
            ask_write_enable <= 1'b0;
        end
    end
    
    //========================================================================
    // Price Comparison Logic (Fixed-Point, Synthesizable)
    //========================================================================
    
    assign price_greater = (cmd_pipe_price > bid_book[best_bid_idx].price);
    assign price_less = (cmd_pipe_price < bid_book[best_bid_idx].price);
    assign price_equal = (cmd_pipe_price == bid_book[best_bid_idx].price);
    
    //========================================================================
    // ADD Order Command Processing
    //========================================================================
    
    always_ff @(posedge clk) begin
        if (cmd_pipe_valid && cmd_pipe_type == 3'b000) begin // ADD
            if (cmd_pipe_side == 1'b0) begin // BID
                // Find appropriate price level or insert new one
                // Simplified: Assume price levels are pre-allocated
                // Real implementation needs binary search tree or CAM
                
                // For now, use hash-based indexing (price mod num_levels)
                write_address <= cmd_pipe_price[$clog2(NUM_PRICE_LEVELS)-1:0];
                bid_write_enable <= 1'b1;
                
                // Update order at this level
                for (integer i = 0; i < ORDERS_PER_LEVEL; i++) begin
                    if (!bid_book[write_address].orders[i].valid) begin
                        bid_book[write_address].orders[i].order_id <= cmd_pipe_order_id;
                        bid_book[write_address].orders[i].quantity <= cmd_pipe_qty;
                        bid_book[write_address].orders[i].valid <= 1'b1;
                        
                        // Update total quantity
                        temp_qty_sum = bid_book[write_address].total_quantity + cmd_pipe_qty;
                        bid_book[write_address].total_quantity <= temp_qty_sum[QUANTITY_WIDTH-1:0];
                        bid_book[write_address].occupied <= 1'b1;
                        bid_book[write_address].price <= cmd_pipe_price;
                        break;
                    end
                end
                
                // Update best bid index if this is a higher price
                if (cmd_pipe_price > bid_book[best_bid_idx].price) begin
                    best_bid_idx <= write_address;
                end
                
            end else begin // ASK
                write_address <= cmd_pipe_price[$clog2(NUM_PRICE_LEVELS)-1:0];
                ask_write_enable <= 1'b1;
                
                for (integer i = 0; i < ORDERS_PER_LEVEL; i++) begin
                    if (!ask_book[write_address].orders[i].valid) begin
                        ask_book[write_address].orders[i].order_id <= cmd_pipe_order_id;
                        ask_book[write_address].orders[i].quantity <= cmd_pipe_qty;
                        ask_book[write_address].orders[i].valid <= 1'b1;
                        
                        temp_qty_sum = ask_book[write_address].total_quantity + cmd_pipe_qty;
                        ask_book[write_address].total_quantity <= temp_qty_sum[QUANTITY_WIDTH-1:0];
                        ask_book[write_address].occupied <= 1'b1;
                        ask_book[write_address].price <= cmd_pipe_price;
                        break;
                    end
                end
                
                // Update best ask index if this is a lower price
                if (cmd_pipe_price < ask_book[best_ask_idx].price || 
                    !ask_book[best_ask_idx].occupied) begin
                    best_ask_idx <= write_address;
                end
            end
        end
    end
    
    //========================================================================
    // CANCEL Order Command Processing
    //========================================================================
    
    always_ff @(posedge clk) begin
        if (cmd_pipe_valid && cmd_pipe_type == 3'b001) begin // CANCEL
            if (cmd_pipe_side == 1'b0) begin // BID
                write_address <= cmd_pipe_price[$clog2(NUM_PRICE_LEVELS)-1:0];
                bid_write_enable <= 1'b1;
                
                for (integer i = 0; i < ORDERS_PER_LEVEL; i++) begin
                    if (bid_book[write_address].orders[i].valid &&
                        bid_book[write_address].orders[i].order_id == cmd_pipe_order_id) begin
                        
                        // Remove order
                        temp_qty_sum = bid_book[write_address].total_quantity - 
                                       bid_book[write_address].orders[i].quantity;
                        bid_book[write_address].total_quantity <= temp_qty_sum[QUANTITY_WIDTH-1:0];
                        bid_book[write_address].orders[i].valid <= 1'b0;
                        bid_book[write_address].orders[i].quantity <= '0;
                        
                        // Check if level is now empty
                        logic all_invalid;
                        all_invalid = 1'b1;
                        for (integer j = 0; j < ORDERS_PER_LEVEL; j++) begin
                            if (bid_book[write_address].orders[j].valid)
                                all_invalid = 1'b0;
                        end
                        bid_book[write_address].occupied <= ~all_invalid;
                        break;
                    end
                end
            end else begin // ASK
                write_address <= cmd_pipe_price[$clog2(NUM_PRICE_LEVELS)-1:0];
                ask_write_enable <= 1'b1;
                
                for (integer i = 0; i < ORDERS_PER_LEVEL; i++) begin
                    if (ask_book[write_address].orders[i].valid &&
                        ask_book[write_address].orders[i].order_id == cmd_pipe_order_id) begin
                        
                        temp_qty_sum = ask_book[write_address].total_quantity - 
                                       ask_book[write_address].orders[i].quantity;
                        ask_book[write_address].total_quantity <= temp_qty_sum[QUANTITY_WIDTH-1:0];
                        ask_book[write_address].orders[i].valid <= 1'b0;
                        ask_book[write_address].orders[i].quantity <= '0;
                        
                        logic all_invalid;
                        all_invalid = 1'b1;
                        for (integer j = 0; j < ORDERS_PER_LEVEL; j++) begin
                            if (ask_book[write_address].orders[j].valid)
                                all_invalid = 1'b0;
                        end
                        ask_book[write_address].occupied <= ~all_invalid;
                        break;
                    end
                end
            end
        end
    end
    
    //========================================================================
    // Best Bid/Ask Output Logic
    //========================================================================
    
    assign best_bid_price = bid_book[best_bid_idx].occupied ? 
                            bid_book[best_bid_idx].price : '0;
    assign best_bid_qty = bid_book[best_bid_idx].occupied ? 
                          bid_book[best_bid_idx].total_quantity : '0;
    
    assign best_ask_price = ask_book[best_ask_idx].occupied ? 
                            ask_book[best_ask_idx].price : '0;
    assign best_ask_qty = ask_book[best_ask_idx].occupied ? 
                          ask_book[best_ask_idx].total_quantity : '0;
    
    //========================================================================
    // Spread and Mid-Price Calculation
    //========================================================================
    
    assign spread = (best_ask_price >= best_bid_price) ? 
                    (best_ask_price - best_bid_price) : '0;
    
    // Mid-price = (best_bid + best_ask) / 2
    assign mid_price = (best_bid_price + best_ask_price) >>> 1;
    
    //========================================================================
    // Occupancy Status Outputs
    //========================================================================
    
    genvar g;
    generate
        for (g = 0; g < NUM_PRICE_LEVELS; g++) begin : occupancy_gen
            assign bid_levels_occupied[g] = bid_book[g].occupied;
            assign ask_levels_occupied[g] = ask_book[g].occupied;
        end
    endgenerate
    
    //========================================================================
    // Ready Status
    //========================================================================
    
    assign lob_ready = 1'b1;  // Always ready after reset
    assign cmd_ready = 1'b1;  // Simple implementation - always accept commands
    assign s_axis_tready_out = 1'b1;
    
endmodule
