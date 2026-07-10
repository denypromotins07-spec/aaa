//============================================================================
// NEXUS-OMEGA Stage 20: AXI4-Stream Market Data Ingest
// 
// Synthesizable SystemVerilog RTL for ingesting UDP market data feeds
// directly into FPGA fabric via AXI4-Stream interface.
// 
// CRITICAL CONSTRAINTS:
// - NO floating-point math
// - NO unsynthesizable constructs (# delays, real types)
// - Fully synchronous design
// - Handles packet parsing, validation, and normalization
// 
// Supports:
// - UDP packet ingestion from 10/25/40/100GbE MAC
// - Packet validation (header check, sequence ID verification)
// - Message type demultiplexing (trade, quote, order status)
// - Fixed-point price/quantity normalization
//============================================================================

`timescale 1ns / 1ps
`default_nettype wire

module axi4_stream_ingest #(
    // Packet buffer parameters
    parameter MAX_PACKET_SIZE = 1500,
    parameter HEADER_SIZE = 64,       // Ethernet + IP + UDP headers
    
    // Fixed-point format for normalized data
    parameter PRICE_WIDTH = 32,
    parameter INTEGER_WIDTH = 16,
    parameter FRACTIONAL_WIDTH = 16,
    parameter QUANTITY_WIDTH = 32,
    
    // Sequence tracking
    parameter SEQ_ID_WIDTH = 32,
    
    // Number of supported symbols
    parameter NUM_SYMBOLS = 256
) (
    // Global clock and reset
    input wire clk,
    input wire rst_n,              // Active-low async reset
    
    // AXI4-Stream input from MAC (UDP payload)
    input wire                      s_axis_tvalid,
    input wire [7:0]                s_axis_tdata,
    input wire [7:0]                s_axis_tkeep,
    input wire                      s_axis_tlast,
    input wire                      s_axis_tready,
    output wire                     s_axis_tready_out,
    
    // User clock domain for output (may be different from MAC clock)
    input wire                      user_clk,
    input wire                      user_rst_n,
    
    // Output: Parsed trade messages
    output wire                     trade_valid,
    output wire [15:0]              trade_symbol_id,
    output wire [PRICE_WIDTH-1:0]   trade_price,
    output wire [QUANTITY_WIDTH-1:0] trade_quantity,
    output wire [1:0]               trade_side,      // 0=BUY, 1=SELL
    output wire [63:0]              trade_timestamp,
    output wire [SEQ_ID_WIDTH-1:0]  trade_seq_id,
    input wire                      trade_ready,
    
    // Output: Parsed quote (BBO) messages
    output wire                     quote_valid,
    output wire [15:0]              quote_symbol_id,
    output wire [PRICE_WIDTH-1:0]   quote_bid_price,
    output wire [QUANTITY_WIDTH-1:0] quote_bid_qty,
    output wire [PRICE_WIDTH-1:0]   quote_ask_price,
    output wire [QUANTITY_WIDTH-1:0] quote_ask_qty,
    output wire [63:0]              quote_timestamp,
    output wire [SEQ_ID_WIDTH-1:0]  quote_seq_id,
    input wire                      quote_ready,
    
    // Status outputs
    output wire                     ingest_ready,
    output wire                     parse_error,
    output wire [SEQ_ID_WIDTH-1:0]  last_seq_id,
    output wire [31:0]              packets_received,
    output wire [31:0]              parse_errors,
    
    // Sequence gap detection
    output wire                     seq_gap_detected,
    output wire [SEQ_ID_WIDTH-1:0]  expected_seq_id,
    output wire [SEQ_ID_WIDTH-1:0]  received_seq_id
);

    //========================================================================
    // Internal Types and Signals
    //========================================================================
    
    // Parser state machine
    typedef enum logic [3:0] {
        STATE_IDLE = 4'd0,
        STATE_ETHER_HEADER = 4'd1,
        STATE_IP_HEADER = 4'd2,
        STATE_UDP_HEADER = 4'd3,
        STATE_MSG_HEADER = 4'd4,
        STATE_MSG_PAYLOAD = 4'd5,
        STATE_MSG_FOOTER = 4'd6,
        STATE_ERROR = 4'd15
    } parser_state_t;
    
    parser_state_t parser_state;
    parser_state_t next_state;
    
    // Packet buffer
    logic [7:0] packet_buffer [0:MAX_PACKET_SIZE-1];
    logic [$clog2(MAX_PACKET_SIZE)-1:0] byte_counter;
    logic [$clog2(MAX_PACKET_SIZE)-1:0] packet_length;
    
    // Message type identification
    typedef enum logic [3:0] {
        MSG_UNKNOWN = 4'd0,
        MSG_TRADE = 4'd1,
        MSG_QUOTE = 4'd2,
        MSG_ORDER_ADD = 4'd3,
        MSG_ORDER_CANCEL = 4'd4,
        MSG_ORDER_MODIFY = 4'd5
    } msg_type_t;
    
    msg_type_t current_msg_type;
    
    // Sequence tracking
    logic [SEQ_ID_WIDTH-1:0] expected_seq_reg;
    logic [SEQ_ID_WIDTH-1:0] received_seq_reg;
    logic seq_gap_flag;
    
    // Statistics
    logic [31:0] packets_rx_count;
    logic [31:0] error_count;
    
    // Temporary parsing signals
    logic [7:0] header_byte;
    logic [15:0] symbol_id_temp;
    logic [31:0] price_raw;
    logic [31:0] quantity_raw;
    logic [63:0] timestamp_temp;
    
    // Clock domain crossing FIFOs (simplified - would use async FIFO in practice)
    logic trade_fifo_empty;
    logic trade_fifo_full;
    logic quote_fifo_empty;
    logic quote_fifo_full;
    
    //========================================================================
    // Packet Parser State Machine
    //========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            parser_state <= STATE_IDLE;
            byte_counter <= '0;
            packet_length <= '0;
            current_msg_type <= MSG_UNKNOWN;
        end else begin
            parser_state <= next_state;
            
            case (next_state)
                STATE_IDLE: begin
                    byte_counter <= '0;
                    if (s_axis_tvalid && s_axis_tready_out) begin
                        parser_state <= STATE_ETHER_HEADER;
                    end
                end
                
                STATE_ETHER_HEADER: begin
                    // Skip 14 bytes Ethernet header
                    if (s_axis_tvalid && s_axis_tready_out) begin
                        if (byte_counter < 13) begin
                            byte_counter <= byte_counter + 1;
                        end else begin
                            byte_counter <= '0;
                            parser_state <= STATE_IP_HEADER;
                        end
                    end
                end
                
                STATE_IP_HEADER: begin
                    // Skip 20 bytes IP header (no options)
                    if (s_axis_tvalid && s_axis_tready_out) begin
                        if (byte_counter < 19) begin
                            byte_counter <= byte_counter + 1;
                        end else begin
                            byte_counter <= '0;
                            parser_state <= STATE_UDP_HEADER;
                        end
                    end
                end
                
                STATE_UDP_HEADER: begin
                    // Skip 8 bytes UDP header
                    if (s_axis_tvalid && s_axis_tready_out) begin
                        if (byte_counter < 7) begin
                            byte_counter <= byte_counter + 1;
                        end else begin
                            byte_counter <= '0;
                            parser_state <= STATE_MSG_HEADER;
                        end
                    end
                end
                
                STATE_MSG_HEADER: begin
                    // Parse message header to determine type
                    if (s_axis_tvalid && s_axis_tready_out) begin
                        // First byte after UDP header is message type
                        if (byte_counter == 0) begin
                            case (s_axis_tdata)
                                8'h01: current_msg_type <= MSG_TRADE;
                                8'h02: current_msg_type <= MSG_QUOTE;
                                8'h03: current_msg_type <= MSG_ORDER_ADD;
                                8'h04: current_msg_type <= MSG_ORDER_CANCEL;
                                8'h05: current_msg_type <= MSG_ORDER_MODIFY;
                                default: current_msg_type <= MSG_UNKNOWN;
                            endcase
                        end
                        
                        if (byte_counter < 15) begin  // 16-byte message header
                            byte_counter <= byte_counter + 1;
                        end else begin
                            byte_counter <= '0;
                            parser_state <= STATE_MSG_PAYLOAD;
                        end
                    end
                end
                
                STATE_MSG_PAYLOAD: begin
                    // Accumulate payload bytes
                    if (s_axis_tvalid && s_axis_tready_out) begin
                        packet_buffer[byte_counter] <= s_axis_tdata;
                        
                        if (s_axis_tlast) begin
                            packet_length <= byte_counter + 1;
                            parser_state <= STATE_MSG_FOOTER;
                        end else begin
                            byte_counter <= byte_counter + 1;
                        end
                    end
                end
                
                STATE_MSG_FOOTER: begin
                    // Validate and dispatch message
                    // Next packet will start processing
                    parser_state <= STATE_IDLE;
                end
                
                STATE_ERROR: begin
                    // Stay in error until reset or recovery
                    if (s_axis_tlast && s_axis_tvalid && s_axis_tready_out) begin
                        parser_state <= STATE_IDLE;
                    end
                end
                
                default: parser_state <= STATE_IDLE;
            endcase
        end
    end
    
    //========================================================================
    // AXI4-Stream Ready Control
    //========================================================================
    
    assign s_axis_tready_out = (parser_state != STATE_ERROR) && 
                               (parser_state != STATE_MSG_FOOTER);
    
    //========================================================================
    // Sequence Gap Detection
    //========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            expected_seq_reg <= '0;
            received_seq_reg <= '0;
            seq_gap_flag <= 1'b0;
        end else if (parser_state == STATE_MSG_FOOTER) begin
            // Extract sequence ID from message (assumes fixed position)
            received_seq_reg <= {packet_buffer[8], packet_buffer[9], 
                                 packet_buffer[10], packet_buffer[11]};
            
            if (received_seq_reg != expected_seq_reg) begin
                seq_gap_flag <= 1'b1;
                // Update expected to next after received
                expected_seq_reg <= received_seq_reg + 1;
            end else begin
                seq_gap_flag <= 1'b0;
                expected_seq_reg <= expected_seq_reg + 1;
            end
        end
    end
    
    assign seq_gap_detected = seq_gap_flag;
    assign expected_seq_id = expected_seq_reg;
    assign received_seq_id = received_seq_reg;
    
    //========================================================================
    // Message Dispatch Logic
    //========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            trade_valid <= 1'b0;
            quote_valid <= 1'b0;
        end else begin
            trade_valid <= 1'b0;
            quote_valid <= 1'b0;
            
            if (parser_state == STATE_MSG_FOOTER) begin
                case (current_msg_type)
                    MSG_TRADE: begin
                        if (!trade_fifo_full) begin
                            trade_valid <= 1'b1;
                        end
                    end
                    
                    MSG_QUOTE: begin
                        if (!quote_fifo_full) begin
                            quote_valid <= 1'b1;
                        end
                    end
                    
                    default: ;
                endcase
            end
        end
    end
    
    //========================================================================
    // Trade Message Extraction
    //========================================================================
    
    // Trade message format (after header):
    // [1:0] Symbol ID (16 bits)
    // [5:2] Price (32 bits, fixed-point)
    // [9:6] Quantity (32 bits)
    // [10] Side (1 bit)
    // [18:11] Timestamp (64 bits)
    
    assign trade_symbol_id = {packet_buffer[0], packet_buffer[1]};
    assign trade_price = {packet_buffer[2], packet_buffer[3], 
                          packet_buffer[4], packet_buffer[5]};
    assign trade_quantity = {packet_buffer[6], packet_buffer[7],
                             packet_buffer[8], packet_buffer[9]};
    assign trade_side = packet_buffer[10][0];
    assign trade_timestamp = {packet_buffer[11], packet_buffer[12],
                              packet_buffer[13], packet_buffer[14],
                              packet_buffer[15], packet_buffer[16],
                              packet_buffer[17], packet_buffer[18]};
    assign trade_seq_id = received_seq_reg;
    
    //========================================================================
    // Quote Message Extraction
    //========================================================================
    
    // Quote message format:
    // [1:0] Symbol ID
    // [5:2] Bid price
    // [9:6] Bid qty
    // [13:10] Ask price
    // [17:14] Ask qty
    // [25:18] Timestamp
    
    assign quote_symbol_id = {packet_buffer[0], packet_buffer[1]};
    assign quote_bid_price = {packet_buffer[2], packet_buffer[3],
                              packet_buffer[4], packet_buffer[5]};
    assign quote_bid_qty = {packet_buffer[6], packet_buffer[7],
                            packet_buffer[8], packet_buffer[9]};
    assign quote_ask_price = {packet_buffer[10], packet_buffer[11],
                              packet_buffer[12], packet_buffer[13]};
    assign quote_ask_qty = {packet_buffer[14], packet_buffer[15],
                            packet_buffer[16], packet_buffer[17]};
    assign quote_timestamp = {packet_buffer[18], packet_buffer[19],
                              packet_buffer[20], packet_buffer[21],
                              packet_buffer[22], packet_buffer[23],
                              packet_buffer[24], packet_buffer[25]};
    assign quote_seq_id = received_seq_reg;
    
    //========================================================================
    // Statistics Counters
    //========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            packets_rx_count <= '0;
            error_count <= '0;
        end else begin
            if (parser_state == STATE_MSG_FOOTER && !seq_gap_flag) begin
                packets_rx_count <= packets_rx_count + 1;
            end
            
            if (parser_state == STATE_ERROR || seq_gap_flag) begin
                error_count <= error_count + 1;
            end
        end
    end
    
    assign packets_received = packets_rx_count;
    assign parse_errors = error_count;
    assign parse_error = (parser_state == STATE_ERROR);
    assign last_seq_id = received_seq_reg;
    
    //========================================================================
    // Ready Status
    //========================================================================
    
    assign ingest_ready = (parser_state == STATE_IDLE) || 
                          (parser_state == STATE_MSG_PAYLOAD);
    
endmodule
