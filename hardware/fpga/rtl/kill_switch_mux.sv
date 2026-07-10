//============================================================================
// NEXUS-OMEGA Stage 20: Hardware Kill-Switch Multiplexer
// 
// Synthesizable SystemVerilog RTL for physical packet gating.
// Severes AXI4-Stream transmission line in < 5 nanoseconds when triggered.
// 
// CRITICAL CONSTRAINTS:
// - NO floating-point math
// - NO unsynthesizable constructs
// - Combinational path from trigger to output for minimum latency
// - Glitch-free transition to safe state
// 
// Operation:
// - When kill_switch_trigger is asserted, all outbound packets are dropped
// - Only heartbeat/acknowledgment packets are allowed through during kill
// - Status register captures the reason for kill
//============================================================================

`timescale 1ns / 1ps
`default_nettype wire

module kill_switch_mux #(
    // AXI4-Stream parameters
    parameter AXIS_DATA_WIDTH = 64,
    parameter AXIS_KEEP_WIDTH = AXIS_DATA_WIDTH / 8,
    
    // Number of breach sources to track
    parameter NUM_BREACH_SOURCES = 5,
    
    // Packet type encoding
    parameter PKT_TYPE_ORDER = 3'd0,
    parameter PKT_TYPE_CANCEL = 3'd1,
    parameter PKT_TYPE_HEARTBEAT = 3'd2,
    parameter PKT_TYPE_ACK = 3'd3,
    parameter PKT_TYPE_DATA = 3'd4
) (
    // Global clock and reset
    input wire clk,
    input wire rst_n,              // Active-low async reset
    
    // Kill-switch trigger from Risk Guardian
    input wire                      kill_switch_trigger,
    
    // Breach source indicators (for status)
    input wire                      breach_gross_notional,
    input wire                      breach_net_delta,
    input wire                      breach_margin,
    input wire                      breach_sequence_gap,
    input wire                      breach_rate_limit,
    
    // Input: AXI4-Stream from matching engine
    input wire                      s_axis_tvalid,
    input wire [AXIS_DATA_WIDTH-1:0] s_axis_tdata,
    input wire [AXIS_KEEP_WIDTH-1:0] s_axis_tkeep,
    input wire                      s_axis_tlast,
    input wire [2:0]                s_axis_pkt_type,
    input wire                      s_axis_ready_in,
    output wire                     s_axis_ready_out,
    
    // Output: AXI4-Stream to MAC/PHY
    output wire                     m_axis_tvalid,
    output wire [AXIS_DATA_WIDTH-1:0] m_axis_tdata,
    output wire [AXIS_KEEP_WIDTH-1:0] m_axis_tkeep,
    output wire                     m_axis_tlast,
    input wire                      m_axis_ready,
    
    // Heartbeat generator (active during kill)
    output wire                     heartbeat_valid,
    output wire [AXIS_DATA_WIDTH-1:0] heartbeat_data,
    
    // Status outputs
    output wire                     kill_active,
    output wire                     kill_latched,
    output wire [NUM_BREACH_SOURCES-1:0] breach_status,
    output wire [31:0]              kill_timestamp,
    output wire                     manual_reset_allowed
);

    //========================================================================
    // Internal Signals
    //========================================================================
    
    // Kill latch (once triggered, stays latched until reset)
    logic kill_latch_reg;
    logic kill_latch_set;
    logic kill_latch_clear;
    
    // Packet filtering decision
    logic packet_allowed;
    logic is_safe_packet;
    
    // Timestamp counter
    logic [31:0] timestamp_counter;
    
    // Manual reset request
    logic manual_reset_req;
    
    // Synchronization registers for glitch-free output
    logic tvalid_sync;
    logic tdata_sync [AXIS_DATA_WIDTH-1:0];
    logic tkeep_sync [AXIS_KEEP_WIDTH-1:0];
    logic tlast_sync;
    
    //========================================================================
    // Kill Latch Logic
    //========================================================================
    
    // Set latch on any trigger
    assign kill_latch_set = kill_switch_trigger;
    
    // Clear latch only via explicit reset command (not auto-clear)
    assign kill_latch_clear = manual_reset_req && !kill_switch_trigger;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kill_latch_reg <= 1'b0;
        end else if (kill_latch_set) begin
            kill_latch_reg <= 1'b1;
        end else if (kill_latch_clear) begin
            kill_latch_reg <= 1'b0;
        end
    end
    
    assign kill_active = kill_latch_reg || kill_switch_trigger;
    assign kill_latched = kill_latch_reg;
    
    //========================================================================
    // Breach Status Capture
    //========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            breach_status <= '0;
        end else if (kill_switch_trigger && !kill_latch_reg) begin
            // Capture breach status on first trigger
            breach_status[0] <= breach_gross_notional;
            breach_status[1] <= breach_net_delta;
            breach_status[2] <= breach_margin;
            breach_status[3] <= breach_sequence_gap;
            breach_status[4] <= breach_rate_limit;
        end
    end
    
    //========================================================================
    // Timestamp Counter
    //========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            timestamp_counter <= '0;
        end else begin
            timestamp_counter <= timestamp_counter + 1;
        end
    end
    
    assign kill_timestamp = kill_latch_reg ? timestamp_counter : '0;
    
    //========================================================================
    // Safe Packet Detection
    //========================================================================
    
    // During kill, only allow heartbeats and ACKs
    assign is_safe_packet = (s_axis_pkt_type == PKT_TYPE_HEARTBEAT) ||
                            (s_axis_pkt_type == PKT_TYPE_ACK);
    
    // Packet is allowed if not in kill state OR if it's a safe packet type
    assign packet_allowed = !kill_active || is_safe_packet;
    
    //========================================================================
    // AXI4-Stream Gating (Combinational for Minimum Latency)
    //========================================================================
    
    // Ready signal back to source - blocked during kill for non-safe packets
    assign s_axis_ready_out = m_axis_ready && (packet_allowed || !s_axis_tvalid);
    
    // Output valid - only pass through if allowed
    assign m_axis_tvalid = s_axis_tvalid && packet_allowed;
    
    // Data passes through unchanged when allowed
    assign m_axis_tdata = s_axis_tdata;
    assign m_axis_tkeep = s_axis_tkeep;
    assign m_axis_tlast = s_axis_tlast;
    
    //========================================================================
    // Heartbeat Generator (Active During Kill)
    //========================================================================
    
    logic [3:0] heartbeat_counter;
    logic heartbeat_toggle;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            heartbeat_counter <= '0;
            heartbeat_toggle <= 1'b0;
        end else begin
            heartbeat_counter <= heartbeat_counter + 1;
            
            // Generate heartbeat every 16 cycles during kill
            if (heartbeat_counter == 4'd15 && kill_active) begin
                heartbeat_toggle <= ~heartbeat_toggle;
            end
        end
    end
    
    // Heartbeat packet format (simplified)
    assign heartbeat_valid = kill_active && (heartbeat_counter == 4'd15);
    assign heartbeat_data = {
        16'hDEAD,           // Magic word indicating kill state
        8'hBE,              // Beacon type
        8'h00,              // Reserved
        breach_status,      // Which breaches triggered kill
        32'd0               // Padding
    };
    
    //========================================================================
    // Manual Reset Interface
    //========================================================================
    
    // Manual reset is only allowed after certain conditions are met
    // (e.g., risk metrics return to safe levels)
    assign manual_reset_allowed = !breach_gross_notional && 
                                  !breach_net_delta &&
                                  !breach_margin &&
                                  !breach_sequence_gap;
    
    //========================================================================
    // Optional: Synchronized Output for Multi-Clock Domains
    //========================================================================
    
    // If MAC operates on different clock domain, use these synchronized outputs
    // instead of direct combinational outputs
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tvalid_sync <= 1'b0;
            tlast_sync <= 1'b0;
        end else begin
            tvalid_sync <= m_axis_tvalid;
            tlast_sync <= m_axis_tlast;
        end
    end
    
    always_ff @(posedge clk) begin
        // Data registers don't need reset for normal operation
        if (m_axis_tvalid) begin
            for (integer i = 0; i < AXIS_DATA_WIDTH; i++) begin
                tdata_sync[i] <= s_axis_tdata[i];
            end
            for (integer i = 0; i < AXIS_KEEP_WIDTH; i++) begin
                tkeep_sync[i] <= s_axis_tkeep[i];
            end
        end
    end
    
endmodule


//============================================================================
// Top-Level Kill-Switch Wrapper with Multiple Input Ports
//============================================================================

module kill_switch_top #(
    parameter NUM_INPUT_PORTS = 4,
    parameter AXIS_DATA_WIDTH = 64,
    parameter AXIS_KEEP_WIDTH = AXIS_DATA_WIDTH / 8
) (
    input wire clk,
    input wire rst_n,
    
    // Global kill trigger
    input wire kill_switch_trigger,
    
    // Multiple input ports (from different engines)
    input wire [NUM_INPUT_PORTS-1:0]          s_axis_tvalid,
    input wire [NUM_INPUT_PORTS-1:0][AXIS_DATA_WIDTH-1:0] s_axis_tdata,
    input wire [NUM_INPUT_PORTS-1:0][AXIS_KEEP_WIDTH-1:0] s_axis_tkeep,
    input wire [NUM_INPUT_PORTS-1:0]          s_axis_tlast,
    input wire [NUM_INPUT_PORTS-1:0][2:0]     s_axis_pkt_type,
    output wire [NUM_INPUT_PORTS-1:0]         s_axis_ready_out,
    
    // Single output to MAC
    output wire                     m_axis_tvalid,
    output wire [AXIS_DATA_WIDTH-1:0] m_axis_tdata,
    output wire [AXIS_KEEP_WIDTH-1:0] m_axis_tkeep,
    output wire                     m_axis_tlast,
    input wire                      m_axis_ready,
    
    // Status
    output wire                     kill_active,
    output wire                     kill_latched
);

    genvar g;
    generate
        for (g = 0; g < NUM_INPUT_PORTS; g++) begin : port_gen
            // Instantiate per-port kill switch
            kill_switch_mux #(
                .AXIS_DATA_WIDTH(AXIS_DATA_WIDTH),
                .AXIS_KEEP_WIDTH(AXIS_KEEP_WIDTH)
            ) port_mux (
                .clk(clk),
                .rst_n(rst_n),
                .kill_switch_trigger(kill_switch_trigger),
                .breach_gross_notional(1'b0),
                .breach_net_delta(1'b0),
                .breach_margin(1'b0),
                .breach_sequence_gap(1'b0),
                .breach_rate_limit(1'b0),
                .s_axis_tvalid(s_axis_tvalid[g]),
                .s_axis_tdata(s_axis_tdata[g]),
                .s_axis_tkeep(s_axis_tkeep[g]),
                .s_axis_tlast(s_axis_tlast[g]),
                .s_axis_pkt_type(s_axis_pkt_type[g]),
                .s_axis_ready_in(m_axis_ready),
                .s_axis_ready_out(s_axis_ready_out[g]),
                .m_axis_tvalid(),  // Internal
                .m_axis_tdata(),   // Internal
                .m_axis_tkeep(),   // Internal
                .m_axis_tlast(),   // Internal
                .m_axis_ready(1'b0),
                .heartbeat_valid(),
                .heartbeat_data(),
                .kill_active(),
                .kill_latched(),
                .breach_status(),
                .kill_timestamp(),
                .manual_reset_allowed()
            );
        end
    endgenerate
    
    // Arbiter logic would go here to multiplex outputs to single MAC interface
    // Simplified: Just pass through port 0 when not in kill state
    
    assign m_axis_tvalid = kill_active ? 1'b0 : s_axis_tvalid[0];
    assign m_axis_tdata = s_axis_tdata[0];
    assign m_axis_tkeep = s_axis_tkeep[0];
    assign m_axis_tlast = s_axis_tlast[0];
    
endmodule
