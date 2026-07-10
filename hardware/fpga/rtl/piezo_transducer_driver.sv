// Piezoelectric Transducer Driver for Ultrasonic Phased Array
// FPGA RTL Implementation for NEXUS-OMEGA Stage 31
//
// This module generates precise phase-shifted PWM signals for driving
// hundreds of piezoelectric transducers in an ultrasonic phased array.
// Uses direct digital synthesis (DDS) for frequency generation and
// configurable phase accumulators for beam steering.

`timescale 1ns / 1ps

module piezo_transducer_driver #(
    parameter NUM_TRANSDUCERS = 256,
    parameter PHASE_BITS = 16,
    parameter FREQ_BITS = 32,
    parameter PWM_BITS = 10,
    parameter CLK_FREQ = 100_000_000,      // 100 MHz system clock
    parameter MAX_FREQUENCY = 10_000_000   // 10 MHz max ultrasonic freq
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    
    // Configuration interface
    input wire config_valid,
    input wire [7:0] transducer_addr,
    input wire [PHASE_BITS-1:0] phase_data,
    input wire [PWM_BITS-1:0] amplitude_data,
    input wire write_enable,
    
    // Frequency control (global for all transducers)
    input wire [FREQ_BITS-1:0] frequency_word,
    input wire frequency_update,
    
    // Phase lock reference
    input wire phase_lock_ref,
    input wire phase_lock_enable,
    
    // Status outputs
    output reg phase_locked,
    output reg [15:0] error_status,
    
    // Transducer drive outputs (serialized for high pin count)
    output wire [NUM_TRANSDUCERS-1:0] pwm_out,
    output wire data_valid
);

    // =========================================================================
    // Internal Registers and Wires
    // =========================================================================
    
    // Phase accumulator for each transducer (distributed RAM inference)
    reg [PHASE_BITS-1:0] phase_accum [0:NUM_TRANSDUCERS-1];
    reg [PWM_BITS-1:0] amplitude_reg [0:NUM_TRANSDUCERS-1];
    reg [PHASE_BITS-1:0] phase_offset [0:NUM_TRANSDUCERS-1];
    
    // Global phase accumulator (reference)
    reg [FREQ_BITS-1:0] global_phase_acc;
    
    // Configuration state machine
    localparam CFG_IDLE = 2'b00;
    localparam CFG_WRITE = 2'b01;
    localparam CFG_VERIFY = 2'b10;
    localparam CFG_DONE = 2'b11;
    
    reg [1:0] cfg_state;
    reg [7:0] cfg_counter;
    
    // Phase lock monitoring
    reg [31:0] lock_timer;
    reg [15:0] phase_error_accum;
    
    // Output pipeline
    reg [NUM_TRANSDUCERS-1:0] pwm_out_reg;
    reg data_valid_reg;
    
    // =========================================================================
    // Global Phase Accumulator (DDS Core)
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            global_phase_acc <= {FREQ_BITS{1'b0}};
        end else if (enable) begin
            if (frequency_update) begin
                global_phase_acc <= global_phase_acc + frequency_word;
            end else begin
                global_phase_acc <= global_phase_acc + frequency_word;
            end
        end
    end
    
    // Extract top bits for sine lookup (or direct PWM comparison)
    wire [PHASE_BITS-1:0] global_phase = global_phase_acc[FREQ_BITS-1:FREQ_BITS-PHASE_BITS];
    
    // =========================================================================
    // Configuration Write Interface
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cfg_state <= CFG_IDLE;
            cfg_counter <= 8'd0;
        end else begin
            case (cfg_state)
                CFG_IDLE: begin
                    if (config_valid && write_enable) begin
                        cfg_state <= CFG_WRITE;
                        cfg_counter <= 8'd0;
                    end
                end
                
                CFG_WRITE: begin
                    // Write phase offset to addressed transducer
                    if (transducer_addr < NUM_TRANSDUCERS) begin
                        phase_offset[transducer_addr] <= phase_data;
                        amplitude_reg[transducer_addr] <= amplitude_data;
                    end
                    cfg_state <= CFG_VERIFY;
                end
                
                CFG_VERIFY: begin
                    // Verify write (could read back in real implementation)
                    cfg_state <= CFG_DONE;
                end
                
                CFG_DONE: begin
                    cfg_state <= CFG_IDLE;
                end
            endcase
        end
    end
    
    // =========================================================================
    // Per-Transducer Phase Accumulators
    // =========================================================================
    
    genvar i;
    generate
        for (i = 0; i < NUM_TRANSDUCERS; i = i + 1) begin : transducer_array
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    phase_accum[i] <= {PHASE_BITS{1'b0}};
                end else if (enable) begin
                    // Each transducer has its own phase accumulator
                    // with individual phase offset for beam steering
                    phase_accum[i] <= phase_accum[i] + frequency_word[PHASE_BITS-1:0];
                    
                    // Apply phase offset (wraps automatically)
                    if (phase_accum[i] >= (1 << PHASE_BITS) - phase_offset[i]) begin
                        phase_accum[i] <= phase_accum[i] - (1 << PHASE_BITS);
                    end
                end
            end
        end
    endgenerate
    
    // =========================================================================
    // PWM Generation (Comparator-based)
    // =========================================================================
    
    reg [PWM_BITS-1:0] pwm_counter;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pwm_counter <= {PWM_BITS{1'b0}};
        end else begin
            pwm_counter <= pwm_counter + 1'b1;
        end
    end
    
    // Generate PWM for each transducer
    generate
        for (i = 0; i < NUM_TRANSDUCERS; i = i + 1) begin : pwm_gen
            assign pwm_out[i] = (pwm_counter < amplitude_reg[i]) && 
                                (phase_accum[i][PHASE_BITS-1] == 1'b1) &&
                                enable;
        end
    endgenerate
    
    // =========================================================================
    // Phase Lock Monitoring
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase_locked <= 1'b0;
            lock_timer <= 32'd0;
            phase_error_accum <= 16'd0;
        end else if (phase_lock_enable) begin
            // Monitor phase relationship with reference
            if (phase_lock_ref) begin
                // Check if global phase is within tolerance of reference
                if (global_phase_acc[FREQ_BITS-1:FREQ_BITS-8] < 8'd10 || 
                    global_phase_acc[FREQ_BITS-1:FREQ_BITS-8] > 8'd246) begin
                    phase_error_accum <= phase_error_accum + 1'b1;
                end
                
                // Lock achieved if error accumulation is low
                if (phase_error_accum < 16'd100) begin
                    phase_locked <= 1'b1;
                end else begin
                    phase_locked <= 1'b0;
                end
                
                // Reset error accumulator periodically
                if (lock_timer >= 32'd100000) begin
                    lock_timer <= 32'd0;
                    phase_error_accum <= 16'd0;
                end else begin
                    lock_timer <= lock_timer + 1'b1;
                end
            end
        end else begin
            phase_locked <= 1'b0;
        end
    end
    
    // =========================================================================
    // Error Status Register
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            error_status <= 16'd0;
        end else begin
            error_status[0] <= ~enable;           // Bit 0: Disabled
            error_status[1] <= ~phase_locked;     // Bit 1: Not locked
            error_status[2] <= (cfg_state == CFG_WRITE);  // Bit 2: Configuring
            error_status[3] <= (frequency_word > MAX_FREQUENCY); // Bit 3: Freq overflow
            error_status[4] <= (amplitude_data > 1023); // Bit 4: Amplitude overflow
            // Bits 5-15 reserved
        end
    end
    
    // =========================================================================
    // Data Valid Output
    // =========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_valid_reg <= 1'b0;
        end else begin
            // Pulse data_valid when PWM outputs are stable
            data_valid_reg <= enable && phase_locked;
        end
    end
    
    assign pwm_out = pwm_out_reg;
    assign data_valid = data_valid_reg;
    
endmodule

// =========================================================================
// Testbench for Piezoelectric Transducer Driver
// =========================================================================

`ifdef SIMULATION

module piezo_transducer_driver_tb;

    localparam NUM_TRANSDUCERS = 16;
    localparam PHASE_BITS = 16;
    localparam FREQ_BITS = 32;
    localparam PWM_BITS = 10;
    
    reg clk;
    reg rst_n;
    reg enable;
    reg config_valid;
    reg [7:0] transducer_addr;
    reg [PHASE_BITS-1:0] phase_data;
    reg [PWM_BITS-1:0] amplitude_data;
    reg write_enable;
    reg [FREQ_BITS-1:0] frequency_word;
    reg frequency_update;
    reg phase_lock_ref;
    reg phase_lock_enable;
    
    wire phase_locked;
    wire [15:0] error_status;
    wire [NUM_TRANSDUCERS-1:0] pwm_out;
    wire data_valid;
    
    // Instantiate DUT
    piezo_transducer_driver #(
        .NUM_TRANSDUCERS(NUM_TRANSDUCERS),
        .PHASE_BITS(PHASE_BITS),
        .FREQ_BITS(FREQ_BITS),
        .PWM_BITS(PWM_BITS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .config_valid(config_valid),
        .transducer_addr(transducer_addr),
        .phase_data(phase_data),
        .amplitude_data(amplitude_data),
        .write_enable(write_enable),
        .frequency_word(frequency_word),
        .frequency_update(frequency_update),
        .phase_lock_ref(phase_lock_ref),
        .phase_lock_enable(phase_lock_enable),
        .phase_locked(phase_locked),
        .error_status(error_status),
        .pwm_out(pwm_out),
        .data_valid(data_valid)
    );
    
    // Clock generation (100 MHz)
    initial clk = 0;
    always #5 clk = ~clk;
    
    // Test sequence
    initial begin
        $display("Starting Piezo Transducer Driver Testbench");
        
        // Reset
        rst_n = 0;
        enable = 0;
        #100;
        rst_n = 1;
        #50;
        
        // Enable module
        enable = 1;
        
        // Configure frequency (40 kHz target)
        // frequency_word = (target_freq / clk_freq) * 2^FREQ_BITS
        frequency_word = 32'd1717987;  // ~40 kHz
        frequency_update = 1;
        #10;
        frequency_update = 0;
        
        // Configure phase offsets for beam steering
        config_valid = 1;
        write_enable = 1;
        
        for (integer i = 0; i < NUM_TRANSDUCERS; i = i + 1) begin
            transducer_addr = i;
            phase_data = i * 1000;  // Progressive phase shift
            amplitude_data = 512;   // 50% duty cycle
            #10;
        end
        
        config_valid = 0;
        write_enable = 0;
        
        // Enable phase locking
        phase_lock_enable = 1;
        phase_lock_ref = 1;
        
        // Run simulation
        #10000;
        
        $display("Testbench Complete");
        $finish;
    end
    
    // Monitoring
    always @(posedge phase_locked) begin
        $display("[%0t] Phase Locked", $time);
    end
    
    always @(negedge phase_locked) begin
        $display("[%0t] Phase Unlocked", $time);
    end
    
endmodule

`endif
