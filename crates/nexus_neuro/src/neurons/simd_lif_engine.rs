//! SIMD-Accelerated Leaky Integrate-and-Fire (LIF) Neuron Engine
//! 
//! Implements vectorized LIF neuron state updates using Rust's std::simd
//! for AVX2/AVX-512 acceleration. Updates 8-16 neurons simultaneously.
//!
//! LIF Differential Equation: τ * dV/dt = -(V - V_rest) + R * I(t)
//! Discrete update: V[t+1] = V[t] + dt/τ * (-(V[t] - V_rest) + R * I[t])

use std::arch::x86_64::*;
use std::simd::{f32x8, f32x16, num::SimdFloat, Simd, Mask};
use std::sync::atomic::{AtomicU64, Ordering};

/// SIMD lane width for AVX2 (8 x f32)
pub const SIMD_LANE_WIDTH_AVX2: usize = 8;

/// SIMD lane width for AVX-512 (16 x f32)
pub const SIMD_LANE_WIDTH_AVX512: usize = 16;

/// Default membrane time constant τ (milliseconds)
pub const DEFAULT_TAU_MS: f32 = 20.0;

/// Default resting potential (mV)
pub const DEFAULT_V_REST: f32 = -70.0;

/// Default spike threshold (mV)
pub const DEFAULT_V_THRESHOLD: f32 = -55.0;

/// Default reset potential after spike (mV)
pub const DEFAULT_V_RESET: f32 = -75.0;

/// Default refractory period (microseconds)
pub const DEFAULT_REFRACTORY_US: u32 = 2000;

/// State of a single LIF neuron
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct LifNeuronState {
    /// Membrane potential (mV)
    pub v_mem: f32,
    /// Resting potential (mV)
    pub v_rest: f32,
    /// Spike threshold (mV)
    pub v_threshold: f32,
    /// Reset potential (mV)
    pub v_reset: f32,
    /// Membrane time constant τ (ms)
    pub tau_ms: f32,
    /// Refractory timer (μs remaining)
    pub refractory_timer_us: u32,
    /// Last spike timestamp (μs)
    pub last_spike_ts_us: u64,
    /// Spike count (lifetime)
    pub spike_count: u64,
    /// Padding for alignment
    _padding: [u8; 4],
}

impl LifNeuronState {
    #[inline]
    pub fn new(
        v_rest: f32,
        v_threshold: f32,
        v_reset: f32,
        tau_ms: f32,
        refractory_us: u32,
    ) -> Self {
        Self {
            v_mem: v_rest,
            v_rest,
            v_threshold,
            v_reset,
            tau_ms,
            refractory_timer_us: 0,
            last_spike_ts_us: 0,
            spike_count: 0,
            _padding: [0; 4],
        }
    }

    /// Create with default parameters
    #[inline]
    pub fn default_params() -> Self {
        Self::new(
            DEFAULT_V_REST,
            DEFAULT_V_THRESHOLD,
            DEFAULT_V_RESET,
            DEFAULT_TAU_MS,
            DEFAULT_REFRACTORY_US,
        )
    }

    /// Check if neuron is in refractory period
    #[inline]
    pub fn is_refractory(&self) -> bool {
        self.refractory_timer_us > 0
    }

    /// Check if neuron should spike
    #[inline]
    pub fn should_spike(&self) -> bool {
        !self.is_refractory() && self.v_mem >= self.v_threshold
    }
}

/// Packed neuron state for SIMD processing (8 neurons)
#[derive(Debug, Clone, Copy)]
#[repr(C, align(32))]
pub struct SimdLifNeurons8 {
    /// Membrane potentials (8 neurons)
    pub v_mem: f32x8,
    /// Resting potentials
    pub v_rest: f32x8,
    /// Thresholds
    pub v_threshold: f32x8,
    /// Reset potentials
    pub v_reset: f32x8,
    /// Time constants
    pub tau_ms: f32x8,
    /// Refractory timers (packed as f32 for SIMD)
    pub refractory_timer: f32x8,
    /// Inverse tau for efficient division (1/τ)
    pub inv_tau: f32x8,
}

impl SimdLifNeurons8 {
    #[inline]
    pub fn from_array(neurons: &[LifNeuronState; 8]) -> Self {
        let mut v_mem = [0.0f32; 8];
        let mut v_rest = [0.0f32; 8];
        let mut v_threshold = [0.0f32; 8];
        let mut v_reset = [0.0f32; 8];
        let mut tau_ms = [0.0f32; 8];
        let mut refractory = [0.0f32; 8];
        let mut inv_tau = [0.0f32; 8];

        for i in 0..8 {
            v_mem[i] = neurons[i].v_mem;
            v_rest[i] = neurons[i].v_rest;
            v_threshold[i] = neurons[i].v_threshold;
            v_reset[i] = neurons[i].v_reset;
            tau_ms[i] = neurons[i].tau_ms;
            refractory[i] = neurons[i].refractory_timer_us as f32;
            inv_tau[i] = 1.0 / neurons[i].tau_ms;
        }

        Self {
            v_mem: f32x8::from(v_mem),
            v_rest: f32x8::from(v_rest),
            v_threshold: f32x8::from(v_threshold),
            v_reset: f32x8::from(v_reset),
            tau_ms: f32x8::from(tau_ms),
            refractory_timer: f32x8::from(refractory),
            inv_tau: f32x8::from(inv_tau),
        }
    }

    /// Convert back to array of neuron states
    #[inline]
    pub fn to_array(&self) -> [LifNeuronState; 8] {
        let v_mem_arr = self.v_mem.to_array();
        let v_rest_arr = self.v_rest.to_array();
        let v_threshold_arr = self.v_threshold.to_array();
        let v_reset_arr = self.v_reset.to_array();
        let tau_arr = self.tau_ms.to_array();
        let refractory_arr = self.refractory_timer.to_array();

        let mut neurons = [LifNeuronState::default_params(); 8];
        for i in 0..8 {
            neurons[i].v_mem = v_mem_arr[i];
            neurons[i].v_rest = v_rest_arr[i];
            neurons[i].v_threshold = v_threshold_arr[i];
            neurons[i].v_reset = v_reset_arr[i];
            neurons[i].tau_ms = tau_arr[i];
            neurons[i].refractory_timer_us = refractory_arr[i] as u32;
        }
        neurons
    }
}

/// SIMD-accelerated LIF neuron engine
pub struct SimdLifEngine {
    /// Number of neurons (must be multiple of SIMD_LANE_WIDTH)
    neuron_count: usize,
    /// Padded neuron count for SIMD alignment
    padded_count: usize,
    /// Neuron states (padded to SIMD lane width)
    neurons: Vec<LifNeuronState>,
    /// Input currents (padded)
    input_currents: Vec<f32>,
    /// Spike flags output (one bit per neuron)
    spike_flags: Vec<bool>,
    /// Simulation timestep (μs)
    dt_us: f32,
    /// Total spikes generated
    total_spikes: AtomicU64,
    /// Total timesteps simulated
    timesteps_simulated: AtomicU64,
}

impl SimdLifEngine {
    /// Create a new SIMD LIF engine
    /// # Arguments
    /// * `neuron_count` - Number of neurons (will be padded to SIMD_LANE_WIDTH_AVX2 multiple)
    /// * `dt_us` - Simulation timestep in microseconds
    pub fn new(neuron_count: usize, dt_us: f32) -> Result<Self, &'static str> {
        if neuron_count == 0 {
            return Err("Neuron count must be positive");
        }
        if dt_us <= 0.0 {
            return Err("Timestep must be positive");
        }

        // Pad to SIMD lane width (AVX2 = 8)
        let padding = (SIMD_LANE_WIDTH_AVX2 - (neuron_count % SIMD_LANE_WIDTH_AVX2))
            % SIMD_LANE_WIDTH_AVX2;
        let padded_count = neuron_count + padding;

        let mut neurons = Vec::with_capacity(padded_count);
        let mut input_currents = Vec::with_capacity(padded_count);

        for i in 0..padded_count {
            if i < neuron_count {
                neurons.push(LifNeuronState::default_params());
                input_currents.push(0.0);
            } else {
                // Padding neurons (inactive)
                let mut pad_neuron = LifNeuronState::default_params();
                pad_neuron.v_threshold = f32::MAX; // Never spike
                neurons.push(pad_neuron);
                input_currents.push(0.0);
            }
        }

        Ok(Self {
            neuron_count,
            padded_count,
            neurons,
            input_currents,
            spike_flags: vec![false; padded_count],
            dt_us,
            total_spikes: AtomicU64::new(0),
            timesteps_simulated: AtomicU64::new(0),
        })
    }

    /// Set input current for a specific neuron
    #[inline]
    pub fn set_input_current(&mut self, neuron_idx: usize, current: f32) {
        if neuron_idx < self.neuron_count {
            self.input_currents[neuron_idx] = current;
        }
    }

    /// Set input currents for all neurons
    #[inline]
    pub fn set_input_currents(&mut self, currents: &[f32]) {
        let len = currents.len().min(self.neuron_count);
        self.input_currents[..len].copy_from_slice(&currents[..len]);
    }

    /// Perform one simulation step using SIMD vectorization
    /// Returns slice of spike flags (true = neuron spiked)
    #[inline]
    pub fn step(&mut self) -> &[bool] {
        self.step_avx2();
        self.timesteps_simulated.fetch_add(1, Ordering::Relaxed);
        &self.spike_flags[..self.neuron_count]
    }

    /// AVX2-optimized simulation step (8 neurons at a time)
    #[inline]
    fn step_avx2(&mut self) {
        let dt_ms = self.dt_us / 1000.0; // Convert μs to ms
        let dt_vec = f32x8::splat(dt_ms);

        let mut spike_count_this_step = 0u64;

        // Process 8 neurons at a time
        for chunk_start in (0..self.padded_count).step_by(SIMD_LANE_WIDTH_AVX2) {
            // Load neuron state into SIMD registers
            let mut v_mem = [0.0f32; 8];
            let mut v_rest = [0.0f32; 8];
            let mut v_threshold = [0.0f32; 8];
            let mut v_reset = [0.0f32; 8];
            let mut tau_inv = [0.0f32; 8];
            let mut refractory = [0.0f32; 8];
            let mut input = [0.0f32; 8];

            for i in 0..8 {
                let idx = chunk_start + i;
                v_mem[i] = self.neurons[idx].v_mem;
                v_rest[i] = self.neurons[idx].v_rest;
                v_threshold[i] = self.neurons[idx].v_threshold;
                v_reset[i] = self.neurons[idx].v_reset;
                tau_inv[i] = 1.0 / self.neurons[idx].tau_ms;
                refractory[i] = self.neurons[idx].refractory_timer_us as f32;
                input[i] = self.input_currents[idx];
            }

            let v_mem_vec = f32x8::from(v_mem);
            let v_rest_vec = f32x8::from(v_rest);
            let v_threshold_vec = f32x8::from(v_threshold);
            let v_reset_vec = f32x8::from(v_reset);
            let tau_inv_vec = f32x8::from(tau_inv);
            let refractory_vec = f32x8::from(refractory);
            let input_vec = f32x8::from(input);

            // LIF update: V[t+1] = V[t] + dt/τ * (-(V[t] - V_rest) + R * I[t])
            // Assuming R = 1 for simplicity (can be parameterized)
            let dv = dt_vec * tau_inv_vec * (-(v_mem_vec - v_rest_vec) + input_vec);
            let v_mem_new = v_mem_vec + dv;

            // Decrement refractory timer
            let refractory_new = (refractory_vec - dt_vec).max(f32x8::splat(0.0));

            // Detect spikes: V >= threshold AND not refractory
            let above_threshold = v_mem_new.simd_ge(v_threshold_vec);
            let not_refractory = refractory_vec.simd_eq(f32x8::splat(0.0));
            let spike_mask = above_threshold & not_refractory;

            // Reset membrane potential for spiking neurons
            let v_mem_reset = v_mem_new.select(
                spike_mask,
                v_reset_vec,
            );

            // Set refractory period for spiking neurons
            let refractory_reset = refractory_new.select(
                spike_mask,
                f32x8::splat(DEFAULT_REFRACTORY_US as f32),
                refractory_new,
            );

            // Store results back
            let v_mem_arr = v_mem_reset.to_array();
            let refractory_arr = refractory_reset.to_array();
            let spike_arr = spike_mask.to_array();

            for i in 0..8 {
                let idx = chunk_start + i;
                self.neurons[idx].v_mem = v_mem_arr[i];
                self.neurons[idx].refractory_timer_us = refractory_arr[i] as u32;
                self.spike_flags[idx] = spike_arr[i];

                if spike_arr[i] && idx < self.neuron_count {
                    self.neurons[idx].spike_count += 1;
                    self.neurons[idx].last_spike_ts_us = 
                        self.timesteps_simulated.load(Ordering::Relaxed) * self.dt_us as u64;
                    spike_count_this_step += 1;
                }
            }
        }

        self.total_spikes.fetch_add(spike_count_this_step, Ordering::Relaxed);
    }

    /// Get spike flags from last step
    #[inline]
    pub fn get_spike_flags(&self) -> &[bool] {
        &self.spike_flags[..self.neuron_count]
    }

    /// Get membrane potentials
    #[inline]
    pub fn get_membrane_potentials(&self) -> &[f32] {
        &self.neurons.iter()
            .take(self.neuron_count)
            .map(|n| n.v_mem)
            .collect::<Vec<f32>>()
    }

    /// Get statistics
    #[inline]
    pub fn stats(&self) -> LifEngineStats {
        LifEngineStats {
            neuron_count: self.neuron_count,
            padded_count: self.padded_count,
            total_spikes: self.total_spikes.load(Ordering::Relaxed),
            timesteps_simulated: self.timesteps_simulated.load(Ordering::Relaxed),
        }
    }

    /// Reset all neurons to resting state
    #[inline]
    pub fn reset(&mut self) {
        for i in 0..self.neuron_count {
            self.neurons[i].v_mem = self.neurons[i].v_rest;
            self.neurons[i].refractory_timer_us = 0;
            self.spike_flags[i] = false;
        }
        self.input_currents.fill(0.0);
    }

    /// Reset a specific neuron
    #[inline]
    pub fn reset_neuron(&mut self, idx: usize) {
        if idx < self.neuron_count {
            self.neurons[idx].v_mem = self.neurons[idx].v_rest;
            self.neurons[idx].refractory_timer_us = 0;
            self.spike_flags[idx] = false;
        }
    }
}

/// Statistics about LIF engine operation
#[derive(Debug, Clone, Copy)]
pub struct LifEngineStats {
    pub neuron_count: usize,
    pub padded_count: usize,
    pub total_spikes: u64,
    pub timesteps_simulated: u64,
}

/// Batch processor for large-scale SNN simulation
pub struct LifBatchProcessor {
    /// Multiple SIMD engines for parallel processing
    engines: Vec<SimdLifEngine>,
    /// Global timestep counter
    global_timestep: u64,
}

impl LifBatchProcessor {
    #[inline]
    pub fn new(total_neurons: usize, dt_us: f32) -> Result<Self, &'static str> {
        // Split into multiple engines for better cache locality
        const NEURONS_PER_ENGINE: usize = 1024;
        let engine_count = (total_neurons + NEURONS_PER_ENGINE - 1) / NEURONS_PER_ENGINE;
        
        let mut engines = Vec::with_capacity(engine_count);
        let mut remaining = total_neurons;

        while remaining > 0 {
            let batch_size = remaining.min(NEURONS_PER_ENGINE);
            engines.push(SimdLifEngine::new(batch_size, dt_us)?);
            remaining -= batch_size;
        }

        Ok(Self {
            engines,
            global_timestep: 0,
        })
    }

    /// Step all engines synchronously
    #[inline]
    pub fn step_all(&mut self) {
        for engine in &mut self.engines {
            engine.step();
        }
        self.global_timestep += 1;
    }

    /// Get total neuron count
    #[inline]
    pub fn total_neurons(&self) -> usize {
        self.engines.iter().map(|e| e.stats().neuron_count).sum()
    }

    /// Get aggregate statistics
    #[inline]
    pub fn aggregate_stats(&self) -> BatchStats {
        let mut total_spikes = 0u64;
        let mut total_timesteps = 0u64;
        let mut total_neurons = 0usize;

        for engine in &self.engines {
            let stats = engine.stats();
            total_spikes += stats.total_spikes;
            total_timesteps = total_timesteps.max(stats.timesteps_simulated);
            total_neurons += stats.neuron_count;
        }

        BatchStats {
            engine_count: self.engines.len(),
            total_neurons,
            total_spikes,
            global_timestep: self.global_timestep,
        }
    }
}

/// Batch processing statistics
#[derive(Debug, Clone, Copy)]
pub struct BatchStats {
    pub engine_count: usize,
    pub total_neurons: usize,
    pub total_spikes: u64,
    pub global_timestep: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lif_neuron_creation() {
        let neuron = LifNeuronState::default_params();
        assert_eq!(neuron.v_mem, DEFAULT_V_REST);
        assert_eq!(neuron.v_threshold, DEFAULT_V_THRESHOLD);
        assert!(!neuron.is_refractory());
    }

    #[test]
    fn test_simd_lif_engine_creation() {
        let engine = SimdLifEngine::new(16, 100.0).unwrap();
        assert_eq!(engine.neuron_count, 16);
        assert_eq!(engine.padded_count, 16); // Already aligned
    }

    #[test]
    fn test_simd_lif_engine_padding() {
        let engine = SimdLifEngine::new(10, 100.0).unwrap();
        assert_eq!(engine.neuron_count, 10);
        assert_eq!(engine.padded_count, 16); // Padded to next multiple of 8
    }

    #[test]
    fn test_lif_simulation_step() {
        let mut engine = SimdLifEngine::new(8, 100.0).unwrap();
        
        // Apply strong input current to drive spiking
        let currents = [100.0f32; 8];
        engine.set_input_currents(&currents);
        
        // Run several steps
        for _ in 0..100 {
            engine.step();
        }
        
        // Should have generated some spikes
        let stats = engine.stats();
        assert!(stats.total_spikes > 0);
    }

    #[test]
    fn test_lif_refractory_period() {
        let mut engine = SimdLifEngine::new(8, 100.0).unwrap();
        
        // Drive neuron to spike
        engine.set_input_current(0, 100.0);
        
        // Run until spike
        let mut spiked = false;
        for _ in 0..200 {
            let spikes = engine.step();
            if spikes[0] {
                spiked = true;
                break;
            }
        }
        
        assert!(spiked);
        
        // Neuron should now be refractory
        assert!(engine.neurons[0].is_refractory());
    }

    #[test]
    fn test_batch_processor() {
        let processor = LifBatchProcessor::new(2048, 100.0).unwrap();
        assert_eq!(processor.total_neurons(), 2048);
        assert!(processor.aggregate_stats().engine_count > 0);
    }
}
