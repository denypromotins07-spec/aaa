//! Membrane Potential Update Module
//! 
//! Low-level membrane potential update kernels for LIF neurons.
//! Provides both scalar and SIMD-optimized update functions.

use std::simd::{f32x8, num::SimdFloat};

/// Scalar membrane potential update (fallback for non-SIMD paths)
/// LIF: dV/dt = (-(V - V_rest) + R * I) / τ
#[inline(always)]
pub fn update_membrane_potential_scalar(
    v_mem: f32,
    v_rest: f32,
    input_current: f32,
    tau_ms: f32,
    dt_ms: f32,
    resistance: f32,
) -> f32 {
    let dv = dt_ms * (-(v_mem - v_rest) + resistance * input_current) / tau_ms;
    v_mem + dv
}

/// SIMD vectorized membrane potential update for 8 neurons
#[inline(always)]
pub fn update_membrane_potential_simd8(
    v_mem: f32x8,
    v_rest: f32x8,
    input_current: f32x8,
    tau_inv: f32x8,
    dt_ms: f32,
    resistance: f32,
) -> f32x8 {
    let dt_vec = f32x8::splat(dt_ms);
    let resistance_vec = f32x8::splat(resistance);
    
    // dV = dt * (-(V - V_rest) + R * I) / τ
    let dv = dt_vec * (-(v_mem - v_rest) + resistance_vec * input_current) * tau_inv;
    v_mem + dv
}

/// Apply spike reset to membrane potentials
/// Returns (reset_v_mem, spike_occurred_mask)
#[inline(always)]
pub fn apply_spike_reset_simd8(
    v_mem: f32x8,
    v_threshold: f32x8,
    v_reset: f32x8,
    refractory_timer: f32x8,
    dt_ms: f32,
    refractory_period_ms: f32,
) -> (f32x8, f32x8) {
    let dt_vec = f32x8::splat(dt_ms);
    let refractory_period_vec = f32x8::splat(refractory_period_ms);
    
    // Decrement refractory timer
    let refractory_new = (refractory_timer - dt_vec).max(f32x8::splat(0.0));
    
    // Detect spikes: V >= threshold AND not in refractory
    let above_threshold = v_mem.simd_ge(v_threshold);
    let not_refractory = refractory_timer.simd_eq(f32x8::splat(0.0));
    let spike_mask = above_threshold & not_refractory;
    
    // Reset membrane potential for spiking neurons
    let v_mem_reset = v_mem.select(spike_mask, v_reset);
    
    // Set refractory period for spiking neurons
    let refractory_reset = refractory_new.select(
        spike_mask,
        refractory_period_vec,
        refractory_new,
    );
    
    (v_mem_reset, refractory_reset)
}

/// Batch membrane potential update with adaptive timestep
pub struct MembraneUpdateBatch {
    /// Current membrane potentials
    v_mem: Vec<f32>,
    /// Resting potentials
    v_rest: Vec<f32>,
    /// Thresholds
    v_threshold: Vec<f32>,
    /// Reset potentials
    v_reset: Vec<f32>,
    /// Inverse time constants
    tau_inv: Vec<f32>,
    /// Refractory timers
    refractory_timer: Vec<f32>,
    /// Spike flags
    spiked: Vec<bool>,
    /// Neuron count
    count: usize,
}

impl MembraneUpdateBatch {
    #[inline]
    pub fn new(count: usize) -> Self {
        Self {
            v_mem: vec![0.0f32; count],
            v_rest: vec![0.0f32; count],
            v_threshold: vec![0.0f32; count],
            v_reset: vec![0.0f32; count],
            tau_inv: vec![1.0f32; count],
            refractory_timer: vec![0.0f32; count],
            spiked: vec![false; count],
            count,
        }
    }

    /// Initialize neuron parameters
    #[inline]
    pub fn initialize_neurons(
        &mut self,
        v_rest: f32,
        v_threshold: f32,
        v_reset: f32,
        tau_ms: f32,
    ) {
        for i in 0..self.count {
            self.v_mem[i] = v_rest;
            self.v_rest[i] = v_rest;
            self.v_threshold[i] = v_threshold;
            self.v_reset[i] = v_reset;
            self.tau_inv[i] = 1.0 / tau_ms;
            self.refractory_timer[i] = 0.0;
        }
    }

    /// Perform one update step using SIMD where possible
    #[inline]
    pub fn step(&mut self, input_currents: &[f32], dt_ms: f32, resistance: f32) {
        let resistance_vec = f32x8::splat(resistance);
        let dt_vec = f32x8::splat(dt_ms);
        let refractory_period_vec = f32x8::splat(2.0); // 2ms default

        // Process in chunks of 8
        let simd_count = (self.count / 8) * 8;

        for chunk_start in (0..simd_count).step_by(8) {
            // Load data
            let mut v_mem_arr = [0.0f32; 8];
            let mut v_rest_arr = [0.0f32; 8];
            let mut tau_inv_arr = [0.0f32; 8];
            let mut refractory_arr = [0.0f32; 8];
            let mut input_arr = [0.0f32; 8];
            let mut threshold_arr = [0.0f32; 8];
            let mut reset_arr = [0.0f32; 8];

            for i in 0..8 {
                let idx = chunk_start + i;
                v_mem_arr[i] = self.v_mem[idx];
                v_rest_arr[i] = self.v_rest[idx];
                tau_inv_arr[i] = self.tau_inv[idx];
                refractory_arr[i] = self.refractory_timer[idx];
                input_arr[i] = input_currents.get(idx).copied().unwrap_or(0.0);
                threshold_arr[i] = self.v_threshold[idx];
                reset_arr[i] = self.v_reset[idx];
            }

            let v_mem = f32x8::from(v_mem_arr);
            let v_rest = f32x8::from(v_rest_arr);
            let tau_inv = f32x8::from(tau_inv_arr);
            let refractory = f32x8::from(refractory_arr);
            let input = f32x8::from(input_arr);
            let threshold = f32x8::from(threshold_arr);
            let reset = f32x8::from(reset_arr);

            // Update membrane potential
            let dv = dt_vec * (-(v_mem - v_rest) + resistance_vec * input) * tau_inv;
            let v_mem_new = v_mem + dv;

            // Apply spike reset
            let refractory_new = (refractory - dt_vec).max(f32x8::splat(0.0));
            let above_threshold = v_mem_new.simd_ge(threshold);
            let not_refractory = refractory.simd_eq(f32x8::splat(0.0));
            let spike_mask = above_threshold & not_refractory;

            let v_mem_reset = v_mem_new.select(spike_mask, reset);
            let refractory_reset = refractory_new.select(
                spike_mask,
                refractory_period_vec,
                refractory_new,
            );

            // Store results
            let v_mem_result = v_mem_reset.to_array();
            let refractory_result = refractory_reset.to_array();
            let spike_result = spike_mask.to_array();

            for i in 0..8 {
                let idx = chunk_start + i;
                self.v_mem[idx] = v_mem_result[i];
                self.refractory_timer[idx] = refractory_result[i];
                self.spiked[idx] = spike_result[i];
            }
        }

        // Handle remaining neurons (scalar fallback)
        for idx in simd_count..self.count {
            if self.refractory_timer[idx] > 0.0 {
                self.refractory_timer[idx] -= dt_ms;
                self.spiked[idx] = false;
            } else {
                self.v_mem[idx] = update_membrane_potential_scalar(
                    self.v_mem[idx],
                    self.v_rest[idx],
                    input_currents.get(idx).copied().unwrap_or(0.0),
                    1.0 / self.tau_inv[idx],
                    dt_ms,
                    resistance,
                );

                if self.v_mem[idx] >= self.v_threshold[idx] {
                    self.v_mem[idx] = self.v_reset[idx];
                    self.refractory_timer[idx] = refractory_period_ms();
                    self.spiked[idx] = true;
                } else {
                    self.spiked[idx] = false;
                }
            }
        }
    }

    /// Get spike flags
    #[inline]
    pub fn get_spikes(&self) -> &[bool] {
        &self.spiked[..self.count]
    }

    /// Get membrane potentials
    #[inline]
    pub fn get_potentials(&self) -> &[f32] {
        &self.v_mem[..self.count]
    }

    /// Reset all neurons
    #[inline]
    pub fn reset(&mut self) {
        for i in 0..self.count {
            self.v_mem[i] = self.v_rest[i];
            self.refractory_timer[i] = 0.0;
            self.spiked[i] = false;
        }
    }
}

/// Default refractory period in milliseconds
#[inline(always)]
const fn refractory_period_ms() -> f32 {
    2.0
}

/// Exponential integrate-and-fire (EIF) model extension
/// Adds exponential nonlinearity near threshold: dV/dt = (-(V-V_rest) + Δ*exp((V-V_t)/Δ) + RI)/τ
#[inline(always)]
pub fn update_eif_membrane_potential(
    v_mem: f32,
    v_rest: f32,
    v_threshold: f32,
    input_current: f32,
    tau_ms: f32,
    dt_ms: f32,
    delta_t: f32, // Sharpness parameter
    resistance: f32,
) -> f32 {
    let exp_term = delta_t * ((v_mem - v_threshold) / delta_t).exp();
    let dv = dt_ms * (-(v_mem - v_rest) + exp_term + resistance * input_current) / tau_ms;
    (v_mem + dv).min(v_threshold * 2.0) // Clamp to prevent runaway
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_update() {
        let v_mem = -70.0;
        let v_rest = -70.0;
        let input = 10.0;
        let tau = 20.0;
        let dt = 0.1;
        let r = 1.0;

        let new_v = update_membrane_potential_scalar(v_mem, v_rest, input, tau, dt, r);
        
        // Should increase from resting potential
        assert!(new_v > v_mem);
        assert!(new_v < v_rest + 1.0); // Small increase
    }

    #[test]
    fn test_batch_initialize() {
        let mut batch = MembraneUpdateBatch::new(16);
        batch.initialize_neurons(-70.0, -55.0, -75.0, 20.0);

        for v in batch.get_potentials() {
            assert_eq!(*v, -70.0);
        }
    }

    #[test]
    fn test_batch_step_with_input() {
        let mut batch = MembraneUpdateBatch::new(8);
        batch.initialize_neurons(-70.0, -55.0, -75.0, 20.0);

        let inputs = [50.0f32; 8];
        
        for _ in 0..50 {
            batch.step(&inputs, 0.1, 1.0);
        }

        // Some neurons should have spiked with strong input
        let spikes = batch.get_spikes();
        assert!(spikes.iter().any(|&s| s));
    }

    #[test]
    fn test_eif_update() {
        let v_mem = -60.0;
        let new_v = update_eif_membrane_potential(
            v_mem, -70.0, -55.0, 10.0, 20.0, 0.1, 2.0, 1.0,
        );
        // EIF should show exponential acceleration near threshold
        assert!(new_v > v_mem);
    }
}
