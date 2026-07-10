//! Spike-Timing-Dependent Plasticity (STDP) Learning Rule
//! 
//! Implements biological Hebbian learning where synaptic weights are updated
//! based on the relative timing of pre- and post-synaptic spikes.
//! 
//! LTP (Long-Term Potentiation): Pre before Post → strengthen synapse
//! LTD (Long-Term Depression): Post before Pre → weaken synapse
//!
//! CRITICAL: Uses atomic operations for thread-safe parallel weight updates
//! and strict weight bounding to prevent STDP runaway explosions.

use std::sync::atomic::{AtomicI64, Ordering};

/// Fixed-point scale factor for weight representation (12-bit fractional)
pub const WEIGHT_SCALE: i64 = 4096;

/// Maximum synaptic weight (scaled fixed-point)
pub const MAX_WEIGHT: i64 = WEIGHT_SCALE; // 1.0 in fixed-point

/// Minimum synaptic weight (scaled fixed-point, never goes to zero)
pub const MIN_WEIGHT: i64 = WEIGHT_SCALE / 100; // 0.01 in fixed-point

/// Default LTP amplitude (weight increase per causal spike pair)
pub const DEFAULT_LTP_AMPLITUDE: i64 = WEIGHT_SCALE / 1000; // 0.001

/// Default LTD amplitude (weight decrease per anti-causal spike pair)
pub const DEFAULT_LTD_AMPLITUDE: i64 = WEIGHT_SCALE / 1250; // 0.0008

/// Default LTP time constant (microseconds)
pub const DEFAULT_LTP_TAU_US: u64 = 20_000; // 20ms

/// Default LTD time constant (microseconds)
pub const DEFAULT_LTD_TAU_US: u64 = 20_000; // 20ms

/// STDP configuration parameters
#[derive(Debug, Clone, Copy)]
pub struct StdpConfig {
    /// LTP amplitude (fixed-point)
    pub ltp_amplitude: i64,
    /// LTD amplitude (fixed-point)
    pub ltd_amplitude: i64,
    /// LTP time constant (μs)
    pub ltp_tau_us: u64,
    /// LTD time constant (μs)
    pub ltd_tau_us: u64,
    /// Weight decay rate per timestep (fixed-point, e.g., WEIGHT_SCALE/10000 for 0.01%)
    pub weight_decay: i64,
    /// Homeostatic scaling target firing rate (spikes/second)
    pub target_firing_rate: f32,
    /// Homeostatic scaling rate
    pub homeostatic_rate: f32,
    /// Whether to use weight-dependent STDP (soft bounds)
    pub weight_dependent: bool,
}

impl Default for StdpConfig {
    #[inline]
    fn default() -> Self {
        Self {
            ltp_amplitude: DEFAULT_LTP_AMPLITUDE,
            ltd_amplitude: DEFAULT_LTD_AMPLITUDE,
            ltp_tau_us: DEFAULT_LTP_TAU_US,
            ltd_tau_us: DEFAULT_LTD_TAU_US,
            weight_decay: WEIGHT_SCALE / 10000, // 0.01% per step
            target_firing_rate: 10.0, // 10 Hz target
            homeostatic_rate: 0.001,
            weight_dependent: true,
        }
    }
}

/// Synaptic weight with atomic operations for thread-safe updates
pub struct AtomicSynapticWeight {
    /// Weight value in fixed-point format
    value: AtomicI64,
}

impl AtomicSynapticWeight {
    #[inline]
    pub fn new(initial_weight: f32) -> Self {
        let fixed_point = (initial_weight * WEIGHT_SCALE as f32) as i64;
        let clamped = fixed_point.clamp(MIN_WEIGHT, MAX_WEIGHT);
        Self {
            value: AtomicI64::new(clamped),
        }
    }

    /// Get current weight as f32
    #[inline]
    pub fn get(&self) -> f32 {
        self.value.load(Ordering::Acquire) as f32 / WEIGHT_SCALE as f32
    }

    /// Get raw fixed-point value
    #[inline]
    pub fn get_fixed(&self) -> i64 {
        self.value.load(Ordering::Acquire)
    }

    /// Atomically add delta to weight with bounds checking
    /// Returns the new weight value (fixed-point)
    #[inline]
    pub fn atomic_add_bounded(&self, delta: i64) -> i64 {
        loop {
            let current = self.value.load(Ordering::Acquire);
            let mut new_value = current + delta;

            // Hard bounds clipping
            if new_value > MAX_WEIGHT {
                new_value = MAX_WEIGHT;
            } else if new_value < MIN_WEIGHT {
                new_value = MIN_WEIGHT;
            }

            // Atomic compare-and-swap
            match self.value.compare_exchange_weak(
                current,
                new_value,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return new_value,
                Err(_) => continue, // Retry on contention
            }
        }
    }

    /// Weight-dependent update (soft bounds - Oja's rule style)
    /// LTP is scaled by (1 - w/w_max), LTD is scaled by (w/w_max)
    #[inline]
    pub fn atomic_add_weight_dependent(&self, ltp_delta: i64, ltd_delta: i64) -> i64 {
        loop {
            let current = self.value.load(Ordering::Acquire);
            
            // Soft bound scaling
            let ltp_scaled = if ltp_delta > 0 {
                ltp_delta * (MAX_WEIGHT - current) / MAX_WEIGHT
            } else {
                ltp_delta
            };

            let ltd_scaled = if ltd_delta > 0 {
                ltd_delta * current / MAX_WEIGHT
            } else {
                ltd_delta
            };

            let mut new_value = current + ltp_scaled - ltd_scaled;

            // Still enforce hard bounds as safety net
            if new_value > MAX_WEIGHT {
                new_value = MAX_WEIGHT;
            } else if new_value < MIN_WEIGHT {
                new_value = MIN_WEIGHT;
            }

            match self.value.compare_exchange_weak(
                current,
                new_value,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return new_value,
                Err(_) => continue,
            }
        }
    }

    /// Apply homeostatic scaling to prevent weight explosion
    #[inline]
    pub fn apply_homeostatic_scaling(&self, actual_firing_rate: f32, target: f32, rate: f32) {
        if actual_firing_rate <= 0.0 {
            return;
        }

        let error = (actual_firing_rate - target) / target;
        let scale_factor = 1.0 - rate * error;

        loop {
            let current = self.value.load(Ordering::Acquire);
            let scaled = (current as f32 * scale_factor) as i64;
            let clamped = scaled.clamp(MIN_WEIGHT, MAX_WEIGHT);

            match self.value.compare_exchange_weak(
                current,
                clamped,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }
    }

    /// Reset weight to initial value
    #[inline]
    pub fn reset(&self, initial_weight: f32) {
        let fixed_point = (initial_weight * WEIGHT_SCALE as f32) as i64;
        let clamped = fixed_point.clamp(MIN_WEIGHT, MAX_WEIGHT);
        self.value.store(clamped, Ordering::Release);
    }
}

/// STDP learning rule implementation
pub struct StdpRule {
    config: StdpConfig,
    /// Pre-computed LTP decay lookup table (for performance)
    ltp_decay_table: Vec<i64>,
    /// Pre-computed LTD decay lookup table
    ltd_decay_table: Vec<i64>,
    /// Time resolution for lookup tables (μs)
    table_resolution_us: u64,
}

impl StdpRule {
    /// Create a new STDP rule with given configuration
    #[inline]
    pub fn new(config: StdpConfig) -> Self {
        // Build decay lookup tables for fast exponential computation
        // Table covers up to 5 * tau (effectively zero after that)
        let max_tau = config.ltp_tau_us.max(config.ltd_tau_us);
        let table_duration_us = max_tau * 5;
        let table_resolution_us = 100; // 100μs resolution
        let table_size = (table_duration_us / table_resolution_us) as usize;

        let mut ltp_decay_table = Vec::with_capacity(table_size);
        let mut ltd_decay_table = Vec::with_capacity(table_size);

        for i in 0..table_size {
            let dt_us = (i as u64) * table_resolution_us;
            
            // Exponential decay: A * exp(-dt/τ)
            let ltp_decay = Self::exp_decay(config.ltp_amplitude, config.ltp_tau_us, dt_us);
            let ltd_decay = Self::exp_decay(config.ltd_amplitude, config.ltd_tau_us, dt_us);
            
            ltp_decay_table.push(ltp_decay);
            ltd_decay_table.push(ltd_decay);
        }

        Self {
            config,
            ltp_decay_table,
            ltd_decay_table,
            table_resolution_us,
        }
    }

    /// Fast exponential decay using lookup table
    #[inline]
    fn exp_decay(amplitude: i64, tau_us: u64, dt_us: u64) -> i64 {
        // Use integer approximation: exp(-x) ≈ (1 - x/n)^n for large n
        // Or simpler: linear approximation for small dt, exponential falloff for large
        if dt_us >= tau_us * 5 {
            return 0;
        }

        // More accurate: use fixed-point exponential approximation
        let ratio = (dt_us * WEIGHT_SCALE / tau_us) as i64;
        // exp(-ratio) approximation using Taylor series (first few terms)
        // For better accuracy, use lookup or CORDIC
        
        // Simple but effective: linear interpolation of exponential
        let normalized_dt = (dt_us * 1024 / tau_us) as i64;
        if normalized_dt <= 0 {
            amplitude
        } else if normalized_dt >= 1024 {
            0
        } else {
            // Approximate exp(-x) where x = normalized_dt/1024 * 5
            let x_times_256 = (normalized_dt * 5 * 256) / 1024;
            // exp(-x) ≈ 256 - x + x²/2 - ... (scaled by 256)
            let exp_approx = 256 - x_times_256 + (x_times_256 * x_times_256) / 512;
            (amplitude * exp_approx) / 256
        }
    }

    /// Calculate STDP weight update from spike timing difference
    /// dt_us > 0 means pre before post (LTP)
    /// dt_us < 0 means post before pre (LTD)
    #[inline]
    pub fn calculate_update(&self, dt_us: i64) -> i64 {
        if dt_us == 0 {
            return 0;
        }

        let abs_dt = dt_us.unsigned_abs();
        let table_idx = (abs_dt / self.table_resolution_us) as usize;

        if dt_us > 0 {
            // LTP: pre before post
            if table_idx < self.ltp_decay_table.len() {
                self.ltp_decay_table[table_idx]
            } else {
                0
            }
        } else {
            // LTD: post before pre
            if table_idx < self.ltd_decay_table.len() {
                -self.ltd_decay_table[table_idx] // Negative for depression
            } else {
                0
            }
        }
    }

    /// Apply STDP update to a synapse
    #[inline]
    pub fn apply_update(&self, weight: &AtomicSynapticWeight, dt_us: i64) -> i64 {
        let delta = self.calculate_update(dt_us);
        
        if self.config.weight_dependent {
            // Split delta into LTP and LTD components for weight-dependent scaling
            let ltp_delta = delta.max(0);
            let ltd_delta = (-delta).max(0);
            weight.atomic_add_weight_dependent(ltp_delta, ltd_delta)
        } else {
            weight.atomic_add_bounded(delta)
        }
    }

    /// Apply weight decay (synaptic scaling)
    #[inline]
    pub fn apply_decay(&self, weight: &AtomicSynapticWeight) {
        if self.config.weight_decay > 0 {
            let current = weight.get_fixed();
            let decay = (current * self.config.weight_decay) / WEIGHT_SCALE;
            weight.atomic_add_bounded(-decay);
        }
    }

    /// Get configuration
    #[inline]
    pub fn config(&self) -> StdpConfig {
        self.config
    }
}

/// Pair-based STDP tracker for accumulating spike pairs
pub struct StdpPairAccumulator {
    /// Recent pre-synaptic spike timestamps (circular buffer)
    pre_spikes: Vec<u64>,
    /// Recent post-synaptic spike timestamps
    post_spikes: Vec<u64>,
    /// Maximum age for spike pairs (μs)
    max_age_us: u64,
    /// Current write indices
    pre_idx: usize,
    post_idx: usize,
}

impl StdpPairAccumulator {
    #[inline]
    pub fn new(max_spikes: usize, max_age_us: u64) -> Self {
        Self {
            pre_spikes: vec![0; max_spikes],
            post_spikes: vec![0; max_spikes],
            max_age_us,
            pre_idx: 0,
            post_idx: 0,
        }
    }

    /// Record a pre-synaptic spike
    #[inline]
    pub fn record_pre_spike(&mut self, timestamp_us: u64) {
        self.pre_spikes[self.pre_idx] = timestamp_us;
        self.pre_idx = (self.pre_idx + 1) % self.pre_spikes.len();
    }

    /// Record a post-synaptic spike
    #[inline]
    pub fn record_post_spike(&mut self, timestamp_us: u64) {
        self.post_spikes[self.post_idx] = timestamp_us;
        self.post_idx = (self.post_idx + 1) % self.post_spikes.len();
    }

    /// Get all valid spike pairs with their timing differences
    /// Returns Vec of (pre_timestamp, post_timestamp, dt_us)
    #[inline]
    pub fn get_valid_pairs(&self, current_time_us: u64) -> Vec<(u64, u64, i64)> {
        let mut pairs = Vec::new();

        for &pre_ts in &self.pre_spikes {
            if pre_ts == 0 || current_time_us.saturating_sub(pre_ts) > self.max_age_us {
                continue;
            }

            for &post_ts in &self.post_spikes {
                if post_ts == 0 || current_time_us.saturating_sub(post_ts) > self.max_age_us {
                    continue;
                }

                let dt = post_ts as i64 - pre_ts as i64;
                if dt.abs() as u64 <= self.max_age_us {
                    pairs.push((pre_ts, post_ts, dt));
                }
            }
        }

        pairs
    }

    /// Clear old spikes outside the time window
    #[inline]
    pub fn clear_old_spikes(&mut self, current_time_us: u64) {
        let cutoff = current_time_us.saturating_sub(self.max_age_us);
        
        for ts in &mut self.pre_spikes {
            if *ts < cutoff {
                *ts = 0;
            }
        }
        
        for ts in &mut self.post_spikes {
            if *ts < cutoff {
                *ts = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_weight_creation() {
        let weight = AtomicSynapticWeight::new(0.5);
        assert!((weight.get() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_atomic_weight_bounds() {
        let weight = AtomicSynapticWeight::new(0.5);
        
        // Try to exceed maximum
        weight.atomic_add_bounded(WEIGHT_SCALE);
        assert!(weight.get() <= 1.0);
        
        // Try to go below minimum
        for _ in 0..1000 {
            weight.atomic_add_bounded(-WEIGHT_SCALE / 100);
        }
        assert!(weight.get() >= 0.01);
    }

    #[test]
    fn test_stdp_ltp() {
        let config = StdpConfig::default();
        let rule = StdpRule::new(config);
        
        // Pre before post by 10ms should cause LTP
        let dt_us = 10_000;
        let update = rule.calculate_update(dt_us);
        assert!(update > 0);
    }

    #[test]
    fn test_stdp_ltd() {
        let config = StdpConfig::default();
        let rule = StdpRule::new(config);
        
        // Post before pre by 10ms should cause LTD
        let dt_us = -10_000;
        let update = rule.calculate_update(dt_us);
        assert!(update < 0);
    }

    #[test]
    fn test_stdp_decay_with_time() {
        let config = StdpConfig::default();
        let rule = StdpRule::new(config);
        
        let update_near = rule.calculate_update(1000); // 1ms
        let update_far = rule.calculate_update(50_000); // 50ms
        
        // Closer spikes should have stronger effect
        assert!(update_near > update_far);
    }

    #[test]
    fn test_weight_dependent_update() {
        let weight = AtomicSynapticWeight::new(0.9); // Near max
        let config = StdpConfig::default();
        let rule = StdpRule::new(config);
        
        // Apply LTP when weight is already high
        rule.apply_update(&weight, 10_000);
        
        // Should be bounded close to 1.0 but not exceed
        assert!(weight.get() <= 1.0);
    }

    #[test]
    fn test_pair_accumulator() {
        let mut acc = StdpPairAccumulator::new(10, 100_000);
        
        acc.record_pre_spike(1000);
        acc.record_post_spike(1100);
        
        let pairs = acc.get_valid_pairs(2000);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].2, 100); // dt = 100us
    }
}
