//! Eligibility Traces for STDP Learning
//! 
//! Implements trace-based STDP where each spike leaves an exponentially
//! decaying "trace" that interacts with future/past spikes from other neurons.
//! This is more efficient than pair-based matching for large networks.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// Default eligibility trace time constant (microseconds)
pub const DEFAULT_TRACE_TAU_US: u64 = 50_000; // 50ms

/// Fixed-point scale for trace values (12-bit fractional)
pub const TRACE_SCALE: i64 = 4096;

/// Eligibility trace state for a single neuron
#[derive(Debug, Clone, Copy)]
pub struct EligibilityTrace {
    /// Current trace value (fixed-point)
    pub value: i64,
    /// Last update timestamp (μs)
    pub last_update_us: u64,
    /// Trace decay constant (μs)
    pub tau_us: u64,
}

impl EligibilityTrace {
    #[inline]
    pub fn new(tau_us: u64) -> Self {
        Self {
            value: 0,
            last_update_us: 0,
            tau_us,
        }
    }

    /// Create with default time constant
    #[inline]
    pub fn default_trace() -> Self {
        Self::new(DEFAULT_TRACE_TAU_US)
    }

    /// Update trace value at given timestamp
    /// Applies exponential decay since last update, then adds spike contribution
    #[inline]
    pub fn update(&mut self, timestamp_us: u64, spike_contribution: i64) {
        if self.last_update_us == 0 {
            self.value = spike_contribution;
            self.last_update_us = timestamp_us;
            return;
        }

        let dt_us = timestamp_us.saturating_sub(self.last_update_us);
        
        // Apply exponential decay: trace *= exp(-dt/τ)
        self.value = Self::decay_trace(self.value, self.tau_us, dt_us);
        
        // Add new spike contribution
        self.value = self.value.saturating_add(spike_contribution);
        
        // Clamp to valid range
        self.value = self.value.clamp(0, TRACE_SCALE * 4); // Max 4x base scale
        
        self.last_update_us = timestamp_us;
    }

    /// Get current decayed trace value at given timestamp
    #[inline]
    pub fn get_decayed(&self, timestamp_us: u64) -> i64 {
        if self.last_update_us == 0 || self.value == 0 {
            return 0;
        }

        let dt_us = timestamp_us.saturating_sub(self.last_update_us);
        Self::decay_trace(self.value, self.tau_us, dt_us)
    }

    /// Fast exponential decay approximation for traces
    #[inline]
    fn decay_trace(value: i64, tau_us: u64, dt_us: u64) -> i64 {
        if dt_us == 0 {
            return value;
        }

        // For large dt, trace effectively zero
        if dt_us >= tau_us * 5 {
            return 0;
        }

        // Linear approximation of exponential decay
        // exp(-dt/τ) ≈ 1 - dt/τ for small dt
        // Use fixed-point arithmetic
        let ratio = ((dt_us * TRACE_SCALE as u64) / tau_us) as i64;
        
        if ratio >= TRACE_SCALE {
            return 0;
        }

        // value * (1 - dt/τ) = value - value * dt/τ
        value - (value * ratio) / TRACE_SCALE
    }

    /// Reset trace to zero
    #[inline]
    pub fn reset(&mut self) {
        self.value = 0;
        self.last_update_us = 0;
    }

    /// Set trace time constant
    #[inline]
    pub fn set_tau(&mut self, tau_us: u64) {
        self.tau_us = tau_us;
    }
}

/// Pre-synaptic eligibility trace accumulator
pub struct PreSynapticTrace {
    /// Trace value (atomic for concurrent access)
    trace: AtomicI64,
    /// Last update timestamp
    last_update: AtomicU64,
    /// Time constant
    tau_us: u64,
    /// Spike count for normalization
    spike_count: AtomicU64,
}

impl PreSynapticTrace {
    #[inline]
    pub fn new(tau_us: u64) -> Self {
        Self {
            trace: AtomicI64::new(0),
            last_update: AtomicU64::new(0),
            tau_us,
            spike_count: AtomicU64::new(0),
        }
    }

    /// Record a pre-synaptic spike
    #[inline]
    pub fn record_spike(&self, timestamp_us: u64, amplitude: i64) {
        let mut current_trace = self.trace.load(Ordering::Acquire);
        let last_update = self.last_update.load(Ordering::Acquire);

        if last_update > 0 {
            let dt_us = timestamp_us.saturating_sub(last_update);
            current_trace = EligibilityTrace::decay_trace(current_trace, self.tau_us, dt_us);
        }

        current_trace = current_trace.saturating_add(amplitude);
        current_trace = current_trace.clamp(0, TRACE_SCALE * 4);

        self.trace.store(current_trace, Ordering::Release);
        self.last_update.store(timestamp_us, Ordering::Release);
        self.spike_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current trace value (decayed to current time)
    #[inline]
    pub fn get(&self, timestamp_us: u64) -> i64 {
        let current_trace = self.trace.load(Ordering::Acquire);
        let last_update = self.last_update.load(Ordering::Acquire);

        if last_update == 0 || current_trace == 0 {
            return 0;
        }

        let dt_us = timestamp_us.saturating_sub(last_update);
        EligibilityTrace::decay_trace(current_trace, self.tau_us, dt_us)
    }

    /// Get spike count
    #[inline]
    pub fn spike_count(&self) -> u64 {
        self.spike_count.load(Ordering::Relaxed)
    }

    /// Reset trace
    #[inline]
    pub fn reset(&self) {
        self.trace.store(0, Ordering::Release);
        self.last_update.store(0, Ordering::Release);
        self.spike_count.store(0, Ordering::Relaxed);
    }
}

/// Post-synaptic eligibility trace (for backpropagating action potentials)
pub struct PostSynapticTrace {
    /// Membrane trace (fast decay)
    membrane_trace: AtomicI64,
    /// Calcium trace (slow decay, integrates spikes)
    calcium_trace: AtomicI64,
    /// Fast time constant (μs)
    tau_fast_us: u64,
    /// Slow time constant (μs)
    tau_slow_us: u64,
    /// Last update timestamp
    last_update: AtomicU64,
}

impl PostSynapticTrace {
    #[inline]
    pub fn new(tau_fast_us: u64, tau_slow_us: u64) -> Self {
        Self {
            membrane_trace: AtomicI64::new(0),
            calcium_trace: AtomicI64::new(0),
            tau_fast_us,
            tau_slow_us,
            last_update: AtomicU64::new(0),
        }
    }

    /// Default configuration (20ms fast, 100ms slow)
    #[inline]
    pub fn default_traces() -> Self {
        Self::new(20_000, 100_000)
    }

    /// Record a post-synaptic spike
    #[inline]
    pub fn record_spike(&self, timestamp_us: u64) {
        let last_update = self.last_update.load(Ordering::Acquire);
        let dt_us = if last_update > 0 {
            timestamp_us.saturating_sub(last_update)
        } else {
            0
        };

        // Decay both traces
        let mut mem_trace = self.membrane_trace.load(Ordering::Acquire);
        let mut calc_trace = self.calcium_trace.load(Ordering::Acquire);

        mem_trace = EligibilityTrace::decay_trace(mem_trace, self.tau_fast_us, dt_us);
        calc_trace = EligibilityTrace::decay_trace(calc_trace, self.tau_slow_us, dt_us);

        // Add spike contributions
        mem_trace = mem_trace.saturating_add(TRACE_SCALE); // Full contribution to fast trace
        calc_trace = calc_trace.saturating_add(TRACE_SCALE / 4); // Partial to slow trace

        // Clamp
        mem_trace = mem_trace.clamp(0, TRACE_SCALE * 4);
        calc_trace = calc_trace.clamp(0, TRACE_SCALE * 8);

        self.membrane_trace.store(mem_trace, Ordering::Release);
        self.calcium_trace.store(calc_trace, Ordering::Release);
        self.last_update.store(timestamp_us, Ordering::Release);
    }

    /// Get combined eligibility signal for plasticity
    #[inline]
    pub fn get_eligibility(&self, timestamp_us: u64) -> i64 {
        let last_update = self.last_update.load(Ordering::Acquire);
        let dt_us = if last_update > 0 {
            timestamp_us.saturating_sub(last_update)
        } else {
            0
        };

        let mem_trace = self.membrane_trace.load(Ordering::Acquire);
        let calc_trace = self.calcium_trace.load(Ordering::Acquire);

        let decayed_mem = EligibilityTrace::decay_trace(mem_trace, self.tau_fast_us, dt_us);
        let decayed_calc = EligibilityTrace::decay_trace(calc_trace, self.tau_slow_us, dt_us);

        // Combined eligibility: weighted sum of fast and slow traces
        (decayed_mem + decayed_calc / 2).clamp(0, TRACE_SCALE * 4)
    }

    /// Reset traces
    #[inline]
    pub fn reset(&self) {
        self.membrane_trace.store(0, Ordering::Release);
        self.calcium_trace.store(0, Ordering::Release);
        self.last_update.store(0, Ordering::Release);
    }
}

/// Triplet STDP rule using eligibility traces
/// More biologically realistic than pair-based STDP
pub struct TripletStdpTraces {
    /// Pre-synaptic trace (r1)
    pre_trace: PreSynapticTrace,
    /// Post-synaptic fast trace (o1)
    post_fast: PostSynapticTrace,
    /// Post-synaptic slow trace (o2)
    post_slow: PostSynapticTrace,
    /// LTP rate (A2+)
    ltp_rate: i64,
    /// LTD rate (A2-)
    ltd_rate: i64,
}

impl TripletStdpTraces {
    #[inline]
    pub fn new(ltp_rate: i64, ltd_rate: i64) -> Self {
        Self {
            pre_trace: PreSynapticTrace::new(DEFAULT_TRACE_TAU_US),
            post_fast: PostSynapticTrace::new(20_000, 100_000),
            post_slow: PostSynapticTrace::new(50_000, 200_000),
            ltp_rate,
            ltd_rate,
        }
    }

    /// Record a pre-synaptic spike and compute LTD contribution
    #[inline]
    pub fn record_pre_spike(&self, timestamp_us: u64) -> i64 {
        // Get post-synaptic eligibility for LTD
        let post_eligibility = self.post_slow.get_eligibility(timestamp_us);
        
        // Record pre-synaptic spike
        self.pre_trace.record_spike(timestamp_us, TRACE_SCALE);
        
        // LTD proportional to post-synaptic trace
        -(post_eligibility * self.ltd_rate) / TRACE_SCALE
    }

    /// Record a post-synaptic spike and compute LTP contribution
    #[inline]
    pub fn record_post_spike(&self, timestamp_us: u64) -> i64 {
        // Get pre-synaptic trace for LTP
        let pre_trace = self.pre_trace.get(timestamp_us);
        
        // Record post-synaptic spikes
        self.post_fast.record_spike(timestamp_us);
        self.post_slow.record_spike(timestamp_us);
        
        // LTP proportional to pre-synaptic trace
        (pre_trace * self.ltp_rate) / TRACE_SCALE
    }

    /// Get total weight change from triplet interactions
    #[inline]
    pub fn compute_weight_change(&self, timestamp_us: u64, is_pre: bool) -> i64 {
        if is_pre {
            self.record_pre_spike(timestamp_us)
        } else {
            self.record_post_spike(timestamp_us)
        }
    }

    /// Reset all traces
    #[inline]
    pub fn reset(&self) {
        self.pre_trace.reset();
        self.post_fast.reset();
        self.post_slow.reset();
    }
}

/// Batch eligibility trace manager for multiple synapses
pub struct EligibilityTraceManager {
    /// Pre-synaptic traces for each input
    pre_traces: Vec<PreSynapticTrace>,
    /// Post-synaptic traces for each output
    post_traces: Vec<PostSynapticTrace>,
}

impl EligibilityTraceManager {
    #[inline]
    pub fn new(n_inputs: usize, n_outputs: usize) -> Self {
        Self {
            pre_traces: (0..n_inputs)
                .map(|_| PreSynapticTrace::new(DEFAULT_TRACE_TAU_US))
                .collect(),
            post_traces: (0..n_outputs)
                .map(|_| PostSynapticTrace::default_traces())
                .collect(),
        }
    }

    /// Record pre-synaptic spike
    #[inline]
    pub fn record_pre(&self, input_idx: usize, timestamp_us: u64) {
        if input_idx < self.pre_traces.len() {
            self.pre_traces[input_idx].record_spike(timestamp_us, TRACE_SCALE);
        }
    }

    /// Record post-synaptic spike
    #[inline]
    pub fn record_post(&self, output_idx: usize, timestamp_us: u64) {
        if output_idx < self.post_traces.len() {
            self.post_traces[output_idx].record_spike(timestamp_us);
        }
    }

    /// Get eligibility for specific synapse
    #[inline]
    pub fn get_synapse_eligibility(&self, pre_idx: usize, post_idx: usize, timestamp_us: u64) -> i64 {
        if pre_idx >= self.pre_traces.len() || post_idx >= self.post_traces.len() {
            return 0;
        }

        let pre_trace = self.pre_traces[pre_idx].get(timestamp_us);
        let post_trace = self.post_traces[post_idx].get_eligibility(timestamp_us);

        // Hebbian interaction: product of pre and post traces
        (pre_trace * post_trace) / TRACE_SCALE
    }

    /// Clear all traces
    #[inline]
    pub fn clear_all(&self) {
        for trace in &self.pre_traces {
            trace.reset();
        }
        for trace in &self.post_traces {
            trace.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eligibility_trace_update() {
        let mut trace = EligibilityTrace::default_trace();
        
        trace.update(1000, TRACE_SCALE);
        assert_eq!(trace.value, TRACE_SCALE);
        
        // After one time constant, should decay to ~37%
        trace.update(1000 + DEFAULT_TRACE_TAU_US, 0);
        assert!(trace.value < TRACE_SCALE / 2);
    }

    #[test]
    fn test_presynaptic_trace() {
        let trace = PreSynapticTrace::new(DEFAULT_TRACE_TAU_US);
        
        trace.record_spike(1000, TRACE_SCALE);
        assert_eq!(trace.spike_count(), 1);
        assert!(trace.get(1000) > 0);
        
        // Should decay over time
        assert!(trace.get(100_000) < trace.get(1000));
    }

    #[test]
    fn test_postsynaptic_traces() {
        let trace = PostSynapticTrace::default_traces();
        
        trace.record_spike(1000);
        
        // Both traces should be positive
        assert!(trace.get_eligibility(1000) > 0);
        
        // Fast trace decays quicker than slow
        let fast_decay = 50_000;
        let eligibility_at_decay = trace.get_eligibility(1000 + fast_decay);
        assert!(eligibility_at_decay < trace.get_eligibility(1000));
    }

    #[test]
    fn test_triplet_stdp() {
        let triplet = TripletStdpTraces::new(100, 80);
        
        // Pre before post sequence
        triplet.record_pre_spike(1000);
        let ltp = triplet.record_post_spike(1100);
        
        // Should produce LTP (positive)
        assert!(ltp > 0);
    }

    #[test]
    fn test_trace_manager() {
        let manager = EligibilityTraceManager::new(4, 2);
        
        manager.record_pre(0, 1000);
        manager.record_post(0, 1100);
        
        let eligibility = manager.get_synapse_eligibility(0, 0, 1100);
        assert!(eligibility > 0);
    }
}
