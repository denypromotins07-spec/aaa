//! Hawkes Process Intensity Calculator
//! 
//! Implements a real-time Hawkes Process to model self-exciting trade arrivals.
//! Detects micro-momentum bursts before they appear on price charts.
//! Zero-allocation implementation with fixed-size event history.

use nexus_core::memory::arena::BumpAllocator;
use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum number of events to track for Hawkes calculation
pub const MAX_HAWKES_EVENTS: usize = 1024;

/// Cache-line padded event record
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct HawkesEvent {
    /// Event timestamp in nanoseconds
    ts: u64,
    /// Event magnitude (e.g., volume, price impact)
    magnitude: f64,
    /// Is this a buy-initiated event?
    is_buy: bool,
    /// Padding
    _padding: [u8; 7],
}

impl Default for HawkesEvent {
    fn default() -> Self {
        Self {
            ts: 0,
            magnitude: 0.0,
            is_buy: false,
            _padding: [0u8; 7],
        }
    }
}

/// Hawkes Process parameters
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct HawkesParams {
    /// Base intensity (background rate)
    pub mu: f64,
    /// Excitation factor (how much each event increases intensity)
    pub alpha: f64,
    /// Decay rate (how quickly excitation decays)
    pub beta: f64,
    /// Kernel type: 0=exponential, 1=power-law
    pub kernel_type: u8,
    /// Power-law exponent (if kernel_type=1)
    pub power_exponent: f64,
    /// Padding
    _padding: [u8; 50],
}

impl Default for HawkesParams {
    fn default() -> Self {
        // Typical parameters for trade arrival modeling
        Self {
            mu: 0.1,      // Base rate of 0.1 events per unit time
            alpha: 0.8,   // Strong self-excitation
            beta: 1.0,    // Decay rate
            kernel_type: 0, // Exponential kernel
            power_exponent: 1.5,
            _padding: [0u8; 50],
        }
    }
}

/// Hawkes Process Intensity Calculator
pub struct HawkesIntensity {
    /// Event ring buffer
    events: [HawkesEvent; MAX_HAWKES_EVENTS],
    /// Write index
    write_idx: usize,
    /// Event count
    count: usize,
    /// Current intensity estimate
    current_intensity: f64,
    /// Last calculation timestamp
    last_ts: u64,
    /// Parameters
    params: HawkesParams,
    /// Separate buy/sell intensities
    buy_intensity: f64,
    sell_intensity: f64,
    /// Atomic counter for lock-free reads
    intensity_counter: AtomicU64,
}

unsafe impl Send for HawkesIntensity {}
unsafe impl Sync for HawkesIntensity {}

impl HawkesIntensity {
    pub fn new(_allocator: &BumpAllocator, params: HawkesParams) -> Self {
        Self {
            events: [HawkesEvent::default(); MAX_HAWKES_EVENTS],
            write_idx: 0,
            count: 0,
            current_intensity: params.mu,
            last_ts: 0,
            params,
            buy_intensity: params.mu / 2.0,
            sell_intensity: params.mu / 2.0,
            intensity_counter: AtomicU64::new(0),
        }
    }

    /// Process a new trade event - zero allocation
    #[inline]
    pub fn on_trade(&mut self, ts: u64, magnitude: f64, is_buy: bool) {
        // Store event
        let event = HawkesEvent {
            ts,
            magnitude,
            is_buy,
            _padding: [0u8; 7],
        };
        
        self.events[self.write_idx] = event;
        self.write_idx = (self.write_idx + 1) % MAX_HAWKES_EVENTS;
        if self.count < MAX_HAWKES_EVENTS {
            self.count += 1;
        }

        // Update intensity
        self.update_intensity(ts);
    }

    /// Update intensity using Hawkes process formula
    /// λ(t) = μ + Σ α * exp(-β * (t - t_i))
    #[inline]
    fn update_intensity(&mut self, current_ts: u64) {
        if self.count == 0 {
            self.current_intensity = self.params.mu;
            return;
        }

        let mut total_excitation = 0.0;
        let mut buy_excitation = 0.0;
        let mut sell_excitation = 0.0;

        // Sum over recent events (limited window for efficiency)
        let max_lookback = self.count.min(256); // Limit to recent events
        
        for i in 0..max_lookback {
            let idx = (self.write_idx + MAX_HAWKES_EVENTS - i - 1) % MAX_HAWKES_EVENTS;
            let event = &self.events[idx];
            
            if event.ts == 0 || current_ts <= event.ts {
                continue;
            }

            let dt = (current_ts - event.ts) as f64 / 1_000_000_000.0; // Convert to seconds
            
            // Exponential kernel: α * exp(-β * dt)
            let excitation = if self.params.kernel_type == 0 {
                self.params.alpha * (-self.params.beta * dt).exp()
            } else {
                // Power-law kernel
                let denom = 1.0 + dt.powf(self.params.power_exponent);
                self.params.alpha / denom
            };

            let weighted_excitation = excitation * event.magnitude;
            total_excitation += weighted_excitation;
            
            if event.is_buy {
                buy_excitation += weighted_excitation;
            } else {
                sell_excitation += weighted_excitation;
            }
        }

        self.current_intensity = self.params.mu + total_excitation;
        self.buy_intensity = self.params.mu / 2.0 + buy_excitation;
        self.sell_intensity = self.params.mu / 2.0 + sell_excitation;
        self.last_ts = current_ts;

        // Update atomic counter for lock-free reads
        self.intensity_counter.store(
            self.current_intensity.to_bits(),
            Ordering::Release,
        );
    }

    /// Get current total intensity (lock-free)
    #[inline]
    pub fn get_intensity(&self) -> f64 {
        f64::from_bits(self.intensity_counter.load(Ordering::Acquire))
    }

    /// Get buy-side intensity
    #[inline]
    pub fn get_buy_intensity(&self) -> f64 {
        self.buy_intensity
    }

    /// Get sell-side intensity
    #[inline]
    pub fn get_sell_intensity(&self) -> f64 {
        self.sell_intensity
    }

    /// Get intensity ratio (buy/sell) - indicates directional pressure
    #[inline]
    pub fn get_intensity_ratio(&self) -> f64 {
        let epsilon = 1e-10;
        self.buy_intensity / (self.sell_intensity + epsilon)
    }

    /// Get normalized intensity (relative to base rate)
    #[inline]
    pub fn get_normalized_intensity(&self) -> f64 {
        self.current_intensity / self.params.mu
    }

    /// Detect momentum burst - returns true if intensity exceeds threshold
    #[inline]
    pub fn is_burst_detected(&self, threshold_multiplier: f64) -> bool {
        self.current_intensity > self.params.mu * threshold_multiplier
    }

    /// Get the excitation level (how much above baseline)
    #[inline]
    pub fn get_excitation_level(&self) -> f64 {
        (self.current_intensity - self.params.mu) / self.params.mu
    }

    /// Reset the process
    #[inline]
    pub fn reset(&mut self) {
        self.events = [HawkesEvent::default(); MAX_HAWKES_EVENTS];
        self.write_idx = 0;
        self.count = 0;
        self.current_intensity = self.params.mu;
        self.buy_intensity = self.params.mu / 2.0;
        self.sell_intensity = self.params.mu / 2.0;
        self.last_ts = 0;
    }

    /// Update parameters
    #[inline]
    pub fn set_params(&mut self, params: HawkesParams) {
        self.params = params;
    }
}

/// Burst detection result
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Default)]
pub struct BurstSignal {
    /// Is a burst detected?
    pub is_burst: bool,
    /// Burst direction (positive = buy pressure, negative = sell pressure)
    pub direction: f64,
    /// Intensity level
    pub intensity: f64,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Timestamp
    pub ts: u64,
    /// Padding
    _padding: [u8; 24],
}

impl BurstSignal {
    #[inline]
    pub fn from_hawkes(hawkes: &HawkesIntensity, ts: u64, threshold: f64) -> Self {
        let is_burst = hawkes.is_burst_detected(threshold);
        let ratio = hawkes.get_intensity_ratio();
        
        // Direction: log ratio centered at 0
        let direction = if ratio > 1.0 {
            (ratio).ln().min(2.0)
        } else {
            -(1.0 / ratio).ln().max(-2.0)
        };

        let confidence = if is_burst {
            (hawkes.get_excitation_level() / threshold).min(1.0)
        } else {
            0.0
        };

        Self {
            is_burst,
            direction,
            intensity: hawkes.get_intensity(),
            confidence,
            ts,
            _padding: [0u8; 24],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_core::memory::arena::BumpAllocator;

    #[test]
    fn test_hawkes_base_intensity() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let params = HawkesParams {
            mu: 0.1,
            alpha: 0.8,
            beta: 1.0,
            ..Default::default()
        };
        let hawkes = HawkesIntensity::new(&allocator, params);

        // Initial intensity should be base rate
        assert!((hawkes.get_intensity() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_hawkes_self_excitation() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let params = HawkesParams {
            mu: 0.1,
            alpha: 0.8,
            beta: 0.5,
            ..Default::default()
        };
        let mut hawkes = HawkesIntensity::new(&allocator, params);

        // Simulate a burst of trades
        let base_ts = 1_000_000_000_000u64;
        for i in 0..10 {
            let ts = base_ts + i * 1_000_000; // 1ms apart
            hawkes.on_trade(ts, 1.0, i % 2 == 0);
        }

        // Intensity should increase due to self-excitation
        assert!(hawkes.get_intensity() > 0.1);
        
        // Buy and sell intensities should be roughly equal
        let ratio = hawkes.get_intensity_ratio();
        assert!((ratio - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_burst_detection() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let params = HawkesParams {
            mu: 0.1,
            alpha: 0.9,
            beta: 0.3,
            ..Default::default()
        };
        let mut hawkes = HawkesIntensity::new(&allocator, params);

        // Many rapid trades
        let base_ts = 1_000_000_000_000u64;
        for i in 0..50 {
            let ts = base_ts + i * 100_000; // 0.1ms apart
            hawkes.on_trade(ts, 1.0, true); // All buys
        }

        // Should detect burst
        assert!(hawkes.is_burst_detected(3.0));
        
        // Direction should be positive (buy pressure)
        assert!(hawkes.get_intensity_ratio() > 1.0);
    }

    #[test]
    fn test_intensity_decay() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let params = HawkesParams {
            mu: 0.1,
            alpha: 0.8,
            beta: 2.0, // Fast decay
            ..Default::default()
        };
        let mut hawkes = HawkesIntensity::new(&allocator, params);

        // Single trade
        hawkes.on_trade(1_000_000_000_000, 1.0, true);
        let intensity_immediate = hawkes.get_intensity();

        // Wait (simulate time passing)
        hawkes.on_trade(5_000_000_000_000, 0.0, false); // Zero magnitude, just to trigger update
        let intensity_later = hawkes.get_intensity();

        // Intensity should decay toward base rate
        assert!(intensity_later < intensity_immediate);
    }
}
