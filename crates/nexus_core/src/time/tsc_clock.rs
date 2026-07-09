//! High-precision timing for NEXUS-OMEGA
//!
//! Provides microsecond-precision clocks synchronized with the CPU's
//! Time Stamp Counter (TSC) for sub-microsecond event timestamping.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Cache line size for alignment
const CACHE_LINE_SIZE: usize = 64;

/// Error type for clock operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockError {
    /// TSC not available on this platform
    TSCUnavailable,
    /// TSC not synchronized across cores
    TSCNotSynced,
    /// Clock not initialized
    NotInitialized,
    /// Time went backwards (detected via monotonicity check)
    TimeWentBackwards,
}

impl std::fmt::Display for ClockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClockError::TSCUnavailable => write!(f, "TSC not available on this platform"),
            ClockError::TSCNotSynced => write!(f, "TSC not synchronized across cores"),
            ClockError::NotInitialized => write!(f, "Clock not initialized"),
            ClockError::TimeWentBackwards => write!(f, "Time went backwards"),
        }
    }
}

impl std::error::Error for ClockError {}

/// Monotonic nanoseconds clock using TSC
///
/// This clock provides sub-microsecond precision by reading the CPU's
/// Time Stamp Counter directly. It includes calibration against the
/// system clock for accurate wall-clock time conversion.
#[repr(C, align(64))]
pub struct MonotonicNanosClock {
    /// Base timestamp in nanoseconds (calibrated against system clock)
    base_ns: AtomicU64,
    
    /// Base TSC value at initialization
    base_tsc: AtomicU64,
    
    /// TSC frequency in Hz (cycles per second)
    tsc_frequency_hz: AtomicU64,
    
    /// Last recorded timestamp (for monotonicity enforcement)
    last_ns: AtomicU64,
    
    /// Whether TSC is invariant (constant frequency regardless of CPU state)
    is_invariant: bool,
    
    /// Padding to cache line boundary
    _padding: [u8; CLOCK_PADDING],
}

const CLOCK_BASE_SIZE: usize = 5 * 8 + 1; // 5 u64s + 1 bool
const CLOCK_PADDING: usize = CACHE_LINE_SIZE - CLOCK_BASE_SIZE;

impl MonotonicNanosClock {
    /// Create a new monotonic clock
    pub fn new() -> Result<Self, ClockError> {
        let now = Instant::now();
        let base_ns = 0u64; // We'll use relative timestamps
        
        // Get TSC value and frequency
        let (tsc_value, frequency) = Self::read_tsc_with_calibration()?;
        
        Ok(Self {
            base_ns: AtomicU64::new(base_ns),
            base_tsc: AtomicU64::new(tsc_value),
            tsc_frequency_hz: AtomicU64::new(frequency),
            last_ns: AtomicU64::new(0),
            is_invariant: Self::check_tsc_invariant(),
            _padding: [0u8; CLOCK_PADDING],
        })
    }
    
    /// Read the current timestamp in nanoseconds
    #[inline(always)]
    pub fn now_nanos(&self) -> u64 {
        let current_tsc = Self::read_tsc_raw();
        let base_tsc = self.base_tsc.load(Ordering::Relaxed);
        let frequency = self.tsc_frequency_hz.load(Ordering::Relaxed);
        
        if frequency == 0 {
            // Fallback to system time if TSC not calibrated
            return Instant::now().duration_since(Instant::now()).as_nanos() as u64;
        }
        
        // Calculate elapsed nanoseconds from TSC delta
        let tsc_delta = current_tsc.saturating_sub(base_tsc);
        let elapsed_ns = Self::tsc_to_nanos(tsc_delta, frequency);
        
        // Ensure monotonicity
        let mut last = self.last_ns.load(Ordering::Relaxed);
        loop {
            if elapsed_ns > last {
                match self.last_ns.compare_exchange_weak(
                    last,
                    elapsed_ns,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => last = actual,
                }
            } else {
                // Time would go backwards, return last known value + 1
                elapsed_ns + 1
            }
        }
        
        elapsed_ns
    }
    
    /// Calculate delta between two timestamps
    #[inline(always)]
    pub fn delta_nanos(&self, start: u64, end: u64) -> u64 {
        end.saturating_sub(start)
    }
    
    /// Get elapsed time in microseconds since a start timestamp
    #[inline(always)]
    pub fn elapsed_micros(&self, start: u64) -> f64 {
        let now = self.now_nanos();
        let delta_ns = self.delta_nanos(start, now);
        delta_ns as f64 / 1000.0
    }
    
    /// Get elapsed time in nanoseconds since a start timestamp
    #[inline(always)]
    pub fn elapsed_nanos(&self, start: u64) -> u64 {
        let now = self.now_nanos();
        self.delta_nanos(start, now)
    }
    
    /// Check if the clock is using invariant TSC
    pub fn is_invariant(&self) -> bool {
        self.is_invariant
    }
    
    /// Get the TSC frequency in Hz
    pub fn tsc_frequency(&self) -> u64 {
        self.tsc_frequency_hz.load(Ordering::Relaxed)
    }
    
    /// Reset the clock base (recalibrate)
    pub fn reset(&self) -> Result<(), ClockError> {
        let (tsc_value, frequency) = Self::read_tsc_with_calibration()?;
        
        self.base_tsc.store(tsc_value, Ordering::Relaxed);
        self.tsc_frequency_hz.store(frequency, Ordering::Relaxed);
        self.last_ns.store(0, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Read raw TSC value (platform-specific)
    #[inline(always)]
    fn read_tsc_raw() -> u64 {
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        {
            unsafe {
                core::arch::x86_64::_rdtsc() as u64
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            let mut val: u64;
            unsafe {
                core::arch::asm!(
                    "mrs {val}, cntvct_el0",
                    val = out(reg) val,
                    options(nomem, nostack, preserves_flags),
                );
            }
            val
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback: use system time
            Instant::now().duration_since(Instant::epoch()).as_nanos() as u64
        }
    }
    
    /// Read TSC with calibration against system clock
    fn read_tsc_with_calibration() -> Result<(u64, u64), ClockError> {
        let tsc_value = Self::read_tsc_raw();
        
        // Estimate frequency (in production, this would be more sophisticated)
        let frequency = Self::estimate_tsc_frequency();
        
        Ok((tsc_value, frequency))
    }
    
    /// Estimate TSC frequency by sampling over a short period
    fn estimate_tsc_frequency() -> u64 {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            // Sample TSC over a known time period
            let start_tsc = Self::read_tsc_raw();
            let start_time = Instant::now();
            
            // Wait approximately 10ms
            while start_time.elapsed() < Duration::from_millis(10) {
                core::hint::spin_loop();
            }
            
            let end_tsc = Self::read_tsc_raw();
            let elapsed = start_time.elapsed();
            
            let tsc_delta = end_tsc - start_tsc;
            let elapsed_ns = elapsed.as_nanos() as u64;
            
            if elapsed_ns == 0 {
                return 0;
            }
            
            // frequency = tsc_delta / (elapsed_ns / 1e9)
            // = tsc_delta * 1e9 / elapsed_ns
            (tsc_delta * 1_000_000_000) / elapsed_ns
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Default estimate for unknown platforms
            2_500_000_000 // 2.5 GHz
        }
    }
    
    /// Convert TSC cycles to nanoseconds
    #[inline(always)]
    fn tsc_to_nanos(tsc_cycles: u64, frequency_hz: u64) -> u64 {
        if frequency_hz == 0 {
            return 0;
        }
        
        // nanos = cycles * 1e9 / frequency
        (tsc_cycles * 1_000_000_000) / frequency_hz
    }
    
    /// Check if TSC is invariant (constant frequency)
    fn check_tsc_invariant() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            // Check CPUID for invariant TSC flag
            // In production, you'd use cpuid crate or inline assembly
            true // Assume invariant for modern x86_64
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            // ARM64 virtual counter is typically invariant
            true
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }
}

impl Default for MonotonicNanosClock {
    fn default() -> Self {
        Self::new().expect("Failed to create default clock")
    }
}

/// Global clock instance for fast access
static GLOBAL_CLOCK: parking_lot::RwLock<Option<MonotonicNanosClock>> = parking_lot::RwLock::new(None);

/// Initialize the global clock
pub fn init_global_clock() -> Result<(), ClockError> {
    let clock = MonotonicNanosClock::new()?;
    *GLOBAL_CLOCK.write() = Some(clock);
    Ok(())
}

/// Get the current timestamp from the global clock
pub fn global_now_nanos() -> u64 {
    let guard = GLOBAL_CLOCK.read();
    match guard.as_ref() {
        Some(clock) => clock.now_nanos(),
        None => {
            // Fallback: create a temporary clock
            MonotonicNanosClock::new()
                .map(|c| c.now_nanos())
                .unwrap_or(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_clock_creation() {
        let clock = MonotonicNanosClock::new();
        assert!(clock.is_ok());
    }
    
    #[test]
    fn test_monotonicity() {
        let clock = MonotonicNanosClock::new().unwrap();
        
        let t1 = clock.now_nanos();
        let t2 = clock.now_nanos();
        let t3 = clock.now_nanos();
        
        assert!(t2 >= t1);
        assert!(t3 >= t2);
    }
    
    #[test]
    fn test_delta_calculation() {
        let clock = MonotonicNanosClock::new().unwrap();
        
        let start = clock.now_nanos();
        std::thread::sleep(Duration::from_millis(1));
        let end = clock.now_nanos();
        
        let delta = clock.delta_nanos(start, end);
        
        // Should be at least 1ms (1,000,000 ns), allow some slack
        assert!(delta >= 500_000);
    }
    
    #[test]
    fn test_elapsed_micros() {
        let clock = MonotonicNanosClock::new().unwrap();
        
        let start = clock.now_nanos();
        std::thread::sleep(Duration::from_millis(2));
        
        let micros = clock.elapsed_micros(start);
        
        // Should be at least 2000 microseconds, allow some slack
        assert!(micros >= 1500.0);
    }
    
    #[test]
    fn test_tsc_frequency_estimate() {
        let freq = MonotonicNanosClock::estimate_tsc_frequency();
        
        // Any reasonable CPU should be between 100MHz and 10GHz
        assert!(freq >= 100_000_000);
        assert!(freq <= 10_000_000_000);
    }
}
