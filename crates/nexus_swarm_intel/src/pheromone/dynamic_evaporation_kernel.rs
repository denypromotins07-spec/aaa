//! Dynamic Evaporation Kernel for Pheromone Decay.
//! 
//! Implements regime-adaptive pheromone evaporation where the decay rate ρ
//! is linked to market volatility and macro regime conditions.

use nexus_types::market::{MarketRegime, VolatilityState};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, Duration};

/// Default evaporation rate for normal market conditions
pub const DEFAULT_EVAPORATION_RATE: f64 = 0.1;

/// Minimum evaporation rate (never fully stop evaporation)
pub const MIN_EVAPORATION_RATE: f64 = 0.01;

/// Maximum evaporation rate (during flash crashes)
pub const MAX_EVAPORATION_RATE: f64 = 0.95;

/// Evaporation parameters tuned for different market regimes
#[derive(Debug, Clone, Copy)]
pub struct EvaporationParams {
    /// Base evaporation rate
    pub base_rho: f64,
    /// Volatility sensitivity multiplier
    pub volatility_sensitivity: f64,
    /// Regime-specific adjustment
    pub regime_adjustment: f64,
    /// Minimum time between updates (ns)
    pub update_interval_ns: u64,
}

impl Default for EvaporationParams {
    fn default() -> Self {
        Self {
            base_rho: DEFAULT_EVAPORATION_RATE,
            volatility_sensitivity: 0.5,
            regime_adjustment: 0.0,
            update_interval_ns: 1_000_000, // 1ms
        }
    }
}

/// Dynamic evaporation kernel that adapts ρ based on market conditions
pub struct DynamicEvaporationKernel {
    params: EvaporationParams,
    current_rho: f64,
    last_update_time: Instant,
    update_count: AtomicU64,
    /// Rolling window of recent volatility readings
    volatility_history: [f64; 8],
    volatility_idx: usize,
}

impl DynamicEvaporationKernel {
    pub fn new(params: EvaporationParams) -> Self {
        Self {
            params,
            current_rho: params.base_rho,
            last_update_time: Instant::now(),
            update_count: AtomicU64::new(0),
            volatility_history: [0.0; 8],
            volatility_idx: 0,
        }
    }

    /// Calculate dynamic evaporation rate based on market conditions
    pub fn calculate_evaporation_rate(
        &mut self,
        regime: MarketRegime,
        volatility: VolatilityState,
    ) -> f64 {
        let now = Instant::now();
        
        // Rate limit updates
        if now.duration_since(self.last_update_time).as_nanos() as u64 < self.params.update_interval_ns {
            return self.current_rho;
        }

        self.last_update_time = now;

        // Update volatility history
        self.volatility_history[self.volatility_idx] = volatility.realized_vol;
        self.volatility_idx = (self.volatility_idx + 1) % 8;

        // Calculate average recent volatility
        let avg_volatility: f64 = self.volatility_history.iter().sum::<f64>() / 8.0;

        // Base evaporation from params
        let mut rho = self.params.base_rho;

        // Volatility adjustment: higher volatility = faster evaporation
        // This forces the swarm to forget old paths quickly in turbulent markets
        let vol_factor = 1.0 + (avg_volatility * self.params.volatility_sensitivity);
        rho *= vol_factor;

        // Regime-specific adjustments
        rho += match regime {
            MarketRegime::Normal => 0.0,
            MarketRegime::TrendingUp => -0.02, // Slightly slower, trends persist
            MarketRegime::TrendingDown => 0.03, // Faster, downtrends can reverse quickly
            MarketRegime::HighVolatility => 0.15, // Much faster evaporation
            MarketRegime::FlashCrash => 0.4, // Extreme evaporation to force exploration
            MarketRegime::LiquidityCrisis => 0.25, // Fast evaporation
            MarketRegime::MeanReverting => -0.05, // Slower, mean reversion is predictable
        };

        // Apply regime adjustment from params
        rho += self.params.regime_adjustment;

        // Clamp to valid range
        self.current_rho = rho.clamp(MIN_EVAPORATION_RATE, MAX_EVAPORATION_RATE);
        
        self.update_count.fetch_add(1, Ordering::Relaxed);
        self.current_rho
    }

    /// Get current evaporation rate without recalculating
    pub fn current_rate(&self) -> f64 {
        self.current_rho
    }

    /// Get number of updates performed
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }

    /// Force recalculation regardless of timing
    pub fn force_recalculate(
        &mut self,
        regime: MarketRegime,
        volatility: VolatilityState,
    ) -> f64 {
        self.last_update_time = Instant::now() - Duration::from_secs(1);
        self.calculate_evaporation_rate(regime, volatility)
    }

    /// Get volatility-adjusted decay factor for single step
    pub fn decay_factor(&self) -> f64 {
        1.0 - self.current_rho
    }

    /// Apply evaporation to a pheromone value directly
    pub fn apply_to_pheromone(&self, pheromone: f64) -> f64 {
        (pheromone * self.decay_factor()).clamp(0.001, 1.0)
    }
}

/// Batch evaporation processor for SIMD optimization
pub struct BatchEvaporationProcessor {
    kernel: DynamicEvaporationKernel,
}

impl BatchEvaporationProcessor {
    pub fn new(kernel: DynamicEvaporationKernel) -> Self {
        Self { kernel }
    }

    /// Process batch of pheromone values with current evaporation rate
    pub fn process_batch(&self, pheromones: &mut [f64]) {
        let decay = self.kernel.decay_factor();
        
        for p in pheromones.iter_mut() {
            *p = (*p * decay).clamp(0.001, 1.0);
        }
    }

    /// Process with SIMD acceleration (AVX2)
    #[cfg(target_arch = "x86_64")]
    pub fn process_batch_simd(&self, pheromones: &mut [f64]) {
        use std::arch::x86_64::*;

        const LANE_WIDTH: usize = 4; // AVX processes 4 f64 at once
        let decay = self.kernel.decay_factor();
        
        // Broadcast decay factor to all lanes
        unsafe {
            let decay_vec = _mm256_set1_pd(decay);
            let min_vec = _mm256_set1_pd(0.001);
            let max_vec = _mm256_set1_pd(1.0);

            let len = pheromones.len();
            let simd_len = len - (len % LANE_WIDTH);

            for i in (0..simd_len).step_by(LANE_WIDTH) {
                let ptr = pheromones.as_mut_ptr().add(i);
                let mut v = _mm256_loadu_pd(ptr);
                v = _mm256_mul_pd(v, decay_vec);
                v = _mm256_max_pd(v, min_vec);
                v = _mm256_min_pd(v, max_vec);
                _mm256_storeu_pd(ptr, v);
            }

            // Handle remainder
            for i in simd_len..len {
                pheromones[i] = (pheromones[i] * decay).clamp(0.001, 1.0);
            }
        }
    }

    /// Get reference to underlying kernel
    pub fn kernel(&self) -> &DynamicEvaporationKernel {
        &self.kernel
    }

    /// Get mutable reference to underlying kernel
    pub fn kernel_mut(&mut self) -> &mut DynamicEvaporationKernel {
        &mut self.kernel
    }
}

/// Evaporation scheduler for asynchronous execution
pub struct EvaporationScheduler {
    kernel: DynamicEvaporationKernel,
    next_execution_time: Instant,
    execution_interval: Duration,
    enabled: bool,
}

impl EvaporationScheduler {
    pub fn new(kernel: DynamicEvaporationKernel, interval_ms: u64) -> Self {
        Self {
            kernel,
            next_execution_time: Instant::now(),
            execution_interval: Duration::from_millis(interval_ms),
            enabled: true,
        }
    }

    /// Check if evaporation should be executed now
    pub fn should_execute(&self) -> bool {
        self.enabled && Instant::now() >= self.next_execution_time
    }

    /// Mark evaporation as executed and schedule next
    pub fn mark_executed(&mut self) {
        self.next_execution_time = Instant::now() + self.execution_interval;
    }

    /// Get kernel for execution
    pub fn kernel(&self) -> &DynamicEvaporationKernel {
        &self.kernel
    }

    /// Get mutable kernel for updates
    pub fn kernel_mut(&mut self) -> &mut DynamicEvaporationKernel {
        &mut self.kernel
    }

    /// Enable/disable scheduler
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Update execution interval
    pub fn set_interval(&mut self, interval_ms: u64) {
        self.execution_interval = Duration::from_millis(interval_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_types::market::{MarketRegime, VolatilityState};

    #[test]
    fn test_default_evaporation() {
        let mut kernel = DynamicEvaporationKernel::new(EvaporationParams::default());
        let regime = MarketRegime::Normal;
        let volatility = VolatilityState {
            realized_vol: 0.2,
            implied_vol: 0.25,
            vol_of_vol: 0.1,
        };

        let rho = kernel.calculate_evaporation_rate(regime, volatility);
        assert!(rho >= MIN_EVAPORATION_RATE);
        assert!(rho <= MAX_EVAPORATION_RATE);
    }

    #[test]
    fn test_flash_crash_evaporation() {
        let mut kernel = DynamicEvaporationKernel::new(EvaporationParams::default());
        let regime = MarketRegime::FlashCrash;
        let volatility = VolatilityState {
            realized_vol: 2.0, // Extreme volatility
            implied_vol: 3.0,
            vol_of_vol: 1.5,
        };

        let rho = kernel.force_recalculate(regime, volatility);
        // Should be very high during flash crash
        assert!(rho > 0.5);
    }

    #[test]
    fn test_mean_reverting_evaporation() {
        let mut kernel = DynamicEvaporationKernel::new(EvaporationParams::default());
        let regime = MarketRegime::MeanReverting;
        let volatility = VolatilityState {
            realized_vol: 0.1, // Low volatility
            implied_vol: 0.15,
            vol_of_vol: 0.05,
        };

        let rho = kernel.force_recalculate(regime, volatility);
        // Should be lower than default for mean reverting
        assert!(rho < DEFAULT_EVAPORATION_RATE);
    }

    #[test]
    fn test_decay_factor_application() {
        let mut kernel = DynamicEvaporationKernel::new(EvaporationParams::default());
        let regime = MarketRegime::Normal;
        let volatility = VolatilityState {
            realized_vol: 0.2,
            implied_vol: 0.25,
            vol_of_vol: 0.1,
        };

        kernel.calculate_evaporation_rate(regime, volatility);
        let decay = kernel.decay_factor();

        let initial_pheromone = 0.8;
        let evaporated = kernel.apply_to_pheromone(initial_pheromone);
        
        assert!(evaporated < initial_pheromone);
        assert!(evaporated >= 0.001);
    }

    #[test]
    fn test_batch_processor() {
        let kernel = DynamicEvaporationKernel::new(EvaporationParams::default());
        let mut processor = BatchEvaporationProcessor::new(kernel);

        let mut pheromones = vec![0.5, 0.6, 0.7, 0.8, 0.9];
        processor.process_batch(&mut pheromones);

        // All should be reduced
        for (i, &p) in pheromones.iter().enumerate() {
            let expected_base = [0.5, 0.6, 0.7, 0.8, 0.9][i];
            assert!(p < expected_base);
            assert!(p >= 0.001);
        }
    }

    #[test]
    fn test_scheduler_timing() {
        let kernel = DynamicEvaporationKernel::new(EvaporationParams::default());
        let mut scheduler = EvaporationScheduler::new(kernel, 100); // 100ms

        assert!(!scheduler.should_execute()); // Just created, should wait
        
        // Simulate time passing
        std::thread::sleep(Duration::from_millis(110));
        assert!(scheduler.should_execute());

        scheduler.mark_executed();
        assert!(!scheduler.should_execute()); // Reset after execution
    }
}
