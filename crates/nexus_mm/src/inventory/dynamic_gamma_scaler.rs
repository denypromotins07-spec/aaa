//! Dynamic Risk Aversion (Gamma) Scaler linked to EVT Tail-Risk Engine.
//! Adjusts gamma based on market stress conditions from Stage 11.
//! Zero-allocation, no unwrap/expect in hot paths.

use crate::pde::hjb_equation::AvellanedaStoikovParams;

/// Error types for gamma scaling
#[derive(Debug, Clone, PartialEq)]
pub enum GammaScalerError {
    InvalidBaseGamma,
    EvtDataUnavailable,
    NumericalOverflow,
}

/// Configuration for dynamic gamma scaling
#[derive(Debug, Clone)]
pub struct GammaScalerConfig {
    /// Base risk aversion (gamma_0)
    pub base_gamma: f64,
    /// Minimum gamma (floor during calm markets)
    pub min_gamma: f64,
    /// Maximum gamma (cap during extreme stress)
    pub max_gamma: f64,
    /// Sensitivity to EVT tail probability
    pub evt_sensitivity: f64,
    /// Sensitivity to volatility regime
    pub vol_sensitivity: f64,
    /// Lookback window for rolling statistics (in ticks)
    pub lookback_window: usize,
}

impl Default for GammaScalerConfig {
    fn default() -> Self {
        Self {
            base_gamma: 0.1,
            min_gamma: 0.01,
            max_gamma: 10.0,
            evt_sensitivity: 5.0,
            vol_sensitivity: 2.0,
            lookback_window: 1000,
        }
    }
}

/// Rolling statistics tracker for zero-allocation updates
pub struct RollingStats {
    count: usize,
    sum: f64,
    sum_sq: f64,
    min_val: f64,
    max_val: f64,
    window_size: usize,
    /// Circular buffer for sliding window (pre-allocated)
    buffer: Vec<f64>,
    head: usize,
}

impl RollingStats {
    pub fn new(window_size: usize) -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            min_val: f64::MAX,
            max_val: f64::MIN,
            window_size,
            buffer: vec![0.0; window_size],
            head: 0,
        }
    }
    
    #[inline(always)]
    pub fn update(&mut self, value: f64) {
        if self.count >= self.window_size {
            // Remove oldest value from circular buffer
            let old_value = self.buffer[self.head];
            self.sum -= old_value;
            self.sum_sq -= old_value * old_value;
        } else {
            self.count += 1;
        }
        
        // Add new value
        self.buffer[self.head] = value;
        self.sum += value;
        self.sum_sq += value * value;
        
        // Update min/max (simplified - full implementation would use deque)
        self.min_val = self.min_val.min(value);
        self.max_val = self.max_val.max(value);
        
        // Advance circular buffer head
        self.head = (self.head + 1) % self.window_size;
    }
    
    #[inline(always)]
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum / self.count as f64
    }
    
    #[inline(always)]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let mean = self.mean();
        (self.sum_sq / self.count as f64) - (mean * mean)
    }
    
    #[inline(always)]
    pub fn std_dev(&self) -> f64 {
        self.variance().max(0.0).sqrt()
    }
    
    #[inline(always)]
    pub fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0;
        self.sum_sq = 0.0;
        self.min_val = f64::MAX;
        self.max_val = f64::MIN;
        self.head = 0;
    }
}

/// Dynamic Gamma Scaler that adjusts risk aversion based on market conditions
pub struct DynamicGammaScaler {
    config: GammaScalerConfig,
    /// Rolling statistics for volatility estimation
    vol_stats: RollingStats,
    /// Current scaled gamma value
    current_gamma: f64,
    /// Last update timestamp (nanoseconds)
    last_update_ns: u64,
}

impl DynamicGammaScaler {
    pub fn new(config: GammaScalerConfig) -> Result<Self, GammaScalerError> {
        if config.base_gamma <= 0.0 || config.min_gamma <= 0.0 
            || config.max_gamma < config.min_gamma {
            return Err(GammaScalerError::InvalidBaseGamma);
        }
        
        Ok(Self {
            config,
            vol_stats: RollingStats::new(config.lookback_window),
            current_gamma: config.base_gamma,
            last_update_ns: 0,
        })
    }
    
    /// Update with new volatility observation
    #[inline(always)]
    pub fn update_volatility(&mut self, volatility: f64, timestamp_ns: u64) {
        self.vol_stats.update(volatility);
        self.last_update_ns = timestamp_ns;
    }
    
    /// Calculate dynamic gamma based on EVT tail probability and volatility
    /// 
    /// Formula: gamma = gamma_0 * exp(evt_sensitivity * P_tail + vol_sensitivity * z_score)
    /// 
    /// # Arguments
    /// * `evt_tail_prob` - Probability of tail event from Stage 11 EVT engine (0 to 1)
    /// * `current_vol` - Current realized volatility
    /// 
    /// Returns clamped gamma value in [min_gamma, max_gamma]
    #[inline(always)]
    pub fn calculate_gamma(
        &self,
        evt_tail_prob: f64,
        current_vol: f64,
    ) -> f64 {
        // Validate inputs
        if evt_tail_prob < 0.0 || evt_tail_prob > 1.0 {
            return self.config.base_gamma.clamp(self.config.min_gamma, self.config.max_gamma);
        }
        
        // Calculate volatility z-score
        let vol_mean = self.vol_stats.mean();
        let vol_std = self.vol_stats.std_dev();
        
        let vol_zscore = if vol_std > 1e-15 {
            (current_vol - vol_mean) / vol_std
        } else {
            0.0
        };
        
        // Calculate exponential adjustment
        let evt_component = self.config.evt_sensitivity * evt_tail_prob;
        let vol_component = self.config.vol_sensitivity * vol_zscore.max(0.0);
        
        let adjustment_factor = (evt_component + vol_component).exp();
        
        // Apply to base gamma
        let scaled_gamma = self.config.base_gamma * adjustment_factor;
        
        // Clamp to valid range to prevent numerical overflow
        scaled_gamma.clamp(self.config.min_gamma, self.config.max_gamma)
    }
    
    /// Update and get new gamma value
    #[inline(always)]
    pub fn update_and_get_gamma(
        &mut self,
        evt_tail_prob: f64,
        current_vol: f64,
        timestamp_ns: u64,
    ) -> f64 {
        self.update_volatility(current_vol, timestamp_ns);
        self.current_gamma = self.calculate_gamma(evt_tail_prob, current_vol);
        self.current_gamma
    }
    
    /// Get current gamma value
    #[inline(always)]
    pub const fn current_gamma(&self) -> f64 {
        self.current_gamma
    }
    
    /// Check if market is in high-stress regime
    #[inline(always)]
    pub fn is_high_stress(&self, evt_tail_prob: f64) -> bool {
        evt_tail_prob > 0.05 || self.current_gamma > self.config.base_gamma * 2.0
    }
    
    /// Get stress level (0.0 = calm, 1.0 = extreme stress)
    #[inline(always)]
    pub fn stress_level(&self, evt_tail_prob: f64) -> f64 {
        let gamma_ratio = (self.current_gamma - self.config.base_gamma) 
            / (self.config.max_gamma - self.config.base_gamma);
        let evt_component = evt_tail_prob;
        
        ((gamma_ratio + evt_component) / 2.0).clamp(0.0, 1.0)
    }
    
    /// Reset scaler to base state
    pub fn reset(&mut self) {
        self.vol_stats.reset();
        self.current_gamma = self.config.base_gamma;
        self.last_update_ns = 0;
    }
}

/// Integration with Avellaneda-Stoikov parameters
pub struct AdaptiveAvellanedaStoikov {
    base_params: AvellanedaStoikovParams,
    gamma_scaler: DynamicGammaScaler,
}

impl AdaptiveAvellanedaStoikov {
    pub fn new(
        base_params: AvellanedaStoikovParams,
        gamma_config: GammaScalerConfig,
    ) -> Result<Self, GammaScalerError> {
        let mut config = gamma_config;
        config.base_gamma = base_params.gamma;
        config.min_gamma = config.min_gamma.min(base_params.gamma);
        config.max_gamma = config.max_gamma.max(base_params.gamma);
        
        let gamma_scaler = DynamicGammaScaler::new(config)?;
        
        Ok(Self {
            base_params,
            gamma_scaler,
        })
    }
    
    /// Get parameters with dynamically adjusted gamma
    #[inline(always)]
    pub fn get_adaptive_params(&self, evt_tail_prob: f64, current_vol: f64) -> AvellanedaStoikovParams {
        let adaptive_gamma = self.gamma_scaler.calculate_gamma(evt_tail_prob, current_vol);
        
        AvellanedaStoikovParams {
            gamma: adaptive_gamma,
            ..self.base_params.clone()
        }
    }
    
    /// Update and get adaptive parameters
    #[inline(always)]
    pub fn update_and_get_params(
        &mut self,
        evt_tail_prob: f64,
        current_vol: f64,
        timestamp_ns: u64,
    ) -> AvellanedaStoikovParams {
        let adaptive_gamma = self.gamma_scaler.update_and_get_gamma(
            evt_tail_prob,
            current_vol,
            timestamp_ns,
        );
        
        AvellanedaStoikovParams {
            gamma: adaptive_gamma,
            ..self.base_params.clone()
        }
    }
    
    /// Get current stress level
    #[inline(always)]
    pub fn stress_level(&self, evt_tail_prob: f64) -> f64 {
        self.gamma_scaler.stress_level(evt_tail_prob)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gamma_scaler_basic() {
        let config = GammaScalerConfig {
            base_gamma: 0.1,
            min_gamma: 0.01,
            max_gamma: 10.0,
            evt_sensitivity: 5.0,
            vol_sensitivity: 2.0,
            lookback_window: 100,
        };
        
        let mut scaler = DynamicGammaScaler::new(config).unwrap();
        
        // Calm market: low tail prob, normal vol
        let gamma_calm = scaler.calculate_gamma(0.01, 0.02);
        assert!(gamma_calm >= 0.01);
        assert!(gamma_calm <= 10.0);
        
        // Stressful market: high tail prob
        for _ in 0..50 {
            scaler.update_volatility(0.02, 0);
        }
        let gamma_stress = scaler.calculate_gamma(0.3, 0.1);
        
        // Stressed gamma should be higher than calm
        assert!(gamma_stress > gamma_calm);
    }
    
    #[test]
    fn test_gamma_clamping() {
        let config = GammaScalerConfig {
            base_gamma: 0.1,
            min_gamma: 0.01,
            max_gamma: 1.0,
            evt_sensitivity: 10.0,
            vol_sensitivity: 5.0,
            lookback_window: 100,
        };
        
        let scaler = DynamicGammaScaler::new(config).unwrap();
        
        // Extreme stress should not exceed max_gamma
        let gamma_extreme = scaler.calculate_gamma(1.0, 1.0);
        assert!(gamma_extreme <= 1.0);
        
        // Very calm should not go below min_gamma
        let gamma_calm = scaler.calculate_gamma(0.0, 0.001);
        assert!(gamma_calm >= 0.01);
    }
    
    #[test]
    fn test_rolling_stats() {
        let mut stats = RollingStats::new(10);
        
        for i in 1..=10 {
            stats.update(i as f64);
        }
        
        assert!((stats.mean() - 5.5).abs() < 1e-10);
        assert!(stats.std_dev() > 0.0);
    }
}
