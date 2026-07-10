//! VPIN Spread Adjuster for Adverse Selection Protection.
//! Integrates with Stage 7 VPIN engine to widen spreads during toxic flow.
//! Zero-allocation, no unwrap/expect in hot paths.

/// Error types for VPIN operations
#[derive(Debug, Clone, PartialEq)]
pub enum VpinError {
    InvalidVpinValue,
    ConfigurationError,
}

/// Configuration for VPIN-based spread adjustment
#[derive(Debug, Clone)]
pub struct VpinSpreadConfig {
    /// Base spread multiplier
    pub base_multiplier: f64,
    /// Maximum spread multiplier during extreme toxicity
    pub max_multiplier: f64,
    /// VPIN threshold for starting to widen spreads
    pub vpin_threshold: f64,
    /// Sensitivity factor (how aggressively to widen)
    pub sensitivity: f64,
    /// Asymmetric adjustment: extra widening on toxic side
    pub asymmetric_factor: f64,
}

impl Default for VpinSpreadConfig {
    fn default() -> Self {
        Self {
            base_multiplier: 1.0,
            max_multiplier: 10.0,
            vpin_threshold: 0.3,
            sensitivity: 2.0,
            asymmetric_factor: 1.5,
        }
    }
}

/// Result of spread adjustment calculation
#[derive(Debug, Clone, Copy)]
pub struct AdjustedSpreads {
    /// Adjusted bid spread (distance from mid to bid)
    pub bid_spread: f64,
    /// Adjusted ask spread (distance from mid to ask)
    pub ask_spread: f64,
    /// Applied multiplier
    pub multiplier: f64,
    /// Toxicity direction: positive = buy-side toxic, negative = sell-side toxic
    pub toxicity_direction: f64,
}

/// VPIN Spread Adjuster
pub struct VpinSpreadAdjuster {
    config: VpinSpreadConfig,
    /// Current VPIN value (0 to 1)
    current_vpin: f64,
    /// Signed VPIN: positive for buy imbalance, negative for sell imbalance
    signed_vpin: f64,
    /// Rolling average VPIN for smoothing
    rolling_vpin: f64,
    /// Smoothing factor for exponential moving average
    smoothing_alpha: f64,
}

impl VpinSpreadAdjuster {
    pub fn new(config: VpinSpreadConfig, smoothing_alpha: f64) -> Result<Self, VpinError> {
        if smoothing_alpha < 0.0 || smoothing_alpha > 1.0 {
            return Err(VpinError::ConfigurationError);
        }
        
        Ok(Self {
            config,
            current_vpin: 0.0,
            signed_vpin: 0.0,
            rolling_vpin: 0.0,
            smoothing_alpha,
        })
    }
    
    /// Update with new VPIN value from Stage 7 engine
    #[inline(always)]
    pub fn update_vpin(&mut self, vpin: f64, buy_volume_imbalance: f64) -> Result<(), VpinError> {
        if vpin < 0.0 || vpin > 1.0 {
            return Err(VpinError::InvalidVpinValue);
        }
        
        self.current_vpin = vpin;
        // Signed VPIN: direction matters for asymmetric adjustment
        self.signed_vpin = vpin * buy_volume_imbalance.clamp(-1.0, 1.0);
        
        // Exponential moving average for smoothing
        self.rolling_vpin = self.smoothing_alpha * vpin 
            + (1.0 - self.smoothing_alpha) * self.rolling_vpin;
        
        Ok(())
    }
    
    /// Calculate adjusted spreads based on current VPIN
    /// 
    /// # Arguments
    /// * `base_bid_spread` - Normal bid spread (without toxicity adjustment)
    /// * `base_ask_spread` - Normal ask spread (without toxicity adjustment)
    /// 
    /// Returns adjusted spreads that widen when VPIN indicates toxic flow
    #[inline(always)]
    pub fn calculate_adjusted_spreads(
        &self,
        base_bid_spread: f64,
        base_ask_spread: f64,
    ) -> AdjustedSpreads {
        let vpin = self.rolling_vpin.max(self.current_vpin);
        
        // Check if VPIN is below threshold (no adjustment needed)
        if vpin <= self.config.vpin_threshold {
            return AdjustedSpreads {
                bid_spread: base_bid_spread * self.config.base_multiplier,
                ask_spread: base_ask_spread * self.config.base_multiplier,
                multiplier: self.config.base_multiplier,
                toxicity_direction: self.signed_vpin,
            };
        }
        
        // Calculate multiplier based on how far VPIN exceeds threshold
        let excess_vpin = (vpin - self.config.vpin_threshold) 
            / (1.0 - self.config.vpin_threshold);
        
        let base_multiplier = self.config.base_multiplier 
            + self.config.sensitivity * excess_vpin;
        
        let multiplier = base_multiplier.min(self.config.max_multiplier);
        
        // Apply asymmetric adjustment based on toxicity direction
        let (bid_multiplier, ask_multiplier) = if self.signed_vpin > 0.0 {
            // Buy-side toxicity: widen ask more (protect against informed buyers)
            let asym = 1.0 + self.config.asymmetric_factor * self.signed_vpin;
            (multiplier, multiplier * asym)
        } else if self.signed_vpin < 0.0 {
            // Sell-side toxicity: widen bid more (protect against informed sellers)
            let asym = 1.0 + self.config.asymmetric_factor * (-self.signed_vpin);
            (multiplier * asym, multiplier)
        } else {
            (multiplier, multiplier)
        };
        
        AdjustedSpreads {
            bid_spread: base_bid_spread * bid_multiplier,
            ask_spread: base_ask_spread * ask_multiplier,
            multiplier,
            toxicity_direction: self.signed_vpin,
        }
    }
    
    /// Get toxicity level (0.0 = calm, 1.0 = extremely toxic)
    #[inline(always)]
    pub fn toxicity_level(&self) -> f64 {
        self.rolling_vpin.max(self.current_vpin)
    }
    
    /// Check if market is currently toxic
    #[inline(always)]
    pub fn is_toxic(&self) -> bool {
        self.rolling_vpin > self.config.vpin_threshold 
            || self.current_vpin > self.config.vpin_threshold
    }
    
    /// Get recommended action based on toxicity
    #[inline(always)]
    pub fn get_recommendation(&self) -> ToxicityAction {
        let vpin = self.rolling_vpin.max(self.current_vpin);
        
        if vpin < self.config.vpin_threshold {
            ToxicityAction::NormalQuoting
        } else if vpin < 0.5 {
            ToxicityAction::WidenSpreads
        } else if vpin < 0.7 {
            ToxicityAction::ReduceSize
        } else {
            ToxicityAction::HaltQuoting
        }
    }
    
    /// Reset VPIN tracker
    pub fn reset(&mut self) {
        self.current_vpin = 0.0;
        self.signed_vpin = 0.0;
        self.rolling_vpin = 0.0;
    }
}

/// Recommended action based on toxicity level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToxicityAction {
    NormalQuoting,
    WidenSpreads,
    ReduceSize,
    HaltQuoting,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_normal_market() {
        let config = VpinSpreadConfig::default();
        let mut adjuster = VpinSpreadAdjuster::new(config, 0.3).unwrap();
        
        adjuster.update_vpin(0.1, 0.0).unwrap();
        
        let adjusted = adjuster.calculate_adjusted_spreads(0.01, 0.01);
        
        // Low VPIN should not widen spreads much
        assert!(adjusted.multiplier <= 1.1);
    }
    
    #[test]
    fn test_toxic_buy_side() {
        let config = VpinSpreadConfig::default();
        let mut adjuster = VpinSpreadAdjuster::new(config, 0.3).unwrap();
        
        // High VPIN with buy imbalance
        adjuster.update_vpin(0.8, 0.9).unwrap();
        
        let adjusted = adjuster.calculate_adjusted_spreads(0.01, 0.01);
        
        // Spreads should be significantly widened
        assert!(adjusted.multiplier > 2.0);
        // Ask spread should be wider than bid (buy-side toxicity)
        assert!(adjusted.ask_spread > adjusted.bid_spread);
    }
    
    #[test]
    fn test_toxic_sell_side() {
        let config = VpinSpreadConfig::default();
        let mut adjuster = VpinSpreadAdjuster::new(config, 0.3).unwrap();
        
        // High VPIN with sell imbalance
        adjuster.update_vpin(0.8, -0.9).unwrap();
        
        let adjusted = adjuster.calculate_adjusted_spreads(0.01, 0.01);
        
        // Bid spread should be wider than ask (sell-side toxicity)
        assert!(adjusted.bid_spread > adjusted.ask_spread);
    }
}
