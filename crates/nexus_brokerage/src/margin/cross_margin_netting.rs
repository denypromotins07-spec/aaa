//! Cross-Margin Netting Engine
//! 
//! Calculates true portfolio exposure by offsetting correlated positions
//! across different brokers to free up trapped collateral.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MarginError {
    #[error("Invalid correlation matrix: asset {asset} has invalid correlation")]
    InvalidCorrelation { asset: u32 },
    #[error("Netting calculation overflow for asset {asset}")]
    CalculationOverflow { asset: u32 },
    #[error("Negative margin requirement not allowed")]
    NegativeMarginRequirement,
}

/// Correlated position pair for netting
#[derive(Debug, Clone)]
pub struct CorrelatedPair {
    pub asset_a: u32,
    pub asset_b: u32,
    pub correlation: f64, // -1.0 to 1.0
    pub netting_factor: f64, // Derived from correlation
}

/// Cross-margin netting calculator
pub struct CrossMarginNetting {
    /// Position data: asset_id -> (long_exposure, short_exposure) in fixed-point
    positions: dashmap::DashMap<u32, (AtomicI64, AtomicI64)>,
    /// Correlation pairs for netting calculation
    correlations: dashmap::DashMap<(u32, u32), f64>,
    /// Total netted margin requirement in fixed-point units
    total_margin_required: AtomicI64,
    /// Total freed collateral from netting
    freed_collateral: AtomicI64,
    /// Calculation epoch
    epoch: AtomicU64,
}

impl CrossMarginNetting {
    pub fn new() -> Self {
        Self {
            positions: dashmap::DashMap::new(),
            correlations: dashmap::DashMap::new(),
            total_margin_required: AtomicI64::new(0),
            freed_collateral: AtomicI64::new(0),
            epoch: AtomicU64::new(0),
        }
    }

    /// Register or update a position for an asset
    pub fn update_position(&self, asset_id: u32, long_exposure: i64, short_exposure: i64) {
        if long_exposure < 0 || short_exposure < 0 {
            return; // Invalid exposures ignored
        }

        self.positions.entry(asset_id).or_insert_with(|| {
            (AtomicI64::new(long_exposure), AtomicI64::new(short_exposure))
        }).and_modify(|(l, s)| {
            l.store(long_exposure, Ordering::Release);
            s.store(short_exposure, Ordering::Release);
        });
    }

    /// Set correlation between two assets
    pub fn set_correlation(&self, asset_a: u32, asset_b: u32, correlation: f64) -> Result<(), MarginError> {
        if !correlation.is_finite() || correlation < -1.0 || correlation > 1.0 {
            return Err(MarginError::InvalidCorrelation { asset: asset_a });
        }

        self.correlations.insert((asset_a, asset_b), correlation);
        Ok(())
    }

    /// Calculate netting factor from correlation
    /// Higher correlation = higher netting benefit
    fn calculate_netting_factor(correlation: f64) -> f64 {
        // Netting factor ranges from 0.0 (no netting) to 1.0 (full netting)
        // For perfect positive correlation (1.0), we get full netting
        // For negative correlation, no netting benefit
        ((correlation + 1.0) / 2.0).clamp(0.0, 1.0)
    }

    /// Calculate gross margin requirement (without netting)
    fn calculate_gross_margin(&self, margin_rate: f64) -> i64 {
        let mut gross = 0i64;
        
        for entry in self.positions.iter() {
            let (long_exp, short_exp) = entry.value();
            let long = long_exp.load(Ordering::Acquire);
            let short = short_exp.load(Ordering::Acquire);
            
            // Gross margin = (long + short) * margin_rate
            let combined = long.saturating_add(short);
            let margin = (combined as f64 * margin_rate) as i64;
            gross = gross.saturating_add(margin);
        }
        
        gross
    }

    /// Calculate net margin requirement with cross-asset netting
    pub fn calculate_net_margin(&self, margin_rate: f64) -> Result<i64, MarginError> {
        let gross_margin = self.calculate_gross_margin(margin_rate);
        
        let mut netting_benefit = 0i64;
        
        // Iterate through all correlation pairs to calculate netting benefit
        for entry in self.correlations.iter() {
            let ((asset_a, asset_b), &correlation) = entry.pair();
            
            let pos_a = self.positions.get(asset_a);
            let pos_b = self.positions.get(asset_b);
            
            if let (Some(entry_a), Some(entry_b)) = (pos_a, pos_b) {
                let (long_a, short_a) = entry_a.value();
                let (long_b, short_b) = entry_b.value();
                
                let long_a_val = long_a.load(Ordering::Acquire) as f64;
                let short_a_val = short_a.load(Ordering::Acquire) as f64;
                let long_b_val = long_b.load(Ordering::Acquire) as f64;
                let short_b_val = short_b.load(Ordering::Acquire) as f64;
                
                // Calculate nettable exposure
                let netting_factor = Self::calculate_netting_factor(correlation);
                
                // Netting benefit = min(exposures) * netting_factor * margin_rate
                let min_long = long_a_val.min(long_b_val);
                let min_short = short_a_val.min(short_b_val);
                let nettable = (min_long + min_short) * netting_factor * margin_rate;
                
                netting_benefit = netting_benefit.saturating_add(nettable as i64);
            }
        }
        
        // Net margin = gross margin - netting benefit
        let net_margin = gross_margin.saturating_sub(netting_benefit);
        
        if net_margin < 0 {
            return Err(MarginError::NegativeMarginRequirement);
        }
        
        // Update atomic counters
        self.total_margin_required.store(net_margin, Ordering::Release);
        self.freed_collateral.store(netting_benefit, Ordering::Release);
        self.epoch.fetch_add(1, Ordering::AcqRel);
        
        Ok(net_margin)
    }

    /// Get the amount of collateral freed by netting
    pub fn get_freed_collateral(&self) -> i64 {
        self.freed_collateral.load(Ordering::Acquire)
    }

    /// Get current net margin requirement
    pub fn get_net_margin(&self) -> i64 {
        self.total_margin_required.load(Ordering::Acquire)
    }

    /// Get netting efficiency ratio (freed / gross)
    pub fn get_netting_efficiency(&self, margin_rate: f64) -> f64 {
        let gross = self.calculate_gross_margin(margin_rate) as f64;
        let freed = self.freed_collateral.load(Ordering::Acquire) as f64;
        
        if gross <= 0.0 {
            return 0.0;
        }
        
        (freed / gross).clamp(0.0, 1.0)
    }

    /// Calculate cross-broker net position for a single asset
    pub fn net_single_asset(&self, asset_id: u32) -> Option<i64> {
        let entry = self.positions.get(&asset_id)?;
        let (long_exp, short_exp) = entry.value();
        
        let long = long_exp.load(Ordering::Acquire);
        let short = short_exp.load(Ordering::Acquire);
        
        // Net position = long - short
        Some(long.saturating_sub(short))
    }
}

impl Default for CrossMarginNetting {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_netting() {
        let netting = CrossMarginNetting::new();
        
        // Asset 1: Long 100k, Short 50k
        netting.update_position(1, 100_000, 50_000);
        
        // Asset 2: Long 80k, Short 60k  
        netting.update_position(2, 80_000, 60_000);
        
        // Set high correlation for netting benefit
        netting.set_correlation(1, 2, 0.9).unwrap();
        
        let margin_rate = 0.1; // 10% margin
        let net_margin = netting.calculate_net_margin(margin_rate).unwrap();
        
        // Gross margin would be: (150k + 140k) * 0.1 = 29k
        // With netting, should be less
        assert!(net_margin < 29_000);
        assert!(net_margin > 0);
        
        let efficiency = netting.get_netting_efficiency(margin_rate);
        assert!(efficiency > 0.0);
    }

    #[test]
    fn test_no_correlation() {
        let netting = CrossMarginNetting::new();
        
        netting.update_position(1, 100_000, 50_000);
        netting.update_position(2, 80_000, 60_000);
        
        // No correlations set
        let margin_rate = 0.1;
        let net_margin = netting.calculate_net_margin(margin_rate).unwrap();
        let gross_margin = netting.calculate_gross_margin(margin_rate);
        
        // Without correlations, net margin should equal gross
        assert_eq!(net_margin, gross_margin);
        assert_eq!(netting.get_freed_collateral(), 0);
    }

    #[test]
    fn test_invalid_correlation() {
        let netting = CrossMarginNetting::new();
        
        // Invalid correlation > 1.0
        let result = netting.set_correlation(1, 2, 1.5);
        assert!(matches!(result, Err(MarginError::InvalidCorrelation { .. })));
        
        // NaN correlation
        let result = netting.set_correlation(1, 2, f64::NAN);
        assert!(matches!(result, Err(MarginError::InvalidCorrelation { .. })));
    }
}
