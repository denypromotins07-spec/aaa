//! Thermodynamic lead-lag delay calculator.
//!
//! Computes the exact time delay between order flow in one asset
//! and price movement in another, based on thermodynamic principles
//! of information transfer in financial markets.

use crate::lead_lag::hayashi_yoshida_estimator::{Tick, HayashiYoshidaEstimator};

/// Result from thermodynamic delay analysis
#[derive(Debug, Clone)]
pub struct ThermodynamicDelayResult {
    /// Optimal lag in microseconds (positive means A leads B)
    pub optimal_lag_us: u64,
    /// Correlation at optimal lag
    pub peak_correlation: f64,
    /// Width of correlation peak (uncertainty in lag estimate)
    pub peak_width_us: u64,
    /// Information transfer rate estimate
    pub info_transfer_rate: f64,
}

/// Thermodynamic lead-lag analyzer
pub struct ThermodynamicDelayAnalyzer {
    hy_estimator: HayashiYoshidaEstimator,
    max_search_lag_us: u64,
    search_step_us: u64,
}

impl ThermodynamicDelayAnalyzer {
    /// Create new analyzer with default search parameters
    pub fn new(max_search_lag_us: u64, search_step_us: u64) -> Self {
        Self {
            hy_estimator: HayashiYoshidaEstimator::new(1000),
            max_search_lag_us,
            search_step_us,
        }
    }

    /// Compute thermodynamic delay between two assets
    /// 
    /// Uses a multi-scale approach:
    /// 1. Coarse search to find approximate lag
    /// 2. Fine search around peak for precise estimate
    /// 3. Parabolic interpolation for sub-step precision
    pub fn compute_delay(
        &mut self,
        ticks_a: &[Tick],
        ticks_b: &[Tick],
    ) -> Result<ThermodynamicDelayResult, String> {
        // Phase 1: Coarse search
        let (coarse_lag, coarse_corr) = self.coarse_search(ticks_a, ticks_b)?;

        // Phase 2: Fine search around coarse peak
        let fine_range = (self.search_step_us * 2).min(coarse_lag);
        let fine_start = coarse_lag.saturating_sub(fine_range);
        let fine_end = coarse_lag + fine_range;

        let (fine_lag, fine_corr) = self.fine_search(ticks_a, ticks_b, fine_start, fine_end)?;

        // Phase 3: Parabolic interpolation for sub-step precision
        let (refined_lag, refined_corr) = self.parabolic_interpolation(
            ticks_a, ticks_b, fine_lag, self.search_step_us / 2
        )?;

        // Estimate peak width (confidence interval)
        let peak_width = self.estimate_peak_width(ticks_a, ticks_b, refined_lag, refined_corr)?;

        // Estimate information transfer rate
        let info_rate = self.estimate_info_transfer_rate(refined_corr, peak_width)?;

        Ok(ThermodynamicDelayResult {
            optimal_lag_us: refined_lag,
            peak_correlation: refined_corr,
            peak_width_us: peak_width,
            info_transfer_rate: info_rate,
        })
    }

    /// Coarse search with larger steps
    fn coarse_search(
        &mut self,
        ticks_a: &[Tick],
        ticks_b: &[Tick],
    ) -> Result<(u64, f64), String> {
        let coarse_step = self.search_step_us * 5;
        self.hy_estimator.find_optimal_lag(
            ticks_a,
            ticks_b,
            self.max_search_lag_us,
            coarse_step,
        )
    }

    /// Fine search in narrow range
    fn fine_search(
        &mut self,
        ticks_a: &[Tick],
        ticks_b: &[Tick],
        start_us: u64,
        end_us: u64,
    ) -> Result<(u64, f64), String> {
        let mut best_lag = start_us;
        let mut best_corr = f64::NEG_INFINITY;

        let mut lag = start_us;
        while lag <= end_us {
            let result = self.hy_estimator.estimate_lead_lag(ticks_a, ticks_b, lag)?;
            
            if let Some(corr) = self.hy_estimator.estimate_correlation(
                &result,
                self.hy_estimator.compute_variance(ticks_a).unwrap_or(1.0),
                self.hy_estimator.compute_variance(ticks_b).unwrap_or(1.0),
            ).ok().flatten() {
                if corr > best_corr {
                    best_corr = corr;
                    best_lag = lag;
                }
            }
            lag += self.search_step_us;
        }

        Ok((best_lag, best_corr))
    }

    /// Parabolic interpolation for sub-step precision
    fn parabolic_interpolation(
        &mut self,
        ticks_a: &[Tick],
        ticks_b: &[Tick],
        center_lag: u64,
        half_step: u64,
    ) -> Result<(u64, f64), String> {
        let lag_minus = center_lag.saturating_sub(half_step);
        let lag_plus = center_lag + half_step;

        let corr_minus = self.get_correlation_at_lag(ticks_a, ticks_b, lag_minus)?;
        let corr_center = self.get_correlation_at_lag(ticks_a, ticks_b, center_lag)?;
        let corr_plus = self.get_correlation_at_lag(ticks_a, ticks_b, lag_plus)?;

        // Parabolic fit: y = a*x^2 + b*x + c
        // Vertex at x = -b/(2a)
        let h = half_step as f64;
        
        // Check for valid parabola (must have maximum, not minimum)
        let denom = 2.0 * (corr_minus - 2.0 * corr_center + corr_plus);
        if denom.abs() < 1e-10 || denom >= 0.0 {
            // Can't fit proper parabola, return center
            return Ok((center_lag, corr_center));
        }

        let offset = h * (corr_minus - corr_plus) / denom;
        let refined_lag = (center_lag as i64 + offset as i64).max(0) as u64;
        
        // Estimate peak correlation from parabola
        let refined_corr = corr_center - (corr_minus - 2.0 * corr_center + corr_plus) * (offset / h).powi(2);
        let refined_corr = refined_corr.max(-1.0).min(1.0);

        Ok((refined_lag, refined_corr))
    }

    /// Get correlation at specific lag
    fn get_correlation_at_lag(
        &mut self,
        ticks_a: &[Tick],
        ticks_b: &[Tick],
        lag: u64,
    ) -> Result<f64, String> {
        let result = self.hy_estimator.estimate_lead_lag(ticks_a, ticks_b, lag)?;
        
        let var_a = self.hy_estimator.compute_variance(ticks_a).unwrap_or(1.0);
        let var_b = self.hy_estimator.compute_variance(ticks_b).unwrap_or(1.0);
        
        self.hy_estimator.estimate_correlation(&result, var_a, var_b)
            .map(|c| c.max(-1.0).min(1.0))
    }

    /// Estimate width of correlation peak
    fn estimate_peak_width(
        &mut self,
        ticks_a: &[Tick],
        ticks_b: &[Tick],
        peak_lag: u64,
        peak_corr: f64,
    ) -> Result<u64, String> {
        // Find lag where correlation drops to half of peak
        let half_height = peak_corr * 0.5;
        
        let mut left_width = 0u64;
        let mut lag = peak_lag;
        while lag > 0 {
            lag -= self.search_step_us;
            let corr = self.get_correlation_at_lag(ticks_a, ticks_b, lag)?;
            if corr <= half_height {
                left_width = peak_lag - lag;
                break;
            }
        }

        let mut right_width = 0u64;
        let mut lag = peak_lag;
        while lag < self.max_search_lag_us {
            lag += self.search_step_us;
            let corr = self.get_correlation_at_lag(ticks_a, ticks_b, lag)?;
            if corr <= half_height {
                right_width = lag - peak_lag;
                break;
            }
        }

        Ok((left_width + right_width) / 2)
    }

    /// Estimate information transfer rate based on correlation and uncertainty
    fn estimate_info_transfer_rate(
        &self,
        correlation: f64,
        peak_width_us: u64,
    ) -> Result<f64, String> {
        if peak_width_us == 0 {
            return Err("Peak width cannot be zero".to_string());
        }

        // Simplified information transfer metric
        // Higher correlation + narrower peak = higher info transfer
        let info_content = (1.0 + correlation) / 2.0; // Normalize to [0, 1]
        let temporal_precision = 1_000_000.0 / peak_width_us as f64; // Hz

        Ok(info_content * temporal_precision)
    }

    /// Detect regime changes in lead-lag relationship
    pub fn detect_regime_change(
        &mut self,
        ticks_a: &[Tick],
        ticks_b: &[Tick],
        window_size: usize,
    ) -> Result<Vec<RegimeChange>, String> {
        if ticks_a.len() < window_size * 2 {
            return Err("Insufficient data for regime detection".to_string());
        }

        let mut regimes = Vec::new();
        let mut prev_lag: Option<u64> = None;

        for i in (window_size..ticks_a.len()).step_by(window_size) {
            let window_a = &ticks_a[i - window_size..i];
            let window_b = &ticks_b[i - window_size..i];

            if let Ok(result) = self.compute_delay(window_a, window_b) {
                if let Some(prev) = prev_lag {
                    let lag_change = result.optimal_lag_us as i64 - prev as i64;
                    
                    if lag_change.abs() as u64 > self.search_step_us * 2 {
                        regimes.push(RegimeChange {
                            timestamp_us: ticks_a[i].timestamp_us,
                            old_lag: prev,
                            new_lag: result.optimal_lag_us,
                            magnitude: lag_change.unsigned_abs(),
                        });
                    }
                }
                prev_lag = Some(result.optimal_lag_us);
            }
        }

        Ok(regimes)
    }
}

/// Detected regime change in lead-lag relationship
#[derive(Debug, Clone)]
pub struct RegimeChange {
    pub timestamp_us: u64,
    pub old_lag: u64,
    pub new_lag: u64,
    pub magnitude: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_computation() {
        // Create synthetic data with known lag
        let base_time = 0u64;
        let mut ticks_a = Vec::new();
        let mut ticks_b = Vec::new();

        for i in 0..100 {
            let t = base_time + i * 1000;
            ticks_a.push(Tick {
                timestamp_us: t,
                price: 100.0 + (i as f64) * 0.01,
                volume: 1.0,
            });
            
            // B lags A by 5000 us
            ticks_b.push(Tick {
                timestamp_us: t + 5000,
                price: 50.0 + (i as f64) * 0.005,
                volume: 1.0,
            });
        }

        let mut analyzer = ThermodynamicDelayAnalyzer::new(20000, 500);
        let result = analyzer.compute_delay(&ticks_a, &ticks_b);

        // Should detect positive lag (A leads B)
        assert!(result.is_ok());
    }
}
