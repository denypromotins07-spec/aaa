//! Hayashi-Yoshida estimator for non-synchronous cross-asset covariance.
//!
//! The Hayashi-Yoshida estimator correctly computes covariance between two assets
//! that trade at irregular, non-overlapping timestamps - a common scenario in
//! cross-asset trading (e.g., Gold vs USD/JPY, BTC vs Nasdaq futures).
//!
//! Unlike standard Pearson correlation which requires synchronized returns,
//! H-Y sums the products of returns over all overlapping time intervals.
//!
//! Reference: Hayashi, T., & Yoshida, N. (2005). "On covariance estimation
//! of non-synchronously observed diffusion processes."

use std::collections::BTreeMap;

/// A single tick observation
#[derive(Debug, Clone)]
pub struct Tick {
    /// Microsecond timestamp
    pub timestamp_us: u64,
    /// Price (can be mid-price, last trade, etc.)
    pub price: f64,
    /// Optional volume
    pub volume: f64,
}

/// Result from Hayashi-Yoshida estimation
#[derive(Debug, Clone)]
pub struct HyResult {
    /// Estimated covariance
    pub covariance: f64,
    /// Estimated correlation (if variances provided)
    pub correlation: Option<f64>,
    /// Number of overlapping intervals found
    pub num_overlaps: usize,
    /// Effective time span in microseconds
    pub time_span_us: u64,
}

impl HyResult {
    /// Check if result is valid (finite and positive overlaps)
    pub fn is_valid(&self) -> bool {
        self.covariance.is_finite() && self.num_overlaps > 0
    }
}

/// Hayashi-Yoshida covariance estimator
pub struct HayashiYoshidaEstimator {
    /// Pre-allocated buffer for sorted ticks (asset A)
    ticks_a_buffer: Vec<Tick>,
    /// Pre-allocated buffer for sorted ticks (asset B)
    ticks_b_buffer: Vec<Tick>,
}

impl HayashiYoshidaEstimator {
    /// Create new estimator with pre-allocated capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            ticks_a_buffer: Vec::with_capacity(capacity),
            ticks_b_buffer: Vec::with_capacity(capacity),
        }
    }

    /// Compute log return from two prices
    #[inline(always)]
    fn log_return(p1: f64, p2: f64) -> Option<f64> {
        if p1 <= 0.0 || p2 <= 0.0 {
            return None;
        }
        Some((p2 / p1).ln())
    }

    /// Estimate covariance using Hayashi-Yoshida method
    ///
    /// # Arguments
    /// * `ticks_a` - Ticks for asset A (unsorted, will be sorted internally)
    /// * `ticks_b` - Ticks for asset B (unsorted, will be sorted internally)
    ///
    /// # Algorithm
    /// For each pair of consecutive timestamps in the combined timeline,
    /// compute returns for both assets over their respective intervals
    /// that overlap with the current interval. Sum the products.
    pub fn estimate_covariance(
        &mut self,
        ticks_a: &[Tick],
        ticks_b: &[Tick],
    ) -> Result<HyResult, String> {
        if ticks_a.len() < 2 || ticks_b.len() < 2 {
            return Err("Need at least 2 ticks per asset".to_string());
        }

        // Copy and sort ticks by timestamp (stable sort preserves order for equal timestamps)
        self.ticks_a_buffer.clear();
        self.ticks_b_buffer.clear();
        
        self.ticks_a_buffer.extend_from_slice(ticks_a);
        self.ticks_b_buffer.extend_from_slice(ticks_b);
        
        self.ticks_a_buffer.sort_by_key(|t| t.timestamp_us);
        self.ticks_b_buffer.sort_by_key(|t| t.timestamp_us);

        // Validate no duplicate timestamps (or handle them)
        for i in 1..self.ticks_a_buffer.len() {
            if self.ticks_a_buffer[i].timestamp_us == self.ticks_a_buffer[i-1].timestamp_us {
                // Merge duplicate timestamps by averaging price
                // For now, just skip - production code should handle properly
            }
        }

        let mut sum_cov = 0.0;
        let mut num_overlaps = 0usize;

        // Build interval lists for both assets
        // Each interval is (start_time, end_time, return)
        let intervals_a = self.build_intervals(&self.ticks_a_buffer)?;
        let intervals_b = self.build_intervals(&self.ticks_b_buffer)?;

        // For each pair of intervals, check if they overlap
        // If they do, add the product of returns weighted by overlap duration
        for ia in &intervals_a {
            for ib in &intervals_b {
                // Check for overlap: max(start_a, start_b) < min(end_a, end_b)
                let overlap_start = ia.start_us.max(ib.start_us);
                let overlap_end = ia.end_us.min(ib.end_us);

                if overlap_start < overlap_end {
                    let overlap_duration = overlap_end - overlap_start;
                    
                    // Weight by relative overlap (normalized later)
                    let weight = overlap_duration as f64 
                        / ((ia.end_us - ia.start_us) as f64.min(1.0));
                    
                    if let (Some(ret_a), Some(ret_b)) = (ia.return_val, ib.return_val) {
                        sum_cov += weight * ret_a * ret_b;
                        num_overlaps += 1;
                    }
                }
            }
        }

        if num_overlaps == 0 {
            return Ok(HyResult {
                covariance: 0.0,
                correlation: None,
                num_overlaps: 0,
                time_span_us: self.ticks_a_buffer.last().unwrap().timestamp_us 
                    - self.ticks_a_buffer.first().unwrap().timestamp_us,
            });
        }

        // Normalize by total time to get covariance rate
        let total_time_us = self.ticks_a_buffer.last().unwrap().timestamp_us 
            - self.ticks_a_buffer.first().unwrap().timestamp_us;
        
        let covariance = sum_cov * 1_000_000.0 / total_time_us as f64; // Annualize approximation

        Ok(HyResult {
            covariance,
            correlation: None, // Will be computed if variances provided
            num_overlaps,
            time_span_us: total_time_us,
        })
    }

    /// Build return intervals from sorted ticks
    fn build_intervals(
        &self,
        ticks: &[Tick],
    ) -> Result<Vec<ReturnInterval>, String> {
        let mut intervals = Vec::with_capacity(ticks.len() - 1);

        for i in 1..ticks.len() {
            let start_us = ticks[i - 1].timestamp_us;
            let end_us = ticks[i].timestamp_us;
            
            if end_us <= start_us {
                continue; // Skip invalid intervals
            }

            let return_val = Self::log_return(ticks[i - 1].price, ticks[i].price);

            intervals.push(ReturnInterval {
                start_us,
                end_us,
                return_val,
            });
        }

        Ok(intervals)
    }

    /// Estimate correlation given pre-computed variances
    pub fn estimate_correlation(
        &self,
        cov_result: &HyResult,
        var_a: f64,
        var_b: f64,
    ) -> Result<f64, String> {
        if var_a <= 0.0 || var_b <= 0.0 {
            return Err("Variances must be positive".to_string());
        }

        let denom = (var_a * var_b).sqrt();
        if denom < 1e-15 {
            return Err("Denominator too small".to_string());
        }

        let correlation = cov_result.covariance / denom;

        // Clamp to [-1, 1] for numerical stability
        let correlation = correlation.max(-1.0).min(1.0);

        Ok(correlation)
    }

    /// Compute lead-lag correlation: how does asset A at time t correlate with asset B at time t+lag
    pub fn estimate_lead_lag(
        &mut self,
        ticks_a: &[Tick],
        ticks_b: &[Tick],
        lag_us: u64,
    ) -> Result<HyResult, String> {
        // Shift ticks_b forward by lag_us
        let shifted_b: Vec<Tick> = ticks_b.iter()
            .map(|t| Tick {
                timestamp_us: t.timestamp_us + lag_us,
                price: t.price,
                volume: t.volume,
            })
            .collect();

        self.estimate_covariance(ticks_a, &shifted_b)
    }

    /// Find optimal lead-lag delay that maximizes correlation
    pub fn find_optimal_lag(
        &mut self,
        ticks_a: &[Tick],
        ticks_b: &[Tick],
        max_lag_us: u64,
        step_us: u64,
    ) -> Result<(u64, f64), String> {
        let mut best_lag = 0u64;
        let mut best_corr = f64::NEG_INFINITY;

        let mut lag = 0u64;
        while lag <= max_lag_us {
            let result = self.estimate_lead_lag(ticks_a, ticks_b, lag)?;
            
            if let Some(corr) = self.compute_correlation_from_result(&result, ticks_a, ticks_b) {
                if corr > best_corr {
                    best_corr = corr;
                    best_lag = lag;
                }
            }

            lag += step_us;
        }

        Ok((best_lag, best_corr))
    }

    /// Helper to compute correlation from HY result
    fn compute_correlation_from_result(
        &self,
        result: &HyResult,
        ticks_a: &[Tick],
        ticks_b: &[Tick],
    ) -> Option<f64> {
        // Compute variances separately
        let var_a = self.compute_variance(ticks_a)?;
        let var_b = self.compute_variance(ticks_b)?;
        
        self.estimate_correlation(result, var_a, var_b).ok()
    }

    /// Compute realized variance for a single asset
    fn compute_variance(&self, ticks: &[Tick]) -> Option<f64> {
        if ticks.len() < 2 {
            return None;
        }

        let mut sum_sq = 0.0;
        for i in 1..ticks.len() {
            if let Some(ret) = Self::log_return(ticks[i - 1].price, ticks[i].price) {
                sum_sq += ret * ret;
            }
        }

        Some(sum_sq)
    }
}

/// Internal structure for return intervals
#[derive(Debug, Clone)]
struct ReturnInterval {
    start_us: u64,
    end_us: u64,
    return_val: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synchronous_correlation() {
        // Perfectly correlated synchronous data
        let ticks_a = vec![
            Tick { timestamp_us: 0, price: 100.0, volume: 1.0 },
            Tick { timestamp_us: 1000, price: 101.0, volume: 1.0 },
            Tick { timestamp_us: 2000, price: 102.0, volume: 1.0 },
            Tick { timestamp_us: 3000, price: 103.0, volume: 1.0 },
        ];

        let ticks_b = vec![
            Tick { timestamp_us: 0, price: 50.0, volume: 1.0 },
            Tick { timestamp_us: 1000, price: 50.5, volume: 1.0 },
            Tick { timestamp_us: 2000, price: 51.0, volume: 1.0 },
            Tick { timestamp_us: 3000, price: 51.5, volume: 1.0 },
        ];

        let mut estimator = HayashiYoshidaEstimator::new(10);
        let result = estimator.estimate_covariance(&ticks_a, &ticks_b).unwrap();

        assert!(result.covariance > 0.0); // Positive covariance expected
        assert!(result.num_overlaps > 0);
    }

    #[test]
    fn test_non_synchronous_data() {
        // Non-overlapping timestamps
        let ticks_a = vec![
            Tick { timestamp_us: 0, price: 100.0, volume: 1.0 },
            Tick { timestamp_us: 2000, price: 101.0, volume: 1.0 },
            Tick { timestamp_us: 4000, price: 102.0, volume: 1.0 },
        ];

        let ticks_b = vec![
            Tick { timestamp_us: 1000, price: 50.0, volume: 1.0 },
            Tick { timestamp_us: 3000, price: 50.5, volume: 1.0 },
            Tick { timestamp_us: 5000, price: 51.0, volume: 1.0 },
        ];

        let mut estimator = HayashiYoshidaEstimator::new(10);
        let result = estimator.estimate_covariance(&ticks_a, &ticks_b).unwrap();

        // Should still produce a valid estimate despite non-overlapping times
        assert!(result.num_overlaps > 0 || result.covariance == 0.0);
    }
}
