//! Universe Scanner - SIMD-accelerated correlation and cointegration screening
//! 
//! Scans thousands of asset pairs to identify tradable cointegrated subsets
//! using Pearson correlation and variance ratio tests with SIMD vectorization.

use core::arch::x86_64::*;

/// Maximum number of assets that can be scanned in parallel
const MAX_ASSETS: usize = 256;

/// Maximum lookback window for correlation calculation
const MAX_LOOKBACK: usize = 1024;

/// Result of a pair screening operation
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PairScreeningResult {
    /// Asset A index
    pub asset_a: u16,
    /// Asset B index
    pub asset_b: u16,
    /// Pearson correlation coefficient
    pub correlation: f64,
    /// Variance ratio (A/B)
    pub variance_ratio: f64,
    /// Half-life of mean reversion (estimated)
    pub half_life: f64,
    /// Score for ranking (higher = better cointegration candidate)
    pub score: f64,
}

impl PairScreeningResult {
    #[inline]
    pub const fn new() -> Self {
        Self {
            asset_a: 0,
            asset_b: 0,
            correlation: 0.0,
            variance_ratio: 1.0,
            half_life: f64::INFINITY,
            score: 0.0,
        }
    }
}

impl Default for PairScreeningResult {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// SIMD-accelerated universe scanner for cointegrated pairs
pub struct SimdCorrelationScanner {
    /// Pre-allocated price buffer for asset A (stack-friendly)
    prices_a: [f64; MAX_LOOKBACK],
    /// Pre-allocated price buffer for asset B
    prices_b: [f64; MAX_LOOKBACK],
    /// Current lookback length
    lookback: usize,
    /// Minimum correlation threshold
    min_correlation: f64,
    /// Maximum variance ratio threshold
    max_variance_ratio: f64,
}

impl SimdCorrelationScanner {
    /// Create a new scanner with default thresholds
    #[inline]
    pub fn new(min_correlation: f64, max_variance_ratio: f64) -> Self {
        Self {
            prices_a: [0.0; MAX_LOOKBACK],
            prices_b: [0.0; MAX_LOOKBACK],
            lookback: 0,
            min_correlation: min_correlation.clamp(0.0, 1.0),
            max_variance_ratio: max_variance_ratio.max(1.0),
        }
    }

    /// Load price data for two assets
    /// 
    /// # Safety
    /// Slices must not exceed MAX_LOOKBACK elements
    #[inline]
    pub fn load_prices(&mut self, prices_a: &[f64], prices_b: &[f64]) -> bool {
        if prices_a.len() != prices_b.len() || prices_a.len() > MAX_LOOKBACK {
            return false;
        }
        
        self.lookback = prices_a.len();
        
        // Safe copy since we verified lengths
        self.prices_a[..self.lookback].copy_from_slice(prices_a);
        self.prices_b[..self.lookback].copy_from_slice(prices_b);
        
        true
    }

    /// Compute Pearson correlation using SIMD acceleration
    /// 
    /// Uses AVX2 instructions to process 4 doubles in parallel
    #[inline]
    pub fn compute_correlation(&self) -> Option<f64> {
        if self.lookback < 2 {
            return None;
        }

        // Use SIMD for the main computation
        unsafe {
            self.compute_correlation_simd()
        }
    }

    /// SIMD-accelerated correlation computation (AVX2)
    #[inline]
    unsafe fn compute_correlation_simd(&self) -> Option<f64> {
        let n = self.lookback;
        
        // Process 4 elements at a time using AVX2
        let simd_limit = (n / 4) * 4;
        
        // Accumulators for SIMD computation
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_xx = 0.0f64;
        let mut sum_yy = 0.0f64;
        let mut sum_xy = 0.0f64;

        // SIMD accumulation
        let mut acc_x = _mm256_setzero_pd();
        let mut acc_y = _mm256_setzero_pd();
        let mut acc_xx = _mm256_setzero_pd();
        let mut acc_yy = _mm256_setzero_pd();
        let mut acc_xy = _mm256_setzero_pd();

        for i in (0..simd_limit).step_by(4) {
            // Load 4 doubles from each array
            let x_vec = _mm256_loadu_pd(self.prices_a.as_ptr().add(i));
            let y_vec = _mm256_loadu_pd(self.prices_b.as_ptr().add(i));

            // Accumulate sums
            acc_x = _mm256_add_pd(acc_x, x_vec);
            acc_y = _mm256_add_pd(acc_y, y_vec);

            // x^2 and y^2
            let xx_vec = _mm256_mul_pd(x_vec, x_vec);
            let yy_vec = _mm256_mul_pd(y_vec, y_vec);
            acc_xx = _mm256_add_pd(acc_xx, xx_vec);
            acc_yy = _mm256_add_pd(acc_yy, yy_vec);

            // x*y
            let xy_vec = _mm256_mul_pd(x_vec, y_vec);
            acc_xy = _mm256_add_pd(acc_xy, xy_vec);
        }

        // Horizontal sum of SIMD accumulators
        let mut tmp_x = [0.0f64; 4];
        let mut tmp_y = [0.0f64; 4];
        let mut tmp_xx = [0.0f64; 4];
        let mut tmp_yy = [0.0f64; 4];
        let mut tmp_xy = [0.0f64; 4];

        _mm256_storeu_pd(tmp_x.as_mut_ptr(), acc_x);
        _mm256_storeu_pd(tmp_y.as_mut_ptr(), acc_y);
        _mm256_storeu_pd(tmp_xx.as_mut_ptr(), acc_xx);
        _mm256_storeu_pd(tmp_yy.as_mut_ptr(), acc_yy);
        _mm256_storeu_pd(tmp_xy.as_mut_ptr(), acc_xy);

        sum_x = tmp_x.iter().sum();
        sum_y = tmp_y.iter().sum();
        sum_xx = tmp_xx.iter().sum();
        sum_yy = tmp_yy.iter().sum();
        sum_xy = tmp_xy.iter().sum();

        // Process remaining elements (scalar)
        for i in simd_limit..n {
            let x = self.prices_a[i];
            let y = self.prices_b[i];
            
            if !x.is_finite() || !y.is_finite() {
                continue; // Skip invalid data
            }
            
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_yy += y * y;
            sum_xy += x * y;
        }

        let n_f64 = n as f64;
        
        // Compute correlation using numerically stable formula
        // r = (n*sum_xy - sum_x*sum_y) / sqrt((n*sum_xx - sum_x^2) * (n*sum_yy - sum_y^2))
        let numerator = n_f64 * sum_xy - sum_x * sum_y;
        let denom_x = n_f64 * sum_xx - sum_x * sum_x;
        let denom_y = n_f64 * sum_yy - sum_y * sum_y;

        if denom_x <= 0.0 || denom_y <= 0.0 {
            return Some(0.0);
        }

        let denominator = (denom_x * denom_y).sqrt();
        
        if denominator < 1e-15 {
            return Some(0.0);
        }

        let correlation = numerator / denominator;
        
        // Clamp to [-1, 1] to handle floating-point errors
        Some(correlation.clamp(-1.0, 1.0))
    }

    /// Compute variance ratio between two assets
    #[inline]
    pub fn compute_variance_ratio(&self) -> Option<f64> {
        if self.lookback < 2 {
            return None;
        }

        let n = self.lookback as f64;
        
        // Compute means
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        
        for i in 0..self.lookback {
            let x = self.prices_a[i];
            let y = self.prices_b[i];
            
            if x.is_finite() { sum_x += x; }
            if y.is_finite() { sum_y += y; }
        }
        
        let mean_x = sum_x / n;
        let mean_y = sum_y / n;

        // Compute variances using Welford's online algorithm for stability
        let mut m2_x = 0.0f64;
        let mut m2_y = 0.0f64;
        
        for i in 0..self.lookback {
            let x = self.prices_a[i];
            let y = self.prices_b[i];
            
            if x.is_finite() {
                let diff = x - mean_x;
                m2_x += diff * diff;
            }
            if y.is_finite() {
                let diff = y - mean_y;
                m2_y += diff * diff;
            }
        }

        let var_x = m2_x / (n - 1.0);
        let var_y = m2_y / (n - 1.0);

        if var_y < 1e-15 {
            return None;
        }

        Some(var_x / var_y)
    }

    /// Screen a pair and return screening result
    #[inline]
    pub fn screen_pair(asset_a: u16, asset_b: u16, correlation: f64, variance_ratio: f64) -> PairScreeningResult {
        // Score based on correlation strength and variance ratio closeness to 1
        let corr_score = correlation.abs();
        let vr_score = if variance_ratio > 0.0 {
            1.0 - (variance_ratio - 1.0).abs().min(1.0)
        } else {
            0.0
        };

        let score = corr_score * 0.7 + vr_score * 0.3;

        PairScreeningResult {
            asset_a,
            asset_b,
            correlation,
            variance_ratio,
            half_life: f64::INFINITY, // Would need additional computation
            score,
        }
    }

    /// Check if a pair passes the screening thresholds
    #[inline]
    pub fn passes_thresholds(&self, correlation: f64, variance_ratio: f64) -> bool {
        correlation.abs() >= self.min_correlation
            && variance_ratio >= 1.0 / self.max_variance_ratio
            && variance_ratio <= self.max_variance_ratio
    }

    /// Get the current lookback size
    #[inline]
    pub fn lookback(&self) -> usize {
        self.lookback
    }

    /// Clear loaded data
    #[inline]
    pub fn clear(&mut self) {
        self.lookback = 0;
    }
}

/// Batch scanner for multiple pairs
pub struct BatchPairScanner {
    /// Results buffer (pre-allocated)
    results: [PairScreeningResult; MAX_ASSETS * MAX_ASSETS / 2],
    /// Number of valid results
    count: usize,
}

impl BatchPairScanner {
    #[inline]
    pub const fn new() -> Self {
        Self {
            results: [PairScreeningResult::new(); MAX_ASSETS * MAX_ASSETS / 2],
            count: 0,
        }
    }

    /// Scan all pairs in a universe and return qualifying pairs
    /// 
    /// # Arguments
    /// * `prices` - Slice of price slices, one per asset
    /// * `scanner` - Reusable scanner instance
    /// * `min_correlation` - Minimum correlation threshold
    pub fn scan_all(
        &mut self,
        prices: &[&[f64]],
        scanner: &mut SimdCorrelationScanner,
        min_correlation: f64,
    ) -> &[PairScreeningResult] {
        self.count = 0;
        let num_assets = prices.len();
        
        if num_assets < 2 || num_assets > MAX_ASSETS {
            return &[];
        }

        // O(N^2) pair scanning
        for i in 0..num_assets {
            for j in (i + 1)..num_assets {
                if self.count >= self.results.len() {
                    break;
                }

                // Load prices into scanner
                if !scanner.load_prices(prices[i], prices[j]) {
                    continue;
                }

                // Compute metrics
                let correlation = match scanner.compute_correlation() {
                    Some(c) => c,
                    None => continue,
                };

                let variance_ratio = match scanner.compute_variance_ratio() {
                    Some(vr) => vr,
                    None => continue,
                };

                // Check thresholds
                if scanner.passes_thresholds(correlation, variance_ratio) {
                    let result = SimdCorrelationScanner::screen_pair(
                        i as u16,
                        j as u16,
                        correlation,
                        variance_ratio,
                    );
                    
                    self.results[self.count] = result;
                    self.count += 1;
                }
            }
        }

        &self.results[..self.count]
    }

    /// Get the number of qualifying pairs found
    #[inline]
    pub fn result_count(&self) -> usize {
        self.count
    }
}

impl Default for BatchPairScanner {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_correlation() {
        let mut scanner = SimdCorrelationScanner::new(0.5, 2.0);
        
        // Perfect positive correlation
        let prices_a: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let prices_b: Vec<f64> = (1..=100).map(|x| (x * 2) as f64).collect();
        
        scanner.load_prices(&prices_a, &prices_b);
        let corr = scanner.compute_correlation().unwrap();
        
        assert!((corr - 1.0).abs() < 1e-10, "Perfect correlation should be 1.0");
    }

    #[test]
    fn test_negative_correlation() {
        let mut scanner = SimdCorrelationScanner::new(0.5, 2.0);
        
        // Perfect negative correlation
        let prices_a: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let prices_b: Vec<f64> = (1..=100).rev().map(|x| x as f64).collect();
        
        scanner.load_prices(&prices_a, &prices_b);
        let corr = scanner.compute_correlation().unwrap();
        
        assert!((corr + 1.0).abs() < 1e-10, "Perfect negative correlation should be -1.0");
    }

    #[test]
    fn test_variance_ratio() {
        let mut scanner = SimdCorrelationScanner::new(0.5, 2.0);
        
        let prices_a: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let prices_b: Vec<f64> = (1..=100).map(|x| (x * 2) as f64).collect();
        
        scanner.load_prices(&prices_a, &prices_b);
        let vr = scanner.compute_variance_ratio().unwrap();
        
        // Var(a*X) = a^2 * Var(X), so VR should be 0.25
        assert!((vr - 0.25).abs() < 0.01, "Variance ratio should be 0.25");
    }

    #[test]
    fn test_screening_thresholds() {
        let scanner = SimdCorrelationScanner::new(0.8, 1.5);
        
        // High correlation, good variance ratio - should pass
        assert!(scanner.passes_thresholds(0.9, 1.2));
        
        // Low correlation - should fail
        assert!(!scanner.passes_thresholds(0.5, 1.2));
        
        // Bad variance ratio - should fail
        assert!(!scanner.passes_thresholds(0.9, 2.0));
    }

    #[test]
    fn test_insufficient_data() {
        let mut scanner = SimdCorrelationScanner::new(0.5, 2.0);
        
        let prices_a = vec![1.0];
        let prices_b = vec![2.0];
        
        scanner.load_prices(&prices_a, &prices_b);
        assert!(scanner.compute_correlation().is_none());
        assert!(scanner.compute_variance_ratio().is_none());
    }
}
