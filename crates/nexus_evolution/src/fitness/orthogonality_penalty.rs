//! Orthogonality Penalty Calculator
//! 
//! Calculates the correlation between candidate strategies and existing
//! portfolio alphas to enforce diversification in the genetic algorithm.

use std::collections::HashMap;

/// Cached alpha signal for orthogonality calculation
#[derive(Debug, Clone)]
pub struct AlphaSignal {
    pub id: u64,
    pub name: String,
    /// Signal values over the evaluation period
    pub values: Vec<f64>,
    /// Asset class this alpha targets
    pub asset_class: AssetClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetClass {
    Equities,
    FixedIncome,
    Commodities,
    FX,
    Derivatives,
    Crypto,
}

/// Result of orthogonality analysis
#[derive(Debug, Clone)]
pub struct OrthogonalityResult {
    /// Maximum absolute correlation with any existing alpha
    pub max_correlation: f64,
    /// Mean absolute correlation across all alphas
    pub mean_correlation: f64,
    /// Number of alphas with correlation > threshold
    pub high_corr_count: usize,
    /// Orthogonality score (1.0 = perfectly orthogonal, 0.0 = fully correlated)
    pub orthogonality_score: f64,
    /// Details of correlations per alpha
    pub correlations: Vec<(u64, f64)>,
}

impl OrthogonalityResult {
    pub fn perfect() -> Self {
        Self {
            max_correlation: 0.0,
            mean_correlation: 0.0,
            high_corr_count: 0,
            orthogonality_score: 1.0,
            correlations: Vec::new(),
        }
    }
}

/// Calculator for strategy orthogonality against portfolio alphas
pub struct OrthogonalityCalculator {
    /// Existing portfolio alphas for comparison
    portfolio_alphas: Vec<AlphaSignal>,
    /// Correlation threshold for "high correlation" flag
    high_corr_threshold: f64,
    /// Minimum samples required for valid correlation
    min_samples: usize,
}

impl OrthogonalityCalculator {
    pub fn new(portfolio_alphas: Vec<AlphaSignal>) -> Self {
        Self {
            portfolio_alphas,
            high_corr_threshold: 0.7,
            min_samples: 30,
        }
    }

    /// Set the high correlation threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.high_corr_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set minimum samples for valid calculation
    pub fn with_min_samples(mut self, min: usize) -> Self {
        self.min_samples = min;
        self
    }

    /// Add an alpha to the portfolio
    pub fn add_alpha(&mut self, alpha: AlphaSignal) {
        self.portfolio_alphas.push(alpha);
    }

    /// Remove an alpha from the portfolio by ID
    pub fn remove_alpha(&mut self, id: u64) {
        self.portfolio_alphas.retain(|a| a.id != id);
    }

    /// Calculate orthogonality of a candidate signal against portfolio
    pub fn calculate_orthogonality(&self, candidate: &[f64]) -> OrthogonalityResult {
        if self.portfolio_alphas.is_empty() {
            return OrthogonalityResult::perfect();
        }

        if candidate.len() < self.min_samples {
            // Not enough data for valid correlation
            return OrthogonalityResult {
                max_correlation: 1.0,
                mean_correlation: 1.0,
                high_corr_count: self.portfolio_alphas.len(),
                orthogonality_score: 0.0,
                correlations: Vec::new(),
            };
        }

        let mut correlations: Vec<(u64, f64)> = Vec::with_capacity(self.portfolio_alphas.len());
        let mut max_corr = 0.0f64;
        let mut sum_abs_corr = 0.0f64;
        let mut high_corr_count = 0usize;

        for alpha in &self.portfolio_alphas {
            // Align lengths
            let len = candidate.len().min(alpha.values.len());
            if len < self.min_samples {
                continue;
            }

            let corr = self.pearson_correlation(
                &candidate[..len],
                &alpha.values[..len],
            );

            let abs_corr = corr.abs();
            correlations.push((alpha.id, corr));

            if abs_corr > max_corr {
                max_corr = abs_corr;
            }

            sum_abs_corr += abs_corr;

            if abs_corr > self.high_corr_threshold {
                high_corr_count += 1;
            }
        }

        let mean_corr = if correlations.is_empty() {
            0.0
        } else {
            sum_abs_corr / correlations.len() as f64
        };

        // Orthogonality score: 1.0 - weighted combination of max and mean correlation
        let orthogonality_score = (1.0 - max_corr).max(0.0) * 0.6 
            + (1.0 - mean_corr).max(0.0) * 0.4;

        OrthogonalityResult {
            max_correlation: max_corr,
            mean_correlation: mean_corr,
            high_corr_count,
            orthogonality_score,
            correlations,
        }
    }

    /// Calculate orthogonality specifically for a given asset class
    pub fn calculate_orthogonality_by_class(
        &self,
        candidate: &[f64],
        asset_class: AssetClass,
    ) -> OrthogonalityResult {
        let filtered_alphas: Vec<AlphaSignal> = self.portfolio_alphas
            .iter()
            .filter(|a| a.asset_class == asset_class)
            .cloned()
            .collect();

        if filtered_alphas.is_empty() {
            return OrthogonalityResult::perfect();
        }

        let mut calc = OrthogonalityCalculator::new(filtered_alphas);
        calc.high_corr_threshold = self.high_corr_threshold;
        calc.min_samples = self.min_samples;
        calc.calculate_orthogonality(candidate)
    }

    /// Fast Pearson correlation coefficient
    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len());
        if n < self.min_samples {
            return 0.0;
        }

        // Calculate means
        let mean_x = x[..n].iter().sum::<f64>() / n as f64;
        let mean_y = y[..n].iter().sum::<f64>() / n as f64;

        // Calculate covariance and standard deviations
        let mut cov = 0.0f64;
        let mut var_x = 0.0f64;
        let mut var_y = 0.0f64;

        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let std_x = var_x.sqrt();
        let std_y = var_y.sqrt();

        if std_x < 1e-10 || std_y < 1e-10 {
            return 0.0;
        }

        cov / (std_x * std_y)
    }

    /// Get all alphas targeting a specific asset class
    pub fn get_alphas_by_class(&self, asset_class: AssetClass) -> Vec<&AlphaSignal> {
        self.portfolio_alphas
            .iter()
            .filter(|a| a.asset_class == asset_class)
            .collect()
    }

    /// Get total number of portfolio alphas
    pub fn alpha_count(&self) -> usize {
        self.portfolio_alphas.len()
    }

    /// Clear all portfolio alphas
    pub fn clear(&mut self) {
        self.portfolio_alphas.clear();
    }
}

/// Builder for creating OrthogonalityCalculator with fluent API
pub struct OrthogonalityBuilder {
    alphas: Vec<AlphaSignal>,
    threshold: f64,
    min_samples: usize,
}

impl OrthogonalityBuilder {
    pub fn new() -> Self {
        Self {
            alphas: Vec::new(),
            threshold: 0.7,
            min_samples: 30,
        }
    }

    pub fn add_alpha(mut self, alpha: AlphaSignal) -> Self {
        self.alphas.push(alpha);
        self
    }

    pub fn add_alphas<I>(mut self, alphas: I) -> Self
    where
        I: IntoIterator<Item = AlphaSignal>,
    {
        self.alphas.extend(alphas);
        self
    }

    pub fn threshold(mut self, t: f64) -> Self {
        self.threshold = t.clamp(0.0, 1.0);
        self
    }

    pub fn min_samples(mut self, m: usize) -> Self {
        self.min_samples = m.max(2);
        self
    }

    pub fn build(self) -> OrthogonalityCalculator {
        let mut calc = OrthogonalityCalculator::new(self.alphas);
        calc.high_corr_threshold = self.threshold;
        calc.min_samples = self.min_samples;
        calc
    }
}

impl Default for OrthogonalityBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_orthogonality() {
        let calc = OrthogonalityCalculator::new(Vec::new());
        let result = calc.calculate_orthogonality(&[1.0, 2.0, 3.0]);
        
        assert!((result.orthogonality_score - 1.0).abs() < 1e-10);
        assert_eq!(result.max_correlation, 0.0);
    }

    #[test]
    fn test_perfect_correlation() {
        let alpha = AlphaSignal {
            id: 1,
            name: "test".to_string(),
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            asset_class: AssetClass::Equities,
        };

        let calc = OrthogonalityCalculator::new(vec![alpha]);
        
        // Same signal = perfect correlation
        let result = calc.calculate_orthogonality(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        
        assert!((result.max_correlation - 1.0).abs() < 1e-5);
        assert!((result.orthogonality_score - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_negative_correlation() {
        let alpha = AlphaSignal {
            id: 1,
            name: "test".to_string(),
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            asset_class: AssetClass::Equities,
        };

        let calc = OrthogonalityCalculator::new(vec![alpha]);
        
        // Inverse signal = negative correlation
        let result = calc.calculate_orthogonality(&[5.0, 4.0, 3.0, 2.0, 1.0]);
        
        assert!((result.max_correlation - 1.0).abs() < 1e-5); // Abs value is 1
        assert!(result.correlations.first().map_or(false, |(_, c)| *c < -0.9));
    }

    #[test]
    fn test_builder_pattern() {
        let alpha = AlphaSignal {
            id: 1,
            name: "test".to_string(),
            values: vec![1.0, 2.0, 3.0],
            asset_class: AssetClass::FX,
        };

        let calc = OrthogonalityBuilder::new()
            .add_alpha(alpha)
            .threshold(0.5)
            .min_samples(10)
            .build();

        assert!((calc.high_corr_threshold - 0.5).abs() < 1e-10);
        assert_eq!(calc.min_samples, 10);
    }
}
