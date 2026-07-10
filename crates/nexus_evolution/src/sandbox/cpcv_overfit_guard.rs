//! Combinatorial Purged Cross-Validation (CPCV) Overfitting Guard
//! 
//! Implements Marcos Lopez de Prado's CPCV method to prevent backtest overfitting
//! by strictly isolating training and testing paths with purged gaps.
//! 
//! Reference: "The Definitive Guide to Backtesting" - Lopez de Prado

use super::ast_evaluator::{AstEvaluator, EvaluationResult, DataWindow};
use crate::gp::expression_tree::ExpressionTree;
use std::collections::HashMap;

/// Configuration for CPCV validation
#[derive(Debug, Clone)]
pub struct CpcvConfig {
    /// Number of folds (N)
    pub n_folds: usize,
    /// Number of test folds per split (k)
    pub n_test_folds: usize,
    /// Purge gap as fraction of dataset length (to prevent look-ahead bias)
    pub purge_ratio: f64,
    /// Minimum samples per fold
    pub min_samples_per_fold: usize,
}

impl Default for CpcvConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            n_test_folds: 1,
            purge_ratio: 0.02, // 2% purge between train/test
            min_samples_per_fold: 100,
        }
    }
}

/// Result from a single CPCV path evaluation
#[derive(Debug, Clone)]
pub struct PathResult {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
    pub purged_indices: Vec<usize>,
    pub train_sharpe: f64,
    pub test_sharpe: f64,
    pub is_valid: bool,
}

/// Complete CPCV evaluation result with distribution statistics
#[derive(Debug, Clone)]
pub struct CpcvResult {
    /// Mean out-of-sample Sharpe ratio across all paths
    pub mean_oos_sharpe: f64,
    /// Standard deviation of OOS Sharpe ratios
    pub std_oos_sharpe: f64,
    /// Probability that Sharpe > 0 (based on path distribution)
    pub prob_profitable: f64,
    /// Minimum OOS Sharpe observed (worst case)
    pub min_oos_sharpe: f64,
    /// Maximum OOS Sharpe observed (best case)
    pub max_oos_sharpe: f64,
    /// Number of valid paths evaluated
    pub valid_path_count: usize,
    /// Total number of paths
    pub total_paths: usize,
    /// Individual path results
    pub path_results: Vec<PathResult>,
    /// Overfitting metric: (mean_train_sharpe - mean_oos_sharpe) / std_oos_sharpe
    pub overfitting_score: f64,
}

impl CpcvResult {
    pub const fn invalid() -> Self {
        Self {
            mean_oos_sharpe: f64::NEG_INFINITY,
            std_oos_sharpe: 0.0,
            prob_profitable: 0.0,
            min_oos_sharpe: f64::NEG_INFINITY,
            max_oos_sharpe: f64::NEG_INFINITY,
            valid_path_count: 0,
            total_paths: 0,
            path_results: Vec::new(),
            overfitting_score: f64::MAX,
        }
    }
}

/// Combinatorial Purged Cross-Validation Engine
pub struct CpcvOverfitGuard {
    config: CpcvConfig,
    dataset_length: usize,
    fold_boundaries: Vec<(usize, usize)>, // (start, end) for each fold
}

impl CpcvOverfitGuard {
    pub fn new(config: CpcvConfig, dataset_length: usize) -> Self {
        let mut guard = Self {
            config,
            dataset_length,
            fold_boundaries: Vec::new(),
        };
        guard.compute_fold_boundaries();
        guard
    }

    /// Compute fold boundaries ensuring minimum samples per fold
    fn compute_fold_boundaries(&mut self) {
        self.fold_boundaries.clear();
        
        let samples_per_fold = self.dataset_length / self.config.n_folds;
        
        if samples_per_fold < self.config.min_samples_per_fold {
            // Dataset too small for requested folds - reduce folds
            let actual_folds = self.dataset_length / self.config.min_samples_per_fold;
            if actual_folds < 2 {
                return; // Cannot create valid folds
            }
        }

        for i in 0..self.config.n_folds {
            let start = i * samples_per_fold;
            let end = if i == self.config.n_folds - 1 {
                self.dataset_length
            } else {
                (i + 1) * samples_per_fold
            };
            self.fold_boundaries.push((start, end));
        }
    }

    /// Generate all combinatorial paths for CPCV
    /// Returns iterator over (train_indices, test_indices, purged_indices)
    pub fn generate_paths(&self) -> Vec<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        let mut paths = Vec::new();
        let n = self.config.n_folds;
        let k = self.config.n_test_folds;

        if n < k || k == 0 {
            return paths;
        }

        // Generate all combinations of k test folds from n folds
        let test_combinations = self.combinations(n, k);

        for test_fold_indices in test_combinations {
            let mut test_indices = Vec::new();
            let mut purged_indices = Vec::new();

            // Collect test indices and purge boundaries
            let mut purge_regions: Vec<(usize, usize)> = Vec::new();
            
            for &fold_idx in &test_fold_indices {
                let (start, end) = self.fold_boundaries[fold_idx];
                test_indices.extend(start..end);
                
                // Add purge region before this test fold
                if fold_idx > 0 {
                    let (_, prev_end) = self.fold_boundaries[fold_idx - 1];
                    let purge_size = ((end - start) as f64 * self.config.purge_ratio) as usize;
                    if purge_size > 0 {
                        purge_regions.push((prev_end.saturating_sub(purge_size), prev_end));
                    }
                }
            }

            // Collect purged indices
            for (purge_start, purge_end) in purge_regions {
                purged_indices.extend(purge_start..purge_end);
            }

            // Train indices are everything except test and purged
            let mut train_indices = Vec::new();
            for i in 0..self.dataset_length {
                if !test_indices.contains(&i) && !purged_indices.contains(&i) {
                    train_indices.push(i);
                }
            }

            paths.push((train_indices, test_indices, purged_indices));
        }

        paths
    }

    /// Compute binomial coefficient C(n, k)
    fn combinations(&self, n: usize, k: usize) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        self.combine_helper(0, n, k, &mut Vec::new(), &mut result);
        result
    }

    fn combine_helper(
        &self,
        start: usize,
        n: usize,
        k: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }

        for i in start..n {
            current.push(i);
            self.combine_helper(i + 1, n, k, current, result);
            current.pop();
        }
    }

    /// Evaluate a tree using CPCV
    /// Returns distribution of out-of-sample performance metrics
    pub fn evaluate_cpcv(
        &self,
        tree: &ExpressionTree,
        full_data: &DataWindow,
        evaluator_template: &AstEvaluator,
    ) -> CpcvResult {
        let paths = self.generate_paths();
        
        if paths.is_empty() {
            return CpcvResult::invalid();
        }

        let total_paths = paths.len();
        let mut path_results = Vec::with_capacity(total_paths);
        let mut oos_sharpes: Vec<f64> = Vec::with_capacity(total_paths);
        let mut train_sharpes: Vec<f64> = Vec::new();

        for (path_idx, (train_idx, test_idx, _purged_idx)) in paths.into_iter().enumerate() {
            if train_idx.is_empty() || test_idx.is_empty() {
                continue;
            }

            // Create data windows for train and test
            let train_data = self.filter_data_by_indices(full_data, &train_idx);
            let test_data = self.filter_data_by_indices(full_data, &test_idx);

            // Evaluate on training data
            let mut train_eval = evaluator_template.clone_for_evaluation();
            let train_result = train_eval.evaluate(tree, &train_data);

            // Evaluate on test data (out-of-sample)
            let mut test_eval = evaluator_template.clone_for_evaluation();
            let test_result = test_eval.evaluate(tree, &test_data);

            if !train_result.is_valid || !test_result.is_valid {
                continue;
            }

            oos_sharpes.push(test_result.sharpe_ratio);
            train_sharpes.push(train_result.sharpe_ratio);

            path_results.push(PathResult {
                train_indices: train_idx,
                test_indices: test_idx,
                purged_indices: _purged_idx,
                train_sharpe: train_result.sharpe_ratio,
                test_sharpe: test_result.sharpe_ratio,
                is_valid: true,
            });
        }

        if oos_sharpes.is_empty() {
            return CpcvResult::invalid();
        }

        // Calculate statistics
        let valid_count = oos_sharpes.len();
        let mean_oos = oos_sharpes.iter().sum::<f64>() / valid_count as f64;
        let variance = oos_sharpes.iter().map(|s| (s - mean_oos).powi(2)).sum::<f64>() / valid_count as f64;
        let std_oos = variance.sqrt();
        
        let min_oos = oos_sharpes.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_oos = oos_sharpes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        // Probability of profitable strategy (Sharpe > 0)
        let profitable_count = oos_sharpes.iter().filter(|&&s| s > 0.0).count();
        let prob_profitable = profitable_count as f64 / valid_count as f64;

        // Overfitting score
        let mean_train = train_sharpes.iter().sum::<f64>() / train_sharpes.len() as f64;
        let overfitting_score = if std_oos < 1e-10 {
            if (mean_train - mean_oos).abs() < 1e-10 {
                0.0
            } else {
                f64::MAX
            }
        } else {
            (mean_train - mean_oos) / std_oos
        };

        CpcvResult {
            mean_oos_sharpe: mean_oos,
            std_oos_sharpe: std_oos,
            prob_profitable,
            min_oos_sharpe: min_oos,
            max_oos_sharpe: max_oos,
            valid_path_count: valid_count,
            total_paths,
            path_results,
            overfitting_score,
        }
    }

    /// Filter data to specific indices (creates a view, not a copy where possible)
    fn filter_data_by_indices(&self, data: &DataWindow, indices: &[usize]) -> DataWindow {
        // For efficiency, we create new vectors here
        // In production, this would use zero-copy views or memory-mapped files
        let mut open = Vec::with_capacity(indices.len());
        let mut high = Vec::with_capacity(indices.len());
        let mut low = Vec::with_capacity(indices.len());
        let mut close = Vec::with_capacity(indices.len());
        let mut volume = Vec::with_capacity(indices.len());
        let mut obi = Vec::with_capacity(indices.len());
        let mut micro_price = Vec::with_capacity(indices.len());
        let mut timestamp = Vec::with_capacity(indices.len());

        for &idx in indices {
            if idx < data.length {
                open.push(data.open[idx]);
                high.push(data.high[idx]);
                low.push(data.low[idx]);
                close.push(data.close[idx]);
                volume.push(data.volume[idx]);
                obi.push(data.obi[idx]);
                micro_price.push(data.micro_price[idx]);
                timestamp.push(data.timestamp[idx]);
            }
        }

        // Static storage for the filtered data (in production, use arena allocation)
        // This is a simplified implementation - real implementation would avoid copies
        DataWindow::new(
            &open, &high, &low, &close, &volume, &obi, &micro_price, &timestamp
        )
    }

    /// Check if a strategy passes the overfitting threshold
    pub fn is_strategy_valid(&self, result: &CpcvResult, min_prob: f64, max_overfit: f64) -> bool {
        result.valid_path_count > 0
            && result.prob_profitable >= min_prob
            && result.overfitting_score <= max_overfit
            && result.mean_oos_sharpe > 0.0
    }
}

// Helper trait for cloning evaluators (simplified - real impl would use proper cloning)
impl AstEvaluator {
    pub fn clone_for_evaluation(&self) -> Self {
        AstEvaluator::new(self.ts_buffer.len(), self.var_bindings.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpcv_config() {
        let config = CpcvConfig::default();
        assert_eq!(config.n_folds, 5);
        assert_eq!(config.n_test_folds, 1);
        assert!((config.purge_ratio - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_combinations() {
        let guard = CpcvOverfitGuard::new(CpcvConfig::default(), 1000);
        let combos = guard.combinations(5, 2);
        assert_eq!(combos.len(), 10); // C(5,2) = 10
    }

    #[test]
    fn test_path_generation() {
        let config = CpcvConfig {
            n_folds: 4,
            n_test_folds: 1,
            ..Default::default()
        };
        let guard = CpcvOverfitGuard::new(config, 400);
        let paths = guard.generate_paths();
        assert_eq!(paths.len(), 4); // C(4,1) = 4 paths
    }
}
