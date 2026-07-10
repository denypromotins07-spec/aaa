//! Lock-free telemetry aggregator for shadow model evaluation
//!
//! Uses atomic operations to aggregate metrics without mutex contention.

use crate::MLOpsError;
use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};

/// Lock-free telemetry aggregator
pub struct TelemetryAggregator {
    /// Total predictions made
    prediction_count: AtomicU64,
    /// Sum of squared errors (fixed-point * 1e9)
    sum_squared_error: AtomicI64,
    /// Sum of absolute errors (fixed-point * 1e9)
    sum_absolute_error: AtomicI64,
    /// Hypothetical PnL (fixed-point * 1e9)
    hypothetical_pnl: AtomicI64,
    /// Sum of hypothetical returns for Sharpe calculation
    sum_returns: AtomicI64,
    /// Sum of squared returns for Sharpe calculation
    sum_sq_returns: AtomicI64,
}

impl TelemetryAggregator {
    /// Create new telemetry aggregator
    pub fn new() -> Self {
        Self {
            prediction_count: AtomicU64::new(0),
            sum_squared_error: AtomicI64::new(0),
            sum_absolute_error: AtomicI64::new(0),
            hypothetical_pnl: AtomicI64::new(0),
            sum_returns: AtomicI64::new(0),
            sum_sq_returns: AtomicI64::new(0),
        }
    }

    /// Record a prediction and its outcome (lock-free)
    pub fn record_prediction(&self, prediction: f64) -> Result<(), MLOpsError> {
        // In production, this would also receive actual outcome
        // For now, just increment counter
        self.prediction_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Record prediction error (prediction - actual)
    pub fn record_error(&self, error: f64) -> Result<(), MLOpsError> {
        // Convert to fixed-point (scale by 1e9 to preserve precision)
        let error_fp = (error * 1e9) as i64;
        let sq_error_fp = (error * error * 1e9) as i64;
        let abs_error_fp = error.abs().mul_ceil(1e9) as i64;

        self.sum_squared_error.fetch_add(sq_error_fp, Ordering::Relaxed);
        self.sum_absolute_error.fetch_add(abs_error_fp, Ordering::Relaxed);

        Ok(())
    }

    /// Record hypothetical PnL from shadow trading
    pub fn record_pnl(&self, pnl: f64) -> Result<(), MLOpsError> {
        let pnl_fp = (pnl * 1e9) as i64;
        
        self.hypothetical_pnl.fetch_add(pnl_fp, Ordering::Relaxed);
        self.sum_returns.fetch_add(pnl_fp, Ordering::Relaxed);
        self.sum_sq_returns.fetch_add((pnl * pnl * 1e9) as i64, Ordering::Relaxed);

        Ok(())
    }

    /// Get total prediction count
    pub fn prediction_count(&self) -> u64 {
        self.prediction_count.load(Ordering::Relaxed)
    }

    /// Get mean squared error
    pub fn mse(&self) -> f64 {
        let count = self.prediction_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        
        let sum_sq = self.sum_squared_error.load(Ordering::Relaxed) as f64;
        sum_sq / count as f64 / 1e9
    }

    /// Get mean absolute error
    pub fn mae(&self) -> f64 {
        let count = self.prediction_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        
        let sum_abs = self.sum_absolute_error.load(Ordering::Relaxed) as f64;
        sum_abs / count as f64 / 1e9
    }

    /// Get total hypothetical PnL
    pub fn total_pnl(&self) -> f64 {
        let pnl = self.hypothetical_pnl.load(Ordering::Relaxed) as f64;
        pnl / 1e9
    }

    /// Get annualized Sharpe ratio (assuming daily returns)
    pub fn sharpe_ratio(&self) -> f64 {
        let count = self.prediction_count.load(Ordering::Relaxed);
        if count < 2 {
            return 0.0;
        }

        let n = count as f64;
        let sum_ret = self.sum_returns.load(Ordering::Relaxed) as f64 / 1e9;
        let sum_sq_ret = self.sum_sq_returns.load(Ordering::Relaxed) as f64 / 1e9;

        let mean_ret = sum_ret / n;
        let variance = (sum_sq_ret / n) - (mean_ret * mean_ret);

        if variance <= 0.0 {
            return 0.0;
        }

        let std_dev = variance.sqrt();
        let daily_sharpe = mean_ret / std_dev;

        // Annualize (assuming 252 trading days)
        daily_sharpe * 252.0_f64.sqrt()
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.prediction_count.store(0, Ordering::Relaxed);
        self.sum_squared_error.store(0, Ordering::Relaxed);
        self.sum_absolute_error.store(0, Ordering::Relaxed);
        self.hypothetical_pnl.store(0, Ordering::Relaxed);
        self.sum_returns.store(0, Ordering::Relaxed);
        self.sum_sq_returns.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_aggregation() {
        let telemetry = TelemetryAggregator::new();

        for i in 0..100 {
            let error = (i % 10) as f64 * 0.01 - 0.045;
            telemetry.record_error(error).unwrap();
            telemetry.record_pnl(error * 100.0).unwrap();
        }

        assert_eq!(telemetry.prediction_count(), 100);
        assert!(telemetry.mse() > 0.0);
        assert!(telemetry.mae() > 0.0);
    }

    #[test]
    fn test_lock_free_concurrent_access() {
        use std::thread;
        
        let telemetry = std::sync::Arc::new(TelemetryAggregator::new());
        let mut handles = vec![];

        for t in 0..10 {
            let tel_clone = std::sync::Arc::clone(&telemetry);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let error = (t * 100 + i) as f64 * 0.001;
                    tel_clone.record_error(error).unwrap();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(telemetry.prediction_count(), 1000);
    }
}
