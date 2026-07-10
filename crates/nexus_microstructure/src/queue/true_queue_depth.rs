//! True Queue Depth calculator combining survival analysis with displayed volume.
//! 
//! Multiplies displayed volume at each price level by its survival probability
//! to provide the actual executable liquidity rather than spoofed liquidity.

use std::collections::HashMap;
use parking_lot::RwLock;
use crate::queue::survival_analysis::KaplanMeierEstimator;

/// True queue depth estimate for a price level
#[derive(Debug, Clone)]
pub struct QueueDepthEstimate {
    pub price_level: i64,
    pub displayed_volume: u64,
    pub survival_probability: f64,
    pub true_executable_volume: u64,
    pub confidence: f64,
    pub is_spoofed: bool,
}

/// True Queue Depth Calculator
pub struct TrueQueueDepthCalculator {
    survival_estimator: KaplanMeierEstimator,
    current_book_state: RwLock<HashMap<i64, u64>>, // price -> displayed volume
    last_estimates: RwLock<HashMap<i64, QueueDepthEstimate>>,
    /// Threshold below which queue is considered spoofed
    spoof_threshold: f64,
}

impl TrueQueueDepthCalculator {
    /// Create a new true queue depth calculator
    pub fn new(survival_estimator: KaplanMeierEstimator, spoof_threshold: f64) -> Self {
        Self {
            survival_estimator,
            current_book_state: RwLock::new(HashMap::with_capacity(256)),
            last_estimates: RwLock::new(HashMap::new()),
            spoof_threshold,
        }
    }

    /// Update the current order book state
    #[inline]
    pub fn update_book_state(&self, price_level: i64, displayed_volume: u64) {
        let mut book = self.current_book_state.write();
        if displayed_volume == 0 {
            book.remove(&price_level);
        } else {
            book.insert(price_level, displayed_volume);
        }
    }

    /// Calculate true queue depth for all price levels
    pub fn calculate_all_depths(&self, max_time_ms: f64) -> Vec<QueueDepthEstimate> {
        let book = self.current_book_state.read();
        let mut estimates = Vec::with_capacity(book.len());
        
        for (&price_level, &displayed_volume) in book.iter() {
            if let Some(survival_prob) = self.survival_estimator.get_survival_at_time(price_level, max_time_ms) {
                let true_volume = (displayed_volume as f64 * survival_prob) as u64;
                let is_spoofed = survival_prob < self.spoof_threshold;
                
                // Estimate confidence based on survival estimator's data quality
                let confidence = if survival_prob > 0.8 {
                    0.9
                } else if survival_prob > 0.5 {
                    0.7
                } else {
                    0.5
                };
                
                let estimate = QueueDepthEstimate {
                    price_level,
                    displayed_volume,
                    survival_probability: survival_prob,
                    true_executable_volume: true_volume,
                    confidence,
                    is_spoofed,
                };
                
                estimates.push(estimate);
            } else {
                // No survival data available, assume full volume but low confidence
                let estimate = QueueDepthEstimate {
                    price_level,
                    displayed_volume,
                    survival_probability: 1.0,
                    true_executable_volume: displayed_volume,
                    confidence: 0.3, // Low confidence due to lack of data
                    is_spoofed: false,
                };
                estimates.push(estimate);
            }
        }
        
        // Cache results
        let mut last = self.last_estimates.write();
        last.clear();
        for est in &estimates {
            last.insert(est.price_level, est.clone());
        }
        
        estimates
    }

    /// Get true queue depth for a specific price level
    pub fn get_depth(&self, price_level: i64, max_time_ms: f64) -> Option<QueueDepthEstimate> {
        // Check cache first
        {
            let cached = self.last_estimates.read();
            if let Some(est) = cached.get(&price_level) {
                return Some(est.clone());
            }
        }
        
        // Calculate fresh estimate
        let book = self.current_book_state.read();
        let displayed_volume = *book.get(&price_level)?;
        
        let survival_prob = self.survival_estimator.get_survival_at_time(price_level, max_time_ms)?;
        let true_volume = (displayed_volume as f64 * survival_prob) as u64;
        let is_spoofed = survival_prob < self.spoof_threshold;
        
        Some(QueueDepthEstimate {
            price_level,
            displayed_volume,
            survival_probability: survival_prob,
            true_executable_volume: true_volume,
            confidence: 0.5,
            is_spoofed,
        })
    }

    /// Get total true depth across multiple price levels
    pub fn get_aggregate_depth(&self, levels: &[i64], max_time_ms: f64) -> u64 {
        let mut total = 0u64;
        
        for &level in levels {
            if let Some(depth) = self.get_depth(level, max_time_ms) {
                total += depth.true_executable_volume;
            }
        }
        
        total
    }

    /// Get ratio of true to displayed depth (liquidity quality metric)
    pub fn get_liquidity_quality_ratio(&self, max_time_ms: f64) -> f64 {
        let estimates = self.calculate_all_depths(max_time_ms);
        
        if estimates.is_empty() {
            return 1.0;
        }
        
        let total_displayed: u64 = estimates.iter().map(|e| e.displayed_volume).sum();
        let total_true: u64 = estimates.iter().map(|e| e.true_executable_volume).sum();
        
        if total_displayed == 0 {
            return 1.0;
        }
        
        total_true as f64 / total_displayed as f64
    }

    /// Detect spoofed levels (very low survival probability)
    pub fn detect_spoofed_levels(&self, max_time_ms: f64) -> Vec<i64> {
        let estimates = self.calculate_all_depths(max_time_ms);
        estimates.into_iter()
            .filter(|e| e.is_spoofed)
            .map(|e| e.price_level)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::survival_analysis::{KaplanMeierEstimator, QueueEvent};

    #[test]
    fn test_true_queue_depth() {
        let survival = KaplanMeierEstimator::new(5, 0.95).unwrap();
        
        // Add some survival data showing 50% survival rate
        for i in 0..20 {
            let time_ms = i as f64 * 10.0;
            let event = if i % 2 == 0 { QueueEvent::Fill } else { QueueEvent::Canceled };
            survival.record_event(100, time_ms, event, 100);
        }
        
        let calculator = TrueQueueDepthCalculator::new(survival, 0.3);
        calculator.update_book_state(100, 1000);
        
        let depths = calculator.calculate_all_depths(100.0);
        assert!(!depths.is_empty());
        
        let depth = depths.iter().find(|d| d.price_level == 100).unwrap();
        assert!(depth.true_executable_volume <= depth.displayed_volume);
    }
}
