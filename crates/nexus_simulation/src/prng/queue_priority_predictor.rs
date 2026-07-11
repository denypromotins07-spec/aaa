//! Queue Priority Predictor
//! 
//! Predicts queue tie-breaking order once PRNG state is cracked.
//! Enables perfect limit order positioning for execution priority.

use core::fmt;

/// Represents a predicted queue position
#[derive(Debug, Clone)]
pub struct QueuePrediction {
    /// Order ID that will be next in queue
    pub predicted_next_order_id: u64,
    /// Our predicted position (1 = first in queue)
    pub our_position: usize,
    /// Total orders ahead of us
    pub orders_ahead: usize,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Estimated time until execution (microseconds)
    pub estimated_execution_time_us: u64,
}

/// Configuration for queue prediction
#[derive(Debug, Clone, Copy)]
pub struct QueuePredictorConfig {
    /// Minimum confidence threshold for acting on predictions
    pub min_confidence_threshold: f32,
    /// Maximum queue depth to track
    pub max_queue_depth: usize,
    /// Lookahead window for predictions (number of trades)
    pub lookahead_trades: usize,
}

impl Default for QueuePredictorConfig {
    fn default() -> Self {
        Self {
            min_confidence_threshold: 0.8,
            max_queue_depth: 10_000,
            lookahead_trades: 100,
        }
    }
}

/// Internal queue state tracking
struct QueueState {
    /// Simulated PRNG state for tie-breaking
    prng_state: Option<Vec<u64>>,
    /// Current queue depth at each price level
    queue_depths: std::collections::BTreeMap<i64, usize>,
    /// Order IDs in queue (by price level)
    order_queue: std::collections::BTreeMap<i64, Vec<u64>>,
}

impl QueueState {
    fn new() -> Self {
        Self {
            prng_state: None,
            queue_depths: std::collections::BTreeMap::new(),
            order_queue: std::collections::BTreeMap::new(),
        }
    }
}

/// Queue Priority Predictor using cracked PRNG state
pub struct QueuePriorityPredictor {
    config: QueuePredictorConfig,
    state: QueueState,
    our_orders: Vec<(i64, u64)>, // (price_level, order_id)
}

impl QueuePriorityPredictor {
    pub fn new(config: QueuePredictorConfig) -> Self {
        Self {
            config,
            state: QueueState::new(),
            our_orders: Vec::new(),
        }
    }

    /// Set the cracked PRNG state for prediction
    pub fn set_prng_state(&mut self, state: Vec<u64>) {
        self.state.prng_state = Some(state);
    }

    /// Update queue depth for a price level
    pub fn update_queue_depth(&mut self, price_level: i64, depth: usize) {
        let clamped_depth = depth.min(self.config.max_queue_depth);
        self.state.queue_depths.insert(price_level, clamped_depth);
        
        // Initialize order queue if needed
        self.state.order_queue.entry(price_level).or_default();
    }

    /// Add an observed order to the queue simulation
    pub fn add_order_to_queue(&mut self, price_level: i64, order_id: u64, timestamp_ns: u64) {
        let queue = self.state.order_queue.entry(price_level).or_default();
        
        if queue.len() >= self.config.max_queue_depth {
            queue.remove(0);
        }
        
        queue.push(order_id);
        
        // Update depth
        *self.state.queue_depths.entry(price_level).or_insert(0) = queue.len();
    }

    /// Register our own order for position tracking
    pub fn register_our_order(&mut self, price_level: i64, order_id: u64) {
        self.our_orders.push((price_level, order_id));
    }

    /// Predict queue position for our orders
    pub fn predict_positions(&self) -> Vec<QueuePrediction> {
        let mut predictions = Vec::new();

        for &(price_level, order_id) in &self.our_orders {
            let queue = match self.state.order_queue.get(&price_level) {
                Some(q) => q,
                None => continue,
            };

            // Find our position in queue
            let our_index = queue.iter().position(|&id| id == order_id);
            
            if let Some(idx) = our_index {
                let orders_ahead = idx;
                let depth = queue.len();
                
                // Calculate confidence based on PRNG state availability and queue stability
                let base_confidence = if self.state.prng_state.is_some() { 0.9 } else { 0.5 };
                let depth_factor = 1.0 - (orders_ahead as f32 / depth as f32).min(1.0) * 0.2;
                let confidence = (base_confidence * depth_factor).max(0.0).min(1.0);

                // Estimate execution time based on recent trade velocity
                // Simplified: assume 1 trade per millisecond average
                let estimated_execution_time_us = (orders_ahead as u64) * 1000;

                let prediction = QueuePrediction {
                    predicted_next_order_id: queue.first().copied().unwrap_or(0),
                    our_position: idx + 1,
                    orders_ahead,
                    confidence,
                    estimated_execution_time_us,
                };

                predictions.push(prediction);
            }
        }

        predictions
    }

    /// Check if we should modify/cancel an order based on predicted position
    pub fn should_modify_order(&self, order_id: u64) -> QueueAction {
        let prediction = self.predict_positions()
            .into_iter()
            .find(|p| {
                // Find prediction for this order
                self.our_orders.iter().any(|&(_, id)| id == order_id)
            });

        match prediction {
            Some(p) if p.confidence < self.config.min_confidence_threshold => {
                QueueAction::CancelAndReposition
            }
            Some(p) if p.orders_ahead > self.config.max_queue_depth / 2 => {
                QueueAction::MoveToBetterPrice
            }
            Some(p) if p.our_position == 1 => {
                QueueAction::HoldPosition
            }
            _ => QueueAction::Wait,
        }
    }

    /// Simulate tie-breaking using cracked PRNG
    pub fn simulate_tiebreak(&self, seed: u64, num_contenders: usize) -> Option<usize> {
        let prng_state = self.state.prng_state.as_ref()?;
        
        // Simple LCG-based tie-break simulation
        let mut state = seed;
        for _ in 0..10 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        }
        
        let winner_index = (state as usize) % num_contenders;
        Some(winner_index)
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.state = QueueState::new();
        self.our_orders.clear();
    }

    /// Get current queue depths
    pub fn get_queue_depths(&self) -> &std::collections::BTreeMap<i64, usize> {
        &self.state.queue_depths
    }
}

/// Recommended action for an order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueAction {
    /// Keep order as-is
    HoldPosition,
    /// Wait for more information
    Wait,
    /// Cancel and re-submit at same price
    CancelAndReposition,
    /// Move to better price level
    MoveToBetterPrice,
    /// Aggressively cross spread
    CrossSpread,
}

/// Errors from queue prediction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueuePredictionError {
    NoPrngState,
    InvalidQueueDepth,
    OrderNotFound,
}

impl fmt::Display for QueuePredictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueuePredictionError::NoPrngState => write!(f, "PRNG state not set"),
            QueuePredictionError::InvalidQueueDepth => write!(f, "Invalid queue depth"),
            QueuePredictionError::OrderNotFound => write!(f, "Order not found in queue"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_prediction() {
        let config = QueuePredictorConfig::default();
        let mut predictor = QueuePriorityPredictor::new(config);

        // Set up a queue at price level 0
        for i in 0..10 {
            predictor.add_order_to_queue(0, i + 100, i * 1000);
        }

        // Register our order
        predictor.register_our_order(0, 105);

        let predictions = predictor.predict_positions();
        
        assert!(!predictions.is_empty());
        assert_eq!(predictions[0].our_position, 6); // We're 6th in queue
        assert_eq!(predictions[0].orders_ahead, 5);
    }

    #[test]
    fn test_tiebreak_simulation() {
        let config = QueuePredictorConfig::default();
        let mut predictor = QueuePriorityPredictor::new(config);
        
        // Set a mock PRNG state
        predictor.set_prng_state(vec![12345, 67890]);
        
        let result = predictor.simulate_tiebreak(42, 10);
        assert!(result.is_some());
        assert!(result.unwrap() < 10);
    }
}
