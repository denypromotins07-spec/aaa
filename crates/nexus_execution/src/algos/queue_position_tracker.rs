//! Queue Position Tracker for Exchange Matching Engine
//! 
//! Calculates queue priority and detects hidden iceberg orders
//! to optimize order placement and avoid adverse selection.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Order book level information for queue tracking
#[derive(Debug, Clone)]
pub struct OrderBookLevel {
    pub price: i64,
    pub visible_size: i64,
    pub total_size_estimate: i64, // Including estimated hidden size
    pub our_order_size: i64,      // Our position in the queue
    pub our_queue_position: i64,  // Estimated position (0 = front)
}

/// Queue state at a specific price level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueuePriority {
    /// At the front of the queue (best position)
    Front,
    /// In the middle of the queue
    Middle,
    /// Near the back of the queue
    Back,
    /// Behind a suspected hidden iceberg
    BehindIceberg,
}

/// Detection result for hidden icebergs
#[derive(Debug, Clone)]
pub struct IcebergDetection {
    pub is_detected: bool,
    pub confidence: u32, // 0-10000 basis points
    pub estimated_hidden_size: i64,
    pub evidence: Vec<&'static str>,
}

/// Configuration for queue tracker
#[derive(Debug, Clone)]
pub struct QueueTrackerConfig {
    /// Number of consecutive fills at same price to trigger iceberg suspicion
    pub fills_for_iceberg_detection: u32,
    /// Minimum size ratio to consider as hidden (hidden/visible)
    pub min_hidden_ratio: u32, // basis points (e.g., 5000 = 50%)
    /// Time window for analyzing fills (milliseconds)
    pub analysis_window_ms: u64,
    /// Queue position threshold for "front" classification
    pub front_queue_threshold_pct: u32,
    /// Queue position threshold for "back" classification
    pub back_queue_threshold_pct: u32,
}

impl Default for QueueTrackerConfig {
    fn default() -> Self {
        Self {
            fills_for_iceberg_detection: 3,
            min_hidden_ratio: 5000, // 50%
            analysis_window_ms: 1000, // 1 second
            front_queue_threshold_pct: 1000, // 10%
            back_queue_threshold_pct: 8000, // 80%
        }
    }
}

/// Recent fill event for analysis
#[derive(Debug, Clone)]
struct FillEvent {
    timestamp: Instant,
    price: i64,
    quantity: i64,
    was_at_best: bool,
}

/// Queue Position Tracker
pub struct QueuePositionTracker {
    config: QueueTrackerConfig,
    /// Recent fills for iceberg detection
    recent_fills: VecDeque<FillEvent>,
    /// Current order book levels we're tracking
    tracked_levels: Vec<OrderBookLevel>,
    /// Detected icebergs by price level
    detected_icebergs: std::collections::HashMap<i64, IcebergDetection>,
    /// Statistics
    stats: QueueTrackerStats,
}

#[derive(Debug, Clone, Default)]
pub struct QueueTrackerStats {
    pub total_analyses: u64,
    pub icebergs_detected: u64,
    pub queue_jumps_triggered: u64,
    pub average_queue_position_pct: u32,
}

impl QueuePositionTracker {
    pub fn new(config: QueueTrackerConfig) -> Self {
        Self {
            config,
            recent_fills: VecDeque::with_capacity(100),
            tracked_levels: Vec::with_capacity(10),
            detected_icebergs: std::collections::HashMap::new(),
            stats: QueueTrackerStats::default(),
        }
    }

    /// Record a fill event for analysis
    pub fn record_fill(&mut self, price: i64, quantity: i64, was_at_best: bool) {
        let event = FillEvent {
            timestamp: Instant::now(),
            price,
            quantity,
            was_at_best,
        };

        self.recent_fills.push_back(event);

        // Cleanup old events outside analysis window
        let cutoff = Instant::now() - Duration::from_millis(self.config.analysis_window_ms);
        while let Some(front) = self.recent_fills.front() {
            if front.timestamp < cutoff {
                self.recent_fills.pop_front();
            } else {
                break;
            }
        }

        // Analyze for iceberg detection
        self.analyze_iceberg_patterns();
    }

    /// Update our order's queue position
    pub fn update_our_position(&mut self, price: i64, our_size: i64, queue_position: i64, total_size: i64) {
        let level = OrderBookLevel {
            price,
            visible_size: total_size,
            total_size_estimate: total_size,
            our_order_size: our_size,
            our_queue_position: queue_position,
        };

        // Update or add tracked level
        if let Some(existing) = self.tracked_levels.iter_mut().find(|l| l.price == price) {
            *existing = level;
        } else {
            self.tracked_levels.push(level);
        }
    }

    /// Analyze recent fills for hidden iceberg patterns
    fn analyze_iceberg_patterns(&mut self) {
        self.stats.total_analyses += 1;

        // Group fills by price level
        let mut fills_by_price: std::collections::HashMap<i64, Vec<&FillEvent>> = 
            std::collections::HashMap::new();
        
        for fill in &self.recent_fills {
            fills_by_price.entry(fill.price).or_default().push(fill);
        }

        // Check each price level for iceberg patterns
        for (price, fills) in &fills_by_price {
            if fills.len() >= self.config.fills_for_iceberg_detection as usize {
                let total_filled: i64 = fills.iter().map(|f| f.quantity).sum();
                
                // Get visible size at this level (if tracked)
                let visible_size = self.tracked_levels
                    .iter()
                    .find(|l| l.price == *price)
                    .map(|l| l.visible_size)
                    .unwrap_or(0);

                // Check if total filled significantly exceeds visible size
                if visible_size > 0 && total_filled > visible_size {
                    let hidden_size = total_filled - visible_size;
                    let hidden_ratio = (hidden_size * 10000) / visible_size;

                    if hidden_ratio >= self.config.min_hidden_ratio {
                        let detection = IcebergDetection {
                            is_detected: true,
                            confidence: (hidden_ratio.min(10000)) as u32,
                            estimated_hidden_size: hidden_size,
                            evidence: vec![
                                "Multiple fills exceed visible size",
                                "Consistent refilling at price level",
                            ],
                        };

                        self.detected_icebergs.insert(*price, detection);
                        self.stats.icebergs_detected += 1;

                        log::debug!(
                            "Hidden iceberg detected at price {}: {} hidden units ({}% confidence)",
                            price,
                            hidden_size,
                            detection.confidence / 100
                        );
                    }
                }
            }
        }
    }

    /// Get queue priority classification for our order
    pub fn get_queue_priority(&self, price: i64) -> QueuePriority {
        // First check if we're behind a detected iceberg
        if let Some(iceberg) = self.detected_icebergs.get(&price) {
            if iceberg.is_detected && iceberg.confidence >= 7000 {
                return QueuePriority::BehindIceberg;
            }
        }

        // Then classify based on queue position
        if let Some(level) = self.tracked_levels.iter().find(|l| l.price == price) {
            if level.visible_size <= 0 {
                return QueuePriority::Back;
            }

            let position_pct = ((level.our_queue_position * 10000) / level.visible_size) as u32;

            if position_pct <= self.config.front_queue_threshold_pct {
                QueuePriority::Front
            } else if position_pct >= self.config.back_queue_threshold_pct {
                QueuePriority::Back
            } else {
                QueuePriority::Middle
            }
        } else {
            QueuePriority::Middle // Default if not tracked
        }
    }

    /// Check if we should cancel and requeue (jump the queue)
    /// Returns true if queue jump is recommended
    pub fn should_jump_queue(&self, price: i64, time_in_queue_ms: u64) -> bool {
        let priority = self.get_queue_priority(price);

        match priority {
            QueuePriority::BehindIceberg => {
                // Always jump if behind confirmed iceberg
                self.stats.queue_jumps_triggered += 1;
                true
            }
            QueuePriority::Back => {
                // Jump if we've been waiting too long at the back
                if time_in_queue_ms > self.config.analysis_window_ms * 2 {
                    self.stats.queue_jumps_triggered += 1;
                    true
                } else {
                    false
                }
            }
            _ => false, // Don't jump from front or middle positions
        }
    }

    /// Get recommended action based on queue analysis
    pub fn get_recommended_action(&self, price: i64, time_in_queue_ms: u64) -> QueueAction {
        let priority = self.get_queue_priority(price);

        match priority {
            QueuePriority::Front => QueueAction::Hold,
            QueuePriority::Middle => {
                if time_in_queue_ms > self.config.analysis_window_ms {
                    QueueAction::ConsiderReprice
                } else {
                    QueueAction::Hold
                }
            }
            QueuePriority::Back => {
                if time_in_queue_ms > self.config.analysis_window_ms * 2 {
                    QueueAction::RepriceBetter
                } else {
                    QueueAction::Monitor
                }
            }
            QueuePriority::BehindIceberg => QueueAction::CancelAndRequeue,
        }
    }

    /// Clear detected icebergs older than the analysis window
    pub fn cleanup_old_detections(&mut self) {
        // Keep only icebergs detected within the analysis window
        // (In practice, you'd track detection timestamps)
        self.detected_icebergs.clear();
    }

    /// Get statistics
    pub fn get_stats(&self) -> QueueTrackerStats {
        self.stats.clone()
    }

    /// Get all detected icebergs
    pub fn get_detected_icebergs(&self) -> &std::collections::HashMap<i64, IcebergDetection> {
        &self.detected_icebergs
    }
}

/// Recommended action based on queue analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueAction {
    /// Keep order as-is
    Hold,
    /// Monitor the situation
    Monitor,
    /// Consider repricing slightly
    ConsiderReprice,
    /// Reprice to better position
    RepriceBetter,
    /// Cancel and immediately requeue at better price
    CancelAndRequeue,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iceberg_detection() {
        let mut tracker = QueuePositionTracker::new(QueueTrackerConfig {
            fills_for_iceberg_detection: 3,
            min_hidden_ratio: 5000, // 50%
            ..Default::default()
        });

        // Set up a level with 1000 visible size
        tracker.update_our_position(50000, 100, 50, 1000);

        // Record multiple fills that exceed visible size
        tracker.record_fill(50000, 400, true);
        tracker.record_fill(50000, 400, true);
        tracker.record_fill(50000, 400, true);

        // Should detect iceberg (1200 filled vs 1000 visible = 20% hidden)
        let detections = tracker.get_detected_icebergs();
        assert!(detections.contains_key(&50000));
        
        let detection = detections.get(&50000).unwrap();
        assert!(detection.is_detected);
        assert!(detection.confidence > 0);
    }

    #[test]
    fn test_queue_priority_classification() {
        let mut tracker = QueuePositionTracker::default();

        // Set up level where we're at the front (position 50 of 1000 = 5%)
        tracker.update_our_position(50000, 100, 50, 1000);
        
        let priority = tracker.get_queue_priority(50000);
        assert_eq!(priority, QueuePriority::Front);

        // Set up level where we're in the middle (position 500 of 1000 = 50%)
        tracker.update_our_position(50001, 100, 500, 1000);
        
        let priority = tracker.get_queue_priority(50001);
        assert_eq!(priority, QueuePriority::Middle);

        // Set up level where we're at the back (position 900 of 1000 = 90%)
        tracker.update_our_position(50002, 100, 900, 1000);
        
        let priority = tracker.get_queue_priority(50002);
        assert_eq!(priority, QueuePriority::Back);
    }

    #[test]
    fn test_queue_jump_recommendation() {
        let mut tracker = QueuePositionTracker::new(QueueTrackerConfig {
            fills_for_iceberg_detection: 3,
            ..Default::default()
        });

        // Simulate being behind an iceberg
        tracker.update_our_position(50000, 100, 500, 1000);
        
        // Create iceberg detection
        tracker.record_fill(50000, 500, true);
        tracker.record_fill(50000, 500, true);
        tracker.record_fill(50000, 500, true);

        // Should recommend queue jump when behind iceberg
        assert!(tracker.should_jump_queue(50000, 1000));

        let action = tracker.get_recommended_action(50000, 1000);
        assert_eq!(action, QueueAction::CancelAndRequeue);
    }
}
