//! Iceberg order detector monitoring L3 order-by-order feed.
//! 
//! Identifies hidden volume that is being systematically replenished at the best bid/ask.

use std::collections::HashMap;
use parking_lot::RwLock;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum IcebergError {
    #[error("Invalid detection threshold")]
    InvalidThreshold,
}

/// Detected iceberg order information
#[derive(Debug, Clone)]
pub struct IcebergDetection {
    pub price_level: i64,
    pub visible_volume: u64,
    pub estimated_total_volume: u64,
    pub refresh_count: usize,
    pub confidence: f64,
    pub first_detected_ns: u64,
    pub last_refresh_ns: u64,
}

/// Tracking state for a potential iceberg
struct IcebergTracker {
    visible_volume: u64,
    total_executed: u64,
    refresh_count: usize,
    first_seen_ns: u64,
    last_activity_ns: u64,
    suspected_iceberg: bool,
}

impl IcebergTracker {
    fn new(volume: u64, timestamp_ns: u64) -> Self {
        Self {
            visible_volume: volume,
            total_executed: 0,
            refresh_count: 0,
            first_seen_ns: timestamp_ns,
            last_activity_ns: timestamp_ns,
            suspected_iceberg: false,
        }
    }
}

/// Configuration for iceberg detection
pub struct IcebergConfig {
    /// Minimum refresh count to confirm iceberg
    pub min_refreshes: usize,
    /// Volume tolerance for detecting refresh (percentage)
    pub volume_tolerance_pct: f64,
    /// Maximum time between refreshes (nanoseconds)
    pub max_refresh_interval_ns: u64,
}

impl Default for IcebergConfig {
    fn default() -> Self {
        Self {
            min_refreshes: 3,
            volume_tolerance_pct: 0.2, // 20% tolerance
            max_refresh_interval_ns: 1_000_000_000, // 1 second
        }
    }
}

/// Iceberg Order Detector for hidden liquidity identification
pub struct IcebergDetector {
    config: IcebergConfig,
    trackers: RwLock<HashMap<i64, IcebergTracker>>,
    confirmed_icebergs: RwLock<HashMap<i64, IcebergDetection>>,
}

impl IcebergDetector {
    /// Create a new iceberg detector
    pub fn new(config: IcebergConfig) -> Result<Self, IcebergError> {
        if config.min_refreshes == 0 || config.volume_tolerance_pct < 0.0 {
            return Err(IcebergError::InvalidThreshold);
        }
        
        Ok(Self {
            config,
            trackers: RwLock::new(HashMap::with_capacity(64)),
            confirmed_icebergs: RwLock::new(HashMap::new()),
        })
    }

    /// Process an order book update (L3 data)
    #[inline]
    pub fn process_update(&self, price_level: i64, new_volume: u64, timestamp_ns: u64, executed_volume: u64) {
        let mut trackers = self.trackers.write();
        
        let tracker = trackers.entry(price_level).or_insert_with(|| {
            IcebergTracker::new(new_volume, timestamp_ns)
        });
        
        // Check if this looks like a refresh
        let is_refresh = self.detect_refresh(tracker, new_volume, timestamp_ns);
        
        if is_refresh {
            tracker.refresh_count += 1;
            tracker.last_activity_ns = timestamp_ns;
            
            // Check if we have enough refreshes to confirm iceberg
            if tracker.refresh_count >= self.config.min_refreshes && !tracker.suspected_iceberg {
                tracker.suspected_iceberg = true;
                
                // Estimate total volume
                let avg_visible = (tracker.visible_volume + new_volume) / 2;
                let estimated_total = avg_visible * (tracker.refresh_count as u64 + 1);
                
                let confidence = ((tracker.refresh_count - self.config.min_refreshes) as f64 / 
                    self.config.min_refreshes as f64).min(1.0);
                
                let detection = IcebergDetection {
                    price_level,
                    visible_volume: new_volume,
                    estimated_total_volume: estimated_total,
                    refresh_count: tracker.refresh_count,
                    confidence,
                    first_detected_ns: tracker.first_seen_ns,
                    last_refresh_ns: timestamp_ns,
                };
                
                self.confirmed_icebergs.write().insert(price_level, detection);
            }
        } else {
            // Normal update
            tracker.visible_volume = new_volume;
            tracker.total_executed += executed_volume;
            tracker.last_activity_ns = timestamp_ns;
        }
    }

    /// Detect if a volume refresh occurred
    fn detect_refresh(&self, tracker: &IcebergTracker, new_volume: u64, timestamp_ns: u64) -> bool {
        // Check time window
        if timestamp_ns - tracker.last_activity_ns > self.config.max_refresh_interval_ns {
            return false;
        }
        
        // Check if volume increased significantly after execution
        let volume_increase = new_volume as f64 / tracker.visible_volume as f64;
        let expected_range = 1.0 - self.config.volume_tolerance_pct..=1.0 + self.config.volume_tolerance_pct;
        
        // Refresh detected if volume jumps back to original size after partial execution
        if tracker.total_executed > 0 && !expected_range.contains(&volume_increase) {
            // Volume jumped back up after being executed - likely iceberg refresh
            return true;
        }
        
        false
    }

    /// Get all confirmed iceberg orders
    pub fn get_confirmed_icebergs(&self) -> Vec<IcebergDetection> {
        self.confirmed_icebergs.read().values().cloned().collect()
    }

    /// Check if a specific price level has a confirmed iceberg
    pub fn is_iceberg(&self, price_level: i64) -> bool {
        self.confirmed_icebergs.read().contains_key(&price_level)
    }

    /// Get estimated hidden volume at a price level
    pub fn get_hidden_volume(&self, price_level: i64) -> Option<u64> {
        let icebergs = self.confirmed_icebergs.read();
        icebergs.get(&price_level).map(|d| {
            d.estimated_total_volume.saturating_sub(d.visible_volume)
        })
    }

    /// Clear stale icebergs (no activity for specified duration)
    pub fn clear_stale(&self, max_age_ns: u64, current_time_ns: u64) {
        let mut trackers = self.trackers.write();
        let mut confirmed = self.confirmed_icebergs.write();
        
        trackers.retain(|_, t| {
            current_time_ns - t.last_activity_ns <= max_age_ns
        });
        
        confirmed.retain(|price, _| {
            trackers.contains_key(price)
        });
    }

    /// Reset detector state
    pub fn reset(&self) {
        self.trackers.write().clear();
        self.confirmed_icebergs.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iceberg_detection() {
        let detector = IcebergDetector::new(IcebergConfig::default()).unwrap();
        
        // Simulate iceberg: visible 100, gets executed, refreshes to 100 repeatedly
        let base_time = 1_000_000_000u64;
        
        // Initial placement
        detector.process_update(100, 100, base_time, 0);
        
        // Execute 50, refresh back to 100
        detector.process_update(100, 100, base_time + 100_000_000, 50);
        detector.process_update(100, 100, base_time + 200_000_000, 50);
        detector.process_update(100, 100, base_time + 300_000_000, 50);
        
        // Should now be detected as iceberg
        assert!(detector.is_iceberg(100));
        
        let icebergs = detector.get_confirmed_icebergs();
        assert!(!icebergs.is_empty());
        assert!(icebergs[0].refresh_count >= 3);
    }
}
