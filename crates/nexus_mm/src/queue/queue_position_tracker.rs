//! Queue Position Tracker for Limit Orders.
//! Monitors exact volume ahead in L2/L3 order book.
//! Zero-allocation, no unwrap/expect in hot paths.

/// Error types for queue tracking
#[derive(Debug, Clone, PartialEq)]
pub enum QueueError {
    InvalidLevel,
    OrderNotFound,
    BookStale,
}

/// Queue position information
#[derive(Debug, Clone, Copy)]
pub struct QueuePosition {
    /// Volume ahead of our order (in base units)
    pub volume_ahead: u64,
    /// Total volume at price level (including ours)
    pub total_volume: u64,
    /// Our order size
    pub our_size: u64,
    /// Position rank (1 = front of queue)
    pub rank: u32,
    /// Estimated fill probability (0 to 1)
    pub fill_probability: f64,
}

impl QueuePosition {
    pub const fn new(
        volume_ahead: u64,
        total_volume: u64,
        our_size: u64,
        rank: u32,
        fill_probability: f64,
    ) -> Self {
        Self {
            volume_ahead,
            total_volume,
            our_size,
            rank,
            fill_probability,
        }
    }
}

/// Level data in order book
#[derive(Debug, Clone, Copy)]
pub struct PriceLevel {
    /// Price (in tick units)
    pub price_ticks: i64,
    /// Total volume at level
    pub volume: u64,
    /// Number of orders at level
    pub order_count: u32,
    /// Volume from aggressive orders (likely to cancel)
    pub flicker_volume: u64,
}

impl PriceLevel {
    pub const fn new(price_ticks: i64, volume: u64, order_count: u32) -> Self {
        Self {
            price_ticks,
            volume,
            order_count,
            flicker_volume: 0,
        }
    }
}

/// Queue Position Tracker
pub struct QueuePositionTracker {
    /// Our order ID
    our_order_id: Option<u64>,
    /// Our order side (true = bid, false = ask)
    our_is_bid: bool,
    /// Our order price (in ticks)
    our_price_ticks: i64,
    /// Our order size
    our_size: u64,
    /// Pre-allocated buffer for bid levels (zero-copy)
    bid_levels: Vec<PriceLevel>,
    /// Pre-allocated buffer for ask levels
    ask_levels: Vec<PriceLevel>,
    /// Number of valid bid levels
    bid_count: usize,
    /// Number of valid ask levels
    ask_count: usize,
    /// Last update timestamp (nanoseconds)
    last_update_ns: u64,
    /// Staleness threshold (nanoseconds)
    staleness_threshold_ns: u64,
}

impl QueuePositionTracker {
    pub fn new(max_levels: usize, staleness_threshold_ms: u64) -> Self {
        Self {
            our_order_id: None,
            our_is_bid: true,
            our_price_ticks: 0,
            our_size: 0,
            bid_levels: vec![PriceLevel::new(0, 0, 0); max_levels],
            ask_levels: vec![PriceLevel::new(0, 0, 0); max_levels],
            bid_count: 0,
            ask_count: 0,
            last_update_ns: 0,
            staleness_threshold_ns: staleness_threshold_ms * 1_000_000,
        }
    }
    
    /// Update order book levels (zero-copy, caller provides sorted levels)
    #[inline(always)]
    pub fn update_book_levels(
        &mut self,
        is_bid: bool,
        levels: &[PriceLevel],
        timestamp_ns: u64,
    ) -> Result<(), QueueError> {
        let target_levels = if is_bid {
            &mut self.bid_levels[..]
        } else {
            &mut self.ask_levels[..]
        };
        
        let copy_count = levels.len().min(target_levels.len());
        
        for i in 0..copy_count {
            target_levels[i] = levels[i];
        }
        
        if is_bid {
            self.bid_count = copy_count;
        } else {
            self.ask_count = copy_count;
        }
        
        self.last_update_ns = timestamp_ns;
        
        Ok(())
    }
    
    /// Set our order details
    #[inline(always)]
    pub fn set_our_order(
        &mut self,
        order_id: u64,
        is_bid: bool,
        price_ticks: i64,
        size: u64,
    ) {
        self.our_order_id = Some(order_id);
        self.our_is_bid = is_bid;
        self.our_price_ticks = price_ticks;
        self.our_size = size;
    }
    
    /// Clear our order
    #[inline(always)]
    pub fn clear_our_order(&mut self) {
        self.our_order_id = None;
    }
    
    /// Calculate current queue position
    #[inline(always)]
    pub fn calculate_position(&self) -> Result<QueuePosition, QueueError> {
        let order_id = self.our_order_id.ok_or(QueueError::OrderNotFound)?;
        
        // Check for stale data
        if self.last_update_ns == 0 {
            return Err(QueueError::BookStale);
        }
        
        let levels = if self.our_is_bid {
            &self.bid_levels[..self.bid_count]
        } else {
            &self.ask_levels[..self.ask_count]
        };
        
        let mut volume_ahead: u64 = 0;
        let mut found_level = false;
        let mut rank: u32 = 1;
        
        // Iterate through levels to find our position
        for level in levels {
            if level.price_ticks == self.our_price_ticks {
                found_level = true;
                
                // Estimate our rank within the level
                // Assume orders are roughly equal size for simplicity
                let avg_order_size = if level.order_count > 0 {
                    level.volume / level.order_count as u64
                } else {
                    self.our_size
                };
                
                if avg_order_size > 0 {
                    rank = ((volume_ahead / avg_order_size) + 1) as u32;
                }
                
                break;
            }
            
            // Accumulate volume ahead (better prices execute first)
            if self.our_is_bid {
                // For bids, higher prices are better
                if level.price_ticks > self.our_price_ticks {
                    volume_ahead = volume_ahead.saturating_add(level.volume);
                }
            } else {
                // For asks, lower prices are better
                if level.price_ticks < self.our_price_ticks {
                    volume_ahead = volume_ahead.saturating_add(level.volume);
                }
            }
        }
        
        if !found_level {
            // Our price not in book - we're behind all displayed volume
            return Ok(QueuePosition::new(
                volume_ahead,
                self.our_size,
                self.our_size,
                rank,
                0.0,
            ));
        }
        
        // Find total volume at our level
        let mut total_at_level: u64 = 0;
        for level in levels {
            if level.price_ticks == self.our_price_ticks {
                total_at_level = level.volume;
                break;
            }
        }
        
        // Simple fill probability estimate based on queue position
        let fill_prob = if total_at_level > 0 {
            let position_in_queue = volume_ahead.min(total_at_level);
            1.0 - (position_in_queue as f64 / total_at_level as f64)
        } else {
            0.0
        };
        
        Ok(QueuePosition::new(
            volume_ahead,
            total_at_level.max(self.our_size),
            self.our_size,
            rank,
            fill_prob.clamp(0.0, 1.0),
        ))
    }
    
    /// Check if book data is stale
    #[inline(always)]
    pub fn is_stale(&self, current_time_ns: u64) -> bool {
        current_time_ns - self.last_update_ns > self.staleness_threshold_ns
    }
    
    /// Get volume at a specific price level
    #[inline(always)]
    pub fn get_volume_at_price(&self, price_ticks: i64, is_bid: bool) -> Option<u64> {
        let levels = if is_bid {
            &self.bid_levels[..self.bid_count]
        } else {
            &self.ask_levels[..self.ask_count]
        };
        
        for level in levels {
            if level.price_ticks == price_ticks {
                return Some(level.volume);
            }
        }
        
        None
    }
    
    /// Reset tracker
    pub fn reset(&mut self) {
        self.our_order_id = None;
        self.bid_count = 0;
        self.ask_count = 0;
        self.last_update_ns = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_queue_position_basic() {
        let mut tracker = QueuePositionTracker::new(10, 1000);
        
        // Set up bid book: best bid at 100 with 1000 volume
        let levels = vec![
            PriceLevel::new(100, 1000, 10),
            PriceLevel::new(99, 500, 5),
        ];
        tracker.update_book_levels(true, &levels, 1_000_000_000).unwrap();
        
        // Place our order at best bid
        tracker.set_our_order(1, true, 100, 100);
        
        let pos = tracker.calculate_position().unwrap();
        
        assert!(pos.rank >= 1);
        assert!(pos.fill_probability > 0.0);
    }
    
    #[test]
    fn test_no_order_set() {
        let tracker = QueuePositionTracker::new(10, 1000);
        assert!(tracker.calculate_position().is_err());
    }
}
