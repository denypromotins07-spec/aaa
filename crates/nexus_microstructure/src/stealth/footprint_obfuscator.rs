//! Footprint Obfuscator for hiding trading patterns from adversarial HFT firms.
//! 
//! Dynamically randomizes order sizes, time delays, and limit price offsets
//! using cryptographically secure randomness to prevent pattern recognition.

use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;
use crate::stealth::cryptographic_stealth_rng::CryptographicStealthRng;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ObfuscationError {
    #[error("Invalid obfuscation parameters")]
    InvalidParameters,
}

/// Obfuscated order parameters
#[derive(Debug, Clone)]
pub struct ObfuscatedOrder {
    /// Original intended size
    pub original_size: u64,
    /// Obfuscated size (randomized)
    pub obfuscated_size: u64,
    /// Original price offset (in ticks)
    pub original_price_offset: i64,
    /// Obfuscated price offset
    pub obfuscated_price_offset: i64,
    /// Delay to wait before submission (nanoseconds)
    pub delay_ns: u64,
    /// Split into child orders
    pub child_orders: Vec<ChildOrder>,
}

/// Child order for order splitting
#[derive(Debug, Clone)]
pub struct ChildOrder {
    pub size: u64,
    pub price_offset: i64,
    pub delay_ns: u64,
}

/// Configuration for footprint obfuscation
pub struct ObfuscationConfig {
    /// Maximum size deviation percentage (0-1)
    pub max_size_deviation: f64,
    /// Maximum price offset deviation (in ticks)
    pub max_tick_deviation: i64,
    /// Maximum delay (nanoseconds)
    pub max_delay_ns: u64,
    /// Minimum number of child orders for splits
    pub min_child_orders: usize,
    /// Maximum number of child orders for splits
    pub max_child_orders: usize,
    /// Probability of splitting an order (0-1)
    pub split_probability: f64,
}

impl Default for ObfuscationConfig {
    fn default() -> Self {
        Self {
            max_size_deviation: 0.2, // ±20%
            max_tick_deviation: 2,
            max_delay_ns: 10_000_000, // 10ms
            min_child_orders: 2,
            max_child_orders: 5,
            split_probability: 0.3,
        }
    }
}

/// Footprint Obfuscator for stealth execution
pub struct FootprintObfuscator {
    config: ObfuscationConfig,
    rng: CryptographicStealthRng,
    /// Order counter for seed variation
    order_counter: AtomicU64,
    /// Current obfuscation strategy index (rotated periodically)
    strategy_index: RwLock<usize>,
}

impl FootprintObfuscator {
    /// Create a new footprint obfuscator
    pub fn new(config: ObfuscationConfig, seed_entropy: &[u8]) -> Result<Self, ObfuscationError> {
        if config.max_size_deviation < 0.0 || config.max_size_deviation > 1.0 {
            return Err(ObfuscationError::InvalidParameters);
        }
        if config.split_probability < 0.0 || config.split_probability > 1.0 {
            return Err(ObfuscationError::InvalidParameters);
        }
        if config.min_child_orders > config.max_child_orders {
            return Err(ObfuscationError::InvalidParameters);
        }
        
        let rng = CryptographicStealthRng::new(seed_entropy)?;
        
        Ok(Self {
            config,
            rng,
            order_counter: AtomicU64::new(0),
            strategy_index: RwLock::new(0),
        })
    }

    /// Obfuscate an order to hide trading footprint
    pub fn obfuscate(&self, original_size: u64, price_offset: i64) -> Result<ObfuscatedOrder, ObfuscationError> {
        let counter = self.order_counter.fetch_add(1, Ordering::Relaxed);
        
        // Generate random values using cryptographic RNG (Audit Fix #3 - non-blocking)
        let size_factor = self.rng.next_f64();
        let tick_variation = self.rng.next_i64_range(-self.config.max_tick_deviation as i64, self.config.max_tick_deviation as i64);
        let delay = if self.config.max_delay_ns > 0 {
            self.rng.next_u64_range(0, self.config.max_delay_ns)
        } else {
            0
        };
        
        // Calculate obfuscated size with bounded deviation
        let deviation_multiplier = 1.0 + (size_factor - 0.5) * 2.0 * self.config.max_size_deviation;
        let obfuscated_size = ((original_size as f64 * deviation_multiplier) as u64).max(1);
        
        // Calculate obfuscated price offset
        let obfuscated_price_offset = price_offset.saturating_add(tick_variation);
        
        // Decide whether to split into child orders
        let split_roll = self.rng.next_f64();
        let child_orders = if split_roll < self.config.split_probability && original_size >= 100 {
            self.generate_child_orders(original_size, price_offset)
        } else {
            vec![ChildOrder {
                size: obfuscated_size,
                price_offset: obfuscated_price_offset,
                delay_ns: delay,
            }]
        };
        
        Ok(ObfuscatedOrder {
            original_size,
            obfuscated_size,
            original_price_offset: price_offset,
            obfuscated_price_offset,
            delay_ns: delay,
            child_orders,
        })
    }

    /// Generate child orders for order splitting
    fn generate_child_orders(&self, total_size: u64, base_price_offset: i64) -> Vec<ChildOrder> {
        let num_children = self.rng.next_usize_range(
            self.config.min_child_orders,
            self.config.max_child_orders + 1,
        );
        
        let mut child_orders = Vec::with_capacity(num_children);
        let mut remaining_size = total_size;
        
        for i in 0..num_children {
            let is_last = i == num_children - 1;
            
            // Random size for this child
            let child_size = if is_last {
                remaining_size // Ensure we use all size
            } else {
                let fraction = self.rng.next_f64();
                let size = ((fraction * remaining_size as f64) as u64).max(1);
                remaining_size -= size;
                size
            };
            
            // Slight price variation per child
            let price_var = self.rng.next_i64_range(-1, 2);
            let child_price = base_price_offset.saturating_add(price_var);
            
            // Staggered delays
            let delay = self.rng.next_u64_range(0, self.config.max_delay_ns / num_children as u64);
            
            child_orders.push(ChildOrder {
                size: child_size,
                price_offset: child_price,
                delay_ns: delay,
            });
        }
        
        child_orders
    }

    /// Rotate obfuscation strategy (call periodically)
    pub fn rotate_strategy(&self) {
        let mut idx = self.strategy_index.write();
        *idx = (*idx + 1) % 4;
    }

    /// Get current strategy index
    pub fn get_strategy_index(&self) -> usize {
        *self.strategy_index.read()
    }

    /// Reset the obfuscator
    pub fn reset(&self, new_seed: &[u8]) -> Result<(), ObfuscationError> {
        self.rng.reseed(new_seed)?;
        self.order_counter.store(0, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_obfuscation() {
        let config = ObfuscationConfig::default();
        let obfuscator = FootprintObfuscator::new(config, b"test_seed_123").unwrap();
        
        let original_size = 1000u64;
        let price_offset = 5i64;
        
        let obfuscated = obfuscator.obfuscate(original_size, price_offset).unwrap();
        
        // Size should be within bounds
        let min_size = (original_size as f64 * (1.0 - config.max_size_deviation)) as u64;
        let max_size = (original_size as f64 * (1.0 + config.max_size_deviation)) as u64;
        
        assert!(obfuscated.obfuscated_size >= min_size);
        assert!(obfuscated.obfuscated_size <= max_size);
        
        // Price offset should be close to original
        let price_diff = (obfuscated.obfuscated_price_offset - price_offset).abs();
        assert!(price_diff <= config.max_tick_deviation);
    }

    #[test]
    fn test_order_splitting() {
        // Run multiple times to trigger split probability
        let config = ObfuscationConfig {
            split_probability: 1.0, // Force splitting
            ..Default::default()
        };
        let obfuscator = FootprintObfuscator::new(config, b"test_seed").unwrap();
        
        let obfuscated = obfuscator.obfuscate(1000, 5).unwrap();
        
        // Should have child orders
        assert!(!obfuscated.child_orders.is_empty());
        
        // Total child size should equal original
        let total_child_size: u64 = obfuscated.child_orders.iter().map(|c| c.size).sum();
        assert_eq!(total_child_size, 1000);
    }
}
