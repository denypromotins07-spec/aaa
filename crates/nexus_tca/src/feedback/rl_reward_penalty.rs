//! RL Reward Penalty - Zero-copy TCA feedback to RL environment
//! 
//! Writes TCA metrics directly into shared memory for RL agent consumption,
//! enabling the agent to learn from real-world market impact.

use std::sync::atomic::{AtomicI64, AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RLPenaltyError {
    #[error("Shared memory not initialized")]
    SharedMemoryNotInitialized,
    #[error("Invalid penalty value: {reason}")]
    InvalidPenalty { reason: String },
    #[error("Buffer overflow: index {index} exceeds capacity {capacity}")]
    BufferOverflow { index: usize, capacity: usize },
}

/// RL reward penalty configuration
#[derive(Debug, Clone)]
pub struct RLPenaltyConfig {
    /// Weight for implementation shortfall penalty
    pub is_weight: f64,
    /// Weight for delay cost penalty
    pub delay_weight: f64,
    /// Weight for market impact penalty
    pub impact_weight: f64,
    /// Weight for spread cost penalty
    pub spread_weight: f64,
    /// Maximum penalty per step (clipping threshold)
    pub max_penalty_bps: i64,
    /// Penalty decay factor for historical penalties
    pub decay_factor: f64,
}

impl Default for RLPenaltyConfig {
    fn default() -> Self {
        Self {
            is_weight: 1.0,
            delay_weight: 0.5,
            impact_weight: 0.8,
            spread_weight: 0.3,
            max_penalty_bps: 100, // Cap at 100 bps
            decay_factor: 0.95,
        }
    }
}

/// Shared memory buffer for zero-copy RL communication
/// In production, this would point to actual shared memory segment
pub struct RLSharedMemory {
    /// Latest penalty in basis points (signed: positive = bad, negative = good)
    current_penalty_bps: AtomicI64,
    /// Cumulative penalty over episode
    cumulative_penalty_bps: AtomicI64,
    /// Step count within episode
    step_count: AtomicU64,
    /// Historical penalties (circular buffer indices)
    history_write_idx: AtomicU64,
    /// Enabled flag
    enabled: AtomicBool,
    /// Capacity of history buffer
    history_capacity: usize,
}

impl RLSharedMemory {
    pub fn new(history_capacity: usize) -> Self {
        Self {
            current_penalty_bps: AtomicI64::new(0),
            cumulative_penalty_bps: AtomicI64::new(0),
            step_count: AtomicU64::new(0),
            history_write_idx: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
            history_capacity,
        }
    }

    /// Write current penalty to shared memory (zero-copy if using actual shm)
    pub fn write_penalty(&self, penalty_bps: i64) -> Result<(), RLPenaltyError> {
        if !self.enabled.load(Ordering::Acquire) {
            return Ok(());
        }

        self.current_penalty_bps.store(penalty_bps, Ordering::Release);
        
        // Update cumulative
        self.cumulative_penalty_bps.fetch_add(penalty_bps, Ordering::Relaxed);
        
        // Increment step
        self.step_count.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Read current penalty (for RL agent)
    pub fn read_penalty(&self) -> i64 {
        self.current_penalty_bps.load(Ordering::Acquire)
    }

    /// Get cumulative penalty for episode
    pub fn get_cumulative_penalty(&self) -> i64 {
        self.cumulative_penalty_bps.load(Ordering::Acquire)
    }

    /// Get step count
    pub fn get_step_count(&self) -> u64 {
        self.step_count.load(Ordering::Acquire)
    }

    /// Reset for new episode
    pub fn reset_episode(&self) {
        self.current_penalty_bps.store(0, Ordering::Release);
        self.cumulative_penalty_bps.store(0, Ordering::Release);
        self.step_count.store(0, Ordering::Release);
    }

    /// Enable/disable penalty writing
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    /// Check if enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }
}

/// RL Reward Penalty Calculator
pub struct RLRewardPenalty {
    config: RLPenaltyConfig,
    shared_memory: Arc<RLSharedMemory>,
    /// Scale factor for fixed-point calculations
    scale: i64,
}

impl RLRewardPenalty {
    pub fn new(config: RLPenaltyConfig, shared_memory: Arc<RLSharedMemory>, scale: i64) -> Self {
        Self {
            config,
            shared_memory,
            scale,
        }
    }

    /// Calculate and write penalty based on TCA results
    pub fn calculate_and_write_penalty(
        &self,
        shortfall_bps: i64,
        delay_cost: i64,
        market_impact: i64,
        spread_cost: i64,
        notional: i64,
    ) -> Result<i64, RLPenaltyError> {
        if notional <= 0 {
            return Err(RLPenaltyError::InvalidPenalty {
                reason: "Notional must be positive".to_string(),
            });
        }

        // Convert costs to basis points
        let delay_bps = (delay_cost.abs() * 10_000 / notional).abs();
        let impact_bps = (market_impact.abs() * 10_000 / notional).abs();
        let spread_bps = (spread_cost.abs() * 10_000 / notional).abs();

        // Calculate weighted penalty
        let raw_penalty = (self.config.is_weight * shortfall_bps as f64
            + self.config.delay_weight * delay_bps as f64
            + self.config.impact_weight * impact_bps as f64
            + self.config.spread_weight * spread_bps as f64) as i64;

        // Apply clipping
        let clipped_penalty = raw_penalty.clamp(-self.config.max_penalty_bps, self.config.max_penalty_bps);

        // Write to shared memory
        self.shared_memory.write_penalty(clipped_penalty)?;

        Ok(clipped_penalty)
    }

    /// Calculate penalty for favorable execution (negative penalty = reward bonus)
    pub fn calculate_bonus(
        &self,
        favorable_bps: i64,
    ) -> Result<i64, RLPenaltyError> {
        if favorable_bps > 0 {
            return Err(RLPenaltyError::InvalidPenalty {
                reason: "Favorable execution should have negative bps".to_string(),
            });
        }

        // Bonus is simply the negative penalty (capped)
        let bonus = favorable_bps.clamp(-self.config.max_penalty_bps, 0);
        
        self.shared_memory.write_penalty(bonus)?;
        
        Ok(bonus)
    }

    /// Get reference to shared memory for direct RL access
    pub fn get_shared_memory(&self) -> Arc<RLSharedMemory> {
        Arc::clone(&self.shared_memory)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: RLPenaltyConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn get_config(&self) -> &RLPenaltyConfig {
        &self.config
    }
}

/// Builder for creating RL penalty system
pub struct RLPenaltyBuilder {
    config: RLPenaltyConfig,
    history_capacity: usize,
    scale: i64,
}

impl RLPenaltyBuilder {
    pub fn new() -> Self {
        Self {
            config: RLPenaltyConfig::default(),
            history_capacity: 1000,
            scale: 1_000_000,
        }
    }

    pub fn config(mut self, config: RLPenaltyConfig) -> Self {
        self.config = config;
        self
    }

    pub fn history_capacity(mut self, capacity: usize) -> Self {
        self.history_capacity = capacity;
        self
    }

    pub fn scale(mut self, scale: i64) -> Self {
        self.scale = scale;
        self
    }

    pub fn build(self) -> RLRewardPenalty {
        let shared_memory = Arc::new(RLSharedMemory::new(self.history_capacity));
        RLRewardPenalty::new(self.config, shared_memory, self.scale)
    }
}

impl Default for RLPenaltyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_penalty_calculation() {
        let config = RLPenaltyConfig::default();
        let shm = Arc::new(RLSharedMemory::new(100));
        let penalty = RLRewardPenalty::new(config, shm, 1_000_000);

        let result = penalty.calculate_and_write_penalty(
            50,  // shortfall_bps
            20,  // delay_cost_bps equivalent
            15,  // impact_bps equivalent
            10,  // spread_bps equivalent
            1_000_000, // notional
        ).unwrap();

        // Penalty should be positive and bounded
        assert!(result > 0);
        assert!(result <= penalty.config.max_penalty_bps);
    }

    #[test]
    fn test_bonus_calculation() {
        let config = RLPenaltyConfig::default();
        let shm = Arc::new(RLSharedMemory::new(100));
        let penalty = RLRewardPenalty::new(config, shm, 1_000_000);

        // Favorable execution: -30 bps
        let result = penalty.calculate_bonus(-30).unwrap();

        // Should be negative (bonus)
        assert!(result < 0);
        assert!(result >= -penalty.config.max_penalty_bps);
    }

    #[test]
    fn test_shared_memory_persistence() {
        let config = RLPenaltyConfig::default();
        let shm = Arc::new(RLSharedMemory::new(100));
        let penalty = RLRewardPenalty::new(config.clone(), shm.clone(), 1_000_000);

        // Write penalty
        penalty.calculate_and_write_penalty(50, 10, 10, 5, 1_000_000).unwrap();

        // Read from shared memory directly
        assert!(shm.read_penalty() > 0);
        assert_eq!(shm.get_step_count(), 1);

        // Write another
        penalty.calculate_and_write_penalty(30, 5, 5, 3, 1_000_000).unwrap();
        assert_eq!(shm.get_step_count(), 2);
        assert!(shm.get_cumulative_penalty() > 0);
    }

    #[test]
    fn test_episode_reset() {
        let config = RLPenaltyConfig::default();
        let shm = Arc::new(RLSharedMemory::new(100));
        let penalty = RLRewardPenalty::new(config, shm.clone(), 1_000_000);

        // Accumulate some penalty
        penalty.calculate_and_write_penalty(50, 10, 10, 5, 1_000_000).unwrap();
        penalty.calculate_and_write_penalty(30, 5, 5, 3, 1_000_000).unwrap();

        assert!(shm.get_cumulative_penalty() > 0);
        assert_eq!(shm.get_step_count(), 2);

        // Reset episode
        shm.reset_episode();

        assert_eq!(shm.get_cumulative_penalty(), 0);
        assert_eq!(shm.get_step_count(), 0);
    }

    #[test]
    fn test_disabled_penalty() {
        let config = RLPenaltyConfig::default();
        let shm = Arc::new(RLSharedMemory::new(100));
        let penalty = RLRewardPenalty::new(config, shm.clone(), 1_000_000);

        // Disable
        shm.set_enabled(false);

        // Writing should succeed but not update
        let initial = shm.read_penalty();
        penalty.calculate_and_write_penalty(50, 10, 10, 5, 1_000_000).unwrap();

        // When disabled, penalty should remain at initial value
        assert_eq!(shm.read_penalty(), initial);
    }
}
