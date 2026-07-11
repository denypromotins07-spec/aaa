//! Wavefunction Collapse Lock for NEXUS-OMEGA
//! 
//! Ensures the AI's observation is locked to a specific branch
//! after wavefunction collapse, preventing decoherence-induced drift.
//! 
//! This module implements the final stage of quantum branch selection.

use core::fmt;
use alloc::{vec::Vec};

/// Represents a locked wavefunction state
#[derive(Debug, Clone, Copy)]
pub struct LockedWavefunction {
    /// Branch ID that was collapsed into
    pub collapsed_branch_id: u128,
    /// Lock timestamp (s)
    pub lock_time: f64,
    /// Expected decoherence time (s)
    pub expected_decoherence: f64,
    /// Lock strength (0-1)
    pub lock_strength: f64,
    /// Whether lock is stable
    pub is_stable: bool,
}

/// Configuration for the collapse lock
#[derive(Debug, Clone, Copy)]
pub struct CollapseLockConfig {
    /// Minimum lock strength required
    pub min_lock_strength: f64,
    /// Maximum acceptable decoherence rate
    pub max_decoherence_rate: f64,
    /// Lock duration before refresh needed (s)
    pub lock_duration: f64,
}

impl Default for CollapseLockConfig {
    fn default() -> Self {
        Self {
            min_lock_strength: 0.99,
            max_decoherence_rate: 1e-9,
            lock_duration: 1e-3, // 1ms
        }
    }
}

/// The Wavefunction Collapse Lock
pub struct WavefunctionCollapseLock {
    config: CollapseLockConfig,
    /// Currently locked state
    locked_state: Option<LockedWavefunction>,
    /// Total locks performed
    total_locks: u64,
    /// Lock failures due to decoherence
    decoherence_failures: u64,
    /// History of branch IDs for cycle detection
    branch_history: Vec<u128>,
}

impl WavefunctionCollapseLock {
    pub fn new(config: CollapseLockConfig) -> Self {
        Self {
            config,
            locked_state: None,
            total_locks: 0,
            decoherence_failures: 0,
            branch_history: Vec::new(),
        }
    }

    /// Attempt to lock onto a collapsed branch
    /// Returns Result to avoid unwrap() in hot paths
    pub fn lock(&mut self, branch_id: u128, current_time: f64, decoherence_time: f64) 
        -> Result<(), LockError> 
    {
        // Check for rapid branch cycling (potential instability)
        if self.detect_branch_cycling(branch_id) {
            return Err(LockError::BranchCyclingDetected);
        }

        let lock_strength = Self::calculate_lock_strength(decoherence_time, self.config.lock_duration);
        
        if lock_strength < self.config.min_lock_strength {
            return Err(LockError::InsufficientLockStrength(lock_strength));
        }

        let locked = LockedWavefunction {
            collapsed_branch_id: branch_id,
            lock_time: current_time,
            expected_decoherence: decoherence_time,
            lock_strength,
            is_stable: true,
        };

        self.locked_state = Some(locked);
        self.total_locks += 1;
        self.branch_history.push(branch_id);

        // Limit history size
        if self.branch_history.len() > 100 {
            self.branch_history.remove(0);
        }

        Ok(())
    }

    fn calculate_lock_strength(decoherence_time: f64, lock_duration: f64) -> f64 {
        // Lock strength inversely proportional to decoherence rate
        // Stronger lock when decoherence time >> lock duration
        if decoherence_time <= 0.0 {
            return 0.0;
        }
        let ratio = lock_duration / decoherence_time;
        (-ratio).exp() // Exponential decay model
    }

    fn detect_branch_cycling(&self, new_branch_id: u128) -> bool {
        if self.branch_history.len() < 3 {
            return false;
        }

        // Check if we're rapidly cycling between same branches
        let recent_count = self.branch_history.iter()
            .rev()
            .take(5)
            .filter(|&&id| id == new_branch_id)
            .count();

        recent_count >= 3 // Cycling if same branch appears 3+ times in last 5
    }

    /// Verify lock stability
    pub fn verify_lock(&self, elapsed_time: f64) -> Result<bool, LockError> {
        let locked = self.locked_state.ok_or(LockError::NoActiveLock)?;

        if !locked.is_stable {
            return Ok(false);
        }

        // Check if elapsed time exceeds expected decoherence
        if elapsed_time > locked.expected_decoherence {
            self.decoherence_failures += 1;
            return Ok(false);
        }

        // Recalculate lock strength based on elapsed time
        let remaining_ratio = 1.0 - (elapsed_time / locked.expected_decoherence);
        let current_strength = locked.lock_strength * remaining_ratio;

        if current_strength < self.config.min_lock_strength {
            return Ok(false);
        }

        Ok(true)
    }

    /// Refresh the lock to extend duration
    pub fn refresh_lock(&mut self, current_time: f64) -> Result<(), LockError> {
        let mut locked = self.locked_state.ok_or(LockError::NoActiveLock)?;

        locked.lock_time = current_time;
        locked.is_stable = true;

        self.locked_state = Some(locked);
        Ok(())
    }

    /// Get the currently locked branch ID
    pub fn locked_branch_id(&self) -> Option<u128> {
        self.locked_state.map(|s| s.collapsed_branch_id)
    }

    /// Get lock success rate
    pub fn lock_success_rate(&self) -> f64 {
        if self.total_locks == 0 {
            return 0.0;
        }
        let successful = self.total_locks - self.decoherence_failures;
        successful as f64 / self.total_locks as f64
    }

    /// Clear the lock
    pub fn unlock(&mut self) {
        self.locked_state = None;
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.locked_state = None;
        self.total_locks = 0;
        self.decoherence_failures = 0;
        self.branch_history.clear();
    }
}

/// Errors that can occur in wavefunction locking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockError {
    NoActiveLock,
    InsufficientLockStrength(f64),
    BranchCyclingDetected,
    DecoherenceExceeded,
    LockUnstable,
}

impl fmt::Display for LockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LockError::NoActiveLock => write!(f, "No active wavefunction lock"),
            LockError::InsufficientLockStrength(s) => {
                write!(f, "Lock strength {} below minimum", s)
            }
            LockError::BranchCyclingDetected => {
                write!(f, "Rapid branch cycling detected (instability)")
            }
            LockError::DecoherenceExceeded => {
                write!(f, "Decoherence time exceeded")
            }
            LockError::LockUnstable => write!(f, "Lock is unstable"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lock_creation() {
        let config = CollapseLockConfig::default();
        let lock = WavefunctionCollapseLock::new(config);
        assert!(lock.locked_branch_id().is_none());
        assert_eq!(lock.total_locks, 0);
    }

    #[test]
    fn test_successful_lock() {
        let config = CollapseLockConfig::default();
        let mut lock = WavefunctionCollapseLock::new(config);

        // Long decoherence time should give strong lock
        let result = lock.lock(12345, 0.0, 1.0);
        assert!(result.is_ok());
        assert_eq!(lock.locked_branch_id(), Some(12345));
    }

    #[test]
    fn test_weak_lock_rejected() {
        let config = CollapseLockConfig::default();
        let mut lock = WavefunctionCollapseLock::new(config);

        // Very short decoherence time should fail
        let result = lock.lock(12345, 0.0, 1e-10);
        assert!(result.is_err());
    }

    #[test]
    fn test_lock_verification() {
        let config = CollapseLockConfig::default();
        let mut lock = WavefunctionCollapseLock::new(config);

        lock.lock(12345, 0.0, 1.0).unwrap();

        // Verify at t=0.1s (should still be stable)
        assert!(lock.verify_lock(0.1).unwrap());

        // Verify at t=2.0s (past decoherence time)
        assert!(!lock.verify_lock(2.0).unwrap());
    }

    #[test]
    fn test_branch_cycling_detection() {
        let config = CollapseLockConfig::default();
        let mut lock = WavefunctionCollapseLock::new(config);

        // Add some history
        for _ in 0..5 {
            lock.branch_history.push(99999);
        }

        // Try to lock same branch - should detect cycling
        let result = lock.lock(99999, 0.0, 1.0);
        assert_eq!(result, Err(LockError::BranchCyclingDetected));
    }
}
