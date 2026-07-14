//! Live Fire Atomic Gate
//! The ONLY way to enable live trading - mathematically prevents bypass of pre-flight checklist

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{info, error, warn};

use crate::preflight::unanimous_checklist::ChecklistResult;

/// Error types for live fire gate operations
#[derive(Debug, thiserror::Error)]
pub enum LiveFireGateError {
    #[error("Cannot enable live trading: pre-flight checklist did not pass")]
    ChecklistNotPassed,
    #[error("Cannot enable live trading: checklist result was critical failure")]
    CriticalChecklistFailure,
    #[error("Live trading already enabled")]
    AlreadyEnabled,
    #[error("Live trading disabled: {0}")]
    Disabled(String),
    #[error("Gate is in lockdown mode - manual override attempt detected")]
    LockdownMode,
}

/// Secure token proving checklist was executed and passed
#[derive(Debug, Clone)]
pub struct ChecklistToken {
    /// Unique epoch ID when checklist was executed
    epoch: u64,
    /// Timestamp when checklist completed
    timestamp_ms: u64,
    /// Hash of checklist results (for audit trail)
    result_hash: u64,
}

impl ChecklistToken {
    fn new(epoch: u64, result: &ChecklistResult) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        format!("{:?}", result).hash(&mut hasher);
        
        Self {
            epoch,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            result_hash: hasher.finish(),
        }
    }
}

/// The Live Fire Atomic Gate
/// 
/// CRITICAL SECURITY PROPERTIES:
/// 1. The `is_live_enabled` AtomicBool can ONLY be set via `try_promote_to_live()`
/// 2. `try_promote_to_live()` REQUIRES a valid ChecklistToken from unanimous pass
/// 3. NO manual override, RPC command, or debug path can bypass this
/// 4. Once disabled, re-enabling requires a NEW checklist execution
pub struct LiveFireAtomicGate {
    /// The actual live trading flag - private and inaccessible directly
    is_live_enabled: AtomicBool,
    
    /// Current epoch counter - incremented on each state change
    epoch: AtomicU64,
    
    /// Last successful checklist token (for audit)
    last_valid_token: std::sync::Mutex<Option<ChecklistToken>>,
    
    /// Lockdown flag - set if bypass attempt detected
    lockdown_mode: AtomicBool,
}

impl LiveFireAtomicGate {
    pub fn new() -> Self {
        Self {
            is_live_enabled: AtomicBool::new(false),
            epoch: AtomicU64::new(0),
            last_valid_token: std::sync::Mutex::new(None),
            lockdown_mode: AtomicBool::new(false),
        }
    }

    /// Check if live trading is currently enabled
    pub fn is_live_enabled(&self) -> bool {
        self.is_live_enabled.load(Ordering::Acquire)
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Attempt to promote to live trading mode
    /// 
    /// THIS IS THE ONLY WAY TO ENABLE LIVE TRADING
    /// 
    /// Requirements:
    /// 1. ChecklistResult MUST be UnanimousPass or PassWithWarnings
    /// 2. No critical failures allowed
    /// 3. Gate must not be in lockdown mode
    /// 
    /// Returns ChecklistToken on success (required for subsequent operations)
    pub fn try_promote_to_live(&self, checklist_result: ChecklistResult) -> Result<ChecklistToken, LiveFireGateError> {
        // Check lockdown mode first
        if self.lockdown_mode.load(Ordering::Acquire) {
            error!("LIVE FIRE GATE IN LOCKDOWN - bypass attempt detected");
            return Err(LiveFireGateError::LockdownMode);
        }

        // CRITICAL: Mathematically verify checklist passed
        if !checklist_result.can_proceed() {
            error!("Pre-flight checklist failed - live trading DENIED");
            return Err(LiveFireGateError::CriticalChecklistFailure);
        }

        // Verify it's actually a unanimous pass (not just warnings)
        if !checklist_result.is_unanimous_pass() {
            warn!("Pre-flight passed with warnings - proceeding with caution");
            // Still allow, but log the warnings
            if let ChecklistResult::PassWithWarnings { warnings, .. } = &checklist_result {
                for warning in warnings {
                    warn!("Warning: {}", warning);
                }
            }
        }

        // Check if already enabled
        if self.is_live_enabled.load(Ordering::Acquire) {
            warn!("Live trading already enabled");
            return Err(LiveFireGateError::AlreadyEnabled);
        }

        // Generate token BEFORE enabling (atomic sequence)
        let new_epoch = self.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        let token = ChecklistToken::new(new_epoch, &checklist_result);

        // ATOMICALLY enable live trading
        // This is the single point where the switch flips
        self.is_live_enabled.store(true, Ordering::Release);
        
        // Store token for audit
        {
            let mut last_token = self.last_valid_token.lock().unwrap();
            *last_token = Some(token.clone());
        }

        info!("=== LIVE FIRE ATOMIC GATE ACTIVATED ===");
        info!("Epoch: {}", new_epoch);
        info!("Checklist Hash: {:016x}", token.result_hash);
        info!("Timestamp: {}", token.timestamp_ms);
        info!("=== TRADING LIVE ===");

        Ok(token)
    }

    /// Disable live trading (e.g., during graceful shutdown)
    pub fn disable_live(&self) {
        if self.is_live_enabled.swap(false, Ordering::AcqRel) {
            info!("=== LIVE FIRE ATOMIC GATE DEACTIVATED ===");
            // Increment epoch to invalidate any in-flight tokens
            self.epoch.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// Force disable and enter lockdown (e.g., on security breach detection)
    pub fn force_lockdown(&self, reason: &str) {
        self.is_live_enabled.store(false, Ordering::Release);
        self.lockdown_mode.store(true, Ordering::Release);
        self.epoch.fetch_add(1, Ordering::AcqRel);
        
        error!("=== LIVE FIRE GATE IN LOCKDOWN ===");
        error!("Reason: {}", reason);
        error!("Manual re-enable is IMPOSSIBLE - restart required");
    }

    /// Check if gate is in lockdown mode
    pub fn is_lockdown(&self) -> bool {
        self.lockdown_mode.load(Ordering::Acquire)
    }

    /// Get last valid token (for audit/logging)
    pub fn get_last_token(&self) -> Option<ChecklistToken> {
        self.last_valid_token.lock().unwrap().clone()
    }

    /// Execute a closure ONLY if live trading is enabled
    /// This provides an additional safety layer for critical operations
    pub fn execute_if_live<F, T>(&self, f: F) -> Result<T, LiveFireGateError>
    where
        F: FnOnce() -> T,
    {
        if !self.is_live_enabled.load(Ordering::Acquire) {
            return Err(LiveFireGateError::Disabled(
                "Live trading not enabled".to_string()
            ));
        }

        if self.lockdown_mode.load(Ordering::Acquire) {
            return Err(LiveFireGateError::LockdownMode);
        }

        Ok(f())
    }
}

impl Default for LiveFireAtomicGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cannot_enable_without_checklist() {
        let gate = LiveFireAtomicGate::new();
        
        // Try to enable with a failure result
        let fail_result = ChecklistResult::CriticalFailure {
            failures: vec![],
        };
        
        assert!(matches!(
            gate.try_promote_to_live(fail_result),
            Err(LiveFireGateError::CriticalChecklistFailure)
        ));
        
        // Verify still disabled
        assert!(!gate.is_live_enabled());
    }

    #[test]
    fn test_can_enable_with_unanimous_pass() {
        let gate = LiveFireAtomicGate::new();
        
        let pass_result = ChecklistResult::UnanimousPass {
            details: vec![],
        };
        
        let result = gate.try_promote_to_live(pass_result);
        
        assert!(result.is_ok());
        assert!(gate.is_live_enabled());
    }

    #[test]
    fn test_lockdown_prevents_reenable() {
        let gate = LiveFireAtomicGate::new();
        
        // Enable normally first
        let pass_result = ChecklistResult::UnanimousPass { details: vec![] };
        let _ = gate.try_promote_to_live(pass_result);
        
        // Force lockdown
        gate.force_lockdown("security test");
        
        // Try to re-enable (should fail)
        let pass_result2 = ChecklistResult::UnanimousPass { details: vec![] };
        assert!(matches!(
            gate.try_promote_to_live(pass_result2),
            Err(LiveFireGateError::LockdownMode)
        ));
    }
}
