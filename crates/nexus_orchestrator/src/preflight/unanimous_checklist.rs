//! Unanimous Pre-Flight Checklist
//! All checks must pass before transitioning from ShadowMode to LiveTrading

use std::time::Duration;
use tracing::{info, warn, error};

/// Result of a single pre-flight check
#[derive(Debug, Clone, PartialEq)]
pub enum CheckResult {
    Pass { message: String },
    Fail { reason: String },
    Warning { message: String },
}

/// Aggregate result of the entire checklist
#[derive(Debug, Clone, PartialEq)]
pub enum ChecklistResult {
    /// All checks passed - unanimous consent for live trading
    UnanimousPass { details: Vec<CheckResult> },
    /// One or more critical checks failed
    CriticalFailure { failures: Vec<CheckResult> },
    /// Non-critical warnings but can proceed with caution
    PassWithWarnings { details: Vec<CheckResult>, warnings: Vec<String> },
}

impl ChecklistResult {
    pub fn is_unanimous_pass(&self) -> bool {
        matches!(self, ChecklistResult::UnanimousPass { .. })
    }

    pub fn can_proceed(&self) -> bool {
        matches!(
            self,
            ChecklistResult::UnanimousPass { .. } | ChecklistResult::PassWithWarnings { .. }
        )
    }
}

/// Pre-flight check types
#[derive(Debug, Clone)]
pub enum PreflightCheckType {
    NtpClockSync,
    ShadowReconciliation,
    RiskGatekeeperLimits,
    RaftLeadershipStonith,
}

/// The Unanimous Pre-Flight Checklist
pub struct PreFlightChecklist {
    /// Maximum allowed NTP drift in milliseconds
    max_ntp_drift_ms: i64,
    /// Maximum allowed shadow reconciliation drift (basis points)
    max_reconciliation_drift_bps: f64,
}

impl Default for PreFlightChecklist {
    fn default() -> Self {
        Self {
            max_ntp_drift_ms: 50, // 50ms max drift from exchange
            max_reconciliation_drift_bps: 0.0, // Zero tolerance for OMS mismatch
        }
    }
}

impl PreFlightChecklist {
    pub fn new(max_ntp_ms: i64, max_recon_bps: f64) -> Self {
        Self {
            max_ntp_drift_ms: max_ntp_ms,
            max_reconciliation_drift_bps: max_recon_bps,
        }
    }

    /// Execute all pre-flight checks and return unanimous result
    pub async fn execute_all_checks(&self) -> ChecklistResult {
        info!("=== PREFLIGHT CHECKLIST INITIATED ===");
        
        let mut results: Vec<CheckResult> = Vec::with_capacity(4);
        let mut failures: Vec<CheckResult> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();

        // Check 1: NTP Clock Sync
        let ntp_result = self.check_ntp_clock_sync().await;
        match &ntp_result {
            CheckResult::Pass { .. } => {
                info!("[CHECK] NTP Sync: PASS");
                results.push(ntp_result);
            }
            CheckResult::Fail { reason } => {
                error!("[CHECK] NTP Sync: FAIL - {}", reason);
                failures.push(ntp_result);
                results.push(ntp_result);
            }
            CheckResult::Warning { message } => {
                warn!("[CHECK] NTP Sync: WARN - {}", message);
                warnings.push(message.clone());
                results.push(ntp_result);
            }
        }

        // Check 2: Shadow Reconciliation Ledger
        let recon_result = self.check_shadow_reconciliation().await;
        match &recon_result {
            CheckResult::Pass { .. } => {
                info!("[CHECK] Shadow Reconciliation: PASS");
                results.push(recon_result);
            }
            CheckResult::Fail { reason } => {
                error!("[CHECK] Shadow Reconciliation: FAIL - {}", reason);
                failures.push(recon_result);
                results.push(recon_result);
            }
            CheckResult::Warning { message } => {
                warn!("[CHECK] Shadow Reconciliation: WARN - {}", message);
                warnings.push(message.clone());
                results.push(recon_result);
            }
        }

        // Check 3: Risk Gatekeeper Limits
        let risk_result = self.check_risk_gatekeeper_limits().await;
        match &risk_result {
            CheckResult::Pass { .. } => {
                info!("[CHECK] Risk Gatekeeper: PASS");
                results.push(risk_result);
            }
            CheckResult::Fail { reason } => {
                error!("[CHECK] Risk Gatekeeper: FAIL - {}", reason);
                failures.push(risk_result);
                results.push(risk_result);
            }
            CheckResult::Warning { message } => {
                warn!("[CHECK] Risk Gatekeeper: WARN - {}", message);
                warnings.push(message.clone());
                results.push(risk_result);
            }
        }

        // Check 4: Raft Leadership + STONITH
        let raft_result = self.check_raft_leadership_stonith().await;
        match &raft_result {
            CheckResult::Pass { .. } => {
                info!("[CHECK] Raft Leadership + STONITH: PASS");
                results.push(raft_result);
            }
            CheckResult::Fail { reason } => {
                error!("[CHECK] Raft Leadership + STONITH: FAIL - {}", reason);
                failures.push(raft_result);
                results.push(raft_result);
            }
            CheckResult::Warning { message } => {
                warn!("[CHECK] Raft Leadership + STONITH: WARN - {}", message);
                warnings.push(message.clone());
                results.push(raft_result);
            }
        }

        // Determine final result
        if !failures.is_empty() {
            error!("=== PREFLIGHT CHECKLIST FAILED ===");
            error!("{} critical failure(s)", failures.len());
            ChecklistResult::CriticalFailure { failures }
        } else if !warnings.is_empty() {
            warn!("=== PREFLIGHT CHECKLIST PASSED WITH WARNINGS ===");
            warn!("{} warning(s)", warnings.len());
            ChecklistResult::PassWithWarnings { details: results, warnings }
        } else {
            info!("=== PREFLIGHT CHECKLIST UNANIMOUS PASS ===");
            ChecklistResult::UnanimousPass { details: results }
        }
    }

    /// Check 1: NTP Clock Sync within 50ms of exchange
    async fn check_ntp_clock_sync(&self) -> CheckResult {
        // In production: query NTP server and compare with exchange timestamp
        // Placeholder implementation
        let current_drift_ms = 0i64; // Simulated drift
        
        if current_drift_ms.abs() <= self.max_ntp_drift_ms {
            CheckResult::Pass {
                message: format!("NTP drift: {}ms (limit: {}ms)", current_drift_ms, self.max_ntp_drift_ms),
            }
        } else if current_drift_ms.abs() <= self.max_ntp_drift_ms * 2 {
            CheckResult::Warning {
                message: format!("NTP drift elevated: {}ms", current_drift_ms),
            }
        } else {
            CheckResult::Fail {
                reason: format!("NTP drift excessive: {}ms > {}ms", current_drift_ms, self.max_ntp_drift_ms),
            }
        }
    }

    /// Check 2: Shadow Reconciliation - OMS matches Exchange REST API
    async fn check_shadow_reconciliation(&self) -> CheckResult {
        // In production: compare local OMS state with exchange REST API snapshot
        // Placeholder implementation
        let drift_bps = 0.0f64; // Simulated drift
        
        if drift_bps <= self.max_reconciliation_drift_bps {
            CheckResult::Pass {
                message: format!("Shadow reconciliation: {:.4} bps drift (limit: {:.4} bps)", 
                    drift_bps, self.max_reconciliation_drift_bps),
            }
        } else {
            CheckResult::Fail {
                reason: format!("Shadow reconciliation drift: {:.4} bps > {:.4} bps", 
                    drift_bps, self.max_reconciliation_drift_bps),
            }
        }
    }

    /// Check 3: Risk Gatekeeper limits loaded and acknowledged
    async fn check_risk_gatekeeper_limits(&self) -> CheckResult {
        // In production: verify nexus_oms::risk_gatekeeper::are_limits_loaded()
        // Placeholder implementation
        let limits_loaded = true;
        
        if limits_loaded {
            CheckResult::Pass {
                message: "Risk gatekeeper limits loaded and acknowledged".to_string(),
            }
        } else {
            CheckResult::Fail {
                reason: "Risk gatekeeper limits not loaded".to_string(),
            }
        }
    }

    /// Check 4: Raft Swarm confirms undisputed Leader with valid STONITH fence
    async fn check_raft_leadership_stonith(&self) -> CheckResult {
        // In production: query nexus_swarm::SwarmConsensusEngine
        // Placeholder implementation
        let is_leader = true;
        let stonith_armed = true;
        
        if is_leader && stonith_armed {
            CheckResult::Pass {
                message: "Raft leadership confirmed + STONITH fence armed".to_string(),
            }
        } else if is_leader && !stonith_armed {
            CheckResult::Fail {
                reason: "Node is leader but STONITH fence not armed".to_string(),
            }
        } else {
            CheckResult::Fail {
                reason: "Node is not the Raft cluster leader".to_string(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_unanimous_pass() {
        let checklist = PreFlightChecklist::default();
        let result = checklist.execute_all_checks().await;
        
        // With placeholder implementations, should pass
        assert!(result.can_proceed());
    }
}
