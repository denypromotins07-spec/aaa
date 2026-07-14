//! Shadow to Live Promoter
//! Orchestrates the transition from ShadowMode to LiveTrading

use std::sync::Arc;
use tracing::{info, warn, error};

use crate::lifecycle::global_fsm::{GlobalLifecycleFSM, OrchestratorState};
use crate::preflight::unanimous_checklist::{PreFlightChecklist, ChecklistResult};
use crate::preflight::live_fire_atomic_gate::{LiveFireAtomicGate, LiveFireGateError};

/// Error types for promoter operations
#[derive(Debug, thiserror::Error)]
pub enum PromoterError {
    #[error("Not in ShadowMode - current state: {0}")]
    NotInShadowMode(String),
    #[error("Pre-flight checklist failed: {0}")]
    ChecklistFailed(String),
    #[error("Failed to enable live gate: {0}")]
    GateEnableFailed(LiveFireGateError),
    #[error("FSM transition failed")]
    TransitionFailed,
    #[error("System in catastrophic state")]
    CatastrophicState,
}

/// Shadow to Live Promoter
/// 
/// Manages the safe transition from ShadowMode to LiveTrading:
/// 1. Verify current state is ShadowMode
/// 2. Execute pre-flight checklist
/// 3. Enable live fire atomic gate
/// 4. Transition FSM to LiveTrading
pub struct ShadowToLivePromoter {
    fsm: GlobalLifecycleFSM,
    checklist: PreFlightChecklist,
    live_gate: Arc<LiveFireAtomicGate>,
}

impl ShadowToLivePromoter {
    pub fn new(
        fsm: GlobalLifecycleFSM,
        checklist: PreFlightChecklist,
        live_gate: Arc<LiveFireAtomicGate>,
    ) -> Self {
        Self {
            fsm,
            checklist,
            live_gate,
        }
    }

    /// Execute the full promotion sequence
    pub async fn promote_to_live(&self) -> Result<(), PromoterError> {
        info!("=== SHADOW TO LIVE PROMOTION INITIATED ===");

        // Step 1: Verify we're in ShadowMode
        let current_state = self.fsm.current_state();
        if current_state != OrchestratorState::ShadowMode {
            error!("Cannot promote: not in ShadowMode (current: {})", current_state);
            return Err(PromoterError::NotInShadowMode(current_state.to_string()));
        }
        info!("[PROMOTE] Verified: State is ShadowMode");

        // Step 2: Transition to PreFlightChecks
        if !self.fsm.transition(OrchestratorState::ShadowMode, OrchestratorState::PreFlightChecks) {
            error!("Failed to transition to PreFlightChecks");
            return Err(PromoterError::TransitionFailed);
        }
        info!("[PROMOTE] FSM: ShadowMode -> PreFlightChecks");

        // Step 3: Execute pre-flight checklist
        info!("[PROMOTE] Executing pre-flight checklist...");
        let checklist_result = self.checklist.execute_all_checks().await;

        match &checklist_result {
            ChecklistResult::UnanimousPass { .. } => {
                info!("[PROMOTE] Pre-flight: UNANIMOUS PASS");
            }
            ChecklistResult::PassWithWarnings { warnings, .. } => {
                warn!("[PROMOTE] Pre-flight: PASS WITH WARNINGS");
                for w in warnings {
                    warn!("[PROMOTE]   - {}", w);
                }
            }
            ChecklistResult::CriticalFailure { failures } => {
                error!("[PROMOTE] Pre-flight: CRITICAL FAILURE");
                for f in failures {
                    error!("[PROMOTE]   - {:?}", f);
                }
                
                // Transition back to ShadowMode on failure
                self.fsm.transition(
                    OrchestratorState::PreFlightChecks,
                    OrchestratorState::ShadowMode,
                );
                
                return Err(PromoterError::ChecklistFailed(
                    format!("{} critical failure(s)", failures.len())
                ));
            }
        }

        // Step 4: Enable live fire atomic gate
        info!("[PROMOTE] Enabling live fire atomic gate...");
        match self.live_gate.try_promote_to_live(checklist_result) {
            Ok(token) => {
                info!("[PROMOTE] Live gate enabled - Epoch: {}", token.epoch);
            }
            Err(err) => {
                error!("[PROMOTE] Failed to enable live gate: {}", err);
                
                // Transition back to ShadowMode
                self.fsm.transition(
                    OrchestratorState::PreFlightChecks,
                    OrchestratorState::ShadowMode,
                );
                
                return Err(PromoterError::GateEnableFailed(err));
            }
        }

        // Step 5: Transition to LiveTrading
        if !self.fsm.transition(OrchestratorState::PreFlightChecks, OrchestratorState::LiveTrading) {
            error!("[PROMOTE] Failed to transition to LiveTrading");
            
            // Critical: disable live gate if transition fails
            self.live_gate.disable_live();
            
            return Err(PromoterError::TransitionFailed);
        }
        info!("[PROMOTE] FSM: PreFlightChecks -> LiveTrading");

        info!("=== SHADOW TO LIVE PROMOTION COMPLETE ===");
        info!("=== NEXUS-OMEGA IS NOW LIVE TRADING ===");

        Ok(())
    }

    /// Abort promotion and return to ShadowMode
    pub fn abort_promotion(&self, reason: &str) {
        warn!("Aborting promotion: {}", reason);
        
        // Disable live gate if it was enabled
        self.live_gate.disable_live();
        
        // Transition back to ShadowMode
        self.fsm.transition(
            OrchestratorState::PreFlightChecks,
            OrchestratorState::ShadowMode,
        );
        
        info!("Promotion aborted - returned to ShadowMode");
    }

    /// Get reference to live gate for external checks
    pub fn live_gate(&self) -> &LiveFireAtomicGate {
        &self.live_gate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_promotion_flow() {
        let fsm = GlobalLifecycleFSM::new();
        let checklist = PreFlightChecklist::default();
        let live_gate = Arc::new(LiveFireAtomicGate::new());
        
        // First transition to ShadowMode (normally done by boot sequence)
        fsm.transition(OrchestratorState::Bootstrapping, OrchestratorState::ShadowMode);
        
        let promoter = ShadowToLivePromoter::new(fsm, checklist, live_gate.clone());
        
        // This would succeed with mocked dependencies
        // For unit test, verify structure compiles
        assert!(promoter.live_gate().is_live_enabled() == false);
    }
}
