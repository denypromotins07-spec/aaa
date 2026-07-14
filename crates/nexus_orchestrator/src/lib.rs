//! NEXUS-OMEGA Master Orchestrator
//! 
//! Governs the global lifecycle of the trading system:
//! - Sequential boot topology with strict ordering
//! - Pre-flight checklist validation
//! - Live fire atomic gating
//! - Graceful shutdown and catastrophic halt

pub mod lifecycle;
pub mod preflight;

pub use lifecycle::{GlobalLifecycleFSM, OrchestratorState, SequentialBootTopology, BootError};
pub use preflight::{PreFlightChecklist, ChecklistResult, LiveFireAtomicGate, ShadowToLivePromoter};

/// Master Orchestrator - Central coordinator for all subsystems
pub struct MasterOrchestrator {
    fsm: GlobalLifecycleFSM,
    boot_topology: SequentialBootTopology,
    preflight_checklist: PreFlightChecklist,
    live_gate: LiveFireAtomicGate,
}

impl MasterOrchestrator {
    /// Create a new Master Orchestrator instance
    pub fn new() -> Self {
        let fsm = GlobalLifecycleFSM::new();
        let boot_topology = SequentialBootTopology::new(fsm.clone());
        let preflight_checklist = PreFlightChecklist::default();
        let live_gate = LiveFireAtomicGate::new();
        
        Self {
            fsm,
            boot_topology,
            preflight_checklist,
            live_gate,
        }
    }

    /// Get reference to the global FSM
    pub fn fsm(&self) -> &GlobalLifecycleFSM {
        &self.fsm
    }

    /// Get reference to the live fire gate
    pub fn live_gate(&self) -> &LiveFireAtomicGate {
        &self.live_gate
    }

    /// Execute the complete boot sequence
    pub async fn boot(&self) -> Result<(), BootError> {
        self.boot_topology.execute_boot_sequence().await
    }

    /// Promote from shadow mode to live trading
    pub async fn promote_to_live(&self) -> Result<(), crate::preflight::PromoterError> {
        use std::sync::Arc;
        
        let promoter = ShadowToLivePromoter::new(
            self.fsm.clone(),
            self.preflight_checklist.clone(),
            Arc::new(self.live_gate.clone()),
        );
        
        promoter.promote_to_live().await
    }

    /// Get current orchestrator state
    pub fn current_state(&self) -> OrchestratorState {
        self.fsm.current_state()
    }

    /// Check if currently in live trading mode
    pub fn is_live_trading(&self) -> bool {
        self.fsm.is_live_trading()
    }

    /// Initiate graceful shutdown
    pub fn shutdown(&self) {
        use lifecycle::OrchestratorState;
        
        let current = self.fsm.current_state();
        if current == OrchestratorState::LiveTrading {
            self.live_gate.disable_live();
        }
        
        self.fsm.transition(current, OrchestratorState::GracefulShutdown);
    }

    /// Force catastrophic halt
    pub fn catastrophic_halt(&self, reason: &str) {
        tracing::error!("CATASTROPHIC HALT: {}", reason);
        self.live_gate.force_lockdown(reason);
        self.fsm.force_transition(OrchestratorState::CatastrophicHalt);
    }
}

impl Default for MasterOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let orchestrator = MasterOrchestrator::new();
        assert_eq!(orchestrator.current_state(), OrchestratorState::Bootstrapping);
        assert!(!orchestrator.is_live_trading());
    }
}
