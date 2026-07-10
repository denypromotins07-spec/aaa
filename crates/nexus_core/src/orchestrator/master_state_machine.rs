// STAGE 25: CHAPTER 4 - MASTER STATE MACHINE
/// Governs global lifecycle of NEXUS-OMEGA with strict priority hierarchy
/// States: Initializing, DataSyncing, PaperTrading, LiveShadow, LiveExecution, Halt, CatastrophicFailure

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// System states with strict priority ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum SystemState {
    CatastrophicFailure = 0, // Highest priority - immediate halt
    Halt = 1,                // Second highest - manual intervention required
    Initializing = 2,        // Boot sequence
    DataSyncing = 3,         // Synchronizing market data
    PaperTrading = 4,        // Simulation mode
    LiveShadow = 5,          // Live data, no execution
    LiveExecution = 6,       // Full live trading
}

impl SystemState {
    /// Get state name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            SystemState::CatastrophicFailure => "CatastrophicFailure",
            SystemState::Halt => "Halt",
            SystemState::Initializing => "Initializing",
            SystemState::DataSyncing => "DataSyncing",
            SystemState::PaperTrading => "PaperTrading",
            SystemState::LiveShadow => "LiveShadow",
            SystemState::LiveExecution => "LiveExecution",
        }
    }

    /// Check if state allows trading
    pub fn can_trade(&self) -> bool {
        matches!(self, SystemState::LiveExecution | SystemState::PaperTrading)
    }

    /// Check if state is terminal (requires manual reset)
    pub fn is_terminal(&self) -> bool {
        matches!(self, SystemState::CatastrophicFailure | SystemState::Halt)
    }
}

/// State transition commands
#[derive(Debug, Clone, PartialEq)]
pub enum StateCommand {
    Initialize,
    StartDataSync,
    EnterPaperTrading,
    EnterLiveShadow,
    EnterLiveExecution,
    Halt(String), // With reason
    CatastrophicFailure(String),
    PromoteToLeader, // From Stage 22 Swarm
    AlignmentVeto(String), // From Stage 24 Super-Ego
}

/// Transition result
#[derive(Debug, Clone)]
pub struct TransitionResult {
    pub from_state: SystemState,
    pub to_state: SystemState,
    pub command: StateCommand,
    pub success: bool,
    pub rejected_reason: Option<String>,
    pub timestamp_ns: u64,
}

/// Master State Machine state
pub struct MasterStateMachineState {
    pub current_state: AtomicU8,
    pub transition_count: AtomicU64,
    pub last_transition_time_ns: AtomicU64,
    pub is_locked: AtomicBool, // Prevents concurrent transitions
}

impl Default for MasterStateMachineState {
    fn default() -> Self {
        Self {
            current_state: AtomicU8::new(SystemState::Initializing as u8),
            transition_count: AtomicU64::new(0),
            last_transition_time_ns: AtomicU64::new(0),
            is_locked: AtomicBool::new(false),
        }
    }
}

/// Master State Machine
/// Implements strict priority hierarchy for state transitions
pub struct MasterStateMachine {
    state: std::sync::Arc<MasterStateMachineState>,
    transition_log: std::sync::Mutex<Vec<TransitionResult>>,
    chaos_mode_flag: AtomicBool,
}

impl MasterStateMachine {
    pub fn new() -> Self {
        Self {
            state: std::sync::Arc::new(MasterStateMachineState::default()),
            transition_log: std::sync::Mutex::new(Vec::new()),
            chaos_mode_flag: AtomicBool::new(false),
        }
    }

    /// Activate chaos mode (for testing)
    pub fn activate_chaos_mode(&self) {
        self.chaos_mode_flag.store(true, Ordering::SeqCst);
    }

    /// Deactivate chaos mode
    pub fn deactivate_chaos_mode(&self) {
        self.chaos_mode_flag.store(false, Ordering::SeqCst);
    }

    /// Get current system state
    pub fn get_current_state(&self) -> SystemState {
        let state_u8 = self.state.current_state.load(Ordering::SeqCst);
        // Safe conversion since we control all state values
        match state_u8 {
            0 => SystemState::CatastrophicFailure,
            1 => SystemState::Halt,
            2 => SystemState::Initializing,
            3 => SystemState::DataSyncing,
            4 => SystemState::PaperTrading,
            5 => SystemState::LiveShadow,
            6 => SystemState::LiveExecution,
            _ => SystemState::Halt, // Default to safe state
        }
    }

    /// Process a state transition command
    /// Returns TransitionResult with success/failure information
    pub fn process_command(&self, command: StateCommand) -> TransitionResult {
        let now = Instant::now();
        let now_ns = now.duration_since(Instant::EPOCH).as_nanos() as u64;

        // Try to acquire lock (non-blocking)
        if self.state.is_locked.swap(true, Ordering::SeqCst) {
            // Another transition in progress
            return TransitionResult {
                from_state: self.get_current_state(),
                to_state: self.get_current_state(),
                command: command.clone(),
                success: false,
                rejected_reason: Some("Concurrent transition in progress".to_string()),
                timestamp_ns: now_ns,
            };
        }

        let from_state = self.get_current_state();
        let (to_state, success, rejected_reason) = self.evaluate_transition(from_state, &command);

        if success && to_state != from_state {
            // Execute transition
            self.state.current_state.store(to_state as u8, Ordering::SeqCst);
            self.state.transition_count.fetch_add(1, Ordering::Relaxed);
            self.state.last_transition_time_ns.store(now_ns, Ordering::Relaxed);
        }

        // Release lock
        self.state.is_locked.store(false, Ordering::SeqCst);

        let result = TransitionResult {
            from_state,
            to_state,
            command: command.clone(),
            success,
            rejected_reason,
            timestamp_ns: now_ns,
        };

        // Log transition
        let mut log = self.transition_log.lock().unwrap();
        log.push(result.clone());

        result
    }

    /// Evaluate if a transition is allowed based on priority hierarchy
    fn evaluate_transition(
        &self,
        from: SystemState,
        command: &StateCommand,
    ) -> (SystemState, bool, Option<String>) {
        // CRITICAL: Priority hierarchy prevents undefined behavior
        // Higher priority commands always override lower priority states

        match command {
            StateCommand::CatastrophicFailure(reason) => {
                // Always allowed - highest priority
                (SystemState::CatastrophicFailure, true, None)
            }
            StateCommand::Halt(reason) => {
                // Always allowed unless already in CatastrophicFailure
                if from == SystemState::CatastrophicFailure {
                    (from, false, Some("Cannot halt from CatastrophicFailure".to_string()))
                } else {
                    (SystemState::Halt, true, None)
                }
            }
            StateCommand::AlignmentVeto(reason) => {
                // Stage 24 Super-Ego veto - forces Halt
                (SystemState::Halt, true, None)
            }
            StateCommand::Initialize => {
                // Only allowed from terminal states
                if from.is_terminal() {
                    (SystemState::Initializing, true, None)
                } else {
                    (from, false, Some("Can only initialize from terminal state".to_string()))
                }
            }
            StateCommand::StartDataSync => {
                // Only from Initializing
                if from == SystemState::Initializing {
                    (SystemState::DataSyncing, true, None)
                } else {
                    (from, false, Some("Can only start data sync from Initializing".to_string()))
                }
            }
            StateCommand::EnterPaperTrading => {
                // From DataSyncing or LiveShadow
                if from == SystemState::DataSyncing || from == SystemState::LiveShadow {
                    (SystemState::PaperTrading, true, None)
                } else {
                    (from, false, Some("Invalid source state for PaperTrading".to_string()))
                }
            }
            StateCommand::EnterLiveShadow => {
                // From DataSyncing or PaperTrading
                if from == SystemState::DataSyncing || from == SystemState::PaperTrading {
                    (SystemState::LiveShadow, true, None)
                } else {
                    (from, false, Some("Invalid source state for LiveShadow".to_string()))
                }
            }
            StateCommand::EnterLiveExecution => {
                // Only from LiveShadow (safe progression)
                if from == SystemState::LiveShadow {
                    (SystemState::LiveExecution, true, None)
                } else {
                    (from, false, Some("Can only enter LiveExecution from LiveShadow".to_string()))
                }
            }
            StateCommand::PromoteToLeader => {
                // Stage 22 Swarm promotion - only if not in terminal state
                if from.is_terminal() {
                    (from, false, Some("Cannot promote to leader from terminal state".to_string()))
                } else {
                    // Stay in current state but mark as leader (handled externally)
                    (from, true, None)
                }
            }
        }
    }

    /// Check if system can trade
    pub fn can_trade(&self) -> bool {
        self.get_current_state().can_trade()
    }

    /// Check if system is in terminal state
    pub fn is_terminal(&self) -> bool {
        self.get_current_state().is_terminal()
    }

    /// Get transition statistics
    pub fn get_stats(&self) -> StateMachineStats {
        StateMachineStats {
            current_state: self.get_current_state(),
            transition_count: self.state.transition_count.load(Ordering::Relaxed),
            last_transition_time_ns: self.state.last_transition_time_ns.load(Ordering::Relaxed),
            is_locked: self.state.is_locked.load(Ordering::Relaxed),
            chaos_mode: self.chaos_mode_flag.load(Ordering::SeqCst),
        }
    }

    /// Get recent transition log
    pub fn get_transition_log(&self, limit: usize) -> Vec<TransitionResult> {
        let log = self.transition_log.lock().unwrap();
        log.iter().rev().take(limit).cloned().collect()
    }

    /// Force reset to Initializing (only for testing)
    pub fn force_reset(&self) {
        if self.chaos_mode_flag.load(Ordering::SeqCst) {
            self.state.current_state.store(SystemState::Initializing as u8, Ordering::SeqCst);
            self.state.transition_count.store(0, Ordering::Relaxed);
            self.transition_log.lock().unwrap().clear();
        }
    }
}

/// State machine statistics
#[derive(Debug, Clone)]
pub struct StateMachineStats {
    pub current_state: SystemState,
    pub transition_count: u64,
    pub last_transition_time_ns: u64,
    pub is_locked: bool,
    pub chaos_mode: bool,
}

impl Default for MasterStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_state_progression() {
        let sm = MasterStateMachine::new();
        
        assert_eq!(sm.get_current_state(), SystemState::Initializing);

        // Initialize -> DataSyncing
        let result = sm.process_command(StateCommand::StartDataSync);
        assert!(result.success);
        assert_eq!(sm.get_current_state(), SystemState::DataSyncing);

        // DataSyncing -> LiveShadow
        let result = sm.process_command(StateCommand::EnterLiveShadow);
        assert!(result.success);
        assert_eq!(sm.get_current_state(), SystemState::LiveShadow);

        // LiveShadow -> LiveExecution
        let result = sm.process_command(StateCommand::EnterLiveExecution);
        assert!(result.success);
        assert_eq!(sm.get_current_state(), SystemState::LiveExecution);
        assert!(sm.can_trade());
    }

    #[test]
    fn test_halt_priority() {
        let sm = MasterStateMachine::new();
        
        // Go to LiveExecution
        sm.process_command(StateCommand::StartDataSync);
        sm.process_command(StateCommand::EnterLiveShadow);
        sm.process_command(StateCommand::EnterLiveExecution);
        
        // Halt should work from any state
        let result = sm.process_command(StateCommand::Halt("Test".to_string()));
        assert!(result.success);
        assert_eq!(sm.get_current_state(), SystemState::Halt);
        assert!(!sm.can_trade());
    }

    #[test]
    fn test_catastrophic_failure_priority() {
        let sm = MasterStateMachine::new();
        
        // Even from Halt, CatastrophicFailure takes precedence
        sm.process_command(StateCommand::Halt("Test".to_string()));
        let result = sm.process_command(StateCommand::CatastrophicFailure("Critical".to_string()));
        assert!(result.success);
        assert_eq!(sm.get_current_state(), SystemState::CatastrophicFailure);
    }

    #[test]
    fn test_invalid_transition() {
        let sm = MasterStateMachine::new();
        
        // Cannot go directly from Initializing to LiveExecution
        let result = sm.process_command(StateCommand::EnterLiveExecution);
        assert!(!result.success);
        assert!(result.rejected_reason.is_some());
        assert_eq!(sm.get_current_state(), SystemState::Initializing);
    }
}
