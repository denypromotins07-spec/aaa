//! OMS State Replicator for NEXUS-OMEGA Swarm
//! 
//! Replicates Order Management System state from leader to followers
//! via the Raft log, ensuring all nodes maintain identical state.

use crate::{ConsensusError, ConsensusResult, LogEntry, LogEntryType, LogIndex, LockFreeRaftLog};
use std::sync::Arc;

/// Represents the current state of the OMS
#[derive(Debug, Clone, Default)]
pub struct OMSState {
    /// Map of order_id -> order state (serialized)
    pub orders: std::collections::HashMap<String, Vec<u8>>,
    /// Map of symbol -> net position
    pub positions: std::collections::HashMap<String, i64>,
    /// Current PnL
    pub pnl: i64,
    /// Last applied log index
    pub last_applied_index: LogIndex,
}

/// Replicates OMS state changes via the Raft log
pub struct OMSStateReplicator {
    raft_log: Arc<LockFreeRaftLog>,
    local_state: Arc<std::sync::RwLock<OMSState>>,
    node_id: u64,
}

unsafe impl Send for OMSStateReplicator {}
unsafe impl Sync for OMSStateReplicator {}

impl OMSStateReplicator {
    pub fn new(raft_log: Arc<LockFreeRaftLog>, node_id: u64) -> Self {
        Self {
            raft_log,
            local_state: Arc::new(std::sync::RwLock::new(OMSState::default())),
            node_id,
        }
    }

    /// Record an order submission in the Raft log
    pub fn record_order_submission(&self, order_id: String, order_data: Vec<u8>) -> ConsensusResult<LogIndex> {
        let entry = LogEntry::new(
            0, // Index will be assigned by append
            1, // Term should be set by caller
            LogEntryType::OrderSubmission { order_id },
            order_data,
        );
        
        self.raft_log.append(entry)
    }

    /// Record an order cancellation in the Raft log
    pub fn record_order_cancellation(&self, order_id: String) -> ConsensusResult<LogIndex> {
        let entry = LogEntry::new(
            0,
            1,
            LogEntryType::OrderCancellation { order_id },
            vec![],
        );
        
        self.raft_log.append(entry)
    }

    /// Record a position update in the Raft log
    pub fn record_position_update(&self, symbol: String, delta: i64) -> ConsensusResult<LogIndex> {
        let data = delta.to_le_bytes().to_vec();
        let entry = LogEntry::new(
            0,
            1,
            LogEntryType::PositionUpdate { symbol, delta },
            data,
        );
        
        self.raft_log.append(entry)
    }

    /// Record a PnL update in the Raft log
    pub fn record_pnl_update(&self, delta: i64) -> ConsensusResult<LogIndex> {
        let data = delta.to_le_bytes().to_vec();
        let entry = LogEntry::new(
            0,
            1,
            LogEntryType::PnLUpdate { delta },
            data,
        );
        
        self.raft_log.append(entry)
    }

    /// Apply a log entry to the local state (called by followers)
    pub fn apply_entry(&self, entry: &LogEntry) -> ConsensusResult<()> {
        if !entry.verify_integrity() {
            return Err(ConsensusError::LogCorruption {
                index: entry.index,
                reason: "CRC32 verification failed on apply".to_string(),
            });
        }

        let mut state = self.local_state.write().unwrap();

        match &entry.entry_type {
            LogEntryType::NoOp => {}
            
            LogEntryType::OrderSubmission { order_id } => {
                state.orders.insert(order_id.clone(), entry.data.clone());
            }
            
            LogEntryType::OrderCancellation { order_id } => {
                state.orders.remove(order_id);
            }
            
            LogEntryType::PositionUpdate { symbol, delta } => {
                let current = state.positions.entry(symbol.clone()).or_insert(0);
                *current += delta;
            }
            
            LogEntryType::PnLUpdate { delta } => {
                state.pnl += delta;
            }
            
            LogEntryType::StateSnapshot { .. } => {
                // Snapshot application handled separately
            }
        }

        state.last_applied_index = entry.index;
        Ok(())
    }

    /// Apply entries from a range (used during log replay)
    pub fn apply_range(&self, start: LogIndex, end: LogIndex) -> ConsensusResult<()> {
        for i in start..end {
            if let Some(entry) = self.raft_log.get(i) {
                self.apply_entry(&entry)?;
            }
        }
        Ok(())
    }

    /// Get the current local state
    pub fn get_state(&self) -> OMSState {
        self.local_state.read().unwrap().clone()
    }

    /// Get a specific order's data
    pub fn get_order(&self, order_id: &str) -> Option<Vec<u8>> {
        let state = self.local_state.read().unwrap();
        state.orders.get(order_id).cloned()
    }

    /// Get the current position for a symbol
    pub fn get_position(&self, symbol: &str) -> i64 {
        let state = self.local_state.read().unwrap();
        *state.positions.get(symbol).unwrap_or(&0)
    }

    /// Get the current PnL
    pub fn get_pnl(&self) -> i64 {
        let state = self.local_state.read().unwrap();
        state.pnl
    }

    /// Get the last applied log index
    pub fn get_last_applied_index(&self) -> LogIndex {
        let state = self.local_state.read().unwrap();
        state.last_applied_index
    }

    /// Reset state from a snapshot
    pub fn apply_snapshot(&self, snapshot: OMSState) -> ConsensusResult<()> {
        let mut state = self.local_state.write().unwrap();
        *state = snapshot;
        Ok(())
    }

    /// Check if replicator is caught up with the log
    pub fn is_caught_up(&self) -> bool {
        let state = self.local_state.read().unwrap();
        let commit_index = self.raft_log.get_commit_index();
        state.last_applied_index >= commit_index
    }

    /// Get replication lag in entries
    pub fn get_lag(&self) -> LogIndex {
        let state = self.local_state.read().unwrap();
        let commit_index = self.raft_log.get_commit_index();
        commit_index.saturating_sub(state.last_applied_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SwarmConfig;

    #[test]
    fn test_order_replication() {
        let config = SwarmConfig::default();
        let raft_log = Arc::new(LockFreeRaftLog::new(&config));
        let replicator = OMSStateReplicator::new(Arc::clone(&raft_log), 1);

        // Record an order
        let order_id = "order_123".to_string();
        let order_data = vec![1, 2, 3, 4];
        
        let index = replicator.record_order_submission(order_id.clone(), order_data.clone()).unwrap();
        
        // Apply the entry
        if let Some(entry) = raft_log.get(index) {
            replicator.apply_entry(&entry).unwrap();
        }
        
        // Verify state
        assert_eq!(replicator.get_order("order_123"), Some(order_data));
    }

    #[test]
    fn test_position_replication() {
        let config = SwarmConfig::default();
        let raft_log = Arc::new(LockFreeRaftLog::new(&config));
        let replicator = OMSStateReplicator::new(Arc::clone(&raft_log), 1);

        // Record position updates
        replicator.record_position_update("BTCUSD".to_string(), 100).unwrap();
        replicator.record_position_update("BTCUSD".to_string(), -50).unwrap();
        
        // Apply entries
        let commit_index = raft_log.get_commit_index();
        replicator.apply_range(0, commit_index).unwrap();
        
        // Verify net position
        assert_eq!(replicator.get_position("BTCUSD"), 50);
    }

    #[test]
    fn test_pnl_replication() {
        let config = SwarmConfig::default();
        let raft_log = Arc::new(LockFreeRaftLog::new(&config));
        let replicator = OMSStateReplicator::new(Arc::clone(&raft_log), 1);

        // Record PnL updates
        replicator.record_pnl_update(1000).unwrap();
        replicator.record_pnl_update(-500).unwrap();
        replicator.record_pnl_update(200).unwrap();
        
        // Apply entries
        let commit_index = raft_log.get_commit_index();
        replicator.apply_range(0, commit_index).unwrap();
        
        // Verify PnL
        assert_eq!(replicator.get_pnl(), 700);
    }
}
