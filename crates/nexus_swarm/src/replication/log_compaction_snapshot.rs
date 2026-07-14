//! Log Compaction and Snapshotting for NEXUS-OMEGA Swarm
//! 
//! Implements background snapshotting to compress the Raft log
//! and prevent unbounded memory growth.

use crate::{ConsensusError, ConsensusResult, LogEntry, LogEntryType, LogIndex, LockFreeRaftLog, OMSState, SwarmConfig};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A compressed snapshot of the OMS state at a specific log index
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    /// Log index at which this snapshot was taken
    pub index: LogIndex,
    /// Term at the time of snapshot
    pub term: u64,
    /// Timestamp when snapshot was created
    pub timestamp_ns: u64,
    /// Serialized OMS state
    pub oms_state: OMSState,
    /// CRC32 checksum of the snapshot data
    pub crc32: u32,
}

impl StateSnapshot {
    pub fn new(index: LogIndex, term: u64, oms_state: OMSState) -> Self {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        // Create a deterministic serialization for CRC calculation
        let mut hasher_data = Vec::new();
        hasher_data.extend_from_slice(&index.to_le_bytes());
        hasher_data.extend_from_slice(&term.to_le_bytes());
        hasher_data.extend_from_slice(&(oms_state.pnl.to_le_bytes()));
        
        for (key, value) in &oms_state.orders {
            hasher_data.extend_from_slice(key.as_bytes());
            hasher_data.extend_from_slice(value);
        }
        
        for (key, value) in &oms_state.positions {
            hasher_data.extend_from_slice(key.as_bytes());
            hasher_data.extend_from_slice(&value.to_le_bytes());
        }
        
        let crc32 = crc32fast::hash(&hasher_data);
        
        Self {
            index,
            term,
            timestamp_ns,
            oms_state,
            crc32,
        }
    }

    pub fn verify_integrity(&self) -> bool {
        let recomputed = {
            let mut hasher_data = Vec::new();
            hasher_data.extend_from_slice(&self.index.to_le_bytes());
            hasher_data.extend_from_slice(&self.term.to_le_bytes());
            hasher_data.extend_from_slice(&(self.oms_state.pnl.to_le_bytes()));
            
            for (key, value) in &self.oms_state.orders {
                hasher_data.extend_from_slice(key.as_bytes());
                hasher_data.extend_from_slice(value);
            }
            
            for (key, value) in &self.oms_state.positions {
                hasher_data.extend_from_slice(key.as_bytes());
                hasher_data.extend_from_slice(&value.to_le_bytes());
            }
            
            crc32fast::hash(&hasher_data)
        };
        
        self.crc32 == recomputed
    }
}

/// Handles log compaction and snapshot creation
pub struct LogCompactionSnapshot {
    raft_log: Arc<LockFreeRaftLog>,
    config: SwarmConfig,
    last_snapshot_index: std::sync::atomic::AtomicU64,
    last_snapshot_time: std::sync::Mutex<Option<Instant>>,
    pending_snapshot: std::sync::Mutex<Option<StateSnapshot>>,
}

unsafe impl Send for LogCompactionSnapshot {}
unsafe impl Sync for LogCompactionSnapshot {}

impl LogCompactionSnapshot {
    pub fn new(raft_log: Arc<LockFreeRaftLog>, config: SwarmConfig) -> Self {
        Self {
            raft_log,
            config,
            last_snapshot_index: std::sync::atomic::AtomicU64::new(0),
            last_snapshot_time: std::sync::Mutex::new(None),
            pending_snapshot: std::sync::Mutex::new(None),
        }
    }

    /// Check if compaction is needed based on log size or time interval
    pub fn needs_compaction(&self) -> bool {
        // Check by entry count
        if self.raft_log.needs_compaction(self.config.max_log_entries_before_compaction) {
            return true;
        }
        
        // Check by time interval
        let last_snapshot = self.last_snapshot_time.lock().unwrap();
        if let Some(last_time) = *last_snapshot {
            if last_time.elapsed() >= Duration::from_millis(self.config.snapshot_interval_ms) {
                return true;
            }
        }
        
        false
    }

    /// Create a snapshot of the current state
    pub fn create_snapshot(&self, oms_state: OMSState) -> ConsensusResult<StateSnapshot> {
        let commit_index = self.raft_log.get_commit_index();
        
        // Get term at commit index
        let term = self.raft_log.get_term_at(commit_index).unwrap_or(0);
        
        let snapshot = StateSnapshot::new(commit_index, term, oms_state);
        
        if !snapshot.verify_integrity() {
            return Err(ConsensusError::LogCorruption {
                index: commit_index,
                reason: "Snapshot CRC verification failed".to_string(),
            });
        }
        
        Ok(snapshot)
    }

    /// Apply a snapshot and truncate the log
    pub fn apply_snapshot(&self, snapshot: StateSnapshot) -> ConsensusResult<()> {
        if !snapshot.verify_integrity() {
            return Err(ConsensusError::LogCorruption {
                index: snapshot.index,
                reason: "Snapshot integrity check failed".to_string(),
            });
        }

        // Truncate log before snapshot index
        self.raft_log.truncate_before(snapshot.index);
        
        // Update last snapshot tracking
        self.last_snapshot_index.store(snapshot.index, std::sync::atomic::Ordering::Release);
        *self.last_snapshot_time.lock().unwrap() = Some(Instant::now());
        
        tracing::info!(
            "Applied snapshot at index {} with term {}",
            snapshot.index,
            snapshot.term
        );
        
        Ok(())
    }

    /// Get the last snapshot index
    pub fn get_last_snapshot_index(&self) -> LogIndex {
        self.last_snapshot_index.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Run compaction in background
    pub async fn run_background_compaction(
        self: Arc<Self>,
        oms_state_provider: Arc<dyn Fn() -> OMSState + Send + Sync>,
    ) {
        loop {
            tokio::time::sleep(Duration::from_millis(
                self.config.snapshot_interval_ms / 2
            )).await;
            
            if self.needs_compaction() {
                let oms_state = oms_state_provider();
                
                match self.create_snapshot(oms_state) {
                    Ok(snapshot) => {
                        if let Err(e) = self.apply_snapshot(snapshot) {
                            tracing::error!("Failed to apply snapshot: {:?}", e);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to create snapshot: {:?}", e);
                    }
                }
            }
        }
    }

    /// Compact the log up to a specific index
    pub fn compact_up_to(&self, index: LogIndex, oms_state: OMSState) -> ConsensusResult<()> {
        let commit_index = self.raft_log.get_commit_index();
        
        if index > commit_index {
            return Err(ConsensusError::LogCorruption {
                index,
                reason: "Cannot compact beyond commit index".to_string(),
            });
        }
        
        let snapshot = StateSnapshot::new(index, 0, oms_state);
        self.apply_snapshot(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_creation() {
        let config = SwarmConfig::default();
        let raft_log = Arc::new(LockFreeRaftLog::new(&config));
        let compactor = LogCompactionSnapshot::new(Arc::clone(&raft_log), config);
        
        let mut oms_state = OMSState::default();
        oms_state.pnl = 1000;
        oms_state.positions.insert("BTCUSD".to_string(), 50);
        
        let snapshot = compactor.create_snapshot(oms_state.clone()).unwrap();
        
        assert!(snapshot.verify_integrity());
        assert_eq!(snapshot.oms_state.pnl, 1000);
    }

    #[test]
    fn test_snapshot_integrity_corruption() {
        let config = SwarmConfig::default();
        let raft_log = Arc::new(LockFreeRaftLog::new(&config));
        let compactor = LogCompactionSnapshot::new(Arc::clone(&raft_log), config);
        
        let mut oms_state = OMSState::default();
        oms_state.pnl = 1000;
        
        let mut snapshot = compactor.create_snapshot(oms_state).unwrap();
        
        // Corrupt the snapshot
        snapshot.oms_state.pnl = 9999;
        
        assert!(!snapshot.verify_integrity());
    }

    #[test]
    fn test_snapshot_application() {
        let config = SwarmConfig::default();
        let raft_log = Arc::new(LockFreeRaftLog::new(&config));
        let compactor = LogCompactionSnapshot::new(Arc::clone(&raft_log), config.clone());
        
        // First write some entries
        use crate::LogEntryType;
        for i in 0..100 {
            let entry = LogEntry::new(i, 1, LogEntryType::NoOp, vec![i as u8]);
            raft_log.append(entry).unwrap();
        }
        raft_log.commit(100).unwrap();
        
        assert_eq!(raft_log.get_commit_index(), 100);
        
        // Create and apply snapshot
        let oms_state = OMSState::default();
        let snapshot = compactor.create_snapshot(oms_state).unwrap();
        compactor.apply_snapshot(snapshot).unwrap();
        
        // Log should be truncated
        assert_eq!(compactor.get_last_snapshot_index(), 100);
    }
}
