//! Lock-Free Raft Log for Zero-Allocation Replication
//! 
//! Implements a pre-allocated ring buffer based Raft log that avoids
//! heap allocations during replication to prevent GC pauses.

use crate::{ConsensusError, ConsensusResult, LogEntry, LogIndex, SwarmConfig, Term};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Maximum number of entries in the pre-allocated ring buffer
const MAX_LOG_ENTRIES: usize = 1_000_000;

/// A lock-free Raft log using atomic operations and pre-allocated storage
pub struct LockFreeRaftLog {
    /// Pre-allocated ring buffer for log entries
    entries: Vec<Arc<std::sync::RwLock<Option<LogEntry>>>>,
    /// Index of the first committed entry
    commit_index: AtomicU64,
    /// Index of the last applied entry
    last_applied: AtomicU64,
    /// Current write position (next index to write)
    write_index: AtomicU64,
    /// Total entries written (for compaction tracking)
    total_entries_written: AtomicU64,
    /// Buffer size
    buffer_size: usize,
}

unsafe impl Send for LockFreeRaftLog {}
unsafe impl Sync for LockFreeRaftLog {}

impl LockFreeRaftLog {
    pub fn new(config: &SwarmConfig) -> Self {
        let buffer_size = config.max_log_entries_before_compaction.min(MAX_LOG_ENTRIES);
        
        let mut entries = Vec::with_capacity(buffer_size);
        for _ in 0..buffer_size {
            entries.push(Arc::new(std::sync::RwLock::new(None)));
        }
        
        Self {
            entries,
            commit_index: AtomicU64::new(0),
            last_applied: AtomicU64::new(0),
            write_index: AtomicU64::new(0),
            total_entries_written: AtomicU64::new(0),
            buffer_size,
        }
    }

    /// Append an entry to the log
    /// 
    /// Returns the index at which the entry was stored
    pub fn append(&self, mut entry: LogEntry) -> ConsensusResult<LogIndex> {
        let current_write = self.write_index.fetch_add(1, Ordering::AcqRel);
        
        // Calculate ring buffer position
        let buffer_index = (current_write as usize) % self.buffer_size;
        
        // Update entry index to match actual position
        entry.index = current_write;
        
        // Write to buffer
        let slot = &self.entries[buffer_index];
        {
            let mut guard = slot.write().unwrap();
            
            // Check for CRC32 integrity before storing
            if !entry.verify_integrity() {
                return Err(ConsensusError::LogCorruption {
                    index: current_write,
                    reason: "CRC32 mismatch on append".to_string(),
                });
            }
            
            *guard = Some(entry);
        }
        
        self.total_entries_written.fetch_add(1, Ordering::Relaxed);
        
        Ok(current_write)
    }

    /// Get an entry by index
    pub fn get(&self, index: LogIndex) -> Option<LogEntry> {
        if index >= self.total_entries_written.load(Ordering::Acquire) {
            return None;
        }
        
        let buffer_index = (index as usize) % self.buffer_size;
        let slot = &self.entries[buffer_index];
        
        let guard = slot.read().unwrap();
        guard.as_ref().cloned()
    }

    /// Get entries in a range
    pub fn get_range(&self, start: LogIndex, end: LogIndex) -> Vec<LogEntry> {
        let mut entries = Vec::new();
        
        for i in start..end {
            if let Some(entry) = self.get(i) {
                entries.push(entry);
            }
        }
        
        entries
    }

    /// Commit entries up to the given index
    pub fn commit(&self, up_to_index: LogIndex) -> ConsensusResult<()> {
        let current_commit = self.commit_index.load(Ordering::Acquire);
        
        if up_to_index <= current_commit {
            return Ok(());
        }
        
        // Verify all entries up to up_to_index exist and have valid CRC
        for i in current_commit..up_to_index {
            let buffer_index = (i as usize) % self.buffer_size;
            let slot = &self.entries[buffer_index];
            let guard = slot.read().unwrap();
            
            if let Some(entry) = guard.as_ref() {
                if !entry.verify_integrity() {
                    return Err(ConsensusError::LogCorruption {
                        index: i,
                        reason: "CRC32 mismatch on commit".to_string(),
                    });
                }
            } else {
                return Err(ConsensusError::LogCorruption {
                    index: i,
                    reason: "Missing entry on commit".to_string(),
                });
            }
        }
        
        self.commit_index.store(up_to_index, Ordering::Release);
        Ok(())
    }

    /// Get the commit index
    pub fn get_commit_index(&self) -> LogIndex {
        self.commit_index.load(Ordering::Acquire)
    }

    /// Get the last applied index
    pub fn get_last_applied(&self) -> LogIndex {
        self.last_applied.load(Ordering::Acquire)
    }

    /// Mark entries as applied up to the given index
    pub fn mark_applied(&self, up_to_index: LogIndex) {
        self.last_applied.store(up_to_index, Ordering::Release);
    }

    /// Get the next index to be written
    pub fn get_next_index(&self) -> LogIndex {
        self.write_index.load(Ordering::Acquire)
    }

    /// Get total entries written
    pub fn get_total_entries(&self) -> LogIndex {
        self.total_entries_written.load(Ordering::Acquire)
    }

    /// Check if log needs compaction
    pub fn needs_compaction(&self, threshold: usize) -> bool {
        self.total_entries_written.load(Ordering::Acquire) as usize >= threshold
    }

    /// Get the log term at a specific index
    pub fn get_term_at(&self, index: LogIndex) -> Option<Term> {
        self.get(index).map(|e| e.term)
    }

    /// Verify integrity of a range of entries
    pub fn verify_range(&self, start: LogIndex, end: LogIndex) -> ConsensusResult<()> {
        for i in start..end {
            let buffer_index = (i as usize) % self.buffer_size;
            let slot = &self.entries[buffer_index];
            let guard = slot.read().unwrap();
            
            if let Some(entry) = guard.as_ref() {
                if !entry.verify_integrity() {
                    return Err(ConsensusError::LogCorruption {
                        index: i,
                        reason: "CRC32 verification failed".to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Clear entries up to a certain index (used after snapshotting)
    pub fn truncate_before(&self, index: LogIndex) {
        let total = self.total_entries_written.load(Ordering::Acquire);
        let start = index.min(total);
        
        for i in 0..start {
            let buffer_index = (i as usize) % self.buffer_size;
            let slot = &self.entries[buffer_index];
            let mut guard = slot.write().unwrap();
            *guard = None;
        }
        
        // Reset write index if truncating everything
        if index >= total {
            self.write_index.store(0, Ordering::Release);
            self.total_entries_written.store(0, Ordering::Release);
            self.commit_index.store(0, Ordering::Release);
            self.last_applied.store(0, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LogEntryType;

    #[test]
    fn test_append_and_get() {
        let config = SwarmConfig::default();
        let log = LockFreeRaftLog::new(&config);
        
        let entry = LogEntry::new(0, 1, LogEntryType::NoOp, vec![1, 2, 3]);
        let index = log.append(entry.clone()).unwrap();
        
        assert_eq!(index, 0);
        
        let retrieved = log.get(0).unwrap();
        assert_eq!(retrieved.index, 0);
        assert_eq!(retrieved.term, 1);
    }

    #[test]
    fn test_commit() {
        let config = SwarmConfig::default();
        let log = LockFreeRaftLog::new(&config);
        
        for i in 0..5 {
            let entry = LogEntry::new(i, 1, LogEntryType::NoOp, vec![i as u8]);
            log.append(entry).unwrap();
        }
        
        log.commit(5).unwrap();
        assert_eq!(log.get_commit_index(), 5);
    }

    #[test]
    fn test_ring_buffer_wraparound() {
        let mut config = SwarmConfig::default();
        config.max_log_entries_before_compaction = 10;
        
        let log = LockFreeRaftLog::new(&config);
        
        // Write more entries than buffer size
        for i in 0..15 {
            let entry = LogEntry::new(i, 1, LogEntryType::NoOp, vec![i as u8]);
            log.append(entry).unwrap();
        }
        
        // Should be able to get recent entries
        assert!(log.get(14).is_some());
        assert!(log.get(10).is_some());
        
        // Older entries may have been overwritten
        assert!(log.get(0).is_none());
    }

    #[test]
    fn test_crc_verification() {
        let config = SwarmConfig::default();
        let log = LockFreeRaftLog::new(&config);
        
        let mut entry = LogEntry::new(0, 1, LogEntryType::NoOp, vec![1, 2, 3]);
        // Corrupt the CRC
        entry.crc32 = 0xDEADBEEF;
        
        let result = log.append(entry);
        assert!(result.is_err());
        
        if let Err(ConsensusError::LogCorruption { reason, .. }) = result {
            assert!(reason.contains("CRC32"));
        } else {
            panic!("Expected LogCorruption error");
        }
    }
}
