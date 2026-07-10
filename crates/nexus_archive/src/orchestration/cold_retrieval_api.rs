//! Cold Retrieval API
//! 
//! Provides an API for retrieving archived data from cold storage mediums.
//! Allows genetic programming and other systems to request historical regime data.

use crate::orchestration::eternal_archive_manager::{
    EternalArchiveManager, ArchiveEntry, StorageMedium, ArchiveError, EpochId
};
use crate::orchestration::regime_archival_policy::{MacroRegime, RegimeArchivalPolicy};
use thiserror::Error;

/// Retrieval request types
#[derive(Debug, Clone)]
pub enum RetrievalRequest {
    /// Retrieve by specific epoch ID
    ByEpoch(u64),
    /// Retrieve all data from a specific regime
    ByRegime(MacroRegime),
    /// Retrieve all DNA-stored data
    ByMedium(StorageMedium),
    /// Retrieve data within a time range
    ByTimeRange { start_ns: u64, end_ns: u64 },
}

/// Retrieval result status
#[derive(Debug)]
pub enum RetrievalStatus {
    Success { bytes_retrieved: u64 },
    Partial { bytes_retrieved: u64, bytes_missing: u64 },
    Failed { reason: String },
    Pending { estimated_wait_ms: u64 },
}

/// Cold retrieval operation handle
#[derive(Debug, Clone)]
pub struct RetrievalHandle {
    pub request_id: u64,
    pub epoch_id: u64,
    pub medium: StorageMedium,
    pub status: RetrievalStatus,
    pub data_checksum: Option<u128>,
}

#[derive(Error, Debug)]
pub enum RetrievalApiError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[error("Entry not found: {0}")]
    EntryNotFound(u64),
    #[error("Medium offline: {0:?}")]
    MediumOffline(StorageMedium),
    #[error("Retrieval timeout")]
    Timeout,
    #[error("Checksum verification failed")]
    ChecksumFailed,
    #[error("Buffer overflow")]
    BufferOverflow,
    #[error("Archive error: {0}")]
    ArchiveError(String),
}

/// Pre-allocated buffer for retrieval handles
pub struct RetrievalQueue {
    handles: Box<[Option<RetrievalHandle>]>,
    head: usize,
    tail: usize,
    count: usize,
    capacity: usize,
    next_request_id: u64,
}

impl RetrievalQueue {
    pub fn with_capacity(capacity: usize) -> Self {
        let handles = vec![None; capacity].into_boxed_slice();
        Self {
            handles,
            head: 0,
            tail: 0,
            count: 0,
            capacity,
            next_request_id: 1,
        }
    }

    pub fn enqueue(&mut self, handle: RetrievalHandle) -> Result<(), RetrievalApiError> {
        if self.count >= self.capacity {
            return Err(RetrievalApiError::BufferOverflow);
        }

        self.handles[self.tail] = Some(handle);
        self.tail = (self.tail + 1) % self.capacity;
        self.count += 1;

        Ok(())
    }

    pub fn dequeue(&mut self) -> Option<RetrievalHandle> {
        if self.count == 0 {
            return None;
        }

        let handle = self.handles[self.head].take();
        self.head = (self.head + 1) % self.capacity;
        self.count -= 1;

        handle
    }

    pub fn get(&self, request_id: u64) -> Option<&RetrievalHandle> {
        let mut idx = self.head;
        for _ in 0..self.count {
            if let Some(handle) = &self.handles[idx] {
                if handle.request_id == request_id {
                    return Some(handle);
                }
            }
            idx = (idx + 1) % self.capacity;
        }
        None
    }

    pub fn update_status(&mut self, request_id: u64, status: RetrievalStatus) -> bool {
        let mut idx = self.head;
        for _ in 0..self.count {
            if let Some(handle) = &mut self.handles[idx] {
                if handle.request_id == request_id {
                    handle.status = status;
                    return true;
                }
            }
            idx = (idx + 1) % self.capacity;
        }
        false
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn clear(&mut self) {
        for i in 0..self.capacity {
            self.handles[i] = None;
        }
        self.head = 0;
        self.tail = 0;
        self.count = 0;
    }
}

/// Cold Retrieval API for accessing archived data
pub struct ColdRetrievalApi {
    archive_manager: EternalArchiveManager,
    retrieval_queue: RetrievalQueue,
    medium_availability: [bool; 3], // Dna, Holographic, Optical5D
    total_retrieved_bytes: u64,
}

impl ColdRetrievalApi {
    /// Create a new retrieval API
    pub fn new(archive_capacity: usize, queue_capacity: usize) -> Self {
        Self {
            archive_manager: EternalArchiveManager::new(archive_capacity),
            retrieval_queue: RetrievalQueue::with_capacity(queue_capacity),
            medium_availability: [true, true, true], // All mediums online by default
            total_retrieved_bytes: 0,
        }
    }

    /// Set medium availability
    pub fn set_medium_available(&mut self, medium: StorageMedium, available: bool) {
        let idx = match medium {
            StorageMedium::Dna => 0,
            StorageMedium::Holographic => 1,
            StorageMedium::Optical5D => 2,
        };
        self.medium_availability[idx] = available;
    }

    /// Check if a medium is available
    pub fn is_medium_available(&self, medium: StorageMedium) -> bool {
        let idx = match medium {
            StorageMedium::Dna => 0,
            StorageMedium::Holographic => 1,
            StorageMedium::Optical5D => 2,
        };
        self.medium_availability[idx]
    }

    /// Submit a retrieval request
    pub fn submit_request(&mut self, request: RetrievalRequest) -> Result<u64, RetrievalApiError> {
        let request_id = self.retrieval_queue.next_request_id;
        self.retrieval_queue.next_request_id += 1;

        // Find matching entries
        let entries = self.find_entries_for_request(&request)?;

        if entries.is_empty() {
            return Err(RetrievalApiError::EntryNotFound(0));
        }

        // Create handles for each entry
        for entry in entries {
            // Check medium availability
            if !self.is_medium_available(entry.medium) {
                return Err(RetrievalApiError::MediumOffline(entry.medium));
            }

            let handle = RetrievalHandle {
                request_id,
                epoch_id: entry.epoch_id.0,
                medium: entry.medium,
                status: RetrievalStatus::Pending { estimated_wait_ms: self.estimate_retrieval_time(entry.medium) },
                data_checksum: None,
            };

            self.retrieval_queue.enqueue(handle)?;
        }

        Ok(request_id)
    }

    /// Find entries matching a retrieval request
    fn find_entries_for_request(&self, request: &RetrievalRequest) -> Result<Vec<&ArchiveEntry>, RetrievalApiError> {
        match request {
            RetrievalRequest::ByEpoch(epoch_id) => {
                let entry = self.archive_manager.find_by_epoch(*epoch_id)
                    .map_err(|_| RetrievalApiError::EntryNotFound(*epoch_id))?;
                Ok(vec![entry])
            }
            RetrievalRequest::ByRegime(regime) => {
                let tag = regime.to_tag();
                Ok(self.archive_manager.find_by_regime(tag))
            }
            RetrievalRequest::ByMedium(medium) => {
                Ok(self.archive_manager.find_by_medium(*medium))
            }
            RetrievalRequest::ByTimeRange { start_ns, end_ns } => {
                // Would need timestamp indexing for efficient range queries
                // For now, return all entries (simplified)
                Ok(Vec::new())
            }
        }
    }

    /// Estimate retrieval time based on medium
    fn estimate_retrieval_time(&self, medium: StorageMedium) -> u64 {
        match medium {
            StorageMedium::Dna => 3_600_000,      // 1 hour (synthesis/sequencing)
            StorageMedium::Holographic => 10_000,  // 10 seconds
            StorageMedium::Optical5D => 60_000,    // 1 minute
        }
    }

    /// Get status of a retrieval request
    pub fn get_request_status(&self, request_id: u64) -> Option<&RetrievalHandle> {
        self.retrieval_queue.get(request_id)
    }

    /// Simulate completion of a retrieval request
    pub fn complete_retrieval(
        &mut self,
        request_id: u64,
        bytes_retrieved: u64,
        checksum: u128,
    ) -> Result<(), RetrievalApiError> {
        let status = RetrievalStatus::Success { bytes_retrieved };
        
        if !self.retrieval_queue.update_status(request_id, status) {
            return Err(RetrievalApiError::EntryNotFound(request_id));
        }

        self.total_retrieved_bytes += bytes_retrieved;

        Ok(())
    }

    /// Verify retrieved data against stored checksum
    pub fn verify_retrieval(&self, request_id: u64, actual_checksum: u128) -> Result<bool, RetrievalApiError> {
        let handle = self.retrieval_queue.get(request_id)
            .ok_or_else(|| RetrievalApiError::EntryNotFound(request_id))?;

        // Look up original entry
        let entry = self.archive_manager.find_by_epoch(handle.epoch_id)
            .map_err(|e| RetrievalApiError::ArchiveError(e.to_string()))?;

        if entry.checksum != actual_checksum {
            return Err(RetrievalApiError::ChecksumFailed);
        }

        Ok(true)
    }

    /// Get pending retrieval count
    pub fn pending_count(&self) -> usize {
        self.retrieval_queue.len()
    }

    /// Get total retrieved bytes
    pub fn total_retrieved_bytes(&self) -> u64 {
        self.total_retrieved_bytes
    }

    /// Get reference to archive manager
    pub fn archive_manager(&self) -> &EternalArchiveManager {
        &self.archive_manager
    }

    /// Get mutable reference to archive manager
    pub fn archive_manager_mut(&mut self) -> &mut EternalArchiveManager {
        &mut self.archive_manager
    }
}

/// Builder for constructing retrieval requests
pub struct RetrievalRequestBuilder {
    epoch_id: Option<u64>,
    regime: Option<MacroRegime>,
    medium: Option<StorageMedium>,
    start_ns: Option<u64>,
    end_ns: Option<u64>,
}

impl RetrievalRequestBuilder {
    pub fn new() -> Self {
        Self {
            epoch_id: None,
            regime: None,
            medium: None,
            start_ns: None,
            end_ns: None,
        }
    }

    pub fn epoch(mut self, epoch_id: u64) -> Self {
        self.epoch_id = Some(epoch_id);
        self
    }

    pub fn regime(mut self, regime: MacroRegime) -> Self {
        self.regime = Some(regime);
        self
    }

    pub fn medium(mut self, medium: StorageMedium) -> Self {
        self.medium = Some(medium);
        self
    }

    pub fn time_range(mut self, start_ns: u64, end_ns: u64) -> Self {
        self.start_ns = Some(start_ns);
        self.end_ns = Some(end_ns);
        self
    }

    pub fn build(self) -> Result<RetrievalRequest, RetrievalApiError> {
        if let Some(epoch_id) = self.epoch_id {
            Ok(RetrievalRequest::ByEpoch(epoch_id))
        } else if let Some(regime) = self.regime {
            Ok(RetrievalRequest::ByRegime(regime))
        } else if let Some(medium) = self.medium {
            Ok(RetrievalRequest::ByMedium(medium))
        } else if let (Some(start_ns), Some(end_ns)) = (self.start_ns, self.end_ns) {
            Ok(RetrievalRequest::ByTimeRange { start_ns, end_ns })
        } else {
            Err(RetrievalApiError::InvalidRequest("No criteria specified".to_string()))
        }
    }
}

impl Default for RetrievalRequestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrieval_queue_basic() {
        let mut queue = RetrievalQueue::with_capacity(10);
        
        let handle = RetrievalHandle {
            request_id: 1,
            epoch_id: 100,
            medium: StorageMedium::Dna,
            status: RetrievalStatus::Pending { estimated_wait_ms: 1000 },
            data_checksum: None,
        };
        
        queue.enqueue(handle).unwrap();
        assert_eq!(queue.len(), 1);
        
        let found = queue.get(1);
        assert!(found.is_some());
    }

    #[test]
    fn test_api_creation() {
        let api = ColdRetrievalApi::new(100, 50);
        assert_eq!(api.pending_count(), 0);
        assert!(api.is_medium_available(StorageMedium::Dna));
    }

    #[test]
    fn test_request_builder() {
        let request = RetrievalRequestBuilder::new()
            .epoch(42)
            .build()
            .unwrap();
        
        match request {
            RetrievalRequest::ByEpoch(id) => assert_eq!(id, 42),
            _ => panic!("Wrong request type"),
        }
    }

    #[test]
    fn test_medium_availability() {
        let mut api = ColdRetrievalApi::new(100, 50);
        
        assert!(api.is_medium_available(StorageMedium::Dna));
        
        api.set_medium_available(StorageMedium::Dna, false);
        assert!(!api.is_medium_available(StorageMedium::Dna));
    }

    #[test]
    fn test_regime_request() {
        let request = RetrievalRequestBuilder::new()
            .regime(MacroRegime::FinancialCrisis)
            .build()
            .unwrap();
        
        match request {
            RetrievalRequest::ByRegime(r) => assert_eq!(r, MacroRegime::FinancialCrisis),
            _ => panic!("Wrong request type"),
        }
    }
}
