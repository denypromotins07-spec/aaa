//! Eternal Archive Manager
//! 
//! Central orchestrator for managing data lifecycle across all cold storage mediums.
//! Coordinates archival operations and tracks data locations.

use thiserror::Error;

/// Storage medium types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageMedium {
    Dna,
    Holographic,
    Optical5D,
}

/// Data epoch identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpochId(pub u64);

/// Archive entry metadata
#[derive(Debug, Clone)]
pub struct ArchiveEntry {
    pub epoch_id: EpochId,
    pub medium: StorageMedium,
    pub location_id: u64,
    pub size_bytes: u64,
    pub checksum: u128,
    pub archived_timestamp_ns: u64,
    pub regime_tag: u32,
}

#[derive(Error, Debug)]
pub enum ArchiveError {
    #[error("Archive entry not found: {0}")]
    EntryNotFound(u64),
    #[error("Storage medium unavailable: {0:?}")]
    MediumUnavailable(StorageMedium),
    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: u128, actual: u128 },
    #[error("Archival failed: {0}")]
    ArchivalFailed(String),
    #[error("Buffer overflow")]
    BufferOverflow,
    #[error("Invalid epoch ID")]
    InvalidEpochId,
}

/// Pre-allocated buffer for archive entries
pub struct ArchiveCatalog {
    entries: Box<[Option<ArchiveEntry>]>,
    len: usize,
    capacity: usize,
}

impl ArchiveCatalog {
    pub fn with_capacity(capacity: usize) -> Self {
        let entries = vec![None; capacity].into_boxed_slice();
        Self { entries, len: 0, capacity }
    }

    pub fn add(&mut self, entry: ArchiveEntry) -> Result<(), ArchiveError> {
        if self.len >= self.capacity {
            return Err(ArchiveError::BufferOverflow);
        }
        
        // Find empty slot or append
        let mut found_slot = None;
        for i in 0..self.len {
            if self.entries[i].is_none() {
                found_slot = Some(i);
                break;
            }
        }
        
        let idx = found_slot.unwrap_or(self.len);
        if idx >= self.capacity {
            return Err(ArchiveError::BufferOverflow);
        }
        
        self.entries[idx] = Some(entry);
        if idx == self.len {
            self.len += 1;
        }
        
        Ok(())
    }

    pub fn get(&self, epoch_id: u64) -> Option<&ArchiveEntry> {
        for entry in self.entries.iter().flatten() {
            if entry.epoch_id.0 == epoch_id {
                return Some(entry);
            }
        }
        None
    }

    pub fn remove(&mut self, epoch_id: u64) -> Option<ArchiveEntry> {
        for entry in self.entries.iter_mut() {
            if let Some(e) = entry {
                if e.epoch_id.0 == epoch_id {
                    return entry.take();
                }
            }
        }
        None
    }

    pub fn iter(&self) -> impl Iterator<Item = &ArchiveEntry> {
        self.entries.iter().flatten()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        for i in 0..self.len {
            self.entries[i] = None;
        }
        self.len = 0;
    }
}

/// Eternal Archive Manager for coordinating all storage mediums
pub struct EternalArchiveManager {
    catalog: ArchiveCatalog,
    next_location_id: u64,
    current_epoch: EpochId,
    total_archived_bytes: u64,
}

impl EternalArchiveManager {
    /// Create a new archive manager with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            catalog: ArchiveCatalog::with_capacity(capacity),
            next_location_id: 1,
            current_epoch: EpochId(0),
            total_archived_bytes: 0,
        }
    }

    /// Register a new archival operation
    pub fn register_archival(
        &mut self,
        epoch_id: u64,
        medium: StorageMedium,
        size_bytes: u64,
        checksum: u128,
        regime_tag: u32,
    ) -> Result<u64, ArchiveError> {
        let location_id = self.next_location_id;
        self.next_location_id += 1;

        let entry = ArchiveEntry {
            epoch_id: EpochId(epoch_id),
            medium,
            location_id,
            size_bytes,
            checksum,
            archived_timestamp_ns: 0, // Would be set from system clock
            regime_tag,
        };

        self.catalog.add(entry)?;
        self.total_archived_bytes += size_bytes;

        Ok(location_id)
    }

    /// Find an entry by epoch ID
    pub fn find_by_epoch(&self, epoch_id: u64) -> Result<&ArchiveEntry, ArchiveError> {
        self.catalog.get(epoch_id)
            .ok_or_else(|| ArchiveError::EntryNotFound(epoch_id))
    }

    /// Find all entries for a specific medium
    pub fn find_by_medium(&self, medium: StorageMedium) -> Vec<&ArchiveEntry> {
        self.catalog.iter()
            .filter(|e| e.medium == medium)
            .collect()
    }

    /// Find all entries with a specific regime tag
    pub fn find_by_regime(&self, regime_tag: u32) -> Vec<&ArchiveEntry> {
        self.catalog.iter()
            .filter(|e| e.regime_tag == regime_tag)
            .collect()
    }

    /// Verify checksum of an entry
    pub fn verify_checksum(&self, epoch_id: u64, actual_checksum: u128) -> Result<bool, ArchiveError> {
        let entry = self.find_by_epoch(epoch_id)?;
        if entry.checksum != actual_checksum {
            return Err(ArchiveError::ChecksumMismatch {
                expected: entry.checksum,
                actual: actual_checksum,
            });
        }
        Ok(true)
    }

    /// Get statistics about the archive
    pub fn get_statistics(&self) -> ArchiveStatistics {
        let mut dna_count = 0u64;
        let mut holographic_count = 0u64;
        let mut optical_count = 0u64;
        let mut dna_bytes = 0u64;
        let mut holographic_bytes = 0u64;
        let mut optical_bytes = 0u64;

        for entry in self.catalog.iter() {
            match entry.medium {
                StorageMedium::Dna => {
                    dna_count += 1;
                    dna_bytes += entry.size_bytes;
                }
                StorageMedium::Holographic => {
                    holographic_count += 1;
                    holographic_bytes += entry.size_bytes;
                }
                StorageMedium::Optical5D => {
                    optical_count += 1;
                    optical_bytes += entry.size_bytes;
                }
            }
        }

        ArchiveStatistics {
            total_entries: self.catalog.len() as u64,
            total_bytes: self.total_archived_bytes,
            dna_entries: dna_count,
            holographic_entries: holographic_count,
            optical_entries: optical_count,
            dna_bytes,
            holographic_bytes,
            optical_bytes,
        }
    }

    /// Set current epoch
    pub fn set_current_epoch(&mut self, epoch_id: u64) {
        self.current_epoch = EpochId(epoch_id);
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> EpochId {
        self.current_epoch
    }

    /// Get total archived bytes
    pub fn total_archived_bytes(&self) -> u64 {
        self.total_archived_bytes
    }
}

/// Archive statistics
#[derive(Debug, Default)]
pub struct ArchiveStatistics {
    pub total_entries: u64,
    pub total_bytes: u64,
    pub dna_entries: u64,
    pub holographic_entries: u64,
    pub optical_entries: u64,
    pub dna_bytes: u64,
    pub holographic_bytes: u64,
    pub optical_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_basic() {
        let mut catalog = ArchiveCatalog::with_capacity(10);
        
        let entry = ArchiveEntry {
            epoch_id: EpochId(1),
            medium: StorageMedium::Dna,
            location_id: 1,
            size_bytes: 1000,
            checksum: 0x12345678,
            archived_timestamp_ns: 1000000,
            regime_tag: 1,
        };
        
        catalog.add(entry).unwrap();
        assert_eq!(catalog.len(), 1);
        
        let found = catalog.get(1).unwrap();
        assert_eq!(found.epoch_id.0, 1);
    }

    #[test]
    fn test_archive_manager_registration() {
        let mut manager = EternalArchiveManager::new(100);
        
        let location = manager.register_archival(
            1,
            StorageMedium::Dna,
            10000,
            0xABCD,
            42,
        ).unwrap();
        
        assert_eq!(location, 1);
        
        let entry = manager.find_by_epoch(1).unwrap();
        assert_eq!(entry.location_id, 1);
        assert_eq!(entry.size_bytes, 10000);
    }

    #[test]
    fn test_find_by_regime() {
        let mut manager = EternalArchiveManager::new(100);
        
        manager.register_archival(1, StorageMedium::Dna, 1000, 0x1, 10).unwrap();
        manager.register_archival(2, StorageMedium::Holographic, 2000, 0x2, 10).unwrap();
        manager.register_archival(3, StorageMedium::Optical5D, 3000, 0x3, 20).unwrap();
        
        let regime_10 = manager.find_by_regime(10);
        assert_eq!(regime_10.len(), 2);
        
        let regime_20 = manager.find_by_regime(20);
        assert_eq!(regime_20.len(), 1);
    }

    #[test]
    fn test_statistics() {
        let mut manager = EternalArchiveManager::new(100);
        
        manager.register_archival(1, StorageMedium::Dna, 1000, 0x1, 1).unwrap();
        manager.register_archival(2, StorageMedium::Dna, 2000, 0x2, 1).unwrap();
        manager.register_archival(3, StorageMedium::Holographic, 3000, 0x3, 1).unwrap();
        
        let stats = manager.get_statistics();
        assert_eq!(stats.total_entries, 3);
        assert_eq!(stats.dna_entries, 2);
        assert_eq!(stats.holographic_entries, 1);
        assert_eq!(stats.total_bytes, 6000);
    }
}
