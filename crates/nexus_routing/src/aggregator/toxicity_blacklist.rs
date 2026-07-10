//! Toxicity Blacklist - VPIN-based venue filtering
//! 
//! Cross-references VPIN (Volume-Synchronized Probability of Informed Trading)
//! metrics to automatically blacklist toxic venues exhibiting informed flow.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ToxicityError {
    #[error("VPIN value out of range: {vpin}")]
    InvalidVPIN { vpin: f64 },
    #[error("Venue {venue_id} permanently blacklisted")]
    PermanentlyBlacklisted { venue_id: u32 },
    #[error("Venue {venue_id} temporarily blacklisted until {until:?}")]
    TemporarilyBlacklisted { venue_id: u32, until: Instant },
}

/// Venue toxicity status
#[derive(Debug, Clone)]
pub struct VenueToxicityStatus {
    pub venue_id: u32,
    pub vpin: f64,              // Current VPIN value (0.0 to 1.0)
    pub is_blacklisted: bool,
    pub blacklist_until: Option<Instant>,
    pub is_permanent: bool,
    pub last_updated: Instant,
    pub violation_count: u32,
}

/// Toxicity Router with VPIN tracking
pub struct ToxicityRouter {
    /// VPIN values per venue
    venue_vpin: dashmap::DashMap<u32, f64>,
    /// Blacklist status per venue
    blacklist_status: dashmap::DashMap<u32, VenueToxicityStatus>,
    /// VPIN threshold for temporary blacklist
    temp_threshold: AtomicU64, // Stored as basis points (e.g., 7000 = 0.70)
    /// VPIN threshold for permanent blacklist
    perm_threshold: AtomicU64,
    /// Default temporary blacklist duration
    default_blacklist_duration: AtomicU64, // Seconds
    /// Enabled flag
    enabled: AtomicBool,
    /// Total blacklists issued
    blacklist_count: AtomicU64,
}

impl ToxicityRouter {
    pub fn new() -> Self {
        Self {
            venue_vpin: dashmap::DashMap::new(),
            blacklist_status: dashmap::DashMap::new(),
            temp_threshold: AtomicU64::new(7000), // 0.70 default
            perm_threshold: AtomicU64::new(9000), // 0.90 default
            default_blacklist_duration: AtomicU64::new(300), // 5 minutes
            enabled: AtomicBool::new(true),
            blacklist_count: AtomicU64::new(0),
        }
    }

    /// Update VPIN value for a venue
    pub fn update_vpin(&self, venue_id: u32, vpin: f64) -> Result<(), ToxicityError> {
        if vpin < 0.0 || vpin > 1.0 {
            return Err(ToxicityError::InvalidVPIN { vpin });
        }

        self.venue_vpin.insert(venue_id, vpin);

        // Check if this triggers a blacklist
        self.check_and_update_blacklist(venue_id, vpin)?;

        Ok(())
    }

    /// Get current VPIN for a venue
    pub fn get_vpin(&self, venue_id: u32) -> Option<f64> {
        self.venue_vpin.get(&venue_id).map(|entry| *entry.value())
    }

    /// Check and update blacklist status based on VPIN
    fn check_and_update_blacklist(&self, venue_id: u32, vpin: f64) -> Result<(), ToxicityError> {
        let temp_thresh = self.temp_threshold.load(Ordering::Acquire) as f64 / 10_000.0;
        let perm_thresh = self.perm_threshold.load(Ordering::Acquire) as f64 / 10_000.0;

        let mut status = self.blacklist_status.entry(venue_id).or_insert_with(|| {
            VenueToxicityStatus {
                venue_id,
                vpin,
                is_blacklisted: false,
                blacklist_until: None,
                is_permanent: false,
                last_updated: Instant::now(),
                violation_count: 0,
            }
        });

        status.vpin = vpin;
        status.last_updated = Instant::now();

        if vpin >= perm_thresh {
            // Permanent blacklist
            status.is_blacklisted = true;
            status.is_permanent = true;
            status.blacklist_until = None;
            status.violation_count += 1;
            self.blacklist_count.fetch_add(1, Ordering::Relaxed);
            return Err(ToxicityError::PermanentlyBlacklisted { venue_id });
        } else if vpin >= temp_thresh {
            // Temporary blacklist
            let duration = Duration::from_secs(self.default_blacklist_duration.load(Ordering::Acquire));
            status.is_blacklisted = true;
            status.is_permanent = false;
            status.blacklist_until = Some(Instant::now() + duration);
            status.violation_count += 1;
            self.blacklist_count.fetch_add(1, Ordering::Relaxed);
            return Err(ToxicityError::TemporarilyBlacklisted {
                venue_id,
                until: status.blacklist_until.unwrap(),
            });
        } else {
            // Clear blacklist if VPIN normalized
            status.is_blacklisted = false;
            status.is_permanent = false;
            status.blacklist_until = None;
        }

        Ok(())
    }

    /// Check if venue is currently safe to route to
    pub fn is_venue_safe(&self, venue_id: u32) -> Result<bool, ToxicityError> {
        if !self.enabled.load(Ordering::Acquire) {
            return Ok(true); // Bypass when disabled
        }

        let status_entry = self.blacklist_status.get(&venue_id);
        
        if let Some(status) = status_entry {
            if status.is_permanent {
                return Err(ToxicityError::PermanentlyBlacklisted { venue_id });
            }

            if let Some(until) = status.blacklist_until {
                if Instant::now() < until {
                    return Err(ToxicityError::TemporarilyBlacklisted {
                        venue_id,
                        until,
                    });
                } else {
                    // Blacklist expired - clear it
                    drop(status_entry);
                    self.clear_blacklist(venue_id);
                }
            }
        }

        Ok(true)
    }

    /// Manually clear a venue's blacklist status
    pub fn clear_blacklist(&self, venue_id: u32) {
        if let Some(mut status) = self.blacklist_status.get_mut(&venue_id) {
            status.is_blacklisted = false;
            status.is_permanent = false;
            status.blacklist_until = None;
        }
    }

    /// Manually blacklist a venue
    pub fn manual_blacklist(&self, venue_id: u32, permanent: bool) {
        let mut status = self.blacklist_status.entry(venue_id).or_insert_with(|| {
            VenueToxicityStatus {
                venue_id,
                vpin: 0.0,
                is_blacklisted: true,
                blacklist_until: None,
                is_permanent: permanent,
                last_updated: Instant::now(),
                violation_count: 1,
            }
        });

        status.is_blacklisted = true;
        status.is_permanent = permanent;
        
        if !permanent {
            let duration = Duration::from_secs(self.default_blacklist_duration.load(Ordering::Acquire));
            status.blacklist_until = Some(Instant::now() + duration);
        }

        self.blacklist_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Set temporary blacklist threshold (basis points)
    pub fn set_temp_threshold(&self, threshold_bps: u64) {
        self.temp_threshold.store(threshold_bps, Ordering::Release);
    }

    /// Set permanent blacklist threshold (basis points)
    pub fn set_perm_threshold(&self, threshold_bps: u64) {
        self.perm_threshold.store(threshold_bps, Ordering::Release);
    }

    /// Set default blacklist duration (seconds)
    pub fn set_blacklist_duration(&self, duration_secs: u64) {
        self.default_blacklist_duration.store(duration_secs, Ordering::Release);
    }

    /// Enable/disable toxicity routing
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    /// Get all venue toxicity statuses
    pub fn get_all_statuses(&self) -> Vec<VenueToxicityStatus> {
        self.blacklist_status
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get blacklist count
    pub fn get_blacklist_count(&self) -> u64 {
        self.blacklist_count.load(Ordering::Acquire)
    }
}

impl Default for ToxicityRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vpin_update() {
        let router = ToxicityRouter::new();
        
        // Normal VPIN
        assert!(router.update_vpin(1, 0.50).is_ok());
        assert_eq!(router.get_vpin(1), Some(0.50));
        
        // Invalid VPIN
        assert!(matches!(router.update_vpin(1, 1.5), Err(ToxicityError::InvalidVPIN { .. })));
    }

    #[test]
    fn test_temporary_blacklist() {
        let router = ToxicityRouter::new();
        router.set_temp_threshold(7000); // 0.70
        
        // High VPIN triggers temporary blacklist
        let result = router.update_vpin(1, 0.75);
        assert!(matches!(result, Err(ToxicityError::TemporarilyBlacklisted { .. })));
        
        // Venue should be unsafe
        assert!(matches!(router.is_venue_safe(1), Err(ToxicityError::TemporarilyBlacklisted { .. })));
    }

    #[test]
    fn test_permanent_blacklist() {
        let router = ToxicityRouter::new();
        router.set_perm_threshold(9000); // 0.90
        
        // Very high VPIN triggers permanent blacklist
        let result = router.update_vpin(1, 0.95);
        assert!(matches!(result, Err(ToxicityError::PermanentlyBlacklisted { .. })));
        
        // Venue should be permanently unsafe
        assert!(matches!(router.is_venue_safe(1), Err(ToxicityError::PermanentlyBlacklisted { .. })));
    }

    #[test]
    fn test_manual_blacklist() {
        let router = ToxicityRouter::new();
        
        // Manual permanent blacklist
        router.manual_blacklist(1, true);
        assert!(matches!(router.is_venue_safe(1), Err(ToxicityError::PermanentlyBlacklisted { .. })));
        
        // Manual temporary blacklist
        router.manual_blacklist(2, false);
        assert!(matches!(router.is_venue_safe(2), Err(ToxicityError::TemporarilyBlacklisted { .. })));
    }

    #[test]
    fn test_disabled_router() {
        let router = ToxicityRouter::new();
        router.set_enabled(false);
        
        // Even with high VPIN, should be safe when disabled
        let _ = router.update_vpin(1, 0.95);
        assert!(router.is_venue_safe(1).unwrap());
    }

    #[test]
    fn test_blacklist_expiry() {
        let router = ToxicityRouter::new();
        router.set_temp_threshold(7000);
        router.set_blacklist_duration(1); // 1 second for testing
        
        // Trigger temporary blacklist
        let _ = router.update_vpin(1, 0.75);
        assert!(matches!(router.is_venue_safe(1), Err(_)));
        
        // Wait for expiry
        std::thread::sleep(Duration::from_secs(2));
        
        // Should be safe now
        assert!(router.is_venue_safe(1).unwrap());
    }
}
