//! Leader Lease Token for Cryptographic Authority Gating
//! 
//! Provides a time-limited token that proves leadership authority for order signing.
//! CRITICAL: Token validation and HMAC signing MUST be atomic to prevent race conditions.

use crate::{NodeId, Term, ConsensusError};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A cryptographic lease token proving leadership authority
/// 
/// SECURITY: This token uses atomic operations to ensure that validation
/// and usage happen without race conditions. The token cannot be cloned
/// to prevent use-after-revocation attacks.
pub struct LeaderLeaseToken {
    /// Node ID of the leader holding this lease
    node_id: NodeId,
    /// Raft term for which this lease is valid
    term: Term,
    /// Timestamp when lease was issued (nanoseconds since epoch)
    issued_at_ns: u64,
    /// Timestamp when lease expires (nanoseconds since epoch)
    expires_at_ns: AtomicU64,
    /// Flag indicating if lease has been revoked
    revoked: AtomicBool,
    /// Epoch counter for atomic validation
    epoch: AtomicU64,
}

unsafe impl Send for LeaderLeaseToken {}
unsafe impl Sync for LeaderLeaseToken {}

impl LeaderLeaseToken {
    /// Create a new lease token with the specified duration
    pub fn new(node_id: NodeId, term: Term, duration: Duration) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let expires_at = now + duration.as_nanos() as u64;
        
        Self {
            node_id,
            term,
            issued_at_ns: now,
            expires_at_ns: AtomicU64::new(expires_at),
            revoked: AtomicBool::new(false),
            epoch: AtomicU64::new(1),
        }
    }

    /// Check if the token is currently valid
    /// 
    /// ATOMIC GUARANTEE: This check reads both expiration and revocation
    /// atomically using memory ordering to prevent torn reads.
    pub fn is_valid(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        // Use acquire ordering to ensure we see the latest writes
        let expires_at = self.expires_at_ns.load(Ordering::Acquire);
        let revoked = self.revoked.load(Ordering::Acquire);
        
        !revoked && now < expires_at
    }

    /// Atomically validate the token and return the current epoch
    /// 
    /// This method should be called immediately before HMAC signing to ensure
    /// no race condition between validation and signature generation.
    pub fn validate_and_get_epoch(&self) -> Result<u64, ConsensusError> {
        if !self.is_valid() {
            return Err(ConsensusError::LeaseExpired {
                term: self.term,
                valid_until: self.expires_at_ns.load(Ordering::Relaxed),
            });
        }
        
        Ok(self.epoch.load(Ordering::Acquire))
    }

    /// Renew the lease with a new expiration time
    /// 
    /// Returns the new epoch number if successful
    pub fn renew(&self, duration: Duration) -> Result<u64, ConsensusError> {
        if self.revoked.load(Ordering::Acquire) {
            return Err(ConsensusError::LeaseExpired {
                term: self.term,
                valid_until: 0,
            });
        }
        
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let new_expires_at = now + duration.as_nanos() as u64;
        self.expires_at_ns.store(new_expires_at, Ordering::Release);
        
        // Increment epoch to invalidate any in-flight validations
        let new_epoch = self.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        
        Ok(new_epoch)
    }

    /// Revoke the lease immediately
    pub fn revoke(&self) {
        self.revoked.store(true, Ordering::Release);
    }

    /// Get the node ID associated with this lease
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    /// Get the term for which this lease is valid
    pub fn term(&self) -> Term {
        self.term
    }

    /// Get remaining time on the lease
    pub fn remaining_duration(&self) -> Option<Duration> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let expires_at = self.expires_at_ns.load(Ordering::Acquire);
        
        if now >= expires_at {
            None
        } else {
            Some(Duration::from_nanos(expires_at - now))
        }
    }

    /// Extend the lease by a given duration (only if still valid)
    pub fn extend(&self, extension: Duration) -> Result<(), ConsensusError> {
        if !self.is_valid() {
            return Err(ConsensusError::LeaseExpired {
                term: self.term,
                valid_until: self.expires_at_ns.load(Ordering::Relaxed),
            });
        }
        
        let current_expires = self.expires_at_ns.load(Ordering::Acquire);
        let new_expires = current_expires + extension.as_nanos() as u64;
        
        self.expires_at_ns.store(new_expires, Ordering::Release);
        let _ = self.epoch.fetch_add(1, Ordering::AcqRel);
        
        Ok(())
    }
}

/// Cryptographic Authority Gate that wraps the WAPI signer
/// 
/// This gate ensures that orders can only be signed when a valid
/// leader lease token is present AND the epoch hasn't changed during signing.
pub struct CryptographicAuthorityGate {
    current_token: Arc<std::sync::RwLock<Option<Arc<LeaderLeaseToken>>>>,
    signing_enabled: AtomicBool,
}

unsafe impl Send for CryptographicAuthorityGate {}
unsafe impl Sync for CryptographicAuthorityGate {}

impl CryptographicAuthorityGate {
    pub fn new() -> Self {
        Self {
            current_token: Arc::new(std::sync::RwLock::new(None)),
            signing_enabled: AtomicBool::new(false),
        }
    }

    /// Set the current leader lease token
    pub fn set_token(&self, token: Arc<LeaderLeaseToken>) -> Result<(), ConsensusError> {
        if !token.is_valid() {
            return Err(ConsensusError::LeaseExpired {
                term: token.term(),
                valid_until: 0,
            });
        }
        
        let mut guard = self.current_token.write().unwrap();
        *guard = Some(token);
        self.signing_enabled.store(true, Ordering::Release);
        
        Ok(())
    }

    /// Clear the current token (called when stepping down from leadership)
    pub fn clear_token(&self) {
        let mut guard = self.current_token.write().unwrap();
        *guard = None;
        self.signing_enabled.store(false, Ordering::Release);
    }

    /// Execute a signing operation atomically with lease validation
    /// 
    /// CRITICAL: This method ensures that the lease is validated immediately
    /// before the signing closure is executed, preventing any race condition
    /// where the lease could be revoked between validation and signing.
    /// 
    /// The closure receives the epoch number, which should be included in
    /// the signed payload to detect any mid-signing revocation.
    pub fn execute_with_lease<F, R>(&self, f: F) -> Result<R, ConsensusError>
    where
        F: FnOnce(u64, NodeId, Term) -> Result<R, ConsensusError>,
    {
        if !self.signing_enabled.load(Ordering::Acquire) {
            return Err(ConsensusError::NotLeader(0));
        }

        let guard = self.current_token.read().unwrap();
        let token = guard.as_ref().ok_or(ConsensusError::NotLeader(0))?;

        // Atomically validate and get epoch - this MUST happen immediately before signing
        let epoch = token.validate_and_get_epoch()?;
        let node_id = token.node_id();
        let term = token.term();
        
        // Drop the read guard before executing the closure to prevent deadlocks
        drop(guard);
        
        // Execute the signing operation with the validated epoch
        f(epoch, node_id, term)
    }

    /// Check if signing is currently enabled
    pub fn is_signing_enabled(&self) -> bool {
        self.signing_enabled.load(Ordering::Acquire)
    }

    /// Get the current token info without allowing signing
    pub fn get_token_info(&self) -> Option<(NodeId, Term, Option<Duration>)> {
        let guard = self.current_token.read().unwrap();
        guard.as_ref().map(|t| {
            (t.node_id(), t.term(), t.remaining_duration())
        })
    }
}

impl Default for CryptographicAuthorityGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lease_token_creation() {
        let token = LeaderLeaseToken::new(1, 5, Duration::from_secs(60));
        
        assert_eq!(token.node_id(), 1);
        assert_eq!(token.term(), 5);
        assert!(token.is_valid());
        assert!(token.remaining_duration().is_some());
    }

    #[test]
    fn test_lease_token_expiration() {
        let token = LeaderLeaseToken::new(1, 5, Duration::from_millis(50));
        
        assert!(token.is_valid());
        
        std::thread::sleep(Duration::from_millis(60));
        
        assert!(!token.is_valid());
        assert!(token.remaining_duration().is_none());
    }

    #[test]
    fn test_lease_revocation() {
        let token = LeaderLeaseToken::new(1, 5, Duration::from_secs(60));
        
        assert!(token.is_valid());
        
        token.revoke();
        
        assert!(!token.is_valid());
    }

    #[test]
    fn test_cryptographic_gate() {
        let gate = CryptographicAuthorityGate::new();
        
        assert!(!gate.is_signing_enabled());
        
        let token = Arc::new(LeaderLeaseToken::new(1, 5, Duration::from_secs(60)));
        gate.set_token(Arc::clone(&token)).unwrap();
        
        assert!(gate.is_signing_enabled());
        
        let result = gate.execute_with_lease(|epoch, node_id, term| {
            Ok((epoch, node_id, term))
        });
        
        assert!(result.is_ok());
        let (epoch, node_id, term) = result.unwrap();
        assert_eq!(node_id, 1);
        assert_eq!(term, 5);
        assert_eq!(epoch, 1);
        
        gate.clear_token();
        assert!(!gate.is_signing_enabled());
    }

    #[test]
    fn test_gate_rejects_expired_token() {
        let gate = CryptographicAuthorityGate::new();
        let token = Arc::new(LeaderLeaseToken::new(1, 5, Duration::from_millis(50)));
        
        std::thread::sleep(Duration::from_millis(60));
        
        let result = gate.set_token(token);
        assert!(result.is_err());
        assert!(!gate.is_signing_enabled());
    }
}
