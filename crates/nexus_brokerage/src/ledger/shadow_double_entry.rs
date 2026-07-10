//! Shadow Double-Entry Ledger with lock-free accounting
//! 
//! Implements a double-entry bookkeeping system that tracks cash, margin, and positions
//! across multiple Prime Brokers using atomic operations for thread safety.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::collections::HashMap;
use dashmap::DashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LedgerError {
    #[error("Insufficient balance: required {required}, available {available}")]
    InsufficientBalance { required: i64, available: i64 },
    #[error("Double-entry violation: debits {debits} != credits {credits}")]
    DoubleEntryViolation { debits: i64, credits: i64 },
    #[error("Reconciliation mismatch: shadow {shadow}, broker {broker}")]
    ReconciliationMismatch { shadow: i64, broker: i64 },
    #[error("Torn read detected at epoch {epoch}")]
    TornReadDetected { epoch: u64 },
}

/// Atomic account balance with separate debit/credit tracking
#[derive(Debug)]
struct AtomicBalance {
    /// Net balance (credits - debits) in fixed-point units
    net: AtomicI64,
    /// Total debits in fixed-point units
    total_debits: AtomicI64,
    /// Total credits in fixed-point units
    total_credits: AtomicI64,
    /// Epoch counter for detecting torn reads
    epoch: AtomicU64,
}

impl AtomicBalance {
    fn new(initial: i64) -> Self {
        Self {
            net: AtomicI64::new(initial),
            total_debits: AtomicI64::new(0),
            total_credits: AtomicI64::new(0),
            epoch: AtomicU64::new(0),
        }
    }

    fn debit(&self, amount: i64) -> Result<(), LedgerError> {
        if amount < 0 {
            return Err(LedgerError::DoubleEntryViolation {
                debits: amount,
                credits: 0,
            });
        }

        // Lock-free CAS loop for atomic update
        let mut current_epoch = self.epoch.load(Ordering::Acquire);
        loop {
            let current_net = self.net.load(Ordering::Acquire);
            
            // Check sufficient balance before debit
            if current_net < amount {
                return Err(LedgerError::InsufficientBalance {
                    required: amount,
                    available: current_net,
                });
            }

            let new_epoch = current_epoch.wrapping_add(1);
            
            // Update epoch first to signal ongoing modification
            if self.epoch.compare_exchange_weak(
                current_epoch,
                new_epoch,
                Ordering::AcqRel,
                Ordering::Acquire,
            ).is_err() {
                current_epoch = self.epoch.load(Ordering::Acquire);
                continue;
            }

            // Perform debit operation
            self.net.store(current_net - amount, Ordering::Release);
            self.total_debits.fetch_add(amount, Ordering::AcqRel);

            // Finalize with new epoch
            self.epoch.store(new_epoch.wrapping_add(1), Ordering::Release);
            return Ok(());
        }
    }

    fn credit(&self, amount: i64) -> Result<(), LedgerError> {
        if amount < 0 {
            return Err(LedgerError::DoubleEntryViolation {
                debits: 0,
                credits: amount,
            });
        }

        let mut current_epoch = self.epoch.load(Ordering::Acquire);
        loop {
            let current_net = self.net.load(Ordering::Acquire);
            let new_epoch = current_epoch.wrapping_add(1);

            if self.epoch.compare_exchange_weak(
                current_epoch,
                new_epoch,
                Ordering::AcqRel,
                Ordering::Acquire,
            ).is_err() {
                current_epoch = self.epoch.load(Ordering::Acquire);
                continue;
            }

            self.net.store(current_net + amount, Ordering::Release);
            self.total_credits.fetch_add(amount, Ordering::AcqRel);

            self.epoch.store(new_epoch.wrapping_add(1), Ordering::Release);
            return Ok(());
        }
    }

    fn get_balance(&self) -> Result<(i64, u64), LedgerError> {
        let epoch_start = self.epoch.load(Ordering::Acquire);
        let net = self.net.load(Ordering::Acquire);
        let epoch_end = self.epoch.load(Ordering::Acquire);

        // Detect torn read: if epoch changed during read, data may be inconsistent
        if epoch_start != epoch_end {
            return Err(LedgerError::TornReadDetected { epoch: epoch_end });
        }

        Ok((net, epoch_end))
    }

    fn verify_double_entry(&self) -> Result<(), LedgerError> {
        let debits = self.total_debits.load(Ordering::Acquire);
        let credits = self.total_credits.load(Ordering::Acquire);
        let net = self.net.load(Ordering::Acquire);

        // In double-entry: net should equal credits - debits
        if net != credits - debits {
            return Err(LedgerError::DoubleEntryViolation { debits, credits });
        }

        Ok(())
    }
}

/// Shadow ledger tracking multiple accounts across brokers
pub struct ShadowLedger {
    /// Map of broker_id -> account_id -> balance
    accounts: DashMap<(u32, u32), AtomicBalance>,
    /// Position tracking: broker_id -> asset_id -> net_position (fixed-point)
    positions: DashMap<(u32, u32), AtomicI64>,
    /// Global epoch for cross-account consistency checks
    global_epoch: AtomicU64,
}

impl ShadowLedger {
    pub fn new() -> Self {
        Self {
            accounts: DashMap::new(),
            positions: DashMap::new(),
            global_epoch: AtomicU64::new(0),
        }
    }

    /// Initialize an account with starting balance
    pub fn init_account(&self, broker_id: u32, account_id: u32, initial_balance: i64) {
        self.accounts.insert((broker_id, account_id), AtomicBalance::new(initial_balance));
    }

    /// Record a debit transaction (e.g., buying an asset)
    pub fn debit(&self, broker_id: u32, account_id: u32, amount: i64) -> Result<u64, LedgerError> {
        let entry = self.accounts.get(&(broker_id, account_id)).ok_or_else(|| {
            LedgerError::InsufficientBalance { required: amount, available: 0 }
        })?;

        entry.value().debit(amount)?;
        
        let epoch = self.global_epoch.fetch_add(1, Ordering::AcqRel);
        Ok(epoch)
    }

    /// Record a credit transaction (e.g., selling an asset)
    pub fn credit(&self, broker_id: u32, account_id: u32, amount: i64) -> Result<u64, LedgerError> {
        let entry = self.accounts.get(&(broker_id, account_id)).ok_or_else(|| {
            LedgerError::DoubleEntryViolation { debits: 0, credits: amount }
        })?;

        entry.value().credit(amount)?;
        
        let epoch = self.global_epoch.fetch_add(1, Ordering::AcqRel);
        Ok(epoch)
    }

    /// Update position for an asset at a specific broker
    pub fn update_position(&self, broker_id: u32, asset_id: u32, delta: i64) -> Result<(), LedgerError> {
        let key = (broker_id, asset_id);
        
        // Use entry API for atomic insert-or-update
        let mut current = self.positions.entry(key).or_insert_with(|| AtomicI64::new(0));
        current.fetch_add(delta, Ordering::AcqRel);
        
        Ok(())
    }

    /// Get current balance with torn-read detection
    pub fn get_balance(&self, broker_id: u32, account_id: u32) -> Result<(i64, u64), LedgerError> {
        let entry = self.accounts.get(&(broker_id, account_id)).ok_or_else(|| {
            LedgerError::InsufficientBalance { required: 0, available: 0 }
        })?;

        entry.value().get_balance()
    }

    /// Get current position for an asset
    pub fn get_position(&self, broker_id: u32, asset_id: u32) -> i64 {
        self.positions
            .get(&(broker_id, asset_id))
            .map(|entry| entry.load(Ordering::Acquire))
            .unwrap_or(0)
    }

    /// Verify double-entry integrity for a specific account
    pub fn verify_account_integrity(&self, broker_id: u32, account_id: u32) -> Result<(), LedgerError> {
        let entry = self.accounts.get(&(broker_id, account_id)).ok_or_else(|| {
            LedgerError::DoubleEntryViolation { debits: 0, credits: 0 }
        })?;

        entry.value().verify_double_entry()
    }

    /// Reconcile shadow ledger against broker-reported state
    pub fn reconcile(&self, broker_id: u32, account_id: u32, broker_balance: i64) -> Result<(), LedgerError> {
        let (shadow_balance, _epoch) = self.get_balance(broker_id, account_id)?;

        if shadow_balance != broker_balance {
            return Err(LedgerError::ReconciliationMismatch {
                shadow: shadow_balance,
                broker: broker_balance,
            });
        }

        Ok(())
    }

    /// Get all positions across all brokers for an asset
    pub fn get_cross_broker_position(&self, asset_id: u32) -> i64 {
        let mut total = 0i64;
        for entry in self.positions.iter() {
            let ((broker_id, aid), value) = entry.pair();
            if *aid == asset_id {
                total += value.load(Ordering::Acquire);
            }
        }
        total
    }
}

impl Default for ShadowLedger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_entry_basic() {
        let ledger = ShadowLedger::new();
        ledger.init_account(1, 100, 1_000_000); // 1M initial

        // Debit 100k
        assert!(ledger.debit(1, 100, 100_000).is_ok());
        
        // Credit 50k
        assert!(ledger.credit(1, 100, 50_000).is_ok());

        // Verify balance: 1M - 100k + 50k = 950k
        let (balance, _) = ledger.get_balance(1, 100).expect("Should have balance");
        assert_eq!(balance, 950_000);

        // Verify double-entry integrity
        assert!(ledger.verify_account_integrity(1, 100).is_ok());
    }

    #[test]
    fn test_insufficient_balance() {
        let ledger = ShadowLedger::new();
        ledger.init_account(1, 100, 100_000);

        // Try to debit more than available
        let result = ledger.debit(1, 100, 150_000);
        assert!(matches!(result, Err(LedgerError::InsufficientBalance { .. })));
    }

    #[test]
    fn test_reconciliation() {
        let ledger = ShadowLedger::new();
        ledger.init_account(1, 100, 500_000);
        ledger.debit(1, 100, 100_000).unwrap();

        // Correct reconciliation
        assert!(ledger.reconcile(1, 100, 400_000).is_ok());

        // Incorrect reconciliation
        let result = ledger.reconcile(1, 100, 450_000);
        assert!(matches!(result, Err(LedgerError::ReconciliationMismatch { .. })));
    }
}
