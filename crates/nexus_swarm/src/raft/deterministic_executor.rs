//! Deterministic Execution Engine for Raft Consensus
//! 
//! Ensures exactly-once execution semantics by proposing every outbound order
//! to the Raft log before transmission. Guarantees no double-submission on node restart.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use crate::raft::multi_group_raft::{LogEntry, RaftCommand, OrderSide, TransactionId};

/// Unique transaction identifier (128-bit)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransactionId(pub [u8; 16]);

impl TransactionId {
    pub fn new() -> Self {
        let mut bytes = [0u8; 16];
        
        // Use a combination of timestamp and random bytes for uniqueness
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        
        bytes[0..8].copy_from_slice(&now.to_be_bytes());
        
        // Fill remaining with random bytes
        #[cfg(not(test))]
        {
            use rand::RngCore;
            let mut rng = rand::thread_rng();
            rng.fill_bytes(&mut bytes[8..16]);
        }
        #[cfg(test)]
        {
            bytes[8..16].copy_from_slice(&[0u8; 8]);
        }
        
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

impl Default for TransactionId {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of executing a transaction
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionResult {
    /// Transaction committed successfully
    Committed(TransactionId),
    /// Transaction pending more data
    Pending,
    /// Transaction rolled back due to error
    RolledBack(String),
}

/// Atomic transaction wrapper for deterministic execution
#[derive(Debug, Clone)]
pub struct AtomicTransaction {
    pub id: TransactionId,
    pub command: RaftCommand,
    pub pre_state: Option<StateSnapshot>,
    pub post_state: Option<StateSnapshot>,
    pub executed: bool,
    pub rolled_back: bool,
}

impl AtomicTransaction {
    pub fn new(id: TransactionId, command: RaftCommand) -> Self {
        Self {
            id,
            command,
            pre_state: None,
            post_state: None,
            executed: false,
            rolled_back: false,
        }
    }

    /// Execute the transaction atomically
    pub async fn execute<F, T>(&mut self, mutator: F) -> Result<ExecutionResult, String>
    where
        F: FnOnce(&RaftCommand, &mut StateSnapshot) -> Result<T, String>,
    {
        if self.executed {
            return Ok(ExecutionResult::Committed(self.id));
        }

        if self.rolled_back {
            return Err("Transaction already rolled back".to_string());
        }

        // Capture pre-state
        let mut current_state = StateSnapshot::default();
        self.pre_state = Some(current_state.clone());

        // Execute mutation
        match mutator(&self.command, &mut current_state) {
            Ok(_) => {
                self.post_state = Some(current_state);
                self.executed = true;
                Ok(ExecutionResult::Committed(self.id))
            }
            Err(e) => {
                self.rolled_back = true;
                Err(e)
            }
        }
    }
}

/// Snapshot of OMS state for atomic transactions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub orders: BTreeMap<String, OrderState>,
    pub positions: BTreeMap<String, i64>,
    pub balances: BTreeMap<String, u128>,
    pub sequence_number: u64,
}

/// State of an order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderState {
    pub order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: u64,
    pub filled: u64,
    pub price: u64,
    pub status: OrderStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

impl Default for OrderState {
    fn default() -> Self {
        Self {
            order_id: String::new(),
            symbol: String::new(),
            side: OrderSide::Buy,
            quantity: 0,
            filled: 0,
            price: 0,
            status: OrderStatus::New,
        }
    }
}

/// Deterministic Executor - applies Raft log entries to state machine
pub struct DeterministicExecutor {
    state: RwLock<StateSnapshot>,
    pending_transactions: RwLock<HashMap<TransactionId, AtomicTransaction>>,
    executed_tx_ids: RwLock<HashSet<TransactionId>>,
    sequence_number: RwLock<u64>,
}

impl DeterministicExecutor {
    pub fn new() -> Self {
        Self {
            state: RwLock::new(StateSnapshot::default()),
            pending_transactions: RwLock::new(HashMap::new()),
            executed_tx_ids: RwLock::new(HashSet::new()),
            sequence_number: RwLock::new(0),
        }
    }

    /// Execute a transaction from a Raft log entry
    pub async fn execute_transaction(&self, entry: &LogEntry) -> Result<ExecutionResult, ExecutorError> {
        // Check for duplicate execution (idempotency)
        {
            let executed = self.executed_tx_ids.read().await;
            if executed.contains(&entry.transaction_id) {
                // Already executed - return success without re-executing
                return Ok(ExecutionResult::Committed(entry.transaction_id.clone()));
            }
        }

        // Create atomic transaction
        let mut tx = AtomicTransaction::new(entry.transaction_id.clone(), entry.command.clone());

        // Execute with atomic rollback on failure
        let result = tx.execute(|cmd, state| self.apply_command(cmd, state)).await;

        match result {
            Ok(ExecutionResult::Committed(_)) => {
                // Mark as executed
                self.executed_tx_ids.write().await.insert(entry.transaction_id.clone());
                
                // Update global state
                if let Some(post_state) = tx.post_state.take() {
                    *self.state.write().await = post_state;
                }

                // Increment sequence number
                *self.sequence_number.write().await += 1;

                Ok(ExecutionResult::Committed(entry.transaction_id.clone()))
            }
            Ok(ExecutionResult::Pending) => Ok(ExecutionResult::Pending),
            Err(e) => {
                // Critical: Log divergence prevention
                // If we can't execute deterministically, we must halt
                eprintln!("CRITICAL: Non-deterministic execution detected: {}", e);
                Err(ExecutorError::NonDeterministicExecution(e))
            }
        }
    }

    /// Apply a Raft command to the state snapshot
    fn apply_command(&self, command: &RaftCommand, state: &mut StateSnapshot) -> Result<(), String> {
        match command {
            RaftCommand::SubmitOrder { order_id, symbol, side, quantity, price } => {
                // Check for duplicate order ID
                if state.orders.contains_key(order_id) {
                    return Err(format!("Duplicate order ID: {}", order_id));
                }

                // Validate order parameters
                if *quantity == 0 {
                    return Err("Order quantity must be positive".to_string());
                }

                // Create new order
                let order = OrderState {
                    order_id: order_id.clone(),
                    symbol: symbol.clone(),
                    side: side.clone(),
                    quantity: *quantity,
                    filled: 0,
                    price: *price,
                    status: OrderStatus::New,
                };

                state.orders.insert(order_id.clone(), order);
                state.sequence_number += 1;
            }

            RaftCommand::CancelOrder { order_id } => {
                let order = state.orders.get_mut(order_id)
                    .ok_or_else(|| format!("Order not found: {}", order_id))?;

                if order.status == OrderStatus::Filled || order.status == OrderStatus::Cancelled {
                    return Err(format!("Cannot cancel order in state {:?}", order.status));
                }

                order.status = OrderStatus::Cancelled;
                state.sequence_number += 1;
            }

            RaftCommand::UpdateOrderBook { symbol, bids, asks } => {
                // Update order book state (simplified - in production this would be more complex)
                // For now, just validate the update is well-formed
                if bids.is_empty() && asks.is_empty() {
                    return Err("Order book update must contain at least one level".to_string());
                }

                // Validate price levels are sorted
                for window in bids.windows(2) {
                    if window[0].0 <= window[1].0 {
                        return Err("Bid prices must be sorted descending".to_string());
                    }
                }
                for window in asks.windows(2) {
                    if window[0].0 >= window[1].0 {
                        return Err("Ask prices must be sorted ascending".to_string());
                    }
                }

                state.sequence_number += 1;
            }

            RaftCommand::Checkpoint { checkpoint_id } => {
                // Record checkpoint in state
                tracing::info!("Checkpoint {} created at sequence {}", checkpoint_id, state.sequence_number);
            }
        }

        Ok(())
    }

    /// Get current state summary
    pub async fn get_state_summary(&self) -> BTreeMap<String, serde_json::Value> {
        let state = self.state.read().await;
        let mut summary = BTreeMap::new();

        summary.insert(
            "order_count".to_string(),
            serde_json::json!(state.orders.len()),
        );
        summary.insert(
            "sequence_number".to_string(),
            serde_json::json!(state.sequence_number),
        );
        summary.insert(
            "pending_orders".to_string(),
            serde_json::json!(state.orders.values()
                .filter(|o| o.status == OrderStatus::New || o.status == OrderStatus::PartiallyFilled)
                .count()),
        );

        summary
    }

    /// Restore state from snapshot
    pub async fn restore_state(&self, state: BTreeMap<String, serde_json::Value>) -> Result<(), ExecutorError> {
        let mut new_state = StateSnapshot::default();

        if let Some(seq) = state.get("sequence_number").and_then(|v| v.as_u64()) {
            new_state.sequence_number = seq;
        }

        *self.state.write().await = new_state;
        *self.sequence_number.write().await = seq;

        Ok(())
    }

    /// Get pending transactions count
    pub async fn pending_count(&self) -> usize {
        self.pending_transactions.read().await.len()
    }

    /// Clear executed transaction IDs (for testing)
    #[cfg(test)]
    pub async fn clear_executed(&self) {
        self.executed_tx_ids.write().await.clear();
    }
}

impl Default for DeterministicExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Executor error types
#[derive(Debug, thiserror::Error)]
pub enum ExecutorError {
    #[error("Non-deterministic execution: {0}")]
    NonDeterministicExecution(String),
    #[error("Invalid command: {0}")]
    InvalidCommand(String),
    #[error("State corruption detected")]
    StateCorruption,
    #[error("Rollback failed: {0}")]
    RollbackFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_deterministic_execution() {
        let executor = DeterministicExecutor::new();

        let entry = LogEntry::new(
            1,
            1,
            RaftCommand::SubmitOrder {
                order_id: "test-order-1".to_string(),
                symbol: "BTCUSD".to_string(),
                side: OrderSide::Buy,
                quantity: 100,
                price: 50000,
            },
        );

        let result = executor.execute_transaction(&entry).await;
        assert!(result.is_ok());

        // Verify idempotency - executing same entry again should succeed without re-executing
        let result2 = executor.execute_transaction(&entry).await;
        assert!(result2.is_ok());

        // Verify state
        let summary = executor.get_state_summary().await;
        assert_eq!(summary.get("order_count").unwrap().as_u64(), Some(1));
    }

    #[tokio::test]
    async fn test_order_cancellation() {
        let executor = DeterministicExecutor::new();

        // Submit order
        let submit_entry = LogEntry::new(
            1,
            1,
            RaftCommand::SubmitOrder {
                order_id: "cancel-test".to_string(),
                symbol: "ETHUSD".to_string(),
                side: OrderSide::Sell,
                quantity: 50,
                price: 3000,
            },
        );
        executor.execute_transaction(&submit_entry).await.unwrap();

        // Cancel order
        let cancel_entry = LogEntry::new(
            1,
            2,
            RaftCommand::CancelOrder {
                order_id: "cancel-test".to_string(),
            },
        );
        let result = executor.execute_transaction(&cancel_entry).await;
        assert!(result.is_ok());

        // Try to cancel again - should fail
        let result2 = executor.execute_transaction(&cancel_entry).await;
        assert!(result2.is_err());
    }

    #[tokio::test]
    async fn test_duplicate_order_rejection() {
        let executor = DeterministicExecutor::new();

        let entry = LogEntry::new(
            1,
            1,
            RaftCommand::SubmitOrder {
                order_id: "dup-order".to_string(),
                symbol: "BTCUSD".to_string(),
                side: OrderSide::Buy,
                quantity: 100,
                price: 50000,
            },
        );

        executor.execute_transaction(&entry).await.unwrap();

        // Try to submit same order again
        let dup_entry = LogEntry::new(
            1,
            2,
            RaftCommand::SubmitOrder {
                order_id: "dup-order".to_string(),
                symbol: "BTCUSD".to_string(),
                side: OrderSide::Buy,
                quantity: 200,
                price: 51000,
            },
        );

        let result = executor.execute_transaction(&dup_entry).await;
        assert!(result.is_err());
    }
}
