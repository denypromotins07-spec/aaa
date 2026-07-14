//! Phantom Fill Detector - Detects dropped WebSocket packets and state drift.
//! 
//! CRITICAL: If the exchange reports a fill that the local OMS missed due to
//! a dropped WebSocket packet, this detector MUST instantly halt trading and
//! trigger the kill switch to prevent catastrophic unhedged exposure.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{error, warn, info};

use crate::{ReconcileError, ExchangeStateSnapshot, FillReport};
use super::lock_free_oms_snapshot::LockFreeOMSSnapshot;

/// Result of state reconciliation
#[derive(Debug, Clone, PartialEq)]
pub enum ReconcileResult {
    /// States match within tolerance
    Ok,
    
    /// Minor timing difference (acceptable)
    MinorDrift { diff_scaled: i128 },
    
    /// Critical state drift detected - HALT TRADING
    CriticalDrift { 
        local_balance: i128,
        exchange_balance: i128,
        diff_scaled: i128,
    },
    
    /// Phantom fill detected - order filled on exchange but not in local OMS
    PhantomFill {
        order_id: String,
        expected_filled: i128,
        actual_filled: i128,
    },
}

/// Configuration for phantom fill detection
#[derive(Debug, Clone)]
pub struct PhantomFillConfig {
    /// Tolerance for balance differences (in scaled units, default 100 = 0.00000100)
    pub balance_tolerance_scaled: i128,
    
    /// Whether to trigger kill switch on phantom fill
    pub halt_on_phantom_fill: bool,
    
    /// Maximum acceptable fill latency before considering it phantom (ms)
    pub max_fill_latency_ms: u64,
}

impl Default for PhantomFillConfig {
    fn default() -> Self {
        Self {
            balance_tolerance_scaled: 100,  // 0.00000100 with 8 decimal scaling
            halt_on_phantom_fill: true,
            max_fill_latency_ms: 1000,
        }
    }
}

/// Statistics about reconciliation operations
#[derive(Debug, Clone, Default)]
pub struct ReconcileStats {
    pub total_reconciliations: u64,
    pub successful_reconciliations: u64,
    pub minor_drift_count: u64,
    pub critical_drift_count: u64,
    pub phantom_fills_detected: u64,
    pub kill_switch_triggers: u64,
}

/// Phantom Fill Detector that compares exchange state with local OMS
pub struct PhantomFillDetector {
    config: PhantomFillConfig,
    oms_snapshot: Arc<LockFreeOMSSnapshot>,
    
    /// Flag indicating if kill switch should be triggered
    kill_switch_triggered: AtomicBool,
    
    /// Counter for total reconciliations
    reconcile_count: AtomicU64,
    
    /// Statistics
    stats: Arc<parking_lot::RwLock<ReconcileStats>>,
    
    /// Optional callback to trigger kill switch (set by caller)
    kill_switch_callback: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl PhantomFillDetector {
    pub fn new(
        config: PhantomFillConfig,
        oms_snapshot: Arc<LockFreeOMSSnapshot>,
    ) -> Self {
        Self {
            config,
            oms_snapshot,
            kill_switch_triggered: AtomicBool::new(false),
            reconcile_count: AtomicU64::new(0),
            stats: Arc::new(parking_lot::RwLock::new(ReconcileStats::default())),
            kill_switch_callback: None,
        }
    }
    
    /// Set callback to trigger kill switch when drift is detected
    pub fn set_kill_switch_callback<F>(&mut self, callback: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.kill_switch_callback = Some(Arc::new(callback));
    }
    
    /// Check if kill switch has been triggered
    pub fn is_kill_switch_triggered(&self) -> bool {
        self.kill_switch_triggered.load(Ordering::SeqCst)
    }
    
    /// Reset kill switch trigger (only after manual intervention)
    pub fn reset_kill_switch(&self) {
        self.kill_switch_triggered.store(false, Ordering::SeqCst);
    }
    
    /// Get current statistics
    pub fn get_stats(&self) -> ReconcileStats {
        self.stats.read().clone()
    }
    
    /// Main reconciliation function - compare exchange state with local OMS
    /// 
    /// Returns ReconcileResult indicating the status of reconciliation.
    /// If CriticalDrift or PhantomFill is returned, trading should halt immediately.
    pub fn reconcile(&self, exchange_state: &ExchangeStateSnapshot) -> ReconcileResult {
        self.reconcile_count.fetch_add(1, Ordering::Relaxed);
        
        // Get tear-free OMS snapshot
        let oms_state = self.oms_snapshot.snapshot_blocking();
        
        // Calculate exchange total balance
        let exchange_total: i128 = exchange_state.balances.values().sum();
        
        // Compare balances
        let balance_diff = oms_state.total_balance - exchange_total;
        let abs_diff = balance_diff.abs();
        
        if abs_diff > self.config.balance_tolerance_scaled {
            error!(
                "CRITICAL_STATE_DRIFT: local={}, exchange={}, diff={}",
                oms_state.total_balance,
                exchange_total,
                balance_diff
            );
            
            {
                let mut stats = self.stats.write();
                stats.critical_drift_count += 1;
            }
            
            self.trigger_kill_switch();
            
            return ReconcileResult::CriticalDrift {
                local_balance: oms_state.total_balance,
                exchange_balance: exchange_total,
                diff_scaled: balance_diff,
            };
        }
        
        // Check for phantom fills by comparing order states
        if let Some(phantom) = self.detect_phantom_fills(exchange_state) {
            warn!("Phantom fill detected: {:?}", phantom);
            
            {
                let mut stats = self.stats.write();
                stats.phantom_fills_detected += 1;
            }
            
            if self.config.halt_on_phantom_fill {
                self.trigger_kill_switch();
            }
            
            return ReconcileResult::PhantomFill {
                order_id: phantom.order_id,
                expected_filled: phantom.expected_filled,
                actual_filled: phantom.actual_filled,
            };
        }
        
        // Check for minor drift (acceptable timing differences)
        if abs_diff > 0 {
            let mut stats = self.stats.write();
            stats.minor_drift_count += 1;
            
            return ReconcileResult::MinorDrift { 
                diff_scaled: balance_diff 
            };
        }
        
        // Full match
        {
            let mut stats = self.stats.write();
            stats.successful_reconciliations += 1;
        }
        
        ReconcileResult::Ok
    }
    
    /// Detect phantom fills by comparing order filled quantities
    fn detect_phantom_fills(
        &self, 
        exchange_state: &ExchangeStateSnapshot
    ) -> Option<PhantomFillInfo> {
        // In production, this would compare against a local order book
        // For now, we check if any exchange order has filled_qty > 0
        // while our OMS shows no corresponding position change
        
        for order in &exchange_state.active_orders {
            if order.filled_qty > 0 {
                // Check if OMS position reflects this fill
                // This is simplified - production would track per-order fills
                
                let position = exchange_state.positions.get(&order.symbol);
                
                if let Some(pos) = position {
                    // Simple heuristic: if position qty doesn't match expected
                    // from filled orders, we may have a phantom fill
                    
                    // In production, this would maintain a map of order_id -> expected_fill
                    // and compare against actual exchange fills
                    
                    // Placeholder detection logic
                    if order.filled_qty > pos.qty && order.status == "FILLED" {
                        return Some(PhantomFillInfo {
                            order_id: order.order_id.clone(),
                            expected_filled: pos.qty,
                            actual_filled: order.filled_qty,
                        });
                    }
                }
            }
        }
        
        None
    }
    
    /// Trigger the kill switch
    fn trigger_kill_switch(&self) {
        self.kill_switch_triggered.store(true, Ordering::SeqCst);
        
        {
            let mut stats = self.stats.write();
            stats.kill_switch_triggers += 1;
        }
        
        error!("KILL SWITCH TRIGGERED - State drift detected");
        
        // Call callback if set
        if let Some(ref callback) = self.kill_switch_callback {
            callback();
        }
    }
}

#[derive(Debug, Clone)]
pub struct PhantomFillInfo {
    pub order_id: String,
    pub expected_filled: i128,
    pub actual_filled: i128,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_reconcile_matching_states() {
        let config = PhantomFillConfig::default();
        let oms = Arc::new(LockFreeOMSSnapshot::new());
        
        // Set up matching state
        oms.update_state(100_000_000, 50_000_000, 1_000_000, 2, 12345);
        
        let detector = PhantomFillDetector::new(config, oms);
        
        let mut balances = HashMap::new();
        balances.insert("USDT".to_string(), 100_000_000);
        
        let exchange_state = ExchangeStateSnapshot {
            balances,
            positions: HashMap::new(),
            active_orders: vec![],
            server_time_ms: 0,
            epoch_id: 0,
        };
        
        let result = detector.reconcile(&exchange_state);
        assert_eq!(result, ReconcileResult::Ok);
    }
    
    #[test]
    fn test_reconcile_critical_drift() {
        let config = PhantomFillConfig::default();
        let oms = Arc::new(LockFreeOMSSnapshot::new());
        
        // Set up state with significant drift
        oms.update_state(100_000_000, 50_000_000, 1_000_000, 2, 12345);
        
        let detector = PhantomFillDetector::new(config, oms);
        
        let mut balances = HashMap::new();
        // Exchange shows different balance (drift of 1000 > tolerance of 100)
        balances.insert("USDT".to_string(), 99_998_000);
        
        let exchange_state = ExchangeStateSnapshot {
            balances,
            positions: HashMap::new(),
            active_orders: vec![],
            server_time_ms: 0,
            epoch_id: 0,
        };
        
        let result = detector.reconcile(&exchange_state);
        assert!(matches!(result, ReconcileResult::CriticalDrift { .. }));
        assert!(detector.is_kill_switch_triggered());
    }
    
    #[test]
    fn test_kill_switch_callback() {
        let config = PhantomFillConfig::default();
        let oms = Arc::new(LockFreeOMSSnapshot::new());
        oms.update_state(100_000_000, 50_000_000, 1_000_000, 2, 12345);
        
        let mut detector = PhantomFillDetector::new(config, oms);
        
        let triggered = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let triggered_clone = Arc::clone(&triggered);
        
        detector.set_kill_switch_callback(move || {
            triggered_clone.store(true, Ordering::SeqCst);
        });
        
        let mut balances = HashMap::new();
        balances.insert("USDT".to_string(), 50_000_000);  // Large drift
        
        let exchange_state = ExchangeStateSnapshot {
            balances,
            positions: HashMap::new(),
            active_orders: vec![],
            server_time_ms: 0,
            epoch_id: 0,
        };
        
        let _ = detector.reconcile(&exchange_state);
        
        assert!(triggered.load(Ordering::SeqCst));
        assert!(detector.is_kill_switch_triggered());
    }
}
