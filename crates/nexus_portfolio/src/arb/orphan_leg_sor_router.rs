//! Orphan Leg SOR Router - Aggressively flattens orphaned positions.
//! 
//! When an orphan leg is detected, this router uses limit orders with strict
//! slippage bounds to flatten the exposure. NEVER uses market orders during flash crashes.

use std::sync::atomic::{AtomicBool, AtomicI128, Ordering};
use crate::margin::fixed_point_pnl::FixedPoint;

/// Maximum slippage allowed during flatten (in basis points)
const MAX_SLIPPAGE_BPS: i128 = 50; // 0.5% max slippage

/// State of the SOR router
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SorRouterState {
    Idle,
    CalculatingLimits,
    SendingCancel,
    SendingLimitOrders,
    FlatteningComplete,
    Failed,
}

/// Command to flatten an orphan leg
#[derive(Debug, Clone)]
pub struct FlattenCommand {
    /// Exchange ID where the orphan exists
    pub exchange_id: u8,
    /// Symbol to flatten
    pub symbol: [u8; 16],
    /// Size to flatten (positive = long, negative = short)
    pub size_scaled: i128,
    /// Reference price (scaled)
    pub reference_price_scaled: i128,
    /// Is long position?
    pub is_long: bool,
}

/// Result of flatten attempt
#[derive(Debug, Clone)]
pub enum FlattenResult {
    Success,
    Partial { filled_scaled: i128, remaining_scaled: i128 },
    Failed(&'static str),
}

/// SOR router for orphan leg flattening
pub struct OrphanLegSorRouter {
    current_state: AtomicU8,
    /// Size successfully flattened
    flattened_size: AtomicI128,
    /// Flag indicating operation in progress
    operation_in_progress: AtomicBool,
    /// Slippage exceeded flag
    slippage_exceeded: AtomicBool,
}

// Map u8 to SorRouterState
impl OrphanLegSorRouter {
    pub fn new() -> Self {
        Self {
            current_state: AtomicU8::new(SorRouterState::Idle as u8),
            flattened_size: AtomicI128::new(0),
            operation_in_progress: AtomicBool::new(false),
            slippage_exceeded: AtomicBool::new(false),
        }
    }

    /// Initiate flatten operation
    pub fn initiate_flatten(&self, command: &FlattenCommand) -> bool {
        let expected = SorRouterState::Idle as u8;
        let desired = SorRouterState::CalculatingLimits as u8;
        
        match self.current_state.compare_exchange(
            expected,
            desired,
            Ordering::SeqCst,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                self.operation_in_progress.store(true, Ordering::SeqCst);
                true
            }
            Err(_) => false,
        }
    }

    /// Calculate limit price with slippage bounds
    /// CRITICAL: Uses limit orders only, never market orders
    pub fn calculate_limit_price(&self, command: &FlattenCommand) -> Result<i128, &'static str> {
        let ref_price = FixedPoint::from_scaled(command.reference_price_scaled);
        
        // Calculate max acceptable slippage
        let slippage_bps = FixedPoint::from_scaled(MAX_SLIPPAGE_BPS * 1_000_000_000_000i128); // Convert bps to scaled
        let hundred_percent = FixedPoint::from_scaled(10_000 * 1_000_000_000_000i128); // 10000 bps = 100%
        
        if command.is_long {
            // Selling: limit price = reference * (1 - slippage)
            // We accept getting up to X% less than reference
            let one = FixedPoint::from_scaled(1_000_000_000_000_000_000i128);
            let slippage_factor = slippage_bps.checked_div(&hundred_percent)
                .ok_or("Slippage calculation overflow")?;
            
            let limit_factor = one.checked_sub(&slippage_factor)
                .ok_or("Limit factor underflow")?;
            
            let limit_price = ref_price.checked_mul(&limit_factor)
                .ok_or("Limit price calculation overflow")?;
            
            Ok(limit_price.to_scaled())
        } else {
            // Buying back short: limit price = reference * (1 + slippage)
            // We accept paying up to X% more than reference
            let one = FixedPoint::from_scaled(1_000_000_000_000_000_000i128);
            let slippage_factor = slippage_bps.checked_div(&hundred_percent)
                .ok_or("Slippage calculation overflow")?;
            
            let limit_factor = one.checked_add(&slippage_factor)
                .ok_or("Limit factor overflow")?;
            
            let limit_price = ref_price.checked_mul(&limit_factor)
                .ok_or("Limit price calculation overflow")?;
            
            Ok(limit_price.to_scaled())
        }
    }

    /// Record partial fill
    pub fn record_fill(&self, filled_scaled: i128) {
        self.flattened_size.fetch_add(filled_scaled, Ordering::Relaxed);
    }

    /// Check if flatten is complete
    pub fn check_flatten_complete(&self, target_size: i128) -> FlattenResult {
        let flattened = self.flattened_size.load(Ordering::Relaxed);
        
        if flattened >= target_size {
            self.current_state.store(SorRouterState::FlatteningComplete as u8, Ordering::SeqCst);
            self.operation_in_progress.store(false, Ordering::SeqCst);
            FlattenResult::Success
        } else if flattened > 0 {
            let remaining = target_size.saturating_sub(flattened);
            FlattenResult::Partial {
                filled_scaled: flattened,
                remaining_scaled: remaining,
            }
        } else {
            FlattenResult::Failed("No fills received")
        }
    }

    /// Mark operation as failed due to slippage
    pub fn mark_slippage_exceeded(&self) {
        self.slippage_exceeded.store(true, Ordering::SeqCst);
        self.current_state.store(SorRouterState::Failed as u8, Ordering::SeqCst);
        self.operation_in_progress.store(false, Ordering::SeqCst);
    }

    /// Get current state
    pub fn get_state(&self) -> SorRouterState {
        self.u8_to_state(self.current_state.load(Ordering::Relaxed))
    }

    /// Check if operation is in progress
    pub fn is_operation_in_progress(&self) -> bool {
        self.operation_in_progress.load(Ordering::Relaxed)
    }

    /// Get total flattened size
    pub fn get_flattened_size(&self) -> i128 {
        self.flattened_size.load(Ordering::Relaxed)
    }

    /// Reset router for next operation
    pub fn reset(&self) {
        self.current_state.store(SorRouterState::Idle as u8, Ordering::SeqCst);
        self.flattened_size.store(0, Ordering::Relaxed);
        self.operation_in_progress.store(false, Ordering::Relaxed);
        self.slippage_exceeded.store(false, Ordering::Relaxed);
    }

    fn u8_to_state(&self, val: u8) -> SorRouterState {
        match val {
            0 => SorRouterState::Idle,
            1 => SorRouterState::CalculatingLimits,
            2 => SorRouterState::SendingCancel,
            3 => SorRouterState::SendingLimitOrders,
            4 => SorRouterState::FlatteningComplete,
            5 => SorRouterState::Failed,
            _ => SorRouterState::Idle,
        }
    }
}

impl Default for OrphanLegSorRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_limit_price_calculation_long_flatten() {
        let router = OrphanLegSorRouter::new();
        
        let command = FlattenCommand {
            exchange_id: 1,
            symbol: [0; 16],
            size_scaled: 1_000_000_000_000_000_000i128,
            reference_price_scaled: 50_000 * 1_000_000_000_000_000_000i128, // $50k
            is_long: true,
        };
        
        let limit_price = router.calculate_limit_price(&command).unwrap();
        
        // Expected: $50k * (1 - 0.005) = $49,750
        let expected = 49_750 * 1_000_000_000_000_000_000i128;
        
        // Allow small rounding difference
        let diff = (limit_price - expected).abs();
        assert!(diff < 1_000_000_000i128); // Within 0.000001
    }

    #[test]
    fn test_limit_price_calculation_short_flatten() {
        let router = OrphanLegSorRouter::new();
        
        let command = FlattenCommand {
            exchange_id: 1,
            symbol: [0; 16],
            size_scaled: -1_000_000_000_000_000_000i128,
            reference_price_scaled: 50_000 * 1_000_000_000_000_000_000i128,
            is_long: false,
        };
        
        let limit_price = router.calculate_limit_price(&command).unwrap();
        
        // Expected: $50k * (1 + 0.005) = $50,250
        let expected = 50_250 * 1_000_000_000_000_000_000i128;
        
        let diff = (limit_price - expected).abs();
        assert!(diff < 1_000_000_000i128);
    }

    #[test]
    fn test_flatten_state_transitions() {
        let router = OrphanLegSorRouter::new();
        
        let command = FlattenCommand {
            exchange_id: 1,
            symbol: [0; 16],
            size_scaled: 1_000_000_000_000_000_000i128,
            reference_price_scaled: 50_000 * 1_000_000_000_000_000_000i128,
            is_long: true,
        };
        
        assert_eq!(router.get_state(), SorRouterState::Idle);
        
        assert!(router.initiate_flatten(&command));
        assert_eq!(router.get_state(), SorRouterState::CalculatingLimits);
        assert!(router.is_operation_in_progress());
        
        // Simulate fills
        router.record_fill(500_000_000_000_000_000i128);
        router.record_fill(500_000_000_000_000_000i128);
        
        let result = router.check_flatten_complete(1_000_000_000_000_000_000i128);
        assert!(matches!(result, FlattenResult::Success));
        assert!(!router.is_operation_in_progress());
    }
}
