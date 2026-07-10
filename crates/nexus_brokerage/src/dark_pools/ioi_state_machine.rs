//! Indication of Interest (IOI) State Machine for Dark Pool Liquidity Discovery
//! 
//! Implements a state machine for broadcasting encrypted IOIs to dark pools
//! to find hidden block liquidity without leaking trading intent.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum IOIError {
    #[error("Invalid state transition: from {from:?} to {to:?}")]
    InvalidTransition { from: IOIState, to: IOIState },
    #[error("IOI expired before execution")]
    IOIExpired,
    #[error("Price improvement not achieved: dark_price {dark_price}, lit_nbbo {lit_nbbo}")]
    NoPriceImprovement { dark_price: i64, lit_nbbo: i64 },
    #[error("Toxicity threshold exceeded: toxicity {toxicity} > threshold {threshold}")]
    ToxicityExceeded { toxicity: f64, threshold: f64 },
    #[error("Order size exceeds dark pool maximum")]
    SizeExceeded,
}

/// IOI states representing the lifecycle of a dark pool inquiry
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IOIState {
    /// Initial state - preparing IOI parameters
    Initializing,
    /// IOI broadcast to dark pools, awaiting responses
    Broadcasted,
    /// Received conditional order indications from dark pools
    ResponsesReceived,
    /// Evaluating price improvement against lit NBBO
    EvaluatingImprovement,
    /// Executing against selected dark pool
    Executing,
    /// Execution complete
    Completed,
    /// Cancelled or expired
    Cancelled,
}

/// Indication of Interest structure
#[derive(Debug, Clone)]
pub struct IndicationOfInterest {
    pub ioi_id: u64,
    pub asset_id: u32,
    pub side: Side,
    pub quantity: i64,
    pub min_qty: i64,
    pub max_qty: i64,
    pub peg_offset: i64, // Fixed-point offset from mid
    pub improvement_required: i64, // Minimum price improvement in fixed-point
    pub expiry: Instant,
    pub state: IOIState,
    pub target_dark_pools: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Response from a dark pool to an IOI
#[derive(Debug, Clone)]
pub struct DarkPoolResponse {
    pub dark_pool_id: u32,
    pub available_qty: i64,
    pub peg_price: i64, // Fixed-point pegged price
    pub improvement_offered: i64, // Fixed-point improvement over NBBO
    pub conditional_order_id: u64,
    pub expiry: Instant,
}

/// IOI State Machine with atomic state transitions
pub struct IOIStateMachine {
    ioi: IndicationOfInterest,
    responses: Vec<DarkPoolResponse>,
    state: AtomicU64, // Encoded IOIState as u64
    created_at: Instant,
    updated_at: AtomicU64, // Microseconds since epoch
    is_active: AtomicBool,
}

impl IOIStateMachine {
    pub fn new(ioi: IndicationOfInterest) -> Self {
        let now = Instant::now();
        Self {
            ioi,
            responses: Vec::new(),
            state: AtomicU64::new(IOIState::Initializing as u64),
            created_at: now,
            updated_at: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
        }
    }

    /// Get current state
    pub fn get_state(&self) -> IOIState {
        match self.state.load(Ordering::Acquire) {
            0 => IOIState::Initializing,
            1 => IOIState::Broadcasted,
            2 => IOIState::ResponsesReceived,
            3 => IOIState::EvaluatingImprovement,
            4 => IOIState::Executing,
            5 => IOIState::Completed,
            6 => IOIState::Cancelled,
            _ => IOIState::Cancelled, // Default to cancelled for invalid states
        }
    }

    /// Attempt state transition with validation
    fn transition(&self, new_state: IOIState) -> Result<(), IOIError> {
        let current = self.get_state();
        
        // Validate state transitions
        let valid = match (current, new_state) {
            (IOIState::Initializing, IOIState::Broadcasted) => true,
            (IOIState::Broadcasted, IOIState::ResponsesReceived) => true,
            (IOIState::Broadcasted, IOIState::Cancelled) => true,
            (IOIState::ResponsesReceived, IOIState::EvaluatingImprovement) => true,
            (IOIState::ResponsesReceived, IOIState::Cancelled) => true,
            (IOIState::EvaluatingImprovement, IOIState::Executing) => true,
            (IOIState::EvaluatingImprovement, IOIState::Cancelled) => true,
            (IOIState::Executing, IOIState::Completed) => true,
            (IOIState::Executing, IOIState::Cancelled) => true,
            _ => false,
        };

        if !valid {
            return Err(IOIError::InvalidTransition {
                from: current,
                to: new_state,
            });
        }

        self.state.store(new_state as u64, Ordering::Release);
        self.updated_at.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_micros() as u64,
            Ordering::Release,
        );
        
        Ok(())
    }

    /// Transition to Broadcasted state
    pub fn broadcast(&self) -> Result<(), IOIError> {
        if self.is_expired() {
            return Err(IOIError::IOIExpired);
        }
        self.transition(IOIState::Broadcasted)
    }

    /// Add response from dark pool
    pub fn add_response(&self, response: DarkPoolResponse) -> Result<(), IOIError> {
        if self.get_state() != IOIState::Broadcasted {
            return Err(IOIError::InvalidTransition {
                from: self.get_state(),
                to: IOIState::ResponsesReceived,
            });
        }

        // Validate response
        if response.available_qty < self.ioi.min_qty {
            return Err(IOIError::SizeExceeded);
        }

        self.responses.push(response);
        Ok(())
    }

    /// Mark responses as received
    pub fn finalize_responses(&self) -> Result<(), IOIError> {
        if self.responses.is_empty() {
            return Err(IOIError::InvalidTransition {
                from: self.get_state(),
                to: IOIState::ResponsesReceived,
            });
        }
        self.transition(IOIState::ResponsesReceived)
    }

    /// Evaluate price improvement for a specific response
    pub fn evaluate_improvement(
        &self,
        response_idx: usize,
        lit_best_bid: i64,
        lit_best_ask: i64,
    ) -> Result<bool, IOIError> {
        if self.get_state() != IOIState::ResponsesReceived {
            return Err(IOIError::InvalidTransition {
                from: self.get_state(),
                to: IOIState::EvaluatingImprovement,
            });
        }

        let response = self.responses.get(response_idx)
            .ok_or(IOIError::SizeExceeded)?;

        // Calculate lit NBBO midpoint
        let lit_midpoint = lit_best_bid.saturating_add(lit_best_ask) / 2;

        // Check if dark pool offers price improvement
        let has_improvement = match self.ioi.side {
            Side::Buy => {
                // For buys: dark price should be below lit ask
                response.peg_price < lit_best_ask.saturating_sub(self.ioi.improvement_required)
            }
            Side::Sell => {
                // For sells: dark price should be above lit bid
                response.peg_price > lit_best_bid.saturating_add(self.ioi.improvement_required)
            }
        };

        if !has_improvement {
            return Err(IOIError::NoPriceImprovement {
                dark_price: response.peg_price,
                lit_nbbo: lit_midpoint,
            });
        }

        Ok(true)
    }

    /// Select best response and begin execution
    pub fn select_and_execute(&self, response_idx: usize) -> Result<&DarkPoolResponse, IOIError> {
        self.transition(IOIState::EvaluatingImprovement)?;
        self.transition(IOIState::Executing)?;

        self.responses.get(response_idx)
            .ok_or(IOIError::SizeExceeded)
    }

    /// Mark execution complete
    pub fn complete(&self) -> Result<(), IOIError> {
        self.transition(IOIState::Completed)?;
        self.is_active.store(false, Ordering::Release);
        Ok(())
    }

    /// Cancel the IOI
    pub fn cancel(&self) -> Result<(), IOIError> {
        self.transition(IOIState::Cancelled)?;
        self.is_active.store(false, Ordering::Release);
        Ok(())
    }

    /// Check if IOI has expired
    pub fn is_expired(&self) -> bool {
        Instant::now() > self.ioi.expiry
    }

    /// Check if IOI is still active
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire) && !self.is_expired()
    }

    /// Get number of responses received
    pub fn response_count(&self) -> usize {
        self.responses.len()
    }

    /// Get best response by improvement offered
    pub fn get_best_response(&self) -> Option<&DarkPoolResponse> {
        self.responses.iter().max_by(|a, b| {
            a.improvement_offered.cmp(&b.improvement_offered)
        })
    }
}

/// Builder for creating IOIs with proper defaults
pub struct IOIBuilder {
    ioi_id: u64,
    asset_id: u32,
    side: Side,
    quantity: i64,
    min_qty: i64,
    max_qty: i64,
    peg_offset: i64,
    improvement_required: i64,
    ttl: Duration,
    target_dark_pools: Vec<u32>,
}

impl IOIBuilder {
    pub fn new(ioi_id: u64, asset_id: u32, side: Side, quantity: i64) -> Self {
        Self {
            ioi_id,
            asset_id,
            side,
            quantity,
            min_qty: quantity / 10, // 10% minimum fill
            max_qty: quantity,
            peg_offset: 0,
            improvement_required: 100, // 100 micro-units minimum improvement
            ttl: Duration::from_secs(30),
            target_dark_pools: Vec::new(),
        }
    }

    pub fn peg_offset(mut self, offset: i64) -> Self {
        self.peg_offset = offset;
        self
    }

    pub fn improvement_required(mut self, improvement: i64) -> Self {
        self.improvement_required = improvement;
        self
    }

    pub fn ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    pub fn target_pools(mut self, pools: Vec<u32>) -> Self {
        self.target_dark_pools = pools;
        self
    }

    pub fn build(self) -> IndicationOfInterest {
        IndicationOfInterest {
            ioi_id: self.ioi_id,
            asset_id: self.asset_id,
            side: self.side,
            quantity: self.quantity,
            min_qty: self.min_qty,
            max_qty: self.max_qty,
            peg_offset: self.peg_offset,
            improvement_required: self.improvement_required,
            expiry: Instant::now() + self.ttl,
            state: IOIState::Initializing,
            target_dark_pools: self.target_dark_pools,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ioi_lifecycle() {
        let ioi = IOIBuilder::new(1, 100, Side::Buy, 10000)
            .improvement_required(50)
            .ttl(Duration::from_secs(60))
            .build();

        let machine = IOIStateMachine::new(ioi);

        // Initial state
        assert_eq!(machine.get_state(), IOIState::Initializing);
        assert!(machine.is_active());

        // Broadcast
        assert!(machine.broadcast().is_ok());
        assert_eq!(machine.get_state(), IOIState::Broadcasted);

        // Add response
        let response = DarkPoolResponse {
            dark_pool_id: 1,
            available_qty: 5000,
            peg_price: 50000000,
            improvement_offered: 75,
            conditional_order_id: 100,
            expiry: Instant::now() + Duration::from_secs(10),
        };
        assert!(machine.add_response(response).is_ok());
        assert!(machine.finalize_responses().is_ok());
        assert_eq!(machine.get_state(), IOIState::ResponsesReceived);
    }

    #[test]
    fn test_invalid_transition() {
        let ioi = IOIBuilder::new(1, 100, Side::Buy, 10000).build();
        let machine = IOIStateMachine::new(ioi);

        // Try invalid transition: Initializing -> Executing
        let result = machine.transition(IOIState::Executing);
        assert!(matches!(result, Err(IOIError::InvalidTransition { .. })));
    }

    #[test]
    fn test_cancellation() {
        let ioi = IOIBuilder::new(1, 100, Side::Buy, 10000).build();
        let machine = IOIStateMachine::new(ioi);

        machine.broadcast().unwrap();
        assert!(machine.cancel().is_ok());
        assert_eq!(machine.get_state(), IOIState::Cancelled);
        assert!(!machine.is_active());
    }
}
