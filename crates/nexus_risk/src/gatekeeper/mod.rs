//! Chapter 1: Pre-Trade Risk Gatekeeper - Zero-Allocation Hot Path
//! 
//! This module implements the core gatekeeper components that sit strictly between
//! the Alpha Engine and the Execution OMS. Every outbound order MUST pass through
//! this filter before reaching the WAPI Signer.
//! 
//! CRITICAL: The Execution OMS CANNOT sign or send an order unless the Risk Gatekeeper
//! returns a `RiskApproved` token. This is enforced at the type level.

pub mod pre_trade_filter;
pub mod price_collar;
pub mod position_accumulator;

// Re-export key types for integration
pub use pre_trade_filter::{PreTradeRiskGatekeeper, RiskApproved, RiskViolation};
pub use price_collar::PriceCollarValidator;
pub use position_accumulator::AtomicPositionAccumulator;
