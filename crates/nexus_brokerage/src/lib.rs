//! Stage 13: Multi-Venue Orchestration, Prime Brokerage Ledger & Dark Pool Routing
//! 
//! This crate provides:
//! - Shadow double-entry ledger for cross-broker reconciliation
//! - Cross-margin netting engine
//! - FIX 4.4 zero-copy parsing
//! - Dark pool IOI state machines
//! - Venue normalization and toxicity routing

pub mod ledger;
pub mod margin;
pub mod reconciliation;
pub mod dark_pools;

pub use ledger::shadow_double_entry::*;
pub use margin::cross_margin_netting::*;
pub use reconciliation::fix_position_parser::*;
pub use dark_pools::fix_tag_value_parser::*;
pub use dark_pools::ioi_state_machine::*;
pub use dark_pools::conditional_pegging::*;
