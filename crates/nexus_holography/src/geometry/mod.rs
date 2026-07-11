//! Chapter 1: AdS/CFT Market Correspondence & Hyperbolic Bulk Geometry
//! 
//! Models the global liquidity universe using Anti-de Sitter (AdS) space geometry.
//! Lit Exchanges are the conformal boundary; Dark Pools form the hyperbolic "Bulk".

pub mod ads_cft_dictionary;
pub mod poincare_metric;
pub mod boundary_operator_map;

pub use ads_cft_dictionary::*;
pub use poincare_metric::*;
pub use boundary_operator_map::*;
