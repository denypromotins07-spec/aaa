//! Nexus Simulation - Level-of-Detail (LOD) & Volatility Surface Rendering
//! 
//! Exploits exchange compute budget limitations that cause reduced fidelity
//! updates for illiquid options and deep OTM strikes.

pub mod lod_fidelity_scanner;
pub mod compute_budget_profiler;
pub mod stale_surface_exploiter;
