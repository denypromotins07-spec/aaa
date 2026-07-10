//! Nexus Routing Library - Smart Order Routing and Latency Arbitrage

pub mod sor_engine;
pub mod venue_latency_model;
pub mod stale_quote_sniper;

pub use sor_engine::{SorEngine, VenueId, VenueQuote, VenueMetrics, MAX_VENUES};
pub use venue_latency_model::VenueLatencyModel;
pub use stale_quote_sniper::{StaleQuoteSniper, StaleQuoteOpportunity, SymbolState, MAX_SYMBOLS};
