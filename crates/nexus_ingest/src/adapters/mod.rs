//! Chapter 1: Live Exchange Adapters Module
//!
//! This module provides production-grade WebSocket adapters for connecting
//! to real exchange market data feeds.

pub mod binance_ws_manager;
pub mod reconnect_state_machine;

pub use binance_ws_manager::{
    BinanceWsConfig, ConnectionState, LiveExchangeAdapter, WsStats,
};
pub use reconnect_state_machine::{
    ReconnectPhase, ReconnectStateMachine, ReconnectStats, SubscriptionChannel,
};
