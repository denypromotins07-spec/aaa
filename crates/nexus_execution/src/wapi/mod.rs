//! WAPI Module - WebSocket API for Exchange Execution
//! 
//! Provides zero-allocation HMAC signing, clock synchronization,
//! and resilient WebSocket connection management.

pub mod wapi_connection_manager;
pub mod zero_alloc_hmac_signer;
pub mod recvwindow_clock_sync;

pub use wapi_connection_manager::{
    WapiConnectionManager,
    WapiConfig,
    ConnectionState,
    ConnectionEvent,
};

pub use zero_alloc_hmac_signer::{
    ZeroAllocHmacSigner,
    HmacBuffer,
    HmacSignerError,
    WapiRequestBuilder,
    HMAC_OUTPUT_SIZE,
    HMAC_HEX_SIZE,
};

pub use recvwindow_clock_sync::{
    RecvWindowClockSyncDaemon,
    ClockSyncConfig,
    ClockSyncStatus,
    ClockSyncStats,
};
