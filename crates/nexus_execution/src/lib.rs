//! NEXUS-OMEGA Execution Library
//! 
//! Smart Order Routing, Iceberg Sniper, Queue Position Tracking,
//! and zero-allocation WAPI execution.

pub mod wapi;
pub mod algos;
pub mod sor;

pub use wapi::{
    WapiConnectionManager, WapiConfig, ConnectionState, ConnectionEvent,
    ZeroAllocHmacSigner, HmacBuffer, HmacSignerError, WapiRequestBuilder,
    RecvWindowClockSyncDaemon, ClockSyncConfig, ClockSyncStatus,
};

pub use algos::{
    IcebergSniper, IcebergConfig, IcebergState, IcebergSlice, SliceState, OrderSide,
    QueuePositionTracker, QueueTrackerConfig, QueuePriority, QueueAction, IcebergDetection,
};

pub use sor::{
    SmartOrderRouter, SorConfig, ExecutionDecision, RoutingStrategy,
};

/// Execution Engine - main orchestrator
pub struct ExecutionEngine {
    pub wapi_manager: Option<WapiConnectionManager>,
    pub clock_sync: Option<RecvWindowClockSyncDaemon>,
    pub iceberg_sniper: Option<IcebergSniper>,
    pub queue_tracker: Option<QueuePositionTracker>,
    pub smart_router: Option<SmartOrderRouter>,
}

impl ExecutionEngine {
    pub fn new() -> Self {
        Self {
            wapi_manager: None,
            clock_sync: None,
            iceberg_sniper: None,
            queue_tracker: None,
            smart_router: None,
        }
    }

    pub fn with_full_config(
        wapi_config: WapiConfig,
        sor_config: SorConfig,
        iceberg_config: IcebergConfig,
        queue_config: QueueTrackerConfig,
    ) -> Self {
        Self {
            wapi_manager: None, // Initialized asynchronously
            clock_sync: Some(RecvWindowClockSyncDaemon::new(ClockSyncConfig::default())),
            iceberg_sniper: Some(IcebergSniper::new(iceberg_config, 0)),
            queue_tracker: Some(QueuePositionTracker::new(queue_config)),
            smart_router: Some(SmartOrderRouter::new(sor_config)),
        }
    }
}

impl Default for ExecutionEngine {
    fn default() -> Self {
        Self::new()
    }
}
