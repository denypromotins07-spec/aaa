//! OMS Reconciliation Module
//! 
//! Handles execution report processing and race condition resolution.

pub mod execution_report_handler;
pub mod race_condition_resolver;

pub use execution_report_handler::{
    ExecutionReport,
    ExecutionReportType,
    ExecutionReportResult,
    ExecutionReportReconciler,
    ReconciliationStats,
};

pub use race_condition_resolver::{
    RaceConditionResolver,
    RaceConditionResolution,
    AuthoritativeAction,
    PendingCancel,
    FillEvent,
    TimestampNs,
    ResolverStats,
};
