// STAGE 23: Computational Law, RegTech Compliance & Immutable Audit Trails
// Root library file for nexus_legal crate

pub mod mar;
pub mod compliance;
pub mod zk;
pub mod audit;

// Re-export main types
pub use mar::wash_trade_graph::{WashTradeGraph, WashTradeConfig, ExecutionNode, ExecutionId, AssetClass, Side};
pub use mar::tarjan_cycle_detector::TarjanDetector;
pub use mar::spoofing_self_check::{SpoofingMonitor, SpoofingConfig};

pub use compliance::dag_rule_engine::{ComplianceState, ComplianceFlag, ComplianceResult, CompiledDag, DagBuilder};
pub use compliance::regsho_locator::{RegShoEngine, RegShoConfig, LocateRecord, TickTestState};
pub use compliance::mifid_tick_size::{MifidEngine, MifidConfig, TickSizeTable, OrderSide};

pub use zk::halo2_compliance_circuit::{MaxDrawdownCircuit, SharpeRatioCircuit, OtrCircuit, ComplianceProof};

pub use audit::lock_free_merkle::{LockFreeMerkleTree, AuditEvent, MerkleProof};
