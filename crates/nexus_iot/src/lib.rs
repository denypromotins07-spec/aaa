//! NEXUS-IOT: Decentralized Physical Infrastructure (DePIN) Module
//! 
//! Ingests IoT telemetry from DePIN networks, processes edge compute filters,
//! validates sensor consensus, and generates alpha signals from physical data.

pub mod protocols;
pub mod edge;
pub mod consensus;
pub mod dsp;
pub mod alpha;
pub mod geospatial;
pub mod maritime;

// Re-export main types
pub use protocols::zero_alloc_mqtt_parser::{MqttPacket, MqttError};
pub use edge::wasm_jit_sandbox::{WasmJitSandbox, WasmSandboxConfig, WasmError};
pub use edge::coap_udp_listener::{CoapUdpListener, CoapPacket, CoapError};
pub use consensus::bft_sensor_validation::{BftSensorValidator, SensorReading, ConsensusError};
pub use consensus::sybil_detection_graph::{SybilDetectionGraph, SensorNode, SpatiotemporalCluster};
pub use consensus::spatiotemporal_quarantine::{SpatiotemporalQuarantine, QuarantineReason};
pub use dsp::simd_fft_vibration::{SimdFftProcessor, FftError};
pub use dsp::acoustic_anomaly_detector::{AcousticAnomalyDetector, AnomalyReport};
pub use alpha::predictive_maintenance_kalman::{PredictiveMaintenanceFilter, AssetState, MaintenanceAlert};
pub use geospatial::retail_foot_traffic::{RetailFootTrafficEstimator, FootTrafficEstimate};
pub use geospatial::connected_vehicle_telemetry::{ConnectedVehicleProcessor, EconomicActivityIndex};
pub use maritime::ais_draft_tracker::{AisDraftTracker, CommodityFlowEstimate};
