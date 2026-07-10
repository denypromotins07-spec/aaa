//! NEXUS-OMEGA Neuromorphic Computing Module
//! 
//! Stage 28: Spiking Neural Networks (SNNs), Event-Camera Vision,
//! Leaky Integrate-and-Fire (LIF) neurons, and Spike-Timing-Dependent Plasticity (STDP)

pub mod ingestion;
pub mod neurons;
pub mod encoding;
pub mod plasticity;

/// Re-exports for convenience
pub use ingestion::aer_zero_copy_parser::{AerPacket, AerParser, AerParseError};
pub use ingestion::spike_rasterizer::{SpikeRasterizer, SpikeTrain};
pub use neurons::simd_lif_engine::{SimdLifEngine, LifNeuronState};
pub use encoding::time_to_first_spike::{TtfsEncoder, SpikeLatency};
pub use plasticity::stdp_learning_rule::{StdpRule, StdpConfig};
