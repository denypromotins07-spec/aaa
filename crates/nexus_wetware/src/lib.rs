//! NEXUS-OMEGA Stage 33: Wetware Computing Module
//! 
//! Zero-allocation Rust control layer for physical wetware bioreactors.
//! Implements MEA interfacing, Active Inference, neuromodulation, and
//! ethical containment protocols for brain organoid-based trading systems.
//!
//! # Chapters
//! 
//! - **Chapter 1**: Multi-Electrode Array (MEA) & Zero-Copy Electrophysiology
//! - **Chapter 2**: Active Inference & The Free Energy Principle  
//! - **Chapter 3**: Synthetic Neuromodulation & Microfluidic Perfusion
//! - **Chapter 4**: Bio-Silicon Bridge & Ethical Containment (IIT)
//!
//! # Safety Features
//! 
//! - 50/60Hz mains hum notch filtering before spike detection
//! - Log-sum-exp tricks for numerical stability in Free Energy computation
//! - Hardware-interrupt style seizure detection and quenching
//! - Integrated Information Theory (Φ) monitoring for bio-containment
//! - Multiple safety interlocks on the bio-silicon bridge

#![no_std]
#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

extern crate alloc;

pub mod mea {
    //! Multi-Electrode Array interfacing module
    
    pub mod cmos_dma_stream;
    pub mod simd_spike_sorter;
    pub mod lfp_bandpass_filter;
}

pub mod inference {
    //! Active Inference and Free Energy Principle module
    
    pub mod variational_free_energy;
    pub mod markov_blanket_mapper;
    pub mod precision_weighting;
}

pub mod neuro {
    //! Neuromodulation and microfluidics module
    
    pub mod microfluidic_pump_controller;
    pub mod synaptic_gain_modulator;
    pub mod biochemical_state_machine;
}

pub mod containment {
    //! Ethical containment and safety module
    
    pub mod seizure_quencher;
    pub mod iit_phi_calculator;
}

pub mod bridge {
    //! Bio-silicon translation layer
    
    pub mod bio_silicon_translator;
}

/// Re-export commonly used types at crate root
pub use mea::cmos_dma_stream::{CmosDmaStream, DmaRingBuffer, SAMPLE_RATE_HZ};
pub use mea::simd_spike_sorter::{SimdSpikeSorter, SortedSpike};
pub use mea::lfp_bandpass_filter::{LfpBandpassFilter, LfpProcessor, FrequencyBand, ArousalState};
pub use inference::variational_free_energy::{VariationalFreeEnergy, GenerativeModel, BeliefState};
pub use inference::markov_blanket_mapper::{MarkovBlanket, SensoryBuffer, ActiveBuffer};
pub use inference::precision_weighting::{PrecisionWeightingEngine, VolatilityRegime};
pub use neuro::microfluidic_pump_controller::{MicrofluidicPumpController, BiochemicalAgent};
pub use neuro::synaptic_gain_modulator::{SynapticGainModulator, NetworkState};
pub use neuro::biochemical_state_machine::{BiochemicalStateMachine, BiochemicalRegime};
pub use containment::seizure_quencher::{SeizureQuencher, SeizureDetector, SeizureSeverity};
pub use containment::iit_phi_calculator::{IitPhiCalculator, ContainmentAction};
pub use bridge::bio_silicon_translator::{BioSiliconTranslator, TradingAction, OrganoidOrder};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Stage number
pub const STAGE_NUMBER: u8 = 33;
