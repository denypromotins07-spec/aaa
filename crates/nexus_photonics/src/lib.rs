//! NEXUS-OMEGA Stage 32: Silicon Photonics, Optical Matrix Multiplication & Light-Speed Interconnects
//!
//! This crate provides:
//! - Chapter 1: Optical Neural Networks (ONN) & MZI Mesh Compilation
//! - Chapter 2: Microring Resonator Weight Banks & WDM Crossbars
//! - Chapter 3: Photonic Time-Stretch ADC & Femtosecond Timestamping
//! - Chapter 4: Co-Packaged Optics (CPO) & Thermo-Optic Resonance Locking

pub mod adc;
pub mod compute;
pub mod onn;
pub mod tuning;

/// Re-export key types for convenience
pub use onn::clements_decomposition::ClementsDecomposer;
pub use onn::mzi_mesh_compiler::MziMeshCompiler;
pub use compute::microring_weight_bank::MicroringWeightBank;
pub use compute::wdm_crossbar_router::WdmCrossbarRouter;
pub use adc::photonic_time_stretch::PhotonicTimeStretchAdc;
pub use adc::femtosecond_timestamp::FemtosecondTimestampEngine;
pub use tuning::thermo_optic_phase_shifter::ThermoOpticPhaseShifter;
pub use tuning::dithering_lock_in_amplifier::DitheringLockInAmplifier;
