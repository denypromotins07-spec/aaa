//! NEXUS-OMEGA Stage 30: Holographic Data Storage, DNA Computing & 5D Optical Memory Archiving
//! 
//! This crate implements exotic physics-based storage mediums for century-scale data archival:
//! - Holographic Volumetric Storage with Bragg grating simulation
//! - Synthetic DNA Data Storage with biological error correction
//! - 5D Optical Nano-grating (Superman Crystal) memory
//! - Cold Storage orchestration for regime-based archival

#![no_std]

extern crate alloc;

pub mod holographic;
pub mod dna;
pub mod optical_5d;
pub mod orchestration;

// Re-export main types for convenience
pub use holographic::{VolumetricPageEncoder, BraggGratingSimulator, HolographicMultiplexer};
pub use dna::{NucleotideEncoder, HomopolymerAvoidanceEncoder, ReedSolomonGF4};
pub use optical_5d::{FemtosecondVoxelMapper, NanoGratingOrientation, JonesCalculusDecoder};
pub use orchestration::{EternalArchiveManager, RegimeArchivalPolicy, ColdRetrievalApi};
