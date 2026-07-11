//! NEXUS-OMEGA Stage 36: Space Economy, Orbital Debris Contagion & Asteroid Mining Derivatives
//! 
//! This crate implements zero-allocation Rust solvers for:
//! - SGP4/TLE orbital propagation
//! - Conjunction Data Message covariance matrices
//! - Delta-V discounted commodity forwards
//! - Kessler Syndrome collisional cascade PDEs

#![no_std]

extern crate alloc;

pub mod orbital {
    pub mod sgp4_propagator;
    pub mod conjunction_covariance;
}

pub mod derivatives {
    pub mod station_keeping_pricer;
}

pub mod telemetry {
    pub mod acoustic_vibration_filter;
    pub mod bayesian_success_predictor;
}

pub mod alpha {
    pub mod payload_capacity_arb;
}

pub mod mining {
    pub mod spectroscopic_yield;
    pub mod delta_v_manifold_calculator;
    pub mod asteroid_forward_pricer;
}

pub mod contagion {
    pub mod kessler_boltzmann_pde;
    pub mod debris_cloud_expansion;
    pub mod terrestrial_fallback_arb;
}

// Re-export main types for convenience
pub use orbital::sgp4_propagator::{SGP4State, TLE, ECIState, SGP4Error};
pub use orbital::conjunction_covariance::{ConjunctionEngine, CovarianceMatrix3D, ConjunctionData};
pub use mining::spectroscopic_yield::{SpectralType, MineralComposition};
pub use mining::delta_v_manifold_calculator::{DeltaVBudget, OrbitalElements};
pub use contagion::kessler_boltzmann_pde::{DebrisDensityField, KesslerBoltzmannSolver};
