//! NEXUS-OMEGA Stage 48: Dyson Sphere Economy, Stellar Lifting & Kardashev Type II Alpha
//! 
//! This crate implements the physics and economics of a Kardashev Type II civilization:
//! - Magnetohydrodynamic stellar lifting for plasma extraction
//! - Symplectic N-body integration for Dyson swarm orbital mechanics  
//! - Stefan-Boltzmann thermodynamic manifolds for Matrioshka brains
//! - Nicoll-Dyson phased arrays for interstellar energy beaming

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![forbid(unsafe_code)]

pub mod stellar;
pub mod orbital;
pub mod thermodynamics;
pub mod beaming;
pub mod derivatives;
pub mod arbitrage;

/// Re-export commonly used types
pub use stellar::mhd_plasma_solver::{MHDState, MHDSolver, MHDError, StellarConstants};
pub use stellar::magnetic_containment::{MagneticContainmentField, PlasmaExtractor};
pub use orbital::symplectic_nbody_integrator::{BodyState, LeapfrogIntegrator, YoshidaIntegrator};
pub use orbital::resonance_stabilizer::{OrbitalSlot, ResonanceStabilizer};
pub use thermodynamics::stefan_boltzmann_manifold::{MatrioshkaShell, StefanBoltzmannManifold};
pub use thermodynamics::carnot_compute_yield::{CarnotComputePricer, ComputeClass};
pub use beaming::nicoll_dyson_phased_array::{NicollDysonArray, PhasedArrayConfig};
pub use beaming::relativistic_doppler_attenuation::{RelativisticProbe, AttenuationCalculator};
pub use arbitrage::interstellar_energy_carry::{InterstellarEnergyArbitrage, BussardRamjet};
