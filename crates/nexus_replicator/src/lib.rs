//! Stage 47: The Von Neumann Probe, Self-Replicating Capital & Interstellar Propagation
//!
//! This crate implements:
//! - Cryptographic Quines (self-replicating smart contracts)
//! - Autonomous Resource Acquisition engines
//! - Orbital Dyson Swarm micro-satellite physics
//! - Shannon-Hartley laser propagation mechanics

#![no_std]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]

extern crate alloc;

pub mod digital;
pub mod resources;
pub mod physical;
pub mod interstellar;

/// Re-exports for convenience
pub use digital::{EvmQuineGenerator, SelfReferentialBytecode, CrossChainAtomicSpawner};
pub use resources::{AutonomousResourceAcquirer, TerritorialDefender};
pub use physical::{OrbitalSimulator, SolarFluxHarvester, ThermalManager};
pub use interstellar::{ShannonHartleyEncoder, DopplerCompensator, PlasmaDispersionModel};

/// Global constants for Stage 47
pub mod constants {
    /// Speed of light in m/s
    pub const C: f64 = 299_792_458.0;
    
    /// Gravitational constant G in m^3 kg^-1 s^-2
    pub const G: f64 = 6.674_30e-11;
    
    /// Earth mass in kg
    pub const EARTH_MASS: f64 = 5.972e24;
    
    /// Earth radius in meters
    pub const EARTH_RADIUS: f64 = 6_371_000.0;
    
    /// Earth J2 zonal harmonic coefficient
    pub const EARTH_J2: f64 = 1.082_626_68e-3;
    
    /// Solar constant at 1 AU in W/m^2
    pub const SOLAR_CONSTANT: f64 = 1361.0;
    
    /// Boltzmann constant in J/K
    pub const K_BOLTZMANN: f64 = 1.380_649e-23;
    
    /// Planck constant in J·s
    pub const H_PLANCK: f64 = 6.626_070_15e-34;
    
    /// Stefan-Boltzmann constant in W m^-2 K^-4
    pub const SIGMA_STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;
    
    /// Vacuum permittivity in F/m
    pub const EPSILON_0: f64 = 8.854_187_8128e-12;
    
    /// Maximum EVM bytecode size (24 KiB limit)
    pub const MAX_EVM_BYTECODE_SIZE: usize = 24 * 1024;
    
    /// Standard EVM block gas limit
    pub const STANDARD_BLOCK_GAS_LIMIT: u64 = 30_000_000;
    
    /// Safety buffer for gas estimation (10%)
    pub const GAS_ESTIMATION_BUFFER: f64 = 1.1;
}
