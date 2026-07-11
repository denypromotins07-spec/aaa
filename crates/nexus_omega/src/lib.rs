//! NEXUS-OMEGA Stage 45: The Omega Point, Teleological Attractors & Final State Boundary Conditions
//! 
//! This crate implements advanced financial mathematics based on:
//! - Takens' Embedding Theorem for phase-space reconstruction
//! - Lyapunov spectrum analysis for chaos detection
//! - Kaplan-Yorke dimension for fractal attractor characterization
//! - Wick-rotated Kolmogorov equations for imaginary-time pricing
//! - Hartle-Hawking no-boundary proposal applied to derivatives
//! - Landauer principle for thermodynamic limits on computation
//! - Eschatological derivatives for end-of-history hedging

#![no_std]

extern crate alloc;

pub mod attractors {
    pub mod takens_phase_space;
    pub mod lyapunov_spectrum;
    pub mod kaplan_yorke_dimension;
}

pub mod teleology {
    pub mod wick_rotated_kolmogorov;
    pub mod imaginary_time_propagator;
    pub mod final_state_boundary;
}

pub mod thermodynamics {
    pub mod landauer_limit_calculator;
    pub mod omega_point_metric;
    pub mod strategy_mutation_trigger;
}

pub mod derivatives {
    pub mod eschatological_option_pricer;
    pub mod paradigm_transition_swap;
}

pub mod hedging {
    pub mod teleological_barbell;
}

// Re-export main types for convenience
pub use attractors::takens_phase_space::{TakensEmbedding, TakensConfig, MutualInformation, FoldingArtifactDetector};
pub use attractors::lyapunov_spectrum::{LyapunovCalculator, LyapunovConfig, LyapunovSpectrum, ChaosDetector};
pub use attractors::kaplan_yorke_dimension::{KaplanYorkeCalculator, KaplanYorkeConfig, KaplanYorkeResult, MarketRigidityDetector};

pub use teleology::wick_rotated_kolmogorov::{WickKolmogorovSolver, WickKolmogorovConfig, Complex, HartleHawkingPricer};
pub use teleology::imaginary_time_propagator::{ImaginaryTimePropagator, ImaginaryTimeConfig, PathIntegralCalculator};
pub use teleology::final_state_boundary::{FinalStateBoundarySolver, FinalStateConfig, FinalBoundaryType};

pub use thermodynamics::landauer_limit_calculator::{LandauerCalculator, LandauerResult, FemtoJoule, MarketEfficiencyTracker};
pub use thermodynamics::omega_point_metric::{OmegaPointMetric, OmegaPointConfig, OmegaPointResult};
pub use thermodynamics::strategy_mutation_trigger::{StrategyMutationTrigger, MutationTriggerConfig};

pub use derivatives::eschatological_option_pricer::{EschatologicalOptionPricer, EschatologicalEvent};
pub use derivatives::paradigm_transition_swap::{ParadigmTransitionSwap, ParadigmShiftDetection};

pub use hedging::teleological_barbell::{TeleologicalBarbellEngine, BarbellPortfolio, BarbellAnalysis};
