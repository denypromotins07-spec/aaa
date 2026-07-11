//! NEXUS-OMEGA Stage 43: Memetic Warfare, Narrative Topology & Cognitive Contagion Routing
//! 
//! This crate implements advanced financial narrative analysis using:
//! - Epidemiological SIR models for meme propagation
//! - Riemannian manifold curvature for semantic paradigm shifts  
//! - Soros Reflexivity theory via coupled ODEs
//! - Spectral graph analysis for botnet detection

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![forbid(unsafe_code)]

pub mod epidemiology {
    //! Chapter 1: Memetic Epidemiology & Financial SIR Models
    
    pub mod financial_sir_ode;
    pub mod viral_r0_calculator;
    pub mod stiff_runge_kutta;
    
    pub use financial_sir_ode::*;
    pub use viral_r0_calculator::*;
    pub use stiff_runge_kutta::*;
}

pub mod topology {
    //! Chapter 2: Narrative Topology & Semantic Manifold Shifts
    
    pub mod semantic_manifold_curvature;
    pub mod ricci_flow_evolution;
    pub mod narrative_paradigm_shift;
    
    pub use semantic_manifold_curvature::*;
    pub use ricci_flow_evolution::*;
    pub use narrative_paradigm_shift::*;
}

pub mod reflexivity {
    //! Chapter 3: Soros Reflexivity & Coupled Narrative-Liquidity ODEs
    
    pub mod coupled_ode_solver;
    pub mod jacobian_eigenvalue;
    pub mod bubble_inflection_detector;
    
    pub use coupled_ode_solver::*;
    pub use jacobian_eigenvalue::*;
    pub use bubble_inflection_detector::*;
}

pub mod warfare {
    //! Chapter 4: Memetic Warfare & Spectral Astroturfing Detection
    
    pub mod spectral_astroturfing_detector;
    pub mod fiedler_vector_graph;
    pub mod botnet_pump_short;
    
    pub use spectral_astroturfing_detector::*;
    pub use fiedler_vector_graph::*;
    pub use botnet_pump_short::*;
}

/// Re-export key types at crate root for convenience
pub use epidemiology::{FinancialSirModel, ViralR0Calculator, SirState, RadauIIASolver};
pub use topology::{RicciCurvatureCalculator, ManifoldConfig, ParadigmShiftDetector};
pub use reflexivity::{CoupledODESolver, LyapunovAnalyzer, BubbleInflectionDetector};
pub use warfare::{SpectralAstroturfingDetector, FiedlerVectorAnalyzer, BotnetPumpShortStrategy};
