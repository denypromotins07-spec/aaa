//! NEXUS Longevity Module - Stage 37 of 50
//! 
//! Longevity Economics, Actuarial Genomics & Bio-Financial Derivatives
//! 
//! This module implements zero-allocation Rust solvers for:
//! - Epigenetic clock acceleration (Horvath, Hannum, PhenoAge)
//! - Polygenic risk score (PRS) manifolds with elastic net regularization
//! - Stochastic mortality models (Lee-Carter, CBD) with Kalman filtering
//! - Longevity derivative pricing (mortality bonds, survival swaps)
//! - Bio-financial arbitrage between genomic signals and actuarial tables

#![no_std]

extern crate alloc;

pub mod genomics {
    pub mod simd_fasta_parser;
    pub mod elastic_net_prs;
    pub mod homomorphic_privacy_router;
}

pub mod epigenetics {
    pub mod horvath_clock_solver;
    pub mod methylation_beta_processor;
}

pub mod mortality {
    pub mod lee_carter_kalman;
    pub mod cbd_older_age_extension;
    pub mod cohort_correlation_cholesky;
}

pub mod derivatives {
    pub mod affine_longevity_bond;
    pub mod mortality_swap_pricer;
}

pub mod alpha {
    pub mod senolytic_trial_arb;
    pub mod bio_financial_arb_engine;
}

// Re-export main types for convenience
pub use genomics::simd_fasta_parser::{SimdFastqParser, ReadBuffer, FastaParseError};
pub use genomics::elastic_net_prs::{PrsCalculator, PrsState, ElasticNetRegularizer, RiskCategory};
pub use genomics::homomorphic_privacy_router::{HomomorphicPrsRouter, EncryptedValue, PrivacyAuditLog};

pub use epigenetics::horvath_clock_solver::{HorvathClockSolver, BetaValue, HorvathCoefficients};
pub use epigenetics::methylation_beta_processor::{MethylationBetaProcessor, ProbeIntensities, QcMetrics};

pub use mortality::lee_carter_kalman::{LeeCarterKalmanModel, KalmanFilterState, LeeCarterParams};
pub use mortality::cohort_correlation_cholesky::{CohortCorrelationModel, CorrelationMatrix, CholeskyFactor};

pub use derivatives::affine_longevity_bond::{AffineLongevityBondPricer, LongevityBondPrice, AffineState};
pub use derivatives::mortality_swap_pricer::{MortalitySwapPricer, MortalitySwapTerms, MortalitySwapValuation};

pub use alpha::senolytic_trial_arb::{SenolyticTrialArb, SenolyticTrialResult, LongevityTradingSignal};
pub use alpha::bio_financial_arb_engine::{BioFinancialArbEngine, BiologicalAgeSignal, LongevityArbSignal};
