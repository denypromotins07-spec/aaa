//! NEXUS-OMEGA Stage 40: Temporal Mechanics, CTCs & Retrocausal Alpha
//! 
//! This crate implements advanced quantum-inspired trading mechanisms:
//! - Weak Measurements for order book probing without collapse
//! - Deutsch Closed Timelike Curves for paradox resolution
//! - Wheeler-Feynman Absorber Theory for time-symmetric market impact
//! - Transactional Interpretation for retrocausal smart order routing

#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

pub mod weak;
pub mod ctc;
pub mod absorber;
pub mod transactional;

/// Re-export main types for convenience
pub use weak::{
    weak_value_amplifier::{WeakValueAmplifier, WeakMeasurementResult},
    post_selection_filter::PostSelectionFilter,
    hidden_intent_extractor::{HiddenIntentExtractor, InstitutionalIntent, IntentDirection},
};

pub use ctc::{
    deutsch_density_matrix::{DensityMatrix, Complex, CTCResult},
    fixed_point_iteration::{FixedPointSolver, FixedPointResult, ConvergenceMethod},
    paradox_slippage_resolver::{ParadoxSlippageResolver, ParadoxResolution, ParadoxType, ExecutionParams},
};

pub use absorber::{
    wheeler_feynman_green::{WheelerFeynmanGreen, GreenFunctionResult, GreenComponent},
    advanced_potential_liquidity::{AdvancedPotentialLiquidity, AdvancedPotentialField, LiquidityCluster},
    time_symmetric_impact::{TimeSymmetricImpactModel, TimeSymmetricImpact, TradeRecord},
};

pub use transactional::{
    offer_wave_emitter::{OfferWaveEmitter, OfferWave, OfferWaveComponent},
    confirmation_wave_receiver::{ConfirmationWaveReceiver, ConfirmationWave, ConfirmationSource},
    retrocausal_handshake::{RetrocausalHandshakeCalculator, TransactionResult, HandshakeState},
};
