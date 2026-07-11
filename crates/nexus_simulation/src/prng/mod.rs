//! Nexus Simulation - PRNG State Extraction & Queue Prediction
//! 
//! Implements Z3 SMT solver bridge for reverse-engineering exchange PRNG states
//! used in queue tie-breaking and randomized block trade allocation.

pub mod z3_state_extractor;
pub mod mersenne_twister_cracker;
pub mod queue_priority_predictor;
