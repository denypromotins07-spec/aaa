//! Fencing Module Declaration

pub mod stonith_executor;
pub mod quorum_witness;
pub mod kamikaze_protocol;

pub use stonith_executor::{STONITHFencer, FencingResult};
pub use quorum_witness::{QuorumWitness, WitnessState};
pub use kamikaze_protocol::{KamikazeProtocol, KamikazeState, integrate_with_kill_switch};
