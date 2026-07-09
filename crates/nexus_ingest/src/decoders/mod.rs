//! Chapter 4: Zero-Copy Protocol Decoders
//!
//! This module provides ultra-fast, zero-copy decoders for exchange protocols.

pub mod simd_json_ws;
pub mod cme_mdp3;
pub mod fix_protocol;

pub use simd_json_ws::{SimdJsonParser, WebSocketFrame};
pub use cme_mdp3::{CmeMdp3Decoder, Mdp3Message};
pub use fix_protocol::{FixParser, FixField};
