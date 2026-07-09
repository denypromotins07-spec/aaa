//! Nexus Adapters Library - Zero-Allocation Exchange Adapters

pub mod zero_alloc_buffer_writer;
pub mod binance_ws_signer;
pub mod fix_binary_encoder;

pub use zero_alloc_buffer_writer::{NetworkBuffer, MAX_MESSAGE_SIZE};
pub use binance_ws_signer::{BinanceWsBuilder, HmacSigner};
pub use fix_binary_encoder::{FixEncoder, tags, msg_types, sides, order_types, tif};
