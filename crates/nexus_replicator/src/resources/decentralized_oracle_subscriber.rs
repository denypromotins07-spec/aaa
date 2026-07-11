//! Decentralized Oracle Subscriber Module
pub use super::autonomous_compute_buyer::{ResourceError, ResourceResult};

use alloc::vec::Vec;
use core::marker::PhantomData;

/// Oracle subscription configuration
#[derive(Debug, Clone)]
pub struct OracleConfig {
    /// Oracle provider ID
    pub provider_id: [u8; 32],
    /// Data feed IDs to subscribe to
    pub feed_ids: Vec<[u8; 32]>,
    /// Update frequency in seconds
    pub update_frequency_seconds: u64,
    /// Maximum price per update (atto-tokens)
    pub max_price_per_update: u128,
}

/// Oracle subscriber for decentralized data feeds
pub struct OracleSubscriber<'a> {
    config: Option<OracleConfig>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> OracleSubscriber<'a> {
    pub fn new() -> Self {
        Self {
            config: None,
            _marker: PhantomData,
        }
    }

    pub fn with_config(config: OracleConfig) -> Self {
        Self {
            config: Some(config),
            _marker: PhantomData,
        }
    }

    pub fn subscribe(&mut self, _config: OracleConfig) -> ResourceResult<()> {
        self.config = Some(_config);
        Ok(())
    }

    pub fn is_subscribed(&self) -> bool {
        self.config.is_some()
    }
}

impl Default for OracleSubscriber<'_> {
    fn default() -> Self {
        Self::new()
    }
}
