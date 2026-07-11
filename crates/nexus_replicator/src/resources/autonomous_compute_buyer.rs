//! Autonomous Compute Buyer Module
//!
//! Implements autonomous acquisition of decentralized compute resources
//! (Akash, Golem, etc.) using crypto-native micropayments.

use alloc::vec::Vec;
use core::marker::PhantomData;

/// Error types for resource acquisition
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceError {
    InsufficientFunds,
    ProviderUnavailable,
    InvalidResourceSpec,
    PaymentFailed,
    SLAViolation,
}

/// Result type for resource operations
pub type ResourceResult<T> = Result<T, ResourceError>;

/// Compute resource specification
#[derive(Debug, Clone)]
pub struct ComputeSpec {
    /// CPU cores required
    pub cpu_cores: u32,
    /// Memory in GB
    pub memory_gb: u32,
    /// Storage in GB
    pub storage_gb: u32,
    /// GPU requirement (optional)
    pub gpu_count: u32,
    /// Duration in seconds
    pub duration_seconds: u64,
}

impl ComputeSpec {
    /// Create a new compute spec
    pub fn new(cpu_cores: u32, memory_gb: u32, storage_gb: u32) -> Self {
        Self {
            cpu_cores,
            memory_gb,
            storage_gb,
            gpu_count: 0,
            duration_seconds: 3600, // Default 1 hour
        }
    }

    /// Validate the spec
    pub fn validate(&self) -> ResourceResult<()> {
        if self.cpu_cores == 0 || self.memory_gb == 0 || self.storage_gb == 0 {
            return Err(ResourceError::InvalidResourceSpec);
        }
        Ok(())
    }

    /// Calculate estimated cost (simplified pricing model)
    pub fn estimate_cost(&self, price_per_core_hour: u128) -> u128 {
        let hours = self.duration_seconds / 3600 + 1;
        (self.cpu_cores as u128) * hours * price_per_core_hour
    }
}

/// Decentralized compute provider
#[derive(Debug, Clone)]
pub struct ComputeProvider {
    /// Provider ID
    pub id: [u8; 32],
    /// Available CPU cores
    pub available_cores: u32,
    /// Available memory in GB
    pub available_memory_gb: u32,
    /// Price per core-hour in atto-tokens
    pub price_per_core_hour: u128,
    /// Reputation score (0-100)
    pub reputation: u8,
}

impl ComputeProvider {
    /// Check if provider can fulfill the spec
    pub fn can_fulfill(&self, spec: &ComputeSpec) -> bool {
        self.available_cores >= spec.cpu_cores
            && self.available_memory_gb >= spec.memory_gb
            && self.reputation > 50 // Minimum reputation threshold
    }
}

/// Payment channel state
#[derive(Debug, Clone)]
pub struct PaymentChannel {
    /// Channel ID
    pub id: [u8; 32],
    /// Total balance in atto-tokens
    pub balance: u128,
    /// Amount spent
    pub spent: u128,
    /// Recipient address
    pub recipient: [u8; 32],
    /// Expiry block height
    pub expiry_block: u64,
}

impl PaymentChannel {
    /// Create a new payment channel
    pub fn new(id: [u8; 32], balance: u128, recipient: [u8; 32], expiry_block: u64) -> Self {
        Self {
            id,
            balance,
            spent: 0,
            recipient,
            expiry_block,
        }
    }

    /// Get remaining balance
    pub fn remaining(&self) -> u128 {
        self.balance.saturating_sub(self.spent)
    }

    /// Process a micropayment
    pub fn pay(&mut self, amount: u128) -> ResourceResult<()> {
        if amount > self.remaining() {
            return Err(ResourceError::InsufficientFunds);
        }
        self.spent = self.spent.saturating_add(amount);
        Ok(())
    }
}

/// Autonomous Resource Acquirer
pub struct AutonomousResourceAcquirer<'a> {
    /// Available payment channels
    channels: Vec<PaymentChannel>,
    /// Known compute providers
    providers: Vec<ComputeProvider>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> AutonomousResourceAcquirer<'a> {
    /// Create a new acquirer
    pub fn new() -> Self {
        Self {
            channels: Vec::new(),
            providers: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Add a payment channel
    pub fn add_channel(&mut self, channel: PaymentChannel) {
        self.channels.push(channel);
    }

    /// Register a compute provider
    pub fn register_provider(&mut self, provider: ComputeProvider) {
        self.providers.push(provider);
    }

    /// Find best provider for a given spec
    pub fn find_best_provider(&self, spec: &ComputeSpec) -> ResourceResult<&ComputeProvider> {
        spec.validate()?;

        let mut best_provider: Option<&ComputeProvider> = None;
        let mut best_score: u32 = 0;

        for provider in &self.providers {
            if provider.can_fulfill(spec) {
                // Score = reputation * 100 - price_factor
                let price_factor = (provider.price_per_core_hour / 1_000_000) as u32;
                let score = (provider.reputation as u32) * 100 - price_factor;

                if score > best_score {
                    best_score = score;
                    best_provider = Some(provider);
                }
            }
        }

        best_provider.ok_or(ResourceError::ProviderUnavailable)
    }

    /// Acquire compute resources autonomously
    pub fn acquire_compute(
        &mut self,
        spec: &ComputeSpec,
    ) -> ResourceResult<ComputeAllocation> {
        let provider = self.find_best_provider(spec)?;
        
        // Find a suitable payment channel
        let cost = spec.estimate_cost(provider.price_per_core_hour);
        
        let channel = self.channels
            .iter_mut()
            .find(|c| c.remaining() >= cost && c.recipient == provider.id)
            .ok_or(ResourceError::InsufficientFunds)?;

        // Process payment
        channel.pay(cost)?;

        // Create allocation
        let allocation = ComputeAllocation {
            provider_id: provider.id,
            spec: spec.clone(),
            total_cost: cost,
            channel_id: channel.id,
            status: AllocationStatus::Active,
        };

        Ok(allocation)
    }

    /// Get total available balance across all channels
    pub fn total_balance(&self) -> u128 {
        self.channels.iter().map(|c| c.remaining()).sum()
    }

    /// Get number of registered providers
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }
}

impl Default for AutonomousResourceAcquirer<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute allocation result
#[derive(Debug, Clone)]
pub struct ComputeAllocation {
    /// Provider ID
    pub provider_id: [u8; 32],
    /// Resource specification
    pub spec: ComputeSpec,
    /// Total cost in atto-tokens
    pub total_cost: u128,
    /// Payment channel ID
    pub channel_id: [u8; 32],
    /// Allocation status
    pub status: AllocationStatus,
}

/// Allocation status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStatus {
    Pending,
    Active,
    Completed,
    Failed,
}

/// Resource acquisition strategy
#[derive(Debug, Clone)]
pub enum AcquisitionStrategy {
    /// Cheapest first
    CostMinimization,
    /// Best reputation first
    QualityMaximization,
    /// Balanced approach
    Balanced,
}

impl AcquisitionStrategy {
    /// Score a provider based on strategy
    pub fn score_provider(&self, provider: &ComputeProvider, spec: &ComputeSpec) -> u32 {
        if !provider.can_fulfill(spec) {
            return 0;
        }

        match self {
            AcquisitionStrategy::CostMinimization => {
                // Lower price = higher score
                let max_price = 1_000_000_000;
                ((max_price.saturating_sub(provider.price_per_core_hour)) / 10_000_000) as u32
            }
            AcquisitionStrategy::QualityMaximization => {
                (provider.reputation as u32) * 100
            }
            AcquisitionStrategy::Balanced => {
                let quality_score = (provider.reputation as u32) * 50;
                let price_score = ((1_000_000_000u128.saturating_sub(provider.price_per_core_hour))
                    / 20_000_000) as u32;
                quality_score + price_score
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_spec_validation() {
        let valid_spec = ComputeSpec::new(4, 8, 100);
        assert!(valid_spec.validate().is_ok());

        let invalid_spec = ComputeSpec::new(0, 8, 100);
        assert_eq!(invalid_spec.validate(), Err(ResourceError::InvalidResourceSpec));
    }

    #[test]
    fn test_payment_channel() {
        let mut channel = PaymentChannel::new([1u8; 32], 1_000_000, [2u8; 32], 1000);
        assert_eq!(channel.remaining(), 1_000_000);

        assert!(channel.pay(500_000).is_ok());
        assert_eq!(channel.remaining(), 500_000);

        assert_eq!(channel.pay(600_000), Err(ResourceError::InsufficientFunds));
    }

    #[test]
    fn test_provider_selection() {
        let mut acquirer = AutonomousResourceAcquirer::new();
        
        let provider1 = ComputeProvider {
            id: [1u8; 32],
            available_cores: 8,
            available_memory_gb: 16,
            price_per_core_hour: 100_000_000,
            reputation: 90,
        };

        let provider2 = ComputeProvider {
            id: [2u8; 32],
            available_cores: 4,
            available_memory_gb: 8,
            price_per_core_hour: 50_000_000,
            reputation: 70,
        };

        acquirer.register_provider(provider1);
        acquirer.register_provider(provider2);

        let spec = ComputeSpec::new(4, 8, 50);
        let best = acquirer.find_best_provider(&spec);
        assert!(best.is_ok());
    }
}
