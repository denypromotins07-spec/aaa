//! Sandwich Attack Protection Router
//! 
//! Simulates mempool state to detect front-running attempts before broadcasting.
//! Implements protection strategies against sandwich attacks by analyzing pending transactions.

use thiserror::Error;
use alloc::vec::Vec;
use alloc::string::String;

/// Minimum price improvement threshold to proceed (basis points)
const MIN_PRICE_IMPROVEMENT_BP: u16 = 50; // 0.5%

/// Maximum acceptable slippage (basis points)
const MAX_SLIPPAGE_BP: u16 = 300; // 3%

#[derive(Error, Debug)]
pub enum SandwichError {
    #[error("Front-run detected: pending tx at {pending_price} vs expected {expected_price}")]
    FrontRunDetected { pending_price: u64, expected_price: u64 },
    #[error("Back-run risk: MEV bot likely to follow")]
    BackRunRisk,
    #[error("Slippage exceeds maximum: {slippage_bp} > {max_bp}")]
    ExcessiveSlippage { slippage_bp: u16, max_bp: u16 },
    #[error("Mempool simulation failed")]
    SimulationFailed,
    #[error("Invalid route configuration")]
    InvalidRoute,
}

pub type Result<T> = core::result::Result<T, SandwichError>;

/// Pending transaction in mempool with MEV potential
#[derive(Clone, Debug)]
pub struct PendingTx {
    pub hash: [u8; 32],
    pub from: [u8; 20],
    pub to: Option<[u8; 20]>,
    pub value: u64,
    pub gas_price: u64,
    pub input_size: usize,
    /// Estimated MEV profit in basis points
    pub mev_potential_bp: u16,
}

/// Route for DEX swap with price impact analysis
#[derive(Clone, Debug)]
pub struct SwapRoute {
    pub path: Vec<[u8; 20]>, // Token addresses
    pub dex: DexType,
    pub expected_output: u64,
    pub price_impact_bp: u16,
    pub gas_estimate: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DexType {
    UniswapV2,
    UniswapV3,
    SushiSwap,
    Curve,
    Balancer,
}

/// Analysis result for a potential trade
#[derive(Debug)]
pub struct TradeAnalysis {
    /// Whether sandwich attack is detected
    pub sandwich_risk: SandwichRiskLevel,
    /// Recommended action
    pub recommendation: TradeRecommendation,
    /// Expected slippage after protection
    pub protected_slippage_bp: u16,
    /// MEV bots likely to target this trade
    pub targeted_by_mev: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SandwichRiskLevel {
    None,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeRecommendation {
    /// Proceed with normal parameters
    Proceed,
    /// Reduce size to avoid detection
    ReduceSize,
    /// Use private RPC / Flashbots
    UsePrivateRpc,
    /// Split into multiple smaller trades
    SplitTrade,
    /// Abort - too risky
    Abort,
}

/// Sandwich Protection Router state
pub struct SandwichProtectionRouter {
    /// Pending transactions from mempool
    pending_txs: Vec<PendingTx>,
    /// Known MEV bot addresses
    known_mev_bots: Vec<[u8; 20]>,
    /// Recent sandwich attacks observed
    recent_attacks: u32,
    /// Current gas price for priority calculation
    current_base_fee: u64,
}

impl SandwichProtectionRouter {
    /// Create a new protection router
    pub fn new() -> Self {
        Self {
            pending_txs: Vec::with_capacity(256),
            known_mev_bots: Vec::with_capacity(64),
            recent_attacks: 0,
            current_base_fee: 0,
        }
    }

    /// Add known MEV bot address for tracking
    pub fn add_mev_bot(&mut self, address: [u8; 20]) {
        if !self.known_mev_bots.contains(&address) {
            self.known_mev_bots.push(address);
        }
    }

    /// Update mempool state with new pending transactions
    pub fn update_mempool(&mut self, txs: impl Iterator<Item = PendingTx>) {
        self.pending_txs.clear();
        self.pending_txs.extend(txs);
    }

    /// Set current base fee
    pub fn set_base_fee(&mut self, fee: u64) {
        self.current_base_fee = fee;
    }

    /// Analyze a potential trade for sandwich risk
    pub fn analyze_trade(&self, route: &SwapRoute, amount_in: u64) -> Result<TradeAnalysis> {
        // Check for front-running transactions in mempool
        let front_run_risk = self.detect_front_run(route, amount_in)?;
        
        // Check for back-running risk
        let back_run_risk = self.detect_back_run(route, amount_in);
        
        // Calculate overall risk level
        let sandwich_risk = self.calculate_risk_level(front_run_risk, back_run_risk);
        
        // Determine recommendation
        let recommendation = self.get_recommendation(&sandwich_risk, route.price_impact_bp);
        
        // Calculate protected slippage
        let protected_slippage = self.calculate_protected_slippage(
            route.price_impact_bp,
            &sandwich_risk,
        );
        
        // Check if targeted by MEV
        let targeted = self.is_targeted_by_mev(route, amount_in);

        Ok(TradeAnalysis {
            sandwich_risk,
            recommendation,
            protected_slippage_bp: protected_slippage,
            targeted_by_mev: targeted,
        })
    }

    /// Detect potential front-running transactions
    fn detect_front_run(&self, route: &SwapRoute, amount_in: u64) -> Result<bool> {
        // Look for pending transactions with similar routes
        for tx in &self.pending_txs {
            // Large transactions to same DEX/router
            if tx.input_size > 200 && tx.gas_price > self.current_base_fee {
                // High MEV potential indicates possible sandwich setup
                if tx.mev_potential_bp > MIN_PRICE_IMPROVEMENT_BP {
                    // Check if this could front-run our trade
                    let would_front_run = tx.gas_price > self.current_base_fee * 110 / 100;
                    if would_front_run {
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }

    /// Detect potential back-running
    fn detect_back_run(&self, route: &SwapRoute, amount_in: u64) -> bool {
        // Check for MEV bots watching this DEX
        for tx in &self.pending_txs {
            if self.known_mev_bots.contains(&tx.from) {
                // MEV bot active in mempool
                if tx.input_size > 100 {
                    return true;
                }
            }
        }
        false
    }

    /// Calculate overall risk level
    fn calculate_risk_level(&self, front_run: bool, back_run: bool) -> SandwichRiskLevel {
        match (front_run, back_run) {
            (false, false) => SandwichRiskLevel::None,
            (false, true) => SandwichRiskLevel::Low,
            (true, false) => SandwichRiskLevel::Medium,
            (true, true) => SandwichRiskLevel::High,
        }
    }

    /// Get trade recommendation based on risk
    fn get_recommendation(&self, risk: &SandwichRiskLevel, slippage_bp: u16) -> TradeRecommendation {
        // If slippage already high, be more conservative
        if slippage_bp > MAX_SLIPPAGE_BP {
            return TradeRecommendation::Abort;
        }

        match risk {
            SandwichRiskLevel::None => TradeRecommendation::Proceed,
            SandwichRiskLevel::Low => {
                if slippage_bp > 100 {
                    TradeRecommendation::ReduceSize
                } else {
                    TradeRecommendation::Proceed
                }
            }
            SandwichRiskLevel::Medium => TradeRecommendation::UsePrivateRpc,
            SandwichRiskLevel::High => TradeRecommendation::SplitTrade,
            SandwichRiskLevel::Critical => TradeRecommendation::Abort,
        }
    }

    /// Calculate slippage with protection applied
    fn calculate_protected_slippage(&self, base_slippage: u16, risk: &SandwichRiskLevel) -> u16 {
        let multiplier = match risk {
            SandwichRiskLevel::None => 100,
            SandwichRiskLevel::Low => 120,
            SandwichRiskLevel::Medium => 150,
            SandwichRiskLevel::High => 200,
            SandwichRiskLevel::Critical => 500,
        };
        
        let adjusted = (base_slippage as u32 * multiplier as u32 / 100) as u16;
        adjusted.min(MAX_SLIPPAGE_BP)
    }

    /// Check if trade is likely targeted by MEV bots
    fn is_targeted_by_mev(&self, route: &SwapRoute, amount_in: u64) -> bool {
        // Large trades on popular DEXes are prime targets
        let is_large = amount_in > 1_000_000_000_000_000_000u64; // > 1 ETH equivalent
        
        let is_popular_dex = matches!(
            route.dex,
            DexType::UniswapV2 | DexType::UniswapV3 | DexType::SushiSwap
        );
        
        is_large && is_popular_dex
    }

    /// Simulate execution with mempool state
    pub fn simulate_execution(&self, route: &SwapRoute, amount_in: u64) -> Result<u64> {
        // Apply price impact from our trade
        let base_output = route.expected_output;
        
        // Check if any pending tx would affect our output
        for tx in &self.pending_txs {
            if tx.mev_potential_bp > 0 {
                // Simulate the impact of this pending tx executing first
                let impact = (base_output as u64 * tx.mev_potential_bp as u64 / 10000) as u64;
                if impact > base_output / 10 {
                    // Significant impact detected
                    return Err(SandwichError::FrontRunDetected {
                        pending_price: tx.gas_price,
                        expected_price: self.current_base_fee,
                    });
                }
            }
        }
        
        Ok(base_output)
    }

    /// Record a detected sandwich attack for statistics
    pub fn record_attack(&mut self) {
        self.recent_attacks = self.recent_attacks.saturating_add(1);
    }

    /// Get recent attack count
    pub const fn recent_attacks(&self) -> u32 {
        self.recent_attacks
    }
}

impl Default for SandwichProtectionRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let router = SandwichProtectionRouter::new();
        assert_eq!(router.recent_attacks(), 0);
    }

    #[test]
    fn test_no_risk_analysis() {
        let mut router = SandwichProtectionRouter::new();
        router.set_base_fee(100_000_000_000u64); // 100 gwei
        
        let route = SwapRoute {
            path: vec![[1u8; 20], [2u8; 20]],
            dex: DexType::UniswapV3,
            expected_output: 1_000_000_000_000_000_000u64,
            price_impact_bp: 50,
            gas_estimate: 150_000,
        };
        
        let analysis = router.analyze_trade(&route, 1_000_000_000_000_000_000u64).unwrap();
        assert_eq!(analysis.sandwich_risk, SandwichRiskLevel::None);
        assert_eq!(analysis.recommendation, TradeRecommendation::Proceed);
    }

    #[test]
    fn test_high_slippage_abort() {
        let router = SandwichProtectionRouter::new();
        
        let route = SwapRoute {
            path: vec![[1u8; 20], [2u8; 20]],
            dex: DexType::UniswapV2,
            expected_output: 1_000_000_000_000_000_000u64,
            price_impact_bp: 400, // Very high
            gas_estimate: 150_000,
        };
        
        let analysis = router.analyze_trade(&route, 1_000_000_000_000_000_000u64).unwrap();
        assert_eq!(analysis.recommendation, TradeRecommendation::Abort);
    }

    #[test]
    fn test_mev_bot_tracking() {
        let mut router = SandwichProtectionRouter::new();
        router.add_mev_bot([0x42u8; 20]);
        
        assert!(router.known_mev_bots.contains(&[0x42u8; 20]));
    }
}
