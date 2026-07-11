//! Flow State Arbitrage Engine - trades based on neural engagement signals.
//! Shorts high-fatigue platforms, goes long on flow-inducing platforms.

use crate::bci::flow_state_theta_beta::{CognitiveState, AggregateFlowState};

/// Maximum platforms tracked
pub const MAX_PLATFORMS: usize = 32;

/// Platform position type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionType {
    Long,
    Short,
    Neutral,
}

/// Platform signal result
#[derive(Debug, Clone)]
pub struct PlatformSignal {
    pub platform_id: u32,
    pub position: PositionType,
    pub signal_strength: f32,
    pub engagement_score: f32,
    pub fatigue_score: f32,
    pub recommended_allocation: f32,
}

impl PlatformSignal {
    pub const fn new() -> Self {
        Self {
            platform_id: 0,
            position: PositionType::Neutral,
            signal_strength: 0.0,
            engagement_score: 0.0,
            fatigue_score: 0.0,
            recommended_allocation: 0.0,
        }
    }
}

impl Default for PlatformSignal {
    fn default() -> Self {
        Self::new()
    }
}

/// Flow state arbitrage engine result
#[derive(Debug, Clone)]
pub struct ArbitrageResult {
    pub signals: [PlatformSignal; MAX_PLATFORMS],
    pub num_signals: usize,
    pub portfolio_alpha: f32,
    pub sharpe_ratio: f32,
    pub max_drawdown_risk: f32,
}

impl ArbitrageResult {
    pub const fn new() -> Self {
        Self {
            signals: [PlatformSignal::new(); MAX_PLATFORMS],
            num_signals: 0,
            portfolio_alpha: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown_risk: 0.0,
        }
    }
}

impl Default for ArbitrageResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Main flow state arbitrage engine
pub struct FlowStateArbitrageEngine {
    result: ArbitrageResult,
    /// Risk-free rate
    risk_free_rate: f32,
    /// Risk aversion parameter
    risk_aversion: f32,
    /// Position limits
    max_position_pct: f32,
}

impl FlowStateArbitrageEngine {
    pub fn new() -> Self {
        Self {
            result: ArbitrageResult::new(),
            risk_free_rate: 0.03,
            risk_aversion: 0.5,
            max_position_pct: 0.2,
        }
    }

    /// Configure engine parameters
    pub fn configure(&mut self, rf_rate: f32, risk_aversion: f32, max_pos: f32) {
        self.risk_free_rate = rf_rate.max(0.0);
        self.risk_aversion = risk_aversion.clamp(0.0, 1.0);
        self.max_position_pct = max_pos.clamp(0.0, 1.0);
    }

    /// Generate trading signals from flow state data
    pub fn generate_signals(&mut self, platform_states: &[(u32, &AggregateFlowState)]) -> &ArbitrageResult {
        self.result.num_signals = platform_states.len().min(MAX_PLATFORMS);
        
        let mut total_alpha = 0.0f32;
        let mut total_variance = 0.0f32;

        for (i, &(platform_id, state)) in platform_states.iter().enumerate().take(MAX_PLATFORMS) {
            let engagement = state.population_engagement;
            let fatigue = state.population_fatigue;
            
            // Signal strength: long high engagement, short high fatigue
            let signal = engagement - fatigue * 2.0;
            
            // Determine position
            let position = if signal > 0.3 {
                PositionType::Long
            } else if signal < -0.3 {
                PositionType::Short
            } else {
                PositionType::Neutral
            };

            // Calculate recommended allocation based on signal strength
            let allocation = signal.abs().min(self.max_position_pct) * 
                if position == PositionType::Long { 1.0 } else { -1.0 };

            // Expected alpha from this position
            let expected_return = signal * 0.1; // Simplified alpha model
            total_alpha += expected_return * allocation;
            total_variance += (signal * 0.2).powi(2); // Simplified variance

            self.result.signals[i] = PlatformSignal {
                platform_id,
                position,
                signal_strength: signal,
                engagement_score: engagement,
                fatigue_score: fatigue,
                recommended_allocation: allocation,
            };
        }

        self.result.portfolio_alpha = total_alpha;
        
        // Sharpe ratio
        let volatility = total_variance.sqrt();
        if volatility > 1e-6 {
            self.result.sharpe_ratio = (total_alpha - self.risk_free_rate) / volatility;
        }

        // Max drawdown risk estimate
        self.result.max_drawdown_risk = volatility * 2.5;

        &self.result
    }

    /// Get aggregate signal across all platforms
    pub fn aggregate_signal(&self) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..self.result.num_signals {
            sum += self.result.signals[i].signal_strength;
        }
        if self.result.num_signals > 0 {
            sum / self.result.num_signals as f32
        } else {
            0.0
        }
    }

    /// Get portfolio alpha
    #[inline]
    pub const fn portfolio_alpha(&self) -> f32 {
        self.result.portfolio_alpha
    }

    /// Get Sharpe ratio
    #[inline]
    pub const fn sharpe_ratio(&self) -> f32 {
        self.result.sharpe_ratio
    }
}

impl Default for FlowStateArbitrageEngine {
    fn default() -> Self {
        Self::new()
    }
}
