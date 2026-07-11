//! Base-Reality Arbitrage Engine
//! 
//! Combines all ontological glitch detection into a unified arbitrage system.
//! Coordinates sequence ID rollovers, timestamp races, and other simulation artifacts
//! to identify and exploit exchange software/hardware flaws.

use core::fmt;

/// Represents a base-reality arbitrage opportunity
#[derive(Debug, Clone)]
pub struct RealityArbitrageOpportunity {
    /// Unique identifier for this opportunity
    pub id: u64,
    /// Type of glitch being exploited
    pub glitch_type: GlitchType,
    /// Expected profit in basis points
    pub expected_profit_bps: f32,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Time window for exploitation (microseconds)
    pub window_us: u64,
    /// Risk level (1-10)
    pub risk_level: u8,
    /// Recommended action
    pub recommended_action: ArbitrageAction,
}

/// Types of exploitable glitches
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlitchType {
    /// Sequence ID rollover
    SequenceRollover,
    /// Timestamp resolution race
    TimestampRace,
    /// Floating-point rounding error
    RoundingError,
    /// Queue tie-breaking prediction
    QueuePrediction,
    /// Multiple glitches combined
    Composite,
}

/// Recommended arbitrage actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArbitrageAction {
    /// Aggressive market order
    MarketOrder,
    /// Passive limit order at specific price
    LimitOrder,
    /// Cancel and replace
    CancelReplace,
    /// Do nothing (monitor only)
    Monitor,
    /// Hedge existing position
    Hedge,
}

/// Configuration for base-reality arbitrage
#[derive(Debug, Clone, Copy)]
pub struct RealityArbitrageConfig {
    /// Minimum confidence threshold for execution
    pub min_confidence_threshold: f32,
    /// Maximum risk level acceptable
    pub max_risk_level: u8,
    /// Minimum expected profit (bps)
    pub min_profit_bps: f32,
    /// Enable composite glitch detection
    pub enable_composite_detection: bool,
}

impl Default for RealityArbitrageConfig {
    fn default() -> Self {
        Self {
            min_confidence_threshold: 0.7,
            max_risk_level: 5,
            min_profit_bps: 1.0,
            enable_composite_detection: true,
        }
    }
}

/// Unified Base-Reality Arbitrage Engine
pub struct BaseRealityArbitrageur {
    config: RealityArbitrageConfig,
    opportunities: Vec<RealityArbitrageOpportunity>,
    next_opportunity_id: u64,
    total_arbitrage_count: usize,
    total_profit_bps: f32,
}

impl BaseRealityArbitrageur {
    pub const fn new(config: RealityArbitrageConfig) -> Self {
        Self {
            config,
            opportunities: Vec::new(),
            next_opportunity_id: 1,
            total_arbitrage_count: 0,
            total_profit_bps: 0.0,
        }
    }

    /// Register a potential arbitrage opportunity from any source
    pub fn register_opportunity(
        &mut self,
        glitch_type: GlitchType,
        expected_profit_bps: f32,
        confidence: f32,
        window_us: u64,
        risk_level: u8,
    ) -> Option<&RealityArbitrageOpportunity> {
        // Filter by configuration thresholds
        if confidence < self.config.min_confidence_threshold {
            return None;
        }
        
        if risk_level > self.config.max_risk_level {
            return None;
        }
        
        if expected_profit_bps < self.config.min_profit_bps {
            return None;
        }

        // Determine recommended action based on glitch type and parameters
        let action = self.determine_action(glitch_type, expected_profit_bps, window_us);

        let opportunity = RealityArbitrageOpportunity {
            id: self.next_opportunity_id,
            glitch_type,
            expected_profit_bps,
            confidence,
            window_us,
            risk_level,
            recommended_action: action,
        };

        self.next_opportunity_id += 1;
        self.opportunities.push(opportunity);
        
        self.opportunities.last()
    }

    /// Determine the best action for a given opportunity
    fn determine_action(
        &self,
        glitch_type: GlitchType,
        profit_bps: f32,
        window_us: u64,
    ) -> ArbitrageAction {
        match glitch_type {
            GlitchType::SequenceRollover => {
                // Short window - aggressive action
                if window_us < 100 {
                    ArbitrageAction::MarketOrder
                } else {
                    ArbitrageAction::LimitOrder
                }
            }
            GlitchType::TimestampRace => {
                // Very short window - must be aggressive
                if window_us < 50 && profit_bps > 5.0 {
                    ArbitrageAction::MarketOrder
                } else {
                    ArbitrageAction::Monitor
                }
            }
            GlitchType::RoundingError => {
                // Medium window - can use passive orders
                ArbitrageAction::LimitOrder
            }
            GlitchType::QueuePrediction => {
                // Depends on queue position
                if profit_bps > 2.0 {
                    ArbitrageAction::CancelReplace
                } else {
                    ArbitrageAction::Monitor
                }
            }
            GlitchType::Composite => {
                // Multiple signals - higher confidence, can be more aggressive
                if profit_bps > 3.0 {
                    ArbitrageAction::MarketOrder
                } else {
                    ArbitrageAction::Hedge
                }
            }
        }
    }

    /// Get the highest priority opportunity
    pub fn get_best_opportunity(&self) -> Option<&RealityArbitrageOpportunity> {
        self.opportunities
            .iter()
            .filter(|o| o.confidence >= self.config.min_confidence_threshold)
            .max_by(|a, b| {
                // Priority = confidence * profit / risk
                let score_a = (a.confidence as f32 * a.expected_profit_bps) / a.risk_level as f32;
                let score_b = (b.confidence as f32 * b.expected_profit_bps) / b.risk_level as f32;
                score_a.partial_cmp(&score_b).unwrap_or(core::cmp::Ordering::Equal)
            })
    }

    /// Execute an opportunity (simulate execution tracking)
    pub fn execute_opportunity(&mut self, opportunity_id: u64) -> Result<ExecutionResult, ArbitrageError> {
        let opportunity = self.opportunities
            .iter()
            .find(|o| o.id == opportunity_id)
            .ok_or(ArbitrageError::OpportunityNotFound)?;

        // Simulate execution result
        let result = ExecutionResult {
            opportunity_id,
            executed: true,
            actual_profit_bps: opportunity.expected_profit_bps * 0.9, // Assume 90% fill efficiency
            execution_time_us: opportunity.window_us / 2,
        };

        self.total_arbitrage_count += 1;
        self.total_profit_bps += result.actual_profit_bps;

        // Remove executed opportunity
        self.opportunities.retain(|o| o.id != opportunity_id);

        Ok(result)
    }

    /// Combine multiple glitch signals into a composite opportunity
    pub fn combine_signals(
        &mut self,
        signals: &[GlitchType],
    ) -> Option<&RealityArbitrageOpportunity> {
        if !self.config.enable_composite_detection || signals.is_empty() {
            return None;
        }

        // Calculate combined confidence and profit
        let base_confidence: f32 = signals.len() as f32 * 0.3;
        let confidence = base_confidence.min(0.95);
        
        let base_profit: f32 = signals.len() as f32 * 2.0;
        let profit_bps = base_profit.min(50.0);
        
        // Use minimum window across all signals
        let window_us = 100; // Default for composite

        self.register_opportunity(
            GlitchType::Composite,
            profit_bps,
            confidence,
            window_us,
            4, // Medium risk for composite
        )
    }

    /// Get statistics
    pub fn get_statistics(&self) -> ArbitrageStatistics {
        ArbitrageStatistics {
            total_opportunities: self.opportunities.len(),
            total_executed: self.total_arbitrage_count,
            total_profit_bps: self.total_profit_bps,
            average_confidence: if !self.opportunities.is_empty() {
                self.opportunities.iter().map(|o| o.confidence).sum::<f32>() 
                    / self.opportunities.len() as f32
            } else {
                0.0
            },
        }
    }

    /// Clear all opportunities
    pub fn clear(&mut self) {
        self.opportunities.clear();
    }
}

/// Result of an arbitrage execution
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub opportunity_id: u64,
    pub executed: bool,
    pub actual_profit_bps: f32,
    pub execution_time_us: u64,
}

/// Statistics about arbitrage activity
#[derive(Debug, Clone)]
pub struct ArbitrageStatistics {
    pub total_opportunities: usize,
    pub total_executed: usize,
    pub total_profit_bps: f32,
    pub average_confidence: f32,
}

/// Errors from arbitrage operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArbitrageError {
    OpportunityNotFound,
    BelowThreshold,
    RiskTooHigh,
    WindowExpired,
}

impl fmt::Display for ArbitrageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArbitrageError::OpportunityNotFound => write!(f, "Opportunity not found"),
            ArbitrageError::BelowThreshold => write!(f, "Profit below minimum threshold"),
            ArbitrageError::RiskTooHigh => write!(f, "Risk level exceeds maximum"),
            ArbitrageError::WindowExpired => write!(f, "Arbitrage window has expired"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opportunity_registration() {
        let config = RealityArbitrageConfig::default();
        let mut arb = BaseRealityArbitrageur::new(config);

        let opp = arb.register_opportunity(
            GlitchType::SequenceRollover,
            5.0,
            0.8,
            100,
            3,
        );

        assert!(opp.is_some());
        assert_eq!(opp.unwrap().glitch_type, GlitchType::SequenceRollover);
    }

    #[test]
    fn test_threshold_filtering() {
        let config = RealityArbitrageConfig {
            min_confidence_threshold: 0.9,
            ..Default::default()
        };
        let mut arb = BaseRealityArbitrageur::new(config);

        // Low confidence should be filtered
        let opp = arb.register_opportunity(
            GlitchType::TimestampRace,
            10.0,
            0.5,
            50,
            2,
        );

        assert!(opp.is_none());
    }

    #[test]
    fn test_best_opportunity_selection() {
        let config = RealityArbitrageConfig::default();
        let mut arb = BaseRealityArbitrageur::new(config);

        arb.register_opportunity(GlitchType::RoundingError, 2.0, 0.7, 200, 2);
        arb.register_opportunity(GlitchType::SequenceRollover, 5.0, 0.9, 50, 3);

        let best = arb.get_best_opportunity();
        assert!(best.is_some());
        assert_eq!(best.unwrap().glitch_type, GlitchType::SequenceRollover);
    }

    #[test]
    fn test_statistics() {
        let config = RealityArbitrageConfig::default();
        let mut arb = BaseRealityArbitrageur::new(config);

        arb.register_opportunity(GlitchType::TimestampRace, 3.0, 0.8, 100, 2);
        
        let stats = arb.get_statistics();
        assert_eq!(stats.total_opportunities, 1);
        assert_eq!(stats.total_executed, 0);
    }
}
