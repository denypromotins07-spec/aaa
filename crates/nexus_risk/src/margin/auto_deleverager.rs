//! Auto-Deleveraging Engine.
//! 
//! Automatically reduces position risk when margin thresholds are breached,
//! prioritizing illiquid positions to minimize market impact.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Epsilon for comparisons
const EPSILON: f64 = 1e-9;

/// Deleverage action to be executed
#[derive(Debug, Clone)]
pub struct DeleverageAction {
    /// Symbol to reduce
    pub symbol: String,
    /// Side to reduce (opposite of current position)
    pub side: i8,
    /// Amount to reduce in base units
    pub reduce_amount: u64,
    /// Priority score (higher = more urgent)
    pub priority: f64,
    /// Reason for deleveraging
    pub reason: DeleverageReason,
}

/// Reason for triggering deleveraging
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeleverageReason {
    /// Margin ratio exceeded warning threshold
    MarginWarning,
    /// Margin ratio exceeded critical threshold
    MarginCritical,
    /// Liquidation price too close to mark price
    LiquidationRisk,
    /// Exchange initiated ADL (Auto-Deleverage)
    ExchangeADL,
    /// Manual intervention
    Manual,
}

/// Position risk metrics for prioritization
#[derive(Debug, Clone)]
pub struct PositionRiskMetrics {
    /// Symbol
    pub symbol: String,
    /// Current position size
    pub size: u64,
    /// Side (positive = long, negative = short)
    pub side: i8,
    /// Distance to liquidation as percentage
    pub liq_buffer_pct: f64,
    /// Margin ratio contribution
    pub margin_contribution: f64,
    /// Estimated market impact per unit reduced
    pub market_impact_factor: f64,
    /// Risk score (higher = should be reduced first)
    pub risk_score: f64,
}

/// Configuration for auto-deleveraging
#[derive(Debug, Clone)]
pub struct AutoDeleverageConfig {
    /// Margin ratio that triggers warning-level deleveraging
    pub warning_threshold: f64,
    /// Margin ratio that triggers aggressive deleveraging
    pub critical_threshold: f64,
    /// Minimum liquidation buffer before triggering (as %)
    pub min_liq_buffer_pct: f64,
    /// Maximum single reduction as % of position
    pub max_reduction_pct: f64,
    /// Minimum delay between actions (milliseconds)
    pub action_delay_ms: u64,
}

impl Default for AutoDeleverageConfig {
    fn default() -> Self {
        Self {
            warning_threshold: 0.7,      // 70% margin utilization
            critical_threshold: 0.85,    // 85% - aggressive action needed
            min_liq_buffer_pct: 0.05,    // 5% buffer minimum
            max_reduction_pct: 0.25,     // Max 25% of position at once
            action_delay_ms: 100,        // 100ms between actions
        }
    }
}

/// Result of deleveraging analysis
#[derive(Debug, Clone)]
pub struct DeleverageAnalysis {
    /// Whether deleveraging is needed
    pub needs_action: bool,
    /// Recommended actions (sorted by priority)
    pub recommended_actions: Vec<DeleverageAction>,
    /// Current margin ratio
    pub current_margin_ratio: f64,
    /// Projected margin ratio after actions
    pub projected_margin_ratio: f64,
}

/// Auto-Deleveraging Engine
/// 
/// Monitors portfolio risk and generates deleveraging actions
/// when margin thresholds are breached.
pub struct AutoDeleveragingEngine {
    /// Configuration
    config: AutoDeleverageConfig,
    /// Whether engine is enabled
    enabled: AtomicBool,
    /// Count of actions generated
    actions_generated: AtomicU64,
    /// Count of actions executed
    actions_executed: AtomicU64,
    /// Timestamp of last action
    last_action_timestamp_ns: AtomicU64,
}

unsafe impl Send for AutoDeleveragingEngine {}
unsafe impl Sync for AutoDeleveragingEngine {}

impl AutoDeleveragingEngine {
    /// Create a new auto-deleveraging engine.
    pub fn new(config: AutoDeleverageConfig) -> Self {
        Self {
            config,
            enabled: AtomicBool::new(false),
            actions_generated: AtomicU64::new(0),
            actions_executed: AtomicU64::new(0),
            last_action_timestamp_ns: AtomicU64::new(0),
        }
    }

    /// Enable or disable the engine.
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }

    /// Check if engine is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Analyze positions and generate deleveraging recommendations.
    /// 
    /// # Arguments
    /// * `positions` - Current position risk metrics
    /// * `current_margin_ratio` - Current portfolio margin ratio
    /// * `timestamp_ns` - Current timestamp
    #[inline]
    pub fn analyze(&self, positions: &[PositionRiskMetrics], current_margin_ratio: f64, timestamp_ns: u64) -> DeleverageAnalysis {
        if !self.is_enabled() {
            return DeleverageAnalysis {
                needs_action: false,
                recommended_actions: vec![],
                current_margin_ratio,
                projected_margin_ratio: current_margin_ratio,
            };
        }

        let mut actions = Vec::new();
        let mut projected_ratio = current_margin_ratio;

        // Determine trigger level
        let (trigger_reason, target_ratio) = if current_margin_ratio >= self.config.critical_threshold {
            (DeleverageReason::MarginCritical, self.config.warning_threshold)
        } else if current_margin_ratio >= self.config.warning_threshold {
            (DeleverageReason::MarginWarning, self.config.warning_threshold - 0.05)
        } else {
            // Check liquidation buffers
            let low_buffer_positions: Vec<&PositionRiskMetrics> = positions
                .iter()
                .filter(|p| p.liq_buffer_pct < self.config.min_liq_buffer_pct && p.liq_buffer_pct > 0.0)
                .collect();
            
            if !low_buffer_positions.is_empty() {
                (DeleverageReason::LiquidationRisk, self.config.warning_threshold - 0.1)
            } else {
                // No action needed
                return DeleverageAnalysis {
                    needs_action: false,
                    recommended_actions: vec![],
                    current_margin_ratio,
                    projected_margin_ratio,
                };
            }
        };

        // Sort positions by risk score (highest first)
        let mut sorted_positions: Vec<&PositionRiskMetrics> = positions.iter().collect();
        sorted_positions.sort_by(|a, b| b.risk_score.partial_cmp(&a.risk_score).unwrap_or(std::cmp::Ordering::Equal));

        // Generate reduction actions
        for position in sorted_positions {
            if projected_ratio <= target_ratio {
                break;
            }

            // Calculate reduction amount
            let max_reduce = (position.size as f64 * self.config.max_reduction_pct) as u64;
            if max_reduce == 0 || position.size == 0 {
                continue;
            }

            // Estimate margin relief from this reduction
            let margin_relief = position.margin_contribution * (max_reduce as f64 / position.size as f64);
            let estimated_ratio_reduction = margin_relief;

            // Adjust reduction based on market impact
            let impact_adjusted_reduce = if position.market_impact_factor > 1.0 {
                // High impact - reduce less
                (max_reduce as f64 / position.market_impact_factor) as u64
            } else {
                max_reduce
            };

            if impact_adjusted_reduce > 0 {
                let priority = position.risk_score * (1.0 + (self.config.critical_threshold - current_margin_ratio));
                
                actions.push(DeleverageAction {
                    symbol: position.symbol.clone(),
                    side: -position.side, // Reduce by taking opposite side
                    reduce_amount: impact_adjusted_reduce,
                    priority,
                    reason: trigger_reason,
                });

                projected_ratio -= estimated_ratio_reduction.min(0.01); // Conservative estimate
            }
        }

        // Sort actions by priority
        actions.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));

        let needs_action = !actions.is_empty();
        
        if needs_action {
            self.actions_generated.fetch_add(actions.len() as u64, Ordering::Relaxed);
        }

        DeleverageAnalysis {
            needs_action,
            recommended_actions: actions,
            current_margin_ratio,
            projected_margin_ratio: projected_ratio.max(0.0),
        }
    }

    /// Record that an action was executed.
    #[inline]
    pub fn record_execution(&self, timestamp_ns: u64) {
        self.actions_executed.fetch_add(1, Ordering::Relaxed);
        self.last_action_timestamp_ns.store(timestamp_ns, Ordering::Relaxed);
    }

    /// Check if enough time has passed since last action.
    #[inline]
    pub fn can_take_action(&self, current_timestamp_ns: u64) -> bool {
        let last_ts = self.last_action_timestamp_ns.load(Ordering::Relaxed);
        if last_ts == 0 {
            return true;
        }
        let elapsed_ms = (current_timestamp_ns - last_ts) / 1_000_000;
        elapsed_ms >= self.config.action_delay_ms
    }

    /// Get statistics.
    pub fn stats(&self) -> DeleverageStats {
        DeleverageStats {
            enabled: self.is_enabled(),
            actions_generated: self.actions_generated.load(Ordering::Relaxed),
            actions_executed: self.actions_executed.load(Ordering::Relaxed),
            config: self.config.clone(),
        }
    }

    /// Calculate risk score for a position.
    /// 
    /// Higher scores indicate positions that should be reduced first.
    #[inline]
    pub fn calculate_risk_score(
        liq_buffer_pct: f64,
        margin_contribution: f64,
        market_impact: f64,
    ) -> f64 {
        // Score components:
        // 1. Inverse of liquidation buffer (closer to liq = higher score)
        let liq_score = if liq_buffer_pct > EPSILON {
            1.0 / liq_buffer_pct
        } else {
            1000.0 // Immediate danger
        };

        // 2. Margin contribution (higher = more impact on portfolio)
        let margin_score = margin_contribution * 10.0;

        // 3. Market impact penalty (higher impact = lower priority to avoid slippage)
        let impact_penalty = 1.0 / market_impact.max(1.0);

        // Combined score
        liq_score * 0.5 + margin_score * 0.3 + impact_penalty * 0.2
    }
}

/// Statistics from the deleveraging engine
#[derive(Debug, Clone)]
pub struct DeleverageStats {
    pub enabled: bool,
    pub actions_generated: u64,
    pub actions_executed: u64,
    pub config: AutoDeleverageConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_action_when_below_threshold() {
        let engine = AutoDeleveragingEngine::new(AutoDeleverageConfig::default());
        engine.set_enabled(true);

        let positions = vec![
            PositionRiskMetrics {
                symbol: "BTCUSD".to_string(),
                size: 1000,
                side: 1,
                liq_buffer_pct: 0.20,
                margin_contribution: 0.3,
                market_impact_factor: 1.0,
                risk_score: 0.5,
            },
        ];

        let result = engine.analyze(&positions, 0.5, 1000);

        assert!(!result.needs_action);
        assert!(result.recommended_actions.is_empty());
    }

    #[test]
    fn test_action_when_above_warning() {
        let engine = AutoDeleveragingEngine::new(AutoDeleverageConfig::default());
        engine.set_enabled(true);

        let positions = vec![
            PositionRiskMetrics {
                symbol: "BTCUSD".to_string(),
                size: 1000,
                side: 1,
                liq_buffer_pct: 0.10,
                margin_contribution: 0.4,
                market_impact_factor: 1.0,
                risk_score: 0.8,
            },
        ];

        let result = engine.analyze(&positions, 0.75, 1000);

        assert!(result.needs_action);
        assert!(!result.recommended_actions.is_empty());
        assert_eq!(result.recommended_actions[0].reason, DeleverageReason::MarginWarning);
    }

    #[test]
    fn test_critical_triggers_aggressive_action() {
        let engine = AutoDeleveragingEngine::new(AutoDeleverageConfig::default());
        engine.set_enabled(true);

        let positions = vec![
            PositionRiskMetrics {
                symbol: "BTCUSD".to_string(),
                size: 1000,
                side: 1,
                liq_buffer_pct: 0.03,
                margin_contribution: 0.5,
                market_impact_factor: 1.0,
                risk_score: 0.9,
            },
        ];

        let result = engine.analyze(&positions, 0.90, 1000);

        assert!(result.needs_action);
        assert_eq!(result.recommended_actions[0].reason, DeleverageReason::MarginCritical);
    }

    #[test]
    fn test_priority_ordering() {
        let engine = AutoDeleveragingEngine::new(AutoDeleverageConfig::default());
        engine.set_enabled(true);

        let positions = vec![
            PositionRiskMetrics {
                symbol: "LOWRISK".to_string(),
                size: 1000,
                side: 1,
                liq_buffer_pct: 0.15,
                margin_contribution: 0.2,
                market_impact_factor: 1.0,
                risk_score: 0.3,
            },
            PositionRiskMetrics {
                symbol: "HIGHRISK".to_string(),
                size: 1000,
                side: 1,
                liq_buffer_pct: 0.02,
                margin_contribution: 0.4,
                market_impact_factor: 1.0,
                risk_score: 0.9,
            },
        ];

        let result = engine.analyze(&positions, 0.80, 1000);

        assert!(result.needs_action);
        assert!(result.recommended_actions.len() >= 1);
        // HIGHRISK should be first due to higher risk score
        assert_eq!(result.recommended_actions[0].symbol, "HIGHRISK");
    }

    #[test]
    fn test_market_impact_adjustment() {
        let engine = AutoDeleveragingEngine::new(AutoDeleverageConfig::default());
        engine.set_enabled(true);

        let positions = vec![
            PositionRiskMetrics {
                symbol: "ILLIQUID".to_string(),
                size: 1000,
                side: 1,
                liq_buffer_pct: 0.02,
                margin_contribution: 0.4,
                market_impact_factor: 5.0, // High impact
                risk_score: 0.9,
            },
        ];

        let result = engine.analyze(&positions, 0.80, 1000);

        assert!(result.needs_action);
        // Reduction should be adjusted down due to high market impact
        let action = &result.recommended_actions[0];
        assert!(action.reduce_amount < (1000.0 * 0.25) as u64);
    }

    #[test]
    fn test_risk_score_calculation() {
        // Close to liquidation should have high score
        let score_close = AutoDeleveragingEngine::calculate_risk_score(0.01, 0.3, 1.0);
        let score_far = AutoDeleveragingEngine::calculate_risk_score(0.30, 0.3, 1.0);
        
        assert!(score_close > score_far);

        // High margin contribution should increase score
        let score_high_margin = AutoDeleveragingEngine::calculate_risk_score(0.10, 0.5, 1.0);
        let score_low_margin = AutoDeleveragingEngine::calculate_risk_score(0.10, 0.1, 1.0);
        
        assert!(score_high_margin > score_low_margin);
    }

    #[test]
    fn test_action_delay() {
        let engine = AutoDeleveragingEngine::new(AutoDeleverageConfig::default());

        // First action should be allowed
        assert!(engine.can_take_action(1000));
        engine.record_execution(1000);

        // Immediate second action should be blocked
        assert!(!engine.can_take_action(1050)); // Only 50ms later

        // After delay, should be allowed
        assert!(engine.can_take_action(1000 + 100_000_000)); // 100ms later
    }

    #[test]
    fn test_disabled_engine() {
        let engine = AutoDeleveragingEngine::new(AutoDeleverageConfig::default());
        // Engine is disabled by default

        let positions = vec![
            PositionRiskMetrics {
                symbol: "BTCUSD".to_string(),
                size: 1000,
                side: 1,
                liq_buffer_pct: 0.01,
                margin_contribution: 0.5,
                market_impact_factor: 1.0,
                risk_score: 1.0,
            },
        ];

        let result = engine.analyze(&positions, 0.95, 1000);

        assert!(!result.needs_action);
        assert!(result.recommended_actions.is_empty());
    }
}
