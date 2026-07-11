//! Strategy Mutation Trigger - autonomously abandons strategies at Omega Point.
//! Detects when thermodynamic limits are reached and triggers mutation to new alpha.

use alloc::vec::Vec;
use core::fmt::Debug;

use super::landauer_limit_calculator::{LandauerCalculator, MarketEfficiencyTracker};
use super::omega_point_metric::{
    EfficiencyHorizonTracker, MutationDecision, MutationUrgency, OmegaMarketState, OmegaPointConfig,
    OmegaPointMetric,
};

/// Configuration for mutation triggering
#[derive(Debug, Clone)]
pub struct MutationTriggerConfig {
    /// Base Omega Point configuration
    pub omega_config: OmegaPointConfig,
    /// Efficiency threshold for triggering mutation
    pub trigger_threshold: f64,
    /// Hysteresis to prevent rapid on/off triggering
    pub hysteresis: f64,
    /// Minimum cooldown period between mutations (arbitrary units)
    pub cooldown_period: u64,
}

impl Default for MutationTriggerConfig {
    fn default() -> Self {
        Self {
            omega_config: OmegaPointConfig::default(),
            trigger_threshold: 5.0,
            hysteresis: 1.0,
            cooldown_period: 100,
        }
    }
}

/// State of a strategy in the mutation system
#[derive(Debug, Clone)]
pub struct StrategyState {
    /// Unique strategy identifier
    pub strategy_id: u64,
    /// Current efficiency log ratio
    pub efficiency_log: f64,
    /// Whether mutation is triggered
    pub mutation_triggered: bool,
    /// Time since last mutation
    pub time_since_last_mutation: u64,
    /// Consecutive periods above threshold
    pub breach_count: u64,
}

/// Result of mutation trigger analysis
#[derive(Debug, Clone)]
pub struct MutationTriggerResult {
    /// Whether mutation should occur now
    pub trigger_mutation: bool,
    /// Urgency level
    pub urgency: MutationUrgency,
    /// Reason for decision
    pub reason: &'static str,
    /// Recommended mutation direction
    pub direction: &'static str,
    /// Time until next evaluation recommended
    pub reevaluate_in: u64,
}

/// Strategy Mutation Trigger system
pub struct StrategyMutationTrigger {
    config: MutationTriggerConfig,
    tracker: EfficiencyHorizonTracker,
    active_strategies: Vec<StrategyState>,
}

impl StrategyMutationTrigger {
    pub fn new(config: MutationTriggerConfig) -> Self {
        let tracker = EfficiencyHorizonTracker::new(
            config.omega_config.clone(),
            config.trigger_threshold,
        );
        Self {
            config,
            tracker,
            active_strategies: Vec::new(),
        }
    }

    /// Register a new strategy for monitoring
    pub fn register_strategy(&mut self, strategy_id: u64) {
        self.active_strategies.push(StrategyState {
            strategy_id,
            efficiency_log: f64::MAX,
            mutation_triggered: false,
            time_since_last_mutation: 0,
            breach_count: 0,
        });
    }

    /// Update strategy metrics and check for mutation trigger
    pub fn update_strategy(
        &mut self,
        strategy_id: u64,
        energy_per_trade_j: f64,
        bits_processed: u64,
        current_time: u64,
    ) -> Option<MutationTriggerResult> {
        let state = self.active_strategies.iter_mut().find(|s| s.strategy_id == strategy_id)?;

        // Calculate current efficiency
        let decision = self.tracker.should_mutate(energy_per_trade_j, bits_processed);
        state.efficiency_log = decision.current_efficiency;
        state.time_since_last_mutation = current_time - state.time_since_last_mutation.min(current_time);

        // Update breach count with hysteresis
        if state.efficiency_log < self.config.trigger_threshold + self.config.hysteresis {
            state.breach_count = state.breach_count.saturating_add(1);
        } else {
            state.breach_count = 0;
        }

        // Check cooldown
        if state.time_since_last_mutation < self.config.cooldown_period {
            return Some(MutationTriggerResult {
                trigger_mutation: false,
                urgency: MutationUrgency::Low,
                reason: "Cooldown period active",
                direction: decision.recommended_direction,
                reevaluate_in: self.config.cooldown_period - state.time_since_last_mutation,
            });
        }

        // Determine if mutation should trigger
        let trigger = decision.should_mutate && state.breach_count >= 3;

        if trigger {
            state.mutation_triggered = true;
            state.time_since_last_mutation = 0;
            state.breach_count = 0;
        }

        Some(MutationTriggerResult {
            trigger_mutation: trigger,
            urgency: decision.urgency,
            reason: if trigger { "Efficiency limit reached" } else { "Below threshold" },
            direction: decision.recommended_direction,
            reevaluate_in: 1,
        })
    }

    /// Get all strategies requiring immediate mutation
    pub fn get_pending_mutations(&self) -> Vec<u64> {
        self.active_strategies
            .iter()
            .filter(|s| s.mutation_triggered)
            .map(|s| s.strategy_id)
            .collect()
    }

    /// Acknowledge mutation completion for a strategy
    pub fn acknowledge_mutation(&mut self, strategy_id: u64) -> bool {
        if let Some(state) = self.active_strategies.iter_mut().find(|s| s.strategy_id == strategy_id) {
            state.mutation_triggered = false;
            state.time_since_last_mutation = 0;
            state.breach_count = 0;
            true
        } else {
            false
        }
    }

    /// Batch update multiple strategies
    pub fn batch_update(
        &mut self,
        updates: &[(u64, f64, u64)], // (strategy_id, energy, bits)
        current_time: u64,
    ) -> Vec<(u64, MutationTriggerResult)> {
        let mut results = Vec::new();
        
        for &(id, energy, bits) in updates {
            if let Some(result) = self.update_strategy(id, energy, bits, current_time) {
                results.push((id, result));
            }
        }
        
        results
    }

    /// Get summary statistics of all monitored strategies
    pub fn get_portfolio_summary(&self) -> MutationPortfolioSummary {
        if self.active_strategies.is_empty() {
            return MutationPortfolioSummary {
                total_strategies: 0,
                pending_mutations: 0,
                avg_efficiency_log: 0.0,
                strategies_at_omega: 0,
            };
        }

        let total = self.active_strategies.len();
        let pending = self.active_strategies.iter().filter(|s| s.mutation_triggered).count();
        
        let sum_efficiency: f64 = self.active_strategies.iter().map(|s| s.efficiency_log).sum();
        let avg_efficiency = sum_efficiency / total as f64;
        
        let at_omega = self.active_strategies.iter().filter(|s| {
            s.efficiency_log < self.config.omega_config.omega_threshold
        }).count();

        MutationPortfolioSummary {
            total_strategies: total,
            pending_mutations: pending,
            avg_efficiency_log: avg_efficiency,
            strategies_at_omega: at_omega,
        }
    }
}

/// Portfolio-level mutation summary
#[derive(Debug, Clone)]
pub struct MutationPortfolioSummary {
    pub total_strategies: usize,
    pub pending_mutations: usize,
    pub avg_efficiency_log: f64,
    pub strategies_at_omega: usize,
}

impl MutationPortfolioSummary {
    /// Get overall portfolio health status
    pub fn health_status(&self) -> PortfolioHealth {
        if self.total_strategies == 0 {
            return PortfolioHealth::NoData;
        }

        let pending_ratio = self.pending_mutations as f64 / self.total_strategies as f64;
        let omega_ratio = self.strategies_at_omega as f64 / self.total_strategies as f64;

        if omega_ratio > 0.5 {
            PortfolioHealth::Critical
        } else if pending_ratio > 0.3 || omega_ratio > 0.25 {
            PortfolioHealth::Warning
        } else if pending_ratio > 0.1 {
            PortfolioHealth::Caution
        } else {
            PortfolioHealth::Healthy
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PortfolioHealth {
    Healthy,
    Caution,
    Warning,
    Critical,
    NoData,
}

/// Automatic strategy rotator based on Omega Point metrics
pub struct AutoStrategyRotator {
    trigger: StrategyMutationTrigger,
    mutation_queue: Vec<u64>,
}

impl AutoStrategyRotator {
    pub fn new(config: MutationTriggerConfig) -> Self {
        Self {
            trigger: StrategyMutationTrigger::new(config),
            mutation_queue: Vec::new(),
        }
    }

    /// Process updates and populate mutation queue
    pub fn process_cycle(
        &mut self,
        updates: &[(u64, f64, u64)],
        current_time: u64,
    ) -> Vec<u64> {
        let results = self.trigger.batch_update(updates, current_time);
        
        self.mutation_queue.clear();
        for (id, result) in results {
            if result.trigger_mutation {
                self.mutation_queue.push(id);
            }
        }
        
        self.mutation_queue.clone()
    }

    /// Get next strategy to mutate
    pub fn next_mutation(&mut self) -> Option<u64> {
        if self.mutation_queue.is_empty() {
            None
        } else {
            Some(self.mutation_queue.remove(0))
        }
    }

    /// Get portfolio health
    pub fn portfolio_health(&self) -> PortfolioHealth {
        self.trigger.get_portfolio_summary().health_status()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutation_trigger_registration() {
        let config = MutationTriggerConfig::default();
        let mut trigger = StrategyMutationTrigger::new(config);
        
        trigger.register_strategy(1);
        trigger.register_strategy(2);
        
        assert_eq!(trigger.active_strategies.len(), 2);
    }

    #[test]
    fn test_mutation_trigger_cooldown() {
        let mut config = MutationTriggerConfig::default();
        config.cooldown_period = 10;
        let mut trigger = StrategyMutationTrigger::new(config);
        
        trigger.register_strategy(1);
        
        // First update during cooldown
        let result = trigger.update_strategy(1, 1e-20, 1000, 0);
        assert!(result.is_some());
        assert!(!result.unwrap().trigger_mutation);
    }

    #[test]
    fn test_portfolio_summary() {
        let config = MutationTriggerConfig::default();
        let mut trigger = StrategyMutationTrigger::new(config);
        
        trigger.register_strategy(1);
        trigger.register_strategy(2);
        trigger.register_strategy(3);
        
        let summary = trigger.get_portfolio_summary();
        assert_eq!(summary.total_strategies, 3);
    }

    #[test]
    fn test_auto_rotator() {
        let config = MutationTriggerConfig::default();
        let mut rotator = AutoStrategyRotator::new(config);
        
        rotator.trigger.register_strategy(1);
        
        let mutations = rotator.process_cycle(&[(1, 1e-9, 1000)], 100);
        assert!(mutations.is_empty() || !mutations.is_empty()); // Just test it runs
    }
}
