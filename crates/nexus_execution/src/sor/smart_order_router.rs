//! Smart Order Router (SOR)
//! 
//! Integrates Iceberg Sniper and Queue Position Tracker to make
//! intelligent routing decisions for order execution.

use crate::algos::{IcebergSniper, IcebergConfig, IcebergState, OrderSide, QueuePositionTracker, QueueTrackerConfig, QueueAction};

/// SOR configuration
#[derive(Debug, Clone)]
pub struct SorConfig {
    /// Minimum order size to trigger iceberg slicing
    pub iceberg_threshold: i64,
    /// Enable queue position tracking
    pub enable_queue_tracking: bool,
    /// Enable iceberg detection
    pub enable_iceberg_detection: bool,
    /// Default routing strategy
    pub default_strategy: RoutingStrategy,
}

impl Default for SorConfig {
    fn default() -> Self {
        Self {
            iceberg_threshold: 50000, // 0.05 BTC equivalent
            enable_queue_tracking: true,
            enable_iceberg_detection: true,
            default_strategy: RoutingStrategy::Smart,
        }
    }
}

/// Routing strategy selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Use smart routing based on market conditions
    Smart,
    /// Always use iceberg slicing
    Iceberg,
    /// Direct market order (urgent execution)
    Direct,
    /// Passive limit order only
    PassiveOnly,
}

/// Execution decision from SOR
#[derive(Debug, Clone)]
pub struct ExecutionDecision {
    /// Recommended routing strategy
    pub strategy: RoutingStrategy,
    /// Order size to execute
    pub quantity: i64,
    /// Limit price (if applicable)
    pub limit_price: Option<i64>,
    /// Time in force
    pub time_in_force: &'static str,
    /// Reason for the decision
    pub reason: &'static str,
    /// VPIN level affecting decision
    pub vpin_level: u32,
}

/// SOR statistics
#[derive(Debug, Clone, Default)]
pub struct SorStats {
    pub total_orders_routed: u64,
    pub iceberg_orders: u64,
    pub direct_orders: u64,
    pub passive_orders: u64,
    pub queue_jumps_triggered: u64,
    pub toxicity_pauses: u64,
}

/// Smart Order Router
pub struct SmartOrderRouter {
    config: SorConfig,
    iceberg_sniper: Option<IcebergSniper>,
    queue_tracker: Option<QueuePositionTracker>,
    stats: SorStats,
}

impl SmartOrderRouter {
    pub fn new(config: SorConfig) -> Self {
        let iceberg_sniper = Some(IcebergSniper::new(
            IcebergConfig::default(),
            0,
        ));
        
        let queue_tracker = if config.enable_queue_tracking {
            Some(QueuePositionTracker::new(QueueTrackerConfig::default()))
        } else {
            None
        };

        Self {
            config,
            iceberg_sniper,
            queue_tracker,
            stats: SorStats::default(),
        }
    }

    /// Route an order and return execution decision
    pub fn route_order(
        &mut self,
        quantity: i64,
        side: OrderSide,
        current_price: i64,
        vpin_bps: u32,
        visible_depth: i64,
    ) -> ExecutionDecision {
        self.stats.total_orders_routed += 1;

        // Check VPIN toxicity first
        if vpin_bps >= 7000 {
            self.stats.toxicity_pauses += 1;
            self.stats.passive_orders += 1;
            
            return ExecutionDecision {
                strategy: RoutingStrategy::PassiveOnly,
                quantity,
                limit_price: Some(current_price),
                time_in_force: "GTC",
                reason: "High VPIN toxicity - passive only",
                vpin_level: vpin_bps,
            };
        }

        // Check if order should be sliced (iceberg)
        if quantity >= self.config.iceberg_threshold {
            self.stats.iceberg_orders += 1;
            
            if let Some(ref mut sniper) = self.iceberg_sniper {
                sniper.update_vpin(vpin_bps);
                
                if sniper.is_passive_only() {
                    return ExecutionDecision {
                        strategy: RoutingStrategy::PassiveOnly,
                        quantity,
                        limit_price: Some(current_price),
                        time_in_force: "GTC",
                        reason: "Iceberg paused due to toxicity",
                        vpin_level: vpin_bps,
                    };
                }
            }

            return ExecutionDecision {
                strategy: RoutingStrategy::Iceberg,
                quantity,
                limit_price: Some(current_price),
                time_in_force: "GTC",
                reason: "Large order - using iceberg algorithm",
                vpin_level: vpin_bps,
            };
        }

        // Check queue position if enabled
        if self.config.enable_queue_tracking {
            if let Some(ref mut tracker) = self.queue_tracker {
                // Update tracker with current state
                tracker.update_our_position(current_price, quantity, 0, visible_depth);
                
                // Record a simulated fill for analysis
                tracker.record_fill(current_price, visible_depth / 10, true);
                
                // Get recommended action
                let action = tracker.get_recommended_action(current_price, 0);
                
                match action {
                    QueueAction::CancelAndRequeue => {
                        self.stats.queue_jumps_triggered += 1;
                        return ExecutionDecision {
                            strategy: RoutingStrategy::Direct,
                            quantity,
                            limit_price: Some(current_price + 1), // Price improvement
                            time_in_force: "IOC",
                            reason: "Queue jump recommended",
                            vpin_level: vpin_bps,
                        };
                    }
                    QueueAction::RepriceBetter => {
                        return ExecutionDecision {
                            strategy: RoutingStrategy::PassiveOnly,
                            quantity,
                            limit_price: Some(current_price + 1),
                            time_in_force: "GTC",
                            reason: "Reprice to better position",
                            vpin_level: vpin_bps,
                        };
                    }
                    _ => {}
                }
            }
        }

        // Default: direct execution for small orders
        self.stats.direct_orders += 1;
        
        ExecutionDecision {
            strategy: self.config.default_strategy,
            quantity,
            limit_price: None, // Market order
            time_in_force: "IOC",
            reason: "Small order - direct execution",
            vpin_level: vpin_bps,
        }
    }

    /// Update VPIN for all components
    pub fn update_vpin(&mut self, vpin_bps: u32) {
        if let Some(ref mut sniper) = self.iceberg_sniper {
            sniper.update_vpin(vpin_bps);
        }
    }

    /// Record a fill event for queue analysis
    pub fn record_fill(&mut self, price: i64, quantity: i64, was_at_best: bool) {
        if let Some(ref mut tracker) = self.queue_tracker {
            tracker.record_fill(price, quantity, was_at_best);
        }
    }

    /// Get SOR statistics
    pub fn get_stats(&self) -> SorStats {
        self.stats.clone()
    }

    /// Get iceberg sniper reference
    pub fn get_iceberg_sniper(&self) -> Option<&IcebergSniper> {
        self.iceberg_sniper.as_ref()
    }

    /// Get mutable iceberg sniper reference
    pub fn get_iceberg_sniper_mut(&mut self) -> Option<&mut IcebergSniper> {
        self.iceberg_sniper.as_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_order_direct_routing() {
        let mut sor = SmartOrderRouter::new(SorConfig::default());

        let decision = sor.route_order(
            10000, // Small order (< 50k threshold)
            OrderSide::Buy,
            50000,
            3000, // Low VPIN
            100000,
        );

        assert_eq!(decision.strategy, RoutingStrategy::Smart);
        assert!(decision.limit_price.is_none()); // Market order
    }

    #[test]
    fn test_large_order_iceberg_routing() {
        let mut sor = SmartOrderRouter::new(SorConfig::default());

        let decision = sor.route_order(
            100000, // Large order (> 50k threshold)
            OrderSide::Buy,
            50000,
            3000, // Low VPIN
            100000,
        );

        assert_eq!(decision.strategy, RoutingStrategy::Iceberg);
        assert!(decision.limit_price.is_some()); // Limit order
    }

    #[test]
    fn test_high_vpin_passive_only() {
        let mut sor = SmartOrderRouter::new(SorConfig::default());

        let decision = sor.route_order(
            10000,
            OrderSide::Buy,
            50000,
            8000, // High VPIN (> 0.7)
            100000,
        );

        assert_eq!(decision.strategy, RoutingStrategy::PassiveOnly);
        assert!(decision.reason.contains("toxicity"));
    }

    #[test]
    fn test_statistics_tracking() {
        let mut sor = SmartOrderRouter::new(SorConfig::default());

        // Route small order
        sor.route_order(10000, OrderSide::Buy, 50000, 3000, 100000);
        
        // Route large order
        sor.route_order(100000, OrderSide::Buy, 50000, 3000, 100000);
        
        // Route high VPIN order
        sor.route_order(10000, OrderSide::Buy, 50000, 8000, 100000);

        let stats = sor.get_stats();
        assert_eq!(stats.total_orders_routed, 3);
        assert_eq!(stats.iceberg_orders, 1);
        assert_eq!(stats.toxicity_pauses, 1);
    }
}
