//! Cross-Venue Margin Aggregator.
//! 
//! Nets offsetting positions across multiple exchanges to calculate
//! true portfolio exposure and optimize collateral usage.

use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;

/// Epsilon for comparisons
const EPSILON: f64 = 1e-9;

/// Venue identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Venue {
    Binance,
    Bybit,
    Deribit,
    OKX,
    Coinbase,
    Kraken,
    Custom(u8),
}

impl Venue {
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Binance => "binance",
            Self::Bybit => "bybit",
            Self::Deribit => "deribit",
            Self::OKX => "okx",
            Self::Coinbase => "coinbase",
            Self::Kraken => "kraken",
            Self::Custom(_) => "custom",
        }
    }
}

/// Position on a specific venue
#[derive(Debug, Clone)]
pub struct VenuePosition {
    /// Venue where position is held
    pub venue: Venue,
    /// Symbol (normalized across venues)
    pub symbol: String,
    /// Position size (positive = long, negative = short)
    pub signed_size: f64,
    /// Entry price in USD
    pub entry_price: f64,
    /// Current mark price in USD
    pub mark_price: f64,
    /// Collateral posted for this position
    pub collateral: f64,
    /// Unrealized P&L
    pub unrealized_pnl: f64,
}

/// Netted position after cross-venue aggregation
#[derive(Debug, Clone)]
pub struct NettedPosition {
    /// Normalized symbol
    pub symbol: String,
    /// Gross long size (sum of all long positions)
    pub gross_long: f64,
    /// Gross short size (sum of all short positions)
    pub gross_short: f64,
    /// Net position size
    pub net_size: f64,
    /// Total collateral across all venues
    pub total_collateral: f64,
    /// Net unrealized P&L
    pub net_unrealized_pnl: f64,
    /// Whether this represents a hedged position
    pub is_hedged: bool,
    /// Hedge ratio (min(long, short) / max(long, short))
    pub hedge_ratio: f64,
}

/// Portfolio-level margin summary
#[derive(Debug, Clone)]
pub struct PortfolioMarginSummary {
    /// Total initial margin required (gross)
    pub gross_initial_margin: f64,
    /// Total initial margin after netting benefits
    pub netted_initial_margin: f64,
    /// Total maintenance margin required
    pub total_maintenance_margin: f64,
    /// Total collateral posted across all venues
    pub total_collateral: f64,
    /// Free collateral available
    pub free_collateral: f64,
    /// Portfolio margin ratio
    pub margin_ratio: f64,
    /// Net delta exposure in USD
    pub net_delta_usd: f64,
    /// Count of venues with positions
    pub venue_count: usize,
    /// Count of unique symbols
    pub symbol_count: usize,
}

/// Cross-Venue Margin Aggregator
/// 
/// Aggregates positions across multiple exchanges, identifies
/// offsetting positions, and calculates netted margin requirements.
pub struct CrossVenueMarginAggregator {
    /// Positions by venue and symbol
    positions: parking_lot::Mutex<HashMap<(Venue, String), VenuePosition>>,
    /// Count of aggregations performed
    aggregation_count: AtomicU64,
    /// Total collateral freed through netting
    collateral_freed: AtomicU64, // Stored as bits for atomic f64
}

unsafe impl Send for CrossVenueMarginAggregator {}
unsafe impl Sync for CrossVenueMarginAggregator {}

impl CrossVenueMarginAggregator {
    /// Create a new cross-venue aggregator.
    pub fn new() -> Self {
        Self {
            positions: parking_lot::Mutex::new(HashMap::new()),
            aggregation_count: AtomicU64::new(0),
            collateral_freed: AtomicU64::new(0),
        }
    }

    /// Update or add a position.
    #[inline]
    pub fn update_position(&self, position: VenuePosition) {
        let key = (position.venue, position.symbol.clone());
        let mut positions = self.positions.lock();
        positions.insert(key, position);
    }

    /// Remove a position.
    #[inline]
    pub fn remove_position(&self, venue: Venue, symbol: &str) {
        let mut positions = self.positions.lock();
        positions.remove(&(venue, symbol.to_string()));
    }

    /// Get all positions for a specific symbol across venues.
    #[inline]
    pub fn get_symbol_positions(&self, symbol: &str) -> Vec<VenuePosition> {
        let positions = self.positions.lock();
        positions
            .iter()
            .filter(|((_, s), _)| s == symbol)
            .map(|(_, p)| p.clone())
            .collect()
    }

    /// Calculate netted position for a symbol.
    #[inline]
    pub fn calculate_netted_position(&self, symbol: &str) -> Option<NettedPosition> {
        let venue_positions = self.get_symbol_positions(symbol);
        
        if venue_positions.is_empty() {
            return None;
        }

        let mut gross_long = 0.0;
        let mut gross_short = 0.0;
        let mut total_collateral = 0.0;
        let mut net_unrealized_pnl = 0.0;

        for pos in &venue_positions {
            total_collateral += pos.collateral;
            net_unrealized_pnl += pos.unrealized_pnl;

            if pos.signed_size > 0.0 {
                gross_long += pos.signed_size;
            } else {
                gross_short += pos.signed_size.abs();
            }
        }

        let net_size = gross_long - gross_short;
        let is_hedged = gross_long > EPSILON && gross_short > EPSILON;
        
        let max_side = gross_long.max(gross_short);
        let min_side = gross_long.min(gross_short);
        let hedge_ratio = if max_side > EPSILON {
            min_side / max_side
        } else {
            0.0
        };

        Some(NettedPosition {
            symbol: symbol.to_string(),
            gross_long,
            gross_short,
            net_size,
            total_collateral,
            net_unrealized_pnl,
            is_hedged,
            hedge_ratio,
        })
    }

    /// Calculate portfolio-wide margin summary with netting benefits.
    pub fn calculate_portfolio_summary(&self) -> PortfolioMarginSummary {
        let positions = self.positions.lock();
        
        // Group by symbol
        let mut by_symbol: HashMap<String, Vec<&VenuePosition>> = HashMap::new();
        for ((_, symbol), position) in positions.iter() {
            by_symbol.entry(symbol.clone()).or_default().push(position);
        }

        let mut gross_initial_margin = 0.0;
        let mut netted_initial_margin = 0.0;
        let mut total_maintenance_margin = 0.0;
        let mut total_collateral = 0.0;
        let mut net_delta_usd = 0.0;

        for (symbol, venue_positions) in &by_symbol {
            let mut symbol_gross_long = 0.0;
            let mut symbol_gross_short = 0.0;
            let mut symbol_collateral = 0.0;

            for pos in venue_positions {
                let notional = pos.mark_price * pos.signed_size.abs();
                symbol_collateral += pos.collateral;
                
                // Assume 10% IM requirement (simplified)
                let im_requirement = notional * 0.10;
                let mm_requirement = notional * 0.05; // 5% MM

                gross_initial_margin += im_requirement;
                total_maintenance_margin += mm_requirement;
                total_collateral += pos.collateral;

                if pos.signed_size > 0.0 {
                    symbol_gross_long += pos.signed_size;
                    net_delta_usd += pos.signed_size * pos.mark_price;
                } else {
                    symbol_gross_short += pos.signed_size.abs();
                    net_delta_usd -= pos.signed_size.abs() * pos.mark_price;
                }
            }

            // Apply netting benefit for hedged positions
            // Only pay margin on the larger side
            let net_notional = (symbol_gross_long - symbol_gross_short).abs() 
                * venue_positions.first().map(|p| p.mark_price).unwrap_or(1.0);
            
            // For hedged positions, use reduced margin (e.g., 20% of gross)
            let is_hedged = symbol_gross_long > EPSILON && symbol_gross_short > EPSILON;
            if is_hedged {
                let hedge_ratio = symbol_gross_long.min(symbol_gross_short) 
                    / symbol_gross_long.max(symbol_gross_short).max(EPSILON);
                
                // Margin reduction based on hedge quality
                let reduction_factor = 1.0 - (hedge_ratio * 0.8); // Up to 80% reduction
                netted_initial_margin += gross_initial_margin * reduction_factor;
            } else {
                netted_initial_margin += gross_initial_margin;
            }

            let _ = symbol; // Suppress unused warning
        }

        let free_collateral = total_collateral - total_maintenance_margin;
        let margin_ratio = if total_collateral > EPSILON {
            total_maintenance_margin / total_collateral
        } else {
            1.0
        };

        self.aggregation_count.fetch_add(1, Ordering::Relaxed);
        
        // Track collateral freed
        let freed = gross_initial_margin - netted_initial_margin;
        if freed > 0.0 {
            self.collateral_freed.store(f64::to_bits(freed), Ordering::Relaxed);
        }

        PortfolioMarginSummary {
            gross_initial_margin,
            netted_initial_margin,
            total_maintenance_margin,
            total_collateral,
            free_collateral,
            margin_ratio,
            net_delta_usd,
            venue_count: by_symbol.values().flatten().map(|p| p.venue).collect::<std::collections::HashSet<_>>().len(),
            symbol_count: by_symbol.len(),
        }
    }

    /// Identify opportunities for margin optimization.
    pub fn find_optimization_opportunities(&self) -> Vec<OptimizationOpportunity> {
        let positions = self.positions.lock();
        let mut opportunities = Vec::new();

        // Group by symbol
        let mut by_symbol: HashMap<String, Vec<&VenuePosition>> = HashMap::new();
        for ((_, symbol), position) in positions.iter() {
            by_symbol.entry(symbol.clone()).or_default().push(position);
        }

        for (symbol, venue_positions) in &by_symbol {
            if venue_positions.len() < 2 {
                continue;
            }

            let mut longs: Vec<&VenuePosition> = venue_positions.iter()
                .filter(|p| p.signed_size > 0.0)
                .copied()
                .collect();
            let mut shorts: Vec<&VenuePosition> = venue_positions.iter()
                .filter(|p| p.signed_size < 0.0)
                .copied()
                .collect();

            if !longs.is_empty() && !shorts.is_empty() {
                // Found a hedged position - check if we can optimize
                let total_long: f64 = longs.iter().map(|p| p.signed_size).sum();
                let total_short: f64 = shorts.iter().map(|p| p.signed_size.abs()).sum();
                
                let hedge_ratio = total_long.min(total_short) / total_long.max(total_short).max(EPSILON);
                
                if hedge_ratio > 0.3 {
                    // Significant hedge - could potentially move to portfolio margin
                    let current_collateral: f64 = venue_positions.iter().map(|p| p.collateral).sum();
                    let potential_savings = current_collateral * hedge_ratio * 0.5; // Estimate 50% savings on hedged portion

                    opportunities.push(OptimizationOpportunity {
                        symbol: symbol.clone(),
                        opportunity_type: OptimizationType::PortfolioMargin,
                        potential_savings,
                        description: format!(
                            "Hedged position with {:.1}% hedge ratio. Consider portfolio margin.",
                            hedge_ratio * 100.0
                        ),
                    });
                }

                // Check for collateral imbalance (one venue has excess)
                let avg_collateral_ratio = current_collateral / venue_positions.len() as f64;
                for pos in venue_positions {
                    if pos.collateral > avg_collateral_ratio * 1.5 {
                        opportunities.push(OptimizationOpportunity {
                            symbol: symbol.clone(),
                            opportunity_type: OptimizationType::CollateralRebalance,
                            potential_savings: 0.0, // Hard to quantify
                            description: format!(
                                "{} has excess collateral (${:.2}). Consider rebalancing.",
                                pos.venue.as_str(),
                                pos.collateral
                            ),
                        });
                    }
                }
            }
        }

        opportunities
    }

    /// Get aggregation statistics.
    pub fn stats(&self) -> AggregatorStats {
        let positions = self.positions.lock();
        AggregatorStats {
            total_positions: positions.len(),
            aggregation_count: self.aggregation_count.load(Ordering::Relaxed),
            collateral_freed: f64::from_bits(self.collateral_freed.load(Ordering::Relaxed)),
        }
    }
}

impl Default for CrossVenueMarginAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Type of optimization opportunity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationType {
    /// Move to portfolio margin account
    PortfolioMargin,
    /// Rebalance collateral between venues
    CollateralRebalance,
    /// Close offsetting positions to free capital
    CloseHedge,
    /// Increase leverage on under-utilized positions
    LeverageOptimize,
}

/// An optimization opportunity identified by the aggregator
#[derive(Debug, Clone)]
pub struct OptimizationOpportunity {
    pub symbol: String,
    pub opportunity_type: OptimizationType,
    pub potential_savings: f64,
    pub description: String,
}

/// Statistics from the aggregator
#[derive(Debug, Clone)]
pub struct AggregatorStats {
    pub total_positions: usize,
    pub aggregation_count: u64,
    pub collateral_freed: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_position() {
        let agg = CrossVenueMarginAggregator::new();
        
        agg.update_position(VenuePosition {
            venue: Venue::Binance,
            symbol: "BTCUSD".to_string(),
            signed_size: 1.0,
            entry_price: 50_000.0,
            mark_price: 50_000.0,
            collateral: 5_000.0,
            unrealized_pnl: 0.0,
        });

        let netted = agg.calculate_netted_position("BTCUSD").unwrap();
        
        assert_eq!(netted.net_size, 1.0);
        assert!(!netted.is_hedged);
        assert_eq!(netted.hedge_ratio, 0.0);
    }

    #[test]
    fn test_cross_venue_hedge() {
        let agg = CrossVenueMarginAggregator::new();
        
        // Long on Binance
        agg.update_position(VenuePosition {
            venue: Venue::Binance,
            symbol: "BTCUSD".to_string(),
            signed_size: 1.0,
            entry_price: 50_000.0,
            mark_price: 50_000.0,
            collateral: 5_000.0,
            unrealized_pnl: 0.0,
        });

        // Short on Bybit (same size - perfect hedge)
        agg.update_position(VenuePosition {
            venue: Venue::Bybit,
            symbol: "BTCUSD".to_string(),
            signed_size: -1.0,
            entry_price: 50_000.0,
            mark_price: 50_000.0,
            collateral: 5_000.0,
            unrealized_pnl: 0.0,
        });

        let netted = agg.calculate_netted_position("BTCUSD").unwrap();
        
        assert_eq!(netted.gross_long, 1.0);
        assert_eq!(netted.gross_short, 1.0);
        assert_eq!(netted.net_size, 0.0);
        assert!(netted.is_hedged);
        assert_eq!(netted.hedge_ratio, 1.0);
    }

    #[test]
    fn test_portfolio_summary_with_netting() {
        let agg = CrossVenueMarginAggregator::new();
        
        // Add hedged positions
        agg.update_position(VenuePosition {
            venue: Venue::Binance,
            symbol: "BTCUSD".to_string(),
            signed_size: 2.0,
            entry_price: 50_000.0,
            mark_price: 50_000.0,
            collateral: 10_000.0,
            unrealized_pnl: 0.0,
        });

        agg.update_position(VenuePosition {
            venue: Venue::Bybit,
            symbol: "BTCUSD".to_string(),
            signed_size: -1.0,
            entry_price: 50_000.0,
            mark_price: 50_000.0,
            collateral: 5_000.0,
            unrealized_pnl: 0.0,
        });

        let summary = agg.calculate_portfolio_summary();
        
        // Net delta should reflect the net long position
        assert!(summary.net_delta_usd > 0.0);
        
        // Netted margin should be less than gross due to hedge
        assert!(summary.netted_initial_margin <= summary.gross_initial_margin);
    }

    #[test]
    fn test_optimization_detection() {
        let agg = CrossVenueMarginAggregator::new();
        
        // Create a hedged position
        agg.update_position(VenuePosition {
            venue: Venue::Binance,
            symbol: "ETHUSD".to_string(),
            signed_size: 10.0,
            entry_price: 3_000.0,
            mark_price: 3_000.0,
            collateral: 3_000.0,
            unrealized_pnl: 0.0,
        });

        agg.update_position(VenuePosition {
            venue: Venue::Bybit,
            symbol: "ETHUSD".to_string(),
            signed_size: -8.0,
            entry_price: 3_000.0,
            mark_price: 3_000.0,
            collateral: 2_400.0,
            unrealized_pnl: 0.0,
        });

        let opportunities = agg.find_optimization_opportunities();
        
        // Should detect the hedge as an optimization opportunity
        assert!(opportunities.iter().any(|o| o.opportunity_type == OptimizationType::PortfolioMargin));
    }

    #[test]
    fn test_multiple_symbols() {
        let agg = CrossVenueMarginAggregator::new();
        
        agg.update_position(VenuePosition {
            venue: Venue::Binance,
            symbol: "BTCUSD".to_string(),
            signed_size: 1.0,
            entry_price: 50_000.0,
            mark_price: 50_000.0,
            collateral: 5_000.0,
            unrealized_pnl: 0.0,
        });

        agg.update_position(VenuePosition {
            venue: Venue::Binance,
            symbol: "ETHUSD".to_string(),
            signed_size: 10.0,
            entry_price: 3_000.0,
            mark_price: 3_000.0,
            collateral: 3_000.0,
            unrealized_pnl: 0.0,
        });

        let summary = agg.calculate_portfolio_summary();
        
        assert_eq!(summary.symbol_count, 2);
        assert_eq!(summary.venue_count, 1);
    }

    #[test]
    fn test_remove_position() {
        let agg = CrossVenueMarginAggregator::new();
        
        agg.update_position(VenuePosition {
            venue: Venue::Binance,
            symbol: "BTCUSD".to_string(),
            signed_size: 1.0,
            entry_price: 50_000.0,
            mark_price: 50_000.0,
            collateral: 5_000.0,
            unrealized_pnl: 0.0,
        });

        assert!(agg.calculate_netted_position("BTCUSD").is_some());

        agg.remove_position(Venue::Binance, "BTCUSD");
        
        assert!(agg.calculate_netted_position("BTCUSD").is_none());
    }
}
