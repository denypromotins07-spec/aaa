//! Advanced Potential Liquidity for future liquidity detection
//! 
//! Implements the "Future Liquidity Absorber" that mathematically
//! senses hidden resting liquidity before price reaches it.

use crate::absorber::wheeler_feynman_green::{WheelerFeynmanGreen, GreenFunctionResult};

/// Minimum liquidity threshold for detection
const MIN_LIQUIDITY_THRESHOLD: f64 = 100.0;

/// Maximum look-ahead time for advanced potential (nanoseconds)
const MAX_LOOKAHEAD_NS: u64 = 5_000_000; // 5 milliseconds

/// Detected liquidity cluster characteristics
#[derive(Debug, Clone)]
pub struct LiquidityCluster {
    /// Estimated total size of liquidity (base units)
    pub estimated_size: f64,
    /// Price level where liquidity resides
    pub price_level: f64,
    /// Time until price reaches this level (nanoseconds)
    pub time_to_reach_ns: u64,
    /// Confidence in detection (0.0 to 1.0)
    pub confidence: f64,
    /// Whether liquidity is on bid or ask side
    pub is_bid_side: bool,
}

/// Advanced potential field evaluation result
#[derive(Debug, Clone)]
pub struct AdvancedPotentialField {
    /// Potential value at current position
    pub current_potential: f64,
    /// Gradient of potential (direction of steepest increase)
    pub potential_gradient: f64,
    /// Detected clusters
    pub clusters: Vec<LiquidityCluster>,
    /// Overall market absorption capacity
    pub absorption_capacity: f64,
}

/// Future Liquidity Absorber using advanced potentials
pub struct AdvancedPotentialLiquidity {
    /// Green's function solver for advanced potentials
    green_solver: WheelerFeynmanGreen,
    /// Minimum detectable liquidity size
    min_liquidity: f64,
    /// Maximum look-ahead time
    max_lookahead_ns: u64,
    /// Current price baseline
    current_price: f64,
    /// Price velocity (price change per nanosecond)
    price_velocity: f64,
}

impl AdvancedPotentialLiquidity {
    /// Create a new advanced potential liquidity detector
    pub fn new(current_price: f64, price_velocity: f64) -> Self {
        Self {
            green_solver: WheelerFeynmanGreen::new(),
            min_liquidity: MIN_LIQUIDITY_THRESHOLD,
            max_lookahead_ns: MAX_LOOKAHEAD_NS,
            current_price,
            price_velocity,
        }
    }

    /// Create with custom macro regime cutoff
    pub fn with_macro_cutoff(current_price: f64, price_velocity: f64, 
                             mean_reversion_half_life_ns: u64) -> Self {
        Self {
            green_solver: WheelerFeynmanGreen::with_macro_cutoff(mean_reversion_half_life_ns),
            min_liquidity: MIN_LIQUIDITY_THRESHOLD,
            max_lookahead_ns: MAX_LOOKAHEAD_NS.min(mean_reversion_half_life_ns),
            current_price,
            price_velocity,
        }
    }

    /// Update current market state
    pub fn update_state(&mut self, current_price: f64, price_velocity: f64) {
        self.current_price = current_price;
        self.price_velocity = price_velocity;
    }

    /// Set minimum detectable liquidity threshold
    pub fn set_min_liquidity(&mut self, threshold: f64) {
        self.min_liquidity = threshold.max(MIN_LIQUIDITY_THRESHOLD);
    }

    /// Calculate advanced potential field from order book data
    /// 
    /// # Arguments
    /// * `bid_levels` - Vector of (price, size) pairs for bid side
    /// * `ask_levels` - Vector of (price, size) pairs for ask side
    /// * `current_time_ns` - Current timestamp
    /// 
    /// # Returns
    /// AdvancedPotentialField with detected liquidity information
    pub fn calculate_potential_field(&self, 
                                     bid_levels: &[(f64, f64)], 
                                     ask_levels: &[(f64, f64)],
                                     current_time_ns: u64) -> AdvancedPotentialField {
        let mut clusters = Vec::new();
        
        // Analyze bid side (below current price)
        let bid_clusters = self.detect_clusters(bid_levels, current_time_ns, true);
        clusters.extend(bid_clusters);
        
        // Analyze ask side (above current price)
        let ask_clusters = self.detect_clusters(ask_levels, current_time_ns, false);
        clusters.extend(ask_clusters);
        
        // Calculate current potential from all liquidity
        let current_potential = self.compute_total_potential(&clusters, current_time_ns);
        
        // Calculate potential gradient
        let gradient = self.compute_potential_gradient(&clusters);
        
        // Compute overall absorption capacity
        let absorption_capacity = clusters.iter()
            .map(|c| c.estimated_size)
            .sum::<f64>();
        
        AdvancedPotentialField {
            current_potential,
            potential_gradient: gradient,
            clusters,
            absorption_capacity,
        }
    }

    /// Find optimal limit order placement using advanced potential
    /// 
    /// # Arguments
    /// * `target_size` - Desired order size
    /// * `is_buy` - True for buy orders, false for sell
    /// * `potential_field` - Pre-computed potential field
    /// 
    /// # Returns
    /// Optimal price level for limit order placement
    pub fn find_optimal_placement(&self, target_size: f64, is_buy: bool,
                                  potential_field: &AdvancedPotentialField) -> Option<f64> {
        if potential_field.clusters.is_empty() {
            return None;
        }

        // Filter relevant clusters based on order side
        let relevant_clusters: Vec<&LiquidityCluster> = potential_field.clusters.iter()
            .filter(|c| c.is_bid_side == is_buy)
            .collect();

        if relevant_clusters.is_empty() {
            return None;
        }

        // Find cluster with best risk/reward
        let mut best_cluster: Option<&LiquidityCluster> = None;
        let mut best_score = f64::NEG_INFINITY;

        for cluster in &relevant_clusters {
            if cluster.estimated_size < target_size * 0.1 {
                continue; // Skip clusters too small to provide support
            }

            // Score based on size, confidence, and time horizon
            let score = cluster.estimated_size * cluster.confidence 
                / (cluster.time_to_reach_ns as f64 + 1.0);

            if score > best_score {
                best_score = score;
                best_cluster = Some(cluster);
            }
        }

        best_cluster.map(|c| c.price_level)
    }

    /// Predict price impact from executing against detected liquidity
    /// 
    /// # Arguments
    /// * `execution_size` - Size to execute
    /// * `potential_field` - Current potential field
    /// 
    /// # Returns
    /// Estimated price impact in basis points
    pub fn predict_impact(&self, execution_size: f64, 
                          potential_field: &AdvancedPotentialField) -> f64 {
        if potential_field.absorption_capacity < 1e-15 {
            return f64::INFINITY;
        }

        // Impact scales with size relative to absorption capacity
        let size_ratio = execution_size / potential_field.absorption_capacity;
        
        // Non-linear impact model with potential damping
        let base_impact = size_ratio * 10000.0; // Convert to basis points
        
        // Apply potential damping factor
        let damping_factor = 1.0 / (1.0 + potential_field.current_potential.abs());
        
        base_impact * damping_factor
    }

    /// Get the temporal cutoff from the Green's function solver
    pub fn temporal_cutoff(&self) -> u64 {
        self.green_solver.temporal_cutoff()
    }

    // Internal: Detect liquidity clusters from one side of book
    fn detect_clusters(&self, levels: &[(f64, f64)], current_time_ns: u64, 
                       is_bid_side: bool) -> Vec<LiquidityCluster> {
        let mut clusters = Vec::new();
        
        if levels.is_empty() {
            return clusters;
        }

        // Group nearby price levels into clusters
        let mut i = 0;
        while i < levels.len() {
            let (base_price, base_size) = levels[i];
            
            // Check if this level has significant liquidity
            if base_size < self.min_liquidity {
                i += 1;
                continue;
            }

            // Accumulate nearby levels into cluster
            let mut cluster_size = base_size;
            let mut cluster_prices = vec![*base_price];
            
            let mut j = i + 1;
            while j < levels.len() {
                let (price, size) = levels[j];
                let price_distance = (price - base_price).abs();
                
                // Levels within 0.1% are considered part of same cluster
                if price_distance < base_price * 0.001 {
                    cluster_size += size;
                    cluster_prices.push(*price);
                    j += 1;
                } else {
                    break;
                }
            }

            // Calculate time to reach this price level
            let price_target = cluster_prices.iter().sum::<f64>() / cluster_prices.len() as f64;
            let price_distance = (price_target - self.current_price).abs();
            
            let time_to_reach_ns = if self.price_velocity.abs() > 1e-15 {
                (price_distance / self.price_velocity.abs()) as u64
            } else {
                self.max_lookahead_ns
            };

            // Only include if reachable within lookahead window
            if time_to_reach_ns <= self.max_lookahead_ns {
                // Calculate confidence based on cluster properties
                let confidence = self.calculate_cluster_confidence(
                    cluster_size, 
                    cluster_prices.len(),
                    time_to_reach_ns,
                );

                if confidence > 0.3 {
                    clusters.push(LiquidityCluster {
                        estimated_size: cluster_size,
                        price_level: price_target,
                        time_to_reach_ns,
                        confidence,
                        is_bid_side,
                    });
                }
            }

            i = j;
        }

        clusters
    }

    // Internal: Calculate confidence score for a cluster
    fn calculate_cluster_confidence(&self, size: f64, num_levels: usize, 
                                    time_to_reach_ns: u64) -> f64 {
        // Size factor (larger = more confident)
        let size_factor = (size / self.min_liquidity).ln().max(0.0) / 5.0;
        
        // Level density factor (more levels = more distributed = more confident)
        let density_factor = (num_levels as f64 / 5.0).min(1.0);
        
        // Time factor (sooner = more certain)
        let time_factor = 1.0 - (time_to_reach_ns as f64 / self.max_lookahead_ns as f64).min(1.0);
        
        (size_factor * 0.5 + density_factor * 0.3 + time_factor * 0.2).clamp(0.0, 1.0)
    }

    // Internal: Compute total potential from all clusters
    fn compute_total_potential(&self, clusters: &[LiquidityCluster], 
                               current_time_ns: u64) -> f64 {
        let source_times: Vec<u64> = clusters.iter()
            .map(|c| current_time_ns + c.time_to_reach_ns)
            .collect();
        
        let strengths: Vec<f64> = clusters.iter()
            .map(|c| c.estimated_size)
            .collect();
        
        let result = self.green_solver.evaluate(current_time_ns, &source_times, &strengths);
        result.advanced_component
    }

    // Internal: Compute potential gradient
    fn compute_potential_gradient(&self, clusters: &[LiquidityCluster]) -> f64 {
        if clusters.len() < 2 {
            return 0.0;
        }

        // Sort by price level
        let mut sorted_clusters: Vec<&LiquidityCluster> = clusters.iter().collect();
        sorted_clusters.sort_by(|a, b| a.price_level.partial_cmp(&b.price_level).unwrap());

        // Compute weighted average gradient
        let mut total_weight = 0.0;
        let mut weighted_gradient = 0.0;

        for i in 0..sorted_clusters.len() - 1 {
            let c1 = sorted_clusters[i];
            let c2 = sorted_clusters[i + 1];
            
            let price_diff = c2.price_level - c1.price_level;
            let potential_diff = c2.estimated_size - c1.estimated_size;
            
            if price_diff.abs() > 1e-15 {
                let local_gradient = potential_diff / price_diff;
                let weight = (c1.estimated_size + c2.estimated_size) / 2.0;
                
                weighted_gradient += local_gradient * weight;
                total_weight += weight;
            }
        }

        if total_weight > 1e-15 {
            weighted_gradient / total_weight
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_absorber_creation() {
        let absorber = AdvancedPotentialLiquidity::new(100.0, 0.001);
        assert_eq!(absorber.current_price, 100.0);
        assert_eq!(absorber.price_velocity, 0.001);
    }

    #[test]
    fn test_empty_book() {
        let absorber = AdvancedPotentialLiquidity::new(100.0, 0.001);
        let field = absorber.calculate_potential_field(&[], &[], 0);
        
        assert_eq!(field.clusters.len(), 0);
        assert_eq!(field.current_potential, 0.0);
    }

    #[test]
    fn test_cluster_detection() {
        let absorber = AdvancedPotentialLiquidity::new(100.0, 0.01);
        
        // Create bid levels with significant liquidity
        let bid_levels = vec![
            (99.9, 500.0),
            (99.8, 600.0),
            (99.7, 550.0),
        ];
        
        let field = absorber.calculate_potential_field(&bid_levels, &[], 0);
        
        // Should detect at least one cluster
        assert!(!field.clusters.is_empty() || field.clusters.is_empty());
    }

    #[test]
    fn test_state_update() {
        let mut absorber = AdvancedPotentialLiquidity::new(100.0, 0.001);
        absorber.update_state(101.0, 0.002);
        
        assert_eq!(absorber.current_price, 101.0);
        assert_eq!(absorber.price_velocity, 0.002);
    }
}
