//! Water Rights Alpha Module
//! Cross-references aquifer depletion with water futures and agricultural markets

use alloc::vec::Vec;
use core::fmt;

/// Error types for water rights alpha
#[derive(Debug, Clone, PartialEq)]
pub enum WaterRightsError {
    DataUnavailable,
    InvalidRegion,
    MarketClosed,
}

impl fmt::Display for WaterRightsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DataUnavailable => write!(f, "Market data unavailable"),
            Self::InvalidRegion => write!(f, "Invalid geographic region"),
            Self::MarketClosed => write!(f, "Market is closed"),
        }
    }
}

/// Agricultural commodity types sensitive to water scarcity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaterSensitiveCommodity {
    Almonds,
    Rice,
    Cotton,
    Corn,
    Wheat,
    Soybeans,
    Cattle,
}

/// Water market quote
#[derive(Debug, Clone)]
pub struct WaterMarketQuote {
    pub region: &'static str,
    pub price_per_af: f64,  // Price per acre-foot
    pub volume_available: f64,
    pub bid: f64,
    pub ask: f64,
    pub timestamp_us: u64,
}

/// Commodity exposure to water stress
#[derive(Debug, Clone)]
pub struct CommodityWaterExposure {
    pub commodity: WaterSensitiveCommodity,
    /// Water intensity (af/ton)
    pub water_intensity: f64,
    /// Regional production concentration (0-1)
    pub regional_concentration: f64,
    /// Irrigation dependency (0-1)
    pub irrigation_dependency: f64,
}

/// Alpha signal for water-related trades
#[derive(Debug, Clone)]
pub struct WaterRightsAlphaSignal {
    pub signal_type: SignalType,
    pub target: &'static str,
    pub direction: Direction,
    pub strength: f64,
    pub expected_return: f64,
    pub time_horizon_days: usize,
    pub confidence: f64,
    pub rationale: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    WaterFutures,
    AgriculturalCommodity,
    Equity,
    Bond,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Long,
    Short,
}

/// Water Rights Alpha Engine
pub struct WaterRightsAlphaEngine {
    /// Current water market quotes by region
    water_quotes: alloc::collections::BTreeMap<&'static str, WaterMarketQuote>,
    /// Commodity exposures
    commodity_exposures: alloc::collections::BTreeMap<WaterSensitiveCommodity, CommodityWaterExposure>,
    /// Aquifer stress levels by region
    aquifer_stress: alloc::collections::BTreeMap<&'static str, f64>,
    /// Historical signals generated
    signal_history: Vec<WaterRightsAlphaSignal>,
}

impl WaterRightsAlphaEngine {
    pub fn new() -> Self {
        let mut commodity_exposures = alloc::collections::BTreeMap::new();

        // Initialize commodity water exposures
        commodity_exposures.insert(
            WaterSensitiveCommodity::Almonds,
            CommodityWaterExposure {
                commodity: WaterSensitiveCommodity::Almonds,
                water_intensity: 5.8,  // af/ton
                regional_concentration: 0.85,  // California produces 85% of US almonds
                irrigation_dependency: 1.0,
            },
        );
        commodity_exposures.insert(
            WaterSensitiveCommodity::Rice,
            CommodityWaterExposure {
                commodity: WaterSensitiveCommodity::Rice,
                water_intensity: 3.5,
                regional_concentration: 0.5,
                irrigation_dependency: 0.95,
            },
        );
        commodity_exposures.insert(
            WaterSensitiveCommodity::Cotton,
            CommodityWaterExposure {
                commodity: WaterSensitiveCommodity::Cotton,
                water_intensity: 2.8,
                regional_concentration: 0.4,
                irrigation_dependency: 0.7,
            },
        );
        commodity_exposures.insert(
            WaterSensitiveCommodity::Corn,
            CommodityWaterExposure {
                commodity: WaterSensitiveCommodity::Corn,
                water_intensity: 1.2,
                regional_concentration: 0.3,
                irrigation_dependency: 0.4,
            },
        );

        Self {
            water_quotes: alloc::collections::BTreeMap::new(),
            commodity_exposures,
            aquifer_stress: alloc::collections::BTreeMap::new(),
            signal_history: Vec::new(),
        }
    }

    /// Update water market quote
    pub fn update_water_quote(&mut self, quote: WaterMarketQuote) {
        self.water_quotes.insert(quote.region, quote);
    }

    /// Update aquifer stress level for a region
    pub fn update_aquifer_stress(&mut self, region: &'static str, stress_level: f64) {
        self.aquifer_stress.insert(region, stress_level.clamp(0.0, 1.0));
    }

    /// Generate alpha signals based on current state
    pub fn generate_signals(&mut self) -> Vec<WaterRightsAlphaSignal> {
        let mut signals = Vec::new();

        // Generate water futures signals
        for (&region, &stress) in &self.aquifer_stress {
            if stress > 0.6 {
                signals.push(WaterRightsAlphaSignal {
                    signal_type: SignalType::WaterFutures,
                    target: region,
                    direction: Direction::Long,
                    strength: stress,
                    expected_return: stress * 0.15,  // Up to 15% expected return
                    time_horizon_days: 90,
                    confidence: stress * 0.8,
                    rationale: "Aquifer depletion driving water scarcity premium",
                });
            }
        }

        // Generate agricultural commodity signals
        for (&commodity, exposure) in &self.commodity_exposures {
            let commodity_name = match commodity {
                WaterSensitiveCommodity::Almonds => "Almonds",
                WaterSensitiveCommodity::Rice => "Rice",
                WaterSensitiveCommodity::Cotton => "Cotton",
                WaterSensitiveCommodity::Corn => "Corn",
                WaterSensitiveCommodity::Wheat => "Wheat",
                WaterSensitiveCommodity::Soybeans => "Soybeans",
                WaterSensitiveCommodity::Cattle => "Cattle",
            };

            // Check if high stress in concentrated production regions
            let effective_stress = self.calculate_effective_stress(exposure);

            if effective_stress > 0.5 {
                let combined_score = exposure.water_intensity * exposure.irrigation_dependency * effective_stress;

                signals.push(WaterRightsAlphaSignal {
                    signal_type: SignalType::AgriculturalCommodity,
                    target: commodity_name,
                    direction: Direction::Long,
                    strength: combined_score.clamp(0.0, 1.0),
                    expected_return: combined_score * 0.10,
                    time_horizon_days: 180,
                    confidence: combined_score * 0.7,
                    rationale: "Water scarcity impacting production costs",
                });
            }
        }

        // Store signals in history
        self.signal_history.extend(signals.clone());

        signals
    }

    /// Calculate effective stress considering regional concentration
    fn calculate_effective_stress(&self, exposure: &CommodityWaterExposure) -> f64 {
        let mut weighted_stress = 0.0;
        let mut total_weight = 0.0;

        for (&region, &stress) in &self.aquifer_stress {
            // Weight by regional importance (simplified)
            let weight = if region.contains("California") {
                exposure.regional_concentration
            } else if region.contains("Ogallala") {
                0.3
            } else {
                0.1
            };

            weighted_stress += stress * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_stress / total_weight
        } else {
            0.0
        }
    }

    /// Get optimal portfolio allocation based on signals
    pub fn get_portfolio_allocation(&self, max_water_exposure: f64) -> Vec<PortfolioPosition> {
        let mut positions = Vec::new();

        // Aggregate signals by target
        let mut target_scores: alloc::collections::BTreeMap<&str, f64> = alloc::collections::BTreeMap::new();

        for signal in &self.signal_history {
            let entry = target_scores.entry(signal.target).or_insert(0.0);
            *entry += signal.strength * signal.confidence;
        }

        // Create positions sorted by score
        let mut sorted_targets: Vec<_> = target_scores.iter().collect();
        sorted_targets.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(core::cmp::Ordering::Equal));

        let mut remaining_budget = max_water_exposure;

        for (target, score) in sorted_targets {
            if remaining_budget <= 0.0 {
                break;
            }

            let position_size = (score * 0.1).min(remaining_budget);
            positions.push(PortfolioPosition {
                target,
                size: position_size,
                side: Direction::Long,
            });

            remaining_budget -= position_size;
        }

        positions
    }

    /// Get signal history count
    pub fn signal_count(&self) -> usize {
        self.signal_history.len()
    }
}

impl Default for WaterRightsAlphaEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Portfolio position
#[derive(Debug, Clone)]
pub struct PortfolioPosition {
    pub target: &'static str,
    pub size: f64,
    pub side: Direction,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_generation() {
        let mut engine = WaterRightsAlphaEngine::new();

        // Set high stress in California
        engine.update_aquifer_stress("California Central Valley", 0.75);

        // Add water quote
        let quote = WaterMarketQuote {
            region: "California Central Valley",
            price_per_af: 2500.0,
            volume_available: 10000.0,
            bid: 2400.0,
            ask: 2600.0,
            timestamp_us: 1_000_000_000_000,
        };
        engine.update_water_quote(quote);

        let signals = engine.generate_signals();
        assert!(!signals.is_empty());

        // Should have water futures signal
        let water_signals: Vec<_> = signals.iter()
            .filter(|s| s.signal_type == SignalType::WaterFutures)
            .collect();
        assert!(!water_signals.is_empty());
    }
}
