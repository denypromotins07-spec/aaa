//! Terrestrial Fallback Arbitrage Module
//! 
//! Trades terrestrial fiber-optic and subsea cable equities
//! when LEO becomes unusable due to Kessler Syndrome.

use super::kessler_boltzmann_pde::DebrisDensityField;
use super::debris_cloud_expansion::DebrisCloudState;

/// Error types for terrestrial fallback arb
#[derive(Debug, Clone, Copy)]
pub enum TerrestrialArbError {
    InvalidEquityPrice(f64),
    InvalidKesslerThreshold(f64),
    MarketDataStale,
    NumericalInstability,
}

impl core::fmt::Display for TerrestrialArbError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TerrestrialArbError::InvalidEquityPrice(p) => write!(f, "Invalid equity price: {}", p),
            TerrestrialArbError::InvalidKesslerThreshold(t) => {
                write!(f, "Invalid Kessler threshold: {}", t)
            }
            TerrestrialArbError::MarketDataStale => write!(f, "Market data is stale"),
            TerrestrialArbError::NumericalInstability => write!(f, "Numerical instability"),
        }
    }
}

/// Trading signal for terrestrial fallback
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TerrestrialSignal {
    LongFiberOptic,
    ShortFiberOptic,
    LongSubseaCable,
    ShortSubseaCable,
    LongLEOSatellites,
    ShortLEOSatellites,
    Hold,
}

/// Market snapshot for terrestrial arbitrage
#[derive(Debug, Clone, Copy)]
pub struct TerrestrialMarketSnapshot {
    pub fiber_optic_equity_price: f64,
    pub subsea_cable_equity_price: f64,
    pub leo_satellite_equity_price: f64,
    pub satellite_insurance_premium: f64,
    pub timestamp: f64,
}

/// Terrestrial fallback arbitrage engine
pub struct TerrestrialFallbackArbitrage {
    pub kessler_threshold: f64,
    pub leo_exposure_factor: f64,
    pub terrestrial_substitute_factor: f64,
}

impl TerrestrialFallbackArbitrage {
    /// Create new arbitrage engine
    pub fn new(kessler_threshold: f64) -> Result<Self, TerrestrialArbError> {
        if kessler_threshold <= 0.0 || kessler_threshold > 1e6 {
            return Err(TerrestrialArbError::InvalidKesslerThreshold(kessler_threshold));
        }
        
        Ok(Self {
            kessler_threshold,
            leo_exposure_factor: 0.8,
            terrestrial_substitute_factor: 1.2,
        })
    }
    
    /// Assess Kessler risk from debris field
    pub fn assess_kessler_risk(
        &self,
        field: &DebrisDensityField,
    ) -> (f64, bool) {
        let total_density: f64 = field.data.iter().sum();
        let avg_density = total_density / field.data.len() as f64;
        
        let risk_ratio = avg_density / self.kessler_threshold;
        let kessler_active = risk_ratio > 1.0;
        
        (risk_ratio, kessler_active)
    }
    
    /// Generate trading signals based on Kessler risk
    pub fn generate_signals(
        &self,
        field: &DebrisDensityField,
        market: &TerrestrialMarketSnapshot,
    ) -> Result<Vec<TerrestrialSignal>, TerrestrialArbError> {
        if market.fiber_optic_equity_price <= 0.0 
            || market.subsea_cable_equity_price <= 0.0 
            || market.leo_satellite_equity_price <= 0.0 {
            return Err(TerrestrialArbError::InvalidEquityPrice(
                market.fiber_optic_equity_price.min(market.subsea_cable_equity_price)
                    .min(market.leo_satellite_equity_price)
            ));
        }
        
        let (risk_ratio, kessler_active) = self.assess_kessler_risk(field);
        
        let mut signals = Vec::new();
        
        if kessler_active {
            // LEO satellites become risky - short
            signals.push(TerrestrialSignal::ShortLEOSatellites);
            
            // Terrestrial alternatives become valuable - long
            signals.push(TerrestrialSignal::LongFiberOptic);
            signals.push(TerrestrialSignal::LongSubseaCable);
            
            // Insurance premiums should spike
            if market.satellite_insurance_premium < 10_000_000.0 {
                // Insurance underpriced relative to risk
                signals.push(TerrestrialSignal::LongFiberOptic); // Proxy for insurance shorts
            }
        } else if risk_ratio > 0.5 {
            // Elevated but not critical risk
            signals.push(TerrestrialSignal::ShortLEOSatellites);
            signals.push(TerrestrialSignal::Hold); // Wait on terrestrial longs
        } else {
            // Normal conditions
            signals.push(TerrestrialSignal::Hold);
            signals.push(TerrestrialSignal::Hold);
            signals.push(TerrestrialSignal::Hold);
        }
        
        Ok(signals)
    }
    
    /// Calculate expected return from Kessler event
    pub fn calculate_expected_return(
        &self,
        field: &DebrisDensityField,
        investment_horizon_years: f64,
    ) -> f64 {
        let (risk_ratio, _) = self.assess_kessler_risk(field);
        
        // Probability of Kessler syndrome within horizon
        let kessler_prob = 1.0 - (-risk_ratio * investment_horizon_years * 0.1).exp();
        
        // Expected return: gain from terrestrial longs minus loss from satellite shorts
        let terrestrial_gain = kessler_prob * self.terrestrial_substitute_factor;
        let satellite_loss = kessler_prob * self.leo_exposure_factor;
        
        terrestrial_gain - satellite_loss
    }
    
    /// Update exposure factors based on market conditions
    pub fn update_factors(&mut self, leo_market_cap: f64, terrestrial_market_cap: f64) {
        if leo_market_cap > 0.0 && terrestrial_market_cap > 0.0 {
            let ratio = leo_market_cap / terrestrial_market_cap;
            self.leo_exposure_factor = (ratio * 0.5).min(0.9);
            self.terrestrial_substitute_factor = 1.0 + (ratio * 0.3).min(0.5);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_arbitrage_engine() {
        let arb = TerrestrialFallbackArbitrage::new(1000.0).unwrap();
        
        let mut field = DebrisDensityField::new(20, 18).unwrap();
        // Populate with high density to trigger Kessler
        for i in 0..field.altitude_bins {
            for j in 0..field.inclination_bins {
                field.set(i, j, 2000.0);
            }
        }
        
        let market = TerrestrialMarketSnapshot {
            fiber_optic_equity_price: 100.0,
            subsea_cable_equity_price: 150.0,
            leo_satellite_equity_price: 50.0,
            satellite_insurance_premium: 1_000_000.0,
            timestamp: 0.0,
        };
        
        let signals = arb.generate_signals(&field, &market);
        assert!(signals.is_ok());
        
        let sigs = signals.unwrap();
        assert!(sigs.contains(&TerrestrialSignal::ShortLEOSatellites));
    }
    
    #[test]
    fn test_kessler_assessment() {
        let arb = TerrestrialFallbackArbitrage::new(1000.0).unwrap();
        
        let mut field = DebrisDensityField::new(20, 18).unwrap();
        for i in 0..field.altitude_bins {
            for j in 0..field.inclination_bins {
                field.set(i, j, 500.0);
            }
        }
        
        let (ratio, active) = arb.assess_kessler_risk(&field);
        assert!(ratio > 0.0);
        assert!(!active); // Below threshold
    }
}
