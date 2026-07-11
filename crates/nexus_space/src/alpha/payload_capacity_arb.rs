//! Payload Capacity Arbitrage Module
//! 
//! Trades launch capacity futures and satellite constellation equities
//! based on reusability certification events.

use super::bayesian_success_predictor::{BayesianSuccessPredictor, LaunchPhase, TelemetryObservation};

/// Error types for payload arbitrage
#[derive(Debug, Clone, Copy)]
pub enum PayloadArbError {
    InvalidCapacity(f64),
    InvalidPrice(f64),
    MarketDataStale,
    NumericalInstability,
}

impl core::fmt::Display for PayloadArbError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PayloadArbError::InvalidCapacity(c) => write!(f, "Invalid capacity: {}", c),
            PayloadArbError::InvalidPrice(p) => write!(f, "Invalid price: {}", p),
            PayloadArbError::MarketDataStale => write!(f, "Market data is stale"),
            PayloadArbError::NumericalInstability => write!(f, "Numerical instability"),
        }
    }
}

/// Launch vehicle reusability status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReusabilityStatus {
    Expendable,
    RapidReuse,      // < 24 hour turnaround
    StandardReuse,   // < 1 week turnaround
    ExtendedRefurb,  // > 1 week refurbishment
    Lost,
}

/// Market data snapshot
#[derive(Debug, Clone, Copy)]
pub struct MarketSnapshot {
    pub launch_future_price: f64,
    pub satellite_equity_price: f64,
    pub insurance_premium: f64,
    pub timestamp: f64,
}

/// Trading signal generation
#[derive(Debug, Clone, Copy)]
pub enum TradingSignal {
    LongFutures,
    ShortFutures,
    LongSatelliteEquity,
    ShortSatelliteEquity,
    Hold,
}

/// Payload capacity arbitrage engine
pub struct PayloadCapacityArbitrage {
    predictor: BayesianSuccessPredictor,
    baseline_launch_cost: f64,
    reuse_discount_factor: f64,
}

impl PayloadCapacityArbitrage {
    /// Create new arbitrage engine
    pub fn new(baseline_launch_cost: f64) -> Result<Self, PayloadArbError> {
        if baseline_launch_cost <= 0.0 {
            return Err(PayloadArbError::InvalidPrice(baseline_launch_cost));
        }
        
        Ok(Self {
            predictor: BayesianSuccessPredictor::new(),
            baseline_launch_cost,
            reuse_discount_factor: 0.7, // 30% discount for rapid reuse
        })
    }
    
    /// Process telemetry and generate trading signals
    pub fn process_telemetry(
        &mut self,
        obs: &TelemetryObservation,
        market: &MarketSnapshot,
    ) -> Result<TradingSignal, PayloadArbError> {
        // Update success probability
        let success_prob = self.predictor.update(obs)
            .map_err(|_| PayloadArbError::NumericalInstability)?;
        
        // Determine reusability likelihood
        let reuse_status = self.estimate_reusability(obs, success_prob);
        
        // Calculate fair value of launch capacity
        let fair_value = self.calculate_fair_value(reuse_status);
        
        // Generate signal based on mispricing
        let signal = self.generate_signal(fair_value, market);
        
        Ok(signal)
    }
    
    /// Estimate reusability status from telemetry
    fn estimate_reusability(&self, obs: &TelemetryObservation, success_prob: f64) -> ReusabilityStatus {
        if success_prob < 0.5 {
            return ReusabilityStatus::Lost;
        }
        
        // Simplified heuristic based on landing acceleration profile
        match obs.phase {
            LaunchPhase::Separation | LaunchPhase::SECO => {
                if obs.acceleration_ms2.abs() < 3.0 {
                    ReusabilityStatus::RapidReuse
                } else if obs.acceleration_ms2.abs() < 6.0 {
                    ReusabilityStatus::StandardReuse
                } else {
                    ReusabilityStatus::ExtendedRefurb
                }
            }
            _ => ReusabilityStatus::Expendable,
        }
    }
    
    /// Calculate fair value of launch capacity
    fn calculate_fair_value(&self, status: ReusabilityStatus) -> f64 {
        let discount = match status {
            ReusabilityStatus::Expendable => 1.0,
            ReusabilityStatus::RapidReuse => self.reuse_discount_factor * 0.8,
            ReusabilityStatus::StandardReuse => self.reuse_discount_factor,
            ReusabilityStatus::ExtendedRefurb => self.reuse_discount_factor * 1.2,
            ReusabilityStatus::Lost => 0.0,
        };
        
        self.baseline_launch_cost * discount
    }
    
    /// Generate trading signal based on mispricing
    fn generate_signal(&self, fair_value: f64, market: &MarketSnapshot) -> TradingSignal {
        let threshold = 0.05; // 5% mispricing threshold
        
        let mispricing = (fair_value - market.launch_future_price) / market.launch_future_price;
        
        if mispricing > threshold {
            TradingSignal::LongFutures
        } else if mispricing < -threshold {
            TradingSignal::ShortFutures
        } else {
            // Satellite equity signals based on launch cost expectations
            if mispricing < -0.02 {
                // Cheap launches benefit satellite operators
                TradingSignal::LongSatelliteEquity
            } else if mispricing > 0.02 {
                TradingSignal::ShortSatelliteEquity
            } else {
                TradingSignal::Hold
            }
        }
    }
    
    /// Get current success probability
    pub fn success_probability(&self) -> f64 {
        self.predictor.success_probability()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_arbitrage_engine() {
        let mut arb = PayloadCapacityArbitrage::new(50_000_000.0).unwrap();
        
        let obs = TelemetryObservation {
            phase: LaunchPhase::Liftoff,
            velocity_ms: 50.0,
            altitude_km: 1.0,
            acceleration_ms2: 12.0,
            vibration_level: 0.2,
            timestamp: 5.0,
        };
        
        let market = MarketSnapshot {
            launch_future_price: 45_000_000.0,
            satellite_equity_price: 100.0,
            insurance_premium: 5_000_000.0,
            timestamp: 5.0,
        };
        
        let signal = arb.process_telemetry(&obs, &market);
        assert!(signal.is_ok());
    }
}
