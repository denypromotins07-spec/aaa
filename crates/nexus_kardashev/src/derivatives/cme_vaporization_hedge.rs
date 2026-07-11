//! CME Vaporization Hedge Derivative Pricer
//! 
//! Implements financial derivatives for hedging against Coronal Mass Ejection
//! damage to orbital infrastructure. Uses MHD-based probability models
//! to price insurance contracts and energy storage futures.

use crate::stellar::mhd_plasma_solver::{CMEEvent, CMEHedgeParams};
use num_traits::{Float, Zero};

/// Types of CME-related derivatives
#[derive(Debug, Clone)]
pub enum CMEDerivative {
    /// Insurance contract paying out on hardware vaporization
    VaporizationInsurance {
        contract_id: u64,
        notional_usd: f64,
        coverage_fraction: f64,
        expiration_days: u32,
    },
    /// Energy storage futures (price spikes when generation damaged)
    EnergyStorageFuture {
        contract_id: u64,
        mwh_quantity: f64,
        delivery_date_days: u32,
        strike_price_usd: f64,
    },
    /// Orbital slot lease with CME protection clause
    ProtectedOrbitalLease {
        slot_id: u64,
        daily_rate_usd: f64,
        protection_level: ProtectionLevel,
        lease_duration_days: u32,
    },
}

/// Levels of CME protection for orbital slots
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProtectionLevel {
    /// Basic warning system only
    Basic,
    /// Active magnetic deflection available
    ActiveDeflection,
    /// Full Faraday cage + redundancy
    FullShielding,
}

impl ProtectionLevel {
    /// Effectiveness factor for reducing expected loss
    pub fn effectiveness(&self) -> f64 {
        match self {
            ProtectionLevel::Basic => 0.1,
            ProtectionLevel::ActiveDeflection => 0.5,
            ProtectionLevel::FullShielding => 0.9,
        }
    }
    
    /// Daily cost multiplier for protection
    pub fn cost_multiplier(&self) -> f64 {
        match self {
            ProtectionLevel::Basic => 1.0,
            ProtectionLevel::ActiveDeflection => 2.5,
            ProtectionLevel::FullShielding => 5.0,
        }
    }
}

/// CME risk assessment output
#[derive(Debug, Clone)]
pub struct CMERiskAssessment<T> {
    pub event_probability: T,
    pub expected_energy_release_joules: T,
    pub time_to_impact_hours: T,
    pub affected_orbital_slots: Vec<u64>,
    pub recommended_hedge_ratio: f64,
}

/// CME Vaporization Hedge Pricer
pub struct CMEVaporizationPricer<T> {
    risk_free_rate: T,
    base_volatility: T,
    stellar_activity_index: T,
}

impl<T: Float + Zero> CMEVaporizationPricer<T> {
    pub fn new(risk_free_rate: T, base_volatility: T) -> Self {
        Self {
            risk_free_rate,
            base_volatility,
            stellar_activity_index: T::from(1.0).unwrap_or_else(|| T::one()),
        }
    }
    
    /// Update stellar activity index from MHD observations
    pub fn update_stellar_activity(&mut self, cme_events: &[CMEEvent<T>]) {
        if cme_events.is_empty() {
            self.stellar_activity_index = T::from(0.5).unwrap_or_else(|| T::one() / T::from(2).unwrap());
            return;
        }
        
        // Weighted average of event energies
        let total_energy: T = cme_events.iter().map(|e| e.estimated_energy).fold(
            T::zero(),
            |acc, e| acc + e,
        );
        
        let avg_energy = total_energy / T::from(cme_events.len() as f64).unwrap();
        
        // Normalize to activity index (1.0 = solar maximum, 0.1 = minimum)
        let reference_energy = T::from(1e25).unwrap_or_else(|| T::one());
        self.stellar_activity_index = (T::one() + (avg_energy / reference_energy).ln())
            .max(T::from(0.1).unwrap_or_else(|| T::one() / T::from(10).unwrap()))
            .min(T::from(10.0).unwrap());
    }
    
    /// Calculate fair premium for vaporization insurance
    pub fn price_vaporization_insurance(
        &self,
        params: &CMEHedgeParams,
    ) -> InsuranceQuote {
        let base_premium = params.calculate_insurance_premium();
        
        // Adjust for stellar activity
        let activity_factor = self.stellar_activity_index.to_f64().unwrap_or(1.0);
        
        // Adjust for protection level (assume basic for now)
        let protection_factor = ProtectionLevel::Basic.effectiveness();
        
        // Net expected loss after protection
        let adjusted_loss = params.expected_loss_fraction * (1.0 - protection_factor);
        let adjusted_premium = params.hardware_value_usd * adjusted_loss 
                             * params.cme_probability * activity_factor;
        
        // Risk loading based on time urgency
        let urgency_loading = if params.time_to_impact_hours < 24.0 {
            2.0
        } else if params.time_to_impact_hours < 72.0 {
            1.5
        } else {
            1.0
        };
        
        // Profit margin
        let profit_margin = 1.2;
        
        InsuranceQuote {
            premium_usd: adjusted_premium * urgency_loading * profit_margin,
            base_premium,
            activity_adjustment: activity_factor,
            coverage_amount: params.hardware_value_usd * params.expected_loss_fraction,
            confidence_interval: (0.85, 0.95),
        }
    }
    
    /// Price energy storage future option
    pub fn price_energy_future(
        &self,
        mwh_quantity: f64,
        strike_price: f64,
        days_to_delivery: u32,
        current_spot_price: f64,
        cme_probability: f64,
    ) -> OptionQuote {
        // Black-Scholes style pricing with CME jump component
        
        let dt = days_to_delivery as f64 / 365.0;
        let sqrt_t = dt.sqrt();
        
        // Adjusted volatility includes CME jump risk
        let cme_jump_vol = cme_probability * 2.0; // Jump adds significant vol
        let total_vol = self.base_volatility.to_f64().unwrap_or(0.3) + cme_jump_vol;
        
        // Forward price with CME premium
        let rf = self.risk_free_rate.to_f64().unwrap_or(0.05);
        let forward_price = current_spot_price * (1.0 + rf * dt) * (1.0 + cme_probability * 0.5);
        
        // Simplified Black-Scholes call price
        let d1 = if total_vol * sqrt_t > 0.0 {
            (forward_price / strike_price).ln() / (total_vol * sqrt_t) 
                + 0.5 * total_vol * sqrt_t
        } else {
            0.0
        };
        let d2 = d1 - total_vol * sqrt_t;
        
        // Cumulative normal distribution approximation
        let n_d1 = cumulative_normal(d1);
        let n_d2 = cumulative_normal(d2);
        
        let call_price = forward_price * n_d1 - strike_price * (-rf * dt).exp() * n_d2;
        
        OptionQuote {
            option_premium: call_price * mwh_quantity,
            underlying_forward: forward_price,
            implied_volatility: total_vol,
            delta: n_d1,
            gamma: 0.0, // Would need second derivative
            vega: mwh_quantity * sqrt_t * normal_pdf(d1),
        }
    }
    
    /// Calculate optimal hedge ratio for a portfolio of orbital assets
    pub fn calculate_portfolio_hedge_ratio(
        &self,
        assets: &[CMEHedgeParams],
        correlation_matrix: &[f64],
    ) -> HedgeRatio {
        let n_assets = assets.len();
        if n_assets == 0 {
            return HedgeRatio {
                energy_future_weight: 0.0,
                insurance_weight: 0.0,
                unhedged_risk: 1.0,
            };
        }
        
        // Total portfolio value at risk
        let total_var: f64 = assets.iter()
            .map(|a| a.hardware_value_usd * a.expected_loss_fraction * a.cme_probability)
            .sum();
        
        // Optimal hedge minimizes variance of hedged portfolio
        // Simple heuristic: hedge proportionally to individual risks
        let mut total_hedge_notional = 0.0;
        let mut total_insurance_premium = 0.0;
        
        for asset in assets {
            let hedge_ratio = asset.calculate_energy_hedge_ratio();
            total_hedge_notional += asset.hardware_value_usd * hedge_ratio;
            
            let quote = self.price_vaporization_insurance(asset);
            total_insurance_premium += quote.premium_usd;
        }
        
        // Diversification benefit from correlation
        let avg_correlation = if n_assets > 1 {
            let sum_corr: f64 = correlation_matrix.iter().sum();
            sum_corr / (n_assets * (n_assets - 1)) as f64
        } else {
            1.0
        };
        
        let diversification_factor = 1.0 - avg_correlation * 0.5;
        
        HedgeRatio {
            energy_future_weight: total_hedge_notional / total_var.max(1.0),
            insurance_weight: total_insurance_premium / total_var.max(1.0),
            unhedged_risk: diversification_factor,
        }
    }
    
    /// Generate CME risk assessment from detected events
    pub fn assess_risk_from_events(
        &self,
        events: &[CMEEvent<T>],
        orbital_positions: &[(u64, T, T, T)],  // (slot_id, x, y, z)
    ) -> CMERiskAssessment<T> {
        if events.is_empty() {
            return CMERiskAssessment {
                event_probability: T::zero(),
                expected_energy_release_joules: T::zero(),
                time_to_impact_hours: T::zero(),
                affected_orbital_slots: Vec::new(),
                recommended_hedge_ratio: 0.0,
            };
        }
        
        // Aggregate event probabilities
        let total_probability: T = events.iter()
            .map(|e| e.probability)
            .fold(T::zero(), |acc, p| acc + p);
        
        let avg_probability = total_probability / T::from(events.len() as f64).unwrap();
        
        // Total expected energy
        let total_energy: T = events.iter()
            .map(|e| e.estimated_energy)
            .fold(T::zero(), |acc, e| acc + e);
        
        // Find affected orbital slots (simplified - would need actual trajectory calc)
        let affected_slots: Vec<u64> = orbital_positions
            .iter()
            .filter_map(|(slot_id, _, _, _)| {
                // Assume all inner slots affected for now
                Some(*slot_id)
            })
            .collect();
        
        // Recommended hedge ratio proportional to risk
        let hedge_ratio = avg_probability.to_f64().unwrap_or(0.0) * 0.8;
        
        CMERiskAssessment {
            event_probability: avg_probability,
            expected_energy_release_joules: total_energy,
            time_to_impact_hours: T::from(48.0).unwrap(),  // Would calculate from trajectory
            affected_orbital_slots,
            recommended_hedge_ratio: hedge_ratio.min(1.0),
        }
    }
}

/// Insurance quote with breakdown
#[derive(Debug, Clone)]
pub struct InsuranceQuote {
    pub premium_usd: f64,
    pub base_premium: f64,
    pub activity_adjustment: f64,
    pub coverage_amount: f64,
    pub confidence_interval: (f64, f64),
}

/// Option quote with Greeks
#[derive(Debug, Clone)]
pub struct OptionQuote {
    pub option_premium: f64,
    pub underlying_forward: f64,
    pub implied_volatility: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
}

/// Portfolio hedge ratios
#[derive(Debug, Clone)]
pub struct HedgeRatio {
    pub energy_future_weight: f64,
    pub insurance_weight: f64,
    pub unhedged_risk: f64,
}

/// Cumulative standard normal distribution
fn cumulative_normal(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    
    let t = 1.0 / (1.0 + 0.2316419 * x_abs);
    let d = 0.3989423 * (-x_abs * x_abs / 2.0).exp();
    
    let poly = t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    
    0.5 * (1.0 + sign * (1.0 - d * poly))
}

/// Standard normal PDF
fn normal_pdf(x: f64) -> f64 {
    (2.0 * std::f64::consts::PI).sqrt().recip() * (-x * x / 2.0).exp()
}

/// Autonomous CME hedging agent
pub struct CMEHedgingAgent<T> {
    pricer: CMEVaporizationPricer<T>,
    portfolio_value: f64,
    max_hedge_fraction: f64,
}

impl<T: Float + Zero + Copy> CMEHedgingAgent<T> {
    pub fn new(pricer: CMEVaporizationPricer<T>, portfolio_value: f64) -> Self {
        Self {
            pricer,
            portfolio_value,
            max_hedge_fraction: 0.5,  // Max 50% of portfolio on hedges
        }
    }
    
    /// Execute optimal hedge based on current CME risk
    pub fn execute_hedge(
        &mut self,
        events: &[CMEEvent<T>],
        assets: &[CMEHedgeParams],
    ) -> HedgeExecution {
        let risk_assessment = self.pricer.assess_risk_from_events(
            events,
            &assets.iter().map(|a| (a.orbital_slot_id, T::zero(), T::zero(), T::zero())).collect::<Vec<_>>()
        );
        
        let hedge_ratio = self.pricer.calculate_portfolio_hedge_ratio(assets, &[]);
        
        // Calculate hedge amounts
        let max_hedge_budget = self.portfolio_value * self.max_hedge_fraction;
        
        let insurance_budget = max_hedge_budget * hedge_ratio.insurance_weight 
                             / (hedge_ratio.energy_future_weight + hedge_ratio.insurance_weight + 1e-10);
        let energy_budget = max_hedge_budget - insurance_budget;
        
        // Generate execution orders
        let mut insurance_orders = Vec::new();
        let mut energy_orders = Vec::new();
        
        for asset in assets {
            let quote = self.pricer.price_vaporization_insurance(asset);
            if quote.premium_usd <= insurance_budget / assets.len() as f64 {
                insurance_orders.push(InsuranceOrder {
                    asset_id: asset.orbital_slot_id,
                    premium: quote.premium_usd,
                    coverage: quote.coverage_amount,
                });
            }
            
            let future_quote = self.pricer.price_energy_future(
                100.0,  // MWh
                50.0,   // Strike
                30,     // Days
                45.0,   // Spot
                asset.cme_probability,
            );
            
            if future_quote.option_premium <= energy_budget / assets.len() as f64 {
                energy_orders.push(EnergyOrder {
                    asset_id: asset.orbital_slot_id,
                    mwh: 100.0,
                    premium: future_quote.option_premium,
                    delta: future_quote.delta,
                });
            }
        }
        
        HedgeExecution {
            risk_assessment,
            insurance_orders,
            energy_orders,
            total_cost: insurance_budget + energy_budget,
            expected_protection: 1.0 - hedge_ratio.unhedged_risk,
        }
    }
}

/// Executed insurance order
#[derive(Debug, Clone)]
pub struct InsuranceOrder {
    pub asset_id: u64,
    pub premium: f64,
    pub coverage: f64,
}

/// Executed energy order
#[derive(Debug, Clone)]
pub struct EnergyOrder {
    pub asset_id: u64,
    pub mwh: f64,
    pub premium: f64,
    pub delta: f64,
}

/// Complete hedge execution report
#[derive(Debug, Clone)]
pub struct HedgeExecution {
    pub risk_assessment: CMERiskAssessment<T>,
    pub insurance_orders: Vec<InsuranceOrder>,
    pub energy_orders: Vec<EnergyOrder>,
    pub total_cost: f64,
    pub expected_protection: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pricer_initialization() {
        type F = f64;
        let pricer = CMEVaporizationPricer::new(F::from(0.05).unwrap(), F::from(0.3).unwrap());
        
        assert!(pricer.risk_free_rate > F::zero());
        assert!(pricer.base_volatility > F::zero());
    }
    
    #[test]
    fn test_insurance_quote() {
        type F = f64;
        let pricer = CMEVaporizationPricer::new(F::from(0.05).unwrap(), F::from(0.3).unwrap());
        
        let params = CMEHedgeParams {
            orbital_slot_id: 1,
            hardware_value_usd: 1e9,
            deflector_capability: 0.5,
            time_to_impact_hours: 48.0,
            cme_probability: 0.3,
            expected_loss_fraction: 0.8,
        };
        
        let quote = pricer.price_vaporization_insurance(&params);
        
        assert!(quote.premium_usd > 0.0);
        assert!(quote.coverage_amount > 0.0);
    }
    
    #[test]
    fn test_cumulative_normal() {
        // N(0) should be 0.5
        let n_zero = cumulative_normal(0.0);
        assert!((n_zero - 0.5).abs() < 0.001);
        
        // N(∞) should approach 1
        let n_large = cumulative_normal(10.0);
        assert!(n_large > 0.99);
        
        // N(-∞) should approach 0
        let n_neg = cumulative_normal(-10.0);
        assert!(n_neg < 0.01);
    }
}
