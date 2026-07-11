//! Cognitive Fatigue Swap - derivative paying out based on aggregate neural exhaustion.
//! Allows advertisers/platforms to hedge against user burnout.

/// Maximum counterparties supported
pub const MAX_COUNTERPARTIES: usize = 16;

/// Swap leg type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SwapLegType {
    Fixed,
    FloatingNeuralExhaustion,
}

/// Cognitive fatigue swap result
#[derive(Debug, Clone)]
pub struct FatigueSwapResult {
    pub present_value: f32,
    pub fixed_leg_pv: f32,
    pub floating_leg_pv: f32,
    pub fair_spread: f32,
    pub expected_payout: f32,
    pub variance: f32,
}

impl FatigueSwapResult {
    pub const fn new() -> Self {
        Self {
            present_value: 0.0,
            fixed_leg_pv: 0.0,
            floating_leg_pv: 0.0,
            fair_spread: 0.0,
            expected_payout: 0.0,
            variance: 0.0,
        }
    }
}

impl Default for FatigueSwapResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Main cognitive fatigue swap pricer
pub struct CognitiveFatigueSwap {
    /// Notional amount
    notional: f32,
    /// Fixed rate
    fixed_rate: f32,
    /// Maturity in months
    maturity_months: u32,
    /// Payment frequency (months)
    payment_frequency: u32,
    /// Result
    result: FatigueSwapResult,
    /// Neural exhaustion curve
    exhaustion_rates: [f32; 12],
}

impl CognitiveFatigueSwap {
    pub fn new() -> Self {
        Self {
            notional: 1_000_000.0,
            fixed_rate: 0.05,
            maturity_months: 12,
            payment_frequency: 3,
            result: FatigueSwapResult::new(),
            exhaustion_rates: [0.0; 12],
        }
    }

    /// Configure swap parameters
    pub fn configure(&mut self, notional: f32, fixed_rate: f32, maturity_months: u32) {
        self.notional = notional.max(0.0);
        self.fixed_rate = fixed_rate.max(0.0);
        self.maturity_months = maturity_months.max(1);
    }

    /// Set neural exhaustion forward curve
    pub fn set_exhaustion_curve(&mut self, rates: &[f32; 12]) {
        self.exhaustion_rates = *rates;
    }

    /// Price the swap
    pub fn price(&mut self) -> &FatigueSwapResult {
        let num_payments = self.maturity_months / self.payment_frequency;
        let dt = self.payment_frequency as f32 / 12.0;
        
        // Fixed leg PV
        let mut fixed_pv = 0.0f32;
        for i in 1..=num_payments {
            let t = i as f32 * dt;
            let discount = (-0.03 * t).exp(); // Assume 3% risk-free
            fixed_pv += self.fixed_rate * dt * discount;
        }
        fixed_pv *= self.notional;

        // Floating leg PV (based on neural exhaustion)
        let mut floating_pv = 0.0f32;
        let mut expected_exhaustion = 0.0f32;
        let mut exhaustion_sq = 0.0f32;

        for i in 1..=num_payments {
            let idx = ((i * self.payment_frequency) as usize).min(11);
            let exhaustion_rate = self.exhaustion_rates[idx];
            let t = i as f32 * dt;
            let discount = (-0.03 * t).exp();
            
            floating_pv += exhaustion_rate * dt * discount;
            expected_exhaustion += exhaustion_rate;
            exhaustion_sq += exhaustion_rate * exhaustion_rate;
        }
        floating_pv *= self.notional;

        self.result.fixed_leg_pv = fixed_pv;
        self.result.floating_leg_pv = floating_pv;
        self.result.present_value = floating_pv - fixed_pv;
        
        // Fair spread makes swap value zero
        if fixed_pv > 1e-6 {
            self.result.fair_spread = floating_pv / (fixed_pv / self.fixed_rate);
        }

        // Expected payout and variance
        self.result.expected_payout = (floating_pv - fixed_pv) / num_payments as f32;
        
        let mean_exhaustion = expected_exhaustion / num_payments as f32;
        let mean_sq = exhaustion_sq / num_payments as f32;
        self.result.variance = (mean_sq - mean_exhaustion * mean_exhaustion).max(0.0);

        &self.result
    }

    /// Get mark-to-market value
    #[inline]
    pub const fn mtm(&self) -> f32 {
        self.result.present_value
    }

    /// Get fair spread
    #[inline]
    pub const fn fair_spread(&self) -> f32 {
        self.result.fair_spread
    }
}

impl Default for CognitiveFatigueSwap {
    fn default() -> Self {
        Self::new()
    }
}
