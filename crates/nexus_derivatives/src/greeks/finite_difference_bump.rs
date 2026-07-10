//! Finite Difference Method (FDM) for exotic option Greeks
//! 
//! Uses bump-and-reprice to calculate Greeks when analytical formulas don't exist.
//! Zero-allocation design using pre-allocated bump buffers from Stage 1 BumpAllocator.

use crate::pricing::black_scholes_fast::{BSParams, OptionType, bs_price};

/// Default bump sizes for finite difference
const DEFAULT_SPOT_BUMP_PCT: f64 = 0.01; // 1%
const DEFAULT_VOL_BUMP_PCT: f64 = 0.01;  // 1%
const DEFAULT_TIME_BUMP_DAYS: f64 = 1.0;

/// Configuration for finite difference calculations
#[derive(Debug, Clone, Copy)]
pub struct BumpConfig {
    /// Spot bump size as percentage (e.g., 0.01 for 1%)
    pub spot_bump_pct: f64,
    /// Volatility bump size as percentage
    pub vol_bump_pct: f64,
    /// Time bump in days
    pub time_bump_days: f64,
    /// Rate bump in basis points
    pub rate_bump_bp: f64,
    /// Use central difference (more accurate) vs forward difference
    pub use_central_difference: bool,
}

impl Default for BumpConfig {
    fn default() -> Self {
        Self {
            spot_bump_pct: DEFAULT_SPOT_BUMP_PCT,
            vol_bump_pct: DEFAULT_VOL_BUMP_PCT,
            time_bump_days: DEFAULT_TIME_BUMP_DAYS,
            rate_bump_bp: 1.0,
            use_central_difference: true,
        }
    }
}

impl BumpConfig {
    #[inline]
    pub const fn new(
        spot_bump_pct: f64,
        vol_bump_pct: f64,
        time_bump_days: f64,
        rate_bump_bp: f64,
        use_central: bool,
    ) -> Self {
        Self {
            spot_bump_pct,
            vol_bump_pct,
            time_bump_days,
            rate_bump_bp,
            use_central_difference: use_central,
        }
    }
}

/// Pre-allocated bump state for zero-allocation FDM
#[derive(Debug, Clone)]
pub struct BumpState {
    /// Base parameters (original)
    base_params: BSParams,
    /// Up-bumped spot parameters
    spot_up_params: BSParams,
    /// Down-bumped spot parameters
    spot_down_params: BSParams,
    /// Up-bumped vol parameters
    vol_up_params: BSParams,
    /// Down-bumped vol parameters
    vol_down_params: BSParams,
    /// Down-bumped time parameters
    time_down_params: BSParams,
    /// Up-bumped rate parameters
    rate_up_params: BSParams,
    /// Down-bumped rate parameters
    rate_down_params: BSParams,
}

impl BumpState {
    /// Create new bump state with pre-allocated parameter copies
    #[inline]
    pub fn new(base: &BSParams, config: &BumpConfig) -> Self {
        let spot_bump = base.spot * config.spot_bump_pct;
        let vol_bump = base.volatility * config.vol_bump_pct;
        let time_bump = config.time_bump_days / 365.0;
        let rate_bump = config.rate_bump_bp / 10000.0;
        
        Self {
            base_params: *base,
            spot_up_params: BSParams { spot: base.spot + spot_bump, ..*base },
            spot_down_params: BSParams { spot: base.spot - spot_bump, ..*base },
            vol_up_params: BSParams { volatility: base.volatility + vol_bump, ..*base },
            vol_down_params: BSParams { volatility: base.volatility - vol_bump, ..*base },
            time_down_params: BSParams { time_to_expiry: (base.time_to_expiry - time_bump).max(0.0), ..*base },
            rate_up_params: BSParams { risk_free_rate: base.risk_free_rate + rate_bump, ..*base },
            rate_down_params: BSParams { risk_free_rate: base.risk_free_rate - rate_bump, ..*base },
        }
    }
    
    /// Update all bumped states from new base parameters
    #[inline]
    pub fn update(&mut self, base: &BSParams, config: &BumpConfig) {
        let spot_bump = base.spot * config.spot_bump_pct;
        let vol_bump = base.volatility * config.vol_bump_pct;
        let time_bump = config.time_bump_days / 365.0;
        let rate_bump = config.rate_bump_bp / 10000.0;
        
        self.base_params = *base;
        self.spot_up_params.spot = base.spot + spot_bump;
        self.spot_down_params.spot = base.spot - spot_bump;
        self.vol_up_params.volatility = base.volatility + vol_bump;
        self.vol_down_params.volatility = base.volatility - vol_bump;
        self.time_down_params.time_to_expiry = (base.time_to_expiry - time_bump).max(0.0);
        self.rate_up_params.risk_free_rate = base.risk_free_rate + rate_bump;
        self.rate_down_params.risk_free_rate = base.risk_free_rate - rate_bump;
    }
}

/// Finite Difference Engine for calculating Greeks numerically
pub struct FiniteDifferenceEngine {
    /// Bump configuration
    config: BumpConfig,
    /// Pre-allocated bump state (zero allocation during calculation)
    bump_state: Option<BumpState>,
    /// Reusable price buffer
    price_buffer: [f64; 8],
}

impl Default for FiniteDifferenceEngine {
    fn default() -> Self {
        Self::new(BumpConfig::default())
    }
}

impl FiniteDifferenceEngine {
    /// Create a new FDM engine with given configuration
    #[inline]
    pub fn new(config: BumpConfig) -> Self {
        Self {
            config,
            bump_state: None,
            price_buffer: [0.0; 8],
        }
    }
    
    /// Calculate Delta using finite difference
    /// 
    /// Central difference: Δ ≈ (V(S+h) - V(S-h)) / (2h)
    /// Forward difference: Δ ≈ (V(S+h) - V(S)) / h
    #[inline]
    pub fn calculate_delta(&mut self, params: &BSParams, option_type: OptionType) -> f64 {
        self.ensure_bump_state(params);
        
        let bump_state = self.bump_state.as_ref().unwrap();
        
        let price_up = bs_price(&bump_state.spot_up_params, option_type).price;
        let price_down = bs_price(&bump_state.spot_down_params, option_type).price;
        
        if self.config.use_central_difference {
            // Central difference (O(h²) accuracy)
            let h = bump_state.spot_up_params.spot - bump_state.spot_down_params.spot;
            (price_up - price_down) / h
        } else {
            // Forward difference (O(h) accuracy)
            let h = bump_state.spot_up_params.spot - params.spot;
            (price_up - bs_price(params, option_type).price) / h
        }
    }
    
    /// Calculate Gamma using finite difference
    /// 
    /// Γ ≈ (Δ_up - Δ_down) / h ≈ (V(S+h) - 2V(S) + V(S-h)) / h²
    #[inline]
    pub fn calculate_gamma(&mut self, params: &BSParams, option_type: OptionType) -> f64 {
        self.ensure_bump_state(params);
        
        let bump_state = self.bump_state.as_ref().unwrap();
        
        let price_up = bs_price(&bump_state.spot_up_params, option_type).price;
        let price_base = bs_price(params, option_type).price;
        let price_down = bs_price(&bump_state.spot_down_params, option_type).price;
        
        let h = bump_state.spot_up_params.spot - params.spot;
        
        // Second-order central difference
        (price_up - 2.0 * price_base + price_down) / (h * h)
    }
    
    /// Calculate Vega using finite difference
    /// 
    /// ν ≈ (V(σ+h) - V(σ-h)) / (2h)
    #[inline]
    pub fn calculate_vega(&mut self, params: &BSParams, option_type: OptionType) -> f64 {
        self.ensure_bump_state(params);
        
        let bump_state = self.bump_state.as_ref().unwrap();
        
        let price_up = bs_price(&bump_state.vol_up_params, option_type).price;
        let price_down = bs_price(&bump_state.vol_down_params, option_type).price;
        
        if self.config.use_central_difference {
            let h = bump_state.vol_up_params.volatility - bump_state.vol_down_params.volatility;
            (price_up - price_down) / h * 0.01 // Per 1% move
        } else {
            let h = bump_state.vol_up_params.volatility - params.volatility;
            (price_up - bs_price(params, option_type).price) / h * 0.01
        }
    }
    
    /// Calculate Theta using finite difference
    /// 
    /// Θ ≈ (V(T-h) - V(T)) / h (per day)
    #[inline]
    pub fn calculate_theta(&mut self, params: &BSParams, option_type: OptionType) -> f64 {
        self.ensure_bump_state(params);
        
        let bump_state = self.bump_state.as_ref().unwrap();
        
        let price_base = bs_price(params, option_type).price;
        let price_down = bs_price(&bump_state.time_down_params, option_type).price;
        
        // Time bump in years
        let h = self.config.time_bump_days;
        
        // Theta per day (negative of price change as time passes)
        (price_down - price_base) / h
    }
    
    /// Calculate Rho using finite difference
    /// 
    /// ρ ≈ (V(r+h) - V(r-h)) / (2h)
    #[inline]
    pub fn calculate_rho(&mut self, params: &BSParams, option_type: OptionType) -> f64 {
        self.ensure_bump_state(params);
        
        let bump_state = self.bump_state.as_ref().unwrap();
        
        let price_up = bs_price(&bump_state.rate_up_params, option_type).price;
        let price_down = bs_price(&bump_state.rate_down_params, option_type).price;
        
        if self.config.use_central_difference {
            let h = bump_state.rate_up_params.risk_free_rate - bump_state.rate_down_params.risk_free_rate;
            (price_up - price_down) / h * 0.01 // Per 1% move
        } else {
            let h = bump_state.rate_up_params.risk_free_rate - params.risk_free_rate;
            (price_up - bs_price(params, option_type).price) / h * 0.01
        }
    }
    
    /// Calculate all Greeks at once (efficient - reuses bumped prices)
    pub fn calculate_all_greeks(
        &mut self,
        params: &BSParams,
        option_type: OptionType,
    ) -> FdmGreeks {
        self.ensure_bump_state(params);
        
        let bump_state = self.bump_state.as_ref().unwrap();
        let h_spot = bump_state.spot_up_params.spot - bump_state.spot_down_params.spot;
        let h_vol = bump_state.vol_up_params.volatility - bump_state.vol_down_params.volatility;
        let h_time = self.config.time_bump_days;
        let h_rate = bump_state.rate_up_params.risk_free_rate - bump_state.rate_down_params.risk_free_rate;
        
        // Compute all bumped prices
        self.price_buffer[0] = bs_price(&bump_state.spot_up_params, option_type).price;
        self.price_buffer[1] = bs_price(&bump_state.spot_down_params, option_type).price;
        self.price_buffer[2] = bs_price(&bump_state.vol_up_params, option_type).price;
        self.price_buffer[3] = bs_price(&bump_state.vol_down_params, option_type).price;
        self.price_buffer[4] = bs_price(&bump_state.time_down_params, option_type).price;
        self.price_buffer[5] = bs_price(&bump_state.rate_up_params, option_type).price;
        self.price_buffer[6] = bs_price(&bump_state.rate_down_params, option_type).price;
        self.price_buffer[7] = bs_price(params, option_type).price;
        
        let p_base = self.price_buffer[7];
        
        // Delta (central)
        let delta = (self.price_buffer[0] - self.price_buffer[1]) / h_spot;
        
        // Gamma (central second derivative)
        let gamma = (self.price_buffer[0] - 2.0 * p_base + self.price_buffer[1]) 
            / ((h_spot / 2.0) * (h_spot / 2.0));
        
        // Vega (central, per 1%)
        let vega = (self.price_buffer[2] - self.price_buffer[3]) / h_vol * 0.01;
        
        // Theta (per day)
        let theta = (self.price_buffer[4] - p_base) / h_time;
        
        // Rho (central, per 1%)
        let rho = (self.price_buffer[5] - self.price_buffer[6]) / h_rate * 0.01;
        
        FdmGreeks {
            delta,
            gamma,
            vega,
            theta,
            rho,
        }
    }
    
    /// Ensure bump state is initialized
    #[inline]
    fn ensure_bump_state(&mut self, params: &BSParams) {
        match &mut self.bump_state {
            Some(state) => state.update(params, &self.config),
            None => self.bump_state = Some(BumpState::new(params, &self.config)),
        }
    }
    
    /// Get current bump configuration
    #[inline]
    pub fn config(&self) -> &BumpConfig {
        &self.config
    }
    
    /// Update bump configuration
    #[inline]
    pub fn set_config(&mut self, config: BumpConfig) {
        self.config = config;
        self.bump_state = None; // Force re-initialization
    }
}

/// Greeks result from FDM calculation
#[derive(Debug, Clone, Copy)]
pub struct FdmGreeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
}

impl FdmGreeks {
    #[inline]
    pub const fn new(delta: f64, gamma: f64, vega: f64, theta: f64, rho: f64) -> Self {
        Self { delta, gamma, vega, theta, rho }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::greeks::analytical_greeks::calculate_greeks;
    
    #[test]
    fn test_fdm_delta_vs_analytical() {
        let params = BSParams {
            spot: 100.0,
            strike: 100.0,
            time_to_expiry: 0.25,
            risk_free_rate: 0.05,
            volatility: 0.2,
            dividend_yield: 0.0,
        };
        
        let mut fdm = FiniteDifferenceEngine::default();
        let fdm_delta = fdm.calculate_delta(&params, OptionType::Call);
        
        let analytical = calculate_greeks(&params, OptionType::Call);
        let analytical_delta = analytical.first.delta;
        
        // FDM should be within 1% of analytical
        let rel_error = (fdm_delta - analytical_delta).abs() / analytical_delta.abs();
        assert!(rel_error < 0.01, "FDM delta error too large: {}%", rel_error * 100.0);
    }
    
    #[test]
    fn test_fdm_gamma_vs_analytical() {
        let params = BSParams::default();
        
        let mut fdm = FiniteDifferenceEngine::default();
        let fdm_gamma = fdm.calculate_gamma(&params, OptionType::Call);
        
        let analytical = calculate_greeks(&params, OptionType::Call);
        let analytical_gamma = analytical.second.gamma;
        
        // Gamma comparison (more sensitive to bump size)
        let rel_error = (fdm_gamma - analytical_gamma).abs() / analytical_gamma.abs();
        assert!(rel_error < 0.05, "FDM gamma error too large: {}%", rel_error * 100.0);
    }
    
    #[test]
    fn test_fdm_all_greeks() {
        let params = BSParams::default();
        
        let mut fdm = FiniteDifferenceEngine::default();
        let greeks = fdm.calculate_all_greeks(&params, OptionType::Call);
        
        assert!(greeks.delta > 0.0, "Call delta should be positive");
        assert!(greeks.gamma > 0.0, "Gamma should be positive");
        assert!(greeks.vega > 0.0, "Vega should be positive");
        assert!(greeks.theta < 0.0, "Long option theta should be negative");
    }
    
    #[test]
    fn test_central_vs_forward_difference() {
        let params = BSParams::default();
        
        let mut fdm_central = FiniteDifferenceEngine::new(BumpConfig {
            use_central_difference: true,
            ..Default::default()
        });
        
        let mut fdm_forward = FiniteDifferenceEngine::new(BumpConfig {
            use_central_difference: false,
            ..Default::default()
        });
        
        let delta_central = fdm_central.calculate_delta(&params, OptionType::Call);
        let delta_forward = fdm_forward.calculate_delta(&params, OptionType::Call);
        
        // Central difference should be more accurate (closer to analytical)
        let analytical = calculate_greeks(&params, OptionType::Call).first.delta;
        
        let error_central = (delta_central - analytical).abs();
        let error_forward = (delta_forward - analytical).abs();
        
        assert!(error_central < error_forward, 
            "Central difference should be more accurate");
    }
    
    #[test]
    fn test_zero_allocation_update() {
        let mut fdm = FiniteDifferenceEngine::default();
        let params1 = BSParams::default();
        let params2 = BSParams { spot: 105.0, ..Default::default() };
        
        // First call initializes bump state
        let _ = fdm.calculate_delta(&params1, OptionType::Call);
        
        // Second call should reuse bump state (no allocation)
        let _ = fdm.calculate_delta(&params2, OptionType::Call);
        
        // Verify bump state was updated
        let bump_state = fdm.bump_state.as_ref().unwrap();
        assert!((bump_state.base_params.spot - params2.spot).abs() < 1e-10);
    }
}
