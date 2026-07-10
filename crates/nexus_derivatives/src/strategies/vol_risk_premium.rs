//! Volatility Risk Premium (VRP) harvesting strategy
//! 
//! Systematically sells overpriced implied volatility and buys realized volatility.
//! Implements gamma scalping to monetize the vol premium.

use crate::pricing::black_scholes_fast::{BSParams, OptionType, bs_price};
use crate::greeks::analytical_greeks::{calculate_greeks, FullGreeks};

/// Minimum VRP threshold for trade entry (in vol points)
const MIN_VRP_THRESHOLD: f64 = 0.02; // 2%

/// Maximum position size as fraction of portfolio
const MAX_POSITION_SIZE: f64 = 0.1;

/// Signal from VRP analysis
#[derive(Debug, Clone, Copy)]
pub struct VrpSignal {
    /// Implied volatility
    pub implied_vol: f64,
    /// Realized volatility estimate
    pub realized_vol: f64,
    /// VRP = IV - RV
    pub vrp: f64,
    /// Signal strength (-1 to 1)
    pub signal_strength: f64,
    /// Recommended action: true = sell vol, false = buy vol
    pub sell_volatility: bool,
    /// Expected P&L from gamma scalping
    pub expected_gamma_pnl: f64,
}

impl VrpSignal {
    #[inline]
    pub const fn new(
        implied_vol: f64,
        realized_vol: f64,
        vrp: f64,
        signal_strength: f64,
        sell_volatility: bool,
        expected_gamma_pnl: f64,
    ) -> Self {
        Self {
            implied_vol,
            realized_vol,
            vrp,
            signal_strength,
            sell_volatility,
            expected_gamma_pnl,
        }
    }
    
    /// Check if signal meets minimum threshold
    #[inline]
    pub fn is_tradeable(&self) -> bool {
        self.vrp.abs() >= MIN_VRP_THRESHOLD && self.signal_strength.abs() >= 0.3
    }
}

/// Volatility Risk Premium analyzer and trader
pub struct VolatilityRiskPremium {
    /// Lookback period for realized vol calculation (days)
    lookback_days: usize,
    /// VRP threshold for entry
    entry_threshold: f64,
    /// Exit threshold (when VRP mean-reverts)
    exit_threshold: f64,
    /// Current realized vol estimate
    realized_vol: f64,
    /// Realized vol calculation buffer (pre-allocated)
    returns_buffer: Vec<f64>,
}

impl Default for VolatilityRiskPremium {
    fn default() -> Self {
        Self::new(20, MIN_VRP_THRESHOLD, 0.005)
    }
}

impl VolatilityRiskPremium {
    /// Create a new VRP analyzer
    #[inline]
    pub fn new(lookback_days: usize, entry_threshold: f64, exit_threshold: f64) -> Self {
        Self {
            lookback_days,
            entry_threshold,
            exit_threshold,
            realized_vol: 0.0,
            returns_buffer: Vec::with_capacity(lookback_days),
        }
    }
    
    /// Update realized volatility from price series
    /// 
    /// # Arguments
    /// * `prices` - Slice of historical prices (most recent last)
    /// * `daily_scale` - Annualization factor (e.g., sqrt(252) for daily)
    pub fn update_realized_vol(&mut self, prices: &[f64], daily_scale: f64) {
        if prices.len() < 2 {
            return;
        }
        
        self.returns_buffer.clear();
        
        // Calculate log returns
        for i in 1..prices.len().min(self.lookback_days + 1) {
            let ret = (prices[i] / prices[i - 1]).ln();
            self.returns_buffer.push(ret);
        }
        
        if self.returns_buffer.is_empty() {
            return;
        }
        
        // Calculate standard deviation
        let mean = self.returns_buffer.iter().sum::<f64>() / self.returns_buffer.len() as f64;
        let variance = self.returns_buffer.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / self.returns_buffer.len() as f64;
        
        self.realized_vol = variance.sqrt() * daily_scale;
    }
    
    /// Set realized vol directly (from external source)
    #[inline]
    pub fn set_realized_vol(&mut self, rv: f64) {
        self.realized_vol = rv;
    }
    
    /// Analyze VRP and generate trading signal
    /// 
    /// # Arguments
    /// * `implied_vol` - Current implied volatility from market
    /// * `params` - Option parameters for gamma calculation
    /// * `option_type` - Call or Put
    /// 
    /// # Returns
    /// VrpSignal with recommendation
    pub fn analyze(&self, implied_vol: f64, params: &BSParams, option_type: OptionType) -> VrpSignal {
        let vrp = implied_vol - self.realized_vol;
        
        // Normalize signal strength
        let signal_strength = (vrp / implied_vol).clamp(-1.0, 1.0);
        
        // Determine direction: positive VRP = sell vol, negative = buy vol
        let sell_volatility = vrp > 0.0;
        
        // Estimate gamma scalping P&L
        let expected_gamma_pnl = self.estimate_gamma_scalp_pnl(params, option_type, implied_vol);
        
        VrpSignal::new(
            implied_vol,
            self.realized_vol,
            vrp,
            signal_strength,
            sell_volatility,
            expected_gamma_pnl,
        )
    }
    
    /// Estimate expected P&L from gamma scalping
    /// 
    /// Gamma scalping profit ≈ 0.5 * Γ * S² * (σ_realized² - σ_implied²) * T
    fn estimate_gamma_scalp_pnl(&self, params: &BSParams, option_type: OptionType, iv: f64) -> f64 {
        let greeks = calculate_greeks(params, option_type);
        let gamma = greeks.second.gamma;
        
        // Expected variance difference
        let var_diff = self.realized_vol.powi(2) - iv.powi(2);
        
        // Gamma scalping P&L approximation
        // Positive when realized > implied (long gamma profits)
        0.5 * gamma * params.spot.powi(2) * var_diff * params.time_to_expiry
    }
    
    /// Generate optimal strike for VRP trade
    /// 
    /// Typically sells OTM options where VRP is highest
    pub fn find_optimal_strike(&self, spot: f64, target_delta: f64) -> f64 {
        // Simplified: use delta approximation
        // For puts: strike ≈ spot * e^(N^-1(delta) * σ * √T)
        // For calls: strike ≈ spot * e^(-N^-1(1-delta) * σ * √T)
        
        // Rough approximation: 25 delta put ≈ 95% of spot
        // 25 delta call ≈ 105% of spot
        if target_delta < 0.5 {
            spot * (1.0 - (0.5 - target_delta) * 0.2)
        } else {
            spot * (1.0 + (target_delta - 0.5) * 0.2)
        }
    }
    
    /// Check if existing position should be exited
    /// 
    /// Exit when VRP mean-reverts below threshold
    pub fn should_exit(&self, current_vrp: f64, position_vrp: f64) -> bool {
        // Exit when VRP has mean-reverted significantly
        let pnl_fraction = 1.0 - (current_vrp / position_vrp);
        pnl_fraction > 0.5 || current_vrp.abs() < self.exit_threshold
    }
    
    /// Get current realized vol estimate
    #[inline]
    pub fn realized_vol(&self) -> f64 {
        self.realized_vol
    }
    
    /// Get lookback period
    #[inline]
    pub fn lookback_days(&self) -> usize {
        self.lookback_days
    }
}

/// VRP trade execution parameters
#[derive(Debug, Clone)]
pub struct VrpTrade {
    /// Option type
    pub option_type: OptionType,
    /// Strike price
    pub strike: f64,
    /// Expiry (time to expiry in years)
    pub time_to_expiry: f64,
    /// Number of contracts (negative = short)
    pub quantity: i64,
    /// Entry implied volatility
    pub entry_iv: f64,
    /// Entry VRP level
    pub entry_vrp: f64,
    /// Stop-loss VRP level
    pub stop_loss_vrp: f64,
    /// Take-profit VRP level
    pub take_profit_vrp: f64,
}

impl VrpTrade {
    #[inline]
    pub fn new(
        option_type: OptionType,
        strike: f64,
        time_to_expiry: f64,
        quantity: i64,
        entry_iv: f64,
        entry_vrp: f64,
    ) -> Self {
        Self {
            option_type,
            strike,
            time_to_expiry,
            quantity,
            entry_iv,
            entry_vrp,
            stop_loss_vrp: entry_vrp * 0.5, // Stop at 50% mean reversion
            take_profit_vrp: 0.0, // Exit at zero VRP
        }
    }
    
    /// Check if trade should be stopped out
    #[inline]
    pub fn check_stop_loss(&self, current_vrp: f64) -> bool {
        if self.entry_vrp > 0.0 {
            // Short vol trade: stop if VRP increases
            current_vrp > self.entry_vrp * 1.5
        } else {
            // Long vol trade: stop if VRP decreases further
            current_vrp < self.entry_vrp * 1.5
        }
    }
    
    /// Check if trade should take profit
    #[inline]
    pub fn check_take_profit(&self, current_vrp: f64) -> bool {
        if self.entry_vrp > 0.0 {
            current_vrp <= self.take_profit_vrp
        } else {
            current_vrp >= self.take_profit_vrp
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vrp_signal_generation() {
        let mut vrp = VolatilityRiskPremium::default();
        vrp.set_realized_vol(0.25); // 25% realized vol
        
        let params = BSParams {
            spot: 100.0,
            strike: 100.0,
            time_to_expiry: 0.25,
            risk_free_rate: 0.05,
            volatility: 0.30, // 30% implied vol
            dividend_yield: 0.0,
        };
        
        let signal = vrp.analyze(0.30, &params, OptionType::Call);
        
        assert_eq!(signal.vrp, 0.05); // 30% - 25% = 5%
        assert!(signal.sell_volatility); // Positive VRP = sell vol
        assert!(signal.is_tradeable()); // 5% > 2% threshold
    }
    
    #[test]
    fn test_realized_vol_calculation() {
        let mut vrp = VolatilityRiskPremium::new(5, 0.02, 0.005);
        
        // Simulate price series with known volatility
        let prices = vec![100.0, 101.0, 99.5, 102.0, 100.5, 101.5];
        let daily_scale = (252.0_f64).sqrt();
        
        vrp.update_realized_vol(&prices, daily_scale);
        
        assert!(vrp.realized_vol() > 0.0);
        assert!(vrp.realized_vol() < 2.0); // Sanity check
    }
    
    #[test]
    fn test_gamma_scalp_pnl() {
        let mut vrp = VolatilityRiskPremium::default();
        vrp.set_realized_vol(0.30); // Higher realized than implied
        
        let params = BSParams::default();
        
        let pnl = vrp.estimate_gamma_scalp_pnl(&params, OptionType::Call, 0.20);
        
        // Should be positive when realized > implied (long gamma profits)
        assert!(pnl > 0.0, "Gamma scalp PnL should be positive");
    }
    
    #[test]
    fn test_exit_logic() {
        let vrp = VolatilityRiskPremium::default();
        
        // Entered at 5% VRP, now at 2%
        let should_exit = vrp.should_exit(0.02, 0.05);
        assert!(should_exit, "Should exit on significant mean reversion");
        
        // Still at 4.5% VRP (little mean reversion)
        let should_hold = vrp.should_exit(0.045, 0.05);
        assert!(!should_hold, "Should hold when little mean reversion");
    }
    
    #[test]
    fn test_trade_stop_loss() {
        let trade = VrpTrade::new(
            OptionType::Call,
            105.0,
            0.25,
            -10, // Short 10 calls
            0.30,
            0.05, // Entered at 5% VRP
        );
        
        // VRP increased to 8% (trade going against us)
        assert!(trade.check_stop_loss(0.08), "Should stop loss on increasing VRP");
        
        // VRP decreased to 3% (trade working)
        assert!(!trade.check_stop_loss(0.03), "Should not stop loss on decreasing VRP");
    }
}
