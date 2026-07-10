//! Dispersion trading strategy
//! 
//! Trades the spread between index implied volatility and the weighted
//! average implied volatility of constituent components.

use crate::pricing::black_scholes_fast::{BSParams, OptionType};

/// Maximum number of constituents supported
const MAX_CONSTITUENTS: usize = 100;

/// Signal from dispersion analysis
#[derive(Debug, Clone, Copy)]
pub struct DispersionSignal {
    /// Index implied volatility
    pub index_iv: f64,
    /// Weighted average component IV (correlation-adjusted)
    pub component_iv: f64,
    /// Implied correlation
    pub implied_correlation: f64,
    /// Fair value correlation
    pub fair_correlation: f64,
    /// Signal strength (-1 to 1)
    pub signal_strength: f64,
    /// Recommended action: true = long dispersion (long comp vol, short index vol)
    pub long_dispersion: bool,
    /// Expected P&L per unit notional
    pub expected_pnl: f64,
}

impl DispersionSignal {
    #[inline]
    pub const fn new(
        index_iv: f64,
        component_iv: f64,
        implied_corr: f64,
        fair_corr: f64,
        signal_strength: f64,
        long_dispersion: bool,
        expected_pnl: f64,
    ) -> Self {
        Self {
            index_iv,
            component_iv,
            implied_correlation: implied_corr,
            fair_correlation: fair_corr,
            signal_strength,
            long_dispersion,
            expected_pnl,
        }
    }
    
    /// Check if signal meets minimum threshold
    #[inline]
    pub fn is_tradeable(&self) -> bool {
        self.signal_strength.abs() >= 0.3 
            && (self.index_iv - self.component_iv).abs() >= 0.02
    }
}

/// Constituent in the index basket
#[derive(Debug, Clone, Copy)]
pub struct Constituent {
    /// Symbol/name identifier
    pub id: u32,
    /// Weight in index (sum to 1.0)
    pub weight: f64,
    /// Implied volatility
    pub implied_vol: f64,
    /// Realized volatility
    pub realized_vol: f64,
}

impl Constituent {
    #[inline]
    pub const fn new(id: u32, weight: f64, implied_vol: f64, realized_vol: f64) -> Self {
        Self { id, weight, implied_vol, realized_vol }
    }
}

/// Index volatility basket for dispersion calculation
#[derive(Debug, Clone)]
pub struct IndexVolBasket {
    /// Index name/identifier
    pub index_id: u32,
    /// Index-level implied volatility
    pub index_iv: f64,
    /// Constituents with their vols and weights
    pub constituents: [Option<Constituent>; MAX_CONSTITUENTS],
    /// Number of active constituents
    pub count: usize,
    /// Pre-computed weighted average component IV
    pub weighted_comp_iv: f64,
}

impl Default for IndexVolBasket {
    fn default() -> Self {
        Self::new(0)
    }
}

impl IndexVolBasket {
    /// Create a new basket for an index
    #[inline]
    pub const fn new(index_id: u32) -> Self {
        Self {
            index_id,
            index_iv: 0.0,
            constituents: [None; MAX_CONSTITUENTS],
            count: 0,
            weighted_comp_iv: 0.0,
        }
    }
    
    /// Add a constituent to the basket
    pub fn add_constituent(&mut self, constituent: Constituent) -> bool {
        if self.count >= MAX_CONSTITUENTS {
            return false;
        }
        
        self.constituents[self.count] = Some(constituent);
        self.count += 1;
        self.recalculate_weighted_iv();
        true
    }
    
    /// Update index IV
    #[inline]
    pub fn set_index_iv(&mut self, iv: f64) {
        self.index_iv = iv;
    }
    
    /// Recalculate weighted average component IV
    fn recalculate_weighted_iv(&mut self) {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        
        for i in 0..self.count {
            if let Some(c) = self.constituents[i] {
                weighted_sum += c.weight * c.implied_vol;
                total_weight += c.weight;
            }
        }
        
        if total_weight > 0.0 {
            self.weighted_comp_iv = weighted_sum / total_weight;
        }
    }
    
    /// Get weighted component IV
    #[inline]
    pub fn weighted_component_iv(&self) -> f64 {
        self.weighted_comp_iv
    }
}

/// Dispersion trader - analyzes and trades index vs component vol spread
pub struct DispersionTrader {
    /// Historical average correlation
    historical_avg_corr: f64,
    /// Minimum signal threshold
    min_signal_threshold: f64,
    /// Correlation mean-reversion speed
    corr_mean_reversion: f64,
}

impl Default for DispersionTrader {
    fn default() -> Self {
        Self::new(0.6, 0.02, 0.1)
    }
}

impl DispersionTrader {
    /// Create a new dispersion trader
    /// 
    /// # Arguments
    /// * `hist_avg_corr` - Historical average correlation (e.g., 0.6 for equities)
    /// * `min_threshold` - Minimum signal strength to trade
    /// * `mean_reversion` - Speed of correlation mean reversion
    #[inline]
    pub fn new(hist_avg_corr: f64, min_threshold: f64, mean_reversion: f64) -> Self {
        Self {
            historical_avg_corr: hist_avg_corr.clamp(0.0, 1.0),
            min_signal_threshold: min_threshold,
            corr_mean_reversion: mean_reversion,
        }
    }
    
    /// Analyze dispersion opportunity
    /// 
    /// # Arguments
    /// * `basket` - Index volatility basket
    /// 
    /// # Returns
    /// DispersionSignal with recommendation
    pub fn analyze(&self, basket: &IndexVolBasket) -> DispersionSignal {
        let index_iv = basket.index_iv;
        let comp_iv = basket.weighted_comp_iv;
        
        // Calculate implied correlation
        // σ_index² ≈ Σ w_i² σ_i² + ΣΣ w_i w_j σ_i σ_j ρ_ij
        // Simplified: ρ_implied ≈ (σ_index² - Σ w_i² σ_i²) / (ΣΣ w_i w_j σ_i σ_j)
        let implied_corr = self.calculate_implied_correlation(basket);
        
        // Fair value correlation (mean-reverting to historical average)
        let fair_corr = self.historical_avg_corr;
        
        // Signal based on correlation gap
        let corr_gap = implied_corr - fair_corr;
        let signal_strength = corr_gap.clamp(-1.0, 1.0);
        
        // Long dispersion when implied correlation is too high
        // (component vol cheap relative to index vol)
        let long_dispersion = implied_corr > fair_corr + self.min_signal_threshold;
        
        // Expected P&L from correlation mean reversion
        let expected_pnl = self.estimate_dispersion_pnl(basket, implied_corr, fair_corr);
        
        DispersionSignal::new(
            index_iv,
            comp_iv,
            implied_corr,
            fair_corr,
            signal_strength,
            long_dispersion,
            expected_pnl,
        )
    }
    
    /// Calculate implied correlation from index and component vols
    fn calculate_implied_correlation(&self, basket: &IndexVolBasket) -> f64 {
        if basket.count < 2 || basket.index_iv <= 0.0 {
            return self.historical_avg_corr;
        }
        
        // Sum of weighted squared vols
        let mut sum_w2v2 = 0.0;
        let mut sum_wv = 0.0;
        
        for i in 0..basket.count {
            if let Some(c) = basket.constituents[i] {
                sum_w2v2 += c.weight * c.weight * c.implied_vol * c.implied_vol;
                sum_wv += c.weight * c.implied_vol;
            }
        }
        
        // σ_index² = Σ w_i² σ_i² + (1 - ρ) * ΣΣ w_i w_j σ_i σ_j (simplified)
        // Solve for ρ
        let index_var = basket.index_iv * basket.index_iv;
        
        if sum_w2v2 >= index_var {
            return 0.0; // Edge case
        }
        
        // Approximate cross terms
        let cross_terms = sum_wv * sum_wv - sum_w2v2;
        
        if cross_terms <= 0.0 {
            return 0.0;
        }
        
        // ρ ≈ (σ_index² - Σ w_i² σ_i²) / cross_terms
        let implied_corr = (index_var - sum_w2v2) / cross_terms;
        
        implied_corr.clamp(0.0, 1.0)
    }
    
    /// Estimate P&L from dispersion trade
    fn estimate_dispersion_pnl(
        &self,
        basket: &IndexVolBasket,
        implied_corr: f64,
        fair_corr: f64,
    ) -> f64 {
        // P&L ≈ Notional * (ρ_implied - ρ_fair) * Vega * Vol
        let vol_spread = basket.index_iv - basket.weighted_comp_iv;
        let corr_gap = implied_corr - fair_corr;
        
        // Simplified: assume vega of 0.1 per vol point
        0.1 * corr_gap * vol_spread * 100.0
    }
    
    /// Calculate optimal hedge ratio for dispersion trade
    pub fn calculate_hedge_ratio(&self, basket: &IndexVolBasket, constituent_idx: usize) -> f64 {
        if constituent_idx >= basket.count {
            return 0.0;
        }
        
        if let Some(c) = basket.constituents[constituent_idx] {
            // Hedge ratio = weight * (σ_comp / σ_index) * correlation
            let corr = self.calculate_implied_correlation(basket);
            c.weight * (c.implied_vol / basket.index_iv.max(0.001)) * corr
        } else {
            0.0
        }
    }
    
    /// Get historical average correlation
    #[inline]
    pub fn historical_avg_correlation(&self) -> f64 {
        self.historical_avg_corr
    }
    
    /// Update historical correlation estimate
    #[inline]
    pub fn update_historical_correlation(&mut self, new_corr: f64) {
        // Exponential moving average update
        self.historical_avg_corr = 
            (1.0 - self.corr_mean_reversion) * self.historical_avg_corr 
            + self.corr_mean_reversion * new_corr.clamp(0.0, 1.0);
    }
}

/// Dispersion trade execution parameters
#[derive(Debug, Clone)]
pub struct DispersionTrade {
    /// Index to trade
    pub index_id: u32,
    /// Notional amount
    pub notional: f64,
    /// Long component strikes (array)
    pub long_strikes: Vec<f64>,
    /// Short index strike
    pub short_strike: f64,
    /// Entry implied correlation
    pub entry_implied_corr: f64,
    /// Exit target correlation
    pub exit_target_corr: f64,
    /// Stop loss correlation level
    pub stop_loss_corr: f64,
}

impl DispersionTrade {
    #[inline]
    pub fn new(
        index_id: u32,
        notional: f64,
        long_strikes: Vec<f64>,
        short_strike: f64,
        entry_corr: f64,
    ) -> Self {
        Self {
            index_id,
            notional,
            long_strikes,
            short_strike,
            entry_implied_corr: entry_corr,
            exit_target_corr: 0.6, // Mean revert to 0.6
            stop_loss_corr: entry_corr * 1.5, // Stop at 50% adverse move
        }
    }
    
    /// Check if trade should exit
    pub fn check_exit(&self, current_corr: f64, long_dispersion: bool) -> bool {
        if long_dispersion {
            // Long dispersion: profit when corr decreases
            current_corr <= self.exit_target_corr || current_corr >= self.stop_loss_corr
        } else {
            // Short dispersion: profit when corr increases
            current_corr >= self.exit_target_corr || current_corr <= self.stop_loss_corr
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_implied_correlation_calculation() {
        let mut basket = IndexVolBasket::new(1);
        basket.set_index_iv(0.25);
        
        // Add two equal-weight constituents
        basket.add_constituent(Constituent::new(1, 0.5, 0.20, 0.18)).unwrap();
        basket.add_constituent(Constituent::new(2, 0.5, 0.22, 0.20)).unwrap();
        
        let trader = DispersionTrader::default();
        let implied_corr = trader.calculate_implied_correlation(&basket);
        
        assert!(implied_corr >= 0.0 && implied_corr <= 1.0);
    }
    
    #[test]
    fn test_dispersion_signal_generation() {
        let mut basket = IndexVolBasket::new(1);
        basket.set_index_iv(0.30);
        
        basket.add_constituent(Constituent::new(1, 0.5, 0.25, 0.23)).unwrap();
        basket.add_constituent(Constituent::new(2, 0.5, 0.27, 0.25)).unwrap();
        
        let trader = DispersionTrader::new(0.6, 0.02, 0.1);
        let signal = trader.analyze(&basket);
        
        assert!(signal.index_iv > 0.0);
        assert!(signal.component_iv > 0.0);
        assert!(signal.index_iv > signal.component_iv); // Index vol > component vol
    }
    
    #[test]
    fn test_long_dispersion_signal() {
        let mut basket = IndexVolBasket::new(1);
        basket.set_index_iv(0.35); // High index vol
        
        basket.add_constituent(Constituent::new(1, 0.5, 0.20, 0.18)).unwrap();
        basket.add_constituent(Constituent::new(2, 0.5, 0.22, 0.20)).unwrap();
        
        let trader = DispersionTrader::new(0.5, 0.02, 0.1);
        let signal = trader.analyze(&basket);
        
        // Should signal long dispersion (components cheap vs index)
        assert!(signal.long_dispersion);
    }
    
    #[test]
    fn test_hedge_ratio() {
        let mut basket = IndexVolBasket::new(1);
        basket.set_index_iv(0.25);
        
        basket.add_constituent(Constituent::new(1, 0.3, 0.20, 0.18)).unwrap();
        basket.add_constituent(Constituent::new(2, 0.7, 0.28, 0.26)).unwrap();
        
        let trader = DispersionTrader::default();
        
        let ratio_0 = trader.calculate_hedge_ratio(&basket, 0);
        let ratio_1 = trader.calculate_hedge_ratio(&basket, 1);
        
        assert!(ratio_0 > 0.0 && ratio_0 < 1.0);
        assert!(ratio_1 > 0.0 && ratio_1 < 1.0);
    }
    
    #[test]
    fn test_correlation_mean_reversion() {
        let mut trader = DispersionTrader::new(0.6, 0.02, 0.2);
        
        // Update with higher correlation
        trader.update_historical_correlation(0.8);
        
        let new_avg = trader.historical_avg_correlation();
        
        // Should have moved toward 0.8 but not fully there
        assert!(new_avg > 0.6 && new_avg < 0.8);
    }
}
