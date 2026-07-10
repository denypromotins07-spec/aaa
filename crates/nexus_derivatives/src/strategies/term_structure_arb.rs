//! Term structure arbitrage strategy
//! 
//! Trades the VIX-style term structure (contango/backwardation) of crypto
//! perpetual funding rates vs. quarterly futures basis.

use crate::pricing::black_scholes_fast::{BSParams, OptionType};

/// Number of tenor buckets in term structure
const MAX_TENORS: usize = 12;

/// Term structure state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContangoBackwardation {
    /// Normal: longer-dated prices higher (positive carry)
    Contango,
    /// Inverted: shorter-dated prices higher (negative carry)
    Backwardation,
    /// Flat: no significant slope
    Flat,
}

/// Signal from term structure analysis
#[derive(Debug, Clone, Copy)]
pub struct TermStructureSignal {
    /// Current term structure state
    pub structure: ContangoBackwardation,
    /// Slope of term structure (annualized %)
    pub slope: f64,
    /// Curvature (second derivative)
    pub curvature: f64,
    /// Fair value slope (mean-reversion target)
    pub fair_slope: f64,
    /// Signal strength (-1 to 1)
    pub signal_strength: f64,
    /// Recommended action: true = long front/short back, false = opposite
    pub long_front_short_back: bool,
    /// Expected P&L from convergence
    pub expected_pnl: f64,
}

impl TermStructureSignal {
    #[inline]
    pub const fn new(
        structure: ContangoBackwardation,
        slope: f64,
        curvature: f64,
        fair_slope: f64,
        signal_strength: f64,
        long_front_short_back: bool,
        expected_pnl: f64,
    ) -> Self {
        Self {
            structure,
            slope,
            curvature,
            fair_slope,
            signal_strength,
            long_front_short_back,
            expected_pnl,
        }
    }
    
    /// Check if signal meets minimum threshold
    #[inline]
    pub fn is_tradeable(&self) -> bool {
        self.signal_strength.abs() >= 0.3 
            && self.slope.abs() >= 0.05 // 5% annualized
    }
}

/// Tenor point in term structure
#[derive(Debug, Clone, Copy)]
pub struct TenorPoint {
    /// Time to maturity in years
    pub time: f64,
    /// Price or rate at this tenor
    pub value: f64,
    /// Volume at this tenor
    pub volume: f64,
    /// Open interest
    pub open_interest: f64,
}

impl TenorPoint {
    #[inline]
    pub const fn new(time: f64, value: f64, volume: f64, oi: f64) -> Self {
        Self { time, value, volume, open_interest: oi }
    }
}

/// Term structure curve for analysis
#[derive(Debug, Clone)]
pub struct TermStructureCurve {
    /// Tenor points (sorted by time)
    pub tenors: [Option<TenorPoint>; MAX_TENORS],
    /// Number of active tenors
    pub count: usize,
    /// Spot price reference
    pub spot: f64,
    /// Timestamp of last update (nanos)
    pub last_update_ns: u64,
}

impl Default for TermStructureCurve {
    fn default() -> Self {
        Self::new(100.0)
    }
}

impl TermStructureCurve {
    /// Create a new term structure curve
    #[inline]
    pub const fn new(spot: f64) -> Self {
        Self {
            tenors: [None; MAX_TENORS],
            count: 0,
            spot,
            last_update_ns: 0,
        }
    }
    
    /// Add a tenor point (maintains sorted order)
    pub fn add_tenor(&mut self, tenor: TenorPoint) -> bool {
        if self.count >= MAX_TENORS {
            return false;
        }
        
        // Find insertion position
        let pos = self.find_insertion_position(tenor.time);
        
        // Shift existing points
        for i in (pos + 1..=self.count).rev() {
            self.tenors[i] = self.tenors[i - 1];
        }
        
        self.tenors[pos] = Some(tenor);
        self.count += 1;
        
        // Update timestamp
        self.last_update_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        true
    }
    
    /// Find insertion position using binary search
    fn find_insertion_position(&self, time: f64) -> usize {
        let mut lo = 0;
        let mut hi = self.count;
        
        while lo < hi {
            let mid = (lo + hi) / 2;
            if let Some(t) = self.tenors[mid] {
                if t.time < time {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            } else {
                hi = mid;
            }
        }
        
        lo
    }
    
    /// Get interpolated value at a given time
    pub fn interpolate(&self, time: f64) -> Option<f64> {
        if self.count < 2 {
            return None;
        }
        
        // Find bracketing tenors
        for i in 0..self.count - 1 {
            if let (Some(t1), Some(t2)) = (self.tenors[i], self.tenors[i + 1]) {
                if t1.time <= time && t2.time >= time {
                    // Linear interpolation
                    let weight = (time - t1.time) / (t2.time - t1.time);
                    return Some(t1.value * (1.0 - weight) + t2.value * weight);
                }
            }
        }
        
        // Extrapolate from nearest points
        if time < self.tenors[0]?.time {
            return Some(self.tenors[0]?.value);
        }
        if time > self.tenors[self.count - 1]?.time {
            return Some(self.tenors[self.count - 1]?.value);
        }
        
        None
    }
    
    /// Calculate slope between two tenors
    pub fn calculate_slope(&self, t1: f64, t2: f64) -> Option<f64> {
        let v1 = self.interpolate(t1)?;
        let v2 = self.interpolate(t2)?;
        
        let dt = t2 - t1;
        if dt <= 0.0 {
            return None;
        }
        
        // Annualized slope
        Some((v2 - v1) / dt * (v1.abs() + 0.001).recip())
    }
    
    /// Get front month value
    #[inline]
    pub fn front_value(&self) -> Option<f64> {
        self.tenors.iter().flatten().next().map(|t| t.value)
    }
    
    /// Get back month value
    #[inline]
    pub fn back_value(&self) -> Option<f64> {
        self.tenors.iter().flatten().last().map(|t| t.value)
    }
}

/// Term structure arbitrage analyzer
pub struct TermStructureArb {
    /// Historical average slope (mean-reversion target)
    fair_slope: f64,
    /// Slope mean-reversion speed
    mean_reversion_speed: f64,
    /// Minimum signal threshold
    min_threshold: f64,
    /// Funding rate for perpetuals
    current_funding_rate: f64,
}

impl Default for TermStructureArb {
    fn default() -> Self {
        Self::new(0.0, 0.1, 0.05)
    }
}

impl TermStructureArb {
    /// Create a new term structure arb analyzer
    /// 
    /// # Arguments
    /// * `fair_slope` - Historical average slope (e.g., 0 for symmetric)
    /// * `mean_reversion` - Speed of mean reversion (0-1)
    /// * `min_threshold` - Minimum signal to trade
    #[inline]
    pub fn new(fair_slope: f64, mean_reversion: f64, min_threshold: f64) -> Self {
        Self {
            fair_slope,
            mean_reversion_speed: mean_reversion.clamp(0.0, 1.0),
            min_threshold: min_threshold,
            current_funding_rate: 0.0,
        }
    }
    
    /// Set current perpetual funding rate
    #[inline]
    pub fn set_funding_rate(&mut self, rate: f64) {
        self.current_funding_rate = rate;
    }
    
    /// Analyze term structure and generate signal
    /// 
    /// # Arguments
    /// * `curve` - Current term structure curve
    /// 
    /// # Returns
    /// TermStructureSignal with recommendation
    pub fn analyze(&self, curve: &TermStructureCurve) -> TermStructureSignal {
        if curve.count < 2 {
            return TermStructureSignal::new(
                ContangoBackwardation::Flat,
                0.0,
                0.0,
                self.fair_slope,
                0.0,
                false,
                0.0,
            );
        }
        
        // Get front and back values
        let front = curve.front_value().unwrap_or(curve.spot);
        let back = curve.back_value().unwrap_or(curve.spot);
        
        // Calculate slope
        let slope = if let Some(s) = curve.calculate_slope(0.083, 1.0) {
            s // 1 month to 1 year
        } else {
            (back - front) / front
        };
        
        // Determine structure type
        let structure = if slope > self.min_threshold {
            ContangoBackwardation::Contango
        } else if slope < -self.min_threshold {
            ContangoBackwardation::Backwardation
        } else {
            ContangoBackwardation::Flat
        };
        
        // Calculate curvature (second derivative approximation)
        let curvature = self.calculate_curvature(curve);
        
        // Signal strength based on deviation from fair value
        let slope_gap = slope - self.fair_slope;
        let signal_strength = (slope_gap / slope.abs().max(0.01)).clamp(-1.0, 1.0);
        
        // Trade direction: bet on mean reversion
        // If in contango (slope > 0), short front / long back
        // If in backwardation (slope < 0), long front / short back
        let long_front_short_back = slope < self.fair_slope - self.min_threshold;
        
        // Expected P&L from convergence
        let expected_pnl = self.estimate_convergence_pnl(slope, curve);
        
        TermStructureSignal::new(
            structure,
            slope,
            curvature,
            self.fair_slope,
            signal_strength,
            long_front_short_back,
            expected_pnl,
        )
    }
    
    /// Calculate curvature of term structure
    fn calculate_curvature(&self, curve: &TermStructureCurve) -> f64 {
        if curve.count < 3 {
            return 0.0;
        }
        
        // Get three points for second derivative
        let t1 = 0.083; // 1 month
        let t2 = 0.25;  // 3 months
        let t3 = 0.5;   // 6 months
        
        let v1 = curve.interpolate(t1).unwrap_or(curve.spot);
        let v2 = curve.interpolate(t2).unwrap_or(curve.spot);
        let v3 = curve.interpolate(t3).unwrap_or(curve.spot);
        
        // Second derivative approximation
        let h1 = t2 - t1;
        let h2 = t3 - t2;
        
        if h1 <= 0.0 || h2 <= 0.0 {
            return 0.0;
        }
        
        let d1 = (v2 - v1) / h1;
        let d2 = (v3 - v2) / h2;
        
        (d2 - d1) / ((h1 + h2) / 2.0)
    }
    
    /// Estimate P&L from term structure convergence
    fn estimate_convergence_pnl(&self, slope: f64, curve: &TermStructureCurve) -> f64 {
        // P&L ≈ Notional * (slope - fair_slope) * Duration
        let slope_gap = slope - self.fair_slope;
        let duration = 0.25; // Assume 3-month holding period
        
        slope_gap * duration * 100.0 // Scaled for readability
    }
    
    /// Compare perpetual funding rate to futures implied rate
    /// 
    /// Returns the funding-futures basis (positive = funding rich)
    pub fn funding_futures_basis(&self, curve: &TermStructureCurve) -> f64 {
        let front = curve.front_value().unwrap_or(curve.spot);
        let annualized_basis = (front - curve.spot) / curve.spot * 12.0; // Monthly to annual
        
        // Funding rate is typically 8-hour, annualize it
        let annualized_funding = self.current_funding_rate * 3.0 * 365.0;
        
        annualized_funding - annualized_basis
    }
    
    /// Update fair slope estimate with new observation
    #[inline]
    pub fn update_fair_slope(&mut self, observed_slope: f64) {
        self.fair_slope = (1.0 - self.mean_reversion_speed) * self.fair_slope
            + self.mean_reversion_speed * observed_slope;
    }
}

/// Term structure trade execution parameters
#[derive(Debug, Clone)]
pub struct TermStructureTrade {
    /// Front leg (nearby contract)
    pub front_leg: Leg,
    /// Back leg (deferred contract)
    pub back_leg: Leg,
    /// Entry slope
    pub entry_slope: f64,
    /// Exit target slope
    pub exit_target_slope: f64,
    /// Stop loss slope level
    pub stop_loss_slope: f64,
}

/// Trade leg specification
#[derive(Debug, Clone, Copy)]
pub struct Leg {
    /// Contract identifier
    pub contract_id: u32,
    /// Quantity (positive = long, negative = short)
    pub quantity: i64,
    /// Entry price
    pub entry_price: f64,
}

impl Leg {
    #[inline]
    pub const fn new(contract_id: u32, quantity: i64, entry_price: f64) -> Self {
        Self { contract_id, quantity, entry_price }
    }
}

impl TermStructureTrade {
    #[inline]
    pub fn new(
        front_leg: Leg,
        back_leg: Leg,
        entry_slope: f64,
    ) -> Self {
        Self {
            front_leg,
            back_leg,
            entry_slope,
            exit_target_slope: 0.0, // Mean revert to flat
            stop_loss_slope: entry_slope * 1.5, // 50% adverse move
        }
    }
    
    /// Check if trade should exit
    pub fn check_exit(&self, current_slope: f64) -> bool {
        // Exit on mean reversion or stop loss
        let pnl_fraction = 1.0 - (current_slope / self.entry_slope).abs();
        
        pnl_fraction > 0.5 || current_slope.abs() > self.stop_loss_slope.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_contango_detection() {
        let mut curve = TermStructureCurve::new(100.0);
        
        // Add upward sloping curve (contango)
        curve.add_tenor(TenorPoint::new(0.083, 101.0, 1000.0, 5000.0)).unwrap();
        curve.add_tenor(TenorPoint::new(0.25, 102.0, 800.0, 4000.0)).unwrap();
        curve.add_tenor(TenorPoint::new(0.5, 104.0, 600.0, 3000.0)).unwrap();
        curve.add_tenor(TenorPoint::new(1.0, 108.0, 400.0, 2000.0)).unwrap();
        
        let arb = TermStructureArb::default();
        let signal = arb.analyze(&curve);
        
        assert_eq!(signal.structure, ContangoBackwardation::Contango);
        assert!(signal.slope > 0.0);
    }
    
    #[test]
    fn test_backwardation_detection() {
        let mut curve = TermStructureCurve::new(100.0);
        
        // Add downward sloping curve (backwardation)
        curve.add_tenor(TenorPoint::new(0.083, 99.0, 1000.0, 5000.0)).unwrap();
        curve.add_tenor(TenorPoint::new(0.25, 98.0, 800.0, 4000.0)).unwrap();
        curve.add_tenor(TenorPoint::new(0.5, 97.0, 600.0, 3000.0)).unwrap();
        curve.add_tenor(TenorPoint::new(1.0, 95.0, 400.0, 2000.0)).unwrap();
        
        let arb = TermStructureArb::default();
        let signal = arb.analyze(&curve);
        
        assert_eq!(signal.structure, ContangoBackwardation::Backwardation);
        assert!(signal.slope < 0.0);
    }
    
    #[test]
    fn test_signal_direction() {
        let mut curve = TermStructureCurve::new(100.0);
        
        // Strong contango
        curve.add_tenor(TenorPoint::new(0.083, 105.0, 1000.0, 5000.0)).unwrap();
        curve.add_tenor(TenorPoint::new(1.0, 120.0, 400.0, 2000.0)).unwrap();
        
        let arb = TermStructureArb::new(0.0, 0.1, 0.05);
        let signal = arb.analyze(&curve);
        
        // Should signal short front / long back (bet on mean reversion)
        assert!(!signal.long_front_short_back);
    }
    
    #[test]
    fn test_interpolation() {
        let mut curve = TermStructureCurve::new(100.0);
        
        curve.add_tenor(TenorPoint::new(0.25, 102.0, 1000.0, 5000.0)).unwrap();
        curve.add_tenor(TenorPoint::new(0.5, 104.0, 800.0, 4000.0)).unwrap();
        
        // Interpolate at 0.375 (midpoint)
        let interp = curve.interpolate(0.375).unwrap();
        
        assert!((interp - 103.0).abs() < 0.1); // Should be ~103
    }
    
    #[test]
    fn test_funding_futures_basis() {
        let mut arb = TermStructureArb::default();
        arb.set_funding_rate(0.0001); // 0.01% per 8 hours
        
        let mut curve = TermStructureCurve::new(100.0);
        curve.add_tenor(TenorPoint::new(0.083, 101.0, 1000.0, 5000.0)).unwrap();
        
        let basis = arb.funding_futures_basis(&curve);
        
        // Funding should be compared to futures basis
        assert!(basis.is_finite());
    }
}
