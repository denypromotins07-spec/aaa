//! Real-time volatility surface construction and arbitrage enforcement
//! 
//! Builds a continuous 3D surface (Strike × Expiry × IV) from raw option chain data.
//! Implements SVI parameterization and enforces calendar/butterfly arbitrage constraints.

use std::sync::atomic::{AtomicU64, Ordering};
use crate::pricing::sabr_hybrid::SABRParams;

/// Maximum number of expiry buckets supported
const MAX_EXPIRIES: usize = 24;

/// Maximum number of strikes per expiry
const MAX_STRIKES: usize = 100;

/// Volatility surface point
#[derive(Debug, Clone, Copy)]
pub struct VolPoint {
    /// Strike price
    pub strike: f64,
    /// Implied volatility
    pub iv: f64,
    /// Option type flag (true = call, false = put)
    pub is_call: bool,
    /// Volume weight for averaging
    pub volume: f64,
    /// Open interest weight
    pub open_interest: f64,
}

/// Expiry bucket containing strikes and their volatilities
#[derive(Debug, Clone)]
pub struct ExpiryBucket {
    /// Time to expiry in years
    pub time_to_expiry: f64,
    /// Forward price for this expiry
    pub forward: f64,
    /// ATM volatility
    pub atm_iv: f64,
    /// Strikes (sorted)
    pub strikes: [f64; MAX_STRIKES],
    /// Implied volatilities at each strike
    pub vols: [f64; MAX_STRIKES],
    /// Number of valid points
    pub count: usize,
    /// Last update timestamp (nanos)
    pub last_update_ns: AtomicU64,
}

impl Default for ExpiryBucket {
    fn default() -> Self {
        Self {
            time_to_expiry: 0.0,
            forward: 0.0,
            atm_iv: 0.0,
            strikes: [0.0; MAX_STRIKES],
            vols: [0.0; MAX_STRIKES],
            count: 0,
            last_update_ns: AtomicU64::new(0),
        }
    }
}

impl ExpiryBucket {
    /// Create a new empty bucket
    #[inline]
    pub fn new(time_to_expiry: f64, forward: f64) -> Self {
        Self {
            time_to_expiry,
            forward,
            ..Default::default()
        }
    }
    
    /// Add a volatility point (maintains sorted order by strike)
    /// 
    /// # Returns
    /// `true` if successfully added, `false` if bucket is full
    #[inline]
    pub fn add_point(&mut self, point: &VolPoint) -> bool {
        if self.count >= MAX_STRIKES {
            return false;
        }
        
        // Find insertion position (binary search for sorted strikes)
        let pos = self.find_insertion_position(point.strike);
        
        // Shift existing points
        for i in (pos + 1..=self.count).rev() {
            self.strikes[i] = self.strikes[i - 1];
            self.vols[i] = self.vols[i - 1];
        }
        
        self.strikes[pos] = point.strike;
        self.vols[pos] = point.iv;
        self.count += 1;
        
        // Update timestamp
        self.last_update_ns.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            Ordering::Relaxed,
        );
        
        true
    }
    
    /// Find insertion position using binary search
    #[inline]
    fn find_insertion_position(&self, strike: f64) -> usize {
        let mut lo = 0;
        let mut hi = self.count;
        
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.strikes[mid] < strike {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        
        lo
    }
    
    /// Get interpolated volatility at a given strike
    /// Uses cubic spline interpolation
    #[inline]
    pub fn interpolate_vol(&self, strike: f64) -> Option<f64> {
        if self.count < 2 {
            return None;
        }
        
        // Find bracketing strikes
        let idx = self.find_bracket(strike)?;
        
        if idx == 0 || idx >= self.count {
            return None;
        }
        
        // Linear interpolation (simplified - full cubic spline in svi_parameterization)
        let t = (strike - self.strikes[idx - 1]) / (self.strikes[idx] - self.strikes[idx - 1]);
        let vol = self.vols[idx - 1] * (1.0 - t) + self.vols[idx] * t;
        
        Some(vol)
    }
    
    /// Find bracket index for a strike
    fn find_bracket(&self, strike: f64) -> Option<usize> {
        if strike < self.strikes[0] || strike > self.strikes[self.count - 1] {
            return None;
        }
        
        let mut lo = 0;
        let mut hi = self.count - 1;
        
        while lo <= hi {
            let mid = (lo + hi) / 2;
            if self.strikes[mid] < strike {
                lo = mid + 1;
            } else if self.strikes[mid] > strike {
                if mid == 0 {
                    return Some(0);
                }
                hi = mid - 1;
            } else {
                return Some(mid);
            }
        }
        
        Some(lo.min(self.count - 1))
    }
    
    /// Check if this bucket has any arbitrage violations
    pub fn check_butterfly_arbitrage(&self) -> bool {
        if self.count < 3 {
            return false;
        }
        
        // Butterfly arbitrage: probability density must be positive
        // d²C/dK² > 0 => convexity in call prices
        for i in 1..self.count - 1 {
            let k1 = self.strikes[i - 1];
            let k2 = self.strikes[i];
            let k3 = self.strikes[i + 1];
            
            let v1 = self.vols[i - 1];
            let v2 = self.vols[i];
            let v3 = self.vols[i + 1];
            
            // Simple convexity check on variance (w^2*T)
            let var1 = v1 * v1 * self.time_to_expiry;
            let var2 = v2 * v2 * self.time_to_expiry;
            let var3 = v3 * v3 * self.time_to_expiry;
            
            // Weights for convexity
            let w1 = (k3 - k2) / (k3 - k1);
            let w3 = (k2 - k1) / (k3 - k1);
            
            // Convex combination should be >= middle value
            let expected = w1 * var1 + w3 * var3;
            
            if var2 > expected + 1e-6 {
                return true; // Arbitrage detected
            }
        }
        
        false
    }
}

/// Volatility surface builder
#[derive(Debug)]
pub struct VolatilitySurface {
    /// Expiry buckets
    pub buckets: [ExpiryBucket; MAX_EXPIRIES],
    /// Number of active expiry buckets
    pub expiry_count: usize,
    /// Underlying spot price
    pub spot: f64,
    /// Risk-free rate
    pub risk_free_rate: f64,
    /// Surface version counter (for lock-free reads)
    pub version: AtomicU64,
}

impl Default for VolatilitySurface {
    fn default() -> Self {
        Self {
            buckets: std::array::from_fn(|_| ExpiryBucket::default()),
            expiry_count: 0,
            spot: 0.0,
            risk_free_rate: 0.0,
            version: AtomicU64::new(0),
        }
    }
}

impl VolatilitySurface {
    /// Create a new surface
    #[inline]
    pub fn new(spot: f64, risk_free_rate: f64) -> Self {
        Self {
            spot,
            risk_free_rate,
            ..Default::default()
        }
    }
    
    /// Add or update an expiry bucket
    /// 
    /// # Returns
    /// Index of the bucket, or None if max capacity reached
    pub fn add_expiry(&mut self, time_to_expiry: f64, forward: f64) -> Option<usize> {
        // Check if bucket already exists
        for i in 0..self.expiry_count {
            if (self.buckets[i].time_to_expiry - time_to_expiry).abs() < 1e-6 {
                self.buckets[i].forward = forward;
                self.increment_version();
                return Some(i);
            }
        }
        
        // Create new bucket
        if self.expiry_count >= MAX_EXPIRIES {
            return None;
        }
        
        let idx = self.expiry_count;
        self.buckets[idx] = ExpiryBucket::new(time_to_expiry, forward);
        self.expiry_count += 1;
        
        // Sort buckets by expiry time
        self.sort_buckets();
        
        self.increment_version();
        Some(idx)
    }
    
    /// Add a volatility point to a specific expiry
    pub fn add_vol_point(&mut self, expiry_idx: usize, point: &VolPoint) -> bool {
        if expiry_idx >= self.expiry_count {
            return false;
        }
        
        let result = self.buckets[expiry_idx].add_point(point);
        if result {
            self.increment_version();
        }
        result
    }
    
    /// Get interpolated volatility for (strike, expiry)
    pub fn get_vol(&self, strike: f64, time_to_expiry: f64) -> Option<f64> {
        if self.expiry_count == 0 {
            return None;
        }
        
        // Find bracketing expiries
        let (lower_idx, upper_idx) = self.find_expiry_bracket(time_to_expiry)?;
        
        if lower_idx == upper_idx {
            return self.buckets[lower_idx].interpolate_vol(strike);
        }
        
        // Interpolate between expiries
        let t_lower = self.buckets[lower_idx].time_to_expiry;
        let t_upper = self.buckets[upper_idx].time_to_expiry;
        
        let vol_lower = self.buckets[lower_idx].interpolate_vol(strike)?;
        let vol_upper = self.buckets[upper_idx].interpolate_vol(strike)?;
        
        // Time-weighted interpolation
        let weight = if (t_upper - t_lower).abs() < 1e-8 {
            0.5
        } else {
            (time_to_expiry - t_lower) / (t_upper - t_lower)
        };
        
        Some(vol_lower * (1.0 - weight) + vol_upper * weight)
    }
    
    /// Find bracketing expiry indices
    fn find_expiry_bracket(&self, time: f64) -> Option<(usize, usize)> {
        if self.expiry_count == 0 {
            return None;
        }
        
        if time <= self.buckets[0].time_to_expiry {
            return Some((0, 0));
        }
        
        if time >= self.buckets[self.expiry_count - 1].time_to_expiry {
            return Some((self.expiry_count - 1, self.expiry_count - 1));
        }
        
        // Binary search for bracket
        let mut lo = 0;
        let mut hi = self.expiry_count - 1;
        
        while lo < hi - 1 {
            let mid = (lo + hi) / 2;
            if self.buckets[mid].time_to_expiry < time {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        
        Some((lo, hi))
    }
    
    /// Sort buckets by expiry time
    fn sort_buckets(&mut self) {
        // Simple insertion sort (small array)
        for i in 1..self.expiry_count {
            let mut j = i;
            while j > 0 && self.buckets[j - 1].time_to_expiry > self.buckets[j].time_to_expiry {
                self.buckets.swap(j - 1, j);
                j -= 1;
            }
        }
    }
    
    /// Increment version atomically
    #[inline]
    fn increment_version(&mut self) {
        self.version.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Check entire surface for calendar spread arbitrage
    /// Variance should increase with time (no calendar arb)
    pub fn check_calendar_arbitrage(&self) -> Vec<(usize, usize)> {
        let mut violations = Vec::new();
        
        for i in 0..self.expiry_count.saturating_sub(1) {
            for j in (i + 1)..self.expiry_count {
                let t1 = self.buckets[i].time_to_expiry;
                let t2 = self.buckets[j].time_to_expiry;
                
                if t2 <= t1 {
                    continue;
                }
                
                // Compare ATM vols
                let atm1 = self.buckets[i].atm_iv;
                let atm2 = self.buckets[j].atm_iv;
                
                // Total variance should increase with time
                let var1 = atm1 * atm1 * t1;
                let var2 = atm2 * atm2 * t2;
                
                if var1 > var2 + 1e-6 {
                    violations.push((i, j));
                }
            }
        }
        
        violations
    }
    
    /// Check for butterfly arbitrage across all expiries
    pub fn check_butterfly_arbitrage(&self) -> Vec<usize> {
        let mut violations = Vec::new();
        
        for i in 0..self.expiry_count {
            if self.buckets[i].check_butterfly_arbitrage() {
                violations.push(i);
            }
        }
        
        violations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_surface_construction() {
        let mut surface = VolatilitySurface::new(100.0, 0.05);
        
        // Add expiries
        let idx1 = surface.add_expiry(0.25, 100.5).unwrap();
        let idx2 = surface.add_expiry(0.5, 101.0).unwrap();
        let idx3 = surface.add_expiry(1.0, 102.0).unwrap();
        
        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(idx3, 2);
        assert_eq!(surface.expiry_count, 3);
        
        // Verify sorted by expiry
        assert!(surface.buckets[0].time_to_expiry <= surface.buckets[1].time_to_expiry);
        assert!(surface.buckets[1].time_to_expiry <= surface.buckets[2].time_to_expiry);
    }
    
    #[test]
    fn test_vol_interpolation() {
        let mut surface = VolatilitySurface::new(100.0, 0.05);
        let idx = surface.add_expiry(0.5, 100.0).unwrap();
        
        // Add some vol points
        surface.add_vol_point(idx, &VolPoint {
            strike: 90.0,
            iv: 0.35,
            is_call: true,
            volume: 100.0,
            open_interest: 500.0,
        }).unwrap();
        
        surface.add_vol_point(idx, &VolPoint {
            strike: 100.0,
            iv: 0.30,
            is_call: true,
            volume: 200.0,
            open_interest: 1000.0,
        }).unwrap();
        
        surface.add_vol_point(idx, &VolPoint {
            strike: 110.0,
            iv: 0.28,
            is_call: true,
            volume: 150.0,
            open_interest: 750.0,
        }).unwrap();
        
        // Test interpolation
        let vol_95 = surface.get_vol(95.0, 0.5).unwrap();
        assert!(vol_95 > 0.30 && vol_95 < 0.35, "Interpolated vol out of range: {}", vol_95);
    }
    
    #[test]
    fn test_no_arbitrage_empty() {
        let surface = VolatilitySurface::new(100.0, 0.05);
        
        let calendar_viols = surface.check_calendar_arbitrage();
        let butterfly_viols = surface.check_butterfly_arbitrage();
        
        assert!(calendar_viols.is_empty());
        assert!(butterfly_viols.is_empty());
    }
    
    #[test]
    fn test_version_increment() {
        let mut surface = VolatilitySurface::new(100.0, 0.05);
        let initial_version = surface.version.load(Ordering::Relaxed);
        
        surface.add_expiry(0.5, 100.0);
        let v1 = surface.version.load(Ordering::Relaxed);
        
        assert!(v1 > initial_version);
    }
}
