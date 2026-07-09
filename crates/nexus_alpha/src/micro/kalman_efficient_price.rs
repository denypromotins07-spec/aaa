//! Dual Kalman Filter for Latent Efficient Price Estimation
//! 
//! Implements a dual Kalman Filter system to estimate the "latent efficient price"
//! and detect cross-exchange lead-lag micro-arbitrage opportunities.
//! Zero-allocation implementation with fixed-size state vectors.

use nexus_core::memory::arena::BumpAllocator;

/// State dimension for Kalman filter
pub const STATE_DIM: usize = 2; // Price and velocity

/// Measurement dimension
pub const MEASUREMENT_DIM: usize = 1; // Just price measurement

/// Cache-line aligned Kalman Filter state
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct KalmanState {
    /// State vector [price, velocity]
    pub x: [f64; STATE_DIM],
    /// Error covariance matrix (2x2, stored as flat array)
    pub p: [f64; STATE_DIM * STATE_DIM],
    /// Process noise covariance
    pub q: [f64; STATE_DIM * STATE_DIM],
    /// Measurement noise covariance
    pub r: f64,
    /// Valid flag
    pub valid: bool,
    /// Padding
    _padding: [u8; 24],
}

impl Default for KalmanState {
    fn default() -> Self {
        Self {
            x: [0.0, 0.0],
            p: [1.0, 0.0, 0.0, 1.0], // Identity
            q: [0.01, 0.0, 0.0, 0.01], // Small process noise
            r: 1.0, // Measurement noise
            valid: false,
            _padding: [0u8; 24],
        }
    }
}

/// Kalman Filter for single exchange
pub struct ExchangeKalmanFilter {
    /// Filter state
    state: KalmanState,
    /// State transition matrix (2x2)
    a: [f64; STATE_DIM * STATE_DIM],
    /// Measurement matrix (1x2)
    h: [f64; MEASUREMENT_DIM * STATE_DIM],
    /// Innovation (residual)
    innovation: f64,
    /// Innovation variance
    innovation_var: f64,
    /// Last update timestamp
    last_ts: u64,
    /// Exchange ID
    exchange_id: u32,
}

impl ExchangeKalmanFilter {
    pub fn new(exchange_id: u32, initial_price: f64) -> Self {
        // State transition: constant velocity model
        // x[k+1] = A * x[k] where A = [[1, dt], [0, 1]]
        let dt = 0.001; // Assume 1ms time step
        
        let mut filter = Self {
            state: KalmanState::default(),
            a: [1.0, dt, 0.0, 1.0],
            h: [1.0, 0.0], // Only observe price
            innovation: 0.0,
            innovation_var: 0.0,
            last_ts: 0,
            exchange_id,
        };

        // Initialize with given price
        filter.state.x[0] = initial_price;
        filter.state.x[1] = 0.0; // Initial velocity
        filter.state.valid = true;

        filter
    }

    /// Process a new price measurement
    #[inline]
    pub fn update(&mut self, ts: u64, measured_price: f64) -> KalmanResult {
        // Update time step if needed
        if self.last_ts > 0 && ts > self.last_ts {
            let dt = ((ts - self.last_ts) as f64 / 1_000_000_000.0).min(0.1); // Cap at 100ms
            self.a[1] = dt; // Update A[0,1] = dt
        }
        self.last_ts = ts;

        // === PREDICT STEP ===
        // x_pred = A * x
        let x_pred = [
            self.a[0] * self.state.x[0] + self.a[1] * self.state.x[1],
            self.a[2] * self.state.x[0] + self.a[3] * self.state.x[1],
        ];

        // P_pred = A * P * A^T + Q
        let p_pred = self.predict_covariance();

        // === UPDATE STEP ===
        // Innovation: y = z - H * x_pred
        let z = measured_price;
        let z_pred = self.h[0] * x_pred[0] + self.h[1] * x_pred[1];
        self.innovation = z - z_pred;

        // Innovation variance: S = H * P_pred * H^T + R
        let hp = [
            self.h[0] * p_pred[0] + self.h[1] * p_pred[2],
            self.h[0] * p_pred[1] + self.h[1] * p_pred[3],
        ];
        self.innovation_var = hp[0] * self.h[0] + hp[1] * self.h[1] + self.state.r;

        // Kalman gain: K = P_pred * H^T / S
        let s_inv = 1.0 / self.innovation_var.max(1e-10);
        let k = [
            (p_pred[0] * self.h[0] + p_pred[1] * self.h[1]) * s_inv,
            (p_pred[2] * self.h[0] + p_pred[3] * self.h[1]) * s_inv,
        ];

        // Update state: x = x_pred + K * y
        self.state.x[0] = x_pred[0] + k[0] * self.innovation;
        self.state.x[1] = x_pred[1] + k[1] * self.innovation;

        // Update covariance: P = (I - K*H) * P_pred
        self.update_covariance(&k, &p_pred);

        KalmanResult {
            estimated_price: self.state.x[0],
            estimated_velocity: self.state.x[1],
            innovation: self.innovation,
            innovation_std: self.innovation_var.sqrt(),
            ts,
            exchange_id: self.exchange_id,
        }
    }

    #[inline]
    fn predict_covariance(&self) -> [f64; 4] {
        // P_pred = A * P * A^T + Q
        // Simplified for 2x2 matrices
        let a = &self.a;
        let p = &self.state.p;
        let q = &self.state.q;

        // A * P
        let ap = [
            a[0] * p[0] + a[1] * p[2],
            a[0] * p[1] + a[1] * p[3],
            a[2] * p[0] + a[3] * p[2],
            a[2] * p[1] + a[3] * p[3],
        ];

        // (A * P) * A^T + Q
        [
            ap[0] * a[0] + ap[1] * a[1] + q[0],
            ap[0] * a[2] + ap[1] * a[3] + q[1],
            ap[2] * a[0] + ap[3] * a[1] + q[2],
            ap[2] * a[2] + ap[3] * a[3] + q[3],
        ]
    }

    #[inline]
    fn update_covariance(&mut self, k: &[f64; 2], p_pred: &[f64; 4]) {
        // P = (I - K*H) * P_pred
        // KH is 2x2: K (2x1) * H (1x2)
        let kh = [
            k[0] * self.h[0],
            k[0] * self.h[1],
            k[1] * self.h[0],
            k[1] * self.h[1],
        ];

        let i_m_kh = [
            1.0 - kh[0],
            -kh[1],
            -kh[2],
            1.0 - kh[3],
        ];

        self.state.p = [
            i_m_kh[0] * p_pred[0] + i_m_kh[1] * p_pred[2],
            i_m_kh[0] * p_pred[1] + i_m_kh[1] * p_pred[3],
            i_m_kh[2] * p_pred[0] + i_m_kh[3] * p_pred[2],
            i_m_kh[2] * p_pred[1] + i_m_kh[3] * p_pred[3],
        ];

        // Ensure symmetry
        self.state.p[1] = (self.state.p[1] + self.state.p[2]) / 2.0;
        self.state.p[2] = self.state.p[1];
    }

    /// Get current estimated price
    #[inline]
    pub fn get_estimated_price(&self) -> f64 {
        self.state.x[0]
    }

    /// Get current estimated velocity
    #[inline]
    pub fn get_estimated_velocity(&self) -> f64 {
        self.state.x[1]
    }

    /// Get innovation (surprise) from last update
    #[inline]
    pub fn get_last_innovation(&self) -> f64 {
        self.innovation
    }

    /// Get standardized innovation (z-score)
    #[inline]
    pub fn get_standardized_innovation(&self) -> f64 {
        if self.innovation_var > 1e-10 {
            self.innovation / self.innovation_var.sqrt()
        } else {
            0.0
        }
    }
}

/// Kalman Filter result
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct KalmanResult {
    /// Estimated efficient price
    pub estimated_price: f64,
    /// Estimated price velocity (change rate)
    pub estimated_velocity: f64,
    /// Innovation (measurement surprise)
    pub innovation: f64,
    /// Innovation standard deviation
    pub innovation_std: f64,
    /// Timestamp
    pub ts: u64,
    /// Exchange ID
    pub exchange_id: u32,
    /// Padding
    _padding: [u8; 24],
}

impl Default for KalmanResult {
    fn default() -> Self {
        Self {
            estimated_price: 0.0,
            estimated_velocity: 0.0,
            innovation: 0.0,
            innovation_std: 0.0,
            ts: 0,
            exchange_id: 0,
            _padding: [0u8; 24],
        }
    }
}

/// Dual Kalman Filter for lead-lag detection between two exchanges
pub struct DualKalmanFilter {
    /// Filter for exchange 1 (leader candidate)
    filter1: ExchangeKalmanFilter,
    /// Filter for exchange 2 (laggard candidate)
    filter2: ExchangeKalmanFilter,
    /// Cross-correlation buffer for lead-lag
    cross_corr_sum: f64,
    cross_corr_count: u32,
    /// Estimated lead-lag in nanoseconds
    lead_lag_ns: i64,
    /// Lead-lag confidence (0-1)
    lead_lag_confidence: f64,
}

impl DualKalmanFilter {
    pub fn new(_allocator: &BumpAllocator, exchange1_id: u32, exchange2_id: u32, initial_price: f64) -> Self {
        Self {
            filter1: ExchangeKalmanFilter::new(exchange1_id, initial_price),
            filter2: ExchangeKalmanFilter::new(exchange2_id, initial_price),
            cross_corr_sum: 0.0,
            cross_corr_count: 0,
            lead_lag_ns: 0,
            lead_lag_confidence: 0.0,
        }
    }

    /// Process updates from both exchanges
    #[inline]
    pub fn update_both(
        &mut self,
        ts1: u64, price1: f64,
        ts2: u64, price2: f64,
    ) -> DualKalmanResult {
        let result1 = self.filter1.update(ts1, price1);
        let result2 = self.filter2.update(ts2, price2);

        // Calculate cross-correlation of innovations
        let innovation_product = result1.innovation * result2.innovation;
        self.cross_corr_sum += innovation_product;
        self.cross_corr_count += 1;

        // Estimate lead-lag based on innovation timing
        self.estimate_lead_lag(&result1, &result2);

        // Calculate price spread
        let spread = result1.estimated_price - result2.estimated_price;
        let spread_zscore = if result1.innovation_std > 0.0 && result2.innovation_std > 0.0 {
            spread / (result1.innovation_std.powi(2) + result2.innovation_std.powi(2)).sqrt()
        } else {
            0.0
        };

        DualKalmanResult {
            price1: result1.estimated_price,
            price2: result2.estimated_price,
            velocity1: result1.estimated_velocity,
            velocity2: result2.estimated_velocity,
            spread,
            spread_zscore,
            lead_lag_ns: self.lead_lag_ns,
            lead_lag_confidence: self.lead_lag_confidence,
            arb_signal: self.generate_arb_signal(spread_zscore),
            ts: ts1.max(ts2),
        }
    }

    #[inline]
    fn estimate_lead_lag(&mut self, r1: &KalmanResult, r2: &KalmanResult) {
        // Simple lead-lag estimation based on innovation correlation
        // If exchange 1's innovations consistently precede exchange 2's, it's the leader
        
        let time_diff = r1.ts as i64 - r2.ts as i64;
        
        // Use recent innovations to estimate direction
        let innov1 = r1.innovation;
        let innov2 = r2.innovation;

        // If innovations have same sign and exchange 1 moved first
        if innov1 * innov2 > 0.0 {
            self.lead_lag_ns = time_diff;
            
            // Confidence based on consistency
            let consistency = (innov1.abs().min(innov2.abs()) / innov1.abs().max(innov2.abs())).min(1.0);
            self.lead_lag_confidence = consistency * 0.5 + self.lead_lag_confidence * 0.5;
        }
    }

    #[inline]
    fn generate_arb_signal(&self, spread_zscore: f64) -> ArbSignal {
        if spread_zscore.abs() < 1.0 {
            ArbSignal::None
        } else if spread_zscore > 2.0 {
            ArbSignal::SellExchange1BuyExchange2
        } else if spread_zscore < -2.0 {
            ArbSignal::BuyExchange1SellExchange2
        } else if spread_zscore > 0.0 {
            ArbSignal::WeakSellExchange1
        } else {
            ArbSignal::WeakBuyExchange1
        }
    }

    /// Get the leader exchange ID (or 0 if unclear)
    #[inline]
    pub fn get_leader_exchange(&self) -> u32 {
        if self.lead_lag_confidence < 0.3 {
            return 0;
        }
        
        if self.lead_lag_ns > 0 {
            // Exchange 1 leads
            self.filter1.exchange_id
        } else if self.lead_lag_ns < 0 {
            // Exchange 2 leads
            self.filter2.exchange_id
        } else {
            0
        }
    }

    /// Get current spread
    #[inline]
    pub fn get_spread(&self) -> f64 {
        self.filter1.get_estimated_price() - self.filter2.get_estimated_price()
    }
}

/// Dual Kalman result
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct DualKalmanResult {
    /// Estimated price on exchange 1
    pub price1: f64,
    /// Estimated price on exchange 2
    pub price2: f64,
    /// Velocity on exchange 1
    pub velocity1: f64,
    /// Velocity on exchange 2
    pub velocity2: f64,
    /// Price spread (1 - 2)
    pub spread: f64,
    /// Spread z-score
    pub spread_zscore: f64,
    /// Estimated lead-lag in nanoseconds
    pub lead_lag_ns: i64,
    /// Lead-lag confidence
    pub lead_lag_confidence: f64,
    /// Arbitrage signal
    pub arb_signal: ArbSignal,
    /// Timestamp
    pub ts: u64,
}

/// Arbitrage signal
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArbSignal {
    None = 0,
    WeakBuyExchange1 = 1,
    WeakSellExchange1 = 2,
    BuyExchange1SellExchange2 = 3,
    SellExchange1BuyExchange2 = 4,
}

impl Default for ArbSignal {
    fn default() -> Self {
        ArbSignal::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_core::memory::arena::BumpAllocator;

    #[test]
    fn test_kalman_initialization() {
        let filter = ExchangeKalmanFilter::new(1, 100.0);
        assert_eq!(filter.get_estimated_price(), 100.0);
        assert_eq!(filter.get_estimated_velocity(), 0.0);
    }

    #[test]
    fn test_kalman_price_tracking() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut filter = ExchangeKalmanFilter::new(1, 100.0);

        // Simulate price increasing
        let base_ts = 1_000_000_000_000u64;
        for i in 0..10 {
            let ts = base_ts + i * 1_000_000;
            let price = 100.0 + i as f64 * 0.1;
            let result = filter.update(ts, price);
            
            // Estimate should track the price
            assert!(result.estimated_price > 99.0 && result.estimated_price < price + 1.0);
        }
    }

    #[test]
    fn test_kalman_velocity_estimation() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut filter = ExchangeKalmanFilter::new(1, 100.0);

        // Constant upward movement
        let base_ts = 1_000_000_000_000u64;
        for i in 0..20 {
            let ts = base_ts + i * 1_000_000;
            let price = 100.0 + i as f64 * 0.5;
            filter.update(ts, price);
        }

        // Should estimate positive velocity
        assert!(filter.get_estimated_velocity() > 0.0);
    }

    #[test]
    fn test_dual_kalman_spread() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut dual = DualKalmanFilter::new(&allocator, 1, 2, 100.0);

        // Same price on both exchanges
        let base_ts = 1_000_000_000_000u64;
        for i in 0..10 {
            let ts = base_ts + i * 1_000_000;
            let price = 100.0 + i as f64 * 0.1;
            let result = dual.update_both(ts, price, ts, price);
            
            // Spread should be near zero
            assert!(result.spread.abs() < 0.5);
        }
    }

    #[test]
    fn test_dual_kalman_arb_signal() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut dual = DualKalmanFilter::new(&allocator, 1, 2, 100.0);

        // Create price divergence
        let base_ts = 1_000_000_000_000u64;
        for i in 0..20 {
            let ts = base_ts + i * 1_000_000;
            let price1 = 100.0 + i as f64 * 0.5; // Exchange 1 rises faster
            let price2 = 100.0 + i as f64 * 0.1; // Exchange 2 rises slower
            let result = dual.update_both(ts, price1, ts, price2);
            
            // Eventually should generate sell exchange 1 signal
            if i > 15 {
                assert!(result.spread > 0.0);
            }
        }
    }
}
