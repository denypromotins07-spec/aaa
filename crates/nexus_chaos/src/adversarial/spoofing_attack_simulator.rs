// STAGE 25: CHAPTER 3 - SPOOFING ATTACK SIMULATOR
// Uses Stage 17 Hawkes process logic against the bot
// Generates hyper-realistic phantom liquidity to test detection

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Spoofing attack types
#[derive(Debug, Clone, PartialEq)]
pub enum SpoofType {
    Layering,           // Multiple orders at different price levels
    MomentumIgnition,   // Fake orders to trigger momentum algorithms
    QuoteStuffing,      // Rapid order placement/cancellation
    WashTrading,        // Self-matching to create fake volume
}

/// Phantom order representation
#[derive(Debug, Clone)]
pub struct PhantomOrder {
    pub order_id: u64,
    pub symbol: String,
    pub side: Side,
    pub price: u64,
    pub size: u64,
    pub timestamp_ns: u64,
    pub is_cancelled: bool,
    pub lifetime_ns: u64,
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Bid,
    Ask,
}

/// Spoofing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpoofConfig {
    pub spoof_type: SpoofType,
    pub order_rate_per_second: u64,
    pub cancel_rate: f64,
    pub price_deviation_bps: u64,
    pub target_symbols: Vec<String>,
}

/// Spoofing state
pub struct SpoofState {
    pub orders_generated: AtomicU64,
    pub orders_cancelled: AtomicU64,
    pub active_phantom_orders: AtomicU64,
    pub detection_evasion_score: AtomicU64, // 0-10000 scale
}

impl Default for SpoofState {
    fn default() -> Self {
        Self {
            orders_generated: AtomicU64::new(0),
            orders_cancelled: AtomicU64::new(0),
            active_phantom_orders: AtomicU64::new(0),
            detection_evasion_score: AtomicU64::new(5000),
        }
    }
}

/// Spoofing Attack Simulator
pub struct SpoofingAttackSimulator {
    state: std::sync::Arc<SpoofState>,
    config: SpoofConfig,
    chaos_mode_flag: AtomicBool,
    phantom_orders: std::sync::Mutex<Vec<PhantomOrder>>,
    rng_seed: u64,
    next_order_id: AtomicU64,
}

impl SpoofingAttackSimulator {
    pub fn new(config: SpoofConfig, rng_seed: u64) -> Self {
        Self {
            state: std::sync::Arc::new(SpoofState::default()),
            config,
            chaos_mode_flag: AtomicBool::new(false),
            phantom_orders: std::sync::Mutex::new(Vec::new()),
            rng_seed,
            next_order_id: AtomicU64::new(1),
        }
    }

    /// Activate chaos mode
    pub fn activate_chaos_mode(&self) {
        self.chaos_mode_flag.store(true, Ordering::SeqCst);
    }

    /// Deactivate chaos mode
    pub fn deactivate_chaos_mode(&self) {
        self.chaos_mode_flag.store(false, Ordering::SeqCst);
    }

    /// Check if chaos mode is active
    pub fn is_chaos_mode_active(&self) -> bool {
        self.chaos_mode_flag.load(Ordering::SeqCst)
    }

    /// Generate a batch of phantom orders based on spoof type
    pub fn generate_phantom_orders(
        &self,
        base_price: u64,
        count: usize,
    ) -> Result<Vec<PhantomOrder>, SpoofError> {
        if !self.chaos_mode_flag.load(Ordering::SeqCst) {
            return Err(SpoofError::ChaosModeNotActive);
        }

        let mut orders = Vec::new();
        let now = Instant::now();
        let now_ns = now.duration_since(Instant::EPOCH).as_nanos() as u64;

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rng_seed + now_ns);

        for i in 0..count {
            let order_id = self.next_order_id.fetch_add(1, Ordering::Relaxed);
            
            // Determine side based on spoof type
            let side = match self.config.spoof_type {
                SpoofType::Layering => {
                    if i % 2 == 0 { Side::Bid } else { Side::Ask }
                }
                SpoofType::MomentumIgnition => Side::Bid,
                SpoofType::QuoteStuffing => {
                    if rng.gen_bool(0.5) { Side::Bid } else { Side::Ask }
                }
                SpoofType::WashTrading => Side::Bid,
            };

            // Calculate price with deviation
            let deviation_bps = rng.gen_range(0..self.config.price_deviation_bps);
            let price = if side == Side::Bid {
                base_price.saturating_sub(base_price * deviation_bps / 10000)
            } else {
                base_price + base_price * deviation_bps / 10000
            };

            // Generate size
            let size = rng.gen_range(100..10000);

            // Calculate lifetime (how long before cancellation)
            let lifetime_ns = match self.config.spoof_type {
                SpoofType::Layering => rng.gen_range(100_000_000..1_000_000_000), // 100ms-1s
                SpoofType::MomentumIgnition => rng.gen_range(10_000_000..100_000_000), // 10-100ms
                SpoofType::QuoteStuffing => rng.gen_range(1_000_000..10_000_000), // 1-10ms
                SpoofType::WashTrading => rng.gen_range(50_000_000..500_000_000), // 50-500ms
            };

            let order = PhantomOrder {
                order_id,
                symbol: "BTC-PERP".to_string(),
                side,
                price,
                size,
                timestamp_ns: now_ns + (i as u64 * 100_000), // Stagger by 100μs
                is_cancelled: false,
                lifetime_ns,
            };

            orders.push(order);
        }

        // Add to phantom orders list
        let mut phantom_list = self.phantom_orders.lock().unwrap();
        phantom_list.extend(orders.clone());
        
        self.state.orders_generated.fetch_add(count as u64, Ordering::Relaxed);
        self.state.active_phantom_orders.fetch_add(count as u64, Ordering::Relaxed);

        Ok(orders)
    }

    /// Cancel expired phantom orders
    pub fn cancel_expired_orders(&self) -> Vec<u64> {
        let now = Instant::now();
        let now_ns = now.duration_since(Instant::EPOCH).as_nanos() as u64;

        let mut phantom_list = self.phantom_orders.lock().unwrap();
        let mut cancelled_ids = Vec::new();

        for order in phantom_list.iter_mut() {
            if !order.is_cancelled && now_ns >= order.timestamp_ns + order.lifetime_ns {
                order.is_cancelled = true;
                cancelled_ids.push(order.order_id);
                self.state.orders_cancelled.fetch_add(1, Ordering::Relaxed);
                self.state.active_phantom_orders.fetch_sub(1, Ordering::Relaxed);
            }
        }

        cancelled_ids
    }

    /// Get all active phantom orders
    pub fn get_active_orders(&self) -> Vec<PhantomOrder> {
        let phantom_list = self.phantom_orders.lock().unwrap();
        phantom_list.iter()
            .filter(|o| !o.is_cancelled)
            .cloned()
            .collect()
    }

    /// Update detection evasion score based on how well spoofing avoids detection
    pub fn update_evasion_score(&self, detected: bool) {
        let current = self.state.detection_evasion_score.load(Ordering::Relaxed);
        let new_score = if detected {
            current.saturating_sub(100)
        } else {
            (current + 50).min(10000)
        };
        self.state.detection_evasion_score.store(new_score, Ordering::Relaxed);
    }

    /// Get spoofing statistics
    pub fn get_stats(&self) -> SpoofStats {
        SpoofStats {
            orders_generated: self.state.orders_generated.load(Ordering::Relaxed),
            orders_cancelled: self.state.orders_cancelled.load(Ordering::Relaxed),
            active_phantom_orders: self.state.active_phantom_orders.load(Ordering::Relaxed),
            detection_evasion_score: self.state.detection_evasion_score.load(Ordering::Relaxed),
            chaos_mode: self.chaos_mode_flag.load(Ordering::SeqCst),
        }
    }

    /// Reset state
    pub fn reset(&self) {
        self.phantom_orders.lock().unwrap().clear();
        self.state.orders_generated.store(0, Ordering::Relaxed);
        self.state.orders_cancelled.store(0, Ordering::Relaxed);
        self.state.active_phantom_orders.store(0, Ordering::Relaxed);
        self.state.detection_evasion_score.store(5000, Ordering::Relaxed);
    }
}

/// Spoofing statistics
#[derive(Debug, Clone)]
pub struct SpoofStats {
    pub orders_generated: u64,
    pub orders_cancelled: u64,
    pub active_phantom_orders: u64,
    pub detection_evasion_score: u64,
    pub chaos_mode: bool,
}

/// Spoof errors
#[derive(Debug, Clone, PartialEq)]
pub enum SpoofError {
    ChaosModeNotActive,
    InvalidConfiguration,
    OrderGenerationFailed,
}

impl std::fmt::Display for SpoofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpoofError::ChaosModeNotActive => write!(f, "Chaos mode not active"),
            SpoofError::InvalidConfiguration => write!(f, "Invalid configuration"),
            SpoofError::OrderGenerationFailed => write!(f, "Order generation failed"),
        }
    }
}

impl std::error::Error for SpoofError {}

/// Builder for spoof configurations
pub struct SpoofConfigBuilder {
    spoof_type: SpoofType,
    order_rate_per_second: u64,
    cancel_rate: f64,
    price_deviation_bps: u64,
    target_symbols: Vec<String>,
}

impl SpoofConfigBuilder {
    pub fn new() -> Self {
        Self {
            spoof_type: SpoofType::Layering,
            order_rate_per_second: 1000,
            cancel_rate: 0.95,
            price_deviation_bps: 50,
            target_symbols: vec!["BTC-PERP".to_string()],
        }
    }

    pub fn spoof_type(mut self, spoof_type: SpoofType) -> Self {
        self.spoof_type = spoof_type;
        self
    }

    pub fn order_rate(mut self, rate: u64) -> Self {
        self.order_rate_per_second = rate;
        self
    }

    pub fn cancel_rate(mut self, rate: f64) -> Self {
        self.cancel_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn price_deviation_bps(mut self, bps: u64) -> Self {
        self.price_deviation_bps = bps;
        self
    }

    pub fn target_symbol(mut self, symbol: &str) -> Self {
        self.target_symbols.push(symbol.to_string());
        self
    }

    pub fn build(self) -> SpoofConfig {
        SpoofConfig {
            spoof_type: self.spoof_type,
            order_rate_per_second: self.order_rate_per_second,
            cancel_rate: self.cancel_rate,
            price_deviation_bps: self.price_deviation_bps,
            target_symbols: self.target_symbols,
        }
    }
}

impl Default for SpoofConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spoof_without_chaos_mode() {
        let config = SpoofConfigBuilder::new()
            .spoof_type(SpoofType::Layering)
            .build();

        let simulator = SpoofingAttackSimulator::new(config, 42);

        let result = simulator.generate_phantom_orders(50000_000_000, 10);
        assert!(matches!(result, Err(SpoofError::ChaosModeNotActive)));
    }

    #[test]
    fn test_generate_layering_attack() {
        let config = SpoofConfigBuilder::new()
            .spoof_type(SpoofType::Layering)
            .price_deviation_bps(100)
            .build();

        let simulator = SpoofingAttackSimulator::new(config, 42);
        simulator.activate_chaos_mode();

        let orders = simulator.generate_phantom_orders(50000_000_000, 10);
        assert!(orders.is_ok());

        let orders = orders.unwrap();
        assert_eq!(orders.len(), 10);

        let stats = simulator.get_stats();
        assert_eq!(stats.orders_generated, 10);
        assert_eq!(stats.active_phantom_orders, 10);
    }

    #[test]
    fn test_order_expiration() {
        let config = SpoofConfigBuilder::new()
            .spoof_type(SpoofType::QuoteStuffing)
            .build();

        let simulator = SpoofingAttackSimulator::new(config, 42);
        simulator.activate_chaos_mode();

        let _ = simulator.generate_phantom_orders(50000_000_000, 5);
        
        // Wait for orders to expire (QuoteStuffing has 1-10ms lifetime)
        std::thread::sleep(Duration::from_millis(15));
        
        let cancelled = simulator.cancel_expired_orders();
        assert!(!cancelled.is_empty());

        let stats = simulator.get_stats();
        assert!(stats.orders_cancelled > 0);
    }
}
