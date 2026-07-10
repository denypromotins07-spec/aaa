//! Proxy Hedge Fallback
//! 
//! Uses highly liquid proxy assets to hedge exposed legs when
//! the original paired asset becomes unavailable.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Maximum number of proxy candidates supported
const MAX_PROXIES: usize = 8;

/// Information about a proxy hedge candidate
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ProxyCandidate {
    /// Asset identifier (e.g., symbol hash)
    pub asset_id: u64,
    /// Correlation with target asset (-1 to 1)
    pub correlation: f64,
    /// Liquidity score (higher = more liquid)
    pub liquidity_score: f64,
    /// Hedge ratio relative to target
    pub hedge_ratio: f64,
    /// Whether currently available for trading
    pub available: bool,
}

impl Default for ProxyCandidate {
    #[inline]
    fn default() -> Self {
        Self {
            asset_id: 0,
            correlation: 0.0,
            liquidity_score: 0.0,
            hedge_ratio: 1.0,
            available: false,
        }
    }
}

/// Result of proxy hedge selection
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ProxyHedgeResult {
    /// Whether a suitable proxy was found
    pub found: bool,
    /// Selected proxy asset ID
    pub proxy_asset_id: u64,
    /// Recommended hedge quantity
    pub hedge_qty: i64,
    /// Expected slippage in basis points
    pub expected_slippage_bps: u16,
    /// Confidence score (0-1)
    pub confidence: f64,
}

impl Default for ProxyHedgeResult {
    #[inline]
    fn default() -> Self {
        Self {
            found: false,
            proxy_asset_id: 0,
            hedge_qty: 0,
            expected_slippage_bps: 0,
            confidence: 0.0,
        }
    }
}

/// Configuration for proxy hedging
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ProxyHedgeConfig {
    /// Minimum correlation threshold for proxy selection
    pub min_correlation: f64,
    /// Minimum liquidity score
    pub min_liquidity_score: f64,
    /// Maximum acceptable slippage (bps)
    pub max_slippage_bps: u16,
    /// Hedge ratio buffer (multiply by this for safety margin)
    pub hedge_buffer: f64,
}

impl ProxyHedgeConfig {
    #[inline]
    pub const fn standard() -> Self {
        Self {
            min_correlation: 0.85,
            min_liquidity_score: 0.5,
            max_slippage_bps: 100,
            hedge_buffer: 1.05, // 5% buffer
        }
    }

    #[inline]
    pub const fn conservative() -> Self {
        Self {
            min_correlation: 0.95,
            min_liquidity_score: 0.8,
            max_slippage_bps: 50,
            hedge_buffer: 1.10,
        }
    }
}

impl Default for ProxyHedgeConfig {
    #[inline]
    fn default() -> Self {
        Self::standard()
    }
}

/// Proxy Hedge Fallback Engine
/// 
/// Selects and manages proxy hedges when primary leg fails.
pub struct ProxyHedgeFallback {
    /// Available proxy candidates
    proxies: [ProxyCandidate; MAX_PROXIES],
    /// Number of valid proxies
    n_proxies: usize,
    /// Configuration
    config: ProxyHedgeConfig,
    /// Whether proxy hedging is enabled
    enabled: AtomicBool,
    /// Last successful hedge timestamp
    last_hedge_ts: AtomicU64,
    /// Total hedges executed
    hedge_count: AtomicU64,
}

impl ProxyHedgeFallback {
    /// Create a new proxy hedge fallback engine
    #[inline]
    pub fn new(config: ProxyHedgeConfig) -> Self {
        Self {
            proxies: [ProxyCandidate::default(); MAX_PROXIES],
            n_proxies: 0,
            config,
            enabled: AtomicBool::new(true),
            last_hedge_ts: AtomicU64::new(0),
            hedge_count: AtomicU64::new(0),
        }
    }

    /// Register a proxy candidate
    /// 
    /// # Arguments
    /// * `asset_id` - Unique identifier for the proxy asset
    /// * `correlation` - Historical correlation with target
    /// * `liquidity_score` - Relative liquidity measure
    /// * `hedge_ratio` - Typical hedge ratio
    /// 
    /// Returns true if successfully registered, false if full
    #[inline]
    pub fn register_proxy(
        &mut self,
        asset_id: u64,
        correlation: f64,
        liquidity_score: f64,
        hedge_ratio: f64,
    ) -> bool {
        if self.n_proxies >= MAX_PROXIES {
            return false;
        }

        if !correlation.is_finite() || !liquidity_score.is_finite() || !hedge_ratio.is_finite() {
            return false;
        }

        self.proxies[self.n_proxies] = ProxyCandidate {
            asset_id,
            correlation: correlation.clamp(-1.0, 1.0),
            liquidity_score: liquidity_score.max(0.0),
            hedge_ratio,
            available: true,
        };

        self.n_proxies += 1;
        true
    }

    /// Update availability of a proxy
    #[inline]
    pub fn set_proxy_availability(&mut self, asset_id: u64, available: bool) {
        for i in 0..self.n_proxies {
            if self.proxies[i].asset_id == asset_id {
                self.proxies[i].available = available;
                break;
            }
        }
    }

    /// Select the best proxy for hedging an exposed position
    /// 
    /// # Arguments
    /// * `exposed_qty` - Quantity of the exposed leg (positive = long, negative = short)
    /// * `target_price` - Price of the original target asset
    /// * `proxy_prices` - Slice of (asset_id, price) pairs for proxies
    #[inline]
    pub fn select_best_proxy(
        &self,
        exposed_qty: i64,
        target_price: f64,
        proxy_prices: &[(u64, f64)],
    ) -> ProxyHedgeResult {
        if !self.enabled.load(Ordering::Relaxed) || self.n_proxies == 0 {
            return ProxyHedgeResult::default();
        }

        if !target_price.is_finite() || target_price <= 0.0 {
            return ProxyHedgeResult::default();
        }

        let mut best_idx: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;

        // Score each available proxy
        for i in 0..self.n_proxies {
            let proxy = &self.proxies[i];

            if !proxy.available {
                continue;
            }

            // Check minimum thresholds
            if proxy.correlation < self.config.min_correlation {
                continue;
            }
            if proxy.liquidity_score < self.config.min_liquidity_score {
                continue;
            }

            // Get current price for this proxy
            let proxy_price = match proxy_prices.iter().find(|(id, _)| *id == proxy.asset_id) {
                Some((_, price)) => *price,
                None => continue,
            };

            if !proxy_price.is_finite() || proxy_price <= 0.0 {
                continue;
            }

            // Score = correlation * liquidity * |correlation| (prefer same-direction correlation)
            let direction_score = if (exposed_qty > 0 && proxy.correlation > 0) ||
                                     (exposed_qty < 0 && proxy.correlation < 0) {
                1.0 // Need opposite direction hedge
            } else {
                -1.0
            };

            let score = proxy.correlation.abs() * proxy.liquidity_score * direction_score;

            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }

        let idx = match best_idx {
            Some(i) => i,
            None => return ProxyHedgeResult::default(),
        };

        let proxy = self.proxies[idx];

        // Calculate hedge quantity
        // hedge_qty = -exposed_qty * hedge_ratio * (target_price / proxy_price) * buffer
        let proxy_price = match proxy_prices.iter().find(|(id, _)| *id == proxy.asset_id) {
            Some((_, price)) => *price,
            None => return ProxyHedgeResult::default(),
        };

        let base_hedge_qty = -(exposed_qty as f64) 
            * proxy.hedge_ratio 
            * (target_price / proxy_price)
            * self.config.hedge_buffer;

        let hedge_qty = base_hedge_qty.round() as i64;

        // Estimate slippage based on liquidity score and size
        let estimated_slippage_bps = ((1.0 - proxy.liquidity_score) * 200.0) as u16;

        // Confidence based on correlation strength
        let confidence = proxy.correlation.abs();

        ProxyHedgeResult {
            found: true,
            proxy_asset_id: proxy.asset_id,
            hedge_qty,
            expected_slippage_bps: estimated_slippage_bps.min(self.config.max_slippage_bps),
            confidence,
        }
    }

    /// Record a successful hedge execution
    #[inline]
    pub fn record_hedge(&self, timestamp_us: u64) {
        self.last_hedge_ts.store(timestamp_us, Ordering::Relaxed);
        self.hedge_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the number of registered proxies
    #[inline]
    pub fn proxy_count(&self) -> usize {
        self.n_proxies
    }

    /// Enable or disable proxy hedging
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if proxy hedging is enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Get the timestamp of the last hedge
    #[inline]
    pub fn last_hedge_timestamp(&self) -> u64 {
        self.last_hedge_ts.load(Ordering::Relaxed)
    }

    /// Get the total number of hedges executed
    #[inline]
    pub fn hedge_count(&self) -> u64 {
        self.hedge_count.load(Ordering::Relaxed)
    }

    /// Clear all registered proxies
    #[inline]
    pub fn clear_proxies(&mut self) {
        self.proxies = [ProxyCandidate::default(); MAX_PROXIES];
        self.n_proxies = 0;
    }

    /// Get the best proxy without executing (for preview)
    #[inline]
    pub fn preview_best_proxy(&self) -> Option<&ProxyCandidate> {
        if self.n_proxies == 0 || !self.enabled.load(Ordering::Relaxed) {
            return None;
        }

        let mut best_idx: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;

        for i in 0..self.n_proxies {
            let proxy = &self.proxies[i];
            if !proxy.available {
                continue;
            }

            let score = proxy.correlation.abs() * proxy.liquidity_score;
            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }

        best_idx.map(|i| &self.proxies[i])
    }
}

impl Default for ProxyHedgeFallback {
    #[inline]
    fn default() -> Self {
        Self::new(ProxyHedgeConfig::standard())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_proxies() {
        let mut fallback = ProxyHedgeFallback::new(ProxyHedgeConfig::standard());
        
        assert!(fallback.register_proxy(1, 0.95, 0.8, 1.0));
        assert!(fallback.register_proxy(2, 0.90, 0.7, 0.9));
        assert_eq!(fallback.proxy_count(), 2);
    }

    #[test]
    fn test_max_proxies() {
        let mut fallback = ProxyHedgeFallback::new(ProxyHedgeConfig::standard());
        
        for i in 0..MAX_PROXIES {
            assert!(fallback.register_proxy(i as u64, 0.9, 0.5, 1.0));
        }
        
        // Should fail after max
        assert!(!fallback.register_proxy(999, 0.9, 0.5, 1.0));
    }

    #[test]
    fn test_select_best_proxy() {
        let mut fallback = ProxyHedgeFallback::new(ProxyHedgeConfig::standard());
        
        // Register two proxies - second one is better
        fallback.register_proxy(1, 0.85, 0.5, 1.0); // Lower correlation and liquidity
        fallback.register_proxy(2, 0.95, 0.9, 1.0); // Higher correlation and liquidity
        
        let proxy_prices = vec![(1, 100.0), (2, 100.0)];
        let result = fallback.select_best_proxy(100, 100.0, &proxy_prices);
        
        assert!(result.found);
        assert_eq!(result.proxy_asset_id, 2); // Should select better proxy
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_hedge_direction() {
        let mut fallback = ProxyHedgeFallback::new(ProxyHedgeConfig::standard());
        
        // Positive correlation proxy
        fallback.register_proxy(1, 0.95, 0.8, 1.0);
        
        let proxy_prices = vec![(1, 100.0)];
        
        // Long exposure -> need short hedge (negative qty)
        let result = fallback.select_best_proxy(100, 100.0, &proxy_prices);
        assert!(result.found);
        assert!(result.hedge_qty < 0);
        
        // Short exposure -> need long hedge (positive qty)
        let result = fallback.select_best_proxy(-100, 100.0, &proxy_prices);
        assert!(result.found);
        assert!(result.hedge_qty > 0);
    }

    #[test]
    fn test_unavailable_proxy() {
        let mut fallback = ProxyHedgeFallback::new(ProxyHedgeConfig::standard());
        
        fallback.register_proxy(1, 0.95, 0.8, 1.0);
        fallback.set_proxy_availability(1, false);
        
        let proxy_prices = vec![(1, 100.0)];
        let result = fallback.select_best_proxy(100, 100.0, &proxy_prices);
        
        assert!(!result.found); // No available proxies
    }

    #[test]
    fn test_below_threshold_correlation() {
        let mut fallback = ProxyHedgeFallback::new(ProxyHedgeConfig::standard());
        
        // Below minimum correlation threshold (0.85)
        fallback.register_proxy(1, 0.70, 0.9, 1.0);
        
        let proxy_prices = vec![(1, 100.0)];
        let result = fallback.select_best_proxy(100, 100.0, &proxy_prices);
        
        assert!(!result.found);
    }

    #[test]
    fn test_enable_disable() {
        let mut fallback = ProxyHedgeFallback::new(ProxyHedgeConfig::standard());
        fallback.register_proxy(1, 0.95, 0.8, 1.0);
        
        fallback.set_enabled(false);
        
        let proxy_prices = vec![(1, 100.0)];
        let result = fallback.select_best_proxy(100, 100.0, &proxy_prices);
        
        assert!(!result.found);
        assert!(!fallback.is_enabled());
    }

    #[test]
    fn test_hedge_recording() {
        let fallback = ProxyHedgeFallback::new(ProxyHedgeConfig::standard());
        
        assert_eq!(fallback.hedge_count(), 0);
        
        fallback.record_hedge(1000);
        assert_eq!(fallback.hedge_count(), 1);
        assert_eq!(fallback.last_hedge_timestamp(), 1000);
    }
}
