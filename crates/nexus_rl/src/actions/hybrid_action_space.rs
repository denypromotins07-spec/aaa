//! Hybrid Action Space for RL Trading Agent
//! 
//! Defines a complex, multi-dimensional action space combining discrete and continuous
//! components: order type, price offset, and size fraction.

use std::fmt;

/// Order type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Market = 0,
    Limit = 1,
    PostOnly = 2,
    IOC = 3,
    FOK = 4,
}

impl OrderType {
    /// Convert from u8 with validation
    #[inline]
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(OrderType::Market),
            1 => Some(OrderType::Limit),
            2 => Some(OrderType::PostOnly),
            3 => Some(OrderType::IOC),
            4 => Some(OrderType::FOK),
            _ => None,
        }
    }
    
    /// Check if order type requires price
    #[inline]
    pub const fn requires_price(self) -> bool {
        matches!(self, OrderType::Limit | OrderType::PostOnly)
    }
}

/// Side of the trade
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeSide {
    Buy = 0,
    Sell = 1,
    Hold = 2,
}

impl TradeSide {
    #[inline]
    pub fn from_i32(val: i32) -> Option<Self> {
        match val {
            0 => Some(TradeSide::Buy),
            1 => Some(TradeSide::Sell),
            2 => Some(TradeSide::Hold),
            _ => None,
        }
    }
    
    #[inline]
    pub const fn is_neutral(self) -> bool {
        matches!(self, TradeSide::Hold)
    }
}

/// Hybrid action representation for trading
#[derive(Debug, Clone, Copy)]
pub struct HybridAction {
    /// Trade side (Buy/Sell/Hold)
    pub side: TradeSide,
    /// Order type (Market/Limit/etc.)
    pub order_type: OrderType,
    /// Price offset in ticks from mid-price (for limit orders)
    pub price_offset_ticks: i16,
    /// Size as fraction of max position (0.0 to 1.0)
    pub size_fraction: f32,
    /// Asset index in the universe
    pub asset_idx: u8,
}

impl HybridAction {
    /// Create a new hybrid action
    #[inline]
    pub const fn new(
        side: TradeSide,
        order_type: OrderType,
        price_offset_ticks: i16,
        size_fraction: f32,
        asset_idx: u8,
    ) -> Self {
        Self {
            side,
            order_type,
            price_offset_ticks,
            size_fraction,
            asset_idx,
        }
    }
    
    /// Create a HOLD action (no-op)
    #[inline]
    pub const fn hold() -> Self {
        Self {
            side: TradeSide::Hold,
            order_type: OrderType::Market,
            price_offset_ticks: 0,
            size_fraction: 0.0,
            asset_idx: 0,
        }
    }
    
    /// Validate action parameters
    #[inline]
    pub fn is_valid(&self) -> bool {
        // Size must be in valid range
        if self.size_fraction < 0.0 || self.size_fraction > 1.0 {
            return false;
        }
        
        // Limit orders must have non-negative price offset
        if self.order_type.requires_price() && self.price_offset_ticks < 0 {
            return false;
        }
        
        // Hold actions don't need other validation
        if self.side.is_neutral() {
            return true;
        }
        
        true
    }
    
    /// Get actual limit price given mid-price and tick size
    #[inline]
    pub fn get_limit_price(&self, mid_price: f64, tick_size: f64) -> f64 {
        if !self.order_type.requires_price() {
            return mid_price;
        }
        
        let offset = self.price_offset_ticks as f64 * tick_size;
        
        match self.side {
            TradeSide::Buy => mid_price - offset,
            TradeSide::Sell => mid_price + offset,
            TradeSide::Hold => mid_price,
        }
    }
    
    /// Get actual order size given max size
    #[inline]
    pub fn get_order_size(&self, max_size: f64) -> f64 {
        if self.side.is_neutral() {
            return 0.0;
        }
        max_size * self.size_fraction as f64
    }
    
    /// Encode action into a compact byte representation for transmission
    #[inline]
    pub fn encode(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        
        buf[0] = self.side as u8;
        buf[1] = self.order_type as u8;
        buf[2..4].copy_from_slice(&self.price_offset_ticks.to_ne_bytes());
        buf[4..8].copy_from_slice(&self.size_fraction.to_ne_bytes());
        buf[8] = self.asset_idx;
        // Remaining bytes reserved
        
        buf
    }
    
    /// Decode action from byte representation
    #[inline]
    pub fn decode(buf: &[u8; 16]) -> Option<Self> {
        let side = TradeSide::from_i32(buf[0] as i32)?;
        let order_type = OrderType::from_u8(buf[1])?;
        let price_offset_ticks = i16::from_ne_bytes([buf[2], buf[3]]);
        let size_fraction = f32::from_ne_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let asset_idx = buf[8];
        
        Some(Self {
            side,
            order_type,
            price_offset_ticks,
            size_fraction,
            asset_idx,
        })
    }
}

impl Default for HybridAction {
    fn default() -> Self {
        Self::hold()
    }
}

impl fmt::Display for HybridAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.side.is_neutral() {
            write!(f, "HOLD")
        } else {
            write!(
                f,
                "{} {} @ offset={}ticks, size={:.2}%",
                match self.side {
                    TradeSide::Buy => "BUY",
                    TradeSide::Sell => "SELL",
                    TradeSide::Hold => "HOLD",
                },
                match self.order_type {
                    OrderType::Market => "MKT",
                    OrderType::Limit => "LMT",
                    OrderType::PostOnly => "POST",
                    OrderType::IOC => "IOC",
                    OrderType::FOK => "FOK",
                },
                self.price_offset_ticks,
                self.size_fraction * 100.0
            )
        }
    }
}

/// Action space configuration
#[derive(Debug, Clone)]
pub struct ActionSpaceConfig {
    /// Number of assets in the universe
    pub num_assets: usize,
    /// Maximum price offset in ticks
    pub max_price_offset_ticks: i16,
    /// Minimum size fraction step
    pub size_step: f32,
    /// Allowed order types
    pub allowed_order_types: Vec<OrderType>,
}

impl Default for ActionSpaceConfig {
    fn default() -> Self {
        Self {
            num_assets: 10,
            max_price_offset_ticks: 100,
            size_step: 0.01,
            allowed_order_types: vec![
                OrderType::Market,
                OrderType::Limit,
                OrderType::PostOnly,
            ],
        }
    }
}

impl ActionSpaceConfig {
    /// Calculate total discrete action space size
    pub fn discrete_action_count(&self) -> usize {
        // 3 sides * num_order_types * (max_offset + 1) * (1/size_step + 1) * num_assets
        let sides = 3; // Buy, Sell, Hold
        let order_types = self.allowed_order_types.len();
        let offsets = self.max_price_offset_ticks as usize + 1;
        let sizes = (1.0 / self.size_step) as usize + 1;
        
        sides * order_types * offsets * sizes * self.num_assets
    }
    
    /// Sample a random action (for exploration)
    pub fn sample(&self, rng: &mut impl rand::Rng) -> HybridAction {
        use rand::Rng;
        
        let side = match rng.gen_range(0..3) {
            0 => TradeSide::Buy,
            1 => TradeSide::Sell,
            _ => TradeSide::Hold,
        };
        
        let order_type = self.allowed_order_types
            .get(rng.gen_range(0..self.allowed_order_types.len()))
            .copied()
            .unwrap_or(OrderType::Market);
        
        let price_offset = if order_type.requires_price() {
            rng.gen_range(0..=self.max_price_offset_ticks)
        } else {
            0
        };
        
        let size_fraction = if side.is_neutral() {
            0.0
        } else {
            (rng.gen::<f32>() * (1.0 / self.size_step)).floor() * self.size_step
        };
        
        let asset_idx = rng.gen_range(0..self.num_assets as u8);
        
        HybridAction::new(side, order_type, price_offset, size_fraction, asset_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hybrid_action_creation() {
        let action = HybridAction::new(
            TradeSide::Buy,
            OrderType::Limit,
            5,
            0.5,
            0,
        );
        
        assert_eq!(action.side, TradeSide::Buy);
        assert_eq!(action.order_type, OrderType::Limit);
        assert_eq!(action.price_offset_ticks, 5);
        assert!(action.is_valid());
    }
    
    #[test]
    fn test_hold_action() {
        let action = HybridAction::hold();
        assert!(action.side.is_neutral());
        assert!(action.is_valid());
    }
    
    #[test]
    fn test_encode_decode() {
        let original = HybridAction::new(
            TradeSide::Sell,
            OrderType::PostOnly,
            10,
            0.25,
            3,
        );
        
        let encoded = original.encode();
        let decoded = HybridAction::decode(&encoded).unwrap();
        
        assert_eq!(original.side, decoded.side);
        assert_eq!(original.order_type, decoded.order_type);
        assert_eq!(original.price_offset_ticks, decoded.price_offset_ticks);
        assert!((original.size_fraction - decoded.size_fraction).abs() < 1e-6);
        assert_eq!(original.asset_idx, decoded.asset_idx);
    }
    
    #[test]
    fn test_limit_price_calculation() {
        let buy_action = HybridAction::new(
            TradeSide::Buy,
            OrderType::Limit,
            5,
            0.5,
            0,
        );
        
        let mid_price = 100.0;
        let tick_size = 0.01;
        
        // Buy limit should be below mid
        let buy_price = buy_action.get_limit_price(mid_price, tick_size);
        assert!((buy_price - 99.95).abs() < 1e-6);
        
        let sell_action = HybridAction::new(
            TradeSide::Sell,
            OrderType::Limit,
            5,
            0.5,
            0,
        );
        
        // Sell limit should be above mid
        let sell_price = sell_action.get_limit_price(mid_price, tick_size);
        assert!((sell_price - 100.05).abs() < 1e-6);
    }
    
    #[test]
    fn test_invalid_actions() {
        // Negative size fraction
        let invalid1 = HybridAction::new(
            TradeSide::Buy,
            OrderType::Market,
            0,
            -0.1,
            0,
        );
        assert!(!invalid1.is_valid());
        
        // Size > 1.0
        let invalid2 = HybridAction::new(
            TradeSide::Buy,
            OrderType::Market,
            0,
            1.5,
            0,
        );
        assert!(!invalid2.is_valid());
    }
}
