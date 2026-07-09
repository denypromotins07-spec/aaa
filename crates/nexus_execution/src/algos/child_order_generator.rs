//! Child Order Generator for algorithmic execution.
//! Generates child orders from parent orders based on execution algo state.

use nexus_oms::{FixedPoint, Side, OrderType, OrderId};
use std::sync::atomic::{AtomicU64, Ordering};

const SCALE: i64 = 100_000_000;

/// Child order request
#[derive(Debug, Clone, Copy)]
pub struct ChildOrderRequest {
    pub parent_id: OrderId,
    pub side: Side,
    pub order_type: OrderType,
    pub price: FixedPoint,
    pub quantity: FixedPoint,
    pub venue_id: u32,
    pub is_ioc: bool,
    pub sequence: u64,
}

/// Execution algo type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgoType {
    Iceberg,
    Pov,
    Vwap,
    Twap,
    Sniper,
}

/// Child order generator state
pub struct ChildOrderGenerator {
    /// Current algo type
    algo_type: AlgoType,
    /// Parent order ID
    parent_id: OrderId,
    /// Side
    side: Side,
    /// Total parent quantity
    total_qty: FixedPoint,
    /// Remaining parent quantity
    remaining_qty: FixedPoint,
    /// Generated child count
    children_generated: AtomicU64,
    /// Filled child count
    children_filled: AtomicU64,
    /// Last child order ID
    last_child_id: AtomicU64,
    /// Sequence number
    sequence: AtomicU64,
    /// Default venue
    default_venue: u32,
}

impl ChildOrderGenerator {
    #[inline]
    pub fn new(
        algo_type: AlgoType,
        parent_id: OrderId,
        side: Side,
        total_qty: FixedPoint,
        default_venue: u32,
    ) -> Self {
        Self {
            algo_type,
            parent_id,
            side,
            total_qty,
            remaining_qty: total_qty,
            children_generated: AtomicU64::new(0),
            children_filled: AtomicU64::new(0),
            last_child_id: AtomicU64::new(0),
            sequence: AtomicU64::new(0),
            default_venue,
        }
    }

    /// Generate next child order for iceberg
    #[inline]
    pub fn generate_iceberg_child(&self, clip_qty: FixedPoint, limit_price: FixedPoint) -> Option<ChildOrderRequest> {
        if self.remaining_qty.is_zero() {
            return None;
        }

        let qty = clip_qty.min(self.remaining_qty);
        if qty.is_zero() {
            return None;
        }

        let child_id = self.last_child_id.fetch_add(1, Ordering::Relaxed);
        let seq = self.sequence.fetch_add(1, Ordering::Relaxed);

        Some(ChildOrderRequest {
            parent_id: self.parent_id,
            side: self.side,
            order_type: OrderType::Limit,
            price: limit_price,
            quantity: qty,
            venue_id: self.default_venue,
            is_ioc: false,
            sequence: seq,
        })
    }

    /// Generate next child order for POV
    #[inline]
    pub fn generate_pov_child(&self, market_vol: FixedPoint, participation: FixedPoint, 
                               limit_price: FixedPoint, max_qty: FixedPoint) -> Option<ChildOrderRequest> {
        if self.remaining_qty.is_zero() {
            return None;
        }

        // Calculate target size based on market volume and participation rate
        let target_qty = market_vol * participation;
        let qty = target_qty.min(max_qty).min(self.remaining_qty);
        
        if qty.is_zero() {
            return None;
        }

        let child_id = self.last_child_id.fetch_add(1, Ordering::Relaxed);
        let seq = self.sequence.fetch_add(1, Ordering::Relaxed);

        Some(ChildOrderRequest {
            parent_id: self.parent_id,
            side: self.side,
            order_type: OrderType::Limit,
            price: limit_price,
            quantity: qty,
            venue_id: self.default_venue,
            is_ioc: false,
            sequence: seq,
        })
    }

    /// Generate sniper IOC order for stale quote detection
    #[inline]
    pub fn generate_sniper_child(&self, qty: FixedPoint, price: FixedPoint, venue_id: u32) -> Option<ChildOrderRequest> {
        if self.remaining_qty.is_zero() {
            return None;
        }

        let actual_qty = qty.min(self.remaining_qty);
        if actual_qty.is_zero() {
            return None;
        }

        let child_id = self.last_child_id.fetch_add(1, Ordering::Relaxed);
        let seq = self.sequence.fetch_add(1, Ordering::Relaxed);

        Some(ChildOrderRequest {
            parent_id: self.parent_id,
            side: self.side,
            order_type: OrderType::IOC,
            price,
            quantity: actual_qty,
            venue_id,
            is_ioc: true,
            sequence: seq,
        })
    }

    /// Update remaining quantity after child fill
    #[inline]
    pub fn on_child_fill(&mut self, fill_qty: FixedPoint) -> Result<(), &'static str> {
        if fill_qty.is_zero() {
            return Err("Fill quantity cannot be zero");
        }

        if fill_qty > self.remaining_qty {
            return Err("Fill exceeds remaining quantity");
        }

        self.remaining_qty = self.remaining_qty - fill_qty;
        self.children_filled.fetch_add(1, Ordering::Relaxed);
        self.sequence.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Get remaining quantity
    #[inline]
    pub fn get_remaining_qty(&self) -> FixedPoint {
        self.remaining_qty
    }

    /// Check if complete
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.remaining_qty.is_zero()
    }

    /// Get children generated count
    #[inline]
    pub fn get_children_generated(&self) -> u64 {
        self.children_generated.load(Ordering::Relaxed)
    }

    /// Get children filled count
    #[inline]
    pub fn get_children_filled(&self) -> u64 {
        self.children_filled.load(Ordering::Relaxed)
    }

    /// Get sequence number
    #[inline]
    pub fn get_sequence(&self) -> u64 {
        self.sequence.load(Ordering::Relaxed)
    }

    /// Get algo type
    #[inline]
    pub fn get_algo_type(&self) -> AlgoType {
        self.algo_type
    }

    /// Get parent ID
    #[inline]
    pub fn get_parent_id(&self) -> OrderId {
        self.parent_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iceberg_child_generation() {
        let gen = ChildOrderGenerator::new(
            AlgoType::Iceberg,
            OrderId::new(1),
            Side::Buy,
            FixedPoint::from_int(1000),
            1,
        );

        let child = gen.generate_iceberg_child(
            FixedPoint::from_int(100),
            FixedPoint::from_int(50),
        );

        assert!(child.is_some());
        let child = child.unwrap();
        assert_eq!(child.parent_id, OrderId::new(1));
        assert_eq!(child.quantity.to_f64(), 100.0);
        assert_eq!(child.price.to_f64(), 50.0);
        assert_eq!(child.side, Side::Buy);
    }

    #[test]
    fn test_pov_child_generation() {
        let gen = ChildOrderGenerator::new(
            AlgoType::Pov,
            OrderId::new(2),
            Side::Sell,
            FixedPoint::from_int(500),
            2,
        );

        let child = gen.generate_pov_child(
            FixedPoint::from_int(10000), // Market volume
            FixedPoint::from_fractional(5_000_000), // 5% participation
            FixedPoint::from_int(55),
            FixedPoint::from_int(200), // Max qty
        );

        assert!(child.is_some());
        let child = child.unwrap();
        // 5% of 10000 = 500, clamped to max 200
        assert_eq!(child.quantity.to_f64(), 200.0);
    }

    #[test]
    fn test_sniper_child_generation() {
        let gen = ChildOrderGenerator::new(
            AlgoType::Sniper,
            OrderId::new(3),
            Side::Buy,
            FixedPoint::from_int(100),
            1,
        );

        let child = gen.generate_sniper_child(
            FixedPoint::from_int(50),
            FixedPoint::from_int(49),
            3,
        );

        assert!(child.is_some());
        let child = child.unwrap();
        assert_eq!(child.order_type, OrderType::IOC);
        assert_eq!(child.is_ioc, true);
        assert_eq!(child.venue_id, 3);
    }
}
