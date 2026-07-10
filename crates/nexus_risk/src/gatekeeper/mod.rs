//! Chapter 1: Pre-Trade Risk Gatekeeper
//! 
//! Lock-free observer pattern for validating outbound orders before they reach the network.

pub mod pre_trade_interceptor;
pub mod fat_finger_collars;
pub mod lock_free_order_queue;
