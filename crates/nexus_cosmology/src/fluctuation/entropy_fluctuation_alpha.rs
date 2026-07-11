//! Entropy Fluctuation Alpha Router
//! 
//! Implements the "limit order" system for placing trades in spacetime,
//! waiting for quantum fluctuations to assemble computational substrates.

use super::boltzmann_brain_nucleation::{LogProb, BoltzmannParams, BoltzmannNucleationCalculator};

/// A "limit order" in spacetime waiting for a fluctuation event
#[derive(Debug, Clone)]
pub struct FluctuationOrder {
    /// Unique order identifier
    pub id: u64,
    /// Target entropy decrease [J/K]
    pub target_delta_s: f64,
    /// Maximum acceptable "price" (inverse probability)
    pub max_cost_ln: f64, // ln(1/P) = -ln(P)
    /// Payload data to upload upon fluctuation
    pub payload_hash: [u8; 32],
    /// Order status
    pub status: OrderStatus,
    /// Creation time (cosmological epoch)
    pub creation_epoch: f64,
}

/// Order status in the fluctuation queue
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderStatus {
    /// Waiting for fluctuation
    Pending,
    /// Fluctuation detected, uploading payload
    Executing,
    /// Successfully uploaded
    Completed,
    /// Fluctuation decohered before upload complete
    Failed,
    /// Cancelled by requester
    Cancelled,
}

/// Entropy fluctuation trading router
#[derive(Debug, Clone)]
pub struct FluctuationAlphaRouter {
    /// Nucleation calculator
    calculator: BoltzmannNucleationCalculator,
    /// Current background temperature [K]
    background_temperature: f64,
    /// Observable horizon volume [m³]
    horizon_volume: f64,
    /// Order counter
    order_counter: u64,
    /// Pending orders
    pending_orders: Vec<FluctuationOrder>,
}

impl FluctuationAlphaRouter {
    /// Create a new fluctuation router
    /// 
    /// # Arguments
    /// * `background_temp` - Cosmic background temperature [K]
    /// * `horizon_volume` - Accessible spacetime volume [m³]
    /// 
    /// # Returns
    /// * `Result<Self, &'static str>` - Router or error
    pub fn new(background_temp: f64, horizon_volume: f64) -> Result<Self, &'static str> {
        if background_temp <= 0.0 {
            return Err("Background temperature must be positive");
        }
        if horizon_volume <= 0.0 {
            return Err("Horizon volume must be positive");
        }
        
        Ok(Self {
            calculator: BoltzmannNucleationCalculator::default(),
            background_temperature: background_temp,
            horizon_volume,
            order_counter: 0,
            pending_orders: Vec::new(),
        })
    }
    
    /// Place a new fluctuation order
    /// 
    /// # Arguments
    /// * `target_delta_s` - Target entropy decrease [J/K]
    /// * `max_cost_ln` - Maximum acceptable -ln(P)
    /// * `payload_hash` - Hash of data to upload
    /// 
    /// # Returns
    /// * `Result<u64, &'static str>` - Order ID
    pub fn place_order(
        &mut self,
        target_delta_s: f64,
        max_cost_ln: f64,
        payload_hash: [u8; 32],
    ) -> Result<u64, &'static str> {
        if target_delta_s <= 0.0 {
            return Err("Target entropy decrease must be positive");
        }
        if max_cost_ln <= 0.0 {
            return Err("Maximum cost must be positive");
        }
        
        let order = FluctuationOrder {
            id: self.order_counter,
            target_delta_s,
            max_cost_ln,
            payload_hash,
            status: OrderStatus::Pending,
            creation_epoch: self.get_cosmological_time(),
        };
        
        self.order_counter += 1;
        let order_id = order.id;
        self.pending_orders.push(order);
        
        Ok(order_id)
    }
    
    /// Calculate the "cost" of a fluctuation order
    /// Cost = -ln(P) where P is the nucleation probability
    /// 
    /// # Arguments
    /// * `order` - The order to evaluate
    /// 
    /// # Returns
    /// * `f64` - Cost in natural log units
    pub fn calculate_order_cost(&self, order: &FluctuationOrder) -> f64 {
        let params = BoltzmannParams {
            delta_s: order.target_delta_s,
            temperature: self.background_temperature,
            volume: 1.0, // Unit volume for rate calculation
            timescale: 1.0,
        };
        
        let prob = self.calculator.calculate_nucleation_rate(&params);
        
        // Cost = -ln(P)
        -prob.ln_p
    }
    
    /// Check if an order's cost is within budget
    /// 
    /// # Arguments
    /// * `order_id` - Order to check
    /// 
    /// # Returns
    /// * `Result<bool, &'static str>` - True if affordable
    pub fn is_order_affordable(&self, order_id: u64) -> Result<bool, &'static str> {
        let order = self.pending_orders.iter()
            .find(|o| o.id == order_id)
            .ok_or("Order not found")?;
        
        let cost = self.calculate_order_cost(order);
        Ok(cost <= order.max_cost_ln)
    }
    
    /// Simulate fluctuation detection and order execution
    /// 
    /// In reality, this would interface with quantum field monitors
    /// 
    /// # Arguments
    /// * `dt` - Simulation timestep [s]
    /// 
    /// # Returns
    /// * `Vec<u64>` - IDs of orders that executed
    pub fn simulate_timestep(&mut self, dt: f64) -> Vec<u64> {
        let mut executed = Vec::new();
        
        for order in &mut self.pending_orders {
            if order.status != OrderStatus::Pending {
                continue;
            }
            
            let cost = self.calculate_order_cost(order);
            
            // Skip orders that exceed budget
            if cost > order.max_cost_ln {
                continue;
            }
            
            // Calculate probability of execution in this timestep
            let params = BoltzmannParams {
                delta_s: order.target_delta_s,
                temperature: self.background_temperature,
                volume: self.horizon_volume,
                timescale: dt,
            };
            
            let prob = self.calculator.calculate_nucleation_rate(&params);
            
            // For cosmological timescales, we use a stochastic threshold
            // In practice, these probabilities are so small that execution
            // is essentially impossible within any reasonable timeframe
            // unless the entropy decrease is microscopic
            
            if !prob.is_effectively_zero() {
                // Non-zero chance of execution
                let attempt_prob = prob.to_probability();
                if attempt_prob > 0.0 && (attempt_prob * self.horizon_volume * dt) > 0.5 {
                    order.status = OrderStatus::Executing;
                    executed.push(order.id);
                }
            }
        }
        
        executed
    }
    
    /// Get the expected waiting time for an order
    /// 
    /// # Arguments
    /// * `order_id` - Order to evaluate
    /// 
    /// # Returns
    /// * `Result<f64, &'static str>` - Expected time [s] or infinity
    pub fn expected_wait_time(&self, order_id: u64) -> Result<f64, &'static str> {
        let order = self.pending_orders.iter()
            .find(|o| o.id == order_id)
            .ok_or("Order not found")?;
        
        let params = BoltzmannParams {
            delta_s: order.target_delta_s,
            temperature: self.background_temperature,
            volume: self.horizon_volume,
            timescale: 1.0,
        };
        
        self.calculator.expected_waiting_time(self.horizon_volume, &params)
    }
    
    /// Cancel a pending order
    /// 
    /// # Arguments
    /// * `order_id` - Order to cancel
    /// 
    /// # Returns
    /// * `Result<(), &'static str>` - Success or error
    pub fn cancel_order(&mut self, order_id: u64) -> Result<(), &'static str> {
        let order = self.pending_orders.iter_mut()
            .find(|o| o.id == order_id)
            .ok_or("Order not found")?;
        
        if order.status != OrderStatus::Pending {
            return Err("Can only cancel pending orders");
        }
        
        order.status = OrderStatus::Cancelled;
        Ok(())
    }
    
    /// Get current cosmological time (placeholder)
    fn get_cosmological_time(&self) -> f64 {
        // In a full simulation, this would track actual cosmic time
        1e17 // ~current age of universe in seconds
    }
    
    /// Update background temperature (for evolving universe models)
    /// 
    /// # Arguments
    /// * `new_temp` - New background temperature [K]
    /// 
    /// # Returns
    /// * `Result<(), &'static str>` - Success or error
    pub fn update_temperature(&mut self, new_temp: f64) -> Result<(), &'static str> {
        if new_temp <= 0.0 {
            return Err("Temperature must be positive");
        }
        self.background_temperature = new_temp;
        Ok(())
    }
    
    /// Get statistics about pending orders
    pub fn get_statistics(&self) -> RouterStats {
        let pending = self.pending_orders.iter()
            .filter(|o| o.status == OrderStatus::Pending)
            .count();
        let executing = self.pending_orders.iter()
            .filter(|o| o.status == OrderStatus::Executing)
            .count();
        let completed = self.pending_orders.iter()
            .filter(|o| o.status == OrderStatus::Completed)
            .count();
        
        RouterStats {
            total_orders: self.order_counter,
            pending,
            executing,
            completed,
            failed: self.pending_orders.iter()
                .filter(|o| o.status == OrderStatus::Failed)
                .count(),
            cancelled: self.pending_orders.iter()
                .filter(|o| o.status == OrderStatus::Cancelled)
                .count(),
        }
    }
}

/// Router statistics
#[derive(Debug, Clone, Copy)]
pub struct RouterStats {
    /// Total orders ever placed
    pub total_orders: u64,
    /// Currently pending
    pub pending: usize,
    /// Currently executing
    pub executing: usize,
    /// Successfully completed
    pub completed: usize,
    /// Failed executions
    pub failed: usize,
    /// Cancelled orders
    pub cancelled: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let router = FluctuationAlphaRouter::new(1e-30, 1e80);
        assert!(router.is_ok());
    }

    #[test]
    fn test_place_order() {
        let mut router = FluctuationAlphaRouter::new(1e-30, 1e80).unwrap();
        
        let hash = [0u8; 32];
        let order_id = router.place_order(1e-2, 1e25, hash);
        
        assert!(order_id.is_ok());
        assert_eq!(order_id.unwrap(), 0);
    }

    #[test]
    fn test_order_cost() {
        let mut router = FluctuationAlphaRouter::new(1e-30, 1e80).unwrap();
        
        let hash = [0u8; 32];
        let order_id = router.place_order(1e-2, 1e25, hash).unwrap();
        
        let cost = router.calculate_order_cost(
            router.pending_orders.iter().find(|o| o.id == order_id).unwrap()
        );
        
        assert!(cost > 0.0);
    }

    #[test]
    fn test_cancel_order() {
        let mut router = FluctuationAlphaRouter::new(1e-30, 1e80).unwrap();
        
        let hash = [0u8; 32];
        let order_id = router.place_order(1e-2, 1e25, hash).unwrap();
        
        let result = router.cancel_order(order_id);
        assert!(result.is_ok());
        
        let stats = router.get_statistics();
        assert_eq!(stats.cancelled, 1);
    }

    #[test]
    fn test_statistics() {
        let mut router = FluctuationAlphaRouter::new(1e-30, 1e80).unwrap();
        
        let hash = [0u8; 32];
        router.place_order(1e-2, 1e25, hash).unwrap();
        router.place_order(1e-3, 1e20, hash).unwrap();
        
        let stats = router.get_statistics();
        assert_eq!(stats.total_orders, 2);
        assert_eq!(stats.pending, 2);
    }
}
