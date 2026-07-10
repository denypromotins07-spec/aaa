//! Geographic Workload Shifter for Energy Cost Optimization
//! 
//! Shifts compute workloads across geographically distributed nodes
//! based on real-time energy pricing and carbon intensity.

use core::fmt;

/// Maximum number of geographic regions supported
const MAX_REGIONS: usize = 32;
/// Minimum workload shift threshold (percentage cost difference)
const MIN_SHIFT_THRESHOLD: f64 = 15.0;
/// Network latency penalty threshold (ms)
const LATENCY_PENALTY_THRESHOLD_MS: u64 = 50;

/// Errors in geographic workload shifting
#[derive(Debug, Clone, PartialEq)]
pub enum WorkloadShiftError {
    InvalidRegionId,
    InsufficientCapacity,
    NetworkLatencyTooHigh,
    CostCalculationFailed,
    MigrationBlocked,
    RegionUnavailable,
}

impl fmt::Display for WorkloadShiftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkloadShiftError::InvalidRegionId => write!(f, "Invalid region identifier"),
            WorkloadShiftError::InsufficientCapacity => write!(f, "Target region has insufficient capacity"),
            WorkloadShiftError::NetworkLatencyTooHigh => write!(f, "Network latency exceeds threshold"),
            WorkloadShiftError::CostCalculationFailed => write!(f, "Energy cost calculation failed"),
            WorkloadShiftError::MigrationBlocked => write!(f, "Workload migration blocked"),
            WorkloadShiftError::RegionUnavailable => write!(f, "Target region unavailable"),
        }
    }
}

/// Energy pricing data for a region
#[derive(Debug, Clone, Copy)]
pub struct EnergyPricing {
    /// Current price ($/MWh)
    pub current_price: f64,
    /// Predicted price in 1 hour ($/MWh)
    pub predicted_price_1h: f64,
    /// Predicted price in 6 hours ($/MWh)
    pub predicted_price_6h: f64,
    /// Carbon intensity (gCO2/kWh)
    pub carbon_intensity: f64,
    /// Grid frequency stability (Hz deviation)
    pub frequency_deviation: f64,
}

impl Default for EnergyPricing {
    fn default() -> Self {
        Self {
            current_price: 50.0,
            predicted_price_1h: 50.0,
            predicted_price_6h: 50.0,
            carbon_intensity: 400.0,
            frequency_deviation: 0.0,
        }
    }
}

/// Regional compute node state
#[derive(Debug, Clone, Copy)]
pub struct RegionalNode {
    /// Region identifier
    pub region_id: u8,
    /// Available compute capacity (normalized 0-1)
    pub available_capacity: f64,
    /// Current utilization (0-1)
    pub utilization: f64,
    /// Network latency to primary node (ms)
    pub latency_ms: u64,
    /// Node is accepting migrations
    pub accepting_migrations: bool,
    /// Energy pricing for this region
    pub pricing: EnergyPricing,
}

impl Default for RegionalNode {
    fn default() -> Self {
        Self {
            region_id: 0,
            available_capacity: 1.0,
            utilization: 0.0,
            latency_ms: 0,
            accepting_migrations: true,
            pricing: EnergyPricing::default(),
        }
    }
}

/// Workload descriptor for migration
#[derive(Debug, Clone, Copy)]
pub struct WorkloadDescriptor {
    /// Unique workload identifier
    pub id: u64,
    /// Compute requirements (normalized 0-1)
    pub compute_requirement: f64,
    /// Memory requirements (GB)
    pub memory_gb: f64,
    /// Priority (higher = more critical)
    pub priority: u8,
    /// Current region
    pub current_region: u8,
    /// Migration allowed flag
    pub migration_allowed: bool,
    /// Last migration timestamp (hours ago)
    pub last_migration_hours: u64,
}

impl Default for WorkloadDescriptor {
    fn default() -> Self {
        Self {
            id: 0,
            compute_requirement: 0.1,
            memory_gb: 8.0,
            priority: 128,
            current_region: 0,
            migration_allowed: true,
            last_migration_hours: 24,
        }
    }
}

/// Geographic workload shifter
pub struct GeographicWorkloadShifter {
    /// Regional nodes
    nodes: [RegionalNode; MAX_REGIONS],
    /// Number of active regions
    active_regions: usize,
    /// Primary region ID
    primary_region: u8,
}

impl GeographicWorkloadShifter {
    /// Create a new workload shifter
    pub fn new(active_regions: usize, primary_region: u8) -> Result<Self, WorkloadShiftError> {
        if active_regions == 0 || active_regions > MAX_REGIONS {
            return Err(WorkloadShiftError::InvalidRegionId);
        }
        if primary_region as usize >= active_regions {
            return Err(WorkloadShiftError::InvalidRegionId);
        }

        let mut nodes = [RegionalNode::default(); MAX_REGIONS];
        for i in 0..active_regions {
            nodes[i].region_id = i as u8;
            nodes[i].accepting_migrations = true;
        }

        Ok(Self {
            nodes,
            active_regions,
            primary_region,
        })
    }

    /// Update regional node state
    pub fn update_node_state(
        &mut self,
        region_id: u8,
        available_capacity: f64,
        utilization: f64,
        latency_ms: u64,
        pricing: EnergyPricing,
    ) -> Result<(), WorkloadShiftError> {
        if region_id as usize >= self.active_regions {
            return Err(WorkloadShiftError::InvalidRegionId);
        }

        let node = &mut self.nodes[region_id as usize];
        node.available_capacity = available_capacity.clamp(0.0, 1.0);
        node.utilization = utilization.clamp(0.0, 1.0);
        node.latency_ms = latency_ms;
        node.pricing = pricing;

        // Mark unavailable if capacity too low or latency too high
        node.accepting_migrations = available_capacity > 0.1 && latency_ms < LATENCY_PENALTY_THRESHOLD_MS;

        Ok(())
    }

    /// Evaluate if workload should be migrated
    pub fn evaluate_migration(&self, workload: &WorkloadDescriptor) -> Option<u8> {
        if !workload.migration_allowed {
            return None;
        }

        // Check migration cooldown (minimum 1 hour between migrations)
        if workload.last_migration_hours < 1 {
            return None;
        }

        let current_region = workload.current_region as usize;
        if current_region >= self.active_regions {
            return None;
        }

        let current_node = &self.nodes[current_region];
        let current_cost = current_node.pricing.current_price;

        // Find best alternative region
        let mut best_region: Option<u8> = None;
        let mut best_savings = 0.0;

        for i in 0..self.active_regions {
            if i == current_region {
                continue;
            }

            let candidate = &self.nodes[i];
            if !candidate.accepting_migrations {
                continue;
            }

            // Check capacity
            if candidate.available_capacity < workload.compute_requirement {
                continue;
            }

            // Calculate effective cost including latency penalty
            let latency_penalty = if candidate.latency_ms > LATENCY_PENALTY_THRESHOLD_MS {
                (candidate.latency_ms - LATENCY_PENALTY_THRESHOLD_MS) as f64 * 0.1
            } else {
                0.0
            };

            let effective_cost = candidate.pricing.current_price + latency_penalty;
            let savings = current_cost - effective_cost;

            // Only consider if savings exceed threshold
            if savings > current_cost * MIN_SHIFT_THRESHOLD / 100.0 {
                if savings > best_savings {
                    best_savings = savings;
                    best_region = Some(i as u8);
                }
            }
        }

        best_region
    }

    /// Execute workload migration
    pub fn migrate_workload(&mut self, workload: &mut WorkloadDescriptor, target_region: u8)
        -> Result<(), WorkloadShiftError>
    {
        if target_region as usize >= self.active_regions {
            return Err(WorkloadShiftError::InvalidRegionId);
        }

        let target_node = &self.nodes[target_region as usize];
        if !target_node.accepting_migrations {
            return Err(WorkloadShiftError::RegionUnavailable);
        }

        if target_node.available_capacity < workload.compute_requirement {
            return Err(WorkloadShiftError::InsufficientCapacity);
        }

        // In real implementation, this would:
        // 1. Serialize workload state
        // 2. Transfer to target region
        // 3. Resume execution
        // 4. Update routing tables

        let old_region = workload.current_region;
        workload.current_region = target_region;
        workload.last_migration_hours = 0;

        // Update node capacities
        self.nodes[old_region as usize].available_capacity += workload.compute_requirement;
        self.nodes[target_region as usize].available_capacity -= workload.compute_requirement;
        self.nodes[target_region as usize].utilization += workload.compute_requirement;

        Ok(())
    }

    /// Get total energy cost across all regions
    pub fn get_total_energy_cost(&self) -> f64 {
        let mut total_cost = 0.0;
        for i in 0..self.active_regions {
            let node = &self.nodes[i];
            total_cost += node.utilization * node.pricing.current_price;
        }
        total_cost
    }

    /// Get total carbon footprint
    pub fn get_total_carbon_footprint(&self) -> f64 {
        let mut total_carbon = 0.0;
        for i in 0..self.active_regions {
            let node = &self.nodes[i];
            total_carbon += node.utilization * node.pricing.carbon_intensity;
        }
        total_carbon
    }

    /// Optimize workload distribution for minimum cost
    pub fn optimize_for_cost(&mut self, workloads: &mut [WorkloadDescriptor]) -> Result<f64, WorkloadShiftError> {
        let mut total_savings = 0.0;

        // Sort workloads by priority (lower priority first for migration)
        workloads.sort_by(|a, b| b.priority.cmp(&a.priority));

        for workload in workloads.iter_mut() {
            if let Some(target) = self.evaluate_migration(workload) {
                let old_cost = self.nodes[workload.current_region as usize].pricing.current_price 
                    * workload.compute_requirement;
                
                if let Ok(_) = self.migrate_workload(workload, target) {
                    let new_cost = self.nodes[target as usize].pricing.current_price 
                        * workload.compute_requirement;
                    total_savings += old_cost - new_cost;
                }
            }
        }

        Ok(total_savings)
    }

    /// Optimize workload distribution for minimum carbon
    pub fn optimize_for_carbon(&mut self, workloads: &mut [WorkloadDescriptor]) -> Result<f64, WorkloadShiftError> {
        let mut carbon_reduction = 0.0;

        // Sort workloads by priority (lower priority first)
        workloads.sort_by(|a, b| b.priority.cmp(&a.priority));

        for workload in workloads.iter_mut() {
            let current_region = workload.current_region as usize;
            let current_carbon = self.nodes[current_region].pricing.carbon_intensity;

            // Find lowest carbon region with capacity
            let mut best_region: Option<u8> = None;
            let mut lowest_carbon = current_carbon;

            for i in 0..self.active_regions {
                if i == current_region {
                    continue;
                }

                let candidate = &self.nodes[i];
                if !candidate.accepting_migrations {
                    continue;
                }

                if candidate.available_capacity < workload.compute_requirement {
                    continue;
                }

                if candidate.pricing.carbon_intensity < lowest_carbon {
                    lowest_carbon = candidate.pricing.carbon_intensity;
                    best_region = Some(i as u8);
                }
            }

            if let Some(target) = best_region {
                if let Ok(_) = self.migrate_workload(workload, target) {
                    carbon_reduction += (current_carbon - lowest_carbon) * workload.compute_requirement;
                }
            }
        }

        Ok(carbon_reduction)
    }

    /// Get node by region ID
    pub fn get_node(&self, region_id: u8) -> Option<&RegionalNode> {
        if region_id as usize < self.active_regions {
            Some(&self.nodes[region_id as usize])
        } else {
            None
        }
    }

    /// Get active region count
    pub fn active_regions(&self) -> usize {
        self.active_regions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shifter_creation() {
        let shifter = GeographicWorkloadShifter::new(4, 0);
        assert!(shifter.is_ok());
    }

    #[test]
    fn test_invalid_region_count() {
        let shifter = GeographicWorkloadShifter::new(0, 0);
        assert_eq!(shifter.unwrap_err(), WorkloadShiftError::InvalidRegionId);
    }

    #[test]
    fn test_node_update() {
        let mut shifter = GeographicWorkloadShifter::new(4, 0).unwrap();
        
        let pricing = EnergyPricing {
            current_price: 30.0,
            ..Default::default()
        };
        
        let result = shifter.update_node_state(1, 0.8, 0.2, 20, pricing);
        assert!(result.is_ok());
        
        let node = shifter.get_node(1).unwrap();
        assert_eq!(node.pricing.current_price, 30.0);
    }

    #[test]
    fn test_migration_evaluation() {
        let mut shifter = GeographicWorkloadShifter::new(4, 0).unwrap();
        
        // Set region 0 to high price
        let pricing_high = EnergyPricing {
            current_price: 100.0,
            ..Default::default()
        };
        shifter.update_node_state(0, 0.5, 0.5, 10, pricing_high).unwrap();
        
        // Set region 1 to low price
        let pricing_low = EnergyPricing {
            current_price: 40.0,
            ..Default::default()
        };
        shifter.update_node_state(1, 0.8, 0.2, 15, pricing_low).unwrap();
        
        let workload = WorkloadDescriptor {
            current_region: 0,
            compute_requirement: 0.2,
            migration_allowed: true,
            last_migration_hours: 24,
            ..Default::default()
        };
        
        let target = shifter.evaluate_migration(&workload);
        assert_eq!(target, Some(1));
    }
}
