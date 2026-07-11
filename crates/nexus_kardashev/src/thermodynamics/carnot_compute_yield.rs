//! Carnot Compute Yield Curve for Matrioshka Brains
//! 
//! Implements thermodynamic compute pricing based on Carnot efficiency
//! limits across the temperature gradient of nested Dyson shells.

use crate::thermodynamics::stefan_boltzmann_manifold::{MatrioshkaShell, StefanBoltzmannManifold, ThermoError};
use num_traits::{Float, Zero};

/// Compute task classification by latency/temperature requirements
#[derive(Debug, Clone, Copy)]
pub enum ComputeClass {
    /// High-frequency trading: requires hot inner shells for speed
    HFT,
    /// General computation: mid-range shells
    General,
    /// Batch processing: can run on cooler outer shells
    Batch,
    /// Cold storage: outermost shells near background temperature
    ColdStorage,
}

impl ComputeClass {
    /// Preferred temperature range for each class (Kelvin)
    pub fn preferred_temp_range(&self) -> (f64, f64) {
        match self {
            ComputeClass::HFT => (500.0, 1500.0),
            ComputeClass::General => (200.0, 500.0),
            ComputeClass::Batch => (50.0, 200.0),
            ComputeClass::ColdStorage => (3.0, 50.0),
        }
    }
    
    /// Latency sensitivity factor (higher = more sensitive to temperature)
    pub fn latency_sensitivity(&self) -> f64 {
        match self {
            ComputeClass::HFT => 10.0,
            ComputeClass::General => 1.0,
            ComputeClass::Batch => 0.1,
            ComputeClass::ColdStorage => 0.01,
        }
    }
}

/// Carnot compute yield curve pricer
pub struct CarnotComputePricer<T> {
    base_compute_price: T,  // $/op at reference temperature
    reference_temp: T,
}

impl<T: Float + Copy + Zero> CarnotComputePricer<T> {
    pub fn new(base_price: T, reference_temp: T) -> Self {
        Self {
            base_compute_price: base_price,
            reference_temp,
        }
    }
    
    /// Calculate Carnot efficiency between two temperatures
    /// η = 1 - T_cold/T_hot (must have T_hot > T_cold)
    pub fn calculate_carnot_efficiency(&self, t_hot: T, t_cold: T) -> Result<T, ThermoError> {
        if t_hot <= t_cold || t_cold <= T::zero() {
            return Err(ThermoError::TemperatureViolation {
                t_hot: t_hot.to_f64().unwrap_or(0.0),
                t_cold: t_cold.to_f64().unwrap_or(0.0),
            });
        }
        
        let one = T::one();
        let efficiency = one - t_cold / t_hot;
        
        // Validate efficiency is in [0, 1]
        if efficiency < T::zero() || efficiency > one {
            return Err(ThermoError::InvalidCarnotEfficiency {
                eta: efficiency.to_f64().unwrap_or(0.0),
            });
        }
        
        Ok(efficiency)
    }
    
    /// Price compute operation based on shell temperature
    pub fn price_compute_at_shell(
        &self,
        shell: &MatrioshkaShell<T>,
        heat_sink_temp: T,
        operations: u64,
    ) -> Result<ComputeQuote, ThermoError> {
        // Verify second law compliance
        if shell.temperature <= heat_sink_temp {
            return Err(ThermoError::TemperatureViolation {
                t_hot: shell.temperature.to_f64().unwrap_or(0.0),
                t_cold: heat_sink_temp.to_f64().unwrap_or(0.0),
            });
        }
        
        // Carnot efficiency limits maximum work extraction
        let carnot_eta = self.calculate_carnot_efficiency(shell.temperature, heat_sink_temp)?;
        
        // Temperature-dependent compute efficiency
        // Hotter shells: faster clock speeds but higher energy cost
        // Colder shells: slower but more energy-efficient
        
        let ops_t = T::from(operations as f64).unwrap();
        
        // Base cost adjusted by Carnot factor
        // Higher efficiency = lower cost per op
        let efficiency_factor = T::one() / (T::one() + carnot_eta);
        let base_cost = self.base_compute_price * ops_t * efficiency_factor;
        
        // Temperature premium: hotter shells cost more due to cooling requirements
        let temp_ratio = shell.temperature / self.reference_temp;
        let temp_premium = if temp_ratio > T::one() {
            temp_ratio - T::one()
        } else {
            T::zero()
        };
        
        let total_cost = base_cost * (T::one() + temp_premium * T::from(0.1).unwrap());
        
        Ok(ComputeQuote {
            total_cost_usd: total_cost.to_f64().unwrap_or(0.0),
            cost_per_op: (total_cost / ops_t).to_f64().unwrap_or(0.0),
            carnot_efficiency: carnot_eta.to_f64().unwrap_or(0.0),
            shell_temperature: shell.temperature.to_f64().unwrap_or(0.0),
            heat_sink_temp: heat_sink_temp.to_f64().unwrap_or(0.0),
        })
    }
    
    /// Find optimal shell for a compute class
    pub fn find_optimal_shell_for_class(
        &self,
        manifold: &StefanBoltzmannManifold<T>,
        compute_class: ComputeClass,
        heat_sink_temp: T,
    ) -> Option<OptimalShellAssignment<T>> {
        let (temp_min, temp_max) = compute_class.preferred_temp_range();
        let temp_min_t = T::from(temp_min).unwrap();
        let temp_max_t = T::from(temp_max).unwrap();
        
        let mut best_shell: Option<&MatrioshkaShell<T>> = None;
        let mut best_score = T::zero();
        
        for shell in &manifold.shells {
            // Check if shell is in preferred temperature range
            if shell.temperature >= temp_min_t && shell.temperature <= temp_max_t {
                // Score based on compute efficiency and cost
                if let Ok(carnot_eta) = self.calculate_carnot_efficiency(shell.temperature, heat_sink_temp) {
                    let score = carnot_eta * shell.compute_efficiency;
                    
                    if score > best_score {
                        best_score = score;
                        best_shell = Some(shell);
                    }
                }
            }
        }
        
        best_shell.map(|shell| OptimalShellAssignment {
            shell_id: shell.shell_id,
            temperature: shell.temperature,
            compute_efficiency: shell.compute_efficiency,
            score: best_score,
            compute_class,
        })
    }
    
    /// Calculate arbitrage value of moving compute between shells
    pub fn calculate_migration_arbitrage(
        &self,
        source_shell: &MatrioshkaShell<T>,
        target_shell: &MatrioshkaShell<T>,
        heat_sink_temp: T,
        operations: u64,
    ) -> Option<MigrationArbitrage> {
        let source_quote = self.price_compute_at_shell(source_shell, heat_sink_temp, operations).ok()?;
        let target_quote = self.price_compute_at_shell(target_shell, heat_sink_temp, operations).ok()?;
        
        let cost_diff = source_quote.total_cost_usd - target_quote.total_cost_usd;
        let migration_cost = operations as f64 * 1e-12;  // Data transfer cost
        
        let net_benefit = cost_diff - migration_cost;
        
        if net_benefit.abs() < 1e-6 {
            return None;  // Not worth migrating
        }
        
        Some(MigrationArbitrage {
            source_shell: source_shell.shell_id,
            target_shell: target_shell.shell_id,
            cost_savings: net_benefit.max(0.0),
            latency_change: if target_shell.temperature > source_shell.temperature {
                "decreased".to_string()
            } else {
                "increased".to_string()
            },
            recommendation: if net_benefit > 0.0 { "migrate" } else { "hold" }.to_string(),
        })
    }
}

/// Quote for compute operation
#[derive(Debug, Clone)]
pub struct ComputeQuote {
    pub total_cost_usd: f64,
    pub cost_per_op: f64,
    pub carnot_efficiency: f64,
    pub shell_temperature: f64,
    pub heat_sink_temp: f64,
}

/// Optimal shell assignment result
#[derive(Debug, Clone)]
pub struct OptimalShellAssignment<T> {
    pub shell_id: u32,
    pub temperature: T,
    pub compute_efficiency: T,
    pub score: T,
    pub compute_class: ComputeClass,
}

/// Migration arbitrage opportunity
#[derive(Debug, Clone)]
pub struct MigrationArbitrage {
    pub source_shell: u32,
    pub target_shell: u32,
    pub cost_savings: f64,
    pub latency_change: String,
    pub recommendation: String,
}

/// Thermodynamic compute allocator
pub struct ThermoComputeAllocator<T> {
    pricer: CarnotComputePricer<T>,
    manifold: StefanBoltzmannManifold<T>,
    heat_sink_temp: T,
}

impl<T: Float + Copy + Zero> ThermoComputeAllocator<T> {
    pub fn new(pricer: CarnotComputePricer<T>, manifold: StefanBoltzmannManifold<T>, heat_sink_temp: T) -> Self {
        Self {
            pricer,
            manifold,
            heat_sink_temp,
        }
    }
    
    /// Allocate compute workload to optimal shell
    pub fn allocate_workload(&self, operations: u64, compute_class: ComputeClass) -> Result<AllocationResult, ThermoError> {
        let assignment = self.pricer.find_optimal_shell_for_class(
            &self.manifold,
            compute_class,
            self.heat_sink_temp,
        ).ok_or_else(|| ThermoError::EnergyImbalance { imbalance: 0.0 })?;
        
        let shell = self.manifold.get_shell(assignment.shell_id)
            .ok_or_else(|| ThermoError::EnergyImbalance { imbalance: 0.0 })?;
        
        let quote = self.pricer.price_compute_at_shell(shell, self.heat_sink_temp, operations)?;
        
        Ok(AllocationResult {
            assigned_shell: assignment.shell_id,
            operations,
            compute_class,
            total_cost: quote.total_cost_usd,
            estimated_latency_ms: self.estimate_latency(shell, operations),
        })
    }
    
    fn estimate_latency(&self, shell: &MatrioshkaShell<T>, operations: u64) -> f64 {
        // Simplified: latency inversely proportional to temperature
        // (hotter = faster clock speeds)
        let temp_k = shell.temperature.to_f64().unwrap_or(300.0);
        let base_latency = 1.0;  // 1ms baseline
        let temp_factor = 1000.0 / temp_k;  // Faster at higher temps
        
        base_latency * temp_factor * (operations as f64 / 1e9)
    }
}

/// Allocation result
#[derive(Debug, Clone)]
pub struct AllocationResult {
    pub assigned_shell: u32,
    pub operations: u64,
    pub compute_class: ComputeClass,
    pub total_cost: f64,
    pub estimated_latency_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_carnot_efficiency() {
        type F = f64;
        let pricer = CarnotComputePricer::new(F::from(1e-9).unwrap(), F::from(300.0).unwrap());
        
        let t_hot = F::from(1000.0).unwrap();
        let t_cold = F::from(300.0).unwrap();
        
        let eta = pricer.calculate_carnot_efficiency(t_hot, t_cold).unwrap();
        
        // η = 1 - 300/1000 = 0.7
        assert!((eta - 0.7).abs() < 0.01);
    }
    
    #[test]
    fn test_carnot_violation_detection() {
        type F = f64;
        let pricer = CarnotComputePricer::new(F::from(1e-9).unwrap(), F::from(300.0).unwrap());
        
        let t_hot = F::from(300.0).unwrap();
        let t_cold = F::from(500.0).unwrap();  // Cold is hotter than hot!
        
        let result = pricer.calculate_carnot_efficiency(t_hot, t_cold);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_compute_class_temp_ranges() {
        let hft_range = ComputeClass::HFT.preferred_temp_range();
        assert!(hft_range.0 < hft_range.1);
        assert!(hft_range.0 > 100.0);  // HFT needs hot shells
    }
}
