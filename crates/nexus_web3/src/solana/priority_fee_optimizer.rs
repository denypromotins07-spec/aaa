//! Priority Fee Optimizer for Solana/Jito
//! 
//! Dynamically calculates optimal micro-lamport tips based on:
//! - Real-time Jito bundle competition
//! - Compute unit estimates
//! - Network congestion metrics
//! - Predictive tip escalation for lag compensation

use thiserror::Error;
use alloc::vec::Vec;

/// Minimum tip per CU to be considered by validators
pub const MIN_TIP_PER_CU: u64 = 10;

/// Maximum tip per CU to prevent overpayment
pub const MAX_TIP_PER_CU: u64 = 10_000;

/// Lag compensation threshold in milliseconds
const LAG_COMPENSATION_THRESHOLD_MS: u64 = 50;

/// Tip escalation factor when lag detected
const LAG_ESCALATION_FACTOR: u64 = 150; // 150% of calculated tip

#[derive(Error, Debug)]
pub enum PriorityFeeError {
    #[error("Invalid compute unit estimate")]
    InvalidComputeUnits,
    #[error("Network data unavailable")]
    NetworkDataUnavailable,
    #[error("Tip calculation overflow")]
    TipOverflow,
}

pub type Result<T> = core::result::Result<T, PriorityFeeError>;

/// Compute unit estimation for common instruction types
#[derive(Clone, Copy, Debug)]
pub struct CuEstimate {
    pub instruction_type: InstructionType,
    pub estimated_cu: u64,
    pub variance_percent: u8, // 0-100
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InstructionType {
    Transfer,
    Swap,
    Stake,
    Vote,
    CustomProgram,
}

/// Network congestion level
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CongestionLevel {
    Low,
    Medium,
    High,
    Extreme,
}

/// Priority Fee Optimizer state
pub struct PriorityFeeOptimizer {
    /// Base tip per CU from Jito metrics
    base_tip_per_cu: u64,
    /// Current network congestion
    congestion: CongestionLevel,
    /// Last update timestamp (ms since epoch)
    last_update_ms: u64,
    /// Lag compensation multiplier (basis points)
    lag_multiplier_bp: u16,
    /// Recent tip success rate (0-100)
    success_rate: u8,
}

impl PriorityFeeOptimizer {
    /// Create a new optimizer with default values
    pub fn new() -> Self {
        Self {
            base_tip_per_cu: MIN_TIP_PER_CU,
            congestion: CongestionLevel::Low,
            last_update_ms: 0,
            lag_multiplier_bp: 10000, // 100% = no adjustment
            success_rate: 100,
        }
    }

    /// Update with fresh Jito metrics
    pub fn update_jito_metrics(&mut self, tip_per_cu: u64, success_rate: u8) {
        self.base_tip_per_cu = tip_per_cu.clamp(MIN_TIP_PER_CU, MAX_TIP_PER_CU);
        self.success_rate = success_rate.min(100);
        self.last_update_ms = current_timestamp_ms();
    }

    /// Update network congestion level
    pub fn update_congestion(&mut self, congestion: CongestionLevel) {
        self.congestion = congestion;
    }

    /// Check if metrics are stale (lag detection)
    pub fn is_stale(&self) -> bool {
        let now = current_timestamp_ms();
        let age_ms = now.saturating_sub(self.last_update_ms);
        age_ms > LAG_COMPENSATION_THRESHOLD_MS
    }

    /// Apply lag compensation if metrics are stale
    fn apply_lag_compensation(&mut self) {
        if self.is_stale() {
            // Increase multiplier by escalation factor
            self.lag_multiplier_bp = self.lag_multiplier_bp
                .saturating_mul(LAG_ESCALATION_FACTOR as u16)
                .min(50000); // Cap at 500%
        } else {
            // Reset to normal
            self.lag_multiplier_bp = 10000;
        }
    }

    /// Calculate optimal tip for given compute units
    pub fn calculate_optimal_tip(&mut self, compute_units: u64) -> Result<u64> {
        if compute_units == 0 {
            return Err(PriorityFeeError::InvalidComputeUnits);
        }

        // Apply lag compensation first
        self.apply_lag_compensation();

        // Base calculation
        let base_tip = self.base_tip_per_cu
            .checked_mul(compute_units)
            .ok_or(PriorityFeeError::TipOverflow)?;

        // Apply congestion multiplier
        let congestion_multiplier = match self.congestion {
            CongestionLevel::Low => 100u16,
            CongestionLevel::Medium => 150,
            CongestionLevel::High => 250,
            CongestionLevel::Extreme => 500,
        };

        // Apply success rate adjustment (lower success = higher tip needed)
        let success_adjustment = if self.success_rate < 80 {
            100 + (80 - self.success_rate) as u16 * 2
        } else {
            100
        };

        // Combined multiplier: congestion * lag * success
        let combined_multiplier = (congestion_multiplier as u32)
            .saturating_mul(self.lag_multiplier_bp as u32)
            .saturating_mul(success_adjustment as u32);

        // Divide by 10000 (basis points normalization)
        let adjusted_tip = (base_tip as u128)
            .saturating_mul(combined_multiplier as u128)
            .saturating_div(100_000_000) as u64;

        Ok(adjusted_tip.clamp(
            MIN_TIP_PER_CU.saturating_mul(compute_units),
            MAX_TIP_PER_CU.saturating_mul(compute_units),
        ))
    }

    /// Estimate compute units for an instruction type
    pub fn estimate_cu(&self, instr_type: InstructionType) -> CuEstimate {
        match instr_type {
            InstructionType::Transfer => CuEstimate {
                instruction_type: instr_type,
                estimated_cu: 150,
                variance_percent: 5,
            },
            InstructionType::Swap => CuEstimate {
                instruction_type: instr_type,
                estimated_cu: 50_000,
                variance_percent: 20,
            },
            InstructionType::Stake => CuEstimate {
                instruction_type: instr_type,
                estimated_cu: 3_000,
                variance_percent: 10,
            },
            InstructionType::Vote => CuEstimate {
                instruction_type: instr_type,
                estimated_cu: 200,
                variance_percent: 5,
            },
            InstructionType::CustomProgram => CuEstimate {
                instruction_type: instr_type,
                estimated_cu: 100_000,
                variance_percent: 50,
            },
        }
    }

    /// Get current tip per CU recommendation
    pub const fn current_tip_per_cu(&self) -> u64 {
        self.base_tip_per_cu
    }

    /// Get congestion level
    pub const fn congestion_level(&self) -> CongestionLevel {
        self.congestion
    }
}

impl Default for PriorityFeeOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let optimizer = PriorityFeeOptimizer::new();
        assert_eq!(optimizer.current_tip_per_cu(), MIN_TIP_PER_CU);
        assert_eq!(optimizer.congestion_level(), CongestionLevel::Low);
    }

    #[test]
    fn test_cu_estimation() {
        let optimizer = PriorityFeeOptimizer::new();
        
        let transfer_cu = optimizer.estimate_cu(InstructionType::Transfer);
        assert_eq!(transfer_cu.estimated_cu, 150);
        
        let swap_cu = optimizer.estimate_cu(InstructionType::Swap);
        assert_eq!(swap_cu.estimated_cu, 50_000);
    }

    #[test]
    fn test_tip_calculation() {
        let mut optimizer = PriorityFeeOptimizer::new();
        optimizer.update_jito_metrics(100, 95);
        
        let tip = optimizer.calculate_optimal_tip(50_000).unwrap();
        assert!(tip >= MIN_TIP_PER_CU * 50_000);
        assert!(tip <= MAX_TIP_PER_CU * 50_000);
    }

    #[test]
    fn test_zero_cu_error() {
        let optimizer = PriorityFeeOptimizer::new();
        let result = optimizer.calculate_optimal_tip(0);
        assert!(matches!(result, Err(PriorityFeeError::InvalidComputeUnits)));
    }

    #[test]
    fn test_congestion_impact() {
        let mut optimizer_low = PriorityFeeOptimizer::new();
        optimizer_low.update_jito_metrics(100, 95);
        optimizer_low.update_congestion(CongestionLevel::Low);
        
        let mut optimizer_high = PriorityFeeOptimizer::new();
        optimizer_high.update_jito_metrics(100, 95);
        optimizer_high.update_congestion(CongestionLevel::High);
        
        let tip_low = optimizer_low.calculate_optimal_tip(50_000).unwrap();
        let tip_high = optimizer_high.calculate_optimal_tip(50_000).unwrap();
        
        assert!(tip_high > tip_low, "High congestion should require higher tips");
    }
}
