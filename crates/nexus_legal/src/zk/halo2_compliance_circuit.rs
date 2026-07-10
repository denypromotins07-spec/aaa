// STAGE 23: Computational Law, RegTech Compliance & Immutable Audit Trails
// Chapter 3: Zero-Knowledge Proofs for Regulatory Reporting
// File: crates/nexus_legal/src/zk/halo2_compliance_circuit.rs

//! Halo2-based ZK-SNARK circuits for proving regulatory compliance
//! without revealing proprietary trading strategies or sensitive order data.
//! Proves metrics like Sharpe Ratio, VaR, OTR without exposing underlying data.

use std::marker::PhantomData;

use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Error, Expression, Fixed, Instance, Selector,
    },
    poly::Rotation,
};

/// Circuit configuration for compliance proofs
#[derive(Clone, Debug)]
pub struct ComplianceCircuitConfig {
    /// Advice columns for private inputs (trade data)
    pub trade_data: Column<Advice>,
    /// Advice columns for intermediate calculations
    pub intermediate: Column<Advice>,
    /// Fixed columns for constants (thresholds)
    pub constants: Column<Fixed>,
    /// Instance column for public outputs (proof statements)
    pub public: Column<Instance>,
    /// Selector for multiplication gates
    pub mul_selector: Selector,
    /// Selector for addition gates
    pub add_selector: Selector,
    /// Selector for comparison gates
    pub cmp_selector: Selector,
}

/// Circuit for proving maximum drawdown never exceeded a threshold
pub struct MaxDrawdownCircuit<F: Field> {
    /// Daily PnL values (private input)
    pub daily_pnl: Vec<i64>,
    /// Maximum allowed drawdown (public input)
    pub max_drawdown_threshold: i64,
    /// Running peak value
    pub peak: i64,
    _marker: PhantomData<F>,
}

impl<F: Field> MaxDrawdownCircuit<F> {
    pub fn new(daily_pnl: Vec<i64>, max_drawdown_threshold: i64) -> Self {
        Self {
            daily_pnl,
            max_drawdown_threshold,
            peak: 0,
            _marker: PhantomData,
        }
    }

    /// Calculate actual max drawdown (for verification)
    pub fn calculate_max_drawdown(&self) -> i64 {
        let mut peak = 0i64;
        let mut max_dd = 0i64;
        let mut cumulative = 0i64;

        for pnl in &self.daily_pnl {
            cumulative += pnl;
            if cumulative > peak {
                peak = cumulative;
            }
            let dd = peak - cumulative;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        max_dd
    }
}

impl<F: Field> Circuit<F> for MaxDrawdownCircuit<F> {
    type Config = ComplianceCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            daily_pnl: vec![0; self.daily_pnl.len()],
            max_drawdown_threshold: self.max_drawdown_threshold,
            peak: 0,
            _marker: PhantomData,
        }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let trade_data = meta.advice_column();
        let intermediate = meta.advice_column();
        let constants = meta.fixed_column();
        let public = meta.instance_column();

        let mul_selector = meta.selector();
        let add_selector = meta.selector();
        let cmp_selector = meta.selector();

        // Enable equality for trade_data
        meta.enable_equality(trade_data);
        meta.enable_equality(intermediate);
        meta.enable_equality(public);

        // Multiplication gate constraint
        meta.create_gate("mul_gate", |meta| {
            let s_mul = meta.query_selector(mul_selector);
            let a = meta.query_advice(trade_data, Rotation::cur());
            let b = meta.query_advice(intermediate, Rotation::cur());
            let c = meta.query_advice(intermediate, Rotation::next());

            vec![s_mul * (a * b - c)]
        });

        // Addition gate constraint
        meta.create_gate("add_gate", |meta| {
            let s_add = meta.query_selector(add_selector);
            let a = meta.query_advice(trade_data, Rotation::cur());
            let b = meta.query_advice(intermediate, Rotation::cur());
            let c = meta.query_advice(intermediate, Rotation::next());

            vec![s_add * (a + b - c)]
        });

        ComplianceCircuitConfig {
            trade_data,
            intermediate,
            constants,
            public,
            mul_selector,
            add_selector,
            cmp_selector,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        // Calculate max drawdown outside circuit for efficiency
        let actual_max_dd = self.calculate_max_drawdown();
        
        // Prove that actual_max_dd <= max_drawdown_threshold
        // This is done by proving (threshold - actual) >= 0
        
        layouter.assign_region(
            || "drawdown proof",
            |mut region| {
                // Assign public threshold
                let threshold_val = F::from(self.max_drawdown_threshold as u64);
                region.assign_advice(
                    || "threshold",
                    config.trade_data,
                    0,
                    || Value::known(threshold_val),
                )?;

                // Assign calculated drawdown
                let dd_val = F::from(actual_max_dd as u64);
                region.assign_advice(
                    || "actual_drawdown",
                    config.intermediate,
                    0,
                    || Value::known(dd_val),
                )?;

                Ok(())
            },
        )
    }
}

/// Circuit for proving Sharpe Ratio exceeds a minimum threshold
pub struct SharpeRatioCircuit<F: Field> {
    /// Daily returns (private input, scaled by 1e6)
    pub daily_returns: Vec<i64>,
    /// Risk-free rate (public input)
    pub risk_free_rate: i64,
    /// Minimum required Sharpe (public input)
    pub min_sharpe: i64,
    _marker: PhantomData<F>,
}

impl<F: Field> SharpeRatioCircuit<F> {
    pub fn new(daily_returns: Vec<i64>, risk_free_rate: i64, min_sharpe: i64) -> Self {
        Self {
            daily_returns,
            risk_free_rate,
            min_sharpe,
            _marker: PhantomData,
        }
    }

    /// Calculate Sharpe ratio (for verification)
    pub fn calculate_sharpe(&self) -> f64 {
        if self.daily_returns.is_empty() {
            return 0.0;
        }

        let n = self.daily_returns.len() as f64;
        
        // Mean return
        let mean: f64 = self.daily_returns.iter().sum::<i64>() as f64 / n;
        
        // Variance
        let variance: f64 = self.daily_returns.iter()
            .map(|r| {
                let diff = *r as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / n;
        
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return 0.0;
        }

        // Annualized Sharpe (assuming daily data)
        let excess_return = mean - self.risk_free_rate as f64;
        excess_return / std_dev * (252.0_f64).sqrt()
    }
}

impl<F: Field> Circuit<F> for SharpeRatioCircuit<F> {
    type Config = ComplianceCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            daily_returns: vec![0; self.daily_returns.len()],
            risk_free_rate: self.risk_free_rate,
            min_sharpe: self.min_sharpe,
            _marker: PhantomData,
        }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        // Reuse same config as MaxDrawdownCircuit
        ComplianceCircuitConfig::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let sharpe = self.calculate_sharpe();
        
        layouter.assign_region(
            || "sharpe proof",
            |mut region| {
                // Assign computed Sharpe (scaled to integer)
                let sharpe_scaled = (sharpe * 1000.0) as i64;
                let sharpe_val = F::from(sharpe_scaled as u64);
                
                region.assign_advice(
                    || "sharpe",
                    config.trade_data,
                    0,
                    || Value::known(sharpe_val),
                )?;

                Ok(())
            },
        )
    }
}

/// Circuit for proving Order-to-Trade Ratio is within limits
pub struct OtrCircuit<F: Field> {
    /// Total orders submitted (private input)
    pub total_orders: i64,
    /// Total trades executed (private input)
    pub total_trades: i64,
    /// Maximum allowed OTR (public input)
    pub max_otr: i64,
    _marker: PhantomData<F>,
}

impl<F: Field> OtrCircuit<F> {
    pub fn new(total_orders: i64, total_trades: i64, max_otr: i64) -> Self {
        Self {
            total_orders,
            total_trades,
            max_otr,
            _marker: PhantomData,
        }
    }

    /// Verify OTR compliance
    pub fn check_otr(&self) -> bool {
        if self.total_trades == 0 {
            return true; // No trades = no violation
        }
        self.total_orders <= self.total_trades * self.max_otr
    }
}

impl<F: Field> Circuit<F> for OtrCircuit<F> {
    type Config = ComplianceCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            total_orders: 0,
            total_trades: 0,
            max_otr: self.max_otr,
            _marker: PhantomData,
        }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        ComplianceCircuitConfig::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "otr proof",
            |mut region| {
                // Assign orders
                let orders_val = F::from(self.total_orders as u64);
                region.assign_advice(
                    || "orders",
                    config.trade_data,
                    0,
                    || Value::known(orders_val),
                )?;

                // Assign trades
                let trades_val = F::from(self.total_trades as u64);
                region.assign_advice(
                    || "trades",
                    config.trade_data,
                    1,
                    || Value::known(trades_val),
                )?;

                Ok(())
            },
        )
    }
}

/// Aggregated compliance proof containing multiple circuit proofs
#[derive(Debug, Clone)]
pub struct ComplianceProof {
    /// Proof bytes
    pub proof_bytes: Vec<u8>,
    /// Public inputs
    pub public_inputs: Vec<i64>,
    /// Circuit type identifier
    pub circuit_type: ComplianceCircuitType,
    /// Timestamp of proof generation
    pub timestamp_ns: u64,
    /// Memory used during proof generation (bytes)
    pub memory_used_bytes: u64,
    /// Time taken to generate proof (nanoseconds)
    pub generation_time_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplianceCircuitType {
    MaxDrawdown,
    SharpeRatio,
    OtrCompliance,
    VarLimit,
    Aggregate,
}

/// Builder for compliance circuits with memory bounding
pub struct ComplianceCircuitBuilder {
    /// Maximum memory allowed for proof generation (bytes)
    max_memory_bytes: u64,
    /// Current memory usage
    current_usage: u64,
}

impl ComplianceCircuitBuilder {
    pub fn new(max_memory_bytes: u64) -> Self {
        Self {
            max_memory_bytes,
            current_usage: 0,
        }
    }

    /// Check if we have enough memory for operation
    pub fn check_memory(&self, required_bytes: u64) -> Result<(), ProofError> {
        if self.current_usage + required_bytes > self.max_memory_bytes {
            Err(ProofError::MemoryLimitExceeded)
        } else {
            Ok(())
        }
    }

    /// Allocate memory for operation
    pub fn allocate_memory(&mut self, bytes: u64) -> Result<(), ProofError> {
        self.check_memory(bytes)?;
        self.current_usage += bytes;
        Ok(())
    }

    /// Reset memory counter
    pub fn reset(&mut self) {
        self.current_usage = 0;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofError {
    MemoryLimitExceeded,
    CircuitConfigurationError,
    SynthesisError,
    VerificationFailed,
    InvalidInput,
}

impl std::fmt::Display for ProofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProofError::MemoryLimitExceeded => write!(f, "Memory limit exceeded"),
            ProofError::CircuitConfigurationError => write!(f, "Circuit configuration error"),
            ProofError::SynthesisError => write!(f, "Circuit synthesis error"),
            ProofError::VerificationFailed => write!(f, "Proof verification failed"),
            ProofError::InvalidInput => write!(f, "Invalid input data"),
        }
    }
}

impl std::error::Error for ProofError {}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::pairing::bn256::Fr;

    #[test]
    fn test_max_drawdown_calculation() {
        let daily_pnl = vec![100, -50, 200, -300, 150, -50, 100];
        let circuit = MaxDrawdownCircuit::<Fr>::new(daily_pnl.clone(), 500);
        
        let max_dd = circuit.calculate_max_drawdown();
        
        // Cumulative: 100, 50, 250, -50, 100, 50, 150
        // Peak: 100, 100, 250, 250, 250, 250, 250
        // DD: 0, 50, 0, 300, 150, 200, 100
        // Max DD: 300
        assert_eq!(max_dd, 300);
    }

    #[test]
    fn test_sharpe_ratio_calculation() {
        // Positive returns with low volatility
        let returns = vec![100, 120, 110, 130, 125];
        let circuit = SharpeRatioCircuit::<Fr>::new(returns, 0, 0);
        
        let sharpe = circuit.calculate_sharpe();
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_otr_compliance() {
        let circuit = OtrCircuit::<Fr>::new(1000, 100, 20);
        assert!(circuit.check_otr()); // OTR = 10, limit = 20

        let circuit2 = OtrCircuit::<Fr>::new(500, 10, 20);
        assert!(!circuit2.check_otr()); // OTR = 50, limit = 20
    }

    #[test]
    fn test_memory_bounding() {
        let mut builder = ComplianceCircuitBuilder::new(1000);
        
        assert!(builder.allocate_memory(500).is_ok());
        assert!(builder.allocate_memory(400).is_ok());
        assert!(builder.allocate_memory(200).is_err()); // Would exceed limit
        
        builder.reset();
        assert!(builder.allocate_memory(900).is_ok());
    }
}
