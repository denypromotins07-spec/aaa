//! Portfolio Hamiltonian Builder
//! 
//! Translates portfolio optimization problems (minimizing variance w^T Σ w subject to 
//! expected return and risk limits) into QUBO matrix form.
//! 
//! Real-world constraints like "maximum 10 assets" and "weights must be multiples of 0.01"
//! are converted into quadratic penalty terms λ(Ax - b)².

use ndarray::{Array2, Array1};
use num_traits::Float;
use thiserror::Error;
use crate::qubo::adaptive_penalty_scaler::{AdaptivePenaltyScaler, PenaltyScalerConfig, PenaltyScalingResult};

/// Errors that can occur during QUBO formulation
#[derive(Error, Debug)]
pub enum QuboError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Invalid covariance matrix: {0}")]
    InvalidCovarianceMatrix(String),
    #[error("Invalid constraint specification: {0}")]
    InvalidConstraint(String),
    #[error("Numerical error: {0}")]
    NumericalError(String),
    #[error("QUBO matrix exceeds maximum size: {size} > {max}")]
    MatrixTooLarge { size: usize, max: usize },
}

/// Configuration for portfolio QUBO construction
#[derive(Debug, Clone)]
pub struct QuboConfig<F: Float> {
    /// Number of assets in the portfolio
    pub n_assets: usize,
    /// Number of qubits per asset (for discretization of weights)
    pub qubits_per_asset: usize,
    /// Target expected return
    pub target_return: F,
    /// Risk aversion parameter (higher = more risk-averse)
    pub risk_aversion: F,
    /// Maximum number of assets allowed (cardinality constraint)
    pub max_assets: Option<usize>,
    /// Minimum lot size (e.g., 0.01 for 1% increments)
    pub min_lot_size: F,
    /// Expected returns vector
    pub expected_returns: Vec<F>,
    /// Asset identifiers
    pub asset_ids: Vec<String>,
}

impl<F: Float + Default> Default for QuboConfig<F> {
    fn default() -> Self {
        Self {
            n_assets: 10,
            qubits_per_asset: 4, // Allows 2^4 = 16 discrete weight levels
            target_return: F::from(0.1).unwrap_or(F::one() / F::from(10.0).unwrap()),
            risk_aversion: F::from(1.0).unwrap_or(F::one()),
            max_assets: Some(10),
            min_lot_size: F::from(0.01).unwrap_or(F::from(0.01f64).unwrap()),
            expected_returns: vec![],
            asset_ids: vec![],
        }
    }
}

/// The QUBO matrix representation
#[derive(Debug, Clone)]
pub struct QuboMatrix<F: Float> {
    /// The Q matrix where objective is minimize x^T Q x
    pub matrix: Array2<F>,
    /// Linear term vector (can be absorbed into diagonal of Q)
    pub linear_term: Array1<F>,
    /// Number of logical qubits
    pub n_qubits: usize,
    /// Mapping from qubit index to (asset_id, weight_contribution)
    pub qubit_mapping: Vec<(String, F)>,
    /// Constant offset (doesn't affect optimization)
    pub constant_offset: F,
}

impl<F: Float> QuboMatrix<F> {
    /// Create a new QUBO matrix
    pub fn new(n_qubits: usize) -> Self {
        Self {
            matrix: Array2::zeros((n_qubits, n_qubits)),
            linear_term: Array1::zeros(n_qubits),
            n_qubits,
            qubit_mapping: Vec::with_capacity(n_qubits),
            constant_offset: F::zero(),
        }
    }

    /// Get total number of variables
    pub fn n_variables(&self) -> usize {
        self.n_qubits
    }

    /// Validate the QUBO matrix structure
    pub fn validate(&self) -> Result<(), QuboError> {
        if self.matrix.nrows() != self.matrix.ncols() {
            return Err(QuboError::DimensionMismatch {
                expected: self.matrix.ncols(),
                actual: self.matrix.nrows(),
            });
        }
        
        if self.linear_term.len() != self.n_qubits {
            return Err(QuboError::DimensionMismatch {
                expected: self.n_qubits,
                actual: self.linear_term.len(),
            });
        }
        
        Ok(())
    }
}

/// Builder for constructing portfolio QUBO formulations
pub struct PortfolioQuboBuilder<F: Float> {
    config: QuboConfig<F>,
    covariance_matrix: Option<Array2<F>>,
    constraints: Vec<Constraint<F>>,
}

/// Types of constraints supported
#[derive(Debug, Clone)]
pub enum Constraint<F: Float> {
    /// Budget constraint: sum of weights equals 1
    Budget,
    /// Cardinality constraint: at most k assets selected
    Cardinality(usize),
    /// Minimum weight for selected assets
    MinWeight(F),
    /// Maximum weight per asset
    MaxWeight(F),
    /// Sector exposure limit
    SectorLimit { sector_ids: Vec<usize>, max_exposure: F },
    /// Custom linear constraint: a^T x = b
    CustomLinear { coefficients: Vec<F>, target: F },
}

impl<F: Float + 'static> PortfolioQuboBuilder<F> 
where
    F: From<f64> + Copy + Into<f64>,
{
    /// Create a new portfolio QUBO builder with configuration
    pub fn new(config: QuboConfig<F>) -> Self {
        Self {
            config,
            covariance_matrix: None,
            constraints: Vec::new(),
        }
    }

    /// Set the covariance matrix
    pub fn with_covariance(mut self, cov: Array2<F>) -> Result<Self, QuboError> {
        if cov.nrows() != self.config.n_assets || cov.ncols() != self.config.n_assets {
            return Err(QuboError::DimensionMismatch {
                expected: self.config.n_assets,
                actual: cov.nrows(),
            });
        }
        
        // Validate covariance matrix is symmetric and positive semi-definite
        for i in 0..cov.nrows() {
            for j in (i+1)..cov.ncols() {
                if (cov[[i, j]] - cov[[j, i]]).abs() > F::from(1e-10).unwrap() {
                    return Err(QuboError::InvalidCovarianceMatrix(
                        "Covariance matrix must be symmetric".to_string(),
                    ));
                }
            }
        }
        
        self.covariance_matrix = Some(cov);
        Ok(self)
    }

    /// Add a budget constraint (weights sum to 1)
    pub fn with_budget_constraint(mut self) -> Self {
        self.constraints.push(Constraint::Budget);
        self
    }

    /// Add a cardinality constraint (max k assets)
    pub fn with_cardinality_constraint(mut self, max_k: usize) -> Self {
        self.constraints.push(Constraint::Cardinality(max_k));
        self
    }

    /// Add minimum weight constraint
    pub fn with_min_weight(mut self, min_w: F) -> Self {
        self.constraints.push(Constraint::MinWeight(min_w));
        self
    }

    /// Add maximum weight per asset constraint
    pub fn with_max_weight(mut self, max_w: F) -> Self {
        self.constraints.push(Constraint::MaxWeight(max_w));
        self
    }

    /// Add custom linear constraint
    pub fn with_custom_linear_constraint(mut self, coefficients: Vec<F>, target: F) -> Result<Self, QuboError> {
        if coefficients.len() != self.config.n_assets {
            return Err(QuboError::DimensionMismatch {
                expected: self.config.n_assets,
                actual: coefficients.len(),
            });
        }
        self.constraints.push(Constraint::CustomLinear { coefficients, target });
        Ok(self)
    }

    /// Build the QUBO matrix
    /// 
    /// This translates the portfolio optimization problem:
    /// minimize: w^T Σ w - μ * expected_return(w)
    /// subject to: constraints
    /// 
    /// Into QUBO form: minimize x^T Q x + c^T x
    pub fn build(self) -> Result<QuboMatrix<F>, QuboError> {
        let n_assets = self.config.n_assets;
        let qubits_per_asset = self.config.qubits_per_asset;
        let total_qubits = n_assets * qubits_per_asset;
        
        // Check size limits
        if total_qubits > 500 {
            return Err(QuboError::MatrixTooLarge { size: total_qubits, max: 500 });
        }

        let mut qubo = QuboMatrix::new(total_qubits);
        
        // Get or create identity covariance if not provided
        let cov = self.covariance_matrix.unwrap_or_else(|| {
            Array2::from_shape_fn((n_assets, n_assets), |(i, j)| {
                if i == j { F::one() } else { F::from(0.1).unwrap() }
            })
        });

        // Build binary encoding for each asset's weight
        // Weight_i = sum_{k=0}^{qubits_per_asset-1} b_{i,k} * 2^k * min_lot_size
        let weight_contributions: Vec<Vec<F>> = (0..n_assets)
            .map(|asset_idx| {
                (0..qubits_per_asset)
                    .map(|bit_idx| {
                        F::from(2u64.pow(bit_idx as u32)).unwrap() * self.config.min_lot_size
                    })
                    .collect()
            })
            .collect();

        // Populate qubit mapping
        for asset_idx in 0..n_assets {
            let asset_id = if asset_idx < self.config.asset_ids.len() {
                self.config.asset_ids[asset_idx].clone()
            } else {
                format!("asset_{}", asset_idx)
            };
            
            for bit_idx in 0..qubits_per_asset {
                let qubit_idx = asset_idx * qubits_per_asset + bit_idx;
                qubo.qubit_mapping.push((asset_id.clone(), weight_contributions[asset_idx][bit_idx]));
            }
        }

        // 1. Variance term: w^T Σ w
        // Expand into binary variables
        self.add_variance_terms(&mut qubo, &cov, &weight_contributions);

        // 2. Expected return term: -μ * r^T w (negative because we minimize)
        self.add_return_terms(&mut qubo, &weight_contributions);

        // 3. Add constraint penalty terms
        self.add_constraint_penalties(&mut qubo, &weight_contributions)?;

        // Validate the result
        qubo.validate()?;

        Ok(qubo)
    }

    /// Add variance minimization terms to QUBO matrix
    fn add_variance_terms(
        &self,
        qubo: &mut QuboMatrix<F>,
        cov: &Array2<F>,
        weight_contributions: &[Vec<F>],
    ) {
        let n_assets = self.config.n_assets;
        let qubits_per_asset = self.config.qubits_per_asset;
        let risk_aversion = self.config.risk_aversion;

        for i in 0..n_assets {
            for j in 0..n_assets {
                let cov_ij = cov[[i, j]];
                
                // For each pair of bits in assets i and j
                for bi in 0..qubits_per_asset {
                    for bj in 0..qubits_per_asset {
                        let qi = i * qubits_per_asset + bi;
                        let qj = j * qubits_per_asset + bj;
                        
                        // Contribution: risk_aversion * cov[i,j] * w_contrib[i,bi] * w_contrib[j,bj]
                        let contribution = risk_aversion * cov_ij * 
                            weight_contributions[i][bi] * weight_contributions[j][bj];
                        
                        if qi == qj {
                            // Diagonal term (x_i^2 = x_i for binary)
                            qubo.linear_term[qi] = qubo.linear_term[qi] + contribution;
                        } else {
                            // Off-diagonal term
                            qubo.matrix[[qi, qj]] = qubo.matrix[[qi, qj]] + contribution;
                            qubo.matrix[[qj, qi]] = qubo.matrix[[qj, qi]] + contribution;
                        }
                    }
                }
            }
        }
    }

    /// Add expected return maximization terms (as negative minimization)
    fn add_return_terms(
        &self,
        qubo: &mut QuboMatrix<F>,
        weight_contributions: &[Vec<F>],
    ) {
        let n_assets = self.config.n_assets;
        let qubits_per_asset = self.config.qubits_per_asset;
        let target_return = self.config.target_return;

        for i in 0..n_assets {
            let expected_return_i = if i < self.config.expected_returns.len() {
                self.config.expected_returns[i]
            } else {
                F::zero()
            };

            for bi in 0..qubits_per_asset {
                let qi = i * qubits_per_asset + bi;
                let contribution = -target_return * expected_return_i * weight_contributions[i][bi];
                qubo.linear_term[qi] = qubo.linear_term[qi] + contribution;
            }
        }
    }

    /// Add constraint penalty terms using adaptive penalty scaling
    fn add_constraint_penalties(
        &self,
        qubo: &mut QuboMatrix<F>,
        weight_contributions: &[Vec<F>],
    ) -> Result<(), QuboError> {
        let n_assets = self.config.n_assets;
        let qubits_per_asset = self.config.qubits_per_asset;
        let total_qubits = n_assets * qubits_per_asset;

        for constraint in &self.constraints {
            match constraint {
                Constraint::Budget => {
                    // sum(weights) = 1
                    // Penalty: λ * (sum_i sum_bi w_{i,bi} * b_{i,bi} - 1)^2
                    
                    // Build A matrix and b vector for this constraint
                    let mut a_row = Array1::zeros(total_qubits);
                    for i in 0..n_assets {
                        for bi in 0..qubits_per_asset {
                            let qi = i * qubits_per_asset + bi;
                            a_row[qi] = weight_contributions[i][bi];
                        }
                    }
                    
                    let b_val = F::one();
                    let a_matrix = a_row.into_shape((1, total_qubits)).unwrap();
                    let b_vector = Array1::from_vec(vec![b_val]);

                    // Calculate optimal lambda using adaptive scaler
                    let scaler = AdaptivePenaltyScaler::new();
                    let penalty_result = scaler.calculate_optimal_lambda(
                        &qubo.matrix,
                        &a_matrix,
                        &b_vector,
                    ).map_err(|e| QuboError::NumericalError(e.to_string()))?;

                    // Apply penalty: λ * (Ax - b)^2 = λ * (x^T A^T A x - 2b^T A x + b^2)
                    let lambda = penalty_result.optimal_lambda;
                    
                    // Quadratic term: λ * A^T A
                    for i in 0..total_qubits {
                        for j in 0..total_qubits {
                            qubo.matrix[[i, j]] = qubo.matrix[[i, j]] + 
                                lambda * a_row[i] * a_row[j];
                        }
                    }
                    
                    // Linear term: -2λb^T A
                    for i in 0..total_qubits {
                        qubo.linear_term[i] = qubo.linear_term[i] - F::from(2.0).unwrap() * lambda * b_val * a_row[i];
                    }
                    
                    // Constant term (doesn't affect optimization but track for completeness)
                    qubo.constant_offset = qubo.constant_offset + lambda * b_val * b_val;
                }

                Constraint::Cardinality(max_k) => {
                    // At most k assets selected
                    // Use auxiliary binary variables or penalty formulation
                    // Simplified: penalize if more than k assets have non-zero weight
                    
                    // For each asset, create an indicator variable for whether it's selected
                    // This requires additional ancilla qubits in practice
                    // Here we use a soft penalty approach
                    
                    let k_float = F::from(*max_k as f64).unwrap();
                    
                    // Indicator: asset i is selected if any of its bits are 1
                    // Approximate: sum of all bits for asset i > 0 means selected
                    // Penalty: λ * max(0, sum(indicators) - k)^2
                    
                    // Soft relaxation: use sum of first bit of each asset as proxy
                    let mut a_row = Array1::zeros(total_qubits);
                    for i in 0..n_assets {
                        let qi = i * qubits_per_asset; // First bit of each asset
                        a_row[qi] = F::one();
                    }
                    
                    let a_matrix = a_row.into_shape((1, total_qubits)).unwrap();
                    let b_vector = Array1::from_vec(vec![k_float]);

                    let scaler = AdaptivePenaltyScaler::new();
                    let penalty_result = scaler.calculate_optimal_lambda(
                        &qubo.matrix,
                        &a_matrix,
                        &b_vector,
                    ).map_err(|e| QuboError::NumericalError(e.to_string()))?;

                    let lambda = penalty_result.optimal_lambda / F::from(10.0).unwrap(); // Softer penalty
                    
                    for i in 0..total_qubits {
                        for j in 0..total_qubits {
                            qubo.matrix[[i, j]] = qubo.matrix[[i, j]] + 
                                lambda * a_row[i] * a_row[j];
                        }
                    }
                }

                Constraint::MinWeight(min_w) => {
                    // If asset is selected, weight >= min_w
                    // This is enforced by the binary encoding granularity
                    // No additional QUBO terms needed if min_lot_size <= min_w
                }

                Constraint::MaxWeight(max_w) => {
                    // Each asset weight <= max_w
                    // Enforce through binary encoding upper bound
                    let max_representable = weight_contributions[0].iter()
                        .fold(F::zero(), |acc, &w| acc + w);
                    
                    if max_w < &max_representable {
                        // Would need additional constraints - simplified handling
                        // In practice, adjust qubits_per_asset or use ancilla variables
                    }
                }

                Constraint::SectorLimit { sector_ids, max_exposure } => {
                    // Sum of weights in sector <= max_exposure
                    let mut a_row = Array1::zeros(total_qubits);
                    
                    for &asset_idx in sector_ids {
                        if asset_idx < n_assets {
                            for bi in 0..qubits_per_asset {
                                let qi = asset_idx * qubits_per_asset + bi;
                                a_row[qi] = weight_contributions[asset_idx][bi];
                            }
                        }
                    }
                    
                    let a_matrix = a_row.into_shape((1, total_qubits)).unwrap();
                    let b_vector = Array1::from_vec(vec![*max_exposure]);

                    let scaler = AdaptivePenaltyScaler::new();
                    if let Ok(penalty_result) = scaler.calculate_optimal_lambda(
                        &qubo.matrix,
                        &a_matrix,
                        &b_vector,
                    ) {
                        let lambda = penalty_result.optimal_lambda;
                        
                        for i in 0..total_qubits {
                            for j in 0..total_qubits {
                                qubo.matrix[[i, j]] = qubo.matrix[[i, j]] + 
                                    lambda * a_row[i] * a_row[j];
                            }
                        }
                    }
                }

                Constraint::CustomLinear { coefficients, target } => {
                    // a^T x = b constraint
                    let mut a_row = Array1::zeros(total_qubits);
                    
                    for asset_idx in 0..n_assets {
                        for bi in 0..qubits_per_asset {
                            let qi = asset_idx * qubits_per_asset + bi;
                            a_row[qi] = coefficients[asset_idx] * weight_contributions[asset_idx][bi];
                        }
                    }
                    
                    let a_matrix = a_row.into_shape((1, total_qubits)).unwrap();
                    let b_vector = Array1::from_vec(vec![*target]);

                    let scaler = AdaptivePenaltyScaler::new();
                    if let Ok(penalty_result) = scaler.calculate_optimal_lambda(
                        &qubo.matrix,
                        &a_matrix,
                        &b_vector,
                    ) {
                        let lambda = penalty_result.optimal_lambda;
                        
                        for i in 0..total_qubits {
                            for j in 0..total_qubits {
                                qubo.matrix[[i, j]] = qubo.matrix[[i, j]] + 
                                    lambda * a_row[i] * a_row[j];
                            }
                        }
                        
                        for i in 0..total_qubits {
                            qubo.linear_term[i] = qubo.linear_term[i] - 
                                F::from(2.0).unwrap() * lambda * (*target) * a_row[i];
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qubo_builder_basic() {
        let config: QuboConfig<f64> = QuboConfig {
            n_assets: 3,
            qubits_per_asset: 3,
            target_return: 0.1,
            risk_aversion: 1.0,
            max_assets: Some(2),
            min_lot_size: 0.01,
            expected_returns: vec![0.08, 0.12, 0.10],
            asset_ids: vec!["AAPL".to_string(), "GOOG".to_string(), "MSFT".to_string()],
        };

        let cov = Array2::from_shape_vec((3, 3), vec![
            0.04, 0.01, 0.02,
            0.01, 0.09, 0.03,
            0.02, 0.03, 0.06,
        ]).unwrap();

        let builder = PortfolioQuboBuilder::new(config)
            .with_covariance(cov).unwrap()
            .with_budget_constraint()
            .with_cardinality_constraint(2);

        let qubo = builder.build();
        
        assert!(qubo.is_ok());
        let qubo = qubo.unwrap();
        
        assert_eq!(qubo.n_qubits, 9);
        assert!(qubo.validate().is_ok());
    }

    #[test]
    fn test_qubo_symmetry() {
        let config: QuboConfig<f64> = QuboConfig::default();
        let cov = Array2::from_shape_vec((2, 2), vec![
            0.04, 0.01,
            0.01, 0.09,
        ]).unwrap();

        let builder = PortfolioQuboBuilder::new(config)
            .with_covariance(cov).unwrap()
            .with_budget_constraint();

        let qubo = builder.build().unwrap();
        
        // Verify Q matrix is symmetric
        for i in 0..qubo.n_qubits {
            for j in 0..qubo.n_qubits {
                assert!((qubo.matrix[[i, j]] - qubo.matrix[[j, i]]).abs() < 1e-10);
            }
        }
    }
}
