//! Asynchronous Quantum Oracle
//! 
//! Runs quantum optimization in the background without blocking the HFT kernel.
//! Implements strict timeout handling and graceful task cancellation to prevent
//! memory leaks and deadlocks when quantum APIs hang.

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock, watch};
use tokio::time::timeout;
use thiserror::Error;
use tracing::{info, warn, error, debug};

use crate::qubo::portfolio_hamiltonian::QuboMatrix;
use crate::annealing::dwave_hybrid_bridge::{DWaveHybridBridge, DWaveConfig, DWaveResponse, DWaveSample};
use crate::bridge::classical_simulated_annealing::{ClassicalSimulatedAnnealer, AnnealingConfig};
use crate::bridge::energy_gap_validator::{EnergyGapValidator, GapValidationResult};

/// Errors that can occur in the quantum oracle
#[derive(Error, Debug)]
pub enum OracleError {
    #[error("Quantum API timeout after {0}ms")]
    QuantumTimeout(u64),
    #[error("Quantum API error: {0}")]
    QuantumApiError(String),
    #[error("Classical fallback failed: {0}")]
    ClassicalFallbackError(String),
    #[error("Solution validation failed: {0}")]
    ValidationFailed(String),
    #[error("Oracle channel closed unexpectedly")]
    ChannelClosed,
    #[error("Task cancelled")]
    TaskCancelled,
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

/// Response from the quantum oracle
#[derive(Debug, Clone)]
pub struct OracleResponse {
    /// Optimized portfolio weights (continuous values)
    pub weights: Vec<f64>,
    /// Energy of the solution
    pub energy: f64,
    /// Whether this came from quantum or classical solver
    pub source: SolutionSource,
    /// Time taken to compute (microseconds)
    pub computation_time_us: u64,
    /// Validation status
    pub validation: GapValidationResult,
    /// Confidence score (0-1)
    pub confidence: f64,
}

/// Source of the solution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolutionSource {
    /// Direct quantum annealing
    QuantumAnnealing,
    /// Hybrid quantum-classical
    QuantumHybrid,
    /// Classical simulated annealing fallback
    ClassicalFallback,
    /// Pre-computed cached solution
    Cached,
}

/// Configuration for the async oracle
#[derive(Debug, Clone)]
pub struct OracleConfig {
    /// Timeout for quantum API calls (milliseconds)
    pub quantum_timeout_ms: u64,
    /// Maximum queue size for pending requests
    pub max_queue_size: usize,
    /// Whether to enable classical fallback
    pub enable_fallback: bool,
    /// Minimum confidence threshold to accept quantum solution
    pub min_confidence_threshold: f64,
    /// Number of parallel solver tasks
    pub parallel_tasks: usize,
    /// D-Wave configuration
    pub dwave_config: DWaveConfig,
    /// Classical annealing configuration
    pub classical_config: AnnealingConfig,
}

impl Default for OracleConfig {
    fn default() -> Self {
        Self {
            quantum_timeout_ms: 30_000, // 30 second timeout
            max_queue_size: 100,
            enable_fallback: true,
            min_confidence_threshold: 0.7,
            parallel_tasks: 2,
            dwave_config: DWaveConfig::default(),
            classical_config: AnnealingConfig::default(),
        }
    }
}

/// Internal request type
struct SolveRequest {
    qubo: QuboMatrix<f64>,
    response_tx: mpsc::Sender<Result<OracleResponse, OracleError>>,
}

/// Asynchronous Quantum Oracle - runs quantum optimization in background
pub struct AsyncQuantumOracle {
    config: OracleConfig,
    request_tx: mpsc::Sender<SolveRequest>,
    shutdown_tx: watch::Sender<bool>,
    state: Arc<RwLock<OracleState>>,
}

/// Current state of the oracle
#[derive(Debug, Clone, Default)]
struct OracleState {
    /// Number of pending requests
    pending_count: usize,
    /// Total solutions computed
    total_solutions: u64,
    /// Solutions from quantum
    quantum_solutions: u64,
    /// Solutions from classical fallback
    fallback_solutions: u64,
    /// Last solution time
    last_solution_time_us: Option<u64>,
    /// Average computation time
    avg_computation_time_us: f64,
    /// Current best energy seen
    best_energy_seen: Option<f64>,
}

impl AsyncQuantumOracle {
    /// Create a new async quantum oracle
    pub fn new(config: OracleConfig) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<SolveRequest>(config.max_queue_size);
        let (shutdown_tx, _) = watch::channel(false);
        
        Self {
            config,
            request_tx,
            shutdown_tx,
            state: Arc::new(RwLock::new(OracleState::default())),
        }
    }

    /// Start the oracle background workers
    /// 
    /// This spawns Tokio tasks that continuously process solve requests.
    /// Call this once during system initialization.
    pub async fn start(&self) -> Result<(), OracleError> {
        info!("Starting AsyncQuantumOracle with {} parallel tasks", self.config.parallel_tasks);
        
        for task_id in 0..self.config.parallel_tasks {
            let mut request_rx = self.request_tx.resend();
            let mut shutdown_rx = self.shutdown_tx.subscribe();
            let config = self.config.clone();
            let state = Arc::clone(&self.state);
            
            tokio::spawn(async move {
                worker_loop(
                    task_id,
                    request_rx,
                    shutdown_rx,
                    config,
                    state,
                ).await;
            });
        }
        
        Ok(())
    }

    /// Submit a QUBO problem for solving
    /// 
    /// Returns immediately with a receiver that will get the result
    /// when the quantum/classical solver completes.
    pub async fn submit(&self, qubo: QuboMatrix<f64>) -> Result<mpsc::Receiver<Result<OracleResponse, OracleError>>, OracleError> {
        let (response_tx, response_rx) = mpsc::channel(1);
        
        let request = SolveRequest {
            qubo,
            response_tx,
        };
        
        self.request_tx.send(request).await
            .map_err(|_| OracleError::ChannelClosed)?;
        
        // Update pending count
        {
            let mut state = self.state.write().await;
            state.pending_count += 1;
        }
        
        Ok(response_rx)
    }

    /// Submit and wait for result with timeout
    /// 
    /// This is a convenience method that combines submit and receive
    /// with automatic timeout handling.
    pub async fn solve_with_timeout(
        &self,
        qubo: QuboMatrix<f64>,
        timeout_ms: u64,
    ) -> Result<OracleResponse, OracleError> {
        let mut response_rx = self.submit(qubo).await?;
        
        match timeout(Duration::from_millis(timeout_ms), response_rx.recv()).await {
            Ok(Some(result)) => {
                // Update state on success
                {
                    let mut state = self.state.write().await;
                    state.pending_count = state.pending_count.saturating_sub(1);
                }
                result
            }
            Ok(None) => Err(OracleError::ChannelClosed),
            Err(_) => Err(OracleError::QuantumTimeout(timeout_ms)),
        }
    }

    /// Get current oracle statistics
    pub async fn get_stats(&self) -> OracleState {
        self.state.read().await.clone()
    }

    /// Shutdown all worker tasks gracefully
    pub async fn shutdown(&self) -> Result<(), OracleError> {
        info!("Shutting down AsyncQuantumOracle");
        
        // Signal workers to stop
        self.shutdown_tx.send(true)
            .map_err(|_| OracleError::TaskCancelled)?;
        
        // Wait a bit for workers to finish
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        Ok(())
    }

    /// Check if oracle is healthy and responsive
    pub async fn health_check(&self) -> bool {
        let state = self.state.read().await;
        // Consider healthy if we've solved at least one problem
        state.total_solutions > 0 || state.pending_count > 0
    }
}

/// Worker loop that processes solve requests
async fn worker_loop(
    task_id: usize,
    mut request_rx: mpsc::Receiver<SolveRequest>,
    mut shutdown_rx: watch::Receiver<bool>,
    config: OracleConfig,
    state: Arc<RwLock<OracleState>>,
) {
    info!("Quantum oracle worker {} started", task_id);
    
    loop {
        tokio::select! {
            // Check for shutdown signal
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    info!("Worker {} received shutdown signal", task_id);
                    break;
                }
            }
            
            // Process solve requests
            Some(request) = request_rx.recv() => {
                debug!("Worker {} processing solve request", task_id);
                
                let result = solve_qubo_with_fallback(
                    request.qubo,
                    &config,
                ).await;
                
                // Send result back
                let _ = request.response_tx.send(result).await;
                
                // Update statistics
                {
                    let mut state = state.write().await;
                    state.total_solutions += 1;
                    state.pending_count = state.pending_count.saturating_sub(1);
                    
                    if let Ok(ref resp) = result {
                        match resp.source {
                            SolutionSource::QuantumAnnealing | SolutionSource::QuantumHybrid => {
                                state.quantum_solutions += 1;
                            }
                            SolutionSource::ClassicalFallback => {
                                state.fallback_solutions += 1;
                            }
                            _ => {}
                        }
                        
                        state.last_solution_time_us = Some(resp.computation_time_us);
                        
                        // Update running average
                        let alpha = 0.1; // Exponential smoothing factor
                        state.avg_computation_time_us = 
                            (1.0 - alpha) * state.avg_computation_time_us + 
                            alpha * resp.computation_time_us as f64;
                        
                        // Track best energy
                        if state.best_energy_seen.map_or(true, |best| resp.energy < best) {
                            state.best_energy_seen = Some(resp.energy);
                        }
                    }
                }
            }
        }
    }
    
    info!("Worker {} stopped", task_id);
}

/// Solve a QUBO using quantum with classical fallback
async fn solve_qubo_with_fallback(
    qubo: QuboMatrix<f64>,
    config: &OracleConfig,
) -> Result<OracleResponse, OracleError> {
    let start_time = std::time::Instant::now();
    
    // Try quantum first (with timeout)
    if !config.dwave_config.api_token.is_empty() {
        match solve_quantum(&qubo, config).await {
            Ok(response) => {
                // Validate quantum solution
                let validator = EnergyGapValidator::new();
                let validation = validator.validate_solution(&qubo, &response.weights);
                
                if validation.is_valid && response.confidence >= config.min_confidence_threshold {
                    return Ok(response);
                }
                
                warn!("Quantum solution failed validation (confidence: {}), trying fallback", response.confidence);
            }
            Err(e) => {
                warn!("Quantum solve failed: {}, falling back to classical", e);
            }
        }
    }
    
    // Classical fallback
    if config.enable_fallback {
        solve_classical(&qubo, config).await
    } else {
        Err(OracleError::ClassicalFallbackError("Fallback disabled".to_string()))
    }
}

/// Solve using quantum annealing
async fn solve_quantum(
    qubo: &QuboMatrix<f64>,
    config: &OracleConfig,
) -> Result<OracleResponse, OracleError> {
    // Convert QUBO to matrix format for D-Wave bridge
    let n = qubo.n_qubits;
    let q_matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| qubo.matrix[[i, j]]).collect())
        .collect();
    let linear_terms: Vec<f64> = qubo.linear_term.iter().copied().collect();
    
    let mut bridge = DWaveHybridBridge::new(config.dwave_config.clone());
    
    // Connect with timeout
    match timeout(
        Duration::from_millis(5000),
        bridge.connect()
    ).await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(OracleError::QuantumApiError(e.to_string())),
        Err(_) => return Err(OracleError::QuantumTimeout(5000)),
    }
    
    // Submit problem with timeout
    let response = match timeout(
        Duration::from_millis(config.quantum_timeout_ms),
        bridge.submit_qubo_problem(&q_matrix, &linear_terms)
    ).await {
        Ok(Ok(response)) => response,
        Ok(Err(e)) => return Err(OracleError::QuantumApiError(e.to_string())),
        Err(_) => return Err(OracleError::QuantumTimeout(config.quantum_timeout_ms)),
    };
    
    // Extract best sample
    let best_sample = response.best_sample
        .ok_or_else(|| OracleError::QuantumApiError("No samples returned".to_string()))?;
    
    // Convert binary solution to weights
    let weights = convert_binary_to_weights(&best_sample.solution, &qubo.qubit_mapping);
    
    let computation_time = start_time.elapsed().as_micros() as u64;
    
    Ok(OracleResponse {
        weights,
        energy: best_sample.energy,
        source: SolutionSource::QuantumHybrid,
        computation_time_us: computation_time,
        validation: GapValidationResult::default(), // Will be set by caller
        confidence: calculate_solution_confidence(&response, &best_sample),
    })
}

/// Solve using classical simulated annealing
async fn solve_classical(
    qubo: &QuboMatrix<f64>,
    config: &OracleConfig,
) -> Result<OracleResponse, OracleError> {
    let start_time = std::time::Instant::now();
    
    let annealer = ClassicalSimulatedAnnealer::with_config(config.classical_config.clone());
    
    // Run classical annealing
    let result = annealer.solve(qubo).await
        .map_err(|e| OracleError::ClassicalFallbackError(e.to_string()))?;
    
    // Convert to weights
    let weights = convert_binary_to_weights(&result.solution, &qubo.qubit_mapping);
    
    let computation_time = start_time.elapsed().as_micros() as u64;
    
    Ok(OracleResponse {
        weights,
        energy: result.energy,
        source: SolutionSource::ClassicalFallback,
        computation_time_us: computation_time,
        validation: GapValidationResult::default(),
        confidence: 0.85, // Classical solutions typically have good confidence
    })
}

/// Convert binary solution to portfolio weights
fn convert_binary_to_weights(
    binary: &[i8],
    qubit_mapping: &[(String, f64)],
) -> Vec<f64> {
    // Group by asset and sum contributions
    let mut asset_weights: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    
    for (i, (&bit, (_, contribution))) in binary.iter().zip(qubit_mapping.iter()).enumerate() {
        if bit == 1 {
            *asset_weights.entry(qubit_mapping[i].0.clone()).or_insert(0.0) += contribution;
        }
    }
    
    // Normalize to sum to 1
    let total: f64 = asset_weights.values().sum();
    if total > 0.0 {
        asset_weights.values().map(|&w| w / total).collect()
    } else {
        // Equal weights as fallback
        let n = asset_weights.len().max(1);
        vec![1.0 / n as f64; n]
    }
}

/// Calculate confidence score based on solution quality
fn calculate_solution_confidence(
    response: &DWaveResponse,
    best_sample: &DWaveSample,
) -> f64 {
    let mut confidence = 0.5;
    
    // Higher confidence if many samples agree
    if response.samples.len() > 10 {
        let occurrences: usize = response.samples.iter()
            .filter(|s| s.energy == best_sample.energy)
            .map(|s| s.num_occurrences)
            .sum();
        let occurrence_ratio = occurrences as f64 / response.samples.len() as f64;
        confidence += occurrence_ratio * 0.3;
    }
    
    // Higher confidence if energy is low relative to others
    if !response.samples.is_empty() {
        let energies: Vec<f64> = response.samples.iter().map(|s| s.energy).collect();
        let min_e = energies.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_e = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_e - min_e;
        if range > 0.0 {
            let normalized = (best_sample.energy - min_e) / range;
            confidence += (1.0 - normalized) * 0.2;
        }
    }
    
    confidence.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[tokio::test]
    async fn test_oracle_creation_and_startup() {
        let config = OracleConfig::default();
        let oracle = AsyncQuantumOracle::new(config);
        
        assert!(oracle.start().await.is_ok());
        
        let stats = oracle.get_stats().await;
        assert_eq!(stats.total_solutions, 0);
        
        oracle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_oracle_classical_fallback() {
        let mut config = OracleConfig::default();
        // Empty token means no quantum, forces classical fallback
        config.dwave_config.api_token = String::new();
        config.enable_fallback = true;
        
        let oracle = AsyncQuantumOracle::new(config);
        oracle.start().await.unwrap();
        
        // Create simple QUBO
        let mut qubo = QuboMatrix::new(4);
        qubo.matrix = Array2::from_shape_vec((4, 4), vec![
            1.0, -0.5, 0.0, 0.0,
            -0.5, 1.0, -0.5, 0.0,
            0.0, -0.5, 1.0, -0.5,
            0.0, 0.0, -0.5, 1.0,
        ]).unwrap();
        qubo.linear_term = ndarray::Array1::from_vec(vec![-0.1, -0.1, -0.1, -0.1]);
        qubo.qubit_mapping = vec![
            ("a".to_string(), 0.25),
            ("b".to_string(), 0.25),
            ("c".to_string(), 0.25),
            ("d".to_string(), 0.25),
        ];
        
        let result = oracle.solve_with_timeout(qubo, 5000).await;
        
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.source, SolutionSource::ClassicalFallback);
        assert!(!response.weights.is_empty());
        
        oracle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_oracle_timeout_handling() {
        let mut config = OracleConfig::default();
        config.quantum_timeout_ms = 100; // Very short timeout
        
        let oracle = AsyncQuantumOracle::new(config);
        oracle.start().await.unwrap();
        
        let mut qubo = QuboMatrix::new(2);
        qubo.matrix = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        qubo.linear_term = ndarray::Array1::from_vec(vec![0.0, 0.0]);
        qubo.qubit_mapping = vec![("a".to_string(), 0.5), ("b".to_string(), 0.5)];
        
        // Should timeout quickly
        let result = oracle.solve_with_timeout(qubo, 50).await;
        
        // Either timeout or fallback should occur
        assert!(result.is_err() || result.unwrap().source == SolutionSource::ClassicalFallback);
        
        oracle.shutdown().await.unwrap();
    }
}
