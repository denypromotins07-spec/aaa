//! Integrated Information Theory (IIT) Phi Calculator
//! 
//! Approximates the Φ (Phi) value of the organoid's causal network
//! to detect emergent complex dynamics that may indicate sentience-like
//! behavior. Triggers bio-containment halt if threshold exceeded.

use core::f64;

/// Maximum number of nodes in the causal network
pub const MAX_NODES: usize = 256;

/// Default Phi threshold for containment trigger
const DEFAULT_PHI_THRESHOLD: f64 = 0.5;

/// Critical Phi threshold (immediate halt)
const CRITICAL_PHI_THRESHOLD: f64 = 1.0;

/// Error types for IIT computation
#[derive(Debug, Clone, Copy)]
pub enum IitError {
    InvalidNodeIndex,
    MatrixDimensionMismatch,
    ComputationFailed,
    ThresholdExceeded,
    NotInitialized,
}

/// Causal influence matrix (transition probability matrix)
#[repr(C, align(64))]
pub struct CausalMatrix {
    /// Transition probabilities [to x from]
    probabilities: [[f64; MAX_NODES]; MAX_NODES],
    /// Number of valid nodes
    num_nodes: usize,
}

impl CausalMatrix {
    /// Create a new causal matrix
    pub fn new(num_nodes: usize) -> Self {
        let mut matrix = Self {
            probabilities: [[0.0; MAX_NODES]; MAX_NODES],
            num_nodes: num_nodes.min(MAX_NODES),
        };
        
        // Initialize with identity (no causal influence)
        for i in 0..matrix.num_nodes {
            matrix.probabilities[i][i] = 1.0;
        }
        
        matrix
    }

    /// Set transition probability
    pub fn set_transition(&mut self, from: usize, to: usize, prob: f64) -> Result<(), IitError> {
        if from >= self.num_nodes || to >= self.num_nodes {
            return Err(IitError::InvalidNodeIndex);
        }
        
        if prob < 0.0 || prob > 1.0 {
            return Err(IitError::ComputationFailed);
        }
        
        self.probabilities[to][from] = prob;
        Ok(())
    }

    /// Get transition probability
    #[inline]
    pub fn get_transition(&self, from: usize, to: usize) -> Option<f64> {
        if from < self.num_nodes && to < self.num_nodes {
            Some(self.probabilities[to][from])
        } else {
            None
        }
    }

    /// Normalize columns (ensure probabilities sum to 1)
    pub fn normalize(&mut self) {
        for from in 0..self.num_nodes {
            let sum: f64 = self.probabilities[..self.num_nodes]
                .iter()
                .map(|row| row[from])
                .sum();
            
            if sum > 1e-10 {
                for to in 0..self.num_nodes {
                    self.probabilities[to][from] /= sum;
                }
            }
        }
    }
}

/// Perturbation state for MIP computation
#[derive(Clone, Copy)]
pub struct PerturbationState {
    /// Node states (binary)
    states: u128, // Supports up to 128 nodes directly
    /// Probability of this state
    probability: f64,
}

/// Integrated Information Calculator
pub struct IitPhiCalculator {
    /// Causal matrix
    causal_matrix: CausalMatrix,
    /// Current Phi estimate
    current_phi: f64,
    /// Running average Phi
    running_phi: f64,
    /// Phi history for trend detection
    phi_history: [f64; 32],
    history_idx: usize,
    /// Containment threshold
    containment_threshold: f64,
    /// Critical threshold
    critical_threshold: f64,
    /// Containment triggered flag
    containment_triggered: bool,
    /// Last computed time
    last_compute_ns: u64,
}

impl IitPhiCalculator {
    /// Create a new IIT calculator
    pub fn new(num_nodes: usize) -> Self {
        Self {
            causal_matrix: CausalMatrix::new(num_nodes),
            current_phi: 0.0,
            running_phi: 0.0,
            phi_history: [0.0; 32],
            history_idx: 0,
            containment_threshold: DEFAULT_PHI_THRESHOLD,
            critical_threshold: CRITICAL_PHI_THRESHOLD,
            containment_triggered: false,
            last_compute_ns: 0,
        }
    }

    /// Update causal matrix from spike correlation data
    pub fn update_from_correlations(
        &mut self,
        correlations: &[[f64; MAX_NODES]; MAX_NODES],
        num_nodes: usize,
    ) -> Result<(), IitError> {
        if num_nodes > self.causal_matrix.num_nodes {
            return Err(IitError::MatrixDimensionMismatch);
        }

        // Convert correlations to causal influences
        // This is a simplified approximation - full IIT uses perturbation analysis
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i != j {
                    // Normalize correlation to probability range
                    let causal_strength = ((correlations[i][j] + 1.0) / 2.0).min(1.0).max(0.0);
                    self.causal_matrix.set_transition(i, j, causal_strength)?;
                }
            }
        }

        self.causal_matrix.normalize();
        Ok(())
    }

    /// Compute Φ using simplified MIP (Minimum Information Partition) approach
    /// Full IIT 3.0/4.0 computation is computationally prohibitive for real-time
    pub fn compute_phi(&mut self) -> Result<f64, IitError> {
        let n = self.causal_matrix.num_nodes;
        if n == 0 {
            return Ok(0.0);
        }

        // Simplified Φ computation using effective information
        // Φ ≈ Σ (EI_whole - EI_parts) / normalization
        
        let mut phi = 0.0;

        // Compute effective information for the whole system
        let ei_whole = self.compute_effective_information()?;

        // Compute EI for various partitions (simplified - just bipartitions)
        let mut min_ei_parts = f64::INFINITY;
        
        // Try different bipartitions
        for split in 0..n {
            let ei_parts = self.compute_partitioned_ei(split)?;
            if ei_parts < min_ei_parts {
                min_ei_parts = ei_parts;
            }
        }

        // Φ is the difference (information lost due to partitioning)
        phi = ei_whole - min_ei_parts;
        phi = phi.max(0.0); // Φ cannot be negative

        // Normalize by system size
        phi /= n as f64;

        self.current_phi = phi;
        self.running_phi = 0.9 * self.running_phi + 0.1 * phi;

        // Store in history
        self.phi_history[self.history_idx] = phi;
        self.history_idx = (self.history_idx + 1) % 32;

        // Check thresholds
        self.check_containment();

        Ok(phi)
    }

    /// Compute effective information for the whole system
    fn compute_effective_information(&self) -> Result<f64, IitError> {
        let n = self.causal_matrix.num_nodes;
        let mut ei = 0.0;

        // EI = H(posterior) - H(prior)
        // Simplified: sum of mutual information between nodes
        
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let p = self.causal_matrix.get_transition(i, j).unwrap_or(0.0);
                    if p > 1e-10 && p < 1.0 - 1e-10 {
                        // Mutual information contribution
                        ei += p * (p.ln() / 2.0f64.ln()); // Bits
                    }
                }
            }
        }

        Ok(ei.abs())
    }

    /// Compute EI for a partitioned system
    fn compute_partitioned_ei(&self, split_point: usize) -> Result<f64, IitError> {
        let n = self.causal_matrix.num_nodes;
        let mut ei = 0.0;

        // Only count within-partition connections
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    // Check if both nodes are in same partition
                    let same_partition = (i < split_point && j < split_point)
                        || (i >= split_point && j >= split_point);
                    
                    if same_partition {
                        let p = self.causal_matrix.get_transition(i, j).unwrap_or(0.0);
                        if p > 1e-10 && p < 1.0 - 1e-10 {
                            ei += p * (p.ln() / 2.0f64.ln());
                        }
                    }
                }
            }
        }

        Ok(ei.abs())
    }

    /// Check if Phi exceeds containment thresholds
    fn check_containment(&mut self) {
        if self.current_phi >= self.critical_threshold {
            self.containment_triggered = true;
        } else if self.current_phi >= self.containment_threshold {
            // Check trend - rising Phi is concerning
            let trend = self.compute_phi_trend();
            if trend > 0.01 {
                self.containment_triggered = true;
            }
        }
    }

    /// Compute Phi trend over recent history
    fn compute_phi_trend(&self) -> f64 {
        let valid = self.history_idx.min(16);
        if valid < 2 {
            return 0.0;
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for i in 0..valid {
            let x = i as f64;
            let y = self.phi_history[i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let n = valid as f64;
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        
        if slope.is_nan() {
            0.0
        } else {
            slope
        }
    }

    /// Get current Phi value
    #[inline]
    pub fn get_current_phi(&self) -> f64 {
        self.current_phi
    }

    /// Get running average Phi
    #[inline]
    pub fn get_running_phi(&self) -> f64 {
        self.running_phi
    }

    /// Check if containment should be triggered
    #[inline]
    pub fn should_trigger_containment(&self) -> bool {
        self.containment_triggered
    }

    /// Reset containment flag (after manual intervention)
    pub fn reset_containment(&mut self) {
        self.containment_triggered = false;
    }

    /// Set containment threshold
    pub fn set_containment_threshold(&mut self, threshold: f64) {
        self.containment_threshold = threshold.max(0.01).min(10.0);
    }

    /// Get causal matrix reference
    pub fn causal_matrix_mut(&mut self) -> &mut CausalMatrix {
        &mut self.causal_matrix
    }
}

/// Bio-containment action types
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum ContainmentAction {
    /// No action needed
    None = 0,
    /// Warning issued
    Warning = 1,
    /// Reduce connectivity (lower stimulation)
    ReduceConnectivity = 2,
    /// Partial isolation
    PartialIsolation = 3,
    /// Full bio-containment halt
    FullHalt = 4,
}

/// Containment protocol manager
pub struct ContainmentProtocolManager {
    /// IIT calculator
    phi_calculator: IitPhiCalculator,
    /// Current containment level
    current_level: ContainmentAction,
    /// Escalation counter
    escalation_count: u32,
    /// Time at current level (ms)
    time_at_level_ms: u64,
    /// Minimum time before escalation (ms)
    min_escalation_time_ms: u64,
}

impl ContainmentProtocolManager {
    /// Create a new protocol manager
    pub fn new(num_nodes: usize) -> Self {
        Self {
            phi_calculator: IitPhiCalculator::new(num_nodes),
            current_level: ContainmentAction::None,
            escalation_count: 0,
            time_at_level_ms: 0,
            min_escalation_time_ms: 5000, // 5 seconds minimum
        }
    }

    /// Evaluate and potentially escalate containment
    pub fn evaluate(&mut self, timestamp_ms: u64) -> Result<ContainmentAction, IitError> {
        // Compute Phi
        self.phi_calculator.compute_phi()?;

        let phi = self.phi_calculator.get_current_phi();
        let should_contain = self.phi_calculator.should_trigger_containment();

        if !should_contain {
            // De-escalate if safe
            if self.current_level != ContainmentAction::None {
                self.time_at_level_ms += timestamp_ms;
                if self.time_at_level_ms >= self.min_escalation_time_ms {
                    self.deescalate();
                }
            }
            return Ok(self.current_level);
        }

        // Determine appropriate level based on Phi
        let target_level = if phi >= self.phi_calculator.critical_threshold {
            ContainmentAction::FullHalt
        } else if phi >= self.phi_calculator.containment_threshold * 1.5 {
            ContainmentAction::PartialIsolation
        } else if phi >= self.phi_calculator.containment_threshold {
            ContainmentAction::ReduceConnectivity
        } else {
            ContainmentAction::Warning
        };

        // Escalate if needed
        if self.should_escalate(target_level) {
            self.escalate_to(target_level);
        }

        self.time_at_level_ms += timestamp_ms;

        Ok(self.current_level)
    }

    /// Check if escalation is warranted
    fn should_escalate(&self, target: ContainmentAction) -> bool {
        let target_priority = target as u8;
        let current_priority = self.current_level as u8;
        
        target_priority > current_priority 
            && self.time_at_level_ms >= self.min_escalation_time_ms
    }

    /// Escalate to a higher containment level
    fn escalate_to(&mut self, level: ContainmentAction) {
        self.current_level = level;
        self.escalation_count += 1;
        self.time_at_level_ms = 0;
    }

    /// De-escalate containment
    fn deescalate(&mut self) {
        match self.current_level {
            ContainmentAction::FullHalt => self.current_level = ContainmentAction::PartialIsolation,
            ContainmentAction::PartialIsolation => self.current_level = ContainmentAction::ReduceConnectivity,
            ContainmentAction::ReduceConnectivity => self.current_level = ContainmentAction::Warning,
            ContainmentAction::Warning => self.current_level = ContainmentAction::None,
            ContainmentAction::None => {}
        }
        self.time_at_level_ms = 0;
    }

    /// Get current containment level
    pub fn get_containment_level(&self) -> ContainmentAction {
        self.current_level
    }

    /// Get Phi calculator reference
    pub fn phi_calculator_mut(&mut self) -> &mut IitPhiCalculator {
        &mut self.phi_calculator
    }

    /// Trigger immediate full halt
    pub fn trigger_full_halt(&mut self) {
        self.current_level = ContainmentAction::FullHalt;
        self.phi_calculator.containment_triggered = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_matrix_initialization() {
        let matrix = CausalMatrix::new(8);
        assert_eq!(matrix.num_nodes, 8);
        
        // Diagonal should be 1.0 (identity)
        for i in 0..8 {
            assert!((matrix.get_transition(i, i).unwrap() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_phi_computation_zero() {
        let mut calculator = IitPhiCalculator::new(8);
        
        // Identity matrix should have near-zero Phi
        let phi = calculator.compute_phi().unwrap();
        assert!(phi >= 0.0);
    }

    #[test]
    fn test_containment_trigger() {
        let mut calculator = IitPhiCalculator::new(8);
        calculator.set_containment_threshold(0.001); // Very low threshold
        
        // Manually set high Phi
        calculator.current_phi = 1.0;
        calculator.check_containment();
        
        assert!(calculator.should_trigger_containment());
    }

    #[test]
    fn test_protocol_manager_escalation() {
        let mut manager = ContainmentProtocolManager::new(8);
        
        // Initial state
        assert_eq!(manager.get_containment_level(), ContainmentAction::None);
        
        // Trigger full halt
        manager.trigger_full_halt();
        assert_eq!(manager.get_containment_level(), ContainmentAction::FullHalt);
    }
}
