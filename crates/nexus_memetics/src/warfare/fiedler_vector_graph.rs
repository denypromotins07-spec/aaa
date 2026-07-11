//! Fiedler Vector Graph Analysis for Botnet Detection
//! 
//! Computes and analyzes the Fiedler vector (second eigenvector of graph Laplacian)
//! to identify structural signatures of coordinated bot campaigns.

use crate::warfare::spectral_astroturfing_detector::{PropagationNode, PropagationEdge, AccountType};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FiedlerError {
    #[error("Graph too small for Fiedler analysis")]
    GraphTooSmall,
    #[error("Disconnected graph components detected")]
    DisconnectedGraph,
    #[error("Eigenvalue computation failed")]
    EigenFailure,
}

/// Detailed Fiedler vector analysis result
#[derive(Debug, Clone)]
pub struct FiedlerAnalysis {
    /// The Fiedler value (algebraic connectivity)
    pub fiedler_value: f64,
    /// The Fiedler vector itself
    pub fiedler_vector: Vec<f64>,
    /// Node partition based on Fiedler vector signs
    pub partition: NodePartition,
    /// Structural interpretation
    pub interpretation: FiedlerInterpretation,
    /// Botnet likelihood score
    pub botnet_score: f64,
}

/// Partition of nodes based on Fiedler vector
#[derive(Debug, Clone)]
pub struct NodePartition {
    /// Nodes with positive Fiedler values
    pub positive_set: Vec<usize>,
    /// Nodes with negative Fiedler values
    pub negative_set: Vec<usize>,
    /// Ratio of sizes between partitions
    pub size_ratio: f64,
    /// Whether partition shows hub-and-spoke pattern
    pub is_hub_spoke: bool,
}

/// Interpretation of Fiedler structure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiedlerInterpretation {
    /// Well-connected organic graph
    OrganicClustered,
    /// Two distinct communities
    BimodalCommunity,
    /// Star/hub topology (botnet indicator)
    HubSpoke,
    /// Chain/line topology
    LinearChain,
    /// Random/unstructured
    Unstructured,
}

/// Fiedler vector analyzer
pub struct FiedlerVectorAnalyzer {
    /// Threshold for considering graph disconnected
    disconnect_threshold: f64,
    /// Threshold for hub-spoke detection
    hub_spoke_threshold: f64,
}

impl FiedlerVectorAnalyzer {
    pub fn new(disconnect_threshold: f64, hub_spoke_threshold: f64) -> Self {
        Self {
            disconnect_threshold,
            hub_spoke_threshold,
        }
    }

    /// Build Laplacian matrix from graph
    pub fn build_laplacian(&self, nodes: &[PropagationNode], edges: &[PropagationEdge]) -> DMatrix<f64> {
        let n = nodes.len();
        let mut adj = DMatrix::zeros(n, n);

        for edge in edges {
            if edge.from < n && edge.to < n {
                adj[(edge.from, edge.to)] = edge.weight;
                adj[(edge.to, edge.from)] = edge.weight;
            }
        }

        // Build degree matrix
        let mut laplacian = DMatrix::zeros(n, n);
        for i in 0..n {
            let deg: f64 = (0..n).map(|j| adj[(i, j)]).sum();
            laplacian[(i, i)] = deg;
        }

        // L = D - A
        laplacian -= adj;
        laplacian
    }

    /// Compute Fiedler vector and value
    pub fn compute_fiedler(
        &self,
        laplacian: &DMatrix<f64>,
    ) -> Result<(f64, Vec<f64>), FiedlerError> {
        let n = laplacian.nrows();
        
        if n < 3 {
            return Err(FiedlerError::GraphTooSmall);
        }

        let eigen = SymmetricEigen::new(laplacian.clone());
        
        // Check for disconnection (first eigenvalue should be ~0)
        if eigen.eigenvalues[0].abs() > self.disconnect_threshold {
            return Err(FiedlerError::DisconnectedGraph);
        }

        let fiedler_value = eigen.eigenvalues[1];
        
        // Check for near-zero Fiedler value (disconnected or nearly so)
        if fiedler_value < self.disconnect_threshold {
            return Err(FiedlerError::DisconnectedGraph);
        }

        let fiedler_vector: Vec<f64> = (0..n).map(|i| eigen.eigenvectors[(i, 1)]).collect();

        Ok((fiedler_value, fiedler_vector))
    }

    /// Analyze Fiedler vector structure
    pub fn analyze_partition(&self, fiedler_vector: &[f64], nodes: &[PropagationNode]) -> NodePartition {
        let positive: Vec<usize> = fiedler_vector
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0.0)
            .map(|(i, _)| i)
            .collect();

        let negative: Vec<usize> = fiedler_vector
            .iter()
            .enumerate()
            .filter(|(_, &v)| v <= 0.0)
            .map(|(i, _)| i)
            .collect();

        let pos_count = positive.len() as f64;
        let neg_count = negative.len() as f64;
        let size_ratio = if neg_count > 0.0 { pos_count / neg_count } else { f64::INFINITY };

        // Detect hub-spoke pattern: one side much smaller than other
        let is_hub_spoke = size_ratio > 5.0 || size_ratio < 0.2;

        NodePartition {
            positive_set: positive,
            negative_set: negative,
            size_ratio,
            is_hub_spoke,
        }
    }

    /// Interpret Fiedler structure
    pub fn interpret_structure(
        &self,
        fiedler_value: f64,
        partition: &NodePartition,
        fiedler_vector: &[f64],
    ) -> FiedlerInterpretation {
        // High algebraic connectivity = well-connected
        if fiedler_value > 1.0 {
            return FiedlerInterpretation::OrganicClustered;
        }

        // Check for bimodal distribution
        let variance = self.compute_variance(fiedler_vector);
        let mean = fiedler_vector.iter().sum::<f64>() / fiedler_vector.len() as f64;
        
        // Bimodal if values cluster around two distinct means
        let bimodality = self.detect_bimodality(fiedler_vector, mean);

        if bimodality > 0.7 {
            return FiedlerInterpretation::BimodalCommunity;
        }

        // Hub-spoke detection
        if partition.is_hub_spoke {
            return FiedlerInterpretation::HubSpoke;
        }

        // Check for linear chain (Fiedler vector has monotonic pattern)
        if self.is_monotonic(fiedler_vector) {
            return FiedlerInterpretation::LinearChain;
        }

        FiedlerInterpretation::Unstructured
    }

    fn compute_variance(&self, values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
    }

    fn detect_bimodality(&self, values: &[f64], overall_mean: f64) -> f64 {
        // Split by sign and check if each group clusters tightly
        let positive: Vec<f64> = values.iter().copied().filter(|&x| x > 0.0).collect();
        let negative: Vec<f64> = values.iter().copied().filter(|&x| x <= 0.0).collect();

        if positive.len() < 2 || negative.len() < 2 {
            return 0.0;
        }

        let pos_mean = positive.iter().sum::<f64>() / positive.len() as f64;
        let neg_mean = negative.iter().sum::<f64>() / negative.len() as f64;

        // Bimodality score based on separation of means relative to within-group variance
        let pos_var = self.compute_variance(&positive);
        let neg_var = self.compute_variance(&negative);
        
        let separation = (pos_mean - neg_mean).abs();
        let avg_var = (pos_var + neg_var) / 2.0;

        if avg_var < 1e-10 {
            return 1.0;
        }

        (separation / avg_var.sqrt()).min(1.0)
    }

    fn is_monotonic(&self, values: &[f64]) -> bool {
        if values.len() < 3 {
            return false;
        }

        let mut increasing = true;
        let mut decreasing = true;

        for i in 1..values.len() {
            if values[i] < values[i - 1] {
                increasing = false;
            }
            if values[i] > values[i - 1] {
                decreasing = false;
            }
        }

        increasing || decreasing
    }

    /// Compute botnet score from Fiedler analysis
    pub fn compute_botnet_score(
        &self,
        fiedler_value: f64,
        partition: &NodePartition,
        interpretation: FiedlerInterpretation,
        nodes: &[PropagationNode],
    ) -> f64 {
        let mut score = 0.0;

        // Low algebraic connectivity = potential botnet
        if fiedler_value < 0.1 {
            score += 0.3;
        } else if fiedler_value < 0.3 {
            score += 0.15;
        }

        // Hub-spoke is strong botnet indicator
        if partition.is_hub_spoke {
            score += 0.3;
        }

        // Hub-spoke interpretation
        if interpretation == FiedlerInterpretation::HubSpoke {
            score += 0.25;
        }

        // Known bot accounts in central positions
        let bot_in_positive = partition.positive_set
            .iter()
            .filter(|&&i| nodes.get(i).map_or(false, |n| n.account_type == AccountType::Bot))
            .count();
        
        let bot_in_negative = partition.negative_set
            .iter()
            .filter(|&&i| nodes.get(i).map_or(false, |n| n.account_type == AccountType::Bot))
            .count();

        // If bots concentrated in smaller partition, higher score
        let smaller_size = partition.positive_set.len().min(partition.negative_set.len());
        if smaller_size > 0 {
            let bot_concentration = (bot_in_positive.max(bot_in_negative) as f64) / smaller_size as f64;
            score += bot_concentration * 0.15;
        }

        score.min(1.0)
    }

    /// Full Fiedler analysis pipeline
    pub fn full_analysis(
        &self,
        nodes: &[PropagationNode],
        edges: &[PropagationEdge],
    ) -> Result<FiedlerAnalysis, FiedlerError> {
        let laplacian = self.build_laplacian(nodes, edges);
        let (fiedler_value, fiedler_vector) = self.compute_fiedler(&laplacian)?;
        
        let partition = self.analyze_partition(&fiedler_vector, nodes);
        let interpretation = self.interpret_structure(fiedler_value, &partition, &fiedler_vector);
        let botnet_score = self.compute_botnet_score(fiedler_value, &partition, interpretation, nodes);

        Ok(FiedlerAnalysis {
            fiedler_value,
            fiedler_vector,
            partition,
            interpretation,
            botnet_score,
        })
    }
}

/// Streaming Fiedler tracker for evolving graphs
pub struct StreamingFiedlerTracker {
    analyzer: FiedlerVectorAnalyzer,
    history: Vec<FiedlerAnalysis>,
    max_history: usize,
}

impl StreamingFiedlerTracker {
    pub fn new(analyzer: FiedlerVectorAnalyzer, max_history: usize) -> Self {
        Self {
            analyzer,
            history: Vec::new(),
            max_history,
        }
    }

    /// Update with new graph state
    pub fn update(&mut self, nodes: &[PropagationNode], edges: &[PropagationEdge]) -> Option<FiedlerAnalysis> {
        match self.analyzer.full_analysis(nodes, edges) {
            Ok(analysis) => {
                self.history.push(analysis.clone());
                if self.history.len() > self.max_history {
                    self.history.remove(0);
                }
                Some(analysis)
            }
            Err(_) => None,
        }
    }

    /// Detect sudden changes in graph structure
    pub fn detect_structural_break(&self) -> Option<(usize, f64)> {
        if self.history.len() < 2 {
            return None;
        }

        for i in 1..self.history.len() {
            let prev_val = self.history[i - 1].fiedler_value;
            let curr_val = self.history[i].fiedler_value;
            
            // Significant drop in algebraic connectivity
            if prev_val > 0.5 && curr_val < 0.2 {
                return Some((i, (prev_val - curr_val) / prev_val));
            }
        }

        None
    }

    /// Trend in botnet score
    pub fn botnet_trend(&self) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }

        let first = self.history.first()?.botnet_score;
        let last = self.history.last()?.botnet_score;
        
        Some(last - first)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fiedler_computation() {
        let analyzer = FiedlerVectorAnalyzer::new(1e-6, 5.0);
        
        // Simple path graph
        let nodes = vec![
            PropagationNode { id: 0, first_exposure: 0.0, share_count: 1, account_type: AccountType::Unknown },
            PropagationNode { id: 1, first_exposure: 0.1, share_count: 1, account_type: AccountType::Unknown },
            PropagationNode { id: 2, first_exposure: 0.2, share_count: 1, account_type: AccountType::Unknown },
            PropagationNode { id: 3, first_exposure: 0.3, share_count: 1, account_type: AccountType::Unknown },
        ];

        let edges = vec![
            PropagationEdge { from: 0, to: 1, time_delay: 0.1, weight: 1.0 },
            PropagationEdge { from: 1, to: 2, time_delay: 0.1, weight: 1.0 },
            PropagationEdge { from: 2, to: 3, time_delay: 0.1, weight: 1.0 },
        ];

        let laplacian = analyzer.build_laplacian(&nodes, &edges);
        let result = analyzer.compute_fiedler(&laplacian);
        
        assert!(result.is_ok());
    }

    #[test]
    fn test_hub_spoke_detection() {
        let analyzer = FiedlerVectorAnalyzer::new(1e-6, 3.0);
        
        // Star graph (hub at center)
        let nodes = (0..10)
            .map(|i| PropagationNode {
                id: i,
                first_exposure: i as f64 * 0.1,
                share_count: if i == 0 { 9 } else { 1 },
                account_type: if i == 0 { AccountType::Bot } else { AccountType::Unknown },
            })
            .collect::<Vec<_>>();

        let edges = (1..10)
            .map(|i| PropagationEdge { from: 0, to: i, time_delay: 0.1, weight: 1.0 })
            .collect::<Vec<_>>();

        let result = analyzer.full_analysis(&nodes, &edges);
        
        assert!(result.is_ok());
        let analysis = result.unwrap();
        
        // Should detect hub-spoke structure
        assert!(analysis.partition.is_hub_spoke || analysis.botnet_score > 0.3);
    }
}
