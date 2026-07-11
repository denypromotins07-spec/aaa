//! Spectral Astroturfing Detector using Graph Laplacian Analysis
//! 
//! Distinguishes organic narrative propagation from coordinated botnet campaigns
//! by analyzing the spectral properties of information propagation graphs.

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use thiserror::Error;
use std::collections::{HashMap, HashSet};

#[derive(Error, Debug)]
pub enum AstroturfingError {
    #[error("Graph is disconnected: cannot compute Fiedler vector")]
    DisconnectedGraph,
    #[error("Insufficient nodes for spectral analysis: need at least 3, got {actual}")]
    InsufficientNodes { actual: usize },
    #[error("Eigenvalue computation failed")]
    EigenvalueFailure,
    #[error("Numerical instability detected")]
    NumericalInstability,
}

/// Node in the narrative propagation graph
#[derive(Debug, Clone)]
pub struct PropagationNode {
    pub id: usize,
    /// Timestamp of first exposure to narrative
    pub first_exposure: f64,
    /// Number of times shared/posted
    pub share_count: u32,
    /// Account type classification
    pub account_type: AccountType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccountType {
    Unknown,
    Retail,
    Institutional,
    Bot,
    Influencer,
}

/// Edge in the propagation graph (information flow)
#[derive(Debug, Clone)]
pub struct PropagationEdge {
    pub from: usize,
    pub to: usize,
    /// Time delay between exposures
    pub time_delay: f64,
    /// Strength of connection (e.g., follower relationship)
    pub weight: f64,
}

/// Result of astroturfing analysis
#[derive(Debug, Clone)]
pub struct AstroturfingAnalysis {
    /// Fiedler value (algebraic connectivity)
    pub fiedler_value: f64,
    /// Fiedler vector (second eigenvector of Laplacian)
    pub fiedler_vector: Option<Vec<f64>>,
    /// Classification confidence [0, 1]
    pub botnet_confidence: f64,
    /// Estimated bot fraction
    pub estimated_bot_fraction: f64,
    /// Graph structural metrics
    pub structural_metrics: GraphMetrics,
    /// Classification result
    pub classification: PropagationClassification,
}

impl AstroturfingAnalysis {
    /// Check if narrative is likely astroturfed
    pub fn is_astroturfed(&self, threshold: f64) -> bool {
        self.botnet_confidence >= threshold
    }

    /// Recommended trading action based on classification
    pub fn recommended_action(&self) -> AstroturfingAction {
        match self.classification {
            PropagationClassification::Organic => AstroturfingAction::None,
            PropagationClassification::LikelyOrganic => AstroturfingAction::Monitor,
            PropagationClassification::Suspicious => AstroturfingAction::ReduceExposure,
            PropagationClassification::LikelyBotnet => AstroturfingAction::Short,
            PropagationClassification::ConfirmedBotnet => AstroturfingAction::AggressiveShort,
        }
    }
}

/// Structural metrics of the propagation graph
#[derive(Debug, Clone)]
pub struct GraphMetrics {
    /// Number of nodes
    pub node_count: usize,
    /// Number of edges
    pub edge_count: usize,
    /// Average degree
    pub avg_degree: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Diameter (longest shortest path)
    pub diameter: Option<usize>,
    /// Density
    pub density: f64,
}

/// Classification of propagation pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropagationClassification {
    /// Clearly organic human propagation
    Organic,
    /// Mostly organic with some suspicious elements
    LikelyOrganic,
    /// Mixed signals, unclear origin
    Suspicious,
    /// Likely coordinated bot campaign
    LikelyBotnet,
    /// Confirmed astroturfing/botnet
    ConfirmedBotnet,
}

/// Trading action based on astroturfing detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AstroturfingAction {
    /// No action needed
    None,
    /// Monitor for changes
    Monitor,
    /// Reduce exposure to affected asset
    ReduceExposure,
    /// Initiate short position
    Short,
    /// Aggressively short
    AggressiveShort,
}

/// Spectral graph analyzer for astroturfing detection
pub struct SpectralAstroturfingDetector {
    /// Threshold for "low" algebraic connectivity (botnet indicator)
    low_connectivity_threshold: f64,
    /// Minimum nodes for reliable analysis
    min_nodes: usize,
}

impl SpectralAstroturfingDetector {
    pub fn new(low_connectivity_threshold: f64, min_nodes: usize) -> Self {
        Self {
            low_connectivity_threshold,
            min_nodes,
        }
    }

    /// Build adjacency matrix from edges
    fn build_adjacency_matrix(&self, nodes: &[PropagationNode], edges: &[PropagationEdge]) -> DMatrix<f64> {
        let n = nodes.len();
        let mut adj = DMatrix::zeros(n, n);

        for edge in edges {
            if edge.from < n && edge.to < n {
                adj[(edge.from, edge.to)] = edge.weight;
                adj[(edge.to, edge.from)] = edge.weight; // Undirected
            }
        }

        adj
    }

    /// Build degree matrix
    fn build_degree_matrix(&self, adj: &DMatrix<f64>) -> DMatrix<f64> {
        let n = adj.nrows();
        let mut deg = DMatrix::zeros(n, n);

        for i in 0..n {
            let d: f64 = (0..n).map(|j| adj[(i, j)]).sum();
            deg[(i, i)] = d;
        }

        deg
    }

    /// Compute graph Laplacian L = D - A
    fn compute_laplacian(&self, adj: &DMatrix<f64>) -> DMatrix<f64> {
        let deg = self.build_degree_matrix(adj);
        deg - adj
    }

    /// Compute normalized Laplacian L_norm = D^(-1/2) L D^(-1/2)
    fn compute_normalized_laplacian(&self, laplacian: &DMatrix<f64>, degree: &DMatrix<f64>) -> DMatrix<f64> {
        let n = laplacian.nrows();
        
        // Compute D^(-1/2)
        let mut d_inv_sqrt = DMatrix::zeros(n, n);
        for i in 0..n {
            let d = degree[(i, i)];
            if d > 1e-10 {
                d_inv_sqrt[(i, i)] = 1.0 / d.sqrt();
            }
        }

        // L_norm = D^(-1/2) L D^(-1/2)
        &d_inv_sqrt * laplacian * &d_inv_sqrt
    }

    /// Compute Fiedler value and vector (second smallest eigenvalue/eigenvector)
    fn compute_fiedler(&self, laplacian: &DMatrix<f64>) -> Result<(f64, Vec<f64>), AstroturfingError> {
        let n = laplacian.nrows();
        
        if n < 3 {
            return Err(AstroturfingError::InsufficientNodes { actual: n });
        }

        // Use symmetric eigenvalue decomposition
        let eigen = SymmetricEigen::new(laplacian.clone());
        
        // Eigenvalues are sorted in ascending order for SymmetricEigen
        let eigenvalues = &eigen.eigenvalues;
        let eigenvectors = &eigen.eigenvectors;

        // First eigenvalue should be ~0 for connected graph
        if eigenvalues[0].abs() > 1e-6 {
            // Graph may be disconnected or numerical issues
            // Continue anyway but note it
        }

        // Fiedler value is the second smallest
        let fiedler_value = eigenvalues[1];
        
        // Fiedler vector is corresponding eigenvector
        let fiedler_vector: Vec<f64> = (0..n).map(|i| eigenvectors[(i, 1)]).collect();

        if fiedler_value < 1e-10 {
            return Err(AstroturfingError::DisconnectedGraph);
        }

        Ok((fiedler_value, fiedler_vector))
    }

    /// Analyze propagation graph for astroturfing signatures
    pub fn analyze(
        &self,
        nodes: &[PropagationNode],
        edges: &[PropagationEdge],
    ) -> Result<AstroturfingAnalysis, AstroturfingError> {
        if nodes.len() < self.min_nodes {
            return Err(AstroturfingError::InsufficientNodes { actual: nodes.len() });
        }

        // Build matrices
        let adj = self.build_adjacency_matrix(nodes, edges);
        let laplacian = self.compute_laplacian(&adj);
        let degree = self.build_degree_matrix(&adj);

        // Compute Fiedler value and vector
        let (fiedler_value, fiedler_vector) = self.compute_fiedler(&laplacian)?;

        // Compute structural metrics
        let metrics = self.compute_graph_metrics(nodes, edges, &adj);

        // Classify based on spectral and structural features
        let classification = self.classify_propagation(fiedler_value, &metrics, nodes);

        // Estimate bot fraction
        let bot_count: usize = nodes.iter().filter(|n| n.account_type == AccountType::Bot).count();
        let estimated_bot_fraction = bot_count as f64 / nodes.len() as f64;

        // Calculate botnet confidence
        let botnet_confidence = self.compute_botnet_confidence(fiedler_value, &metrics, estimated_bot_fraction);

        Ok(AstroturfingAnalysis {
            fiedler_value,
            fiedler_vector: Some(fiedler_vector),
            botnet_confidence,
            estimated_bot_fraction,
            structural_metrics: metrics,
            classification,
        })
    }

    fn compute_graph_metrics(
        &self,
        nodes: &[PropagationNode],
        edges: &[PropagationEdge],
        adj: &DMatrix<f64>,
    ) -> GraphMetrics {
        let n = nodes.len();
        let m = edges.len();

        // Average degree
        let total_degree: f64 = (0..n).map(|i| {
            (0..n).map(|j| if adj[(i, j)] > 0.0 { 1 } else { 0 }).sum::<usize>() as f64
        }).sum();
        let avg_degree = total_degree / n as f64;

        // Density
        let max_edges = n * (n - 1) / 2;
        let density = if max_edges > 0 {
            m as f64 / max_edges as f64
        } else {
            0.0
        };

        // Clustering coefficient (simplified)
        let clustering = self.compute_clustering_coefficient(adj, n);

        GraphMetrics {
            node_count: n,
            edge_count: m,
            avg_degree,
            clustering_coefficient: clustering,
            diameter: None, // Would require BFS for exact calculation
            density,
        }
    }

    fn compute_clustering_coefficient(&self, adj: &DMatrix<f64>, n: usize) -> f64 {
        let mut total_local_clustering = 0.0;
        let mut count = 0;

        for i in 0..n {
            // Get neighbors
            let neighbors: Vec<usize> = (0..n)
                .filter(|&j| adj[(i, j)] > 0.0 && i != j)
                .collect();

            let k = neighbors.len();
            if k < 2 {
                continue;
            }

            // Count triangles
            let mut triangles = 0;
            for &a in &neighbors {
                for &b in &neighbors {
                    if a < b && adj[(a, b)] > 0.0 {
                        triangles += 1;
                    }
                }
            }

            let possible_triangles = k * (k - 1) / 2;
            total_local_clustering += triangles as f64 / possible_triangles as f64;
            count += 1;
        }

        if count > 0 {
            total_local_clustering / count as f64
        } else {
            0.0
        }
    }

    fn classify_propagation(
        &self,
        fiedler_value: f64,
        metrics: &GraphMetrics,
        nodes: &[PropagationNode],
    ) -> PropagationClassification {
        // Botnets typically have:
        // - Low algebraic connectivity (low Fiedler value)
        // - Tree-like structure (low clustering)
        // - High density of connections from few central nodes

        let bot_count: usize = nodes.iter().filter(|n| n.account_type == AccountType::Bot).count();
        let known_bot_ratio = bot_count as f64 / nodes.len() as f64;

        // Score based on multiple features
        let mut score = 0.0;

        // Low Fiedler value indicates poor connectivity (botnet-like)
        if fiedler_value < self.low_connectivity_threshold {
            score += 0.4;
        } else if fiedler_value < self.low_connectivity_threshold * 2.0 {
            score += 0.2;
        }

        // Low clustering coefficient indicates tree-like structure
        if metrics.clustering_coefficient < 0.1 {
            score += 0.3;
        } else if metrics.clustering_coefficient < 0.3 {
            score += 0.15;
        }

        // High density with low clustering is suspicious
        if metrics.density > 0.5 && metrics.clustering_coefficient < 0.2 {
            score += 0.2;
        }

        // Known bot accounts
        score += known_bot_ratio * 0.3;

        // Classify based on score
        if score >= 0.8 {
            PropagationClassification::ConfirmedBotnet
        } else if score >= 0.6 {
            PropagationClassification::LikelyBotnet
        } else if score >= 0.4 {
            PropagationClassification::Suspicious
        } else if score >= 0.2 {
            PropagationClassification::LikelyOrganic
        } else {
            PropagationClassification::Organic
        }
    }

    fn compute_botnet_confidence(
        &self,
        fiedler_value: f64,
        metrics: &GraphMetrics,
        bot_fraction: f64,
    ) -> f64 {
        let mut confidence = 0.0;

        // Fiedler-based confidence
        if fiedler_value < self.low_connectivity_threshold {
            confidence += 0.4;
        }

        // Structure-based confidence
        if metrics.clustering_coefficient < 0.2 {
            confidence += 0.3;
        }

        // Bot fraction confidence
        confidence += bot_fraction * 0.3;

        confidence.min(1.0)
    }
}

/// Streaming detector for real-time astroturfing monitoring
pub struct StreamingAstroturfingMonitor {
    detector: SpectralAstroturfingDetector,
    recent_nodes: Vec<PropagationNode>,
    recent_edges: Vec<PropagationEdge>,
    max_window_size: usize,
    last_analysis: Option<AstroturfingAnalysis>,
}

impl StreamingAstroturfingMonitor {
    pub fn new(detector: SpectralAstroturfingDetector, max_window_size: usize) -> Self {
        Self {
            detector,
            recent_nodes: Vec::new(),
            recent_edges: Vec::new(),
            max_window_size,
            last_analysis: None,
        }
    }

    /// Add a new propagation event
    pub fn add_event(&mut self, from_node: PropagationNode, to_node: PropagationNode, time_delay: f64) {
        self.recent_nodes.push(from_node);
        self.recent_nodes.push(to_node);
        
        // Deduplicate nodes by ID
        self.recent_nodes.sort_by_key(|n| n.id);
        self.recent_nodes.dedup_by_key(|n| n.id);

        self.recent_edges.push(PropagationEdge {
            from: self.recent_nodes.iter().position(|n| n.id == from_node.id).unwrap_or(0),
            to: self.recent_nodes.iter().position(|n| n.id == to_node.id).unwrap_or(0),
            time_delay,
            weight: 1.0,
        });

        // Trim window
        if self.recent_nodes.len() > self.max_window_size {
            let remove_count = self.recent_nodes.len() - self.max_window_size;
            self.recent_nodes.drain(0..remove_count);
            
            // Re-index edges
            let valid_ids: HashSet<usize> = self.recent_nodes.iter().map(|n| n.id).collect();
            self.recent_edges.retain(|e| {
                // Keep edges that reference valid nodes
                true // Simplified - would need proper re-indexing
            });
        }
    }

    /// Run analysis on current window
    pub fn analyze(&mut self) -> Option<AstroturfingAnalysis> {
        if self.recent_nodes.len() < self.detector.min_nodes {
            return None;
        }

        match self.detector.analyze(&self.recent_nodes, &self.recent_edges) {
            Ok(analysis) => {
                self.last_analysis = Some(analysis.clone());
                Some(analysis)
            }
            Err(_) => None,
        }
    }

    /// Get latest classification
    pub fn latest_classification(&self) -> Option<PropagationClassification> {
        self.last_analysis.as_ref().map(|a| a.classification)
    }

    /// Check if current state indicates astroturfing
    pub fn is_astroturfing_detected(&self, threshold: f64) -> bool {
        self.last_analysis
            .as_ref()
            .map(|a| a.is_astroturfed(threshold))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_detection() {
        let detector = SpectralAstroturfingDetector::new(0.1, 5);

        // Create a simple tree-like graph (botnet signature)
        let nodes = vec![
            PropagationNode { id: 0, first_exposure: 0.0, share_count: 10, account_type: AccountType::Bot },
            PropagationNode { id: 1, first_exposure: 0.1, share_count: 1, account_type: AccountType::Unknown },
            PropagationNode { id: 2, first_exposure: 0.1, share_count: 1, account_type: AccountType::Unknown },
            PropagationNode { id: 3, first_exposure: 0.2, share_count: 1, account_type: AccountType::Unknown },
            PropagationNode { id: 4, first_exposure: 0.2, share_count: 1, account_type: AccountType::Unknown },
        ];

        let edges = vec![
            PropagationEdge { from: 0, to: 1, time_delay: 0.1, weight: 1.0 },
            PropagationEdge { from: 0, to: 2, time_delay: 0.1, weight: 1.0 },
            PropagationEdge { from: 0, to: 3, time_delay: 0.2, weight: 1.0 },
            PropagationEdge { from: 0, to: 4, time_delay: 0.2, weight: 1.0 },
        ];

        let result = detector.analyze(&nodes, &edges);
        assert!(result.is_ok());
    }
}
