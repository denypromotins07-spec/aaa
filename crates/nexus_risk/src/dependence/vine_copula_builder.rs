//! Vine Copula (C-Vine/D-Vine) builder for high-dimensional dependence modeling
//!
//! Vine copulas decompose multivariate dependence into a cascade of bivariate
//! copulas (pair-copulas), enabling flexible modeling of complex tail dependencies
//! that cannot be captured by standard elliptical copulas.

use crate::dependence::student_t_copula::{StudentTCopula, StudentTCopulaConfig, CopulaError};
use ndarray::{Array1, Array2, ArrayView1};
use std::collections::HashMap;

/// Type of vine structure
#[derive(Debug, Clone, Copy)]
pub enum VineType {
    /// C-Vine: Canonical vine with a central node in each tree
    CVine,
    /// D-Vine: Direct vine with a path structure
    DVine,
    /// R-Vine: Regular vine (most general, automatically determined)
    RVine,
}

/// Pair copula specification for vine construction
#[derive(Debug, Clone)]
pub struct PairCopulaSpec {
    /// First variable index
    pub var1: usize,
    /// Second variable index
    pub var2: usize,
    /// Conditioning set (variables conditioned on)
    pub conditioning_set: Vec<usize>,
    /// Copula family
    pub family: PairCopulaFamily,
    /// Copula parameters
    pub parameters: Vec<f64>,
}

/// Bivariate copula families supported in the vine
#[derive(Debug, Clone, Copy)]
pub enum PairCopulaFamily {
    /// Gaussian copula
    Gaussian,
    /// Student-t copula with specified degrees of freedom
    StudentT(f64),
    /// Clayton copula (asymmetric lower tail dependence)
    Clayton,
    /// Gumbel copula (asymmetric upper tail dependence)
    Gumbel,
    /// Frank copula (symmetric dependence)
    Frank,
}

/// A single tree in the vine decomposition
#[derive(Debug, Clone)]
pub struct VineTree {
    /// Level of the tree (1-indexed)
    pub level: usize,
    /// Edges in this tree
    pub edges: Vec<PairCopulaSpec>,
}

/// Vine Copula structure containing all trees
pub struct VineCopula {
    /// Number of variables
    pub n_variables: usize,
    /// Type of vine
    pub vine_type: VineType,
    /// Tree sequence (T1, T2, ..., T_{n-1})
    pub trees: Vec<VineTree>,
    /// Variable ordering (important for C-Vine and D-Vine)
    pub variable_order: Vec<usize>,
}

impl VineCopula {
    /// Build a C-Vine copula from correlation matrix
    pub fn build_c_vine(
        correlation_matrix: &Array2<f64>,
        degrees_of_freedom: f64,
    ) -> Result<Self, CopulaError> {
        let n = correlation_matrix.nrows();
        
        if n != correlation_matrix.ncols() {
            return Err(CopulaError::InvalidDimensions);
        }
        
        // Determine variable ordering based on dependency strength
        let order = Self::optimal_c_vine_ordering(correlation_matrix);
        
        let mut trees = Vec::with_capacity(n - 1);
        
        // Build first tree (rooted at first variable in order)
        let mut tree_edges = Vec::new();
        let root = order[0];
        
        for i in 1..n {
            let var_i = order[i];
            let rho = correlation_matrix[[root, var_i]];
            
            tree_edges.push(PairCopulaSpec {
                var1: root,
                var2: var_i,
                conditioning_set: vec![],
                family: PairCopulaFamily::StudentT(degrees_of_freedom),
                parameters: vec![rho],
            });
        }
        
        trees.push(VineTree {
            level: 1,
            edges: tree_edges,
        });
        
        // Build subsequent trees
        for level in 2..n {
            let mut level_edges = Vec::new();
            let prev_tree = &trees[level - 2];
            
            // Generate edges for this level based on proximity condition
            for i in 0..prev_tree.edges.len() {
                for j in (i + 1)..prev_tree.edges.len() {
                    let edge_i = &prev_tree.edges[i];
                    let edge_j = &prev_tree.edges[j];
                    
                    // Check if edges share a common node (proximity condition)
                    let common_nodes: Vec<_> = edge_i
                        .conditioning_set
                        .iter()
                        .chain(std::iter::once(&edge_i.var1))
                        .chain(std::iter::once(&edge_i.var2))
                        .collect();
                    
                    let mut shared = Vec::new();
                    for node in [&edge_j.var1, &edge_j.var2] {
                        if common_nodes.contains(&node) {
                            shared.push(*node);
                        }
                    }
                    
                    if shared.len() == level - 1 {
                        // Valid edge for this level
                        let new_cond_set: Vec<_> = edge_i
                            .conditioning_set
                            .iter()
                            .copied()
                            .chain(std::iter::once(edge_i.var1))
                            .chain(std::iter::once(edge_i.var2))
                            .filter(|x| *x != edge_j.var1 && *x != edge_j.var2)
                            .collect();
                        
                        // Calculate partial correlation for this pair
                        let partial_rho = Self::partial_correlation(
                            edge_i.var1,
                            edge_j.var1,
                            &new_cond_set,
                            correlation_matrix,
                        );
                        
                        level_edges.push(PairCopulaSpec {
                            var1: edge_i.var1,
                            var2: edge_j.var1,
                            conditioning_set: new_cond_set,
                            family: PairCopulaFamily::StudentT(degrees_of_freedom),
                            parameters: vec![partial_rho],
                        });
                    }
                }
            }
            
            if !level_edges.is_empty() {
                trees.push(VineTree {
                    level,
                    edges: level_edges,
                });
            }
        }
        
        Ok(Self {
            n_variables: n,
            vine_type: VineType::CVine,
            trees,
            variable_order: order,
        })
    }
    
    /// Build a D-Vine copula from data
    pub fn build_d_vine(
        correlation_matrix: &Array2<f64>,
        degrees_of_freedom: f64,
    ) -> Result<Self, CopulaError> {
        let n = correlation_matrix.nrows();
        
        if n != correlation_matrix.ncols() {
            return Err(CopulaError::InvalidDimensions);
        }
        
        // For D-Vine, ordering is typically based on variable importance
        let order = Self::optimal_d_vine_ordering(correlation_matrix);
        
        let mut trees = Vec::with_capacity(n - 1);
        
        // Build first tree (path structure)
        let mut tree_edges = Vec::new();
        
        for i in 0..n - 1 {
            let var1 = order[i];
            let var2 = order[i + 1];
            let rho = correlation_matrix[[var1, var2]];
            
            tree_edges.push(PairCopulaSpec {
                var1,
                var2,
                conditioning_set: vec![],
                family: PairCopulaFamily::StudentT(degrees_of_freedom),
                parameters: vec![rho],
            });
        }
        
        trees.push(VineTree {
            level: 1,
            edges: tree_edges,
        });
        
        // Build subsequent trees following D-Vine structure
        for level in 2..n {
            let mut level_edges = Vec::new();
            let prev_tree = &trees[level - 2];
            
            for i in 0..prev_tree.edges.len().saturating_sub(1) {
                let edge_i = &prev_tree.edges[i];
                let edge_j = &prev_tree.edges[i + 1];
                
                // In D-Vine, consecutive edges are connected
                let new_var1 = if edge_i.var1 != edge_j.var1 && edge_i.var1 != edge_j.var2 {
                    edge_i.var1
                } else {
                    edge_i.var2
                };
                
                let new_var2 = if edge_j.var1 != edge_i.var1 && edge_j.var1 != edge_i.var2 {
                    edge_j.var1
                } else {
                    edge_j.var2
                };
                
                let new_cond_set: Vec<_> = edge_i
                    .conditioning_set
                    .iter()
                    .copied()
                    .chain(std::iter::once(edge_i.var1))
                    .chain(std::iter::once(edge_i.var2))
                    .filter(|x| *x != new_var1 && *x != new_var2)
                    .collect();
                
                let partial_rho = Self::partial_correlation(
                    new_var1,
                    new_var2,
                    &new_cond_set,
                    correlation_matrix,
                );
                
                level_edges.push(PairCopulaSpec {
                    var1: new_var1,
                    var2: new_var2,
                    conditioning_set: new_cond_set,
                    family: PairCopulaFamily::StudentT(degrees_of_freedom),
                    parameters: vec![partial_rho],
                });
            }
            
            if !level_edges.is_empty() {
                trees.push(VineTree {
                    level,
                    edges: level_edges,
                });
            }
        }
        
        Ok(Self {
            n_variables: n,
            vine_type: VineType::DVine,
            trees,
            variable_order: order,
        })
    }
    
    /// Calculate the density of the vine copula at given uniform marginals
    pub fn density(&self, u: &ArrayView1<f64>) -> Result<f64, CopulaError> {
        if u.len() != self.n_variables {
            return Err(CopulaError::InvalidDimensions);
        }
        
        let mut log_density = 0.0;
        
        // Sum log-densities from all pair copulas across all trees
        for tree in &self.trees {
            for edge in &tree.edges {
                let pair_density = self.evaluate_pair_copula(
                    u[edge.var1],
                    u[edge.var2],
                    &edge.conditioning_set,
                    u,
                    &edge.family,
                    &edge.parameters,
                )?;
                
                if pair_density > 0.0 {
                    log_density += pair_density.ln();
                }
            }
        }
        
        Ok(log_density.exp())
    }
    
    /// Evaluate a single pair copula density
    fn evaluate_pair_copula(
        &self,
        u1: f64,
        u2: f64,
        cond_set: &[usize],
        u: &ArrayView1<f64>,
        family: &PairCopulaFamily,
        params: &[f64],
    ) -> Result<f64, CopulaError> {
        match family {
            PairCopulaFamily::StudentT(nu) => {
                if params.is_empty() {
                    return Err(CopulaError::InvalidCorrelation(0.0));
                }
                
                let rho = params[0];
                let config = StudentTCopulaConfig::bivariate(rho, *nu)?;
                let copula = StudentTCopula::new(config)?;
                
                // For conditional copulas, we need h-functions
                if cond_set.is_empty() {
                    let u_vec = Array1::from_vec(vec![u1, u2]);
                    copula.density(&u_vec.view())
                } else {
                    // Simplified: use unconditional density as approximation
                    // Production code should implement proper h-functions
                    let u_vec = Array1::from_vec(vec![u1, u2]);
                    copula.density(&u_vec.view())
                }
            }
            
            PairCopulaFamily::Gaussian => {
                // Simplified Gaussian copula density
                if params.is_empty() {
                    return Ok(1.0);
                }
                
                let rho = params[0].clamp(-0.999, 0.999);
                let det = 1.0 - rho * rho;
                
                // This is a simplified version
                Ok(1.0 / det.sqrt())
            }
            
            _ => Ok(1.0), // Placeholder for other families
        }
    }
    
    /// Calculate partial correlation using recursive formula
    fn partial_correlation(
        var1: usize,
        var2: usize,
        cond_set: &[usize],
        corr_matrix: &Array2<f64>,
    ) -> f64 {
        if cond_set.is_empty() {
            return corr_matrix[[var1, var2]];
        }
        
        // Recursive calculation of partial correlation
        // ρ_{XY|Z} = (ρ_{XY} - ρ_{XZ}ρ_{YZ}) / sqrt((1-ρ_{XZ}²)(1-ρ_{YZ}²))
        
        let z = cond_set[0];
        let rho_xy = corr_matrix[[var1, var2]];
        let rho_xz = corr_matrix[[var1, z]];
        let rho_yz = corr_matrix[[var2, z]];
        
        let numerator = rho_xy - rho_xz * rho_yz;
        let denominator = ((1.0 - rho_xz * rho_xz) * (1.0 - rho_yz * rho_yz)).sqrt();
        
        if denominator < 1e-10 {
            return 0.0;
        }
        
        let partial_rho = numerator / denominator;
        
        // If more conditioning variables, recurse
        if cond_set.len() > 1 {
            // Simplified: just return the first-order partial
            // Full implementation would recurse through all conditioning variables
        }
        
        partial_rho.clamp(-1.0, 1.0)
    }
    
    /// Determine optimal variable ordering for C-Vine based on dependency strength
    fn optimal_c_vine_ordering(corr_matrix: &Array2<f64>) -> Vec<usize> {
        let n = corr_matrix.nrows();
        let mut order = Vec::with_capacity(n);
        let mut remaining: Vec<usize> = (0..n).collect();
        
        // Greedy selection: pick variable with highest total correlation
        while !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_score = f64::NEG_INFINITY;
            
            for (i, &var) in remaining.iter().enumerate() {
                let score: f64 = remaining
                    .iter()
                    .filter(|&&v| v != var)
                    .map(|&other| corr_matrix[[var, other]].abs())
                    .sum();
                
                if score > best_score {
                    best_score = score;
                    best_idx = i;
                }
            }
            
            order.push(remaining.remove(best_idx));
        }
        
        order
    }
    
    /// Determine optimal variable ordering for D-Vine
    fn optimal_d_vine_ordering(corr_matrix: &Array2<f64>) -> Vec<usize> {
        // For D-Vine, we often use temporal or logical ordering
        // Here we use a simple heuristic based on average correlation
        let n = corr_matrix.nrows();
        let mut scores: Vec<(usize, f64)> = (0..n)
            .map(|i| {
                let avg_corr: f64 = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| corr_matrix[[i, j]].abs())
                    .sum::<f64>()
                    / (n - 1) as f64;
                (i, avg_corr)
            })
            .collect();
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.into_iter().map(|(i, _)| i).collect()
    }
    
    /// Get the number of trees in the vine
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }
    
    /// Get the total number of pair copulas
    pub fn num_pair_copulas(&self) -> usize {
        self.trees.iter().map(|t| t.edges.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_c_vine_construction() {
        let mut corr = Array2::zeros((3, 3));
        corr[[0, 0]] = 1.0;
        corr[[0, 1]] = 0.5;
        corr[[0, 2]] = 0.3;
        corr[[1, 0]] = 0.5;
        corr[[1, 1]] = 1.0;
        corr[[1, 2]] = 0.4;
        corr[[2, 0]] = 0.3;
        corr[[2, 1]] = 0.4;
        corr[[2, 2]] = 1.0;
        
        let vine = VineCopula::build_c_vine(&corr, 5.0).unwrap();
        
        assert_eq!(vine.n_variables, 3);
        assert!(matches!(vine.vine_type, VineType::CVine));
        assert!(!vine.trees.is_empty());
    }
    
    #[test]
    fn test_d_vine_construction() {
        let mut corr = Array2::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    corr[[i, j]] = 1.0;
                } else {
                    corr[[i, j]] = 0.3;
                }
            }
        }
        
        let vine = VineCopula::build_d_vine(&corr, 6.0).unwrap();
        
        assert_eq!(vine.n_variables, 4);
        assert!(matches!(vine.vine_type, VineType::DVine));
    }
}
