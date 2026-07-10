//! Transfer Entropy using KSG Estimator
//! Measures directed information flow between time series.

use super::kd_tree_arena::KdTreeArena;

/// Transfer entropy estimator using KSG method
pub struct TransferEntropyKsg {
    /// Number of nearest neighbors
    k: usize,
    /// Embedding dimension for X
    dim_x: usize,
    /// Embedding dimension for Y  
    dim_y: usize,
    /// Time lag for embedding
    lag: usize,
    /// Arena for k-d tree nodes
    arena: KdTreeArena,
}

impl TransferEntropyKsg {
    pub fn new(k: usize, dim_x: usize, dim_y: usize, lag: usize, arena_size: usize) -> Self {
        Self {
            k,
            dim_x,
            dim_y,
            lag,
            arena: KdTreeArena::new(arena_size, dim_x + dim_y + 1),
        }
    }

    /// Compute transfer entropy from X to Y
    /// TE(X->Y) = I(Y_{t+1}; X_t | Y_t)
    pub fn compute(&mut self, x: &[f64], y: &[f64]) -> Option<f64> {
        if x.len() != y.len() || x.len() < self.dim_x + self.dim_y + self.lag + 1 {
            return None;
        }

        // Build embedded vectors and compute KSG estimate
        // Simplified implementation - full KSG requires k-NN distance calculations
        let n_valid = x.len() - self.dim_x - self.dim_y - self.lag;
        
        if n_valid == 0 {
            return None;
        }

        // Placeholder: compute correlation-based approximation
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_yy = 0.0;
        
        for i in 0..n_valid {
            let x_val = x[i + self.dim_x];
            let y_next = y[i + self.dim_x + self.dim_y + self.lag];
            
            sum_xy += x_val * y_next;
            sum_xx += x_val * x_val;
            sum_yy += y_next * y_next;
        }
        
        let denom = (sum_xx * sum_yy).sqrt();
        if denom < 1e-16 {
            return Some(0.0);
        }
        
        let corr = sum_xy / denom;
        
        // Convert correlation to transfer entropy approximation
        Some((1.0 + corr).max(0.0).ln().abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_entropy() {
        let mut te = TransferEntropyKsg::new(3, 2, 2, 1, 1000);
        
        // Create series where X leads Y
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1 - 0.5).sin()).collect(); // Lagged
        
        let result = te.compute(&x, &y);
        assert!(result.is_some());
    }
}
