//! Atomic Weight Updater for Thread-Safe STDP
//! 
//! Provides lock-free, atomic operations for updating synaptic weights
//! in parallel SNN simulations. Prevents race conditions when multiple
//! spikes hit the same synapse simultaneously.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use crate::plasticity::stdp_learning_rule::{WEIGHT_SCALE, MAX_WEIGHT, MIN_WEIGHT};

/// Lock-free synaptic weight matrix with atomic updates
pub struct AtomicWeightMatrix {
    /// Flattened weight matrix (row-major: pre_idx * n_post + post_idx)
    weights: Vec<AtomicI64>,
    /// Number of pre-synaptic neurons
    n_pre: usize,
    /// Number of post-synaptic neurons
    n_post: usize,
    /// Update counter for statistics
    update_count: AtomicU64,
    /// Contention counter (CAS failures)
    contention_count: AtomicU64,
}

impl AtomicWeightMatrix {
    /// Create a new weight matrix
    #[inline]
    pub fn new(n_pre: usize, n_post: usize, initial_weight: f32) -> Self {
        let total_weights = n_pre * n_post;
        let initial_fixed = ((initial_weight * WEIGHT_SCALE as f32) as i64)
            .clamp(MIN_WEIGHT, MAX_WEIGHT);

        let weights: Vec<AtomicI64> = (0..total_weights)
            .map(|_| AtomicI64::new(initial_fixed))
            .collect();

        Self {
            weights,
            n_pre,
            n_post,
            update_count: AtomicU64::new(0),
            contention_count: AtomicU64::new(0),
        }
    }

    /// Get linear index from pre/post indices
    #[inline]
    fn get_index(&self, pre_idx: usize, post_idx: usize) -> Option<usize> {
        if pre_idx < self.n_pre && post_idx < self.n_post {
            Some(pre_idx * self.n_post + post_idx)
        } else {
            None
        }
    }

    /// Get weight value as f32
    #[inline]
    pub fn get_weight(&self, pre_idx: usize, post_idx: usize) -> Option<f32> {
        self.get_index(pre_idx, post_idx).map(|idx| {
            self.weights[idx].load(Ordering::Acquire) as f32 / WEIGHT_SCALE as f32
        })
    }

    /// Get raw fixed-point weight
    #[inline]
    pub fn get_weight_fixed(&self, pre_idx: usize, post_idx: usize) -> Option<i64> {
        self.get_index(pre_idx, post_idx)
            .map(|idx| self.weights[idx].load(Ordering::Acquire))
    }

    /// Atomically add delta to weight with bounds checking
    /// Returns true if update succeeded, false on repeated contention
    #[inline]
    pub fn atomic_add_weight(
        &self,
        pre_idx: usize,
        post_idx: usize,
        delta: i64,
    ) -> Result<i64, &'static str> {
        let idx = match self.get_index(pre_idx, post_idx) {
            Some(i) => i,
            None => return Err("Invalid indices"),
        };

        let mut retries = 0;
        const MAX_RETRIES: u32 = 100;

        loop {
            let current = self.weights[idx].load(Ordering::Acquire);
            
            // Apply delta with hard bounds
            let mut new_value = current + delta;
            if new_value > MAX_WEIGHT {
                new_value = MAX_WEIGHT;
            } else if new_value < MIN_WEIGHT {
                new_value = MIN_WEIGHT;
            }

            match self.weights[idx].compare_exchange_weak(
                current,
                new_value,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.update_count.fetch_add(1, Ordering::Relaxed);
                    return Ok(new_value);
                }
                Err(_) => {
                    retries += 1;
                    if retries >= MAX_RETRIES {
                        self.contention_count.fetch_add(1, Ordering::Relaxed);
                        return Err("Contention limit exceeded");
                    }
                    // Exponential backoff could be added here for high contention
                }
            }
        }
    }

    /// Weight-dependent atomic update (Oja's rule style soft bounds)
    #[inline]
    pub fn atomic_add_weight_dependent(
        &self,
        pre_idx: usize,
        post_idx: usize,
        ltp_delta: i64,
        ltd_delta: i64,
    ) -> Result<i64, &'static str> {
        let idx = match self.get_index(pre_idx, post_idx) {
            Some(i) => i,
            None => return Err("Invalid indices"),
        };

        let mut retries = 0;
        const MAX_RETRIES: u32 = 100;

        loop {
            let current = self.weights[idx].load(Ordering::Acquire);

            // Soft bound scaling
            let ltp_scaled = if ltp_delta > 0 {
                ltp_delta * (MAX_WEIGHT - current) / MAX_WEIGHT
            } else {
                ltp_delta
            };

            let ltd_scaled = if ltd_delta > 0 {
                ltd_delta * current / MAX_WEIGHT
            } else {
                ltd_delta
            };

            let mut new_value = current + ltp_scaled - ltd_scaled;

            // Hard bounds safety net
            if new_value > MAX_WEIGHT {
                new_value = MAX_WEIGHT;
            } else if new_value < MIN_WEIGHT {
                new_value = MIN_WEIGHT;
            }

            match self.weights[idx].compare_exchange_weak(
                current,
                new_value,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.update_count.fetch_add(1, Ordering::Relaxed);
                    return Ok(new_value);
                }
                Err(_) => {
                    retries += 1;
                    if retries >= MAX_RETRIES {
                        self.contention_count.fetch_add(1, Ordering::Relaxed);
                        return Err("Contention limit exceeded");
                    }
                }
            }
        }
    }

    /// Multiply weight by scalar (for homeostatic scaling)
    #[inline]
    pub fn atomic_multiply_weight(
        &self,
        pre_idx: usize,
        post_idx: usize,
        scalar_numerator: i64,
        scalar_denominator: i64,
    ) -> Result<i64, &'static str> {
        let idx = match self.get_index(pre_idx, post_idx) {
            Some(i) => i,
            None => return Err("Invalid indices"),
        };

        let mut retries = 0;
        const MAX_RETRIES: u32 = 100;

        loop {
            let current = self.weights[idx].load(Ordering::Acquire);
            let new_value = (current * scalar_numerator) / scalar_denominator;
            let clamped = new_value.clamp(MIN_WEIGHT, MAX_WEIGHT);

            match self.weights[idx].compare_exchange_weak(
                current,
                clamped,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(clamped),
                Err(_) => {
                    retries += 1;
                    if retries >= MAX_RETRIES {
                        self.contention_count.fetch_add(1, Ordering::Relaxed);
                        return Err("Contention limit exceeded");
                    }
                }
            }
        }
    }

    /// Apply global homeostatic scaling to all weights
    #[inline]
    pub fn apply_global_scaling(&self, scale_factor_num: i64, scale_factor_den: i64) {
        for weight in &self.weights {
            let mut current = weight.load(Ordering::Acquire);
            
            loop {
                let scaled = (current * scale_factor_num) / scale_factor_den;
                let clamped = scaled.clamp(MIN_WEIGHT, MAX_WEIGHT);

                match weight.compare_exchange_weak(
                    current,
                    clamped,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(prev) => current = prev, // Retry with updated value
                }
            }
        }
    }

    /// Get statistics
    #[inline]
    pub fn stats(&self) -> WeightMatrixStats {
        WeightMatrixStats {
            n_pre: self.n_pre,
            n_post: self.n_post,
            total_weights: self.n_pre * self.n_post,
            update_count: self.update_count.load(Ordering::Relaxed),
            contention_count: self.contention_count.load(Ordering::Relaxed),
        }
    }

    /// Get average weight across all synapses
    #[inline]
    pub fn average_weight(&self) -> f32 {
        let sum: i64 = self.weights.iter()
            .map(|w| w.load(Ordering::Relaxed))
            .sum();
        (sum / (self.n_pre * self.n_post) as i64) as f32 / WEIGHT_SCALE as f32
    }

    /// Reset all weights to initial value
    #[inline]
    pub fn reset(&self, initial_weight: f32) {
        let initial_fixed = ((initial_weight * WEIGHT_SCALE as f32) as i64)
            .clamp(MIN_WEIGHT, MAX_WEIGHT);
        
        for weight in &self.weights {
            weight.store(initial_fixed, Ordering::Release);
        }
        
        self.update_count.store(0, Ordering::Relaxed);
        self.contention_count.store(0, Ordering::Relaxed);
    }
}

/// Statistics about weight matrix operations
#[derive(Debug, Clone, Copy)]
pub struct WeightMatrixStats {
    pub n_pre: usize,
    pub n_post: usize,
    pub total_weights: usize,
    pub update_count: u64,
    pub contention_count: u64,
}

/// Sparse atomic weight updater for efficient sparse connectivity
pub struct SparseAtomicWeights {
    /// Pre-synaptic indices for each connection
    pre_indices: Vec<usize>,
    /// Post-synaptic indices for each connection
    post_indices: Vec<usize>,
    /// Weight values (atomic)
    weights: Vec<AtomicI64>,
    /// Number of connections
    n_connections: usize,
}

impl SparseAtomicWeights {
    /// Create sparse weight matrix from connection list
    #[inline]
    pub fn new(connections: &[(usize, usize)], initial_weight: f32) -> Self {
        let initial_fixed = ((initial_weight * WEIGHT_SCALE as f32) as i64)
            .clamp(MIN_WEIGHT, MAX_WEIGHT);

        let pre_indices: Vec<usize> = connections.iter().map(|(pre, _)| *pre).collect();
        let post_indices: Vec<usize> = connections.iter().map(|(_, post)| *post).collect();
        let weights: Vec<AtomicI64> = connections
            .iter()
            .map(|_| AtomicI64::new(initial_fixed))
            .collect();

        Self {
            pre_indices,
            post_indices,
            weights,
            n_connections: connections.len(),
        }
    }

    /// Find connection index for given pre/post pair (linear search)
    #[inline]
    fn find_connection(&self, pre_idx: usize, post_idx: usize) -> Option<usize> {
        for i in 0..self.n_connections {
            if self.pre_indices[i] == pre_idx && self.post_indices[i] == post_idx {
                return Some(i);
            }
        }
        None
    }

    /// Get weight for specific connection
    #[inline]
    pub fn get_weight(&self, conn_idx: usize) -> Option<f32> {
        if conn_idx < self.n_connections {
            Some(self.weights[conn_idx].load(Ordering::Acquire) as f32 / WEIGHT_SCALE as f32)
        } else {
            None
        }
    }

    /// Update weight for specific connection
    #[inline]
    pub fn update_weight(&self, conn_idx: usize, delta: i64) -> Result<i64, &'static str> {
        if conn_idx >= self.n_connections {
            return Err("Invalid connection index");
        }

        let mut retries = 0;
        const MAX_RETRIES: u32 = 100;

        loop {
            let current = self.weights[conn_idx].load(Ordering::Acquire);
            let mut new_value = current + delta;
            
            if new_value > MAX_WEIGHT {
                new_value = MAX_WEIGHT;
            } else if new_value < MIN_WEIGHT {
                new_value = MIN_WEIGHT;
            }

            match self.weights[conn_idx].compare_exchange_weak(
                current,
                new_value,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(new_value),
                Err(_) => {
                    retries += 1;
                    if retries >= MAX_RETRIES {
                        return Err("Contention limit exceeded");
                    }
                }
            }
        }
    }

    /// Get number of connections
    #[inline]
    pub fn n_connections(&self) -> usize {
        self.n_connections
    }

    /// Iterate over all connections (pre, post, weight)
    #[inline]
    pub fn iter_connections(&self) -> impl Iterator<Item = (usize, usize, f32)> + '_ {
        (0..self.n_connections).filter_map(move |i| {
            let weight = self.weights[i].load(Ordering::Relaxed) as f32 / WEIGHT_SCALE as f32;
            Some((self.pre_indices[i], self.post_indices[i], weight))
        })
    }
}

/// Batch weight updater for applying multiple updates efficiently
pub struct BatchWeightUpdater<'a> {
    weight_matrix: &'a AtomicWeightMatrix,
    /// Pending updates: (pre_idx, post_idx, delta)
    pending_updates: Vec<(usize, usize, i64)>,
    /// Maximum batch size before flush
    max_batch_size: usize,
}

impl<'a> BatchWeightUpdater<'a> {
    #[inline]
    pub fn new(weight_matrix: &'a AtomicWeightMatrix, max_batch_size: usize) -> Self {
        Self {
            weight_matrix,
            pending_updates: Vec::with_capacity(max_batch_size),
            max_batch_size,
        }
    }

    /// Queue an update
    #[inline]
    pub fn queue_update(&mut self, pre_idx: usize, post_idx: usize, delta: i64) {
        self.pending_updates.push((pre_idx, post_idx, delta));
        
        // Auto-flush if batch is full
        if self.pending_updates.len() >= self.max_batch_size {
            let _ = self.flush();
        }
    }

    /// Flush all pending updates
    #[inline]
    pub fn flush(&mut self) -> Result<usize, &'static str> {
        let mut success_count = 0;
        
        for (pre_idx, post_idx, delta) in self.pending_updates.drain(..) {
            if self.weight_matrix.atomic_add_weight(pre_idx, post_idx, delta).is_ok() {
                success_count += 1;
            }
        }
        
        Ok(success_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_matrix_creation() {
        let matrix = AtomicWeightMatrix::new(10, 10, 0.5);
        assert_eq!(matrix.stats().total_weights, 100);
        assert!((matrix.average_weight() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_atomic_weight_update() {
        let matrix = AtomicWeightMatrix::new(4, 4, 0.5);
        
        // Add positive delta
        let result = matrix.atomic_add_weight(0, 0, WEIGHT_SCALE / 10);
        assert!(result.is_ok());
        
        let new_weight = matrix.get_weight(0, 0).unwrap();
        assert!(new_weight > 0.5);
    }

    #[test]
    fn test_weight_bounds_enforcement() {
        let matrix = AtomicWeightMatrix::new(4, 4, 0.5);
        
        // Try to exceed maximum
        for _ in 0..100 {
            let _ = matrix.atomic_add_weight(0, 0, WEIGHT_SCALE);
        }
        
        assert!(matrix.get_weight(0, 0).unwrap() <= 1.0);
        
        // Try to go below minimum
        for _ in 0..100 {
            let _ = matrix.atomic_add_weight(0, 0, -WEIGHT_SCALE);
        }
        
        assert!(matrix.get_weight(0, 0).unwrap() >= 0.01);
    }

    #[test]
    fn test_sparse_weights() {
        let connections = vec![(0, 0), (0, 2), (1, 1), (2, 0)];
        let sparse = SparseAtomicWeights::new(&connections, 0.3);
        
        assert_eq!(sparse.n_connections(), 4);
        
        // Update first connection
        let result = sparse.update_weight(0, WEIGHT_SCALE / 10);
        assert!(result.is_ok());
        
        let weight = sparse.get_weight(0).unwrap();
        assert!(weight > 0.3);
    }

    #[test]
    fn test_batch_updater() {
        let matrix = AtomicWeightMatrix::new(8, 8, 0.5);
        let mut batch = BatchWeightUpdater::new(&matrix, 10);
        
        // Queue multiple updates
        for i in 0..5 {
            batch.queue_update(i, i, WEIGHT_SCALE / 20);
        }
        
        // Flush and verify
        let success = batch.flush().unwrap();
        assert_eq!(success, 5);
    }
}
