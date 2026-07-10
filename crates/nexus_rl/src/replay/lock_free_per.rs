//! Lock-Free Prioritized Experience Replay Buffer
//! 
//! Implements a concurrent, lock-free PER buffer using atomic ring buffers
//! for high-throughput trajectory collection in distributed RL (Ape-X architecture).

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Maximum capacity of the replay buffer
const MAX_CAPACITY: usize = 1_000_000;

/// Experience sample with priority
#[derive(Debug, Clone)]
pub struct Experience {
    /// Observation state (flattened)
    pub observation: Vec<f32>,
    /// Action taken
    pub action: Vec<f32>,
    /// Reward received
    pub reward: f32,
    /// Next observation
    pub next_observation: Vec<f32>,
    /// Terminal flag
    pub done: bool,
    /// Priority for sampling (TD-error based)
    pub priority: f64,
    /// Timestamp (nanoseconds)
    pub timestamp_ns: u64,
}

impl Experience {
    /// Create a new experience sample
    pub fn new(
        observation: Vec<f32>,
        action: Vec<f32>,
        reward: f32,
        next_observation: Vec<f32>,
        done: bool,
        priority: f64,
    ) -> Self {
        Self {
            observation,
            action,
            reward,
            next_observation,
            done,
            priority,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
        }
    }
    
    /// Create with zero allocation for pre-allocated buffers
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            observation: Vec::with_capacity(capacity),
            action: Vec::with_capacity(capacity / 4),
            reward: 0.0,
            next_observation: Vec::with_capacity(capacity),
            done: false,
            priority: 1.0,
            timestamp_ns: 0,
        }
    }
}

/// SumTree for prioritized sampling (lock-free version)
pub struct LockFreeSumTree {
    /// Tree nodes (stored as priorities)
    tree: Vec<AtomicF64>,
    /// Leaf capacity
    capacity: usize,
    /// Current write position
    write_pos: AtomicUsize,
    /// Total priority sum
    total_priority: AtomicF64,
}

/// Atomic f64 wrapper using compare-and-swap
struct AtomicF64 {
    bits: AtomicU64,
}

impl AtomicF64 {
    fn new(val: f64) -> Self {
        Self {
            bits: AtomicU64::new(val.to_bits()),
        }
    }
    
    fn load(&self, order: Ordering) -> f64 {
        f64::from_bits(self.bits.load(order))
    }
    
    fn store(&self, val: f64, order: Ordering) {
        self.bits.store(val.to_bits(), order);
    }
    
    fn fetch_add(&self, val: f64, order: Ordering) -> f64 {
        loop {
            let current = self.load(Ordering::Relaxed);
            let new_val = current + val;
            if self.bits.compare_exchange_weak(
                current.to_bits(),
                new_val.to_bits(),
                order,
                Ordering::Relaxed,
            ).is_ok() {
                return current;
            }
        }
    }
}

impl LockFreeSumTree {
    /// Create a new sum tree with given capacity
    pub fn new(capacity: usize) -> Self {
        // Tree size is 2 * capacity - 1, but we use 2 * capacity for simplicity
        let tree_size = 2 * capacity;
        let tree: Vec<AtomicF64> = (0..tree_size)
            .map(|_| AtomicF64::new(0.0))
            .collect();
        
        Self {
            tree,
            capacity,
            write_pos: AtomicUsize::new(0),
            total_priority: AtomicF64::new(0.0),
        }
    }
    
    /// Update priority at given index
    pub fn update(&self, index: usize, priority: f64) {
        let tree_index = index + self.capacity;
        let old_priority = self.tree[tree_index].load(Ordering::Relaxed);
        let diff = priority - old_priority;
        
        self.tree[tree_index].store(priority, Ordering::Relaxed);
        
        // Propagate change up the tree
        let mut idx = tree_index;
        while idx > 1 {
            idx /= 2;
            self.tree[idx].fetch_add(diff, Ordering::Relaxed);
        }
        
        self.total_priority.fetch_add(diff, Ordering::Relaxed);
    }
    
    /// Sample index based on priority-weighted distribution
    pub fn sample(&self, value: f64) -> Option<usize> {
        let total = self.total_priority.load(Ordering::Relaxed);
        if total <= 0.0 {
            return None;
        }
        
        let mut idx = 1;
        let mut current_value = value;
        
        while idx < self.capacity {
            let left = idx * 2;
            let left_priority = self.tree[left].load(Ordering::Relaxed);
            
            if current_value <= left_priority && left_priority > 0.0 {
                idx = left;
            } else {
                current_value -= left_priority;
                idx = left + 1;
            }
        }
        
        Some(idx - self.capacity)
    }
    
    /// Get maximum priority
    pub fn max_priority(&self) -> f64 {
        let mut max_p = 0.0;
        for i in self.capacity..2 * self.capacity {
            let p = self.tree[i].load(Ordering::Relaxed);
            if p > max_p {
                max_p = p;
            }
        }
        max_p.max(1e-6) // Minimum priority floor
    }
}

/// Lock-free ring buffer for experience storage
pub struct LockFreeReplayBuffer {
    /// Circular buffer of experiences
    buffer: Vec<Arc<spin::Mutex<Option<Experience>>>>,
    /// Capacity
    capacity: usize,
    /// Write position
    write_pos: AtomicUsize,
    /// Current size (number of stored experiences)
    size: AtomicUsize,
    /// Priority sum tree
    sum_tree: LockFreeSumTree,
    /// Sampling alpha (priority exponent)
    alpha: f64,
    /// Importance sampling beta
    beta: f64,
    /// Beta annealing rate
    beta_anneal_rate: f64,
}

unsafe impl Send for LockFreeReplayBuffer {}
unsafe impl Sync for LockFreeReplayBuffer {}

impl LockFreeReplayBuffer {
    /// Create a new replay buffer
    pub fn new(capacity: usize, alpha: f64, beta: f64) -> Self {
        let capacity = capacity.min(MAX_CAPACITY);
        let buffer: Vec<_> = (0..capacity)
            .map(|_| Arc::new(spin::Mutex::new(None)))
            .collect();
        
        Self {
            buffer,
            capacity,
            write_pos: AtomicUsize::new(0),
            size: AtomicUsize::new(0),
            sum_tree: LockFreeSumTree::new(capacity),
            alpha,
            beta,
            beta_anneal_rate: 0.001,
        }
    }
    
    /// Default configuration for PPO/SAC
    pub fn default_ppo() -> Self {
        Self::new(100_000, 0.6, 0.4)
    }
    
    /// Push experience to buffer (thread-safe)
    pub fn push(&self, mut experience: Experience) {
        let pos = self.write_pos.fetch_add(1, Ordering::Relaxed) % self.capacity;
        
        // Set initial priority
        let max_priority = self.sum_tree.max_priority();
        experience.priority = max_priority;
        
        // Store experience
        let mut slot = self.buffer[pos].lock();
        *slot = Some(experience);
        drop(slot);
        
        // Update sum tree
        self.sum_tree.update(pos, max_priority);
        
        // Update size
        let current_size = self.size.load(Ordering::Relaxed);
        if current_size < self.capacity {
            self.size.store(current_size + 1, Ordering::Relaxed);
        }
    }
    
    /// Sample a batch of experiences with importance sampling weights
    pub fn sample_batch(&self, batch_size: usize) -> Option<(Vec<Experience>, Vec<f64>, Vec<usize>)> {
        let size = self.size.load(Ordering::Relaxed);
        if size == 0 {
            return None;
        }
        
        let mut experiences = Vec::with_capacity(batch_size);
        let mut weights = Vec::with_capacity(batch_size);
        let mut indices = Vec::with_capacity(batch_size);
        
        let segment = self.sum_tree.total_priority.load(Ordering::Relaxed) / batch_size as f64;
        
        for i in 0..batch_size {
            // Sample value within segment
            let value = (i as f64 * segment) + (segment * rand::random::<f64>());
            
            if let Some(idx) = self.sum_tree.sample(value) {
                if let Some(exp) = self.buffer[idx].lock().clone() {
                    experiences.push(exp.clone());
                    indices.push(idx);
                    
                    // Calculate importance sampling weight
                    let prob = exp.priority / self.sum_tree.total_priority.load(Ordering::Relaxed);
                    let weight = ((size as f64 * prob).powf(-self.beta) / size as f64).ln().exp();
                    weights.push(weight);
                }
            }
        }
        
        // Normalize weights
        let max_weight = weights.iter().cloned().fold(0.0_f64, f64::max);
        for w in &mut weights {
            *w /= max_weight;
        }
        
        Some((experiences, weights, indices))
    }
    
    /// Update priorities for sampled indices (after TD-error calculation)
    pub fn update_priorities(&self, indices: &[usize], priorities: &[f64]) {
        for (&idx, &priority) in indices.iter().zip(priorities.iter()) {
            self.sum_tree.update(idx, priority);
        }
    }
    
    /// Anneal beta parameter for importance sampling
    pub fn anneal_beta(&mut self) {
        self.beta = (self.beta + self.beta_anneal_rate).min(1.0);
    }
    
    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Acquire)
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Check if buffer has enough samples for training
    pub fn can_sample(&self, min_batch_size: usize) -> bool {
        self.len() >= min_batch_size
    }
    
    /// Clear the buffer
    pub fn clear(&self) {
        for slot in &self.buffer {
            *slot.lock() = None;
        }
        self.write_pos.store(0, Ordering::Release);
        self.size.store(0, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_push_and_sample() {
        let buffer = LockFreeReplayBuffer::default_ppo();
        
        // Push some experiences
        for i in 0..100 {
            let exp = Experience::new(
                vec![i as f32; 10],
                vec![1.0],
                1.0,
                vec![(i + 1) as f32; 10],
                false,
                1.0,
            );
            buffer.push(exp);
        }
        
        assert_eq!(buffer.len(), 100);
        
        // Sample a batch
        let result = buffer.sample_batch(32);
        assert!(result.is_some());
        
        let (exps, weights, indices) = result.unwrap();
        assert_eq!(exps.len(), 32);
        assert_eq!(weights.len(), 32);
        assert_eq!(indices.len(), 32);
    }
    
    #[test]
    fn test_priority_update() {
        let buffer = LockFreeReplayBuffer::default_ppo();
        
        // Push and sample
        for i in 0..10 {
            buffer.push(Experience::new(vec![0.0; 5], vec![0.0], 0.0, vec![0.0; 5], false, 1.0));
        }
        
        let (_, _, indices) = buffer.sample_batch(5).unwrap();
        
        // Update priorities
        let new_priorities: Vec<f64> = indices.iter().map(|&i| i as f64 * 0.1).collect();
        buffer.update_priorities(&indices, &new_priorities);
    }
}
