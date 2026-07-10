//! Continuous Batching Queue for LLM Inference
//!
//! This module implements a dynamic batching system that groups incoming
//! inference requests into optimal GPU batch sizes without introducing
//! artificial latency delays.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex, Notify};
use tracing::{info, debug, warn};

/// Maximum queue size before backpressure is applied
const MAX_QUEUE_SIZE: usize = 10_000;

/// Default maximum batch size for GPU inference
const DEFAULT_MAX_BATCH_SIZE: usize = 32;

/// Default maximum wait time for batching (microseconds)
const DEFAULT_BATCH_TIMEOUT_US: u64 = 100;

/// An inference request in the queue
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Unique request ID
    pub id: u64,
    /// Token IDs (input sequence)
    pub input_ids: Vec<u32>,
    /// Maximum new tokens to generate
    pub max_new_tokens: u32,
    /// Priority (higher = more urgent)
    pub priority: u8,
    /// Timestamp when request was created
    pub created_at: Instant,
    /// Response channel for sending results
    pub response_tx: Option<mpsc::Sender<InferenceResponse>>,
}

/// Response from an inference operation
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    /// Original request ID
    pub request_id: u64,
    /// Generated token IDs
    pub output_ids: Vec<u32>,
    /// Time to first token (microseconds)
    pub ttft_us: u64,
    /// Total inference time (microseconds)
    pub total_time_us: u64,
}

/// A batch of requests ready for GPU inference
pub struct InferenceBatch {
    /// Requests in this batch
    pub requests: Vec<InferenceRequest>,
    /// Flattened input IDs for all requests
    pub flattened_ids: Vec<u32>,
    /// Sequence lengths for each request
    pub seq_lengths: Vec<u32>,
    /// Batch creation timestamp
    pub created_at: Instant,
}

impl InferenceBatch {
    /// Create a new batch from a list of requests
    pub fn new(requests: Vec<InferenceRequest>) -> Self {
        let mut flattened_ids = Vec::new();
        let mut seq_lengths = Vec::new();

        for req in &requests {
            let len = req.input_ids.len() as u32;
            seq_lengths.push(len);
            flattened_ids.extend_from_slice(&req.input_ids);
        }

        Self {
            requests,
            flattened_ids,
            seq_lengths,
            created_at: Instant::now(),
        }
    }

    /// Get the batch size (number of requests)
    #[inline]
    pub fn batch_size(&self) -> usize {
        self.requests.len()
    }

    /// Get total number of tokens in the batch
    #[inline]
    pub fn total_tokens(&self) -> usize {
        self.flattened_ids.len()
    }
}

/// Configuration for the continuous batching queue
#[derive(Debug, Clone)]
pub struct BatchingConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Minimum batch size to trigger immediate execution
    pub min_batch_size: usize,
    /// Maximum wait time for batching
    pub batch_timeout: Duration,
    /// Enable priority scheduling
    pub enable_priority: bool,
}

impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            min_batch_size: 4,
            batch_timeout: Duration::from_micros(DEFAULT_BATCH_TIMEOUT_US),
            enable_priority: true,
        }
    }
}

/// Continuous batching queue manager
pub struct ContinuousBatchingQueue {
    /// Pending requests waiting to be batched
    pending: Arc<Mutex<VecDeque<InferenceRequest>>>,
    /// Notification for new requests
    notify: Arc<Notify>,
    /// Configuration
    config: BatchingConfig,
    /// Statistics
    stats: Arc<Mutex<BatchingStats>>,
}

/// Statistics for the batching queue
#[derive(Debug, Default)]
pub struct BatchingStats {
    /// Total requests processed
    pub total_requests: u64,
    /// Total batches created
    pub total_batches: u64,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Average wait time (microseconds)
    pub avg_wait_time_us: f64,
    /// Requests dropped due to queue full
    pub dropped_requests: u64,
}

impl ContinuousBatchingQueue {
    /// Create a new continuous batching queue
    pub fn new(config: BatchingConfig) -> Self {
        Self {
            pending: Arc::new(Mutex::new(VecDeque::with_capacity(config.max_batch_size))),
            notify: Arc::new(Notify::new()),
            config,
            stats: Arc::new(Mutex::new(BatchingStats::default())),
        }
    }

    /// Submit a new inference request
    pub async fn submit(&self, request: InferenceRequest) -> Result<(), &'static str> {
        let mut pending = self.pending.lock().await;

        if pending.len() >= MAX_QUEUE_SIZE {
            let mut stats = self.stats.lock().await;
            stats.dropped_requests += 1;
            return Err("Queue is full");
        }

        pending.push_back(request);
        
        // Notify the batching loop
        self.notify.notify_one();

        Ok(())
    }

    /// Get the next batch of requests
    /// Returns None if no requests are available and timeout expires
    pub async fn get_next_batch(&self) -> Option<InferenceBatch> {
        let start = Instant::now();
        
        loop {
            {
                let mut pending = self.pending.lock().await;
                
                // Check if we have enough requests for a batch
                if pending.len() >= self.config.min_batch_size {
                    // Take up to max_batch_size requests
                    let count = std::cmp::min(pending.len(), self.config.max_batch_size);
                    let requests: Vec<InferenceRequest> = pending.drain(..count).collect();
                    
                    if !requests.is_empty() {
                        let mut stats = self.stats.lock().await;
                        stats.total_requests += requests.len() as u64;
                        stats.total_batches += 1;
                        
                        // Update average batch size
                        let n = stats.total_batches as f64;
                        stats.avg_batch_size = 
                            (stats.avg_batch_size * (n - 1.0) + requests.len() as f64) / n;
                        
                        return Some(InferenceBatch::new(requests));
                    }
                }
                
                // If we have some requests but not enough, check timeout
                if !pending.is_empty() && start.elapsed() >= self.config.batch_timeout {
                    let count = pending.len();
                    let requests: Vec<InferenceRequest> = pending.drain(..count).collect();
                    
                    if !requests.is_empty() {
                        let mut stats = self.stats.lock().await;
                        stats.total_requests += requests.len() as u64;
                        stats.total_batches += 1;
                        
                        let n = stats.total_batches as f64;
                        stats.avg_batch_size = 
                            (stats.avg_batch_size * (n - 1.0) + requests.len() as f64) / n;
                        
                        let wait_time = start.elapsed().as_micros() as f64;
                        stats.avg_wait_time_us = 
                            (stats.avg_wait_time_us * (n - 1.0) + wait_time) / n;
                        
                        return Some(InferenceBatch::new(requests));
                    }
                }
            }
            
            // Wait for notification or timeout
            let timeout = self.config.batch_timeout.saturating_sub(start.elapsed());
            if timeout.is_zero() {
                return None;
            }
            
            tokio::select! {
                _ = self.notify.notified() => {
                    continue;
                }
                _ = tokio::time::sleep(timeout) => {
                    // Timeout expired, try to create a batch with what we have
                    continue;
                }
            }
        }
    }

    /// Get current queue length
    pub async fn queue_length(&self) -> usize {
        let pending = self.pending.lock().await;
        pending.len()
    }

    /// Get statistics
    pub async fn get_stats(&self) -> BatchingStats {
        let stats = self.stats.lock().await;
        stats.clone()
    }

    /// Clear all pending requests
    pub async fn clear(&self) {
        let mut pending = self.pending.lock().await;
        pending.clear();
    }
}

/// Priority-aware request scheduler
pub struct PriorityScheduler {
    /// High priority queue
    high_priority: VecDeque<InferenceRequest>,
    /// Normal priority queue
    normal_priority: VecDeque<InferenceRequest>,
    /// Low priority queue
    low_priority: VecDeque<InferenceRequest>,
}

impl PriorityScheduler {
    pub fn new() -> Self {
        Self {
            high_priority: VecDeque::new(),
            normal_priority: VecDeque::new(),
            low_priority: VecDeque::new(),
        }
    }

    /// Add a request to the appropriate queue based on priority
    pub fn enqueue(&mut self, request: InferenceRequest) {
        match request.priority {
            0..=2 => self.low_priority.push_back(request),
            3..=4 => self.normal_priority.push_back(request),
            5..=u8::MAX => self.high_priority.push_back(request),
        }
    }

    /// Get the next request (highest priority first)
    pub fn dequeue(&mut self) -> Option<InferenceRequest> {
        self.high_priority.pop_front()
            .or_else(|| self.normal_priority.pop_front())
            .or_else(|| self.low_priority.pop_front())
    }

    /// Get total number of queued requests
    pub fn len(&self) -> usize {
        self.high_priority.len() + self.normal_priority.len() + self.low_priority.len()
    }

    /// Check if any queue has requests
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drain up to `max` requests respecting priority order
    pub fn drain_up_to(&mut self, max: usize) -> Vec<InferenceRequest> {
        let mut result = Vec::with_capacity(max);
        
        // First, take from high priority
        while result.len() < max && let Some(req) = self.high_priority.pop_front() {
            result.push(req);
        }
        
        // Then from normal priority
        while result.len() < max && let Some(req) = self.normal_priority.pop_front() {
            result.push(req);
        }
        
        // Finally from low priority
        while result.len() < max && let Some(req) = self.low_priority.pop_front() {
            result.push(req);
        }
        
        result
    }
}

impl Default for PriorityScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_batching() {
        let config = BatchingConfig {
            max_batch_size: 4,
            min_batch_size: 2,
            batch_timeout: Duration::from_millis(10),
            ..Default::default()
        };
        
        let queue = ContinuousBatchingQueue::new(config);
        
        // Submit some requests
        for i in 0..3 {
            let request = InferenceRequest {
                id: i,
                input_ids: vec![1, 2, 3],
                max_new_tokens: 10,
                priority: 3,
                created_at: Instant::now(),
                response_tx: None,
            };
            queue.submit(request).await.unwrap();
        }
        
        // Get a batch
        let batch = queue.get_next_batch().await;
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert!(batch.batch_size() >= 2);
        assert!(batch.batch_size() <= 4);
    }

    #[test]
    fn test_priority_scheduler() {
        let mut scheduler = PriorityScheduler::new();
        
        // Add requests with different priorities
        for i in 0..6 {
            scheduler.enqueue(InferenceRequest {
                id: i,
                input_ids: vec![1],
                max_new_tokens: 1,
                priority: if i < 2 { 5 } else if i < 4 { 3 } else { 1 },
                created_at: Instant::now(),
                response_tx: None,
            });
        }
        
        // Dequeue should respect priority
        let first = scheduler.dequeue().unwrap();
        assert!(first.priority >= 5);
        
        let second = scheduler.dequeue().unwrap();
        assert!(second.priority >= 5);
        
        let third = scheduler.dequeue().unwrap();
        assert!(third.priority >= 3 && third.priority < 5);
    }

    #[test]
    fn test_batch_creation() {
        let requests = vec![
            InferenceRequest {
                id: 1,
                input_ids: vec![1, 2, 3],
                max_new_tokens: 10,
                priority: 3,
                created_at: Instant::now(),
                response_tx: None,
            },
            InferenceRequest {
                id: 2,
                input_ids: vec![4, 5],
                max_new_tokens: 5,
                priority: 3,
                created_at: Instant::now(),
                response_tx: None,
            },
        ];
        
        let batch = InferenceBatch::new(requests);
        assert_eq!(batch.batch_size(), 2);
        assert_eq!(batch.total_tokens(), 5);
        assert_eq!(batch.seq_lengths, vec![3, 2]);
        assert_eq!(batch.flattened_ids, vec![1, 2, 3, 4, 5]);
    }
}
