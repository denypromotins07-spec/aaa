//! Continuous Batching Queue
//! 
//! Dynamically groups incoming text payloads into optimal GPU batch sizes
//! without introducing artificial latency delays.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex};
use crate::inference::trt_llm_ffi_bridge::{InferenceRequest, InferenceResponse, InferenceError};

/// Configuration for the continuous batching queue
#[derive(Debug, Clone)]
pub struct BatchingConfig {
    pub max_batch_size: usize,
    pub max_wait_time: Duration,
    pub min_batch_size: usize,
}

impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_wait_time: Duration::from_micros(500),
            min_batch_size: 1,
        }
    }
}

/// Pending request with arrival time
struct PendingRequest {
    request: InferenceRequest,
    arrived_at: Instant,
    response_tx: mpsc::Sender<Result<InferenceResponse, InferenceError>>,
}

/// Continuous batching queue for LLM inference
pub struct ContinuousBatchingQueue {
    config: BatchingConfig,
    pending: Arc<Mutex<VecDeque<PendingRequest>>>,
    shutdown_tx: mpsc::Sender<()>,
}

impl ContinuousBatchingQueue {
    /// Create a new continuous batching queue
    pub fn new(config: BatchingConfig) -> Self {
        let (shutdown_tx, _) = mpsc::channel(1);
        
        Self {
            config,
            pending: Arc::new(Mutex::new(VecDeque::with_capacity(config.max_batch_size))),
            shutdown_tx,
        }
    }

    /// Start the batch processor loop
    pub fn start<F, Fut>(&self, infer_fn: F) -> tokio::task::JoinHandle<()>
    where
        F: Fn(Vec<InferenceRequest>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Vec<InferenceResponse>, InferenceError>> + Send,
    {
        let pending = Arc::clone(&self.pending);
        let config = self.config.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            let mut timer = tokio::time::interval(config.max_wait_time);
            timer.tick().await; // Skip first immediate tick

            loop {
                tokio::select! {
                    biased;
                    
                    _ = shutdown_rx.recv() => {
                        break;
                    }
                    
                    _ = timer.tick() => {
                        // Check if we have enough requests to form a batch
                        let batch = {
                            let mut pending_guard = pending.lock().await;
                            
                            if pending_guard.is_empty() {
                                continue;
                            }
                            
                            let now = Instant::now();
                            let mut batch: Vec<PendingRequest> = Vec::new();
                            
                            // Collect requests up to max_batch_size
                            while batch.len() < config.max_batch_size {
                                if let Some(pending_req) = pending_guard.pop_front() {
                                    // Always include if we're at max wait time
                                    let elapsed = now.duration_since(pending_req.arrived_at);
                                    
                                    if elapsed >= config.max_wait_time || 
                                       batch.len() >= config.min_batch_size ||
                                       pending_guard.is_empty() {
                                        batch.push(pending_req);
                                    } else {
                                        // Put it back and wait for more
                                        pending_guard.push_front(pending_req);
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            }
                            
                            batch
                        };
                        
                        if !batch.is_empty() {
                            let requests: Vec<InferenceRequest> = 
                                batch.iter().map(|p| p.request.clone()).collect();
                            
                            let response_txs: Vec<_> = 
                                batch.into_iter().map(|p| p.response_tx).collect();
                            
                            // Run inference
                            match infer_fn(requests).await {
                                Ok(responses) => {
                                    // Send responses back
                                    for (response, tx) in responses.into_iter().zip(response_txs) {
                                        let _ = tx.send(Ok(response)).await;
                                    }
                                }
                                Err(e) => {
                                    // Send error to all waiting requests
                                    for tx in response_txs {
                                        let _ = tx.send(Err(e.clone())).await;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        })
    }

    /// Submit a request for batching
    pub async fn submit(
        &self,
        request: InferenceRequest,
    ) -> Result<mpsc::Receiver<Result<InferenceResponse, InferenceError>>, InferenceError> {
        let (tx, rx) = mpsc::channel(1);
        
        let pending_req = PendingRequest {
            request,
            arrived_at: Instant::now(),
            response_tx: tx,
        };
        
        let mut pending_guard = self.pending.lock().await;
        
        if pending_guard.len() >= self.config.max_batch_size * 2 {
            // Queue is too full, reject
            return Err(InferenceError::BatchTooLarge {
                requested: pending_guard.len(),
                max: self.config.max_batch_size,
            });
        }
        
        pending_guard.push_back(pending_req);
        
        Ok(rx)
    }

    /// Get current queue depth
    pub async fn queue_depth(&self) -> usize {
        self.pending.lock().await.len()
    }
}

/// Batch statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    pub total_batches: u64,
    pub total_requests: u64,
    pub avg_batch_size: f64,
    pub avg_latency_ms: f64,
    pub max_latency_ms: f64,
}

impl BatchStats {
    /// Update statistics with a new batch
    pub fn update(&mut self, batch_size: usize, latency_ms: f64) {
        self.total_batches += 1;
        self.total_requests += batch_size as u64;
        
        let n = self.total_batches as f64;
        self.avg_batch_size = ((n - 1.0) * self.avg_batch_size + batch_size as f64) / n;
        self.avg_latency_ms = ((n - 1.0) * self.avg_latency_ms + latency_ms) / n;
        self.max_latency_ms = self.max_latency_ms.max(latency_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_continuous_batching() {
        let config = BatchingConfig {
            max_batch_size: 4,
            max_wait_time: Duration::from_millis(10),
            min_batch_size: 1,
        };
        
        let queue = ContinuousBatchingQueue::new(config);
        
        // Start batch processor with mock inference
        let _handle = queue.start(|requests| async move {
            Ok(requests
                .iter()
                .map(|req| InferenceResponse {
                    id: req.id,
                    generated_text: b"response".to_vec(),
                    tokens_generated: 5,
                    latency_ms: 1.0,
                    success: true,
                })
                .collect())
        });
        
        // Submit requests
        let mut receivers = Vec::new();
        for i in 0..3 {
            let req = InferenceRequest {
                id: i,
                prompt: b"test".to_vec(),
                max_tokens: 10,
                temperature: 0.7,
                top_p: 0.9,
            };
            let rx = queue.submit(req).await.unwrap();
            receivers.push(rx);
        }
        
        // Wait for responses
        for mut rx in receivers {
            let response = rx.recv().await;
            assert!(response.is_some());
            assert!(response.unwrap().is_ok());
        }
    }
}
