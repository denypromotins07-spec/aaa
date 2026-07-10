//! High-Throughput Async Stream Consumer
//! 
//! Consumes thousands of news articles, tweets, and press releases per second
//! from WebSockets and Kafka streams using Tokio async runtime.
//! 
//! CRITICAL: CPU-heavy tasks are offloaded to spawn_blocking to prevent
//! blocking the async runtime.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, broadcast};
use tokio::task::JoinHandle;
use crossbeam_channel::{bounded, Sender as CrossbeamSender, Receiver as CrossbeamReceiver};
use rayon::prelude::*;

/// Message envelope for ingested text data
#[derive(Debug, Clone)]
pub struct TextMessage {
    pub id: u64,
    pub source: SourceType,
    pub payload: Arc<[u8]>,
    pub timestamp: Instant,
    pub metadata: Option<Arc<[u8]>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceType {
    WebSocket,
    Kafka,
    RestApi,
    File,
}

/// Configuration for the stream consumer
#[derive(Debug, Clone)]
pub struct ConsumerConfig {
    pub max_batch_size: usize,
    pub batch_timeout: Duration,
    pub worker_threads: usize,
    pub channel_capacity: usize,
}

impl Default for ConsumerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 256,
            batch_timeout: Duration::from_micros(100),
            worker_threads: num_cpus::get().max(4),
            channel_capacity: 65536,
        }
    }
}

/// Async Stream Consumer that multiplexes multiple input streams
pub struct AsyncStreamConsumer {
    config: ConsumerConfig,
    tx: mpsc::Sender<TextMessage>,
    rx: Option<mpsc::Receiver<TextMessage>>,
    broadcast_tx: broadcast::Sender<Arc<TextMessage>>,
    shutdown_tx: broadcast::Sender<()>,
    handles: Vec<JoinHandle<()>>,
}

impl AsyncStreamConsumer {
    /// Create a new async stream consumer
    pub fn new(config: ConsumerConfig) -> Self {
        let (tx, rx) = mpsc::channel(config.channel_capacity);
        let (broadcast_tx, _) = broadcast::channel(4096);
        let (shutdown_tx, _) = broadcast::channel(16);
        
        Self {
            config,
            tx,
            rx: Some(rx),
            broadcast_tx,
            shutdown_tx,
            handles: Vec::new(),
        }
    }

    /// Start the consumer with worker threads
    pub fn start(&mut self) -> Result<(), ConsumerError> {
        let rx = self.rx.take().ok_or(ConsumerError::AlreadyStarted)?;
        
        // Spawn batch processor
        let batch_handle = self.spawn_batch_processor(rx);
        self.handles.push(batch_handle);
        
        // Spawn parallel workers for CPU-heavy parsing
        for _ in 0..self.config.worker_threads {
            let worker_handle = self.spawn_parallel_worker();
            self.handles.push(worker_handle);
        }
        
        Ok(())
    }

    /// Spawn batch processor that groups messages
    fn spawn_batch_processor(
        &self,
        mut rx: mpsc::Receiver<TextMessage>,
    ) -> JoinHandle<()> {
        let config = self.config.clone();
        let broadcast_tx = self.broadcast_tx.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        
        tokio::spawn(async move {
            let mut batch: Vec<TextMessage> = Vec::with_capacity(config.max_batch_size);
            let mut timeout = tokio::time::interval(config.batch_timeout);
            
            loop {
                tokio::select! {
                    biased;
                    
                    _ = shutdown_rx.recv() => {
                        break;
                    }
                    
                    _ = timeout.tick() => {
                        if !batch.is_empty() {
                            // Process batch in parallel using Rayon
                            let processed: Vec<_> = batch.par_iter()
                                .filter_map(|msg| {
                                    // Validate message before broadcasting
                                    if !msg.payload.is_empty() {
                                        Some(Arc::new(msg.clone()))
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            
                            for msg in processed {
                                let _ = broadcast_tx.send(msg);
                            }
                            batch.clear();
                        }
                    }
                    
                    msg = rx.recv() => {
                        match msg {
                            Some(m) => {
                                batch.push(m);
                                if batch.len() >= config.max_batch_size {
                                    // Immediate flush on batch full
                                    let processed: Vec<_> = batch.par_iter()
                                        .filter_map(|msg| {
                                            if !msg.payload.is_empty() {
                                                Some(Arc::new(msg.clone()))
                                            } else {
                                                None
                                            }
                                        })
                                        .collect();
                                    
                                    for msg in processed {
                                        let _ = broadcast_tx.send(msg);
                                    }
                                    batch.clear();
                                }
                            }
                            None => break,
                        }
                    }
                }
            }
        })
    }

    /// Spawn parallel worker for CPU-heavy JSON parsing
    fn spawn_parallel_worker(&self) -> JoinHandle<()> {
        let mut broadcast_rx = self.broadcast_tx.subscribe();
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    
                    _ = shutdown_rx.recv() => {
                        break;
                    }
                    
                    msg = broadcast_rx.recv() => {
                        match msg {
                            Ok(msg_arc) => {
                                // Offload heavy JSON parsing to blocking thread
                                let _ = tokio::task::spawn_blocking(move || {
                                    // Parse JSON payload without allocation using simd_json
                                    // This is a placeholder - actual implementation uses simd_json
                                    let _parsed = Self::parse_json_payload(&msg_arc.payload);
                                }).await;
                            }
                            Err(broadcast::error::RecvError::Lagged(n)) => {
                                eprintln!("Consumer lagged by {} messages", n);
                            }
                            Err(broadcast::error::RecvError::Closed) => break,
                        }
                    }
                }
            }
        })
    }

    /// Parse JSON payload using zero-copy techniques
    fn parse_json_payload(payload: &[u8]) -> Option<&[u8]> {
        // In production, this would use simd_json or similar
        // For now, just validate it's valid UTF-8
        if std::str::from_utf8(payload).is_ok() {
            Some(payload)
        } else {
            None
        }
    }

    /// Send a message into the consumer pipeline
    pub async fn send(&self, msg: TextMessage) -> Result<(), ConsumerError> {
        self.tx.send(msg).await
            .map_err(|_| ConsumerError::ChannelClosed)
    }

    /// Try send without blocking
    pub fn try_send(&self, msg: TextMessage) -> Result<(), ConsumerError> {
        self.tx.try_send(msg)
            .map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => ConsumerError::ChannelFull,
                mpsc::error::TrySendError::Closed(_) => ConsumerError::ChannelClosed,
            })
    }

    /// Shutdown all workers gracefully
    pub async fn shutdown(self) -> Result<(), ConsumerError> {
        let _ = self.shutdown_tx.send(());
        
        for handle in self.handles {
            let _ = handle.await;
        }
        
        Ok(())
    }
}

/// Consumer errors
#[derive(Debug, thiserror::Error)]
pub enum ConsumerError {
    #[error("Consumer already started")]
    AlreadyStarted,
    #[error("Channel closed")]
    ChannelClosed,
    #[error("Channel full")]
    ChannelFull,
    #[error("Parse error: {0}")]
    ParseError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stream_consumer() {
        let config = ConsumerConfig {
            max_batch_size: 4,
            batch_timeout: Duration::from_millis(10),
            worker_threads: 2,
            channel_capacity: 1024,
        };
        
        let mut consumer = AsyncStreamConsumer::new(config);
        consumer.start().unwrap();
        
        let msg = TextMessage {
            id: 1,
            source: SourceType::WebSocket,
            payload: Arc::from(b"{\"text\": \"test\"}".as_slice()),
            timestamp: Instant::now(),
            metadata: None,
        };
        
        assert!(consumer.send(msg).await.is_ok());
        
        let _ = consumer.shutdown().await;
    }
}
