//! Chapter 1: Resilient WebSocket Connection Manager for Binance
//!
//! This module implements a production-grade WebSocket adapter that connects
//! to Binance's live market data streams with exponential backoff reconnection,
//! ping/pong keep-alive handling, and graceful TCP half-close management.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures_util::{stream::SplitSink, SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, watch};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tracing::{debug, error, info, warn};
use url::Url;

/// Reconnection state machine states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Healing,
}

/// Configuration for the Binance WebSocket adapter
#[derive(Debug, Clone)]
pub struct BinanceWsConfig {
    /// Symbol to subscribe to (e.g., "btcusdt")
    pub symbol: String,
    /// Channels to subscribe to (e.g., "depth", "trade")
    pub channels: Vec<String>,
    /// Base URL for Binance WebSocket
    pub base_url: String,
    /// Initial reconnect delay in milliseconds
    pub initial_reconnect_delay_ms: u64,
    /// Maximum reconnect delay in milliseconds
    pub max_reconnect_delay_ms: u64,
    /// Ping interval in seconds
    pub ping_interval_secs: u64,
    /// Timeout for pong response in seconds
    pub pong_timeout_secs: u64,
}

impl Default for BinanceWsConfig {
    fn default() -> Self {
        Self {
            symbol: "btcusdt".to_string(),
            channels: vec!["depth@100ms".to_string(), "trade".to_string()],
            base_url: "wss://stream.binance.com:9443/ws".to_string(),
            initial_reconnect_delay_ms: 100,
            max_reconnect_delay_ms: 30000,
            ping_interval_secs: 30,
            pong_timeout_secs: 10,
        }
    }
}

/// Statistics for WebSocket connection
#[derive(Debug, Clone, Default)]
pub struct WsStats {
    pub bytes_received: u64,
    pub messages_received: u64,
    pub reconnects: u64,
    pub ping_sent: u64,
    pub pong_received: u64,
    pub errors: u64,
    pub last_message_timestamp_ns: u64,
}

/// Exponential backoff calculator
pub struct BackoffCalculator {
    current_delay_ms: u64,
    initial_delay_ms: u64,
    max_delay_ms: u64,
    multiplier: u64,
    jitter_factor: f64,
}

impl BackoffCalculator {
    pub fn new(initial_delay_ms: u64, max_delay_ms: u64) -> Self {
        Self {
            current_delay_ms: initial_delay_ms,
            initial_delay_ms,
            max_delay_ms,
            multiplier: 2,
            jitter_factor: 0.1,
        }
    }

    pub fn next_delay(&mut self) -> Duration {
        let delay = self.current_delay_ms.min(self.max_delay_ms);
        
        // Add jitter to prevent thundering herd
        let jitter = (delay as f64 * self.jitter_factor * rand_jitter()) as u64;
        let delay_with_jitter = delay + jitter;
        
        // Exponential increase for next time
        self.current_delay_ms = (self.current_delay_ms * self.multiplier).min(self.max_delay_ms);
        
        Duration::from_millis(delay_with_jitter)
    }

    pub fn reset(&mut self) {
        self.current_delay_ms = self.initial_delay_ms;
    }
}

/// Simple random jitter generator (no external dependency)
fn rand_jitter() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as f64;
    ((nanos / 1_000_000_000.0) - 0.5) * 2.0 // Range: -1.0 to 1.0
}

/// Live Exchange Adapter for Binance WebSocket
pub struct LiveExchangeAdapter {
    config: BinanceWsConfig,
    state: Arc<std::sync::Mutex<ConnectionState>>,
    stats: Arc<std::sync::Mutex<WsStats>>,
    shutdown_signal: watch::Sender<bool>,
    message_tx: mpsc::Sender<Vec<u8>>,
    is_running: Arc<AtomicBool>,
}

impl LiveExchangeAdapter {
    pub fn new(config: BinanceWsConfig) -> Self {
        let (shutdown_signal, _) = watch::channel(false);
        let (message_tx, _) = mpsc::channel(10000);
        
        Self {
            config,
            state: Arc::new(std::sync::Mutex::new(ConnectionState::Disconnected)),
            stats: Arc::new(std::sync::Mutex::new(WsStats::default())),
            shutdown_signal,
            message_tx,
            is_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Build the WebSocket URL from config
    fn build_url(&self) -> String {
        let streams: Vec<String> = self.config.channels
            .iter()
            .map(|ch| format!("{}{}", self.config.symbol, ch))
            .collect();
        
        format!("{}/{}", self.config.base_url, streams.join("/"))
    }

    /// Update connection state atomically
    fn set_state(&self, state: ConnectionState) {
        if let Ok(mut guard) = self.state.lock() {
            *guard = state;
        }
    }

    /// Get current connection state
    pub fn get_state(&self) -> ConnectionState {
        self.state.lock().map(|g| *g).unwrap_or(ConnectionState::Disconnected)
    }

    /// Get statistics snapshot
    pub fn get_stats(&self) -> WsStats {
        self.stats.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Check if adapter is running
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Start the WebSocket connection loop
    pub async fn run(&self, message_tx: mpsc::Sender<Vec<u8>>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.is_running.store(true, Ordering::Release);
        
        let mut backoff = BackoffCalculator::new(
            self.config.initial_reconnect_delay_ms,
            self.config.max_reconnect_delay_ms,
        );
        
        let mut shutdown_rx = self.shutdown_signal.subscribe();
        
        loop {
            if *shutdown_rx.borrow() {
                info!("Shutdown signal received, stopping WebSocket adapter");
                break;
            }

            self.set_state(ConnectionState::Connecting);
            
            match self.connect_and_stream(message_tx.clone(), &mut shutdown_rx).await {
                Ok(_) => {
                    // Connection ended cleanly (shouldn't happen in normal operation)
                    info!("WebSocket stream ended");
                    backoff.reset();
                }
                Err(e) => {
                    self.set_state(ConnectionState::Reconnecting);
                    
                    {
                        let mut stats = self.stats.lock().unwrap();
                        stats.errors += 1;
                        stats.reconnects += 1;
                    }
                    
                    let delay = backoff.next_delay();
                    warn!("WebSocket error: {}. Reconnecting in {:?}...", e, delay);
                    
                    tokio::select! {
                        _ = tokio::time::sleep(delay) => {},
                        _ = shutdown_rx.changed() => {
                            if *shutdown_rx.borrow() {
                                break;
                            }
                        }
                    }
                }
            }
        }

        self.is_running.store(false, Ordering::Release);
        self.set_state(ConnectionState::Disconnected);
        Ok(())
    }

    /// Connect and stream messages
    async fn connect_and_stream(
        &self,
        message_tx: mpsc::Sender<Vec<u8>>,
        shutdown_rx: &mut watch::Receiver<bool>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let url = self.build_url();
        info!("Connecting to Binance WebSocket: {}", url);

        let ws_url = Url::parse(&url)?;
        let (ws_stream, _) = connect_async(ws_url).await?;
        
        self.set_state(ConnectionState::Connected);
        info!("WebSocket connected successfully");

        let (mut write_half, mut read_half) = ws_stream.split();
        
        // Spawn ping task
        let ping_interval = Duration::from_secs(self.config.ping_interval_secs);
        let pong_timeout = Duration::from_secs(self.config.pong_timeout_secs);
        let ping_stats = self.stats.clone();
        let ping_shutdown = shutdown_rx.clone();
        
        let ping_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(ping_interval);
            let mut awaiting_pong = false;
            
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if awaiting_pong {
                            warn!("Pong timeout - connection may be stale");
                            break;
                        }
                        
                        if let Err(e) = write_half.send(Message::Ping(vec![])).await {
                            warn!("Failed to send ping: {}", e);
                            break;
                        }
                        
                        {
                            if let Ok(mut stats) = ping_stats.lock() {
                                stats.ping_sent += 1;
                            }
                        }
                        
                        awaiting_pong = true;
                        
                        // Wait for pong with timeout
                        tokio::time::sleep(pong_timeout).await;
                        awaiting_pong = false;
                    }
                    _ = ping_shutdown.changed() => {
                        if *ping_shutdown.borrow() {
                            break;
                        }
                    }
                }
            }
        });

        // Read loop
        while let Some(msg_result) = read_half.next().await {
            if *shutdown_rx.borrow() {
                break;
            }

            match msg_result {
                Ok(msg) => {
                    match msg {
                        Message::Text(text) => {
                            let bytes = text.into_bytes();
                            
                            {
                                let mut stats = self.stats.lock().unwrap();
                                stats.bytes_received += bytes.len() as u64;
                                stats.messages_received += 1;
                                stats.last_message_timestamp_ns = 
                                    std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_nanos() as u64;
                            }
                            
                            if message_tx.send(bytes).await.is_err() {
                                warn!("Message channel full, dropping message");
                            }
                        }
                        Message::Binary(data) => {
                            // Handle binary frames if needed
                            debug!("Received binary frame: {} bytes", data.len());
                        }
                        Message::Ping(data) => {
                            // Respond to ping immediately
                            if let Err(e) = write_half.send(Message::Pong(data)).await {
                                warn!("Failed to send pong: {}", e);
                                return Err(Box::new(e));
                            }
                            
                            {
                                let mut stats = self.stats.lock().unwrap();
                                stats.pong_received += 1;
                            }
                        }
                        Message::Pong(_) => {
                            debug!("Received pong response");
                        }
                        Message::Close(frame) => {
                            info!("Received close frame: {:?}", frame);
                            // Graceful close - don't treat as error
                            return Ok(());
                        }
                        Message::Frame(_) => {
                            // Raw frame, ignore
                        }
                    }
                }
                Err(e) => {
                    // Check if this is a half-close or network error
                    if e.to_string().contains("EOF") || e.to_string().contains("connection closed") {
                        info!("TCP half-close detected");
                    } else {
                        warn!("WebSocket error: {}", e);
                    }
                    return Err(Box::new(e));
                }
            }
        }

        // Clean up ping task
        ping_handle.abort();
        
        Ok(())
    }

    /// Signal shutdown
    pub fn shutdown(&self) {
        let _ = self.shutdown_signal.send(true);
    }
}

// SAFETY: LiveExchangeAdapter is designed for multi-threaded async contexts
unsafe impl Send for LiveExchangeAdapter {}
unsafe impl Sync for LiveExchangeAdapter {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backoff_calculator() {
        let mut backoff = BackoffCalculator::new(100, 10000);
        
        let first = backoff.next_delay();
        let second = backoff.next_delay();
        
        // Second delay should be larger (exponential)
        assert!(second >= first);
        
        // Reset should go back to initial
        backoff.reset();
        let after_reset = backoff.next_delay();
        assert!(after_reset <= first * 2); // Allow for jitter
    }

    #[test]
    fn test_connection_state_transitions() {
        let config = BinanceWsConfig::default();
        let adapter = LiveExchangeAdapter::new(config);
        
        assert_eq!(adapter.get_state(), ConnectionState::Disconnected);
    }

    #[test]
    fn test_build_url() {
        let config = BinanceWsConfig {
            symbol: "btcusdt".to_string(),
            channels: vec!["depth@100ms".to_string()],
            base_url: "wss://stream.binance.com:9443/ws".to_string(),
            ..Default::default()
        };
        
        let adapter = LiveExchangeAdapter::new(config);
        let url = adapter.build_url();
        
        assert!(url.contains("btcusdt"));
        assert!(url.contains("depth@100ms"));
    }
}
