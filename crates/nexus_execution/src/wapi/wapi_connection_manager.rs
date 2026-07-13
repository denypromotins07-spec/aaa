//! WebSocket API Connection Manager for Exchange Execution
//! 
//! Manages persistent WebSocket connections for order submission and execution reports.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use url::Url;

/// Connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Failed,
}

/// WAPI connection configuration
#[derive(Debug, Clone)]
pub struct WapiConfig {
    pub base_url: String,
    pub api_key: String,
    pub secret_key: String,
    pub recv_window_ms: u64,
    pub max_reconnect_attempts: u32,
    pub reconnect_base_delay_ms: u64,
    pub reconnect_max_delay_ms: u64,
    pub ping_interval_ms: u64,
    pub pong_timeout_ms: u64,
}

impl Default for WapiConfig {
    fn default() -> Self {
        Self {
            base_url: "wss://stream.binance.com:9443/ws".to_string(),
            api_key: String::new(),
            secret_key: String::new(),
            recv_window_ms: 5000,
            max_reconnect_attempts: 10,
            reconnect_base_delay_ms: 100,
            reconnect_max_delay_ms: 10000,
            ping_interval_ms: 30000, // 30 seconds
            pong_timeout_ms: 10000,  // 10 seconds
        }
    }
}

/// Connection event for external notification
#[derive(Debug, Clone)]
pub enum ConnectionEvent {
    Connected,
    Disconnected,
    Reconnecting { attempt: u32, delay_ms: u64 },
    Error(String),
    Message(Message),
}

/// WAPI Connection Manager with exponential backoff reconnection
pub struct WapiConnectionManager {
    config: WapiConfig,
    state: ConnectionState,
    reconnect_attempt: u32,
    last_pong_received: Instant,
    is_connected: Arc<AtomicBool>,
    message_tx: mpsc::UnboundedSender<Message>,
    event_tx: mpsc::UnboundedSender<ConnectionEvent>,
    sequence_num: AtomicU64,
}

impl WapiConnectionManager {
    pub fn new(
        config: WapiConfig,
        message_tx: mpsc::UnboundedSender<Message>,
        event_tx: mpsc::UnboundedSender<ConnectionEvent>,
    ) -> Self {
        Self {
            config,
            state: ConnectionState::Disconnected,
            reconnect_attempt: 0,
            last_pong_received: Instant::now(),
            is_connected: Arc::new(AtomicBool::new(false)),
            message_tx,
            event_tx,
            sequence_num: AtomicU64::new(0),
        }
    }

    /// Get next sequence number (thread-safe, monotonic)
    #[inline]
    pub fn next_sequence(&self) -> u64 {
        self.sequence_num.fetch_add(1, Ordering::Relaxed)
    }

    /// Check if currently connected
    #[inline]
    pub fn is_connected(&self) -> bool {
        self.is_connected.load(Ordering::Relaxed)
    }

    /// Get current connection state
    #[inline]
    pub fn state(&self) -> ConnectionState {
        self.state
    }

    /// Calculate reconnect delay with exponential backoff
    fn calculate_reconnect_delay(&self) -> u64 {
        let base = self.config.reconnect_base_delay_ms;
        let max = self.config.reconnect_max_delay_ms;
        
        // Exponential backoff: base * 2^attempt
        let delay = base.saturating_mul(2u64.saturating_pow(self.reconnect_attempt));
        
        // Add jitter (±10%)
        let jitter = delay / 10;
        let jittered = if rand::random::<bool>() {
            delay.saturating_add(jitter)
        } else {
            delay.saturating_sub(jitter)
        };
        
        jittered.min(max)
    }

    /// Handle connection failure and schedule reconnection
    pub async fn handle_disconnect(&mut self) -> Result<(), &'static str> {
        self.is_connected.store(false, Ordering::Relaxed);
        self.state = ConnectionState::Disconnected;

        if self.reconnect_attempt >= self.config.max_reconnect_attempts {
            self.state = ConnectionState::Failed;
            let _ = self.event_tx.send(ConnectionEvent::Error(
                format!("Max reconnect attempts ({}) exceeded", self.config.max_reconnect_attempts)
            ));
            return Err("Max reconnect attempts exceeded");
        }

        self.reconnect_attempt += 1;
        let delay_ms = self.calculate_reconnect_delay();
        self.state = ConnectionState::Reconnecting;

        let _ = self.event_tx.send(ConnectionEvent::Reconnecting {
            attempt: self.reconnect_attempt,
            delay_ms,
        });

        // Wait before reconnecting
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;

        Ok(())
    }

    /// Establish WebSocket connection
    pub async fn connect(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.state = ConnectionState::Connecting;

        let url = Url::parse(&self.config.base_url)?;
        
        match connect_async(url).await {
            Ok((ws_stream, _)) => {
                self.state = ConnectionState::Connected;
                self.is_connected.store(true, Ordering::Relaxed);
                self.reconnect_attempt = 0;
                self.last_pong_received = Instant::now();

                let _ = self.event_tx.send(ConnectionEvent::Connected);
                
                // Return the stream for the caller to handle
                // In production, this would spawn a task to handle the stream
                drop(ws_stream);
                
                Ok(())
            }
            Err(e) => {
                self.state = ConnectionState::Failed;
                let _ = self.event_tx.send(ConnectionEvent::Error(format!("Connection failed: {}", e)));
                Err(Box::new(e))
            }
        }
    }

    /// Send ping to keep connection alive
    pub async fn send_ping(&self, ws: &tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>) -> Result<(), &'static str> {
        use tokio_tungstenite::tungstenite::Message;
        
        ws.send(Message::Ping(vec![]))
            .map_err(|_| "Failed to send ping")?;
        
        Ok(())
    }

    /// Check if pong timeout has occurred
    pub fn check_pong_timeout(&self) -> bool {
        self.last_pong_received.elapsed() > Duration::from_millis(self.config.pong_timeout_ms)
    }

    /// Reset pong timer on pong received
    pub fn on_pong_received(&mut self) {
        self.last_pong_received = Instant::now();
    }

    /// Send text message through the connection
    pub fn send_message(&self, message: String) -> Result<(), &'static str> {
        if !self.is_connected.load(Ordering::Relaxed) {
            return Err("Not connected");
        }

        let msg = Message::Text(message);
        self.message_tx.send(msg).map_err(|_| "Channel closed")
    }
}

/// Background task for monitoring connection health
pub async fn connection_health_monitor(
    is_connected: Arc<AtomicBool>,
    ping_interval_ms: u64,
    mut event_tx: mpsc::UnboundedSender<ConnectionEvent>,
) {
    let mut interval = tokio::time::interval(Duration::from_millis(ping_interval_ms));
    
    loop {
        interval.tick().await;
        
        if !is_connected.load(Ordering::Relaxed) {
            continue;
        }

        // In production, this would send actual pings through the WebSocket
        // For now, just log that we're checking
        tracing::debug!("Connection health check - connected: {}", is_connected.load(Ordering::Relaxed));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_backoff() {
        let config = WapiConfig {
            reconnect_base_delay_ms: 100,
            reconnect_max_delay_ms: 5000,
            ..Default::default()
        };

        let (msg_tx, _) = mpsc::unbounded_channel();
        let (event_tx, _) = mpsc::unbounded_channel();
        
        let mut manager = WapiConnectionManager::new(config, msg_tx, event_tx);

        // First attempt: ~100ms
        let delay1 = manager.calculate_reconnect_delay();
        assert!(delay1 >= 90 && delay1 <= 110); // Allow jitter

        manager.reconnect_attempt = 1;
        // Second attempt: ~200ms
        let delay2 = manager.calculate_reconnect_delay();
        assert!(delay2 >= 180 && delay2 <= 220);

        manager.reconnect_attempt = 5;
        // Sixth attempt: ~3200ms (100 * 2^5)
        let delay3 = manager.calculate_reconnect_delay();
        assert!(delay3 >= 2800 && delay3 <= 3600);

        manager.reconnect_attempt = 20;
        // Should be capped at max
        let delay4 = manager.calculate_reconnect_delay();
        assert_eq!(delay4, 5000);
    }

    #[test]
    fn test_sequence_numbers() {
        let config = WapiConfig::default();
        let (msg_tx, _) = mpsc::unbounded_channel();
        let (event_tx, _) = mpsc::unbounded_channel();
        
        let manager = WapiConnectionManager::new(config, msg_tx, event_tx);

        assert_eq!(manager.next_sequence(), 0);
        assert_eq!(manager.next_sequence(), 1);
        assert_eq!(manager.next_sequence(), 2);
    }
}
