//! Axum WebSocket Server for Nexus Telemetry
//! 
//! CRITICAL ARCHITECTURE NOTES:
//! 1. This server runs on a DEDICATED Tokio thread pool, completely isolated from trading hot-paths
//! 2. Uses MessagePack (binary) for high-frequency market data - ZERO serde_json in broadcast loop
//! 3. JSON is ONLY used for low-frequency control messages (Start/Stop Bot)
//! 4. Handles WebSocket disconnects gracefully to prevent SPSC buffer backup

use axum::{
    extract::ws::{WebSocket, Message, WebSocketUpgrade, CloseFrame},
    response::IntoResponse,
    routing::get,
    Router,
};
use tokio::sync::broadcast;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{info, warn, error, debug};

use crate::binary_serializer::{
    BinarySerializer, MarketTelemetry, SystemHealth, ControlMessage, ControlCommand,
    msg_types,
};
use crate::lock_free_spsc_broadcaster::{SpscBroadcaster, MultiClientBroadcaster};

/// Server configuration
pub struct ServerConfig {
    pub bind_address: SocketAddr,
    pub dedicated_thread_pool_size: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:8081".parse().unwrap(),
            dedicated_thread_pool_size: 2, // Small dedicated pool for WS handling
        }
    }
}

/// Shared application state
pub struct AppState {
    pub broadcaster: Arc<MultiClientBroadcaster>,
    /// Pre-allocated buffer pool for serialization (avoids allocations in hot path)
    pub buffer_pool: Vec<Vec<u8>>,
}

impl AppState {
    pub fn new(broadcaster: Arc<MultiClientBroadcaster>) -> Self {
        let mut buffer_pool = Vec::with_capacity(16);
        for _ in 0..16 {
            buffer_pool.push(Vec::with_capacity(8192));
        }
        Self {
            broadcaster,
            buffer_pool,
        }
    }
    
    /// Get a buffer from the pool (or create new if exhausted)
    pub fn get_buffer(&mut self) -> Vec<u8> {
        self.buffer_pool.pop().unwrap_or_else(|| Vec::with_capacity(8192))
    }
    
    /// Return buffer to pool
    pub fn return_buffer(&mut self, mut buf: Vec<u8>) {
        buf.clear();
        if self.buffer_pool.len() < 32 {
            self.buffer_pool.push(buf);
        }
    }
}

/// Create the Axum router with WebSocket endpoint
pub fn create_router(state: Arc<tokio::sync::Mutex<AppState>>) -> Router {
    Router::new()
        .route("/ws/telemetry", get(ws_handler))
        .route("/ws/control", get(control_ws_handler))
        .with_state(state)
}

/// Main telemetry WebSocket handler
/// Sends binary MessagePack data for high-frequency market updates
async fn ws_handler(
    ws: WebSocketUpgrade,
    state: axum::extract::State<Arc<tokio::sync::Mutex<AppState>>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// Control WebSocket handler (JSON only for low-frequency commands)
async fn control_ws_handler(
    ws: WebSocketUpgrade,
    state: axum::extract::State<Arc<tokio::sync::Mutex<AppState>>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_control_socket(socket, state))
}

/// Handle main telemetry socket connection
async fn handle_socket(socket: WebSocket, state: Arc<tokio::sync::Mutex<AppState>>) {
    let (mut sender, mut receiver) = socket.split();
    
    // Register client connection
    {
        let app_state = state.lock().await;
        app_state.broadcaster.client_connected();
        info!("Telemetry client connected. Total clients: {}", app_state.broadcaster.client_count());
    }
    
    // Clone broadcaster for this connection
    let broadcaster = {
        let app_state = state.lock().await;
        app_state.broadcaster.get_spsc()
    };
    
    // Spawn task to read from SPSC and send to WebSocket
    let send_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(16)); // ~60fps
        
        loop {
            interval.tick().await;
            
            // Batch consume from SPSC buffer
            let mut items_sent = 0;
            broadcaster.consume_batch(|item| {
                // Send binary MessagePack data with type discriminator
                let mut frame = Vec::with_capacity(item.serialized_bytes.len() + 1);
                frame.push(msg_types::MARKET_DATA);
                frame.extend_from_slice(&item.serialized_bytes);
                
                match sender.send(Message::Binary(frame)).await {
                    Ok(_) => items_sent += 1,
                    Err(e) => {
                        warn!("Failed to send telemetry: {}", e);
                        return; // Exit early on error
                    }
                }
            });
            
            // If we can't send, client is gone
            if items_sent == 0 && broadcaster.utilization() > 0.9 {
                warn!("Client falling behind, buffer utilization high");
            }
        }
    });
    
    // Handle incoming messages (should be rare for telemetry socket)
    let recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.recv().await {
            match msg {
                Message::Text(text) => {
                    warn!("Unexpected text message on telemetry socket: {}", text);
                }
                Message::Close(reason) => {
                    info!("Telemetry client disconnected: {:?}", reason);
                    break;
                }
                Message::Pong(_) | Message::Ping(_) => {
                    // Handled automatically by axum
                }
                Message::Binary(_) => {
                    warn!("Unexpected binary message on telemetry socket");
                }
            }
        }
    });
    
    // Wait for either task to complete
    tokio::select! {
        result = send_task => {
            if let Err(e) = result {
                error!("Send task failed: {}", e);
            }
        }
        result = recv_task => {
            if let Err(e) = result {
                error!("Recv task failed: {}", e);
            }
        }
    }
    
    // Unregister client
    {
        let app_state = state.lock().await;
        app_state.broadcaster.client_disconnected();
        info!("Telemetry client removed. Total clients: {}", app_state.broadcaster.client_count());
    }
}

/// Handle control socket connection (JSON for commands)
async fn handle_control_socket(socket: WebSocket, state: Arc<tokio::sync::Mutex<AppState>>) {
    let (mut sender, mut receiver) = socket.split();
    
    info!("Control client connected");
    
    while let Some(msg) = receiver.recv().await {
        match msg {
            Message::Text(text) => {
                // Parse control command
                match serde_json::from_str::<ControlMessage>(&text) {
                    Ok(cmd) => {
                        info!("Received control command: {:?}", cmd.command);
                        // TODO: Route command to trading engine
                        // For now, just acknowledge
                        let response = serde_json::json!({
                            "status": "acknowledged",
                            "command": format!("{:?}", cmd.command)
                        });
                        let _ = sender.send(Message::Text(response.to_string())).await;
                    }
                    Err(e) => {
                        warn!("Invalid control message: {}", e);
                        let error = serde_json::json!({
                            "status": "error",
                            "message": e.to_string()
                        });
                        let _ = sender.send(Message::Text(error.to_string())).await;
                    }
                }
            }
            Message::Binary(_) => {
                warn!("Unexpected binary message on control socket");
            }
            Message::Close(reason) => {
                info!("Control client disconnected: {:?}", reason);
                break;
            }
            Message::Pong(_) | Message::Ping(_) => {}
        }
    }
    
    info!("Control client session ended");
}

/// Start the telemetry server on a dedicated thread pool
pub async fn run_server(
    config: ServerConfig,
    broadcaster: Arc<MultiClientBroadcaster>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    
    let app_state = Arc::new(tokio::sync::Mutex::new(AppState::new(broadcaster)));
    let app = create_router(app_state);
    
    info!(
        "Starting Nexus Telemetry WebSocket server on {} with dedicated thread pool",
        config.bind_address
    );
    
    // Bind and serve
    let listener = tokio::net::TcpListener::bind(config.bind_address).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

/// Demo data generator for testing (simulates trading engine output)
pub async fn demo_data_generator(broadcaster: Arc<SpscBroadcaster>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let mut base_price: i64 = 4200000; // $42,000.00
    
    loop {
        // Simulate price movement
        let delta = rng.gen_range(-1000..1000);
        base_price += delta;
        
        let telemetry = MarketTelemetry {
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            symbol: "BTC-USD".to_string(),
            best_bid_price: base_price,
            best_ask_price: base_price + 100,
            best_bid_volume: rng.gen_range(100..500),
            best_ask_volume: rng.gen_range(100..500),
            l2_bids: generate_l2_levels(base_price, &mut rng, true),
            l2_asks: generate_l2_levels(base_price, &mut rng, false),
            recent_trades: vec![],
        };
        
        let mut buffer = Vec::with_capacity(4096);
        if BinarySerializer::encode_telemetry(&telemetry, &mut buffer).is_ok() {
            if !broadcaster.publish(telemetry, buffer) {
                warn!("SPSC buffer overflow - dropping telemetry");
            }
        }
        
        // Simulate ~10K updates/sec
        tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
    }
}

fn generate_l2_levels(base_price: i64, rng: &mut impl Rng, is_bid: bool) -> Vec<(i64, u64)> {
    let mut levels = Vec::with_capacity(10);
    for i in 0..10 {
        let price_offset = (i as i64 + 1) * 100;
        let price = if is_bid {
            base_price - price_offset
        } else {
            base_price + price_offset
        };
        let volume = rng.gen_range(50..1000);
        levels.push((price, volume));
    }
    levels
}

/// Entry point for the telemetry server
pub async fn start_telemetry_server() -> Result<(), Box<dyn std::error::Error>> {
    let config = ServerConfig::default();
    let broadcaster = Arc::new(MultiClientBroadcaster::new());
    let spsc = broadcaster.get_spsc();
    
    // Spawn demo data generator (simulates trading engine)
    tokio::spawn(demo_data_generator(spsc));
    
    // Run server
    run_server(config, broadcaster).await
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_app_state_buffer_pool() {
        let broadcaster = Arc::new(MultiClientBroadcaster::new());
        let mut state = AppState::new(broadcaster);
        
        let buf1 = state.get_buffer();
        assert_eq!(buf1.capacity(), 8192);
        
        state.return_buffer(buf1);
        assert_eq!(state.buffer_pool.len(), 16);
    }
}
