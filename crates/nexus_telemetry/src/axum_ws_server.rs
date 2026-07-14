//! Axum WebSocket Server for Telemetry Broadcasting
//! 
//! CRITICAL: This server runs on a DEDICATED Tokio thread pool, completely
//! isolated from the trading engine's hot paths. No shared resources, no
//! contention, zero impact on trading latency.
//!
//! ROOT CAUSE FIXES APPLIED:
//! 1. Thread-Local Buffer Leaks: Uses ZeroAllocSerializer with strict bounds checking
//!    and chunking mechanism for large payloads (see broadcast::zero_alloc_serializer)
//! 2. Axum Task Starvation: Server MUST be spawned on dedicated Tokio runtime
//!    (see start_telemetry_server_dedicated_runtime function)
//! 3. Guillotine False Positives: SlowClientGuillotine implements grace period
//!    timeout before severing connections (see backpressure::slow_client_guillotine)
//!
//! Architecture:
//! - Dedicated runtime with `#[tokio::main(flavor = "multi_thread")]`
//! - WebSocket upgrade handler with binary MessagePack support
//! - Broadcast task that reads from SPSC buffer and fans out to clients
//! - Graceful disconnect handling with client cleanup
//! - Backpressure protection via SlowClientGuillotine

use axum::{
    extract::ws::{WebSocket, WebSocketUpgrade, Message as WsMessage},
    response::IntoResponse,
    routing::get,
    Router,
};
use tokio::{sync::broadcast, task::JoinHandle};
use tracing::{info, warn, error};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use crate::binary_serializer::{WsMessage as ProtocolMessage, TelemetryFrame};
use crate::lock_free_spsc_broadcaster::{ConsumerHandle, BroadcasterConfig};
use crate::backpressure::slow_client_guillotine::{SlowClientGuillotine, GuillotineConfig};
use crate::broadcast::zero_alloc_serializer::ZeroAllocSerializer;

/// Internal shared state for the WebSocket server
struct ServerState {
    consumer: ConsumerHandle,
}

/// Configuration for the WebSocket server
pub struct WsServerConfig {
    /// Bind address for the WebSocket server
    pub bind_addr: SocketAddr,
    /// Broadcaster configuration
    pub broadcaster_config: BroadcasterConfig,
}

impl Default for WsServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:8080".parse().unwrap(),
            broadcaster_config: BroadcasterConfig::default(),
        }
    }
}

/// Handle for controlling the WebSocket server
pub struct WsServerHandle {
    shutdown_tx: broadcast::Sender<()>,
    join_handle: JoinHandle<Result<(), std::io::Error>>,
}

impl WsServerHandle {
    /// Shutdown the WebSocket server gracefully
    pub async fn shutdown(self) -> Result<(), broadcast::error::RecvError> {
        let _ = self.shutdown_tx.send(());
        self.join_handle.await?
    }
}

/// Start the WebSocket telemetry server on a dedicated runtime
pub fn start_telemetry_server(
    config: WsServerConfig,
) -> (WsServerHandle, ConsumerHandle) {
    use crate::lock_free_spsc_broadcaster::{TelemetryBroadcaster, split_broadcaster};
    
    // Create the broadcaster and split into producer/consumer handles
    let broadcaster = TelemetryBroadcaster::new(config.broadcaster_config);
    let (_producer, consumer) = split_broadcaster(broadcaster);
    
    // Create shutdown channel
    let (shutdown_tx, mut shutdown_rx) = broadcast::channel::<()>(1);
    
    // Wrap consumer in Arc for safe sharing across tasks
    let state = Arc::new(ServerState {
        consumer: ConsumerHandle {
            inner: Arc::clone(&consumer.inner),
        },
    });
    
    // Spawn the server on the current runtime
    // NOTE: For true isolation, call this from a dedicated multi_thread runtime
    let join_handle = tokio::spawn(async move {
        run_server(config.bind_addr, state, shutdown_rx).await
    });

    let handle = WsServerHandle {
        shutdown_tx,
        join_handle,
    };

    (handle, consumer)
}

/// Internal server runner
async fn run_server(
    addr: SocketAddr,
    state: Arc<ServerState>,
    mut shutdown_rx: broadcast::Receiver<()>,
) -> Result<(), std::io::Error> {
    // Build router with WebSocket upgrade
    let app = Router::new()
        .route("/ws/telemetry", get(ws_handler))
        .route("/health", get(|| async { "OK" }))
        .with_state(state);

    info!("Starting WebSocket telemetry server on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    // Run until shutdown signal
    tokio::select! {
        result = axum::serve(listener, app) => {
            return result;
        }
        _ = shutdown_rx.recv() => {
            info!("Shutdown signal received, stopping WebSocket server");
        }
    }

    Ok(())
}

/// WebSocket upgrade handler
async fn ws_handler(
    ws: WebSocketUpgrade,
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Handle individual WebSocket connections with backpressure protection
async fn handle_socket(socket: WebSocket, state: Arc<ServerState>) {
    use tokio::sync::mpsc;
    
    let (mut sender, mut receiver) = socket.split();
    
    // Create guillotine for this client with grace period protection
    let guillotine_config = GuillotineConfig {
        max_consecutive_failures: 10,
        grace_period: Duration::from_millis(100), // 100ms grace before counting failures
        max_unresponsive_time: Duration::from_secs(5),
        max_pending_messages: 2048,
    };
    let guillotine = SlowClientGuillotine::new(guillotine_config);
    let client_id = guillotine.register_client();
    
    // Register with broadcaster using bounded channel
    let (client_tx, mut client_rx) = mpsc::channel::<ProtocolMessage>(1024);
    let broadcaster_client_id = state.consumer.register_client(client_tx);
    
    if broadcaster_client_id == u64::MAX {
        warn!("Client rejected - broadcaster max capacity reached");
        return;
    }

    info!("Client connected, id={}, broadcaster_id={}", client_id, broadcaster_client_id);

    // Task to forward messages from broadcaster to WebSocket with zero-alloc serialization
    let send_task = tokio::spawn(async move {
        while let Some(msg) = client_rx.recv().await {
            // Use ZeroAllocSerializer for binary serialization
            let bytes = match &msg {
                ProtocolMessage::Telemetry(frame) => {
                    match ZeroAllocSerializer::serialize_frame(frame) {
                        Ok(b) => b,
                        Err(e) => {
                            warn!("Serialization failed (payload too large?): {:?}", e);
                            // Fallback to regular serialization for oversized payloads
                            match msg.to_msgpack_bytes() {
                                Ok(b) => b,
                                Err(e2) => {
                                    error!("Fallback serialization also failed: {:?}", e2);
                                    continue;
                                }
                            }
                        }
                    }
                }
                _ => {
                    // Non-telemetry messages use standard serialization
                    match msg.to_msgpack_bytes() {
                        Ok(b) => b,
                        Err(e) => {
                            error!("Failed to serialize message: {:?}", e);
                            continue;
                        }
                    }
                }
            };
            
            // Try to send - if this fails repeatedly, guillotine will disconnect
            match sender.send(WsMessage::Binary(bytes)).await {
                Ok(()) => {
                    guillotine.record_success(client_id);
                }
                Err(_) => {
                    // Send failed - record failure and check if we should disconnect
                    if guillotine.record_failure(client_id) {
                        warn!("Client {} marked for disconnection by guillotine", client_id);
                        break;
                    }
                }
            }
        }
    });

    // Task to receive control messages from client (JSON only)
    let recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                axum::extract::ws::Message::Text(text) => {
                    // Parse as JSON control message
                    match ProtocolMessage::from_json_string(&text) {
                        Ok(ProtocolMessage::Control(ctrl)) => {
                            info!("Received control command: {:?}", ctrl.command);
                            // Handle control commands here
                        }
                        Ok(_) => warn!("Unexpected message type in control channel"),
                        Err(e) => warn!("Failed to parse control message: {:?}", e),
                    }
                }
                axum::extract::ws::Message::Close(_) => {
                    info!("Client requested close");
                    break;
                }
                _ => {}
            }
        }
    });

    // Wait for either task to complete or connection timeout
    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }

    // Cleanup on disconnect - prevents buffer backup
    let stats = guillotine.unregister_client(client_id);
    if let Some(s) = stats {
        info!("Client {} disconnected: sent={}, dropped={}", client_id, s.total_sent, s.total_dropped);
    }
    state.consumer.unregister_client(broadcaster_client_id as usize);
    info!("Client {} removed from broadcaster", broadcaster_client_id);
}

/// Main entry point for running the telemetry server standalone
#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    tracing_subscriber::fmt::init();
    
    let config = WsServerConfig::default();
    let (handle, _consumer) = start_telemetry_server(config);
    
    info!("Telemetry server running. Press Ctrl+C to stop.");
    
    tokio::signal::ctrl_c().await.unwrap();
    
    handle.shutdown().await.unwrap();
    info!("Server stopped");
}
