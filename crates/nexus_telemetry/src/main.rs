//! Main entry point for the Nexus Telemetry Server
//! 
//! Run with: cargo run --bin nexus_telemetry_server

use nexus_telemetry::start_telemetry_server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 NEXUS-OMEGA Telemetry Server Starting...");
    println!("   WebSocket Endpoint: ws://localhost:8081/ws/telemetry");
    println!("   Control Endpoint:   ws://localhost:8081/ws/control");
    println!("   Serialization:      MessagePack (binary, zero-allocation)");
    println!();
    
    start_telemetry_server().await?;
    
    Ok(())
}
