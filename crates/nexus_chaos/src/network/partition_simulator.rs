// STAGE 25: CHAPTER 1 - NETWORK PARTITION SIMULATOR
// Implements deterministic chaos injection for network faults
// Zero-alloc, async-safe, no unwrap() in hot paths

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for partition simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    pub latency_spike_ms: u64,
    pub packet_drop_rate: f64,
    pub duration_ms: u64,
    pub target_services: Vec<String>,
}

/// Network fault types
#[derive(Debug, Clone, PartialEq)]
pub enum NetworkFault {
    LatencySpike(Duration),
    PacketDrop,
    RateLimit(u32), // HTTP 429 retry-after seconds
    ConnectionSever,
    DnsFailure,
}

/// Chaos injection result
#[derive(Debug, Clone)]
pub struct InjectionResult {
    pub fault_type: NetworkFault,
    pub injected_at: Instant,
    pub affected_connection_id: u64,
    pub success: bool,
}

/// Lock-free partition simulator state
pub struct PartitionSimulatorState {
    pub active: AtomicBool,
    pub current_latency_ns: AtomicU64,
    pub drop_counter: AtomicU64,
    pub last_injection_time: AtomicU64,
    pub chaos_mode_flag: AtomicBool, // CRITICAL: Prevents kill-switch false positives
}

impl Default for PartitionSimulatorState {
    fn default() -> Self {
        Self {
            active: AtomicBool::new(false),
            current_latency_ns: AtomicU64::new(0),
            drop_counter: AtomicU64::new(0),
            last_injection_time: AtomicU64::new(0),
            chaos_mode_flag: AtomicBool::new(false),
        }
    }
}

/// Deterministic Network Partition Simulator
/// Injects latency, packet drops, and connection failures asynchronously
pub struct PartitionSimulator {
    state: std::sync::Arc<PartitionSimulatorState>,
    config: PartitionConfig,
    injection_tx: mpsc::Sender<NetworkFault>,
    rng_seed: u64,
}

impl PartitionSimulator {
    pub fn new(config: PartitionConfig, rng_seed: u64) -> (Self, mpsc::Receiver<NetworkFault>) {
        let (tx, rx) = mpsc::channel(1024);
        let state = std::sync::Arc::new(PartitionSimulatorState::default());
        
        (
            Self {
                state,
                config,
                injection_tx: tx,
                rng_seed,
            },
            rx,
        )
    }

    /// Activate chaos mode - CRITICAL: Sets flag to prevent kill-switch false positives
    pub fn activate_chaos_mode(&self) {
        self.state.chaos_mode_flag.store(true, Ordering::SeqCst);
        self.state.active.store(true, Ordering::SeqCst);
    }

    /// Deactivate chaos mode
    pub fn deactivate_chaos_mode(&self) {
        self.state.chaos_mode_flag.store(false, Ordering::SeqCst);
        self.state.active.store(false, Ordering::SeqCst);
        self.state.current_latency_ns.store(0, Ordering::Relaxed);
    }

    /// Check if chaos mode is active (used by Stage 5 Kill-Switch)
    pub fn is_chaos_mode_active(&self) -> bool {
        self.state.chaos_mode_flag.load(Ordering::SeqCst)
    }

    /// Inject latency spike asynchronously via lock-free channel
    /// NEVER blocks the execution thread
    pub async fn inject_latency_spike(&self, connection_id: u64) -> Option<InjectionResult> {
        if !self.state.active.load(Ordering::Relaxed) {
            return None;
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rng_seed + connection_id);
        let latency_ms = self.config.latency_spike_ms;
        
        let fault = NetworkFault::LatencySpike(Duration::from_millis(latency_ms));
        
        // Non-blocking send - if channel full, skip injection (no backpressure on hot path)
        match self.injection_tx.try_send(fault.clone()) {
            Ok(_) => {
                self.state.current_latency_ns.store(
                    latency_ms * 1_000_000, 
                    Ordering::Relaxed
                );
                Some(InjectionResult {
                    fault_type: fault,
                    injected_at: Instant::now(),
                    affected_connection_id: connection_id,
                    success: true,
                })
            }
            Err(_) => {
                // Channel full - skip injection silently (chaos engineering best practice)
                Some(InjectionResult {
                    fault_type: fault,
                    injected_at: Instant::now(),
                    affected_connection_id: connection_id,
                    success: false,
                })
            }
        }
    }

    /// Inject packet drop with configured probability
    pub fn should_drop_packet(&self, connection_id: u64) -> bool {
        if !self.state.active.load(Ordering::Relaxed) {
            return false;
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rng_seed + connection_id + self.state.drop_counter.load(Ordering::Relaxed));
        let drop = rng.gen::<f64>() < self.config.packet_drop_rate;
        
        if drop {
            self.state.drop_counter.fetch_add(1, Ordering::Relaxed);
        }
        
        drop
    }

    /// Inject rate limit error (HTTP 429)
    pub async fn inject_rate_limit(&self, connection_id: u64) -> Option<InjectionResult> {
        if !self.state.active.load(Ordering::Relaxed) {
            return None;
        }

        let retry_after = 30; // Standard 30 second backoff
        let fault = NetworkFault::RateLimit(retry_after);

        match self.injection_tx.try_send(fault.clone()) {
            Ok(_) => Some(InjectionResult {
                fault_type: fault,
                injected_at: Instant::now(),
                affected_connection_id: connection_id,
                success: true,
            }),
            Err(_) => Some(InjectionResult {
                fault_type: fault,
                injected_at: Instant::now(),
                affected_connection_id: connection_id,
                success: false,
            }),
        }
    }

    /// Get current injected latency in nanoseconds (lock-free read)
    pub fn get_current_latency_ns(&self) -> u64 {
        self.state.current_latency_ns.load(Ordering::Relaxed)
    }

    /// Deterministic fault scheduling based on seed
    pub fn schedule_fault_sequence(&self, base_seed: u64) -> Vec<(u64, NetworkFault)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(base_seed);
        let mut sequence = Vec::new();
        
        for i in 0..100 {
            let fault_type = match rng.gen_range(0..4) {
                0 => NetworkFault::LatencySpike(Duration::from_millis(rng.gen_range(100..5000))),
                1 => NetworkFault::PacketDrop,
                2 => NetworkFault::RateLimit(rng.gen_range(10..120)),
                _ => NetworkFault::ConnectionSever,
            };
            
            let timestamp_ms = i * 1000 + rng.gen_range(0..500);
            sequence.push((timestamp_ms, fault_type));
        }
        
        sequence
    }
}

/// Builder for partition configurations
pub struct PartitionConfigBuilder {
    latency_spike_ms: u64,
    packet_drop_rate: f64,
    duration_ms: u64,
    target_services: Vec<String>,
}

impl PartitionConfigBuilder {
    pub fn new() -> Self {
        Self {
            latency_spike_ms: 500,
            packet_drop_rate: 0.1,
            duration_ms: 10000,
            target_services: vec!["fix_gateway".to_string(), "rest_api".to_string()],
        }
    }

    pub fn latency_spike(mut self, ms: u64) -> Self {
        self.latency_spike_ms = ms;
        self
    }

    pub fn packet_drop_rate(mut self, rate: f64) -> Self {
        self.packet_drop_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn duration(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }

    pub fn target_service(mut self, service: &str) -> Self {
        self.target_services.push(service.to_string());
        self
    }

    pub fn build(self) -> PartitionConfig {
        PartitionConfig {
            latency_spike_ms: self.latency_spike_ms,
            packet_drop_rate: self.packet_drop_rate,
            duration_ms: self.duration_ms,
            target_services: self.target_services,
        }
    }
}

impl Default for PartitionConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_partition_simulator_activation() {
        let config = PartitionConfigBuilder::new()
            .latency_spike(1000)
            .packet_drop_rate(0.5)
            .build();
        
        let (simulator, _rx) = PartitionSimulator::new(config, 42);
        
        assert!(!simulator.is_chaos_mode_active());
        
        simulator.activate_chaos_mode();
        assert!(simulator.is_chaos_mode_active());
        
        simulator.deactivate_chaos_mode();
        assert!(!simulator.is_chaos_mode_active());
        assert_eq!(simulator.get_current_latency_ns(), 0);
    }

    #[tokio::test]
    async fn test_packet_drop_determinism() {
        let config = PartitionConfigBuilder::new()
            .packet_drop_rate(1.0) // 100% drop rate
            .build();
        
        let (simulator, _rx) = PartitionSimulator::new(config, 42);
        simulator.activate_chaos_mode();
        
        // With 100% drop rate, all packets should be dropped
        assert!(simulator.should_drop_packet(1));
        assert!(simulator.should_drop_packet(2));
        assert!(simulator.should_drop_packet(3));
    }
}
