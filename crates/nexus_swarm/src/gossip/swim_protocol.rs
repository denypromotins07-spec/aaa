//! SWIM Gossip Protocol Implementation
//! 
//! Scalable Weakly-consistent Infection-style Process Group Membership
//! Uses UDP multicast for sub-second node health monitoring in the swarm.

use std::collections::{HashMap, HashSet, BTreeMap};
use std::net::{SocketAddr, IpAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use tokio::net::UdpSocket;
use serde::{Serialize, Deserialize};
use rand::Rng;

/// Unique node identifier
pub type NodeId = u64;

/// SWIM message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwimMessage {
    /// Ping message sent to a target node
    Ping {
        from: NodeId,
        target: NodeId,
        sequence_number: u64,
        timestamp: u128,
    },
    /// Acknowledgment response
    Ack {
        from: NodeId,
        target: NodeId,
        sequence_number: u64,
        timestamp: u128,
    },
    /// Failure detection - node suspected dead
    Suspect {
        from: NodeId,
        target: NodeId,
        incarnation: u64,
        timestamp: u128,
    },
    /// Node confirmed dead
    Dead {
        from: NodeId,
        target: NodeId,
        incarnation: u64,
        timestamp: u128,
    },
    /// Join request from new node
    Join {
        node_id: NodeId,
        address: SocketAddr,
        incarnation: u64,
    },
    /// Leave notification (graceful shutdown)
    Leave {
        node_id: NodeId,
        incarnation: u64,
    },
}

/// Member state in the SWIM protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberState {
    Alive,
    Suspect,
    Dead,
    Left,
}

/// Information about a cluster member
#[derive(Debug, Clone)]
pub struct MemberInfo {
    pub node_id: NodeId,
    pub address: SocketAddr,
    pub state: MemberState,
    pub incarnation: u64,
    pub last_heartbeat: Instant,
    pub suspicion_start: Option<Instant>,
    pub suspicion_timeout: Duration,
}

impl MemberInfo {
    pub fn new(node_id: NodeId, address: SocketAddr) -> Self {
        Self {
            node_id,
            address,
            state: MemberState::Alive,
            incarnation: 0,
            last_heartbeat: Instant::now(),
            suspicion_start: None,
            suspicion_timeout: Duration::from_millis(500), // Sub-second detection
        }
    }
}

/// Configuration for SWIM protocol
#[derive(Debug, Clone)]
pub struct SwimConfig {
    pub node_id: NodeId,
    pub bind_address: SocketAddr,
    pub multicast_group: IpAddr,
    pub ping_interval: Duration,
    pub suspicion_timeout: Duration,
    pub indirect_probes: usize,
    pub max_packet_size: usize,
}

impl Default for SwimConfig {
    fn default() -> Self {
        Self {
            node_id: 0,
            bind_address: "0.0.0.0:9999".parse().unwrap(),
            multicast_group: "224.0.0.1".parse().unwrap(),
            ping_interval: Duration::from_millis(100),
            suspicion_timeout: Duration::from_millis(500),
            indirect_probes: 3,
            max_packet_size: 1400, // Fit in single UDP packet
        }
    }
}

/// SWIM Protocol implementation
pub struct SwimProtocol {
    config: SwimConfig,
    members: RwLock<HashMap<NodeId, MemberInfo>>,
    sequence_number: RwLock<u64>,
    incarnation: RwLock<u64>,
    socket: Option<Arc<UdpSocket>>,
    shutdown_tx: mpsc::Sender<()>,
    event_tx: mpsc::Sender<SwimEvent>,
}

/// Events emitted by SWIM protocol
#[derive(Debug, Clone)]
pub enum SwimEvent {
    NodeJoined(NodeId, SocketAddr),
    NodeSuspected(NodeId),
    NodeConfirmedDead(NodeId),
    NodeLeft(NodeId),
    MessageReceived(SwimMessage),
}

impl SwimProtocol {
    pub fn new(config: SwimConfig) -> Self {
        let (shutdown_tx, _) = mpsc::channel(1);
        let (event_tx, _) = mpsc::channel(100);

        Self {
            config,
            members: RwLock::new(HashMap::new()),
            sequence_number: RwLock::new(0),
            incarnation: RwLock::new(0),
            socket: None,
            shutdown_tx,
            event_tx,
        }
    }

    /// Initialize and start the SWIM protocol
    pub async fn initialize(&mut self) -> Result<(), SwimError> {
        // Create UDP socket
        let socket = UdpSocket::bind(self.config.bind_address).await?;
        
        // Join multicast group if applicable
        if let IpAddr::V4(group) = self.config.multicast_group {
            // Note: Multicast requires specific interface setup
            // This is simplified for demonstration
            let _ = group;
        }

        self.socket = Some(Arc::new(socket));
        
        // Add self to members
        let mut members = self.members.write().await;
        let mut self_info = MemberInfo::new(self.config.node_id, self.config.bind_address);
        self_info.incarnation = *self.incarnation.read().await;
        members.insert(self.config.node_id, self_info);

        Ok(())
    }

    /// Send a ping to a target node
    pub async fn send_ping(&self, target_id: NodeId) -> Result<(), SwimError> {
        let socket = self.socket.as_ref()
            .ok_or_else(|| SwimError::NotInitialized)?;

        let seq = {
            let mut seq = self.sequence_number.write().await;
            *seq += 1;
            *seq
        };

        let msg = SwimMessage::Ping {
            from: self.config.node_id,
            target: target_id,
            sequence_number: seq,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
        };

        let data = bincode::serialize(&msg)
            .map_err(|e| SwimError::SerializationError(e.to_string()))?;

        // Get target address
        let target_addr = {
            let members = self.members.read().await;
            members.get(&target_id)
                .map(|m| m.address)
                .ok_or_else(|| SwimError::NodeNotFound(target_id))?
        };

        socket.send_to(&data, target_addr).await?;
        Ok(())
    }

    /// Send acknowledgment
    pub async fn send_ack(&self, target_id: NodeId, seq: u64) -> Result<(), SwimError> {
        let socket = self.socket.as_ref()
            .ok_or_else(|| SwimError::NotInitialized)?;

        let msg = SwimMessage::Ack {
            from: self.config.node_id,
            target: target_id,
            sequence_number: seq,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
        };

        let data = bincode::serialize(&msg)
            .map_err(|e| SwimError::SerializationError(e.to_string()))?;

        let target_addr = {
            let members = self.members.read().await;
            members.get(&target_id)
                .map(|m| m.address)
                .ok_or_else(|| SwimError::NodeNotFound(target_id))?
        };

        socket.send_to(&data, target_addr).await?;
        Ok(())
    }

    /// Mark a node as suspected
    pub async fn suspect_node(&self, target_id: NodeId) -> Result<(), SwimError> {
        let mut members = self.members.write().await;
        
        if let Some(member) = members.get_mut(&target_id) {
            if member.state == MemberState::Alive {
                member.state = MemberState::Suspect;
                member.suspicion_start = Some(Instant::now());
                
                // Broadcast suspect message
                self.broadcast_suspect(target_id).await?;
            }
        }

        Ok(())
    }

    /// Broadcast suspect message
    async fn broadcast_suspect(&self, target_id: NodeId) -> Result<(), SwimError> {
        let socket = self.socket.as_ref()
            .ok_or_else(|| SwimError::NotInitialized)?;

        let inc = *self.incarnation.read().await;
        let msg = SwimMessage::Suspect {
            from: self.config.node_id,
            target: target_id,
            incarnation: inc,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
        };

        let data = bincode::serialize(&msg)
            .map_err(|e| SwimError::SerializationError(e.to_string()))?;

        let members = self.members.read().await;
        for (_, member) in members.iter() {
            if member.node_id != self.config.node_id && member.state == MemberState::Alive {
                let _ = socket.send_to(&data, member.address).await;
            }
        }

        Ok(())
    }

    /// Confirm a node as dead
    pub async fn confirm_dead(&self, target_id: NodeId) -> Result<(), SwimError> {
        let mut members = self.members.write().await;
        
        if let Some(member) = members.get_mut(&target_id) {
            member.state = MemberState::Dead;
            member.suspicion_start = None;
        }

        // Broadcast dead message
        self.broadcast_dead(target_id).await?;

        Ok(())
    }

    /// Broadcast dead message
    async fn broadcast_dead(&self, target_id: NodeId) -> Result<(), SwimError> {
        let socket = self.socket.as_ref()
            .ok_or_else(|| SwimError::NotInitialized)?;

        let inc = *self.incarnation.read().await;
        let msg = SwimMessage::Dead {
            from: self.config.node_id,
            target: target_id,
            incarnation: inc,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
        };

        let data = bincode::serialize(&msg)
            .map_err(|e| SwimError::SerializationError(e.to_string()))?;

        let members = self.members.read().await;
        for (_, member) in members.iter() {
            if member.node_id != self.config.node_id && member.state == MemberState::Alive {
                let _ = socket.send_to(&data, member.address).await;
            }
        }

        Ok(())
    }

    /// Check for suspicion timeouts and confirm dead nodes
    pub async fn check_suspicion_timeouts(&self) -> Result<(), SwimError> {
        let now = Instant::now();
        let mut to_confirm_dead = Vec::new();

        {
            let members = self.members.read().await;
            for (node_id, member) in members.iter() {
                if member.state == MemberState::Suspect {
                    if let Some(start) = member.suspicion_start {
                        if now.duration_since(start) > member.suspicion_timeout {
                            to_confirm_dead.push(*node_id);
                        }
                    }
                }
            }
        }

        for node_id in to_confirm_dead {
            self.confirm_dead(node_id).await?;
        }

        Ok(())
    }

    /// Get list of alive members
    pub async fn get_alive_members(&self) -> Vec<NodeId> {
        let members = self.members.read().await;
        members.iter()
            .filter(|(_, m)| m.state == MemberState::Alive)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get member count by state
    pub async fn get_member_counts(&self) -> BTreeMap<String, usize> {
        let members = self.members.read().await;
        let mut counts = BTreeMap::new();
        
        let mut alive = 0;
        let mut suspect = 0;
        let mut dead = 0;
        let mut left = 0;

        for (_, member) in members.iter() {
            match member.state {
                MemberState::Alive => alive += 1,
                MemberState::Suspect => suspect += 1,
                MemberState::Dead => dead += 1,
                MemberState::Left => left += 1,
            }
        }

        counts.insert("alive".to_string(), alive);
        counts.insert("suspect".to_string(), suspect);
        counts.insert("dead".to_string(), dead);
        counts.insert("left".to_string(), left);

        counts
    }

    /// Graceful leave from the cluster
    pub async fn leave(&self) -> Result<(), SwimError> {
        let socket = self.socket.as_ref()
            .ok_or_else(|| SwimError::NotInitialized)?;

        let inc = *self.incarnation.read().await;
        let msg = SwimMessage::Leave {
            node_id: self.config.node_id,
            incarnation: inc,
        };

        let data = bincode::serialize(&msg)
            .map_err(|e| SwimError::SerializationError(e.to_string()))?;

        let members = self.members.read().await;
        for (_, member) in members.iter() {
            if member.node_id != self.config.node_id {
                let _ = socket.send_to(&data, member.address).await;
            }
        }

        // Update own state
        {
            let mut members = self.members.write().await;
            if let Some(self_member) = members.get_mut(&self.config.node_id) {
                self_member.state = MemberState::Left;
            }
        }

        Ok(())
    }

    /// Shutdown the protocol
    pub async fn shutdown(&self) -> Result<(), SwimError> {
        self.leave().await?;
        let _ = self.shutdown_tx.send(()).await;
        Ok(())
    }
}

/// SWIM error types
#[derive(Debug, thiserror::Error)]
pub enum SwimError {
    #[error("Protocol not initialized")]
    NotInitialized,
    #[error("Node {0} not found")]
    NodeNotFound(NodeId),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Channel error")]
    ChannelError,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_swim_initialization() {
        let config = SwimConfig {
            node_id: 1,
            bind_address: "127.0.0.1:0".parse().unwrap(),
            ..Default::default()
        };

        let mut protocol = SwimProtocol::new(config);
        let result = protocol.initialize().await;
        assert!(result.is_ok());

        let counts = protocol.get_member_counts().await;
        assert_eq!(*counts.get("alive").unwrap(), 1);
    }

    #[tokio::test]
    async fn test_member_state_transitions() {
        let config = SwimConfig {
            node_id: 1,
            bind_address: "127.0.0.1:0".parse().unwrap(),
            suspicion_timeout: Duration::from_millis(100),
            ..Default::default()
        };

        let mut protocol = SwimProtocol::new(config);
        protocol.initialize().await.unwrap();

        // Add a test member
        {
            let mut members = protocol.members.write().await;
            let test_member = MemberInfo::new(2, "127.0.0.1:9998".parse().unwrap());
            members.insert(2, test_member);
        }

        // Verify initial state
        let counts = protocol.get_member_counts().await;
        assert_eq!(*counts.get("alive").unwrap(), 2);

        // Suspect the member
        protocol.suspect_node(2).await.unwrap();

        let counts = protocol.get_member_counts().await;
        assert_eq!(*counts.get("suspect").unwrap(), 1);

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;
        protocol.check_suspicion_timeouts().await.unwrap();

        let counts = protocol.get_member_counts().await;
        assert_eq!(*counts.get("dead").unwrap(), 1);
    }
}
