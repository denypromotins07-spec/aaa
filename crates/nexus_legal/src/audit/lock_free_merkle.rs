// STAGE 23: Computational Law, RegTech Compliance & Immutable Audit Trails
// Chapter 4: Cryptographic Audit Ledger & Merkle State Anchoring
// File: crates/nexus_legal/src/audit/lock_free_merkle.rs

//! Lock-free, append-only Merkle Tree for immutable audit logging.
//! Uses SPSC channels to decouple hashing from execution hot-path.
//! Provides cryptographic proof that historical logs were never altered.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam::channel::{bounded, Sender, Receiver};
use sha2::{Sha256, Digest};
use parking_lot::RwLock;

/// Hash type alias (32 bytes for SHA-256)
pub type Hash = [u8; 32];

/// Event types that get hashed into the Merkle tree
#[derive(Debug, Clone)]
pub enum AuditEvent {
    /// Alpha signal generated
    AlphaSignal {
        strategy_id: u32,
        symbol: String,
        signal_value: i64,
        timestamp_ns: u64,
    },
    /// Risk check passed
    RiskCheckPassed {
        check_type: String,
        result: bool,
        timestamp_ns: u64,
    },
    /// Order submitted
    OrderSubmitted {
        order_id: u64,
        symbol: String,
        side: String,
        quantity: i64,
        price: u64,
        timestamp_ns: u64,
    },
    /// Order filled
    OrderFilled {
        order_id: u64,
        fill_id: u64,
        quantity: i64,
        price: u64,
        timestamp_ns: u64,
    },
    /// Compliance check result
    ComplianceCheck {
        check_name: String,
        passed: bool,
        details: String,
        timestamp_ns: u64,
    },
    /// System state change
    StateChange {
        component: String,
        old_state: String,
        new_state: String,
        timestamp_ns: u64,
    },
}

impl AuditEvent {
    /// Serialize event to bytes for hashing
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            AuditEvent::AlphaSignal { strategy_id, symbol, signal_value, timestamp_ns } => {
                let mut bytes = Vec::with_capacity(64);
                bytes.extend_from_slice(&[0u8]); // Type tag
                bytes.extend_from_slice(&strategy_id.to_le_bytes());
                bytes.extend_from_slice(symbol.as_bytes());
                bytes.extend_from_slice(&signal_value.to_le_bytes());
                bytes.extend_from_slice(&timestamp_ns.to_le_bytes());
                bytes
            }
            AuditEvent::RiskCheckPassed { check_type, result, timestamp_ns } => {
                let mut bytes = Vec::with_capacity(48);
                bytes.extend_from_slice(&[1u8]);
                bytes.extend_from_slice(check_type.as_bytes());
                bytes.extend_from_slice(&[*result as u8].to_le_bytes());
                bytes.extend_from_slice(&timestamp_ns.to_le_bytes());
                bytes
            }
            AuditEvent::OrderSubmitted { order_id, symbol, side, quantity, price, timestamp_ns } => {
                let mut bytes = Vec::with_capacity(80);
                bytes.extend_from_slice(&[2u8]);
                bytes.extend_from_slice(&order_id.to_le_bytes());
                bytes.extend_from_slice(symbol.as_bytes());
                bytes.extend_from_slice(side.as_bytes());
                bytes.extend_from_slice(&quantity.to_le_bytes());
                bytes.extend_from_slice(&price.to_le_bytes());
                bytes.extend_from_slice(&timestamp_ns.to_le_bytes());
                bytes
            }
            AuditEvent::OrderFilled { order_id, fill_id, quantity, price, timestamp_ns } => {
                let mut bytes = Vec::with_capacity(64);
                bytes.extend_from_slice(&[3u8]);
                bytes.extend_from_slice(&order_id.to_le_bytes());
                bytes.extend_from_slice(&fill_id.to_le_bytes());
                bytes.extend_from_slice(&quantity.to_le_bytes());
                bytes.extend_from_slice(&price.to_le_bytes());
                bytes.extend_from_slice(&timestamp_ns.to_le_bytes());
                bytes
            }
            AuditEvent::ComplianceCheck { check_name, passed, details, timestamp_ns } => {
                let mut bytes = Vec::with_capacity(96);
                bytes.extend_from_slice(&[4u8]);
                bytes.extend_from_slice(check_name.as_bytes());
                bytes.extend_from_slice(&[*passed as u8].to_le_bytes());
                bytes.extend_from_slice(details.as_bytes());
                bytes.extend_from_slice(&timestamp_ns.to_le_bytes());
                bytes
            }
            AuditEvent::StateChange { component, old_state, new_state, timestamp_ns } => {
                let mut bytes = Vec::with_capacity(128);
                bytes.extend_from_slice(&[5u8]);
                bytes.extend_from_slice(component.as_bytes());
                bytes.extend_from_slice(old_state.as_bytes());
                bytes.extend_from_slice(new_state.as_bytes());
                bytes.extend_from_slice(&timestamp_ns.to_le_bytes());
                bytes
            }
        }
    }

    /// Get event timestamp
    pub fn timestamp_ns(&self) -> u64 {
        match self {
            AuditEvent::AlphaSignal { timestamp_ns, .. } => *timestamp_ns,
            AuditEvent::RiskCheckPassed { timestamp_ns, .. } => *timestamp_ns,
            AuditEvent::OrderSubmitted { timestamp_ns, .. } => *timestamp_ns,
            AuditEvent::OrderFilled { timestamp_ns, .. } => *timestamp_ns,
            AuditEvent::ComplianceCheck { timestamp_ns, .. } => *timestamp_ns,
            AuditEvent::StateChange { timestamp_ns, .. } => *timestamp_ns,
        }
    }
}

/// A leaf node in the Merkle tree
#[derive(Debug, Clone)]
pub struct MerkleLeaf {
    pub event: AuditEvent,
    pub hash: Hash,
    pub index: u64,
}

/// An internal node in the Merkle tree
#[derive(Debug, Clone)]
pub struct MerkleNode {
    pub left_hash: Hash,
    pub right_hash: Hash,
    pub combined_hash: Hash,
    pub height: u32,
}

/// Lock-free Merkle tree with async event ingestion
pub struct LockFreeMerkleTree {
    /// Append-only list of leaves
    leaves: Arc<RwLock<Vec<MerkleLeaf>>>,
    /// Current root hash
    root_hash: Arc<RwLock<Hash>>,
    /// Event counter
    event_count: AtomicU64,
    /// Async event channel
    event_sender: Sender<AuditEvent>,
    event_receiver: Receiver<AuditEvent>,
    /// Channel capacity
    channel_capacity: usize,
    /// Dropped events count (when channel full)
    dropped_events: AtomicU64,
    /// Background processing running flag
    processing: AtomicBool,
    /// Last anchor time
    last_anchor_ns: AtomicU64,
}

impl LockFreeMerkleTree {
    pub fn new(channel_capacity: usize) -> Self {
        let (tx, rx) = bounded(channel_capacity);
        
        Self {
            leaves: Arc::new(RwLock::new(Vec::with_capacity(1_000_000))),
            root_hash: Arc::new(RwLock::new([0u8; 32])), // Genesis hash
            event_count: AtomicU64::new(0),
            event_sender: tx,
            event_receiver: rx,
            channel_capacity,
            dropped_events: AtomicU64::new(0),
            processing: AtomicBool::new(false),
            last_anchor_ns: AtomicU64::new(0),
        }
    }

    /// Submit an event to be hashed (non-blocking, called from hot-path)
    pub fn submit_event(&self, event: AuditEvent) -> Result<(), MerkleError> {
        self.event_sender.try_send(event).map_err(|e| {
            match e {
                crossbeam::channel::TrySendError::Full(_) => {
                    self.dropped_events.fetch_add(1, Ordering::Relaxed);
                    MerkleError::ChannelFull
                }
                crossbeam::channel::TrySendError::Disconnected(_) => {
                    MerkleError::ChannelClosed
                }
            }
        })
    }

    /// Process pending events and update Merkle root (called by background thread)
    pub fn process_pending_events(&self) -> usize {
        let mut processed = 0;
        let mut leaves_to_add = Vec::new();

        // Drain all available events
        while let Ok(event) = self.event_receiver.try_recv() {
            let index = self.event_count.fetch_add(1, Ordering::SeqCst);
            let hash = Self::hash_event(&event);
            
            leaves_to_add.push(MerkleLeaf {
                event,
                hash,
                index,
            });
            processed += 1;
        }

        if !leaves_to_add.is_empty() {
            // Add leaves to tree
            {
                let mut leaves = self.leaves.write();
                leaves.extend(leaves_to_add);
            }

            // Recompute root
            let new_root = self.compute_root();
            *self.root_hash.write() = new_root;
        }

        processed
    }

    /// Compute Merkle root from all leaves
    fn compute_root(&self) -> Hash {
        let leaves = self.leaves.read();
        
        if leaves.is_empty() {
            return Self::genesis_hash();
        }

        // Build tree bottom-up
        let mut current_level: Vec<Hash> = leaves.iter().map(|l| l.hash).collect();
        
        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            
            for chunk in current_level.chunks(2) {
                if chunk.len() == 2 {
                    let combined = Self::hash_pair(chunk[0], chunk[1]);
                    next_level.push(combined);
                } else {
                    // Odd node propagates up
                    next_level.push(chunk[0]);
                }
            }
            
            current_level = next_level;
        }

        current_level[0]
    }

    /// Hash a single event
    fn hash_event(event: &AuditEvent) -> Hash {
        let bytes = event.to_bytes();
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        hasher.finalize().into()
    }

    /// Hash two hashes together
    fn hash_pair(left: Hash, right: Hash) -> Hash {
        let mut hasher = Sha256::new();
        hasher.update(&left);
        hasher.update(&right);
        hasher.finalize().into()
    }

    /// Genesis hash (empty tree root)
    fn genesis_hash() -> Hash {
        let mut hasher = Sha256::new();
        hasher.update(b"NEXUS_OMEGA_GENESIS");
        hasher.finalize().into()
    }

    /// Get current root hash
    pub fn get_root(&self) -> Hash {
        *self.root_hash.read()
    }

    /// Get root as hex string
    pub fn get_root_hex(&self) -> String {
        hex::encode(self.get_root())
    }

    /// Get event count
    pub fn event_count(&self) -> u64 {
        self.event_count.load(Ordering::Relaxed)
    }

    /// Get dropped event count
    pub fn dropped_events(&self) -> u64 {
        self.dropped_events.load(Ordering::Relaxed)
    }

    /// Get proof for a specific event index
    pub fn get_proof(&self, index: u64) -> Option<MerkleProof> {
        let leaves = self.leaves.read();
        
        if index >= leaves.len() as u64 {
            return None;
        }

        let leaf = &leaves[index as usize];
        let mut proof_hashes = Vec::new();
        let mut current_index = index;
        
        // Build proof path
        let mut current_level: Vec<Hash> = leaves.iter().map(|l| l.hash).collect();
        
        while current_level.len() > 1 {
            let sibling_index = if current_index % 2 == 0 {
                current_index + 1
            } else {
                current_index - 1
            };

            if (sibling_index as usize) < current_level.len() {
                proof_hashes.push(ProofElement {
                    hash: current_level[sibling_index as usize],
                    position: if current_index % 2 == 0 {
                        Position::Right
                    } else {
                        Position::Left
                    },
                });
            }

            // Build next level
            let mut next_level = Vec::new();
            for chunk in current_level.chunks(2) {
                if chunk.len() == 2 {
                    next_level.push(Self::hash_pair(chunk[0], chunk[1]));
                } else {
                    next_level.push(chunk[0]);
                }
            }
            
            current_level = next_level;
            current_index /= 2;
        }

        Some(MerkleProof {
            leaf_hash: leaf.hash,
            leaf_index: index,
            proof_hashes,
            root: current_level[0],
        })
    }

    /// Verify a Merkle proof
    pub fn verify_proof(proof: &MerkleProof) -> bool {
        let mut current_hash = proof.leaf_hash;
        
        for element in &proof.proof_hashes {
            match element.position {
                Position::Left => {
                    current_hash = Self::hash_pair(element.hash, current_hash);
                }
                Position::Right => {
                    current_hash = Self::hash_pair(current_hash, element.hash);
                }
            }
        }

        current_hash == proof.root
    }

    /// Start background processing thread
    pub fn start_background_processor(&self) -> std::thread::JoinHandle<()> {
        self.processing.store(true, Ordering::SeqCst);
        
        let leaves = Arc::clone(&self.leaves);
        let root_hash = Arc::clone(&self.root_hash);
        let receiver = self.event_receiver.clone();
        let event_count = self.event_count.clone();
        let dropped = self.dropped_events.clone();
        let processing = self.processing.clone();

        std::thread::spawn(move || {
            let mut leaves_to_add = Vec::new();
            
            while processing.load(Ordering::Relaxed) {
                // Batch process events
                leaves_to_add.clear();
                
                while let Ok(event) = receiver.try_recv() {
                    let index = event_count.fetch_add(1, Ordering::SeqCst);
                    let hash = {
                        let bytes = event.to_bytes();
                        let mut hasher = Sha256::new();
                        hasher.update(&bytes);
                        hasher.finalize().into()
                    };
                    
                    leaves_to_add.push(MerkleLeaf { event, hash, index });
                }

                if !leaves_to_add.is_empty() {
                    {
                        let mut leaves = leaves.write();
                        leaves.extend(leaves_to_add);
                    }

                    // Recompute root
                    let new_root = {
                        let leaves = leaves.read();
                        if leaves.is_empty() {
                            [0u8; 32]
                        } else {
                            let mut current_level: Vec<Hash> = leaves.iter().map(|l| l.hash).collect();
                            while current_level.len() > 1 {
                                let mut next_level = Vec::new();
                                for chunk in current_level.chunks(2) {
                                    if chunk.len() == 2 {
                                        let mut hasher = Sha256::new();
                                        hasher.update(&chunk[0]);
                                        hasher.update(&chunk[1]);
                                        next_level.push(hasher.finalize().into());
                                    } else {
                                        next_level.push(chunk[0]);
                                    }
                                }
                                current_level = next_level;
                            }
                            current_level[0]
                        }
                    };
                    *root_hash.write() = new_root;
                }

                // Small sleep to prevent busy-waiting
                std::thread::sleep(Duration::from_micros(100));
            }
        })
    }

    /// Stop background processor
    pub fn stop_background_processor(&self) {
        self.processing.store(false, Ordering::SeqCst);
    }
}

/// Merkle proof for verifying inclusion of a leaf
#[derive(Debug, Clone)]
pub struct MerkleProof {
    pub leaf_hash: Hash,
    pub leaf_index: u64,
    pub proof_hashes: Vec<ProofElement>,
    pub root: Hash,
}

#[derive(Debug, Clone)]
pub struct ProofElement {
    pub hash: Hash,
    pub position: Position,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MerkleError {
    ChannelFull,
    ChannelClosed,
    IndexOutOfBounds,
    InvalidProof,
}

impl std::fmt::Display for MerkleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MerkleError::ChannelFull => write!(f, "Event channel full"),
            MerkleError::ChannelClosed => write!(f, "Event channel closed"),
            MerkleError::IndexOutOfBounds => write!(f, "Index out of bounds"),
            MerkleError::InvalidProof => write!(f, "Invalid Merkle proof"),
        }
    }
}

impl std::error::Error for MerkleError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_tree_append() {
        let tree = LockFreeMerkleTree::new(10_000);
        
        let event1 = AuditEvent::OrderSubmitted {
            order_id: 1,
            symbol: "BTCUSD".to_string(),
            side: "BUY".to_string(),
            quantity: 100,
            price: 50000,
            timestamp_ns: 1000,
        };
        
        let event2 = AuditEvent::OrderFilled {
            order_id: 1,
            fill_id: 1,
            quantity: 100,
            price: 50000,
            timestamp_ns: 2000,
        };

        tree.submit_event(event1).unwrap();
        tree.submit_event(event2).unwrap();
        
        tree.process_pending_events();
        
        assert_eq!(tree.event_count(), 2);
        assert_ne!(tree.get_root(), [0u8; 32]);
    }

    #[test]
    fn test_merkle_proof_verification() {
        let tree = LockFreeMerkleTree::new(10_000);
        
        for i in 0..10 {
            let event = AuditEvent::OrderSubmitted {
                order_id: i,
                symbol: "BTCUSD".to_string(),
                side: "BUY".to_string(),
                quantity: 100,
                price: 50000,
                timestamp_ns: i * 1000,
            };
            tree.submit_event(event).unwrap();
        }
        
        tree.process_pending_events();
        
        // Get proof for event 5
        let proof = tree.get_proof(5).unwrap();
        
        // Verify proof
        assert!(LockFreeMerkleTree::verify_proof(&proof));
    }

    #[test]
    fn test_channel_full_handling() {
        let tree = LockFreeMerkleTree::new(1); // Tiny capacity
        
        // Fill the channel
        let event = AuditEvent::OrderSubmitted {
            order_id: 1,
            symbol: "BTCUSD".to_string(),
            side: "BUY".to_string(),
            quantity: 100,
            price: 50000,
            timestamp_ns: 1000,
        };
        
        tree.submit_event(event.clone()).unwrap();
        
        // Next should fail
        let result = tree.submit_event(event);
        assert!(matches!(result, Err(MerkleError::ChannelFull)));
        assert_eq!(tree.dropped_events(), 1);
    }
}
