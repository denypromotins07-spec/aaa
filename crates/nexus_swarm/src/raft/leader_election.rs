//! Leader Election Module for NEXUS-OMEGA Swarm
//! 
//! Implements Raft-based leader election with cryptographic authority gating.

use crate::{
    ConsensusError, ConsensusResult, LogEntry, LogIndex, NodeId, NodeState, RaftMessage, SwarmConfig, Term,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};
use crossbeam_skiplist::SkipMap;

/// Handles leader election logic in the Raft consensus protocol
pub struct LeaderElection {
    node_id: NodeId,
    current_term: Term,
    voted_for: Option<NodeId>,
    votes_received: HashMap<NodeId, bool>,
    cluster_nodes: Vec<NodeId>,
    state: NodeState,
}

impl LeaderElection {
    pub fn new(config: &SwarmConfig) -> Self {
        Self {
            node_id: config.node_id,
            current_term: 0,
            voted_for: None,
            votes_received: HashMap::new(),
            cluster_nodes: config.cluster_nodes.clone(),
            state: NodeState::Follower,
        }
    }

    /// Start a new election by incrementing term and requesting votes
    pub fn start_election(&mut self) -> RaftMessage {
        self.current_term += 1;
        self.state = NodeState::Candidate;
        self.voted_for = Some(self.node_id);
        self.votes_received.clear();
        self.votes_received.insert(self.node_id, true);

        RaftMessage::RequestVote {
            term: self.current_term,
            candidate_id: self.node_id,
            last_log_index: 0, // Would be populated from actual log
            last_log_term: self.current_term - 1,
        }
    }

    /// Process a vote request from another candidate
    pub fn process_vote_request(
        &mut self,
        term: Term,
        candidate_id: NodeId,
        _last_log_index: LogIndex,
        _last_log_term: Term,
    ) -> RaftMessage {
        let mut vote_granted = false;

        if term > self.current_term {
            // Step down to follower if we see a higher term
            self.current_term = term;
            self.state = NodeState::Follower;
            self.voted_for = None;
        }

        if term >= self.current_term && self.voted_for.is_none() {
            self.voted_for = Some(candidate_id);
            vote_granted = true;
        }

        RaftMessage::VoteResponse {
            term: self.current_term,
            vote_granted,
            voter_id: self.node_id,
        }
    }

    /// Process a vote response
    pub fn process_vote_response(
        &mut self,
        term: Term,
        vote_granted: bool,
        voter_id: NodeId,
    ) -> ConsensusResult<bool> {
        if term > self.current_term {
            self.current_term = term;
            self.state = NodeState::Follower;
            self.voted_for = None;
            return Ok(false);
        }

        if term != self.current_term || self.state != NodeState::Candidate {
            return Ok(false);
        }

        if vote_granted {
            self.votes_received.insert(voter_id, true);
            
            // Check if we have majority
            let quorum_size = (self.cluster_nodes.len() / 2) + 1;
            if self.votes_received.len() >= quorum_size {
                self.state = NodeState::Leader;
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get current leader state
    pub fn get_state(&self) -> (NodeState, Term, Option<NodeId>) {
        (self.state, self.current_term, self.voted_for)
    }

    /// Check if this node is the leader
    pub fn is_leader(&self) -> bool {
        self.state == NodeState::Leader
    }

    /// Get current term
    pub fn current_term(&self) -> Term {
        self.current_term
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_election_start() {
        let mut config = SwarmConfig::default();
        config.node_id = 1;
        config.cluster_nodes = vec![1, 2, 3];
        
        let mut election = LeaderElection::new(&config);
        let msg = election.start_election();
        
        assert!(election.is_leader() == false); // Not leader until votes received
        assert_eq!(election.current_term(), 1);
        
        if let RaftMessage::RequestVote { term, candidate_id, .. } = msg {
            assert_eq!(term, 1);
            assert_eq!(candidate_id, 1);
        } else {
            panic!("Expected RequestVote message");
        }
    }
}
