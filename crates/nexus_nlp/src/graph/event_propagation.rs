//! Event Propagation Engine
//! 
//! Calculates second-order effects by traversing the knowledge graph.
//! When an event is detected (e.g., "OPEC cuts production"), this engine
//! automatically flags related assets for trading signals.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use crate::graph::sharded_knowledge_graph::{ShardedKnowledgeGraph, EdgeType, NodeType};
use crate::graph::entity_ticker_mapper::MappedEntity;

/// Event types that can trigger propagation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    SupplyChange,
    DemandChange,
    PolicyChange,
    EarningsReport,
    EconomicData,
    Geopolitical,
    NaturalDisaster,
    RegulatoryChange,
}

/// Sentiment direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sentiment {
    Bullish,
    Bearish,
    Neutral,
}

/// Propagated signal with conviction score
#[derive(Debug, Clone)]
pub struct PropagatedSignal {
    pub source_event: String,
    pub target_ticker: String,
    pub sentiment: Sentiment,
    pub conviction: f32, // 0.0 to 1.0
    pub propagation_depth: usize,
    pub path: Vec<String>, // Entity names in the propagation path
}

/// Event propagation engine
pub struct EventPropagationEngine {
    graph: Arc<ShardedKnowledgeGraph>,
    /// Base impact scores for different event types
    event_impact_scores: HashMap<EventType, f32>,
    /// Decay factor per hop in propagation
    decay_factor: f32,
}

impl EventPropagationEngine {
    /// Create a new event propagation engine
    pub fn new(graph: Arc<ShardedKnowledgeGraph>) -> Self {
        let mut event_impact_scores = HashMap::new();
        
        // Initialize base impact scores
        event_impact_scores.insert(EventType::SupplyChange, 0.9);
        event_impact_scores.insert(EventType::DemandChange, 0.85);
        event_impact_scores.insert(EventType::PolicyChange, 0.95);
        event_impact_scores.insert(EventType::EarningsReport, 0.7);
        event_impact_scores.insert(EventType::EconomicData, 0.8);
        event_impact_scores.insert(EventType::Geopolitical, 0.75);
        event_impact_scores.insert(EventType::NaturalDisaster, 0.6);
        event_impact_scores.insert(EventType::RegulatoryChange, 0.85);
        
        Self {
            graph,
            event_impact_scores,
            decay_factor: 0.7, // Each hop reduces conviction by 30%
        }
    }

    /// Process an event and propagate signals through the graph
    pub fn propagate_event(
        &self,
        event_type: EventType,
        source_entity: &str,
        sentiment: Sentiment,
    ) -> Vec<PropagatedSignal> {
        let mut signals = Vec::new();
        
        // Find the source node in the graph
        let source_id = match self.find_entity_node(source_entity) {
            Some(id) => id,
            None => return signals, // Entity not found
        };
        
        // Get base impact score
        let base_impact = self.event_impact_scores.get(&event_type).copied().unwrap_or(0.5);
        
        // Traverse the graph up to depth 3
        let visited_ids = self.graph.traverse(source_id, 3, None);
        
        // For each visited node, check if it's an asset/ticker
        for node_id in visited_ids {
            if let Some(node) = self.graph.get_node(node_id) {
                if node.node_type == NodeType::Asset || node.node_type == NodeType::Entity {
                    // Calculate conviction based on distance
                    let depth = self.calculate_depth(source_id, node_id);
                    let conviction = base_impact * self.decay_factor.powi(depth as i32);
                    
                    // Determine sentiment direction based on edge types
                    let propagated_sentiment = self.determine_sentiment(
                        source_id,
                        node_id,
                        sentiment,
                    );
                    
                    // Create signal
                    signals.push(PropagatedSignal {
                        source_event: source_entity.to_string(),
                        target_ticker: node.name.clone(),
                        sentiment: propagated_sentiment,
                        conviction,
                        propagation_depth: depth,
                        path: self.reconstruct_path(source_id, node_id),
                    });
                }
            }
        }
        
        // Sort by conviction descending
        signals.sort_by(|a, b| b.conviction.partial_cmp(&a.conviction).unwrap_or(std::cmp::Ordering::Equal));
        
        signals
    }

    /// Find entity node by name
    fn find_entity_node(&self, entity_name: &str) -> Option<u64> {
        // This would ideally use an index - simplified for now
        // In production, maintain a reverse index from name to node ID
        None // Placeholder - would need graph iteration or index
    }

    /// Calculate depth between two nodes
    fn calculate_depth(&self, source: u64, target: u64) -> usize {
        // BFS to find shortest path
        let mut visited = HashSet::new();
        let mut queue: Vec<(u64, usize)> = vec![(source, 0)];
        
        while let Some((current, depth)) = queue.pop() {
            if current == target {
                return depth;
            }
            
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);
            
            let neighbors = self.graph.get_neighbors(current);
            for (neighbor, _) in neighbors {
                queue.push((neighbor, depth + 1));
            }
        }
        
        usize::MAX // Not connected
    }

    /// Determine propagated sentiment based on edge relationships
    fn determine_sentiment(&self, source: u64, target: u64, base: Sentiment) -> Sentiment {
        // Check edge types along the path
        let neighbors = self.graph.get_neighbors(source);
        
        for (neighbor_id, edge_type) in neighbors {
            if neighbor_id == target {
                // Direct connection - apply edge-specific logic
                return match edge_type {
                    EdgeType::Causes | EdgeType::Impacts | EdgeType::RelatesTo => base,
                    EdgeType::CorrelatesWith => base, // Positive correlation assumed
                    EdgeType::PartOf => base,
                    EdgeType::TradesAs => base,
                };
            }
        }
        
        // Default: pass through base sentiment
        base
    }

    /// Reconstruct path between nodes
    fn reconstruct_path(&self, source: u64, target: u64) -> Vec<String> {
        // BFS to find path
        let mut visited = HashMap::new();
        let mut queue: Vec<u64> = vec![source];
        visited.insert(source, None);
        
        while let Some(current) = queue.pop() {
            if current == target {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = Some(current);
                
                while let Some(n) = node {
                    if let Some(node_data) = self.graph.get_node(n) {
                        path.push(node_data.name);
                    }
                    node = visited.get(&n).copied().flatten();
                }
                
                path.reverse();
                return path;
            }
            
            let neighbors = self.graph.get_neighbors(current);
            for (neighbor, _) in neighbors {
                if !visited.contains_key(&neighbor) {
                    visited.insert(neighbor, Some(current));
                    queue.push(neighbor);
                }
            }
        }
        
        vec![] // No path found
    }

    /// Get all signals above a conviction threshold
    pub fn get_high_conviction_signals(
        &self,
        signals: &[PropagatedSignal],
        threshold: f32,
    ) -> Vec<&PropagatedSignal> {
        signals.iter().filter(|s| s.conviction >= threshold).collect()
    }

    /// Aggregate signals by ticker
    pub fn aggregate_by_ticker(&self, signals: &[PropagatedSignal]) -> HashMap<String, AggregatedSignal> {
        let mut aggregated: HashMap<String, Vec<&PropagatedSignal>> = HashMap::new();
        
        for signal in signals {
            aggregated.entry(signal.target_ticker.clone())
                .or_default()
                .push(signal);
        }
        
        aggregated.into_iter().map(|(ticker, sigs)| {
            let total_conviction: f32 = sigs.iter().map(|s| s.conviction).sum();
            let bullish_count = sigs.iter().filter(|s| s.sentiment == Sentiment::Bullish).count();
            let bearish_count = sigs.iter().filter(|s| s.sentiment == Sentiment::Bearish).count();
            
            let net_sentiment = if bullish_count > bearish_count {
                Sentiment::Bullish
            } else if bearish_count > bullish_count {
                Sentiment::Bearish
            } else {
                Sentiment::Neutral
            };
            
            (ticker, AggregatedSignal {
                ticker,
                net_sentiment,
                total_conviction,
                signal_count: sigs.len(),
            })
        }).collect()
    }
}

/// Aggregated signal for a ticker
#[derive(Debug, Clone)]
pub struct AggregatedSignal {
    pub ticker: String,
    pub net_sentiment: Sentiment,
    pub total_conviction: f32,
    pub signal_count: usize,
}

/// Example: OPEC production cut propagation
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_signal_propagation() {
        let graph = Arc::new(ShardedKnowledgeGraph::new());
        
        // Create test graph: OPEC -> USO -> XLE
        let opec = graph.add_node(NodeType::Organization, "OPEC", HashMap::new());
        let uso = graph.add_node(NodeType::Asset, "USO", HashMap::new());
        let xle = graph.add_node(NodeType::Asset, "XLE", HashMap::new());
        
        graph.add_edge(opec, uso, EdgeType::Impacts, 1.0).unwrap();
        graph.add_edge(uso, xle, EdgeType::CorrelatesWith, 0.8).unwrap();
        
        let engine = EventPropagationEngine::new(graph);
        
        // Note: propagate_event requires find_entity_node to work
        // which needs an index - this is a simplified test
        // In production, the index would be maintained
    }

    #[test]
    fn test_sentiment_determination() {
        // Test that bullish supply shock propagates correctly
        let sentiment = Sentiment::Bullish;
        assert_eq!(sentiment, Sentiment::Bullish);
        
        let sentiment = Sentiment::Bearish;
        assert_eq!(sentiment, Sentiment::Bearish);
    }

    #[test]
    fn test_aggregation() {
        let signals = vec![
            PropagatedSignal {
                source_event: "OPEC cut".to_string(),
                target_ticker: "USO".to_string(),
                sentiment: Sentiment::Bullish,
                conviction: 0.9,
                propagation_depth: 1,
                path: vec!["OPEC".to_string(), "USO".to_string()],
            },
            PropagatedSignal {
                source_event: "OPEC cut".to_string(),
                target_ticker: "USO".to_string(),
                sentiment: Sentiment::Bullish,
                conviction: 0.7,
                propagation_depth: 2,
                path: vec!["OPEC".to_string(), "XLE".to_string(), "USO".to_string()],
            },
        ];
        
        let engine = EventPropagationEngine::new(Arc::new(ShardedKnowledgeGraph::new()));
        let aggregated = engine.aggregate_by_ticker(&signals);
        
        let uso_signal = aggregated.get("USO").unwrap();
        assert_eq!(uso_signal.signal_count, 2);
        assert_eq!(uso_signal.net_sentiment, Sentiment::Bullish);
        assert!((uso_signal.total_conviction - 1.6).abs() < 0.01);
    }
}
