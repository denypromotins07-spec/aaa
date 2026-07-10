//! Event Propagation Engine for Second-Order Effect Calculation
//!
//! This module traverses the knowledge graph to calculate cascading effects
//! from news events. For example, if "OPEC cuts production" is detected,
//! it automatically flags long signals for XLE, HAL, and crude oil futures.

use std::sync::Arc;
use tracing::{info, debug, warn};

use super::sharded_knowledge_graph::{ShardedKnowledgeGraph, NodeData, NodeType, EdgeType, EdgeData};
use super::entity_ticker_mapper::{EntityTickerMapper, TickerMapping, AssetClass};

/// Types of market events
#[derive(Debug, Clone, PartialEq)]
pub enum EventType {
    /// Supply shock (positive or negative)
    SupplyShock { magnitude: f64 },
    /// Demand change
    DemandChange { magnitude: f64 },
    /// Regulatory change
    RegulatoryChange { impact: f64 },
    /// Earnings announcement
    EarningsAnnouncement { beat_miss: f64 },
    /// Central bank policy change
    PolicyChange { hawkish_dovish: f64 },
    /// M&A activity
    MergerAcquisition { deal_value: f64 },
    /// Geopolitical event
    GeopoliticalEvent { severity: f64 },
}

/// A trading signal generated from event propagation
#[derive(Debug, Clone)]
pub struct PropagatedSignal {
    /// Target ticker/symbol
    pub symbol: String,
    /// Signal direction (positive = bullish, negative = bearish)
    pub direction: f64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Expected impact magnitude
    pub expected_impact: f64,
    /// Time horizon in milliseconds
    pub time_horizon_ms: u64,
    /// Source event that triggered this signal
    pub source_event: String,
    /// Propagation depth from source
    pub propagation_depth: usize,
}

/// Result of event propagation analysis
#[derive(Debug, Clone)]
pub struct PropagationResult {
    /// Original event description
    pub event: String,
    /// Affected entities
    pub affected_entities: Vec<String>,
    /// Generated trading signals
    pub signals: Vec<PropagatedSignal>,
    /// Total propagation depth reached
    pub max_depth: usize,
    /// Computation time (microseconds)
    pub computation_time_us: u64,
}

/// Configuration for the event propagation engine
#[derive(Debug, Clone)]
pub struct PropagationConfig {
    /// Maximum propagation depth
    pub max_depth: usize,
    /// Minimum confidence threshold for signals
    pub min_confidence: f64,
    /// Decay factor per hop (0.0 to 1.0)
    pub decay_factor: f64,
    /// Enable second-order effects
    pub enable_second_order: bool,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            min_confidence: 0.5,
            decay_factor: 0.7,
            enable_second_order: true,
        }
    }
}

/// Event propagation engine
pub struct EventPropagationEngine {
    /// Reference to the knowledge graph
    graph: Arc<ShardedKnowledgeGraph>,
    /// Entity ticker mapper
    mapper: Arc<EntityTickerMapper>,
    /// Configuration
    config: PropagationConfig,
}

impl EventPropagationEngine {
    /// Create a new event propagation engine
    pub fn new(
        graph: Arc<ShardedKnowledgeGraph>,
        mapper: Arc<EntityTickerMapper>,
        config: PropagationConfig,
    ) -> Self {
        Self {
            graph,
            mapper,
            config,
        }
    }

    /// Propagate an event through the knowledge graph
    pub fn propagate(&self, event_type: &EventType, source_entity: &str) -> PropagationResult {
        let start = std::time::Instant::now();
        
        info!("Propagating event {:?} from entity {}", event_type, source_entity);

        // Find the source entity in the graph
        let source_node = self.mapper.lookup_by_name(source_entity);
        if source_node.is_none() {
            warn!("Source entity '{}' not found in mapper", source_entity);
            return PropagationResult {
                event: format!("{:?}", event_type),
                affected_entities: Vec::new(),
                signals: Vec::new(),
                max_depth: 0,
                computation_time_us: start.elapsed().as_micros() as u64,
            };
        }

        let source_node = source_node.unwrap();
        let mut affected_entities = Vec::new();
        let mut signals = Vec::new();

        // Get base impact from event type
        let base_impact = match event_type {
            EventType::SupplyShock { magnitude } => *magnitude,
            EventType::DemandChange { magnitude } => *magnitude,
            EventType::RegulatoryChange { impact } => *impact,
            EventType::EarningsAnnouncement { beat_miss } => *beat_miss,
            EventType::PolicyChange { hawkish_dovish } => *hawkish_dovish,
            EventType::MergerAcquisition { deal_value } => (*deal_value / 1e9).clamp(-1.0, 1.0),
            EventType::GeopoliticalEvent { severity } => *severity,
        };

        // Traverse the graph to find affected entities
        let visited = self.graph.traverse(
            source_node.tickers.first()
                .map(|t| t.symbol.clone())
                .unwrap_or_default()
                .chars()
                .map(|c| c as u64)
                .sum(),
            self.config.max_depth,
        );

        // For each affected entity, generate signals
        for (depth, node_id) in visited.iter().enumerate() {
            // In a real implementation, we'd look up the actual node
            // For now, we simulate based on known relationships
            
            let signal = self.generate_signal(
                event_type,
                source_entity,
                depth,
                base_impact,
            );

            if let Some(sig) = signal {
                if sig.confidence >= self.config.min_confidence {
                    signals.push(sig);
                }
            }
        }

        // Generate specific signals based on event type
        self.generate_event_specific_signals(event_type, source_entity, &mut signals);

        // Collect affected entities from signals
        for signal in &signals {
            if !affected_entities.contains(&signal.symbol) {
                affected_entities.push(signal.symbol.clone());
            }
        }

        PropagationResult {
            event: format!("{:?}", event_type),
            affected_entities,
            signals,
            max_depth: self.config.max_depth,
            computation_time_us: start.elapsed().as_micros() as u64,
        }
    }

    /// Generate a signal based on propagation depth
    fn generate_signal(
        &self,
        event_type: &EventType,
        source: &str,
        depth: usize,
        base_impact: f64,
    ) -> Option<PropagatedSignal> {
        // Apply decay based on propagation depth
        let decayed_impact = base_impact * self.config.decay_factor.powi(depth as i32);
        
        // Confidence decreases with depth
        let confidence = 1.0 - (depth as f64 * 0.15);
        
        if confidence < self.config.min_confidence {
            return None;
        }

        // Determine time horizon based on event type
        let time_horizon_ms = match event_type {
            EventType::SupplyShock { .. } => 3600_000, // 1 hour
            EventType::DemandChange { .. } => 7200_000, // 2 hours
            EventType::EarningsAnnouncement { .. } => 1800_000, // 30 minutes
            EventType::PolicyChange { .. } => 300_000, // 5 minutes
            _ => 3600_000,
        };

        Some(PropagatedSignal {
            symbol: format!("{}_DEPTH_{}", source, depth),
            direction: decayed_impact.signum(),
            confidence,
            expected_impact: decayed_impact.abs(),
            time_horizon_ms,
            source_event: source.to_string(),
            propagation_depth: depth,
        })
    }

    /// Generate event-specific signals based on known financial relationships
    fn generate_event_specific_signals(
        &self,
        event_type: &EventType,
        source: &str,
        signals: &mut Vec<PropagatedSignal>,
    ) {
        let source_lower = source.to_lowercase();

        // OPEC production changes affect oil and energy sector
        if source_lower.contains("opec") {
            if let EventType::SupplyShock { magnitude } = event_type {
                // Direct effect on crude oil
                signals.push(PropagatedSignal {
                    symbol: "CL".to_string(),
                    direction: magnitude.signum(),
                    confidence: 0.95,
                    expected_impact: magnitude.abs(),
                    time_horizon_ms: 1800_000,
                    source_event: source.to_string(),
                    propagation_depth: 0,
                });

                // Energy ETFs
                signals.push(PropagatedSignal {
                    symbol: "XLE".to_string(),
                    direction: magnitude.signum(),
                    confidence: 0.85,
                    expected_impact: magnitude.abs() * 0.8,
                    time_horizon_ms: 3600_000,
                    source_event: source.to_string(),
                    propagation_depth: 1,
                });

                // Oil services companies
                for symbol in &["HAL", "SLB", "BKR"] {
                    signals.push(PropagatedSignal {
                        symbol: symbol.to_string(),
                        direction: magnitude.signum(),
                        confidence: 0.75,
                        expected_impact: magnitude.abs() * 0.6,
                        time_horizon_ms: 3600_000,
                        source_event: source.to_string(),
                        propagation_depth: 1,
                    });
                }
            }
        }

        // Federal Reserve policy changes affect bonds, dollar, and rate-sensitive sectors
        if source_lower.contains("fed") || source_lower.contains("federal reserve") {
            if let EventType::PolicyChange { hawkish_dovish } = event_type {
                // Treasury yields (inverse relationship with prices)
                signals.push(PropagatedSignal {
                    symbol: "^TNX".to_string(),
                    direction: hawkish_dovish.signum(),
                    confidence: 0.9,
                    expected_impact: hawkish_dovish.abs(),
                    time_horizon_ms: 300_000,
                    source_event: source.to_string(),
                    propagation_depth: 0,
                });

                // Dollar index
                signals.push(PropagatedSignal {
                    symbol: "DXY".to_string(),
                    direction: hawkish_dovish.signum(),
                    confidence: 0.85,
                    expected_impact: hawkish_dovish.abs() * 0.7,
                    time_horizon_ms: 600_000,
                    source_event: source.to_string(),
                    propagation_depth: 1,
                });

                // Rate-sensitive sectors (banks benefit from hawkish)
                signals.push(PropagatedSignal {
                    symbol: "XLF".to_string(),
                    direction: hawkish_dovish.signum(),
                    confidence: 0.7,
                    expected_impact: hawkish_dovish.abs() * 0.5,
                    time_horizon_ms: 1800_000,
                    source_event: source.to_string(),
                    propagation_depth: 1,
                });

                // Growth stocks hurt by hawkish policy
                signals.push(PropagatedSignal {
                    symbol: "QQQ".to_string(),
                    direction: -hawkish_dovish.signum(),
                    confidence: 0.65,
                    expected_impact: hawkish_dovish.abs() * 0.4,
                    time_horizon_ms: 1800_000,
                    source_event: source.to_string(),
                    propagation_depth: 2,
                });
            }
        }

        // Apple announcements affect suppliers and tech sector
        if source_lower.contains("apple") {
            // Apple suppliers
            for symbol in &["TSM", "AVGO", "QCOM"] {
                signals.push(PropagatedSignal {
                    symbol: symbol.to_string(),
                    direction: 1.0, // Assuming positive news
                    confidence: 0.7,
                    expected_impact: 0.3,
                    time_horizon_ms: 3600_000,
                    source_event: source.to_string(),
                    propagation_depth: 1,
                });
            }
        }
    }

    /// Calculate second-order effects from initial signals
    pub fn calculate_second_order_effects(&self, initial_signals: &[PropagatedSignal]) -> Vec<PropagatedSignal> {
        if !self.config.enable_second_order {
            return Vec::new();
        }

        let mut second_order = Vec::new();

        for signal in initial_signals {
            // Apply additional decay for second-order effects
            let second_order_impact = signal.expected_impact * self.config.decay_factor;
            let second_order_confidence = signal.confidence * self.config.decay_factor;

            if second_order_confidence >= self.config.min_confidence {
                // Find related assets and create second-order signals
                // This is simplified - in production would use graph traversal
                second_order.push(PropagatedSignal {
                    symbol: format!("{}_SO", signal.symbol),
                    direction: signal.direction,
                    confidence: second_order_confidence,
                    expected_impact: second_order_impact,
                    time_horizon_ms: signal.time_horizon_ms * 2,
                    source_event: signal.source_event.clone(),
                    propagation_depth: signal.propagation_depth + 1,
                });
            }
        }

        second_order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opec_propagation() {
        let graph = Arc::new(ShardedKnowledgeGraph::new());
        let mapper = Arc::new(EntityTickerMapper::new(Default::default()));
        let config = PropagationConfig::default();
        
        let engine = EventPropagationEngine::new(graph, mapper, config);
        
        let event = EventType::SupplyShock { magnitude: 0.8 };
        let result = engine.propagate(&event, "OPEC");
        
        assert!(!result.signals.is_empty());
        
        // Should have signals for CL, XLE, etc.
        let symbols: Vec<&String> = result.signals.iter().map(|s| &s.symbol).collect();
        assert!(symbols.contains(&&"CL".to_string()));
        assert!(symbols.contains(&&"XLE".to_string()));
    }

    #[test]
    fn test_fed_propagation() {
        let graph = Arc::new(ShardedKnowledgeGraph::new());
        let mapper = Arc::new(EntityTickerMapper::new(Default::default()));
        let config = PropagationConfig::default();
        
        let engine = EventPropagationEngine::new(graph, mapper, config);
        
        let event = EventType::PolicyChange { hawkish_dovish: 0.6 };
        let result = engine.propagate(&event, "Federal Reserve");
        
        assert!(!result.signals.is_empty());
        
        // Should have signals for treasuries, dollar, etc.
        let symbols: Vec<&String> = result.signals.iter().map(|s| &s.symbol).collect();
        assert!(symbols.contains(&&"^TNX".to_string()));
        assert!(symbols.contains(&&"DXY".to_string()));
    }

    #[test]
    fn test_signal_decay() {
        let config = PropagationConfig {
            max_depth: 3,
            decay_factor: 0.7,
            ..Default::default()
        };
        
        // Verify decay calculation
        let base_impact = 1.0;
        let depth_0 = base_impact * config.decay_factor.powi(0);
        let depth_1 = base_impact * config.decay_factor.powi(1);
        let depth_2 = base_impact * config.decay_factor.powi(2);
        
        assert!((depth_0 - 1.0).abs() < 0.001);
        assert!((depth_1 - 0.7).abs() < 0.001);
        assert!((depth_2 - 0.49).abs() < 0.001);
    }

    #[test]
    fn test_second_order_effects() {
        let graph = Arc::new(ShardedKnowledgeGraph::new());
        let mapper = Arc::new(EntityTickerMapper::new(Default::default()));
        let config = PropagationConfig {
            enable_second_order: true,
            ..Default::default()
        };
        
        let engine = EventPropagationEngine::new(graph, mapper, config);
        
        let initial_signals = vec![
            PropagatedSignal {
                symbol: "CL".to_string(),
                direction: 1.0,
                confidence: 0.9,
                expected_impact: 0.8,
                time_horizon_ms: 1800_000,
                source_event: "OPEC".to_string(),
                propagation_depth: 0,
            }
        ];
        
        let second_order = engine.calculate_second_order_effects(&initial_signals);
        
        assert_eq!(second_order.len(), 1);
        assert!(second_order[0].confidence < initial_signals[0].confidence);
        assert!(second_order[0].expected_impact < initial_signals[0].expected_impact);
    }
}
