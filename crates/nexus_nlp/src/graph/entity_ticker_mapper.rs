//! Entity to Ticker Mapper for Named Entity Recognition (NER)
//!
//! This module maps extracted entities (e.g., "Jerome Powell", "Apple", "OPEC")
//! to specific financial tickers and asset classes.

use std::collections::HashMap;
use std::sync::Arc;
use dashmap::DashMap;
use tracing::{info, debug};

/// Types of financial entities
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntityType {
    /// Company/Corporation
    Company,
    /// Person (e.g., CEO, Fed Chair)
    Person,
    /// Organization (e.g., OPEC, Federal Reserve)
    Organization,
    /// Commodity
    Commodity,
    /// Currency
    Currency,
    /// Economic Indicator
    EconomicIndicator,
    /// Geographic Location
    Location,
    /// Unknown
    Unknown,
}

/// Asset class classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AssetClass {
    Equity,
    FixedIncome,
    Commodity,
    ForeignExchange,
    Derivative,
    Cryptocurrency,
}

/// Mapped ticker information
#[derive(Debug, Clone)]
pub struct TickerMapping {
    /// Primary ticker symbol
    pub symbol: String,
    /// Exchange identifier
    pub exchange: String,
    /// Asset class
    pub asset_class: AssetClass,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Alternative symbols (e.g., options, futures)
    pub alternatives: Vec<String>,
}

/// Entity record with all associated tickers
#[derive(Debug, Clone)]
pub struct EntityRecord {
    /// Canonical name of the entity
    pub name: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Associated tickers
    pub tickers: Vec<TickerMapping>,
    /// Related entities (for graph traversal)
    pub related_entities: Vec<String>,
    /// Last updated timestamp (nanoseconds)
    pub last_updated_ns: u128,
}

impl EntityRecord {
    pub fn new(name: String, entity_type: EntityType) -> Self {
        Self {
            name,
            entity_type,
            tickers: Vec::new(),
            related_entities: Vec::new(),
            last_updated_ns: 0,
        }
    }

    pub fn add_ticker(&mut self, ticker: TickerMapping) {
        self.tickers.push(ticker);
    }
}

/// Configuration for the entity mapper
#[derive(Debug, Clone)]
pub struct EntityMapperConfig {
    /// Enable fuzzy matching
    pub enable_fuzzy: bool,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Maximum number of ticker results
    pub max_results: usize,
}

impl Default for EntityMapperConfig {
    fn default() -> Self {
        Self {
            enable_fuzzy: true,
            min_confidence: 0.7,
            max_results: 5,
        }
    }
}

/// High-speed Named Entity Recognition router
pub struct EntityTickerMapper {
    /// Entity name -> Entity record mapping
    entities: DashMap<String, EntityRecord>,
    /// Alias -> Canonical name mapping
    aliases: DashMap<String, String>,
    /// Ticker -> Entity name reverse mapping
    ticker_to_entity: DashMap<String, String>,
    /// Configuration
    config: EntityMapperConfig,
    /// Statistics
    hits: Arc<std::sync::atomic::AtomicU64>,
    misses: Arc<std::sync::atomic::AtomicU64>,
}

impl EntityTickerMapper {
    /// Create a new entity ticker mapper
    pub fn new(config: EntityMapperConfig) -> Self {
        Self {
            entities: DashMap::new(),
            aliases: DashMap::new(),
            ticker_to_entity: DashMap::new(),
            config,
            hits: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            misses: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Register a new entity with its associated tickers
    pub fn register_entity(&self, mut record: EntityRecord) {
        let name = record.name.clone();
        
        // Index tickers
        for ticker in &record.tickers {
            self.ticker_to_entity.insert(
                ticker.symbol.clone(),
                name.clone()
            );
        }
        
        // Store the entity
        record.last_updated_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        
        self.entities.insert(name.clone(), record);
        info!("Registered entity: {}", name);
    }

    /// Add an alias for an entity
    pub fn add_alias(&self, alias: String, canonical_name: String) {
        self.aliases.insert(alias.to_lowercase(), canonical_name);
    }

    /// Look up an entity by name
    pub fn lookup_by_name(&self, name: &str) -> Option<EntityRecord> {
        // Try exact match first
        if let Some(record) = self.entities.get(name) {
            self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Some(record.value().clone());
        }

        // Try case-insensitive match
        let name_lower = name.to_lowercase();
        if let Some(record) = self.entities.iter().find(|e| e.key().to_lowercase() == name_lower) {
            self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Some(record.value().clone());
        }

        // Try alias lookup
        if let Some(canonical) = self.aliases.get(&name_lower) {
            if let Some(record) = self.entities.get(canonical.as_str()) {
                self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some(record.value().clone());
            }
        }

        self.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        None
    }

    /// Look up entity by ticker symbol
    pub fn lookup_by_ticker(&self, ticker: &str) -> Option<EntityRecord> {
        if let Some(entity_name) = self.ticker_to_entity.get(ticker) {
            self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return self.entities.get(entity_name.as_str()).map(|r| r.value().clone());
        }

        self.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        None
    }

    /// Extract and map entities from text
    pub fn extract_entities(&self, text: &str) -> Vec<(String, EntityRecord)> {
        let mut results = Vec::new();
        
        // Simple keyword-based extraction (in production, this would use NER model)
        for entry in self.entities.iter() {
            let entity_name = entry.key();
            if text.contains(entity_name.as_str()) {
                results.push((entity_name.clone(), entry.value().clone()));
            }
            
            // Also check aliases
            for alias_entry in self.aliases.iter() {
                if alias_entry.value().as_str() == entity_name.as_str() 
                    && text.contains(alias_entry.key().as_str()) 
                {
                    results.push((alias_entry.key().clone(), entry.value().clone()));
                }
            }
        }
        
        results
    }

    /// Get statistics
    pub fn get_stats(&self) -> MapperStats {
        MapperStats {
            total_entities: self.entities.len(),
            total_aliases: self.aliases.len(),
            total_ticker_mappings: self.ticker_to_entity.len(),
            hits: self.hits.load(std::sync::atomic::Ordering::Relaxed),
            misses: self.misses.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Clear all registered entities
    pub fn clear(&self) {
        self.entities.clear();
        self.aliases.clear();
        self.ticker_to_entity.clear();
    }
}

/// Statistics for the entity mapper
#[derive(Debug, Clone)]
pub struct MapperStats {
    pub total_entities: usize,
    pub total_aliases: usize,
    pub total_ticker_mappings: usize,
    pub hits: u64,
    pub misses: u64,
}

impl MapperStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Pre-built entity database for common financial entities
pub mod predefined_entities {
    use super::*;

    /// Initialize mapper with common financial entities
    pub fn initialize_default_mapper(mapper: &EntityTickerMapper) {
        // Tech Companies
        let mut apple = EntityRecord::new("Apple".to_string(), EntityType::Company);
        apple.add_ticker(TickerMapping {
            symbol: "AAPL".to_string(),
            exchange: "NASDAQ".to_string(),
            asset_class: AssetClass::Equity,
            confidence: 1.0,
            alternatives: vec!["AAPL.O".to_string()],
        });
        mapper.register_entity(apple);
        mapper.add_alias("AAPL".to_string(), "Apple".to_string());
        mapper.add_alias("Apple Inc".to_string(), "Apple".to_string());

        let mut microsoft = EntityRecord::new("Microsoft".to_string(), EntityType::Company);
        microsoft.add_ticker(TickerMapping {
            symbol: "MSFT".to_string(),
            exchange: "NASDAQ".to_string(),
            asset_class: AssetClass::Equity,
            confidence: 1.0,
            alternatives: vec!["MSFT.O".to_string()],
        });
        mapper.register_entity(microsoft);
        mapper.add_alias("MSFT".to_string(), "Microsoft".to_string());

        // Organizations
        let mut opec = EntityRecord::new("OPEC".to_string(), EntityType::Organization);
        opec.add_ticker(TickerMapping {
            symbol: "USO".to_string(),
            exchange: "NYSE".to_string(),
            asset_class: AssetClass::Commodity,
            confidence: 0.9,
            alternatives: vec!["CL".to_string(), "BZ".to_string()],
        });
        mapper.register_entity(opec);
        mapper.add_alias("Organization of Petroleum Exporting Countries".to_string(), "OPEC".to_string());

        let mut fed = EntityRecord::new("Federal Reserve".to_string(), EntityType::Organization);
        fed.add_ticker(TickerMapping {
            symbol: "^TNX".to_string(),
            exchange: "CBOE".to_string(),
            asset_class: AssetClass::FixedIncome,
            confidence: 0.8,
            alternatives: vec!["TLT".to_string(), "IEF".to_string()],
        });
        mapper.register_entity(fed);
        mapper.add_alias("Fed".to_string(), "Federal Reserve".to_string());
        mapper.add_alias("FOMC".to_string(), "Federal Reserve".to_string());

        // People
        let mut powell = EntityRecord::new("Jerome Powell".to_string(), EntityType::Person);
        powell.related_entities.push("Federal Reserve".to_string());
        mapper.register_entity(powell);
        mapper.add_alias("Powell".to_string(), "Jerome Powell".to_string());

        // Commodities
        let mut crude = EntityRecord::new("Crude Oil".to_string(), EntityType::Commodity);
        crude.add_ticker(TickerMapping {
            symbol: "CL".to_string(),
            exchange: "NYMEX".to_string(),
            asset_class: AssetClass::Commodity,
            confidence: 1.0,
            alternatives: vec!["USO".to_string(), "UCO".to_string()],
        });
        mapper.register_entity(crude);
        mapper.add_alias("WTI".to_string(), "Crude Oil".to_string());
        mapper.add_alias("West Texas Intermediate".to_string(), "Crude Oil".to_string());

        let mut gold = EntityRecord::new("Gold".to_string(), EntityType::Commodity);
        gold.add_ticker(TickerMapping {
            symbol: "GC".to_string(),
            exchange: "COMEX".to_string(),
            asset_class: AssetClass::Commodity,
            confidence: 1.0,
            alternatives: vec!["GLD".to_string(), "GOLD".to_string()],
        });
        mapper.register_entity(gold);

        info!("Initialized default entity database");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_registration() {
        let mapper = EntityTickerMapper::new(EntityMapperConfig::default());
        
        let mut apple = EntityRecord::new("Apple".to_string(), EntityType::Company);
        apple.add_ticker(TickerMapping {
            symbol: "AAPL".to_string(),
            exchange: "NASDAQ".to_string(),
            asset_class: AssetClass::Equity,
            confidence: 1.0,
            alternatives: vec![],
        });
        
        mapper.register_entity(apple);
        
        let result = mapper.lookup_by_name("Apple");
        assert!(result.is_some());
        assert_eq!(result.unwrap().tickers[0].symbol, "AAPL");
    }

    #[test]
    fn test_alias_lookup() {
        let mapper = EntityTickerMapper::new(EntityMapperConfig::default());
        
        let mut msft = EntityRecord::new("Microsoft".to_string(), EntityType::Company);
        msft.add_ticker(TickerMapping {
            symbol: "MSFT".to_string(),
            exchange: "NASDAQ".to_string(),
            asset_class: AssetClass::Equity,
            confidence: 1.0,
            alternatives: vec![],
        });
        
        mapper.register_entity(msft);
        mapper.add_alias("MSFT".to_string(), "Microsoft".to_string());
        
        let result = mapper.lookup_by_name("MSFT");
        assert!(result.is_some());
    }

    #[test]
    fn test_ticker_lookup() {
        let mapper = EntityTickerMapper::new(EntityMapperConfig::default());
        
        let mut apple = EntityRecord::new("Apple".to_string(), EntityType::Company);
        apple.add_ticker(TickerMapping {
            symbol: "AAPL".to_string(),
            exchange: "NASDAQ".to_string(),
            asset_class: AssetClass::Equity,
            confidence: 1.0,
            alternatives: vec![],
        });
        mapper.register_entity(apple);
        
        let result = mapper.lookup_by_ticker("AAPL");
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "Apple");
    }

    #[test]
    fn test_entity_extraction() {
        let mapper = EntityTickerMapper::new(EntityMapperConfig::default());
        predefined_entities::initialize_default_mapper(&mapper);
        
        let text = "Apple announced new products while Microsoft reported earnings";
        let entities = mapper.extract_entities(text);
        
        assert!(entities.len() >= 2);
    }
}
