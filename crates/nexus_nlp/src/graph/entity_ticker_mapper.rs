//! Entity to Ticker Mapper
//! 
//! Maps extracted entities (e.g., "Jerome Powell", "Apple", "OPEC") 
//! to specific financial tickers and asset classes.

use std::collections::HashMap;
use std::sync::Arc;
use dashmap::DashMap;

/// Entity types for financial NLP
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityType {
    Person,
    Company,
    Organization,
    Commodity,
    Currency,
    Cryptocurrency,
    EconomicIndicator,
    Country,
    CentralBank,
}

/// Asset class categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetClass {
    Equity,
    FixedIncome,
    Commodity,
    Currency,
    Cryptocurrency,
    Derivative,
}

/// Mapped entity with ticker information
#[derive(Debug, Clone)]
pub struct MappedEntity {
    pub entity_name: String,
    pub entity_type: EntityType,
    pub primary_ticker: String,
    pub related_tickers: Vec<String>,
    pub asset_class: AssetClass,
    pub confidence: f32,
}

/// Entity to ticker mapping database
pub struct EntityTickerMapper {
    // Lock-free concurrent map for fast lookups
    entity_map: DashMap<String, MappedEntity>,
    alias_map: DashMap<String, String>, // alias -> canonical name
}

impl EntityTickerMapper {
    /// Create a new entity ticker mapper with default mappings
    pub fn new() -> Self {
        let mapper = Self {
            entity_map: DashMap::new(),
            alias_map: DashMap::new(),
        };
        
        // Initialize with common financial entities
        mapper.initialize_defaults();
        
        mapper
    }

    /// Initialize with default financial entity mappings
    fn initialize_defaults(&self) {
        // Central Bank figures
        self.add_entity(MappedEntity {
            entity_name: "Jerome Powell".to_string(),
            entity_type: EntityType::Person,
            primary_ticker: "USD".to_string(),
            related_tickers: vec!["DXY".to_string(), "TLT".to_string()],
            asset_class: AssetClass::Currency,
            confidence: 1.0,
        });

        // Companies
        self.add_entity(MappedEntity {
            entity_name: "Apple".to_string(),
            entity_type: EntityType::Company,
            primary_ticker: "AAPL".to_string(),
            related_tickers: vec!["QQQ".to_string(), "XLK".to_string()],
            asset_class: AssetClass::Equity,
            confidence: 1.0,
        });

        self.add_entity(MappedEntity {
            entity_name: "Tesla".to_string(),
            entity_type: EntityType::Company,
            primary_ticker: "TSLA".to_string(),
            related_tickers: vec!["QQQ".to_string()],
            asset_class: AssetClass::Equity,
            confidence: 1.0,
        });

        // Organizations
        self.add_entity(MappedEntity {
            entity_name: "OPEC".to_string(),
            entity_type: EntityType::Organization,
            primary_ticker: "USO".to_string(),
            related_tickers: vec!["XLE".to_string(), "HAL".to_string(), "CL=F".to_string()],
            asset_class: AssetClass::Commodity,
            confidence: 1.0,
        });

        self.add_entity(MappedEntity {
            entity_name: "Federal Reserve".to_string(),
            entity_type: EntityType::CentralBank,
            primary_ticker: "DXY".to_string(),
            related_tickers: vec!["TLT".to_string(), "GLD".to_string()],
            asset_class: AssetClass::FixedIncome,
            confidence: 1.0,
        });

        // Commodities
        self.add_entity(MappedEntity {
            entity_name: "Gold".to_string(),
            entity_type: EntityType::Commodity,
            primary_ticker: "GLD".to_string(),
            related_tickers: vec!["GOLD".to_string(), "GC=F".to_string()],
            asset_class: AssetClass::Commodity,
            confidence: 1.0,
        });

        self.add_entity(MappedEntity {
            entity_name: "Bitcoin".to_string(),
            entity_type: EntityType::Cryptocurrency,
            primary_ticker: "BTCUSD".to_string(),
            related_tickers: vec!["GBTC".to_string(), "COIN".to_string()],
            asset_class: AssetClass::Cryptocurrency,
            confidence: 1.0,
        });

        // Add aliases
        self.add_alias("Fed", "Federal Reserve");
        self.add_alias("Powell", "Jerome Powell");
        self.add_alias("AAPL", "Apple");
        self.add_alias("TSLA", "Tesla");
        self.add_alias("Crude Oil", "OPEC");
        self.add_alias("BTC", "Bitcoin");
        self.add_alias("Ether", "Ethereum");
    }

    /// Add an entity to the mapping
    pub fn add_entity(&self, entity: MappedEntity) {
        let name = entity.entity_name.clone();
        self.entity_map.insert(name, entity);
    }

    /// Add an alias for an entity
    pub fn add_alias(&self, alias: &str, canonical: &str) {
        self.alias_map.insert(alias.to_string(), canonical.to_string());
    }

    /// Look up an entity by name
    pub fn lookup(&self, name: &str) -> Option<MappedEntity> {
        // First try direct lookup
        if let Some(entry) = self.entity_map.get(name) {
            return Some(entry.value().clone());
        }

        // Try alias lookup
        if let Some(canonical) = self.alias_map.get(name) {
            return self.entity_map.get(&canonical).map(|e| e.value().clone());
        }

        // Try case-insensitive lookup
        for entry in self.entity_map.iter() {
            if entry.key().eq_ignore_ascii_case(name) {
                return Some(entry.value().clone());
            }
        }

        None
    }

    /// Batch lookup multiple entities
    pub fn batch_lookup(&self, names: &[&str]) -> Vec<Option<MappedEntity>> {
        names.iter().map(|&name| self.lookup(name)).collect()
    }

    /// Get all entities of a specific type
    pub fn get_by_type(&self, entity_type: EntityType) -> Vec<MappedEntity> {
        self.entity_map
            .iter()
            .filter(|entry| entry.value().entity_type == entity_type)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get entity count
    pub fn entity_count(&self) -> usize {
        self.entity_map.len()
    }

    /// Clear all mappings (for testing or reload)
    pub fn clear(&self) {
        self.entity_map.clear();
        self.alias_map.clear();
    }
}

impl Default for EntityTickerMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Named Entity Recognition result
#[derive(Debug, Clone)]
pub struct NerResult {
    pub text: String,
    pub entity: String,
    pub entity_type: EntityType,
    pub start_pos: usize,
    pub end_pos: usize,
    pub confidence: f32,
}

/// Simple keyword-based NER (placeholder for ML-based NER)
pub fn simple_ner(text: &str, mapper: &EntityTickerMapper) -> Vec<NerResult> {
    let mut results = Vec::new();
    
    // Check for known entities in text
    for entry in mapper.entity_map.iter() {
        let entity_name = entry.key();
        if let Some(pos) = text.find(entity_name.as_str()) {
            results.push(NerResult {
                text: text.to_string(),
                entity: entity_name.clone(),
                entity_type: entry.value().entity_type,
                start_pos: pos,
                end_pos: pos + entity_name.len(),
                confidence: entry.value().confidence,
            });
        }
    }

    // Sort by position
    results.sort_by_key(|r| r.start_pos);
    
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_lookup() {
        let mapper = EntityTickerMapper::new();
        
        let apple = mapper.lookup("Apple").unwrap();
        assert_eq!(apple.primary_ticker, "AAPL");
        assert_eq!(apple.asset_class, AssetClass::Equity);
        
        let fed = mapper.lookup("Fed").unwrap();
        assert_eq!(fed.primary_ticker, "DXY");
    }

    #[test]
    fn test_ner_extraction() {
        let mapper = EntityTickerMapper::new();
        let text = "Apple stock rises as Fed announces rate decision";
        
        let results = simple_ner(text, &mapper);
        
        assert!(results.iter().any(|r| r.entity == "Apple"));
        assert!(results.iter().any(|r| r.entity == "Federal Reserve"));
    }

    #[test]
    fn test_batch_lookup() {
        let mapper = EntityTickerMapper::new();
        let names = ["Apple", "Bitcoin", "Unknown"];
        
        let results = mapper.batch_lookup(&names);
        
        assert!(results[0].is_some());
        assert!(results[1].is_some());
        assert!(results[2].is_none());
    }
}
