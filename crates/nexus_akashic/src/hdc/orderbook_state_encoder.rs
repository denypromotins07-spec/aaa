//! Order Book State Encoder for Hyper-Dimensional Computing
//! Encodes the entire L3 order book state into a single hyper-dimensional vector

use crate::hdc::bipolar_vector_generator::{BipolarVector, BipolarVectorGenerator, HDC_DIMENSION, BipolarVectorError};
use crate::hdc::simd_binding_bundling::{bind_vectors, bundle_vectors, permute_vector};

/// Represents a single price level in the order book
#[derive(Debug, Clone)]
pub struct PriceLevel {
    pub price: u64,      // Price in fixed-point (e.g., cents)
    pub size: u64,       // Size in base units
    pub order_count: u32, // Number of orders at this level
    pub side: Side,
}

/// Order book side
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Bid = 0,
    Ask = 1,
}

/// Complete order book snapshot
#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub timestamp_ns: u64,
    pub symbol_id: u64,
}

/// Encoder for converting order book states to hyper-dimensional vectors
pub struct OrderBookStateEncoder {
    generator: BipolarVectorGenerator,
    /// Pre-generated vectors for common price levels
    price_level_cache: Vec<BipolarVector>,
    /// Pre-generated vectors for size buckets
    size_bucket_cache: Vec<BipolarVector>,
    /// Maximum price level index cached
    max_price_index: usize,
    /// Maximum size bucket index cached
    max_size_bucket: usize,
}

impl OrderBookStateEncoder {
    /// Create a new encoder with pre-generated caches
    pub fn new(seed: u64, max_price_levels: usize, max_size_buckets: usize) -> Self {
        let mut generator = BipolarVectorGenerator::new(seed);
        
        // Pre-generate price level vectors
        let mut price_level_cache = Vec::with_capacity(max_price_levels);
        for _ in 0..max_price_levels {
            price_level_cache.push(generator.generate().expect("Failed to generate price vector"));
        }
        
        // Pre-generate size bucket vectors
        let mut size_bucket_cache = Vec::with_capacity(max_size_buckets);
        for _ in 0..max_size_buckets {
            size_bucket_cache.push(generator.generate().expect("Failed to generate size vector"));
        }
        
        Self {
            generator,
            price_level_cache,
            size_bucket_cache,
            max_price_index: max_price_levels,
            max_size_bucket: max_size_buckets,
        }
    }

    /// Encode a single price level to a hyper-dimensional vector
    pub fn encode_price_level(&self, level: &PriceLevel) -> Result<BipolarVector, BipolarVectorError> {
        // Get price vector from cache or generate on-the-fly
        let price_idx = (level.price % self.max_price_index as u64) as usize;
        let price_vec = if price_idx < self.price_level_cache.len() {
            &self.price_level_cache[price_idx]
        } else {
            // Generate deterministically based on price
            return BipolarVector::from_seed(level.price.wrapping_mul(6364136223846793005));
        };

        // Get size bucket vector
        let size_bucket = self.size_to_bucket(level.size);
        let size_vec = if size_bucket < self.size_bucket_cache.len() {
            &self.size_bucket_cache[size_bucket]
        } else {
            return BipolarVector::from_seed(level.size.wrapping_mul(6364136223846793005));
        };

        // Bind price and size
        let bound = bind_vectors(price_vec, size_vec)?;

        // Permute based on side to differentiate bid/ask
        let shift = if level.side == Side::Bid { 0 } else { HDC_DIMENSION / 2 };
        permute_vector(&bound, shift)
    }

    /// Convert size to bucket index (logarithmic scaling)
    fn size_to_bucket(&self, size: u64) -> usize {
        if size == 0 {
            return 0;
        }
        // Logarithmic bucketing: each bucket represents 2x the previous
        let log_size = 64 - size.leading_zeros() as usize;
        log_size.min(self.max_size_bucket - 1)
    }

    /// Encode an entire order book snapshot into a single hyper-dimensional vector
    pub fn encode_orderbook(&self, snapshot: &OrderBookSnapshot) -> Result<BipolarVector, BipolarVectorError> {
        let mut all_levels: Vec<&BipolarVector> = Vec::new();
        let mut temp_vectors: Vec<BipolarVector> = Vec::new();

        // Encode all bid levels
        for level in &snapshot.bids {
            let encoded = self.encode_price_level(level)?;
            temp_vectors.push(encoded);
        }

        // Encode all ask levels
        for level in &snapshot.asks {
            let encoded = self.encode_price_level(level)?;
            temp_vectors.push(encoded);
        }

        // Add timestamp encoding
        let timestamp_vec = BipolarVector::from_seed(snapshot.timestamp_ns)?;
        temp_vectors.push(timestamp_vec);

        // Add symbol ID encoding
        let symbol_vec = BipolarVector::from_seed(snapshot.symbol_id.wrapping_mul(14695981039346656037))?;
        temp_vectors.push(symbol_vec);

        // Create references for bundling
        all_levels.reserve(temp_vectors.len());
        for v in &temp_vectors {
            all_levels.push(v);
        }

        // Bundle all components into a single vector
        bundle_vectors(&all_levels)
    }

    /// Encode order book imbalance (bid vs ask pressure)
    pub fn encode_imbalance(&self, snapshot: &OrderBookSnapshot) -> Result<BipolarVector, BipolarVectorError> {
        let mut bid_vectors: Vec<&BipolarVector> = Vec::new();
        let mut ask_vectors: Vec<&BipolarVector> = Vec::new();
        let mut temp_vectors: Vec<BipolarVector> = Vec::new();

        // Encode bid side
        for level in &snapshot.bids {
            let encoded = self.encode_price_level(level)?;
            temp_vectors.push(encoded);
        }
        
        let bid_count = snapshot.bids.len();
        for i in 0..bid_count {
            bid_vectors.push(&temp_vectors[i]);
        }

        // Encode ask side  
        let ask_start = bid_count;
        for level in &snapshot.asks {
            let encoded = self.encode_price_level(level)?;
            temp_vectors.push(encoded);
        }
        
        let ask_count = snapshot.asks.len();
        for i in 0..ask_count {
            ask_vectors.push(&temp_vectors[ask_start + i]);
        }

        // Bundle each side separately
        let bundled_bids = if !bid_vectors.is_empty() {
            bundle_vectors(&bid_vectors)?
        } else {
            BipolarVector::from_seed(0)?
        };

        let bundled_asks = if !ask_vectors.is_empty() {
            bundle_vectors(&ask_vectors)?
        } else {
            BipolarVector::from_seed(0)?
        };

        // Bind bid and ask bundles to create imbalance representation
        bind_vectors(&bundled_bids, &bundled_asks)
    }

    /// Decode approximate price level from a hyper-dimensional vector
    /// Returns the most likely price bucket index
    pub fn decode_price_bucket(&self, vector: &BipolarVector) -> Option<usize> {
        let mut best_match_idx = None;
        let mut best_similarity = f64::NEG_INFINITY;

        for (idx, cached) in self.price_level_cache.iter().enumerate() {
            let sim = vector.cosine_similarity(cached);
            if sim > best_similarity {
                best_similarity = sim;
                best_match_idx = Some(idx);
            }
        }

        // Only return if similarity is above threshold
        if best_similarity > 0.3 {
            best_match_idx
        } else {
            None
        }
    }

    /// Calculate similarity between two order book states
    pub fn orderbook_similarity(
        &self,
        snapshot1: &OrderBookSnapshot,
        snapshot2: &OrderBookSnapshot,
    ) -> Result<f64, BipolarVectorError> {
        let vec1 = self.encode_orderbook(snapshot1)?;
        let vec2 = self.encode_orderbook(snapshot2)?;
        Ok(vec1.cosine_similarity(&vec2))
    }

    /// Detect regime change by comparing consecutive order book encodings
    pub fn detect_regime_change(
        &self,
        snapshots: &[OrderBookSnapshot],
        threshold: f64,
    ) -> Result<Vec<usize>, BipolarVectorError> {
        if snapshots.len() < 2 {
            return Ok(Vec::new());
        }

        let mut changes = Vec::new();
        let mut prev_vec = self.encode_orderbook(&snapshots[0])?;

        for (i, snapshot) in snapshots.iter().skip(1).enumerate() {
            let curr_vec = self.encode_orderbook(snapshot)?;
            let sim = prev_vec.cosine_similarity(&curr_vec);

            if sim < threshold {
                changes.push(i);
            }

            prev_vec = curr_vec;
        }

        Ok(changes)
    }
}

/// Streaming encoder for incremental order book updates
pub struct StreamingOrderBookEncoder {
    base_encoder: OrderBookStateEncoder,
    current_state: Option<BipolarVector>,
    update_count: usize,
}

impl StreamingOrderBookEncoder {
    pub fn new(seed: u64, max_price_levels: usize, max_size_buckets: usize) -> Self {
        Self {
            base_encoder: OrderBookStateEncoder::new(seed, max_price_levels, max_size_buckets),
            current_state: None,
            update_count: 0,
        }
    }

    /// Apply an incremental update to the current state
    pub fn apply_update(&mut self, level: &PriceLevel) -> Result<(), BipolarVectorError> {
        let level_vec = self.base_encoder.encode_price_level(level)?;
        
        match &self.current_state {
            None => {
                self.current_state = Some(level_vec);
            }
            Some(current) => {
                // Bind the new level info with current state
                let updated = bind_vectors(current, &level_vec)?;
                self.current_state = Some(updated);
            }
        }
        
        self.update_count += 1;
        
        // Apply decay periodically to prevent orthogonality degradation
        if self.update_count % 50 == 0 {
            if let Some(ref mut state) = self.current_state {
                // Note: In production, we'd have a proper decay mechanism here
                // For now, we just track the count
            }
        }
        
        Ok(())
    }

    /// Get the current encoded state
    pub fn get_current_state(&self) -> Option<&BipolarVector> {
        self.current_state.as_ref()
    }

    /// Reset the encoder state
    pub fn reset(&mut self) {
        self.current_state = None;
        self.update_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_snapshot() -> OrderBookSnapshot {
        OrderBookSnapshot {
            bids: vec![
                PriceLevel { price: 10000, size: 100, order_count: 5, side: Side::Bid },
                PriceLevel { price: 9999, size: 200, order_count: 3, side: Side::Bid },
                PriceLevel { price: 9998, size: 150, order_count: 7, side: Side::Bid },
            ],
            asks: vec![
                PriceLevel { price: 10001, size: 120, order_count: 4, side: Side::Ask },
                PriceLevel { price: 10002, size: 180, order_count: 6, side: Side::Ask },
                PriceLevel { price: 10003, size: 90, order_count: 2, side: Side::Ask },
            ],
            timestamp_ns: 1234567890,
            symbol_id: 42,
        }
    }

    #[test]
    fn test_price_level_encoding() {
        let encoder = OrderBookStateEncoder::new(42, 1000, 32);
        let level = PriceLevel {
            price: 10000,
            size: 100,
            order_count: 5,
            side: Side::Bid,
        };

        let encoded = encoder.encode_price_level(&level).unwrap();
        assert_eq!(encoded.as_bits().len(), HDC_DIMENSION / 64);
    }

    #[test]
    fn test_orderbook_encoding() {
        let encoder = OrderBookStateEncoder::new(42, 1000, 32);
        let snapshot = create_test_snapshot();

        let encoded = encoder.encode_orderbook(&snapshot).unwrap();
        assert_eq!(encoded.as_bits().len(), HDC_DIMENSION / 64);
    }

    #[test]
    fn test_orderbook_similarity() {
        let encoder = OrderBookStateEncoder::new(42, 1000, 32);
        let snapshot1 = create_test_snapshot();
        
        let mut snapshot2 = snapshot1.clone();
        snapshot2.timestamp_ns += 1000000; // Slightly different timestamp

        let sim = encoder.orderbook_similarity(&snapshot1, &snapshot2).unwrap();
        
        // Similar snapshots should have high similarity
        assert!(sim > 0.5, "Similarity too low: {}", sim);
    }

    #[test]
    fn test_regime_change_detection() {
        let encoder = OrderBookStateEncoder::new(42, 1000, 32);
        
        let mut snapshots = Vec::new();
        let base = create_test_snapshot();
        
        // Create similar snapshots
        for i in 0..5 {
            let mut snap = base.clone();
            snap.timestamp_ns += i * 1000;
            snapshots.push(snap);
        }
        
        // Create a very different snapshot (regime change)
        let mut different = base.clone();
        different.bids = vec![
            PriceLevel { price: 5000, size: 1000, order_count: 50, side: Side::Bid },
        ];
        different.asks = vec![
            PriceLevel { price: 5001, size: 1000, order_count: 50, side: Side::Ask },
        ];
        snapshots.push(different);
        
        let changes = encoder.detect_regime_change(&snapshots, 0.7).unwrap();
        
        // Should detect the regime change at index 4 (the different snapshot)
        assert!(!changes.is_empty(), "Should detect regime change");
    }

    #[test]
    fn test_streaming_encoder() {
        let mut encoder = StreamingOrderBookEncoder::new(42, 1000, 32);
        
        let level1 = PriceLevel { price: 10000, size: 100, order_count: 5, side: Side::Bid };
        let level2 = PriceLevel { price: 10001, size: 150, order_count: 3, side: Side::Ask };
        
        encoder.apply_update(&level1).unwrap();
        encoder.apply_update(&level2).unwrap();
        
        let state = encoder.get_current_state();
        assert!(state.is_some());
        assert_eq!(state.unwrap().as_bits().len(), HDC_DIMENSION / 64);
        
        encoder.reset();
        assert!(encoder.get_current_state().is_none());
    }

    #[test]
    fn test_bid_ask_differentiation() {
        let encoder = OrderBookStateEncoder::new(42, 1000, 32);
        
        let bid_level = PriceLevel { price: 10000, size: 100, order_count: 5, side: Side::Bid };
        let ask_level = PriceLevel { price: 10000, size: 100, order_count: 5, side: Side::Ask };
        
        let bid_vec = encoder.encode_price_level(&bid_level).unwrap();
        let ask_vec = encoder.encode_price_level(&ask_level).unwrap();
        
        // Bid and ask vectors should be distinguishable
        let sim = bid_vec.cosine_similarity(&ask_vec);
        assert!(sim < 0.8, "Bid/ask vectors not sufficiently differentiated: {}", sim);
    }
}
