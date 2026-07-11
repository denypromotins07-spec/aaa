//! Cosmic String Storage Engine
//! 
//! Implements data encoding into topological defects of spacetime (cosmic strings
//! and magnetic monopoles) for surviving proton decay and black hole evaporation.

/// Topological defect types for storage
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DefectType {
    /// Cosmic string - 1D topological defect
    CosmicString,
    /// Magnetic monopole - 0D point defect
    MagneticMonopole,
    /// Domain wall - 2D sheet defect
    DomainWall,
    /// Texture - non-local defect
    Texture,
}

/// A single bit encoded in a cosmic string
#[derive(Debug, Clone, Copy)]
pub struct CosmicStringBit {
    /// Winding number (+1 or -1 for binary)
    pub winding: i8,
    /// Position along string [m]
    pub position: f64,
    /// String tension μ [kg/m]
    pub tension: f64,
}

impl CosmicStringBit {
    /// Create a new bit from a boolean value
    pub fn from_bool(value: bool, position: f64, tension: f64) -> Self {
        Self {
            winding: if value { 1 } else { -1 },
            position,
            tension,
        }
    }
    
    /// Decode to boolean
    pub fn to_bool(&self) -> bool {
        self.winding > 0
    }
}

/// Data block stored in topological defects
#[derive(Debug, Clone)]
pub struct TopologicalDataBlock {
    /// Unique identifier
    pub id: u64,
    /// Type of defect used
    pub defect_type: DefectType,
    /// Encoded bits
    pub bits: Vec<CosmicStringBit>,
    /// Error correction redundancy (0 to 1)
    pub redundancy: f64,
    /// Creation epoch (cosmological time)
    pub creation_epoch: f64,
}

impl TopologicalDataBlock {
    /// Encode a byte array into cosmic string bits
    /// 
    /// # Arguments
    /// * `id` - Block identifier
    /// * `data` - Data to encode
    /// * `tension` - String tension
    /// * `redundancy` - Error correction overhead
    /// 
    /// # Returns
    /// * `Result<Self, &'static str>` - Encoded block
    pub fn encode_bytes(
        id: u64,
        data: &[u8],
        tension: f64,
        redundancy: f64,
    ) -> Result<Self, &'static str> {
        if tension <= 0.0 {
            return Err("Tension must be positive");
        }
        if redundancy < 0.0 || redundancy > 0.99 {
            return Err("Redundancy must be in [0, 0.99]");
        }
        
        let mut bits = Vec::with_capacity(data.len() * 8);
        let bit_spacing = 1e-35; // Planck length spacing
        
        for (byte_idx, &byte) in data.iter().enumerate() {
            for bit_idx in 0..8 {
                let value = (byte >> bit_idx) & 1 == 1;
                let position = (byte_idx * 8 + bit_idx) as f64 * bit_spacing;
                bits.push(CosmicStringBit::from_bool(value, position, tension));
            }
        }
        
        // Add redundancy bits (simple parity for now)
        if redundancy > 0.0 {
            let n_redundancy = (bits.len() as f64 * redundancy).ceil() as usize;
            for i in 0..n_redundancy {
                // Parity bit over a window
                let window_start = i * (bits.len() / n_redundancy.max(1));
                let window_end = (window_start + bits.len() / n_redundancy.max(1)).min(bits.len());
                
                let parity = bits[window_start..window_end]
                    .iter()
                    .map(|b| b.winding.abs() as u8)
                    .sum::<u8>() % 2;
                
                bits.push(CosmicStringBit::from_bool(parity == 1, 
                    bits.len() as f64 * bit_spacing, tension));
            }
        }
        
        Ok(Self {
            id,
            defect_type: DefectType::CosmicString,
            bits,
            redundancy,
            creation_epoch: 0.0, // Would be actual cosmological time
        })
    }
    
    /// Decode back to bytes
    pub fn decode_bytes(&self) -> Result<Vec<u8>, &'static str> {
        if self.bits.is_empty() {
            return Ok(Vec::new());
        }
        
        // Calculate number of data bits (excluding redundancy)
        let total_bits = self.bits.len();
        let n_redundancy = (total_bits as f64 * self.redundancy).ceil() as usize;
        let n_data_bits = total_bits - n_redundancy;
        let n_bytes = (n_data_bits + 7) / 8;
        
        let mut bytes = vec![0u8; n_bytes];
        
        for (bit_idx, bit) in self.bits.iter().take(n_data_bits).enumerate() {
            let byte_idx = bit_idx / 8;
            let bit_pos = bit_idx % 8;
            if bit.to_bool() {
                bytes[byte_idx] |= 1 << bit_pos;
            }
        }
        
        // TODO: Verify and correct using redundancy bits
        
        Ok(bytes)
    }
    
    /// Get storage density [bits/m]
    pub fn storage_density(&self) -> f64 {
        if self.bits.is_empty() {
            return 0.0;
        }
        
        let first_pos = self.bits.first().map(|b| b.position).unwrap_or(0.0);
        let last_pos = self.bits.last().map(|b| b.position).unwrap_or(0.0);
        let total_length = last_pos - first_pos;
        
        if total_length <= 0.0 {
            return f64::INFINITY; // All bits at same position (quantum limit)
        }
        
        self.bits.len() as f64 / total_length
    }
    
    /// Estimate lifetime before quantum tunneling destroys the encoding
    /// 
    /// For cosmic strings, this is essentially infinite (> age of universe)
    pub fn estimated_lifetime(&self) -> f64 {
        // Cosmic string stability depends on tension
        // Higher tension = more stable
        let avg_tension = self.bits.iter()
            .map(|b| b.tension)
            .sum::<f64>() / self.bits.len().max(1) as f64;
        
        // Tunneling rate ~ exp(-μ/ℏ) which is essentially zero for GUT-scale strings
        // Lifetime effectively infinite
        f64::INFINITY
    }
}

/// Cosmic string storage manager
#[derive(Debug, Clone)]
pub struct CosmicStringStorage {
    /// Stored data blocks
    blocks: Vec<TopologicalDataBlock>,
    /// Block counter
    block_counter: u64,
    /// Default string tension [kg/m]
    default_tension: f64,
    /// Default redundancy
    default_redundancy: f64,
}

impl Default for CosmicStringStorage {
    fn default() -> Self {
        Self {
            blocks: Vec::new(),
            block_counter: 0,
            // GUT-scale cosmic string tension ~ 10^22 kg/m
            default_tension: 1e22,
            default_redundancy: 0.1, // 10% overhead
        }
    }
}

impl CosmicStringStorage {
    /// Store data in a new cosmic string
    /// 
    /// # Arguments
    /// * `data` - Data to store
    /// 
    /// # Returns
    /// * `Result<u64, &'static str>` - Block ID
    pub fn store(&mut self, data: &[u8]) -> Result<u64, &'static str> {
        let block = TopologicalDataBlock::encode_bytes(
            self.block_counter,
            data,
            self.default_tension,
            self.default_redundancy,
        )?;
        
        let id = block.id;
        self.blocks.push(block);
        self.block_counter += 1;
        
        Ok(id)
    }
    
    /// Retrieve stored data by ID
    /// 
    /// # Arguments
    /// * `id` - Block ID
    /// 
    /// # Returns
    /// * `Result<Vec<u8>, &'static str>` - Retrieved data
    pub fn retrieve(&self, id: u64) -> Result<Vec<u8>, &'static str> {
        let block = self.blocks.iter()
            .find(|b| b.id == id)
            .ok_or("Block not found")?;
        
        block.decode_bytes()
    }
    
    /// Verify integrity of a stored block
    /// 
    /// # Arguments
    /// * `id` - Block ID
    /// 
    /// # Returns
    /// * `Result<bool, &'static str>` - True if valid
    pub fn verify_integrity(&self, id: u64) -> Result<bool, &'static str> {
        let block = self.blocks.iter()
            .find(|b| b.id == id)
            .ok_or("Block not found")?;
        
        // Check redundancy bits
        if block.redundancy <= 0.0 {
            return Ok(true); // No redundancy to check
        }
        
        // Simple parity check
        let total_bits = block.bits.len();
        let n_redundancy = (total_bits as f64 * block.redundancy).ceil() as usize;
        let n_data_bits = total_bits - n_redundancy;
        
        for i in 0..n_redundancy {
            let window_start = i * (n_data_bits / n_redundancy.max(1));
            let window_end = (window_start + n_data_bits / n_redundancy.max(1)).min(n_data_bits);
            
            let expected_parity = block.bits[window_start..window_end]
                .iter()
                .map(|b| b.winding.abs() as u8)
                .sum::<u8>() % 2;
            
            let parity_bit_idx = n_data_bits + i;
            if parity_bit_idx < block.bits.len() {
                let actual_parity = if block.bits[parity_bit_idx].to_bool() { 1 } else { 0 };
                if expected_parity != actual_parity {
                    return Ok(false);
                }
            }
        }
        
        Ok(true)
    }
    
    /// Get statistics about stored data
    pub fn get_statistics(&self) -> StorageStats {
        let total_bits: usize = self.blocks.iter().map(|b| b.bits.len()).sum();
        let total_bytes = self.blocks.iter()
            .map(|b| (b.bits.len() as f64 * (1.0 - b.redundancy) / 8.0).ceil() as usize)
            .sum();
        
        StorageStats {
            total_blocks: self.blocks.len(),
            total_bits,
            total_bytes,
            average_redundancy: self.blocks.iter()
                .map(|b| b.redundancy)
                .sum::<f64>() / self.blocks.len().max(1) as f64,
        }
    }
    
    /// Update default tension (for different eras of the universe)
    pub fn set_tension(&mut self, tension: f64) -> Result<(), &'static str> {
        if tension <= 0.0 {
            return Err("Tension must be positive");
        }
        self.default_tension = tension;
        Ok(())
    }
}

/// Storage statistics
#[derive(Debug, Clone, Copy)]
pub struct StorageStats {
    /// Number of stored blocks
    pub total_blocks: usize,
    /// Total bits stored (including redundancy)
    pub total_bits: usize,
    /// Total data bytes (excluding redundancy)
    pub total_bytes: usize,
    /// Average redundancy across blocks
    pub average_redundancy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosmic_string_bit() {
        let bit = CosmicStringBit::from_bool(true, 0.0, 1e22);
        assert_eq!(bit.winding, 1);
        assert!(bit.to_bool());
        
        let bit_false = CosmicStringBit::from_bool(false, 0.0, 1e22);
        assert_eq!(bit_false.winding, -1);
        assert!(!bit_false.to_bool());
    }

    #[test]
    fn test_encode_decode() {
        let data = vec![0x42, 0xAB, 0xCD];
        let block = TopologicalDataBlock::encode_bytes(0, &data, 1e22, 0.1).unwrap();
        
        let decoded = block.decode_bytes().unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_storage_manager() {
        let mut storage = CosmicStringStorage::default();
        
        let data = b"Hello, Heat Death!";
        let id = storage.store(data).unwrap();
        
        let retrieved = storage.retrieve(id).unwrap();
        assert_eq!(retrieved, data);
        
        let valid = storage.verify_integrity(id).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_statistics() {
        let mut storage = CosmicStringStorage::default();
        
        storage.store(b"Test1").unwrap();
        storage.store(b"Test2").unwrap();
        
        let stats = storage.get_statistics();
        assert_eq!(stats.total_blocks, 2);
        assert!(stats.total_bits > 0);
    }
}
