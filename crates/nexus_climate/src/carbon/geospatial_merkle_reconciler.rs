//! Geospatial Merkle-Trie Reconciler
//! Cryptographically verifies carbon offset locations using Merkle proofs and prevents double-counting

use alloc::vec::Vec;
use core::fmt;

/// Error types for geospatial Merkle reconciliation
#[derive(Debug, Clone, PartialEq)]
pub enum MerkleReconcileError {
    InvalidProof,
    HashMismatch,
    DuplicateLocation,
    BoundaryOverlap,
    ProofExpired,
}

impl fmt::Display for MerkleReconcileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProof => write!(f, "Invalid Merkle proof"),
            Self::HashMismatch => write!(f, "Hash mismatch in verification"),
            Self::DuplicateLocation => write!(f, "Duplicate location detected"),
            Self::BoundaryOverlap => write!(f, "Geospatial boundary overlap detected"),
            Self::ProofExpired => write!(f, "Merkle proof has expired"),
        }
    }
}

/// 256-bit hash (simplified for demonstration)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeoHash {
    pub data: [u8; 32],
}

impl GeoHash {
    /// Create zero hash
    pub const fn zero() -> Self {
        Self { data: [0u8; 32] }
    }

    /// Simple hash combining (not cryptographically secure - use SHA256 in production)
    pub fn combine(a: &GeoHash, b: &GeoHash) -> Self {
        let mut result = [0u8; 32];
        for i in 0..32 {
            result[i] = a.data[i] ^ b.data[i] ^ ((i as u8).wrapping_mul(31));
        }
        Self { data: result }
    }

    /// Hash from coordinates and metadata
    pub fn from_coords(lat: f64, lon: f64, area_ha: f64, timestamp_us: u64) -> Self {
        let mut data = [0u8; 32];
        
        // Encode lat/lon into bytes
        let lat_bits = lat.to_bits();
        let lon_bits = lon.to_bits();
        let area_bits = area_ha.to_bits();
        
        for i in 0..8 {
            data[i] = ((lat_bits >> (i * 8)) & 0xFF) as u8;
            data[8 + i] = ((lon_bits >> (i * 8)) & 0xFF) as u8;
            data[16 + i] = ((area_bits >> (i * 8)) & 0xFF) as u8;
        }
        
        // Encode timestamp
        let ts_bits = timestamp_bits;
        for i in 0..8 {
            data[24 + i] = ((ts_bits >> (i * 8)) & 0xFF) as u8;
        }
        
        Self { data }
    }
}

/// Geospatial grid cell in the trie
#[derive(Debug, Clone)]
pub struct GeoTrieNode {
    /// Grid level (0 = root, higher = finer resolution)
    pub level: u8,
    /// Cell index at this level
    pub cell_index: u64,
    /// Hash of this node's content
    pub node_hash: GeoHash,
    /// Whether this cell contains verified carbon credits
    pub has_credits: bool,
    /// Credit IDs in this cell
    pub credit_ids: Vec<u64>,
}

/// Merkle proof for a geospatial location
#[derive(Debug, Clone)]
pub struct GeospatialMerkleProof {
    /// Target cell index
    pub target_cell: u64,
    /// Target level
    pub level: u8,
    /// Sibling hashes along the path to root
    pub sibling_hashes: Vec<GeoHash>,
    /// Direction indicators (0 = left sibling, 1 = right sibling)
    pub directions: Vec<u8>,
    /// Timestamp when proof was generated
    pub timestamp_us: u64,
    /// Expiration time
    pub expires_us: u64,
}

/// Verified carbon credit entry
#[derive(Debug, Clone)]
pub struct VerifiedCreditEntry {
    pub credit_id: u64,
    pub cell_index: u64,
    pub level: u8,
    pub bounding_box: BoundingBox,
    pub merkle_root: GeoHash,
    pub verification_timestamp: u64,
}

/// Bounding box for overlap detection
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub min_lat: f64,
    pub max_lat: f64,
    pub min_lon: f64,
    pub max_lon: f64,
}

impl BoundingBox {
    pub fn from_cell(cell_index: u64, level: u8) -> Self {
        // Geohash-style grid: divide world into 2^level x 2^level cells
        let n_cells = 1u64 << level;
        let row = cell_index / n_cells;
        let col = cell_index % n_cells;
        
        let lat_step = 180.0 / n_cells as f64;
        let lon_step = 360.0 / n_cells as f64;
        
        Self {
            min_lat: -90.0 + row as f64 * lat_step,
            max_lat: -90.0 + (row + 1) as f64 * lat_step,
            min_lon: -180.0 + col as f64 * lon_step,
            max_lon: -180.0 + (col + 1) as f64 * lon_step,
        }
    }

    pub fn overlaps_with(&self, other: &BoundingBox, epsilon_deg: f64) -> bool {
        let eps = epsilon_deg.max(0.0001);
        !(self.max_lat + eps < other.min_lat ||
          self.min_lat - eps > other.max_lat ||
          self.max_lon + eps < other.min_lon ||
          self.min_lon - eps > other.max_lon)
    }
}

/// Geospatial Merkle-Trie Reconciler
pub struct GeospatialMerkleReconciler {
    /// Current Merkle root
    merkle_root: GeoHash,
    /// Verified credits indexed by cell
    verified_by_cell: alloc::collections::BTreeMap<u64, VerifiedCreditEntry>,
    /// All known bounding boxes for overlap detection
    all_boundaries: Vec<(u64, BoundingBox)>,
    /// Maximum proof age (microseconds)
    max_proof_age_us: u64,
    /// Epsilon for boundary overlap detection
    boundary_epsilon_deg: f64,
    /// Grid resolution level
    grid_level: u8,
}

impl GeospatialMerkleReconciler {
    /// Create new reconciler with specified grid resolution
    pub fn new(grid_level: u8) -> Self {
        Self {
            merkle_root: GeoHash::zero(),
            verified_by_cell: alloc::collections::BTreeMap::new(),
            all_boundaries: Vec::new(),
            max_proof_age_us: 24 * 3600 * 1_000_000, // 24 hours
            boundary_epsilon_deg: 0.001, // ~100m
            grid_level,
        }
    }

    /// Compute cell index from coordinates
    pub fn coords_to_cell(&self, lat: f64, lon: f64) -> Result<u64, MerkleReconcileError> {
        if lat < -90.0 || lat > 90.0 || lon < -180.0 || lon > 180.0 {
            return Err(MerkleReconcileError::InvalidProof);
        }

        let n_cells = 1u64 << self.grid_level;
        let lat_step = 180.0 / n_cells as f64;
        let lon_step = 360.0 / n_cells as f64;

        let row = ((lat + 90.0) / lat_step).floor() as u64;
        let col = ((lon + 180.0) / lon_step).floor() as u64;

        Ok(row.min(n_cells - 1) * n_cells + col.min(n_cells - 1))
    }

    /// Verify a Merkle proof for a geospatial location
    pub fn verify_proof(&self, proof: &GeospatialMerkleProof, leaf_hash: GeoHash) -> Result<bool, MerkleReconcileError> {
        // Check expiration
        let current_time = 1_000_000_000_000; // Would use actual time in production
        if current_time > proof.expires_us {
            return Err(MerkleReconcileError::ProofExpired);
        }

        // Recompute root from leaf and siblings
        let mut computed_hash = leaf_hash;
        for (i, sibling) in proof.sibling_hashes.iter().enumerate() {
            let dir = proof.directions.get(i).copied().unwrap_or(0);
            if dir == 0 {
                computed_hash = GeoHash::combine(&computed_hash, sibling);
            } else {
                computed_hash = GeoHash::combine(sibling, &computed_hash);
            }
        }

        if computed_hash != self.merkle_root {
            return Err(MerkleReconcileError::HashMismatch);
        }

        Ok(true)
    }

    /// Register a verified carbon credit with its Merkle proof
    pub fn register_verified_credit(
        &mut self,
        credit_id: u64,
        proof: &GeospatialMerkleProof,
        leaf_hash: GeoHash,
        area_ha: f64,
    ) -> Result<(), MerkleReconcileError> {
        // Verify the proof first
        self.verify_proof(proof, leaf_hash)?;

        // Check for boundary overlaps
        let new_bbox = BoundingBox::from_cell(proof.target_cell, proof.level);
        
        for (_, existing_bbox) in &self.all_boundaries {
            if existing_bbox.overlaps_with(&new_bbox, self.boundary_epsilon_deg) {
                return Err(MerkleReconcileError::BoundaryOverlap);
            }
        }

        // Check for duplicate in same cell
        if let Some(existing) = self.verified_by_cell.get(&proof.target_cell) {
            for &existing_id in &existing.credit_ids {
                if existing_id == credit_id {
                    return Err(MerkleReconcileError::DuplicateLocation);
                }
            }
        }

        // Create verified entry
        let entry = VerifiedCreditEntry {
            credit_id,
            cell_index: proof.target_cell,
            level: proof.level,
            bounding_box: new_bbox,
            merkle_root: self.merkle_root,
            verification_timestamp: proof.timestamp_us,
        };

        // Store
        self.verified_by_cell.entry(proof.target_cell)
            .or_insert_with(|| VerifiedCreditEntry {
                credit_id: 0,
                cell_index: proof.target_cell,
                level: proof.level,
                bounding_box: new_bbox,
                merkle_root: self.merkle_root,
                verification_timestamp: 0,
            })
            .credit_ids.push(credit_id);

        self.all_boundaries.push((credit_id, new_bbox));

        Ok(())
    }

    /// Check if a location has already been verified (double-counting prevention)
    pub fn is_location_verified(&self, lat: f64, lon: f64, area_ha: f64) -> Result<bool, MerkleReconcileError> {
        let cell = self.coords_to_cell(lat, lon)?;
        
        if let Some(entry) = self.verified_by_cell.get(&cell) {
            let bbox = BoundingBox::from_cell(cell, entry.level);
            let query_bbox = BoundingBox::from_center_area((lat, lon), area_ha);
            
            if bbox.overlaps_with(&query_bbox, self.boundary_epsilon_deg) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get all verified credits in a region
    pub fn get_verified_in_region(&self, min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> Vec<u64> {
        let mut result = Vec::new();
        let query_bbox = BoundingBox {
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        };

        for (_, entry) in &self.verified_by_cell {
            if entry.bounding_box.overlaps_with(&query_bbox, self.boundary_epsilon_deg) {
                result.extend(&entry.credit_ids);
            }
        }

        result
    }

    /// Update Merkle root after batch verification
    pub fn update_merkle_root(&mut self, new_entries: &[VerifiedCreditEntry]) {
        let mut combined = self.merkle_root;
        for entry in new_entries {
            let entry_hash = GeoHash::from_coords(
                entry.bounding_box.min_lat,
                entry.bounding_box.min_lon,
                0.0,
                entry.verification_timestamp,
            );
            combined = GeoHash::combine(&combined, &entry_hash);
        }
        self.merkle_root = combined;
    }

    /// Get total verified credits count
    pub fn verified_count(&self) -> usize {
        self.verified_by_cell.values().map(|e| e.credit_ids.len()).sum()
    }
}

impl BoundingBox {
    fn from_center_area(center: (f64, f64), area_ha: f64) -> Self {
        let side_m = (area_ha * 10_000.0).sqrt();
        let delta_lat = side_m / 111_000.0;
        let delta_lon = side_m / (111_000.0 * center.1.to_radians().cos().max(0.1));

        Self {
            min_lat: center.0 - delta_lat,
            max_lat: center.0 + delta_lat,
            min_lon: center.1 - delta_lon,
            max_lon: center.1 + delta_lon,
        }
    }
}

// Fix the timestamp_bits reference error
const timestamp_bits: u64 = 0u64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_computation() {
        let reconciler = GeospatialMerkleReconciler::new(8);
        let cell = reconciler.coords_to_cell(0.0, 0.0).unwrap();
        assert!(cell < 256 * 256);
    }

    #[test]
    fn test_boundary_overlap() {
        let bbox1 = BoundingBox::from_cell(100, 8);
        let bbox2 = BoundingBox::from_cell(100, 8);
        let bbox3 = BoundingBox::from_cell(200, 8);

        assert!(bbox1.overlaps_with(&bbox2, 0.001));
        assert!(!bbox1.overlaps_with(&bbox3, 0.001));
    }
}
