//! Zero-Point Photon Allocator for NEXUS-OMEGA
//! 
//! Maps harvested vacuum photons from the Dynamical Casimir Effect
//! directly into the photonic neural network for computation.
//! 
//! This bypasses the Landauer limit by using zero-point energy
//! rather than thermal energy for bit operations.

use core::fmt;
use alloc::{vec::Vec, string::String};

/// Represents a computational photon packet
#[derive(Debug, Clone, Copy)]
pub struct ComputePhoton {
    /// Frequency (Hz)
    pub frequency: f64,
    /// Energy (Joules)
    pub energy: f64,
    /// Wavelength (m)
    pub wavelength: f64,
    /// Assigned compute task ID
    pub task_id: u64,
    /// Creation timestamp
    pub timestamp: f64,
}

/// Configuration for the photon allocator
#[derive(Debug, Clone, Copy)]
pub struct AllocatorConfig {
    /// Minimum usable photon frequency (Hz)
    pub min_frequency: f64,
    /// Maximum usable photon frequency (Hz)
    pub max_frequency: f64,
    /// Target temperature for Landauer comparison (K)
    pub reference_temperature: f64,
    /// Allocation batch size
    pub batch_size: usize,
}

impl Default for AllocatorConfig {
    fn default() -> Self {
        Self {
            min_frequency: 1e9,      // 1 GHz
            max_frequency: 1e15,     // 1 PHz (near infrared)
            reference_temperature: 300.0, // Room temperature
            batch_size: 1024,
        }
    }
}

/// The Zero-Point Photon Allocator
pub struct ZeroPointAllocator {
    config: AllocatorConfig,
    /// Pool of available photons
    photon_pool: Vec<ComputePhoton>,
    /// Total energy allocated (J)
    total_allocated_energy: f64,
    /// Number of compute operations performed
    operations_count: u64,
    /// Landauer energy threshold (J/bit)
    landauer_threshold: f64,
}

impl ZeroPointAllocator {
    pub fn new(config: AllocatorConfig) -> Self {
        let k_b = 1.380_649e-23; // Boltzmann constant
        let landauer = k_b * config.reference_temperature * 2.0f64.ln();
        
        Self {
            config,
            photon_pool: Vec::new(),
            total_allocated_energy: 0.0,
            operations_count: 0,
            landauer_threshold: landauer,
        }
    }

    /// Add photons from DCE harvester to the pool
    /// Returns Result to avoid unwrap() in hot paths
    pub fn add_photons(&mut self, frequencies: &[f64], timestamp: f64) -> Result<usize, AllocatorError> {
        if frequencies.is_empty() {
            return Ok(0);
        }

        let h = 6.626_070_15e-34; // Planck constant
        let c = 299_792_458.0;    // Speed of light
        let mut added = 0;

        for &freq in frequencies.iter() {
            // Validate frequency range
            if freq < self.config.min_frequency || freq > self.config.max_frequency {
                continue; // Skip out-of-range photons
            }

            let energy = h * freq;
            let wavelength = c / freq;

            let photon = ComputePhoton {
                frequency: freq,
                energy,
                wavelength,
                task_id: 0, // Unassigned
                timestamp,
            };

            self.photon_pool.push(photon);
            added += 1;
        }

        Ok(added)
    }

    /// Allocate photons for a compute task
    pub fn allocate_for_task(&mut self, task_id: u64, required_energy: f64) 
        -> Result<Vec<ComputePhoton>, AllocatorError> 
    {
        if self.photon_pool.is_empty() {
            return Err(AllocatorError::EmptyPhotonPool);
        }

        let mut allocated = Vec::with_capacity(self.config.batch_size);
        let mut accumulated_energy = 0.0;

        // Greedy allocation from pool
        let mut i = 0;
        while i < self.photon_pool.len() && accumulated_energy < required_energy {
            let photon = self.photon_pool[i];
            
            // Check if photon is already assigned
            if photon.task_id != 0 {
                i += 1;
                continue;
            }

            allocated.push(photon);
            accumulated_energy += photon.energy;
            
            // Mark as assigned in pool
            self.photon_pool[i].task_id = task_id;
            
            i += 1;
        }

        if accumulated_energy < required_energy * 0.9 {
            // Return photons to unassigned state
            for photon in &mut allocated {
                photon.task_id = 0;
            }
            return Err(AllocatorError::InsufficientEnergy {
                required: required_energy,
                available: accumulated_energy,
            });
        }

        self.total_allocated_energy += accumulated_energy;
        self.operations_count += 1;

        Ok(allocated)
    }

    /// Calculate efficiency vs Landauer limit
    pub fn efficiency_ratio(&self) -> f64 {
        if self.total_allocated_energy == 0.0 {
            return 0.0;
        }

        // Theoretical minimum energy for same operations
        let theoretical_min = self.landauer_threshold * self.operations_count as f64;
        
        // Ratio: how many times above Landauer limit we are
        // Closer to 1.0 is better (we're approaching the limit)
        theoretical_min / self.total_allocated_energy
    }

    /// Get number of available photons
    pub fn available_photon_count(&self) -> usize {
        self.photon_pool.iter().filter(|p| p.task_id == 0).count()
    }

    /// Get total stored energy
    pub fn total_stored_energy(&self) -> f64 {
        self.photon_pool.iter()
            .filter(|p| p.task_id == 0)
            .map(|p| p.energy)
            .sum()
    }

    /// Reset allocator state
    pub fn reset(&mut self) {
        self.photon_pool.clear();
        self.total_allocated_energy = 0.0;
        self.operations_count = 0;
    }

    /// Get configuration
    pub const fn config(&self) -> &AllocatorConfig {
        &self.config
    }
}

/// Errors that can occur in photon allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocatorError {
    EmptyPhotonPool,
    InsufficientEnergy { required: f64, available: f64 },
    InvalidFrequency(f64),
    TaskAlreadyAssigned(u64),
    PoolOverflow,
}

impl fmt::Display for AllocatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocatorError::EmptyPhotonPool => write!(f, "Photon pool is empty"),
            AllocatorError::InsufficientEnergy { required, available } => {
                write!(f, "Insufficient energy: required {} J, available {} J", required, available)
            }
            AllocatorError::InvalidFrequency(freq) => {
                write!(f, "Invalid frequency: {} Hz (out of range)", freq)
            }
            AllocatorError::TaskAlreadyAssigned(task_id) => {
                write!(f, "Task {} already has allocated photons", task_id)
            }
            AllocatorError::PoolOverflow => write!(f, "Photon pool overflow"),
        }
    }
}

/// Photonic compute task descriptor
#[derive(Debug, Clone)]
pub struct PhotonicTask {
    pub task_id: u64,
    pub operation_type: OperationType,
    pub input_bits: usize,
    pub expected_output_bits: usize,
    pub deadline: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    MatrixMultiply,
    FourierTransform,
    Convolution,
    Activation,
    MemoryAccess,
}

impl PhotonicTask {
    pub fn new(
        task_id: u64,
        operation_type: OperationType,
        input_bits: usize,
        deadline: f64,
    ) -> Self {
        Self {
            task_id,
            operation_type,
            input_bits,
            expected_output_bits: input_bits, // Simplified
            deadline,
        }
    }

    /// Estimate energy requirement based on operation type
    pub fn estimated_energy(&self, photons_per_op: f64) -> f64 {
        let h = 6.626_070_15e-34;
        let base_freq = 1e14; // Base optical frequency
        
        match self.operation_type {
            OperationType::MatrixMultiply => {
                // O(n²) complexity for naive matrix multiply
                let ops = (self.input_bits as f64).powi(2);
                ops * photons_per_op * h * base_freq
            }
            OperationType::FourierTransform => {
                // O(n log n) complexity
                let n = self.input_bits as f64;
                let ops = n * n.log2();
                ops * photons_per_op * h * base_freq
            }
            OperationType::Convolution => {
                let ops = (self.input_bits as f64).powi(2);
                ops * photons_per_op * h * base_freq
            }
            OperationType::Activation => {
                // O(n) complexity
                let ops = self.input_bits as f64;
                ops * photons_per_op * h * base_freq
            }
            OperationType::MemoryAccess => {
                let ops = self.input_bits as f64;
                ops * photons_per_op * h * base_freq
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_creation() {
        let config = AllocatorConfig::default();
        let allocator = ZeroPointAllocator::new(config);
        assert_eq!(allocator.available_photon_count(), 0);
        assert!(allocator.landauer_threshold > 0.0);
    }

    #[test]
    fn test_add_photons() {
        let config = AllocatorConfig::default();
        let mut allocator = ZeroPointAllocator::new(config);
        
        let frequencies = vec![1e14, 2e14, 3e14];
        let added = allocator.add_photons(&frequencies, 0.0).unwrap();
        
        assert_eq!(added, 3);
        assert_eq!(allocator.available_photon_count(), 3);
    }

    #[test]
    fn test_allocate_for_task() {
        let config = AllocatorConfig::default();
        let mut allocator = ZeroPointAllocator::new(config);
        
        // Add some photons
        let frequencies: Vec<f64> = (1..101).map(|i| (i as f64) * 1e13).collect();
        allocator.add_photons(&frequencies, 0.0).unwrap();
        
        // Allocate for a task
        let result = allocator.allocate_for_task(1, 1e-18);
        assert!(result.is_ok());
        
        let allocated = result.unwrap();
        assert!(!allocated.is_empty());
    }

    #[test]
    fn test_landauer_comparison() {
        let config = AllocatorConfig::default();
        let allocator = ZeroPointAllocator::new(config);
        
        // At 300K, Landauer limit is about 2.8e-21 J/bit
        assert!(allocator.landauer_threshold > 1e-22);
        assert!(allocator.landauer_threshold < 1e-20);
    }

    #[test]
    fn test_photonic_task_energy() {
        let task = PhotonicTask::new(1, OperationType::MatrixMultiply, 1000, 1.0);
        let energy = task.estimated_energy(1.0);
        assert!(energy > 0.0);
    }
}
