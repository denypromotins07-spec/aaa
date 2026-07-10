//! Ultrasonic Phased Array Controller for Acoustic Levitation
//! 
//! Calculates phase-shift delays for hundreds of piezoelectric emitters
//! to create 3D acoustic standing waves for particulate levitation.

use core::fmt;
use crate::acoustics::gorkov_potential_solver::{GorkovPotentialSolver, ParticleProperties, GorkovError};

/// Maximum number of transducers in the array
const MAX_TRANSDUCERS: usize = 1024;
/// Speed of sound in air at 20°C (m/s)
const SPEED_OF_SOUND: f64 = 343.0;
/// Minimum operating frequency (Hz)
const MIN_FREQUENCY: f64 = 20_000.0;
/// Maximum operating frequency (Hz)
const MAX_FREQUENCY: f64 = 10_000_000.0;
/// Phase locking tolerance (radians)
const PHASE_LOCK_TOLERANCE: f64 = 0.01;

/// Errors in ultrasonic phased array operations
#[derive(Debug, Clone, PartialEq)]
pub enum PhasedArrayError {
    InvalidTransducerCount,
    InvalidFrequency,
    PhaseCalculationFailed,
    ArrayGeometryError,
    TimingOverflow,
    SynchronizationLost,
}

impl fmt::Display for PhasedArrayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhasedArrayError::InvalidTransducerCount => write!(f, "Invalid number of transducers"),
            PhasedArrayError::InvalidFrequency => write!(f, "Frequency outside valid range"),
            PhasedArrayError::PhaseCalculationFailed => write!(f, "Failed to calculate phase shifts"),
            PhasedArrayError::ArrayGeometryError => write!(f, "Invalid array geometry configuration"),
            PhasedArrayError::TimingOverflow => write!(f, "Timing calculation overflow"),
            PhasedArrayError::SynchronizationLost => write!(f, "Array synchronization lost"),
        }
    }
}

/// Transducer element state
#[derive(Debug, Clone, Copy)]
pub struct TransducerElement {
    /// Position (x, y, z) in meters
    pub position: [f64; 3],
    /// Phase shift in radians
    pub phase_shift: f64,
    /// Amplitude scaling factor (0-1)
    pub amplitude: f64,
    /// Enabled flag
    pub enabled: bool,
    /// Last update timestamp (μs)
    pub last_update_us: u64,
}

impl Default for TransducerElement {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            phase_shift: 0.0,
            amplitude: 1.0,
            enabled: false,
            last_update_us: 0,
        }
    }
}

/// Ultrasonic phased array controller
pub struct UltrasonicPhasedArray {
    /// Array of transducer elements
    transducers: [TransducerElement; MAX_TRANSDUCERS],
    /// Number of active transducers
    active_count: usize,
    /// Operating frequency (Hz)
    frequency: f64,
    /// Wavelength (m)
    wavelength: f64,
    /// Wave number (rad/m)
    wave_number: f64,
    /// Gor'kov potential solver
    gorkov_solver: Option<GorkovPotentialSolver>,
    /// Phase lock status
    phase_locked: bool,
    /// Reference timestamp for synchronization
    reference_time_us: u64,
}

impl UltrasonicPhasedArray {
    /// Create a new phased array with specified transducer count
    pub fn new(transducer_count: usize) -> Result<Self, PhasedArrayError> {
        if transducer_count == 0 || transducer_count > MAX_TRANSDUCERS {
            return Err(PhasedArrayError::InvalidTransducerCount);
        }

        Ok(Self {
            transducers: [TransducerElement::default(); MAX_TRANSDUCERS],
            active_count: 0,
            frequency: 40_000.0, // Default 40kHz
            wavelength: SPEED_OF_SOUND / 40_000.0,
            wave_number: 2.0 * core::f64::consts::PI / (SPEED_OF_SOUND / 40_000.0),
            gorkov_solver: None,
            phase_locked: false,
            reference_time_us: 0,
        })
    }

    /// Initialize array with rectangular grid geometry
    pub fn init_rectangular_grid(
        &mut self,
        rows: usize,
        cols: usize,
        spacing: f64,
    ) -> Result<(), PhasedArrayError> {
        let total = rows * cols;
        if total > MAX_TRANSDUCERS {
            return Err(PhasedArrayError::InvalidTransducerCount);
        }

        let center_x = (cols as f64 - 1.0) * spacing / 2.0;
        let center_y = (rows as f64 - 1.0) * spacing / 2.0;

        let mut idx = 0;
        for row in 0..rows {
            for col in 0..cols {
                let x = col as f64 * spacing - center_x;
                let y = row as f64 * spacing - center_y;
                let z = 0.0;

                self.transducers[idx] = TransducerElement {
                    position: [x, y, z],
                    phase_shift: 0.0,
                    amplitude: 1.0,
                    enabled: true,
                    last_update_us: 0,
                };
                idx += 1;
            }
        }

        self.active_count = total;
        Ok(())
    }

    /// Initialize array with circular geometry
    pub fn init_circular_array(
        &mut self,
        num_rings: usize,
        transducers_per_ring: usize,
        ring_spacing: f64,
    ) -> Result<(), PhasedArrayError> {
        let total = num_rings * transducers_per_ring + 1; // +1 for center
        if total > MAX_TRANSDUCERS {
            return Err(PhasedArrayError::InvalidTransducerCount);
        }

        let mut idx = 0;

        // Center transducer
        self.transducers[0] = TransducerElement {
            position: [0.0, 0.0, 0.0],
            phase_shift: 0.0,
            amplitude: 1.0,
            enabled: true,
            last_update_us: 0,
        };
        idx = 1;

        // Rings
        for ring in 0..num_rings {
            let radius = (ring + 1) as f64 * ring_spacing;
            let angle_step = 2.0 * core::f64::consts::PI / transducers_per_ring as f64;

            for t in 0..transducers_per_ring {
                let angle = t as f64 * angle_step;
                let x = radius * angle.cos();
                let y = radius * angle.sin();
                let z = 0.0;

                self.transducers[idx] = TransducerElement {
                    position: [x, y, z],
                    phase_shift: 0.0,
                    amplitude: 1.0,
                    enabled: true,
                    last_update_us: 0,
                };
                idx += 1;
            }
        }

        self.active_count = total;
        Ok(())
    }

    /// Set operating frequency
    pub fn set_frequency(&mut self, frequency: f64) -> Result<(), PhasedArrayError> {
        if frequency < MIN_FREQUENCY || frequency > MAX_FREQUENCY {
            return Err(PhasedArrayError::InvalidFrequency);
        }

        self.frequency = frequency;
        self.wavelength = SPEED_OF_SOUND / frequency;
        self.wave_number = 2.0 * core::f64::consts::PI / self.wavelength;

        // Update Gor'kov solver if present
        if let Some(ref mut solver) = self.gorkov_solver {
            // Would need to recreate solver with new frequency
            // For now, just invalidate it
            self.gorkov_solver = None;
        }

        // Recalculate phases for new frequency
        self.recalculate_all_phases()?;

        Ok(())
    }

    /// Calculate phase shifts to focus at target point
    pub fn focus_at(&mut self, target: [f64; 3], timestamp_us: u64) -> Result<(), PhasedArrayError> {
        if !self.phase_locked {
            return Err(PhasedArrayError::SynchronizationLost);
        }

        for i in 0..self.active_count {
            if !self.transducers[i].enabled {
                continue;
            }

            // Calculate distance from transducer to target
            let dx = target[0] - self.transducers[i].position[0];
            let dy = target[1] - self.transducers[i].position[1];
            let dz = target[2] - self.transducers[i].position[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();

            // Calculate phase shift needed for constructive interference at target
            // Phase = k * r mod 2π, where k is wave number and r is distance
            let phase = self.wave_number * distance;
            let normalized_phase = phase % (2.0 * core::f64::consts::PI);

            self.transducers[i].phase_shift = normalized_phase;
            self.transducers[i].last_update_us = timestamp_us;
        }

        Ok(())
    }

    /// Calculate phase shifts to create multiple trap points
    pub fn create_multiple_traps(
        &mut self,
        targets: &[[f64; 3]],
        weights: &[f64],
        timestamp_us: u64,
    ) -> Result<(), PhasedArrayError> {
        if targets.len() != weights.len() {
            return Err(PhasedArrayError::ArrayGeometryError);
        }

        for i in 0..self.active_count {
            if !self.transducers[i].enabled {
                continue;
            }

            // Superposition of phases for multiple targets
            let mut total_phase = 0.0;
            let mut total_weight = 0.0;

            for (target_idx, target) in targets.iter().enumerate() {
                let dx = target[0] - self.transducers[i].position[0];
                let dy = target[1] - self.transducers[i].position[1];
                let dz = target[2] - self.transducers[i].position[2];
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                let phase = self.wave_number * distance;
                let weight = weights[target_idx];

                // Weighted phase averaging
                total_phase += phase * weight;
                total_weight += weight;
            }

            if total_weight > 0.0 {
                let avg_phase = total_phase / total_weight;
                self.transducers[i].phase_shift = avg_phase % (2.0 * core::f64::consts::PI);
                self.transducers[i].last_update_us = timestamp_us;
            }
        }

        Ok(())
    }

    /// Recalculate all phase shifts
    fn recalculate_all_phases(&mut self) -> Result<(), PhasedArrayError> {
        // Default focus point at λ/4 above array center
        let focus_z = self.wavelength / 4.0;
        let timestamp_us = 0; // Would use actual timestamp in real implementation
        
        self.focus_at([0.0, 0.0, focus_z], timestamp_us)
    }

    /// Enable phase locking with external reference
    pub fn enable_phase_lock(&mut self, reference_time_us: u64) -> Result<(), PhasedArrayError> {
        self.reference_time_us = reference_time_us;
        self.phase_locked = true;
        Ok(())
    }

    /// Disable phase locking
    pub fn disable_phase_lock(&mut self) {
        self.phase_locked = false;
    }

    /// Check if phase lock is maintained within tolerance
    pub fn check_phase_lock(&self, current_time_us: u64) -> bool {
        if !self.phase_locked {
            return false;
        }

        // Check if any transducer has drifted beyond tolerance
        let elapsed_us = current_time_us - self.reference_time_us;
        let elapsed_s = elapsed_us as f64 / 1_000_000.0;
        
        // Phase drift due to frequency instability
        let expected_phase_drift = 2.0 * core::f64::consts::PI * self.frequency * elapsed_s * 1e-6;
        
        expected_phase_drift.abs() < PHASE_LOCK_TOLERANCE
    }

    /// Get transducer element by index
    pub fn get_transducer(&self, index: usize) -> Option<&TransducerElement> {
        if index >= self.active_count {
            return None;
        }
        Some(&self.transducers[index])
    }

    /// Get all phase shifts as array slice
    pub fn get_phase_shifts(&self) -> &[f64] {
        &self.transducers[..self.active_count]
            .iter()
            .map(|t| t.phase_shift)
            .collect::<Vec<f64>>() // Note: This allocates, would use fixed buffer in production
    }

    /// Set amplitude for specific transducer
    pub fn set_amplitude(&mut self, index: usize, amplitude: f64) -> Result<(), PhasedArrayError> {
        if index >= self.active_count {
            return Err(PhasedArrayError::InvalidTransducerCount);
        }
        if amplitude < 0.0 || amplitude > 1.0 {
            return Err(PhasedArrayError::ArrayGeometryError);
        }

        self.transducers[index].amplitude = amplitude;
        Ok(())
    }

    /// Enable/disable specific transducer
    pub fn set_transducer_enabled(&mut self, index: usize, enabled: bool) -> Result<(), PhasedArrayError> {
        if index >= self.active_count {
            return Err(PhasedArrayError::InvalidTransducerCount);
        }

        self.transducers[index].enabled = enabled;
        Ok(())
    }

    /// Get active transducer count
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Get operating frequency
    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    /// Get wavelength
    pub fn wavelength(&self) -> f64 {
        self.wavelength
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_creation() {
        let array = UltrasonicPhasedArray::new(64);
        assert!(array.is_ok());
    }

    #[test]
    fn test_invalid_transducer_count() {
        let array = UltrasonicPhasedArray::new(0);
        assert_eq!(array.unwrap_err(), PhasedArrayError::InvalidTransducerCount);

        let array = UltrasonicPhasedArray::new(MAX_TRANSDUCERS + 1);
        assert_eq!(array.unwrap_err(), PhasedArrayError::InvalidTransducerCount);
    }

    #[test]
    fn test_rectangular_grid() {
        let mut array = UltrasonicPhasedArray::new(64).unwrap();
        let result = array.init_rectangular_grid(8, 8, 0.00425);
        assert!(result.is_ok());
        assert_eq!(array.active_count(), 64);
    }

    #[test]
    fn test_circular_array() {
        let mut array = UltrasonicPhasedArray::new(100).unwrap();
        let result = array.init_circular_array(3, 10, 0.005);
        assert!(result.is_ok());
        assert_eq!(array.active_count(), 31); // 1 center + 3 rings * 10
    }

    #[test]
    fn test_frequency_change() {
        let mut array = UltrasonicPhasedArray::new(64).unwrap();
        
        let result = array.set_frequency(50_000.0);
        assert!(result.is_ok());
        assert!(array.frequency() == 50_000.0);
    }

    #[test]
    fn test_invalid_frequency() {
        let mut array = UltrasonicPhasedArray::new(64).unwrap();
        
        let result = array.set_frequency(10_000.0);
        assert_eq!(result.unwrap_err(), PhasedArrayError::InvalidFrequency);
    }
}
