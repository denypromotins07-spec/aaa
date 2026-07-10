//! Bragg Grating Simulator
//! 
//! Calculates precise reference beam angles and wavelengths for holographic storage.
//! Ensures minimal cross-talk between adjacent data pages through strict selectivity thresholds.

use thiserror::Error;

/// Speed of light in vacuum (m/s)
pub const C: f64 = 299_792_458.0;

/// Minimum angular separation to prevent cross-talk (radians)
pub const MIN_ANGULAR_SEPARATION: f64 = 1e-6;

/// Maximum acceptable cross-talk ratio
pub const MAX_CROSSTALK_RATIO: f64 = 0.001;

/// Refractive index of typical photopolymer
pub const DEFAULT_REFRACTIVE_INDEX: f64 = 1.52;

#[derive(Error, Debug)]
pub enum BraggError {
    #[error("Invalid wavelength: {0} nm")]
    InvalidWavelength(f64),
    #[error("Invalid angle: {0} radians")]
    InvalidAngle(f64),
    #[error("Cross-talk threshold exceeded: {0}")]
    CrosstalkExceeded(f64),
    #[error("Bragg condition not satisfied")]
    BraggConditionNotMet,
    #[error("Maximum page capacity reached")]
    CapacityExceeded,
}

/// Represents a single holographic page's grating parameters
#[derive(Debug, Clone, Copy)]
pub struct GratingParameters {
    pub page_id: u64,
    pub wavelength_nm: f64,
    pub reference_angle_rad: f64,
    pub object_angle_rad: f64,
    pub grating_spacing_nm: f64,
    pub diffraction_efficiency: f64,
}

impl GratingParameters {
    /// Calculate grating spacing from Bragg condition: 2d sin(θ) = mλ
    pub fn calculate_grating_spacing(wavelength_nm: f64, angle_rad: f64, order: i32) -> Result<f64, BraggError> {
        if wavelength_nm <= 0.0 || wavelength_nm > 2000.0 {
            return Err(BraggError::InvalidWavelength(wavelength_nm));
        }
        
        let sin_theta = angle_rad.sin();
        if sin_theta.abs() < 1e-10 {
            return Err(BraggError::InvalidAngle(angle_rad));
        }
        
        let m = order as f64;
        let d = (m * wavelength_nm) / (2.0 * sin_theta);
        
        if d <= 0.0 {
            return Err(BraggError::BraggConditionNotMet);
        }
        
        Ok(d)
    }

    /// Calculate diffraction efficiency using Kogelnik's coupled wave theory
    pub fn calculate_diffraction_efficiency(
        thickness_um: f64,
        modulation_index: f64,
        wavelength_nm: f64,
        angle_rad: f64,
        refractive_index: f64,
    ) -> f64 {
        let kappa = std::f64::consts::PI * modulation_index / wavelength_nm;
        let delta = 2.0 * std::f64::consts::PI * refractive_index * thickness_um * 1e-6 
            * angle_rad.cos() / wavelength_nm;
        
        let xi = (kappa * kappa - delta * delta).sqrt();
        
        // Coupled wave solution
        if kappa > delta.abs() {
            (kappa * xi.sin()).powi(2)
        } else {
            (kappa * delta.sinh()).powi(2)
        }.clamp(0.0, 1.0)
    }
}

/// Bragg Grating Simulator for multi-page holographic storage
pub struct BraggGratingSimulator {
    pages: Vec<GratingParameters>,
    refractive_index: f64,
    crystal_thickness_um: f64,
    max_pages: usize,
}

impl BraggGratingSimulator {
    /// Create a new simulator with specified crystal properties
    pub fn new(crystal_thickness_um: f64, max_pages: usize) -> Self {
        Self {
            pages: Vec::with_capacity(max_pages.min(1000)),
            refractive_index: DEFAULT_REFRACTIVE_INDEX,
            crystal_thickness_um,
            max_pages,
        }
    }

    /// Allocate a new page with optimal angle/wavelength to minimize cross-talk
    pub fn allocate_page(&mut self, base_wavelength_nm: f64) -> Result<GratingParameters, BraggError> {
        if self.pages.len() >= self.max_pages {
            return Err(BraggError::CapacityExceeded);
        }

        let page_id = self.pages.len() as u64;
        
        // Calculate optimal reference angle based on existing pages
        let reference_angle = if self.pages.is_empty() {
            std::f64::consts::PI / 4.0 // Start at 45 degrees
        } else {
            // Angular multiplexing: increment angle to avoid cross-talk
            let last_angle = match self.pages.last() {
                Some(p) => p.reference_angle_rad,
                None => std::f64::consts::PI / 4.0, // Fallback (should not happen due to is_empty check)
            };
            let min_increment = MIN_ANGULAR_SEPARATION * (self.crystal_thickness_um / 100.0);
            last_angle + min_increment.max(MIN_ANGULAR_SEPARATION)
        };

        // Verify angle is within physical bounds
        if reference_angle <= 0.0 || reference_angle >= std::f64::consts::PI / 2.0 {
            return Err(BraggError::InvalidAngle(reference_angle));
        }

        // Calculate grating spacing using Bragg condition
        let grating_spacing = GratingParameters::calculate_grating_spacing(
            base_wavelength_nm,
            reference_angle,
            1, // First order diffraction
        )?;

        // Calculate object angle (symmetric configuration)
        let object_angle = reference_angle;

        // Calculate diffraction efficiency
        let modulation_index = 0.01; // Typical value for photopolymers
        let diffraction_efficiency = GratingParameters::calculate_diffraction_efficiency(
            self.crystal_thickness_um,
            modulation_index,
            base_wavelength_nm,
            reference_angle,
            self.refractive_index,
        );

        let params = GratingParameters {
            page_id,
            wavelength_nm: base_wavelength_nm,
            reference_angle_rad: reference_angle,
            object_angle_rad: object_angle,
            grating_spacing_nm: grating_spacing,
            diffraction_efficiency,
        };

        // Verify cross-talk with all existing pages
        for existing in &self.pages {
            let crosstalk = self.calculate_crosstalk(&params, existing)?;
            if crosstalk > MAX_CROSSTALK_RATIO {
                return Err(BraggError::CrosstalkExceeded(crosstalk));
            }
        }

        self.pages.push(params);
        Ok(params)
    }

    /// Calculate cross-talk ratio between two pages
    fn calculate_crosstalk(&self, page1: &GratingParameters, page2: &GratingParameters) -> Result<f64, BraggError> {
        let delta_angle = (page1.reference_angle_rad - page2.reference_angle_rad).abs();
        let delta_wavelength = (page1.wavelength_nm - page2.wavelength_nm).abs();

        // Angular selectivity function (sinc^2 approximation)
        let angular_selectivity = {
            let arg = std::f64::consts::PI * self.crystal_thickness_um * 1e-6 
                * delta_angle * self.refractive_index / page1.wavelength_nm;
            if arg.abs() < 1e-10 {
                1.0
            } else {
                (arg.sin() / arg).powi(2)
            }
        };

        // Spectral selectivity function
        let spectral_selectivity = {
            let coherence_length = page1.wavelength_nm.powi(2) / (2.0 * self.crystal_thickness_um * 1e-6);
            let arg = std::f64::consts::PI * delta_wavelength / coherence_length;
            if arg.abs() < 1e-10 {
                1.0
            } else {
                (arg.sin() / arg).powi(2)
            }
        };

        // Combined cross-talk ratio
        let crosstalk = angular_selectivity * spectral_selectivity;
        Ok(crosstalk)
    }

    /// Find the page ID for a given read angle
    pub fn find_page_by_angle(&self, angle_rad: f64, tolerance_rad: f64) -> Option<u64> {
        self.pages.iter().find(|p| {
            (p.reference_angle_rad - angle_rad).abs() < tolerance_rad
        }).map(|p| p.page_id)
    }

    /// Get all allocated pages
    pub fn allocated_pages(&self) -> &[GratingParameters] {
        &self.pages
    }

    /// Calculate total storage capacity in pages
    pub fn estimated_capacity(&self) -> usize {
        // Based on angular range and minimum separation
        let angular_range = std::f64::consts::PI / 2.0 - 0.1; // Avoid grazing angles
        let min_sep = MIN_ANGULAR_SEPARATION * (self.crystal_thickness_um / 100.0);
        (angular_range / min_sep.max(MIN_ANGULAR_SEPARATION)) as usize
    }

    /// Simulate read operation and return diffraction efficiency
    pub fn simulate_read(&self, page_id: u64, read_angle_rad: f64, read_wavelength_nm: f64) -> Result<f64, BraggError> {
        let page = self.pages.iter().find(|p| p.page_id == page_id)
            .ok_or_else(|| BraggError::InvalidAngle(read_angle_rad))?;

        // Calculate deviation from Bragg condition
        let angle_deviation = (read_angle_rad - page.reference_angle_rad).abs();
        let wavelength_deviation = (read_wavelength_nm - page.wavelength_nm).abs();

        // Efficiency drops with deviation
        let angular_factor = (-angle_deviation.powi(2) / (MIN_ANGULAR_SEPARATION.powi(2))).exp();
        let spectral_factor = (-wavelength_deviation.powi(2) / (1.0_f64)).exp();

        Ok(page.diffraction_efficiency * angular_factor * spectral_factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grating_spacing_calculation() {
        let spacing = GratingParameters::calculate_grating_spacing(532.0, std::f64::consts::PI / 4.0, 1).unwrap();
        assert!(spacing > 0.0);
        assert!(spacing < 1000.0); // Reasonable range
    }

    #[test]
    fn test_simulator_allocation() {
        let mut sim = BraggGratingSimulator::new(1000.0, 100);
        let page = sim.allocate_page(532.0).unwrap();
        assert_eq!(page.page_id, 0);
        assert!(page.diffraction_efficiency > 0.0);
    }

    #[test]
    fn test_crosstalk_prevention() {
        let mut sim = BraggGratingSimulator::new(1000.0, 100);
        let _page1 = sim.allocate_page(532.0).unwrap();
        let _page2 = sim.allocate_page(532.0).unwrap();
        
        // Verify cross-talk is below threshold
        if sim.pages.len() >= 2 {
            let crosstalk = sim.calculate_crosstalk(&sim.pages[0], &sim.pages[1]).unwrap();
            assert!(crosstalk <= MAX_CROSSTALK_RATIO);
        }
    }
}
