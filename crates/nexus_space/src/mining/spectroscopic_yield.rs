//! Spectroscopic Yield Estimator for Asteroid Mining
//! 
//! Estimates mineral composition and yield from astronomical spectral data.

/// Error types for spectroscopic analysis
#[derive(Debug, Clone, Copy)]
pub enum SpectroscopyError {
    InvalidAlbedo(f64),
    InvalidWavelength(f64),
    InsufficientData,
    NumericalInstability,
}

impl core::fmt::Display for SpectroscopyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SpectroscopyError::InvalidAlbedo(a) => write!(f, "Invalid albedo: {}", a),
            SpectroscopyError::InvalidWavelength(w) => write!(f, "Invalid wavelength: {}", w),
            SpectroscopyError::InsufficientData => write!(f, "Insufficient spectral data"),
            SpectroscopyError::NumericalInstability => write!(f, "Numerical instability"),
        }
    }
}

/// Asteroid spectral classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectralType {
    C,  // Carbonaceous
    S,  // Silicaceous
    M,  // Metallic
    D,  // Dark primitive
    P,  // Primitive
    E,  // Enstatite
    Unknown,
}

/// Mineral composition estimate
#[derive(Debug, Clone, Copy)]
pub struct MineralComposition {
    pub water_ice_percent: f64,
    pub iron_percent: f64,
    pub nickel_percent: f64,
    pub platinum_group_ppm: f64,
    pub silicate_percent: f64,
    pub carbon_percent: f64,
}

impl MineralComposition {
    /// Create zero composition
    pub fn zeros() -> Self {
        Self {
            water_ice_percent: 0.0,
            iron_percent: 0.0,
            nickel_percent: 0.0,
            platinum_group_ppm: 0.0,
            silicate_percent: 0.0,
            carbon_percent: 0.0,
        }
    }
    
    /// Validate composition sums to ~100%
    pub fn validate(&self) -> Result<(), SpectroscopyError> {
        let total = self.water_ice_percent + self.iron_percent + self.nickel_percent
                  + self.silicate_percent + self.carbon_percent;
        
        if (total - 100.0).abs() > 5.0 {
            return Err(SpectroscopyError::NumericalInstability);
        }
        
        Ok(())
    }
}

/// Spectroscopic yield estimator
pub struct SpectroscopicYieldEstimator {
    pub reference_albedo_c: f64,
    pub reference_albedo_s: f64,
    pub reference_albedo_m: f64,
}

impl SpectroscopicYieldEstimator {
    /// Create new estimator with reference values
    pub fn new() -> Self {
        Self {
            reference_albedo_c: 0.06,  // C-type typical albedo
            reference_albedo_s: 0.20,  // S-type typical albedo
            reference_albedo_m: 0.15,  // M-type typical albedo
        }
    }
    
    /// Classify asteroid from geometric albedo
    pub fn classify_from_albedo(&self, albedo: f64) -> Result<SpectralType, SpectroscopyError> {
        if albedo < 0.0 || albedo > 1.0 {
            return Err(SpectroscopyError::InvalidAlbedo(albedo));
        }
        
        // Simple classification based on albedo ranges
        let class = if albedo < 0.08 {
            SpectralType::C
        } else if albedo < 0.12 {
            SpectralType::P
        } else if albedo < 0.18 {
            SpectralType::M
        } else if albedo < 0.25 {
            SpectralType::S
        } else {
            SpectralType::E
        };
        
        Ok(class)
    }
    
    /// Estimate mineral composition from spectral type and albedo
    pub fn estimate_composition(&self, spectral_type: SpectralType, albedo: f64) -> MineralComposition {
        let mut comp = MineralComposition::zeros();
        
        match spectral_type {
            SpectralType::C => {
                comp.water_ice_percent = 10.0 + albedo * 20.0;
                comp.carbon_percent = 5.0 + albedo * 10.0;
                comp.silicate_percent = 60.0;
                comp.iron_percent = 10.0;
                comp.nickel_percent = 2.0;
                comp.platinum_group_ppm = 10.0;
            }
            SpectralType::S => {
                comp.silicate_percent = 70.0;
                comp.iron_percent = 15.0;
                comp.nickel_percent = 3.0;
                comp.platinum_group_ppm = 50.0;
                comp.water_ice_percent = 2.0;
                comp.carbon_percent = 1.0;
            }
            SpectralType::M => {
                comp.iron_percent = 60.0 + albedo * 20.0;
                comp.nickel_percent = 10.0 + albedo * 5.0;
                comp.platinum_group_ppm = 100.0 + albedo * 200.0;
                comp.silicate_percent = 15.0;
                comp.water_ice_percent = 0.5;
                comp.carbon_percent = 0.1;
            }
            SpectralType::D | SpectralType::P => {
                comp.silicate_percent = 50.0;
                comp.carbon_percent = 10.0;
                comp.water_ice_percent = 15.0;
                comp.iron_percent = 8.0;
                comp.nickel_percent = 1.0;
                comp.platinum_group_ppm = 5.0;
            }
            SpectralType::E => {
                comp.silicate_percent = 80.0;
                comp.iron_percent = 5.0;
                comp.nickel_percent = 1.0;
                comp.water_ice_percent = 0.1;
                comp.carbon_percent = 0.5;
                comp.platinum_group_ppm = 2.0;
            }
            SpectralType::Unknown => {
                // Conservative estimate
                comp.silicate_percent = 60.0;
                comp.iron_percent = 10.0;
                comp.nickel_percent = 2.0;
            }
        }
        
        comp
    }
    
    /// Estimate total yield from diameter and composition
    pub fn estimate_yield(&self, diameter_km: f64, density_g_cm3: f64, comp: &MineralComposition) -> Result<f64, SpectroscopyError> {
        if diameter_km <= 0.0 || density_g_cm3 <= 0.0 {
            return Err(SpectroscopyError::NumericalInstability);
        }
        
        // Volume of sphere (km³)
        let radius_km = diameter_km / 2.0;
        let volume_km3 = (4.0 / 3.0) * std::f64::consts::PI * radius_km.powi(3);
        
        // Convert to mass (tonnes): volume(km³) * density(g/cm³) * 1e9
        let mass_tonnes = volume_km3 * density_g_cm3 * 1e9;
        
        // Extract valuable material mass (PGM in tonnes)
        let pgm_mass = mass_tonnes * (comp.platinum_group_ppm / 1e6);
        
        Ok(pgm_mass)
    }
}

impl Default for SpectroscopicYieldEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_albedo_classification() {
        let estimator = SpectroscopicYieldEstimator::new();
        
        let c_type = estimator.classify_from_albedo(0.05).unwrap();
        assert_eq!(c_type, SpectralType::C);
        
        let m_type = estimator.classify_from_albedo(0.16).unwrap();
        assert_eq!(m_type, SpectralType::M);
    }
    
    #[test]
    fn test_yield_estimation() {
        let estimator = SpectroscopicYieldEstimator::new();
        let comp = estimator.estimate_composition(SpectralType::M, 0.15);
        let yield_tonnes = estimator.estimate_yield(1.0, 5.0, &comp);
        
        assert!(yield_tonnes.is_ok());
        assert!(yield_tonnes.unwrap() > 0.0);
    }
}
