//! Plasma Dispersion Model Module
pub use super::shannon_hartley_laser::{LaserError, LaserResult};

use core::marker::PhantomData;

/// Plasma dispersion model for interstellar medium effects
pub struct PlasmaDispersionModel<'a> {
    _marker: PhantomData<&'a ()>,
}

impl<'a> PlasmaDispersionModel<'a> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Calculate plasma frequency
    pub fn plasma_frequency(&self, electron_density_m3: f64) -> f64 {
        if electron_density_m3 <= 0.0 {
            return 0.0;
        }
        const E_CHARGE: f64 = 1.602_176_634e-19;
        const E_MASS: f64 = 9.109_383_56e-31;
        const EPSILON_0: f64 = 8.854_187_8128e-12;
        
        (electron_density_m3 * E_CHARGE.powi(2) / (E_MASS * EPSILON_0)).sqrt() / (2.0 * core::f64::consts::PI)
    }

    /// Calculate group delay due to plasma dispersion
    pub fn group_delay(&self, freq_hz: f64, distance_m: f64, electron_density_m3: f64) -> f64 {
        let plasma_freq = self.plasma_frequency(electron_density_m3);
        if freq_hz <= plasma_freq {
            return f64::INFINITY; // Signal cannot propagate
        }
        const C: f64 = 299_792_458.0;
        let ratio = plasma_freq / freq_hz;
        distance_m / (C * (1.0 - ratio.powi(2)).sqrt())
    }
}

impl Default for PlasmaDispersionModel<'_> {
    fn default() -> Self {
        Self::new()
    }
}
