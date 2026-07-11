//! Doppler Pre-Compensation Module
pub use super::shannon_hartley_laser::{LaserError, LaserResult};

use core::marker::PhantomData;

/// Doppler compensator for relativistic frequency shifts
pub struct DopplerCompensator<'a> {
    _marker: PhantomData<&'a ()>,
}

impl<'a> DopplerCompensator<'a> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Calculate relativistic Doppler shift factor
    pub fn doppler_factor(&self, velocity_ms: f64) -> f64 {
        const C: f64 = 299_792_458.0;
        let beta = velocity_ms / C;
        if beta.abs() >= 1.0 {
            return 1.0;
        }
        ((1.0 + beta) / (1.0 - beta)).sqrt()
    }

    /// Pre-compensate transmission frequency
    pub fn pre_compensate_frequency(&self, nominal_freq_hz: f64, velocity_ms: f64) -> f64 {
        let factor = self.doppler_factor(velocity_ms);
        if factor <= 0.0 {
            return nominal_freq_hz;
        }
        nominal_freq_hz / factor
    }
}

impl Default for DopplerCompensator<'_> {
    fn default() -> Self {
        Self::new()
    }
}
