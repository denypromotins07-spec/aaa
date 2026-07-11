//! Precision Weighting Engine for Active Inference
//! 
//! Dynamically scales synaptic gain of prediction errors based on
//! market volatility, implementing attention-like mechanisms in
//! the wetware system.

use crate::inference::markov_blanket_mapper::{SensoryBuffer, ActiveBuffer};

/// Maximum number of precision channels
pub const MAX_PRECISION_CHANNELS: usize = 128;

/// Default precision bounds
const PRECISION_MIN: f32 = 0.01;
const PRECISION_MAX: f32 = 100.0;

/// Volatility regime classifications
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum VolatilityRegime {
    Low = 0,      // VIX < 15
    Normal = 1,   // VIX 15-25
    Elevated = 2, // VIX 25-40
    High = 3,     // VIX 40-60
    Extreme = 4,  // VIX > 60
}

impl VolatilityRegime {
    /// Classify volatility from VIX value
    pub fn from_vix(vix: f32) -> Self {
        if vix < 15.0 {
            VolatilityRegime::Low
        } else if vix < 25.0 {
            VolatilityRegime::Normal
        } else if vix < 40.0 {
            VolatilityRegime::Elevated
        } else if vix < 60.0 {
            VolatilityRegime::High
        } else {
            VolatilityRegime::Extreme
        }
    }

    /// Get base precision multiplier for regime
    pub fn base_precision_multiplier(&self) -> f32 {
        match self {
            VolatilityRegime::Low => 0.5,
            VolatilityRegime::Normal => 1.0,
            VolatilityRegime::Elevated => 2.0,
            VolatilityRegime::High => 4.0,
            VolatilityRegime::Extreme => 8.0,
        }
    }

    /// Get recommended learning rate adjustment
    pub fn learning_rate_factor(&self) -> f32 {
        match self {
            VolatilityRegime::Low => 0.5,    // Slow learning in stable markets
            VolatilityRegime::Normal => 1.0,
            VolatilityRegime::Elevated => 1.5,
            VolatilityRegime::High => 2.0,
            VolatilityRegime::Extreme => 0.2, // Reduce learning during chaos
        }
    }
}

/// Error types for precision weighting
#[derive(Debug, Clone, Copy)]
pub enum PrecisionError {
    InvalidChannel,
    OutOfBounds,
    NotInitialized,
    DivergenceDetected,
}

/// Precision parameters for a single channel
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct PrecisionParams {
    /// Current precision value (inverse variance)
    pub precision: f32,
    /// Prior precision (baseline)
    pub prior_precision: f32,
    /// Volatility sensitivity factor
    pub volatility_sensitivity: f32,
    /// Temporal derivative of precision
    pub precision_velocity: f32,
    /// Expected precision under current conditions
    pub expected_precision: f32,
    /// Precision prediction error
    pub prediction_error: f32,
}

impl Default for PrecisionParams {
    fn default() -> Self {
        Self {
            precision: 1.0,
            prior_precision: 1.0,
            volatility_sensitivity: 1.0,
            precision_velocity: 0.0,
            expected_precision: 1.0,
            prediction_error: 0.0,
        }
    }
}

/// Precision Weighting Engine
pub struct PrecisionWeightingEngine {
    /// Per-channel precision parameters
    channels: [PrecisionParams; MAX_PRECISION_CHANNELS],
    /// Number of active channels
    num_channels: usize,
    /// Current volatility regime
    current_regime: VolatilityRegime,
    /// Global precision scaling factor
    global_scale: f32,
    /// Precision update rate (learning rate)
    update_rate: f32,
    /// History of precision values for stability monitoring
    precision_history: [[f32; 16]; MAX_PRECISION_CHANNELS],
    history_idx: usize,
}

impl PrecisionWeightingEngine {
    /// Create a new precision weighting engine
    pub fn new(num_channels: usize) -> Self {
        let mut engine = Self {
            channels: [PrecisionParams::default(); MAX_PRECISION_CHANNELS],
            num_channels: num_channels.min(MAX_PRECISION_CHANNELS),
            current_regime: VolatilityRegime::Normal,
            global_scale: 1.0,
            update_rate: 0.1,
            precision_history: [[0.0; 16]; MAX_PRECISION_CHANNELS],
            history_idx: 0,
        };

        // Initialize all channels with default precision
        for i in 0..engine.num_channels {
            engine.channels[i].precision = 1.0;
            engine.channels[i].prior_precision = 1.0;
        }

        engine
    }

    /// Update precision based on prediction error and volatility
    #[inline]
    pub fn update_precision(
        &mut self,
        channel: usize,
        prediction_error: f32,
        volatility: f32,
    ) -> Result<f32, PrecisionError> {
        if channel >= self.num_channels {
            return Err(PrecisionError::InvalidChannel);
        }

        let params = &mut self.channels[channel];
        
        // Compute expected precision based on volatility
        params.expected_precision = params.prior_precision 
            * (1.0 + params.volatility_sensitivity * volatility.abs());
        
        // Precision prediction error
        params.prediction_error = params.expected_precision - params.precision;
        
        // Update precision using gradient ascent on expected free energy
        let precision_update = self.update_rate 
            * params.prediction_error 
            * prediction_error.abs();
        
        // Apply update with bounds
        let new_precision = (params.precision + precision_update)
            .clamp(PRECISION_MIN, PRECISION_MAX);
        
        // Compute velocity for stability monitoring
        params.precision_velocity = new_precision - params.precision;
        params.precision = new_precision;

        // Store in history
        self.precision_history[channel][self.history_idx] = params.precision;

        Ok(params.precision)
    }

    /// Batch update all channels
    pub fn update_all_channels(
        &mut self,
        prediction_errors: &[f32],
        volatilities: &[f32],
    ) -> Result<(), PrecisionError> {
        if prediction_errors.len() != self.num_channels 
            || volatilities.len() != self.num_channels 
        {
            return Err(PrecisionError::OutOfBounds);
        }

        for i in 0..self.num_channels {
            self.update_precision(i, prediction_errors[i], volatilities[i])?;
        }

        Ok(())
    }

    /// Update global volatility regime
    pub fn update_volatility_regime(&mut self, vix: f32) {
        self.current_regime = VolatilityRegime::from_vix(vix);
        self.global_scale = self.current_regime.base_precision_multiplier();
        
        // Scale all channel precisions by regime factor
        for i in 0..self.num_channels {
            let params = &mut self.channels[i];
            params.precision = (params.prior_precision * self.global_scale)
                .clamp(PRECISION_MIN, PRECISION_MAX);
        }
    }

    /// Set volatility sensitivity for a channel
    pub fn set_volatility_sensitivity(
        &mut self,
        channel: usize,
        sensitivity: f32,
    ) -> Result<(), PrecisionError> {
        if channel >= self.num_channels {
            return Err(PrecisionError::InvalidChannel);
        }

        self.channels[channel].volatility_sensitivity = sensitivity.max(0.0).min(10.0);
        Ok(())
    }

    /// Set prior precision (baseline) for a channel
    pub fn set_prior_precision(
        &mut self,
        channel: usize,
        prior: f32,
    ) -> Result<(), PrecisionError> {
        if channel >= self.num_channels {
            return Err(PrecisionError::InvalidChannel);
        }

        let clamped = prior.clamp(PRECISION_MIN, PRECISION_MAX);
        self.channels[channel].prior_precision = clamped;
        self.channels[channel].precision = clamped;
        Ok(())
    }

    /// Get current precision for a channel
    #[inline]
    pub fn get_precision(&self, channel: usize) -> Option<f32> {
        if channel < self.num_channels {
            Some(self.channels[channel].precision)
        } else {
            None
        }
    }

    /// Get precision-weighted prediction error
    #[inline]
    pub fn get_weighted_error(&self, channel: usize, raw_error: f32) -> Option<f32> {
        self.get_precision(channel).map(|p| p * raw_error)
    }

    /// Check for precision divergence (instability detection)
    pub fn check_divergence(&self, channel: usize, threshold: f32) -> bool {
        if channel >= self.num_channels {
            return false;
        }

        let history = &self.precision_history[channel];
        let valid_count = self.history_idx.min(16);
        
        if valid_count < 2 {
            return false;
        }

        // Compute variance of recent precision values
        let mean: f32 = history[..valid_count].iter().sum::<f32>() / valid_count as f32;
        let variance: f32 = history[..valid_count]
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / valid_count as f32;

        // High variance indicates instability
        variance.sqrt() > threshold
    }

    /// Reset precision to prior for a channel
    pub fn reset_channel(&mut self, channel: usize) -> Result<(), PrecisionError> {
        if channel >= self.num_channels {
            return Err(PrecisionError::InvalidChannel);
        }

        let params = &mut self.channels[channel];
        params.precision = params.prior_precision;
        params.precision_velocity = 0.0;
        params.prediction_error = 0.0;
        
        // Clear history
        self.precision_history[channel] = [0.0; 16];
        
        Ok(())
    }

    /// Reset all channels
    pub fn reset_all(&mut self) {
        for i in 0..self.num_channels {
            let _ = self.reset_channel(i);
        }
        self.history_idx = 0;
    }

    /// Advance history index (call periodically)
    pub fn advance_history(&mut self) {
        self.history_idx = (self.history_idx + 1) % 16;
    }

    /// Get current volatility regime
    #[inline]
    pub fn get_regime(&self) -> VolatilityRegime {
        self.current_regime
    }

    /// Get global precision scale
    #[inline]
    pub fn get_global_scale(&self) -> f32 {
        self.global_scale
    }

    /// Set update rate (learning rate)
    pub fn set_update_rate(&mut self, rate: f32) {
        self.update_rate = rate.max(0.001).min(1.0);
    }

    /// Apply precision to sensory buffer
    pub fn apply_to_sensory_buffer(&self, buffer: &mut SensoryBuffer) {
        for i in 0..self.num_channels.min(buffer.as_slice().len()) {
            let _ = buffer.set_precision(i, self.channels[i].precision);
        }
    }
}

/// Synaptic Gain Modulator for neuromodulation effects
pub struct SynapticGainModulator {
    /// Base synaptic gains per channel
    base_gains: [f32; MAX_PRECISION_CHANNELS],
    /// Current modulated gains
    current_gains: [f32; MAX_PRECISION_CHANNELS],
    /// Neuromodulator levels (dopamine, serotonin, etc.)
    dopamine_level: f32,
    serotonin_level: f32,
    cortisol_level: f32,
    /// Number of channels
    num_channels: usize,
}

impl SynapticGainModulator {
    /// Create a new synaptic gain modulator
    pub fn new(num_channels: usize) -> Self {
        Self {
            base_gains: [1.0; MAX_PRECISION_CHANNELS],
            current_gains: [1.0; MAX_PRECISION_CHANNELS],
            dopamine_level: 0.5,
            serotonin_level: 0.5,
            cortisol_level: 0.0,
            num_channels: num_channels.min(MAX_PRECISION_CHANNELS),
        }
    }

    /// Set neuromodulator level
    pub fn set_neuromodulator(
        &mut self,
        dopamine: Option<f32>,
        serotonin: Option<f32>,
        cortisol: Option<f32>,
    ) {
        if let Some(d) = dopamine {
            self.dopamine_level = d.clamp(0.0, 1.0);
        }
        if let Some(s) = serotonin {
            self.serotonin_level = s.clamp(0.0, 1.0);
        }
        if let Some(c) = cortisol {
            self.cortisol_level = c.clamp(0.0, 1.0);
        }
        
        self.recompute_gains();
    }

    /// Recompute gains based on neuromodulator levels
    fn recompute_gains(&mut self) {
        for i in 0..self.num_channels {
            // Dopamine increases gain (enhances signal)
            // Serotonin stabilizes gain (reduces variance)
            // Cortisol decreases gain (risk-averse state)
            
            let dopaminergic_factor = 0.5 + self.dopamine_level * 1.5; // 0.5 to 2.0
            let serotonergic_factor = 1.0 - self.serotonin_level * 0.3; // 0.7 to 1.0
            let cortisol_factor = 1.0 - self.cortisol_level * 0.7;      // 0.3 to 1.0
            
            self.current_gains[i] = self.base_gains[i] 
                * dopaminergic_factor 
                * serotonergic_factor 
                * cortisol_factor;
            
            // Clamp to reasonable range
            self.current_gains[i] = self.current_gains[i].clamp(0.1, 5.0);
        }
    }

    /// Set base gain for a channel
    pub fn set_base_gain(&mut self, channel: usize, gain: f32) -> Result<(), PrecisionError> {
        if channel >= self.num_channels {
            return Err(PrecisionError::InvalidChannel);
        }

        self.base_gains[channel] = gain.max(0.1).min(10.0);
        self.recompute_gains();
        Ok(())
    }

    /// Get current gain for a channel
    #[inline]
    pub fn get_gain(&self, channel: usize) -> Option<f32> {
        if channel < self.num_channels {
            Some(self.current_gains[channel])
        } else {
            None
        }
    }

    /// Apply gain modulation to a signal
    #[inline]
    pub fn modulate_signal(&self, channel: usize, signal: f32) -> Option<f32> {
        self.get_gain(channel).map(|g| signal * g)
    }

    /// Trigger cortisol analogue protocol (risk-averse state)
    pub fn trigger_cortisol_protocol(&mut self, intensity: f32) {
        let clamped = intensity.clamp(0.0, 1.0);
        self.cortisol_level = clamped;
        // Also reduce dopamine in high-stress situations
        self.dopamine_level = (1.0 - clamped * 0.5).max(0.2);
        self.recompute_gains();
    }

    /// Reset neuromodulator levels to baseline
    pub fn reset_neuromodulators(&mut self) {
        self.dopamine_level = 0.5;
        self.serotonin_level = 0.5;
        self.cortisol_level = 0.0;
        self.recompute_gains();
    }

    /// Get current neuromodulator state
    pub fn get_neuromodulator_state(&self) -> (f32, f32, f32) {
        (self.dopamine_level, self.serotonin_level, self.cortisol_level)
    }
}

/// Adaptive precision controller with homeostatic regulation
pub struct HomeostaticPrecisionController {
    /// Target precision (setpoint)
    target_precision: f32,
    /// Integral term for steady-state error
    integral_error: f32,
    /// Derivative term for damping
    derivative_error: f32,
    /// Previous error
    previous_error: f32,
    /// PID gains
    kp: f32, ki: f32, kd: f32,
    /// Number of channels
    num_channels: usize,
}

impl HomeostaticPrecisionController {
    /// Create a new homeostatic controller
    pub fn new(num_channels: usize, target: f32) -> Self {
        Self {
            target_precision: target,
            integral_error: 0.0,
            derivative_error: 0.0,
            previous_error: 0.0,
            kp: 1.0,
            ki: 0.01,
            kd: 0.1,
            num_channels: num_channels.min(MAX_PRECISION_CHANNELS),
        }
    }

    /// Compute PID control output
    pub fn compute_control(&mut self, current_precision: f32) -> f32 {
        let error = self.target_precision - current_precision;
        
        // Proportional term
        let p_term = self.kp * error;
        
        // Integral term (with anti-windup)
        self.integral_error = (self.integral_error + error).clamp(-10.0, 10.0);
        let i_term = self.ki * self.integral_error;
        
        // Derivative term
        self.derivative_error = error - self.previous_error;
        let d_term = self.kd * self.derivative_error;
        
        self.previous_error = error;
        
        (p_term + i_term + d_term).clamp(-1.0, 1.0)
    }

    /// Set PID gains
    pub fn set_gains(&mut self, kp: f32, ki: f32, kd: f32) {
        self.kp = kp.max(0.0);
        self.ki = ki.max(0.0);
        self.kd = kd.max(0.0);
    }

    /// Reset controller state
    pub fn reset(&mut self) {
        self.integral_error = 0.0;
        self.derivative_error = 0.0;
        self.previous_error = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volatility_regime_classification() {
        assert_eq!(VolatilityRegime::from_vix(10.0), VolatilityRegime::Low);
        assert_eq!(VolatilityRegime::from_vix(20.0), VolatilityRegime::Normal);
        assert_eq!(VolatilityRegime::from_vix(30.0), VolatilityRegime::Elevated);
        assert_eq!(VolatilityRegime::from_vix(50.0), VolatilityRegime::High);
        assert_eq!(VolatilityRegime::from_vix(70.0), VolatilityRegime::Extreme);
    }

    #[test]
    fn test_precision_update() {
        let mut engine = PrecisionWeightingEngine::new(8);
        
        let initial = engine.get_precision(0).unwrap();
        assert!((initial - 1.0).abs() < 1e-6);
        
        // Update with positive prediction error and volatility
        let updated = engine.update_precision(0, 0.5, 0.3).unwrap();
        assert!(updated > initial);
    }

    #[test]
    fn test_synaptic_gain_modulation() {
        let mut modulator = SynapticGainModulator::new(8);
        
        let baseline = modulator.get_gain(0).unwrap();
        
        // Trigger cortisol protocol (stress response)
        modulator.trigger_cortisol_protocol(1.0);
        
        let stressed = modulator.get_gain(0).unwrap();
        assert!(stressed < baseline); // Cortisol should reduce gain
    }

    #[test]
    fn test_homeostatic_controller() {
        let mut controller = HomeostaticPrecisionController::new(8, 1.0);
        
        // Test with below-target precision
        let control = controller.compute_control(0.5);
        assert!(control > 0.0); // Should push up
        
        // Test with above-target precision
        let control = controller.compute_control(1.5);
        assert!(control < 0.0); // Should push down
    }
}
