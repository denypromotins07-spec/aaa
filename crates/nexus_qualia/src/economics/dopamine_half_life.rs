//! Dopamine Half-Life calculator for engagement decay prediction.
//! Computes the exact half-life of dopamine spikes from hedonic treadmill parameters.

use super::hedonic_treadmill_sde::{HedonicTreadmillSde, HedonicError, DEFAULT_MEAN_REVERSION};

/// Dopamine half-life result
#[derive(Debug, Clone)]
pub struct DopamineHalfLifeResult {
    pub half_life_seconds: f32,
    pub quarter_life_seconds: f32,
    pub time_to_baseline_seconds: f32,
    pub decay_constant: f32,
    pub current_deviation: f32,
    pub predicted_decay_path: [f32; 10],
}

impl DopamineHalfLifeResult {
    pub const fn new() -> Self {
        Self {
            half_life_seconds: 0.0,
            quarter_life_seconds: 0.0,
            time_to_baseline_seconds: 0.0,
            decay_constant: 0.0,
            current_deviation: 0.0,
            predicted_decay_path: [0.0; 10],
        }
    }
}

impl Default for DopamineHalfLifeResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Main dopamine half-life calculator
pub struct DopamineHalfLifeCalculator {
    treadmill: HedonicTreadmillSde,
    result: DopamineHalfLifeResult,
}

impl DopamineHalfLifeCalculator {
    pub fn new() -> Self {
        Self {
            treadmill: HedonicTreadmillSde::new(),
            result: DopamineHalfLifeResult::new(),
        }
    }

    /// Configure with mean reversion speed
    pub fn configure(&mut self, theta: f32) -> Result<(), HedonicError> {
        self.treadmill.configure(50.0, theta, 5.0)
    }

    /// Calculate half-life metrics after stimulus
    pub fn calculate(&mut self, initial_spike: f32) -> Result<&DopamineHalfLifeResult, HedonicError> {
        let theta = DEFAULT_MEAN_REVERSION;
        
        // Apply initial spike
        self.treadmill.apply_stimulus(initial_spike);
        
        // Decay constant is theta
        self.result.decay_constant = theta;
        
        // Half-life: t_1/2 = ln(2) / theta
        if theta > 1e-10 {
            self.result.half_life_seconds = core::f32::consts::LN_2 / theta;
            
            // Quarter-life: t_1/4 = ln(4) / theta
            self.result.quarter_life_seconds = (2.0 * core::f32::consts::LN_2) / theta;
            
            // Time to baseline (within 1%): t = ln(100) / theta
            self.result.time_to_baseline_seconds = (2.0 * core::f32::consts::LN_10) / theta;
        } else {
            self.result.half_life_seconds = f32::INFINITY;
            self.result.quarter_life_seconds = f32::INFINITY;
            self.result.time_to_baseline_seconds = f32::INFINITY;
        }

        // Current deviation
        self.result.current_deviation = initial_spike;

        // Predict decay path at regular intervals
        let interval = self.result.half_life_seconds / 10.0;
        for i in 0..10 {
            let t = i as f32 * interval;
            self.result.predicted_decay_path[i] = initial_spike * (-theta * t).exp();
        }

        Ok(&self.result)
    }

    /// Get engagement decay rate for platform churn prediction
    pub fn engagement_decay_rate(&self) -> f32 {
        -self.result.decay_constant
    }

    /// Predict user retention after stimulus
    pub fn predict_retention(&self, time_seconds: f32, initial_engagement: f32) -> f32 {
        initial_engagement * (-self.result.decay_constant * time_seconds).exp()
    }

    /// Get half-life
    #[inline]
    pub const fn half_life(&self) -> f32 {
        self.result.half_life_seconds
    }
}

impl Default for DopamineHalfLifeCalculator {
    fn default() -> Self {
        Self::new()
    }
}
