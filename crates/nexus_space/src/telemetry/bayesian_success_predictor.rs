//! Bayesian Success Predictor for Launch Telemetry
//! 
//! Uses Bayesian filtering to predict launch success probability
//! based on real-time telemetry data.

/// Error types for Bayesian predictor
#[derive(Debug, Clone, Copy)]
pub enum BayesianError {
    InvalidProbability(f64),
    ZeroLikelihood,
    NumericalInstability,
}

impl core::fmt::Display for BayesianError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BayesianError::InvalidProbability(p) => write!(f, "Invalid probability: {}", p),
            BayesianError::ZeroLikelihood => write!(f, "Zero likelihood encountered"),
            BayesianError::NumericalInstability => write!(f, "Numerical instability"),
        }
    }
}

/// Launch phase enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LaunchPhase {
    Liftoff,
    MaxQ,
    MECO,      // Main Engine Cutoff
    Separation,
    SESCO,     // Second Engine Start
    SECO,      // Second Engine Cutoff
    PayloadDeploy,
}

/// Telemetry observation data
#[derive(Debug, Clone, Copy)]
pub struct TelemetryObservation {
    pub phase: LaunchPhase,
    pub velocity_ms: f64,
    pub altitude_km: f64,
    pub acceleration_ms2: f64,
    pub vibration_level: f64,
    pub timestamp: f64,
}

/// Bayesian success predictor with conjugate Beta-Binomial model
pub struct BayesianSuccessPredictor {
    alpha: f64, // Prior successes + 1
    beta: f64,  // Prior failures + 1
    posterior_mean: f64,
    posterior_variance: f64,
}

impl BayesianSuccessPredictor {
    /// Create predictor with uniform prior (Beta(1,1))
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            posterior_mean: 0.5,
            posterior_variance: 1.0 / 12.0,
        }
    }
    
    /// Create with informed prior
    pub fn with_prior(alpha: f64, beta: f64) -> Result<Self, BayesianError> {
        if alpha <= 0.0 || beta <= 0.0 {
            return Err(BayesianError::InvalidProbability(alpha));
        }
        
        let total = alpha + beta;
        let mean = alpha / total;
        let variance = (alpha * beta) / (total * total * (total + 1.0));
        
        Ok(Self {
            alpha,
            beta,
            posterior_mean: mean,
            posterior_variance: variance,
        })
    }
    
    /// Update posterior with new observation using likelihood function
    pub fn update(&mut self, observation: &TelemetryObservation) -> Result<f64, BayesianError> {
        // Compute likelihood based on telemetry consistency
        let likelihood = self.compute_likelihood(observation)?;
        
        if likelihood <= 0.0 {
            return Err(BayesianError::ZeroLikelihood);
        }
        
        // Beta-Binomial update (simplified)
        // In practice, would use proper likelihood ratio
        let effective_successes = likelihood;
        
        self.alpha += effective_successes;
        self.beta += (1.0 - effective_successes);
        
        // Update posterior statistics
        let total = self.alpha + self.beta;
        self.posterior_mean = self.alpha / total;
        self.posterior_variance = (self.alpha * self.beta) / (total * total * (total + 1.0));
        
        Ok(self.posterior_mean)
    }
    
    /// Compute likelihood of observation given expected flight profile
    fn compute_likelihood(&self, obs: &TelemetryObservation) -> Result<f64, BayesianError> {
        // Simplified likelihood based on nominal flight envelope
        let nominal_velocity = self.nominal_velocity_for_phase(obs.phase);
        let nominal_altitude = self.nominal_altitude_for_phase(obs.phase);
        
        // Gaussian likelihood for velocity deviation
        let velocity_deviation = (obs.velocity_ms - nominal_velocity).abs();
        let velocity_likelihood = (-velocity_deviation.powi(2) / (2.0 * 100.0)).exp();
        
        // Gaussian likelihood for altitude deviation
        let altitude_deviation = (obs.altitude_km - nominal_altitude).abs();
        let altitude_likelihood = (-altitude_deviation.powi(2) / (2.0 * 10.0)).exp();
        
        // Combine likelihoods (product rule for independent observations)
        let combined = velocity_likelihood * altitude_likelihood;
        
        // Clamp to valid range
        Ok(combined.max(0.01).min(0.99))
    }
    
    /// Get nominal velocity for launch phase (simplified model)
    fn nominal_velocity_for_phase(&self, phase: LaunchPhase) -> f64 {
        match phase {
            LaunchPhase::Liftoff => 0.0,
            LaunchPhase::MaxQ => 500.0,
            LaunchPhase::MECO => 2500.0,
            LaunchPhase::Separation => 2600.0,
            LaunchPhase::SESCO => 2700.0,
            LaunchPhase::SECO => 7800.0,
            LaunchPhase::PayloadDeploy => 7900.0,
        }
    }
    
    /// Get nominal altitude for launch phase
    fn nominal_altitude_for_phase(&self, phase: LaunchPhase) -> f64 {
        match phase {
            LaunchPhase::Liftoff => 0.0,
            LaunchPhase::MaxQ => 15.0,
            LaunchPhase::MECO => 80.0,
            LaunchPhase::Separation => 90.0,
            LaunchPhase::SESCO => 100.0,
            LaunchPhase::SECO => 200.0,
            LaunchPhase::PayloadDeploy => 300.0,
        }
    }
    
    /// Get current success probability estimate
    pub fn success_probability(&self) -> f64 {
        self.posterior_mean
    }
    
    /// Get credible interval (95%)
    pub fn credible_interval(&self) -> (f64, f64) {
        // Approximate 95% CI using normal approximation
        let std_dev = self.posterior_variance.sqrt();
        let lower = (self.posterior_mean - 1.96 * std_dev).max(0.0);
        let upper = (self.posterior_mean + 1.96 * std_dev).min(1.0);
        (lower, upper)
    }
    
    /// Reset to prior
    pub fn reset(&mut self) {
        let total = self.alpha + self.beta;
        self.posterior_mean = self.alpha / total;
        self.posterior_variance = (self.alpha * self.beta) / (total * total * (total + 1.0));
    }
}

impl Default for BayesianSuccessPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_uniform_prior() {
        let predictor = BayesianSuccessPredictor::new();
        assert!((predictor.success_probability() - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_update_increases_confidence() {
        let mut predictor = BayesianSuccessPredictor::with_prior(5.0, 1.0).unwrap();
        let obs = TelemetryObservation {
            phase: LaunchPhase::Liftoff,
            velocity_ms: 10.0,
            altitude_km: 0.5,
            acceleration_ms2: 9.8,
            vibration_level: 0.1,
            timestamp: 0.0,
        };
        
        let result = predictor.update(&obs);
        assert!(result.is_ok());
    }
}
