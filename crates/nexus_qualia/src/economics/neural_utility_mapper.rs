//! Neural Utility Mapper - maps BCI-derived cognitive strain to financial discount rates.
//! Implements neuro-economic utility functions with asymptotic ceilings to prevent singularities.

/// Maximum cognitive load (biological ceiling)
pub const MAX_COGNITIVE_LOAD: f32 = 100.0;

/// Default baseline utility
pub const BASELINE_UTILITY: f32 = 1.0;

/// Discount rate ceiling (prevents singularity)
pub const MAX_DISCOUNT_RATE: f32 = 0.5;

/// Error types for neural utility mapping
#[derive(Debug, Clone, PartialEq)]
pub enum NeuralUtilityError {
    CognitiveLoadExceeded(f32),
    InvalidNeuralSignal(f32),
    UtilityComputationFailed,
}

impl core::fmt::Display for NeuralUtilityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NeuralUtilityError::CognitiveLoadExceeded(load) => {
                write!(f, "Cognitive load {} exceeds maximum {}", load, MAX_COGNITIVE_LOAD)
            }
            NeuralUtilityError::InvalidNeuralSignal(signal) => {
                write!(f, "Invalid neural signal: {}", signal)
            }
            NeuralUtilityError::UtilityComputationFailed => {
                write!(f, "Utility computation failed")
            }
        }
    }
}

impl std::error::Error for NeuralUtilityError {}

/// Neural utility function result
#[derive(Debug, Clone)]
pub struct NeuralUtilityResult {
    pub raw_utility: f32,
    pub discounted_utility: f32,
    pub discount_rate: f32,
    pub cognitive_load: f32,
    pub neural_friction: f32,
    pub is_saturated: bool,
}

impl NeuralUtilityResult {
    pub const fn new() -> Self {
        Self {
            raw_utility: 0.0,
            discounted_utility: 0.0,
            discount_rate: 0.0,
            cognitive_load: 0.0,
            neural_friction: 0.0,
            is_saturated: false,
        }
    }
}

impl Default for NeuralUtilityResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Main neural utility mapper engine
pub struct NeuralUtilityMapper {
    result: NeuralUtilityResult,
    /// Baseline cognitive load
    baseline_load: f32,
    /// Sensitivity parameter
    sensitivity: f32,
    /// Risk aversion coefficient
    risk_aversion: f32,
}

impl NeuralUtilityMapper {
    pub fn new() -> Self {
        Self {
            result: NeuralUtilityResult::new(),
            baseline_load: 10.0,
            sensitivity: 0.1,
            risk_aversion: 0.5,
        }
    }

    /// Configure mapper parameters
    pub fn configure(&mut self, baseline: f32, sensitivity: f32, risk_aversion: f32) {
        self.baseline_load = baseline.clamp(0.0, MAX_COGNITIVE_LOAD);
        self.sensitivity = sensitivity.max(0.0);
        self.risk_aversion = risk_aversion.clamp(0.0, 1.0);
    }

    /// Map cognitive load to utility with asymptotic ceiling
    pub fn compute_utility(&mut self, cognitive_load: f32, time_horizon: f32) -> Result<&NeuralUtilityResult, NeuralUtilityError> {
        // Validate input
        if cognitive_load < 0.0 {
            return Err(NeuralUtilityError::InvalidNeuralSignal(cognitive_load));
        }

        // Clamp cognitive load to biological maximum (prevents singularity)
        let clamped_load = cognitive_load.min(MAX_COGNITIVE_LOAD);
        self.result.cognitive_load = clamped_load;

        // Compute neural friction (non-linear response)
        let normalized_load = clamped_load / MAX_COGNITIVE_LOAD;
        self.result.neural_friction = normalized_load.powf(2.0 + self.risk_aversion);

        // Compute raw utility using CRRA-like function
        // U = (1 - friction^sensitivity) * baseline
        let friction_factor = 1.0 - self.result.neural_friction.powf(self.sensitivity);
        self.result.raw_utility = BASELINE_UTILITY * friction_factor.max(0.0);

        // Compute discount rate with asymptotic ceiling
        // r = r_max * (1 - exp(-k * load))
        self.result.discount_rate = MAX_DISCOUNT_RATE * (1.0 - (-self.sensitivity * clamped_load).exp());
        self.result.discount_rate = self.result.discount_rate.min(MAX_DISCOUNT_RATE);

        // Apply time discounting
        let discount_factor = (-self.result.discount_rate * time_horizon).exp();
        self.result.discounted_utility = self.result.raw_utility * discount_factor;

        // Check saturation
        self.result.is_saturated = clamped_load >= MAX_COGNITIVE_LOAD * 0.95;

        Ok(&self.result)
    }

    /// Get marginal utility (derivative)
    pub fn marginal_utility(&self, cognitive_load: f32) -> Result<f32, NeuralUtilityError> {
        if cognitive_load < 0.0 || cognitive_load > MAX_COGNITIVE_LOAD {
            return Err(NeuralUtilityError::CognitiveLoadExceeded(cognitive_load));
        }

        let normalized = cognitive_load / MAX_COGNITIVE_LOAD;
        let exponent = 2.0 + self.risk_aversion;
        
        // dU/dL = -baseline * sensitivity * exponent * load^(exponent-1)
        let marginal = -BASELINE_UTILITY * self.sensitivity * exponent * 
            normalized.powf(exponent - 1.0) / MAX_COGNITIVE_LOAD;

        Ok(marginal)
    }

    /// Get current result
    pub const fn current_result(&self) -> &NeuralUtilityResult {
        &self.result
    }
}

impl Default for NeuralUtilityMapper {
    fn default() -> Self {
        Self::new()
    }
}
