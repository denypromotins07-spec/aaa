//! Stochastic Hedonic Treadmill model using Ornstein-Uhlenbeck SDE.
//! Models adaptation to new stimuli with mean-reverting dopamine dynamics.

/// Default mean reversion speed
pub const DEFAULT_MEAN_REVERSION: f32 = 0.1;

/// Default long-term baseline
pub const DEFAULT_BASELINE: f32 = 50.0;

/// Default volatility
pub const DEFAULT_VOLATILITY: f32 = 5.0;

/// Error types for hedonic treadmill
#[derive(Debug, Clone, PartialEq)]
pub enum HedonicError {
    InvalidTimeStep(f32),
    NegativeBaseline,
    NumericalInstability(f32),
}

impl core::fmt::Display for HedonicError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HedonicError::InvalidTimeStep(dt) => write!(f, "Invalid time step: {}", dt),
            HedonicError::NegativeBaseline => write!(f, "Baseline cannot be negative"),
            HedonicError::NumericalInstability(val) => write!(f, "Numerical instability: {}", val),
        }
    }
}

impl std::error::Error for HedonicError {}

/// Hedonic state result
#[derive(Debug, Clone)]
pub struct HedonicState {
    pub current_level: f32,
    pub baseline: f32,
    pub deviation: f32,
    pub half_life: f32,
    pub time_to_baseline: f32,
}

impl HedonicState {
    pub const fn new() -> Self {
        Self {
            current_level: 0.0,
            baseline: 0.0,
            deviation: 0.0,
            half_life: 0.0,
            time_to_baseline: 0.0,
        }
    }
}

impl Default for HedonicState {
    fn default() -> Self {
        Self::new()
    }
}

/// Main hedonic treadmill SDE engine
pub struct HedonicTreadmillSde {
    /// Current dopamine level
    current_level: f32,
    /// Long-term baseline (hedonic set point)
    baseline: f32,
    /// Mean reversion speed (theta)
    theta: f32,
    /// Volatility (sigma)
    sigma: f32,
    /// Result state
    state: HedonicState,
}

impl HedonicTreadmillSde {
    pub fn new() -> Self {
        Self {
            current_level: DEFAULT_BASELINE,
            baseline: DEFAULT_BASELINE,
            theta: DEFAULT_MEAN_REVERSION,
            sigma: DEFAULT_VOLATILITY,
            state: HedonicState::new(),
        }
    }

    /// Configure SDE parameters
    pub fn configure(&mut self, baseline: f32, theta: f32, sigma: f32) -> Result<(), HedonicError> {
        if baseline < 0.0 {
            return Err(HedonicError::NegativeBaseline);
        }
        self.baseline = baseline;
        self.theta = theta.max(0.0);
        self.sigma = sigma.max(0.0);
        Ok(())
    }

    /// Apply stimulus shock (dopamine spike)
    pub fn apply_stimulus(&mut self, magnitude: f32) {
        self.current_level += magnitude;
    }

    /// Advance SDE by one time step using Euler-Maruyama
    pub fn step(&mut self, dt: f32, noise: f32) -> Result<&HedonicState, HedonicError> {
        if dt <= 0.0 {
            return Err(HedonicError::InvalidTimeStep(dt));
        }

        // Ornstein-Uhlenbeck: dX = theta*(baseline - X)*dt + sigma*dW
        let drift = self.theta * (self.baseline - self.current_level) * dt;
        let diffusion = self.sigma * noise * dt.sqrt();
        
        self.current_level += drift + diffusion;
        
        // Ensure non-negative (biological constraint)
        self.current_level = self.current_level.max(0.0);

        // Update state
        self.state.current_level = self.current_level;
        self.state.baseline = self.baseline;
        self.state.deviation = self.current_level - self.baseline;
        self.state.half_life = self.compute_half_life();
        self.state.time_to_baseline = self.compute_time_to_baseline();

        Ok(&self.state)
    }

    /// Simulate full path (zero-alloc, fixed steps)
    pub fn simulate_path(&mut self, num_steps: usize, dt: f32, noise_sequence: &[f32]) -> Result<f32, HedonicError> {
        if noise_sequence.len() < num_steps {
            return Err(HedonicError::NumericalInstability(num_steps as f32));
        }

        let mut cumulative_deviation = 0.0f32;
        
        for i in 0..num_steps {
            self.step(dt, noise_sequence[i])?;
            cumulative_deviation += self.state.deviation.abs();
        }

        Ok(cumulative_deviation / num_steps as f32)
    }

    /// Compute half-life of dopamine spike
    fn compute_half_life(&self) -> f32 {
        // t_1/2 = ln(2) / theta
        if self.theta > 1e-10 {
            core::f32::consts::LN_2 / self.theta
        } else {
            f32::INFINITY
        }
    }

    /// Estimate time to return to baseline (within 5%)
    fn compute_time_to_baseline(&self) -> f32 {
        if self.theta > 1e-10 {
            -(-0.05).ln() / self.theta
        } else {
            f32::INFINITY
        }
    }

    /// Get current level
    #[inline]
    pub const fn current_level(&self) -> f32 {
        self.current_level
    }

    /// Get current state
    #[inline]
    pub const fn state(&self) -> &HedonicState {
        &self.state
    }
}

impl Default for HedonicTreadmillSde {
    fn default() -> Self {
        Self::new()
    }
}
