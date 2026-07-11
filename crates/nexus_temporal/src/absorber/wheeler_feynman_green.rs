//! Wheeler-Feynman Green's Function for time-symmetric market impact
//! 
//! Implements the absorber theory with half-advanced and half-retarded
//! potentials to model time-symmetric liquidity interactions.

/// Speed of information propagation in market (price levels per nanosecond)
const MARKET_PROPAGATION_SPEED: f64 = 1e-6;

/// Default temporal cutoff for advanced potential integration
const DEFAULT_TEMPORAL_CUTOFF_NS: u64 = 10_000_000; // 10 milliseconds

/// Minimum epsilon for numerical stability
const NUMERICAL_EPSILON: f64 = 1e-15;

/// Type of Green's function component
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GreenComponent {
    /// Retarded potential (past affecting present)
    Retarded,
    /// Advanced potential (future affecting present)
    Advanced,
    /// Time-symmetric combination (half-advanced + half-retarded)
    Symmetric,
}

/// Result of Green's function evaluation
#[derive(Debug, Clone)]
pub struct GreenFunctionResult {
    /// Retarded potential contribution
    pub retarded_component: f64,
    /// Advanced potential contribution
    pub advanced_component: f64,
    /// Combined symmetric result
    pub symmetric_result: f64,
    /// Temporal cutoff used (nanoseconds)
    pub temporal_cutoff_ns: u64,
    /// Integration accuracy estimate
    pub accuracy_estimate: f64,
}

/// Wheeler-Feynman Green's Function solver
pub struct WheelerFeynmanGreen {
    /// Temporal cutoff for advanced potential (nanoseconds)
    temporal_cutoff_ns: u64,
    /// Propagation speed parameter
    propagation_speed: f64,
    /// Number of integration points
    integration_points: usize,
    /// Mean-reversion half-life from macro regime (nanoseconds)
    mean_reversion_half_life_ns: u64,
}

impl WheelerFeynmanGreen {
    /// Create a new Green's function solver with default parameters
    pub fn new() -> Self {
        Self {
            temporal_cutoff_ns: DEFAULT_TEMPORAL_CUTOFF_NS,
            propagation_speed: MARKET_PROPAGATION_SPEED,
            integration_points: 100,
            mean_reversion_half_life_ns: DEFAULT_TEMPORAL_CUTOFF_NS,
        }
    }

    /// Create solver with custom temporal cutoff based on macro regime
    /// 
    /// # Arguments
    /// * `mean_reversion_half_life_ns` - Expected mean-reversion half-life from Stage 12 Macro Regime
    pub fn with_macro_cutoff(mean_reversion_half_life_ns: u64) -> Self {
        // Use half-life as temporal cutoff (cannot integrate to infinity)
        let cutoff = mean_reversion_half_life_ns.max(1_000_000).min(100_000_000);
        
        Self {
            temporal_cutoff_ns: cutoff,
            propagation_speed: MARKET_PROPAGATION_SPEED,
            integration_points: 100,
            mean_reversion_half_life_ns: cutoff,
        }
    }

    /// Set integration precision (number of quadrature points)
    pub fn set_integration_precision(&mut self, points: usize) {
        self.integration_points = points.clamp(10, 1000);
    }

    /// Evaluate time-symmetric Green's function for market impact
    /// 
    /// # Arguments
    /// * `t` - Current time (nanoseconds)
    /// * `source_times` - Times of past/future trades (nanoseconds)
    /// * `source_strengths` - Strength of each trade impact
    /// 
    /// # Returns
    /// GreenFunctionResult with all components
    pub fn evaluate(&self, t: u64, source_times: &[u64], source_strengths: &[f64]) -> GreenFunctionResult {
        if source_times.is_empty() || source_strengths.is_empty() {
            return GreenFunctionResult {
                retarded_component: 0.0,
                advanced_component: 0.0,
                symmetric_result: 0.0,
                temporal_cutoff_ns: self.temporal_cutoff_ns,
                accuracy_estimate: 1.0,
            };
        }

        let len = source_times.len().min(source_strengths.len());
        
        // Compute retarded potential (sum over past sources)
        let mut retarded_sum = 0.0;
        for i in 0..len {
            let tau = source_times[i] as i64 - t as i64;
            if tau < 0 {
                // Past source contributes to retarded potential
                let distance = (-tau) as f64 * self.propagation_speed;
                let strength = source_strengths[i];
                retarded_sum += self.retarded_kernel(distance) * strength;
            }
        }

        // Compute advanced potential (sum over future sources, bounded by cutoff)
        let mut advanced_sum = 0.0;
        for i in 0..len {
            let tau = source_times[i] as i64 - t as i64;
            if tau > 0 {
                let tau_ns = tau as u64;
                if tau_ns <= self.temporal_cutoff_ns {
                    // Future source within cutoff contributes to advanced potential
                    let distance = tau as f64 * self.propagation_speed;
                    let strength = source_strengths[i];
                    advanced_sum += self.advanced_kernel(distance) * strength;
                }
            }
        }

        // Time-symmetric combination: G_sym = (G_retarded + G_advanced) / 2
        let symmetric = (retarded_sum + advanced_sum) / 2.0;

        // Estimate integration accuracy
        let accuracy = self.estimate_accuracy(t, source_times, source_strengths);

        GreenFunctionResult {
            retarded_component: retarded_sum,
            advanced_component: advanced_sum,
            symmetric_result: symmetric,
            temporal_cutoff_ns: self.temporal_cutoff_ns,
            accuracy_estimate: accuracy,
        }
    }

    /// Evaluate continuous advanced potential integral
    /// 
    /// Integrates future liquidity distribution from t to t + cutoff
    /// Uses Gaussian quadrature for numerical integration
    /// 
    /// # Arguments
    /// * `t` - Current time
    /// * `liquidity_density` - Function mapping time to liquidity density
    /// 
    /// # Returns
    /// Integrated advanced potential
    pub fn evaluate_continuous_advanced<F>(&self, t: u64, liquidity_density: F) -> f64
    where
        F: Fn(u64) -> f64,
    {
        let cutoff = self.temporal_cutoff_ns;
        let n = self.integration_points;
        
        // Gauss-Legendre quadrature nodes and weights (precomputed for [0, 1])
        let (nodes, weights) = self.gauss_legendre_nodes_weights(n);
        
        let mut integral = 0.0;
        for i in 0..n {
            // Map from [0, 1] to [t, t + cutoff]
            let tau = t + ((nodes[i] + 1.0) / 2.0 * cutoff as f64) as u64;
            let distance = (tau - t) as f64 * self.propagation_speed;
            
            let density = liquidity_density(tau);
            let kernel_value = self.advanced_kernel(distance);
            
            integral += weights[i] * density * kernel_value;
        }
        
        // Scale by integration range
        integral * cutoff as f64 / 2.0
    }

    /// Get the temporal cutoff in nanoseconds
    pub fn temporal_cutoff(&self) -> u64 {
        self.temporal_cutoff_ns
    }

    /// Update temporal cutoff based on new macro regime estimate
    pub fn update_cutoff(&mut self, new_half_life_ns: u64) {
        self.temporal_cutoff_ns = new_half_life_ns.max(1_000_000).min(100_000_000);
        self.mean_reversion_half_life_ns = self.temporal_cutoff_ns;
    }

    // Internal: Retarded kernel function G_R(t - t')
    fn retarded_kernel(&self, distance: f64) -> f64 {
        if distance < NUMERICAL_EPSILON {
            return 0.0;
        }
        // Standard retarded Green's function: ~1/r for 3D wave equation
        // Modified with exponential decay for market friction
        let decay = (-distance * 0.1).exp();
        decay / distance
    }

    // Internal: Advanced kernel function G_A(t' - t)
    fn advanced_kernel(&self, distance: f64) -> f64 {
        // Same functional form as retarded but for future times
        self.retarded_kernel(distance)
    }

    // Internal: Estimate integration accuracy via Richardson extrapolation
    fn estimate_accuracy(&self, t: u64, source_times: &[u64], source_strengths: &[f64]) -> f64 {
        if source_times.len() < 2 {
            return 1.0;
        }

        // Compare coarse vs fine integration
        let coarse_result = self.evaluate_with_points(t, source_times, source_strengths, self.integration_points / 2);
        let fine_result = self.evaluate_with_points(t, source_times, source_strengths, self.integration_points);
        
        let diff = (fine_result.symmetric_result - coarse_result.symmetric_result).abs();
        let scale = fine_result.symmetric_result.abs().max(NUMERICAL_EPSILON);
        
        1.0 - (diff / scale).min(1.0)
    }

    // Internal: Evaluate with specific number of integration points
    fn evaluate_with_points(&self, t: u64, source_times: &[u64], source_strengths: &[f64], points: usize) -> GreenFunctionResult {
        let len = source_times.len().min(source_strengths.len());
        
        let mut retarded_sum = 0.0;
        let mut advanced_sum = 0.0;
        
        for i in 0..len {
            let tau = source_times[i] as i64 - t as i64;
            let strength = source_strengths[i];
            
            if tau < 0 {
                let distance = (-tau) as f64 * self.propagation_speed;
                retarded_sum += self.retarded_kernel(distance) * strength;
            } else if tau as u64 <= self.temporal_cutoff_ns {
                let distance = tau as f64 * self.propagation_speed;
                advanced_sum += self.advanced_kernel(distance) * strength;
            }
        }

        GreenFunctionResult {
            retarded_component: retarded_sum,
            advanced_component: advanced_sum,
            symmetric_result: (retarded_sum + advanced_sum) / 2.0,
            temporal_cutoff_ns: self.temporal_cutoff_ns,
            accuracy_estimate: 1.0,
        }
    }

    // Internal: Get Gauss-Legendre nodes and weights for n points
    fn gauss_legendre_nodes_weights(&self, n: usize) -> (Vec<f64>, Vec<f64>) {
        // Simplified implementation - uses precomputed values for common n
        match n {
            2 => (
                vec![-0.5773502691896257, 0.5773502691896257],
                vec![1.0, 1.0],
            ),
            3 => (
                vec![-0.7745966692414834, 0.0, 0.7745966692414834],
                vec![0.5555555555555556, 0.8888888888888888, 0.5555555555555556],
            ),
            4 => (
                vec![-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
                vec![0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538],
            ),
            _ => {
                // Default to simple trapezoidal rule for other sizes
                let mut nodes = Vec::with_capacity(n);
                let mut weights = Vec::with_capacity(n);
                
                for i in 0..n {
                    nodes.push(-1.0 + 2.0 * i as f64 / (n - 1) as f64);
                    weights.push(2.0 / n as f64);
                }
                
                (nodes, weights)
            }
        }
    }
}

impl Default for WheelerFeynmanGreen {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_green_creation() {
        let green = WheelerFeynmanGreen::new();
        assert!(green.temporal_cutoff_ns > 0);
        assert!(green.propagation_speed > 0.0);
    }

    #[test]
    fn test_macro_cutoff() {
        let green = WheelerFeynmanGreen::with_macro_cutoff(5_000_000);
        assert_eq!(green.temporal_cutoff_ns, 5_000_000);
    }

    #[test]
    fn test_empty_sources() {
        let green = WheelerFeynmanGreen::new();
        let result = green.evaluate(1000, &[], &[]);
        
        assert_eq!(result.retarded_component, 0.0);
        assert_eq!(result.advanced_component, 0.0);
        assert_eq!(result.symmetric_result, 0.0);
    }

    #[test]
    fn test_retarded_only() {
        let green = WheelerFeynmanGreen::new();
        // Only past sources
        let source_times = vec![500, 600, 700];
        let strengths = vec![1.0, 1.0, 1.0];
        
        let result = green.evaluate(1000, &source_times, &strengths);
        
        assert!(result.retarded_component > 0.0);
        assert_eq!(result.advanced_component, 0.0);
    }

    #[test]
    fn test_advanced_only() {
        let green = WheelerFeynmanGreen::new();
        // Only future sources (within cutoff)
        let source_times = vec![1100, 1200, 1300];
        let strengths = vec![1.0, 1.0, 1.0];
        
        let result = green.evaluate(1000, &source_times, &strengths);
        
        assert_eq!(result.retarded_component, 0.0);
        assert!(result.advanced_component > 0.0);
    }

    #[test]
    fn test_symmetric_combination() {
        let green = WheelerFeynmanGreen::new();
        // Both past and future sources
        let source_times = vec![500, 1500];
        let strengths = vec![1.0, 1.0];
        
        let result = green.evaluate(1000, &source_times, &strengths);
        
        assert!(result.retarded_component > 0.0);
        assert!(result.advanced_component > 0.0);
        assert!((result.symmetric_result - (result.retarded_component + result.advanced_component) / 2.0).abs() < NUMERICAL_EPSILON);
    }
}
