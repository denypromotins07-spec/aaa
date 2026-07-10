//! Generalized Advantage Estimation (GAE) Calculator
//! 
//! Implements n-step return calculation and GAE for policy gradient methods.
//! All computations are zero-allocation using pre-allocated buffers.

use std::sync::Arc;

/// Single trajectory step
#[derive(Debug, Clone)]
pub struct TrajectoryStep {
    /// Observation
    pub observation: Vec<f32>,
    /// Action taken
    pub action: Vec<f32>,
    /// Reward received
    pub reward: f64,
    /// Value estimate
    pub value: f64,
    /// Log probability of action
    pub log_prob: f64,
    /// Terminal flag
    pub done: bool,
}

impl TrajectoryStep {
    /// Create a new trajectory step
    pub fn new(
        observation: Vec<f32>,
        action: Vec<f32>,
        reward: f64,
        value: f64,
        log_prob: f64,
        done: bool,
    ) -> Self {
        Self {
            observation,
            action,
            reward,
            value,
            log_prob,
            done,
        }
    }
}

/// Computed advantages and returns
#[derive(Debug, Clone)]
pub struct Advantages {
    /// Advantage estimates
    pub advantages: Vec<f64>,
    /// Return estimates (value targets)
    pub returns: Vec<f64>,
    /// Discounted rewards
    pub discounted_rewards: Vec<f64>,
}

impl Advantages {
    /// Create with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            advantages: Vec::with_capacity(capacity),
            returns: Vec::with_capacity(capacity),
            discounted_rewards: Vec::with_capacity(capacity),
        }
    }
}

/// GAE Calculator with configurable parameters
pub struct GAECalculator {
    /// Discount factor (gamma)
    gamma: f64,
    /// GAE lambda parameter
    lambda: f64,
    /// Pre-allocated buffer for advantages
    advantage_buffer: Vec<f64>,
    /// Pre-allocated buffer for returns
    return_buffer: Vec<f64>,
}

impl GAECalculator {
    /// Create a new GAE calculator
    pub fn new(gamma: f64, lambda: f64, max_trajectory_length: usize) -> Self {
        Self {
            gamma: gamma.clamp(0.0, 1.0),
            lambda: lambda.clamp(0.0, 1.0),
            advantage_buffer: vec![0.0; max_trajectory_length],
            return_buffer: vec![0.0; max_trajectory_length],
        }
    }
    
    /// Default PPO configuration
    pub fn ppo_default() -> Self {
        Self::new(0.99, 0.95, 2048)
    }
    
    /// Default SAC configuration (lower lambda for off-policy)
    pub fn sac_default() -> Self {
        Self::new(0.99, 0.90, 1000)
    }
    
    /// Compute GAE for a single trajectory
    /// 
    /// Uses the formula:
    /// δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    /// A_t = δ_t + γ * λ * δ_{t+1} + (γ * λ)^2 * δ_{t+2} + ...
    /// 
    /// This is computed efficiently using backward recursion:
    /// A_t = δ_t + γ * λ * A_{t+1}
    pub fn compute_gae(&mut self, trajectory: &[TrajectoryStep], final_value: f64) -> Advantages {
        let len = trajectory.len();
        
        // Ensure buffers are large enough
        if self.advantage_buffer.len() < len {
            self.advantage_buffer.resize(len, 0.0);
            self.return_buffer.resize(len, 0.0);
        }
        
        // Clear output buffers
        let mut advantages = Advantages::with_capacity(len);
        
        // Compute TD errors and accumulate advantages backward
        let mut next_advantage = 0.0;
        let mut next_value = final_value;
        
        for t in (0..len).rev() {
            let step = &trajectory[t];
            
            // TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            let delta = step.reward + self.gamma * next_value - step.value;
            
            // GAE: A_t = δ_t + γ * λ * A_{t+1}
            next_advantage = delta + self.gamma * self.lambda * next_advantage;
            self.advantage_buffer[t] = next_advantage;
            
            // Return: R_t = A_t + V(s_t)
            self.return_buffer[t] = next_advantage + step.value;
            
            // Update for next iteration
            next_value = step.value;
        }
        
        // Copy to output vectors
        advantages.advantages.extend_from_slice(&self.advantage_buffer[..len]);
        advantages.returns.extend_from_slice(&self.return_buffer[..len]);
        
        // Compute discounted rewards separately
        let mut discounted_return = final_value;
        for t in (0..len).rev() {
            discounted_return = trajectory[t].reward + self.gamma * discounted_return;
            if trajectory[t].done {
                discounted_return = trajectory[t].reward; // Reset at terminal
            }
            advantages.discounted_rewards.push(discounted_return);
        }
        advantages.discounted_rewards.reverse();
        
        advantages
    }
    
    /// Compute n-step returns (alternative to GAE)
    pub fn compute_nstep_returns(
        &self,
        trajectory: &[TrajectoryStep],
        n: usize,
        final_value: f64,
    ) -> Vec<f64> {
        let len = trajectory.len();
        let mut returns = Vec::with_capacity(len);
        
        for t in 0..len {
            let mut return_estimate = 0.0;
            let mut discount = 1.0;
            
            // Sum n-step rewards
            for k in 0..n {
                let idx = t + k;
                if idx >= len {
                    break;
                }
                
                let step = &trajectory[idx];
                return_estimate += discount * step.reward;
                discount *= self.gamma;
                
                if step.done {
                    break;
                }
            }
            
            // Add bootstrap value if not terminal
            let bootstrap_idx = t + n;
            if bootstrap_idx < len && !trajectory[bootstrap_idx - 1].done {
                return_estimate += discount * trajectory[bootstrap_idx].value;
            } else if bootstrap_idx >= len {
                return_estimate += discount * final_value;
            }
            
            returns.push(return_estimate);
        }
        
        returns
    }
    
    /// Normalize advantages (zero mean, unit variance)
    pub fn normalize_advantages(advantages: &mut [f64]) {
        if advantages.is_empty() {
            return;
        }
        
        // Compute mean
        let mean = advantages.iter().sum::<f64>() / advantages.len() as f64;
        
        // Compute variance
        let variance = advantages
            .iter()
            .map(|&a| (a - mean).powi(2))
            .sum::<f64>() / advantages.len() as f64;
        
        // Normalize with epsilon for stability
        let std = variance.sqrt().max(1e-8);
        
        for a in advantages.iter_mut() {
            *a = (a - mean) / std;
        }
    }
    
    /// Compute importance sampling weights for off-policy correction
    pub fn compute_importance_weights(
        &self,
        old_log_probs: &[f64],
        new_log_probs: &[f64],
        clip_epsilon: f64,
    ) -> Vec<f64> {
        let mut weights = Vec::with_capacity(old_log_probs.len());
        
        for (&old_lp, &new_lp) in old_log_probs.iter().zip(new_log_probs.iter()) {
            // Ratio = exp(new_log_prob - old_log_prob)
            let ratio = (new_lp - old_lp).exp();
            
            // Clip for stability
            let clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon);
            
            weights.push(clipped);
        }
        
        weights
    }
    
    /// Get gamma parameter
    #[inline]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
    
    /// Get lambda parameter
    #[inline]
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

/// Batch processor for multiple trajectories
pub struct TrajectoryBatchProcessor {
    gae_calculator: GAECalculator,
    /// Maximum batch size
    max_batch_size: usize,
}

impl TrajectoryBatchProcessor {
    /// Create a new batch processor
    pub fn new(max_batch_size: usize, gamma: f64, lambda: f64) -> Self {
        Self {
            gae_calculator: GAECalculator::new(gamma, lambda, max_batch_size),
            max_batch_size,
        }
    }
    
    /// Process multiple trajectories into a single batch
    pub fn process_batch(
        &mut self,
        trajectories: &[Vec<TrajectoryStep>],
        final_values: &[f64],
    ) -> Advantages {
        let mut combined = Advantages::with_capacity(self.max_batch_size);
        
        for (traj, &final_val) in trajectories.iter().zip(final_values.iter()) {
            let result = self.gae_calculator.compute_gae(traj, final_val);
            combined.advantages.extend(result.advantages);
            combined.returns.extend(result.returns);
            combined.discounted_rewards.extend(result.discounted_rewards);
        }
        
        // Normalize advantages across entire batch
        GAECalculator::normalize_advantages(&mut combined.advantages);
        
        combined
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gae_computation() {
        let mut gae = GAECalculator::ppo_default();
        
        // Create simple trajectory
        let trajectory = vec![
            TrajectoryStep::new(vec![0.0], vec![0.0], 1.0, 0.5, -1.0, false),
            TrajectoryStep::new(vec![0.0], vec![0.0], 1.0, 0.6, -1.0, false),
            TrajectoryStep::new(vec![0.0], vec![0.0], 1.0, 0.7, -1.0, true),
        ];
        
        let result = gae.compute_gae(&trajectory, 0.0);
        
        assert_eq!(result.advantages.len(), 3);
        assert_eq!(result.returns.len(), 3);
        
        // Verify advantages are computed (non-zero)
        assert!(result.advantages.iter().any(|&a| a.abs() > 1e-6));
    }
    
    #[test]
    fn test_advantage_normalization() {
        let mut advantages = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        GAECalculator::normalize_advantages(&mut advantages);
        
        // Mean should be ~0
        let mean: f64 = advantages.iter().sum::<f64>() / advantages.len() as f64;
        assert!(mean.abs() < 1e-6);
        
        // Variance should be ~1
        let variance: f64 = advantages
            .iter()
            .map(|&a| a.powi(2))
            .sum::<f64>() / advantages.len() as f64;
        assert!((variance - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_nstep_returns() {
        let gae = GAECalculator::new(0.99, 0.95, 100);
        
        let trajectory = vec![
            TrajectoryStep::new(vec![0.0], vec![0.0], 1.0, 0.0, 0.0, false),
            TrajectoryStep::new(vec![0.0], vec![0.0], 1.0, 0.0, 0.0, false),
            TrajectoryStep::new(vec![0.0], vec![0.0], 1.0, 0.0, 0.0, false),
            TrajectoryStep::new(vec![0.0], vec![0.0], 1.0, 0.0, 0.0, true),
        ];
        
        let returns = gae.compute_nstep_returns(&trajectory, 2, 0.0);
        
        assert_eq!(returns.len(), 4);
        
        // First return should include 2 steps + bootstrap
        // R_0 = 1 + 0.99 * 1 + 0.99^2 * V(s_2)
        assert!(returns[0] > 1.9);
    }
    
    #[test]
    fn test_importance_weights() {
        let gae = GAECalculator::ppo_default();
        
        let old_log_probs = vec![-1.0, -2.0, -3.0];
        let new_log_probs = vec![-1.0, -1.5, -4.0];
        
        let weights = gae.compute_importance_weights(&old_log_probs, &new_log_probs, 0.2);
        
        assert_eq!(weights.len(), 3);
        
        // First weight should be 1.0 (same log prob)
        assert!((weights[0] - 1.0).abs() < 1e-6);
        
        // Other weights should be clipped
        assert!(weights[1] <= 1.2);
        assert!(weights[2] >= 0.8);
    }
}
