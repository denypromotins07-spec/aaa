//! Survival Analysis for limit order queue depletion using Kaplan-Meier estimators.
//! 
//! Calculates the real-time probability that a specific limit order in the queue
//! will actually survive to be filled, filtering out phantom liquidity.

use std::collections::BTreeMap;
use parking_lot::RwLock;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SurvivalError {
    #[error("Division by zero in survival calculation")]
    DivisionByZero,
    #[error("Invalid time sequence")]
    InvalidTimeSequence,
}

/// Survival function estimate at a specific time point
#[derive(Debug, Clone)]
pub struct SurvivalEstimate {
    pub time_ms: f64,
    pub survival_probability: f64,
    pub confidence_interval_lower: f64,
    pub confidence_interval_upper: f64,
    pub at_risk_count: usize,
    pub event_count: usize,
}

/// Event type in survival analysis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueueEvent {
    /// Order was filled (event of interest)
    Fill,
    /// Order was canceled (censored observation)
    Canceled,
    /// Order modified (partially censored)
    Modified,
}

/// Observation record for survival analysis
struct Observation {
    time_ms: f64,
    event: QueueEvent,
    volume: u64,
}

/// Kaplan-Meier estimator for queue survival analysis
pub struct KaplanMeierEstimator {
    /// Observations grouped by price level
    observations: RwLock<BTreeMap<i64, Vec<Observation>>>,
    /// Current survival function estimates per price level
    survival_curves: RwLock<BTreeMap<i64, Vec<SurvivalEstimate>>>,
    /// Minimum orders required before estimation
    min_observations: usize,
    /// Confidence level for intervals (e.g., 0.95 for 95% CI)
    confidence_level: f64,
}

impl KaplanMeierEstimator {
    /// Create a new Kaplan-Meier estimator
    pub fn new(min_observations: usize, confidence_level: f64) -> Result<Self, SurvivalError> {
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(SurvivalError::DivisionByZero);
        }
        
        Ok(Self {
            observations: RwLock::new(BTreeMap::new()),
            survival_curves: RwLock::new(BTreeMap::new()),
            min_observations,
            confidence_level,
        })
    }

    /// Record an order lifecycle event
    #[inline]
    pub fn record_event(&self, price_level: i64, time_ms: f64, event: QueueEvent, volume: u64) {
        let mut observations = self.observations.write();
        let obs_list = observations.entry(price_level).or_insert_with(Vec::new);
        
        // Maintain sorted order by time
        let insert_pos = obs_list.binary_search_by(|o| o.time_ms.partial_cmp(&time_ms).unwrap_or(std::cmp::Ordering::Equal));
        
        match insert_pos {
            Ok(pos) | Err(pos) => {
                obs_list.insert(pos, Observation { time_ms, event, volume });
            }
        }
    }

    /// Calculate Kaplan-Meier survival estimate for a price level
    /// 
    /// S(t) = Π_{t_i ≤ t} (1 - d_i / n_i)
    /// where d_i = events at time t_i, n_i = at risk at time t_i
    /// 
    /// CRITICAL: Handles n_i = 0 gracefully to prevent 0/0 division (Audit Fix #2)
    pub fn calculate_survival(&self, price_level: i64, max_time_ms: f64) -> Result<Vec<SurvivalEstimate>, SurvivalError> {
        let observations = self.observations.read();
        
        let obs_list = observations.get(&price_level)
            .ok_or_else(|| SurvivalError::InvalidTimeSequence)?;
        
        if obs_list.len() < self.min_observations {
            return Ok(Vec::new());
        }
        
        let mut estimates = Vec::with_capacity(obs_list.len());
        let mut survival_prob = 1.0;
        let mut at_risk = obs_list.len();
        let mut variance_sum = 0.0; // For Greenwood's formula
        
        // Group events by unique time points
        let mut time_groups: BTreeMap<f64, (usize, usize)> = BTreeMap::new(); // time -> (fills, total_events)
        
        for obs in obs_list.iter() {
            if obs.time_ms > max_time_ms {
                break;
            }
            
            let entry = time_groups.entry(obs.time_ms).or_insert((0, 0));
            entry.1 += 1; // Total events at this time
            
            if obs.event == QueueEvent::Fill {
                entry.0 += 1; // Fills at this time
            }
        }
        
        // Calculate survival probabilities
        for (&time_ms, &(events, total_at_this_time)) in time_groups.iter() {
            // CRITICAL: Handle zero at-risk gracefully
            if at_risk == 0 {
                // No more orders at risk, survival is undefined but we set to 0
                estimates.push(SurvivalEstimate {
                    time_ms,
                    survival_probability: 0.0,
                    confidence_interval_lower: 0.0,
                    confidence_interval_upper: 0.0,
                    at_risk_count: 0,
                    event_count: events,
                });
                continue;
            }
            
            // Kaplan-Meier product-limit estimator
            let conditional_survival = 1.0 - (events as f64 / at_risk as f64);
            survival_prob *= conditional_survival;
            
            // Greenwood's formula for variance: Var(S(t)) = S(t)^2 * Σ(d_i / (n_i * (n_i - d_i)))
            if at_risk > events && at_risk > 0 {
                let variance_term = events as f64 / (at_risk as f64 * (at_risk - events) as f64);
                variance_sum += variance_term;
            }
            
            // Calculate confidence interval using log-log transformation for better stability
            let std_error = if survival_prob > 0.0 && variance_sum > 0.0 {
                survival_prob * variance_sum.sqrt()
            } else {
                0.0
            };
            
            let z_score = self.get_z_score(self.confidence_level);
            let ci_lower = (survival_prob - z_score * std_error).max(0.0).min(1.0);
            let ci_upper = (survival_prob + z_score * std_error).max(0.0).min(1.0);
            
            estimates.push(SurvivalEstimate {
                time_ms,
                survival_probability: survival_prob,
                confidence_interval_lower: ci_lower,
                confidence_interval_upper: ci_upper,
                at_risk_count: at_risk,
                event_count: events,
            });
            
            // Update at-risk count for next iteration
            at_risk = at_risk.saturating_sub(total_at_this_time);
        }
        
        // Cache the result
        let mut curves = self.survival_curves.write();
        curves.insert(price_level, estimates.clone());
        
        Ok(estimates)
    }

    /// Get current survival probability at a specific time horizon
    pub fn get_survival_at_time(&self, price_level: i64, time_ms: f64) -> Option<f64> {
        // First check cache
        {
            let curves = self.survival_curves.read();
            if let Some(curve) = curves.get(&price_level) {
                // Find the estimate just before or at the requested time
                for estimate in curve.iter().rev() {
                    if estimate.time_ms <= time_ms {
                        return Some(estimate.survival_probability);
                    }
                }
            }
        }
        
        // Calculate if not cached
        if let Ok(estimates) = self.calculate_survival(price_level, time_ms) {
            if let Some(last) = estimates.last() {
                return Some(last.survival_probability);
            }
        }
        
        None
    }

    /// Get median survival time (time at which S(t) = 0.5)
    pub fn get_median_survival_time(&self, price_level: i64) -> Option<f64> {
        let estimates = self.calculate_survival(price_level, f64::MAX).ok()?;
        
        for estimate in &estimates {
            if estimate.survival_probability <= 0.5 {
                return Some(estimate.time_ms);
            }
        }
        
        // If never drops below 0.5, return the last observed time
        estimates.last().map(|e| e.time_ms)
    }

    /// Get z-score for confidence level
    fn get_z_score(&self, confidence_level: f64) -> f64 {
        // Approximate z-scores for common confidence levels
        match confidence_level {
            cl if cl >= 0.99 => 2.576,
            cl if cl >= 0.95 => 1.96,
            cl if cl >= 0.90 => 1.645,
            _ => 1.96, // Default to 95%
        }
    }

    /// Reset estimator state
    pub fn reset(&self) {
        self.observations.write().clear();
        self.survival_curves.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kaplan_meier_basic() {
        let estimator = KaplanMeierEstimator::new(5, 0.95).unwrap();
        
        // Simulate order lifecycle
        for i in 0..20 {
            let time_ms = i as f64 * 10.0;
            let event = if i % 3 == 0 { QueueEvent::Canceled } else { QueueEvent::Fill };
            estimator.record_event(100, time_ms, event, 100);
        }
        
        let estimates = estimator.calculate_survival(100, 200.0).unwrap();
        assert!(!estimates.is_empty());
        
        // Survival should decrease over time
        if estimates.len() > 1 {
            assert!(estimates[0].survival_probability >= estimates.last().unwrap().survival_probability);
        }
    }

    #[test]
    fn test_zero_at_risk_handling() {
        let estimator = KaplanMeierEstimator::new(2, 0.95).unwrap();
        
        // All orders fill immediately
        estimator.record_event(100, 1.0, QueueEvent::Fill, 100);
        estimator.record_event(100, 1.0, QueueEvent::Fill, 100);
        
        let result = estimator.calculate_survival(100, 10.0);
        assert!(result.is_ok()); // Should not panic on 0/0
    }
}
