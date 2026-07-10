//! Predictive Maintenance Kalman Filter for Industrial Assets
//! 
//! Uses Kalman filtering to detect early signs of equipment failure.
//! Generates alpha signals for commodity supply shortage prediction.

#[derive(Debug, Clone)]
pub struct AssetState {
    pub asset_id: String,
    pub frequency_hz: f64,
    pub amplitude: f64,
    pub temperature_c: f64,
    pub vibration_rms: f64,
}

#[derive(Debug)]
pub struct KalmanFilter1D {
    x: f64, // State estimate
    p: f64, // Error covariance
    q: f64, // Process noise
    r: f64, // Measurement noise
}

impl KalmanFilter1D {
    pub fn new(initial_value: f64, process_noise: f64, measurement_noise: f64) -> Self {
        Self {
            x: initial_value,
            p: 1.0,
            q: process_noise,
            r: measurement_noise,
        }
    }

    pub fn update(&mut self, measurement: f64) -> f64 {
        // Prediction step
        let p_pred = self.p + self.q;

        // Update step
        let k = p_pred / (p_pred + self.r); // Kalman gain
        self.x = self.x + k * (measurement - self.x);
        self.p = (1.0 - k) * p_pred;

        self.x
    }

    pub fn get_estimate(&self) -> f64 {
        self.x
    }

    pub fn get_uncertainty(&self) -> f64 {
        self.p
    }
}

/// Multi-dimensional Kalman filter for asset health monitoring
pub struct PredictiveMaintenanceFilter {
    frequency_filter: KalmanFilter1D,
    amplitude_filter: KalmanFilter1D,
    temperature_filter: KalmanFilter1D,
    vibration_filter: KalmanFilter1D,
    
    baseline_state: AssetState,
    alert_thresholds: AlertThresholds,
    degradation_rate: f64,
}

#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub frequency_deviation_pct: f64,
    pub amplitude_increase_pct: f64,
    pub temperature_max_c: f64,
    pub vibration_max_g: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            frequency_deviation_pct: 5.0,
            amplitude_increase_pct: 20.0,
            temperature_max_c: 85.0,
            vibration_max_g: 10.0,
        }
    }
}

#[derive(Debug)]
pub enum MaintenanceAlert {
    Normal,
    Warning(String),
    Critical(String),
    ImminentFailure(String),
}

impl PredictiveMaintenanceFilter {
    pub fn new(baseline: AssetState, thresholds: AlertThresholds) -> Self {
        Self {
            frequency_filter: KalmanFilter1D::new(baseline.frequency_hz, 0.001, 0.1),
            amplitude_filter: KalmanFilter1D::new(baseline.amplitude, 0.001, 0.1),
            temperature_filter: KalmanFilter1D::new(baseline.temperature_c, 0.001, 0.5),
            vibration_filter: KalmanFilter1D::new(baseline.vibration_rms, 0.001, 0.2),
            baseline_state: baseline,
            alert_thresholds: thresholds,
            degradation_rate: 0.0,
        }
    }

    /// Process new sensor reading and return maintenance alert
    pub fn process_reading(&mut self, reading: &AssetState) -> MaintenanceAlert {
        // Update Kalman filters
        let freq_est = self.frequency_filter.update(reading.frequency_hz);
        let amp_est = self.amplitude_filter.update(reading.amplitude);
        let temp_est = self.temperature_filter.update(reading.temperature_c);
        let vib_est = self.vibration_filter.update(reading.vibration_rms);

        // Calculate deviations from baseline
        let freq_deviation_pct = ((freq_est - self.baseline_state.frequency_hz).abs() 
            / self.baseline_state.frequency_hz) * 100.0;
        let amp_increase_pct = ((amp_est - self.baseline_state.amplitude).abs() 
            / self.baseline_state.amplitude) * 100.0;

        // Update degradation rate estimate
        self.degradation_rate = 0.95 * self.degradation_rate + 0.05 * freq_deviation_pct;

        // Check thresholds
        let mut alerts = Vec::new();

        if freq_deviation_pct > self.alert_thresholds.frequency_deviation_pct {
            alerts.push(format!("Frequency deviation: {:.1}%", freq_deviation_pct));
        }

        if amp_increase_pct > self.alert_thresholds.amplitude_increase_pct {
            alerts.push(format!("Amplitude increase: {:.1}%", amp_increase_pct));
        }

        if temp_est > self.alert_thresholds.temperature_max_c {
            alerts.push(format!("High temperature: {:.1}°C", temp_est));
        }

        if vib_est > self.alert_thresholds.vibration_max_g {
            alerts.push(format!("High vibration: {:.2}g", vib_est));
        }

        // Determine alert level
        if alerts.is_empty() {
            MaintenanceAlert::Normal
        } else if alerts.len() >= 3 || temp_est > self.alert_thresholds.temperature_max_c + 15.0 {
            MaintenanceAlert::ImminentFailure(alerts.join("; "))
        } else if alerts.len() >= 2 {
            MaintenanceAlert::Critical(alerts.join("; "))
        } else {
            MaintenanceAlert::Warning(alerts.join("; "))
        }
    }

    /// Estimate remaining useful life (RUL) in hours
    pub fn estimate_rul(&self) -> Option<f64> {
        if self.degradation_rate <= 0.0 {
            return None;
        }

        // Simple linear extrapolation
        // Assuming failure at 50% deviation
        let failure_threshold = 50.0;
        let remaining_margin = failure_threshold - self.degradation_rate;
        
        if remaining_margin <= 0.0 {
            return Some(0.0);
        }

        // Estimate hours until failure based on current degradation rate
        let hours_to_failure = remaining_margin / (self.degradation_rate * 0.01);
        Some(hours_to_failure.max(0.0))
    }

    /// Get current state estimates
    pub fn get_current_state(&self) -> AssetState {
        AssetState {
            asset_id: self.baseline_state.asset_id.clone(),
            frequency_hz: self.frequency_filter.get_estimate(),
            amplitude: self.amplitude_filter.get_estimate(),
            temperature_c: self.temperature_filter.get_estimate(),
            vibration_rms: self.vibration_filter.get_estimate(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_filter_smoothing() {
        let mut kf = KalmanFilter1D::new(100.0, 0.001, 0.5);
        
        // Noisy measurements around 100
        let measurements = vec![98.0, 102.0, 99.0, 101.0, 100.0, 97.0, 103.0];
        
        for m in measurements {
            kf.update(m);
        }
        
        let estimate = kf.get_estimate();
        
        // Estimate should be closer to true value than noisy measurements
        assert!((estimate - 100.0).abs() < 2.0);
    }

    #[test]
    fn test_maintenance_alert_generation() {
        let baseline = AssetState {
            asset_id: "MOTOR-001".to_string(),
            frequency_hz: 60.0,
            amplitude: 1.0,
            temperature_c: 45.0,
            vibration_rms: 2.0,
        };

        let mut filter = PredictiveMaintenanceFilter::new(baseline, AlertThresholds::default());

        // Normal reading
        let normal_reading = AssetState {
            asset_id: "MOTOR-001".to_string(),
            frequency_hz: 60.1,
            amplitude: 1.05,
            temperature_c: 46.0,
            vibration_rms: 2.1,
        };
        
        let alert = filter.process_reading(&normal_reading);
        assert!(matches!(alert, MaintenanceAlert::Normal));

        // Abnormal reading with high temperature
        let hot_reading = AssetState {
            asset_id: "MOTOR-001".to_string(),
            frequency_hz: 65.0,
            amplitude: 1.5,
            temperature_c: 90.0,
            vibration_rms: 5.0,
        };
        
        let alert = filter.process_reading(&hot_reading);
        assert!(matches!(alert, MaintenanceAlert::Critical(_) | MaintenanceAlert::Warning(_)));
    }
}
