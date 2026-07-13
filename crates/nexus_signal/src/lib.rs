//! NEXUS-OMEGA Stage 3: Signal Processing & Filtering
//! 
//! This crate provides SIMD-accelerated signal processing including:
//! - Kalman filtering for signal smoothing
//! - Adaptive noise estimation
//! - Zero-allocation rolling statistics

pub mod filters;

pub use filters::{
    SimdKalmanFilter1D, SimdKalmanBatch, KalmanConfig,
    OnlineVarianceEstimator, AdaptiveNoiseEstimator,
    SignalSmoother, SignalSmootherConfig,
};

/// Signal routing output structure for Stage 4 OMS integration
#[derive(Debug, Clone, Copy)]
pub struct ProcessedSignal {
    /// Smoothed OBI signal [-1, 1]
    pub obi: f64,
    /// Smoothed VPIN toxicity [0, 1]
    pub vpin: f64,
    /// Whether market is toxic
    pub is_toxic: bool,
    /// Recommended spread multiplier
    pub spread_multiplier: f64,
    /// Signal timestamp
    pub timestamp_ns: u64,
    /// Signal quality score [0, 1]
    pub quality: f64,
}

impl ProcessedSignal {
    pub fn new(obi: f64, vpin: f64, timestamp_ns: u64) -> Self {
        let is_toxic = vpin > 0.7;
        let spread_multiplier = if is_toxic {
            1.0 + (vpin * 2.0)
        } else {
            1.0
        };
        
        let quality = if is_toxic {
            (1.0 - vpin) * obi.abs()
        } else {
            obi.abs()
        };

        Self {
            obi,
            vpin,
            is_toxic,
            spread_multiplier,
            timestamp_ns,
            quality,
        }
    }

    /// Check if signal is valid for trading
    pub fn is_tradeable(&self, min_quality: f64) -> bool {
        self.quality >= min_quality && !self.is_toxic
    }

    /// Get adjusted signal accounting for toxicity
    pub fn adjusted_signal(&self) -> f64 {
        if self.is_toxic {
            self.obi * (1.0 - self.vpin * 0.5)
        } else {
            self.obi
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processed_signal_basic() {
        let signal = ProcessedSignal::new(0.5, 0.3, 1000);
        
        assert_eq!(signal.obi, 0.5);
        assert_eq!(signal.vpin, 0.3);
        assert!(!signal.is_toxic);
        assert_eq!(signal.spread_multiplier, 1.0);
        assert!(signal.quality > 0.0);
    }

    #[test]
    fn test_processed_signal_toxic() {
        let signal = ProcessedSignal::new(0.8, 0.9, 1000);
        
        assert!(signal.is_toxic);
        assert!(signal.spread_multiplier > 2.0);
        assert!(!signal.is_tradeable(0.1));
    }

    #[test]
    fn test_processed_signal_adjusted() {
        let clean_signal = ProcessedSignal::new(0.6, 0.1, 1000);
        let toxic_signal = ProcessedSignal::new(0.6, 0.8, 1000);
        
        // Clean signal should have full strength
        assert!((clean_signal.adjusted_signal() - 0.6).abs() < 0.01);
        
        // Toxic signal should have reduced strength
        assert!(toxic_signal.adjusted_signal() < 0.6);
    }
}
