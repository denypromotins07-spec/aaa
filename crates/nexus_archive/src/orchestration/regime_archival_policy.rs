//! Regime Archival Policy
//! 
//! Implements regime-based archival policies that trigger when macro economic regimes change.
//! Automatically archives data when significant market regime transitions are detected.

use crate::orchestration::eternal_archive_manager::{StorageMedium, ArchiveError};
use thiserror::Error;

/// Macro economic regime types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroRegime {
    ZeroInterestRate,
    RisingRates,
    HighInflation,
    Deflation,
    Stagflation,
    FinancialCrisis,
    Recovery,
    Expansion,
    Recession,
    Unknown,
}

impl MacroRegime {
    /// Get numeric tag for archival
    pub fn to_tag(self) -> u32 {
        match self {
            MacroRegime::ZeroInterestRate => 1,
            MacroRegime::RisingRates => 2,
            MacroRegime::HighInflation => 3,
            MacroRegime::Deflation => 4,
            MacroRegime::Stagflation => 5,
            MacroRegime::FinancialCrisis => 6,
            MacroRegime::Recovery => 7,
            MacroRegime::Expansion => 8,
            MacroRegime::Recession => 9,
            MacroRegime::Unknown => 0,
        }
    }

    /// Create from numeric tag
    pub fn from_tag(tag: u32) -> Self {
        match tag {
            1 => MacroRegime::ZeroInterestRate,
            2 => MacroRegime::RisingRates,
            3 => MacroRegime::HighInflation,
            4 => MacroRegime::Deflation,
            5 => MacroRegime::Stagflation,
            6 => MacroRegime::FinancialCrisis,
            7 => MacroRegime::Recovery,
            8 => MacroRegime::Expansion,
            9 => MacroRegime::Recession,
            _ => MacroRegime::Unknown,
        }
    }
}

/// Volatility regime for adaptive archival
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolatilityRegime {
    Low,      // VIX < 15
    Normal,   // VIX 15-25
    Elevated, // VIX 25-40
    Crisis,   // VIX > 40
}

impl VolatilityRegime {
    pub fn from_vix(vix: f64) -> Self {
        if vix < 15.0 {
            VolatilityRegime::Low
        } else if vix < 25.0 {
            VolatilityRegime::Normal
        } else if vix < 40.0 {
            VolatilityRegime::Elevated
        } else {
            VolatilityRegime::Crisis
        }
    }
}

#[derive(Error, Debug)]
pub enum PolicyError {
    #[error("Invalid regime transition")]
    InvalidTransition,
    #[error("Archival threshold not met")]
    ThresholdNotMet,
    #[error("Archive error: {0}")]
    ArchiveError(String),
    #[error("Buffer overflow")]
    BufferOverflow,
}

/// Archival decision result
#[derive(Debug, Clone)]
pub struct ArchivalDecision {
    pub should_archive: bool,
    pub priority: u8,          // 0-255, higher = more urgent
    pub recommended_medium: StorageMedium,
    pub reason: &'static str,
}

impl Default for ArchivalDecision {
    fn default() -> Self {
        Self {
            should_archive: false,
            priority: 0,
            recommended_medium: StorageMedium::Optical5D,
            reason: "No archival needed",
        }
    }
}

/// Regime transition tracker
#[derive(Debug, Clone)]
pub struct RegimeTransition {
    pub from_regime: MacroRegime,
    pub to_regime: MacroRegime,
    pub transition_timestamp_ns: u64,
    pub confidence: f64,
}

/// Regime Archival Policy engine
pub struct RegimeArchivalPolicy {
    current_macro_regime: MacroRegime,
    current_volatility_regime: VolatilityRegime,
    last_transition: Option<RegimeTransition>,
    archive_threshold_data_mb: u64,
    accumulated_data_mb: u64,
    pending_archivals: Box<[ArchivalDecision]>,
    pending_count: usize,
}

impl RegimeArchivalPolicy {
    /// Create a new policy engine
    pub fn new(archive_threshold_mb: u64, max_pending: usize) -> Self {
        Self {
            current_macro_regime: MacroRegime::Unknown,
            current_volatility_regime: VolatilityRegime::Normal,
            last_transition: None,
            archive_threshold_data_mb: archive_threshold_mb,
            accumulated_data_mb: 0,
            pending_archivals: vec![ArchivalDecision::default(); max_pending].into_boxed_slice(),
            pending_count: 0,
        }
    }

    /// Update the current macro regime and check for archival triggers
    pub fn update_regime(
        &mut self,
        new_regime: MacroRegime,
        vix: f64,
        timestamp_ns: u64,
    ) -> Result<Option<ArchivalDecision>, PolicyError> {
        let volatility_regime = VolatilityRegime::from_vix(vix);
        self.current_volatility_regime = volatility_regime;

        // Check for regime transition
        if new_regime != self.current_macro_regime {
            let transition = RegimeTransition {
                from_regime: self.current_macro_regime,
                to_regime: new_regime,
                transition_timestamp_ns: timestamp_ns,
                confidence: 0.95, // Would be calculated from regime detector
            };

            self.last_transition = Some(transition);
            self.current_macro_regime = new_regime;

            // Regime transition triggers immediate archival
            return Ok(Some(self.decide_archival_for_transition(new_regime, volatility_regime)));
        }

        // Check accumulated data threshold
        if self.accumulated_data_mb >= self.archive_threshold_data_mb {
            self.accumulated_data_mb = 0;
            return Ok(Some(self.decide_archival_for_threshold()));
        }

        Ok(None)
    }

    /// Decide archival strategy for a regime transition
    fn decide_archival_for_transition(
        &self,
        regime: MacroRegime,
        volatility: VolatilityRegime,
    ) -> ArchivalDecision {
        // Crisis regimes get highest priority and DNA storage (longest lasting)
        match regime {
            MacroRegime::FinancialCrisis | MacroRegime::Recession => ArchivalDecision {
                should_archive: true,
                priority: 255,
                recommended_medium: StorageMedium::Dna,
                reason: "Crisis regime detected - permanent archival required",
            },
            MacroRegime::Stagflation | MacroRegime::HighInflation => ArchivalDecision {
                should_archive: true,
                priority: 200,
                recommended_medium: StorageMedium::Dna,
                reason: "High inflation regime - long-term archival",
            },
            MacroRegime::ZeroInterestRate => ArchivalDecision {
                should_archive: true,
                priority: 150,
                recommended_medium: StorageMedium::Holographic,
                reason: "ZIRP era ended - historical archival",
            },
            _ => {
                // For normal transitions, consider volatility
                match volatility {
                    VolatilityRegime::Crisis => ArchivalDecision {
                        should_archive: true,
                        priority: 180,
                        recommended_medium: StorageMedium::Optical5D,
                        reason: "High volatility - archive for analysis",
                    },
                    VolatilityRegime::Elevated => ArchivalDecision {
                        should_archive: true,
                        priority: 100,
                        recommended_medium: StorageMedium::Holographic,
                        reason: "Elevated volatility - medium-term storage",
                    },
                    _ => ArchivalDecision {
                        should_archive: false,
                        priority: 10,
                        recommended_medium: StorageMedium::Optical5D,
                        reason: "Normal transition - no immediate action",
                    },
                }
            }
        }
    }

    /// Decide archival for threshold-based trigger
    fn decide_archival_for_threshold(&self) -> ArchivalDecision {
        ArchivalDecision {
            should_archive: true,
            priority: 50,
            recommended_medium: StorageMedium::Optical5D,
            reason: "Data threshold reached - routine archival",
        }
    }

    /// Add data volume to accumulator
    pub fn add_data_volume(&mut self, size_mb: u64) -> Result<(), PolicyError> {
        self.accumulated_data_mb += size_mb;
        if self.accumulated_data_mb > self.accumulated_data_mb {
            // Overflow check
            return Err(PolicyError::BufferOverflow);
        }
        Ok(())
    }

    /// Queue an archival decision for processing
    pub fn queue_archival(&mut self, decision: ArchivalDecision) -> Result<(), PolicyError> {
        if self.pending_count >= self.pending_archivals.len() {
            return Err(PolicyError::BufferOverflow);
        }
        self.pending_archivals[self.pending_count] = decision;
        self.pending_count += 1;
        Ok(())
    }

    /// Get pending archivals sorted by priority
    pub fn get_pending_by_priority(&self) -> Vec<&ArchivalDecision> {
        let mut decisions: Vec<&ArchivalDecision> = self.pending_archivals[..self.pending_count]
            .iter()
            .filter(|d| d.should_archive)
            .collect();
        
        decisions.sort_by_key(|d| std::cmp::Reverse(d.priority));
        decisions
    }

    /// Clear processed archivals
    pub fn clear_processed(&mut self, count: usize) {
        if count >= self.pending_count {
            self.pending_count = 0;
        } else {
            // Shift remaining down
            for i in 0..(self.pending_count - count) {
                self.pending_archivals[i] = self.pending_archivals[i + count];
            }
            self.pending_count -= count;
        }
    }

    /// Get current macro regime
    pub fn current_regime(&self) -> MacroRegime {
        self.current_macro_regime
    }

    /// Get current volatility regime
    pub fn current_volatility(&self) -> VolatilityRegime {
        self.current_volatility_regime
    }

    /// Get accumulated data volume
    pub fn accumulated_data_mb(&self) -> u64 {
        self.accumulated_data_mb
    }

    /// Get last regime transition
    pub fn last_transition(&self) -> Option<&RegimeTransition> {
        self.last_transition.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_tag_conversion() {
        assert_eq!(MacroRegime::FinancialCrisis.to_tag(), 6);
        assert_eq!(MacroRegime::from_tag(6), MacroRegime::FinancialCrisis);
    }

    #[test]
    fn test_volatility_from_vix() {
        assert_eq!(VolatilityRegime::from_vix(10.0), VolatilityRegime::Low);
        assert_eq!(VolatilityRegime::from_vix(20.0), VolatilityRegime::Normal);
        assert_eq!(VolatilityRegime::from_vix(35.0), VolatilityRegime::Elevated);
        assert_eq!(VolatilityRegime::from_vix(50.0), VolatilityRegime::Crisis);
    }

    #[test]
    fn test_crisis_regime_archival() {
        let mut policy = RegimeArchivalPolicy::new(1000, 10);
        
        let decision = policy.update_regime(
            MacroRegime::FinancialCrisis,
            45.0,
            1000000,
        ).unwrap().unwrap();
        
        assert!(decision.should_archive);
        assert_eq!(decision.priority, 255);
        assert_eq!(decision.recommended_medium, StorageMedium::Dna);
    }

    #[test]
    fn test_threshold_archival() {
        let mut policy = RegimeArchivalPolicy::new(100, 10);
        
        // Add data to exceed threshold
        policy.add_data_volume(150).unwrap();
        
        // Same regime, but threshold exceeded
        let decision = policy.update_regime(
            MacroRegime::Expansion,
            15.0,
            2000000,
        ).unwrap();
        
        // Should trigger threshold archival
        assert!(policy.accumulated_data_mb() == 0); // Reset after threshold
    }

    #[test]
    fn test_pending_queue() {
        let mut policy = RegimeArchivalPolicy::new(1000, 10);
        
        let decision = ArchivalDecision {
            should_archive: true,
            priority: 100,
            recommended_medium: StorageMedium::Holographic,
            reason: "Test",
        };
        
        policy.queue_archival(decision).unwrap();
        
        let pending = policy.get_pending_by_priority();
        assert_eq!(pending.len(), 1);
    }
}
