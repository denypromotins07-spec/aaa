//! Confirmation Wave Receiver for Transactional Interpretation
//! 
//! Implements the backward-in-time "confirmation wave" representing
//! exchange/liquidity response in Cramer's TI formalism.

use crate::transactional::offer_wave_emitter::OfferWave;

/// Minimum confirmation threshold
const MIN_CONFIRMATION_THRESHOLD: f64 = 1e-8;

/// Maximum number of confirmation sources
const MAX_CONFIRMATION_SOURCES: usize = 128;

/// Single confirmation source from a venue
#[derive(Debug, Clone)]
pub struct ConfirmationSource {
    /// Source venue/liquidity pool ID
    pub venue_id: u32,
    /// Complex amplitude of confirmation (real part)
    pub amplitude_re: f64,
    /// Complex amplitude of confirmation (imaginary part)
    pub amplitude_im: f64,
    /// Available liquidity at this venue
    pub available_liquidity: f64,
    /// Best bid price at venue
    pub best_bid: f64,
    /// Best ask price at venue
    pub best_ask: f64,
    /// Latency from venue (nanoseconds)
    pub latency_ns: u64,
}

/// Aggregated confirmation wave from all venues
#[derive(Debug, Clone)]
pub struct ConfirmationWave {
    /// Individual confirmation sources
    pub sources: Vec<ConfirmationSource>,
    /// Total confirmation amplitude (norm)
    pub total_amplitude: f64,
    /// Phase relationship with offer wave
    pub phase_offset: f64,
    /// Timestamp when confirmation was received
    pub reception_time_ns: u64,
    /// Whether confirmation indicates executable liquidity
    pub is_executable: bool,
}

impl ConfirmationWave {
    /// Calculate interference with an offer wave
    /// 
    /// # Arguments
    /// * `offer_wave` - The corresponding offer wave
    /// * `eval_time_ns` - Time at which to evaluate interference
    /// 
    /// # Returns
    /// Interference amplitude (positive = constructive, negative = destructive)
    pub fn calculate_interference(&self, offer_wave: &OfferWave, eval_time_ns: u64) -> f64 {
        if self.sources.is_empty() {
            return 0.0;
        }

        // Get offer wave value at evaluation time
        let (offer_re, offer_im) = offer_wave.evaluate_at_time(eval_time_ns);
        
        // Sum confirmation amplitudes
        let mut conf_re = 0.0;
        let mut conf_im = 0.0;
        
        for source in &self.sources {
            conf_re += source.amplitude_re;
            conf_im += source.amplitude_im;
        }
        
        // Calculate interference (real part of inner product)
        let interference = offer_re * conf_re + offer_im * conf_im;
        
        interference
    }

    /// Get total available liquidity from all sources
    pub fn total_liquidity(&self) -> f64 {
        self.sources.iter().map(|s| s.available_liquidity).sum()
    }

    /// Get weighted average price from confirmation sources
    pub fn weighted_average_price(&self, reference_price: f64) -> f64 {
        if self.sources.is_empty() {
            return reference_price;
        }

        let total_weight: f64 = self.sources.iter()
            .map(|s| s.available_liquidity)
            .sum();
        
        if total_weight < 1e-15 {
            return reference_price;
        }

        let weighted_sum: f64 = self.sources.iter()
            .map(|s| {
                let mid_price = (s.best_bid + s.best_ask) / 2.0;
                mid_price * s.available_liquidity
            })
            .sum();
        
        weighted_sum / total_weight
    }
}

/// Confirmation Wave Receiver for processing venue responses
pub struct ConfirmationWaveReceiver {
    /// Minimum amplitude for valid confirmations
    min_amplitude: f64,
    /// Maximum allowed latency for confirmations (nanoseconds)
    max_latency_ns: u64,
    /// Number of active confirmation channels
    active_channels: usize,
    /// Received confirmation count
    confirmation_count: u64,
}

impl ConfirmationWaveReceiver {
    /// Create a new confirmation wave receiver
    pub fn new() -> Self {
        Self {
            min_amplitude: MIN_CONFIRMATION_THRESHOLD,
            max_latency_ns: 1_000_000, // 1 millisecond
            active_channels: 0,
            confirmation_count: 0,
        }
    }

    /// Create with custom parameters
    pub fn with_params(min_amplitude: f64, max_latency_ns: u64) -> Self {
        Self {
            min_amplitude: min_amplitude.max(MIN_CONFIRMATION_THRESHOLD),
            max_latency_ns,
            active_channels: 0,
            confirmation_count: 0,
        }
    }

    /// Set minimum confirmation amplitude threshold
    pub fn set_min_amplitude(&mut self, threshold: f64) {
        self.min_amplitude = threshold.max(MIN_CONFIRMATION_THRESHOLD);
    }

    /// Set maximum allowed latency
    pub fn set_max_latency(&mut self, latency_ns: u64) {
        self.max_latency_ns = latency_ns;
    }

    /// Receive and process confirmation from a single venue
    /// 
    /// # Arguments
    /// * `venue_id` - Venue identifier
    /// * `liquidity` - Available liquidity
    /// * `best_bid` - Best bid price
    /// * `best_ask` - Best ask price
    /// * `latency_ns` - Measured latency
    /// * `current_time_ns` - Current timestamp
    /// 
    /// # Returns
    /// ConfirmationSource if valid, None if filtered out
    pub fn receive_confirmation(&mut self,
                                 venue_id: u32,
                                 liquidity: f64,
                                 best_bid: f64,
                                 best_ask: f64,
                                 latency_ns: u64,
                                 current_time_ns: u64) -> Option<ConfirmationSource> {
        // Filter by latency
        if latency_ns > self.max_latency_ns {
            return None;
        }

        // Filter by minimum liquidity
        if liquidity < 1e-10 {
            return None;
        }

        // Validate prices
        if best_bid <= 0.0 || best_ask <= 0.0 || best_bid >= best_ask {
            return None;
        }

        // Calculate confirmation amplitude based on liquidity
        let amplitude = liquidity.sqrt() * 0.1;
        
        if amplitude < self.min_amplitude {
            return None;
        }

        self.active_channels += 1;
        self.confirmation_count += 1;

        Some(ConfirmationSource {
            venue_id,
            amplitude_re: amplitude,
            amplitude_im: 0.0, // Initially real
            available_liquidity: liquidity,
            best_bid,
            best_ask,
            latency_ns,
        })
    }

    /// Aggregate multiple confirmations into a single confirmation wave
    /// 
    /// # Arguments
    /// * `sources` - Vector of confirmation sources
    /// * `offer_wave` - Corresponding offer wave for phase calculation
    /// * `reception_time_ns` - Time when confirmations were received
    /// 
    /// # Returns
    /// Aggregated ConfirmationWave
    pub fn aggregate_confirmations(&self,
                                    sources: Vec<ConfirmationSource>,
                                    offer_wave: &OfferWave,
                                    reception_time_ns: u64) -> ConfirmationWave {
        if sources.is_empty() {
            return ConfirmationWave {
                sources: vec![],
                total_amplitude: 0.0,
                phase_offset: 0.0,
                reception_time_ns,
                is_executable: false,
            };
        }

        // Calculate total amplitude
        let total_amplitude: f64 = sources.iter()
            .map(|s| s.amplitude_re.powi(2) + s.amplitude_im.powi(2))
            .sum::<f64>()
            .sqrt();

        // Calculate phase offset relative to offer wave
        let phase_offset = self.calculate_phase_offset(&sources, offer_wave, reception_time_ns);

        // Determine if liquidity is executable
        let total_liquidity: f64 = sources.iter().map(|s| s.available_liquidity).sum();
        let is_executable = total_liquidity >= offer_wave.order_size * 0.9 
            && total_amplitude >= self.min_amplitude;

        ConfirmationWave {
            sources,
            total_amplitude,
            phase_offset,
            reception_time_ns,
            is_executable,
        }
    }

    /// Check if a confirmation wave forms a valid transaction with offer wave
    /// 
    /// A valid transaction requires:
    /// 1. Constructive interference (amplitude > threshold)
    /// 2. Sufficient liquidity
    /// 3. Acceptable latency
    pub fn validate_transaction(&self, 
                                confirmation: &ConfirmationWave,
                                offer_wave: &OfferWave) -> bool {
        if !confirmation.is_executable {
            return false;
        }

        // Check interference at reception time
        let interference = confirmation.calculate_interference(
            offer_wave, 
            confirmation.reception_time_ns
        );

        // Must have constructive interference
        if interference < self.min_amplitude {
            return false;
        }

        // Verify liquidity covers order
        let total_liquidity = confirmation.total_liquidity();
        if total_liquidity < offer_wave.order_size * 0.9 {
            return false;
        }

        true
    }

    /// Get active channel count
    pub fn active_channels(&self) -> usize {
        self.active_channels
    }

    /// Get total confirmation count
    pub fn confirmation_count(&self) -> u64 {
        self.confirmation_count
    }

    /// Reset receiver state
    pub fn reset(&mut self) {
        self.active_channels = 0;
        // Don't reset confirmation_count (historical stat)
    }

    // Internal: Calculate phase offset between confirmation and offer
    fn calculate_phase_offset(&self, sources: &[ConfirmationSource],
                              offer_wave: &OfferWave,
                              eval_time_ns: u64) -> f64 {
        if sources.is_empty() {
            return 0.0;
        }

        let (offer_re, offer_im) = offer_wave.evaluate_at_time(eval_time_ns);
        
        let conf_re: f64 = sources.iter().map(|s| s.amplitude_re).sum();
        let conf_im: f64 = sources.iter().map(|s| s.amplitude_im).sum();
        
        // Calculate phase difference using atan2
        let offer_phase = offer_im.atan2(offer_re);
        let conf_phase = conf_im.atan2(conf_re);
        
        let mut phase_diff = conf_phase - offer_phase;
        
        // Normalize to [-pi, pi]
        while phase_diff > std::f64::consts::PI {
            phase_diff -= std::f64::consts::TAU;
        }
        while phase_diff < -std::f64::consts::PI {
            phase_diff += std::f64::consts::TAU;
        }
        
        phase_diff
    }
}

impl Default for ConfirmationWaveReceiver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transactional::offer_wave_emitter::OfferWaveEmitter;

    #[test]
    fn test_receiver_creation() {
        let receiver = ConfirmationWaveReceiver::new();
        assert_eq!(receiver.active_channels(), 0);
        assert_eq!(receiver.confirmation_count(), 0);
    }

    #[test]
    fn test_receive_valid_confirmation() {
        let mut receiver = ConfirmationWaveReceiver::new();
        
        let source = receiver.receive_confirmation(
            1,      // Venue ID
            1000.0, // Liquidity
            99.9,   // Best bid
            100.1,  // Best ask
            100000, // Latency (100us)
            1000000, // Current time
        );
        
        assert!(source.is_some());
        let s = source.unwrap();
        assert_eq!(s.venue_id, 1);
        assert!(s.amplitude_re > 0.0);
    }

    #[test]
    fn test_receive_invalid_latency() {
        let mut receiver = ConfirmationWaveReceiver::new();
        
        let source = receiver.receive_confirmation(
            1,
            1000.0,
            99.9,
            100.1,
            10_000_000, // 10ms - exceeds default max
            1000000,
        );
        
        assert!(source.is_none());
    }

    #[test]
    fn test_aggregate_and_validate() {
        let mut emitter = OfferWaveEmitter::new();
        let mut receiver = ConfirmationWaveReceiver::new();
        
        let offer = emitter.emit_offer_wave(true, 100.0, 100.0, &[1], 1000000);
        
        let sources = vec![
            ConfirmationSource {
                venue_id: 1,
                amplitude_re: 1.0,
                amplitude_im: 0.0,
                available_liquidity: 500.0,
                best_bid: 99.9,
                best_ask: 100.1,
                latency_ns: 100000,
            },
            ConfirmationSource {
                venue_id: 2,
                amplitude_re: 0.8,
                amplitude_im: 0.0,
                available_liquidity: 600.0,
                best_bid: 99.85,
                best_ask: 100.15,
                latency_ns: 150000,
            },
        ];
        
        let confirmation = receiver.aggregate_confirmations(sources, &offer, 1000000);
        
        assert!(confirmation.total_amplitude > 0.0);
        assert!(confirmation.is_executable);
        
        // Validate transaction
        assert!(receiver.validate_transaction(&confirmation, &offer));
    }
}
