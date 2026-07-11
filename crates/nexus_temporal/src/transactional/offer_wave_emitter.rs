//! Offer Wave Emitter for Transactional Interpretation
//! 
//! Implements the forward-in-time "offer wave" representing
//! the bot's intent to trade in Cramer's TI formalism.

/// Minimum amplitude threshold for offer waves
const MIN_AMPLITUDE_THRESHOLD: f64 = 1e-10;

/// Maximum number of offer wave components
const MAX_WAVE_COMPONENTS: usize = 256;

/// Offer wave component with amplitude and phase
#[derive(Debug, Clone)]
pub struct OfferWaveComponent {
    /// Complex amplitude of this component
    pub amplitude_re: f64,
    pub amplitude_im: f64,
    /// Frequency component (inverse nanoseconds)
    pub frequency: f64,
    /// Phase offset (radians)
    pub phase: f64,
    /// Target venue/liquidity pool identifier
    pub target_id: u32,
}

/// Complete offer wave emitted by the trading system
#[derive(Debug, Clone)]
pub struct OfferWave {
    /// Components of the offer wave
    pub components: Vec<OfferWaveComponent>,
    /// Total wave amplitude (norm)
    pub total_amplitude: f64,
    /// Emission timestamp (nanoseconds)
    pub emission_time_ns: u64,
    /// Intent type (buy/sell)
    pub is_buy_intent: bool,
    /// Order size represented by this wave
    pub order_size: f64,
    /// Price limit associated with the offer
    pub limit_price: f64,
}

impl OfferWave {
    /// Calculate wave function value at time t
    pub fn evaluate_at_time(&self, t_ns: u64) -> (f64, f64) {
        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        
        let delta_t = (t_ns as i64 - self.emission_time_ns as i64) as f64;
        
        for component in &self.components {
            let omega_t = component.frequency * delta_t + component.phase;
            let cos_val = omega_t.cos();
            let sin_val = omega_t.sin();
            
            sum_re += component.amplitude_re * cos_val - component.amplitude_im * sin_val;
            sum_im += component.amplitude_re * sin_val + component.amplitude_im * cos_val;
        }
        
        (sum_re, sum_im)
    }
    
    /// Get probability density |psi|^2 at time t
    pub fn probability_density(&self, t_ns: u64) -> f64 {
        let (re, im) = self.evaluate_at_time(t_ns);
        re * re + im * im
    }
}

/// Offer Wave Emitter for generating trading intent waves
pub struct OfferWaveEmitter {
    /// Base amplitude for emitted waves
    base_amplitude: f64,
    /// Number of frequency components to emit
    num_components: usize,
    /// Current phase reference
    phase_reference: f64,
    /// Emission counter
    emission_count: u64,
}

impl OfferWaveEmitter {
    /// Create a new offer wave emitter
    pub fn new() -> Self {
        Self {
            base_amplitude: 1.0,
            num_components: 8,
            phase_reference: 0.0,
            emission_count: 0,
        }
    }

    /// Create emitter with custom parameters
    pub fn with_params(base_amplitude: f64, num_components: usize) -> Self {
        Self {
            base_amplitude: base_amplitude.max(MIN_AMPLITUDE_THRESHOLD),
            num_components: num_components.min(MAX_WAVE_COMPONENTS).max(1),
            phase_reference: 0.0,
            emission_count: 0,
        }
    }

    /// Set base amplitude for emitted waves
    pub fn set_base_amplitude(&mut self, amplitude: f64) {
        self.base_amplitude = amplitude.max(MIN_AMPLITUDE_THRESHOLD);
    }

    /// Emit an offer wave for a trading intent
    /// 
    /// # Arguments
    /// * `is_buy` - True for buy intent, false for sell
    /// * `size` - Order size
    /// * `limit_price` - Limit price for the order
    /// * `target_venues` - List of target venue IDs
    /// * `current_time_ns` - Current timestamp
    /// 
    /// # Returns
    /// Generated OfferWave representing the trading intent
    pub fn emit_offer_wave(&mut self,
                           is_buy: bool,
                           size: f64,
                           limit_price: f64,
                           target_venues: &[u32],
                           current_time_ns: u64) -> OfferWave {
        self.emission_count += 1;
        
        // Size-dependent amplitude scaling
        let size_factor = size.sqrt().max(1.0);
        let scaled_amplitude = self.base_amplitude * size_factor;
        
        // Generate frequency components based on order characteristics
        let mut components = Vec::with_capacity(self.num_components.min(target_venues.len().max(1)));
        
        // Primary component at fundamental frequency
        let fundamental_freq = 1.0 / (size * 1000.0).max(1e-6); // Frequency scales with size
        
        for i in 0..self.num_components {
            let frequency = fundamental_freq * (i + 1) as f64;
            let phase = self.phase_reference + (i as f64 * std::f64::consts::PI / 4.0);
            
            // Amplitude decreases for higher harmonics
            let harmonic_amplitude = scaled_amplitude / (i + 1) as f64;
            
            // Distribute across target venues
            let target_id = if target_venues.is_empty() {
                0
            } else {
                target_venues[i % target_venues.len()]
            };
            
            components.push(OfferWaveComponent {
                amplitude_re: harmonic_amplitude,
                amplitude_im: 0.0, // Start with real amplitudes
                frequency,
                phase,
                target_id,
            });
        }
        
        // Calculate total amplitude (L2 norm)
        let total_amplitude = components.iter()
            .map(|c| c.amplitude_re.powi(2) + c.amplitude_im.powi(2))
            .sum::<f64>()
            .sqrt();
        
        // Update phase reference for next emission
        self.phase_reference += std::f64::consts::PI / 8.0;
        if self.phase_reference > std::f64::consts::TAU {
            self.phase_reference -= std::f64::consts::TAU;
        }
        
        OfferWave {
            components,
            total_amplitude,
            emission_time_ns: current_time_ns,
            is_buy_intent: is_buy,
            order_size: size,
            limit_price,
        }
    }

    /// Modulate offer wave based on market conditions
    /// 
    /// # Arguments
    /// * `offer_wave` - Original offer wave
    /// * `volatility` - Current market volatility
    /// * `spread_bps` - Current bid-ask spread in basis points
    /// 
    /// # Returns
    /// Modulated offer wave with adjusted amplitudes
    pub fn modulate_wave(&self, offer_wave: &OfferWave,
                         volatility: f64, spread_bps: f64) -> OfferWave {
        // Volatility damping factor
        let vol_damping = 1.0 / (1.0 + volatility * 10.0);
        
        // Spread penalty (wider spreads reduce effective amplitude)
        let spread_penalty = 1.0 / (1.0 + spread_bps / 100.0);
        
        let modulation_factor = vol_damping * spread_penalty;
        
        let mut modulated_components = Vec::with_capacity(offer_wave.components.len());
        
        for component in &offer_wave.components {
            modulated_components.push(OfferWaveComponent {
                amplitude_re: component.amplitude_re * modulation_factor,
                amplitude_im: component.amplitude_im * modulation_factor,
                frequency: component.frequency,
                phase: component.phase,
                target_id: component.target_id,
            });
        }
        
        let total_amplitude = offer_wave.total_amplitude * modulation_factor;
        
        OfferWave {
            components: modulated_components,
            total_amplitude,
            emission_time_ns: offer_wave.emission_time_ns,
            is_buy_intent: offer_wave.is_buy_intent,
            order_size: offer_wave.order_size,
            limit_price: offer_wave.limit_price,
        }
    }

    /// Get emission statistics
    pub fn emission_count(&self) -> u64 {
        self.emission_count
    }

    /// Reset emitter state
    pub fn reset(&mut self) {
        self.phase_reference = 0.0;
        self.emission_count = 0;
    }
}

impl Default for OfferWaveEmitter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emitter_creation() {
        let emitter = OfferWaveEmitter::new();
        assert!(emitter.base_amplitude >= MIN_AMPLITUDE_THRESHOLD);
        assert_eq!(emitter.emission_count(), 0);
    }

    #[test]
    fn test_emit_basic_wave() {
        let mut emitter = OfferWaveEmitter::new();
        
        let wave = emitter.emit_offer_wave(
            true,  // Buy
            100.0, // Size
            100.0, // Limit price
            &[1, 2, 3], // Target venues
            1000000, // Time
        );
        
        assert!(!wave.components.is_empty());
        assert!(wave.total_amplitude > 0.0);
        assert!(wave.is_buy_intent);
        assert_eq!(wave.order_size, 100.0);
    }

    #[test]
    fn test_wave_evaluation() {
        let mut emitter = OfferWaveEmitter::new();
        
        let wave = emitter.emit_offer_wave(true, 100.0, 100.0, &[1], 1000000);
        
        // Evaluate at emission time
        let (re, im) = wave.evaluate_at_time(1000000);
        assert!(re.abs() > 0.0 || im.abs() > 0.0);
        
        // Probability should be non-negative
        let prob = wave.probability_density(1000000);
        assert!(prob >= 0.0);
    }

    #[test]
    fn test_modulation() {
        let mut emitter = OfferWaveEmitter::new();
        
        let original = emitter.emit_offer_wave(true, 100.0, 100.0, &[1], 1000000);
        let modulated = emitter.modulate_wave(&original, 0.2, 10.0);
        
        // Modulated amplitude should be lower
        assert!(modulated.total_amplitude <= original.total_amplitude);
    }
}
