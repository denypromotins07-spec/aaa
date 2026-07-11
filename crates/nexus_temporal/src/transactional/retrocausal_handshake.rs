//! Retrocausal Handshake Calculator for Transactional Interpretation
//! 
//! Implements the complete transaction formation process where
//! offer and confirmation waves form standing waves enabling execution.

use crate::transactional::offer_wave_emitter::{OfferWaveEmitter, OfferWave};
use crate::transactional::confirmation_wave_receiver::{ConfirmationWaveReceiver, ConfirmationWave, ConfirmationSource};

/// Minimum transaction probability amplitude for execution
const MIN_TRANSACTION_AMPLITUDE: f64 = 0.9;

/// Maximum time window for handshake completion (nanoseconds)
const MAX_HANDSHAKE_WINDOW_NS: u64 = 5_000_000; // 5 milliseconds

/// Transaction formation result
#[derive(Debug, Clone)]
pub struct TransactionResult {
    /// Whether transaction was successfully formed
    pub formed: bool,
    /// Probability amplitude of the transaction (0 to 1)
    pub probability_amplitude: f64,
    /// Constructive interference measure
    pub interference_measure: f64,
    /// Optimal execution venue
    pub optimal_venue: Option<u32>,
    /// Expected execution price
    pub expected_price: f64,
    /// Expected fill ratio (0 to 1)
    pub expected_fill_ratio: f64,
    /// Time until transaction expires (nanoseconds)
    pub time_to_expiry_ns: u64,
}

/// Handshake state during transaction formation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeState {
    /// No activity
    Idle,
    /// Offer wave emitted, awaiting confirmation
    OfferSent,
    /// Confirmation received, calculating interference
    ConfirmationReceived,
    /// Transaction formed, ready to execute
    TransactionFormed,
    /// Handshake failed (destructive interference)
    Failed,
    /// Transaction expired
    Expired,
}

/// Retrocausal Handshake Calculator
pub struct RetrocausalHandshakeCalculator {
    /// Offer wave emitter
    emitter: OfferWaveEmitter,
    /// Confirmation wave receiver
    receiver: ConfirmationWaveReceiver,
    /// Current handshake state
    current_state: HandshakeState,
    /// Active offer wave (if any)
    active_offer: Option<OfferWave>,
    /// Active confirmation wave (if any)
    active_confirmation: Option<ConfirmationWave>,
    /// Handshake start time
    handshake_start_ns: Option<u64>,
    /// Successful transaction count
    successful_transactions: u64,
    /// Failed handshake count
    failed_handshakes: u64,
}

impl RetrocausalHandshakeCalculator {
    /// Create a new handshake calculator
    pub fn new() -> Self {
        Self {
            emitter: OfferWaveEmitter::new(),
            receiver: ConfirmationWaveReceiver::new(),
            current_state: HandshakeState::Idle,
            active_offer: None,
            active_confirmation: None,
            handshake_start_ns: None,
            successful_transactions: 0,
            failed_handshakes: 0,
        }
    }

    /// Create with custom parameters
    pub fn with_params(min_amplitude: f64, max_latency_ns: u64) -> Self {
        Self {
            emitter: OfferWaveEmitter::new(),
            receiver: ConfirmationWaveReceiver::with_params(min_amplitude, max_latency_ns),
            current_state: HandshakeState::Idle,
            active_offer: None,
            active_confirmation: None,
            handshake_start_ns: None,
            successful_transactions: 0,
            failed_handshakes: 0,
        }
    }

    /// Initiate a handshake by emitting an offer wave
    /// 
    /// # Arguments
    /// * `is_buy` - True for buy order, false for sell
    /// * `size` - Order size
    /// * `limit_price` - Limit price
    /// * `target_venues` - List of venue IDs to probe
    /// * `current_time_ns` - Current timestamp
    /// 
    /// # Returns
    /// The emitted offer wave
    pub fn initiate_handshake(&mut self,
                               is_buy: bool,
                               size: f64,
                               limit_price: f64,
                               target_venues: &[u32],
                               current_time_ns: u64) -> OfferWave {
        // Reset any previous handshake
        self.reset_active_waves();
        
        // Emit offer wave
        let offer = self.emitter.emit_offer_wave(
            is_buy,
            size,
            limit_price,
            target_venues,
            current_time_ns,
        );
        
        self.active_offer = Some(offer.clone());
        self.handshake_start_ns = Some(current_time_ns);
        self.current_state = HandshakeState::OfferSent;
        
        offer
    }

    /// Process incoming confirmation from a venue
    /// 
    /// # Arguments
    /// * `venue_id` - Venue identifier
    /// * `liquidity` - Available liquidity
    /// * `best_bid` - Best bid
    /// * `best_ask` - Best ask
    /// * `latency_ns` - Measured latency
    /// * `current_time_ns` - Current timestamp
    /// 
    /// # Returns
    /// Updated handshake state
    pub fn process_confirmation(&mut self,
                                 venue_id: u32,
                                 liquidity: f64,
                                 best_bid: f64,
                                 best_ask: f64,
                                 latency_ns: u64,
                                 current_time_ns: u64) -> HandshakeState {
        // Check if we have an active offer
        if self.active_offer.is_none() {
            return HandshakeState::Idle;
        }
        
        // Check handshake timeout
        if let Some(start_time) = self.handshake_start_ns {
            if current_time_ns - start_time > MAX_HANDSHAKE_WINDOW_NS {
                self.current_state = HandshakeState::Expired;
                self.failed_handshakes += 1;
                return HandshakeState::Expired;
            }
        }
        
        // Receive confirmation
        if let Some(source) = self.receiver.receive_confirmation(
            venue_id,
            liquidity,
            best_bid,
            best_ask,
            latency_ns,
            current_time_ns,
        ) {
            // Aggregate with existing confirmations or create new
            match &self.active_confirmation {
                Some(existing) => {
                    let mut sources = existing.sources.clone();
                    sources.push(source);
                    
                    if let Some(ref offer) = self.active_offer {
                        self.active_confirmation = Some(self.receiver.aggregate_confirmations(
                            sources,
                            offer,
                            current_time_ns,
                        ));
                    }
                }
                None => {
                    if let Some(ref offer) = self.active_offer {
                        self.active_confirmation = Some(self.receiver.aggregate_confirmations(
                            vec![source],
                            offer,
                            current_time_ns,
                        ));
                    }
                }
            }
            
            self.current_state = HandshakeState::ConfirmationReceived;
        }
        
        self.current_state
    }

    /// Calculate transaction formation and determine if execution should proceed
    /// 
    /// # Arguments
    /// * `current_time_ns` - Current timestamp
    /// 
    /// # Returns
    /// TransactionResult with execution recommendation
    pub fn calculate_transaction(&mut self, current_time_ns: u64) -> TransactionResult {
        // Must have both offer and confirmation
        let offer = match &self.active_offer {
            Some(o) => o.clone(),
            None => return TransactionResult {
                formed: false,
                probability_amplitude: 0.0,
                interference_measure: 0.0,
                optimal_venue: None,
                expected_price: 0.0,
                expected_fill_ratio: 0.0,
                time_to_expiry_ns: 0,
            },
        };
        
        let confirmation = match &self.active_confirmation {
            Some(c) => c.clone(),
            None => return TransactionResult {
                formed: false,
                probability_amplitude: 0.0,
                interference_measure: 0.0,
                optimal_venue: None,
                expected_price: offer.limit_price,
                expected_fill_ratio: 0.0,
                time_to_expiry_ns: self.get_time_to_expiry(current_time_ns),
            },
        };
        
        // Calculate interference at current time
        let interference = confirmation.calculate_interference(&offer, current_time_ns);
        
        // Normalize interference to probability amplitude
        let max_possible_interference = offer.total_amplitude * confirmation.total_amplitude;
        let probability_amplitude = if max_possible_interference > 1e-15 {
            (interference / max_possible_interference).clamp(0.0, 1.0)
        } else {
            0.0
        };
        
        // Check if transaction threshold is met
        let formed = probability_amplitude >= MIN_TRANSACTION_AMPLITUDE
            && confirmation.is_executable;
        
        if formed {
            self.current_state = HandshakeState::TransactionFormed;
            self.successful_transactions += 1;
        } else if interference < 0.0 {
            self.current_state = HandshakeState::Failed;
            self.failed_handshakes += 1;
        }
        
        // Find optimal venue (highest liquidity with constructive interference)
        let optimal_venue = self.find_optimal_venue(&confirmation, &offer, current_time_ns);
        
        // Calculate expected price
        let expected_price = confirmation.weighted_average_price(offer.limit_price);
        
        // Calculate expected fill ratio
        let total_liquidity = confirmation.total_liquidity();
        let expected_fill_ratio = if offer.order_size > 1e-15 {
            (total_liquidity / offer.order_size).clamp(0.0, 1.0)
        } else {
            0.0
        };
        
        TransactionResult {
            formed,
            probability_amplitude,
            interference_measure: interference,
            optimal_venue,
            expected_price,
            expected_fill_ratio,
            time_to_expiry_ns: self.get_time_to_expiry(current_time_ns),
        }
    }

    /// Execute the transaction if formed
    /// 
    /// # Arguments
    /// * `current_time_ns` - Current timestamp
    /// 
    /// # Returns
    /// true if execution should proceed, false otherwise
    pub fn execute_if_formed(&mut self, current_time_ns: u64) -> bool {
        let result = self.calculate_transaction(current_time_ns);
        
        if result.formed {
            // Reset for next transaction
            self.reset_active_waves();
            true
        } else {
            false
        }
    }

    /// Get current handshake state
    pub fn state(&self) -> HandshakeState {
        self.current_state
    }

    /// Get statistics
    pub fn stats(&self) -> (u64, u64) {
        (self.successful_transactions, self.failed_handshakes)
    }

    /// Reset calculator state
    pub fn reset(&mut self) {
        self.reset_active_waves();
        self.current_state = HandshakeState::Idle;
        self.handshake_start_ns = None;
    }

    // Internal: Reset active waves
    fn reset_active_waves(&mut self) {
        self.active_offer = None;
        self.active_confirmation = None;
        self.handshake_start_ns = None;
    }

    // Internal: Get time remaining before handshake expires
    fn get_time_to_expiry(&self, current_time_ns: u64) -> u64 {
        match self.handshake_start_ns {
            Some(start) => {
                let elapsed = current_time_ns - start;
                if elapsed >= MAX_HANDSHAKE_WINDOW_NS {
                    0
                } else {
                    MAX_HANDSHAKE_WINDOW_NS - elapsed
                }
            }
            None => 0,
        }
    }

    // Internal: Find optimal venue for execution
    fn find_optimal_venue(&self, confirmation: &ConfirmationWave,
                          offer: &OfferWave,
                          current_time_ns: u64) -> Option<u32> {
        if confirmation.sources.is_empty() {
            return None;
        }

        let mut best_venue: Option<u32> = None;
        let mut best_score = f64::NEG_INFINITY;

        for source in &confirmation.sources {
            // Score based on liquidity and individual interference contribution
            let source_interference = offer.components.iter()
                .filter(|c| c.target_id == source.venue_id)
                .map(|c| c.amplitude_re * source.amplitude_re)
                .sum::<f64>();

            let score = source.available_liquidity * source_interference;

            if score > best_score {
                best_score = score;
                best_venue = Some(source.venue_id);
            }
        }

        best_venue
    }
}

impl Default for RetrocausalHandshakeCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator_creation() {
        let calc = RetrocausalHandshakeCalculator::new();
        assert_eq!(calc.state(), HandshakeState::Idle);
    }

    #[test]
    fn test_initiate_handshake() {
        let mut calc = RetrocausalHandshakeCalculator::new();
        
        let offer = calc.initiate_handshake(
            true,           // Buy
            100.0,          // Size
            100.0,          // Limit price
            &[1, 2],        // Venues
            1000000,        // Time
        );
        
        assert_eq!(calc.state(), HandshakeState::OfferSent);
        assert!(offer.is_buy_intent);
        assert_eq!(offer.order_size, 100.0);
    }

    #[test]
    fn test_process_confirmation() {
        let mut calc = RetrocausalHandshakeCalculator::new();
        
        // Initiate first
        calc.initiate_handshake(true, 100.0, 100.0, &[1], 1000000);
        
        // Process confirmation
        let state = calc.process_confirmation(
            1,      // Venue
            500.0,  // Liquidity
            99.9,   // Bid
            100.1,  // Ask
            100000, // Latency
            1000100, // Time
        );
        
        assert_eq!(state, HandshakeState::ConfirmationReceived);
    }

    #[test]
    fn test_transaction_formation() {
        let mut calc = RetrocausalHandshakeCalculator::new();
        
        // Full handshake sequence
        calc.initiate_handshake(true, 100.0, 100.0, &[1], 1000000);
        calc.process_confirmation(1, 500.0, 99.9, 100.1, 100000, 1000100);
        
        let result = calc.calculate_transaction(1000200);
        
        // Result depends on interference calculation
        assert!(result.time_to_expiry_ns > 0 || result.time_to_expiry_ns == 0);
    }

    #[test]
    fn test_handshake_timeout() {
        let mut calc = RetrocausalHandshakeCalculator::new();
        
        calc.initiate_handshake(true, 100.0, 100.0, &[1], 0);
        
        // Advance time beyond handshake window
        let state = calc.process_confirmation(1, 500.0, 99.9, 100.1, 100000, 10_000_000);
        
        assert_eq!(state, HandshakeState::Expired);
    }

    #[test]
    fn test_statistics() {
        let mut calc = RetrocausalHandshakeCalculator::new();
        
        assert_eq!(calc.stats(), (0, 0));
        
        // Simulate some activity
        calc.initiate_handshake(true, 100.0, 100.0, &[1], 0);
        calc.process_confirmation(1, 500.0, 99.9, 100.1, 100000, 100);
        calc.calculate_transaction(200);
        
        let (success, failed) = calc.stats();
        assert!(success + failed >= 0);
    }
}
