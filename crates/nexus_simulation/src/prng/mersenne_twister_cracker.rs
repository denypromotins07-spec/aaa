//! Mersenne Twister Cracker
//! 
//! Specialized cracker for MT19937 PRNG commonly used in exchanges.
//! Uses observed outputs to reconstruct the 624-word internal state array.

use core::fmt;

/// MT19937 state array size
const MT_STATE_SIZE: usize = 624;

/// MT19937 period parameter
const MT_M: usize = 397;

/// MT19937 magic constants
const MT_MATRIX_A: u32 = 0x9908B0DF;
const MT_UPPER_MASK: u32 = 0x80000000;
const MT_LOWER_MASK: u32 = 0x7FFFFFFF;

/// Represents a cracked MT19937 state
#[derive(Debug, Clone)]
pub struct Mt19937State {
    /// Internal state array (624 words)
    pub state: [u32; MT_STATE_SIZE],
    /// Current index in state array
    pub index: usize,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
}

/// Configuration for MT cracking
#[derive(Debug, Clone, Copy)]
pub struct MtCrackerConfig {
    /// Minimum consecutive outputs required
    pub min_outputs: usize,
    /// Maximum attempts before giving up
    pub max_attempts: usize,
}

impl Default for MtCrackerConfig {
    fn default() -> Self {
        Self {
            min_outputs: MT_STATE_SIZE,
            max_attempts: 100,
        }
    }
}

/// Mersenne Twister cracker
pub struct Mt19937Cracker {
    config: MtCrackerConfig,
    observed_outputs: Vec<u32>,
}

impl Mt19937Cracker {
    pub const fn new(config: MtCrackerConfig) -> Self {
        Self {
            config,
            observed_outputs: Vec::new(),
        }
    }

    /// Add an observed output from the target PRNG
    pub fn add_output(&mut self, value: u32) -> Result<(), MtCrackError> {
        if self.observed_outputs.len() >= MT_STATE_SIZE * 2 {
            return Err(MtCrackError::TooManyOutputs);
        }
        self.observed_outputs.push(value);
        Ok(())
    }

    /// Attempt to crack the MT19937 state
    pub fn crack(&self) -> Result<Mt19937State, MtCrackError> {
        if self.observed_outputs.len() < self.config.min_outputs {
            return Err(MtCrackError::InsufficientOutputs);
        }

        // Reverse the tempering function to get raw state values
        let mut raw_state: Vec<u32> = Vec::with_capacity(self.observed_outputs.len());
        
        for &output in &self.observed_outputs {
            let raw = self.untemper(output)?;
            raw_state.push(raw);
        }

        // Verify state consistency using MT recurrence relation
        if !self.verify_state_consistency(&raw_state) {
            return Err(MtCrackError::InconsistentState);
        }

        // Build state array
        let mut state_array = [0u32; MT_STATE_SIZE];
        for (i, &val) in raw_state.iter().take(MT_STATE_SIZE).enumerate() {
            state_array[i] = val;
        }

        // Calculate confidence based on number of outputs and consistency
        let confidence = ((self.observed_outputs.len() - MT_STATE_SIZE) as f64 / MT_STATE_SIZE as f64)
            .min(1.0)
            .max(0.5);

        Ok(Mt19937State {
            state: state_array,
            index: MT_STATE_SIZE, // Force regeneration on next call
            confidence,
        })
    }

    /// Reverse the MT19937 tempering function
    fn untemper(&self, tempered: u32) -> Result<u32, MtCrackError> {
        let mut y = tempered;

        // Reverse: y ^= (y >> 18)
        // This is its own inverse for the lower 18 bits
        
        // Reverse: y ^= (y << 15) & 0xEFC60000
        let mask = 0xEFC60000u32;
        y ^= (y << 15) & mask;

        // Reverse: y ^= (y << 7) & 0x9D2C5680
        let mask2 = 0x9D2C5680u32;
        y ^= (y << 7) & mask2;
        y ^= (y << 14) & mask2;
        y ^= (y << 21) & mask2;

        // Reverse: y ^= y >> 11
        y ^= y >> 11;
        y ^= y >> 22;

        Ok(y)
    }

    /// Verify that extracted state satisfies MT recurrence relation
    fn verify_state_consistency(&self, state: &[u32]) -> bool {
        if state.len() < MT_STATE_SIZE + 1 {
            return true; // Not enough data to verify
        }

        // Check a few recurrence relations
        // MT recurrence: x[k+N] = x[k+M] ^ ((x[k] ^ x[k+N]) >> 1) ^ A
        let check_count = 10.min(state.len() - MT_STATE_SIZE);
        
        for i in 0..check_count {
            let k = i;
            let expected = state[(k + MT_M) % MT_STATE_SIZE]
                ^ (((state[k] ^ state[(k + MT_STATE_SIZE - 1) % MT_STATE_SIZE]) >> 1)
                    ^ if state[k] & 1 == 0 { 0 } else { MT_MATRIX_A });
            
            let actual = state[(k + MT_STATE_SIZE) % state.len()];
            
            if expected != actual {
                return false;
            }
        }

        true
    }

    /// Predict the next N outputs given a cracked state
    pub fn predict_next(&self, state: &Mt19937State, count: usize) -> Result<Vec<u32>, MtCrackError> {
        if count == 0 {
            return Ok(Vec::new());
        }

        let mut predictions = Vec::with_capacity(count);
        let mut current_state = state.state;
        let mut current_index = state.index;

        for _ in 0..count {
            if current_index >= MT_STATE_SIZE {
                // Generate new batch
                for k in 0..MT_STATE_SIZE {
                    let y = (current_state[k] & MT_UPPER_MASK)
                        | (current_state[(k + 1) % MT_STATE_SIZE] & MT_LOWER_MASK);
                    
                    current_state[k] = current_state[(k + MT_M) % MT_STATE_SIZE]
                        ^ (y >> 1)
                        ^ if y % 2 == 0 { 0 } else { MT_MATRIX_A };
                }
                current_index = 0;
            }

            let y = current_state[current_index];
            current_index += 1;

            // Apply tempering
            let mut z = y;
            z ^= z >> 11;
            z ^= (z << 7) & 0x9D2C5680;
            z ^= (z << 15) & 0xEFC60000;
            z ^= z >> 18;

            predictions.push(z);
        }

        Ok(predictions)
    }

    /// Clear all observed outputs
    pub fn clear(&mut self) {
        self.observed_outputs.clear();
    }

    /// Get number of observed outputs
    pub fn output_count(&self) -> usize {
        self.observed_outputs.len()
    }
}

/// Errors from MT cracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtCrackError {
    InsufficientOutputs,
    TooManyOutputs,
    InconsistentState,
    UntemperFailed,
}

impl fmt::Display for MtCrackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MtCrackError::InsufficientOutputs => write!(f, "Insufficient outputs collected"),
            MtCrackError::TooManyOutputs => write!(f, "Too many outputs (buffer overflow)"),
            MtCrackError::InconsistentState => write!(f, "State fails consistency check"),
            MtCrackError::UntemperFailed => write!(f, "Failed to reverse tempering"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_untemper_temper_roundtrip() {
        let cracker = Mt19937Cracker::new(MtCrackerConfig::default());
        
        // Test value
        let original = 0x12345678u32;
        
        // Temper (forward)
        let mut y = original;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C5680;
        y ^= (y << 15) & 0xEFC60000;
        y ^= y >> 18;
        
        // Untemper (reverse)
        let recovered = cracker.untemper(y).unwrap();
        
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_insufficient_outputs() {
        let config = MtCrackerConfig {
            min_outputs: 624,
            ..Default::default()
        };
        let cracker = Mt19937Cracker::new(config);
        
        let result = cracker.crack();
        assert!(matches!(result, Err(MtCrackError::InsufficientOutputs)));
    }
}
