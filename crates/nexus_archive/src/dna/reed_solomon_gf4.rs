//! Reed-Solomon Error Correction over GF(4)
//! 
//! Implements Reed-Solomon coding specifically for DNA storage using Galois Field GF(4).
//! Uses lookup tables for efficient field arithmetic with XOR-based operations.

use crate::dna::nucleotide_base4_encoder::Base;
use thiserror::Error;

/// GF(4) field size
const GF4_SIZE: usize = 4;

/// GF(4) addition table (XOR operation)
/// Addition in GF(4) is equivalent to XOR of the element indices
const GF4_ADD: [[u8; 4]; 4] = [
    // 0  1  2  3
    [0, 1, 2, 3], // 0
    [1, 0, 3, 2], // 1
    [2, 3, 0, 1], // 2
    [3, 2, 1, 0], // 3
];

/// GF(4) multiplication table
/// Based on primitive polynomial x^2 + x + 1 over GF(2)
/// Elements: 0=00, 1=01, α=10, α²=11 where α is primitive element
const GF4_MUL: [[u8; 4]; 4] = [
    // 0  1  2  3
    [0, 0, 0, 0], // 0
    [0, 1, 2, 3], // 1
    [0, 2, 3, 1], // α (2)
    [0, 3, 1, 2], // α² (3)
];

/// GF(4) logarithm table (for efficient multiplication via log/antilog)
/// log(0) is undefined, represented as 255
const GF4_LOG: [u8; 4] = [255, 0, 1, 2];

/// GF(4) antilogarithm table
const GF4_EXP: [u8; 4] = [1, 2, 3, 1]; // exp wraps around (order 3)

#[derive(Error, Debug)]
pub enum ReedSolomonError {
    #[error("Invalid number of parity symbols: {0}")]
    InvalidParityCount(usize),
    #[error("Message too long: {0} > max {1}")]
    MessageTooLong(usize, usize),
    #[error("Decoding failed: too many errors")]
    TooManyErrors,
    #[error("Invalid codeword length")]
    InvalidCodewordLength,
    #[error("Buffer overflow")]
    BufferOverflow,
}

/// Reed-Solomon codec over GF(4)
/// 
/// For DNA storage, we use small block sizes due to synthesis constraints.
/// Typical configuration: (n=20, k=16, t=2) - 16 data symbols, 4 parity, corrects 2 errors
pub struct ReedSolomonGF4 {
    n: usize, // Total codeword length
    k: usize, // Data symbols
    t: usize, // Error correction capability (t = (n-k)/2)
    generator: Box<[u8]>, // Generator polynomial coefficients
}

impl ReedSolomonGF4 {
    /// Create a new RS codec with specified parameters
    /// 
    /// # Arguments
    /// * `n` - Total codeword length (must be <= 255 for GF(4))
    /// * `k` - Number of data symbols
    /// 
    /// # Constraints
    /// * n - k must be even (for proper error correction)
    /// * t = (n-k)/2 is the error correction capability
    pub fn new(n: usize, k: usize) -> Result<Self, ReedSolomonError> {
        if n > 255 {
            return Err(ReedSolomonError::MessageTooLong(n, 255));
        }
        if k >= n {
            return Err(ReedSolomonError::InvalidParityCount(k));
        }
        
        let parity = n - k;
        if parity % 2 != 0 {
            return Err(ReedSolomonError::InvalidParityCount(parity));
        }
        
        let t = parity / 2;
        
        // Generate generator polynomial
        let generator = Self::generate_polynomial(t)?;
        
        Ok(Self { n, k, t, generator })
    }

    /// Generate the generator polynomial g(x) = (x - α)(x - α²)...(x - α²ᵗ)
    fn generate_polynomial(t: usize) -> Result<Box<[u8]>, ReedSolomonError> {
        // Start with g(x) = 1
        let mut gen = vec![1u8];
        
        for i in 1..=t {
            // Multiply by (x - α^i) = (x + α^i) in GF(4)
            let alpha_i = ((i - 1) % 3 + 1) as u8; // Cycle through 1, 2, 3
            
            // New polynomial: gen(x) * (x + α^i)
            let mut new_gen = vec![0u8; gen.len() + 1];
            
            // gen(x) * x
            for j in 0..gen.len() {
                new_gen[j + 1] = gf4_mul(gen[j], 1);
            }
            
            // gen(x) * α^i
            for j in 0..gen.len() {
                new_gen[j] = gf4_add(new_gen[j], gf4_mul(gen[j], alpha_i));
            }
            
            gen = new_gen;
        }
        
        Ok(gen.into_boxed_slice())
    }

    /// Encode data symbols into a codeword with parity symbols
    /// 
    /// # Arguments
    /// * `data` - Input data symbols (length must be exactly k)
    /// * `output` - Output buffer (length must be at least n)
    /// 
    /// Returns the number of symbols written
    pub fn encode(&self, data: &[Base], output: &mut [Base]) -> Result<usize, ReedSolomonError> {
        if data.len() != self.k {
            return Err(ReedSolomonError::MessageTooLong(data.len(), self.k));
        }
        if output.len() < self.n {
            return Err(ReedSolomonError::BufferOverflow);
        }

        // Convert bases to field elements
        let data_elems: Vec<u8> = data.iter().map(|b| b.to_u8()).collect();
        
        // Systematic encoding: data followed by parity
        // Compute parity using polynomial division
        
        // Initialize remainder with data padded with zeros
        let mut remainder = vec![0u8; self.n];
        for (i, &d) in data_elems.iter().enumerate() {
            remainder[i] = d;
        }
        
        // Polynomial division by generator
        for i in 0..self.k {
            let coef = remainder[i];
            if coef != 0 {
                for j in 0..self.generator.len() {
                    remainder[i + j] = gf4_add(remainder[i + j], gf4_mul(coef, self.generator[j]));
                }
            }
        }
        
        // Copy data to output
        for (i, &d) in data.iter().take(self.k).enumerate() {
            output[i] = d;
        }
        
        // Copy parity symbols
        for i in 0..(self.n - self.k) {
            let parity_val = remainder[self.k + i];
            output[self.k + i] = Base::from_u8(parity_val)
                .map_err(|_| ReedSolomonError::InvalidParityCount(parity_val as usize))?;
        }
        
        Ok(self.n)
    }

    /// Decode a received codeword and correct errors
    /// 
    /// # Arguments
    /// * `received` - Received symbols (may contain errors)
    /// * `corrected` - Output buffer for corrected codeword
    /// 
    /// Returns the number of errors corrected, or error if uncorrectable
    pub fn decode(&self, received: &[Base], corrected: &mut [Base]) -> Result<usize, ReedSolomonError> {
        if received.len() != self.n {
            return Err(ReedSolomonError::InvalidCodewordLength);
        }
        if corrected.len() < self.n {
            return Err(ReedSolomonError::BufferOverflow);
        }

        // Convert to field elements
        let mut codeword: Vec<u8> = received.iter().map(|b| b.to_u8()).collect();
        
        // Compute syndromes
        let syndromes = self.compute_syndromes(&codeword);
        
        // Check if all syndromes are zero (no errors)
        if syndromes.iter().all(|&s| s == 0) {
            for (i, &c) in codeword.iter().enumerate() {
                corrected[i] = Base::from_u8(c)
                    .map_err(|_| ReedSolomonError::InvalidCodewordLength)?;
            }
            return Ok(0);
        }
        
        // Try to correct errors using Peterson-Gorenstein-Zierler algorithm
        let errors_corrected = self.correct_errors(&mut codeword, &syndromes)?;
        
        // Copy corrected codeword
        for (i, &c) in codeword.iter().enumerate() {
            corrected[i] = Base::from_u8(c)
                .map_err(|_| ReedSolomonError::InvalidCodewordLength)?;
        }
        
        Ok(errors_corrected)
    }

    /// Compute syndrome values S_i = r(α^i) for i = 1 to 2t
    fn compute_syndromes(&self, codeword: &[u8]) -> Vec<u8> {
        let mut syndromes = vec![0u8; 2 * self.t];
        
        for i in 0..(2 * self.t) {
            let alpha_power = ((i) % 3 + 1) as u8; // α^(i+1)
            let mut sum = 0u8;
            
            for (j, &c) in codeword.iter().enumerate() {
                // Compute c * (α^(i+1))^j
                let mut power = 1u8;
                for _ in 0..j {
                    power = gf4_mul(power, alpha_power);
                }
                sum = gf4_add(sum, gf4_mul(c, power));
            }
            
            syndromes[i] = sum;
        }
        
        syndromes
    }

    /// Correct errors using syndrome decoding
    fn correct_errors(&self, codeword: &mut [u8], syndromes: &[u8]) -> Result<usize, ReedSolomonError> {
        // For small t, use brute-force error location search
        // This is efficient for DNA storage where t is typically 1-3
        
        if self.t == 1 {
            return self.correct_single_error(codeword, syndromes);
        } else if self.t == 2 {
            return self.correct_double_error(codeword, syndromes);
        }
        
        // General case: simplified Berlekamp-Massey would go here
        // For now, return error for unimplemented cases
        Err(ReedSolomonError::TooManyErrors)
    }

    /// Correct a single error
    fn correct_single_error(&self, codeword: &mut [u8], syndromes: &[u8]) -> Result<usize, ReedSolomonError> {
        if syndromes.is_empty() || syndromes[0] == 0 {
            return Ok(0);
        }
        
        let s1 = syndromes[0];
        let s2 = syndromes.get(1).copied().unwrap_or(0);
        
        if s1 == 0 {
            return Ok(0);
        }
        
        // Error value = s1
        // Error location from s2/s1 ratio
        
        for pos in 0..self.n {
            let alpha_pos = self.gf4_power(pos as u8 + 1);
            let expected_s2 = gf4_mul(s1, alpha_pos);
            
            if expected_s2 == s2 {
                // Found error location
                codeword[pos] = gf4_add(codeword[pos], s1);
                return Ok(1);
            }
        }
        
        Err(ReedSolomonError::TooManyErrors)
    }

    /// Correct up to two errors
    fn correct_double_error(&self, codeword: &mut [u8], syndromes: &[u8]) -> Result<usize, ReedSolomonError> {
        if syndromes.len() < 4 {
            return Err(ReedSolomonError::TooManyErrors);
        }
        
        let s1 = syndromes[0];
        let s2 = syndromes[1];
        let s3 = syndromes[2];
        let s4 = syndromes[3];
        
        // Check for single error first
        if s1 != 0 && s2 != 0 {
            let ratio = self.gf4_divide(s2, s1);
            let mut found_single = true;
            
            for i in 2..4 {
                let expected = gf4_mul(s1, self.gf4_power((i + 1) as u8));
                if syndromes[i] != expected {
                    found_single = false;
                    break;
                }
            }
            
            if found_single {
                return self.correct_single_error(codeword, syndromes);
            }
        }
        
        // Two-error correction would require solving quadratic equation
        // Simplified: try all pairs of positions
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if self.try_two_error_correction(codeword, syndromes, i, j) {
                    return Ok(2);
                }
            }
        }
        
        Err(ReedSolomonError::TooManyErrors)
    }

    /// Attempt to correct errors at two specific positions
    fn try_two_error_correction(&self, codeword: &mut [u8], syndromes: &[u8], pos1: usize, pos2: usize) -> bool {
        let alpha1 = self.gf4_power(pos1 as u8 + 1);
        let alpha2 = self.gf4_power(pos2 as u8 + 1);
        
        // Solve system of equations for error values e1, e2
        // s1 = e1*alpha1 + e2*alpha2
        // s2 = e1*alpha1^2 + e2*alpha2^2
        
        let s1 = syndromes[0];
        let s2 = syndromes[1];
        
        // Brute force: try all combinations of error values
        for e1 in 1..4 {
            for e2 in 1..4 {
                let calc_s1 = gf4_add(gf4_mul(e1, alpha1), gf4_mul(e2, alpha2));
                let calc_s2 = gf4_add(gf4_mul(e1, gf4_mul(alpha1, alpha1)), gf4_mul(e2, gf4_mul(alpha2, alpha2)));
                
                if calc_s1 == s1 && calc_s2 == s2 {
                    // Apply corrections
                    codeword[pos1] = gf4_add(codeword[pos1], e1);
                    codeword[pos2] = gf4_add(codeword[pos2], e2);
                    return true;
                }
            }
        }
        
        false
    }

    /// Compute α^n in GF(4)
    fn gf4_power(&self, n: u8) -> u8 {
        if n == 0 {
            return 1;
        }
        let idx = ((n - 1) % 3) as usize;
        GF4_EXP[idx]
    }

    /// Division in GF(4)
    fn gf4_divide(&self, a: u8, b: u8) -> u8 {
        if b == 0 {
            return 0;
        }
        // a/b = a * b^(-1) = a * b^(2) since b^3 = 1 in GF(4)*
        let b_inv = gf4_mul(b, b);
        gf4_mul(a, b_inv)
    }

    /// Get code parameters
    pub fn parameters(&self) -> (usize, usize, usize) {
        (self.n, self.k, self.t)
    }
}

/// GF(4) addition using lookup table
#[inline]
fn gf4_add(a: u8, b: u8) -> u8 {
    GF4_ADD[a as usize][b as usize]
}

/// GF(4) multiplication using lookup table
#[inline]
fn gf4_mul(a: u8, b: u8) -> u8 {
    GF4_MUL[a as usize][b as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf4_arithmetic() {
        // Addition is XOR-like
        assert_eq!(gf4_add(0, 0), 0);
        assert_eq!(gf4_add(1, 1), 0);
        assert_eq!(gf4_add(2, 3), 1);
        
        // Multiplication
        assert_eq!(gf4_mul(0, 2), 0);
        assert_eq!(gf4_mul(1, 2), 2);
        assert_eq!(gf4_mul(2, 2), 3);
        assert_eq!(gf4_mul(2, 3), 1);
    }

    #[test]
    fn test_rs_encode_decode_no_errors() {
        let rs = ReedSolomonGF4::new(8, 6).unwrap();
        
        let data = [Base::A, Base::C, Base::G, Base::T, Base::A, Base::C];
        let mut codeword = [Base::A; 8];
        
        rs.encode(&data, &mut codeword).unwrap();
        
        let mut decoded = [Base::A; 8];
        let errors = rs.decode(&codeword, &mut decoded).unwrap();
        
        assert_eq!(errors, 0);
        assert_eq!(&decoded[..6], &data);
    }

    #[test]
    fn test_rs_single_error_correction() {
        let rs = ReedSolomonGF4::new(8, 6).unwrap();
        
        let data = [Base::A, Base::C, Base::G, Base::T, Base::A, Base::C];
        let mut codeword = [Base::A; 8];
        
        rs.encode(&data, &mut codeword).unwrap();
        
        // Introduce an error
        codeword[3] = Base::G; // Was T
        
        let mut decoded = [Base::A; 8];
        let errors = rs.decode(&codeword, &mut decoded).unwrap();
        
        assert!(errors >= 1);
        assert_eq!(&decoded[..6], &data);
    }
}
