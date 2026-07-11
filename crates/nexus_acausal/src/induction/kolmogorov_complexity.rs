//! Kolmogorov Complexity Estimator
//! 
//! Provides zero-allocation approximations of algorithmic complexity
//! using compression-based and structural analysis methods.

/// Minimum sample size for meaningful complexity estimation
const MIN_SAMPLE_SIZE: usize = 16;

/// Maximum window size for Lempel-Ziv estimation
const MAX_LZ_WINDOW: usize = 4096;

/// Kolmogorov Complexity estimation result
#[derive(Debug, Clone)]
pub struct KolmogorovEstimate {
    /// Estimated complexity in bits
    pub complexity_bits: f64,
    /// Normalized complexity (0.0 - 1.0)
    pub normalized: f64,
    /// Confidence in estimate (0.0 - 1.0)
    pub confidence: f64,
    /// Method used for estimation
    pub method: ComplexityMethod,
}

/// Method used for complexity estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexityMethod {
    /// Lempel-Ziv compression ratio
    LempelZiv,
    /// Run-length encoding analysis
    RunLength,
    /// Structural pattern counting
    Structural,
    /// Hybrid approach
    Hybrid,
}

/// Kolmogorov Complexity Estimator
pub struct KolmogorovComplexity {
    /// Reusable buffer for zero-alloc operations
    buffer: Vec<u8>,
}

impl KolmogorovComplexity {
    /// Create a new complexity estimator
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(1024),
        }
    }
    
    /// Estimate Kolmogorov complexity of data
    /// 
    /// Returns normalized complexity (0.0 = simple, 1.0 = maximally complex)
    pub fn estimate(&mut self, data: &[u8]) -> Result<KolmogorovEstimate, &'static str> {
        if data.is_empty() {
            return Err("Data cannot be empty");
        }
        
        if data.len() < MIN_SAMPLE_SIZE {
            return Err("Data too short for reliable estimation");
        }
        
        // Use hybrid approach for best accuracy
        let lz_complexity = self.lemper_ziv_estimate(data);
        let rl_complexity = self.run_length_estimate(data);
        let structural = self.structural_estimate(data);
        
        // Weighted average with confidence based on data characteristics
        let lz_weight = 0.5;
        let rl_weight = 0.25;
        let struct_weight = 0.25;
        
        let complexity = lz_weight * lz_complexity.normalized 
                       + rl_weight * rl_complexity.normalized 
                       + struct_weight * structural.normalized;
        
        let confidence = (lz_complexity.confidence + rl_complexity.confidence + structural.confidence) / 3.0;
        
        Ok(KolmogorovEstimate {
            complexity_bits: complexity * data.len() as f64 * 8.0,
            normalized: complexity.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            method: ComplexityMethod::Hybrid,
        })
    }
    
    /// Lempel-Ziv based complexity estimation
    fn lemper_ziv_estimate(&self, data: &[u8]) -> KolmogorovEstimate {
        let mut phrases = 0usize;
        let mut pos = 0usize;
        
        while pos < data.len() && pos < MAX_LZ_WINDOW {
            // Find longest match in previous data
            let mut match_len = 0;
            let search_start = pos.saturating_sub(MAX_LZ_WINDOW);
            
            for start in search_start..pos {
                let mut current_match = 0;
                while pos + current_match < data.len() 
                    && start + current_match < pos 
                    && data[start + current_match] == data[pos + current_match]
                    && current_match < 256 
                {
                    current_match += 1;
                }
                
                if current_match > match_len {
                    match_len = current_match;
                }
            }
            
            if match_len > 0 {
                pos += match_len;
            } else {
                pos += 1;
            }
            
            phrases += 1;
        }
        
        // Normalize by theoretical maximum
        let max_phrases = data.len().min(MAX_LZ_WINDOW);
        let normalized = if max_phrases > 0 {
            phrases as f64 / max_phrases as f64
        } else {
            1.0
        };
        
        // LZ complexity is higher when more phrases are needed
        let complexity = normalized;
        
        KolmogorovEstimate {
            complexity_bits: phrases as f64 * (data.len() as f64).log2(),
            normalized: complexity.clamp(0.0, 1.0),
            confidence: if data.len() > 100 { 0.9 } else { 0.6 },
            method: ComplexityMethod::LempelZiv,
        }
    }
    
    /// Run-length encoding based complexity estimation
    fn run_length_estimate(&self, data: &[u8]) -> KolmogorovEstimate {
        if data.is_empty() {
            return KolmogorovEstimate {
                complexity_bits: 0.0,
                normalized: 0.0,
                confidence: 0.0,
                method: ComplexityMethod::RunLength,
            };
        }
        
        let mut runs = 1usize;
        let mut current_run_len = 1usize;
        
        for i in 1..data.len() {
            if data[i] == data[i - 1] {
                current_run_len += 1;
            } else {
                runs += 1;
                current_run_len = 1;
            }
        }
        
        // More runs = higher complexity
        let max_runs = data.len();
        let normalized = runs as f64 / max_runs as f64;
        
        // Average run length affects confidence
        let avg_run_len = data.len() as f64 / runs as f64;
        let confidence = if avg_run_len > 1.0 { 0.7 } else { 0.4 };
        
        KolmogorovEstimate {
            complexity_bits: (runs as f64).log2() + data.len() as f64,
            normalized: normalized.clamp(0.0, 1.0),
            confidence,
            method: ComplexityMethod::RunLength,
        }
    }
    
    /// Structural pattern-based complexity estimation
    fn structural_estimate(&self, data: &[u8]) -> KolmogorovEstimate {
        if data.is_empty() {
            return KolmogorovEstimate {
                complexity_bits: 0.0,
                normalized: 0.0,
                confidence: 0.0,
                method: ComplexityMethod::Structural,
            };
        }
        
        // Count unique bytes
        let mut unique_bytes = [false; 256];
        let mut unique_count = 0usize;
        
        for &byte in data.iter() {
            if !unique_bytes[byte as usize] {
                unique_bytes[byte as usize] = true;
                unique_count += 1;
            }
        }
        
        // Count transitions (changes between consecutive bytes)
        let mut transitions = 0usize;
        for i in 1..data.len() {
            if data[i] != data[i - 1] {
                transitions += 1;
            }
        }
        
        // Entropy estimation
        let mut byte_counts = [0usize; 256];
        for &byte in data.iter() {
            byte_counts[byte as usize] += 1;
        }
        
        let mut entropy = 0.0f64;
        let len = data.len() as f64;
        for count in byte_counts.iter() {
            if *count > 0 {
                let p = *count as f64 / len;
                entropy -= p * p.log2();
            }
        }
        
        // Normalize entropy (max is log2(256) = 8)
        let normalized_entropy = entropy / 8.0;
        
        // Combine metrics
        let unique_ratio = unique_count as f64 / 256.0;
        let transition_ratio = transitions as f64 / data.len() as f64;
        
        let complexity = (normalized_entropy * 0.5 + unique_ratio * 0.25 + transition_ratio * 0.25).clamp(0.0, 1.0);
        
        KolmogorovEstimate {
            complexity_bits: entropy * data.len() as f64,
            normalized: complexity,
            confidence: if data.len() > 100 { 0.8 } else { 0.5 },
            method: ComplexityMethod::Structural,
        }
    }
    
    /// Get estimated code length from complexity
    pub fn estimate_code_length(&mut self, data: &[u8]) -> Result<usize, &'static str> {
        let estimate = self.estimate(data)?;
        // Convert normalized complexity to approximate code length in bytes
        let code_length = (estimate.complexity_bits / 8.0).ceil() as usize;
        Ok(code_length.max(1))
    }
}

impl Default for KolmogorovComplexity {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_estimator_creation() {
        let estimator = KolmogorovComplexity::new();
        assert!(estimator.buffer.is_empty());
    }
    
    #[test]
    fn test_empty_data_rejected() {
        let mut estimator = KolmogorovComplexity::new();
        let result = estimator.estimate(&[]);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_short_data_rejected() {
        let mut estimator = KolmogorovComplexity::new();
        let result = estimator.estimate(&[1, 2, 3]);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_simple_pattern_low_complexity() {
        let mut estimator = KolmogorovComplexity::new();
        let data = vec![0u8; 100]; // All zeros - very simple
        
        let result = estimator.estimate(&data);
        assert!(result.is_ok());
        let estimate = result.unwrap();
        
        // Simple patterns should have low normalized complexity
        assert!(estimate.normalized < 0.5);
    }
    
    #[test]
    fn test_random_data_high_complexity() {
        let mut estimator = KolmogorovComplexity::new();
        
        // Pseudo-random data with high entropy
        let data: Vec<u8> = (0..100).map(|i| ((i * 17 + 31) % 256) as u8).collect();
        
        let result = estimator.estimate(&data);
        assert!(result.is_ok());
        let estimate = result.unwrap();
        
        // Random-like data should have higher complexity
        assert!(estimate.normalized > 0.5);
    }
    
    #[test]
    fn test_code_length_estimation() {
        let mut estimator = KolmogorovComplexity::new();
        let data = vec![0u8; 100];
        
        let length = estimator.estimate_code_length(&data);
        assert!(length.is_ok());
        assert!(length.unwrap() >= 1);
    }
}
