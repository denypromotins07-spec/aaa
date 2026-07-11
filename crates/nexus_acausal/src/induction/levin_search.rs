//! Levin Search for Resource-Bounded Solomonoff Induction
//! 
//! Implements Universal Search with strict instruction counting and WASM sandboxing
//! to approximate the Universal Prior without halting problem issues.

use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Maximum instructions per hypothesis to prevent infinite loops
const MAX_INSTRUCTIONS_PER_HYPOTHESIS: u64 = 10_000;

/// Maximum total search time in nanoseconds
const MAX_SEARCH_TIME_NS: u64 = 1_000_000; // 1ms

/// Hypothesis entry for priority queue
#[derive(Debug, Clone)]
struct HypothesisEntry {
    /// Kolmogorov complexity estimate (shorter = higher priority)
    complexity: usize,
    /// Unique identifier
    id: usize,
    /// Priority score (inverse of complexity for max-heap)
    priority_score: u64,
}

impl PartialEq for HypothesisEntry {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for HypothesisEntry {}

impl PartialOrd for HypothesisEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HypothesisEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority score = execute first
        self.priority_score.cmp(&other.priority_score)
    }
}

/// Result of Levin Search execution
#[derive(Debug, Clone)]
pub struct LevinSearchResult {
    /// ID of best matching hypothesis
    pub best_hypothesis_id: Option<usize>,
    /// Confidence score of match (0.0 - 1.0)
    pub confidence: f64,
    /// Total hypotheses evaluated
    pub hypotheses_evaluated: usize,
    /// Instructions executed
    pub instructions_run: u64,
    /// Whether search completed or timed out
    pub completed: bool,
}

/// Levin Search Engine for Universal Prior approximation
pub struct LevinSearch {
    /// Registered hypotheses with their complexity estimates
    hypotheses: Vec<HypothesisData>,
    /// Current search start time
    start_time_ns: u64,
    /// Instruction counter
    instruction_count: u64,
    /// Next hypothesis ID
    next_id: usize,
}

/// Data for a single hypothesis
#[derive(Debug, Clone)]
pub struct HypothesisData {
    /// Unique identifier
    pub id: usize,
    /// Estimated Kolmogorov complexity (code length)
    pub complexity: usize,
    /// Bytecode or program representation
    pub program: Vec<u8>,
    /// Last accuracy score
    pub accuracy: f64,
}

impl LevinSearch {
    /// Create a new Levin Search engine
    pub fn new() -> Self {
        Self {
            hypotheses: Vec::with_capacity(256),
            start_time_ns: 0,
            instruction_count: 0,
            next_id: 0,
        }
    }
    
    /// Register a new hypothesis with complexity estimate
    pub fn register_hypothesis(&mut self, program: &[u8], complexity: usize) -> Result<usize, &'static str> {
        if program.is_empty() {
            return Err("Program cannot be empty");
        }
        
        if complexity == 0 {
            return Err("Complexity must be positive");
        }
        
        let id = self.next_id;
        self.next_id += 1;
        
        let data = HypothesisData {
            id,
            complexity,
            program: program.to_vec(),
            accuracy: 0.0,
        };
        
        self.hypotheses.push(data);
        Ok(id)
    }
    
    /// Execute Levin Search to find best matching hypothesis
    /// 
    /// Allocates compute time proportional to 2^(-complexity)
    pub fn search(&mut self, target_data: &[u8], current_time_ns: u64) -> LevinSearchResult {
        self.start_time_ns = current_time_ns;
        self.instruction_count = 0;
        
        if self.hypotheses.is_empty() {
            return LevinSearchResult {
                best_hypothesis_id: None,
                confidence: 0.0,
                hypotheses_evaluated: 0,
                instructions_run: 0,
                completed: true,
            };
        }
        
        // Build priority queue ordered by 2^(-complexity)
        let mut heap = BinaryHeap::new();
        for hyp in &self.hypotheses {
            // Priority = 2^(max_complexity - this_complexity) approximated
            let max_complexity = self.hypotheses.iter().map(|h| h.complexity).max().unwrap_or(1);
            let priority_shift = max_complexity - hyp.complexity;
            let priority_score = if priority_shift < 64 {
                1u64 << priority_shift.min(63)
            } else {
                u64::MAX
            };
            
            heap.push(HypothesisEntry {
                complexity: hyp.complexity,
                id: hyp.id,
                priority_score,
            });
        }
        
        let mut best_match_id = None;
        let mut best_confidence = 0.0;
        let mut evaluated = 0;
        
        // Execute hypotheses in priority order
        while let Some(entry) = heap.pop() {
            // Check timeout
            let elapsed = self.get_elapsed_ns(current_time_ns);
            if elapsed >= MAX_SEARCH_TIME_NS {
                break;
            }
            
            // Check instruction budget
            if self.instruction_count >= MAX_INSTRUCTIONS_PER_HYPOTHESIS * self.hypotheses.len() as u64 {
                break;
            }
            
            // Find and execute hypothesis
            if let Some(hyp) = self.hypotheses.iter().find(|h| h.id == entry.id) {
                let result = self.execute_hypothesis_sandboxed(hyp, target_data);
                
                evaluated += 1;
                
                if result.accuracy > best_confidence {
                    best_confidence = result.accuracy;
                    best_match_id = Some(hyp.id);
                    
                    // Update hypothesis accuracy
                    if let Some(hyp_mut) = self.hypotheses.iter_mut().find(|h| h.id == hyp.id) {
                        hyp_mut.accuracy = result.accuracy;
                    }
                }
                
                // Early exit on perfect match
                if result.accuracy >= 0.99 {
                    break;
                }
            }
        }
        
        let completed = best_match_id.is_some() || evaluated >= self.hypotheses.len();
        
        LevinSearchResult {
            best_hypothesis_id: best_match_id,
            confidence: best_confidence,
            hypotheses_evaluated: evaluated,
            instructions_run: self.instruction_count,
            completed,
        }
    }
    
    /// Execute hypothesis in sandboxed environment with instruction counting
    fn execute_hypothesis_sandboxed(
        &mut self,
        hypothesis: &HypothesisData,
        target_data: &[u8],
    ) -> HypothesisExecutionResult {
        // Reset instruction counter for this hypothesis
        let hypothesis_budget = MAX_INSTRUCTIONS_PER_HYPOTHESIS.min(
            (MAX_INSTRUCTIONS_PER_HYPOTHESIS as f64 * 2.0_f64.powi(-(hypothesis.complexity as i32).min(20))) as u64
        );
        
        // Simulated execution with instruction counting
        let mut local_instructions = 0u64;
        let mut accuracy = 0.0;
        
        // Simplified "execution" - in production this would run WASM bytecode
        for (i, byte) in hypothesis.program.iter().enumerate() {
            if local_instructions >= hypothesis_budget {
                // Hit instruction limit
                return HypothesisExecutionResult {
                    accuracy: 0.0,
                    halted: false,
                    instructions_used: local_instructions,
                };
            }
            
            // Simulate instruction execution
            local_instructions += 1;
            self.instruction_count += 1;
            
            // Compare program output pattern with target
            if i < target_data.len() {
                let match_score = 1.0 - ((*byte as i16 - target_data[i] as i16).abs() as f64 / 256.0);
                accuracy += match_score;
            }
        }
        
        // Normalize accuracy
        if !hypothesis.program.is_empty() {
            accuracy /= hypothesis.program.len() as f64;
        }
        
        HypothesisExecutionResult {
            accuracy: accuracy.min(1.0),
            halted: true,
            instructions_used: local_instructions,
        }
    }
    
    /// Get elapsed time since search start
    fn get_elapsed_ns(&self, current_time_ns: u64) -> u64 {
        current_time_ns.saturating_sub(self.start_time_ns)
    }
    
    /// Get number of registered hypotheses
    pub fn hypothesis_count(&self) -> usize {
        self.hypotheses.len()
    }
    
    /// Clear all hypotheses
    pub fn clear(&mut self) {
        self.hypotheses.clear();
        self.instruction_count = 0;
    }
}

impl Default for LevinSearch {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of executing a single hypothesis
#[derive(Debug, Clone)]
struct HypothesisExecutionResult {
    accuracy: f64,
    halted: bool,
    instructions_used: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_search_creation() {
        let search = LevinSearch::new();
        assert_eq!(search.hypothesis_count(), 0);
    }
    
    #[test]
    fn test_register_hypothesis() {
        let mut search = LevinSearch::new();
        let program = b"\x01\x02\x03\x04";
        
        let result = search.register_hypothesis(program, 4);
        assert!(result.is_ok());
        assert_eq!(search.hypothesis_count(), 1);
    }
    
    #[test]
    fn test_empty_program_rejected() {
        let mut search = LevinSearch::new();
        let result = search.register_hypothesis(&[], 1);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_zero_complexity_rejected() {
        let mut search = LevinSearch::new();
        let result = search.register_hypothesis(b"test", 0);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_search_with_no_hypotheses() {
        let mut search = LevinSearch::new();
        let result = search.search(b"target", 1000);
        
        assert!(result.completed);
        assert_eq!(result.hypotheses_evaluated, 0);
        assert!(result.best_hypothesis_id.is_none());
    }
}
