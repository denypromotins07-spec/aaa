//! Probabilistic Venue Selector for Ant Colony System.
//! 
//! Implements roulette wheel selection and other probabilistic methods
//! for venue selection based on pheromone levels and heuristic visibility.

use nexus_types::market::VenueId;
use rand::Rng;
use thiserror::Error;

/// Probability value normalized to [0, 1]
#[derive(Debug, Clone, Copy)]
pub struct VenueSelectionProb(pub f64);

impl VenueSelectionProb {
    pub fn new(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    pub fn raw(&self) -> f64 {
        self.0
    }
}

/// Scored venue for selection
#[derive(Debug, Clone, Copy)]
pub struct ScoredVenue {
    pub venue_id: VenueId,
    pub score: f64,
    pub cumulative_prob: f64,
}

/// Venue selector implementing multiple selection strategies
pub struct VenueSelector {
    /// Pre-computed cumulative probabilities for O(1) selection
    venues: Vec<ScoredVenue>,
    total_score: f64,
}

impl VenueSelector {
    /// Create a new venue selector from scored venues
    pub fn new(scored_venues: &[(VenueId, f64)]) -> Result<Self, SelectionError> {
        if scored_venues.is_empty() {
            return Err(SelectionError::EmptyCandidateSet);
        }

        let mut venues = Vec::with_capacity(scored_venues.len());
        let mut total_score = 0.0;
        let mut cumulative = 0.0;

        // Validate all scores are non-negative
        for (venue_id, score) in scored_venues {
            if *score < 0.0 {
                return Err(SelectionError::NegativeScore(*score));
            }
            total_score += score;
        }

        if total_score <= 0.0 {
            return Err(SelectionError::ZeroTotalScore);
        }

        // Build cumulative probability distribution
        for (venue_id, score) in scored_venues {
            cumulative += score / total_score;
            venues.push(ScoredVenue {
                venue_id: *venue_id,
                score: *score,
                cumulative_prob: cumulative.min(1.0), // Clamp to prevent floating point errors
            });
        }

        Ok(Self { venues, total_score })
    }

    /// Select a venue using roulette wheel selection
    pub fn select<R: Rng>(&self, rng: &mut R) -> VenueId {
        if self.venues.len() == 1 {
            return self.venues[0].venue_id;
        }

        let r: f64 = rng.gen();
        
        // Binary search for efficiency with large venue sets
        self.binary_search_selection(r)
    }

    /// Binary search implementation for O(log n) selection
    fn binary_search_selection(&self, r: f64) -> VenueId {
        let mut low = 0;
        let mut high = self.venues.len() - 1;

        while low < high {
            let mid = low + (high - low) / 2;
            if self.venues[mid].cumulative_prob < r {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        self.venues[low].venue_id
    }

    /// Select top-k venues by score
    pub fn select_top_k(&self, k: usize) -> Vec<VenueId> {
        let mut sorted: Vec<&ScoredVenue> = self.venues.iter().collect();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        sorted.iter()
            .take(k)
            .map(|v| v.venue_id)
            .collect()
    }

    /// Get the total score of all venues
    pub fn total_score(&self) -> f64 {
        self.total_score
    }

    /// Get number of candidate venues
    pub fn len(&self) -> usize {
        self.venues.len()
    }

    /// Check if selector is empty
    pub fn is_empty(&self) -> bool {
        self.venues.is_empty()
    }

    /// Get probability for a specific venue
    pub fn get_probability(&self, venue_id: VenueId) -> Option<f64> {
        self.venues.iter()
            .find(|v| v.venue_id == venue_id)
            .map(|v| {
                // Calculate actual probability from cumulative differences
                let idx = self.venues.iter().position(|x| x.venue_id == venue_id)?;
                if idx == 0 {
                    Some(v.cumulative_prob)
                } else {
                    Some(v.cumulative_prob - self.venues[idx - 1].cumulative_prob)
                }
            })
            .flatten()
    }
}

/// Stochastic Universal Sampling (SUS) for lower variance selection
pub struct SusSelector {
    venues: Vec<ScoredVenue>,
}

impl SusSelector {
    pub fn new(scored_venues: &[(VenueId, f64)]) -> Result<Self, SelectionError> {
        if scored_venues.is_empty() {
            return Err(SelectionError::EmptyCandidateSet);
        }

        let mut venues = Vec::with_capacity(scored_venues.len());
        let mut total_score = 0.0;
        let mut cumulative = 0.0;

        for (venue_id, score) in scored_venues {
            if *score < 0.0 {
                return Err(SelectionError::NegativeScore(*score));
            }
            total_score += score;
        }

        if total_score <= 0.0 {
            return Err(SelectionError::ZeroTotalScore);
        }

        for (venue_id, score) in scored_venues {
            cumulative += score / total_score;
            venues.push(ScoredVenue {
                venue_id: *venue_id,
                score: *score,
                cumulative_prob: cumulative.min(1.0),
            });
        }

        Ok(Self { venues })
    }

    /// Select n venues using Stochastic Universal Sampling
    /// This provides lower variance than independent roulette wheel selections
    pub fn select_n<R: Rng>(&self, rng: &mut R, n: usize) -> Vec<VenueId> {
        if n == 0 || self.venues.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(n.min(self.venues.len()));
        let pointer_spacing = 1.0 / n as f64;
        let mut pointer = rng.gen::<f64>() * pointer_spacing;

        let mut current_idx = 0;
        for _ in 0..n {
            while pointer > self.venues[current_idx].cumulative_prob {
                current_idx = (current_idx + 1).min(self.venues.len() - 1);
            }
            result.push(self.venues[current_idx].venue_id);
            pointer += pointer_spacing;
        }

        result
    }
}

/// Tournament selection for diversity preservation
pub struct TournamentSelector {
    tournament_size: usize,
}

impl TournamentSelector {
    pub fn new(tournament_size: usize) -> Self {
        Self {
            tournament_size: tournament_size.max(2),
        }
    }

    /// Select a venue using tournament selection
    pub fn select<R: Rng>(&self, rng: &mut R, candidates: &[(VenueId, f64)]) -> Result<VenueId, SelectionError> {
        if candidates.is_empty() {
            return Err(SelectionError::EmptyCandidateSet);
        }

        // Randomly select tournament participants
        let tournament_count = self.tournament_size.min(candidates.len());
        let mut best_venue = candidates[0].0;
        let mut best_score = candidates[0].1;

        for i in 1..tournament_count {
            let idx = rng.gen_range(0..candidates.len());
            if candidates[idx].1 > best_score {
                best_score = candidates[idx].1;
                best_venue = candidates[idx].0;
            }
        }

        Ok(best_venue)
    }
}

/// Errors for selection operations
#[derive(Debug, Error)]
pub enum SelectionError {
    #[error("Empty candidate set")]
    EmptyCandidateSet,
    #[error("Negative score encountered: {0}")]
    NegativeScore(f64),
    #[error("Total score is zero or negative")]
    ZeroTotalScore,
    #[error("Invalid probability distribution")]
    InvalidDistribution,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_venue_selector_creation() {
        let venues = vec![
            (VenueId::new(0), 1.0),
            (VenueId::new(1), 2.0),
            (VenueId::new(2), 3.0),
        ];

        let selector = VenueSelector::new(&venues).unwrap();
        assert_eq!(selector.len(), 3);
        assert!((selector.total_score() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_roulette_wheel_selection_distribution() {
        let venues = vec![
            (VenueId::new(0), 1.0),
            (VenueId::new(1), 1.0),
        ];

        let selector = VenueSelector::new(&venues).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        // Run many selections to verify distribution
        let mut counts = [0usize; 2];
        for _ in 0..1000 {
            let selected = selector.select(&mut rng);
            counts[selected.0 as usize] += 1;
        }

        // With equal scores, each should be selected ~50% of the time
        // Allow some variance due to randomness
        assert!(counts[0] > 400 && counts[0] < 600);
        assert!(counts[1] > 400 && counts[1] < 600);
    }

    #[test]
    fn test_top_k_selection() {
        let venues = vec![
            (VenueId::new(0), 1.0),
            (VenueId::new(1), 5.0),
            (VenueId::new(2), 3.0),
            (VenueId::new(3), 2.0),
        ];

        let selector = VenueSelector::new(&venues).unwrap();
        let top_2 = selector.select_top_k(2);

        assert_eq!(top_2.len(), 2);
        assert!(top_2.contains(&VenueId::new(1))); // Highest score
        assert!(top_2.contains(&VenueId::new(2))); // Second highest
    }

    #[test]
    fn test_sus_selection() {
        let venues = vec![
            (VenueId::new(0), 1.0),
            (VenueId::new(1), 2.0),
            (VenueId::new(2), 3.0),
        ];

        let sus_selector = SusSelector::new(&venues).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let selected = sus_selector.select_n(&mut rng, 3);
        assert_eq!(selected.len(), 3);
        // SUS should select proportionally: venue 2 most often, then 1, then 0
    }

    #[test]
    fn test_tournament_selection() {
        let venues = vec![
            (VenueId::new(0), 1.0),
            (VenueId::new(1), 5.0),
            (VenueId::new(2), 3.0),
        ];

        let tournament_selector = TournamentSelector::new(2);
        let mut rng = StdRng::seed_from_u64(42);

        // Tournament selection should favor higher scores
        let mut best_wins = 0;
        for _ in 0..100 {
            let selected = tournament_selector.select(&mut rng, &venues).unwrap();
            if selected == VenueId::new(1) {
                best_wins += 1;
            }
        }

        // Best venue should win majority of tournaments
        assert!(best_wins > 50);
    }

    #[test]
    fn test_empty_candidate_error() {
        let empty: Vec<(VenueId, f64)> = vec![];
        let result = VenueSelector::new(&empty);
        assert!(matches!(result, Err(SelectionError::EmptyCandidateSet)));
    }

    #[test]
    fn test_negative_score_error() {
        let venues = vec![
            (VenueId::new(0), 1.0),
            (VenueId::new(1), -0.5),
        ];
        let result = VenueSelector::new(&venues);
        assert!(matches!(result, Err(SelectionError::NegativeScore(-0.5))));
    }
}
