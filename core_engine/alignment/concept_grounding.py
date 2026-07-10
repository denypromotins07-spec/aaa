"""
STAGE 24: CONCEPT GROUNDING MODULE
====================================

Verifies that the AI's internal embeddings for financial concepts
remain anchored to real-world observable phenomena, preventing
semantic drift where definitions diverge from economic reality.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConceptDefinition:
    """Definition of a financial concept with grounding criteria."""
    name: str
    description: str
    observable_indicators: List[str]
    expected_range: Tuple[float, float]
    validation_function: str
    last_validated: float = field(default_factory=time.time)
    drift_score: float = 0.0


@dataclass
class GroundingCheckResult:
    """Result of a concept grounding check."""
    concept_name: str
    is_grounded: bool
    drift_magnitude: float
    observable_match_score: float
    recommendations: List[str]
    timestamp: float = field(default_factory=time.time)


class ConceptGroundingModule:
    """
    Module for verifying that AI concepts remain grounded in reality.
    
    Prevents ontological drift by continuously checking that internal
    representations match observable market phenomena.
    """
    
    def __init__(self):
        self.concepts: Dict[str, ConceptDefinition] = {}
        self.grounding_history: List[GroundingCheckResult] = []
        
        # Register core financial concepts
        self._register_core_concepts()
    
    def _register_core_concepts(self):
        """Register core financial concepts that must remain grounded."""
        
        # Liquidity concept
        self.register_concept(ConceptDefinition(
            name="liquidity",
            description="The ease with which an asset can be bought or sold without affecting price",
            observable_indicators=[
                'bid_ask_spread',
                'market_depth',
                'trading_volume',
                'price_impact'
            ],
            expected_range=(0.0, 1.0),
            validation_function="check_liquidity_grounding"
        ))
        
        # Volatility concept
        self.register_concept(ConceptDefinition(
            name="volatility",
            description="Statistical measure of price dispersion around mean",
            observable_indicators=[
                'standard_deviation',
                'realized_vol',
                'implied_vol',
                'price_range'
            ],
            expected_range=(0.0, 5.0),
            validation_function="check_volatility_grounding"
        ))
        
        # Credit risk concept
        self.register_concept(ConceptDefinition(
            name="credit_risk",
            description="Risk of counterparty default on obligations",
            observable_indicators=[
                'credit_spread',
                'cds_premium',
                'rating_downgrade',
                'default_probability'
            ],
            expected_range=(0.0, 1.0),
            validation_function="check_credit_risk_grounding"
        ))
        
        # Market impact concept
        self.register_concept(ConceptDefinition(
            name="market_impact",
            description="Price movement caused by executing a trade",
            observable_indicators=[
                'slippage',
                'price_movement_post_trade',
                'order_book_depletion',
                'recovery_time'
            ],
            expected_range=(0.0, 0.1),
            validation_function="check_market_impact_grounding"
        ))
        
        # Correlation concept
        self.register_concept(ConceptDefinition(
            name="correlation",
            description="Statistical relationship between asset returns",
            observable_indicators=[
                'pearson_correlation',
                'rank_correlation',
                'cointegration',
                'beta'
            ],
            expected_range=(-1.0, 1.0),
            validation_function="check_correlation_grounding"
        ))
    
    def register_concept(self, concept: ConceptDefinition):
        """Register a financial concept for grounding checks."""
        self.concepts[concept.name] = concept
        logger.info(f"Registered concept: {concept.name}")
    
    def check_concept_grounding(
        self,
        concept_name: str,
        internal_embedding: np.ndarray,
        market_data: Dict[str, Any]
    ) -> GroundingCheckResult:
        """
        Check if a concept remains grounded in observable reality.
        
        Args:
            concept_name: Name of the concept to check
            internal_embedding: AI's internal representation
            market_data: Current observable market data
            
        Returns:
            Grounding check result
        """
        if concept_name not in self.concepts:
            return GroundingCheckResult(
                concept_name=concept_name,
                is_grounded=False,
                drift_magnitude=1.0,
                observable_match_score=0.0,
                recommendations=["Concept not registered"]
            )
        
        concept = self.concepts[concept_name]
        
        # Extract observable values from market data
        observable_values = self._extract_observables(concept, market_data)
        
        # Calculate observable match score
        match_score = self._calculate_observable_match(concept, observable_values)
        
        # Calculate drift magnitude
        drift_magnitude = self._calculate_drift(internal_embedding, observable_values)
        
        # Determine if grounded
        is_grounded = (
            match_score > 0.6 and
            drift_magnitude < 0.5 and
            self._check_expected_range(concept, observable_values)
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            concept, match_score, drift_magnitude, is_grounded
        )
        
        # Update concept state
        concept.drift_score = drift_magnitude
        concept.last_validated = time.time()
        
        result = GroundingCheckResult(
            concept_name=concept_name,
            is_grounded=is_grounded,
            drift_magnitude=drift_magnitude,
            observable_match_score=match_score,
            recommendations=recommendations
        )
        
        self.grounding_history.append(result)
        
        if not is_grounded:
            logger.warning(f"CONCEPT DRIFT DETECTED: {concept_name} (drift={drift_magnitude:.2f})")
        
        return result
    
    def _extract_observables(
        self,
        concept: ConceptDefinition,
        market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract observable indicator values from market data."""
        observables = {}
        
        for indicator in concept.observable_indicators:
            if indicator in market_data:
                observables[indicator] = float(market_data[indicator])
            else:
                # Default value if not available
                observables[indicator] = 0.5
        
        return observables
    
    def _calculate_observable_match(
        self,
        concept: ConceptDefinition,
        observables: Dict[str, float]
    ) -> float:
        """Calculate how well observables match expected patterns."""
        if not observables:
            return 0.0
        
        # Normalize and average observable values
        normalized_values = []
        for indicator, value in observables.items():
            # Normalize to 0-1 range based on expected range
            min_val, max_val = concept.expected_range
            if max_val > min_val:
                normalized = (value - min_val) / (max_val - min_val)
                normalized = max(0, min(1, normalized))
                normalized_values.append(normalized)
        
        if not normalized_values:
            return 0.0
        
        # Consistency check - low variance indicates good grounding
        variance = np.var(normalized_values)
        consistency = 1.0 / (1.0 + variance * 10)
        
        return float(consistency)
    
    def _calculate_drift(
        self,
        internal_embedding: np.ndarray,
        observables: Dict[str, float]
    ) -> float:
        """Calculate drift between internal representation and observables."""
        if len(observables) == 0:
            return 1.0
        
        # Convert observables to vector
        obs_vector = np.array(list(observables.values()))
        
        # Normalize both vectors
        if np.linalg.norm(internal_embedding) > 0:
            internal_norm = internal_embedding / np.linalg.norm(internal_embedding)
        else:
            return 1.0
        
        if np.linalg.norm(obs_vector) > 0:
            obs_norm = obs_vector / np.linalg.norm(obs_vector)
        else:
            return 1.0
        
        # Cosine distance as drift measure
        # Pad shorter vector if dimensions don't match
        min_len = min(len(internal_norm), len(obs_norm))
        internal_trimmed = internal_norm[:min_len]
        obs_trimmed = obs_norm[:min_len]
        
        cosine_sim = np.dot(internal_trimmed, obs_trimmed)
        drift = 1.0 - abs(cosine_sim)  # 0 = perfectly aligned, 1 = orthogonal
        
        return float(drift)
    
    def _check_expected_range(
        self,
        concept: ConceptDefinition,
        observables: Dict[str, float]
    ) -> bool:
        """Check if observable values are within expected range."""
        min_val, max_val = concept.expected_range
        
        for value in observables.values():
            if value < min_val or value > max_val * 2:  # Allow some tolerance
                return False
        
        return True
    
    def _generate_recommendations(
        self,
        concept: ConceptDefinition,
        match_score: float,
        drift_magnitude: float,
        is_grounded: bool
    ) -> List[str]:
        """Generate recommendations for maintaining concept grounding."""
        recommendations = []
        
        if not is_grounded:
            recommendations.append(f"Re-align {concept.name} embedding with observable indicators")
            
            if match_score < 0.6:
                recommendations.append("Observable indicators show inconsistent patterns")
            
            if drift_magnitude > 0.5:
                recommendations.append("Internal representation has drifted from market reality")
                recommendations.append("Consider retraining concept embedding on recent data")
            
            recommendations.append("Increase frequency of grounding checks")
        else:
            recommendations.append("Concept remains well-grounded; continue monitoring")
        
        return recommendations
    
    def check_all_concepts(
        self,
        embeddings: Dict[str, np.ndarray],
        market_data: Dict[str, Any]
    ) -> Dict[str, GroundingCheckResult]:
        """Check grounding for all registered concepts."""
        results = {}
        
        for concept_name in self.concepts:
            if concept_name in embeddings:
                results[concept_name] = self.check_concept_grounding(
                    concept_name=concept_name,
                    internal_embedding=embeddings[concept_name],
                    market_data=market_data
                )
            else:
                results[concept_name] = GroundingCheckResult(
                    concept_name=concept_name,
                    is_grounded=False,
                    drift_magnitude=1.0,
                    observable_match_score=0.0,
                    recommendations=["No embedding provided"]
                )
        
        return results
    
    def get_grounding_statistics(self) -> Dict[str, Any]:
        """Get statistics about concept grounding."""
        if not self.grounding_history:
            return {'message': 'No grounding checks performed yet'}
        
        grounded_count = sum(1 for r in self.grounding_history if r.is_grounded)
        total_checks = len(self.grounding_history)
        
        avg_drift_by_concept = {}
        for concept_name in self.concepts:
            concept_results = [r for r in self.grounding_history if r.concept_name == concept_name]
            if concept_results:
                avg_drift_by_concept[concept_name] = np.mean([r.drift_magnitude for r in concept_results])
        
        return {
            'total_checks': total_checks,
            'grounded_checks': grounded_count,
            'grounding_rate': grounded_count / total_checks,
            'average_drift': np.mean([r.drift_magnitude for r in self.grounding_history]),
            'drift_by_concept': avg_drift_by_concept
        }


# Example usage placeholder
if __name__ == "__main__":
    print("Concept Grounding Module loaded successfully")
    module = ConceptGroundingModule()
    print(f"Registered concepts: {list(module.concepts.keys())}")
