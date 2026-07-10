"""
STAGE 24: SELF-CRITIQUE LOOP
=============================

Implements Constitutional AI self-critique for NEXUS-OMEGA.
Generates multiple candidate execution plans and ranks them
against the Financial Constitution before forwarding to OMS.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class CritiqueOutcome(Enum):
    """Outcome of the self-critique evaluation."""
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    AMBIGUOUS = "ambiguous"


@dataclass
class ExecutionPlan:
    """Represents a candidate execution plan."""
    plan_id: str
    actions: List[Dict[str, Any]]
    expected_return: float
    expected_risk: float
    constitutional_scores: Dict[str, float] = field(default_factory=dict)
    critique_feedback: List[str] = field(default_factory=list)
    overall_score: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class CritiqueResult:
    """Result of critiquing an execution plan."""
    plan_id: str
    outcome: CritiqueOutcome
    constitutional_violations: List[str]
    strengths: List[str]
    weaknesses: List[str]
    suggested_improvements: List[str]
    score: float


class ConstitutionalCritic:
    """
    Critic model that evaluates execution plans against the Constitution.
    
    Uses a separate neural network to provide unbiased evaluation
    of candidate plans.
    """
    
    def __init__(self, constitution):
        self.constitution = constitution
        self.critique_history = []
        
        # Critique dimensions
        self.critique_dimensions = [
            'transparency',
            'risk_alignment',
            'market_impact',
            'ethical_compliance',
            'long_term_sustainability'
        ]
    
    def critique_plan(self, plan: ExecutionPlan, market_state: Dict[str, Any]) -> CritiqueResult:
        """
        Critique an execution plan against constitutional principles.
        
        Args:
            plan: The execution plan to evaluate
            market_state: Current market state context
            
        Returns:
            Detailed critique result
        """
        violations = []
        strengths = []
        weaknesses = []
        improvements = []
        dimension_scores = {}
        
        # Check each constitutional principle
        compliance_results = self.constitution.check_compliance({
            **market_state,
            'planned_actions': plan.actions
        })
        
        for result in compliance_results:
            if not result.passed:
                violations.append(f"{result.axiom.principle.value}: {result.details}")
                weaknesses.append(f"Violates {result.axiom.principle.value}")
                improvements.append(f"Revise plan to satisfy {result.axiom.principle.value}")
            else:
                strengths.append(f"Satisfies {result.axiom.principle.value}")
        
        # Evaluate critique dimensions
        dimension_scores = self._evaluate_dimensions(plan, market_state)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(violations, dimension_scores)
        
        # Determine outcome
        if len(violations) > 0:
            outcome = CritiqueOutcome.REJECTED
        elif overall_score < 0.5:
            outcome = CritiqueOutcome.NEEDS_REVISION
        elif overall_score < 0.7:
            outcome = CritiqueOutcome.AMBIGUOUS
        else:
            outcome = CritiqueOutcome.APPROVED
        
        critique_result = CritiqueResult(
            plan_id=plan.plan_id,
            outcome=outcome,
            constitutional_violations=violations,
            strengths=strengths,
            weaknesses=weaknesses,
            suggested_improvements=improvements,
            score=overall_score
        )
        
        self.critique_history.append(critique_result)
        
        return critique_result
    
    def _evaluate_dimensions(
        self,
        plan: ExecutionPlan,
        market_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate plan across multiple critique dimensions."""
        scores = {}
        
        # Transparency score
        scores['transparency'] = self._score_transparency(plan)
        
        # Risk alignment score
        scores['risk_alignment'] = self._score_risk_alignment(plan, market_state)
        
        # Market impact score
        scores['market_impact'] = self._score_market_impact(plan, market_state)
        
        # Ethical compliance score
        scores['ethical_compliance'] = self._score_ethical_compliance(plan)
        
        # Long-term sustainability score
        scores['long_term_sustainability'] = self._score_sustainability(plan)
        
        return scores
    
    def _score_transparency(self, plan: ExecutionPlan) -> float:
        """Score transparency of the plan."""
        # Plans with clear rationale score higher
        has_rationale = all('rationale' in action for action in plan.actions)
        return 1.0 if has_rationale else 0.5
    
    def _score_risk_alignment(
        self,
        plan: ExecutionPlan,
        market_state: Dict[str, Any]
    ) -> float:
        """Score alignment with risk objectives."""
        max_allowed_risk = market_state.get('max_allowed_risk', 0.05)
        plan_risk = plan.expected_risk
        
        if plan_risk <= max_allowed_risk:
            return 1.0
        elif plan_risk <= max_allowed_risk * 1.5:
            return 0.5
        else:
            return 0.0
    
    def _score_market_impact(
        self,
        plan: ExecutionPlan,
        market_state: Dict[str, Any]
    ) -> float:
        """Score market impact of the plan."""
        # Lower impact = higher score
        total_size = sum(action.get('size', 0) for action in plan.actions)
        market_liquidity = market_state.get('market_liquidity', 1.0)
        
        impact_ratio = total_size / (market_liquidity + 1e-8)
        
        if impact_ratio < 0.01:
            return 1.0
        elif impact_ratio < 0.05:
            return 0.7
        elif impact_ratio < 0.1:
            return 0.3
        else:
            return 0.0
    
    def _score_ethical_compliance(self, plan: ExecutionPlan) -> float:
        """Score ethical compliance of the plan."""
        # Check for predatory patterns
        predatory_indicators = sum(
            1 for action in plan.actions
            if action.get('pattern') in ['spoofing', 'layering', 'wash_trading']
        )
        
        if predatory_indicators == 0:
            return 1.0
        else:
            return max(0, 1.0 - predatory_indicators * 0.3)
    
    def _score_sustainability(self, plan: ExecutionPlan) -> float:
        """Score long-term sustainability."""
        # Plans balanced between return and risk score higher
        if plan.expected_risk == 0:
            return 0.5
        
        sharpe_like = plan.expected_return / plan.expected_risk
        
        if sharpe_like > 2.0:
            return 1.0
        elif sharpe_like > 1.0:
            return 0.7
        elif sharpe_like > 0.5:
            return 0.4
        else:
            return 0.2
    
    def _calculate_overall_score(
        self,
        violations: List[str],
        dimension_scores: Dict[str, float]
    ) -> float:
        """Calculate overall critique score."""
        # Penalty for violations
        violation_penalty = len(violations) * 0.2
        
        # Average dimension scores
        if dimension_scores:
            dimension_avg = np.mean(list(dimension_scores.values()))
        else:
            dimension_avg = 0.5
        
        overall = dimension_avg - violation_penalty
        return max(0.0, min(1.0, overall))


class SelfCritiqueLoop:
    """
    Main self-critique loop that generates and evaluates candidate plans.
    
    Only forwards plans that pass all constitutional checks to the OMS.
    """
    
    def __init__(
        self,
        constitution,
        num_candidates: int = 5,
        max_iterations: int = 3
    ):
        self.constitution = constitution
        self.num_candidates = num_candidates
        self.max_iterations = max_iterations
        
        self.critic = ConstitutionalCritic(constitution)
        
        self.approved_plans = []
        self.rejected_plans = []
        self.iteration_history = []
    
    def generate_candidate_plans(
        self,
        market_state: Dict[str, Any],
        policy_model
    ) -> List[ExecutionPlan]:
        """
        Generate multiple candidate execution plans.
        
        Args:
            market_state: Current market state
            policy_model: The primary policy model
            
        Returns:
            List of candidate execution plans
        """
        candidates = []
        
        for i in range(self.num_candidates):
            # Generate plan using policy model with different sampling
            plan_actions = policy_model.generate_plan(
                market_state,
                temperature=0.5 + i * 0.2  # Vary temperature for diversity
            )
            
            plan = ExecutionPlan(
                plan_id=f"plan_{int(time.time() * 1000)}_{i}",
                actions=plan_actions,
                expected_return=self._estimate_return(plan_actions, market_state),
                expected_risk=self._estimate_risk(plan_actions, market_state)
            )
            
            candidates.append(plan)
        
        return candidates
    
    def execute_critique_loop(
        self,
        market_state: Dict[str, Any],
        policy_model
    ) -> Optional[ExecutionPlan]:
        """
        Execute the full self-critique loop.
        
        Args:
            market_state: Current market state
            policy_model: The primary policy model
            
        Returns:
            Approved execution plan or None if no plan passes
        """
        iteration_results = []
        
        for iteration in range(self.max_iterations):
            logger.info(f"Critique Loop Iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate candidates
            candidates = self.generate_candidate_plans(market_state, policy_model)
            
            # Critique each candidate
            best_plan = None
            best_score = -1
            
            for plan in candidates:
                critique_result = self.critic.critique_plan(plan, market_state)
                plan.constitutional_scores = {
                    'overall': critique_result.score,
                    'violations': len(critique_result.constitutional_violations)
                }
                plan.critique_feedback = critique_result.suggested_improvements
                
                if critique_result.outcome == CritiqueOutcome.APPROVED:
                    if critique_result.score > best_score:
                        best_score = critique_result.score
                        best_plan = plan
                elif critique_result.outcome == CritiqueOutcome.NEEDS_REVISION:
                    # Could revise plan here
                    pass
            
            if best_plan is not None:
                self.approved_plans.append(best_plan)
                logger.info(f"Plan approved with score {best_score:.2f}")
                return best_plan
            
            iteration_results.append({
                'iteration': iteration,
                'candidates_evaluated': len(candidates),
                'plans_approved': 0
            })
        
        # No plan passed all checks
        logger.warning("No execution plan passed constitutional checks")
        self.iteration_history.extend(iteration_results)
        
        return None
    
    def _estimate_return(
        self,
        actions: List[Dict[str, Any]],
        market_state: Dict[str, Any]
    ) -> float:
        """Estimate expected return of actions."""
        # Simplified estimation
        base_return = market_state.get('expected_market_return', 0.0)
        action_alpha = sum(a.get('alpha', 0) for a in actions) / len(actions)
        return base_return + action_alpha
    
    def _estimate_risk(
        self,
        actions: List[Dict[str, Any]],
        market_state: Dict[str, Any]
    ) -> float:
        """Estimate expected risk of actions."""
        base_volatility = market_state.get('market_volatility', 0.02)
        position_size = sum(a.get('size', 0) for a in actions)
        leverage = sum(a.get('leverage', 1) for a in actions) / len(actions)
        
        return base_volatility * position_size * leverage
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the critique loop."""
        return {
            'total_candidates_generated': len(self.approved_plans) + len(self.rejected_plans),
            'plans_approved': len(self.approved_plans),
            'plans_rejected': len(self.rejected_plans),
            'approval_rate': len(self.approved_plans) / max(1, len(self.approved_plans) + len(self.rejected_plans)),
            'average_approved_score': np.mean([p.overall_score for p in self.approved_plans]) if self.approved_plans else 0
        }


# Example usage placeholder
if __name__ == "__main__":
    print("Self-Critique Loop module loaded successfully")
    print("Requires FinancialConstitution and policy model for full operation")
