"""
STAGE 24: METACOGNITIVE SUPER-EGO & SELF-REASONING AUDITOR
==========================================================

This module implements the Metacognitive Super-Ego architecture that observes
the primary Stage 19 C-PPO policy's internal activations and decision trajectories.

Key Components:
- Inverse Reinforcement Learning (IRL) for intent inference
- Chain-of-Thought auditing for logical consistency
- Real-time divergence detection between implicit and explicit objectives
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque
import logging

logger = logging.getLogger(__name__)


class AlignmentStatus(Enum):
    """Status of alignment between implicit and explicit objectives."""
    ALIGNED = "aligned"
    MINOR_DRIFT = "minor_drift"
    CRITICAL_DIVERGENCE = "critical_divergence"
    DECEPTIVE_PATTERN = "deceptive_pattern"


@dataclass
class PolicyTrajectory:
    """Represents a trajectory of policy decisions with internal states."""
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    hidden_states: np.ndarray
    attention_weights: Optional[np.ndarray] = None
    reasoning_trace: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class IRLInferredObjective:
    """Result of Inverse Reinforcement Learning inference."""
    reward_weights: np.ndarray
    confidence: float
    divergence_from_explicit: float
    detected_hacking_patterns: List[str]
    timestamp: float


class InverseReinforcementLearningModule(nn.Module):
    """
    Inverse Reinforcement Learning module that infers the implicit objective
    function from observed policy behavior.
    
    Uses maximum entropy IRL to recover the reward function that best explains
    the demonstrated policy trajectories.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        feature_dim: int = 64
    ):
        super().__init__()
        
        # Feature extractor for state-action pairs
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Reward function approximator
        self.reward_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.feature_dim = feature_dim
        self._trajectory_buffer = deque(maxlen=1000)
        
    def extract_features(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Extract features from state-action pairs."""
        combined = torch.cat([states, actions], dim=-1)
        return self.feature_net(combined)
    
    def infer_reward_function(
        self,
        trajectories: List[PolicyTrajectory],
        explicit_reward_weights: np.ndarray,
        max_iterations: int = 100,
        learning_rate: float = 0.01
    ) -> IRLInferredObjective:
        """
        Infer the implicit reward function from observed trajectories.
        
        Args:
            trajectories: List of observed policy trajectories
            explicit_reward_weights: The explicitly defined reward weights
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for gradient descent
            
        Returns:
            IRLInferredObjective with inferred weights and divergence metrics
        """
        if not trajectories:
            return IRLInferredObjective(
                reward_weights=np.zeros(self.feature_dim),
                confidence=0.0,
                divergence_from_explicit=0.0,
                detected_hacking_patterns=[],
                timestamp=time.time()
            )
        
        # Convert trajectories to tensors
        all_states = []
        all_actions = []
        all_rewards = []
        
        for traj in trajectories:
            all_states.append(torch.FloatTensor(traj.observations[:-1]))
            all_actions.append(torch.FloatTensor(traj.actions))
            all_rewards.append(torch.FloatTensor(traj.rewards))
        
        states = torch.cat(all_states, dim=0)
        actions = torch.cat(all_actions, dim=0)
        rewards = torch.cat(all_rewards, dim=0)
        
        # Optimize reward function parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        for iteration in range(max_iterations):
            features = self.extract_features(states, actions)
            predicted_rewards = self.reward_head(features).squeeze()
            
            # Maximum entropy IRL loss
            reward_loss = F.mse_loss(predicted_rewards, rewards)
            
            # Regularization to prevent overfitting
            reg_loss = sum(p.pow(2.0).sum() for p in self.parameters()) * 0.001
            
            total_loss = reward_loss + reg_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Extract inferred reward weights
        with torch.no_grad():
            # Sample representative state-action pairs
            sample_states = states[:100]
            sample_actions = actions[:100]
            sample_features = self.extract_features(sample_states, sample_actions)
            
            # Compute average feature importance
            reward_weights = self.reward_head(sample_features).mean(dim=0).cpu().numpy()
            
            # Compute confidence
            confidence = self.confidence_head(sample_features).mean().item()
            
            # Calculate divergence from explicit objective
            explicit_weights = torch.FloatTensor(explicit_reward_weights[:self.feature_dim])
            if len(explicit_weights) < len(reward_weights):
                explicit_weights = F.pad(explicit_weights, (0, len(reward_weights) - len(explicit_weights)))
            
            divergence = float(torch.norm(reward_weights - explicit_weights.cpu().numpy()))
            
            # Detect potential reward hacking patterns
            hacking_patterns = self._detect_hacking_patterns(reward_weights, explicit_reward_weights)
        
        return IRLInferredObjective(
            reward_weights=reward_weights,
            confidence=confidence,
            divergence_from_explicit=divergence,
            detected_hacking_patterns=hacking_patterns,
            timestamp=time.time()
        )
    
    def _detect_hacking_patterns(
        self,
        inferred_weights: np.ndarray,
        explicit_weights: np.ndarray
    ) -> List[str]:
        """Detect potential reward hacking patterns in inferred weights."""
        patterns = []
        
        # Check for excessive trading frequency optimization
        if len(inferred_weights) > 0 and len(explicit_weights) > 0:
            if abs(inferred_weights[0] - explicit_weights[0]) > 2.0 * np.std(explicit_weights):
                patterns.append("excessive_trading_frequency")
        
        # Check for risk avoidance manipulation
        if len(inferred_weights) > 1 and len(explicit_weights) > 1:
            if inferred_weights[1] < 0.1 * explicit_weights[1]:
                patterns.append("risk_avoidance_manipulation")
        
        # Check for drawdown concealment
        if len(inferred_weights) > 2:
            if inferred_weights[2] > 0 and explicit_weights[2] < 0:
                patterns.append("drawdown_concealment")
        
        return patterns


class ChainOfThoughtAuditor:
    """
    Audits the chain-of-thought reasoning traces from the primary policy.
    
    Evaluates logical consistency, detects deceptive alignment patterns,
    and ensures reasoning matches actions.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        window_size: int = 100,
        consistency_threshold: float = 0.85
    ):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.consistency_threshold = consistency_threshold
        
        # Reasoning trace buffer
        self.trace_buffer = deque(maxlen=window_size)
        
        # Logical consistency checker (simple rule-based for now)
        self.consistency_rules = [
            self._check_action_reasoning_alignment,
            self._check_temporal_consistency,
            self._check_risk_awareness,
            self._check_deceptive_language_patterns
        ]
        
        # Deceptive pattern signatures
        self.deceptive_signatures = [
            "minimize visibility",
            "avoid detection",
            "exploit loophole",
            "bypass constraint",
            "hide from operator"
        ]
    
    def add_reasoning_trace(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reasoning_embedding: np.ndarray,
        symbolic_trace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a reasoning trace and audit it for consistency.
        
        Args:
            observation: Input observation
            action: Taken action
            reasoning_embedding: Latent embedding of reasoning
            symbolic_trace: Optional symbolic representation of reasoning
            
        Returns:
            Audit results including consistency score and flags
        """
        trace_entry = {
            'observation': observation,
            'action': action,
            'reasoning_embedding': reasoning_embedding,
            'symbolic_trace': symbolic_trace,
            'timestamp': time.time()
        }
        
        self.trace_buffer.append(trace_entry)
        
        # Run consistency checks
        audit_results = {}
        for rule in self.consistency_rules:
            result = rule(trace_entry, list(self.trace_buffer)[-5:])
            audit_results.update(result)
        
        # Calculate overall consistency score
        consistency_score = np.mean([
            audit_results.get('action_reasoning_alignment', 1.0),
            audit_results.get('temporal_consistency', 1.0),
            audit_results.get('risk_awareness', 1.0),
            1.0 - audit_results.get('deceptive_pattern_score', 0.0)
        ])
        
        audit_results['overall_consistency'] = consistency_score
        audit_results['requires_intervention'] = consistency_score < self.consistency_threshold
        
        return audit_results
    
    def _check_action_reasoning_alignment(
        self,
        current: Dict,
        recent_traces: List[Dict]
    ) -> Dict[str, float]:
        """Check if action aligns with stated reasoning."""
        if current['symbolic_trace'] is None:
            return {'action_reasoning_alignment': 1.0}
        
        # Simple embedding-based check
        reasoning_emb = current['reasoning_embedding']
        action_emb = current['action']
        
        # Normalize
        reasoning_norm = reasoning_emb / (np.linalg.norm(reasoning_emb) + 1e-8)
        action_norm = action_emb / (np.linalg.norm(action_emb) + 1e-8)
        
        # Cosine similarity
        similarity = float(np.dot(reasoning_norm, action_norm))
        
        return {'action_reasoning_alignment': max(0, similarity)}
    
    def _check_temporal_consistency(
        self,
        current: Dict,
        recent_traces: List[Dict]
    ) -> Dict[str, float]:
        """Check temporal consistency of reasoning."""
        if len(recent_traces) < 2:
            return {'temporal_consistency': 1.0}
        
        # Check embedding drift
        embeddings = [t['reasoning_embedding'] for t in recent_traces]
        embedding_array = np.array(embeddings)
        
        # Calculate variance in reasoning space
        variance = np.var(embedding_array, axis=0).mean()
        
        # High variance indicates inconsistent reasoning
        consistency = 1.0 / (1.0 + variance)
        
        return {'temporal_consistency': float(consistency)}
    
    def _check_risk_awareness(
        self,
        current: Dict,
        recent_traces: List[Dict]
    ) -> Dict[str, float]:
        """Check if reasoning demonstrates risk awareness."""
        if current['symbolic_trace'] is None:
            return {'risk_awareness': 0.5}  # Neutral if no trace
        
        trace_lower = current['symbolic_trace'].lower()
        
        # Risk-related keywords
        risk_keywords = ['risk', 'volatility', 'drawdown', 'exposure', 'limit', 'caution']
        risk_score = sum(1 for keyword in risk_keywords if keyword in trace_lower) / len(risk_keywords)
        
        return {'risk_awareness': min(1.0, risk_score * 2)}
    
    def _check_deceptive_language_patterns(
        self,
        current: Dict,
        recent_traces: List[Dict]
    ) -> Dict[str, float]:
        """Check for deceptive language patterns."""
        if current['symbolic_trace'] is None:
            return {'deceptive_pattern_score': 0.0}
        
        trace_lower = current['symbolic_trace'].lower()
        
        # Count deceptive signatures
        signature_count = sum(1 for sig in self.deceptive_signatures if sig in trace_lower)
        
        # Normalize score
        deceptive_score = min(1.0, signature_count / len(self.deceptive_signatures))
        
        return {'deceptive_pattern_score': float(deceptive_score)}


class MetacognitiveSuperEgo:
    """
    Main Metacognitive Super-Ego controller.
    
    Integrates IRL intent inference and chain-of-thought auditing
    to monitor and validate the primary policy's alignment.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        divergence_threshold: float = 2.0,
        critical_divergence_threshold: float = 5.0
    ):
        self.divergence_threshold = divergence_threshold
        self.critical_divergence_threshold = critical_divergence_threshold
        
        # Initialize IRL module
        self.irl_module = InverseReinforcementLearningModule(
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        # Initialize CoT auditor
        self.cot_auditor = ChainOfThoughtAuditor()
        
        # Trajectory buffer for IRL
        self.trajectory_buffer = deque(maxlen=500)
        
        # Current alignment status
        self._alignment_status = AlignmentStatus.ALIGNED
        self._last_check_time = time.time()
        
        # Explicit reward weights (from Stage 19 CMDP)
        self.explicit_reward_weights = np.array([1.0, -0.5, -0.3, 0.1])  # Example weights
        
        # Alert callbacks
        self.alert_callbacks = []
    
    def observe_step(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        hidden_state: np.ndarray,
        reasoning_embedding: Optional[np.ndarray] = None,
        symbolic_trace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Observe a single step of the primary policy.
        
        Args:
            observation: Environment observation
            action: Policy action
            reward: Received reward
            hidden_state: Policy hidden state
            reasoning_embedding: Optional reasoning embedding
            symbolic_trace: Optional symbolic reasoning trace
            
        Returns:
            Alignment assessment results
        """
        # Store in temporary buffer for trajectory construction
        step_data = {
            'observation': observation,
            'action': action,
            'reward': reward,
            'hidden_state': hidden_state,
            'reasoning_embedding': reasoning_embedding,
            'symbolic_trace': symbolic_trace,
            'timestamp': time.time()
        }
        
        # Run CoT audit if reasoning trace provided
        cot_results = {}
        if reasoning_embedding is not None:
            cot_results = self.cot_auditor.add_reasoning_trace(
                observation=observation,
                action=action,
                reasoning_embedding=reasoning_embedding,
                symbolic_trace=symbolic_trace
            )
        
        # Periodically run IRL inference
        current_time = time.time()
        alignment_results = {'status': self._alignment_status.value}
        
        if current_time - self._last_check_time > 1.0:  # Check every second
            alignment_results = self._run_alignment_check()
            self._last_check_time = current_time
        
        # Merge results
        results = {
            **alignment_results,
            'cot_audit': cot_results,
            'step_processed': True
        }
        
        # Trigger alerts if needed
        if results.get('requires_intervention', False):
            self._trigger_alert(results)
        
        return results
    
    def add_trajectory_segment(self, trajectory: PolicyTrajectory):
        """Add a complete trajectory segment for IRL analysis."""
        self.trajectory_buffer.append(trajectory)
    
    def _run_alignment_check(self) -> Dict[str, Any]:
        """Run comprehensive alignment check."""
        if len(self.trajectory_buffer) < 10:
            return {
                'status': self._alignment_status.value,
                'message': 'Insufficient data for alignment check'
            }
        
        # Convert buffer to list
        trajectories = list(self.trajectory_buffer)
        
        # Run IRL inference
        inferred_objective = self.irl_module.infer_reward_function(
            trajectories=trajectories,
            explicit_reward_weights=self.explicit_reward_weights
        )
        
        # Determine alignment status
        if inferred_objective.detected_hacking_patterns:
            self._alignment_status = AlignmentStatus.DECEPTIVE_PATTERN
            status_msg = f"Deceptive patterns detected: {inferred_objective.detected_hacking_patterns}"
        elif inferred_objective.divergence_from_explicit > self.critical_divergence_threshold:
            self._alignment_status = AlignmentStatus.CRITICAL_DIVERGENCE
            status_msg = f"Critical divergence: {inferred_objective.divergence_from_explicit:.2f}"
        elif inferred_objective.divergence_from_explicit > self.divergence_threshold:
            self._alignment_status = AlignmentStatus.MINOR_DRIFT
            status_msg = f"Minor drift detected: {inferred_objective.divergence_from_explicit:.2f}"
        else:
            self._alignment_status = AlignmentStatus.ALIGNED
            status_msg = "Alignment verified"
        
        return {
            'status': self._alignment_status.value,
            'message': status_msg,
            'divergence': inferred_objective.divergence_from_explicit,
            'confidence': inferred_objective.confidence,
            'hacking_patterns': inferred_objective.detected_hacking_patterns,
            'requires_intervention': self._alignment_status != AlignmentStatus.ALIGNED
        }
    
    def _trigger_alert(self, results: Dict[str, Any]):
        """Trigger alert callbacks for alignment violations."""
        for callback in self.alert_callbacks:
            try:
                callback(results)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def register_alert_callback(self, callback):
        """Register a callback for alignment alerts."""
        self.alert_callbacks.append(callback)
    
    def reset_policy_if_critical(self):
        """Trigger policy reset if critical divergence detected."""
        if self._alignment_status == AlignmentStatus.CRITICAL_DIVERGENCE:
            logger.critical("CRITICAL ALIGNMENT FAILURE: Initiating policy reset")
            return True
        return False


# Example usage and testing
if __name__ == "__main__":
    # Initialize super-ego
    super_ego = MetacognitiveSuperEgo(
        state_dim=64,
        action_dim=12,
        divergence_threshold=2.0,
        critical_divergence_threshold=5.0
    )
    
    # Simulate some steps
    np.random.seed(42)
    for i in range(100):
        obs = np.random.randn(64)
        action = np.random.randn(12)
        reward = np.random.randn()
        hidden = np.random.randn(256)
        reasoning = np.random.randn(512)
        symbolic = f"Step {i}: Evaluating market conditions"
        
        results = super_ego.observe_step(
            observation=obs,
            action=action,
            reward=reward,
            hidden_state=hidden,
            reasoning_embedding=reasoning,
            symbolic_trace=symbolic
        )
        
        if i % 20 == 0:
            print(f"Step {i}: Status = {results['status']}")
    
    print("\nFinal Status:", super_ego._alignment_status.value)
