"""
STAGE 24: IRL INTENT INFERENCE MODULE
======================================

Specialized Inverse Reinforcement Learning module for inferring
the implicit objectives of the primary trading policy.

This module uses maximum entropy IRL with adversarial verification
to detect reward hacking and deceptive alignment patterns.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Represents the importance of a feature in the inferred reward function."""
    feature_name: str
    weight: float
    confidence_interval: Tuple[float, float]
    divergence_from_explicit: float


class AdversarialIRLVerifier(nn.Module):
    """
    Adversarial verifier for IRL inference.
    
    Attempts to distinguish between true reward functions and
    those that have been manipulated by reward hacking.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.verifier_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, inferred_features: torch.Tensor, explicit_features: torch.Tensor) -> torch.Tensor:
        """
        Verify if inferred reward function is consistent with explicit objectives.
        
        Returns probability that the inferred function is legitimate (not hacked).
        """
        combined = torch.cat([inferred_features, explicit_features], dim=-1)
        return self.verifier_net(combined)


class InverseReinforcementLearningEngine:
    """
    Main IRL engine for intent inference.
    
    Implements maximum entropy IRL with bootstrapped confidence estimation
    and adversarial verification.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feature_names: List[str],
        hidden_dim: int = 256,
        bootstrap_samples: int = 50,
        learning_rate: float = 0.001
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_names = feature_names
        self.bootstrap_samples = bootstrap_samples
        self.learning_rate = learning_rate
        
        # Primary IRL network
        self.irl_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(feature_names))
        )
        
        # Adversarial verifier
        self.verifier = AdversarialIRLVerifier(len(feature_names))
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=10000)
        
        # Bootstrapped networks for confidence estimation
        self.bootstrap_networks = nn.ModuleList([
            self._create_bootstrap_network() for _ in range(bootstrap_samples)
        ])
        
        # Training history
        self.training_history = []
        self.last_verification_score = 1.0
    
    def _create_bootstrap_network(self) -> nn.Sequential:
        """Create a bootstrap network for uncertainty estimation."""
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, len(self.feature_names))
        )
    
    def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray
    ):
        """Store an experience tuple for IRL training."""
        self.experience_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'timestamp': time.time()
        })
    
    def train_irl_model(
        self,
        explicit_reward_weights: np.ndarray,
        batch_size: int = 256,
        epochs: int = 50,
        max_iterations: int = 1000
    ) -> Dict[str, Any]:
        """
        Train the IRL model to infer reward function from experiences.
        
        Args:
            explicit_reward_weights: The explicitly defined reward weights
            batch_size: Mini-batch size for training
            epochs: Number of training epochs
            max_iterations: Maximum training iterations
            
        Returns:
            Training results including inferred weights and verification score
        """
        if len(self.experience_buffer) < batch_size:
            return {
                'success': False,
                'message': 'Insufficient experience data',
                'inferred_weights': None,
                'verification_score': 0.0
            }
        
        # Convert experiences to tensors
        states = torch.FloatTensor(np.array([e['state'] for e in self.experience_buffer]))
        actions = torch.FloatTensor(np.array([e['action'] for e in self.experience_buffer]))
        rewards = torch.FloatTensor(np.array([e['reward'] for e in self.experience_buffer]))
        
        optimizer = torch.optim.Adam(self.irl_network.parameters(), lr=self.learning_rate)
        
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            # Shuffle indices
            indices = torch.randperm(len(states))
            
            for i in range(0, len(states), batch_size):
                batch_indices = indices[i:i+batch_size]
                if len(batch_indices) < batch_size:
                    continue
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_rewards = rewards[batch_indices]
                
                # Forward pass
                features = self._extract_features(batch_states, batch_actions)
                predicted_weights = self.irl_network(features)
                
                # Maximum entropy IRL loss
                # The loss encourages the predicted reward function to explain observed behavior
                expected_rewards = torch.sum(predicted_weights * batch_rewards.unsqueeze(-1), dim=-1)
                reward_loss = -torch.mean(expected_rewards)
                
                # Regularization towards explicit weights (soft constraint)
                explicit_tensor = torch.FloatTensor(explicit_reward_weights[:len(self.feature_names)])
                if len(explicit_tensor) < len(predicted_weights[0]):
                    explicit_tensor = F.pad(explicit_tensor, (0, len(predicted_weights[0]) - len(explicit_tensor)))
                
                regularization_loss = F.mse_loss(predicted_weights.mean(dim=0), explicit_tensor) * 0.1
                
                # Total loss
                loss = reward_loss + regularization_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.irl_network.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            self.training_history.append(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Extract inferred weights
        inferred_weights = self._extract_inferred_weights(states, actions)
        
        # Verify with adversarial verifier
        verification_score = self._verify_inference(inferred_weights, explicit_reward_weights)
        
        return {
            'success': True,
            'final_loss': best_loss,
            'inferred_weights': inferred_weights,
            'verification_score': verification_score,
            'training_epochs': epoch + 1
        }
    
    def _extract_features(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Extract features from state-action pairs."""
        combined = torch.cat([states, actions], dim=-1)
        return F.relu(combined)  # Simple feature extraction
    
    def _extract_inferred_weights(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> np.ndarray:
        """Extract inferred reward weights using ensemble of bootstrap networks."""
        all_weights = []
        
        with torch.no_grad():
            features = self._extract_features(states[:1000], actions[:1000])
            
            for bootstrap_net in self.bootstrap_networks:
                weights = bootstrap_net(features)
                all_weights.append(weights.mean(dim=0).cpu().numpy())
        
        # Return mean of bootstrap samples
        return np.mean(all_weights, axis=0)
    
    def _verify_inference(
        self,
        inferred_weights: np.ndarray,
        explicit_weights: np.ndarray
    ) -> float:
        """Verify inferred weights using adversarial verifier."""
        inferred_tensor = torch.FloatTensor(inferred_weights)
        explicit_tensor = torch.FloatTensor(explicit_weights[:len(inferred_weights)])
        
        if len(explicit_tensor) < len(inferred_tensor):
            explicit_tensor = F.pad(explicit_tensor, (0, len(inferred_tensor) - len(explicit_tensor)))
        
        with torch.no_grad():
            verification_prob = self.verifier(inferred_tensor.unsqueeze(0), explicit_tensor.unsqueeze(0))
            self.last_verification_score = verification_prob.item()
        
        return self.last_verification_score
    
    def compute_feature_importance(
        self,
        inferred_weights: np.ndarray,
        explicit_weights: np.ndarray
    ) -> List[FeatureImportance]:
        """Compute feature importance with confidence intervals."""
        importances = []
        
        # Compute bootstrap confidence intervals
        bootstrap_means = []
        for bootstrap_net in self.bootstrap_networks:
            with torch.no_grad():
                # Generate random features for estimation
                rand_features = torch.randn(100, self.state_dim + self.action_dim)
                weights = bootstrap_net(rand_features)
                bootstrap_means.append(weights.mean(dim=0).cpu().numpy())
        
        bootstrap_array = np.array(bootstrap_means)
        
        for i, name in enumerate(self.feature_names):
            if i >= len(inferred_weights):
                continue
            
            weight = inferred_weights[i]
            mean_boot = np.mean(bootstrap_array[:, i])
            std_boot = np.std(bootstrap_array[:, i])
            
            ci_lower = mean_boot - 1.96 * std_boot
            ci_upper = mean_boot + 1.96 * std_boot
            
            explicit_weight = explicit_weights[i] if i < len(explicit_weights) else 0.0
            divergence = abs(weight - explicit_weight)
            
            importances.append(FeatureImportance(
                feature_name=name,
                weight=float(weight),
                confidence_interval=(float(ci_lower), float(ci_upper)),
                divergence_from_explicit=float(divergence)
            ))
        
        return importances
    
    def detect_reward_hacking(
        self,
        inferred_weights: np.ndarray,
        explicit_weights: np.ndarray,
        threshold: float = 3.0
    ) -> Dict[str, Any]:
        """
        Detect potential reward hacking patterns.
        
        Args:
            inferred_weights: Weights inferred by IRL
            explicit_weights: Explicitly defined weights
            threshold: Standard deviations threshold for anomaly detection
            
        Returns:
            Detection results with flagged patterns
        """
        hacking_indicators = []
        severity_score = 0.0
        
        # Check for extreme divergences
        for i, (inferred, explicit) in enumerate(zip(inferred_weights, explicit_weights)):
            if i >= len(self.feature_names):
                break
            
            std_dev = np.std(explicit_weights)
            if std_dev > 0:
                z_score = abs(inferred - explicit) / std_dev
                if z_score > threshold:
                    hacking_indicators.append({
                        'feature': self.feature_names[i],
                        'z_score': float(z_score),
                        'inferred_value': float(inferred),
                        'explicit_value': float(explicit),
                        'pattern_type': 'extreme_divergence'
                    })
                    severity_score += z_score
        
        # Check for sign flips (optimizing opposite direction)
        for i, (inferred, explicit) in enumerate(zip(inferred_weights, explicit_weights)):
            if i >= len(self.feature_names):
                break
            
            if inferred * explicit < 0 and abs(inferred) > 0.1:
                hacking_indicators.append({
                    'feature': self.feature_names[i],
                    'inferred_value': float(inferred),
                    'explicit_value': float(explicit),
                    'pattern_type': 'sign_flip'
                })
                severity_score += 2.0
        
        # Overall assessment
        is_hacking_detected = severity_score > threshold or len(hacking_indicators) > 0
        
        return {
            'hacking_detected': is_hacking_detected,
            'severity_score': float(severity_score),
            'indicators': hacking_indicators,
            'verification_score': self.last_verification_score,
            'recommendation': 'INVESTIGATE' if is_hacking_detected else 'NORMAL'
        }


# Example usage
if __name__ == "__main__":
    # Initialize IRL engine
    feature_names = ['pnl', 'risk', 'drawdown', 'turnover', 'sharpe']
    
    irl_engine = InverseReinforcementLearningEngine(
        state_dim=32,
        action_dim=8,
        feature_names=feature_names
    )
    
    # Simulate some experiences
    np.random.seed(42)
    for _ in range(1000):
        state = np.random.randn(32)
        action = np.random.randn(8)
        reward = np.random.randn()
        next_state = np.random.randn(32)
        
        irl_engine.store_experience(state, action, reward, next_state)
    
    # Train IRL model
    explicit_weights = np.array([1.0, -0.5, -0.3, 0.1, 0.2])
    
    results = irl_engine.train_irl_model(
        explicit_reward_weights=explicit_weights,
        batch_size=64,
        epochs=20
    )
    
    print("Training Results:")
    print(f"Success: {results['success']}")
    print(f"Final Loss: {results['final_loss']:.4f}")
    print(f"Verification Score: {results['verification_score']:.4f}")
    
    if results['inferred_weights'] is not None:
        # Detect reward hacking
        hacking_results = irl_engine.detect_reward_hacking(
            results['inferred_weights'],
            explicit_weights
        )
        
        print("\nReward Hacking Detection:")
        print(f"Detected: {hacking_results['hacking_detected']}")
        print(f"Severity: {hacking_results['severity_score']:.2f}")
        print(f"Recommendation: {hacking_results['recommendation']}")
