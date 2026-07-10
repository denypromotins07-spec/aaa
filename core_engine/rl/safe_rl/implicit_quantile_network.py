"""
NEXUS-OMEGA Stage 19: Implicit Quantile Networks for Distributional RL

This module implements Implicit Quantile Networks (IQN) for Distributional RL.
Instead of predicting a single scalar value, IQN predicts the entire cumulative
distribution function (CDF) of future returns, enabling explicit optimization
of risk measures like CVaR.

Key innovations over standard QR-DQN:
- Implicit quantile representation via neural network
- Arbitrary quantile level sampling during training
- Better sample efficiency through shared representation

Author: NEXUS-OMEGA Architecture
Stage: 19 of 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class FourierEmbedding(nn.Module):
    """
    Fourier embedding for quantile levels.
    
    Maps scalar quantile levels τ ∈ [0, 1] to high-dimensional
    feature space using random Fourier features. This enables
    the network to learn complex quantile-dependent representations.
    """
    
    def __init__(self, input_dim: int = 1, output_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Random Fourier features matrix (fixed, not learned)
        self.register_buffer(
            'B', 
            torch.randn(input_dim, output_dim // 2)
        )
        
        # Learnable output projection
        self.projection = nn.Linear(output_dim, output_dim)
    
    def forward(self, taus: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier embedding of quantile levels.
        
        Args:
            taus: Quantile levels, shape (batch_size, n_taus) or (batch_size,)
        
        Returns:
            Embedded quantiles, shape (batch_size, n_taus, output_dim)
        """
        # Ensure taus has proper shape
        if taus.dim() == 1:
            taus = taus.unsqueeze(-1)  # (batch, 1)
        
        # Add dimension for quantiles if needed
        if taus.dim() == 2:
            taus = taus.unsqueeze(-1)  # (batch, 1, 1)
        
        # Compute random Fourier features
        # cos(2π * B * τ)
        embeddings = torch.cos(2 * np.pi * taus @ self.B)  # (batch, n_taus, output_dim//2)
        
        # Concatenate with sin component
        sin_embeddings = torch.sin(2 * np.pi * taus @ self.B)
        embeddings = torch.cat([embeddings, sin_embeddings], dim=-1)
        
        # Apply ReLU and projection
        embeddings = F.relu(embeddings)
        embeddings = self.projection(embeddings)
        
        return embeddings


class ImplicitQuantileNetwork(nn.Module):
    """
    Implicit Quantile Network (IQN) for distributional RL.
    
    The network takes state s and quantile level τ as input,
    and outputs the τ-quantile of the return distribution Z(s).
    
    Q_θ(s, τ) ≈ F_Z^{-1}(τ | s)
    
    where F_Z is the CDF of the return distribution.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        fourier_dim: int = 128,
        dueling: bool = True,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling
        
        # State encoder (shared across quantiles)
        state_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:
            state_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ])
            prev_dim = hidden_dim
        
        self.state_encoder = nn.Sequential(*state_layers)
        self.state_output_dim = hidden_dims[-1]
        
        # Fourier embedding for quantiles
        self.quantile_embedding = FourierEmbedding(output_dim=fourier_dim)
        
        # Combine state and quantile features
        combined_dim = hidden_dims[-1] + fourier_dim
        
        # Joint processing layers
        joint_layers = []
        prev_dim = combined_dim
        for hidden_dim in hidden_dims[-1:]:
            joint_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ])
            prev_dim = hidden_dim
        
        self.joint_network = nn.Sequential(*joint_layers)
        
        if dueling:
            # Dueling architecture: separate value and advantage streams
            self.value_stream = nn.Linear(hidden_dims[-1], 1)
            self.advantage_stream = nn.Linear(hidden_dims[-1], action_dim)
            
            # Initialize advantage stream to zero for stability
            nn.init.zeros_(self.advantage_stream.weight)
            nn.init.zeros_(self.advantage_stream.bias)
        else:
            # Standard architecture
            self.q_head = nn.Linear(hidden_dims[-1], action_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Orthogonal initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        states: torch.Tensor,
        taus: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through IQN.
        
        Args:
            states: State batch, shape (batch_size, state_dim)
            taus: Quantile levels, shape (batch_size, n_taus)
        
        Returns:
            quantile_values: Q-values for each quantile, shape (batch, n_taus, action_dim)
            quantile_embeddings: Embedded quantile features, shape (batch, n_taus, embed_dim)
        """
        batch_size = states.shape[0]
        n_taus = taus.shape[1] if taus.dim() > 1 else 1
        
        # Encode states (shared representation)
        state_features = self.state_encoder(states)  # (batch, state_feat_dim)
        
        # Embed quantiles
        quantile_feats = self.quantile_embedding(taus)  # (batch, n_taus, fourier_dim)
        
        # Expand state features for each quantile
        state_expanded = state_features.unsqueeze(1).expand(
            -1, n_taus, -1
        )  # (batch, n_taus, state_feat_dim)
        
        # Concatenate state and quantile features
        combined = torch.cat([state_expanded, quantile_feats], dim=-1)
        
        # Joint processing
        joint_features = self.joint_network(combined)
        
        if self.dueling:
            # Dueling decomposition
            values = self.value_stream(joint_features)  # (batch, n_taus, 1)
            advantages = self.advantage_stream(joint_features)  # (batch, n_taus, action_dim)
            
            # Combine: Q = V + (A - mean(A))
            quantile_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        else:
            quantile_values = self.q_head(joint_features)
        
        return quantile_values, quantile_feats
    
    def get_quantile_samples(
        self,
        states: torch.Tensor,
        n_samples: int = 32,
    ) -> torch.Tensor:
        """
        Sample quantiles from the return distribution.
        
        Args:
            states: State batch
            n_samples: Number of quantile samples
        
        Returns:
            Sampled quantile values, shape (batch, n_samples, action_dim)
        """
        # Sample random quantile levels uniformly
        taus = torch.rand(states.shape[0], n_samples, device=states.device)
        
        quantile_values, _ = self.forward(states, taus)
        return quantile_values
    
    def get_distribution_statistics(
        self,
        states: torch.Tensor,
        n_quantiles: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute statistics of the predicted return distribution.
        
        Returns:
            Dictionary with mean, std, skewness, kurtosis, VaR, CVaR
        """
        # Get fine-grained quantile predictions
        taus = torch.linspace(0.01, 0.99, n_quantiles, device=states.device)
        taus = taus.unsqueeze(0).expand(states.shape[0], -1)
        
        quantile_values, _ = self.forward(states, taus)  # (batch, n_quantiles, action_dim)
        
        # Compute statistics
        stats = {}
        
        # Mean (expectation)
        stats['mean'] = quantile_values.mean(dim=1)  # (batch, action_dim)
        
        # Standard deviation
        stats['std'] = quantile_values.std(dim=1)
        
        # Skewness (asymmetry)
        mean = stats['mean'].unsqueeze(1)
        std = stats['std'].unsqueeze(1) + 1e-8
        stats['skewness'] = ((quantile_values - mean) / std).pow(3).mean(dim=1)
        
        # Kurtosis (tail heaviness)
        stats['kurtosis'] = ((quantile_values - mean) / std).pow(4).mean(dim=1) - 3
        
        # Value at Risk (VaR) at different levels
        stats['var_95'] = torch.quantile(quantile_values, 0.05, dim=1)  # Lower tail
        stats['var_99'] = torch.quantile(quantile_values, 0.01, dim=1)
        
        # Conditional VaR (Expected Shortfall)
        sorted_values = torch.sort(quantile_values, dim=1).values
        cvar_indices = max(1, int(0.05 * n_quantiles))
        stats['cvar_95'] = sorted_values[:, :cvar_indices, :].mean(dim=1)
        
        return stats


class Distributional ReplayBuffer:
    """
    Replay buffer optimized for distributional RL training.
    
    Stores full transition tuples and supports efficient
    quantile sampling for IQN training.
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        device: str = 'cpu',
    ):
        self.capacity = capacity
        self.device = device
        
        # Pre-allocate storage
        self.states = torch.zeros(capacity, 0, dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, 0, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.next_states = torch.zeros(capacity, 0, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        
        self.size = 0
        self.position = 0
    
    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        if self.size < self.capacity:
            self.size += 1
        
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(
        self,
        batch_size: int,
        n_taus: int = 32,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a batch with random quantile levels.
        
        Returns:
            Dictionary with states, actions, rewards, next_states, dones, taus
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        # Sample random quantile levels for each transition
        taus = torch.rand(batch_size, n_taus, device=self.device)
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices],
            'taus': taus,
        }
    
    def __len__(self) -> int:
        return self.size


def quantile_regression_loss(
    predicted_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    taus: torch.Tensor,
) -> torch.Tensor:
    """
    Quantile regression loss for distributional RL.
    
    L = E[|τ - 1{δ < 0}| * δ]
    
    where δ = target - predicted
    
    This is the check function (pinball loss) generalized to quantiles.
    
    Args:
        predicted_quantiles: Shape (batch, n_taus_pred, action_dim)
        target_quantiles: Shape (batch, n_taus_target, action_dim)
        taus: Quantile levels for predictions, shape (batch, n_taus_pred)
    
    Returns:
        Scalar loss value
    """
    batch_size = predicted_quantiles.shape[0]
    n_taus_pred = predicted_quantiles.shape[1]
    n_taus_target = target_quantiles.shape[1]
    
    # Compute pairwise differences (huberized version available in quantile_huber_regression.py)
    # δ_ij = target_j - predicted_i
    delta = target_quantiles.unsqueeze(1) - predicted_quantiles.unsqueeze(2)
    # Shape: (batch, n_taus_pred, n_taus_target, action_dim)
    
    # Quantile weights: |τ_i - 1{δ_ij < 0}|
    indicator = (delta < 0).float()
    taus_expanded = taus.unsqueeze(-1).unsqueeze(-1)
    weights = (taus_expanded - indicator).abs()
    
    # Weighted loss
    loss = weights * delta.abs()
    
    # Average over all dimensions
    return loss.mean()
