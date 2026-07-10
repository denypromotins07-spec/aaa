"""
NEXUS-OMEGA Stage 19: Quantile Huber Regression with Crossing Prevention

This module implements Huberized quantile regression loss with explicit
prevention of quantile crossing - a common pathology where predicted
quantiles violate monotonicity (e.g., 90th percentile < 10th percentile).

Prevention techniques:
1. Monotonic network architecture constraints
2. Explicit sorting penalty in loss function
3. Isotonic regression post-processing

Author: NEXUS-OMEGA Architecture
Stage: 19 of 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class MonotonicQuantileNetwork(nn.Module):
    """
    Neural network architecture that enforces monotonic quantile predictions.
    
    Uses cumulative output formulation:
        q_τ = base + Σ_{i=1}^k exp(δ_i) * 1{τ ≥ τ_i}
    
    This guarantees q_τ1 ≤ q_τ2 for τ1 < τ2.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_quantiles: int = 32,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        
        self.n_quantiles = n_quantiles
        self.action_dim = action_dim
        
        # Fixed quantile levels (sorted)
        self.register_buffer(
            'tau_levels',
            torch.linspace(0, 1, n_quantiles + 2)[1:-1]  # Exclude 0 and 1
        )
        
        # Shared state encoder
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Base prediction (minimum quantile)
        self.base_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # Incremental predictions (guaranteed positive via softplus)
        self.delta_head = nn.Linear(hidden_dims[-1], n_quantiles * action_dim)
        
        # Initialize delta weights small for stability
        nn.init.zeros_(self.delta_head.weight)
        nn.init.constant_(self.delta_head.bias, 0.01)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing monotonically increasing quantiles.
        
        Args:
            states: Shape (batch, state_dim)
        
        Returns:
            quantiles: Shape (batch, n_quantiles, action_dim)
        """
        features = self.encoder(states)
        
        # Base value (lowest quantile)
        base = self.base_head(features)  # (batch, action_dim)
        
        # Positive increments
        deltas_raw = self.delta_head(features)  # (batch, n_quantiles * action_dim)
        deltas = F.softplus(deltas_raw).view(-1, self.n_quantiles, self.action_dim)
        
        # Cumulative sum to get quantiles
        quantiles = base.unsqueeze(1) + torch.cumsum(deltas, dim=1)
        
        return quantiles
    
    def get_quantile_levels(self) -> torch.Tensor:
        """Get the fixed quantile levels."""
        return self.tau_levels


def quantile_crossing_loss(
    quantiles: torch.Tensor,
    penalty_weight: float = 1.0,
    margin: float = 1e-4,
) -> Tuple[torch.Tensor, int]:
    """
    Penalty loss for quantile crossing violations.
    
    Encourages strict monotonicity: q_{i+1} > q_i + margin
    
    Args:
        quantiles: Shape (batch, n_quantiles, action_dim)
        penalty_weight: Weight for crossing penalty
        margin: Minimum gap between adjacent quantiles
    
    Returns:
        crossing_loss: Scalar penalty
        n_violations: Number of crossing violations
    """
    if quantiles.dim() < 3:
        return torch.tensor(0.0, device=quantiles.device), 0
    
    # Compute differences between adjacent quantiles
    diffs = quantiles[:, 1:, :] - quantiles[:, :-1, :]  # (batch, n_quantiles-1, action_dim)
    
    # Violations where diff < margin (crossing or too close)
    violations = F.relu(margin - diffs)
    
    n_violations = (violations > 0).sum().item()
    
    # Squared penalty for smooth gradients
    crossing_loss = (violations ** 2).mean() * penalty_weight
    
    return crossing_loss, n_violations


class IsotonicCorrection:
    """
    Post-processing correction using isotonic regression.
    
    Projects predicted quantiles onto the monotonic cone,
    guaranteeing no crossing while minimizing L2 deviation.
    
    Uses Pool Adjacent Violators Algorithm (PAVA).
    """
    
    @staticmethod
    def correct(quantiles: torch.Tensor) -> torch.Tensor:
        """
        Apply isotonic correction to ensure monotonicity.
        
        Args:
            quantiles: Shape (batch, n_quantiles, action_dim)
        
        Returns:
            Corrected quantiles with guaranteed monotonicity
        """
        if quantiles.dim() < 3:
            return quantiles
        
        batch_size, n_quantiles, action_dim = quantiles.shape
        
        corrected = torch.zeros_like(quantiles)
        
        for b in range(batch_size):
            for a in range(action_dim):
                # Extract quantile sequence for this sample/action
                q = quantiles[b, :, a].cpu().numpy()
                
                # Apply PAVA
                corrected_q = IsotonicCorrection._pava(q)
                
                corrected[b, :, a] = torch.from_numpy(corrected_q).to(quantiles.device)
        
        return corrected
    
    @staticmethod
    def _pava(values: np.ndarray) -> np.ndarray:
        """
        Pool Adjacent Violators Algorithm for isotonic regression.
        
        Finds closest non-decreasing sequence in L2 sense.
        """
        n = len(values)
        result = values.copy().astype(float)
        
        # Track block information
        block_sum = result.copy()
        block_count = np.ones(n)
        
        i = 0
        while i < n - 1:
            if result[i] > result[i + 1]:
                # Violation found - merge blocks
                j = i + 1
                while j < n and result[i] > result[j]:
                    j += 1
                
                # Average the violating block
                block_len = j - i
                avg = result[i:j].mean()
                result[i:j] = avg
                
                # Go back to check previous elements
                if i > 0:
                    i -= 1
            else:
                i += 1
        
        return result


class HuberQuantileLoss(nn.Module):
    """
    Huberized quantile regression loss with optional crossing prevention.
    
    Combines:
    1. Standard quantile regression (pinball) loss
    2. Huber smoothing for robustness
    3. Optional crossing penalty
    """
    
    def __init__(
        self,
        huber_delta: float = 1.0,
        crossing_penalty_weight: float = 0.1,
        enable_crossing_prevention: bool = True,
    ):
        super().__init__()
        
        self.huber_delta = huber_delta
        self.crossing_penalty_weight = crossing_penalty_weight
        self.enable_crossing_prevention = enable_crossing_prevention
    
    def forward(
        self,
        predicted_quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        taus: Optional[torch.Tensor] = None,
        apply_correction: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Huber quantile loss with crossing prevention.
        
        Args:
            predicted_quantiles: Shape (batch, n_taus, action_dim)
            target_quantiles: Shape (batch, n_targets, action_dim)
            taus: Quantile levels for predicted
            apply_correction: Whether to apply isotonic correction
        
        Returns:
            total_loss: Combined loss
            loss_info: Dictionary with loss components
        """
        loss_info = {}
        
        # Optionally correct predictions first
        if apply_correction and self.enable_crossing_prevention:
            predicted_quantiles = IsotonicCorrection.correct(predicted_quantiles)
        
        # 1. Huber quantile regression loss
        qr_loss = self._huber_quantile_loss(
            predicted_quantiles,
            target_quantiles,
            taus,
        )
        loss_info['qr_loss'] = qr_loss.item()
        
        # 2. Crossing penalty
        crossing_loss = torch.tensor(0.0, device=predicted_quantiles.device)
        n_violations = 0
        
        if self.enable_crossing_prevention and self.crossing_penalty_weight > 0:
            crossing_loss, n_violations = quantile_crossing_loss(
                predicted_quantiles,
                penalty_weight=self.crossing_penalty_weight,
            )
            loss_info['crossing_loss'] = crossing_loss.item()
            loss_info['n_crossing_violations'] = n_violations
        
        # Total loss
        total_loss = qr_loss + crossing_loss
        
        loss_info['total_loss'] = total_loss.item()
        
        return total_loss, loss_info
    
    def _huber_quantile_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        taus: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Huber-smoothed quantile regression loss.
        
        For δ = target - predicted:
            L = |τ - 1{δ < 0}| * huber(δ)
        """
        batch_size = predicted.shape[0]
        n_pred = predicted.shape[1]
        n_target = target.shape[1]
        
        # Pairwise differences: δ_ij = target_j - predicted_i
        delta = target.unsqueeze(1) - predicted.unsqueeze(2)
        # Shape: (batch, n_pred, n_target, action_dim)
        
        # Huber loss
        abs_delta = delta.abs()
        huber_loss = torch.where(
            abs_delta <= self.huber_delta,
            0.5 * delta.pow(2),
            self.huber_delta * (abs_delta - 0.5 * self.huber_delta)
        )
        
        # Quantile weighting
        if taus is not None:
            taus_expanded = taus.unsqueeze(-1).unsqueeze(-1)
            indicator = (delta < 0).float()
            weights = (taus_expanded - indicator).abs()
            huber_loss = weights * huber_loss
        
        return huber_loss.mean()


class DistributionalMetrics:
    """
    Compute quality metrics for distributional predictions.
    """
    
    @staticmethod
    def check_monotonicity(quantiles: torch.Tensor) -> Tuple[bool, float]:
        """
        Check if quantiles are monotonically increasing.
        
        Returns:
            is_monotonic: Whether all sequences are monotonic
            violation_rate: Fraction of adjacent pairs that violate monotonicity
        """
        if quantiles.dim() < 3:
            return True, 0.0
        
        diffs = quantiles[:, 1:, :] - quantiles[:, :-1, :]
        violations = (diffs < 0).sum().item()
        total_pairs = diffs.numel()
        
        is_monotonic = violations == 0
        violation_rate = violations / max(total_pairs, 1)
        
        return is_monotonic, violation_rate
    
    @staticmethod
    def compute_spread(quantiles: torch.Tensor) -> torch.Tensor:
        """
        Compute inter-quantile spread (uncertainty measure).
        
        Returns:
            spread: Difference between 90th and 10th percentile predictions
        """
        sorted_q, _ = torch.sort(quantiles, dim=1)
        n_q = sorted_q.shape[1]
        
        q90_idx = max(0, int(0.9 * n_q) - 1)
        q10_idx = min(n_q - 1, int(0.1 * n_q))
        
        spread = sorted_q[:, q90_idx, :] - sorted_q[:, q10_idx, :]
        return spread.mean()
    
    @staticmethod
    def calibration_error(
        predicted_quantiles: torch.Tensor,
        realized_returns: torch.Tensor,
        tau_levels: torch.Tensor,
    ) -> float:
        """
        Compute calibration error: how well predicted quantiles match empirical frequencies.
        
        For well-calibrated predictions:
            P(return ≤ q_τ) = τ
        """
        batch_size, n_quantiles, _ = predicted_quantiles.shape
        
        errors = []
        for i, tau in enumerate(tau_levels[:n_quantiles]):
            q_tau = predicted_quantiles[:, i, :].mean(dim=-1)
            
            # Empirical frequency
            empirical_freq = (realized_returns <= q_tau).float().mean().item()
            
            errors.append(abs(empirical_freq - tau.item()))
        
        return np.mean(errors)
