"""
NEXUS-OMEGA Stage 19: CVaR Optimization Objective for Distributional RL

This module implements Conditional Value at Risk (CVaR) optimization
for distributional reinforcement learning. Instead of optimizing expected
return, the agent explicitly optimizes the worst-case α-percentile of
the return distribution, forcing risk-averse behavior.

CVaR_α(Z) = E[Z | Z ≤ VaR_α(Z)]

Where VaR_α is the α-quantile of the return distribution Z.

Key features:
- Differentiable CVaR loss for gradient-based optimization
- Support for multiple risk levels (α ∈ [0.01, 0.5])
- Integration with Implicit Quantile Networks

Author: NEXUS-OMEGA Architecture
Stage: 19 of 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class CVaROptimizer(nn.Module):
    """
    CVaR optimization layer for distributional RL.
    
    This module computes CVaR from quantile predictions and provides
    gradients for end-to-end training.
    
    For a return distribution Z with quantiles {q_i} at levels {τ_i}:
        CVaR_α(Z) ≈ (1/α) * Σ_{τ_i ≤ α} (τ_{i+1} - τ_i) * q_i
    """
    
    def __init__(self, risk_level: float = 0.05):
        """
        Initialize CVaR optimizer.
        
        Args:
            risk_level: α parameter for CVaR (e.g., 0.05 for 5% tail)
        """
        super().__init__()
        self.risk_level = risk_level
        
        if not 0 < risk_level <= 0.5:
            raise ValueError(f"risk_level must be in (0, 0.5], got {risk_level}")
    
    def forward(
        self,
        quantile_values: torch.Tensor,
        quantile_levels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute CVaR from quantile predictions.
        
        Args:
            quantile_values: Shape (batch, n_quantiles, action_dim)
            quantile_levels: Optional τ values, shape (batch, n_quantiles)
                           If None, assumes uniform spacing
        
        Returns:
            CVaR estimates, shape (batch, action_dim)
        """
        batch_size, n_quantiles, _ = quantile_values.shape
        
        if quantile_levels is None:
            # Assume uniform spacing
            quantile_levels = torch.linspace(
                1/(2*n_quantiles), 
                1 - 1/(2*n_quantiles),
                n_quantiles,
                device=quantile_values.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Sort quantiles by value to get proper ordering
        sorted_values, sort_indices = torch.sort(quantile_values, dim=1)
        
        # Get corresponding sorted quantile levels
        sorted_levels = torch.gather(
            quantile_levels.unsqueeze(-1).expand(-1, -1, quantile_values.shape[-1]),
            dim=1,
            index=sort_indices
        )[:, :, 0]  # Take first action dimension for level sorting
        
        # Find indices where τ ≤ α (tail region)
        tail_mask = sorted_levels <= self.risk_level
        
        # Compute CVaR as weighted average of tail quantiles
        # Weights are proportional to quantile spacing
        if n_quantiles > 1:
            # Compute spacing between quantile levels
            half_spacing = (sorted_levels[1:] - sorted_levels[:-1]) / 2
            weights = torch.zeros_like(sorted_levels)
            weights[:, 0] = half_spacing[:, 0]
            weights[:, -1] = half_spacing[:, -1]
            weights[:, 1:-1] = (half_spacing[:, :-1] + half_spacing[:, 1:]) / 2
        else:
            weights = torch.ones_like(sorted_levels) / n_quantiles
        
        # Apply tail mask
        tail_weights = weights * tail_mask.float()
        
        # Normalize weights to sum to 1/α
        weight_sum = tail_weights.sum(dim=1, keepdim=True) + 1e-8
        normalized_weights = tail_weights / weight_sum
        
        # Compute CVaR
        cvar = (normalized_weights.unsqueeze(-1) * sorted_values).sum(dim=1)
        
        return cvar
    
    def compute_cvar_loss(
        self,
        quantile_values: torch.Tensor,
        target_returns: torch.Tensor,
        quantile_levels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute CVaR optimization loss.
        
        The loss encourages high CVaR (risk-adjusted returns).
        
        Args:
            quantile_values: Predicted quantiles, shape (batch, n_quantiles, action_dim)
            target_returns: Target returns for taken actions, shape (batch,)
            quantile_levels: Optional quantile levels
        
        Returns:
            Scalar loss (negative CVaR, to be minimized)
        """
        cvar = self.forward(quantile_values, quantile_levels)
        
        # For the taken actions, extract CVaR
        # Assuming single action dimension or mean over actions
        if cvar.dim() > 1:
            cvar = cvar.mean(dim=-1)
        
        # Negative CVaR (we want to maximize CVaR, so minimize negative)
        return -cvar.mean()


class RiskSensitiveLoss(nn.Module):
    """
    Combined loss function balancing expected return and tail risk.
    
    L = -(1-β) * E[Z] + β * CVaR_α(Z)
    
    Where β controls risk sensitivity:
    - β = 0: Pure expected return optimization (risk-neutral)
    - β = 1: Pure CVaR optimization (extremely risk-averse)
    - β = 0.5: Balanced risk-return tradeoff
    """
    
    def __init__(
        self,
        risk_level: float = 0.05,
        risk_weight: float = 0.3,
        huber_delta: float = 1.0,
    ):
        """
        Initialize risk-sensitive loss.
        
        Args:
            risk_level: α for CVaR computation
            risk_weight: β controlling risk sensitivity
            huber_delta: Delta parameter for Huber loss on quantiles
        """
        super().__init__()
        self.risk_level = risk_level
        self.risk_weight = risk_weight
        self.huber_delta = huber_delta
        
        self.cvar_optimizer = CVaROptimizer(risk_level)
    
    def forward(
        self,
        predicted_quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        predicted_means: Optional[torch.Tensor] = None,
        target_means: Optional[torch.Tensor] = None,
        quantile_levels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined risk-sensitive loss.
        
        Args:
            predicted_quantiles: Shape (batch, n_taus, action_dim)
            target_quantiles: Shape (batch, n_targets, action_dim)
            predicted_means: Optional mean predictions
            target_means: Optional mean targets
            quantile_levels: Quantile levels for predictions
        
        Returns:
            total_loss: Combined loss scalar
            loss_info: Dictionary with individual loss components
        """
        loss_info = {}
        
        # 1. Quantile regression loss (distributional matching)
        qr_loss = self._quantile_huber_loss(
            predicted_quantiles,
            target_quantiles,
            quantile_levels,
        )
        loss_info['qr_loss'] = qr_loss.item()
        
        # 2. CVaR loss (tail risk optimization)
        cvar_loss = self.cvar_optimizer.compute_cvar_loss(
            predicted_quantiles,
            target_quantiles.mean(dim=1) if target_quantiles.dim() > 2 else target_quantiles,
            quantile_levels,
        )
        loss_info['cvar_loss'] = cvar_loss.item()
        
        # 3. Mean squared error (if means provided)
        mse_loss = torch.tensor(0.0, device=predicted_quantiles.device)
        if predicted_means is not None and target_means is not None:
            mse_loss = F.mse_loss(predicted_means, target_means)
            loss_info['mse_loss'] = mse_loss.item()
        
        # Combine losses
        total_loss = (
            (1 - self.risk_weight) * qr_loss
            + self.risk_weight * cvar_loss
            + 0.1 * mse_loss  # Small weight on MSE
        )
        
        loss_info['total_loss'] = total_loss.item()
        loss_info['risk_weight'] = self.risk_weight
        
        return total_loss, loss_info
    
    def _quantile_huber_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        taus: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Huberized quantile regression loss.
        
        More robust than plain quantile loss for outliers.
        """
        batch_size = predicted.shape[0]
        n_pred = predicted.shape[1]
        n_target = target.shape[1]
        
        # Pairwise differences
        delta = target.unsqueeze(1) - predicted.unsqueeze(2)
        
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


class AdaptiveRiskScheduler:
    """
    Dynamically adjusts risk sensitivity based on performance and market conditions.
    
    Increases risk aversion when:
    - Recent returns show high volatility
    - Drawdown exceeds threshold
    - Market regime indicates stress
    
    Decreases risk aversion when:
    - Consistent positive returns
    - Low volatility environment
    """
    
    def __init__(
        self,
        base_risk_weight: float = 0.3,
        min_risk_weight: float = 0.1,
        max_risk_weight: float = 0.7,
        adaptation_rate: float = 0.01,
    ):
        self.base_risk_weight = base_risk_weight
        self.current_risk_weight = base_risk_weight
        self.min_risk_weight = min_risk_weight
        self.max_risk_weight = max_risk_weight
        self.adaptation_rate = adaptation_rate
        
        # Performance tracking
        self.return_history: List[float] = []
        self.volatility_window = 100
    
    def update(
        self,
        current_return: float,
        current_drawdown: float,
        market_volatility: Optional[float] = None,
    ) -> float:
        """
        Update risk weight based on recent performance.
        
        Args:
            current_return: Most recent period return
            current_drawdown: Current drawdown (positive = loss)
            market_volatility: Optional external volatility measure
        
        Returns:
            Updated risk weight
        """
        self.return_history.append(current_return)
        if len(self.return_history) > self.volatility_window:
            self.return_history.pop(0)
        
        # Compute return volatility
        if len(self.return_history) >= 10:
            return_std = np.std(self.return_history)
        else:
            return_std = 0.0
        
        # Adjust risk weight based on volatility
        if return_std > 0.05:  # High volatility
            target_risk = min(self.max_risk_weight, self.current_risk_weight + self.adaptation_rate)
        elif return_std < 0.01:  # Low volatility
            target_risk = max(self.min_risk_weight, self.current_risk_weight - self.adaptation_rate)
        else:
            target_risk = self.base_risk_weight
        
        # Adjust for drawdown
        if current_drawdown > 0.03:  # Significant drawdown
            target_risk = min(self.max_risk_weight, target_risk + 0.05)
        
        # Smooth update
        self.current_risk_weight = (
            (1 - self.adaptation_rate) * self.current_risk_weight
            + self.adaptation_rate * target_risk
        )
        
        return self.current_risk_weight
    
    def get_risk_weight(self) -> float:
        """Get current risk weight."""
        return self.current_risk_weight
    
    def reset(self) -> None:
        """Reset to base configuration."""
        self.current_risk_weight = self.base_risk_weight
        self.return_history = []


def compute_portfolio_cvar(
    returns: torch.Tensor,
    risk_level: float = 0.05,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute CVaR for a portfolio of returns.
    
    Args:
        returns: Shape (n_samples,) or (n_samples, n_assets)
        risk_level: α for CVaR
        weights: Optional asset weights
    
    Returns:
        Portfolio CVaR
    """
    if returns.dim() == 1:
        returns = returns.unsqueeze(-1)
    
    n_samples = returns.shape[0]
    
    # Sort returns
    sorted_returns, _ = torch.sort(returns, dim=0)
    
    # Find cutoff for tail
    tail_size = max(1, int(risk_level * n_samples))
    
    # Extract tail returns
    tail_returns = sorted_returns[:tail_size]
    
    # Compute weighted CVaR
    if weights is not None:
        cvar = (tail_returns * weights.unsqueeze(0)).sum(dim=-1).mean()
    else:
        cvar = tail_returns.mean()
    
    return cvar
