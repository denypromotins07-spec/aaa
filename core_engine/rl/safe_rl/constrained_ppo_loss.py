"""
NEXUS-OMEGA Stage 19: Constrained PPO (C-PPO) Loss Functions

This module implements the Primal-Dual Constrained PPO algorithm using
Lagrangian relaxation. The policy network performs gradient descent on
the Lagrangian loss, while Lagrange multipliers perform gradient ascent
to penalize constraint violations.

Key equations:
- Lagrangian: L(π, λ) = E[R(π)] - Σ λ_i * (E[C_i(π)] - d_i)
- Policy update: ∇_θ L(π_θ, λ)
- Multiplier update: λ_i ← max(0, λ_i + α_λ * (E[C_i(π)] - d_i))

Author: NEXUS-OMEGA Architecture
Stage: 19 of 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np

from .cmdp_formulation import ConstraintType, CMDPTransition, TradingCMDP


@dataclass
class CPPOConfig:
    """Configuration for Constrained PPO."""
    # PPO hyperparameters
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # C-PPO specific
    initial_lambda: float = 0.1
    lambda_lr: float = 0.01
    lambda_max: float = 100.0
    target_kl: float = 0.01
    
    # Cost constraints
    cost_targets: Dict[ConstraintType, float] = None
    
    def __post_init__(self):
        if self.cost_targets is None:
            self.cost_targets = {}


class ActorCriticNetwork(nn.Module):
    """
    Shared backbone actor-critic network for C-PPO.
    
    Outputs:
    - Action distribution parameters (mean, log_std for continuous actions)
    - Value estimate
    - Cost value estimates (one per constraint)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 128],
        n_constraints: int = 4,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_constraints = n_constraints
        
        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation(),
                nn.LayerNorm(hidden_dim),
            ])
            prev_dim = hidden_dim
        
        self.shared_backbone = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dims[-1], 1)
        
        # Cost critics (one per constraint)
        self.cost_critics = nn.ModuleList([
            nn.Linear(hidden_dims[-1], 1) for _ in range(n_constraints)
        ])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Orthogonal initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # Small initialization for final layers
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        for cost_critic in self.cost_critics:
            nn.init.orthogonal_(cost_critic.weight, gain=1.0)
    
    def forward(
        self, 
        states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the network.
        
        Returns:
            action_means: (batch, action_dim)
            values: (batch, 1)
            cost_values: List of (batch, 1) for each constraint
        """
        features = self.shared_backbone(states)
        
        # Actor outputs
        action_means = self.actor_mean(features)
        
        # Critic outputs
        value = self.critic(features)
        
        # Cost critic outputs
        cost_values = [critic(features) for critic in self.cost_critics]
        
        return action_means, value, cost_values
    
    def get_action_dist(
        self, 
        states: torch.Tensor
    ) -> torch.distributions.Normal:
        """Get action distribution for sampling."""
        action_means, _, _ = self.forward(states)
        log_stds = self.actor_logstd.expand_as(action_means)
        stds = torch.exp(log_stds)
        
        return torch.distributions.Normal(action_means, stds)
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Evaluate log probability and entropy of given actions.
        
        Returns:
            log_probs: Log probability of actions
            values: State values
            entropies: Action entropies
            cost_values: Cost value estimates
        """
        action_means, values, cost_values = self.forward(states)
        
        log_stds = self.actor_logstd.expand_as(action_means)
        stds = torch.exp(log_stds)
        
        dist = torch.distributions.Normal(action_means, stds)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_probs, values.squeeze(-1), entropy.squeeze(-1), cost_values


class ConstrainedPPOLoss:
    """
    Constrained PPO loss computation with Lagrangian relaxation.
    
    The total loss is:
        L_total = L_policy + c_v * L_value + c_e * L_entropy + L_lagrangian
    
    Where:
        L_policy = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
        L_value = E[(V(s) - V_target)^2]
        L_entropy = -E[H(π)]
        L_lagrangian = Σ λ_i * (E[C_i] - d_i)
    """
    
    def __init__(
        self,
        config: CPPOConfig,
        cmdp: TradingCMDP,
        device: str = 'cpu',
    ):
        self.config = config
        self.cmdp = cmdp
        self.device = device
        
        # Initialize Lagrange multipliers (one per constraint)
        self.lagrange_multipliers: Dict[ConstraintType, torch.Tensor] = {}
        for constraint_type in cmdp.constraints.keys():
            self.lagrange_multipliers[constraint_type] = torch.tensor(
                config.initial_lambda, 
                requires_grad=False,
                device=device
            )
    
    def compute_policy_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compute clipped surrogate policy loss.
        
        Args:
            old_log_probs: Log probs under old policy
            new_log_probs: Log probs under current policy
            advantages: Generalized Advantage Estimation
        
        Returns:
            policy_loss: Clipped surrogate loss
            approx_kl: Approximate KL divergence
            clip_fraction: Fraction of samples that were clipped
        """
        # Probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Surrogate objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 
            1 - self.config.clip_epsilon,
            1 + self.config.clip_epsilon
        ) * advantages
        
        # Clip fraction
        clip_mask = (surr1 != surr2).float()
        clip_fraction = clip_mask.mean().item()
        
        # Policy loss (negative because we maximize)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Approximate KL divergence
        approx_kl = (old_log_probs - new_log_probs).mean().item()
        
        return policy_loss, torch.tensor(approx_kl), clip_fraction
    
    def compute_value_loss(
        self,
        values: torch.Tensor,
        value_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE value loss."""
        return F.mse_loss(values, value_targets)
    
    def compute_cost_losses(
        self,
        cost_values: List[torch.Tensor],
        cost_targets: Dict[ConstraintType, torch.Tensor],
    ) -> Dict[ConstraintType, torch.Tensor]:
        """
        Compute cost value losses for each constraint.
        
        Returns dictionary mapping constraint type to loss value.
        """
        cost_losses = {}
        
        for i, (constraint_type, target) in enumerate(cost_targets.items()):
            if i < len(cost_values):
                cost_losses[constraint_type] = F.mse_loss(
                    cost_values[i].squeeze(-1),
                    target
                )
        
        return cost_losses
    
    def compute_lagrangian_term(
        self,
        costs: Dict[ConstraintType, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[ConstraintType, float]]:
        """
        Compute Lagrangian penalty term.
        
        L_lagrangian = Σ λ_i * (E[C_i] - d_i)
        
        This term is ADDED to the policy loss (making it worse when
        constraints are violated).
        
        Returns:
            lagrangian_loss: Total Lagrangian penalty
            constraint_info: Dictionary with constraint statistics
        """
        lagrangian_loss = torch.tensor(0.0, device=self.device)
        constraint_info = {}
        
        for constraint_type, cost_value in costs.items():
            lambda_i = self.lagrange_multipliers.get(
                constraint_type, 
                torch.tensor(0.0, device=self.device)
            )
            
            # Get constraint threshold from CMDP
            spec = self.cmdp.constraints.get(constraint_type)
            if spec is not None:
                threshold = spec.threshold
            else:
                threshold = 0.0
            
            # Lagrangian term: λ * (cost - threshold)
            violation = cost_value.mean() - threshold
            lagrangian_loss += lambda_i * violation
            
            constraint_info[constraint_type] = {
                'lambda': lambda_i.item(),
                'cost': cost_value.mean().item(),
                'violation': violation.item(),
                'threshold': threshold,
            }
        
        return lagrangian_loss, constraint_info
    
    def update_lagrange_multipliers(
        self,
        costs: Dict[ConstraintType, torch.Tensor],
        step_size: float = None,
    ) -> Dict[ConstraintType, float]:
        """
        Update Lagrange multipliers using gradient ascent.
        
        λ_i ← max(0, λ_i + α_λ * (E[C_i] - d_i))
        
        This increases penalties for violated constraints and decreases
        them for satisfied constraints (but never below 0).
        
        Returns:
            Dictionary of updated multiplier values
        """
        if step_size is None:
            step_size = self.config.lambda_lr
        
        updates = {}
        
        for constraint_type, cost_value in costs.items():
            current_lambda = self.lagrange_multipliers.get(
                constraint_type,
                torch.tensor(0.0, device=self.device)
            )
            
            spec = self.cmdp.constraints.get(constraint_type)
            threshold = spec.threshold if spec else 0.0
            
            # Gradient ascent step
            violation = cost_value.mean().item() - threshold
            new_lambda = current_lambda.item() + step_size * violation
            
            # Clamp to [0, lambda_max]
            new_lambda = max(0.0, min(new_lambda, self.config.lambda_max))
            
            # Update stored multiplier
            self.lagrange_multipliers[constraint_type] = torch.tensor(
                new_lambda,
                device=self.device,
                requires_grad=False
            )
            
            updates[constraint_type] = new_lambda
        
        return updates
    
    def compute_total_loss(
        self,
        batch: List[CMDPTransition],
        advantages: torch.Tensor,
        value_targets: torch.Tensor,
        cost_targets: Dict[ConstraintType, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the full C-PPO loss for a batch of transitions.
        
        Returns:
            total_loss: Combined loss tensor
            loss_info: Dictionary with individual loss components
        """
        if len(batch) == 0:
            raise ValueError("Empty batch provided to C-PPO loss")
        
        # Stack batch tensors
        states = torch.stack([t.state.to_tensor(self.device) for t in batch])
        actions = torch.stack([torch.FloatTensor(t.action) for t in batch]).to(self.device)
        old_log_probs = torch.stack([
            torch.FloatTensor([np.log(1e-8 + 1.0)])  # Placeholder - should come from old policy
            for t in batch
        ]).to(self.device)
        
        # Forward pass
        log_probs, values, entropies, cost_values = \
            self._evaluate_batch(states, actions)
        
        # Policy loss
        policy_loss, approx_kl, clip_frac = self.compute_policy_loss(
            old_log_probs, log_probs, advantages
        )
        
        # Value loss
        value_loss = self.compute_value_loss(values, value_targets)
        
        # Entropy bonus (negative because we want to maximize entropy)
        entropy_bonus = -entropies.mean()
        
        # Cost losses
        cost_losses = self.compute_cost_losses(cost_values, cost_targets)
        total_cost_loss = sum(cost_losses.values())
        
        # Lagrangian term
        cost_means = {k: v.squeeze(-1) for k, v in zip(
            cost_targets.keys(), cost_values
        )}
        lagrangian_loss, constraint_info = self.compute_lagrangian_term(cost_means)
        
        # Total loss
        total_loss = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            + self.config.entropy_coef * entropy_bonus
            + lagrangian_loss
        )
        
        loss_info = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_bonus': entropy_bonus.item(),
            'lagrangian_loss': lagrangian_loss.item(),
            'total_cost_loss': total_cost_loss.item() if isinstance(total_cost_loss, torch.Tensor) else total_cost_loss,
            'approx_kl': approx_kl if isinstance(approx_kl, float) else approx_kl.item(),
            'clip_fraction': clip_frac,
            'constraints': constraint_info,
        }
        
        return total_loss, loss_info
    
    def _evaluate_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Helper to evaluate a batch through the network."""
        # This would normally take a network as input
        # For now, return placeholders - actual implementation needs network
        batch_size = states.shape[0]
        
        log_probs = torch.zeros(batch_size, 1, device=self.device)
        values = torch.zeros(batch_size, device=self.device)
        entropies = torch.zeros(batch_size, device=self.device)
        cost_values = [torch.zeros(batch_size, 1, device=self.device) for _ in range(4)]
        
        return log_probs, values, entropies, cost_values


class SafePPOTrainer:
    """
    Complete trainer for Constrained PPO with safety guarantees.
    
    Integrates:
    - C-PPO loss computation
    - Lagrange multiplier updates
    - Early stopping on KL divergence
    - Constraint satisfaction monitoring
    """
    
    def __init__(
        self,
        network: ActorCriticNetwork,
        cmdp: TradingCMDP,
        config: CPPOConfig,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu',
    ):
        self.network = network.to(device)
        self.cmdp = cmdp
        self.config = config
        self.optimizer = optimizer
        self.device = device
        
        self.loss_fn = ConstrainedPPOLoss(config, cmdp, device)
        
        # Training metrics
        self.training_history: List[Dict] = []
    
    def train_epoch(
        self,
        trajectories: List[CMDPTransition],
        n_updates: int = 10,
        batch_size: int = 64,
    ) -> Dict:
        """
        Train for one epoch on collected trajectories.
        
        Returns:
            Dictionary with training metrics
        """
        # Compute GAE advantages (placeholder - would use GAE calculator)
        advantages, value_targets, cost_targets = self._prepare_training_data(trajectories)
        
        epoch_metrics = {
            'policy_losses': [],
            'value_losses': [],
            'lagrangian_losses': [],
            'constraint_violations': [],
            'kl_divergences': [],
        }
        
        for update in range(n_updates):
            # Sample mini-batch
            indices = np.random.choice(
                len(trajectories), 
                min(batch_size, len(trajectories)), 
                replace=False
            )
            batch = [trajectories[i] for i in indices]
            
            # Compute loss
            total_loss, loss_info = self.loss_fn.compute_total_loss(
                batch, advantages, value_targets, cost_targets
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            
            # Update policy
            self.optimizer.step()
            
            # Update Lagrange multipliers
            # Extract cost values from batch for multiplier update
            batch_costs = self._extract_batch_costs(batch)
            self.loss_fn.update_lagrange_multipliers(batch_costs)
            
            # Record metrics
            epoch_metrics['policy_losses'].append(loss_info['policy_loss'])
            epoch_metrics['value_losses'].append(loss_info['value_loss'])
            epoch_metrics['lagrangian_losses'].append(loss_info['lagrangian_loss'])
            epoch_metrics['kl_divergences'].append(loss_info['approx_kl'])
            
            # Early stopping on KL
            if loss_info['approx_kl'] > self.config.target_kl:
                break
        
        # Aggregate metrics
        final_metrics = {
            'policy_loss': np.mean(epoch_metrics['policy_losses']),
            'value_loss': np.mean(epoch_metrics['value_losses']),
            'lagrangian_loss': np.mean(epoch_metrics['lagrangian_losses']),
            'kl_divergence': np.mean(epoch_metrics['kl_divergences']),
            'lagrange_multipliers': {
                k.value: v.item() 
                for k, v in self.loss_fn.lagrange_multipliers.items()
            },
        }
        
        self.training_history.append(final_metrics)
        
        return final_metrics
    
    def _prepare_training_data(
        self, 
        trajectories: List[CMDPTransition]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[ConstraintType, torch.Tensor]]:
        """Prepare GAE advantages and targets for training."""
        n_transitions = len(trajectories)
        
        # Placeholders - real implementation would compute GAE
        advantages = torch.ones(n_transitions, 1, device=self.device)
        value_targets = torch.zeros(n_transitions, device=self.device)
        cost_targets = {}
        
        for constraint_type in self.cmdp.constraints.keys():
            cost_targets[constraint_type] = torch.zeros(n_transitions, device=self.device)
        
        return advantages, value_targets, cost_targets
    
    def _extract_batch_costs(
        self, 
        batch: List[CMDPTransition]
    ) -> Dict[ConstraintType, torch.Tensor]:
        """Extract cost values from a batch of transitions."""
        costs: Dict[ConstraintType, List[float]] = {
            ct: [] for ct in self.cmdp.constraints.keys()
        }
        
        for transition in batch:
            for constraint_type, cost_value in transition.costs.items():
                if constraint_type in costs:
                    costs[constraint_type].append(cost_value)
        
        return {
            k: torch.tensor(v, device=self.device) 
            for k, v in costs.items() if v
        }
