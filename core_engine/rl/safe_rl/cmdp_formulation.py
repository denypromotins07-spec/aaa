"""
NEXUS-OMEGA Stage 19: Constrained Markov Decision Process (CMDP) Formulation

This module formulates the trading environment as a CMDP where the objective
is to maximize Sharpe Ratio subject to strict cost constraints:
- Max Drawdown < 5%
- VaR < 2%
- Leverage < 3x

The CMDP is defined as (S, A, P, R, C, γ) where:
- S: State space (market features, portfolio state, risk metrics)
- A: Action space (position sizing, order types)
- P: Transition dynamics
- R: Reward function (Sharpe-adjusted returns)
- C: Cost functions (drawdown, VaR, leverage violations)
- γ: Discount factor

Author: NEXUS-OMEGA Architecture
Stage: 19 of 50
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum, auto
import torch
import torch.nn as nn


class ConstraintType(Enum):
    """Types of constraints enforced in the CMDP."""
    MAX_DRAWDOWN = auto()      # Maximum allowed drawdown
    VALUE_AT_RISK = auto()     # Maximum VaR threshold
    LEVERAGE = auto()          # Maximum leverage ratio
    TURNOVER = auto()          # Maximum position turnover
    EXPOSURE = auto()          # Maximum net exposure
    VOLATILITY = auto()        # Maximum portfolio volatility


@dataclass
class ConstraintSpec:
    """Specification for a single constraint in the CMDP."""
    constraint_type: ConstraintType
    threshold: float           # Maximum allowed value
    name: str
    soft_margin: float = 0.1   # Buffer zone before hard violation
    penalty_weight: float = 1.0  # Initial penalty weight
    
    def is_violated(self, value: float) -> bool:
        """Check if the constraint is violated."""
        return value > self.threshold
    
    def is_in_warning_zone(self, value: float) -> bool:
        """Check if value is in the warning zone (approaching violation)."""
        warning_threshold = self.threshold * (1 - self.soft_margin)
        return warning_threshold < value <= self.threshold
    
    def violation_magnitude(self, value: float) -> float:
        """Calculate how much the constraint is violated."""
        return max(0.0, value - self.threshold)


@dataclass
class CMDPState:
    """
    Complete state representation for the trading CMDP.
    
    Combines market observations, portfolio state, and risk metrics
    into a unified state vector for the RL agent.
    """
    # Market features (from Stage 11/18 feature engineering)
    market_features: np.ndarray  # Shape: (n_features,)
    
    # Portfolio state
    positions: np.ndarray        # Current positions per asset
    cash_balance: float          # Available cash
    total_equity: float          # Total portfolio equity
    
    # Risk metrics (from Stage 5 Risk Engine)
    current_drawdown: float      # Current drawdown from peak
    peak_equity: float           # Historical peak equity
    var_95: float               # 95% Value at Risk
    expected_shortfall: float    # Expected Shortfall (CVaR)
    portfolio_volatility: float  # Realized volatility
    leverage_ratio: float        # Gross exposure / equity
    
    # Constraint tracking
    constraint_costs: Dict[ConstraintType, float] = field(default_factory=dict)
    
    # Metadata
    timestamp: int = 0
    episode_step: int = 0
    
    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """Convert state to PyTorch tensor for neural network input."""
        # Concatenate all numerical features
        feature_list = [
            self.market_features,
            self.positions,
            np.array([self.cash_balance, self.total_equity]),
            np.array([self.current_drawdown, self.peak_equity]),
            np.array([self.var_95, self.expected_shortfall]),
            np.array([self.portfolio_volatility, self.leverage_ratio]),
        ]
        
        # Flatten and concatenate
        flat_features = np.concatenate([f.flatten() for f in feature_list])
        return torch.FloatTensor(flat_features).to(device)
    
    @property
    def state_dict(self) -> Dict[str, float]:
        """Return state as dictionary for constraint checking."""
        return {
            ConstraintType.MAX_DRAWDOWN: abs(self.current_drawdown),
            ConstraintType.VALUE_AT_RISK: self.var_95,
            ConstraintType.LEVERAGE: self.leverage_ratio,
            ConstraintType.VOLATILITY: self.portfolio_volatility,
        }


@dataclass
class CMDPTransition:
    """
    A transition tuple in the CMDP: (s, a, s', r, c, done, info)
    
    Extends standard MDP transitions with cost vector c for constraints.
    """
    state: CMDPState
    action: np.ndarray
    next_state: CMDPState
    reward: float
    costs: Dict[ConstraintType, float]  # Cost for each constraint
    done: bool
    info: Dict = field(default_factory=dict)
    
    def to_training_batch(self) -> Dict[str, torch.Tensor]:
        """Convert transition to training batch format."""
        return {
            'state': self.state.to_tensor(),
            'action': torch.FloatTensor(self.action),
            'next_state': self.next_state.to_tensor(),
            'reward': torch.tensor(self.reward),
            'costs': {k.value: torch.tensor(v) for k, v in self.costs.items()},
            'done': torch.tensor(self.done, dtype=torch.bool),
        }


class TradingCMDP:
    """
    Constrained Markov Decision Process for algorithmic trading.
    
    This class wraps the trading environment and provides:
    1. Constraint cost calculation at each timestep
    2. Safe state validation
    3. Lagrangian penalty computation for C-PPO
    
    The CMDP objective is:
        max_π E[Σ γ^t r(s_t, a_t)] 
        s.t. E[Σ γ^t c_i(s_t, a_t)] ≤ d_i for all constraints i
    
    Where d_i is the constraint threshold.
    """
    
    def __init__(
        self,
        constraints: List[ConstraintSpec],
        discount_factor: float = 0.99,
        cost_discount: float = 0.99,
    ):
        self.constraints = {c.constraint_type: c for c in constraints}
        self.gamma = discount_factor
        self.cost_gamma = cost_discount
        
        # Constraint thresholds for quick access
        self.thresholds = {
            c.constraint_type: c.threshold for c in constraints
        }
    
    def compute_constraint_costs(
        self, 
        state: CMDPState, 
        action: np.ndarray
    ) -> Dict[ConstraintType, float]:
        """
        Compute instantaneous constraint costs for the current state-action pair.
        
        Costs are designed such that:
        - cost = 0 when constraint is satisfied
        - cost > 0 when constraint is violated
        - Cost magnitude reflects severity of violation
        
        Returns:
            Dictionary mapping ConstraintType to cost value
        """
        costs = {}
        state_values = state.state_dict
        
        for constraint_type, spec in self.constraints.items():
            if constraint_type in state_values:
                current_value = state_values[constraint_type]
                
                # Cost is the violation magnitude (0 if satisfied)
                violation = spec.violation_magnitude(current_value)
                
                # Add warning zone penalty (smooth penalty before hard violation)
                if spec.is_in_warning_zone(current_value):
                    warning_penalty = (
                        (current_value - spec.threshold * (1 - spec.soft_margin)) 
                        / (spec.threshold * spec.soft_margin)
                        * 0.1  # Small penalty in warning zone
                    )
                    violation += warning_penalty
                
                costs[constraint_type] = violation
        
        # Special handling for action-dependent constraints
        if ConstraintType.TURNOVER in self.constraints:
            turnover_cost = self._compute_turnover_cost(state, action)
            costs[ConstraintType.TURNOVER] = turnover_cost
        
        if ConstraintType.EXPOSURE in self.constraints:
            exposure_cost = self._compute_exposure_cost(state, action)
            costs[ConstraintType.EXPOSURE] = exposure_cost
        
        return costs
    
    def _compute_turnover_cost(
        self, 
        state: CMDPState, 
        action: np.ndarray
    ) -> float:
        """Compute turnover cost based on position changes."""
        current_positions = state.positions
        new_positions = current_positions + action[:len(current_positions)]
        
        # Turnover = sum of absolute position changes / equity
        position_change = np.sum(np.abs(new_positions - current_positions))
        turnover = position_change / max(state.total_equity, 1e-8)
        
        spec = self.constraints.get(ConstraintType.TURNOVER)
        if spec is None:
            return 0.0
        
        return spec.violation_magnitude(turnover)
    
    def _compute_exposure_cost(
        self, 
        state: CMDPState, 
        action: np.ndarray
    ) -> float:
        """Compute exposure cost based on proposed action."""
        # Estimate new gross exposure after action
        current_positions = state.positions
        new_positions = current_positions + action[:len(current_positions)]
        
        # Assume prices unchanged for immediate exposure estimate
        gross_exposure = np.sum(np.abs(new_positions))
        new_leverage = gross_exposure / max(state.total_equity, 1e-8)
        
        spec = self.constraints.get(ConstraintType.EXPOSURE)
        if spec is None:
            return 0.0
        
        return spec.violation_magnitude(new_leverage)
    
    def compute_lagrangian_cost(
        self, 
        costs: Dict[ConstraintType, float],
        lagrange_multipliers: Dict[ConstraintType, float]
    ) -> float:
        """
        Compute the weighted Lagrangian cost term.
        
        L(λ, π) = Σ λ_i * (E[C_i(π)] - d_i)
        
        This cost is subtracted from the reward during C-PPO training.
        """
        total_cost = 0.0
        for constraint_type, cost_value in costs.items():
            lambda_i = lagrange_multipliers.get(constraint_type, 0.0)
            total_cost += lambda_i * cost_value
        
        return total_cost
    
    def is_state_safe(self, state: CMDPState) -> Tuple[bool, List[ConstraintType]]:
        """
        Check if the current state satisfies all hard constraints.
        
        Returns:
            (is_safe, violated_constraints)
        """
        state_values = state.state_dict
        violated = []
        
        for constraint_type, spec in self.constraints.items():
            if constraint_type in state_values:
                if spec.is_violated(state_values[constraint_type]):
                    violated.append(constraint_type)
        
        return len(violated) == 0, violated
    
    def get_safety_margin(self, state: CMDPState) -> Dict[ConstraintType, float]:
        """
        Calculate safety margin for each constraint.
        
        Positive margin = safe, Negative margin = violated
        """
        state_values = state.state_dict
        margins = {}
        
        for constraint_type, spec in self.constraints.items():
            if constraint_type in state_values:
                current_value = state_values[constraint_type]
                margin = spec.threshold - current_value
                margins[constraint_type] = margin
        
        return margins


class CMDPBatchSampler:
    """
    Sampler for CMDP transitions that respects constraint sparsity.
    
    Ensures balanced sampling of:
    1. Normal transitions (no constraint violations)
    2. Warning transitions (approaching constraints)
    3. Violation transitions (constraint breached)
    """
    
    def __init__(
        self,
        buffer_capacity: int = 100000,
        violation_sample_ratio: float = 0.3,
        warning_sample_ratio: float = 0.3,
    ):
        self.capacity = buffer_capacity
        self.violation_ratio = violation_sample_ratio
        self.warning_ratio = warning_sample_ratio
        
        # Separate buffers for different transition types
        self.normal_buffer: List[CMDPTransition] = []
        self.warning_buffer: List[CMDPTransition] = []
        self.violation_buffer: List[CMDPTransition] = []
    
    def add_transition(self, transition: CMDPTransition) -> None:
        """Add a transition to the appropriate buffer."""
        # Classify transition
        has_violation = any(v > 0 for v in transition.costs.values())
        state_values = transition.state.state_dict
        
        has_warning = False
        if not has_violation:
            for constraint_type, value in state_values.items():
                if constraint_type in self._get_constraint_specs():
                    spec = self._get_constraint_specs()[constraint_type]
                    if spec.is_in_warning_zone(value):
                        has_warning = True
                        break
        
        # Add to appropriate buffer
        if has_violation:
            self.violation_buffer.append(transition)
            self._trim_buffer(self.violation_buffer)
        elif has_warning:
            self.warning_buffer.append(transition)
            self._trim_buffer(self.warning_buffer)
        else:
            self.normal_buffer.append(transition)
            self._trim_buffer(self.normal_buffer)
    
    def _trim_buffer(self, buffer: List) -> None:
        """Trim buffer to capacity using FIFO."""
        while len(buffer) > self.capacity:
            buffer.pop(0)
    
    def _get_constraint_specs(self) -> Dict[ConstraintType, ConstraintSpec]:
        """Get constraint specs (placeholder for integration)."""
        return {}
    
    def sample_batch(
        self, 
        batch_size: int
    ) -> List[CMDPTransition]:
        """Sample a balanced batch of transitions."""
        total_needed = batch_size
        
        # Calculate samples from each buffer
        n_violations = min(
            int(batch_size * self.violation_ratio),
            len(self.violation_buffer)
        )
        n_warnings = min(
            int(batch_size * self.warning_ratio),
            len(self.warning_buffer)
        )
        n_normal = batch_size - n_violations - n_warnings
        n_normal = min(n_normal, len(self.normal_buffer))
        
        # Sample from each buffer
        batch = []
        
        if n_violations > 0:
            indices = np.random.choice(
                len(self.violation_buffer), 
                n_violations, 
                replace=False
            )
            batch.extend([self.violation_buffer[i] for i in indices])
        
        if n_warnings > 0:
            indices = np.random.choice(
                len(self.warning_buffer), 
                n_warnings, 
                replace=False
            )
            batch.extend([self.warning_buffer[i] for i in indices])
        
        if n_normal > 0:
            indices = np.random.choice(
                len(self.normal_buffer), 
                n_normal, 
                replace=False
            )
            batch.extend([self.normal_buffer[i] for i in indices])
        
        return batch
    
    def __len__(self) -> int:
        return len(self.normal_buffer) + len(self.warning_buffer) + len(self.violation_buffer)


# Default constraint specifications for trading
DEFAULT_CONSTRAINTS = [
    ConstraintSpec(
        constraint_type=ConstraintType.MAX_DRAWDOWN,
        threshold=0.05,  # 5% max drawdown
        name="Max Drawdown",
        soft_margin=0.01,
        penalty_weight=10.0,
    ),
    ConstraintSpec(
        constraint_type=ConstraintType.VALUE_AT_RISK,
        threshold=0.02,  # 2% daily VaR
        name="VaR Limit",
        soft_margin=0.005,
        penalty_weight=5.0,
    ),
    ConstraintSpec(
        constraint_type=ConstraintType.LEVERAGE,
        threshold=3.0,  # 3x max leverage
        name="Leverage Cap",
        soft_margin=0.2,
        penalty_weight=2.0,
    ),
    ConstraintSpec(
        constraint_type=ConstraintType.VOLATILITY,
        threshold=0.15,  # 15% annualized vol
        name="Volatility Limit",
        soft_margin=0.02,
        penalty_weight=1.0,
    ),
]


def create_default_cmdp() -> TradingCMDP:
    """Create a TradingCMDP with default constraints."""
    return TradingCMDP(constraints=DEFAULT_CONSTRAINTS)
