"""
NEXUS-OMEGA Stage 19: Lagrange Multiplier Scheduler for C-PPO

This module implements an adaptive penalty scheduler that prevents
Lagrange multipliers from:
1. Diverging to infinity (which would freeze the policy)
2. Collapsing to zero (which would ignore risk constraints)

The scheduler uses multiple techniques:
- Hard clamping to [0, λ_max]
- Adaptive learning rates based on constraint satisfaction
- Exponential moving average smoothing
- Dual-rate adaptation (fast increase, slow decrease)

Author: NEXUS-OMEGA Architecture
Stage: 19 of 50
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

from .cmdp_formulation import ConstraintType, ConstraintSpec, TradingCMDP


class AdaptationMode(Enum):
    """Modes for Lagrange multiplier adaptation."""
    GRADIENT_ASCENT = auto()      # Standard dual gradient ascent
    ADAPTIVE_LR = auto()          # Learning rate adapts to violation history
    DUAL_RATE = auto()            # Fast increase, slow decrease
    SMOOTHED = auto()             # EMA smoothing on updates
    HYSTERESIS = auto()           # Hysteresis to prevent oscillation


@dataclass
class LagrangeSchedulerConfig:
    """Configuration for the Lagrange multiplier scheduler."""
    # Bounds
    lambda_min: float = 0.0
    lambda_max: float = 100.0
    
    # Learning rates
    base_lr: float = 0.01
    max_lr: float = 0.1
    min_lr: float = 0.001
    
    # Adaptation parameters
    ema_alpha: float = 0.1        # Smoothing factor for EMA
    violation_threshold: float = 0.01  # Tolerance for constraint satisfaction
    
    # Dual-rate parameters
    increase_factor: float = 1.5   # Faster increase when violated
    decrease_factor: float = 0.5   # Slower decrease when satisfied
    
    # Hysteresis parameters
    hysteresis_band: float = 0.1   # Band to prevent rapid switching
    
    # Mode
    mode: AdaptationMode = AdaptationMode.ADAPTIVE_LR


class ConstraintTracker:
    """
    Tracks constraint satisfaction history for adaptive scheduling.
    
    Maintains:
    - Running average of constraint violations
    - Consecutive violation count
    - Time since last violation
    - Satisfaction rate
    """
    
    def __init__(self, ema_alpha: float = 0.1):
        self.ema_alpha = ema_alpha
        
        # Exponential moving averages
        self.violation_ema: float = 0.0
        self.satisfaction_rate: float = 1.0
        
        # Counters
        self.consecutive_violations: int = 0
        self.consecutive_satisfactions: int = 0
        self.total_steps: int = 0
        self.violation_steps: int = 0
    
    def update(self, violation_magnitude: float) -> None:
        """
        Update tracker with new violation measurement.
        
        Args:
            violation_magnitude: How much constraint is violated (>0 if violated)
        """
        self.total_steps += 1
        
        is_violated = violation_magnitude > 0
        
        if is_violated:
            self.violation_steps += 1
            self.consecutive_violations += 1
            self.consecutive_satisfactions = 0
        else:
            self.consecutive_violations = 0
            self.consecutive_satisfactions += 1
        
        # Update EMA of violations
        self.violation_ema = (
            self.ema_alpha * violation_magnitude 
            + (1 - self.ema_alpha) * self.violation_ema
        )
        
        # Update satisfaction rate
        self.satisfaction_rate = (
            self.total_steps - self.violation_steps
        ) / self.total_steps
    
    @property
    def is_persistently_violated(self) -> bool:
        """Check if constraint is persistently violated."""
        return self.consecutive_violations >= 5
    
    @property
    def is_stable(self) -> bool:
        """Check if constraint satisfaction is stable."""
        return self.consecutive_satisfactions >= 10
    
    def get_adaptation_signal(self) -> float:
        """
        Get signal for adaptation strength.
        
        Returns value in [0, 1] where:
        - 0 = constraint well satisfied, can decrease λ
        - 1 = constraint severely violated, need to increase λ
        """
        return min(1.0, self.violation_ema * 10)


class LagrangeMultiplierScheduler:
    """
    Adaptive scheduler for Lagrange multipliers in C-PPO.
    
    Prevents divergence and collapse through:
    1. Strict clamping to [λ_min, λ_max]
    2. Adaptive learning rates based on violation history
    3. Dual-rate adaptation (fast increase, slow decrease)
    4. EMA smoothing to reduce oscillations
    5. Hysteresis to prevent rapid switching
    """
    
    def __init__(
        self,
        cmdp: TradingCMDP,
        config: LagrangeSchedulerConfig,
        device: str = 'cpu',
    ):
        self.cmdp = cmdp
        self.config = config
        self.device = device
        
        # Initialize multipliers
        self.multipliers: Dict[ConstraintType, torch.Tensor] = {}
        for constraint_type in cmdp.constraints.keys():
            self.multipliers[constraint_type] = torch.tensor(
                config.base_lr * 10,  # Initial λ value
                dtype=torch.float32,
                device=device,
                requires_grad=False
            )
        
        # Trackers for each constraint
        self.trackers: Dict[ConstraintType, ConstraintTracker] = {
            ct: ConstraintTracker(config.ema_alpha)
            for ct in cmdp.constraints.keys()
        }
        
        # Effective learning rates per constraint
        self.effective_lrs: Dict[ConstraintType, float] = {
            ct: config.base_lr for ct in cmdp.constraints.keys()
        }
        
        # History for monitoring
        self.history: Dict[ConstraintType, List[float]] = {
            ct: [] for ct in cmdp.constraints.keys()
        }
    
    def step(
        self,
        costs: Dict[ConstraintType, float],
        thresholds: Optional[Dict[ConstraintType, float]] = None,
    ) -> Dict[ConstraintType, float]:
        """
        Perform one step of Lagrange multiplier updates.
        
        Args:
            costs: Current constraint cost values
            thresholds: Optional override thresholds (uses CMDP defaults if None)
        
        Returns:
            Dictionary of updated multiplier values
        """
        if thresholds is None:
            thresholds = {
                ct: spec.threshold 
                for ct, spec in self.cmdp.constraints.items()
            }
        
        updates = {}
        
        for constraint_type in self.cmdp.constraints.keys():
            cost = costs.get(constraint_type, 0.0)
            threshold = thresholds.get(constraint_type, 0.0)
            
            # Compute violation
            violation = cost - threshold
            
            # Update tracker
            self.trackers[constraint_type].update(max(0.0, violation))
            
            # Compute update based on mode
            if self.config.mode == AdaptationMode.GRADIENT_ASCENT:
                delta = self._gradient_ascent_update(violation, constraint_type)
            elif self.config.mode == AdaptationMode.ADAPTIVE_LR:
                delta = self._adaptive_lr_update(violation, constraint_type)
            elif self.config.mode == AdaptationMode.DUAL_RATE:
                delta = self._dual_rate_update(violation, constraint_type)
            elif self.config.mode == AdaptationMode.SMOOTHED:
                delta = self._smoothed_update(violation, constraint_type)
            elif self.config.mode == AdaptationMode.HYSTERESIS:
                delta = self._hysteresis_update(violation, constraint_type)
            else:
                delta = self._gradient_ascent_update(violation, constraint_type)
            
            # Apply update with strict clamping
            current = self.multipliers[constraint_type].item()
            new_value = current + delta
            
            # CRITICAL: Strict clamping to prevent divergence
            new_value = max(self.config.lambda_min, min(new_value, self.config.lambda_max))
            
            self.multipliers[constraint_type] = torch.tensor(
                new_value,
                dtype=torch.float32,
                device=self.device,
                requires_grad=False
            )
            
            updates[constraint_type] = new_value
            
            # Record history
            self.history[constraint_type].append(new_value)
            if len(self.history[constraint_type]) > 1000:
                self.history[constraint_type].pop(0)
        
        return updates
    
    def _gradient_ascent_update(
        self, 
        violation: float, 
        constraint_type: ConstraintType
    ) -> float:
        """Standard gradient ascent update."""
        lr = self.effective_lrs[constraint_type]
        return lr * violation
    
    def _adaptive_lr_update(
        self, 
        violation: float, 
        constraint_type: ConstraintType
    ) -> float:
        """
        Adaptive learning rate based on violation history.
        
        Increases LR when persistently violated, decreases when stable.
        """
        tracker = self.trackers[constraint_type]
        base_lr = self.config.base_lr
        
        # Adjust learning rate based on persistence
        if tracker.is_persistently_violated:
            # Increase LR to respond faster
            self.effective_lrs[constraint_type] = min(
                self.config.max_lr,
                self.effective_lrs[constraint_type] * 1.1
            )
        elif tracker.is_stable:
            # Decrease LR for fine-tuning
            self.effective_lrs[constraint_type] = max(
                self.config.min_lr,
                self.effective_lrs[constraint_type] * 0.95
            )
        
        lr = self.effective_lrs[constraint_type]
        return lr * violation
    
    def _dual_rate_update(
        self, 
        violation: float, 
        constraint_type: ConstraintType
    ) -> float:
        """
        Dual-rate update: fast increase when violated, slow decrease when satisfied.
        
        This asymmetric update ensures quick response to violations while
        preventing premature relaxation of constraints.
        """
        lr = self.config.base_lr
        
        if violation > 0:
            # Violated: use faster increase rate
            lr *= self.config.increase_factor
        else:
            # Satisfied: use slower decrease rate
            lr *= self.config.decrease_factor
        
        return lr * violation
    
    def _smoothed_update(
        self, 
        violation: float, 
        constraint_type: ConstraintType
    ) -> float:
        """
        EMA-smoothed update to reduce oscillations.
        
        Applies smoothing to the violation signal before computing update.
        """
        tracker = self.trackers[constraint_type]
        
        # Use EMA of violations instead of raw value
        smoothed_violation = (
            self.config.ema_alpha * violation 
            + (1 - self.config.ema_alpha) * tracker.violation_ema
        )
        
        return self.config.base_lr * smoothed_violation
    
    def _hysteresis_update(
        self, 
        violation: float, 
        constraint_type: ConstraintType
    ) -> float:
        """
        Hysteresis-based update to prevent rapid switching.
        
        Only updates when violation exceeds hysteresis band.
        """
        tracker = self.trackers[constraint_type]
        current_lambda = self.multipliers[constraint_type].item()
        
        # Determine direction
        if violation > self.config.hysteresis_band:
            # Clearly violated: increase λ
            return self.config.base_lr * violation
        elif violation < -self.config.hysteresis_band:
            # Clearly satisfied: decrease λ
            return self.config.base_lr * violation
        else:
            # In hysteresis band: no update
            return 0.0
    
    def get_multipliers(self) -> Dict[ConstraintType, float]:
        """Get current multiplier values."""
        return {
            ct: mult.item() for ct, mult in self.multipliers.items()
        }
    
    def get_multiplier_tensor(
        self, 
        constraint_type: ConstraintType
    ) -> torch.Tensor:
        """Get multiplier as tensor for loss computation."""
        return self.multipliers[constraint_type]
    
    def reset(self) -> None:
        """Reset all multipliers to initial values."""
        initial_value = self.config.base_lr * 10
        for constraint_type in self.cmdp.constraints.keys():
            self.multipliers[constraint_type] = torch.tensor(
                initial_value,
                dtype=torch.float32,
                device=self.device,
                requires_grad=False
            )
            self.trackers[constraint_type] = ConstraintTracker(self.config.ema_alpha)
            self.effective_lrs[constraint_type] = self.config.base_lr
    
    def get_diagnostics(self) -> Dict:
        """
        Get diagnostic information about scheduler state.
        
        Useful for monitoring training stability.
        """
        diagnostics = {
            'multipliers': {},
            'trackers': {},
            'learning_rates': {},
        }
        
        for ct in self.cmdp.constraints.keys():
            diagnostics['multipliers'][ct.value] = self.multipliers[ct].item()
            diagnostics['trackers'][ct.value] = {
                'violation_ema': self.trackers[ct].violation_ema,
                'consecutive_violations': self.trackers[ct].consecutive_violations,
                'consecutive_satisfactions': self.trackers[ct].consecutive_satisfactions,
                'satisfaction_rate': self.trackers[ct].satisfaction_rate,
            }
            diagnostics['learning_rates'][ct.value] = self.effective_lrs[ct]
        
        return diagnostics
    
    def check_divergence(self) -> Tuple[bool, List[ConstraintType]]:
        """
        Check if any multipliers are approaching divergence.
        
        Returns:
            (has_divergence, list of problematic constraints)
        """
        diverging = []
        warning_threshold = self.config.lambda_max * 0.9
        
        for ct, mult in self.multipliers.items():
            if mult.item() > warning_threshold:
                diverging.append(ct)
        
        return len(diverging) > 0, diverging
    
    def check_collapse(self) -> Tuple[bool, List[ConstraintType]]:
        """
        Check if any multipliers have collapsed to near-zero.
        
        Returns:
            (has_collapse, list of problematic constraints)
        """
        collapsing = []
        collapse_threshold = self.config.lambda_min + 0.01
        
        for ct, mult in self.multipliers.items():
            if mult.item() < collapse_threshold:
                # Only flag if constraint should be active
                tracker = self.trackers[ct]
                if tracker.violation_ema > self.config.violation_threshold:
                    collapsing.append(ct)
        
        return len(collapsing) > 0, collapsing


class SafeLagrangianTrainer:
    """
    Wrapper trainer that integrates the scheduler with C-PPO training.
    
    Provides automatic monitoring and intervention for:
    - Multiplier divergence
    - Multiplier collapse
    - Oscillating constraints
    """
    
    def __init__(
        self,
        scheduler: LagrangeMultiplierScheduler,
        divergence_action: str = 'reduce_lr',
        collapse_action: str = 'increase_lr',
    ):
        self.scheduler = scheduler
        self.divergence_action = divergence_action
        self.collapse_action = collapse_action
        
        self.intervention_count = 0
    
    def post_step_check(self) -> Dict:
        """
        Check for issues after scheduler step and apply interventions.
        
        Returns:
            Dictionary with any interventions applied
        """
        interventions = {'divergence': [], 'collapse': [], 'oscillation': []}
        
        # Check for divergence
        has_divergence, diverging = self.scheduler.check_divergence()
        if has_divergence:
            self._handle_divergence(diverging)
            interventions['divergence'] = [ct.value for ct in diverging]
        
        # Check for collapse
        has_collapse, collapsing = self.scheduler.check_collapse()
        if has_collapse:
            self._handle_collapse(collapsing)
            interventions['collapse'] = [ct.value for ct in collapsing]
        
        if any(interventions.values()):
            self.intervention_count += 1
        
        return interventions
    
    def _handle_divergence(self, diverging: List[ConstraintType]) -> None:
        """Handle multiplier divergence."""
        if self.divergence_action == 'reduce_lr':
            for ct in diverging:
                current_lr = self.scheduler.effective_lrs[ct]
                self.scheduler.effective_lrs[ct] = max(
                    self.scheduler.config.min_lr,
                    current_lr * 0.5
                )
        elif self.divergence_action == 'clamp':
            for ct in diverging:
                self.scheduler.multipliers[ct] = torch.tensor(
                    self.scheduler.config.lambda_max * 0.8,
                    device=self.scheduler.device,
                    requires_grad=False
                )
    
    def _handle_collapse(self, collapsing: List[ConstraintType]) -> None:
        """Handle multiplier collapse."""
        if self.collapse_action == 'increase_lr':
            for ct in collapsing:
                current_lr = self.scheduler.effective_lrs[ct]
                self.scheduler.effective_lrs[ct] = min(
                    self.scheduler.config.max_lr,
                    current_lr * 2.0
                )
        elif self.collapse_action == 'boost':
            for ct in collapsing:
                current = self.scheduler.multipliers[ct].item()
                self.scheduler.multipliers[ct] = torch.tensor(
                    max(current * 1.5, 0.1),
                    device=self.scheduler.device,
                    requires_grad=False
                )


def create_default_scheduler(
    cmdp: TradingCMDP,
    device: str = 'cpu',
) -> LagrangeMultiplierScheduler:
    """Create a scheduler with default configuration."""
    config = LagrangeSchedulerConfig(
        mode=AdaptationMode.ADAPTIVE_LR,
        lambda_max=100.0,
        base_lr=0.01,
        ema_alpha=0.1,
    )
    return LagrangeMultiplierScheduler(cmdp, config, device)
