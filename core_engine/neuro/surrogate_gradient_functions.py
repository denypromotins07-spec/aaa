"""
Surrogate Gradient Functions for Spiking Neural Network Backpropagation.

The Heaviside step function used in SNNs is non-differentiable (gradient = 0 everywhere
except at threshold where it's undefined). Surrogate gradients provide a smooth approximation
that allows gradients to flow during backpropagation through time (BPTT).

This module implements multiple surrogate gradient functions with adaptive alpha scheduling
to prevent vanishing gradients during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
import math


@dataclass
class SurrogateGradientConfig:
    """Configuration for surrogate gradient functions."""
    # Type of surrogate gradient function
    function_type: str = "fast_sigmoid"  # fast_sigmoid, multi_gaussian, triangular, piecewise_linear
    
    # Alpha parameter controls the width of the gradient window
    # Larger alpha = wider window = more gradient flow but less biological accuracy
    alpha: float = 1.0
    
    # Minimum alpha to prevent complete gradient vanishing
    min_alpha: float = 0.1
    
    # Maximum alpha to prevent excessive smoothing
    max_alpha: float = 10.0
    
    # Whether to use adaptive alpha scheduling
    adaptive_alpha: bool = True
    
    # Warmup epochs for alpha scheduling
    warmup_epochs: int = 10
    
    # Decay rate for alpha after warmup
    decay_rate: float = 0.95


class SurrogateGradientFunction(torch.autograd.Function):
    """
    Custom autograd function implementing surrogate gradient descent for SNNs.
    
    The forward pass uses the Heaviside step function:
        spike = 1 if membrane_potential > threshold else 0
    
    The backward pass uses a smooth surrogate derivative that allows gradients to flow.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float = 1.0, alpha: float = 1.0, 
                function_type: str = "fast_sigmoid") -> torch.Tensor:
        """
        Forward pass: Heaviside step function.
        
        Args:
            ctx: Context object to save tensors for backward pass
            x: Membrane potential tensor
            threshold: Spike threshold
            alpha: Width parameter for surrogate gradient
            function_type: Type of surrogate gradient function
            
        Returns:
            Binary spike tensor (0 or 1)
        """
        # Save context for backward pass
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        ctx.alpha = alpha
        ctx.function_type = function_type
        
        # Heaviside step function: spike if x > threshold
        spikes = (x > threshold).to(dtype=x.dtype)
        return spikes
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """
        Backward pass: Surrogate gradient computation.
        
        Uses the straight-through estimator (STE) variant where gradients flow
        through the surrogate derivative instead of the actual step function derivative.
        
        Args:
            ctx: Context object with saved tensors
            grad_output: Gradient from upstream layers
            
        Returns:
            Gradient w.r.t. input membrane potential
        """
        x, = ctx.saved_tensors
        threshold = ctx.threshold
        alpha = ctx.alpha
        function_type = ctx.function_type
        
        # Compute surrogate derivative based on function type
        if function_type == "fast_sigmoid":
            grad_approx = _fast_sigmoid_derivative(x, threshold, alpha)
        elif function_type == "multi_gaussian":
            grad_approx = _multi_gaussian_derivative(x, threshold, alpha)
        elif function_type == "triangular":
            grad_approx = _triangular_derivative(x, threshold, alpha)
        elif function_type == "piecewise_linear":
            grad_approx = _piecewise_linear_derivative(x, threshold, alpha)
        else:
            # Default to fast sigmoid
            grad_approx = _fast_sigmoid_derivative(x, threshold, alpha)
        
        # Chain rule: gradient flows through surrogate derivative
        grad_input = grad_output * grad_approx
        
        return grad_input, None, None, None


def _fast_sigmoid_derivative(x: torch.Tensor, threshold: float = 1.0, 
                              alpha: float = 1.0) -> torch.Tensor:
    """
    Fast sigmoid surrogate gradient.
    
    Derivative: alpha / (2 * (|x - threshold| * alpha + 1)^2)
    
    This is computationally efficient and provides reasonable gradient flow.
    The gradient peaks at the threshold and decays quadratically.
    """
    diff = x - threshold
    return alpha / (2.0 * (torch.abs(diff) * alpha + 1.0) ** 2)


def _multi_gaussian_derivative(x: torch.Tensor, threshold: float = 1.0,
                                alpha: float = 1.0) -> torch.Tensor:
    """
    Multi-Gaussian surrogate gradient with multiple peaks.
    
    Uses a sum of Gaussians centered around the threshold to provide
    gradient flow over a wider range while maintaining peak sensitivity.
    
    Derivative: sum_i(w_i * exp(-(x - threshold - mu_i)^2 / (2 * sigma_i^2)))
    
    This helps prevent vanishing gradients by providing multiple gradient peaks.
    """
    # Three Gaussian components
    sigma = 1.0 / alpha
    
    # Center Gaussian at threshold
    g1 = torch.exp(-((x - threshold) ** 2) / (2 * sigma ** 2))
    
    # Side Gaussians for extended gradient flow
    g2 = 0.5 * torch.exp(-((x - threshold - 0.5) ** 2) / (2 * sigma ** 2))
    g3 = 0.5 * torch.exp(-((x - threshold + 0.5) ** 2) / (2 * sigma ** 2))
    
    return (g1 + g2 + g3) / sigma


def _triangular_derivative(x: torch.Tensor, threshold: float = 1.0,
                           alpha: float = 1.0) -> torch.Tensor:
    """
    Triangular surrogate gradient.
    
    Derivative: max(0, 1 - alpha * |x - threshold|)
    
    Provides uniform gradient within a window around the threshold,
    then abruptly cuts off. Simple but effective.
    """
    diff = torch.abs(x - threshold)
    window_width = 1.0 / alpha
    return torch.clamp(1.0 - diff / window_width, min=0.0)


def _piecewise_linear_derivative(x: torch.Tensor, threshold: float = 1.0,
                                  alpha: float = 1.0) -> torch.Tensor:
    """
    Piecewise linear surrogate gradient with different slopes.
    
    Provides steeper gradient near threshold and gentler slope further away.
    """
    diff = x - threshold
    abs_diff = torch.abs(diff)
    
    # Inner region: steep gradient
    inner_mask = abs_diff <= 0.5 / alpha
    inner_grad = alpha * (1.0 - 2.0 * abs_diff)
    
    # Outer region: gentle gradient tail
    outer_mask = (abs_diff > 0.5 / alpha) & (abs_diff <= 1.5 / alpha)
    outer_grad = alpha * 0.5 * (1.5 / alpha - abs_diff)
    
    # Combine regions
    grad = torch.zeros_like(x)
    grad[inner_mask] = inner_grad[inner_mask]
    grad[outer_mask] = outer_grad[outer_mask]
    
    return grad


class AdaptiveAlphaScheduler:
    """
    Adaptive alpha scheduler to prevent vanishing gradients.
    
    During early training, uses larger alpha for wider gradient windows.
    Gradually reduces alpha to improve biological accuracy while maintaining
    sufficient gradient flow.
    """
    
    def __init__(self, config: SurrogateGradientConfig):
        self.config = config
        self.current_alpha = config.alpha
        self.epoch = 0
        self.gradient_magnitude_history = []
        self.warmup_complete = False
        
    def step(self, gradient_magnitude: Optional[float] = None) -> float:
        """
        Update alpha based on current epoch and gradient statistics.
        
        Args:
            gradient_magnitude: Optional observed gradient magnitude for adaptive adjustment
            
        Returns:
            Updated alpha value
        """
        self.epoch += 1
        
        # Track gradient magnitude if provided
        if gradient_magnitude is not None:
            self.gradient_magnitude_history.append(gradient_magnitude)
            # Keep only recent history
            if len(self.gradient_magnitude_history) > 100:
                self.gradient_magnitude_history.pop(0)
        
        # Warmup phase: gradually increase alpha
        if self.epoch <= self.config.warmup_epochs:
            warmup_progress = self.epoch / self.config.warmup_epochs
            self.current_alpha = self.config.min_alpha + warmup_progress * (self.config.alpha - self.config.min_alpha)
        
        # Post-warmup: adaptive adjustment based on gradient magnitude
        elif self.config.adaptive_alpha and gradient_magnitude is not None:
            avg_gradient = sum(self.gradient_magnitude_history) / len(self.gradient_magnitude_history)
            
            # If gradients are too small, increase alpha
            if avg_gradient < 0.01:
                self.current_alpha = min(self.current_alpha * 1.1, self.config.max_alpha)
            # If gradients are healthy, slowly decay alpha
            elif avg_gradient > 0.1:
                self.current_alpha = max(self.current_alpha * self.config.decay_rate, self.config.min_alpha)
        
        # Clamp to valid range
        self.current_alpha = torch.clamp(
            torch.tensor(self.current_alpha),
            self.config.min_alpha,
            self.config.max_alpha
        ).item()
        
        return self.current_alpha
    
    def get_alpha(self) -> float:
        """Get current alpha value."""
        return self.current_alpha
    
    def reset(self):
        """Reset scheduler state."""
        self.epoch = 0
        self.current_alpha = self.config.alpha
        self.gradient_magnitude_history = []
        self.warmup_complete = False


class LearnableSurrogateGradient(nn.Module):
    """
    Surrogate gradient with learnable alpha parameter per neuron.
    
    Allows the network to learn optimal gradient widths for different neurons,
    adapting to their specific activation patterns.
    """
    
    def __init__(self, num_neurons: int, initial_alpha: float = 1.0,
                 min_alpha: float = 0.1, max_alpha: float = 10.0):
        super().__init__()
        
        # Learnable alpha parameter (in log space for stability)
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(initial_alpha)) * torch.ones(num_neurons))
        
        # Constraints
        self.min_log_alpha = math.log(min_alpha)
        self.max_log_alpha = math.log(max_alpha)
        
        # Regularization for alpha smoothness
        self.alpha_smoothness_weight = 0.01
    
    def forward(self, x: torch.Tensor, threshold: float = 1.0, 
                function_type: str = "fast_sigmoid") -> torch.Tensor:
        # Clamp alpha to valid range
        clamped_log_alpha = torch.clamp(self.log_alpha, self.min_log_alpha, self.max_log_alpha)
        alpha = torch.exp(clamped_log_alpha)
        
        # Broadcast alpha if needed
        if x.dim() > 1 and alpha.dim() < x.dim():
            alpha = alpha.view(-1, *[1] * (x.dim() - 1))
        
        return SurrogateGradientFunction.apply(x, threshold, alpha.item(), function_type)
    
    def get_alpha_smoothness_loss(self) -> torch.Tensor:
        """
        Regularization loss to encourage smooth alpha values across neurons.
        """
        if self.log_alpha.numel() < 2:
            return torch.tensor(0.0)
        
        alpha_diff = torch.diff(self.log_alpha)
        return self.alpha_smoothness_weight * torch.mean(alpha_diff ** 2)


def create_surrogate_gradient(config: SurrogateGradientConfig) -> Callable:
    """
    Factory function to create configured surrogate gradient function.
    
    Args:
        config: Configuration for the surrogate gradient
        
    Returns:
        Configured surrogate gradient function
    """
    def surrogate_fn(x: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        return SurrogateGradientFunction.apply(
            x, threshold, config.alpha, config.function_type
        )
    return surrogate_fn


# Convenience functions for common use cases
def fast_sigmoid_spike(x: torch.Tensor, threshold: float = 1.0, alpha: float = 1.0) -> torch.Tensor:
    """Fast sigmoid surrogate gradient spiking function."""
    return SurrogateGradientFunction.apply(x, threshold, alpha, "fast_sigmoid")


def multi_gaussian_spike(x: torch.Tensor, threshold: float = 1.0, alpha: float = 1.0) -> torch.Tensor:
    """Multi-Gaussian surrogate gradient spiking function."""
    return SurrogateGradientFunction.apply(x, threshold, alpha, "multi_gaussian")


def triangular_spike(x: torch.Tensor, threshold: float = 1.0, alpha: float = 1.0) -> torch.Tensor:
    """Triangular surrogate gradient spiking function."""
    return SurrogateGradientFunction.apply(x, threshold, alpha, "triangular")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Surrogate Gradient Functions...")
    
    # Create test tensor
    x = torch.linspace(-2, 4, 1000, requires_grad=True)
    threshold = 1.0
    alpha = 2.0
    
    # Test fast sigmoid
    spikes = fast_sigmoid_spike(x, threshold, alpha)
    loss = spikes.sum()
    loss.backward()
    
    print(f"Input range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Spikes generated: {spikes.sum().item()}")
    print(f"Gradient magnitude: {x.grad.abs().mean().item():.6f}")
    
    # Test adaptive scheduler
    config = SurrogateGradientConfig(adaptive_alpha=True, warmup_epochs=5)
    scheduler = AdaptiveAlphaScheduler(config)
    
    print("\nAdaptive Alpha Schedule:")
    for epoch in range(15):
        grad_mag = 0.05 if epoch < 5 else 0.15  # Simulated gradient magnitudes
        new_alpha = scheduler.step(grad_mag)
        print(f"  Epoch {epoch}: alpha = {new_alpha:.4f}")
