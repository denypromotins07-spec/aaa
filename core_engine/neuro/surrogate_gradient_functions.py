"""
Surrogate Gradient Functions for SNN Backpropagation

The Heaviside step function used for spike generation is non-differentiable.
This module implements surrogate gradient functions that approximate the 
derivative to enable backpropagation through spiking neural networks.

Key insight: During forward pass, use hard threshold (Heaviside).
During backward pass, use smooth approximation for gradient flow.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import math


class SurrogateGradientFunction(torch.autograd.Function):
    """
    Custom autograd function with surrogate gradient for spiking neurons.
    
    Forward pass: Heaviside step function (spike if v > threshold)
    Backward pass: Smooth surrogate derivative for gradient flow
    """
    
    @staticmethod
    def forward(ctx, membrane_potential: torch.Tensor, 
                threshold: float = 1.0,
                surrogate_type: str = 'fast_sigmoid',
                alpha: float = 1.0) -> torch.Tensor:
        """
        Forward pass: Generate spikes using Heaviside step function.
        
        Args:
            membrane_potential: Membrane potential tensor
            threshold: Spike threshold (default 1.0)
            surrogate_type: Type of surrogate gradient ('fast_sigmoid', 'multi_gaussian', 
                           'piecewise_linear', 'arctan')
            alpha: Width parameter for surrogate gradient (larger = wider gradient window)
        
        Returns:
            Spike tensor (0 or 1)
        """
        ctx.save_for_backward(membrane_potential)
        ctx.threshold = threshold
        ctx.surrogate_type = surrogate_type
        ctx.alpha = alpha
        
        # Heaviside step function: spike if V > threshold
        spikes = (membrane_potential > threshold).float()
        return spikes
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """
        Backward pass: Compute surrogate gradient.
        
        The gradient is approximated using a smooth function centered at threshold.
        """
        membrane_potential, = ctx.saved_tensors
        threshold = ctx.threshold
        surrogate_type = ctx.surrogate_type
        alpha = ctx.alpha
        
        # Compute surrogate derivative based on type
        if surrogate_type == 'fast_sigmoid':
            # Fast sigmoid: σ(x) = 1 / (1 + exp(-αx))
            # Derivative: α * σ(x) * (1 - σ(x))
            x = (membrane_potential - threshold) * alpha
            sigmoid_x = torch.sigmoid(x)
            grad_input = alpha * sigmoid_x * (1 - sigmoid_x)
            
        elif surrogate_type == 'multi_gaussian':
            # Multi-Gaussian: Sum of Gaussians for better gradient coverage
            # Prevents vanishing gradients by having multiple peaks
            x = membrane_potential - threshold
            sigma = 1.0 / alpha
            
            # Three Gaussians centered at threshold and ±offset
            offset = 0.5
            g1 = torch.exp(-0.5 * (x / sigma) ** 2)
            g2 = 0.5 * torch.exp(-0.5 * ((x - offset) / sigma) ** 2)
            g3 = 0.5 * torch.exp(-0.5 * ((x + offset) / sigma) ** 2)
            grad_input = (g1 + g2 + g3) * alpha
            
        elif surrogate_type == 'piecewise_linear':
            # Piecewise linear (triangle function)
            # Non-zero gradient in [threshold - 1/α, threshold + 1/α]
            x = membrane_potential - threshold
            width = 1.0 / alpha
            grad_input = torch.clamp(1.0 - torch.abs(x) / width, 0.0, 1.0) * alpha
            
        elif surrogate_type == 'arctan':
            # Arctangent surrogate: derivative of (2/π) * arctan(παx/2)
            x = (membrane_potential - threshold) * alpha
            grad_input = alpha / (1.0 + (math.pi * alpha * x / 2) ** 2)
            
        elif surrogate_type == 'rectangular':
            # Simple rectangular window (supervised learning style)
            x = membrane_potential - threshold
            width = 1.0 / alpha
            grad_input = (torch.abs(x) < width).float() * alpha
            
        else:
            raise ValueError(f"Unknown surrogate type: {surrogate_type}")
        
        # Chain rule: multiply by upstream gradient
        return grad_input * grad_output, None, None, None


def spike_function(membrane_potential: torch.Tensor, 
                   threshold: float = 1.0,
                   surrogate_type: str = 'fast_sigmoid',
                   alpha: float = 1.0,
                   training: bool = True) -> torch.Tensor:
    """
    Apply spiking function with surrogate gradient support.
    
    Args:
        membrane_potential: Input membrane potential
        threshold: Spike threshold
        surrogate_type: Type of surrogate gradient
        alpha: Surrogate gradient width parameter
        training: If True, use surrogate gradient; if False, hard threshold only
    
    Returns:
        Spike tensor
    """
    if training:
        return SurrogateGradientFunction.apply(
            membrane_potential, threshold, surrogate_type, alpha
        )
    else:
        # Inference mode: just use hard threshold
        return (membrane_potential > threshold).float()


class AdaptiveAlphaScheduler:
    """
    Adaptive alpha scheduler to prevent vanishing gradients during BPTT.
    
    Early in training: Use wider alpha for better gradient flow
    Later in training: Narrow alpha for more precise spike timing
    """
    
    def __init__(self, initial_alpha: float = 0.5, 
                 min_alpha: float = 0.1,
                 max_alpha: float = 10.0,
                 warmup_epochs: int = 10,
                 decay_rate: float = 0.95):
        """
        Args:
            initial_alpha: Starting alpha value
            min_alpha: Minimum allowed alpha
            max_alpha: Maximum allowed alpha  
            warmup_epochs: Epochs to gradually increase from initial
            decay_rate: Per-epoch decay after warmup
        """
        self.initial_alpha = initial_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.current_epoch = 0
        self.current_alpha = initial_alpha
        
    def step(self, epoch: Optional[int] = None) -> float:
        """
        Update alpha based on training progress.
        
        Args:
            epoch: Current epoch (if None, increment internal counter)
        
        Returns:
            Updated alpha value
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            progress = self.current_epoch / self.warmup_epochs
            self.current_alpha = self.initial_alpha + (self.max_alpha - self.initial_alpha) * progress
        else:
            # Exponential decay after warmup
            decay_epochs = self.current_epoch - self.warmup_epochs
            self.current_alpha = max(
                self.min_alpha,
                self.max_alpha * (self.decay_rate ** decay_epochs)
            )
        
        return self.current_alpha
    
    def get_alpha(self) -> float:
        """Get current alpha value."""
        return self.current_alpha
    
    def reset(self):
        """Reset scheduler state."""
        self.current_epoch = 0
        self.current_alpha = self.initial_alpha


class LearnableSurrogateGradient(nn.Module):
    """
    Learnable surrogate gradient parameters.
    
    Instead of fixed alpha, learn optimal surrogate gradient shape
    during training for each neuron layer.
    """
    
    def __init__(self, num_neurons: int, 
                 initial_alpha: float = 1.0,
                 alpha_min: float = 0.01,
                 alpha_max: float = 100.0):
        super().__init__()
        
        # Learnable alpha per neuron (in log space for stability)
        self.log_alpha = nn.Parameter(
            torch.full((num_neurons,), math.log(initial_alpha))
        )
        self.alpha_min = math.log(alpha_min)
        self.alpha_max = math.log(alpha_max)
        
    def forward(self, membrane_potential: torch.Tensor) -> torch.Tensor:
        """
        Compute surrogate gradient with learned parameters.
        
        Returns:
            Gradient mask for backpropagation
        """
        # Clamp alpha to valid range
        log_alpha = torch.clamp(self.log_alpha, self.alpha_min, self.alpha_max)
        alpha = torch.exp(log_alpha)
        
        # Broadcast alpha to match membrane potential shape
        while alpha.dim() < membrane_potential.dim():
            alpha = alpha.unsqueeze(-1)
        
        # Fast sigmoid surrogate with learned alpha
        x = membrane_potential * alpha
        sigmoid_x = torch.sigmoid(x)
        grad_mask = alpha * sigmoid_x * (1 - sigmoid_x)
        
        return grad_mask
    
    def get_alpha(self) -> torch.Tensor:
        """Get current alpha values."""
        return torch.exp(torch.clamp(self.log_alpha, self.alpha_min, self.alpha_max))


def compute_surrogate_gradient(membrane_potential: torch.Tensor,
                               threshold: float = 1.0,
                               alpha: float = 1.0,
                               method: str = 'fast_sigmoid') -> torch.Tensor:
    """
    Standalone function to compute surrogate gradient for debugging/analysis.
    
    Args:
        membrane_potential: Membrane potential tensor
        threshold: Spike threshold
        alpha: Gradient width parameter
        method: Gradient computation method
    
    Returns:
        Surrogate gradient tensor
    """
    x = membrane_potential - threshold
    
    if method == 'fast_sigmoid':
        sigmoid_x = torch.sigmoid(alpha * x)
        return alpha * sigmoid_x * (1 - sigmoid_x)
    
    elif method == 'multi_gaussian':
        sigma = 1.0 / alpha
        g1 = torch.exp(-0.5 * (x / sigma) ** 2)
        g2 = 0.5 * torch.exp(-0.5 * ((x - 0.5) / sigma) ** 2)
        g3 = 0.5 * torch.exp(-0.5 * ((x + 0.5) / sigma) ** 2)
        return alpha * (g1 + g2 + g3)
    
    elif method == 'piecewise_linear':
        width = 1.0 / alpha
        return torch.clamp(1.0 - torch.abs(x) / width, 0.0, 1.0) * alpha
    
    elif method == 'arctan':
        return alpha / (1.0 + (math.pi * alpha * x / 2) ** 2)
    
    else:
        raise ValueError(f"Unknown method: {method}")


# Example usage and testing
if __name__ == "__main__":
    # Test surrogate gradient computation
    test_voltage = torch.linspace(-2, 2, 1000)
    
    methods = ['fast_sigmoid', 'multi_gaussian', 'piecewise_linear', 'arctan']
    
    print("Surrogate Gradient Comparison:")
    print("-" * 60)
    
    for method in methods:
        grad = compute_surrogate_gradient(test_voltage, alpha=2.0, method=method)
        print(f"{method:20s}: max={grad.max():.4f}, mean={grad.mean():.4f}, "
              f"nonzero={(grad > 0.01).sum().item()}")
    
    # Test adaptive scheduler
    print("\nAdaptive Alpha Scheduler:")
    scheduler = AdaptiveAlphaScheduler(
        initial_alpha=0.5,
        warmup_epochs=5,
        decay_rate=0.9
    )
    
    for epoch in range(15):
        alpha = scheduler.step(epoch)
        print(f"Epoch {epoch:2d}: alpha = {alpha:.4f}")
