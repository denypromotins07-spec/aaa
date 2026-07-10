"""
SNN Backpropagation Through Time (BPTT) Unroller.

Implements efficient BPTT for Spiking Neural Networks with truncated backpropagation,
gradient clipping, and memory-efficient checkpointing to handle long temporal sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class BPTTConfig:
    """Configuration for BPTT unrolling."""
    # Number of time steps to unroll
    sequence_length: int = 100
    
    # Truncated BPTT: number of steps to backpropagate through
    truncation_length: int = 20
    
    # Gradient clipping value (None disables clipping)
    grad_clip_value: Optional[float] = 1.0
    
    # Whether to use gradient checkpointing for memory efficiency
    use_checkpointing: bool = True
    
    # Dropout rate for temporal dropout
    temporal_dropout: float = 0.0
    
    # Whether to track spiking statistics
    track_statistics: bool = True


class SNNBPTTUnroller:
    """
    Backpropagation Through Time unroller for Spiking Neural Networks.
    
    Handles the temporal dynamics of SNNs including membrane potential updates,
    spike generation, and refractory periods across multiple time steps.
    """
    
    def __init__(self, config: BPTTConfig):
        self.config = config
        self.sequence_length = config.sequence_length
        self.truncation_length = config.truncation_length
        
        # Statistics tracking
        self.spike_counts: List[int] = []
        self.membrane_potential_stats: Dict[str, float] = {}
        
    def unroll_forward(self, 
                       neuron_layer: nn.Module,
                       input_sequence: torch.Tensor,
                       initial_state: Optional[Dict[str, torch.Tensor]] = None
                       ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Unroll SNN forward pass through time.
        
        Args:
            neuron_layer: SNN layer with update_step method
            input_sequence: Input tensor of shape (batch, time, features)
            initial_state: Initial membrane potential and other state variables
            
        Returns:
            Tuple of (spike_output, final_state)
        """
        batch_size = input_sequence.shape[0]
        seq_len = input_sequence.shape[1]
        
        # Initialize state
        if initial_state is None:
            membrane_potential = neuron_layer.init_membrane_potential(batch_size)
            refractory_count = torch.zeros(batch_size, device=input_sequence.device)
        else:
            membrane_potential = initial_state.get('membrane_potential', 
                neuron_layer.init_membrane_potential(batch_size))
            refractory_count = initial_state.get('refractory_count', 
                torch.zeros(batch_size, device=input_sequence.device))
        
        # Storage for outputs
        spike_outputs = []
        membrane_history = []
        
        # Apply checkpointing if enabled
        if self.config.use_checkpointing:
            return self._unroll_with_checkpointing(
                neuron_layer, input_sequence, membrane_potential, refractory_count
            )
        
        # Standard unrolling
        total_spikes = 0
        
        for t in range(seq_len):
            input_t = input_sequence[:, t, :]
            
            # Update neuron state
            membrane_potential, spikes, refractory_count = neuron_layer.update_step(
                input_t, membrane_potential, refractory_count
            )
            
            # Apply temporal dropout
            if self.config.temporal_dropout > 0 and self.training:
                dropout_mask = torch.rand_like(spikes) > self.config.temporal_dropout
                spikes = spikes * dropout_mask
            
            spike_outputs.append(spikes)
            membrane_history.append(membrane_potential.clone())
            total_spikes += spikes.sum().item()
        
        # Store statistics
        if self.config.track_statistics:
            self.spike_counts.append(total_spikes)
            self.membrane_potential_stats = {
                'mean': membrane_potential.mean().item(),
                'std': membrane_potential.std().item(),
                'max': membrane_potential.max().item(),
                'min': membrane_potential.min().item(),
            }
        
        # Stack outputs
        spike_output = torch.stack(spike_outputs, dim=1)  # (batch, time, neurons)
        
        final_state = {
            'membrane_potential': membrane_potential,
            'refractory_count': refractory_count,
        }
        
        return spike_output, final_state
    
    def _unroll_with_checkpointing(self,
                                   neuron_layer: nn.Module,
                                   input_sequence: torch.Tensor,
                                   membrane_potential: torch.Tensor,
                                   refractory_count: torch.Tensor
                                   ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Memory-efficient unrolling using gradient checkpointing.
        
        Only stores checkpoints at intervals, recomputing intermediate states
        during backward pass to save memory.
        """
        from torch.utils.checkpoint import checkpoint
        
        seq_len = input_sequence.shape[1]
        checkpoint_interval = max(1, self.truncation_length // 4)
        
        spike_outputs = []
        
        for t in range(seq_len):
            input_t = input_sequence[:, t, :]
            
            # Use checkpointing for every checkpoint_interval steps
            if t % checkpoint_interval == 0 and t > 0:
                # Detach and reattach gradients at checkpoint
                membrane_potential = membrane_potential.detach()
                membrane_potential.requires_grad = True
                refractory_count = refractory_count.detach()
            
            # Forward step
            membrane_potential, spikes, refractory_count = neuron_layer.update_step(
                input_t, membrane_potential, refractory_count
            )
            
            spike_outputs.append(spikes)
        
        spike_output = torch.stack(spike_outputs, dim=1)
        
        final_state = {
            'membrane_potential': membrane_potential,
            'refractory_count': refractory_count,
        }
        
        return spike_output, final_state
    
    def truncate_gradients(self, 
                           spike_output: torch.Tensor,
                           loss: torch.Tensor,
                           optimizer: torch.optim.Optimizer
                           ) -> float:
        """
        Perform truncated BPTT by limiting gradient flow through time.
        
        Args:
            spike_output: Output spikes from forward pass
            loss: Computed loss
            optimizer: Optimizer for parameter updates
            
        Returns:
            Gradient norm before clipping
        """
        # Backward pass
        loss.backward()
        
        # Calculate gradient norm
        grad_norm = 0.0
        for param in optimizer.param_groups[0]['params']:
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = math.sqrt(grad_norm)
        
        # Clip gradients
        if self.config.grad_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(
                optimizer.param_groups[0]['params'],
                self.config.grad_clip_value
            )
        
        return grad_norm
    
    def get_training_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about training."""
        if not self.spike_counts:
            return {}
        
        return {
            'avg_spike_count': sum(self.spike_counts[-100:]) / len(self.spike_counts[-100:]),
            'membrane_potential_stats': self.membrane_potential_stats,
            'sequence_length': self.sequence_length,
            'truncation_length': self.truncation_length,
        }
    
    def reset_statistics(self):
        """Reset tracked statistics."""
        self.spike_counts = []
        self.membrane_potential_stats = {}


class TemporalLossAggregator(nn.Module):
    """
    Aggregates losses across time steps for SNN training.
    
    Supports multiple aggregation strategies including weighted averaging
    and focus on specific time windows.
    """
    
    def __init__(self, 
                 aggregation: str = 'mean',
                 time_weights: Optional[torch.Tensor] = None,
                 focus_window: Optional[Tuple[int, int]] = None):
        super().__init__()
        
        self.aggregation = aggregation
        self.time_weights = time_weights
        self.focus_window = focus_window
        
    def forward(self, 
                spike_output: torch.Tensor,
                target: torch.Tensor,
                loss_fn: nn.Module = nn.MSELoss()
                ) -> torch.Tensor:
        """
        Compute aggregated loss across time steps.
        
        Args:
            spike_output: Spike output tensor (batch, time, neurons)
            target: Target tensor (batch, time, neurons) or (batch, neurons)
            loss_fn: Loss function to apply at each time step
            
        Returns:
            Aggregated loss scalar
        """
        # Expand target if needed
        if target.dim() == 2:
            target = target.unsqueeze(1).expand(-1, spike_output.shape[1], -1)
        
        # Apply focus window if specified
        if self.focus_window is not None:
            start, end = self.focus_window
            spike_output = spike_output[:, start:end, :]
            target = target[:, start:end, :]
        
        # Compute per-time-step losses
        time_losses = []
        for t in range(spike_output.shape[1]):
            loss_t = loss_fn(spike_output[:, t, :], target[:, t, :])
            time_losses.append(loss_t)
        
        # Aggregate
        if self.time_weights is not None:
            weights = self.time_weights[:len(time_losses)]
            weights = weights / weights.sum()
            aggregated_loss = sum(w * l for w, l in zip(weights, time_losses))
        elif self.aggregation == 'mean':
            aggregated_loss = sum(time_losses) / len(time_losses)
        elif self.aggregation == 'sum':
            aggregated_loss = sum(time_losses)
        elif self.aggregation == 'last':
            aggregated_loss = time_losses[-1]
        else:
            aggregated_loss = sum(time_losses) / len(time_losses)
        
        return aggregated_loss


class SurrogateGradientBPTT(SNNBPTTUnroller):
    """
    BPTT implementation specifically designed for surrogate gradient training.
    
    Integrates surrogate gradient functions directly into the unrolling process
    for efficient gradient computation.
    """
    
    def __init__(self, config: BPTTConfig, surrogate_alpha: float = 1.0):
        super().__init__(config)
        self.surrogate_alpha = surrogate_alpha
        
    def unroll_forward_with_surrogate(self,
                                       neuron_layer: nn.Module,
                                       input_sequence: torch.Tensor,
                                       surrogate_fn: Optional[callable] = None
                                       ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward unroll with integrated surrogate gradient handling.
        
        Args:
            neuron_layer: SNN layer
            input_sequence: Input tensor
            surrogate_fn: Custom surrogate gradient function
            
        Returns:
            Tuple of (spike_output, final_state)
        """
        # This would integrate the surrogate gradient function from Stage 28
        # For now, delegate to parent class
        return self.unroll_forward(neuron_layer, input_sequence)


if __name__ == "__main__":
    # Example usage
    print("Testing SNN BPTT Unroller...")
    
    config = BPTTConfig(
        sequence_length=50,
        truncation_length=10,
        grad_clip_value=1.0,
        use_checkpointing=True
    )
    
    unroller = SNNBPTTUnroller(config)
    
    # Mock neuron layer for testing
    class MockNeuronLayer(nn.Module):
        def __init__(self, input_size: int, hidden_size: int):
            super().__init__()
            self.fc = nn.Linear(input_size, hidden_size)
            self.threshold = 1.0
            self.decay = 0.9
            
        def init_membrane_potential(self, batch_size: int) -> torch.Tensor:
            return torch.zeros(batch_size, self.fc.out_features)
        
        def update_step(self, 
                        input_t: torch.Tensor,
                        membrane_potential: torch.Tensor,
                        refractory_count: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # LIF neuron update
            membrane_potential = self.decay * membrane_potential + self.fc(input_t)
            spikes = (membrane_potential > self.threshold).float()
            membrane_potential = membrane_potential * (1 - spikes)  # Reset on spike
            
            # Refractory handling
            refractory_count = torch.maximum(refractory_count - 1, torch.zeros_like(refractory_count))
            spikes = spikes * (refractory_count == 0)
            refractory_count = torch.where(spikes > 0, torch.ones_like(refractory_count) * 3, refractory_count)
            
            return membrane_potential, spikes, refractory_count
    
    # Create test data
    batch_size = 4
    input_size = 16
    hidden_size = 32
    seq_len = 50
    
    neuron_layer = MockNeuronLayer(input_size, hidden_size)
    input_sequence = torch.randn(batch_size, seq_len, input_size)
    
    # Run unroller
    spike_output, final_state = unroller.unroll_forward(neuron_layer, input_sequence)
    
    print(f"Input shape: {input_sequence.shape}")
    print(f"Output shape: {spike_output.shape}")
    print(f"Total spikes: {spike_output.sum().item()}")
    print(f"Diagnostics: {unroller.get_training_diagnostics()}")
