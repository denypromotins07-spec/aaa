"""
SNN Backpropagation Through Time (BPTT) Unroller

Efficiently unrolls spiking neural networks through time for gradient computation.
Handles memory-efficient BPTT by checkpointing and recomputation strategies
to avoid exhausting GPU VRAM during deep SNN training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import math


@dataclass
class BPTTConfig:
    """Configuration for SNN BPTT unrolling."""
    timesteps: int = 100  # Number of timesteps to unroll
    chunk_size: int = 10  # Chunk size for memory-efficient BPTT
    use_checkpointing: bool = True  # Enable activation checkpointing
    truncate_gradient: bool = True  # Truncate gradients beyond certain timesteps
    truncate_length: int = 50  # Max timesteps for gradient flow
    store_spike_times: bool = True  # Store precise spike times for analysis


class SNNBPTTUnroller:
    """
    Memory-efficient BPTT unroller for Spiking Neural Networks.
    
    Key features:
    - Chunked unrolling to reduce memory footprint
    - Activation checkpointing for very deep unrolls
    - Gradient truncation to prevent vanishing/exploding gradients
    - Support for various surrogate gradient functions
    """
    
    def __init__(self, snn_module: nn.Module, config: BPTTConfig):
        """
        Args:
            snn_module: The SNN module to unroll
            config: BPTT configuration
        """
        self.snn = snn_module
        self.config = config
        self._spike_history: List[Dict[str, torch.Tensor]] = []
        
    def unroll(self, inputs: torch.Tensor, 
               target: Optional[torch.Tensor] = None,
               return_states: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Unroll SNN through time and compute output.
        
        Args:
            inputs: Input tensor of shape (batch, timesteps, features)
                   or (batch, features) for constant input
            target: Optional target for loss computation
            return_states: If True, return final hidden states
        
        Returns:
            outputs: Output tensor (sum/last of timestep outputs)
            loss: Optional loss value
        """
        batch_size = inputs.shape[0]
        
        # Handle input shape
        if inputs.dim() == 2:
            # Constant input across timesteps
            inputs = inputs.unsqueeze(1).expand(-1, self.config.timesteps, -1)
        
        actual_timesteps = inputs.shape[1]
        
        # Initialize state storage
        all_outputs = []
        all_spikes = []
        
        # Get initial hidden state from SNN
        hidden_state = self.snn.init_hidden(batch_size)
        
        # Determine unroll strategy
        if self.config.use_checkpointing and actual_timesteps > self.config.chunk_size:
            outputs, spikes = self._unroll_with_checkpointing(inputs, hidden_state)
        else:
            outputs, spikes = self._unroll_sequential(inputs, hidden_state)
        
        # Aggregate outputs (sum over timesteps for classification)
        if isinstance(outputs, list):
            output = sum(outputs) / len(outputs)
        else:
            output = outputs
        
        # Compute loss if target provided
        loss = None
        if target is not None:
            loss = self._compute_loss(output, target)
        
        if return_states:
            return output, loss, hidden_state
        return output, loss
    
    def _unroll_sequential(self, inputs: torch.Tensor, 
                           hidden_state: dict) -> Tuple[List, List]:
        """Simple sequential unrolling (no checkpointing)."""
        outputs = []
        spikes = []
        
        for t in range(inputs.shape[1]):
            input_t = inputs[:, t]
            
            # Forward step
            output_t, hidden_state, spike_info = self.snn.step(input_t, hidden_state)
            
            outputs.append(output_t)
            if self.config.store_spike_times:
                spikes.append(spike_info)
            
            # Gradient truncation
            if self.config.truncate_gradient and t >= self.config.truncate_length:
                hidden_state = self._detach_hidden(hidden_state)
        
        return outputs, spikes
    
    def _unroll_with_checkpointing(self, inputs: torch.Tensor,
                                    hidden_state: dict) -> Tuple[List, List]:
        """Memory-efficient unrolling with activation checkpointing."""
        outputs = []
        spikes = []
        
        timesteps = inputs.shape[1]
        chunk_size = self.config.chunk_size
        
        for chunk_start in range(0, timesteps, chunk_size):
            chunk_end = min(chunk_start + chunk_size, timesteps)
            
            # Use torch's checkpointing for this chunk
            chunk_outputs, chunk_spikes, hidden_state = self._checkpointed_chunk(
                inputs[:, chunk_start:chunk_end],
                hidden_state
            )
            
            outputs.extend(chunk_outputs)
            spikes.extend(chunk_spikes)
            
            # Detach hidden state between chunks for memory efficiency
            if chunk_end < timesteps and self.config.truncate_gradient:
                hidden_state = self._detach_hidden(hidden_state)
        
        return outputs, spikes
    
    def _checkpointed_chunk(self, chunk_inputs: torch.Tensor,
                            hidden_state: dict) -> Tuple[List, List, dict]:
        """Process a chunk with activation checkpointing."""
        from torch.utils.checkpoint import checkpoint
        
        def step_function(h_state, inputs):
            outputs = []
            spikes = []
            for t in range(inputs.shape[1]):
                out, h_state, spike_info = self.snn.step(inputs[:, t], h_state)
                outputs.append(out)
                spikes.append(spike_info)
            return outputs, spikes, h_state
        
        # Checkpoint the chunk computation
        result = checkpoint(step_function, hidden_state, chunk_inputs, use_reentrant=False)
        return result
    
    def _detach_hidden(self, hidden_state: dict) -> dict:
        """Detach hidden state for gradient truncation."""
        return {k: v.detach() if isinstance(v, torch.Tensor) else v 
                for k, v in hidden_state.items()}
    
    def _compute_loss(self, output: torch.Tensor, 
                      target: torch.Tensor) -> torch.Tensor:
        """Compute loss based on output type."""
        if output.dim() == 1:
            # Binary classification
            return F.binary_cross_entropy_with_logits(output, target.float())
        elif output.dim() == 2:
            # Multi-class classification
            return F.cross_entropy(output, target)
        else:
            # Regression or other
            return F.mse_loss(output, target.float())


class SpikingRNN(nn.Module):
    """
    Spiking Recurrent Neural Network layer.
    
    Implements a recurrent layer with LIF neurons that can be unrolled
    through time for BPTT training.
    """
    
    def __init__(self, input_size: int, hidden_size: int, 
                 threshold: float = 1.0,
                 decay: float = 0.9,
                 surrogate_type: str = 'fast_sigmoid',
                 alpha: float = 1.0):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.decay = decay
        self.surrogate_type = surrogate_type
        self.alpha = alpha
        
        # Recurrent weights
        self.input_proj = nn.Linear(input_size, hidden_size, bias=True)
        self.recurrent_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # Surrogate gradient function
        from surrogate_gradient_functions import SurrogateGradientFunction
        self.spike_fn = SurrogateGradientFunction.apply
    
    def init_hidden(self, batch_size: int) -> dict:
        """Initialize hidden state."""
        device = next(self.parameters()).device
        return {
            'membrane': torch.zeros(batch_size, self.hidden_size, device=device),
            'spike': torch.zeros(batch_size, self.hidden_size, device=device),
            'refractory': torch.zeros(batch_size, self.hidden_size, device=device),
        }
    
    def step(self, input_t: torch.Tensor, 
             hidden_state: dict) -> Tuple[torch.Tensor, dict, dict]:
        """
        Single timestep forward pass.
        
        Args:
            input_t: Input at current timestep
            hidden_state: Previous hidden state
        
        Returns:
            output: Output at current timestep
            new_hidden: Updated hidden state
            spike_info: Information about spikes generated
        """
        membrane = hidden_state['membrane']
        spike = hidden_state['spike']
        refractory = hidden_state['refractory']
        
        # Input current
        input_current = self.input_proj(input_t)
        recurrent_current = self.recurrent_proj(spike)
        total_current = input_current + recurrent_current
        
        # Membrane update with decay
        membrane = self.decay * membrane * (1 - spike) + total_current
        
        # Generate spikes with surrogate gradient
        spike = self.spike_fn(membrane, self.threshold, 
                              self.surrogate_type, self.alpha, self.training)
        
        # Reset membrane after spike
        membrane = membrane * (1 - spike)
        
        # Update refractory counter
        refractory = torch.where(spike.bool(), 
                                  torch.ones_like(refractory) * 3,
                                  torch.clamp(refractory - 1, min=0))
        
        # Output is the spike train projected
        output = self.output_proj(spike)
        
        new_hidden = {
            'membrane': membrane,
            'spike': spike,
            'refractory': refractory,
        }
        
        spike_info = {
            'spike_count': spike.sum(),
            'spike_rate': spike.mean(),
        }
        
        return output, new_hidden, spike_info
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through entire sequence."""
        batch_size = inputs.shape[0]
        hidden = self.init_hidden(batch_size)
        
        outputs = []
        for t in range(inputs.shape[1]):
            output, hidden, _ = self.step(inputs[:, t], hidden)
            outputs.append(output)
        
        return sum(outputs) / len(outputs)


# Example usage
if __name__ == "__main__":
    # Create test SNN
    snn = SpikingRNN(input_size=64, hidden_size=128)
    
    # Configure BPTT
    config = BPTTConfig(
        timesteps=50,
        chunk_size=10,
        use_checkpointing=True,
    )
    
    # Create unroller
    unroller = SNNBPTTUnroller(snn, config)
    
    # Test forward pass
    batch_size = 32
    inputs = torch.randn(batch_size, 50, 64)
    target = torch.randint(0, 2, (batch_size, 128)).float()
    
    output, loss = unroller.unroll(inputs, target)
    
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test with gradient computation
    optimizer = torch.optim.Adam(snn.parameters(), lr=0.001)
    optimizer.zero_grad()
    
    output, loss = unroller.unroll(inputs, target)
    loss.backward()
    
    print(f"Gradient norm: {sum(p.grad.norm().item()**2 for p in snn.parameters() if p.grad is not None)**0.5:.4f}")
    
    optimizer.step()
    print("Training step completed successfully!")
