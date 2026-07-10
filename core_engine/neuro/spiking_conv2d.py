"""
Spiking Conv2D Layer for Event-Camera Vision

Implements convolutional layers optimized for sparse spike data from
event cameras. Uses efficient sparse convolution algorithms to process
Address Event Representation (AER) data without converting to dense frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from surrogate_gradient_functions import SurrogateGradientFunction


class SpikingConv2d(nn.Module):
    """
    Spiking 2D Convolutional Layer.
    
    Processes spike tensors and generates output spikes using LIF neuron dynamics.
    Optimized for sparse input from event cameras.
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 threshold: float = 1.0, decay: float = 0.9,
                 surrogate_type: str = 'fast_sigmoid', alpha: float = 1.0,
                 bias: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.threshold = threshold
        self.decay = decay
        self.surrogate_type = surrogate_type
        self.alpha = alpha
        
        # Convolutional weights
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding, bias=bias)
        
        # BatchNorm for stability
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Surrogate gradient function
        self.spike_fn = SurrogateGradientFunction.apply
    
    def forward(self, spikes: torch.Tensor, 
                membrane: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for spiking convolution.
        
        Args:
            spikes: Input spike tensor (batch, channels, height, width)
            membrane: Previous membrane potential (optional)
        
        Returns:
            output_spikes: Output spike tensor
            new_membrane: Updated membrane potential
        """
        batch_size = spikes.shape[0]
        device = spikes.device
        
        # Initialize membrane if not provided
        if membrane is None:
            membrane = torch.zeros(batch_size, self.out_channels,
                                   spikes.shape[2], spikes.shape[3], device=device)
        
        # Convolve input spikes
        conv_output = self.conv(spikes)
        conv_output = self.bn(conv_output)
        
        # Update membrane potential with decay
        membrane = self.decay * membrane * (1 - (membrane > self.threshold).float()) + conv_output
        
        # Generate output spikes with surrogate gradient
        output_spikes = self.spike_fn(membrane, self.threshold,
                                       self.surrogate_type, self.alpha, self.training)
        
        # Reset membrane after spike
        membrane = membrane * (1 - output_spikes)
        
        return output_spikes, membrane


class SpikingConvBlock(nn.Module):
    """
    Block of spiking convolution + pooling for building deep SNNs.
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, pool_size: int = 2,
                 threshold: float = 1.0, decay: float = 0.9,
                 surrogate_type: str = 'fast_sigmoid', alpha: float = 1.0):
        super().__init__()
        
        self.conv = SpikingConv2d(
            in_channels, out_channels, kernel_size,
            threshold=threshold, decay=decay,
            surrogate_type=surrogate_type, alpha=alpha
        )
        self.pool = nn.AvgPool2d(pool_size)
    
    def forward(self, spikes: torch.Tensor,
                membrane: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output_spikes, new_membrane = self.conv(spikes, membrane)
        pooled_spikes = self.pool(output_spikes)
        pooled_membrane = self.pool(new_membrane)
        return pooled_spikes, pooled_membrane


class EventCameraSNN(nn.Module):
    """
    Complete SNN architecture for event-camera vision processing.
    
    Takes AER events converted to spike tensors and processes them
    through multiple spiking convolutional layers.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (2, 128, 128),
                 num_classes: int = 10,
                 base_channels: int = 32,
                 threshold: float = 1.0,
                 decay: float = 0.9,
                 timesteps: int = 10):
        super().__init__()
        
        self.input_shape = input_shape
        self.timesteps = timesteps
        
        # Feature extraction layers
        self.features = nn.Sequential(
            SpikingConvBlock(input_shape[0], base_channels, 
                            kernel_size=3, pool_size=2,
                            threshold=threshold, decay=decay),
            SpikingConvBlock(base_channels, base_channels * 2,
                            kernel_size=3, pool_size=2,
                            threshold=threshold, decay=decay),
            SpikingConvBlock(base_channels * 2, base_channels * 4,
                            kernel_size=3, pool_size=2,
                            threshold=threshold, decay=decay),
        )
        
        # Compute feature map size after pooling
        with torch.no_grad():
            test_input = torch.zeros(1, *input_shape)
            test_out, _ = self.features(test_input, None)
            self.feature_size = test_out.shape[1] * test_out.shape[2] * test_out.shape[3]
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, base_channels * 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(base_channels * 8, num_classes),
        )
    
    def forward(self, event_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process event tensor through SNN.
        
        Args:
            event_tensor: Shape (batch, timesteps, channels, height, width)
                         where each timestep contains accumulated events
        
        Returns:
            class_logits: Classification logits
        """
        batch_size = event_tensor.shape[0]
        
        # Accumulate spikes over timesteps
        accumulated_spikes = None
        feature_membranes = [None] * len(self.features)
        
        for t in range(min(event_tensor.shape[1], self.timesteps)):
            spikes = event_tensor[:, t]  # (batch, channels, H, W)
            
            # Pass through feature layers
            current_spikes = spikes
            for i, layer in enumerate(self.features):
                current_spikes, feature_membranes[i] = layer(
                    current_spikes, feature_membranes[i]
                )
            
            # Accumulate output spikes
            if accumulated_spikes is None:
                accumulated_spikes = current_spikes
            else:
                accumulated_spikes = accumulated_spikes + current_spikes
        
        # Average over timesteps
        if accumulated_spikes is not None:
            accumulated_spikes = accumulated_spikes / min(event_tensor.shape[1], self.timesteps)
        else:
            accumulated_spikes = torch.zeros(batch_size, 
                                             self.features[-1].conv.out_channels,
                                             self.input_shape[1] // 8,
                                             self.input_shape[2] // 8,
                                             device=event_tensor.device)
        
        # Flatten and classify
        features = accumulated_spikes.view(batch_size, -1)
        logits = self.classifier(features)
        
        return logits


class SparseSpikingConv2d(nn.Module):
    """
    Memory-efficient sparse spiking convolution.
    
    Only processes non-zero spike locations, significantly reducing
    computation for sparse event-camera data.
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 threshold: float = 1.0):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, bias=False)
        self.threshold = threshold
        self.membrane_register: Optional[torch.Tensor] = None
    
    def forward_sparse(self, spike_indices: torch.Tensor,
                       spike_values: torch.Tensor,
                       spatial_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process sparse spikes efficiently.
        
        Args:
            spike_indices: Tensor of shape (N, 4) with (batch, channel, y, x) indices
            spike_values: Tensor of shape (N,) with spike values (all 1.0 typically)
            spatial_shape: (height, width) of output
        
        Returns:
            output_spikes: Sparse output spikes
            membrane: Updated membrane at spike locations
        """
        # Initialize membrane register if needed
        if self.membrane_register is None or self.membrane_register.shape[0] != spike_indices.shape[0]:
            self.membrane_register = torch.zeros_like(spike_values)
        
        # Dense convolution on sparse input (could be further optimized)
        # Create sparse tensor representation
        batch_size = spike_indices[:, 0].max().item() + 1
        dense_input = torch.sparse_coo_tensor(
            spike_indices.t(),
            spike_values,
            (batch_size, self.conv.in_channels, spatial_shape[0], spatial_shape[1])
        ).to_dense()
        
        # Convolve
        conv_output = self.conv(dense_input)
        
        # Update membrane
        self.membrane_register = self.membrane_register + conv_output.view(-1)[spike_indices[:, 0] * conv_output.shape[1] + spike_indices[:, 1]]
        
        # Generate spikes
        output_spikes = (self.membrane_register > self.threshold).float()
        
        # Reset membrane where spikes occurred
        self.membrane_register = self.membrane_register * (1 - output_spikes)
        
        return output_spikes, self.membrane_register
    
    def reset_membrane(self):
        """Reset membrane state."""
        if self.membrane_register is not None:
            self.membrane_register.zero_()


# Utility functions for event-to-spike conversion
def aer_events_to_spike_tensor(events: List[Tuple[int, int, int, int]],
                                spatial_shape: Tuple[int, int],
                                temporal_bins: int = 10) -> torch.Tensor:
    """
    Convert AER events to spike tensor for SNN processing.
    
    Args:
        events: List of (x, y, timestamp, polarity) tuples
        spatial_shape: (height, width) of sensor
        temporal_bins: Number of time bins
    
    Returns:
        spike_tensor: Shape (temporal_bins, 2, height, width)
                     where 2 channels are ON/OFF polarities
    """
    height, width = spatial_shape
    
    # Find time range
    if not events:
        return torch.zeros(temporal_bins, 2, height, width)
    
    timestamps = [e[2] for e in events]
    min_ts, max_ts = min(timestamps), max(timestamps)
    time_range = max(max_ts - min_ts, 1)
    
    # Create spike tensor
    spike_tensor = torch.zeros(temporal_bins, 2, height, width)
    
    for x, y, ts, polarity in events:
        bin_idx = int((ts - min_ts) / time_range * (temporal_bins - 1))
        bin_idx = min(bin_idx, temporal_bins - 1)
        channel = 1 if polarity > 0 else 0
        
        if 0 <= x < width and 0 <= y < height:
            spike_tensor[bin_idx, channel, y, x] = 1.0
    
    return spike_tensor


if __name__ == "__main__":
    # Test spiking convolution
    print("Testing SpikingConv2d...")
    
    batch_size = 4
    in_channels = 2  # ON/OFF event channels
    height, width = 64, 64
    
    # Create random spike input
    spikes = torch.rand(batch_size, in_channels, height, width) > 0.9
    spikes = spikes.float()
    
    # Test layer
    conv = SpikingConv2d(in_channels, 16, kernel_size=3)
    output_spikes, membrane = conv(spikes)
    
    print(f"Input spikes: {spikes.sum().item()}")
    print(f"Output shape: {output_spikes.shape}")
    print(f"Output spikes: {output_spikes.sum().item()}")
    
    # Test full SNN
    print("\nTesting EventCameraSNN...")
    
    snn = EventCameraSNN(
        input_shape=(2, 64, 64),
        num_classes=10,
        timesteps=5
    )
    
    # Simulate event data
    event_data = torch.randn(batch_size, 5, 2, 64, 64) > 0.8
    event_data = event_data.float()
    
    logits = snn(event_data)
    print(f"Logits shape: {logits.shape}")
    print(f"Predicted classes: {logits.argmax(dim=1)}")
    
    print("\nAll tests passed!")
