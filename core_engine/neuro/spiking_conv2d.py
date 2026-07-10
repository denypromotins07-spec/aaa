"""
Spiking Convolutional Neural Network Layers.

Implements convolutional layers for Spiking Neural Networks with event-driven computation,
sparse spike propagation, and efficient GPU utilization for neuromorphic vision tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SpikingConvConfig:
    """Configuration for spiking convolutional layers."""
    # Convolution parameters
    in_channels: int = 3
    out_channels: int = 32
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    
    # Neuron parameters
    threshold: float = 1.0
    decay: float = 0.9
    refractory_period: int = 2
    
    # Surrogate gradient alpha
    surrogate_alpha: float = 1.0
    
    # Whether to use batch normalization
    use_bn: bool = True
    
    # Dropout rate
    dropout: float = 0.0


class SpikingConv2d(nn.Module):
    """
    Spiking 2D Convolutional layer.
    
    Combines standard convolution with Leaky Integrate-and-Fire (LIF) neuron dynamics.
    Processes spike events through convolutional filters and generates output spikes.
    """
    
    def __init__(self, config: SpikingConvConfig):
        super().__init__()
        
        self.config = config
        
        # Convolutional layer
        self.conv = nn.Conv2d(
            config.in_channels,
            config.out_channels,
            config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            bias=not config.use_bn
        )
        
        # Batch normalization (optional)
        self.bn = nn.BatchNorm2d(config.out_channels) if config.use_bn else None
        
        # Initialize weights
        self._initialize_weights()
        
        # State buffers
        self.register_buffer('membrane_potential', None)
        self.register_buffer('refractory_count', None)
        self.register_buffer('spike_threshold', torch.tensor(config.threshold))
        
    def _initialize_weights(self):
        """Initialize convolution weights using He initialization."""
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def reset_state(self, batch_size: int, spatial_shape: Tuple[int, int], device: torch.device):
        """Reset membrane potential and refractory state."""
        c = self.config.out_channels
        h, w = spatial_shape
        
        self.membrane_potential = torch.zeros(batch_size, c, h, w, device=device)
        self.refractory_count = torch.zeros(batch_size, c, h, w, device=device, dtype=torch.int)
    
    def forward(self, 
                input_spikes: torch.Tensor,
                time_window: int = 1
                ) -> torch.Tensor:
        """
        Process input spikes through convolution and generate output spikes.
        
        Args:
            input_spikes: Input spike tensor (batch, time, channels, height, width) or
                         static input (batch, channels, height, width)
            time_window: Number of time steps to process
            
        Returns:
            Output spike tensor (batch, time, out_channels, height, width)
        """
        # Handle static vs temporal input
        if input_spikes.dim() == 4:
            # Static input: treat as single time step
            input_spikes = input_spikes.unsqueeze(1)
        
        batch_size, seq_len, _, height, width = input_spikes.shape
        
        # Initialize state if needed
        if self.membrane_potential is None:
            self.reset_state(batch_size, (height, width), input_spikes.device)
        
        output_spikes = []
        
        for t in range(min(seq_len, time_window)):
            input_t = input_spikes[:, t, :, :, :]
            
            # Convolve input spikes
            conv_output = self.conv(input_t)
            
            # Apply batch normalization
            if self.bn is not None:
                conv_output = self.bn(conv_output)
            
            # Update membrane potential (LIF dynamics)
            self.membrane_potential = self.config.decay * self.membrane_potential + conv_output
            
            # Generate spikes
            output_spikes_t = (self.membrane_potential > self.spike_threshold).float()
            
            # Reset membrane potential for spiking neurons
            self.membrane_potential = self.membrane_potential * (1 - output_spikes_t)
            
            # Handle refractory period
            if self.config.refractory_period > 0:
                self.refractory_count = torch.maximum(
                    self.refractory_count - 1,
                    torch.zeros_like(self.refractory_count)
                )
                
                # Suppress spikes during refractory period
                output_spikes_t = output_spikes_t * (self.refractory_count == 0)
                
                # Set refractory count for newly spiking neurons
                new_spikes = output_spikes_t > 0
                self.refractory_count[new_spikes] = self.config.refractory_period
            
            output_spikes.append(output_spikes_t)
        
        # Stack outputs
        output = torch.stack(output_spikes, dim=1)
        
        return output


class SpikingResidualBlock(nn.Module):
    """
    Residual block for deep spiking CNNs.
    
    Implements skip connections adapted for spiking neural networks,
    helping to train deeper SNN architectures.
    """
    
    def __init__(self, channels: int, kernel_size: int = 3, 
                 threshold: float = 1.0, decay: float = 0.9):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = SpikingConv2d(SpikingConvConfig(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            threshold=threshold,
            decay=decay,
            use_bn=True
        ))
        
        self.conv2 = SpikingConv2d(SpikingConvConfig(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            threshold=threshold,
            decay=decay,
            use_bn=True
        ))
        
        self.threshold = threshold
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Add residual (identity spikes are added to membrane potential)
        # This is a simplified residual implementation for SNNs
        if out.shape == identity.shape:
            out = out + identity * 0.5  # Scale to prevent excessive firing
        
        return out


class SpikingPooling(nn.Module):
    """
    Average pooling layer for spiking neural networks.
    
    Computes average spike rate over pooling regions instead of max pooling,
    which is more biologically plausible for rate-coded SNNs.
    """
    
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply average pooling to spike tensor.
        
        Args:
            x: Spike tensor (batch, time, channels, height, width) or
               (batch, channels, height, width)
               
        Returns:
            Pooled spike tensor
        """
        if x.dim() == 5:
            # Temporal input: apply pooling to each time step
            batch, time, channels, height, width = x.shape
            x = x.view(-1, channels, height, width)
            x = self.pool(x)
            _, _, h_out, w_out = x.shape
            x = x.view(batch, time, -1, h_out, w_out)
        else:
            x = self.pool(x)
        
        return x


class SpikingFlatten(nn.Module):
    """
    Flatten layer for transitioning from conv to fully-connected layers in SNNs.
    
    Preserves temporal dimension while flattening spatial dimensions.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flatten spatial dimensions.
        
        Args:
            x: Input tensor (batch, time, channels, height, width) or
               (batch, channels, height, width)
               
        Returns:
            Flattened tensor (batch, time, features) or (batch, features)
        """
        if x.dim() == 5:
            batch, time, channels, height, width = x.shape
            return x.view(batch, time, channels * height * width)
        elif x.dim() == 4:
            batch, channels, height, width = x.shape
            return x.view(batch, channels * height * width)
        else:
            return x


class SpikingClassifier(nn.Module):
    """
    Complete spiking CNN classifier for neuromorphic vision tasks.
    
    Architecture:
        Conv -> Pool -> Conv -> Pool -> FC -> Output
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 num_classes: int = 10,
                 time_window: int = 10):
        super().__init__()
        
        self.time_window = time_window
        
        # Feature extraction
        self.features = nn.Sequential(
            SpikingConv2d(SpikingConvConfig(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=3,
                padding=1
            )),
            SpikingPooling(2, 2),
            
            SpikingConv2d(SpikingConvConfig(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            )),
            SpikingPooling(2, 2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            SpikingFlatten(),
            nn.Linear(64 * 8 * 8, num_classes),  # Assumes 32x32 input
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify input spike sequence.
        
        Args:
            x: Input spike tensor (batch, time, channels, height, width)
            
        Returns:
            Classification output (summed spike counts per class)
        """
        # Extract features
        x = self.features(x)
        
        # Classify
        x = self.classifier(x)
        
        # Sum spikes over time window for classification decision
        # Each class neuron accumulates spikes; highest count wins
        x = x.sum(dim=1)  # (batch, num_classes)
        
        return x


def compute_spike_rate(spike_tensor: torch.Tensor, 
                       time_window: Optional[int] = None) -> torch.Tensor:
    """
    Compute spike rate from spike tensor.
    
    Args:
        spike_tensor: Binary spike tensor with time dimension
        time_window: Optional time window to average over
        
    Returns:
        Spike rate tensor
    """
    if time_window is not None:
        spike_tensor = spike_tensor[:, :time_window, ...]
    
    return spike_tensor.mean(dim=1)


if __name__ == "__main__":
    # Example usage
    print("Testing Spiking Conv2D...")
    
    config = SpikingConvConfig(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        padding=1
    )
    
    conv_layer = SpikingConv2d(config)
    
    # Create test input (simulated spike events)
    batch_size = 2
    time_steps = 10
    height, width = 32, 32
    
    # Random spike input (Bernoulli distributed)
    input_spikes = (torch.rand(batch_size, time_steps, 3, height, width) > 0.8).float()
    
    # Forward pass
    output = conv_layer(input_spikes, time_window=time_steps)
    
    print(f"Input shape: {input_spikes.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total input spikes: {input_spikes.sum().item()}")
    print(f"Total output spikes: {output.sum().item()}")
    print(f"Spike rate: {output.mean().item():.4f}")
    
    # Test classifier
    classifier = SpikingClassifier(input_channels=3, num_classes=10, time_window=time_steps)
    output_logits = classifier(input_spikes)
    print(f"Classifier output shape: {output_logits.shape}")
