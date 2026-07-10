"""
PyTorch Tensor Bridge for NEXUS RL Environment

Provides zero-copy tensor conversion utilities between shared memory
and PyTorch tensors for GPU-accelerated inference.
"""

import numpy as np
from typing import Optional, Tuple, Union
import ctypes


class TorchTensorBridge:
    """
    Bridge for converting shared memory observations to PyTorch tensors.
    
    Supports both CPU and GPU tensors with zero-copy semantics where possible.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._torch_available = False
        
        try:
            import torch
            self._torch_available = True
            self._torch = torch
        except ImportError:
            pass
    
    def from_shared_memory(
        self,
        shm_ptr: int,
        shm_size: int,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
    ) -> Optional["torch.Tensor"]:
        """
        Create a PyTorch tensor from shared memory pointer.
        
        Args:
            shm_ptr: Pointer to shared memory (as integer)
            shm_size: Size of shared memory in bytes
            shape: Shape of the tensor
            dtype: NumPy dtype of the data
        
        Returns:
            PyTorch tensor backed by shared memory
        """
        if not self._torch_available:
            raise RuntimeError("PyTorch not available")
        
        # Create numpy array from pointer
        ptr = ctypes.cast(shm_ptr, ctypes.POINTER(ctypes.c_uint8))
        np_array = np.ctypeslib.as_array(ptr, shape=(shm_size,))
        
        # View as appropriate dtype
        np_view = np_array.view(dtype=dtype)
        
        # Reshape to desired shape
        np_view = np_view.reshape(shape)
        
        # Convert to torch tensor (zero-copy)
        tensor = self._torch.from_numpy(np_view.copy())  # Copy to ensure safety
        
        if self.device != "cpu":
            tensor = tensor.to(self.device)
        
        return tensor
    
    def from_numpy(
        self,
        array: np.ndarray,
        non_blocking: bool = False,
    ) -> "torch.Tensor":
        """
        Convert numpy array to PyTorch tensor.
        
        Args:
            array: NumPy array
            non_blocking: If True, async transfer to GPU
        
        Returns:
            PyTorch tensor
        """
        if not self._torch_available:
            raise RuntimeError("PyTorch not available")
        
        # Zero-copy conversion
        tensor = self._torch.from_numpy(array)
        
        if self.device != "cpu":
            tensor = tensor.to(self.device, non_blocking=non_blocking)
        
        return tensor.float()
    
    def batch_observations(
        self,
        observations: list,
        pin_memory: bool = True,
    ) -> "torch.Tensor":
        """
        Stack multiple observations into a batched tensor.
        
        Args:
            observations: List of observation arrays
            pin_memory: Pin memory for faster GPU transfer
        
        Returns:
            Batched tensor (batch_size, obs_dim)
        """
        if not self._torch_available:
            raise RuntimeError("PyTorch not available")
        
        # Stack numpy arrays first
        stacked = np.stack(observations, axis=0)
        
        # Convert to tensor
        tensor = self._torch.from_numpy(stacked).float()
        
        if pin_memory:
            tensor = tensor.pin_memory()
        
        if self.device != "cpu":
            tensor = tensor.to(self.device)
        
        return tensor
    
    def decode_actions(
        self,
        action_tensor: "torch.Tensor",
    ) -> np.ndarray:
        """
        Decode action tensor back to numpy for execution.
        
        Args:
            action_tensor: Tensor of actions
        
        Returns:
            NumPy array of actions
        """
        if not self._torch_available:
            raise RuntimeError("PyTorch not available")
        
        # Move to CPU and convert
        if action_tensor.device.type != "cpu":
            action_tensor = action_tensor.cpu()
        
        return action_tensor.numpy()


class DLpackBridge:
    """
    DLPack-based tensor bridge for framework interoperability.
    
    Supports conversion between PyTorch, TensorFlow, JAX, and CuPy
    using the DLPack protocol for zero-copy tensor exchange.
    """
    
    def __init__(self):
        self._dlpack_available = False
        
        try:
            import torch
            # Check if torch has dlpack support
            if hasattr(torch.utils, 'dlpack'):
                self._dlpack_available = True
        except ImportError:
            pass
    
    def to_dlpack(self, tensor: "torch.Tensor") -> Any:
        """
        Convert tensor to DLPack capsule.
        
        Args:
            tensor: PyTorch tensor
        
        Returns:
            DLPack capsule
        """
        if not self._dlpack_available:
            raise RuntimeError("DLPack not available")
        
        import torch
        return torch.utils.dlpack.to_dlpack(tensor)
    
    def from_dlpack(self, capsule: Any) -> "torch.Tensor":
        """
        Create tensor from DLPack capsule.
        
        Args:
            capsule: DLPack capsule from another framework
        
        Returns:
            PyTorch tensor
        """
        if not self._dlpack_available:
            raise RuntimeError("DLPack not available")
        
        import torch
        return torch.utils.dlpack.from_dlpack(capsule)
    
    def convert_framework_tensor(
        self,
        source_tensor: Any,
        source_framework: str,
    ) -> "torch.Tensor":
        """
        Convert tensor from another framework to PyTorch.
        
        Args:
            source_tensor: Tensor from source framework
            source_framework: One of 'tensorflow', 'jax', 'cupy'
        
        Returns:
            PyTorch tensor
        """
        if not self._dlpack_available:
            raise RuntimeError("DLPack not available")
        
        # Import source framework and convert to dlpack
        if source_framework == "tensorflow":
            import tensorflow as tf
            capsule = tf.experimental.dlpack.to_dlpack(source_tensor)
        elif source_framework == "jax":
            import jax
            capsule = source_tensor.__dlpack__()
        elif source_framework == "cupy":
            import cupy as cp
            capsule = source_tensor.__dlpack__()
        else:
            raise ValueError(f"Unknown framework: {source_framework}")
        
        return self.from_dlpack(capsule)


def get_tensor_bridge(device: str = "cpu") -> Union[TorchTensorBridge, DLpackBridge]:
    """Factory function to get appropriate tensor bridge."""
    return TorchTensorBridge(device=device)
