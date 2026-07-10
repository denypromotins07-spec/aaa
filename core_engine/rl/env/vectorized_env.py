"""
Vectorized NEXUS Environment for Batched RL Training

This module implements a vectorized environment that runs multiple Rust RL
environments in parallel, aggregating their states into batched PyTorch tensors
for efficient GPU inference.
"""

import ctypes
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import gymnasium as gym
from concurrent.futures import ThreadPoolExecutor
import threading


class VectorizedNexusEnv:
    """
    Vectorized wrapper that manages multiple NEXUS environments.
    
    This class spawns multiple Rust environment instances and aggregates
    their observations into batched tensors for efficient GPU training.
    """
    
    def __init__(
        self,
        num_envs: int = 8,
        num_assets: int = 10,
        order_book_depth: int = 10,
        feature_dim: int = 64,
    ):
        self.num_envs = num_envs
        self.num_assets = num_assets
        self.order_book_depth = order_book_depth
        self.feature_dim = feature_dim
        
        # Import here to avoid circular imports
        from .nexus_gym_wrapper import NexusGymEnv
        
        # Create individual environments
        self.envs: List[NexusGymEnv] = []
        for i in range(num_envs):
            env = NexusGymEnv(
                env_id=i,
                num_assets=num_assets,
                order_book_depth=order_book_depth,
                feature_dim=feature_dim,
            )
            self.envs.append(env)
        
        # Calculate observation dimension
        obs_dim = (
            order_book_depth * 4 +
            feature_dim +
            num_assets * 3 +
            feature_dim
        )
        
        # Batched observation space
        self.batched_obs_space = (num_envs, obs_dim)
        
        # Action space (same for all envs)
        self.action_space = self.envs[0].action_space
        
        # Thread pool for parallel environment stepping
        self._executor = ThreadPoolExecutor(max_workers=num_envs)
        self._lock = threading.Lock()
        
        # Step counters per environment
        self._step_counts = np.zeros(num_envs, dtype=np.int64)
        
        # Done flags
        self._dones = np.zeros(num_envs, dtype=np.bool_)
    
    def reset(self) -> np.ndarray:
        """
        Reset all environments and return batched observations.
        
        Returns:
            Batched observations: (num_envs, obs_dim)
        """
        # Reset all environments in parallel
        futures = []
        for env in self.envs:
            future = self._executor.submit(env.reset)
            futures.append(future)
        
        # Collect results
        observations = []
        for i, future in enumerate(futures):
            obs, _ = future.result()
            observations.append(obs)
            self._step_counts[i] = 0
            self._dones[i] = False
        
        return np.stack(observations, axis=0)
    
    def step_async(self, actions: np.ndarray) -> None:
        """
        Send actions to all environments (non-blocking).
        
        Args:
            actions: Batched actions (num_envs, action_dim)
        """
        self._pending_futures = []
        for i, env in enumerate(self.envs):
            action = actions[i]
            future = self._executor.submit(env.step, action)
            self._pending_futures.append((i, future))
    
    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Wait for all environment steps to complete.
        
        Returns:
            observations, rewards, dones, truncations, infos
        """
        observations = [None] * self.num_envs
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        truncations = np.zeros(self.num_envs, dtype=np.bool_)
        infos = [{} for _ in range(self.num_envs)]
        
        for i, future in self._pending_futures:
            result = future.result()
            obs, reward, done, truncated, info = result
            observations[i] = obs
            rewards[i] = reward
            dones[i] = done
            truncations[i] = truncated
            infos[i] = info
            self._step_counts[i] = info.get("step", self._step_counts[i])
        
        self._pending_futures = []
        
        return (
            np.stack(observations, axis=0),
            rewards,
            dones,
            truncations,
            infos,
        )
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Execute one step in all environments (blocking).
        
        Args:
            actions: Batched actions (num_envs, action_dim)
        
        Returns:
            observations, rewards, dones, truncations, infos
        """
        self.step_async(actions)
        return self.step_wait()
    
    def get_torch_tensor(self, observations: np.ndarray) -> "torch.Tensor":
        """
        Convert observations to PyTorch tensor for GPU training.
        
        Uses torch.from_blob for zero-copy conversion when possible.
        
        Args:
            observations: numpy array (num_envs, obs_dim)
        
        Returns:
            PyTorch tensor on CPU (can be moved to GPU)
        """
        try:
            import torch
            
            # Zero-copy conversion using from_blob
            tensor = torch.from_numpy(observations)
            return tensor.float()
            
        except ImportError:
            raise RuntimeError("PyTorch not installed. Install with: pip install torch")
    
    def close(self) -> None:
        """Clean up all environments."""
        for env in self.envs:
            env.close()
        self._executor.shutdown(wait=True)
    
    @property
    def unwrapped(self):
        """Return the underlying environments."""
        return self.envs


def make_vectorized_env(
    num_envs: int = 8,
    **kwargs,
) -> VectorizedNexusEnv:
    """Factory function to create vectorized NEXUS environments."""
    return VectorizedNexusEnv(num_envs=num_envs, **kwargs)
