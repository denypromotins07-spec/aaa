"""
Async Trajectory Collector for Ray-based Distributed RL

This module implements a Ray Actor that collects trajectories from vectorized
environments and pushes them to the Rust PER buffer via shared memory.
"""

import ray
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time
import threading


@ray.remote
class TrajectoryCollector:
    """
    Ray Actor for asynchronous trajectory collection.
    
    This actor runs vectorized environments, collects trajectories,
    and pushes experiences to the shared memory PER buffer.
    """
    
    def __init__(
        self,
        num_envs: int = 8,
        num_assets: int = 10,
        rollout_length: int = 128,
        shm_name: str = "nexus_per_shm",
    ):
        self.num_envs = num_envs
        self.num_assets = num_assets
        self.rollout_length = rollout_length
        self.shm_name = shm_name
        
        # Import environment modules
        from .env.vectorized_env import VectorizedNexusEnv
        self.env = VectorizedNexusEnv(
            num_envs=num_envs,
            num_assets=num_assets,
        )
        
        # Trajectory buffers
        self.obs_buffer: List[np.ndarray] = []
        self.action_buffer: List[np.ndarray] = []
        self.reward_buffer: List[float] = []
        self.done_buffer: List[bool] = []
        self.info_buffer: List[Dict] = []
        
        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.episode_rewards: List[float] = []
        
        # Thread lock for buffer access
        self._lock = threading.Lock()
        
        # Running flag
        self._running = False
    
    def collect_rollout(
        self,
        policy_fn,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Collect a rollout of trajectories using the given policy.
        
        Args:
            policy_fn: Function that takes observations and returns actions
            device: Device for policy inference
        
        Returns:
            Dictionary containing trajectory data
        """
        self._running = True
        
        # Reset environments
        obs = self.env.reset()
        
        # Clear buffers
        with self._lock:
            self.obs_buffer.clear()
            self.action_buffer.clear()
            self.reward_buffer.clear()
            self.done_buffer.clear()
            self.info_buffer.clear()
        
        episode_rewards = np.zeros(self.num_envs)
        completed_episodes = 0
        
        for step in range(self.rollout_length):
            # Get actions from policy (release GIL during inference)
            actions = policy_fn(obs)
            
            # Step environments
            next_obs, rewards, dones, truncations, infos = self.env.step(actions)
            
            # Store trajectory data
            with self._lock:
                self.obs_buffer.append(obs.copy())
                self.action_buffer.append(actions.copy())
                self.reward_buffer.append(rewards.copy())
                self.done_buffer.append(dones.copy())
                self.info_buffer.append(infos)
            
            # Update statistics
            episode_rewards += rewards
            self.total_steps += self.num_envs
            
            # Handle episode endings
            for i, (done, truncated) in enumerate(zip(dones, truncations)):
                if done or truncated:
                    completed_episodes += 1
                    self.total_episodes += 1
                    self.episode_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0.0
            
            obs = next_obs
        
        self._running = False
        
        # Convert buffers to arrays
        with self._lock:
            result = {
                "observations": np.stack(self.obs_buffer, axis=1),  # (num_envs, rollout, obs_dim)
                "actions": np.stack(self.action_buffer, axis=1),
                "rewards": np.stack(self.reward_buffer, axis=1),
                "dones": np.stack(self.done_buffer, axis=1),
                "infos": self.info_buffer,
                "episode_rewards": self.episode_rewards[-completed_episodes:] if completed_episodes > 0 else [],
            }
        
        return result
    
    def push_to_per_buffer(
        self,
        trajectory: Dict[str, Any],
        values: np.ndarray,
        log_probs: np.ndarray,
    ) -> int:
        """
        Push trajectory to Rust PER buffer via shared memory.
        
        Args:
            trajectory: Trajectory dictionary from collect_rollout
            values: Value estimates for each step
            log_probs: Log probabilities of actions
        
        Returns:
            Number of experiences pushed
        """
        # In production, this would write to shared memory for Rust to consume
        # For now, we just return the count
        num_experiences = (
            trajectory["observations"].shape[0] * 
            trajectory["observations"].shape[1]
        )
        
        # TODO: Implement shared memory write
        # - Serialize trajectory to shared memory format
        # - Signal Rust consumer via atomic flag
        
        return num_experiences
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        avg_reward = (
            np.mean(self.episode_rewards[-100:]) 
            if len(self.episode_rewards) >= 100 
            else np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        )
        
        return {
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "avg_episode_reward": avg_reward,
            "buffer_size": len(self.obs_buffer),
        }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.total_steps = 0
        self.total_episodes = 0
        self.episode_rewards.clear()
    
    def stop(self) -> None:
        """Stop the collector."""
        self._running = False
        self.env.close()


@ray.remote
class AsyncSampler:
    """
    High-throughput sampler that manages multiple TrajectoryCollectors.
    
    This actor coordinates multiple collector actors to maximize
    sampling throughput for distributed training.
    """
    
    def __init__(
        self,
        num_collectors: int = 4,
        envs_per_collector: int = 8,
        **env_kwargs,
    ):
        self.num_collectors = num_collectors
        self.envs_per_collector = envs_per_collector
        
        # Create collector actors
        self.collectors = [
            TrajectoryCollector.remote(
                num_envs=envs_per_collector,
                **env_kwargs,
            )
            for _ in range(num_collectors)
        ]
        
        # Round-robin index
        self._current_collector = 0
    
    def sample_batch(
        self,
        policy_weights: bytes,
        rollout_length: int = 128,
    ) -> List[ray.ObjectRef]:
        """
        Sample a batch of trajectories from all collectors.
        
        Args:
            policy_weights: Serialized policy weights
            rollout_length: Length of each rollout
        
        Returns:
            List of ObjectRefs to trajectory results
        """
        # Broadcast weights to all collectors
        futures = []
        for collector in self.collectors:
            # In production, update policy weights here
            future = collector.collect_rollout.remote(
                policy_fn=None,  # Would use updated weights
                rollout_length=rollout_length,
            )
            futures.append(future)
        
        return futures
    
    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get stats from all collectors."""
        return ray.get([c.get_stats.remote() for c in self.collectors])
    
    def shutdown(self) -> None:
        """Shutdown all collectors."""
        ray.get([c.stop.remote() for c in self.collectors])


def create_sampler_pool(
    num_samplers: int = 2,
    **sampler_kwargs,
) -> List[AsyncSampler]:
    """Create a pool of async samplers for maximum throughput."""
    return [
        AsyncSampler.remote(**sampler_kwargs)
        for _ in range(num_samplers)
    ]
