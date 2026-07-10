"""
NEXUS Gymnasium Wrapper for RL Trading Environment

This module provides a custom gymnasium.Env wrapper that interfaces with the Rust
RL backend via C-FFI and shared memory, enabling zero-copy state transfer.
"""

import ctypes
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import gymnasium as gym
from gymnasium import spaces


# Load the Rust FFI library
try:
    _nexus_lib = ctypes.CDLL("./libnexus_rl.so")
except OSError:
    _nexus_lib = None

# Define FFI function signatures if library loaded
if _nexus_lib is not None:
    _nexus_lib.nexus_rl_env_create.argtypes = [ctypes.c_uint]
    _nexus_lib.nexus_rl_env_create.restype = ctypes.c_int
    
    _nexus_lib.nexus_rl_env_reset.argtypes = [ctypes.c_int]
    _nexus_lib.nexus_rl_env_reset.restype = ctypes.c_int
    
    _nexus_lib.nexus_rl_env_destroy.argtypes = [ctypes.c_int]
    _nexus_lib.nexus_rl_env_destroy.restype = ctypes.c_int
    
    _nexus_lib.nexus_rl_env_begin_update.argtypes = [ctypes.c_int]
    _nexus_lib.nexus_rl_env_begin_update.restype = ctypes.c_int
    
    _nexus_lib.nexus_rl_env_end_update.argtypes = [ctypes.c_int]
    _nexus_lib.nexus_rl_env_end_update.restype = ctypes.c_int
    
    _nexus_lib.nexus_rl_env_get_shm_ptr.argtypes = [ctypes.c_int]
    _nexus_lib.nexus_rl_env_get_shm_ptr.restype = ctypes.c_void_p
    
    _nexus_lib.nexus_rl_env_get_shm_size.argtypes = [ctypes.c_int]
    _nexus_lib.nexus_rl_env_get_shm_size.restype = ctypes.c_ulonglong


class NexusGymEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    Gymnasium environment wrapper for the NEXUS Rust RL backend.
    
    This environment uses shared memory for zero-copy state transfer between
    Rust (execution) and Python (RL agent) components.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        env_id: int = 0,
        num_assets: int = 10,
        order_book_depth: int = 10,
        feature_dim: int = 64,
        shm_name: Optional[str] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.env_id = env_id
        self.num_assets = num_assets
        self.order_book_depth = order_book_depth
        self.feature_dim = feature_dim
        self.render_mode = render_mode
        self.shm_name = shm_name or f"nexus_rl_shm_{env_id}"
        
        # Calculate observation space dimensions
        # Order book: depth * 4 (bid_price, bid_size, ask_price, ask_size)
        # Market features: feature_dim
        # Portfolio state: num_assets * 3 (position, cash, pnl)
        # Technical indicators: feature_dim
        obs_dim = (
            order_book_depth * 4 +
            feature_dim +
            num_assets * 3 +
            feature_dim
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Action space: [side, order_type, price_offset, size_fraction, asset_idx]
        # For simplicity, we use a flat continuous space that gets decoded
        self.action_space = spaces.Box(
            low=np.array([
                -1.0,   # side: -1=sell, 0=hold, 1=buy
                0.0,    # order_type: 0=market, 1=limit, etc.
                -100.0, # price_offset_ticks
                0.0,    # size_fraction
                0.0,    # asset_idx (normalized)
            ]),
            high=np.array([
                1.0,
                4.0,
                100.0,
                1.0,
                float(num_assets - 1),
            ]),
            dtype=np.float32,
        )
        
        # FFI handle
        self._handle: Optional[int] = None
        self._shm_ptr: Optional[ctypes.c_void_p] = None
        self._shm_size: int = 0
        
        # Shared memory view (numpy array backed by shared memory)
        self._state_view: Optional[np.ndarray] = None
        
        # Initialize environment
        self._init_ffi()
    
    def _init_ffi(self) -> None:
        """Initialize FFI connection to Rust backend."""
        if _nexus_lib is None:
            raise RuntimeError("NEXUS RL library not found. Please build with: cargo build --release")
        
        # Create environment
        self._handle = _nexus_lib.nexus_rl_env_create(self.env_id)
        if self._handle < 0:
            raise RuntimeError(f"Failed to create NEXUS RL environment {self.env_id}")
        
        # Get shared memory pointer
        self._shm_ptr = _nexus_lib.nexus_rl_env_get_shm_ptr(self._handle)
        self._shm_size = _nexus_lib.nexus_rl_env_get_shm_size(self._handle)
        
        if self._shm_ptr and self._shm_size > 0:
            # Create numpy array view of shared memory (zero-copy!)
            shm_array = np.ctypeslib.as_array(
                ctypes.cast(self._shm_ptr, ctypes.POINTER(ctypes.c_uint8)),
                shape=(self._shm_size,)
            )
            # Interpret as float64 for state data
            self._state_view = shm_array.view(dtype=np.float64)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        if self._handle is not None and _nexus_lib is not None:
            # Release GIL during FFI call
            _nexus_lib.nexus_rl_env_reset(self._handle)
        
        # Return initial observation
        obs = self._get_observation()
        return obs, {}
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action array [side, order_type, price_offset, size_fraction, asset_idx]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Decode action and send to Rust backend
        self._send_action(action)
        
        # Begin atomic update (acquire write lock)
        if _nexus_lib is not None and self._handle is not None:
            _nexus_lib.nexus_rl_env_begin_update(self._handle)
        
        # Here would be market simulation / execution logic
        # For now, we just read the updated state
        
        # End atomic update (release write lock)
        if _nexus_lib is not None and self._handle is not None:
            _nexus_lib.nexus_rl_env_end_update(self._handle)
        
        # Get new observation from shared memory (zero-copy read)
        obs = self._get_observation()
        
        # Calculate reward (would come from Rust backend in production)
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = False  # No time limit by default
        
        info = {
            "step": getattr(self, "_step_count", 0),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation from shared memory."""
        if self._state_view is not None:
            # Extract relevant portion of shared memory as observation
            # Layout: [step_counter, writing_flag, episode_step, timestamp, order_book, ...]
            offset = 4  # Skip header fields
            obs_data = self._state_view[offset:offset + self.observation_space.shape[0]]
            return obs_data.astype(np.float32)
        
        # Fallback: return zeros if shared memory not available
        return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def _send_action(self, action: np.ndarray) -> None:
        """Send action to Rust backend."""
        # In production, this would write to a command buffer in shared memory
        # or call FFI functions to transmit the action
        pass
    
    def _calculate_reward(self) -> float:
        """Calculate reward for the current step."""
        # In production, reward comes from Rust backend via shared memory
        return 0.0
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Check for liquidation, max steps, etc.
        return False
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(f"Step: {getattr(self, '_step_count', 0)}")
            print(f"Observation shape: {self.observation_space.shape}")
    
    def close(self) -> None:
        """Clean up resources."""
        if self._handle is not None and _nexus_lib is not None:
            _nexus_lib.nexus_rl_env_destroy(self._handle)
            self._handle = None


def make_nexus_env(
    env_id: int = 0,
    **kwargs,
) -> NexusGymEnv:
    """Factory function to create NEXUS gym environments."""
    return NexusGymEnv(env_id=env_id, **kwargs)
