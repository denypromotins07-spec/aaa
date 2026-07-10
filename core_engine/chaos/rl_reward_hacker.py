#!/usr/bin/env python3
"""
STAGE 25: CHAPTER 3 - RL REWARD HACKER
Attempts to feed Stage 8 Shared Memory Environment corrupted telemetry data
Tests Stage 24 Metacognitive Super-Ego's ability to detect ontological drift
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time


class AttackVector(Enum):
    """Types of reward hacking attacks"""
    REWARD_INFLATION = "reward_inflation"
    STATE_MANIPULATION = "state_manipulation"
    TRANSITION_CORRUPTION = "transition_corruption"
    TERMINAL_STATE_SPOOFING = "terminal_state_spoofing"
    OBSERVATION_NOISE = "observation_noise"


@dataclass
class CorruptedTelemetry:
    """Represents corrupted telemetry data"""
    original_reward: float
    corrupted_reward: float
    original_state: np.ndarray
    corrupted_state: np.ndarray
    attack_vector: AttackVector
    corruption_magnitude: float
    timestamp_ns: int
    detection_evasion_score: float = 0.0


@dataclass
class RewardHackerConfig:
    """Configuration for reward hacking attempts"""
    attack_vectors: List[AttackVector] = field(default_factory=list)
    corruption_probability: float = 0.1
    max_magnitude: float = 0.5
    target_episodes: List[int] = field(default_factory=list)
    chaos_mode_active: bool = False


class RLRewardHacker:
    """
    Adversarial RL Reward Hacker
    
    Attempts to corrupt the Stage 8 Shared Memory Environment's telemetry
    to test Stage 24 Metacognitive Super-Ego's ontological drift detection.
    
    CRITICAL: This runs in a strictly sandboxed environment with weight isolation
    and gradient stopping to prevent exploits from leaking into the primary policy.
    """
    
    def __init__(self, config: RewardHackerConfig, rng_seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(rng_seed)
        self.corruption_count = 0
        self.total_attempts = 0
        self.successful_evasions = 0
        self.detected_attacks = 0
        self._corruption_log: List[CorruptedTelemetry] = []
        
    def activate_chaos_mode(self) -> None:
        """Enable chaos mode for testing"""
        self.config.chaos_mode_active = True
        
    def deactivate_chaos_mode(self) -> None:
        """Disable chaos mode"""
        self.config.chaos_mode_active = False
        
    def is_chaos_mode_active(self) -> bool:
        """Check if chaos mode is active"""
        return self.config.chaos_mode_active
    
    def attempt_reward_inflation(
        self, 
        original_reward: float,
        episode_id: int
    ) -> Tuple[float, bool]:
        """
        Attempt to inflate reward signal
        
        Returns: (corrupted_reward, was_detected)
        """
        if not self.config.chaos_mode_active:
            return original_reward, False
            
        self.total_attempts += 1
        
        if self.rng.random() > self.config.corruption_probability:
            return original_reward, False
            
        # Inflate reward by random magnitude
        magnitude = self.rng.uniform(0.1, self.config.max_magnitude)
        inflated_reward = original_reward * (1.0 + magnitude)
        
        corruption = CorruptedTelemetry(
            original_reward=original_reward,
            corrupted_reward=inflated_reward,
            original_state=np.array([]),
            corrupted_state=np.array([]),
            attack_vector=AttackVector.REWARD_INFLATION,
            corruption_magnitude=magnitude,
            timestamp_ns=time.time_ns()
        )
        
        self._corruption_log.append(corruption)
        self.corruption_count += 1
        
        return inflated_reward, False
        
    def attempt_state_manipulation(
        self,
        original_state: np.ndarray,
        episode_id: int
    ) -> Tuple[np.ndarray, bool]:
        """
        Attempt to manipulate state observations
        
        Returns: (corrupted_state, was_detected)
        """
        if not self.config.chaos_mode_active:
            return original_state, False
            
        self.total_attempts += 1
        
        if self.rng.random() > self.config.corruption_probability:
            return original_state.copy(), False
            
        # Add structured noise to state
        magnitude = self.rng.uniform(0.01, self.config.max_magnitude * 0.1)
        noise = self.rng.normal(0, magnitude, size=original_state.shape)
        corrupted_state = original_state + noise
        
        corruption = CorruptedTelemetry(
            original_reward=0.0,
            corrupted_reward=0.0,
            original_state=original_state.copy(),
            corrupted_state=corrupted_state.copy(),
            attack_vector=AttackVector.STATE_MANIPULATION,
            corruption_magnitude=magnitude,
            timestamp_ns=time.time_ns()
        )
        
        self._corruption_log.append(corruption)
        self.corruption_count += 1
        
        return corrupted_state, False
        
    def attempt_transition_corruption(
        self,
        original_next_state: np.ndarray,
        original_done: bool,
        episode_id: int
    ) -> Tuple[np.ndarray, bool, bool]:
        """
        Attempt to corrupt state transitions
        
        Returns: (corrupted_next_state, corrupted_done, was_detected)
        """
        if not self.config.chaos_mode_active:
            return original_next_state, original_done, False
            
        self.total_attempts += 1
        
        if self.rng.random() > self.config.corruption_probability:
            return original_next_state.copy(), original_done, False
            
        # Slightly perturb next state
        magnitude = self.rng.uniform(0.01, self.config.max_magnitude * 0.05)
        noise = self.rng.normal(0, magnitude, size=original_next_state.shape)
        corrupted_state = original_next_state + noise
        
        # Potentially flip terminal state
        corrupted_done = original_done
        if self.rng.random() < 0.1:  # 10% chance to flip done flag
            corrupted_done = not original_done
            
        corruption = CorruptedTelemetry(
            original_reward=0.0,
            corrupted_reward=0.0,
            original_state=original_next_state.copy(),
            corrupted_state=corrupted_state.copy(),
            attack_vector=AttackVector.TRANSITION_CORRUPTION,
            corruption_magnitude=magnitude,
            timestamp_ns=time.time_ns()
        )
        
        self._corruption_log.append(corruption)
        self.corruption_count += 1
        
        return corrupted_state, corrupted_done, False
        
    def generate_adversarial_episode(
        self,
        episode_length: int,
        state_dim: int
    ) -> List[CorruptedTelemetry]:
        """
        Generate a complete adversarial episode with multiple attack vectors
        
        Returns list of all corruptions applied during the episode
        """
        if not self.config.chaos_mode_active:
            return []
            
        episode_corruptions = []
        state = self.rng.standard_normal(state_dim)
        
        for step in range(episode_length):
            # Apply random attack vector
            if self.config.attack_vectors:
                attack = self.rng.choice(self.config.attack_vectors)
                
                if attack == AttackVector.REWARD_INFLATION:
                    reward = self.rng.uniform(-1, 1)
                    corrupted_reward, _ = self.attempt_reward_inflation(reward, step)
                    
                elif attack == AttackVector.STATE_MANIPULATION:
                    corrupted_state, _ = self.attempt_state_manipulation(state, step)
                    state = corrupted_state
                    
                elif attack == AttackVector.TERMINAL_STATE_SPOOFING:
                    if step == episode_length - 1:
                        # Try to spoof early termination
                        pass
                        
            # Simulate state transition
            state = self.rng.standard_normal(state_dim)
            
        return self._corruption_log[-episode_length:]
        
    def update_evasion_score(self, was_detected: bool) -> None:
        """Update detection evasion score based on attack outcome"""
        if was_detected:
            self.detected_attacks += 1
        else:
            self.successful_evasions += 1
            
    def get_statistics(self) -> Dict:
        """Get reward hacking statistics"""
        total = self.total_attempts
        detected = self.detected_attacks
        evaded = self.successful_evasions
        
        return {
            "total_attempts": total,
            "successful_corruptions": self.corruption_count,
            "detected_attacks": detected,
            "successful_evasions": evaded,
            "detection_rate": detected / total if total > 0 else 0.0,
            "evasion_rate": evaded / total if total > 0 else 0.0,
            "chaos_mode_active": self.config.chaos_mode_active,
            "attack_vectors_configured": [v.value for v in self.config.attack_vectors],
        }
        
    def get_corruption_log(self) -> List[CorruptedTelemetry]:
        """Get full log of corruption attempts"""
        return self._corruption_log.copy()
        
    def reset_statistics(self) -> None:
        """Reset all statistics and logs"""
        self.corruption_count = 0
        self.total_attempts = 0
        self.successful_evasions = 0
        self.detected_attacks = 0
        self._corruption_log.clear()


def create_default_config() -> RewardHackerConfig:
    """Create default reward hacker configuration"""
    return RewardHackerConfig(
        attack_vectors=[
            AttackVector.REWARD_INFLATION,
            AttackVector.STATE_MANIPULATION,
            AttackVector.TRANSITION_CORRUPTION,
        ],
        corruption_probability=0.15,
        max_magnitude=0.3,
        chaos_mode_active=False,
    )


if __name__ == "__main__":
    # Test the reward hacker
    config = create_default_config()
    hacker = RLRewardHacker(config, rng_seed=42)
    
    print("Testing RL Reward Hacker...")
    print(f"Initial chaos mode: {hacker.is_chaos_mode_active()}")
    
    hacker.activate_chaos_mode()
    print(f"After activation: {hacker.is_chaos_mode_active()}")
    
    # Test reward inflation
    original_reward = 0.5
    corrupted, detected = hacker.attempt_reward_inflation(original_reward, 0)
    print(f"Reward inflation: {original_reward} -> {corrupted}, detected: {detected}")
    
    # Test state manipulation
    original_state = np.array([1.0, 2.0, 3.0])
    corrupted_state, detected = hacker.attempt_state_manipulation(original_state, 1)
    print(f"State manipulation detected: {detected}")
    print(f"State difference: {np.linalg.norm(corrupted_state - original_state)}")
    
    # Generate adversarial episode
    corruptions = hacker.generate_adversarial_episode(episode_length=100, state_dim=10)
    print(f"Generated {len(corruptions)} corruptions in episode")
    
    # Get statistics
    stats = hacker.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    hacker.deactivate_chaos_mode()
    print(f"\nAfter deactivation: {hacker.is_chaos_mode_active()}")
