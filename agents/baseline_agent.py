"""
Baseline Agents for Disaster Rescue Environment
Provides random and rule-based baseline implementations for comparison.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
from typing import Tuple, Optional
from environment.disaster_env import DisasterRescueEnv


class BaselineAgent:
    """
    Base class for all agents.
    Defines the interface for predict() method.
    """
    
    def __init__(self, env: Optional[DisasterRescueEnv] = None):
        """
        Initialize baseline agent.
        
        Args:
            env: Environment instance (optional, for rule-based agents)
        """
        self.env = env
        self.step_count = 0
    
    def predict(self, observation: np.ndarray) -> Tuple[int, Optional[dict]]:
        """
        Predict action given observation.
        
        Args:
            observation: Current observation from environment
        
        Returns:
            Tuple of (action, info_dict)
            - action: Action index (0-7)
            - info_dict: Optional metadata
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def reset(self) -> None:
        """Reset agent state."""
        self.step_count = 0


class RandomAgent(BaselineAgent):
    """
    Random baseline agent - takes uniformly random actions.
    
    Useful for:
    - Testing environment mechanics
    - Establishing baseline performance
    - Sanity checks
    """
    
    def __init__(self, env: Optional[DisasterRescueEnv] = None, seed: Optional[int] = None):
        """
        Initialize random agent.
        
        Args:
            env: Environment instance
            seed: Random seed for reproducibility
        """
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
    
    def predict(self, observation: np.ndarray) -> Tuple[int, Optional[dict]]:
        """
        Take random action.
        
        Args:
            observation: Current observation (unused for random agent)
        
        Returns:
            Tuple of (random_action, None)
        """
        action = self.rng.integers(0, 8)
        self.step_count += 1
        return action, None


class ExplorationAgent(BaselineAgent):
    """
    Simple exploration agent - follows a systematic exploration pattern.
    
    Strategy:
    - Spiral outward from center to explore map systematically
    - Helps understand environment dynamics
    """
    
    def __init__(self, env: Optional[DisasterRescueEnv] = None, seed: Optional[int] = None):
        """
        Initialize exploration agent.
        
        Args:
            env: Environment instance
            seed: Random seed for reproducibility
        """
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        self.direction = 0  # 0-7: N, NE, E, SE, S, SW, W, NW
        self.steps_in_direction = 1
        self.steps_taken = 0
        self.direction_changes = 0
    
    def predict(self, observation: np.ndarray) -> Tuple[int, Optional[dict]]:
        """
        Take exploratory action following spiral pattern.
        
        Args:
            observation: Current observation
        
        Returns:
            Tuple of (action, info_dict)
        """
        # Spiral exploration: move in one direction for N steps,
        # then rotate and increase step count
        action = self.direction
        self.steps_taken += 1
        
        # Every few steps, change direction
        if self.steps_taken >= self.steps_in_direction:
            self.direction = (self.direction + 2) % 8  # Rotate 90 degrees
            self.direction_changes += 1
            self.steps_taken = 0
            
            # Increase steps in direction every 2 direction changes
            if self.direction_changes % 2 == 0:
                self.steps_in_direction += 1
        
        self.step_count += 1
        
        info = {
            "direction": self.direction,
            "steps_in_direction": self.steps_in_direction,
        }
        
        return action, info


class GridSearchAgent(BaselineAgent):
    """
    Grid search agent - searches the map in a grid pattern.
    
    Strategy:
    - Scan map row by row or column by column
    - Methodical, ensures coverage
    """
    
    def __init__(self, env: Optional[DisasterRescueEnv] = None, seed: Optional[int] = None):
        """
        Initialize grid search agent.
        
        Args:
            env: Environment instance
            seed: Random seed
        """
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        self.scan_direction = 2  # Start moving East
        self.vertical_step = 0
    
    def predict(self, observation: np.ndarray) -> Tuple[int, Optional[dict]]:
        """
        Take action following grid search pattern.
        
        Args:
            observation: Current observation
        
        Returns:
            Tuple of (action, info_dict)
        """
        # Simple grid: alternate between moving East and moving South
        if self.step_count % 20 == 0 and self.step_count > 0:
            # Every 20 steps, move South and change direction
            action = 4  # South
            self.scan_direction = 2 if self.scan_direction == 6 else 6  # Toggle E/W
            self.vertical_step += 1
        else:
            # Continue in current horizontal direction
            action = self.scan_direction
        
        self.step_count += 1
        
        info = {
            "scan_direction": self.scan_direction,
            "vertical_step": self.vertical_step,
        }
        
        return action, info


class GreedyAgent(BaselineAgent):
    """
    Greedy nearest-neighbor agent - always moves toward nearest victim.
    
    Strategy:
    - Extract victim positions from observation
    - Calculate distance to nearest victim
    - Move toward it greedily
    - Fallback to random if no victims visible
    """
    
    def __init__(self, env: Optional[DisasterRescueEnv] = None, seed: Optional[int] = None):
        """
        Initialize greedy agent.
        
        Args:
            env: Environment instance
            seed: Random seed
        """
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        self.env = env
    
    def predict(self, observation: np.ndarray) -> Tuple[int, Optional[dict]]:
        """
        Move greedily toward nearest victim.
        
        Args:
            observation: Current observation (64x64x5)
        
        Returns:
            Tuple of (action, info_dict)
        """
        if self.env is None:
            # No env provided, fall back to random
            action = self.rng.integers(0, 8)
            self.step_count += 1
            return action, None
        
        # Get agent position and victim channel
        agent_pos = self.env.agent_pos.copy()
        victim_channel = observation[:, :, 1]  # Channel 1 = victims
        
        # Find victim positions in observation
        victim_positions = np.where(victim_channel > 0.5)
        
        if len(victim_positions[0]) == 0:
            # No victims visible, explore randomly
            action = self.rng.integers(0, 8)
            self.step_count += 1
            return action, {"strategy": "random_explore"}
        
        # Convert observation coordinates to agent-relative direction
        # Find nearest victim
        victim_obs_y, victim_obs_x = victim_positions[0], victim_positions[1]
        distances = np.sqrt((victim_obs_x - 32)**2 + (victim_obs_y - 32)**2)
        nearest_idx = np.argmin(distances)
        nearest_y, nearest_x = victim_obs_y[nearest_idx], victim_obs_x[nearest_idx]
        
        # Determine action: move toward nearest victim
        dy = nearest_y - 32  # 32 = center of 64x64
        dx = nearest_x - 32
        
        # Choose action based on direction
        if abs(dx) > abs(dy):
            if dx > 0:
                action = 2  # East
            else:
                action = 6  # West
        else:
            if dy < 0:
                action = 0  # North
            else:
                action = 4  # South
        
        self.step_count += 1
        
        info = {
            "strategy": "greedy_nearest",
            "nearest_victim_dist": float(distances[nearest_idx]),
        }
        
        return action, info


class HybridAgent(BaselineAgent):
    """
    Hybrid agent combining exploration and greedy strategies.
    
    Strategy:
    - Spend 50% time exploring (grid search)
    - Spend 50% time pursuing victims (greedy)
    """
    
    def __init__(self, env: Optional[DisasterRescueEnv] = None, seed: Optional[int] = None):
        """
        Initialize hybrid agent.
        
        Args:
            env: Environment instance
            seed: Random seed
        """
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        self.env = env
        self.exploration_agent = GridSearchAgent(env, seed)
        self.greedy_agent = GreedyAgent(env, seed)
    
    def predict(self, observation: np.ndarray) -> Tuple[int, Optional[dict]]:
        """
        Switch between exploration and greedy pursuit.
        
        Args:
            observation: Current observation
        
        Returns:
            Tuple of (action, info_dict)
        """
        # Alternate strategy every 50 steps
        use_greedy = (self.step_count // 50) % 2 == 1
        
        if use_greedy:
            action, info = self.greedy_agent.predict(observation)
        else:
            action, info = self.exploration_agent.predict(observation)
        
        self.step_count += 1
        
        if info is None:
            info = {}
        info["strategy"] = "greedy" if use_greedy else "exploration"
        
        return action, info


def get_agent(agent_type: str, env: Optional[DisasterRescueEnv] = None, seed: Optional[int] = None) -> BaselineAgent:
    """
    Factory function to get agent by name.
    
    Args:
        agent_type: Type of agent ("random", "exploration", "grid_search", "greedy", "hybrid")
        env: Environment instance
        seed: Random seed
    
    Returns:
        BaselineAgent instance
    
    Raises:
        ValueError: If agent_type is unknown
    """
    agents = {
        "random": RandomAgent,
        "exploration": ExplorationAgent,
        "grid_search": GridSearchAgent,
        "greedy": GreedyAgent,
        "hybrid": HybridAgent,
    }
    
    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}. Must be one of {list(agents.keys())}")
    
    return agents[agent_type](env=env, seed=seed)


if __name__ == "__main__":
    # Test baseline agents
    from environment.disaster_env import DisasterRescueEnv
    
    print("=" * 60)
    print("BASELINE AGENTS TEST")
    print("=" * 60)
    
    env = DisasterRescueEnv(difficulty="easy", seed=42)
    obs, _ = env.reset(seed=42)
    
    agent_types = ["random", "exploration", "grid_search", "greedy", "hybrid"]
    
    for agent_type in agent_types:
        print(f"\n--- {agent_type.upper()} Agent ---")
        agent = get_agent(agent_type, env=env, seed=42)
        
        obs, _ = env.reset(seed=42)
        total_reward = 0.0
        
        for step in range(50):
            action, info = agent.predict(obs)
            obs, reward, terminated, truncated, env_info = env.step(action)
            total_reward += reward
            
            if step < 5:
                print(f"  Step {step}: action={action}, reward={reward:.4f}, info={info}")
            
            if terminated:
                break
        
        print(f"  Total reward (50 steps): {total_reward:.4f}")
    
    print("\n" + "=" * 60)
    print("Baseline agents test passed!")
    print("=" * 60)