"""
Disaster Rescue Environment - OpenEnv Compliant RL Environment
Real-world autonomous drone search & rescue simulation post-disaster.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional
import json
import base64


class DisasterRescueEnv(gym.Env):
    """
    Autonomous Drone Search & Rescue Environment.
    
    Agent controls a drone navigating post-disaster urban terrain to:
    - Locate and rescue survivors
    - Deliver resources/medical supplies
    - Avoid hazards (fires, rubble, unstable structures)
    - Manage limited battery
    
    Real-world inspired constraints:
    - Partial observability (fog of war)
    - Dynamic hazards (aftershocks)
    - Resource scarcity
    - Time pressure
    """
    
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 10,
    }
    
    def __init__(
        self,
        difficulty: str = "easy",
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Disaster Rescue Environment.
        
        Args:
            difficulty: "easy", "medium", or "hard"
            render_mode: "rgb_array" or "human"
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Task-specific parameters
        self.difficulty = difficulty
        self.render_mode = render_mode
        
        # Load difficulty parameters
        self.task_params = self._load_task_params(difficulty)
        
        # Set seed for reproducibility
        if seed is not None:
            self.seed(seed)
        else:
            self.np_random = np.random.default_rng()
        
        # Environment dimensions
        self.map_size = self.task_params["map_size"]
        self.grid_width = self.map_size
        self.grid_height = self.map_size
        
        # Action Space: 8 directions (N, NE, E, SE, S, SW, W, NW)
        self.action_space = spaces.Discrete(8)
        self.action_meanings = {
            0: "North",
            1: "Northeast",
            2: "East",
            3: "Southeast",
            4: "South",
            5: "Southwest",
            6: "West",
            7: "Northwest",
        }
        
        # Observation Space: 64x64x5 channels
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(64, 64, 5),
            dtype=np.float32,
        )
        
        # Initialize state variables
        self.agent_pos = np.array([self.grid_width // 2, self.grid_height // 2], dtype=np.int32)
        self.victims = []  # List of victim positions
        self.hazards = []  # List of hazard positions with intensity
        self.resources = []  # List of resource positions
        self.rescued_victims = set()  # Track rescued victim indices
        self.delivered_resources = 0
        self.battery = self.task_params["battery_capacity"]
        self.max_battery = self.task_params["battery_capacity"]
        self.current_step = 0
        self.max_steps = self.task_params.get("max_steps", 500)
        
        # Tracking for metrics
        self.hazard_collisions = 0
        self.exploration_mask = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        self.victim_found_indices = set()
        
        # Reset to initialize positions
        self.reset()
    
    def _load_task_params(self, difficulty: str) -> Dict[str, Any]:
        """Load task parameters based on difficulty level."""
        params = {
            "easy": {
                "map_size": 64,
                "num_victims": 5,
                "num_hazards": 8,
                "num_resources": 3,
                "visibility_range": 20,
                "fog_density": 0.2,
                "battery_capacity": 1000,
                "aftershock_probability": 0.01,
                "max_steps": 300,
            },
            "medium": {
                "map_size": 96,
                "num_victims": 12,
                "num_hazards": 25,
                "num_resources": 5,
                "visibility_range": 15,
                "fog_density": 0.4,
                "battery_capacity": 800,
                "aftershock_probability": 0.05,
                "max_steps": 400,
            },
            "hard": {
                "map_size": 128,
                "num_victims": 25,
                "num_hazards": 50,
                "num_resources": 8,
                "visibility_range": 10,
                "fog_density": 0.6,
                "battery_capacity": 600,
                "aftershock_probability": 0.1,
                "max_steps": 500,
            },
        }
        return params.get(difficulty, params["easy"])
    
    def seed(self, seed: Optional[int] = None) -> list:
        """Set random seed for reproducibility."""
        self.np_random = np.random.default_rng(seed)
        return [seed]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
        
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            self.seed(seed)
        
        # Reset state
        self.agent_pos = np.array([self.grid_width // 2, self.grid_height // 2], dtype=np.int32)
        self.current_step = 0
        self.battery = self.max_battery
        self.hazard_collisions = 0
        self.rescued_victims = set()
        self.delivered_resources = 0
        self.victim_found_indices = set()
        self.exploration_mask = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        # Spawn victims
        self.victims = []
        for _ in range(self.task_params["num_victims"]):
            victim_pos = self._random_position()
            self.victims.append({"pos": victim_pos, "rescued": False, "idx": len(self.victims)})
        
        # Spawn hazards
        self.hazards = []
        for _ in range(self.task_params["num_hazards"]):
            hazard_pos = self._random_position()
            intensity = self.np_random.uniform(0.3, 1.0)
            self.hazards.append({"pos": hazard_pos, "intensity": intensity})
        
        # Spawn resources
        self.resources = []
        for _ in range(self.task_params["num_resources"]):
            resource_pos = self._random_position()
            self.resources.append({"pos": resource_pos, "collected": False})
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action index (0-7) representing movement direction
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        
        # Movement deltas for 8 directions
        deltas = {
            0: (0, -1),      # North
            1: (1, -1),      # Northeast
            2: (1, 0),       # East
            3: (1, 1),       # Southeast
            4: (0, 1),       # South
            5: (-1, 1),      # Southwest
            6: (-1, 0),      # West
            7: (-1, -1),     # Northwest
        }
        
        dx, dy = deltas[action]
        new_pos = np.array(
            [
                np.clip(self.agent_pos[0] + dx, 0, self.grid_width - 1),
                np.clip(self.agent_pos[1] + dy, 0, self.grid_height - 1),
            ],
            dtype=np.int32,
        )
        
        # Check movement validity
        moved = not np.array_equal(new_pos, self.agent_pos)
        if moved:
            self.agent_pos = new_pos
            self.battery -= 1
        
        # Mark explored cells
        self.exploration_mask[self.agent_pos[1], self.agent_pos[0]] = True
        
        reward = 0.0
        
        # Reward components
        # 1. Time penalty (discourage long episodes)
        reward -= 0.01
        
        # 2. Check for victim rescue
        for victim in self.victims:
            if not victim["rescued"] and np.allclose(self.agent_pos, victim["pos"]):
                victim["rescued"] = True
                self.rescued_victims.add(victim["idx"])
                self.victim_found_indices.add(victim["idx"])
                reward += 0.3  # Large reward for rescue
        
        # 3. Check for resource collection and delivery
        for resource in self.resources:
            if not resource["collected"] and np.allclose(self.agent_pos, resource["pos"]):
                resource["collected"] = True
                reward += 0.1  # Reward for collecting resource
        
        # Bonus: deliver resources to rescued victims
        num_rescued = len(self.rescued_victims)
        if num_rescued > self.delivered_resources:
            self.delivered_resources = num_rescued
            reward += 0.2  # Bonus for resource delivery
        
        # 4. Hazard penalty
        hazard_danger = 0.0
        for hazard in self.hazards:
            if np.allclose(self.agent_pos, hazard["pos"]):
                hazard_danger = max(hazard_danger, hazard["intensity"])
                self.hazard_collisions += 1
        
        if hazard_danger > 0.0:
            reward -= hazard_danger * 0.2
        
        # 5. Exploration bonus (discover new areas)
        exploration_coverage = np.sum(self.exploration_mask) / (self.grid_width * self.grid_height)
        if exploration_coverage > 0.1:
            reward += 0.02
        
        # 6. Battery management penalty
        battery_ratio = self.battery / self.max_battery
        if battery_ratio < 0.2:
            reward -= 0.1
        
        # Random aftershock (dynamic hazard)
        if self.np_random.uniform() < self.task_params["aftershock_probability"]:
            new_hazard_pos = self._random_position()
            self.hazards.append({"pos": new_hazard_pos, "intensity": 0.7})
        
        # Episode termination conditions
        terminated = False
        
        # Battery depleted
        if self.battery <= 0:
            terminated = True
            reward -= 0.5
        
        # All victims rescued
        if len(self.rescued_victims) == len(self.victims):
            terminated = True
            reward += 0.5  # Completion bonus
        
        # Step limit reached
        truncated = self.current_step >= self.max_steps
        
        # Normalize reward to [0, 1]
        reward = np.clip(reward, 0.0, 1.0)
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Generate observation: 64x64x5 grid.
        
        Channels:
        0: Agent position
        1: Victims (alive)
        2: Hazards (intensity)
        3: Resources
        4: Visibility/fog
        """
        # Create observation grid (downsampled to 64x64)
        scale = self.map_size / 64
        obs_size = 64
        obs = np.zeros((obs_size, obs_size, 5), dtype=np.float32)
        
        # Channel 0: Agent position
        agent_scaled = (self.agent_pos / scale).astype(int)
        agent_scaled = np.clip(agent_scaled, 0, obs_size - 1)
        obs[agent_scaled[1], agent_scaled[0], 0] = 1.0
        
        # Channel 1: Victims (alive only)
        for victim in self.victims:
            if not victim["rescued"]:
                pos_scaled = (victim["pos"] / scale).astype(int)
                pos_scaled = np.clip(pos_scaled, 0, obs_size - 1)
                obs[pos_scaled[1], pos_scaled[0], 1] = 1.0
        
        # Channel 2: Hazards (with intensity)
        for hazard in self.hazards:
            pos_scaled = (hazard["pos"] / scale).astype(int)
            pos_scaled = np.clip(pos_scaled, 0, obs_size - 1)
            obs[pos_scaled[1], pos_scaled[0], 2] = min(1.0, hazard["intensity"])
        
        # Channel 3: Resources (uncollected)
        for resource in self.resources:
            if not resource["collected"]:
                pos_scaled = (resource["pos"] / scale).astype(int)
                pos_scaled = np.clip(pos_scaled, 0, obs_size - 1)
                obs[pos_scaled[1], pos_scaled[0], 3] = 1.0
        
        # Channel 4: Visibility (fog of war based on distance from agent)
        visibility_range = self.task_params["visibility_range"] / scale
        for y in range(obs_size):
            for x in range(obs_size):
                dist = np.sqrt((x - agent_scaled[0])**2 + (y - agent_scaled[1])**2)
                visibility = 1.0 if dist <= visibility_range else 0.0
                obs[y, x, 4] = visibility
        
        return obs
    
    def state(self) -> Dict[str, Any]:
        """
        Return full environment state for reproducibility and debugging.
        
        Returns:
            Dictionary with complete state information
        """
        return {
            "agent_pos": self.agent_pos.tolist(),
            "victims": [
                {
                    "pos": v["pos"].tolist(),
                    "rescued": v["rescued"],
                    "idx": v["idx"],
                }
                for v in self.victims
            ],
            "hazards": [
                {
                    "pos": h["pos"].tolist(),
                    "intensity": float(h["intensity"]),
                }
                for h in self.hazards
            ],
            "resources": [
                {
                    "pos": r["pos"].tolist(),
                    "collected": r["collected"],
                }
                for r in self.resources
            ],
            "battery": int(self.battery),
            "max_battery": int(self.max_battery),
            "current_step": int(self.current_step),
            "max_steps": int(self.max_steps),
            "difficulty": self.difficulty,
            "rescued_count": len(self.rescued_victims),
            "total_victims": len(self.victims),
            "hazard_collisions": int(self.hazard_collisions),
            "exploration_coverage": float(np.sum(self.exploration_mask) / (self.grid_width * self.grid_height)),
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info for step/reset."""
        return {
            "battery": self.battery,
            "victims_rescued": len(self.rescued_victims),
            "total_victims": len(self.victims),
            "step": self.current_step,
            "hazard_collisions": self.hazard_collisions,
        }
    
    def _random_position(self) -> np.ndarray:
        """Generate random position on map."""
        return np.array(
            [
                self.np_random.integers(0, self.grid_width),
                self.np_random.integers(0, self.grid_height),
            ],
            dtype=np.int32,
        )
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode is None:
            return None
        
        # Create RGB image
        img = np.ones((self.grid_height, self.grid_width, 3), dtype=np.uint8) * 255
        
        # Draw hazards (red)
        for hazard in self.hazards:
            x, y = hazard["pos"]
            intensity = int(hazard["intensity"] * 255)
            img[y, x] = [intensity, 0, 0]
        
        # Draw victims (blue)
        for victim in self.victims:
            if not victim["rescued"]:
                x, y = victim["pos"]
                img[y, x] = [0, 0, 255]
        
        # Draw resources (green)
        for resource in self.resources:
            if not resource["collected"]:
                x, y = resource["pos"]
                img[y, x] = [0, 255, 0]
        
        # Draw agent (yellow)
        x, y = self.agent_pos
        img[y, x] = [255, 255, 0]
        
        if self.render_mode == "rgb_array":
            return img
        elif self.render_mode == "human":
            print(f"Step {self.current_step} | Battery: {self.battery}/{self.max_battery} | "
                  f"Rescued: {len(self.rescued_victims)}/{len(self.victims)}")
            return None
    
    def close(self):
        """Clean up resources."""
        pass


if __name__ == "__main__":
    # Test environment
    env = DisasterRescueEnv(difficulty="easy")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a few steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.4f}, Terminated: {terminated}")
        if terminated:
            break
    
    print("Environment test passed!")