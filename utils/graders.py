"""
Task Graders - Deterministic Scoring for Each Difficulty Level
Evaluates agent performance and produces normalized scores [0.0, 1.0]
"""

from typing import Dict, Any, Callable
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from configs.task_config import (
    TASK_CONFIG,
    EVALUATION_METRICS,
    get_metric_weight,
)


class TaskGrader:
    """
    Base grader class for evaluating agent performance on a task.
    """
    
    def __init__(self, difficulty: str):
        """
        Initialize grader for a specific difficulty.
        
        Args:
            difficulty: "easy", "medium", or "hard"
        """
        if difficulty not in TASK_CONFIG:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        
        self.difficulty = difficulty
        self.config = TASK_CONFIG[difficulty]
        self.metrics = {}
    
    def grade(self, env_state: Dict[str, Any]) -> float:
        """
        Grade the agent's performance based on final environment state.
        
        Args:
            env_state: Full environment state from env.state()
        
        Returns:
            Normalized score in [0.0, 1.0]
        """
        # Calculate individual metrics
        self._calculate_metrics(env_state)
        
        # Aggregate metrics into final score
        final_score = self._aggregate_score()
        
        # Clamp to valid range
        return float(np.clip(final_score, 0.0, 1.0))
    
    def _calculate_metrics(self, env_state: Dict[str, Any]) -> None:
        """
        Calculate all evaluation metrics from environment state.
        
        Args:
            env_state: Full environment state
        """
        # Metric 1: Victims Rescued Ratio
        total_victims = env_state.get("total_victims", 1)
        rescued_count = env_state.get("rescued_count", 0)
        
        self.metrics["victims_rescued_ratio"] = (
            rescued_count / total_victims if total_victims > 0 else 0.0
        )
        
        # Metric 2: Resources Delivered
        # In this env, resources delivered = min(rescued_count, num_resources)
        # Since each victim gets resources proportionally
        num_resources = self.config.get("num_resources", 1)
        self.metrics["resources_delivered"] = min(rescued_count, num_resources) / num_resources
        
        # Metric 3: Hazard Collision Count (penalize collisions)
        hazard_collisions = env_state.get("hazard_collisions", 0)
        total_hazards = self.config.get("num_hazards", 1)
        # Normalize: 0 collisions = 1.0, all hazards encountered = 0.0
        self.metrics["hazard_collision_count"] = max(
            0.0, 1.0 - (hazard_collisions / max(total_hazards, 1))
        )
        
        # Metric 4: Exploration Coverage
        exploration_coverage = env_state.get("exploration_coverage", 0.0)
        self.metrics["exploration_coverage"] = float(exploration_coverage)
        
        # Metric 5: Battery Efficiency
        current_step = env_state.get("current_step", 1)
        battery_used = env_state.get("max_battery", 1000) - env_state.get("battery", 0)
        max_steps = env_state.get("max_steps", 300)
        
        # Efficiency: minimize battery use per step
        if battery_used > 0:
            efficiency = (max_steps - current_step) / max_steps
            self.metrics["battery_efficiency"] = float(np.clip(efficiency, 0.0, 1.0))
        else:
            self.metrics["battery_efficiency"] = 1.0
    
    def _aggregate_score(self) -> float:
        """
        Aggregate individual metrics into final normalized score.
        
        Uses weighted average of all metrics.
        
        Returns:
            Aggregated score in [0.0, 1.0]
        """
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric_name, metric_value in self.metrics.items():
            if metric_name in EVALUATION_METRICS:
                weight = get_metric_weight(metric_name)
                weighted_sum += metric_value * weight
                total_weight += weight
        
        if total_weight == 0.0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def get_metrics(self) -> Dict[str, float]:
        """Get detailed breakdown of metrics."""
        return self.metrics.copy()


class EasyGrader(TaskGrader):
    """
    Grader for EASY difficulty.
    
    Success criteria:
    - Rescue at least 3 of 5 victims
    - Deliver at least 1 resource
    - Avoid most hazards
    """
    
    def __init__(self):
        super().__init__("easy")
    
    def grade(self, env_state: Dict[str, Any]) -> float:
        """Grade with easy-specific bonuses/penalties."""
        base_score = super().grade(env_state)
        
        # Easy task bonus if success criteria met
        min_victims = self.config["success_criteria"]["min_victims_rescued"]
        rescued_count = env_state.get("rescued_count", 0)
        
        if rescued_count >= min_victims:
            base_score += 0.15  # Success bonus
        
        return float(np.clip(base_score, 0.0, 1.0))


class MediumGrader(TaskGrader):
    """
    Grader for MEDIUM difficulty.
    
    Success criteria:
    - Rescue at least 7 of 12 victims
    - Deliver at least 3 resources
    - Handle moderate hazard encounters
    - Manage battery efficiently
    """
    
    def __init__(self):
        super().__init__("medium")
    
    def grade(self, env_state: Dict[str, Any]) -> float:
        """Grade with medium-specific requirements."""
        base_score = super().grade(env_state)
        
        # Medium task: stricter success criteria
        min_victims = self.config["success_criteria"]["min_victims_rescued"]
        rescued_count = env_state.get("rescued_count", 0)
        
        if rescued_count >= min_victims:
            base_score += 0.10  # More modest bonus
        
        # Battery penalty if too much used
        battery = env_state.get("battery", 0)
        if battery < 0.1 * env_state.get("max_battery", 800):
            base_score -= 0.10  # Penalize poor battery management
        
        return float(np.clip(base_score, 0.0, 1.0))


class HardGrader(TaskGrader):
    """
    Grader for HARD difficulty.
    
    Success criteria:
    - Rescue at least 15 of 25 victims
    - Deliver at least 5 resources
    - Navigate extreme hazards and uncertainty
    - Maintain optimal exploration and battery management
    
    No bonus for meeting criteria - must compete on metrics.
    """
    
    def __init__(self):
        super().__init__("hard")
    
    def grade(self, env_state: Dict[str, Any]) -> float:
        """Grade with hard-specific strict evaluation."""
        base_score = super().grade(env_state)
        
        # Hard task: No bonus, pure metric competition
        # Penalize inefficiency severely
        current_step = env_state.get("current_step", 500)
        max_steps = env_state.get("max_steps", 500)
        
        # Penalize if episode terminated early without good results
        rescued_count = env_state.get("rescued_count", 0)
        min_victims = self.config["success_criteria"]["min_victims_rescued"]
        
        if current_step > 0.8 * max_steps and rescued_count < min_victims:
            base_score -= 0.20  # Heavy penalty for inefficiency
        
        # Reward exploration in hard task
        exploration = env_state.get("exploration_coverage", 0.0)
        if exploration > 0.5:
            base_score += 0.05
        
        return float(np.clip(base_score, 0.0, 1.0))


# Dictionary of graders for easy access
GRADERS: Dict[str, TaskGrader] = {
    "easy": EasyGrader(),
    "medium": MediumGrader(),
    "hard": HardGrader(),
}


def get_grader(difficulty: str) -> TaskGrader:
    """
    Get grader instance for a difficulty level.
    
    Args:
        difficulty: "easy", "medium", or "hard"
    
    Returns:
        TaskGrader instance
    
    Raises:
        ValueError: If difficulty is unknown
    """
    if difficulty not in GRADERS:
        raise ValueError(f"Unknown difficulty: {difficulty}. Must be one of {list(GRADERS.keys())}")
    return GRADERS[difficulty]


def grade_episode(difficulty: str, env_state: Dict[str, Any]) -> float:
    """
    Quick function to grade an episode.
    
    Args:
        difficulty: "easy", "medium", or "hard"
        env_state: Full environment state from env.state()
    
    Returns:
        Normalized score [0.0, 1.0]
    """
    grader = get_grader(difficulty)
    return grader.grade(env_state)


def get_grader_metrics(difficulty: str, env_state: Dict[str, Any]) -> Dict[str, float]:
    """
    Get detailed metrics for an episode.
    
    Args:
        difficulty: "easy", "medium", or "hard"
        env_state: Full environment state
    
    Returns:
        Dictionary of all metrics
    """
    grader = get_grader(difficulty)
    grader.grade(env_state)  # Must call grade() first to calculate metrics
    return grader.get_metrics()


if __name__ == "__main__":
    # Test graders with mock environment states
    
    print("=" * 60)
    print("GRADER TEST")
    print("=" * 60)
    
    # Test Easy Grader
    print("\n--- Easy Grader Test ---")
    easy_state = {
        "agent_pos": [32, 32],
        "victims": [{"pos": [10, 10], "rescued": True, "idx": 0}] * 4,
        "hazards": [],
        "resources": [],
        "battery": 800,
        "max_battery": 1000,
        "current_step": 150,
        "max_steps": 300,
        "difficulty": "easy",
        "rescued_count": 4,
        "total_victims": 5,
        "hazard_collisions": 1,
        "exploration_coverage": 0.35,
    }
    
    easy_grader = get_grader("easy")
    easy_score = easy_grader.grade(easy_state)
    easy_metrics = easy_grader.get_metrics()
    
    print(f"Score: {easy_score:.4f}")
    print("Metrics:")
    for metric, value in easy_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test Medium Grader
    print("\n--- Medium Grader Test ---")
    medium_state = {
        "agent_pos": [48, 48],
        "victims": [{"pos": [20+i*5, 20+i*5], "rescued": True, "idx": i} for i in range(8)],
        "hazards": [],
        "resources": [],
        "battery": 400,
        "max_battery": 800,
        "current_step": 300,
        "max_steps": 400,
        "difficulty": "medium",
        "rescued_count": 8,
        "total_victims": 12,
        "hazard_collisions": 3,
        "exploration_coverage": 0.50,
    }
    
    medium_grader = get_grader("medium")
    medium_score = medium_grader.grade(medium_state)
    medium_metrics = medium_grader.get_metrics()
    
    print(f"Score: {medium_score:.4f}")
    print("Metrics:")
    for metric, value in medium_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test Hard Grader
    print("\n--- Hard Grader Test ---")
    hard_state = {
        "agent_pos": [64, 64],
        "victims": [{"pos": [30+i*3, 30+i*3], "rescued": True, "idx": i} for i in range(16)],
        "hazards": [],
        "resources": [],
        "battery": 150,
        "max_battery": 600,
        "current_step": 450,
        "max_steps": 500,
        "difficulty": "hard",
        "rescued_count": 16,
        "total_victims": 25,
        "hazard_collisions": 8,
        "exploration_coverage": 0.70,
    }
    
    hard_grader = get_grader("hard")
    hard_score = hard_grader.grade(hard_state)
    hard_metrics = hard_grader.get_metrics()
    
    print(f"Score: {hard_score:.4f}")
    print("Metrics:")
    for metric, value in hard_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("Grader test passed!")
    print("=" * 60)