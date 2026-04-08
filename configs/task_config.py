"""
Task Configuration File
Defines task parameters, grading criteria, and difficulty levels.
"""

from typing import Dict, Any

# Task Configuration Dictionary
TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "easy": {
        "id": "DisasterRescue-Easy-v1",
        "difficulty_level": 1,
        "map_size": 64,
        "num_victims": 5,
        "num_hazards": 8,
        "num_resources": 3,
        "visibility_range": 20,
        "fog_density": 0.2,
        "battery_capacity": 1000,
        "aftershock_probability": 0.01,
        "max_steps": 300,
        "time_limit": 300,
        "success_criteria": {
            "min_victims_rescued": 3,
            "min_resources_delivered": 1,
        },
        "baseline_performance": 0.75,
        "description": "Urban rescue in small, manageable district with few hazards",
    },
    "medium": {
        "id": "DisasterRescue-Medium-v1",
        "difficulty_level": 2,
        "map_size": 96,
        "num_victims": 12,
        "num_hazards": 25,
        "num_resources": 5,
        "visibility_range": 15,
        "fog_density": 0.4,
        "battery_capacity": 800,
        "aftershock_probability": 0.05,
        "max_steps": 400,
        "time_limit": 400,
        "success_criteria": {
            "min_victims_rescued": 7,
            "min_resources_delivered": 3,
        },
        "baseline_performance": 0.55,
        "description": "Larger urban area with moderate hazard complexity and uncertainty",
    },
    "hard": {
        "id": "DisasterRescue-Hard-v1",
        "difficulty_level": 3,
        "map_size": 128,
        "num_victims": 25,
        "num_hazards": 50,
        "num_resources": 8,
        "visibility_range": 10,
        "fog_density": 0.6,
        "battery_capacity": 600,
        "aftershock_probability": 0.1,
        "max_steps": 500,
        "time_limit": 500,
        "success_criteria": {
            "min_victims_rescued": 15,
            "min_resources_delivered": 5,
        },
        "baseline_performance": 0.35,
        "description": "Large-scale metro disaster with dynamic hazards, extreme uncertainty",
        "dynamic_hazard_spawning": True,
        "victim_mobility": True,
    },
}

# Reward Component Weights
REWARD_WEIGHTS = {
    "rescue_reward": 0.3,           # Reward per victim rescued
    "time_penalty": 0.01,            # Penalty per timestep
    "safety_penalty": 0.2,           # Penalty per hazard collision
    "resource_delivery": 0.2,        # Reward per resource delivered
    "exploration_bonus": 0.05,       # Bonus for exploration
    "battery_penalty": 0.1,          # Penalty for low battery
}

# Grading Criteria
GRADING_CONFIG = {
    "deterministic": True,
    "seed_reproducibility": True,
    "num_eval_episodes": 20,
    "score_aggregation": "mean",
    "score_range": [0.0, 1.0],
    "passing_threshold": 0.5,
}

# Evaluation Metrics
EVALUATION_METRICS = {
    "victims_rescued_ratio": {
        "description": "Percentage of available victims successfully rescued",
        "range": [0.0, 1.0],
        "weight": 0.4,
    },
    "resources_delivered": {
        "description": "Number of resources delivered to victims",
        "range": [0, None],
        "weight": 0.2,
    },
    "hazard_collision_count": {
        "description": "Number of times agent collided with hazards",
        "range": [0, None],
        "weight": 0.15,
    },
    "exploration_coverage": {
        "description": "Percentage of map explored",
        "range": [0.0, 1.0],
        "weight": 0.15,
    },
    "battery_efficiency": {
        "description": "Actions per unit battery consumed",
        "range": [0.0, 1.0],
        "weight": 0.1,
    },
}

# Difficulty Progression
DIFFICULTY_ORDER = ["easy", "medium", "hard"]

# Default Configuration
DEFAULT_CONFIG = TASK_CONFIG["easy"]


def get_task_config(difficulty: str) -> Dict[str, Any]:
    """
    Get configuration for a specific difficulty level.
    
    Args:
        difficulty: "easy", "medium", or "hard"
    
    Returns:
        Dictionary with task configuration
    """
    if difficulty not in TASK_CONFIG:
        raise ValueError(f"Unknown difficulty: {difficulty}. Must be one of {list(TASK_CONFIG.keys())}")
    return TASK_CONFIG[difficulty]


def get_all_difficulties() -> list:
    """Get list of all available difficulties."""
    return DIFFICULTY_ORDER


def get_reward_weight(component: str) -> float:
    """
    Get reward weight for a specific component.
    
    Args:
        component: Name of reward component
    
    Returns:
        Weight value
    """
    if component not in REWARD_WEIGHTS:
        raise ValueError(f"Unknown reward component: {component}")
    return REWARD_WEIGHTS[component]


def get_metric_weight(metric: str) -> float:
    """
    Get evaluation weight for a specific metric.
    
    Args:
        metric: Name of metric
    
    Returns:
        Weight value
    """
    if metric not in EVALUATION_METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    return EVALUATION_METRICS[metric]["weight"]


if __name__ == "__main__":
    # Test configuration
    print("Available Tasks:")
    for difficulty in get_all_difficulties():
        config = get_task_config(difficulty)
        print(f"  {difficulty}: {config['description']}")
        print(f"    - Map Size: {config['map_size']}x{config['map_size']}")
        print(f"    - Victims: {config['num_victims']}")
        print(f"    - Hazards: {config['num_hazards']}")
        print(f"    - Battery: {config['battery_capacity']}")
        print()
    
    print("Reward Weights:")
    for component, weight in REWARD_WEIGHTS.items():
        print(f"  {component}: {weight}")
    
    print("\nEvaluation Metrics:")
    for metric, config in EVALUATION_METRICS.items():
        print(f"  {metric}: weight={config['weight']}")