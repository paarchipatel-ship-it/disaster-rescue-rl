"""
Evaluation Script
Runs comprehensive evaluations of agents across all difficulty levels.
Generates detailed metrics and performance reports.
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from environment.disaster_env import DisasterRescueEnv
from agents.baseline_agent import get_agent
from utils.graders import get_grader, get_grader_metrics
from configs.task_config import get_all_difficulties, get_task_config


class EvaluationRunner:
    """
    Comprehensive evaluation runner for agents.
    """
    
    def __init__(self, num_episodes: int = 5, seed: Optional[int] = None):
        """
        Initialize evaluation runner.
        
        Args:
            num_episodes: Number of evaluation episodes per task
            seed: Random seed for reproducibility
        """
        self.num_episodes = num_episodes
        self.seed = seed
        self.results = {}
        self.detailed_metrics = {}
    
    def evaluate_agent(
        self,
        agent_type: str,
        difficulty: str,
        num_episodes: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate an agent on a specific difficulty level.
        
        Args:
            agent_type: Type of agent ("random", "exploration", "grid_search", "greedy", "hybrid")
            difficulty: Task difficulty ("easy", "medium", "hard")
            num_episodes: Override default number of episodes
            verbose: Print progress information
        
        Returns:
            Dictionary with evaluation results
        """
        num_episodes = num_episodes or self.num_episodes
        
        env = DisasterRescueEnv(difficulty=difficulty, seed=self.seed)
        agent = get_agent(agent_type, env=env, seed=self.seed)
        grader = get_grader(difficulty)
        
        scores = []
        rewards = []
        detailed_results = []
        
        if verbose:
            print(f"\nEvaluating {agent_type} on {difficulty}...")
        
        for episode in range(num_episodes):
            obs, _ = env.reset(seed=self.seed + episode if self.seed else None)
            agent.reset()
            
            total_reward = 0.0
            step = 0
            max_steps = get_task_config(difficulty)["max_steps"]
            
            # Run episode
            while step < max_steps:
                action, _ = agent.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                step += 1
                
                if terminated or truncated:
                    break
            
            # Grade episode
            env_state = env.state()
            score = grader.grade(env_state)
            metrics = get_grader_metrics(difficulty, env_state)
            
            scores.append(score)
            rewards.append(total_reward)
            
            detailed_results.append({
                "episode": episode,
                "score": float(score),
                "reward": float(total_reward),
                "steps": int(step),
                "metrics": {k: float(v) for k, v in metrics.items()},
            })
            
            if verbose:
                print(f"  Episode {episode + 1}/{num_episodes}: score={score:.4f}, "
                      f"reward={total_reward:.4f}, steps={step}")
        
        # Aggregate results
        result = {
            "agent": agent_type,
            "difficulty": difficulty,
            "num_episodes": num_episodes,
            "scores": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
            },
            "rewards": {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards)),
            },
            "detailed_episodes": detailed_results,
        }
        
        return result
    
    def evaluate_all_agents(
        self,
        agent_types: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple agents across multiple difficulties.
        
        Args:
            agent_types: List of agent types to evaluate (default: all)
            difficulties: List of difficulties to evaluate (default: all)
            verbose: Print progress information
        
        Returns:
            Dictionary mapping agent_type -> difficulty -> results
        """
        if agent_types is None:
            agent_types = ["random", "exploration", "grid_search", "greedy", "hybrid"]
        
        if difficulties is None:
            difficulties = get_all_difficulties()
        
        results = {}
        
        for agent_type in agent_types:
            results[agent_type] = {}
            
            for difficulty in difficulties:
                try:
                    result = self.evaluate_agent(
                        agent_type=agent_type,
                        difficulty=difficulty,
                        verbose=verbose,
                    )
                    results[agent_type][difficulty] = result
                except Exception as e:
                    print(f"[ERROR] Failed to evaluate {agent_type} on {difficulty}: {e}")
                    results[agent_type][difficulty] = {"error": str(e)}
        
        self.results = results
        return results
    
    def print_summary(self) -> None:
        """Print summary of all evaluation results."""
        if not self.results:
            print("No results to display. Run evaluate_all_agents() first.")
            return
        
        print("\n" + "=" * 100)
        print("EVALUATION SUMMARY")
        print("=" * 100)
        
        for agent_type, difficulties in self.results.items():
            print(f"\n{agent_type.upper()}")
            print("-" * 100)
            
            for difficulty, result in difficulties.items():
                if "error" in result:
                    print(f"  {difficulty}: ERROR - {result['error']}")
                else:
                    scores = result["scores"]
                    rewards = result["rewards"]
                    print(f"  {difficulty}:")
                    print(f"    Score: {scores['mean']:.4f} ± {scores['std']:.4f} "
                          f"(min={scores['min']:.4f}, max={scores['max']:.4f})")
                    print(f"    Reward: {rewards['mean']:.4f} ± {rewards['std']:.4f} "
                          f"(min={rewards['min']:.4f}, max={rewards['max']:.4f})")
        
        print("\n" + "=" * 100)
    
    def print_comparison(self, difficulty: str) -> None:
        """
        Print comparison of all agents on a specific difficulty.
        
        Args:
            difficulty: Task difficulty to compare
        """
        if not self.results:
            print("No results to display. Run evaluate_all_agents() first.")
            return
        
        print(f"\n{'=' * 80}")
        print(f"COMPARISON: {difficulty.upper()}")
        print(f"{'=' * 80}")
        print(f"{'Agent':<20} {'Score':<20} {'Reward':<20} {'Std Dev':<15}")
        print("-" * 80)
        
        # Collect all scores
        agent_scores = []
        
        for agent_type, difficulties in self.results.items():
            if difficulty in difficulties:
                result = difficulties[difficulty]
                if "error" not in result:
                    scores = result["scores"]
                    rewards = result["rewards"]
                    
                    agent_scores.append((agent_type, scores["mean"], scores["std"]))
                    
                    print(f"{agent_type:<20} {scores['mean']:<20.4f} {rewards['mean']:<20.4f} "
                          f"{scores['std']:<15.4f}")
        
        # Sort by score (descending)
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("\nRanking:")
        for rank, (agent_type, score, std) in enumerate(agent_scores, 1):
            print(f"  {rank}. {agent_type}: {score:.4f} ± {std:.4f}")
        
        print(f"{'=' * 80}\n")
    
    def export_results_json(self, filepath: str) -> None:
        """
        Export results to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        if not self.results:
            print("No results to export.")
            return
        
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results exported to {filepath}")
    
    def export_results_csv(self, filepath: str) -> None:
        """
        Export results to CSV file.
        
        Args:
            filepath: Path to save CSV file
        """
        if not self.results:
            print("No results to export.")
            return
        
        import csv
        
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(["Agent", "Difficulty", "Mean Score", "Std Dev", 
                           "Min Score", "Max Score", "Mean Reward"])
            
            # Data rows
            for agent_type, difficulties in self.results.items():
                for difficulty, result in difficulties.items():
                    if "error" not in result:
                        scores = result["scores"]
                        rewards = result["rewards"]
                        
                        writer.writerow([
                            agent_type,
                            difficulty,
                            f"{scores['mean']:.4f}",
                            f"{scores['std']:.4f}",
                            f"{scores['min']:.4f}",
                            f"{scores['max']:.4f}",
                            f"{rewards['mean']:.4f}",
                        ])
        
        print(f"Results exported to {filepath}")


def main():
    """
    Main entry point for evaluation.
    
    Environment Variables:
    - NUM_EPISODES: Number of evaluation episodes (default: 5)
    - SEED: Random seed (default: None)
    - AGENTS: Comma-separated list of agents to evaluate (default: all)
    - DIFFICULTIES: Comma-separated list of difficulties (default: all)
    - EXPORT_JSON: Path to export JSON results (optional)
    - EXPORT_CSV: Path to export CSV results (optional)
    """
    # Get configuration from environment
    num_episodes = int(os.getenv("NUM_EPISODES", "5"))
    seed = int(os.getenv("SEED", "0")) if os.getenv("SEED") else None
    
    agents_str = os.getenv("AGENTS", "")
    agents = [a.strip() for a in agents_str.split(",")] if agents_str else None
    
    difficulties_str = os.getenv("DIFFICULTIES", "")
    difficulties = [d.strip() for d in difficulties_str.split(",")] if difficulties_str else None
    
    export_json = os.getenv("EXPORT_JSON")
    export_csv = os.getenv("EXPORT_CSV")
    
    # Create runner
    print(f"[INFO] Starting evaluation (num_episodes={num_episodes}, seed={seed})...")
    runner = EvaluationRunner(num_episodes=num_episodes, seed=seed)
    
    # Run evaluation
    runner.evaluate_all_agents(
        agent_types=agents,
        difficulties=difficulties,
        verbose=True,
    )
    
    # Print summaries
    runner.print_summary()
    
    if difficulties:
        for difficulty in difficulties:
            runner.print_comparison(difficulty)
    else:
        for difficulty in get_all_difficulties():
            runner.print_comparison(difficulty)
    
    # Export results
    if export_json:
        runner.export_results_json(export_json)
    
    if export_csv:
        runner.export_results_csv(export_csv)
    
    print("[INFO] Evaluation complete!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)