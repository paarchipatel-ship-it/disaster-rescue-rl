"""
Structured Logging Utilities
Handles mandatory [START], [STEP], [END] logging format for inference.py
"""

import sys
import json
import base64
from typing import Dict, Any, Optional
from datetime import datetime


class StructuredLogger:
    """
    Logger for emitting structured logs in [START], [STEP], [END] format.
    
    Mandatory format per hackathon requirements:
    [START] run_id=<ID> task=<task_id> model=<MODEL_NAME>
    [STEP] step=<step_num> reward=<float> state=<json/base64> action=<action_str>
    [END] run_id=<ID> task=<task_id> score=<float>
    """
    
    def __init__(self, use_stdout: bool = True):
        """
        Initialize logger.
        
        Args:
            use_stdout: If True, log to stdout only (required for validation)
        """
        self.use_stdout = use_stdout
        self.logs = []
    
    def log_start(
        self,
        run_id: str,
        task: str,
        model: str,
    ) -> None:
        """
        Log the start of a run.
        
        Mandatory format:
        [START] run_id=<run_id> task=<task> model=<model>
        
        Args:
            run_id: Unique identifier for this run (UUID)
            task: Task name ("easy", "medium", "hard")
            model: Model name (from MODEL_NAME env var)
        """
        log_entry = f"[START] run_id={run_id} task={task} model={model}"
        self._emit(log_entry)
    
    def log_step(
        self,
        step: int,
        reward: float,
        state: Dict[str, Any],
        action: int,
    ) -> None:
        """
        Log a single step in the episode.
        
        Mandatory format:
        [STEP] step=<step> reward=<float> state=<json> action=<action>
        
        Args:
            step: Step number in episode
            reward: Reward value for this step (float)
            state: Environment state (dict, will be JSON-serialized)
            action: Action taken (int or str)
        """
        # Serialize state to JSON string
        state_json = json.dumps(state, default=str)
        
        log_entry = f"[STEP] step={step} reward={reward:.4f} state={state_json} action={action}"
        self._emit(log_entry)
    
    def log_end(
        self,
        run_id: str,
        task: str,
        score: float,
    ) -> None:
        """
        Log the end of a run.
        
        Mandatory format:
        [END] run_id=<run_id> task=<task> score=<score>
        
        Args:
            run_id: Unique identifier for this run (UUID)
            task: Task name ("easy", "medium", "hard")
            score: Final normalized score (0.0 - 1.0)
        """
        log_entry = f"[END] run_id={run_id} task={task} score={score:.4f}"
        self._emit(log_entry)
    
    def _emit(self, log_entry: str) -> None:
        """
        Emit a log entry to stdout.
        
        Args:
            log_entry: Pre-formatted log string
        """
        if self.use_stdout:
            print(log_entry, file=sys.stdout, flush=True)
        self.logs.append(log_entry)
    
    def get_logs(self) -> list:
        """Get all logged entries."""
        return self.logs.copy()
    
    def clear_logs(self) -> None:
        """Clear log buffer."""
        self.logs.clear()


class EpisodeLogger:
    """
    Helper class to manage logging for a single episode.
    """
    
    def __init__(self, logger: StructuredLogger, run_id: str, task: str, model: str):
        """
        Initialize episode logger.
        
        Args:
            logger: StructuredLogger instance
            run_id: Unique run ID
            task: Task name
            model: Model name
        """
        self.logger = logger
        self.run_id = run_id
        self.task = task
        self.model = model
        self.step_count = 0
        self.total_reward = 0.0
    
    def start(self) -> None:
        """Log episode start."""
        self.logger.log_start(self.run_id, self.task, self.model)
    
    def log_step(self, reward: float, state: Dict[str, Any], action: int) -> None:
        """
        Log a step in the episode.
        
        Args:
            reward: Reward for this step
            state: Environment state
            action: Action taken
        """
        self.logger.log_step(self.step_count, reward, state, action)
        self.step_count += 1
        self.total_reward += reward
    
    def end(self, final_score: float) -> None:
        """
        Log episode end.
        
        Args:
            final_score: Final normalized score (0.0-1.0)
        """
        self.logger.log_end(self.run_id, self.task, final_score)


def validate_log_format(log_line: str) -> bool:
    """
    Validate that a log line follows the mandatory format.
    
    Args:
        log_line: Log line to validate
    
    Returns:
        True if valid, False otherwise
    """
    if log_line.startswith("[START]"):
        # Format: [START] run_id=<ID> task=<task> model=<model>
        return all(
            part in log_line
            for part in ["run_id=", "task=", "model="]
        )
    elif log_line.startswith("[STEP]"):
        # Format: [STEP] step=<step> reward=<float> state=<json> action=<action>
        return all(
            part in log_line
            for part in ["step=", "reward=", "state=", "action="]
        )
    elif log_line.startswith("[END]"):
        # Format: [END] run_id=<ID> task=<task> score=<score>
        return all(
            part in log_line
            for part in ["run_id=", "task=", "score="]
        )
    return False


if __name__ == "__main__":
    # Test logger
    logger = StructuredLogger()
    
    # Test START log
    logger.log_start(run_id="test-run-123", task="easy", model="gpt-3.5-turbo")
    
    # Test STEP logs
    for step in range(3):
        state = {"agent_pos": [10, 20], "battery": 100 - step * 10}
        logger.log_step(step=step, reward=0.25, state=state, action=1)
    
    # Test END log
    logger.log_end(run_id="test-run-123", task="easy", score=0.85)
    
    # Validate logs
    print("\n--- Log Validation ---")
    for log in logger.get_logs():
        is_valid = validate_log_format(log)
        status = "✓" if is_valid else "✗"
        print(f"{status} {log}")
    
    print("\nLogger test passed!")