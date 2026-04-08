"""
FastAPI Web Application
Provides REST API endpoints for the Disaster Rescue RL Environment.
Compatible with Hugging Face Spaces deployment.
"""

import os
import sys
import json
import uuid
from typing import Dict, Any, Optional, List
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from environment.disaster_env import DisasterRescueEnv
from agents.baseline_agent import get_agent
from utils.graders import get_grader, get_grader_metrics
from configs.task_config import get_all_difficulties, get_task_config
from utils.logger import StructuredLogger


# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class ResetRequest(BaseModel):
    """Request schema for environment reset."""
    difficulty: str = Field("easy", description="Task difficulty: easy, medium, hard")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class ResetResponse(BaseModel):
    """Response schema for environment reset."""
    session_id: str = Field(..., description="Unique session identifier")
    difficulty: str = Field(..., description="Task difficulty")
    observation_shape: List[int] = Field(..., description="Shape of observation")
    action_space: int = Field(..., description="Number of possible actions")
    initial_info: Dict[str, Any] = Field(..., description="Initial environment info")


class StepRequest(BaseModel):
    """Request schema for environment step."""
    session_id: str = Field(..., description="Session identifier")
    action: int = Field(..., description="Action index (0-7)")


class StepResponse(BaseModel):
    """Response schema for environment step."""
    session_id: str = Field(..., description="Session identifier")
    step: int = Field(..., description="Step number")
    reward: float = Field(..., description="Reward for this step")
    terminated: bool = Field(..., description="Episode termination flag")
    truncated: bool = Field(..., description="Episode truncation flag")
    info: Dict[str, Any] = Field(..., description="Additional info")


class StateResponse(BaseModel):
    """Response schema for environment state."""
    session_id: str = Field(..., description="Session identifier")
    state: Dict[str, Any] = Field(..., description="Full environment state")


class RenderResponse(BaseModel):
    """Response schema for rendering."""
    session_id: str = Field(..., description="Session identifier")
    image_base64: str = Field(..., description="Base64-encoded RGB image")


class EvaluateRequest(BaseModel):
    """Request schema for evaluation."""
    agent_type: str = Field("greedy", description="Type of agent")
    difficulty: str = Field("easy", description="Task difficulty")
    num_episodes: int = Field(1, description="Number of episodes")
    seed: Optional[int] = Field(None, description="Random seed")


class EvaluateResponse(BaseModel):
    """Response schema for evaluation."""
    agent_type: str = Field(..., description="Agent type")
    difficulty: str = Field(..., description="Task difficulty")
    scores: Dict[str, float] = Field(..., description="Score statistics")
    rewards: Dict[str, float] = Field(..., description="Reward statistics")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    message: str = Field(..., description="Status message")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Disaster Rescue RL Environment API",
    description="REST API for autonomous drone search & rescue simulation",
    version="1.0.0",
)

# Add CORS middleware for Hugging Face Spaces compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Session Management
# ============================================================================

# In-memory session storage (for demo; use Redis for production)
SESSIONS: Dict[str, DisasterRescueEnv] = {}
SESSION_METADATA: Dict[str, Dict[str, Any]] = {}


def create_session(difficulty: str, seed: Optional[int] = None) -> str:
    """
    Create a new environment session.
    
    Args:
        difficulty: Task difficulty
        seed: Random seed
    
    Returns:
        Session ID
    """
    session_id = str(uuid.uuid4())
    
    env = DisasterRescueEnv(difficulty=difficulty, seed=seed)
    obs, info = env.reset(seed=seed)
    
    SESSIONS[session_id] = env
    SESSION_METADATA[session_id] = {
        "difficulty": difficulty,
        "seed": seed,
        "created_at": str(np.datetime64("now")),
        "step_count": 0,
        "total_reward": 0.0,
    }
    
    return session_id


def get_session(session_id: str) -> DisasterRescueEnv:
    """
    Get environment by session ID.
    
    Args:
        session_id: Session identifier
    
    Returns:
        DisasterRescueEnv instance
    
    Raises:
        HTTPException: If session not found
    """
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return SESSIONS[session_id]


def delete_session(session_id: str) -> None:
    """
    Delete a session.
    
    Args:
        session_id: Session identifier
    """
    if session_id in SESSIONS:
        del SESSIONS[session_id]
    if session_id in SESSION_METADATA:
        del SESSION_METADATA[session_id]


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """
    Root endpoint - API information.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        message="Disaster Rescue RL Environment API is running",
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        message="API is operational",
    )


@app.get("/info")
async def get_info() -> Dict[str, Any]:
    """
    Get API information and available tasks.
    """
    return {
        "api_version": "1.0.0",
        "available_difficulties": get_all_difficulties(),
        "available_agents": ["random", "exploration", "grid_search", "greedy", "hybrid"],
        "action_meanings": {
            0: "North",
            1: "Northeast",
            2: "East",
            3: "Southeast",
            4: "South",
            5: "Southwest",
            6: "West",
            7: "Northwest",
        },
        "observation_shape": [64, 64, 5],
        "observation_channels": {
            0: "Agent position",
            1: "Victims (alive)",
            2: "Hazards (intensity)",
            3: "Resources",
            4: "Visibility (fog)",
        },
        "max_sessions": 100,
        "current_sessions": len(SESSIONS),
    }


# ============================================================================
# Environment Endpoints
# ============================================================================

@app.post("/reset", response_model=ResetResponse)
async def reset_environment(request: ResetRequest) -> ResetResponse:
    """
    Reset environment and create new session.
    
    Args:
        request: ResetRequest with difficulty and seed
    
    Returns:
        ResetResponse with session info
    """
    # Validate difficulty
    if request.difficulty not in get_all_difficulties():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown difficulty: {request.difficulty}"
        )
    
    # Create session
    session_id = create_session(request.difficulty, request.seed)
    env = get_session(session_id)
    obs, info = env.reset(seed=request.seed)
    
    return ResetResponse(
        session_id=session_id,
        difficulty=request.difficulty,
        observation_shape=list(obs.shape),
        action_space=8,
        initial_info=info,
    )


@app.post("/step", response_model=StepResponse)
async def step_environment(request: StepRequest) -> StepResponse:
    """
    Take one step in environment.
    
    Args:
        request: StepRequest with session_id and action
    
    Returns:
        StepResponse with step results
    """
    # Get session
    env = get_session(request.session_id)
    
    # Validate action
    if not 0 <= request.action < 8:
        raise HTTPException(status_code=400, detail="Action must be in range [0, 7]")
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(request.action)
    
    # Update metadata
    SESSION_METADATA[request.session_id]["step_count"] += 1
    SESSION_METADATA[request.session_id]["total_reward"] += float(reward)
    
    return StepResponse(
        session_id=request.session_id,
        step=SESSION_METADATA[request.session_id]["step_count"],
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=info,
    )


@app.get("/state/{session_id}", response_model=StateResponse)
async def get_state(session_id: str) -> StateResponse:
    """
    Get current environment state.
    
    Args:
        session_id: Session identifier
    
    Returns:
        StateResponse with full environment state
    """
    env = get_session(session_id)
    return StateResponse(
        session_id=session_id,
        state=env.state(),
    )


@app.post("/render/{session_id}")
async def render_environment(session_id: str) -> Dict[str, str]:
    """
    Render environment as RGB image (base64).
    
    Args:
        session_id: Session identifier
    
    Returns:
        Dictionary with base64-encoded image
    """
    env = get_session(session_id)
    env.render_mode = "rgb_array"
    img = env.render()
    
    if img is None:
        raise HTTPException(status_code=500, detail="Rendering failed")
    
    # Encode image as base64
    import base64
    from PIL import Image
    from io import BytesIO
    
    pil_img = Image.fromarray(img.astype(np.uint8))
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return {
        "session_id": session_id,
        "image_base64": img_base64,
        "format": "png",
    }


@app.delete("/session/{session_id}")
async def delete_environment(session_id: str) -> Dict[str, str]:
    """
    Delete a session (cleanup).
    
    Args:
        session_id: Session identifier
    
    Returns:
        Success message
    """
    delete_session(session_id)
    return {"message": f"Session {session_id} deleted"}


# ============================================================================
# Evaluation Endpoints
# ============================================================================

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_agent(request: EvaluateRequest) -> EvaluateResponse:
    """
    Evaluate an agent across multiple episodes.
    
    Args:
        request: EvaluateRequest with agent and difficulty
    
    Returns:
        EvaluateResponse with aggregated metrics
    """
    # Validate inputs
    if request.agent_type not in ["random", "exploration", "grid_search", "greedy", "hybrid"]:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {request.agent_type}")
    
    if request.difficulty not in get_all_difficulties():
        raise HTTPException(status_code=400, detail=f"Unknown difficulty: {request.difficulty}")
    
    # Create environment and agent
    env = DisasterRescueEnv(difficulty=request.difficulty, seed=request.seed)
    agent = get_agent(request.agent_type, env=env, seed=request.seed)
    grader = get_grader(request.difficulty)
    
    scores = []
    rewards = []
    
    # Run episodes
    for episode in range(request.num_episodes):
        obs, _ = env.reset(seed=request.seed + episode if request.seed else None)
        agent.reset()
        
        total_reward = 0.0
        step = 0
        max_steps = get_task_config(request.difficulty)["max_steps"]
        
        while step < max_steps:
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
            
            if terminated or truncated:
                break
        
        # Grade episode
        score = grader.grade(env.state())
        scores.append(score)
        rewards.append(total_reward)
    
    return EvaluateResponse(
        agent_type=request.agent_type,
        difficulty=request.difficulty,
        scores={
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        },
        rewards={
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
        },
    )


# ============================================================================
# Session Management Endpoints
# ============================================================================

@app.get("/sessions")
async def list_sessions() -> Dict[str, Any]:
    """
    List all active sessions.
    """
    return {
        "num_sessions": len(SESSIONS),
        "sessions": [
            {
                "session_id": sid,
                "difficulty": SESSION_METADATA[sid]["difficulty"],
                "step_count": SESSION_METADATA[sid]["step_count"],
                "total_reward": SESSION_METADATA[sid]["total_reward"],
                "created_at": SESSION_METADATA[sid]["created_at"],
            }
            for sid in SESSIONS.keys()
        ],
    }


@app.post("/cleanup")
async def cleanup_sessions() -> Dict[str, str]:
    """
    Clean up all sessions (for development/testing).
    """
    num_deleted = len(SESSIONS)
    SESSIONS.clear()
    SESSION_METADATA.clear()
    return {"message": f"Deleted {num_deleted} sessions"}


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    return JSONResponse(
        status_code=500,
        content={"error": f"Internal server error: {str(exc)}"},
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))  # Hugging Face Spaces default
    
    print(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )