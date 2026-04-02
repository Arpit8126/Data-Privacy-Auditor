"""
Data Privacy & Integrity Auditor — FastAPI Server
Exposes the environment over HTTP for OpenEnv compatibility.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from main import DataPrivacyAuditorEnv
from models import AuditAction, AuditObservation, AuditState

app = FastAPI(
    title="Data Privacy & Integrity Auditor",
    version="1.0.0",
    description="OpenEnv RL environment for data privacy auditing",
)

# Single environment instance (stateful per container)
env = DataPrivacyAuditorEnv()


# --- Request / Response schemas -------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"

class StepRequest(BaseModel):
    action_type: str
    row: int = 0
    col: Optional[str] = None
    value: Optional[str] = None

class ObservationResponse(BaseModel):
    table_snapshot: str
    reward: float
    done: bool
    message: str
    remaining_issues: int
    columns: list
    total_rows: int

class StateResponse(BaseModel):
    step_count: int
    total_reward: float
    issues_found: int
    issues_fixed: int
    done: bool


# --- Endpoints ------------------------------------------------------------

@app.post("/reset", response_model=ObservationResponse)
def reset(req: Optional[ResetRequest] = None): # Make the whole request optional
    """Reset the environment for the given task."""
    try:
        # If req is null, default to "easy"
        t_name = req.task_id if (req and req.task_id) else "easy"
        obs = env.reset(task_name=t_name) 
        return _obs_to_dict(obs)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/step", response_model=ObservationResponse)
def step(req: StepRequest):
    """Execute a single action."""
    action = AuditAction(
        action_type=req.action_type,
        row=req.row,
        col=req.col,
        value=req.value,
    )
    obs = env.step(action)
    return _obs_to_dict(obs)


@app.get("/state", response_model=StateResponse)
def get_state():
    """Return the current environment state."""
    s = env.state
    return StateResponse(
        step_count=s.step_count,
        total_reward=s.total_reward,
        issues_found=s.issues_found,
        issues_fixed=s.issues_fixed,
        done=s.done,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# --- Helpers --------------------------------------------------------------

def _obs_to_dict(obs: AuditObservation) -> dict:
    return {
        "table_snapshot": obs.table_snapshot,
        "reward": obs.reward,
        "done": obs.done,
        "message": obs.message,
        "remaining_issues": obs.remaining_issues,
        "columns": obs.columns,
        "total_rows": obs.total_rows,
    }
