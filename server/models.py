"""
Data Privacy & Integrity Auditor — OpenEnv Models
Pydantic models for Action, Observation, and State.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Action — what the agent sends
# ---------------------------------------------------------------------------
class AuditAction(Action):
    """
    An action the agent can take on the dataset.

    action_type: one of 'mask_pii', 'delete_duplicate', 'fix_type'
    row:         0-based row index to target
    col:         column name (required for mask_pii, fix_type)
    value:       replacement value (required for fix_type)
    """
    action_type: str = ""
    row: int = 0
    col: Optional[str] = None
    value: Optional[str] = None

    def __str__(self) -> str:
        parts = [f"{self.action_type}(row={self.row}"]
        if self.col is not None:
            parts.append(f", col={self.col}")
        if self.value is not None:
            parts.append(f", value={self.value}")
        parts.append(")")
        return "".join(parts)


# ---------------------------------------------------------------------------
# Observation — what the environment returns
# ---------------------------------------------------------------------------
class AuditObservation(Observation):
    """
    Returned after every reset() / step().

    table_snapshot:   truncated JSON of the relevant rows (target row ± 2)
    reward:           reward earned by the last action (0.0 on reset)
    done:             whether the episode is finished
    message:          human-readable feedback string
    remaining_issues: how many issues are still open
    columns:          list of column names in the dataset
    total_rows:       total number of rows currently in the dataset
    """
    table_snapshot: str = "[]"
    reward: float = 0.0
    done: bool = False
    message: str = ""
    remaining_issues: int = 0
    columns: List[str] = []
    total_rows: int = 0

    # -- helpers for building truncated snapshots --------------------------

    @staticmethod
    def build_snapshot(df, center_row: int = 0, window: int = 2) -> str:
        """Return a JSON string containing only *center_row ± window* rows.
        Each record also carries its real DataFrame index so the agent can
        address rows unambiguously.
        """
        if df.empty:
            return "[]"

        lo = max(0, center_row - window)
        hi = min(len(df), center_row + window + 1)
        subset = df.iloc[lo:hi].copy()
        subset.insert(0, "__row_index__", range(lo, hi))
        return subset.to_json(orient="records", default_handler=str)


# ---------------------------------------------------------------------------
# State — internal bookkeeping exposed via the state property
# ---------------------------------------------------------------------------
class AuditState(State):
    """Serialisable snapshot of the environment's internal bookkeeping.

    Inherits from openenv State which already has:
      - episode_id: Optional[str]
      - step_count: int (default 0)
    """
    total_reward: float = 0.0
    issues_found: int = 0
    issues_fixed: int = 0
    done: bool = False
