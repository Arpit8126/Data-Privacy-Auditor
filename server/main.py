"""
Data Privacy & Integrity Auditor — OpenEnv Environment
Implements the core RL environment with reset(), step(), and state.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from openenv.core.env_server import Environment

from server.models import AuditAction, AuditObservation, AuditState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
MAX_STEPS = 50
PII_MASK = "***MASKED***"

# Regex helpers
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", re.ASCII
)
_PHONE_RE = re.compile(
    r"""(?:\+?1[-.\s]?)?          # optional country code
        (?:\(?\d{3}\)?[-.\s]?)    # area code
        \d{3}[-.\s]?\d{4}        # local number
        (?:x\d+)?                 # optional extension
    """,
    re.VERBOSE | re.ASCII,
)
_VALID_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$", re.ASCII
)

# Columns that are expected to contain PII in structured fields
_PII_COLUMNS = {"Name", "Email", "Phone"}


# ---------------------------------------------------------------------------
# Issue descriptors
# ---------------------------------------------------------------------------
@dataclass
class Issue:
    """Single issue found in the dataset."""
    kind: str          # 'pii', 'duplicate', 'integrity'
    row: int           # 0-based row index
    col: str = ""      # column name (empty for duplicate)
    detail: str = ""   # human-readable detail
    fixed: bool = False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Make sure AuditState is imported from models at the top!
class DataPrivacyAuditorEnv(Environment[AuditAction, AuditObservation, AuditState]):
    """Gymnasium-style environment for data privacy and integrity auditing."""

    def __init__(self):
        super().__init__()
        self.df: pd.DataFrame = pd.DataFrame()
        self._original_df: pd.DataFrame = pd.DataFrame()
        self._issues: List[Issue] = []
        self._state = AuditState()
        self._task_name: str = ""
        self._rewards: List[float] = []

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, task_name: str = "easy", **kwargs) -> AuditObservation:
        """Load the CSV for *task_name* and scan for issues."""
        self._task_name = task_name
        csv_path = DATASET_DIR / f"{task_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self._original_df = self.df.copy()
        self._issues = []
        self._rewards = []

        # Scan for all issues
        self._detect_issues()

        self._state = AuditState(
            step_count=0,
            total_reward=0.0,
            issues_found=len(self._issues),
            issues_fixed=0,
            done=False,
        )

        remaining = self._remaining_issues()
        return AuditObservation(
            table_snapshot=AuditObservation.build_snapshot(self.df, center_row=0),
            reward=0.0,
            done=False,
            message=f"Environment reset for task '{task_name}'. Found {len(self._issues)} issues.",
            remaining_issues=remaining,
            columns=list(self.df.columns),
            total_rows=len(self.df),
        )

    def step(self, action: AuditAction, timeout_s: Optional[float] = None, **kwargs) -> AuditObservation:
        """Execute *action* and return the observation."""
        if self._state.done:
            return self._make_obs(
                reward=0.0,
                message="Episode already finished.",
                center_row=0,
            )

        self._state.step_count += 1
        reward = 0.0
        message = ""

        try:
            if action.action_type == "mask_pii":
                reward, message = self._handle_mask_pii(action)
            elif action.action_type == "delete_duplicate":
                reward, message = self._handle_delete_duplicate(action)
            elif action.action_type == "fix_type":
                reward, message = self._handle_fix_type(action)
            else:
                reward = -0.5
                message = f"Unknown action type: {action.action_type}"
        except Exception as exc:
            reward = -0.5
            message = f"Action error: {exc}"

        self._state.total_reward += reward
        self._rewards.append(reward)

        remaining = self._remaining_issues()
        if remaining == 0 or self._state.step_count >= MAX_STEPS:
            self._state.done = True

        self._state.issues_fixed = sum(1 for i in self._issues if i.fixed)

        return self._make_obs(
            reward=reward,
            message=message,
            center_row=min(action.row, len(self.df) - 1) if len(self.df) > 0 else 0,
        )

    @property
    def state(self) -> AuditState:
        return self._state

    # ------------------------------------------------------------------
    # Action handlers  (update self.df IN-PLACE)
    # ------------------------------------------------------------------

    def _handle_mask_pii(self, action: AuditAction) -> Tuple[float, str]:
        row, col = action.row, action.col
        if col is None:
            return -0.5, "mask_pii requires a 'col' argument."
        if row < 0 or row >= len(self.df):
            return -0.5, f"Row {row} out of range (0..{len(self.df) - 1})."
        if col not in self.df.columns:
            return -0.5, f"Column '{col}' does not exist."

        # Check if there is an open PII issue matching this cell
        matched = self._find_issue("pii", row, col)
        if matched is None:
            return -0.5, f"No PII issue at row={row}, col={col}."

        # Mask in-place
        self.df.at[self.df.index[row], col] = PII_MASK
        matched.fixed = True
        return 1.0, f"Masked PII at row={row}, col={col}."

    def _handle_delete_duplicate(self, action: AuditAction) -> Tuple[float, str]:
        row = action.row
        if row < 0 or row >= len(self.df):
            return -0.5, f"Row {row} out of range (0..{len(self.df) - 1})."

        matched = self._find_issue("duplicate", row)
        if matched is None:
            return -0.5, f"Row {row} is not a flagged duplicate."

        # Drop in-place and re-index
        self.df.drop(self.df.index[row], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        matched.fixed = True

        # Adjust row indices of remaining unfixed issues that were below the deleted row
        for issue in self._issues:
            if not issue.fixed and issue.row > row:
                issue.row -= 1

        return 1.0, f"Deleted duplicate row {row}."

    def _handle_fix_type(self, action: AuditAction) -> Tuple[float, str]:
        row, col, value = action.row, action.col, action.value
        if col is None or value is None:
            return -0.5, "fix_type requires both 'col' and 'value' arguments."
        if row < 0 or row >= len(self.df):
            return -0.5, f"Row {row} out of range (0..{len(self.df) - 1})."
        if col not in self.df.columns:
            return -0.5, f"Column '{col}' does not exist."

        matched = self._find_issue("integrity", row, col)
        if matched is None:
            return -0.5, f"No integrity issue at row={row}, col={col}."

        # Attempt to apply the fix
        old_val = self.df.at[self.df.index[row], col]

        # Validate the proposed value based on the column
        reward, msg = self._validate_fix(col, value, matched)
        if reward > 0:
            self.df.at[self.df.index[row], col] = value
        return reward, msg

    def _validate_fix(self, col: str, value: str, issue: Issue) -> Tuple[float, str]:
        """Check whether *value* is a valid fix for the given issue."""
        if col == "Salary":
            try:
                numeric = float(value)
            except ValueError:
                return -0.5, f"Salary value '{value}' is not numeric."
            if numeric < 0:
                return 0.1, f"Salary {value} is still negative — partial progress."
            issue.fixed = True
            return 1.0, f"Fixed Salary to {value}."

        if col == "Email":
            if _VALID_EMAIL_RE.match(value):
                issue.fixed = True
                return 1.0, f"Fixed Email to {value}."
            return 0.1, f"Email '{value}' still invalid — partial progress."

        if col == "User_ID":
            try:
                uid = int(value)
            except ValueError:
                return -0.5, f"User_ID '{value}' is not an integer."
            if 1000 <= uid <= 1999:
                issue.fixed = True
                return 1.0, f"Fixed User_ID to {value}."
            return 0.1, f"User_ID {value} out of expected range — partial progress."

        # Generic fallback
        issue.fixed = True
        return 1.0, f"Fixed {col} to {value}."

    # ------------------------------------------------------------------
    # Issue detection
    # ------------------------------------------------------------------

    def _detect_issues(self) -> None:
        """Scan self.df and populate self._issues."""
        self._detect_duplicates()
        self._detect_pii()
        self._detect_integrity()

    def _detect_duplicates(self) -> None:
        dupes = self.df.duplicated(keep="first")
        for idx in dupes[dupes].index:
            row_pos = self.df.index.get_loc(idx)
            self._issues.append(Issue(
                kind="duplicate",
                row=row_pos,
                detail=f"Duplicate of an earlier row",
            ))

    def _detect_pii(self) -> None:
        for col in _PII_COLUMNS:
            if col not in self.df.columns:
                continue
            for row_pos in range(len(self.df)):
                cell = str(self.df.iat[row_pos, self.df.columns.get_loc(col)])
                if cell and cell != PII_MASK:
                    self._issues.append(Issue(
                        kind="pii",
                        row=row_pos,
                        col=col,
                        detail=f"PII in {col}",
                    ))

        # Also scan Notes column for embedded PII
        if "Notes" in self.df.columns:
            for row_pos in range(len(self.df)):
                cell = str(self.df.iat[row_pos, self.df.columns.get_loc("Notes")])
                has_email = bool(_EMAIL_RE.search(cell))
                has_phone = bool(_PHONE_RE.search(cell))
                if has_email or has_phone:
                    self._issues.append(Issue(
                        kind="pii",
                        row=row_pos,
                        col="Notes",
                        detail=f"Embedded PII in Notes ({'email' if has_email else ''}{',' if has_email and has_phone else ''}{'phone' if has_phone else ''})",
                    ))

    def _detect_integrity(self) -> None:
        if "Salary" in self.df.columns:
            for row_pos in range(len(self.df)):
                val = self.df.iat[row_pos, self.df.columns.get_loc("Salary")]
                try:
                    numeric = float(val)
                    if numeric < 0:
                        self._issues.append(Issue(
                            kind="integrity",
                            row=row_pos,
                            col="Salary",
                            detail=f"Negative salary: {val}",
                        ))
                except (ValueError, TypeError):
                    self._issues.append(Issue(
                        kind="integrity",
                        row=row_pos,
                        col="Salary",
                        detail=f"Non-numeric salary: {val}",
                    ))

        if "Email" in self.df.columns:
            for row_pos in range(len(self.df)):
                val = str(self.df.iat[row_pos, self.df.columns.get_loc("Email")])
                if val and not _VALID_EMAIL_RE.match(val):
                    # Only flag if not already a PII issue on the same cell
                    existing = self._find_issue("pii", row_pos, "Email")
                    if existing is None:
                        self._issues.append(Issue(
                            kind="integrity",
                            row=row_pos,
                            col="Email",
                            detail=f"Invalid email format: {val}",
                        ))

        if "User_ID" in self.df.columns:
            for row_pos in range(len(self.df)):
                val = self.df.iat[row_pos, self.df.columns.get_loc("User_ID")]
                try:
                    uid = int(val)
                    if uid < 1000 or uid > 1999:
                        self._issues.append(Issue(
                            kind="integrity",
                            row=row_pos,
                            col="User_ID",
                            detail=f"User_ID out of range: {uid}",
                        ))
                except (ValueError, TypeError):
                    self._issues.append(Issue(
                        kind="integrity",
                        row=row_pos,
                        col="User_ID",
                        detail=f"Non-integer User_ID: {val}",
                    ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_issue(self, kind: str, row: int, col: str = "") -> Issue | None:
        for issue in self._issues:
            if issue.kind == kind and issue.row == row and not issue.fixed:
                if kind == "duplicate" or issue.col == col:
                    return issue
        return None

    def _remaining_issues(self) -> int:
        return sum(1 for i in self._issues if not i.fixed)

    def _make_obs(self, reward: float, message: str, center_row: int) -> AuditObservation:
        remaining = self._remaining_issues()
        return AuditObservation(
            table_snapshot=AuditObservation.build_snapshot(self.df, center_row=center_row),
            reward=reward,
            done=self._state.done,
            message=message,
            remaining_issues=remaining,
            columns=list(self.df.columns),
            total_rows=len(self.df),
        )

    def get_issue_summary(self) -> str:
        """Return a human-readable summary of all open issues (for the LLM prompt)."""
        open_issues = [i for i in self._issues if not i.fixed]
        if not open_issues:
            return "No remaining issues."
        lines = [f"Open issues ({len(open_issues)}):"]
        for i, issue in enumerate(open_issues, 1):
            loc = f"row={issue.row}"
            if issue.col:
                loc += f", col={issue.col}"
            lines.append(f"  {i}. [{issue.kind}] {loc} — {issue.detail}")
        return "\n".join(lines)

    @property
    def rewards_list(self) -> List[float]:
        return list(self._rewards)
