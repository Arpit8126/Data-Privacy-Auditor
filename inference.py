"""
Data Privacy & Integrity Auditor — Inference Script
Drives the environment using the OpenAI-compatible chat completions API
through the Hugging Face inference router.

STDOUT log format (mandatory for OpenEnv validator):
  [START] task=<task> env=data-privacy-auditor model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
from typing import List

from openai import OpenAI

from main import DataPrivacyAuditorEnv
from models import AuditAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENV_NAME = "data-privacy-auditor"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 50
SUCCESS_SCORE_THRESHOLD = 0.8

SYSTEM_PROMPT = """\
You are a Data Privacy & Integrity Auditor agent.
You receive a snapshot of a dataset and a list of open issues.
Your job is to fix every issue by outputting ONE action per turn as a JSON object.

Available actions (pick exactly one):
  {"action_type": "mask_pii",         "row": <int>, "col": "<column>"}
  {"action_type": "delete_duplicate", "row": <int>}
  {"action_type": "fix_type",         "row": <int>, "col": "<column>", "value": "<new_value>"}

Rules:
- row is 0-based.
- For mask_pii: target a cell that contains personal data (Name, Email, Phone, or embedded PII in Notes).
- For delete_duplicate: target the LATER duplicate row (higher index).
- For fix_type: supply a corrected value (e.g., a positive salary, a valid email, a User_ID in 1000-1999).
- Return ONLY the JSON object, no extra text.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool(val: bool) -> str:
    """Return lowercase 'true' or 'false' for OpenEnv validator compliance."""
    return "true" if val else "false"


def parse_action(raw: str) -> AuditAction:
    """Parse the LLM response into an AuditAction.

    Handles responses wrapped in markdown code fences (```json ... ```)
    as well as bare JSON.
    """
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    obj = json.loads(text)
    return AuditAction(
        action_type=obj.get("action_type", ""),
        row=int(obj.get("row", 0)),
        col=obj.get("col"),
        value=obj.get("value"),
    )


def get_model_message(
    client: OpenAI,
    messages: List[dict],
) -> str:
    """Call the chat completions API and return the assistant's content."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=256,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_name: str) -> None:
    """Run a single task end-to-end and print structured logs to stdout."""
    env = DataPrivacyAuditorEnv()
    obs = env.reset(task_name=task_name)

    # MAX_TOTAL_REWARD = 1.0 per issue detected at reset
    max_total_reward = float(max(env.state.issues_found, 1))

    # [START]
    print(
        f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}",
        flush=True,
    )

    messages: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    step_num = 0
    rewards: List[float] = []

    try:
        while not obs.done and step_num < MAX_STEPS:
            step_num += 1

            # Build the user prompt with current state
            user_msg = (
                f"Step {step_num}.\n"
                f"Table snapshot (rows around focus area):\n{obs.table_snapshot}\n\n"
                f"Columns: {obs.columns}\n"
                f"Total rows: {obs.total_rows}\n"
                f"Remaining issues: {obs.remaining_issues}\n\n"
                f"{env.get_issue_summary()}\n\n"
                f"Return your next action as a JSON object."
            )
            messages.append({"role": "user", "content": user_msg})

            error_msg: str | None = None
            action_str = ""

            try:
                raw = get_model_message(client, messages)
                messages.append({"role": "assistant", "content": raw})

                action = parse_action(raw)
                action_str = str(action)
                obs = env.step(action)
                rewards.append(obs.reward)
            except json.JSONDecodeError as e:
                error_msg = f"JSON parse error: {e}"
                rewards.append(0.0)
            except Exception as e:
                error_msg = str(e)
                rewards.append(0.0)

            # [STEP] — done is lowercase, error is null if no error
            print(
                f"[STEP] step={step_num} "
                f"action={action_str} "
                f"reward={rewards[-1]:.2f} "
                f"done={_bool(obs.done)} "
                f"error={error_msg if error_msg else 'null'}",
                flush=True,
            )

    finally:
        # [END] — ALWAYS printed, even on crash
        score = max(0.0, min(1.0, sum(rewards) / max_total_reward))
        success = score >= SUCCESS_SCORE_THRESHOLD
        reward_csv = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={_bool(success)} "
            f"steps={step_num} "
            f"score={score:.3f} "
            f"rewards={reward_csv}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        print(
            "ERROR: HF_TOKEN environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=api_key,
    )

    tasks_to_run = sys.argv[1:] if len(sys.argv) > 1 else TASKS
    for task in tasks_to_run:
        run_task(client, task)


if __name__ == "__main__":
    main()
