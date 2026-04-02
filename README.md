---
title: Data Privacy Auditor
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
---
# Data Privacy & Integrity Auditor

**A Reinforcement Learning Environment for Automated Data Sanitization**

## Environment Description & Motivation

The **Data Privacy & Integrity Auditor** is a production-ready Reinforcement Learning (RL) environment designed to train agents in the critical task of automated data sanitization. Built on the OpenEnv framework, this environment addresses the growing necessity of GDPR and PII (Personally Identifiable Information) compliance in large-scale datasets.

Manual data cleaning is not only time-consuming but also highly prone to human error, introducing significant compliance risks. This environment trains an AI agent to act as an automated auditor, learning to make precise decisions that balance **privacy** (effectively masking sensitive data) with **utility** (maintaining data integrity and structural correctness). 

## Action Space Definition

The agent interacts with the environment by outputting strict JSON objects. The `AuditAction` space consists of three highly specific operations:

*   **`mask_pii`**: Redacts sensitive information. Validated via Regex (targets Name, Email, Phone, or unstructured notes).
*   **`delete_duplicate`**: Removes duplicate entries. The agent must successfully identify and target the later occurrence (higher row index) of a duplicated record.
*   **`fix_type`**: Corrects data integrity errors by supplying a valid replacement value (e.g., converting a negative salary to positive, formatting an email correctly).

**Reward System:**
The environment provides immediate, deterministic feedback to guide the agent's policy:
*   **`+1.0`**: Successful, valid fix of an identified issue.
*   **`+0.1`**: Partial progress (e.g., replacing a string with an integer, but the integer is still out of range).
*   **`-0.5`**: Incorrect actions, out-of-bounds indices, or attempting to fix a non-existent issue.

## Observation Space Definition

The state space is returned to the agent via the `AuditObservation` model. To optimize operations for LLM-based agents, the environment utilizes a **Windowed Snapshot** mechanism.

Instead of passing the entire dataset (which would quickly exhaust context windows and increase latency), the observation provides a truncated JSON snapshot comprising `center_row ± 2` rows. This localized view ensures token efficiency while providing the agent with sufficient structural context to identify duplicates and contextual errors. The observation also includes the current reward, completion status, and a count of remaining issues.

## Task Descriptions & Difficulty

The environment features three curriculum-based difficulty levels to systematically evaluate an agent's auditing capabilities:

| Task Level | Dataset Size | Objective | Description |
| :--- | :--- | :--- | :--- |
| **Easy** | 26 rows | Basic Data Hygiene | Remove 5 explicit duplicate rows and mask structured PII in standard fields (Name, Email, Phone). |
| **Medium** | 20 rows | Unstructured PII Discovery | Identify and mask embedded, unstructured PII (phone numbers, email addresses) hidden within free-text "Notes" columns. |
| **Hard** | 20 rows | Complex Data Integrity | Detect and resolve severe data-integrity violations, including negative salaries, non-numeric salaries ("REDACTED"), out-of-range User_IDs (e.g., 99), and malformed email addresses ("not_an_email.com"). |

## Setup & Usage Instructions

### Prerequisites
Ensure all dependencies are installed via `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 1. Start the Environment Server
Launch the FastAPI server to expose the OpenEnv endpoints. This must be running for the agent to step through the environment.
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 2
```

### 2. Run Inference
The inference script connects the environment to a Hugging Face hosted model. You must provide the mandatory environment variables.

**Required Environment Variables:**
*   `HF_TOKEN`: Your Hugging Face authentication token.
*   `API_BASE_URL`: The OpenAI-compatible endpoint router (default: `https://router.huggingface.co/v1`).
*   `MODEL_NAME`: The model registry name to use (default: `Qwen/Qwen2.5-72B-Instruct`).

**Execution:**
```bash
# Windows (PowerShell)
$env:HF_TOKEN="your_token_here"
python inference.py

# Linux/macOS
HF_TOKEN="your_token_here" python inference.py
```

## Baseline Scores

The designated baseline agent (`Qwen2.5-72B-Instruct` / `Qwen2.5-7B-Instruct`) demonstrates exceptional capability within this environment. 
*   **Easy Task**: Achieves a perfect `1.0` score with a 100% success rate on structured cleaning.
*   **Hard Task**: Handles multidimensional integrity reasoning with high precision, strictly adhering to the JSON output format and correctly tracking row indices despite dataset mutations.
