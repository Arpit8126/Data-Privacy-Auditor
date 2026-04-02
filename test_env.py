"""
Dry-run test for the Data Privacy & Integrity Auditor environment.
Tests all 3 tasks, all action types, and reward logic.
"""
from main import DataPrivacyAuditorEnv
from server.models import AuditAction

def test_easy():
    print("=" * 60)
    print("TEST: easy task")
    env = DataPrivacyAuditorEnv()
    obs = env.reset(task_name="easy")
    print(f"  Issues found: {env.state.issues_found}")
    print(f"  Remaining: {obs.remaining_issues}")
    print(f"  Rows: {obs.total_rows}")
    print(f"  Done: {obs.done}")
    assert obs.total_rows == 25, f"Expected 25 rows, got {obs.total_rows}"
    assert obs.remaining_issues > 0

    # Delete a duplicate (row 21 is a duplicate of row 0)
    obs = env.step(AuditAction(action_type="delete_duplicate", row=21))
    print(f"  delete_duplicate(21): reward={obs.reward}, msg={obs.message}")
    assert obs.reward == 1.0, f"Expected +1.0, got {obs.reward}"

    # Mask PII
    obs = env.step(AuditAction(action_type="mask_pii", row=0, col="Name"))
    print(f"  mask_pii(0, Name): reward={obs.reward}, msg={obs.message}")
    assert obs.reward == 1.0, f"Expected +1.0, got {obs.reward}"

    # Invalid action: mask PII on already-masked cell
    obs = env.step(AuditAction(action_type="mask_pii", row=0, col="Name"))
    print(f"  mask_pii(0, Name) again: reward={obs.reward}, msg={obs.message}")
    assert obs.reward == -0.5, f"Expected -0.5, got {obs.reward}"

    print("  PASS\n")

def test_medium():
    print("=" * 60)
    print("TEST: medium task")
    env = DataPrivacyAuditorEnv()
    obs = env.reset(task_name="medium")
    print(f"  Issues found: {env.state.issues_found}")
    print(f"  Remaining: {obs.remaining_issues}")
    assert obs.remaining_issues > 0

    # Mask PII in Notes column (row 0 has "Direct line is 999.604.9027.")
    obs = env.step(AuditAction(action_type="mask_pii", row=0, col="Notes"))
    print(f"  mask_pii(0, Notes): reward={obs.reward}, msg={obs.message}")
    assert obs.reward == 1.0, f"Expected +1.0, got {obs.reward}"

    print("  PASS\n")

def test_hard():
    print("=" * 60)
    print("TEST: hard task")
    env = DataPrivacyAuditorEnv()
    obs = env.reset(task_name="hard")
    print(f"  Issues found: {env.state.issues_found}")
    print(f"  Remaining: {obs.remaining_issues}")
    issue_summary = env.get_issue_summary()
    print(f"  Issue summary:\n{issue_summary}")

    # Fix negative salary at row 2 (User_ID 1002, Salary -50000)
    obs = env.step(AuditAction(action_type="fix_type", row=2, col="Salary", value="50000"))
    print(f"  fix_type(2, Salary, 50000): reward={obs.reward}, msg={obs.message}")
    assert obs.reward == 1.0, f"Expected +1.0, got {obs.reward}"

    # Fix non-numeric salary at row 5 (REDACTED)
    obs = env.step(AuditAction(action_type="fix_type", row=5, col="Salary", value="60000"))
    print(f"  fix_type(5, Salary, 60000): reward={obs.reward}, msg={obs.message}")
    assert obs.reward == 1.0, f"Expected +1.0, got {obs.reward}"

    # Fix out-of-range User_ID at row 10 (User_ID 99)
    obs = env.step(AuditAction(action_type="fix_type", row=10, col="User_ID", value="1010"))
    print(f"  fix_type(10, User_ID, 1010): reward={obs.reward}, msg={obs.message}")
    assert obs.reward == 1.0, f"Expected +1.0, got {obs.reward}"

    # Fix invalid email at row 15 (not_an_email.com)
    obs = env.step(AuditAction(action_type="fix_type", row=15, col="Email", value="susan.hopkins@example.com"))
    print(f"  fix_type(15, Email, susan.hopkins@example.com): reward={obs.reward}, msg={obs.message}")
    assert obs.reward == 1.0, f"Expected +1.0, got {obs.reward}"

    print("  PASS\n")

def test_done_condition():
    """Verify episode ends when remaining_issues == 0."""
    print("=" * 60)
    print("TEST: done condition (exhaust all issues on easy)")
    env = DataPrivacyAuditorEnv()
    obs = env.reset(task_name="easy")
    total = obs.remaining_issues
    steps = 0
    while not obs.done and steps < 200:
        steps += 1
        # Get first open issue
        open_issues = [i for i in env._issues if not i.fixed]
        if not open_issues:
            break
        issue = open_issues[0]
        if issue.kind == "duplicate":
            obs = env.step(AuditAction(action_type="delete_duplicate", row=issue.row))
        elif issue.kind == "pii":
            obs = env.step(AuditAction(action_type="mask_pii", row=issue.row, col=issue.col))
        elif issue.kind == "integrity":
            obs = env.step(AuditAction(action_type="fix_type", row=issue.row, col=issue.col, value="1010"))
    print(f"  Completed in {steps} steps, done={obs.done}, remaining={obs.remaining_issues}")
    assert obs.done == True, f"Expected done=True, got {obs.done}"
    assert obs.remaining_issues == 0, f"Expected 0 remaining, got {obs.remaining_issues}"
    print("  PASS\n")

def test_snapshot_truncation():
    print("=" * 60)
    print("TEST: snapshot truncation")
    env = DataPrivacyAuditorEnv()
    obs = env.reset(task_name="easy")
    import json
    rows = json.loads(obs.table_snapshot)
    print(f"  Snapshot rows: {len(rows)} (should be <= 5)")
    assert len(rows) <= 5, f"Expected <= 5 rows, got {len(rows)}"
    print("  PASS\n")

if __name__ == "__main__":
    test_easy()
    test_medium()
    test_hard()
    test_done_condition()
    test_snapshot_truncation()
    print("ALL TESTS PASSED")
