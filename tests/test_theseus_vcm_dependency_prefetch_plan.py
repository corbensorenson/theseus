from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_dependency_prefetch_plan as owner  # noqa: E402

REPORT = owner.audit(ROOT / "configs" / "theseus_vcm_dependency_prefetch_plan.json")


def test_schedule_covers_exact_dependency_classes_without_execution() -> None:
    assert REPORT["trigger_state"] == "GREEN"
    assert REPORT["observations"]["locked_prefetch_planned_task_count"] == 48
    assert REPORT["observations"]["no_project_lock_required_task_count"] == 8
    assert REPORT["observations"]["immutable_resolution_required_task_count"] == 6
    assert REPORT["observations"]["prefetch_executions"] == 0
    assert REPORT["parent_target_or_evaluator_executions"] == 0


def test_multilock_tasks_select_the_evaluator_owned_root_and_manager() -> None:
    rows = {row["index"]: row for row in REPORT["rows"]}
    assert rows[4]["governing_lock"]["path"] == "frontend/yarn.lock"
    assert rows[4]["manager"] == "yarn"
    assert rows[41]["governing_lock"]["path"] == "package-lock.json"
    assert rows[41]["manager"] == "npm"
    assert rows[57]["prefetch_planned"] is False
    assert rows[57]["state"] == "NO_PROJECT_LOCK_REQUIRED_STATIC_CLOSURE"


def test_schedule_is_sequential_and_smallest_estimated_graph_first() -> None:
    schedule = REPORT["schedule"]
    assert [row["schedule_ordinal"] for row in schedule] == list(range(1, 49))
    assert all(row["package_estimate_known"] is True for row in schedule)
    estimates = [row["estimated_locked_package_count"] for row in schedule]
    assert estimates == sorted(estimates)
    assert all(row["repository_execution_authorized"] is False for row in schedule)


def test_bun_text_locks_are_measured_instead_of_scheduled_as_zero() -> None:
    rows = {row["index"]: row for row in REPORT["rows"]}
    assert [rows[index]["estimated_locked_package_count"] for index in (46, 48, 50, 61)] == [137, 1033, 3002, 284]
    assert all(rows[index]["package_estimate_method"] == "bun_text_lock_package_arrays" for index in (46, 48, 50, 61))
