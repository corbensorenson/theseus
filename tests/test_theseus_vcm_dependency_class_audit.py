from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_dependency_class_audit as owner  # noqa: E402

REPORT = owner.audit(ROOT / "configs" / "theseus_vcm_dependency_class_audit.json")


def test_all_tasks_receive_a_static_dependency_class_without_execution() -> None:
    assert REPORT["trigger_state"] == "GREEN"
    assert REPORT["observations"]["task_count"] == 62
    assert REPORT["observations"]["tasks_dependency_classified"] == 62
    assert REPORT["observations"]["tasks_evaluator_execution_ready"] == 0
    assert REPORT["parent_target_or_evaluator_executions"] == 0
    assert REPORT["candidate_or_control_calls"] == 0
    assert REPORT["external_reference_calls"] == 0


def test_lock_waivers_are_scoped_to_static_evaluator_closure() -> None:
    waived = {
        row["index"] for row in REPORT["rows"]
        if row["dependency_class"] == "LOCK_NOT_REQUIRED_FOR_STATIC_EVALUATOR_CLOSURE"
    }
    assert waived == {1, 10, 20, 22, 23, 27, 58}
    assert all(not row["external_dependencies_excluding_harness"] for row in REPORT["rows"] if row["index"] in waived)
    assert all(row["evaluator_execution_ready"] is False for row in REPORT["rows"] if row["index"] in waived)


def test_unlocked_external_dependency_tasks_fail_closed_to_immutable_resolution() -> None:
    required = {
        row["index"] for row in REPORT["rows"]
        if row["dependency_class"] == "IMMUTABLE_RESOLUTION_REQUIRED"
    }
    assert required == {12, 13, 16, 25, 35, 56}
    assert all(row["external_dependencies_excluding_harness"] for row in REPORT["rows"] if row["index"] in required)
