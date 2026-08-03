from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_dependency_class_audit_v2 as owner  # noqa: E402

REPORT = owner.audit(ROOT / "configs" / "theseus_vcm_dependency_class_audit_v2.json")


def test_successor_changes_only_cross_ecosystem_task_57_without_execution() -> None:
    assert REPORT["trigger_state"] == "GREEN"
    assert REPORT["changed_indices"] == [57]
    assert REPORT["observations"]["tasks_with_exact_evaluator_ecosystem_lock_receipt"] == 48
    assert REPORT["observations"]["tasks_lock_not_required_for_static_evaluator_closure"] == 8
    assert REPORT["parent_target_or_evaluator_executions"] == 0


def test_task_57_does_not_credit_javascript_lock_to_python_evaluator() -> None:
    row = REPORT["rows"][56]
    assert row["query_language"] == "Python"
    assert row["evaluator_ecosystem_lock_receipts"] == []
    assert row["dependency_class"] == "LOCK_NOT_REQUIRED_FOR_STATIC_EVALUATOR_CLOSURE"
    assert row["external_dependencies_excluding_harness"] == []
    assert set(row["local_aliases_eliminated_from_external_dependencies"]) == {"common", "generated_paths", "merge_train", "tools"}
    assert row["evaluator_execution_ready"] is False
