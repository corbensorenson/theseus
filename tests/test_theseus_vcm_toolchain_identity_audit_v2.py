from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_toolchain_identity_audit as owner  # noqa: E402

REPORT = owner.audit(ROOT / "configs" / "theseus_vcm_toolchain_identity_audit_v2.json")


def test_all_corrected_evaluator_ecosystem_locks_have_exact_managers() -> None:
    assert REPORT["trigger_state"] == "GREEN"
    assert REPORT["observations"]["tasks_with_complete_lock_manager_identity"] == 48
    assert REPORT["observations"]["tasks_with_missing_lock_manager_identity"] == 0
    assert REPORT["missing_tool_task_indices"] == {}
    assert REPORT["parent_target_or_evaluator_executions"] == 0


def test_task_57_no_longer_claims_an_npm_lock_manager_for_python() -> None:
    row = REPORT["rows"][56]
    assert row["query_language"] == "Python"
    assert row["lock_managers"] == []
    assert row["lock_not_required_for_scoped_evaluator"] is True
    assert row["evaluator_execution_ready"] is False
