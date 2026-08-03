from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_toolchain_identity_audit as owner  # noqa: E402

REPORT = owner.audit(ROOT / "configs" / "theseus_vcm_toolchain_identity_audit.json")


def test_exact_local_tools_are_bound_without_repository_or_model_calls() -> None:
    assert REPORT["trigger_state"] == "GREEN"
    assert REPORT["observations"]["available_tool_count"] == 11
    assert REPORT["observations"]["missing_tool_count"] == 1
    assert REPORT["parent_target_or_evaluator_executions"] == 0
    assert REPORT["candidate_or_control_calls"] == 0
    assert REPORT["external_reference_calls"] == 0


def test_only_yarn_locked_task_has_a_manager_identity_residual() -> None:
    assert REPORT["missing_tool_task_indices"] == {"yarn": [4]}
    rows = {row["index"]: row for row in REPORT["rows"]}
    assert rows[4]["missing_lock_managers"] == ["yarn"]
    assert sum(row["lock_manager_identity_complete"] for row in rows.values()) == 48
    assert all(row["evaluator_execution_ready"] is False for row in rows.values())
