from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_toolchain_compatibility_audit_v2 as owner  # noqa: E402

REPORT = owner.audit(ROOT / "configs" / "theseus_vcm_toolchain_compatibility_audit_v2.json")


def test_exact_pnpm_profile_changes_only_task_seven() -> None:
    assert REPORT["trigger_state"] == "GREEN"
    assert REPORT["observations"]["changed_task_indices"] == [7]
    row = next(row for row in REPORT["rows"] if row["index"] == 7)
    assert row["state"] == "COMPATIBLE_DECLARED_REQUIREMENTS"
    assert row["compatible_profile_ids"] == ["node22_20_pnpm10_32_1"]


def test_reclassification_preserves_denominator_and_zero_execution() -> None:
    assert REPORT["observations"]["locked_task_count"] == 48
    assert REPORT["observations"]["compatible_declared_requirement_task_count"] == 19
    assert REPORT["observations"]["no_declared_version_requirement_task_count"] == 16
    assert REPORT["observations"]["incompatible_declared_requirement_task_count"] == 13
    assert REPORT["observations"]["unresolved_requirement_syntax_task_count"] == 0
    assert REPORT["dependency_prefetch_executions"] == 0
    assert REPORT["repository_executions"] == 0
    assert REPORT["candidate_or_control_calls"] == 0
    assert REPORT["external_reference_calls"] == 0


def test_all_non_task_seven_states_match_the_immutable_predecessor() -> None:
    predecessor = json.loads((ROOT / "reports" / "theseus_vcm_toolchain_compatibility_audit.json").read_text())
    before = {row["index"]: row["state"] for row in predecessor["rows"]}
    after = {row["index"]: row["state"] for row in REPORT["rows"]}
    assert {index: state for index, state in before.items() if index != 7} == {
        index: state for index, state in after.items() if index != 7
    }
