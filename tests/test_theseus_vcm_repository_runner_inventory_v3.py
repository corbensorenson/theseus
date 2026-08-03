from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_repository_runner_inventory_v3 as owner  # noqa: E402

REPORT = owner.audit(ROOT / "configs" / "theseus_vcm_repository_runner_inventory_v3.json")


def test_every_repaired_panel_task_has_a_source_bound_runner_without_execution() -> None:
    assert REPORT["trigger_state"] == "GREEN"
    assert REPORT["observations"]["task_count"] == 62
    assert REPORT["observations"]["tasks_with_any_independent_runner_receipt"] == 62
    assert REPORT["residual_indices"] == []
    assert REPORT["parent_target_or_evaluator_executions"] == 0
    assert REPORT["candidate_or_control_calls"] == 0
    assert REPORT["external_reference_calls"] == 0


def test_only_exact_standard_runtime_conventions_close_the_four_residuals() -> None:
    rows = {row["index"]: row for row in REPORT["rows"]}
    expected = {
        10: ("self_contained_node_builtin_selected_file", "node VidCoreLargePlayer/tests/popup-guard.test.mjs"),
        16: ("selected_file_embedded_pytest_command", "python -m pytest dashboard/test_arena_auth.py"),
        20: ("python_unittest_selected_file", "python -m unittest tests/test_codexicon.py"),
        22: ("python_unittest_selected_file", "python -m unittest tests/test_input_validators.py"),
    }
    assert {index for index, row in rows.items() if row["selected_verifier_runtime_receipts"]} == set(expected)
    for index, (kind, command) in expected.items():
        receipt = rows[index]["selected_verifier_runtime_receipts"][0]
        assert receipt["kind"] == kind
        assert receipt["command"] == command
        assert receipt["derivation"] == "selected_verifier_runtime_convention"


def test_nested_manifests_are_relevant_only_when_they_own_selected_paths() -> None:
    rows = {row["index"]: row for row in REPORT["rows"]}
    assert {row["path"] for row in rows[16]["selected_path_relevant_manifests"]} >= {
        "requirements-dev.txt",
        "ops-server/requirements.txt",
    }
    assert rows[10]["selected_path_relevant_manifests"] == []
    assert rows[20]["selected_path_relevant_manifests"] == []
