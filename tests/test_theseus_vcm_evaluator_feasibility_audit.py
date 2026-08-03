from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_vcm_evaluator_feasibility_audit as audit  # noqa: E402


def test_real_feasibility_audit_fails_closed_without_revoking_source_panel() -> None:
    report = audit.audit()
    assert report["trigger_state"] == "RED"
    assert report["state"] == "EVALUATOR_INSTRUMENT_MATERIALIZATION_REQUIRED"
    assert report["source_panel_remains_admitted"] is True
    assert report["evaluator_execution_authorized"] is False
    assert report["python_sandbox_qualified_for_pinned_python_only"] is True
    assert report["language_task_counts"] == {
        "JavaScript": 13,
        "Python": 20,
        "Rust": 13,
        "TypeScript": 16,
    }
    assert report["observations"] == {
        "task_count": 62,
        "primary_executable_verifier_task_count": 62,
        "primary_executable_verifier_path_count": 88,
        "auxiliary_executable_verifier_path_count": 2,
        "nonexecutable_verifier_artifact_count": 19,
        "tasks_with_auxiliary_or_nonexecutable_verifier_artifacts": 8,
        "tasks_with_dependency_or_package_manifest": 0,
        "tasks_with_dependency_lock": 0,
        "tasks_with_transitive_local_source_closure_receipt": 0,
        "tasks_with_independent_runner_command_receipt": 0,
        "tasks_with_parent_fail_target_pass_receipt": 0,
        "evaluator_ready_task_count": 0,
    }
    assert report["parent_target_or_evaluator_executions"] == 0
    assert report["local_model_calls"] == 0
    assert report["external_reference_calls"] == 0


def test_every_task_gap_is_scoped_to_instrument_adequacy() -> None:
    report = audit.audit()
    assert len(report["task_gaps"]) == 62
    assert all(row["evaluator_ready"] is False for row in report["task_gaps"])
    assert all("transitive_local_source_closure_absent" in row["gaps"] for row in report["task_gaps"])
    assert all("independent_runner_command_receipt_absent" in row["gaps"] for row in report["task_gaps"])
    assert report["required_next_instrument"]["inadequate_harness_disposition"] == "INCONCLUSIVE_EXPERIMENT_NOT_TASK_OR_VCM_FAILURE"
