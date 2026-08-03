from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_task14_dual_dependency_closure_audit as owner  # noqa: E402

REPORT = owner.audit(ROOT / "configs" / "theseus_vcm_task14_dual_dependency_closure_audit.json")


def test_audit_rederives_both_lock_store_source_and_phase_closures() -> None:
    assert REPORT["trigger_state"] == "GREEN"
    for label in ("parent", "target"):
        observed = REPORT["observations"][label]
        assert observed["lock_package_count"] == 41
        assert observed["cached_distribution_count"] == 12
        assert observed["retained_store_file_count"] == 371
        assert observed["source_file_count"] == 180
        assert all(REPORT["side_checks"][label].values())
    assert all(REPORT["storage_checks"].values())


def test_audit_executes_nothing_and_preserves_downstream_zeros() -> None:
    assert REPORT["static_audit_only"] is True
    assert REPORT["network_or_dependency_execution_performed"] is False
    for key in ("source_build_executions", "project_installations", "repository_runner_executions", "parent_target_or_evaluator_executions", "candidate_or_control_calls", "external_reference_calls"):
        assert REPORT[key] == 0
