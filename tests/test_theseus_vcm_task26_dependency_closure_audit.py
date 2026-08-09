from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_task26_dependency_closure_audit as owner  # noqa: E402

REPORT = owner.audit(ROOT / "configs" / "theseus_vcm_task26_dependency_closure_audit.json")


def test_audit_rederives_exact_wheel_only_instrument_wall() -> None:
    assert REPORT["trigger_state"] == "GREEN"
    assert REPORT["disposition"] == "INCONCLUSIVE_INSTRUMENT_DEPENDENCY_POLICY_RISK_CLASS"
    assert REPORT["observations"]["lock_package_count"] == 58
    assert REPORT["observations"]["blocking_distribution"] == "proxy-tools==0.1.0"
    assert REPORT["observations"]["blocking_distribution_has_sdist"] is True
    assert REPORT["observations"]["blocking_distribution_wheel_count"] == 0
    assert all(REPORT["lock_checks"].values())
    assert all(REPORT["command_checks"].values())
    assert all(REPORT["source_checks"].values())
    assert all(REPORT["boundary_checks"].values())


def test_audit_executes_nothing_and_grants_no_downstream_authority() -> None:
    assert REPORT["static_audit_only"] is True
    assert REPORT["network_or_dependency_execution_performed"] is False
    for key in (
        "source_build_executions",
        "project_installations",
        "repository_runner_executions",
        "parent_target_or_evaluator_executions",
        "candidate_or_control_calls",
        "external_reference_calls",
    ):
        assert REPORT[key] == 0
