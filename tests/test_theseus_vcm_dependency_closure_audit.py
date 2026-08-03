from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_dependency_closure_audit as owner  # noqa: E402

REPORT = owner.audit(ROOT / "configs" / "theseus_vcm_dependency_closure_audit.json")


def test_independent_audit_rederives_exact_cache_and_source() -> None:
    assert REPORT["trigger_state"] == "GREEN"
    assert REPORT["observations"]["content_blob_count"] == 2
    assert REPORT["observations"]["matched_lock_integrity_count"] == 2
    assert REPORT["observations"]["source_file_count"] == 560
    assert all(REPORT["cache_checks"].values())
    assert all(REPORT["source_checks"].values())


def test_audit_performs_no_dependency_repository_or_model_execution() -> None:
    assert REPORT["static_audit_only"] is True
    assert REPORT["network_or_dependency_execution_performed"] is False
    assert REPORT["repository_runner_executions"] == 0
    assert REPORT["parent_target_or_evaluator_executions"] == 0
    assert REPORT["candidate_or_control_calls"] == 0
    assert REPORT["external_reference_calls"] == 0
