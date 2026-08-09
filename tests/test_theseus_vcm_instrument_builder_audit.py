from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_instrument_builder_audit as audit_owner  # noqa: E402

REPORT = audit_owner.audit(ROOT / "configs" / "theseus_vcm_instrument_builder_audit.json")


def test_committed_risk_chain_is_role_separately_rederived() -> None:
    assert REPORT["trigger_state"] == "GREEN"
    assert REPORT["audited_attempt_count"] == 4
    assert REPORT["audit_kind"] == "role-separated rederivation"
    assert REPORT["qualified_risk_classes"] == [
        "bun_real_lock_install_and_retained_replay",
        "yarn_real_lock_install_and_retained_replay",
        "typescript_real_parent_file_strict_no_emit_mechanics",
        "rust_parent_repository_untrusted_compilation",
    ]


def test_audit_preserves_scoped_caveat_and_executes_nothing() -> None:
    assert "task61_full_project_typecheck_is_inconclusive_missing_generated_parent_source" in REPORT["preserved_caveats"]
    assert REPORT["network_or_dependency_execution_performed"] is False
    for key in ("repository_runner_executions", "parent_target_or_evaluator_executions", "candidate_or_control_calls", "external_reference_calls"):
        assert REPORT[key] == 0
