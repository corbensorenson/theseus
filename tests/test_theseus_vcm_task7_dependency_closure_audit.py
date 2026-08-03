from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_task7_dependency_closure_audit as owner  # noqa: E402
REPORT=owner.audit(ROOT/"configs"/"theseus_vcm_task7_dependency_closure_audit.json")

def test_independent_audit_rederives_pnpm_store_and_source() -> None:
    assert REPORT["trigger_state"]=="GREEN"
    assert REPORT["observations"]["retained_store_file_count"]==9
    assert REPORT["observations"]["pnpm_index_matched_file_count"]==8
    assert REPORT["observations"]["source_file_count"]==244
    assert all(REPORT["store_checks"].values()) and all(REPORT["source_checks"].values())

def test_audit_performs_no_dependency_repository_or_model_execution() -> None:
    assert REPORT["static_audit_only"] is True and REPORT["network_or_dependency_execution_performed"] is False
    for key in ("repository_runner_executions","parent_target_or_evaluator_executions","candidate_or_control_calls","external_reference_calls"): assert REPORT[key]==0
