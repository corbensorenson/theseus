from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_task30_dependency_closure_audit as owner  # noqa:E402
REPORT=owner.audit(ROOT/"configs"/"theseus_vcm_task30_dependency_closure_audit.json")
def test_audit_rederives_all_crates_store_and_source()->None:
 assert REPORT["trigger_state"]=="GREEN" and REPORT["observations"]["lock_checksum_package_count"]==REPORT["observations"]["matched_checksum_count"]==50
 assert REPORT["observations"]["retained_store_file_count"]==2639 and REPORT["observations"]["source_file_count"]==1545
 assert all(REPORT["store_checks"].values()) and all(REPORT["source_checks"].values()) and all(REPORT["phase_checks"].values())
def test_audit_executes_nothing()->None:
 assert REPORT["static_audit_only"] is True and REPORT["network_or_dependency_execution_performed"] is False
 for k in ("repository_build_executions","repository_runner_executions","parent_target_or_evaluator_executions","candidate_or_control_calls","external_reference_calls"):assert REPORT[k]==0
