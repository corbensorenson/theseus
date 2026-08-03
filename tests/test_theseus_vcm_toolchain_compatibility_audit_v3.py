from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_toolchain_compatibility_audit_v3 as owner  # noqa:E402
REPORT=owner.audit(ROOT/"configs"/"theseus_vcm_toolchain_compatibility_audit_v3.json")
def test_rust_profile_changes_exactly_tasks_29_and_36()->None:
 assert REPORT["trigger_state"]=="GREEN" and REPORT["observations"]["changed_task_indices"]==[29,36]
 for index in (29,36):assert next(r for r in REPORT["rows"] if r["index"]==index)["state"]=="COMPATIBLE_DECLARED_REQUIREMENTS"
def test_denominator_and_zero_execution_are_preserved()->None:
 o=REPORT["observations"];assert (o["locked_task_count"],o["compatible_declared_requirement_task_count"],o["no_declared_version_requirement_task_count"],o["incompatible_declared_requirement_task_count"],o["unresolved_requirement_syntax_task_count"])==(48,21,16,11,0)
 assert REPORT["dependency_prefetch_executions"]==REPORT["repository_executions"]==REPORT["candidate_or_control_calls"]==REPORT["external_reference_calls"]==0
