from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_dependency_pair_coverage_audit as owner  # noqa:E402
REPORT=owner.audit(ROOT/"configs"/"theseus_vcm_dependency_pair_coverage_audit.json")
def test_all_parent_target_dependency_pairs_are_classified()->None:
 o=REPORT["observations"];assert REPORT["trigger_state"]=="GREEN" and (o["locked_task_count"],o["identical_pair_task_count"],o["divergent_pair_task_count"],o["missing_pair_member_task_count"])==(48,38,10,0)
 assert o["divergent_task_indices"]==[11,14,17,19,33,46,53,54,61,62] and o["required_distinct_dependency_closure_count"]==58
def test_pair_audit_executes_nothing()->None:
 assert REPORT["static_audit_only"] is True and REPORT["network_or_dependency_execution_performed"] is False and REPORT["repository_executions"]==REPORT["candidate_or_control_calls"]==REPORT["external_reference_calls"]==0
