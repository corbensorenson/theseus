import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_repository_closure_materialization_audit_v2 as owner
def test_host_adequate_closure_successor_is_green():
 r=owner.audit(ROOT/"configs"/"theseus_vcm_repository_closure_materialization_audit_v2.json");assert r["trigger_state"]=="GREEN";assert r["task_count"]==62;assert r["archive_artifact_count"]==124;assert r["replayed_unchanged_task_count"]==61;assert r["replacement_task_count"]==1;assert r["replacement_index"]==13;assert r["replacement_archive_count"]==2;assert all(not row["faults"] for row in r["audited_rows"]);assert r["parent_target_or_evaluator_executions"]==0
