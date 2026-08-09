import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_replacement_resolution as owner
import theseus_vcm_replacement_resolution_audit as auditor
CONFIG=ROOT/"configs"/"theseus_vcm_replacement_resolution.json"
def test_preflight_binds_exact_three_rows_without_execution():
 r=owner.preflight_report(CONFIG);assert r["trigger_state"]=="PAUSED";assert r["execution_performed"] is False;assert r["parent_target_or_evaluator_executions"]==0;assert r["project_selected_output_cap"] is None
def test_live_resolution_and_audit_are_all_or_none():
 cfg=json.loads(CONFIG.read_text());path=ROOT/cfg["report"]
 if not path.is_file():return
 p=json.loads(path.read_text())
 if p.get("trigger_state")!="GREEN":return
 assert p["qualified_task_count"]==3;assert [r["index"] for r in p["rows"]]==[12,13,35];assert p["network_resolution_task_count"]==1;assert p["sealed_receipt_reuse_task_count"]==1;assert p["static_lock_task_count"]==1;assert p["package_installations"]==0;assert p["parent_target_or_evaluator_executions"]==0
 a=auditor.audit(CONFIG);assert a["trigger_state"]=="GREEN";assert a["qualified_task_count"]==3;assert a["network_resolution_task_count"]==1;assert a["sealed_receipt_reuse_task_count"]==1;assert a["static_lock_task_count"]==1
