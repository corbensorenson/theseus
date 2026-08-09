import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_replacement_environment_materializer as owner
import theseus_vcm_replacement_environment_materializer_audit as auditor
CONFIG=ROOT/"configs"/"theseus_vcm_replacement_environment_materializer.json"
def test_preflight_binds_three_replacements_without_execution():
 r=owner.preflight_report(CONFIG);assert r["trigger_state"]=="PAUSED";assert r["execution_performed"] is False;assert r["repository_runner_executions"]==0;assert r["candidate_or_control_calls"]==0;assert r["project_selected_output_cap"] is None
def test_green_environment_receipt_is_all_or_none():
 cfg=json.loads(CONFIG.read_text());path=ROOT/cfg["report"]
 if not path.is_file():return
 r=json.loads(path.read_text())
 if r.get("trigger_state")!="GREEN":return
 assert r["qualified_task_count"]==3;assert [x["index"] for x in r["rows"]]==[12,13,35];assert r["network_enabled_materializations"]==3;assert r["network_denied_replays"]==3;assert r["source_build_executions"]==0;assert r["repository_runner_executions"]==0
 a=auditor.audit(CONFIG);assert a["trigger_state"]=="GREEN";assert a["qualified_task_count"]==3
