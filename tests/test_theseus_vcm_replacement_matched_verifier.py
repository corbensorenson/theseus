import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_replacement_matched_verifier as owner
import theseus_vcm_replacement_matched_verifier_audit as auditor
CONFIG=ROOT/"configs"/"theseus_vcm_replacement_matched_verifier.json"
def test_preflight_is_sealed_and_call_free():
 r=owner.preflight_report(CONFIG);assert r["trigger_state"]=="PAUSED";assert r["execution_performed"] is False;assert r["candidate_or_control_calls"]==0;assert r["external_reference_calls"]==0;assert r["project_selected_output_cap"] is None
def test_common_evaluators_never_overlap_production_transplants():
 cfg,bound,faults=owner.preflight(CONFIG,verify_store=False);assert faults==[];assert [r["index"] for r in cfg["rows"]]==[12,13,35]
 for row in cfg["rows"]:assert set(row["common_evaluator_paths"]).isdisjoint(row["forbidden_transplant_paths"])
def test_green_receipt_rederives_exact_dispositions():
 cfg=json.loads(CONFIG.read_text());path=ROOT/cfg["report"]
 if not path.is_file():return
 r=json.loads(path.read_text())
 if r.get("trigger_state")!="GREEN":return
 assert r["task_count"]==3;assert r["parent_target_or_evaluator_executions"]==6;assert r["network_enabled_calls"]==0;assert r["candidate_or_control_calls"]==0
 a=auditor.audit(CONFIG);assert a["trigger_state"]=="GREEN";assert a["audited_task_count"]==3
