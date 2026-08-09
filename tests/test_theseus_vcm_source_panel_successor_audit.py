import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_source_panel_successor_audit as owner
def test_host_adequate_panel_successor_is_green():
 r=owner.audit(ROOT/"configs"/"theseus_vcm_source_panel_successor_audit.json");assert r["trigger_state"]=="GREEN";assert r["source_panel_admitted"] is True;assert r["assembled_task_count"]==62;assert r["unique_repository_count"]==62;assert r["replacement_indices"]==[13];assert r["preserved_row_count"]==61;assert r["archive_receipt_count"]==248;assert r["selected_source_difference_count"]==62;assert r["selected_verifier_difference_count"]==62;assert r["content_violations"]==[];assert r["assembled_rows"][12]["repository"]=="paulomtts/pyjinhx";assert r["parent_target_or_evaluator_executions"]==0
