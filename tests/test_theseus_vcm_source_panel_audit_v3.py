import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_source_panel_audit_v3 as owner
REPORT=owner.audit(ROOT/"configs/theseus_vcm_source_panel_audit_v3.json")
def test_v3_panel_is_exact_unique_and_english_content_admitted():
    assert REPORT["trigger_state"]=="GREEN" and REPORT["source_panel_admitted"] is True
    assert REPORT["assembled_task_count"]==62 and REPORT["unique_repository_count"]==62
    assert REPORT["english_title_and_selected_content_task_count"]==62
    assert REPORT["replacement_indices"]==[14,19,21,28,32,54,60]
    assert REPORT["content_violations"]==[]
def test_v3_archives_and_changed_bytes_are_complete():
    assert REPORT["archive_receipt_count"]==248
    assert REPORT["selected_source_difference_count"]==62
    assert REPORT["selected_verifier_difference_count"]==62
def test_v3_opens_no_execution_or_model_authority():
    assert REPORT["parent_target_or_evaluator_executions"]==0
    assert REPORT["local_model_calls"]==0 and REPORT["external_reference_calls"]==0
