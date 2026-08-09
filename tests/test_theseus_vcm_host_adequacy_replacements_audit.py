import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_host_adequacy_replacements_audit as owner

def test_host_adequate_replacement_role_audit_is_green():
    report=owner.audit(ROOT/"configs"/"theseus_vcm_host_adequacy_replacements_audit.json")
    assert report["trigger_state"]=="GREEN"
    assert report["replacement_index"]==13
    assert report["repository"]=="paulomtts/pyjinhx"
    assert report["source_disjoint_from_current_panel"] is True
    assert report["archive_receipt_count"]==4
    assert report["source_changed"] is True and report["verifier_changed"] is True
    assert report["explicit_unsupported_host"] is False
    assert report["qualified_rows_rerun"] is False
    assert report["parent_target_or_evaluator_executions"]==0
