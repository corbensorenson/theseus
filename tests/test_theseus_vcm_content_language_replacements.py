import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_content_language_replacements as owner

def test_preflight_binds_exact_seven_slots_and_zero_execution():
    report=owner.preflight(ROOT/"configs/theseus_vcm_content_language_replacements.json")
    assert report["trigger_state"]=="GREEN"
    assert report["replacement_set_admitted"] is False
    assert report["candidate_packet_materialization_opened"] is False

def test_current_task_28_selected_content_is_rejected():
    panel=json.loads((ROOT/"reports/theseus_vcm_source_panel_audit_v2.json").read_text())
    config=json.loads((ROOT/"configs/theseus_vcm_content_language_replacements.json").read_text())
    ranges=[(r["name"],int(r["start"],16),int(r["end"],16)) for r in config["forbidden_unicode_scripts"]]
    violations=owner.selected_content_violations(panel["assembled_rows"][27],ranges,set(config["binary_extensions"]))
    assert violations
    assert any(row["path"]=="tests/test_opening_geography.py" for row in violations)

def test_known_english_task_selected_content_passes():
    panel=json.loads((ROOT/"reports/theseus_vcm_source_panel_audit_v2.json").read_text())
    config=json.loads((ROOT/"configs/theseus_vcm_content_language_replacements.json").read_text())
    ranges=[(r["name"],int(r["start"],16),int(r["end"],16)) for r in config["forbidden_unicode_scripts"]]
    assert owner.selected_content_violations(panel["assembled_rows"][0],ranges,set(config["binary_extensions"]))==[]
