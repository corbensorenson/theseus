import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_repository_closure_materialization as owner
CONFIG=ROOT/"configs/theseus_vcm_repository_closure_materialization_v2.json"
def test_v3_closure_preflight_plans_exact_124_and_no_execution():
    report=owner.preflight(json.loads(CONFIG.read_text()),CONFIG)
    assert report["trigger_state"]=="GREEN" and report["execution_authorized"] is True
    assert report["task_count"]==62 and report["planned_archive_count"]==124
    assert report["parent_target_or_evaluator_executions"]==0
def test_v3_registry_contains_exact_seven_new_repositories():
    panel=json.loads((ROOT/"reports/theseus_vcm_source_panel_audit_v3.json").read_text());registry=owner.transform_panel(panel)
    expected={14:"scrape-badger/scrapebadger-python",19:"cs0lar/treelang",21:"ctsdownloads/easyspeak",28:"microsoft/winml-cli",32:"Rul1an/assay",54:"passionworkeer/obsidian-shared-memory-bus",60:"tsz-org/tsz"}
    assert {i:registry["tasks"][i-1]["repository"] for i in expected}==expected
