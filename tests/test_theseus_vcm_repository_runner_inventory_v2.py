import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_repository_runner_inventory as owner
REPORT=owner.audit(ROOT/"configs/theseus_vcm_repository_runner_inventory_v2.json")
def test_repaired_panel_inventory_covers_all_exact_closures_without_execution():
    assert REPORT["trigger_state"]=="GREEN"
    assert REPORT["observations"]["task_count"]==62
    assert REPORT["observations"]["tasks_with_parent_and_target_source_closure"]==62
    assert REPORT["parent_target_or_evaluator_executions"]==0
def test_repaired_slots_are_present_in_inventory():
    expected={14:"scrape-badger/scrapebadger-python",19:"cs0lar/treelang",21:"ctsdownloads/easyspeak",28:"microsoft/winml-cli",32:"Rul1an/assay",54:"passionworkeer/obsidian-shared-memory-bus",60:"tsz-org/tsz"}
    assert {i:REPORT["rows"][i-1]["repository"] for i in expected}==expected
