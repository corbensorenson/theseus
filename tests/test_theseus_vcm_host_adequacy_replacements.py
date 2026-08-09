import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import theseus_vcm_host_adequacy_replacements as owner

CONFIG=ROOT/"configs"/"theseus_vcm_host_adequacy_replacements.json"


def test_preflight_proves_current_task_13_explicit_host_invalidation():
    report=owner.preflight(CONFIG)
    assert report["trigger_state"]=="GREEN"
    assert report["replacement_set_admitted"] is False
    receipt=report["host_invalidation_receipts"][0]
    assert receipt["index"]==13
    assert receipt["repository"]=="frame-consulting/QuantLibXlOil"
    assert receipt["explicit_unsupported_host"] is True
    assert any("xloil" in hit["pattern"].lower() or "windows" in hit["pattern"].lower() for hit in receipt["hits"])
    assert report["qualified_rows_rerun"] is False


def test_live_success_is_one_source_disjoint_host_adequate_replacement():
    config=json.loads(CONFIG.read_text()); path=ROOT/config["report"]
    if not path.is_file(): return
    report=json.loads(path.read_text())
    if report.get("replacement_set_admitted") is not True:
        assert report.get("selected_repository_count",0)==0
        return
    assert report["trigger_state"]=="GREEN"
    assert [row["index"] for row in report["replacement_rows"]]==[13]
    assert len(report["host_compatibility_receipts"])==1
    assert report["host_compatibility_receipts"][0]["explicit_unsupported_host"] is False
    assert report["qualified_rows_rerun"] is False
    assert report["counters"]["parent_target_or_evaluator_executions"]==0
    assert report["counters"]["local_model_calls"]==0
