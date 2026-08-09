import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_source_panel_audit_v4 as owner

CONFIG = ROOT / "configs" / "theseus_vcm_source_panel_audit_v4.json"


def test_v4_role_separated_panel_replaces_only_three_rows():
    report = owner.audit(CONFIG)
    assert report["trigger_state"] == "GREEN"
    assert report["source_panel_admitted"] is True
    assert report["assembled_task_count"] == 62
    assert report["unique_repository_count"] == 62
    assert report["replacement_indices"] == [12, 13, 35]
    assert report["preserved_row_count"] == 59
    assert report["archive_receipt_count"] == 248
    assert report["selected_source_difference_count"] == 62
    assert report["selected_verifier_difference_count"] == 62
    assert report["content_violations"] == []
    assert report["parent_target_or_evaluator_executions"] == 0
    assert report["local_model_calls"] == 0


def test_v4_rows_preserve_panel_and_language_quotas():
    report = owner.audit(CONFIG)
    prior = json.loads((ROOT / "reports/theseus_vcm_source_panel_audit_v3.json").read_text())
    for index, (old, new) in enumerate(zip(prior["assembled_rows"], report["assembled_rows"]), 1):
        assert (old["panel"], old["query_language"]) == (new["panel"], new["query_language"])
        assert (old == new) is (index not in {12, 13, 35})
