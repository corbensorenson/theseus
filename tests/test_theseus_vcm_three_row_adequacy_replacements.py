import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_three_row_adequacy_replacements as owner
import theseus_vcm_three_row_adequacy_replacements_audit as auditor
import theseus_vcm_three_row_adequacy_replacements_audit_v2 as auditor_v2

CONFIG = ROOT / "configs" / "theseus_vcm_three_row_adequacy_replacements.json"


def test_sealed_config_binds_only_three_inadequate_rows_and_freezes_qualified_rows():
    config = json.loads(CONFIG.read_text())
    assert [row["index"] for row in config["replacement_slots"]] == [12, 13, 35]
    assert config["frozen_qualified_indices"] == [16, 25, 56]
    assert config["authority"]["untrusted_code_execution_authorized"] is False
    assert config["authority"]["local_model_calls_authorized"] is False


def test_live_report_is_all_or_none_and_has_no_forbidden_execution():
    config = json.loads(CONFIG.read_text())
    report_path = ROOT / config["report"]
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text())
    if report["trigger_state"] != "GREEN" or report["replacement_set_admitted"] is not True:
        assert report["replacement_set_admitted"] is False
        assert report.get("selected_repository_count", 0) == 0
        return
    assert report["replacement_set_admitted"] is True
    assert [row["index"] for row in report["replacement_rows"]] == [12, 13, 35]
    assert len({row["repository"] for row in report["replacement_rows"]}) == 3
    assert report["qualified_rows_rerun"] is False
    assert report["counters"]["parent_target_or_evaluator_executions"] == 0
    assert report["counters"]["local_model_calls"] == 0
    assert report["counters"]["external_inference_calls"] == 0


def test_role_separated_audit_successor_accepts_only_a_complete_live_transaction():
    config = json.loads(CONFIG.read_text())
    if not (ROOT / config["report"]).is_file():
        return
    producer = json.loads((ROOT / config["report"]).read_text())
    if producer["trigger_state"] != "GREEN" or producer["replacement_set_admitted"] is not True:
        return
    v1_report = auditor.audit(CONFIG)
    assert v1_report["trigger_state"] == "RED"
    assert v1_report["faults"] == ["forbidden_execution_counter_nonzero"]
    report = auditor_v2.audit(ROOT / "configs" / "theseus_vcm_three_row_adequacy_replacements_audit_v2.json")
    assert report["trigger_state"] == "GREEN"
    assert report["replacement_set_admitted"] is True
    assert report["replacement_indices"] == [12, 13, 35]
    assert report["source_disjoint_from_current_panel"] is True
    assert report["archive_receipt_count"] == 12
    assert report["selected_source_difference_count"] == 3
    assert report["selected_verifier_difference_count"] == 3
    assert report["parent_target_or_evaluator_executions"] == 0
