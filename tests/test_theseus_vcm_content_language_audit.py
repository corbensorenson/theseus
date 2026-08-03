import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_content_language_audit as owner

REPORT = owner.audit(ROOT / "configs/theseus_vcm_content_language_audit.json")


def test_exact_non_english_selected_content_is_rejected():
    assert REPORT["trigger_state"] == "RED"
    assert REPORT["source_panel_admission_remains_valid"] is False
    assert REPORT["violation_indices"] == [14, 19, 21, 28, 32, 54, 60]
    assert REPORT["replacement_required_before_evaluator_execution"] is True


def test_binary_png_fixtures_are_skipped_not_false_positive_text():
    assert REPORT["binary_identities_skipped"] > 0
    assert 61 not in REPORT["violation_indices"]


def test_audit_has_zero_execution_or_model_authority():
    assert REPORT["parent_target_or_evaluator_executions"] == 0
    assert REPORT["candidate_or_control_calls"] == 0
    assert REPORT["external_reference_calls"] == 0
