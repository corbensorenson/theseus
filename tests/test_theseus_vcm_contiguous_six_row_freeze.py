from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_contiguous_six_row_freeze as owner  # noqa: E402
import theseus_vcm_contiguous_six_row_freeze_audit as audit_owner  # noqa: E402

CONFIG = ROOT / "configs" / "theseus_vcm_contiguous_six_row_freeze.json"


def test_six_parent_only_candidate_packets_freeze_without_calls() -> None:
    report, store = owner.freeze(CONFIG)
    assert report["trigger_state"] == "GREEN"
    assert report["row_count"] == 6
    assert report["candidate_visible_field_count"] == 24
    assert report["local_model_calls"] == 0
    assert report["external_reference_calls"] == 0
    assert report["project_selected_quality_token_cap"] is None
    assert len(store["rows"]) == 6
    for row in report["rows"]:
        assert set(row["candidate_surface"]) == {
            "natural_language_request",
            "callable_signature_when_present",
            "broad_parent_effect_root",
            "arm_specific_model_visible_context",
        }
        assert row["allowed_effect_paths_present"] is False
        assert row["selector_has_no_page_or_byte_cap"] is True


def test_role_audit_rederives_six_qualified_rows_and_candidate_bytes() -> None:
    report, store = owner.freeze(CONFIG)
    audit = audit_owner.audit(CONFIG, producer_report=report, store=store)
    assert audit["trigger_state"] == "GREEN"
    assert audit["qualification"]["qualified_task_indices"] == [12, 13, 16, 25, 35, 56]
    assert audit["qualification"]["exact_evaluator_reruns_performed"] == 0
    assert audit["audited_row_count"] == 6
    assert audit["audited_candidate_visible_field_count"] == 24
    assert all(audit["parent_only_rederivation_conclusions"].values())
    assert audit["local_model_calls"] == 0
    assert audit["external_reference_calls"] == 0


def test_contract_has_no_quality_cap_and_reference_is_subscription_only() -> None:
    report, _ = owner.freeze(CONFIG)
    contract = report["frozen_contract"]
    assert contract["instrument_role"] == "unpowered_six_row_real_work_canary_not_claim_denominator"
    assert contract["completion_policy"]["project_selected_quality_token_cap"] is None
    reference = contract["openai_measurement_reference"]
    assert reference["transport"] == "demonstrably_codex_subscription_backed_only"
    assert reference["billable_api_spend_authorized"] is False
    assert reference["omit_if_subscription_provenance_cannot_be_proved"] is True
    assert contract["cost_custody"]["openai_api_spend_authorized_usd"] == 0
    assert len(contract["k3_stop_rules"]) >= 8
