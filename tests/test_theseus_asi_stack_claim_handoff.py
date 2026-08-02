from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_asi_stack_claim_handoff as handoff  # noqa: E402


CONFIG = ROOT / "configs" / "theseus_asi_stack_claim_handoff.json"
CLAIM = "cognitive-compilation-and-semantic-ir.core"


def p4(status: str) -> dict[str, object]:
    return {
        "policy": "project_theseus_p4v2r2r3_terminal_disposition_v1",
        "claim_id": CLAIM,
        "trigger_state": "GREEN",
        "scientific_status": status,
        "scope": "exact synthetic test override",
        "denominators": {
            "tasks": 10,
            "learned_model_calls": 60,
            "project_selected_quality_token_cap": None,
        },
        "adequacy": {
            "information_flow_green": True,
            "mechanics_floor_passed": True,
            "experiment_floor_passed": True,
        },
        "decision_rule": {
            "survivor_effect_rule_passed": status.endswith("D1_ELIGIBLE"),
            "effect_decision_authorized": True,
        },
        "consumption": {
            "eligible_for_D1": status.endswith("D1_ELIGIBLE"),
            "exact_surface_consumed": True,
            "rerun_allowed": False,
        },
        "negative_results": ["retained exact negative result"],
        "maximum_inference": "exact implementation and regime only",
    }


def d1(status: str = "D1_EXACT_IMPLEMENTATION_QUALIFIED") -> dict[str, object]:
    return {
        "policy": "project_theseus_d1_terminal_disposition_v1",
        "claim_id": CLAIM,
        "trigger_state": "GREEN",
        "scientific_status": status,
        "scope": "fresh source-disjoint D1 test override",
        "denominators": {"tasks": 44, "learned_model_calls": 176},
        "adequacy": {"information_flow_green": True},
        "decision_rule": {"effect_decision_authorized": True},
        "consumption": {"exact_surface_consumed": True, "rerun_allowed": False},
        "maximum_inference": "exact D1 implementation and regime only",
    }


def test_book_pin_claim_and_transition_schema_are_exactly_bound() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    binding, faults = handoff.audit_book_binding(config)
    assert faults == []
    assert binding["passed"] is True
    assert binding["commit"] == "17c6ece80f771d3bce5f89c6b85c99ca9b6c2ea0"
    assert binding["chapter_count"] == 84
    assert binding["claim_id"] == CLAIM
    assert binding["current_support_state"] == "argument"
    assert all(row["passed"] for row in binding["artifacts"].values())


def test_preterminal_state_emits_no_book_packet() -> None:
    report = handoff.build_report(CONFIG, p4_override={})
    assert report["trigger_state"] == "PAUSED"
    assert report["packet_ready"] is False
    assert report["book_review_packet"] == {}
    assert report["support_state_effect"] == "none"


def test_superseded_p4_policy_cannot_open_book_handoff() -> None:
    stale = p4("P4V2R2R2_ADEQUATE_NO_SURVIVOR")
    stale["policy"] = "project_theseus_p4v2r2r2_terminal_disposition_v1"
    report = handoff.build_report(CONFIG, p4_override=stale)
    assert report["trigger_state"] == "PAUSED"
    assert report["packet_ready"] is False
    assert report["book_review_packet"] == {}


def test_terminal_non_survivor_opens_review_without_d1_or_support_movement() -> None:
    report = handoff.build_report(
        CONFIG,
        p4_override=p4("P4V2R2R2_ADEQUATE_NO_SURVIVOR"),
    )
    assert report["trigger_state"] == "GREEN"
    assert report["activation_state"] == "READY_FOR_GOVERNED_BOOK_REVIEW_WITHOUT_D1"
    assert report["packet_ready"] is True
    packet = report["book_review_packet"]
    assert packet["d1"] == {}
    assert packet["support_state_effect"] == "none"
    assert packet["review_request"]["automatic_transition_proposed"] is False
    assert packet["review_request"]["book_must_create_separate_evidence_transition_record"] is True


def test_survivor_waits_for_exactly_one_fresh_d1_terminal_result() -> None:
    report = handoff.build_report(
        CONFIG,
        p4_override=p4("P4V2R2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE"),
        d1_override={},
    )
    assert report["trigger_state"] == "PAUSED"
    assert report["activation_state"] == "WAITING_FOR_ONE_FRESH_D1_TERMINAL_RESULT"
    assert report["packet_ready"] is False
    assert report["book_review_packet"] == {}


def test_survivor_and_terminal_d1_open_review_but_never_promote() -> None:
    report = handoff.build_report(
        CONFIG,
        p4_override=p4("P4V2R2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE"),
        d1_override=d1(),
    )
    assert report["trigger_state"] == "GREEN"
    assert report["activation_state"] == "READY_FOR_GOVERNED_BOOK_REVIEW_AFTER_D1"
    packet = report["book_review_packet"]
    assert packet["p4"]["scientific_status"] == (
        "P4V2R2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE"
    )
    assert packet["d1"]["scientific_status"] == "D1_EXACT_IMPLEMENTATION_QUALIFIED"
    assert packet["current_book_support_state"] == "argument"
    assert packet["support_state_effect"] == "none"
    assert report["publication_authority"] == "none"
    assert report["release_authority"] == "none"


def test_wrong_d1_policy_cannot_open_survivor_handoff() -> None:
    stale = d1()
    stale["policy"] = "project_theseus_d1_terminal_disposition_draft"
    report = handoff.build_report(
        CONFIG,
        p4_override=p4("P4V2R2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE"),
        d1_override=stale,
    )
    assert report["trigger_state"] == "PAUSED"
    assert report["packet_ready"] is False


def test_claim_mismatch_in_d1_cannot_open_handoff() -> None:
    wrong = d1()
    wrong["claim_id"] = "another.claim"
    report = handoff.build_report(
        CONFIG,
        p4_override=p4("P4V2R2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE"),
        d1_override=wrong,
    )
    assert report["trigger_state"] == "PAUSED"
    assert report["packet_ready"] is False


def test_config_cannot_grant_automatic_support_or_publication_authority() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    invalid = copy.deepcopy(config)
    invalid["packet_policy"]["automatic_support_transition_proposed"] = True
    invalid["consumer"]["publication_authority"] = "granted"
    faults = handoff.validate_config(invalid)
    assert "forbidden_packet_authority_present" in faults
    assert "publication_authority_present" in faults


def test_public_packet_excludes_candidate_hidden_and_oracle_payload_keys() -> None:
    report = handoff.build_report(
        CONFIG,
        p4_override=p4("INCONCLUSIVE_IMPLEMENTATION"),
    )
    packet = report["book_review_packet"]
    assert packet
    assert handoff.contains_forbidden_packet_key(packet) is False
    serialized = json.dumps(packet, sort_keys=True)
    for forbidden in ("candidate_output", "hidden_tests", "oracle", "solution_body"):
        assert forbidden not in serialized
