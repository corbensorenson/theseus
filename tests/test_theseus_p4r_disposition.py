from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_p4r_disposition as disposition  # noqa: E402


def test_p4r_disposition_recomputes_complete_campaign_custody() -> None:
    report = disposition.build_report()

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["scientific_status"] == "INCONCLUSIVE_IMPLEMENTATION"
    assert report["denominators"]["tasks"] == 10
    assert report["denominators"]["learned_model_calls"] == 60
    assert report["denominators"]["project_selected_quality_token_cap"] is None
    assert len(report["source_identities"]["runtime_receipts"]) == 60
    assert report["termination_custody"]["termination_reason_counts"] == {
        "model_eos": 26,
        "parser_complete": 34,
    }
    assert report["termination_custody"]["safety_ceiling_hits"] == 0
    assert report["termination_custody"]["maximum_generated_tokens"] == 7219


def test_p4r_disposition_fails_closed_on_treatment_adequacy() -> None:
    report = disposition.build_report()

    assert report["arm_totals"][p4.DIRECT]["useful_candidates"] == 1
    assert report["arm_totals"][p4.PLAN]["useful_candidates"] == 2
    assert report["arm_totals"][p4.SEMANTIC]["useful_candidates"] == 0
    assert report["arm_totals"][p4.STATIC]["useful_candidates"] == 0
    assert report["oracle_totals"]["useful_candidates"] == 10
    assert report["adequacy"]["semantic_ir_parse_and_lower"] == "0/10"
    assert report["adequacy"]["mechanics_floor_passed"] is False
    assert report["decision_rule"]["effect_decision_authorized"] is False
    assert report["decision_rule"]["disposition"] == (
        "NOT_EVALUABLE_IMPLEMENTATION_INADEQUATE"
    )
    assert report["next_stage"]["D1_eligible"] is False
    assert report["next_stage"]["book_support_state_effect"] == "none"


def test_committed_disposition_preserves_negative_evidence_scope() -> None:
    report = json.loads(
        (ROOT / "reports" / "theseus_p4r_terminal_disposition.json").read_text(
            encoding="utf-8"
        )
    )

    assert "cannot falsify cognitive compilation" in report["maximum_inference"]
    assert report["consumption"]["eligible_for_exact_rerun"] is False
    assert report["consumption"]["all_ten_tasks_consumed"] is True
