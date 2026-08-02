from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4v2r2r3_disposition as disposition  # noqa: E402


def test_terminal_disposition_is_scoped_inconclusive_implementation() -> None:
    report = disposition.build_report()

    assert report["trigger_state"] == "GREEN"
    assert report["scientific_status"] == "INCONCLUSIVE_IMPLEMENTATION"
    assert report["faults"] == []
    assert report["denominators"]["tasks"] == 10
    assert report["denominators"]["learned_model_calls"] == 60
    assert report["denominators"]["project_selected_quality_token_cap"] is None
    assert report["prompt_continuity"][
        "project_selected_first_artifact_character_cap"
    ] is None
    assert report["adequacy"]["semantic_ir_parse_and_lower"] == "2/10"
    assert report["adequacy"]["semantic_ir_required_floor"] == "8/10"
    assert report["adequacy"]["mechanics_floor_passed"] is False
    assert report["adequacy"]["experiment_floor_passed"] is True
    assert report["decision_rule"]["effect_decision_authorized"] is False
    assert report["decision_rule"]["predeclared"][
        "inconclusive_implementation"
    ]
    assert report["next_stage"]["D1_eligible"] is False


def test_terminal_classification_keeps_context_boundary_invalid() -> None:
    assert disposition.classify_status(
        information_flow_green=True,
        boundary_hits=1,
        mechanics_floor=True,
        experiment_floor=True,
        survivor_rule=True,
    ) == "INVALID_OBSERVATION_CONTEXT_OR_HOST_BOUNDARY"
