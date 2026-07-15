from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neural_seed_local_english_raters import (
    adjudication_keys,
    parse_scores,
    rating_prompt,
    validate_config,
    validate_packet,
)


CONFIG = json.loads((ROOT / "configs/neural_seed_local_english_raters.json").read_text())


def test_config_pins_three_distinct_local_raters() -> None:
    assert validate_config(CONFIG) == []
    cards = [*CONFIG["primary_raters"], CONFIG["adjudicator"]]
    assert len({row["rater_id"] for row in cards}) == 3
    assert len({row["revision"] for row in cards}) == 3
    assert CONFIG["boundaries"]["external_inference_calls"] == 0
    assert CONFIG["consumption_registry"] == "reports/private_functional_consumption_registry.jsonl"


def test_rating_prompt_contains_no_model_or_reference_identity() -> None:
    entry = {
        "prompt": "Revise the earlier plan and keep the budget fixed.",
        "candidate_output": "I will keep the budget fixed and revise only the schedule.",
    }
    prompt = rating_prompt(entry, CONFIG)
    lowered = prompt.lower()
    assert "model_id" not in lowered
    assert "checkpoint" not in lowered
    assert "reference answer" not in lowered
    assert entry["prompt"] in prompt
    assert entry["candidate_output"] in prompt


def test_score_parser_accepts_only_exact_integer_dimension_object() -> None:
    valid = {dimension: 3 for dimension in CONFIG["scoring"]["dimensions"]}
    parsed, error = parse_scores("prefix " + json.dumps(valid) + " suffix", CONFIG)
    assert parsed == valid
    assert error == ""

    invalid = dict(valid)
    invalid["extra"] = 4
    assert parse_scores(json.dumps(invalid), CONFIG)[0] is None
    invalid = dict(valid)
    invalid["clarity"] = 4.5
    assert parse_scores(json.dumps(invalid), CONFIG)[0] is None
    invalid = dict(valid)
    invalid["clarity"] = 5
    assert parse_scores(json.dumps(invalid), CONFIG)[0] is None


def test_adjudication_is_required_only_at_preregistered_delta() -> None:
    base = {dimension: 3 for dimension in CONFIG["scoring"]["dimensions"]}
    far = dict(base)
    far["calibration"] = 1
    judgments = {
        "opaque_a": [
            {"case_id": "case-1", "rater_id": "a", "scores": base},
            {"case_id": "case-1", "rater_id": "b", "scores": far},
            {"case_id": "case-2", "rater_id": "a", "scores": base},
            {"case_id": "case-2", "rater_id": "b", "scores": base},
        ]
    }
    assert adjudication_keys(judgments, CONFIG) == {("opaque_a", "case-1")}


def test_packet_validation_fails_identity_or_reference_exposure() -> None:
    packet = {
        "policy": "project_theseus_blind_english_judgment_packet_v1",
        "trigger_state": "GREEN",
        "item_count": 32,
        "model_identity_present": False,
        "checkpoint_identity_present": False,
        "reference_answer_present": False,
        "items": [{"model_id": "leak"}],
    }
    assert "packet_item_identity_or_reference_exposed" in validate_packet(packet)
