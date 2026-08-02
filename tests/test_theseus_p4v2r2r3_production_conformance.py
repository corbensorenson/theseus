from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4v2r2r3_production_conformance as conformance  # noqa: E402


def test_exact_production_surface_conforms_without_model_or_hidden_calls() -> None:
    report = conformance.run_conformance(include_tokenizer=False)

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["coverage"]["frozen_tasks"] == 10
    assert report["coverage"]["production_first_prompts"] == 30
    assert report["coverage"]["production_repair_prompts"] == 30
    assert report["coverage"]["oracle_parse_lower_apply_visible_passes_required"] == 30
    assert all(
        count >= 1
        for count in report["coverage"]["operation_mechanics"].values()
    )
    assert report["coverage"]["delimiter_round_trips"] == 30
    assert report["coverage"]["malformed_rejections"] == 30
    assert report["candidate_or_control_calls"] == 0
    assert report["hidden_evaluator_calls"] == 0
    assert report["project_selected_quality_token_cap"] is None
