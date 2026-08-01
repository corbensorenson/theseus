from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p2c_disposition as disposition  # noqa: E402


def test_p2c_qualifies_the_instrument_without_claiming_task_or_subsystem_success() -> None:
    report = disposition.build(
        ROOT / "reports" / "theseus_assistant_p2c_click_3578_run.json",
        ROOT / "reports" / "theseus_assistant_p2c_click_3578_evaluation.json",
        ROOT / "configs" / "theseus_assistant_p2c_instrument.json",
        ROOT / "configs" / "theseus_p2c_task_click_3578.json",
    )

    assert report["trigger_state"] == "GREEN"
    assert report["scientific_status"] == "INSTRUMENT_ADEQUATE_TASK_NOT_SOLVED"
    assert report["terminal_disposition"] == "P2C_TERMINAL_INSTRUMENT_ADEQUATE_ZERO_USEFUL"
    assert report["recomputed_checks"]["actual_newline_grammar_transport"] is True
    assert report["recomputed_checks"]["direct_candidate_parseable"] is False
    assert report["recomputed_checks"]["integrated_candidate_parseable"] is True
    assert report["recomputed_checks"]["correctness_evaluated_candidates"] == 1
    assert report["recomputed_checks"]["useful_candidates"] == 0
    assert report["recomputed_checks"]["route_integrity_ready_for_every_call"] is True
    assert report["recomputed_checks"]["rollback_verified"] is True
    assert report["next_stage"]["id"] == "P3_TEN_TASK_MATCHED_RESIDUAL_CAMPAIGN"
    assert report["consumption"]["eligible_for_exact_rerun"] is False
