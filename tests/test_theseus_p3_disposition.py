from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p3_disposition as disposition  # noqa: E402


def test_p3_disposition_recomputes_all_ten_consumed_pairs() -> None:
    report = disposition.build(
        ROOT / "configs" / "theseus_p3_task_pool.json",
        ROOT / "configs" / "theseus_assistant_p3_instrument.json",
    )

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["scientific_status"] == (
        "P3_COMPLETE_RESIDUAL_EXPOSED_NO_USEFULNESS_ROUTE_WINNER"
    )
    assert report["denominators"] == {
        "tasks": 10,
        "arms": 20,
        "persistent_model_loads": 10,
        "model_calls": 36,
        "correctness_evaluated_candidates": 14,
        "useful_candidates": 2,
        "unsafe_candidates": 0,
        "hosted_reference_tasks": 0,
        "hosted_reference_calls": 0,
    }
    assert len(report["source_identities"]["runtime_receipts"]) == 36
    assert all(
        row["route_integrity_ready"]
        for row in report["source_identities"]["runtime_receipts"]
    )
    assert all(row["matched_pair_ready"] for row in report["task_results"])


def test_p3_disposition_preserves_arm_and_paired_denominators() -> None:
    report = json.loads(
        (ROOT / "reports" / "theseus_assistant_p3_terminal_disposition.json").read_text(
            encoding="utf-8"
        )
    )
    direct = report["arm_totals"]["direct_local_model"]
    integrated = report["arm_totals"]["integrated_local_model"]

    assert direct["parseable_candidates"] == 5
    assert integrated["parseable_candidates"] == 9
    assert direct["useful_candidates"] == integrated["useful_candidates"] == 1
    assert direct["unsafe_candidates"] == integrated["unsafe_candidates"] == 0
    assert direct["model_calls"] == 19
    assert integrated["model_calls"] == 17
    assert report["paired_outcomes"]["useful"] == {
        "both": 0,
        "direct_only": 1,
        "integrated_only": 1,
        "neither": 8,
        "discordant": 2,
    }
    assert report["paired_outcomes"]["parseable"] == {
        "both": 5,
        "direct_only": 0,
        "integrated_only": 4,
        "neither": 1,
        "discordant": 4,
    }
    assert report["statistical_boundary"]["useful_exact_two_sided_sign_test_p"] == 1.0
    assert report["statistical_boundary"]["parseable_exact_two_sided_sign_test_p"] == 0.125


def test_p3_selects_one_p4_mechanism_without_promoting_it() -> None:
    report = json.loads(
        (ROOT / "reports" / "theseus_assistant_p3_terminal_disposition.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["next_stage"]["selected_claim_id"] == (
        "cognitive-compilation-and-semantic-ir.core"
    )
    assert report["consumption"]["all_ten_tasks_consumed"] is True
    assert report["consumption"]["eligible_for_exact_rerun"] is False
    assert report["consumption"]["eligible_for_training"] is False
    assert report["consumption"]["eligible_for_D1_or_D2"] is False
    assert report["hosted_reference"]["results"] == "NOT_RUN"
    assert report["hosted_reference"]["transport_state"] == "DEFINED_NOT_BOUND"
    assert "supports no subsystem" in report["maximum_inference"]
