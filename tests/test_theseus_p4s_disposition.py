from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4s_disposition as disposition  # noqa: E402


def test_exact_sign_probability_is_one_sided_and_tie_excluding() -> None:
    assert disposition.exact_sign_probability(0, 0) == 1.0
    assert disposition.exact_sign_probability(3, 3) == 0.125
    assert disposition.exact_sign_probability(3, 4) == 0.3125


def test_task_bootstrap_is_deterministic_and_task_level() -> None:
    first = disposition.bootstrap_interval(
        [1, 1, 0, -1, 1], seed_material="p4s:test", samples=2000
    )
    second = disposition.bootstrap_interval(
        [1, 1, 0, -1, 1], seed_material="p4s:test", samples=2000
    )
    assert first == second
    assert first[0] <= 0.4 <= first[1]


def test_joined_cost_dominance_requires_nonworse_utility_and_all_costs() -> None:
    treatment = {
        "model_calls": 20, "prompt_tokens": 100, "generated_tokens": 100,
        "model_runtime_ms": 1000, "verifier_runtime_ms": 50,
        "rollback_failures": 0,
    }
    cheaper = dict(treatment, generated_tokens=80)
    expensive = dict(treatment, generated_tokens=120)

    assert disposition.cost_dominance(4, treatment, 4, cheaper)[
        "comparator_dominates_treatment"
    ] is True
    assert disposition.cost_dominance(4, treatment, 3, cheaper)[
        "comparator_dominates_treatment"
    ] is False
    assert disposition.cost_dominance(4, treatment, 4, expensive)[
        "comparator_dominates_treatment"
    ] is False


def test_terminal_classification_fails_closed_in_predeclared_order() -> None:
    classify = disposition.classify_status
    assert classify(
        information_flow_green=False, boundary_hits=0,
        mechanics_floor=True, experiment_floor=True, survivor_rule=True,
    ) == "INVALID_INFORMATION_FLOW"
    assert classify(
        information_flow_green=True, boundary_hits=1,
        mechanics_floor=True, experiment_floor=True, survivor_rule=True,
    ) == "INSTRUMENT_INADEQUATE_GENERATION_BOUNDARY_HIT"
    assert classify(
        information_flow_green=True, boundary_hits=0,
        mechanics_floor=False, experiment_floor=True, survivor_rule=True,
    ) == "INCONCLUSIVE_IMPLEMENTATION"
    assert classify(
        information_flow_green=True, boundary_hits=0,
        mechanics_floor=True, experiment_floor=False, survivor_rule=True,
    ) == "INCONCLUSIVE_EXPERIMENT"
    assert classify(
        information_flow_green=True, boundary_hits=0,
        mechanics_floor=True, experiment_floor=True, survivor_rule=True,
    ) == "P4S_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE"
    assert classify(
        information_flow_green=True, boundary_hits=0,
        mechanics_floor=True, experiment_floor=True, survivor_rule=False,
    ) == "P4S_ADEQUATE_NO_SURVIVOR"


def test_no_arbitrary_quality_token_cap_is_encoded_in_disposition() -> None:
    source = (ROOT / "scripts" / "theseus_p4s_disposition.py").read_text(
        encoding="utf-8"
    )
    assert '"project_selected_quality_token_cap": None' in source
    assert "sole numeric boundary" in source
    assert "hit invalidates the observation" in source
