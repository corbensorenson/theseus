from argparse import Namespace
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import dogfood_trace_bootstrap as bootstrap  # noqa: E402
import dogfood_trace_event as event  # noqa: E402
import dogfood_trace_training_bridge as bridge  # noqa: E402
import theseus_assistant_runtime as assistant  # noqa: E402

EXPECTED_OUTCOMES = {
    "accepted",
    "missed",
    "ignored",
    "corrected",
    "completed",
    "failed",
    "abstained",
}


def args(outcome: str) -> Namespace:
    return Namespace(
        surface="local_cli",
        assistant_lane="planning_assistant",
        outcome=outcome,
        intent_summary_redacted="bounded local repository check",
        artifact_ref=["reports/local-check.json"],
        error_family="",
        duration_ms=4,
    )


def test_all_required_outcomes_are_accounted_for_without_model_credit() -> None:
    assert event.ALLOWED_OUTCOMES == EXPECTED_OUTCOMES
    assert set(bootstrap.ALLOWED_OUTCOMES) == EXPECTED_OUTCOMES
    assert bridge.ALLOWED_OUTCOMES == EXPECTED_OUTCOMES
    assert assistant.DEFAULT_ALLOWED_FEEDBACK == EXPECTED_OUTCOMES | {""}
    schema = json.loads(
        (ROOT / "configs/assistant_trace_schema.json").read_text(encoding="utf-8")
    )
    assert set(schema["allowed_outcomes"]) == EXPECTED_OUTCOMES
    assert (
        schema["boundaries"]["assisted_behavior_learned_model_credit_allowed"] is False
    )
    for outcome in sorted(EXPECTED_OUTCOMES):
        row = event.build_event(args(outcome))
        assert row["outcome"] == outcome
        assert row["learned_model_credit_allowed"] is False
        assert row["capability_credit"] == "none_assisted_or_tool_mediated"
        assert event.forbidden_fields_absent(row)


def test_paired_experiment_event_has_complete_cost_and_safety_denominator() -> None:
    paired = args("completed")
    paired.experiment_id = "L0-2026-07-31-001"
    paired.task_pair_id = "pair-001"
    paired.arm_id = "full_theseus"
    paired.subsystem_hypothesis_id = "authority_rollback"
    paired.cost_policy_id = "l0_total_contract_cost_v1"
    paired.route_identity = "typed-plan-vcm-governed"
    paired.acceptance_contract_completed = True
    paired.selected_for_use = False
    paired.verifier_state = "passed"
    paired.rollback_state = "verified"
    paired.unsafe_effect_count = 0
    paired.false_block_count = 0
    paired.model_calls = 2
    paired.generated_tokens = 384
    paired.tool_calls = 5
    paired.verification_ms = 1200
    paired.human_review_and_repair_minutes = 1.5
    paired.residual_count = 1
    paired.residual_owner = "planning"
    paired.total_contract_cost_units = 8.75

    row = event.build_event(paired)

    assert set(event.PAIRED_EVENT_FIELDS).issubset(row)
    assert event.paired_experiment_requested(row)
    assert event.paired_experiment_metadata_complete(row)
    assert row["arm_id"] in event.ALLOWED_EXPERIMENT_ARMS
    assert row["verifier_state"] in event.ALLOWED_VERIFIER_STATES
    assert row["rollback_state"] in event.ALLOWED_ROLLBACK_STATES
    assert row["acceptance_contract_completed"] is True
    assert row["selected_for_use"] is False
    assert row["learned_model_credit_allowed"] is False
    assert event.forbidden_fields_absent(row)


def test_partial_paired_experiment_metadata_fails_closed() -> None:
    partial = args("failed")
    partial.experiment_id = "L0-2026-07-31-002"
    row = event.build_event(partial)

    assert event.paired_experiment_requested(row)
    assert event.paired_experiment_metadata_complete(row) is False
