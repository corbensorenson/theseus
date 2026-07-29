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
