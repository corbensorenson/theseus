from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_semantic_ir_statement_granularity_audit as audit  # noqa: E402


def test_committed_statement_granularity_audit_is_green_and_call_free() -> None:
    report = json.loads(
        (
            ROOT
            / "reports"
            / "theseus_semantic_ir_statement_granularity_audit.json"
        ).read_text(encoding="utf-8")
    )
    comparison = report["task_04_consumed_surface_mechanics_comparison"]

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert comparison["historical_nearest_container_line_span"] == 433
    assert comparison["repaired_exact_statement_line_span"] == 1
    assert comparison["mutation_span_reduction_factor"] == 433.0
    assert len(report["unexposed_and_consumed_parent_surface_addressability"]) == 15
    assert all(value == 0 for value in report["counters"].values())
    assert report["fresh_task_requirements"][
        "consumed_tasks_01_through_04_may_not_be_rerun"
    ] is True


def test_audit_replays_deterministically() -> None:
    expected = json.loads(
        (
            ROOT
            / "reports"
            / "theseus_semantic_ir_statement_granularity_audit.json"
        ).read_text(encoding="utf-8")
    )
    observed = audit.audit(audit.DEFAULT_POOL, audit.DEFAULT_INTERRUPTION)
    for key in (
        "trigger_state",
        "state",
        "faults",
        "task_04_consumed_surface_mechanics_comparison",
        "unexposed_and_consumed_parent_surface_addressability",
        "fresh_task_requirements",
        "counters",
    ):
        assert observed[key] == expected[key]
