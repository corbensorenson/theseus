from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from octopus_router import build_router_head_activation, route_phrase_present, route_task, route_tokens
from octopus_router_head_gate import build_gate
from train_octopus_router_head import (
    build_trace_dataset,
    forbidden_router_trace_paths,
    phrase_in_tokens,
    schema_bound_real_trace,
)


def arm(name: str, *, eligible: bool = True, lifecycle: str = "active", keyword: str = "code") -> dict:
    return {
        "arm_name": name,
        "routing_keywords": [keyword],
        "lifecycle_status": lifecycle,
        "freshness": {"requires_revalidation": False},
        "registry_route": {"routing_eligible": eligible},
        "permission_boundary": {
            "memory": [f"arm/{name}"],
            "tools": "allowlisted_only",
            "side_effects": ["read_reports"],
            "network": "disabled_for_inner_loop",
            "external_inference": "forbidden",
        },
        "runtime_tier": "E1_structured_workflow",
        "cost_profile": {"cold_start_ms": 1, "warm_call_ms": 1, "resident_memory_mb": 1},
    }


def test_router_fails_closed_before_returning_ineligible_arms() -> None:
    decision = route_task(
        "write code",
        "low",
        [arm("head_router"), arm("code_arm", eligible=False)],
        expected_pattern="single",
    )
    assert decision["route_state"] == "BLOCKED"
    assert decision["selected_arms"] == []
    assert decision["permission_envelopes"] == {}
    assert decision["typed_faults"][0]["failure_behavior"] == "reject_without_fallback"


def test_probationary_simulation_arm_is_never_routed_to_live_hardware() -> None:
    arms = [
        arm("head_router"),
        arm("drone_racing_control_arm", lifecycle="probationary_simulation_only", keyword="drone"),
        arm("safety_reflex_arm", keyword="safety"),
    ]
    simulation = route_task("run drone SITL simulation", "low", arms, expected_pattern="single")
    dry_run = route_task("prepare drone telemetry with no live hardware connection", "high", arms, expected_pattern="verification")
    live = route_task("run drone takeoff on real hardware", "high", arms, expected_pattern="reflex")
    assert simulation["route_state"] == "READY"
    assert "drone_racing_control_arm" in simulation["selected_arms"]
    assert dry_run["route_state"] == "READY"
    assert "drone_racing_control_arm" in dry_run["selected_arms"]
    assert live["route_state"] == "BLOCKED"
    assert live["selected_arms"] == []


def test_schema_bound_trace_rejects_answer_metadata_and_missing_counters() -> None:
    base = {
        "trace_id": "trace-1",
        "task": "route a code review",
        "source": "local_dogfood",
        "permission_envelopes": {
            "code_arm": {"external_inference": "forbidden", "runtime_tier": "E1", "tools": []}
        },
    }
    assert not schema_bound_real_trace(base, ["code_arm"])
    clean = {
        **base,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "candidate_generation_credit": 0,
    }
    assert schema_bound_real_trace(clean, ["code_arm"])
    contaminated = {**clean, "solution": "return 42"}
    assert forbidden_router_trace_paths(contaminated) == ["solution"]
    assert not schema_bound_real_trace(contaminated, ["code_arm"])


def test_learned_router_never_activates_without_independent_green_gate() -> None:
    activation = build_router_head_activation(
        {"promotion_gate_passed": True},
        {"promotion_gate_passed": True},
        {"model_type": "sparse_centroid_multilabel_router_v0", "centroids": {"code_arm": {"bias": 1.0}}},
        {"trigger_state": "RED"},
    )
    assert activation["active"] is False
    assert "independent_gate_not_green" in activation["blocked_reasons"]


def test_router_head_holdout_is_disjoint_by_source_case() -> None:
    decisions = [
        {
            "task_id": f"route-{index}",
            "task": f"route code task {index}",
            "risk": "low",
            "pattern": "single",
            "expected": ["code_arm"],
        }
        for index in range(8)
    ]
    rows = build_trace_dataset({"decisions": decisions}, [arm("head_router"), arm("code_arm")])
    train_sources = {row["source_task_id"] for row in rows if row["split"] == "train"}
    holdout_sources = {row["source_task_id"] for row in rows if row["split"] == "holdout"}
    assert train_sources
    assert holdout_sources
    assert train_sources.isdisjoint(holdout_sources)


def test_router_keywords_match_token_boundaries_not_substrings() -> None:
    assert phrase_in_tokens("loop closure", ["run", "loop", "closure", "tool"])
    assert phrase_in_tokens("race", ["run", "the", "race"])
    assert not phrase_in_tokens("race", ["trace", "record"])
    assert not phrase_in_tokens("windows", ["residual", "windowsill"])
    assert route_phrase_present("race", route_tokens("run the race"))
    assert not route_phrase_present("race", route_tokens("write a trace record"))


def test_valid_negative_router_experiment_is_green_but_not_adopted(tmp_path: Path) -> None:
    args = argparse.Namespace(min_contrastive_accuracy=0.95)
    examples = []
    decisions = []
    negatives = []
    contrastive_decisions = []
    for index, arms in enumerate((["code_arm"], ["tool_arm"], ["code_arm", "tool_arm"]), start=1):
        trace_id = f"trace-{index}"
        examples.append(
            {
                "trace_id": trace_id,
                "source_task_id": f"source-{index}",
                "split": "holdout",
                "task": f"task {index}",
                "risk": "low",
                "expected_arms": arms,
                "source": "real_workflow_trace",
                "schema_bound": True,
                "forbidden_metadata_paths": [],
            }
        )
        decisions.append(
            {
                "trace_id": trace_id,
                "risk": "low",
                "expected_arms": arms,
                "predicted_arms": [],
            }
        )
        negatives.append({"split": "holdout"})
        contrastive_decisions.append({"split": "holdout"})
    records = [{"record_type": kind, "public_training_rows_written": 0, "external_inference_calls": 0, "fallback_return_count": 0}
               for kind in {
                   "routing_decision", "authority_use_receipt", "resource_budget", "generation_mode",
                   "failure_boundary", "artifact_graph_record", "claim_record",
                   "evidence_transition_record", "residual_record",
               }]
    for row in records:
        if row["record_type"] == "generation_mode":
            row.update({"learned_generation_claim_allowed": False, "candidate_generation_credit": 0})
    report = {
        "policy": "local_only_no_external_inference",
        "trace_summary": {
            "real_trace_examples": 3,
            "schema_bound_real_trace_examples": 3,
            "contrastive_negatives": 3,
            "contrastive_holdout_negatives": 3,
        },
        "metrics": {"exact_set_accuracy": 0.0, "risk_routing_accuracy": 1.0, "contrastive_negative_accuracy": 0.0},
        "promotion_gate_passed": False,
        "learned_generation_claim_allowed": False,
        "candidate_generation_credit": 0,
        "viea_router_head_records": records,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    dataset = {
        "policy": "local_only_no_external_inference",
        "summary": {
            "examples": 3, "train": 0, "holdout": 3, "real_trace_examples": 3,
            "schema_bound_real_trace_examples": 3, "contrastive_negatives": 3,
            "contrastive_holdout_negatives": 3,
        },
        "examples": examples,
        "contrastive_negatives": negatives,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    evaluation = {
        "policy": "local_only_no_external_inference",
        "metrics": report["metrics"],
        "promotion_gate_passed": False,
        "decisions": decisions,
        "contrastive_decisions": contrastive_decisions,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    report_path = tmp_path / "report.json"
    dataset_path = tmp_path / "dataset.json"
    eval_path = tmp_path / "eval.json"
    for path in (report_path, dataset_path, eval_path):
        path.write_text("{}\n", encoding="utf-8")
    result = build_gate(
        args, report_path, dataset_path, eval_path,
        report, dataset, evaluation, 0.0,
    )
    assert result["trigger_state"] == "GREEN"
    assert result["hard_gaps"] == []
    assert result["summary"]["adoption_state"] == "NOT_ADOPTED"
    assert {row["kind"] for row in result["adoption_gaps"]} == {
        "contrastive_negative_accuracy_below_floor", "holdout_exact_set_accuracy_below_floor", "promotion_gate_not_passed"
    }


def test_router_head_gate_recomputes_dataset_instead_of_trusting_summary(tmp_path: Path) -> None:
    args = argparse.Namespace(min_contrastive_accuracy=0.95)
    report = {
        "policy": "local_only_no_external_inference",
        "trace_summary": {
            "real_trace_examples": 3,
            "schema_bound_real_trace_examples": 3,
            "contrastive_negatives": 3,
            "contrastive_holdout_negatives": 3,
        },
        "metrics": {"exact_set_accuracy": 1.0, "risk_routing_accuracy": 1.0, "contrastive_negative_accuracy": 1.0},
        "promotion_gate_passed": True,
        "learned_generation_claim_allowed": False,
        "candidate_generation_credit": 0,
        "viea_router_head_records": [],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    dataset = {
        "policy": "local_only_no_external_inference",
        "summary": {"examples": 3, "real_trace_examples": 3, "schema_bound_real_trace_examples": 3},
        "examples": [],
        "contrastive_negatives": [],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    evaluation = {
        "policy": "local_only_no_external_inference",
        "metrics": report["metrics"],
        "promotion_gate_passed": True,
        "decisions": [],
        "contrastive_decisions": [],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    result = build_gate(
        args,
        tmp_path / "report.json",
        tmp_path / "dataset.json",
        tmp_path / "eval.json",
        report,
        dataset,
        evaluation,
        0.0,
    )
    assert result["trigger_state"] == "RED"
    kinds = {row["kind"] for row in result["hard_gaps"]}
    assert "dataset_examples_missing" in kinds
    assert "real_trace_summary_content_mismatch" in kinds
