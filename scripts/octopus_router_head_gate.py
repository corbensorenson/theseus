#!/usr/bin/env python3
"""Gate the learned Octopus router head without granting generation credit."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REPORT = "reports/octopus_router_head_report.json"
DEFAULT_DATASET = "reports/octopus_router_trace_dataset.json"
DEFAULT_EVAL = "reports/octopus_router_head_eval.json"
DEFAULT_OUT = "reports/octopus_router_head_gate.json"

REQUIRED_RECORD_TYPES = {
    "routing_decision",
    "authority_use_receipt",
    "resource_budget",
    "generation_mode",
    "failure_boundary",
    "artifact_graph_record",
    "claim_record",
    "evidence_transition_record",
    "residual_record",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--eval", default=DEFAULT_EVAL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--min-contrastive-accuracy", type=float, default=0.95)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    report_path = Path(args.report)
    dataset_path = Path(args.dataset)
    eval_path = Path(args.eval)
    report = read_json(report_path)
    dataset = read_json(dataset_path)
    evaluation = read_json(eval_path)
    gate = build_gate(args, report_path, dataset_path, eval_path, report, dataset, evaluation, started)
    write_json(Path(args.out), gate)
    print(json.dumps(gate_view(gate) if args.gate else gate, indent=2, sort_keys=True))
    return 2 if gate["trigger_state"] == "RED" else 0


def build_gate(
    args: argparse.Namespace,
    report_path: Path,
    dataset_path: Path,
    eval_path: Path,
    report: dict[str, Any],
    dataset: dict[str, Any],
    evaluation: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    trace_summary = dict_value(report.get("trace_summary"))
    dataset_summary = dict_value(dataset.get("summary"))
    metrics = dict_value(evaluation.get("metrics") or report.get("metrics"))
    records = list_dicts(report.get("viea_router_head_records"))
    examples = list_dicts(dataset.get("examples"))
    contrastive_negatives = list_dicts(dataset.get("contrastive_negatives"))
    eval_decisions = list_dicts(evaluation.get("decisions"))
    contrastive_decisions = list_dicts(evaluation.get("contrastive_decisions"))
    observed = {str(row.get("record_type") or "") for row in records}
    missing_records = sorted(REQUIRED_RECORD_TYPES - observed)

    if not report_path.exists():
        hard_gaps.append(gap("report_missing", {"path": str(report_path)}))
    if not dataset_path.exists():
        hard_gaps.append(gap("dataset_missing", {"path": str(dataset_path)}))
    if not eval_path.exists():
        hard_gaps.append(gap("eval_missing", {"path": str(eval_path)}))
    if report.get("policy") != "local_only_no_external_inference":
        hard_gaps.append(gap("report_policy_mismatch", {"policy": report.get("policy")}))
    if dataset.get("policy") != "local_only_no_external_inference":
        hard_gaps.append(gap("dataset_policy_mismatch", {"policy": dataset.get("policy")}))
    if evaluation.get("policy") != "local_only_no_external_inference":
        hard_gaps.append(gap("eval_policy_mismatch", {"policy": evaluation.get("policy")}))
    actual_real = [row for row in examples if row.get("source") == "real_workflow_trace"]
    actual_schema_real = [
        row
        for row in actual_real
        if row.get("schema_bound") is True and not list(row.get("forbidden_metadata_paths") or [])
    ]
    unique_real_tasks = {str(row.get("task") or "").strip() for row in actual_schema_real if str(row.get("task") or "").strip()}
    unique_real_labelsets = {
        tuple(sorted(str(item) for item in row.get("expected_arms") or []))
        for row in actual_schema_real
    }
    if not examples:
        hard_gaps.append(gap("dataset_examples_missing", {"dataset_summary": dataset_summary}))
    if int(trace_summary.get("real_trace_examples") or 0) != len(actual_real):
        hard_gaps.append(gap("real_trace_summary_content_mismatch", {"reported": trace_summary.get("real_trace_examples"), "actual": len(actual_real)}))
    if int(trace_summary.get("schema_bound_real_trace_examples") or 0) != len(actual_schema_real):
        hard_gaps.append(gap("schema_trace_summary_content_mismatch", {"reported": trace_summary.get("schema_bound_real_trace_examples"), "actual": len(actual_schema_real)}))
    if len(actual_real) <= 0:
        hard_gaps.append(gap("real_trace_examples_missing", {"trace_summary": trace_summary}))
    if len(actual_schema_real) <= 0:
        hard_gaps.append(gap("schema_bound_real_trace_examples_missing", {"trace_summary": trace_summary}))
    if len(unique_real_tasks) < 3 or len(unique_real_labelsets) < 2:
        hard_gaps.append(
            gap(
                "real_trace_diversity_below_floor",
                {
                    "unique_task_count": len(unique_real_tasks),
                    "minimum_unique_tasks": 3,
                    "unique_labelset_count": len(unique_real_labelsets),
                    "minimum_unique_labelsets": 2,
                },
            )
        )
    if int(trace_summary.get("contrastive_holdout_negatives") or 0) <= 0:
        hard_gaps.append(gap("contrastive_holdout_negatives_missing", {"trace_summary": trace_summary}))
    if float(metrics.get("contrastive_negative_accuracy") or 0.0) < args.min_contrastive_accuracy:
        hard_gaps.append(
            gap(
                "contrastive_negative_accuracy_below_floor",
                {
                    "observed": metrics.get("contrastive_negative_accuracy"),
                    "required": args.min_contrastive_accuracy,
                    "metrics": metrics,
                },
            )
        )
    if float(metrics.get("exact_set_accuracy") or 0.0) < 0.95:
        hard_gaps.append(gap("holdout_exact_set_accuracy_below_floor", {"metrics": metrics}))
    if float(metrics.get("risk_routing_accuracy") or 0.0) < 1.0:
        hard_gaps.append(gap("risk_routing_accuracy_below_floor", {"metrics": metrics}))
    if report.get("promotion_gate_passed") is not True or evaluation.get("promotion_gate_passed") is not True:
        hard_gaps.append(
            gap(
                "promotion_gate_not_passed",
                {
                    "report_promotion_gate_passed": report.get("promotion_gate_passed"),
                    "eval_promotion_gate_passed": evaluation.get("promotion_gate_passed"),
                },
            )
        )
    if report.get("learned_generation_claim_allowed") is not False:
        hard_gaps.append(gap("learned_generation_boundary_missing", {"learned_generation_claim_allowed": report.get("learned_generation_claim_allowed")}))
    if int(report.get("candidate_generation_credit") or 0) != 0:
        hard_gaps.append(gap("router_head_has_candidate_generation_credit", {"candidate_generation_credit": report.get("candidate_generation_credit")}))
    no_cheat_faults = no_cheat_faults_for([report, dataset, evaluation, *records])
    if no_cheat_faults:
        hard_gaps.append(gap("no_cheat_counter_fault", {"faults": no_cheat_faults[:20]}))
    generation_faults = [
        row
        for row in records
        if row.get("record_type") == "generation_mode"
        and (row.get("learned_generation_claim_allowed") is not False or int(row.get("candidate_generation_credit") or 0) != 0)
    ]
    if generation_faults:
        hard_gaps.append(gap("generation_mode_record_allows_generation_credit", {"count": len(generation_faults)}))
    if missing_records:
        hard_gaps.append(gap("viea_router_head_records_missing", {"missing": missing_records}))
    actual_summary = {
        "examples": len(examples),
        "train": sum(1 for row in examples if row.get("split") == "train"),
        "holdout": sum(1 for row in examples if row.get("split") == "holdout"),
        "real_trace_examples": len(actual_real),
        "schema_bound_real_trace_examples": len(actual_schema_real),
        "contrastive_negatives": len(contrastive_negatives),
        "contrastive_holdout_negatives": sum(1 for row in contrastive_negatives if row.get("split") == "holdout"),
    }
    summary_mismatches = {
        key: {"reported": dataset_summary.get(key), "actual": value}
        for key, value in actual_summary.items()
        if int(dataset_summary.get(key) or 0) != value
    }
    if summary_mismatches:
        hard_gaps.append(gap("dataset_summary_content_mismatch", summary_mismatches))
    if int(dataset_summary.get("contrastive_negatives") or 0) != int(trace_summary.get("contrastive_negatives") or 0):
        hard_gaps.append(
            gap(
                "dataset_report_contrastive_count_mismatch",
                {
                    "dataset": dataset_summary.get("contrastive_negatives"),
                    "report": trace_summary.get("contrastive_negatives"),
                },
            )
        )
    train_sources = {str(row.get("source_task_id") or "") for row in examples if row.get("split") == "train"}
    holdout_sources = {str(row.get("source_task_id") or "") for row in examples if row.get("split") == "holdout"}
    leaked_sources = sorted((train_sources & holdout_sources) - {""})
    if leaked_sources:
        hard_gaps.append(gap("holdout_source_overlap", {"count": len(leaked_sources), "examples": leaked_sources[:20]}))
    expected_holdout_ids = {str(row.get("trace_id") or "") for row in examples if row.get("split") == "holdout"}
    observed_eval_ids = {str(row.get("trace_id") or "") for row in eval_decisions}
    if expected_holdout_ids != observed_eval_ids:
        hard_gaps.append(gap("eval_decision_dataset_mismatch", {"expected_count": len(expected_holdout_ids), "observed_count": len(observed_eval_ids)}))
    recomputed_exact = sum(
        1
        for row in eval_decisions
        if set(row.get("expected_arms") or []) == set(row.get("predicted_arms") or [])
    ) / max(1, len(eval_decisions))
    recomputed_risk = sum(
        1
        for row in eval_decisions
        if row.get("risk") not in {"high", "critical"} or "safety_reflex_arm" in set(row.get("predicted_arms") or [])
    ) / max(1, len(eval_decisions))
    if float(metrics.get("exact_set_accuracy") or 0.0) != round(recomputed_exact, 4):
        hard_gaps.append(gap("exact_accuracy_not_replayable", {"reported": metrics.get("exact_set_accuracy"), "recomputed": recomputed_exact}))
    if float(metrics.get("risk_routing_accuracy") or 0.0) != round(recomputed_risk, 4):
        hard_gaps.append(gap("risk_accuracy_not_replayable", {"reported": metrics.get("risk_routing_accuracy"), "recomputed": recomputed_risk}))
    expected_contrastive_count = actual_summary["contrastive_holdout_negatives"]
    if len(contrastive_decisions) != expected_contrastive_count:
        hard_gaps.append(gap("contrastive_decision_dataset_mismatch", {"expected": expected_contrastive_count, "observed": len(contrastive_decisions)}))

    trigger_state = "GREEN" if not hard_gaps else "RED"
    return {
        "policy": "project_theseus_octopus_router_head_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "report": str(report_path),
            "dataset": str(dataset_path),
            "eval": str(eval_path),
            "real_trace_examples": int(trace_summary.get("real_trace_examples") or 0),
            "schema_bound_real_trace_examples": int(trace_summary.get("schema_bound_real_trace_examples") or 0),
            "contrastive_negatives": int(trace_summary.get("contrastive_negatives") or 0),
            "contrastive_holdout_negatives": int(trace_summary.get("contrastive_holdout_negatives") or 0),
            "contrastive_negative_accuracy": metrics.get("contrastive_negative_accuracy"),
            "exact_set_accuracy": metrics.get("exact_set_accuracy"),
            "risk_routing_accuracy": metrics.get("risk_routing_accuracy"),
            "promotion_gate_passed": bool(report.get("promotion_gate_passed")) and bool(evaluation.get("promotion_gate_passed")),
            "learned_generation_claim_allowed": report.get("learned_generation_claim_allowed"),
            "candidate_generation_credit": int(report.get("candidate_generation_credit") or 0),
            "viea_router_head_record_count": len(records),
            "dataset_example_count": len(examples),
            "actual_schema_bound_real_trace_examples": len(actual_schema_real),
            "unique_schema_bound_real_task_count": len(unique_real_tasks),
            "unique_schema_bound_real_labelset_count": len(unique_real_labelsets),
            "holdout_source_overlap_count": len(leaked_sources),
            "missing_viea_router_head_records": missing_records,
            "hard_gap_count": len(hard_gaps),
            "warning_count": len(warnings),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "metrics": metrics,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "rules": {
            "trace_source": "Learned router-head promotion needs at least one schema-bound real task-to-arm trace, not seeded cases only.",
            "contrastive": "The head must score the true arm set above hard wrong-labelset negatives on holdout examples.",
            "non_generation": "Router-head selections are capability routing evidence only and never learned code-generation evidence.",
        },
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "summary": report.get("summary"),
        "hard_gaps": report.get("hard_gaps"),
        "warnings": report.get("warnings"),
    }


def gap(kind: str, detail: dict[str, Any], *, severity: str = "hard_gap") -> dict[str, Any]:
    return {"kind": kind, "severity": severity, "detail": detail}


def no_cheat_faults_for(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    faults = []
    for index, row in enumerate(rows):
        for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count"):
            if key not in row:
                faults.append({"index": index, "key": key, "value": "missing"})
            elif int(row.get(key) or 0) != 0:
                faults.append({"index": index, "key": key, "value": row.get(key)})
    return faults


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
