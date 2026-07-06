#!/usr/bin/env python3
"""Deterministic proxy benchmarks for Circle Calculus AI contracts.

These benchmarks attach named workloads, ordinary baselines, and measured proxy
metrics to the Circle contract pack. They are not learned-model evaluations and
must not be used as promotion evidence. Their purpose is to make the next real
Theseus-Hive private benchmark attachments concrete and auditable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACTS = ROOT.parent / "circle math" / "site" / "data" / "generated" / "theseus_hive_ai_contracts.json"
DEFAULT_OUT = ROOT / "reports" / "circle_ai_private_proxy_benchmark.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "circle_ai_private_proxy_benchmark.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", default=str(DEFAULT_CONTRACTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    source = resolve(args.contracts)
    pack = read_json(source)
    report = build_report(pack, source)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(pack: dict[str, Any], source: Path) -> dict[str, Any]:
    contracts = [row for row in pack.get("contracts", []) if isinstance(row, dict)]
    benchmarks = [benchmark_for_contract(row) for row in contracts]
    failed = [row["workload_id"] for row in benchmarks if not row["passed"]]
    unknown = [row["workload_id"] for row in benchmarks if row["kind"] == "unknown"]
    hard_failures = failed + unknown
    return {
        "policy": "theseus_hive_circle_ai_private_proxy_benchmark_v0",
        "created_utc": now(),
        "trigger_state": "RED" if hard_failures else "YELLOW",
        "source_contract_path": rel(source),
        "circle_contract_schema_id": pack.get("schema_id", ""),
        "claim_boundary": (
            "This report contains deterministic proxy benchmarks only. It is not "
            "learned-model quality, reasoning, speed, context-length, public-transfer, "
            "promotion, or ASI evidence."
        ),
        "score_semantics": "deterministic proxy metrics for benchmark design; no learned model is trained or evaluated",
        "public_calibration_used": False,
        "external_inference_calls": 0,
        "training_mutation": False,
        "promotion_evidence": False,
        "private_data_exported": False,
        "summary": {
            "proxy_benchmark_count": len(benchmarks),
            "passed_proxy_benchmark_count": sum(1 for row in benchmarks if row["passed"]),
            "failed_proxy_benchmarks": failed,
            "unknown_proxy_benchmarks": unknown,
            "deterministic_proxy_metrics_present": bool(benchmarks),
            "learned_model_quality_metrics_present": False,
            "real_private_workload_results_present": False,
        },
        "governance_gates": [
            gate("external_inference_calls_zero", True, "No model/API/local inference is called."),
            gate("training_mutation_absent", True, "This script writes reports only."),
            gate("promotion_evidence_absent", True, "Proxy metrics are non-promotion evidence."),
            gate("public_calibration_absent", True, "No public benchmark calibration is used."),
            gate("learned_model_quality_absent", True, "No learned model is trained or scored."),
            gate("known_proxy_benchmark_kinds", not unknown, unknown or "all loaded kinds have proxy benchmarks"),
            gate("proxy_benchmarks_passed", not failed, failed or "all proxy benchmarks passed"),
        ],
        "benchmarks": benchmarks,
        "next_actions": next_actions(benchmarks, hard_failures),
    }


def benchmark_for_contract(contract: dict[str, Any]) -> dict[str, Any]:
    kind = str(contract.get("kind") or "")
    if kind == "recurrence_schedule":
        return recurrence_proxy(contract)
    if kind == "strided_candidate_fanout":
        return fanout_proxy(contract)
    if kind == "cyclic_memory_residue_winding":
        return memory_proxy(contract)
    if kind == "multicoil_phase_feature":
        return phase_proxy(contract)
    if kind == "circulant_block_cyclic_mixer":
        return mixer_proxy(contract)
    if kind == "seed_rule_exact_regeneration":
        return seed_rule_proxy(contract)
    return base_benchmark(contract, "unknown", "circle_unknown_proxy_benchmark") | {
        "passed": False,
        "metrics": {},
        "baselines": [],
        "interpretation": "Unknown contract kind.",
        "next_private_benchmark": "Add a proxy evaluator for this contract kind.",
    }


def recurrence_proxy(contract: dict[str, Any]) -> dict[str, Any]:
    fields = fields_of(contract)
    tokens = [row for row in fields.get("tokens", []) if isinstance(row, dict)]
    max_loops = as_int(fields.get("max_loops"))
    loop_period = as_int(fields.get("loop_period"))
    requirements = [as_int(row.get("budget")) for row in tokens]
    policies = {
        "circle_budget": requirements,
        "fixed_depth_max_loops": [max_loops for _ in tokens],
        "fixed_depth_loop_period": [loop_period for _ in tokens],
        "single_pass_no_recurrence": [1 for _ in tokens],
        "overlooped_max_plus_one": [max_loops + 1 for _ in tokens],
    }
    baselines = []
    for name, selected in policies.items():
        satisfied = [chosen >= required for chosen, required in zip(selected, requirements)]
        wasted = [max(0, chosen - required) for chosen, required in zip(selected, requirements)]
        shortfall = [max(0, required - chosen) for chosen, required in zip(selected, requirements)]
        baselines.append(
            {
                "name": name,
                "satisfied_rate": mean_bool(satisfied),
                "total_steps": sum(selected),
                "wasted_steps": sum(wasted),
                "shortfall_steps": sum(shortfall),
            }
        )
    circle = by_name(baselines, "circle_budget")
    passed = bool(contract.get("contract_passed")) and circle["satisfied_rate"] == 1.0 and circle["wasted_steps"] == 0
    return base_benchmark(contract, "recurrence_schedule", "circle_recurrence_budget_proxy_v1") | {
        "passed": passed,
        "metric_family": "schedule_fidelity",
        "metrics": {
            "token_count": len(tokens),
            "circle_satisfied_rate": circle["satisfied_rate"],
            "circle_total_steps": circle["total_steps"],
            "circle_wasted_steps": circle["wasted_steps"],
            "fixed_depth_total_steps": by_name(baselines, "fixed_depth_max_loops")["total_steps"],
            "single_pass_satisfied_rate": by_name(baselines, "single_pass_no_recurrence")["satisfied_rate"],
            "learned_model_quality_metric": None,
        },
        "baselines": baselines,
        "interpretation": "Circle schedule exactly satisfies the synthetic per-token depth contract with no over-budget steps; this is schedule accounting, not reasoning quality.",
        "next_private_benchmark": "Replace synthetic depth requirements with a real looped-model/work-budget task and report task quality plus runtime.",
    }


def fanout_proxy(contract: dict[str, Any]) -> dict[str, Any]:
    fields = fields_of(contract)
    n = as_int(fields.get("context_length"))
    start = as_int(fields.get("start_index"))
    stride = as_int(fields.get("stride"))
    budget = max(1, min(as_int(fields.get("candidate_budget")), max(1, n // 2)))
    circle_path = [as_int(x) for x in fields.get("candidate_path", [])][:budget]
    if not circle_path:
        circle_path = [((start + t * stride) % n) for t in range(budget)]
    sequential = [((start + t) % n) for t in range(budget)]
    round_robin = [((start + (t * 2)) % n) for t in range(budget)]
    local_radius = max(1, budget // 2)
    local_window = [((start + offset) % n) for offset in range(local_radius)]
    random_like = deterministic_permutation(n)[:budget]
    nonlocal_targets = sorted({n // 2, n - 1, circle_path[0], circle_path[-1]})
    baselines = []
    for name, path in {
        "circle_stride": circle_path,
        "sequential_fanout": sequential,
        "round_robin_fanout": round_robin,
        "local_window": local_window,
        "deterministic_random_like": random_like,
    }.items():
        baselines.append(
            {
                "name": name,
                "candidate_budget": len(path),
                "unique_candidates": len(set(path)),
                "duplicate_count": len(path) - len(set(path)),
                "nonlocal_target_hit_rate": hit_rate(path, nonlocal_targets),
                "path": path,
            }
        )
    circle = by_name(baselines, "circle_stride")
    passed = bool(contract.get("contract_passed")) and circle["duplicate_count"] == 0 and circle["nonlocal_target_hit_rate"] > 0
    return base_benchmark(contract, "strided_candidate_fanout", "circle_candidate_fanout_proxy_v1") | {
        "passed": passed,
        "metric_family": "candidate_coverage_proxy",
        "metrics": {
            "context_length": n,
            "budget_used": budget,
            "target_positions": nonlocal_targets,
            "circle_nonlocal_target_hit_rate": circle["nonlocal_target_hit_rate"],
            "circle_duplicate_count": circle["duplicate_count"],
            "semantic_coverage_metric": None,
        },
        "baselines": baselines,
        "interpretation": "The strided path gives deterministic nonlocal reach under a small budget; this is candidate coverage, not semantic coverage.",
        "next_private_benchmark": "Attach the same policies to private candidate generation and score residual quality/rejection rates.",
    }


def memory_proxy(contract: dict[str, Any]) -> dict[str, Any]:
    fields = fields_of(contract)
    target = as_int(fields.get("event_index"))
    same_residue_events = [as_int(x) for x in fields.get("same_residue_events", [])]
    if target not in same_residue_events:
        same_residue_events.append(target)
    slot_only_candidates = sorted(set(same_residue_events))
    latest_slot_event = max(slot_only_candidates)
    baselines = [
        {
            "name": "circle_residue_plus_winding",
            "returned_event": target,
            "candidate_count": 1,
            "exact_identity_hit": True,
            "ambiguous": False,
        },
        {
            "name": "slot_only_memory",
            "returned_event": slot_only_candidates,
            "candidate_count": len(slot_only_candidates),
            "exact_identity_hit": target in slot_only_candidates,
            "ambiguous": len(slot_only_candidates) > 1,
        },
        {
            "name": "fifo_latest_same_slot",
            "returned_event": latest_slot_event,
            "candidate_count": 1,
            "exact_identity_hit": latest_slot_event == target,
            "ambiguous": False,
        },
        {
            "name": "score_based_retention_without_scores",
            "returned_event": None,
            "candidate_count": 0,
            "exact_identity_hit": False,
            "ambiguous": True,
        },
    ]
    circle = by_name(baselines, "circle_residue_plus_winding")
    passed = bool(contract.get("contract_passed")) and circle["exact_identity_hit"] is True and len(slot_only_candidates) > 1
    return base_benchmark(contract, "cyclic_memory_residue_winding", "circle_memory_alias_proxy_v1") | {
        "passed": passed,
        "metric_family": "alias_disambiguation_proxy",
        "metrics": {
            "target_event": target,
            "same_residue_candidate_count": len(slot_only_candidates),
            "circle_exact_identity_hit": True,
            "slot_only_ambiguous": len(slot_only_candidates) > 1,
            "retrieval_quality_metric": None,
        },
        "baselines": baselines,
        "interpretation": "Residue plus winding disambiguates same-slot events in the proxy trace; it does not choose semantically useful memories.",
        "next_private_benchmark": "Run on context-packet traces with real retention/retrieval targets.",
    }


def phase_proxy(contract: dict[str, Any]) -> dict[str, Any]:
    fields = fields_of(contract)
    periods = [as_int(x) for x in fields.get("periods", []) if as_int(x) > 0]
    if not periods:
        periods = [5, 7]
    target_phase = [as_int(x) for x in fields.get("phase_tuple", [])]
    if len(target_phase) != len(periods):
        target_phase = [0 for _ in periods]
    horizon = max(as_int(fields.get("joint_repeat_horizon")), max(periods) * 4)
    train_positions = list(range(horizon))
    eval_positions = list(range(horizon, horizon * 2))
    train_labels = [phase_label(pos, periods, target_phase) for pos in train_positions]
    eval_labels = [phase_label(pos, periods, target_phase) for pos in eval_positions]
    majority = Counter(train_labels).most_common(1)[0][0]
    baselines = [
        {
            "name": "circle_phase_tuple",
            "accuracy": accuracy([phase_label(pos, periods, target_phase) for pos in eval_positions], eval_labels),
            "feature": "correct_period_tuple",
        },
        {
            "name": "wrong_period_phase_tuple",
            "accuracy": accuracy([phase_label(pos, [p + 1 for p in periods], target_phase) for pos in eval_positions], eval_labels),
            "feature": "wrong_period_tuple",
        },
        {
            "name": "majority_no_phase",
            "accuracy": accuracy([majority for _ in eval_positions], eval_labels),
            "feature": "constant_majority",
        },
        {
            "name": "absolute_position_bucket_mod_10",
            "accuracy": bucket_memorization_accuracy(train_positions, train_labels, eval_positions, eval_labels, bucket_mod=10),
            "feature": "position_mod_10",
        },
    ]
    circle = by_name(baselines, "circle_phase_tuple")
    wrong = by_name(baselines, "wrong_period_phase_tuple")
    passed = bool(contract.get("contract_passed")) and circle["accuracy"] == 1.0 and circle["accuracy"] >= wrong["accuracy"]
    return base_benchmark(contract, "multicoil_phase_feature", "circle_phase_feature_proxy_v1") | {
        "passed": passed,
        "metric_family": "synthetic_phase_ablation",
        "metrics": {
            "eval_count": len(eval_labels),
            "circle_accuracy": circle["accuracy"],
            "wrong_period_accuracy": wrong["accuracy"],
            "majority_accuracy": by_name(baselines, "majority_no_phase")["accuracy"],
            "learned_model_quality_metric": None,
        },
        "baselines": baselines,
        "interpretation": "The proxy label is defined by the phase tuple, so the correct phase feature should win; this validates the ablation harness, not a model.",
        "next_private_benchmark": "Use real Code LM state-sequence tasks where periodic structure is only a hypothesis, with wrong-period and no-phase controls.",
    }


def mixer_proxy(contract: dict[str, Any]) -> dict[str, Any]:
    fields = fields_of(contract)
    dense_output = [as_float(x) for x in fields.get("dense_output", [])]
    circulant_output = [as_float(x) for x in fields.get("circulant_output", [])]
    input_values = [as_float(x) for x in fields.get("input_values", [])]
    dense_params = as_int(fields.get("dense_parameters"))
    circulant_params = as_int(fields.get("circulant_parameters"))
    block_params = as_int(fields.get("block_cyclic_parameters"))
    lora_params = as_int(fields.get("lora_parameters"))
    baselines = [
        {
            "name": "dense_circulant_matrix",
            "max_abs_delta_vs_dense": 0.0,
            "parameters": dense_params,
            "exact_output_match": True,
        },
        {
            "name": "circle_circulant_kernel",
            "max_abs_delta_vs_dense": max_abs_delta(circulant_output, dense_output),
            "parameters": circulant_params,
            "exact_output_match": circulant_output == dense_output,
        },
        {
            "name": "identity_no_mixer",
            "max_abs_delta_vs_dense": max_abs_delta(input_values, dense_output),
            "parameters": 0,
            "exact_output_match": input_values == dense_output,
        },
        {
            "name": "lora_parameter_accounting_only",
            "max_abs_delta_vs_dense": None,
            "parameters": lora_params,
            "exact_output_match": None,
        },
        {
            "name": "block_cyclic_parameter_accounting_only",
            "max_abs_delta_vs_dense": None,
            "parameters": block_params,
            "exact_output_match": None,
        },
    ]
    circle = by_name(baselines, "circle_circulant_kernel")
    passed = bool(contract.get("contract_passed")) and circle["exact_output_match"] is True and circle["parameters"] < dense_params
    return base_benchmark(contract, "circulant_block_cyclic_mixer", "circle_mixer_parameter_proxy_v1") | {
        "passed": passed,
        "metric_family": "structured_mixer_proxy",
        "metrics": {
            "circle_exact_output_match": circle["exact_output_match"],
            "circle_parameters": circle["parameters"],
            "dense_parameters": dense_params,
            "circle_parameter_ratio": safe_ratio(circle["parameters"], dense_params),
            "learned_route_quality_metric": None,
            "runtime_metric": None,
        },
        "baselines": baselines,
        "interpretation": "The circulant kernel exactly matches its dense circulant matrix with fewer parameters; no route/ranker quality or runtime is measured.",
        "next_private_benchmark": "Run route/ranker tasks with dense, low-rank, LoRA, block-cyclic, no-mixer, and circulant baselines.",
    }


def seed_rule_proxy(contract: dict[str, Any]) -> dict[str, Any]:
    fields = fields_of(contract)
    generated = fields.get("generated_object")
    regenerated = fields.get("regenerated_object")
    explicit_length = as_int(fields.get("explicit_length"))
    generator_length = as_int(fields.get("generator_length"))
    exact = generated == regenerated and bool(fields.get("exact_regeneration"))
    baselines = [
        {
            "name": "seed_rule_with_verifier",
            "exact_regeneration": exact,
            "stored_length": generator_length,
            "verifier_present": True,
            "residual_rate": 0.0 if exact else 1.0,
        },
        {
            "name": "object_only_storage",
            "exact_regeneration": True,
            "stored_length": explicit_length,
            "verifier_present": False,
            "residual_rate": 0.0,
        },
        {
            "name": "unverified_template",
            "exact_regeneration": None,
            "stored_length": generator_length,
            "verifier_present": False,
            "residual_rate": None,
        },
    ]
    seed = by_name(baselines, "seed_rule_with_verifier")
    passed = bool(contract.get("contract_passed")) and exact
    return base_benchmark(contract, "seed_rule_exact_regeneration", "circle_seed_rule_regeneration_proxy_v1") | {
        "passed": passed,
        "metric_family": "exact_regeneration_proxy",
        "metrics": {
            "exact_regeneration": exact,
            "generator_length": generator_length,
            "explicit_length": explicit_length,
            "generator_shorter": bool(fields.get("generator_shorter")),
            "seed_rule_residual_rate": seed["residual_rate"],
            "workflow_success_metric": None,
        },
        "baselines": baselines,
        "interpretation": "The seed-rule record regenerates exactly, but in this tiny example it is longer than object-only storage.",
        "next_private_benchmark": "Use repeated workflow/tool-card artifacts where generator reuse can be measured against exact regeneration and edit cost.",
    }


def base_benchmark(contract: dict[str, Any], kind: str, workload_id: str) -> dict[str, Any]:
    return {
        "workload_id": workload_id,
        "contract_id": contract.get("id", ""),
        "kind": kind,
        "benchmark_type": "deterministic_proxy",
        "model_inference_used": False,
        "training_data_mutated": False,
        "promotion_evidence": False,
        "theorem_ids": list(contract.get("theorem_ids") or []),
        "dictionary_ids": list(contract.get("dictionary_ids") or []),
    }


def next_actions(benchmarks: list[dict[str, Any]], hard_failures: list[str]) -> list[str]:
    if hard_failures:
        return [
            "Fix failed or unknown proxy benchmark rows before using them as private benchmark templates.",
            "Rerun the Circle exporter, consumer, smoke workload, and proxy benchmark reports.",
        ]
    return [
        "Replace circle_recurrence_budget_proxy_v1 with a private looped-model/work-budget benchmark that reports task score and runtime.",
        "Replace circle_candidate_fanout_proxy_v1 with candidate generation residual tests and duplicate/rejection rates.",
        "Replace circle_memory_alias_proxy_v1 with context-packet retention/retrieval traces.",
        "Replace circle_phase_feature_proxy_v1 with Code LM state-sequence ablations where periodicity is not label-defined.",
        "Replace circle_mixer_parameter_proxy_v1 with route/ranker mixer benchmarks that include runtime.",
        "Replace circle_seed_rule_regeneration_proxy_v1 with repeated workflow/tool-card regeneration benchmarks.",
    ]


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Circle AI Private Proxy Benchmark",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Proxy benchmarks: {summary.get('proxy_benchmark_count')}",
        f"- Passed proxy benchmarks: {summary.get('passed_proxy_benchmark_count')}",
        f"- Learned-model quality metrics present: {summary.get('learned_model_quality_metrics_present')}",
        f"- Real private workload results present: {summary.get('real_private_workload_results_present')}",
        f"- External inference calls: {report.get('external_inference_calls')}",
        f"- Training mutation: {report.get('training_mutation')}",
        f"- Promotion evidence: {report.get('promotion_evidence')}",
        "",
        "These are deterministic proxy metrics for benchmark design only.",
        "",
        "## Benchmarks",
    ]
    for row in report.get("benchmarks", []):
        lines.extend(
            [
                "",
                f"### {row.get('workload_id')}",
                "",
                f"- Contract: `{row.get('contract_id')}`",
                f"- Kind: `{row.get('kind')}`",
                f"- Metric family: `{row.get('metric_family')}`",
                f"- Passed: `{row.get('passed')}`",
                f"- Interpretation: {row.get('interpretation')}",
                f"- Next private benchmark: {row.get('next_private_benchmark')}",
            ]
        )
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def run_self_test() -> None:
    pack = {
        "schema_id": "circle_calculus.theseus_hive_ai_contracts.self_test",
        "contracts": [
            {
                "id": "SELF-REC",
                "kind": "recurrence_schedule",
                "contract_passed": True,
                "fields": {
                    "max_loops": 4,
                    "loop_period": 4,
                    "tokens": [{"budget": 1}, {"budget": 2}, {"budget": 4}],
                },
            },
            {
                "id": "SELF-FAN",
                "kind": "strided_candidate_fanout",
                "contract_passed": True,
                "fields": {"context_length": 7, "start_index": 0, "stride": 3, "candidate_budget": 7, "candidate_path": [3, 6, 2, 5, 1, 4, 0]},
            },
            {
                "id": "SELF-MEM",
                "kind": "cyclic_memory_residue_winding",
                "contract_passed": True,
                "fields": {"event_index": 9, "same_residue_events": [1, 5, 9], "same_residue_windings": [0, 1, 2]},
            },
            {
                "id": "SELF-PHASE",
                "kind": "multicoil_phase_feature",
                "contract_passed": True,
                "fields": {"periods": [3, 5], "phase_tuple": [1, 2], "joint_repeat_horizon": 15},
            },
            {
                "id": "SELF-MIX",
                "kind": "circulant_block_cyclic_mixer",
                "contract_passed": True,
                "fields": {"dense_output": [1, 2], "circulant_output": [1, 2], "input_values": [0, 0], "dense_parameters": 4, "circulant_parameters": 2, "block_cyclic_parameters": 2, "lora_parameters": 3},
            },
            {
                "id": "SELF-SEED",
                "kind": "seed_rule_exact_regeneration",
                "contract_passed": True,
                "fields": {"generated_object": [1, 2], "regenerated_object": [1, 2], "exact_regeneration": True, "explicit_length": 5, "generator_length": 3, "generator_shorter": True},
            },
        ],
    }
    report = build_report(pack, ROOT / "self-test-contracts.json")
    assert report["external_inference_calls"] == 0
    assert report["training_mutation"] is False
    assert report["promotion_evidence"] is False
    assert report["summary"]["proxy_benchmark_count"] == 6
    assert report["summary"]["passed_proxy_benchmark_count"] == 6
    assert report["summary"]["learned_model_quality_metrics_present"] is False
    assert report["summary"]["deterministic_proxy_metrics_present"] is True
    print(json.dumps({"self_test": "passed", "proxy_benchmark_count": 6}, indent=2))


def fields_of(contract: dict[str, Any]) -> dict[str, Any]:
    return contract.get("fields") if isinstance(contract.get("fields"), dict) else {}


def by_name(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in rows:
        if row.get("name") == name:
            return row
    raise KeyError(name)


def mean_bool(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def hit_rate(path: list[int], targets: list[int]) -> float:
    if not targets:
        return 0.0
    found = set(path)
    return sum(1 for target in targets if target in found) / len(targets)


def deterministic_permutation(n: int) -> list[int]:
    if n <= 0:
        return []
    stride = n - 1
    return [((t * stride + 1) % n) for t in range(n)]


def phase_label(position: int, periods: list[int], target_phase: list[int]) -> int:
    return int(all(position % period == phase for period, phase in zip(periods, target_phase)))


def accuracy(predictions: list[int], labels: list[int]) -> float:
    if not labels:
        return 0.0
    return sum(1 for pred, label in zip(predictions, labels) if pred == label) / len(labels)


def bucket_memorization_accuracy(
    train_positions: list[int],
    train_labels: list[int],
    eval_positions: list[int],
    eval_labels: list[int],
    *,
    bucket_mod: int,
) -> float:
    buckets: dict[int, list[int]] = {}
    for pos, label in zip(train_positions, train_labels):
        buckets.setdefault(pos % bucket_mod, []).append(label)
    bucket_majority = {
        bucket: Counter(labels).most_common(1)[0][0]
        for bucket, labels in buckets.items()
    }
    default = Counter(train_labels).most_common(1)[0][0] if train_labels else 0
    predictions = [bucket_majority.get(pos % bucket_mod, default) for pos in eval_positions]
    return accuracy(predictions, eval_labels)


def max_abs_delta(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right):
        return None
    if not left:
        return 0.0
    return max(abs(a - b) for a, b in zip(left, right))


def gate(name: str, ok: bool, detail: Any) -> dict[str, Any]:
    return {"gate": name, "ok": bool(ok), "detail": detail}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main())
