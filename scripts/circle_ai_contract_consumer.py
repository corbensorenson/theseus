#!/usr/bin/env python3
"""Consume Circle Calculus AI contracts as private Theseus-Hive diagnostics.

This script is deliberately report-only. It converts Circle's public-safe
contract fixtures into Theseus-Hive comparison rows against ordinary baselines,
but it does not run model inference, mutate training data, touch promotion
gates, or claim model-quality improvement.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACTS = ROOT.parent / "circle math" / "site" / "data" / "generated" / "theseus_hive_ai_contracts.json"
DEFAULT_OUT = ROOT / "reports" / "circle_ai_contract_consumer.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "circle_ai_contract_consumer.md"
REQUIRED_AXES = (
    "quality",
    "runtime",
    "memory",
    "parameter_count",
    "interpretability",
    "transfer",
    "failure_cases",
)


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
    contract_reports = [evaluate_contract(row) for row in contracts]
    failed_contracts = [row.get("contract_id") for row in contract_reports if not row.get("contract_passed")]
    unknown_contracts = [row.get("contract_id") for row in contract_reports if row.get("kind") == "unknown"]
    axis_gaps = [
        row.get("contract_id")
        for row in contract_reports
        if any(axis not in row.get("axes", {}) for axis in REQUIRED_AXES)
    ]
    hard_failures = failed_contracts + unknown_contracts + axis_gaps
    benchmark_ready = all(row.get("ready_for_private_benchmark_design") for row in contract_reports)
    trigger_state = "RED" if hard_failures else ("YELLOW" if benchmark_ready else "YELLOW")
    return {
        "policy": "theseus_hive_circle_ai_contract_consumer_v0",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "source_contract_path": rel(source),
        "circle_contract_schema_id": pack.get("schema_id", ""),
        "circle_contract_status": pack.get("status", ""),
        "claim_boundary": (
            "Circle contracts are finite structure fixtures for private experiments. "
            "This report is not model-quality, reasoning, speed, context-length, "
            "transfer, promotion, or ASI evidence."
        ),
        "score_semantics": "private planning diagnostic only; benchmark evidence is absent until named workloads attach",
        "public_calibration_used": False,
        "external_inference_calls": 0,
        "training_mutation": False,
        "promotion_evidence": False,
        "private_data_exported": False,
        "summary": {
            "contract_count": len(contracts),
            "contract_reports": len(contract_reports),
            "failed_contracts": failed_contracts,
            "unknown_contracts": unknown_contracts,
            "axis_gap_contracts": axis_gaps,
            "ordinary_baseline_rows": sum(len(row.get("baseline_comparisons", [])) for row in contract_reports),
            "ready_for_private_benchmark_design": benchmark_ready and not hard_failures,
            "benchmark_results_present": False,
        },
        "governance_gates": [
            gate("external_inference_calls_zero", True, "No model/API/local inference is called by this consumer."),
            gate("public_calibration_absent", True, "The Circle pack is used as configuration, not public calibration data."),
            gate("training_mutation_absent", True, "The consumer writes reports only."),
            gate("promotion_evidence_absent", True, "Report is explicitly non-promotion evidence."),
            gate("private_data_export_absent", True, "No private Theseus-Hive data is serialized into the Circle repo."),
            gate("all_contract_fixtures_passed", not failed_contracts, failed_contracts or "all loaded fixtures passed"),
            gate("known_contract_kinds", not unknown_contracts, unknown_contracts or "all loaded kinds have evaluators"),
            gate("axis_coverage_complete", not axis_gaps, axis_gaps or list(REQUIRED_AXES)),
        ],
        "contract_reports": contract_reports,
        "benchmark_requirements": benchmark_requirements(contract_reports),
        "next_actions": next_actions(contract_reports, hard_failures),
    }


def evaluate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    kind = str(contract.get("kind") or "")
    if kind == "recurrence_schedule":
        return evaluate_recurrence(contract)
    if kind == "strided_candidate_fanout":
        return evaluate_fanout(contract)
    if kind == "cyclic_memory_residue_winding":
        return evaluate_memory(contract)
    if kind == "multicoil_phase_feature":
        return evaluate_phase_feature(contract)
    if kind == "circulant_block_cyclic_mixer":
        return evaluate_mixer(contract)
    if kind == "seed_rule_exact_regeneration":
        return evaluate_seed_rule(contract)
    return base_row(contract, "unknown") | {
        "axes": axis_set(
            quality=axis("blocked", "Unknown contract kind; no quality interpretation."),
            runtime=axis("blocked", "Unknown contract kind; no runtime comparison."),
            memory=axis("blocked", "Unknown contract kind; no memory comparison."),
            parameter_count=axis("blocked", "Unknown contract kind; no parameter comparison."),
            interpretability=axis("blocked", "Unknown contract kind; no explanation mapping."),
            transfer=axis("blocked", "Unknown contract kind; no transfer lane."),
            failure_cases=axis("blocked", ["Add an evaluator for this Circle contract kind."]),
        ),
        "baseline_comparisons": [],
        "ready_for_private_benchmark_design": False,
    }


def evaluate_recurrence(contract: dict[str, Any]) -> dict[str, Any]:
    fields = contract.get("fields") if isinstance(contract.get("fields"), dict) else {}
    tokens = [row for row in fields.get("tokens", []) if isinstance(row, dict)]
    max_loops = as_int(fields.get("max_loops"))
    required_steps = as_int(fields.get("required_steps"))
    exit_step = as_int(fields.get("exit_step"))
    contract_active_steps = sum(len(row.get("active_steps") or []) for row in tokens)
    fixed_depth_steps = len(tokens) * max_loops
    active_step_saving = fixed_depth_steps - contract_active_steps
    budgets = [as_int(row.get("budget")) for row in tokens]
    baseline_comparisons = [
        baseline("fixed_depth", "runtime", {
            "steps": fixed_depth_steps,
            "circle_contract_steps": contract_active_steps,
            "deterministic_step_saving": active_step_saving,
            "claim": "accounting_only_not_speedup",
        }),
        baseline("dense_depth", "runtime", {
            "dense_depth_steps": fixed_depth_steps,
            "selected_token_budgets": budgets,
            "claim": "budget_shape_only",
        }),
        baseline("existing_work_budget", "transfer", {
            "mapping": "required_steps and middle_block_budget can become existing work-budget inputs",
            "claim": "consumer_mapping_not_performance",
        }),
        baseline("no_recurrence", "quality", {
            "observable_difference": "no loop_phase, active_steps, or exit_step certificate",
            "claim": "structure_visibility_not_quality",
        }),
    ]
    return base_row(contract, "recurrence_schedule") | {
        "axes": axis_set(
            quality=axis("not_measured", "Requires a named private recursive/looped workload and baseline score."),
            runtime=axis("accounting_fixture", {
                "fixed_depth_steps": fixed_depth_steps,
                "circle_contract_active_steps": contract_active_steps,
                "deterministic_step_saving": active_step_saving,
                "speedup_claimed": False,
            }),
            memory=axis("trace_available", {
                "token_count": len(tokens),
                "tokens_with_active_steps": sum(1 for row in tokens if row.get("active_steps")),
            }),
            parameter_count=axis("not_applicable", "Schedule contract has no model parameter delta."),
            interpretability=axis("high", {
                "loop_period": fields.get("loop_period"),
                "required_steps": required_steps,
                "exit_step": exit_step,
                "overthinking_boundary": fields.get("overthinking_boundary"),
            }),
            transfer=axis("planned_private_eval", "Attach to loop-closure/work-budget admission before making transfer claims."),
            failure_cases=axis("known_limits", [
                "Budget shape may not correlate with task difficulty.",
                "Exit boundary can be wrong for semantic work even if finite arithmetic is correct.",
            ]),
        ),
        "baseline_comparisons": baseline_comparisons,
        "ready_for_private_benchmark_design": contract.get("contract_passed") is True and exit_step == required_steps,
    }


def evaluate_fanout(contract: dict[str, Any]) -> dict[str, Any]:
    fields = contract.get("fields") if isinstance(contract.get("fields"), dict) else {}
    context_length = as_int(fields.get("context_length"))
    candidate_budget = as_int(fields.get("candidate_budget"))
    predicted_reach = as_int(fields.get("predicted_reach"))
    duplicate_count = as_int(fields.get("duplicate_count"))
    coverage_ratio = safe_ratio(predicted_reach, context_length)
    baseline_comparisons = [
        baseline("sequential_fanout", "runtime", {
            "candidate_budget": candidate_budget,
            "coverage_ratio": coverage_ratio,
            "claim": "coverage_order_only",
        }),
        baseline("random_fanout", "interpretability", {
            "circle_path_deterministic": True,
            "random_seed_required": True,
            "claim": "reproducibility_not_quality",
        }),
        baseline("round_robin_fanout", "transfer", {
            "full_coverage_when_gcd_one": bool(fields.get("full_coverage")),
            "gcd": fields.get("gcd"),
            "claim": "coverage_guarantee_not_semantic_coverage",
        }),
        baseline("local_window", "failure_cases", {
            "locality_preserved": False,
            "risk": "strided fanout can skip useful local neighborhoods",
        }),
    ]
    return base_row(contract, "strided_candidate_fanout") | {
        "axes": axis_set(
            quality=axis("not_measured", "Requires private candidate-scoring workload and residual metric."),
            runtime=axis("accounting_fixture", {
                "candidate_budget": candidate_budget,
                "duplicate_count": duplicate_count,
                "unique_positions": predicted_reach,
            }),
            memory=axis("not_applicable", "Fanout path does not change memory by itself."),
            parameter_count=axis("not_applicable", "Fanout policy has no model parameter delta."),
            interpretability=axis("high", {
                "gcd": fields.get("gcd"),
                "predicted_reach": predicted_reach,
                "full_coverage": bool(fields.get("full_coverage")),
            }),
            transfer=axis("planned_private_eval", "Attach to STS/candidate fanout and compare against ordinary admission."),
            failure_cases=axis("known_limits", [
                "Full position coverage is not full semantic coverage.",
                "Non-coprime strides intentionally create partial coverage.",
            ]),
        ),
        "baseline_comparisons": baseline_comparisons,
        "ready_for_private_benchmark_design": contract.get("contract_passed") is True and duplicate_count == 0,
    }


def evaluate_memory(contract: dict[str, Any]) -> dict[str, Any]:
    fields = contract.get("fields") if isinstance(contract.get("fields"), dict) else {}
    same_residue_events = list(fields.get("same_residue_events") or [])
    same_residue_windings = list(fields.get("same_residue_windings") or [])
    alias_count = max(0, len(same_residue_events) - 1)
    baseline_comparisons = [
        baseline("fifo", "transfer", {
            "circle_adds": "residue/winding provenance tags",
            "claim": "diagnostic_visibility_not_retention_win",
        }),
        baseline("lru", "transfer", {
            "circle_adds": "alias source separation for same residue",
            "claim": "diagnostic_visibility_not_retention_win",
        }),
        baseline("score_based_retention", "quality", {
            "score_signal_used": False,
            "claim": "requires named retention workload",
        }),
        baseline("slot_only_memory", "memory", {
            "same_residue_events": same_residue_events,
            "alias_count_exposed": alias_count,
            "winding_values": same_residue_windings,
        }),
    ]
    return base_row(contract, "cyclic_memory_residue_winding") | {
        "axes": axis_set(
            quality=axis("not_measured", "Requires private retention/retrieval workload."),
            runtime=axis("accounting_fixture", {
                "event_count": fields.get("event_count"),
                "bank_size": fields.get("bank_size"),
            }),
            memory=axis("alias_visibility", {
                "residue_slot": fields.get("residue_slot"),
                "winding": fields.get("winding"),
                "max_alias_load": fields.get("max_alias_load"),
                "alias_count_exposed": alias_count,
            }),
            parameter_count=axis("not_applicable", "Memory tags do not imply model parameter changes."),
            interpretability=axis("high", "Residue and winding make slot aliasing explicit."),
            transfer=axis("planned_private_eval", "Attach to context packet memory and routing-state traces."),
            failure_cases=axis("known_limits", [
                "Extra provenance can increase storage and routing complexity.",
                "Alias visibility does not choose which item should be retained.",
            ]),
        ),
        "baseline_comparisons": baseline_comparisons,
        "ready_for_private_benchmark_design": contract.get("contract_passed") is True and alias_count > 0,
    }


def evaluate_phase_feature(contract: dict[str, Any]) -> dict[str, Any]:
    fields = contract.get("fields") if isinstance(contract.get("fields"), dict) else {}
    phase_tuple = fields.get("phase_tuple")
    shifted_phase_tuple = fields.get("shifted_phase_tuple")
    relative_phase = fields.get("relative_phase")
    shifted_relative = fields.get("shifted_relative_phase")
    invariant = phase_tuple == shifted_phase_tuple and relative_phase == shifted_relative
    baseline_comparisons = [
        baseline("existing_position_bucket", "quality", {
            "circle_adds": "explicit period and relative-phase tags",
            "claim": "feature_hypothesis_not_quality",
        }),
        baseline("learned_position", "transfer", {
            "learned_baseline_required": True,
            "circle_feature_deterministic": True,
        }),
        baseline("wrong_period", "failure_cases", {
            "negative_control_required": True,
            "expected_failure_mode": "spurious periodic bias",
        }),
        baseline("no_phase_feature", "interpretability", {
            "phase_tuple_visible": phase_tuple,
            "relative_phase_visible": relative_phase,
        }),
    ]
    return base_row(contract, "multicoil_phase_feature") | {
        "axes": axis_set(
            quality=axis("not_measured", "Requires private state-sequence feature ablation."),
            runtime=axis("accounting_fixture", {
                "period_count": len(fields.get("periods") or []),
                "joint_repeat_horizon": fields.get("joint_repeat_horizon"),
            }),
            memory=axis("small_feature_tag", "Stores finite residues/relative residues only."),
            parameter_count=axis("feature_only", "No model parameter delta unless embedded/learned downstream."),
            interpretability=axis("high", {
                "phase_tuple": phase_tuple,
                "relative_phase": relative_phase,
                "shift_invariant_in_fixture": invariant,
            }),
            transfer=axis("planned_private_eval", "Attach to Code LM state-sequence features with wrong-period controls."),
            failure_cases=axis("known_limits", [
                "Wrong periods can add harmful inductive bias.",
                "Nonperiodic workloads should be negative controls.",
            ]),
        ),
        "baseline_comparisons": baseline_comparisons,
        "ready_for_private_benchmark_design": contract.get("contract_passed") is True and invariant,
    }


def evaluate_mixer(contract: dict[str, Any]) -> dict[str, Any]:
    fields = contract.get("fields") if isinstance(contract.get("fields"), dict) else {}
    dense_parameters = as_int(fields.get("dense_parameters"))
    circulant_parameters = as_int(fields.get("circulant_parameters"))
    dense_adapter_parameters = as_int(fields.get("dense_adapter_parameters"))
    block_cyclic_parameters = as_int(fields.get("block_cyclic_parameters"))
    baseline_comparisons = [
        baseline("dense_mixer", "parameter_count", {
            "dense_parameters": dense_parameters,
            "circulant_parameters": circulant_parameters,
            "circulant_parameter_ratio": fields.get("circulant_parameter_ratio"),
        }),
        baseline("low_rank_mixer", "parameter_count", {
            "requires_baseline_run": True,
            "claim": "not compared by fixture",
        }),
        baseline("lora_adapter", "parameter_count", {
            "lora_parameters": fields.get("lora_parameters"),
            "block_cyclic_parameters": block_cyclic_parameters,
            "block_to_dense_ratio": fields.get("block_to_dense_ratio"),
        }),
        baseline("no_mixer", "quality", {
            "requires_ablation": True,
            "claim": "parameter reduction alone is not a win",
        }),
    ]
    return base_row(contract, "circulant_block_cyclic_mixer") | {
        "axes": axis_set(
            quality=axis("not_measured", "Requires route/ranker/mixer workload quality metric."),
            runtime=axis("not_measured", "Fixture checks arithmetic parity, not runtime."),
            memory=axis("parameter_accounting", {
                "dense_parameters": dense_parameters,
                "circulant_parameters": circulant_parameters,
                "dense_adapter_parameters": dense_adapter_parameters,
                "block_cyclic_parameters": block_cyclic_parameters,
            }),
            parameter_count=axis("accounting_fixture", {
                "circulant_vs_dense_ratio": safe_ratio(circulant_parameters, dense_parameters),
                "block_vs_dense_adapter_ratio": safe_ratio(block_cyclic_parameters, dense_adapter_parameters),
            }),
            interpretability=axis("medium", "Cyclic weight sharing is explicit; downstream feature semantics still need inspection."),
            transfer=axis("planned_private_eval", "Attach to ranker/route-head experiments with dense and low-rank controls."),
            failure_cases=axis("known_limits", [
                "Shift-structured mixers can underfit nonperiodic or position-specific signals.",
                "Parameter reduction is not useful unless quality and runtime stay competitive.",
            ]),
        ),
        "baseline_comparisons": baseline_comparisons,
        "ready_for_private_benchmark_design": contract.get("contract_passed") is True and as_float(fields.get("max_abs_dense_delta")) == 0.0,
    }


def evaluate_seed_rule(contract: dict[str, Any]) -> dict[str, Any]:
    fields = contract.get("fields") if isinstance(contract.get("fields"), dict) else {}
    exact = bool(fields.get("exact_regeneration"))
    generator_shorter = bool(fields.get("generator_shorter"))
    baseline_comparisons = [
        baseline("object_only_storage", "memory", {
            "exact_regeneration": exact,
            "generator_shorter": generator_shorter,
            "explicit_length": fields.get("explicit_length"),
            "generator_length": fields.get("generator_length"),
        }),
        baseline("unverified_template", "interpretability", {
            "verifier_required": True,
            "claim": "seed_rule_must_regenerate_exactly",
        }),
        baseline("freeform_tool_memory", "failure_cases", {
            "risk": "unverified memory can drift from regenerated artifact",
        }),
    ]
    return base_row(contract, "seed_rule_exact_regeneration") | {
        "axes": axis_set(
            quality=axis("not_measured", "Requires workflow regeneration workload and verifier metric."),
            runtime=axis("not_measured", "Generation cost must be measured on named workflows."),
            memory=axis("accounting_fixture", {
                "explicit_length": fields.get("explicit_length"),
                "generator_length": fields.get("generator_length"),
                "generator_shorter": generator_shorter,
            }),
            parameter_count=axis("not_applicable", "Seed-rule object is not a model parameterization."),
            interpretability=axis("high", {
                "artifact_id": fields.get("artifact_id"),
                "iteration_schedule": fields.get("iteration_schedule"),
                "closure_condition": fields.get("closure_condition"),
            }),
            transfer=axis("planned_private_eval", "Attach to CGS/tool-card exact-regeneration checks."),
            failure_cases=axis("known_limits", [
                "For small artifacts, the generator can be longer than the explicit object.",
                "Exact regeneration does not prove semantic usefulness.",
            ]),
        ),
        "baseline_comparisons": baseline_comparisons,
        "ready_for_private_benchmark_design": contract.get("contract_passed") is True and exact,
    }


def base_row(contract: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "contract_id": contract.get("id", ""),
        "kind": kind,
        "contract_passed": bool(contract.get("contract_passed")),
        "status": contract.get("status", ""),
        "source_paper": contract.get("source_paper", ""),
        "theorem_ids": list(contract.get("theorem_ids") or []),
        "dictionary_ids": list(contract.get("dictionary_ids") or []),
        "ordinary_baselines": list(contract.get("ordinary_baselines") or []),
        "theseus_hive_use": contract.get("theseus_hive_use", ""),
        "not_claimed": contract.get("not_claimed", ""),
    }


def axis_set(**axes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return axes


def axis(status: str, evidence: Any) -> dict[str, Any]:
    return {"status": status, "evidence": evidence}


def baseline(name: str, axis_name: str, comparison: dict[str, Any]) -> dict[str, Any]:
    return {"baseline": name, "axis": axis_name, "comparison": comparison}


def gate(name: str, ok: bool, detail: Any) -> dict[str, Any]:
    return {"gate": name, "ok": bool(ok), "detail": detail}


def benchmark_requirements(contract_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in contract_reports:
        rows.append(
            {
                "contract_id": row.get("contract_id"),
                "kind": row.get("kind"),
                "required_before_claim": [
                    "named_private_workload",
                    "ordinary_baseline",
                    "negative_control_where_relevant",
                    "quality_metric",
                    "runtime_metric",
                    "memory_or_parameter_metric",
                    "reproducible_script",
                    "report_path",
                ],
            }
        )
    return rows


def next_actions(contract_reports: list[dict[str, Any]], hard_failures: list[str]) -> list[str]:
    if hard_failures:
        return [
            "Fix failed/unknown Circle contracts before attaching private benchmarks.",
            "Rerun this consumer after the contract pack is regenerated.",
        ]
    kinds = {str(row.get("kind")) for row in contract_reports}
    actions = []
    if "recurrence_schedule" in kinds:
        actions.append("Attach recurrence_schedule to a private looped-model/work-budget smoke workload.")
    if "strided_candidate_fanout" in kinds:
        actions.append("Run strided_candidate_fanout beside sequential/random/round-robin/local-window fanout on a private candidate task.")
    if "cyclic_memory_residue_winding" in kinds:
        actions.append("Add residue-plus-winding tags to an offline context-packet memory trace and compare alias visibility.")
    if "multicoil_phase_feature" in kinds:
        actions.append("Create a private Code LM phase-feature ablation with existing-position, learned-position, wrong-period, and no-phase controls.")
    if "circulant_block_cyclic_mixer" in kinds:
        actions.append("Add a route/ranker mixer microbench that reports quality, runtime, memory, and parameter count separately.")
    if "seed_rule_exact_regeneration" in kinds:
        actions.append("Use seed_rule_exact_regeneration on a repeated workflow/tool-card artifact and measure residual regeneration cost.")
    return actions or ["Add at least one known Circle AI contract kind."]


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Circle AI Contract Consumer",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Contracts loaded: {summary.get('contract_count')}",
        f"- Ordinary baseline rows: {summary.get('ordinary_baseline_rows')}",
        f"- Benchmark results present: {summary.get('benchmark_results_present')}",
        f"- External inference calls: {report.get('external_inference_calls')}",
        f"- Training mutation: {report.get('training_mutation')}",
        f"- Promotion evidence: {report.get('promotion_evidence')}",
        "",
        "This report is private planning evidence only. It does not claim model-quality, reasoning, speed, context-length, transfer, promotion, or ASI progress.",
        "",
        "## Contracts",
    ]
    for row in report.get("contract_reports", []):
        axes = row.get("axes", {})
        lines.extend(
            [
                "",
                f"### {row.get('contract_id')}",
                "",
                f"- Kind: `{row.get('kind')}`",
                f"- Passed fixture: `{row.get('contract_passed')}`",
                f"- Ready for private benchmark design: `{row.get('ready_for_private_benchmark_design')}`",
                f"- Baselines: {', '.join(row.get('ordinary_baselines') or [])}",
                f"- Quality axis: `{get_path(axes, ['quality', 'status'], '')}`",
                f"- Runtime axis: `{get_path(axes, ['runtime', 'status'], '')}`",
                f"- Memory axis: `{get_path(axes, ['memory', 'status'], '')}`",
                f"- Parameter axis: `{get_path(axes, ['parameter_count', 'status'], '')}`",
                f"- Transfer axis: `{get_path(axes, ['transfer', 'status'], '')}`",
            ]
        )
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def run_self_test() -> None:
    pack = {
        "schema_id": "circle_calculus.theseus_hive_ai_contracts.self_test",
        "status": "public_safe_fixture",
        "contracts": [
            {
                "id": "SELF-REC",
                "kind": "recurrence_schedule",
                "status": "fixture",
                "contract_passed": True,
                "ordinary_baselines": ["fixed_depth", "dense_depth", "existing_work_budget", "no_recurrence"],
                "fields": {
                    "max_loops": 4,
                    "required_steps": 2,
                    "exit_step": 2,
                    "loop_period": 4,
                    "tokens": [
                        {"budget": 1, "active_steps": [1]},
                        {"budget": 2, "active_steps": [1, 2]},
                    ],
                },
            },
            {
                "id": "SELF-FAN",
                "kind": "strided_candidate_fanout",
                "status": "fixture",
                "contract_passed": True,
                "ordinary_baselines": ["sequential_fanout", "random_fanout", "round_robin_fanout", "local_window"],
                "fields": {"context_length": 5, "candidate_budget": 5, "predicted_reach": 5, "duplicate_count": 0, "gcd": 1, "full_coverage": True},
            },
            {
                "id": "SELF-MEM",
                "kind": "cyclic_memory_residue_winding",
                "status": "fixture",
                "contract_passed": True,
                "ordinary_baselines": ["fifo", "lru", "score_based_retention", "slot_only_memory"],
                "fields": {"same_residue_events": [1, 5], "same_residue_windings": [0, 1], "event_count": 8, "bank_size": 4, "residue_slot": 1, "winding": 1, "max_alias_load": 2},
            },
            {
                "id": "SELF-PHASE",
                "kind": "multicoil_phase_feature",
                "status": "fixture",
                "contract_passed": True,
                "ordinary_baselines": ["existing_position_bucket", "learned_position", "wrong_period", "no_phase_feature"],
                "fields": {"periods": [2, 3], "phase_tuple": [1, 2], "shifted_phase_tuple": [1, 2], "relative_phase": 1, "shifted_relative_phase": 1, "joint_repeat_horizon": 6},
            },
            {
                "id": "SELF-MIX",
                "kind": "circulant_block_cyclic_mixer",
                "status": "fixture",
                "contract_passed": True,
                "ordinary_baselines": ["dense_mixer", "low_rank_mixer", "lora_adapter", "no_mixer"],
                "fields": {"dense_parameters": 16, "circulant_parameters": 4, "dense_adapter_parameters": 64, "block_cyclic_parameters": 8, "max_abs_dense_delta": 0.0},
            },
            {
                "id": "SELF-SEED",
                "kind": "seed_rule_exact_regeneration",
                "status": "fixture",
                "contract_passed": True,
                "ordinary_baselines": ["object_only_storage", "unverified_template", "freeform_tool_memory"],
                "fields": {"exact_regeneration": True, "generator_shorter": False, "explicit_length": 3, "generator_length": 7},
            },
        ],
    }
    report = build_report(pack, ROOT / "self-test-contracts.json")
    assert report["external_inference_calls"] == 0
    assert report["public_calibration_used"] is False
    assert report["training_mutation"] is False
    assert report["promotion_evidence"] is False
    assert report["summary"]["contract_reports"] == 6
    assert report["summary"]["axis_gap_contracts"] == []
    assert report["summary"]["unknown_contracts"] == []
    assert report["summary"]["ordinary_baseline_rows"] >= 22
    for row in report["contract_reports"]:
        assert all(axis_name in row["axes"] for axis_name in REQUIRED_AXES)
    print(json.dumps({"self_test": "passed", "contract_reports": 6, "ordinary_baseline_rows": report["summary"]["ordinary_baseline_rows"]}, indent=2))


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


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


if __name__ == "__main__":
    raise SystemExit(main())
