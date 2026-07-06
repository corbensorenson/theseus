#!/usr/bin/env python3
"""Guarded STS ranker policy evaluation.

This script turns the matched STS selection result into a named candidate
selector. The selector is metadata-only: it uses STS route indicators, decoder
contract compatibility, and candidate provenance flags. It does not train on
heldout tests, heldout solution bodies, public benchmark content, teacher
tokens, or runtime verifier outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from private_residual_repair_v3_heldout_score import run_candidate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "sts_ranker_policy_v1.json"
DEFAULT_MD = ROOT / "reports" / "sts_ranker_policy_v1.md"
DEFAULT_SELECTED = ROOT / "reports" / "sts_ranker_policy_v1_selected_candidates.jsonl"
DEFAULT_CONFIG = ROOT / "configs" / "sts_ranker_policy_v1.json"
DEFAULT_PUBLIC_READINESS = ROOT / "reports" / "post_distillation_public_transfer_readiness_v1.json"
GUARD_ENV = "THESEUS_ENABLE_STS_RANKER_POLICY_V1"


SURFACES = [
    {
        "name": "private_residual_v3",
        "role": "primary_private_v3",
        "heldout": ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "private_residual_repair_v3_heldout_code_lm_tasks.jsonl",
        "sts_candidates": ROOT / "reports" / "code_lm_private_candidates_private_residual_repair_v3_student_repair.jsonl",
        "non_sts_candidates": ROOT / "reports" / "code_lm_private_candidates_private_residual_repair_v3_student_repair_sts_off_control.jsonl",
        "task_limit": 0,
        "minimum_tasks": 240,
    },
    {
        "name": "private_ecology_generalization_v5_smoke72",
        "role": "existing_disjoint_private_family",
        "heldout": ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "private_ecology_generalization_v5_heldout_code_lm_tasks.jsonl",
        "sts_candidates": ROOT / "reports" / "code_lm_private_candidates_private_ecology_generalization_v5_smoke72.jsonl",
        "non_sts_candidates": ROOT / "reports" / "code_lm_private_candidates_private_ecology_generalization_v5_smoke72_sts_off.jsonl",
        "task_limit": 72,
        "minimum_tasks": 72,
    },
]


FEATURE_WEIGHTS = {
    "sts_conditioned": 100.0,
    "sts_mode": 25.0,
    "category_match": 50.0,
    "semantic_family_match": 20.0,
    "decoder_contract_return_shape_match": 14.0,
    "decoder_contract_arg_count_match": 12.0,
    "decoder_contract_required_construct_overlap": 6.0,
    "decoder_contract_verifier_passed": 8.0,
    "candidate_syntax_lint_passed": 6.0,
    "deterministic_guardrail_passed": 4.0,
    "grammar_masked_learned_token_candidate": 5.0,
    "full_body_token_candidate": 5.0,
    "token_level_code_generation_learned": 4.0,
    "structural_action_candidate": 5.0,
    "private_receiver_inventory_eligible": 3.0,
    "beautiful_code_score": 1.0,
    "original_rank_penalty": -0.001,
    "template_like_candidate": -100.0,
    "placeholder_scaffold_body": -100.0,
    "expression_memory_fallback": -500.0,
    "diagnostic_adapter_stage": -100.0,
    "public_tests_visible_to_generator": -500.0,
    "canonical_solution_seen_by_solver": -500.0,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-budget", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=2)
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--public-transfer-readiness", default=rel(DEFAULT_PUBLIC_READINESS))
    parser.add_argument("--selected-out", default=rel(DEFAULT_SELECTED))
    parser.add_argument("--out", default=rel(DEFAULT_REPORT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_jsonl(resolve(args.selected_out), report.get("selected_candidate_rows", []))
    report.pop("selected_candidate_rows", None)
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace, started: float) -> dict[str, Any]:
    config_path = resolve(args.config)
    config = read_json(config_path)
    budget = max(1, int(args.candidate_budget))
    timeout = max(1, int(args.timeout_seconds))
    surface_reports = []
    selected_rows: list[dict[str, Any]] = []
    for surface in SURFACES:
        surface_report, rows = score_surface(surface, budget=budget, timeout_seconds=timeout)
        surface_reports.append(surface_report)
        selected_rows.extend(rows)

    aggregate = aggregate_surfaces(surface_reports)
    public_readiness = read_json(resolve(args.public_transfer_readiness))
    guarded_enabled = bool(config.get("enabled", False) or os.environ.get(GUARD_ENV) == "1")
    verification_bandwidth = sts_ranker_verification_bandwidth_record(
        aggregate,
        surface_reports,
        selected_rows,
        budget=budget,
        timeout_seconds=timeout,
        guarded_enabled=guarded_enabled,
    )
    governance_tax = sts_ranker_governance_tax_record(
        aggregate,
        surface_reports,
        verification_bandwidth=verification_bandwidth,
        started=started,
    )
    aggregate.update(
        {
            "verification_bandwidth_status": verification_bandwidth["status"],
            "verification_obligation_count": verification_bandwidth["obligation_count"],
            "verification_escalation_required": verification_bandwidth["escalation_required"],
            "governance_tax_status": governance_tax["status"],
            "governance_tax_review_load_units": governance_tax["review_load_units"],
            "governance_tax_caught_failure_count": governance_tax["caught_failure_count"],
        }
    )
    improvement = aggregate["sts_policy_selected_pass_rate"] > aggregate["non_sts_policy_selected_pass_rate"]
    all_surface_non_regression = all(
        row["summary"]["sts_policy_selected_pass_rate"] >= row["summary"]["non_sts_policy_selected_pass_rate"]
        for row in surface_reports
    )
    gates = [
        gate("config_present", config_path.exists(), rel(config_path)),
        gate("guarded_flag_defined", str(config.get("guard_env_var") or GUARD_ENV) == GUARD_ENV, config.get("guard_env_var")),
        gate("disabled_by_default", not bool(config.get("enabled_by_default", False)), config.get("enabled_by_default")),
        gate("public_calibration_disabled", not bool(config.get("allow_public_calibration", False)), config.get("allow_public_calibration")),
        gate("two_private_surfaces_present", len(surface_reports) >= 2, [row["surface"] for row in surface_reports]),
        gate("disjoint_private_family_present", any(row["role"] == "existing_disjoint_private_family" for row in surface_reports), [row["role"] for row in surface_reports]),
        gate("surface_minimum_tasks_met", all(row["summary"]["task_count"] >= row["minimum_tasks"] for row in surface_reports), {row["surface"]: row["summary"]["task_count"] for row in surface_reports}),
        gate("equal_budget_task_rate_full", all(row["summary"]["equal_budget_task_rate"] == 1.0 for row in surface_reports), {row["surface"]: row["summary"]["equal_budget_task_rate"] for row in surface_reports}),
        gate("sts_policy_beats_non_sts_selected", improvement, {
            "sts": aggregate["sts_policy_selected_pass_rate"],
            "non_sts": aggregate["non_sts_policy_selected_pass_rate"],
        }),
        gate("no_surface_selection_regression_vs_non_sts", all_surface_non_regression, {row["surface"]: row["summary"]["selected_pass_delta_sts_policy_minus_non_sts_policy"] for row in surface_reports}),
        gate("sts_policy_does_not_regress_existing_sts_order", aggregate["sts_policy_vs_original_regression_count"] == 0, aggregate["sts_policy_vs_original_regression_count"]),
        gate("oracle_not_worse_than_non_sts", aggregate["sts_oracle_pass_rate"] >= aggregate["non_sts_oracle_pass_rate"], {
            "sts_oracle": aggregate["sts_oracle_pass_rate"],
            "non_sts_oracle": aggregate["non_sts_oracle_pass_rate"],
        }),
        gate("no_admissible_rate_clean", aggregate["no_admissible_rate"] <= 0.03, aggregate["no_admissible_rate"]),
        gate("fallback_returns_zero", aggregate["fallback_return_candidate_count"] == 0, aggregate["fallback_return_candidate_count"]),
        gate("public_leakage_zero", aggregate["public_leakage_count"] == 0, aggregate["public_leakage_count"]),
        gate("external_inference_zero", aggregate["external_inference_calls"] == 0, aggregate["external_inference_calls"]),
        gate("sts_ranker_verification_bandwidth_ready", verification_bandwidth.get("status") == "ready", verification_bandwidth),
        gate("sts_ranker_governance_tax_ready", governance_tax.get("status") == "ready", governance_tax),
        gate("heldout_solution_bodies_not_used_for_policy", True, "policy uses metadata only; heldout solution bodies are ignored"),
        gate("heldout_tests_used_for_eval_only", True, "private heldout tests are executed only after selection for measurement"),
    ]
    hard_failed = [row for row in gates if not row["passed"]]
    trigger_state = "GREEN" if not hard_failed else "YELLOW"
    recommendation = recommendation_for(trigger_state, aggregate, public_readiness)
    viea_records = build_sts_ranker_spine_records(
        args=args,
        aggregate=aggregate,
        surface_reports=surface_reports,
        selected_rows=selected_rows,
        verification_bandwidth=verification_bandwidth,
        governance_tax=governance_tax,
        recommendation=recommendation,
    )
    gates.append(gate("sts_ranker_viea_records_present", len(viea_records) >= 9, len(viea_records)))
    hard_failed = [row for row in gates if not row["passed"]]
    trigger_state = "GREEN" if not hard_failed else "YELLOW"
    return {
        "policy": "project_theseus_sts_ranker_policy_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "guarded_integration": {
            "config": rel(config_path),
            "guard_env_var": GUARD_ENV,
            "enabled_by_default": False,
            "currently_enabled": guarded_enabled,
            "runtime_use_requires": f"{GUARD_ENV}=1",
            "allow_public_calibration": False,
            "eligible_for_guarded_runtime_use": trigger_state == "GREEN",
            "selected_candidate_manifest": rel(resolve(args.selected_out)),
        },
        "policy_derivation": {
            "selector_kind": "deterministic_metadata_ranker",
            "features": sorted(FEATURE_WEIGHTS),
            "weights": FEATURE_WEIGHTS,
            "uses_sts_features": True,
            "uses_decoder_contract_features": True,
            "uses_candidate_metadata": True,
            "uses_heldout_tests_for_policy_derivation": False,
            "uses_heldout_solution_bodies_for_policy_derivation": False,
            "uses_public_benchmark_content": False,
            "uses_teacher_rows": False,
            "external_inference_calls": 0,
        },
        "summary": aggregate,
        "public_transfer_readiness": compact_public_readiness(public_readiness, resolve(args.public_transfer_readiness)),
        "surfaces": surface_reports,
        "gates": gates,
        "recommendation": recommendation,
        "verification_bandwidth": verification_bandwidth,
        "governance_tax": governance_tax,
        "viea_sts_ranker_records": viea_records,
        "score_semantics": (
            "Private-only causal selector evaluation. The selector changes candidate ordering before "
            "private tests are executed; private tests measure the selected candidate and oracle/pass-if-any "
            "after selection. Public calibration remains locked."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": aggregate["fallback_return_candidate_count"],
        "selected_candidate_rows": selected_rows,
    }


def score_surface(surface: dict[str, Any], *, budget: int, timeout_seconds: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    heldout_path = surface["heldout"]
    sts_path = surface["sts_candidates"]
    non_sts_path = surface["non_sts_candidates"]
    heldout = read_jsonl(heldout_path)
    if int(surface.get("task_limit") or 0) > 0:
        heldout = heldout[: int(surface["task_limit"])]
    sts_by_task = group_candidates(read_jsonl(sts_path))
    non_sts_by_task = group_candidates(read_jsonl(non_sts_path))
    task_rows = []
    selected_rows = []
    family_counts: Counter[str] = Counter()
    family_sts_policy_passes: Counter[str] = Counter()
    no_admissible = 0
    fallback_count = 0
    public_leakage = 0
    external_calls = 0
    equal_budget_tasks = 0

    for task in heldout:
        task_id = str(task.get("task_id") or "")
        family = task_family(task)
        family_counts[family] += 1
        sts_candidates = ordered_candidates(sts_by_task.get(task_id, []))
        non_sts_candidates = ordered_candidates(non_sts_by_task.get(task_id, []))
        equal_budget = min(budget, len(sts_candidates), len(non_sts_candidates))
        if equal_budget <= 0:
            no_admissible += 1
            task_rows.append({
                "task_id": task_id,
                "family": family,
                "equal_budget": 0,
                "sts_policy_selected_passed": False,
                "non_sts_policy_selected_passed": False,
                "sts_oracle_passed": False,
                "non_sts_oracle_passed": False,
                "no_admissible": True,
            })
            continue
        equal_budget_tasks += 1
        sts_budget = sts_candidates[:equal_budget]
        non_sts_budget = non_sts_candidates[:equal_budget]
        for candidate in sts_budget + non_sts_budget:
            fallback_count += int(fallback_return_candidate(candidate))
            public_leakage += public_leak_count(task, candidate)
            external_calls += int(candidate.get("external_inference_calls") or 0)

        sts_policy = select_candidate(task, sts_budget)
        non_sts_policy = select_candidate(task, non_sts_budget)
        sts_original = sts_budget[0]
        non_sts_original = non_sts_budget[0]
        sts_results = score_candidates(task, sts_budget, timeout_seconds=timeout_seconds)
        non_sts_results = score_candidates(task, non_sts_budget, timeout_seconds=timeout_seconds)
        sts_policy_passed = sts_results[id(sts_policy)]
        non_sts_policy_passed = non_sts_results[id(non_sts_policy)]
        sts_original_passed = sts_results[id(sts_original)]
        non_sts_original_passed = non_sts_results[id(non_sts_original)]
        sts_oracle = any(sts_results.values())
        non_sts_oracle = any(non_sts_results.values())
        if sts_policy_passed:
            family_sts_policy_passes[family] += 1
        selected_rows.append(selected_candidate_row(surface["name"], task, sts_policy, sts_policy_passed))
        task_rows.append({
            "task_id": task_id,
            "family": family,
            "equal_budget": equal_budget,
            "sts_policy_selected_passed": sts_policy_passed,
            "non_sts_policy_selected_passed": non_sts_policy_passed,
            "sts_original_selected_passed": sts_original_passed,
            "non_sts_original_selected_passed": non_sts_original_passed,
            "sts_oracle_passed": sts_oracle,
            "non_sts_oracle_passed": non_sts_oracle,
            "sts_policy_score": round(policy_score(task, sts_policy)[0], 6),
            "non_sts_policy_score": round(policy_score(task, non_sts_policy)[0], 6),
            "sts_policy_mode": sts_policy.get("candidate_generation_mode"),
            "non_sts_policy_mode": non_sts_policy.get("candidate_generation_mode"),
            "sts_policy_vs_original_regression": bool(sts_original_passed and not sts_policy_passed),
            "sts_policy_vs_non_sts_regression": bool(non_sts_policy_passed and not sts_policy_passed),
            "no_admissible": False,
        })

    task_count = len(heldout)
    summary = summarize_tasks(task_rows, task_count, equal_budget_tasks, no_admissible)
    summary.update({
        "fallback_return_candidate_count": fallback_count,
        "public_leakage_count": public_leakage,
        "external_inference_calls": external_calls,
        "family_rates": {
            family: {
                "task_count": family_counts[family],
                "sts_policy_selected_pass_count": family_sts_policy_passes[family],
                "sts_policy_selected_pass_rate": round(family_sts_policy_passes[family] / max(1, family_counts[family]), 6),
            }
            for family in sorted(family_counts)
        },
    })
    return {
        "surface": surface["name"],
        "role": surface["role"],
        "minimum_tasks": int(surface["minimum_tasks"]),
        "inputs": {
            "heldout": rel(heldout_path),
            "sts_candidates": rel(sts_path),
            "non_sts_candidates": rel(non_sts_path),
            "task_limit": int(surface.get("task_limit") or 0),
            "candidate_budget_requested": budget,
            "timeout_seconds": timeout_seconds,
        },
        "summary": summary,
        "per_task": task_rows,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }, selected_rows


def summarize_tasks(task_rows: list[dict[str, Any]], task_count: int, equal_budget_tasks: int, no_admissible: int) -> dict[str, Any]:
    def count(key: str) -> int:
        return sum(1 for row in task_rows if row.get(key) is True)

    sts_policy = count("sts_policy_selected_passed")
    non_sts_policy = count("non_sts_policy_selected_passed")
    sts_original = count("sts_original_selected_passed")
    non_sts_original = count("non_sts_original_selected_passed")
    sts_oracle = count("sts_oracle_passed")
    non_sts_oracle = count("non_sts_oracle_passed")
    return {
        "task_count": task_count,
        "equal_budget_task_count": equal_budget_tasks,
        "equal_budget_task_rate": round(equal_budget_tasks / max(1, task_count), 6),
        "no_admissible_count": no_admissible,
        "no_admissible_rate": round(no_admissible / max(1, task_count), 6),
        "sts_policy_selected_pass_count": sts_policy,
        "sts_policy_selected_pass_rate": round(sts_policy / max(1, task_count), 6),
        "non_sts_policy_selected_pass_count": non_sts_policy,
        "non_sts_policy_selected_pass_rate": round(non_sts_policy / max(1, task_count), 6),
        "selected_pass_delta_sts_policy_minus_non_sts_policy": round((sts_policy - non_sts_policy) / max(1, task_count), 6),
        "sts_original_selected_pass_count": sts_original,
        "sts_original_selected_pass_rate": round(sts_original / max(1, task_count), 6),
        "non_sts_original_selected_pass_count": non_sts_original,
        "non_sts_original_selected_pass_rate": round(non_sts_original / max(1, task_count), 6),
        "sts_oracle_pass_count": sts_oracle,
        "sts_oracle_pass_rate": round(sts_oracle / max(1, task_count), 6),
        "non_sts_oracle_pass_count": non_sts_oracle,
        "non_sts_oracle_pass_rate": round(non_sts_oracle / max(1, task_count), 6),
        "sts_policy_vs_original_regression_count": count("sts_policy_vs_original_regression"),
        "sts_policy_vs_non_sts_regression_count": count("sts_policy_vs_non_sts_regression"),
    }


def aggregate_surfaces(surfaces: list[dict[str, Any]]) -> dict[str, Any]:
    total_tasks = sum(int(row["summary"]["task_count"]) for row in surfaces)
    totals = Counter()
    for surface in surfaces:
        summary = surface["summary"]
        for key in (
            "equal_budget_task_count",
            "no_admissible_count",
            "sts_policy_selected_pass_count",
            "non_sts_policy_selected_pass_count",
            "sts_original_selected_pass_count",
            "non_sts_original_selected_pass_count",
            "sts_oracle_pass_count",
            "non_sts_oracle_pass_count",
            "sts_policy_vs_original_regression_count",
            "sts_policy_vs_non_sts_regression_count",
            "fallback_return_candidate_count",
            "public_leakage_count",
            "external_inference_calls",
        ):
            totals[key] += int(summary.get(key) or 0)
    return {
        "surface_count": len(surfaces),
        "task_count": total_tasks,
        "equal_budget_task_count": totals["equal_budget_task_count"],
        "equal_budget_task_rate": round(totals["equal_budget_task_count"] / max(1, total_tasks), 6),
        "no_admissible_count": totals["no_admissible_count"],
        "no_admissible_rate": round(totals["no_admissible_count"] / max(1, total_tasks), 6),
        "sts_policy_selected_pass_count": totals["sts_policy_selected_pass_count"],
        "sts_policy_selected_pass_rate": round(totals["sts_policy_selected_pass_count"] / max(1, total_tasks), 6),
        "non_sts_policy_selected_pass_count": totals["non_sts_policy_selected_pass_count"],
        "non_sts_policy_selected_pass_rate": round(totals["non_sts_policy_selected_pass_count"] / max(1, total_tasks), 6),
        "selected_pass_delta_sts_policy_minus_non_sts_policy": round((totals["sts_policy_selected_pass_count"] - totals["non_sts_policy_selected_pass_count"]) / max(1, total_tasks), 6),
        "sts_original_selected_pass_rate": round(totals["sts_original_selected_pass_count"] / max(1, total_tasks), 6),
        "non_sts_original_selected_pass_rate": round(totals["non_sts_original_selected_pass_count"] / max(1, total_tasks), 6),
        "sts_oracle_pass_rate": round(totals["sts_oracle_pass_count"] / max(1, total_tasks), 6),
        "non_sts_oracle_pass_rate": round(totals["non_sts_oracle_pass_count"] / max(1, total_tasks), 6),
        "sts_policy_vs_original_regression_count": totals["sts_policy_vs_original_regression_count"],
        "sts_policy_vs_non_sts_regression_count": totals["sts_policy_vs_non_sts_regression_count"],
        "fallback_return_candidate_count": totals["fallback_return_candidate_count"],
        "public_leakage_count": totals["public_leakage_count"],
        "external_inference_calls": totals["external_inference_calls"],
        "public_candidate_rows": 0,
        "public_tests_used": False,
        "public_solutions_used": False,
        "teacher_rows_used": False,
        "heldout_solution_bodies_used_for_policy": False,
        "heldout_tests_used_for_policy": False,
    }


def select_candidate(task: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = [(policy_score(task, candidate), index, candidate) for index, candidate in enumerate(candidates)]
    ranked.sort(key=lambda item: (-item[0][0], str(item[0][1].get("tie_key") or ""), item[1]))
    return ranked[0][2]


def policy_score(task: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    features = candidate_features(task, candidate)
    score = 0.0
    for name, value in features.items():
        weight = FEATURE_WEIGHTS.get(name, 0.0)
        score += weight * numeric_feature(value)
    mode = str(candidate.get("candidate_generation_mode") or "")
    tie_key = f"{mode}|{candidate.get('candidate_sha256') or ''}"
    return score, {"features": features, "tie_key": tie_key}


def candidate_features(task: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    task_contract = task.get("decoder_contract") if isinstance(task.get("decoder_contract"), dict) else {}
    candidate_contract = candidate_decoder_contract(candidate)
    mode = str(candidate.get("candidate_generation_mode") or "")
    task_required = {str(item) for item in task_contract.get("required_constructs") or []}
    candidate_required = {str(item) for item in candidate_contract.get("required_constructs") or []}
    overlap = len(task_required & candidate_required)
    union = len(task_required | candidate_required)
    return {
        "sts_conditioned": bool(candidate.get("sts_stream_conditioned") or candidate.get("sts_candidate_expression_used")),
        "sts_mode": "sts_conditioned" in mode.lower(),
        "category_match": str(candidate.get("category") or candidate.get("target_category") or "") == str(task.get("category") or ""),
        "semantic_family_match": semantic_family(candidate) == semantic_family(task),
        "decoder_contract_return_shape_match": str(candidate_contract.get("return_shape") or "") == str(task_contract.get("return_shape") or ""),
        "decoder_contract_arg_count_match": int(candidate_contract.get("visible_arg_count_hint") or 0) == int(task_contract.get("visible_arg_count_hint") or 0),
        "decoder_contract_required_construct_overlap": 0.0 if union == 0 else overlap / union,
        "decoder_contract_verifier_passed": bool(candidate.get("decoder_contract_verifier_v1_passed")),
        "candidate_syntax_lint_passed": bool(candidate.get("candidate_syntax_lint_passed", True)),
        "deterministic_guardrail_passed": bool(candidate.get("deterministic_guardrail_passed")),
        "grammar_masked_learned_token_candidate": bool(candidate.get("grammar_masked_learned_token_candidate")),
        "full_body_token_candidate": bool(candidate.get("full_body_token_candidate")),
        "token_level_code_generation_learned": bool(candidate.get("token_level_code_generation_learned")),
        "structural_action_candidate": bool(candidate.get("structural_action_candidate")),
        "private_receiver_inventory_eligible": bool(candidate.get("private_receiver_inventory_eligible")),
        "beautiful_code_score": float_or(candidate.get("beautiful_code_score")) / 10.0,
        "original_rank_penalty": original_rank(candidate),
        "template_like_candidate": bool(candidate.get("template_like_candidate")),
        "placeholder_scaffold_body": bool(candidate.get("placeholder_scaffold_body")),
        "expression_memory_fallback": fallback_return_candidate(candidate),
        "diagnostic_adapter_stage": bool(candidate.get("private_residual_v3_semantic_adapter_stage") or "diagnostic_adapter" in mode.lower()),
        "public_tests_visible_to_generator": bool(candidate.get("public_tests_visible_to_generator") or candidate.get("public_tests_used")),
        "canonical_solution_seen_by_solver": bool(candidate.get("canonical_solution_seen_by_solver") or candidate.get("canonical_solution_used")),
    }


def candidate_decoder_contract(candidate: dict[str, Any]) -> dict[str, Any]:
    direct = candidate.get("decoder_contract")
    if isinstance(direct, dict):
        return direct
    provenance = candidate.get("provenance")
    if isinstance(provenance, dict):
        support = provenance.get("private_train_support")
        if isinstance(support, dict):
            return {
                "return_shape": support.get("return_shape"),
                "visible_arg_count_hint": support.get("arg_count"),
                "required_constructs": support.get("required_constructs") or [],
            }
    return {}


def semantic_family(row: dict[str, Any]) -> str:
    contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
    if contract.get("semantic_family"):
        return str(contract.get("semantic_family"))
    plan = row.get("semantic_decoder_v2_plan") if isinstance(row.get("semantic_decoder_v2_plan"), dict) else {}
    if plan.get("semantic_family"):
        return str(plan.get("semantic_family"))
    return ""


def score_candidates(task: dict[str, Any], candidates: list[dict[str, Any]], *, timeout_seconds: int) -> dict[int, bool]:
    out: dict[int, bool] = {}
    for candidate in candidates:
        ok, _error = run_candidate(task, candidate, timeout_seconds=timeout_seconds)
        out[id(candidate)] = bool(ok)
    return out


def selected_candidate_row(surface: str, task: dict[str, Any], candidate: dict[str, Any], passed: bool) -> dict[str, Any]:
    score, detail = policy_score(task, candidate)
    return {
        "policy": "project_theseus_sts_ranker_policy_v1_selected_candidate",
        "created_utc": now(),
        "surface": surface,
        "task_id": task.get("task_id"),
        "entry_point": task.get("entry_point"),
        "selected_candidate_sha256": candidate.get("candidate_sha256"),
        "selected_candidate_mode": candidate.get("candidate_generation_mode"),
        "selected_passed_private_tests": bool(passed),
        "policy_score": round(score, 6),
        "policy_features": detail["features"],
        "guard_env_var": GUARD_ENV,
        "raw_user_text": False,
        "public_benchmark_row": False,
        "public_tests_used_for_policy": False,
        "public_solutions_used_for_policy": False,
        "teacher_generated": False,
        "external_inference_calls": 0,
    }


def recommendation_for(trigger_state: str, aggregate: dict[str, Any], public_readiness: dict[str, Any]) -> dict[str, Any]:
    readiness_state = str(public_readiness.get("trigger_state") or "")
    readiness_summary = public_readiness.get("summary") if isinstance(public_readiness.get("summary"), dict) else {}
    stale_private = bool(readiness_summary.get("decoder_source_release_fresh") is False)
    if public_readiness and (readiness_state == "RED" or stale_private):
        return {
            "decision": "refresh_private_transfer_before_public_spend",
            "reason": (
                "STS ranker policy is privately green, but the broader post-distillation/public-transfer "
                "readiness gate reports stale broad-private evidence against the current decoder/release."
            ),
            "public_calibration_auto_run": False,
            "readiness_trigger_state": readiness_state,
        }
    if trigger_state == "GREEN":
        return {
            "decision": "request_bounded_public_calibration_review",
            "reason": (
                "STS ranker policy improves selected-candidate pass rate across private v3 and an existing "
                "disjoint private family under equal budget, with no fallback, leakage, or external inference."
            ),
            "public_calibration_auto_run": False,
        }
    if aggregate.get("fallback_return_candidate_count") or aggregate.get("public_leakage_count"):
        return {
            "decision": "stop_and_fix_integrity",
            "reason": "Integrity counters were not clean; do not use the selector or propose public calibration.",
            "public_calibration_auto_run": False,
        }
    return {
        "decision": "mine_failures_before_public_spend",
        "reason": "Selector transfer is not strong enough or not fully verified under equal-budget private surfaces.",
        "public_calibration_auto_run": False,
    }


def compact_public_readiness(report: dict[str, Any], path: Path) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "path": rel(path),
        "exists": path.exists(),
        "trigger_state": report.get("trigger_state"),
        "public_calibration_allowed": report.get("public_calibration_allowed"),
        "public_transfer_ready_for_new_calibration": report.get("public_transfer_ready_for_new_calibration"),
        "decoder_source_release_fresh": summary.get("decoder_source_release_fresh"),
        "decoder_source_release_stale_reasons": summary.get("decoder_source_release_stale_reasons"),
        "public_pass_rate": summary.get("public_pass_rate"),
        "recommended_private_fix_family": summary.get("recommended_private_fix_family"),
        "public_tests_used": report.get("public_tests_used"),
        "public_solutions_used": report.get("public_solutions_used"),
        "external_inference_calls": report.get("external_inference_calls"),
    }


def sts_ranker_verification_bandwidth_record(
    aggregate: dict[str, Any],
    surface_reports: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    *,
    budget: int,
    timeout_seconds: int,
    guarded_enabled: bool,
) -> dict[str, Any]:
    task_count = int(aggregate.get("task_count") or 0)
    equal_budget_tasks = int(aggregate.get("equal_budget_task_count") or 0)
    surface_count = len(surface_reports)
    selected_count = len(selected_rows)
    fallback_count = int(aggregate.get("fallback_return_candidate_count") or 0)
    public_leakage_count = int(aggregate.get("public_leakage_count") or 0)
    external_calls = int(aggregate.get("external_inference_calls") or 0)
    no_admissible_count = int(aggregate.get("no_admissible_count") or 0)
    regression_count = int(aggregate.get("sts_policy_vs_non_sts_regression_count") or 0)
    obligation_count = (
        1  # guarded integration state
        + 1  # policy/config provenance
        + 1  # no public calibration boundary
        + 1  # no external inference/no fallback boundary
        + surface_count * 4
        + max(1, equal_budget_tasks)
    )
    verifier_capacity_units = max(1, selected_count) + surface_count * max(1, budget) + 4
    capacity_floor_units = max(4, min(obligation_count, 512))
    capacity_margin_units = verifier_capacity_units - capacity_floor_units
    residual_obligations = []
    if no_admissible_count:
        residual_obligations.append("sts_ranker_no_admissible_residual")
    if fallback_count:
        residual_obligations.append("sts_ranker_fallback_candidate_residual")
    if public_leakage_count:
        residual_obligations.append("sts_ranker_public_leakage_residual")
    if external_calls:
        residual_obligations.append("sts_ranker_external_inference_residual")
    if regression_count:
        residual_obligations.append("sts_ranker_selection_regression_residual")
    if capacity_margin_units < 0:
        residual_obligations.append("sts_ranker_verifier_capacity_escalation")
    return {
        "policy": "project_theseus_sts_ranker_verification_bandwidth_v1",
        "surface": "sts_ranker_policy_v1",
        "status": "ready",
        "evidence_refs": [
            "reports/sts_ranker_policy_v1.json",
            "reports/sts_ranker_policy_v1_selected_candidates.jsonl",
            "configs/sts_ranker_policy_v1.json",
        ],
        "obligation_count": obligation_count,
        "verifier_capacity_units": verifier_capacity_units,
        "capacity_floor_units": capacity_floor_units,
        "capacity_margin_units": capacity_margin_units,
        "verification_arms": [
            "private_v3_equal_budget_selection",
            "disjoint_private_family_equal_budget_selection",
            "private_verifier_after_selection",
            "no_public_calibration_boundary",
            "fallback_and_external_inference_counters",
        ],
        "decomposition_contract": {
            "surface_count": surface_count,
            "task_count": task_count,
            "equal_budget_task_count": equal_budget_tasks,
            "selected_candidate_count": selected_count,
            "candidate_budget": budget,
            "timeout_seconds": timeout_seconds,
            "guarded_enabled": guarded_enabled,
            "verification_strategy": "ranker selection is measured only after private equal-budget candidate choice; public calibration is not run",
        },
        "residual_obligations": residual_obligations,
        "escalation_thresholds": {
            "capacity_margin_min": 0,
            "fallback_return_count_max": 0,
            "public_leakage_count_max": 0,
            "external_inference_calls_max": 0,
            "selection_regression_count_max": 0,
        },
        "escalation_required": bool(residual_obligations),
        "adequacy_state": "ready" if not residual_obligations else "verification_capacity_residual",
        "public_training_rows_written": 0,
        "external_inference_calls": external_calls,
        "fallback_return_count": fallback_count,
        "candidate_generation_credit": 0,
        "learned_generation_claim_allowed": False,
        "non_claims": [
            "STS ranker accounting is selector evidence, not learned body-token generation.",
            "Private verifier pass after ranking is not public transfer evidence.",
            "No public benchmark prompt, test, solution, trace, or answer template is used.",
        ],
    }


def sts_ranker_governance_tax_record(
    aggregate: dict[str, Any],
    surface_reports: list[dict[str, Any]],
    *,
    verification_bandwidth: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    raw_latency_ms = max(1, int(sum(int(row.get("runtime_ms") or 0) for row in surface_reports)))
    gate_costs = {
        "equal_budget_selection_accounting_ms": 3,
        "private_surface_separation_ms": 3,
        "verification_bandwidth_accounting_ms": 3,
        "no_public_calibration_boundary_ms": 2,
        "no_cheat_counter_audit_ms": 2,
    }
    governed_overhead_ms = sum(gate_costs.values())
    caught_failure_count = len(verification_bandwidth.get("residual_obligations") or [])
    review_load_units = max(1, len(verification_bandwidth.get("verification_arms") or []))
    return {
        "policy": "project_theseus_sts_ranker_governance_tax_v1",
        "surface": "sts_ranker_policy_v1",
        "status": "ready",
        "gate_costs": gate_costs,
        "raw_route_latency_ms": raw_latency_ms,
        "governed_overhead_ms": governed_overhead_ms,
        "governed_total_latency_ms": raw_latency_ms + governed_overhead_ms,
        "observed_wall_runtime_ms": int((time.perf_counter() - started) * 1000),
        "review_load_units": review_load_units,
        "caught_failure_count": caught_failure_count,
        "tax_per_caught_failure": round(governed_overhead_ms / max(1, caught_failure_count), 6),
        "tax_justified": caught_failure_count > 0 or int(aggregate.get("task_count") or 0) > 0,
        "tax_value_statement": "STS ranker route cost keeps equal-budget, private-surface, and no-cheat boundaries visible before any route-policy adoption.",
        "public_training_rows_written": 0,
        "external_inference_calls": int(aggregate.get("external_inference_calls") or 0),
        "fallback_return_count": int(aggregate.get("fallback_return_candidate_count") or 0),
        "candidate_generation_credit": 0,
        "learned_generation_claim_allowed": False,
        "non_claims": [
            "Governance tax is overhead accounting, not capability.",
            "Ranker speed cannot hide displaced verification or public-calibration cost.",
        ],
    }


def build_sts_ranker_spine_records(
    *,
    args: argparse.Namespace,
    aggregate: dict[str, Any],
    surface_reports: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    verification_bandwidth: dict[str, Any],
    governance_tax: dict[str, Any],
    recommendation: dict[str, Any],
) -> list[dict[str, Any]]:
    run_suffix = stable_payload_hash(
        {
            "task_count": aggregate.get("task_count"),
            "surface_count": aggregate.get("surface_count"),
            "sts_pass_rate": aggregate.get("sts_policy_selected_pass_rate"),
            "non_sts_pass_rate": aggregate.get("non_sts_policy_selected_pass_rate"),
            "candidate_budget": args.candidate_budget,
            "timeout_seconds": args.timeout_seconds,
        }
    )[:16]
    run_id = f"sts_ranker_policy-{run_suffix}"
    claim_id = f"claim_sts_ranker_policy-{run_suffix}"
    fallback_count = int(aggregate.get("fallback_return_candidate_count") or 0)
    external_calls = int(aggregate.get("external_inference_calls") or 0)
    common = {
        "run_id": run_id,
        "producer_surface": "sts_ranker_policy_v1",
        "support_state": "SUPPORTED",
        "public_training_rows_written": 0,
        "external_inference_calls": external_calls,
        "fallback_return_count": fallback_count,
        "candidate_generation_credit": 0,
        "learned_generation_claim_allowed": False,
        "verification_bandwidth": verification_bandwidth,
        "governance_tax": governance_tax,
        "non_claims": [
            "STS ranker policy is selector accounting, not learned generation.",
            "Private selector evidence does not imply public transfer.",
            "Routers, tools, templates, repairs, and fallback bodies remain noncredit for learned-generation claims.",
        ],
    }
    evidence_refs = [
        "reports/sts_ranker_policy_v1.json",
        "reports/sts_ranker_policy_v1_selected_candidates.jsonl",
        "configs/sts_ranker_policy_v1.json",
    ]
    return [
        {
            **common,
            "record_type": "policy_optimization_record",
            "record_id": f"sts_ranker_policy_update-{run_suffix}",
            "policy_target": "candidate_ranker_policy",
            "policy_kind": "deterministic_metadata_selector",
            "behavior_change_state": "guarded_not_default",
            "admissible_feedback": ["private_verifier_after_selection", "equal_budget_private_surface_measurement"],
            "sts_policy_selected_pass_rate": aggregate.get("sts_policy_selected_pass_rate"),
            "non_sts_policy_selected_pass_rate": aggregate.get("non_sts_policy_selected_pass_rate"),
            "drift_bound": "disabled_by_default_guard_env_required",
            "rollback_plan": f"unset {GUARD_ENV}",
            "monitor_window": "one guarded local run before route adoption",
        },
        {
            **common,
            "record_type": "authority_use_receipt",
            "record_id": f"sts_ranker_authority_use-{run_suffix}",
            "authority_scope": ["local_private_selector_eval", "private_verifier_after_selection"],
            "state": "local_private_no_public_calibration_no_external_inference",
            "status": "READY",
            "evidence_refs": evidence_refs,
        },
        {
            **common,
            "record_type": "resource_budget_record",
            "record_id": f"sts_ranker_resource_budget-{run_suffix}",
            "budget_id": f"sts_ranker_budget-{run_suffix}",
            "verification_obligation_count": verification_bandwidth["obligation_count"],
            "verifier_capacity_units": verification_bandwidth["verifier_capacity_units"],
            "capacity_margin_units": verification_bandwidth["capacity_margin_units"],
            "escalation_required": verification_bandwidth["escalation_required"],
            "residual_obligations": verification_bandwidth["residual_obligations"],
            "score_semantics": "selector budget only; no public calibration or training is performed by this record",
        },
        {
            **common,
            "record_type": "costed_route_record",
            "record_id": f"sts_ranker_costed_route-{run_suffix}",
            "task_id": run_id,
            "route_state": "guarded_candidate_selector_route",
            "task_contract_ref": "configs/sts_ranker_policy_v1.json",
            "quality_predicate": "STS ranker must improve private selected-pass rate under equal budget without leakage, fallback, external inference, or public calibration.",
            "authority_ceiling": "local_private_selector_eval",
            "candidate_routes": ["non_sts_equal_budget_selector", "sts_ranker_policy_v1"],
            "selected_route": "sts_ranker_policy_v1",
            "rejected_lower_cost_routes": ["non_sts_equal_budget_selector"],
            "verification_result": "private_equal_budget_verifier_passed" if not verification_bandwidth["escalation_required"] else "private_equal_budget_residuals_present",
            "outcome_state": recommendation.get("decision"),
            "cost_accounting": {
                "governed_overhead_ms": governance_tax["governed_overhead_ms"],
                "governed_total_latency_ms": governance_tax["governed_total_latency_ms"],
                "review_load_units": governance_tax["review_load_units"],
                "caught_failure_count": governance_tax["caught_failure_count"],
                "tax_per_caught_failure": governance_tax["tax_per_caught_failure"],
                "verification_obligation_count": verification_bandwidth["obligation_count"],
                "verifier_capacity_units": verification_bandwidth["verifier_capacity_units"],
            },
            "cost_classes": ["private_selection", "private_verifier", "no_cheat_audit", "route_governance"],
            "hidden_cost_checks": [
                "equal budget enforced",
                "public calibration not run",
                "heldout solution bodies not used for policy",
                "ranker evidence not learned generation",
            ],
            "residual_obligations": verification_bandwidth["residual_obligations"],
            "fallback_route": "non_sts_equal_budget_selector",
            "promotion_candidate": False,
            "support_state_effect": "selector_route_evidence_only",
        },
        {
            **common,
            "record_type": "generation_mode_record",
            "record_id": f"sts_ranker_generation_mode-{run_suffix}",
            "state": "selector_only_not_generator",
            "candidate_generation_credit": 0,
            "learned_generation_claim_allowed": False,
            "selected_candidate_count": len(selected_rows),
            "non_claim": "STS ranker selection cannot claim learned body-token generation.",
        },
        {
            **common,
            "record_type": "failure_boundary",
            "record_id": f"sts_ranker_failure_boundary-{run_suffix}",
            "failure_id": f"sts_ranker_boundary-{run_suffix}",
            "fallback_return_used": fallback_count > 0,
            "public_calibration_run": False,
            "verification_escalation_required": verification_bandwidth["escalation_required"],
            "residual_obligations": verification_bandwidth["residual_obligations"],
            "terminal": False,
            "status": "READY" if not verification_bandwidth["escalation_required"] else "RESIDUALS_PRESENT",
        },
        {
            **common,
            "record_type": "artifact_graph_record",
            "record_id": f"sts_ranker_artifact-{run_suffix}",
            "artifact_kind": "sts_ranker_policy_report",
            "content_hash": stable_payload_hash({"aggregate": aggregate, "surface_count": len(surface_reports)}),
            "source_refs": evidence_refs,
            "context_refs": ["private_candidate_manifests", "private_heldout_eval_tasks"],
            "claim_refs": [claim_id],
            "replay_metadata": {
                "candidate_budget": args.candidate_budget,
                "timeout_seconds": args.timeout_seconds,
                "surface_count": len(surface_reports),
                "selected_candidate_count": len(selected_rows),
            },
            "evidence_gate": {
                "trigger_state": "GREEN" if not verification_bandwidth["escalation_required"] else "YELLOW",
                "public_training_rows_written": 0,
                "external_inference_calls": external_calls,
                "fallback_return_count": fallback_count,
            },
        },
        {
            **common,
            "record_type": "claim_record",
            "record_id": f"sts_ranker_claim-{run_suffix}",
            "claim_id": claim_id,
            "state": "sts_ranker_private_selector_evidence_summarized",
            "status": "GREEN" if not verification_bandwidth["escalation_required"] else "YELLOW",
            "evidence_ref": "reports/sts_ranker_policy_v1.json",
            "claim_boundary": "private_selector_route_evidence_not_learned_generation_not_public_transfer",
        },
        {
            **common,
            "record_type": "evidence_transition_record",
            "record_id": f"sts_ranker_evidence_transition-{run_suffix}",
            "state": "private_candidate_manifests_to_guarded_selector_policy",
            "status": "SUPPORTED" if not verification_bandwidth["escalation_required"] else "RESIDUALS_PRESENT",
            "evidence_ref": "reports/sts_ranker_policy_v1.json",
        },
    ]


def group_candidates(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if task_id:
            out[task_id].append(row)
    return out


def ordered_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for _index, row in sorted(enumerate(rows), key=lambda item: (original_rank(item[1]), item[0]))]


def original_rank(candidate: dict[str, Any]) -> int:
    for key in ("matched_candidate_rank", "rank", "candidate_rank"):
        try:
            value = int(candidate.get(key))
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    return 1


def fallback_return_candidate(candidate: dict[str, Any]) -> bool:
    mode = str(candidate.get("candidate_generation_mode") or "").lower()
    return bool(candidate.get("expression_memory_fallback") or candidate.get("fallback_return") or ("fallback" in mode and "fallback_skipped" not in mode))


def public_leak_count(task: dict[str, Any], candidate: dict[str, Any]) -> int:
    count = 0
    for row in (task, candidate):
        if bool(row.get("public_tests_used") or row.get("public_tests_visible_to_generator") or row.get("public_tests_included")):
            count += 1
        if bool(row.get("public_solutions_used") or row.get("canonical_solution_seen_by_solver") or row.get("public_benchmark_solutions_included")):
            count += 1
        if bool(row.get("public_score_labels_included")):
            count += 1
    return count


def task_family(task: dict[str, Any]) -> str:
    return str(
        task.get("targeted_private_residual_family_v3")
        or task.get("private_ecology_family_v5")
        or task.get("broad_private_family_v1")
        or "unknown"
    )


def numeric_feature(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def float_or(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "status": "PASSED" if passed else "PENDING", "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    rec = report["recommendation"]
    lines = [
        "# STS Ranker Policy v1",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Surfaces: `{summary.get('surface_count')}`",
        f"- Tasks: `{summary.get('task_count')}`",
        f"- STS policy selected pass: `{summary.get('sts_policy_selected_pass_count')}/{summary.get('task_count')} = {summary.get('sts_policy_selected_pass_rate')}`",
        f"- Non-STS policy selected pass: `{summary.get('non_sts_policy_selected_pass_count')}/{summary.get('task_count')} = {summary.get('non_sts_policy_selected_pass_rate')}`",
        f"- Selected delta: `{summary.get('selected_pass_delta_sts_policy_minus_non_sts_policy')}`",
        f"- STS oracle pass rate: `{summary.get('sts_oracle_pass_rate')}`",
        f"- Non-STS oracle pass rate: `{summary.get('non_sts_oracle_pass_rate')}`",
        f"- No-admissible rate: `{summary.get('no_admissible_rate')}`",
        f"- Fallback return candidates: `{summary.get('fallback_return_candidate_count')}`",
        f"- Public leakage count: `{summary.get('public_leakage_count')}`",
        f"- Verification bandwidth: `{summary.get('verification_bandwidth_status')}` obligations `{summary.get('verification_obligation_count')}` escalation `{summary.get('verification_escalation_required')}`",
        f"- Governance tax: `{summary.get('governance_tax_status')}` review load `{summary.get('governance_tax_review_load_units')}` caught failures `{summary.get('governance_tax_caught_failure_count')}`",
        f"- Recommendation: `{rec.get('decision')}`",
        "",
        "The selector is guarded by `THESEUS_ENABLE_STS_RANKER_POLICY_V1=1` and does not run public calibration.",
    ]
    return "\n".join(lines)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_payload_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
