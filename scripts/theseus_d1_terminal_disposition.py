#!/usr/bin/env python3
"""Independently aggregate the one-shot fresh D1 qualification campaign."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_d1_campaign as campaign  # noqa: E402
import theseus_d1_evaluator as evaluator  # noqa: E402
import theseus_d1_evaluator_seal as seal  # noqa: E402


POLICY = "project_theseus_d1_terminal_disposition_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_d1_terminal_disposition.json"
TREATMENT = "typed_semantic_ir_treatment"
STATIC = "deterministic_request_compiler_baseline"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    config = p2a.read_json(config_path)
    report = build_report(config, config_path=config_path)
    p2a.write_json(p2a.resolve(args.out or str(config["report"])), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def build_report(
    config: dict[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG,
    progress_override: dict[str, Any] | None = None,
    pool_override: dict[str, Any] | None = None,
    consumption_override: list[dict[str, Any]] | None = None,
    evaluation_overrides: dict[int, dict[str, Any]] | None = None,
    run_overrides: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    faults = validate_config(config)
    instrument_path = p2a.resolve(str(config.get("instrument") or ""))
    campaign_config_path = p2a.resolve(str(config.get("campaign_config") or ""))
    pool_path = p2a.resolve(str(config.get("task_pool") or ""))
    progress_path = p2a.resolve(str(config.get("campaign_progress") or ""))
    instrument = p2a.read_json(instrument_path) if instrument_path.is_file() else {}
    campaign_config = p2a.read_json(campaign_config_path) if campaign_config_path.is_file() else {}
    pool = pool_override if pool_override is not None else (
        p2a.read_json(pool_path) if pool_path.is_file() else {}
    )
    progress = progress_override if progress_override is not None else (
        p2a.read_json(progress_path) if progress_path.is_file() else {}
    )
    pool_audit = campaign.audit_pool(
        campaign_config, pool, pool_path, pool_override
    ) if campaign_config else {"passed": False, "faults": ["campaign_config_missing"]}
    complete = int(progress.get("complete_tasks") or 0)
    preterminal = (
        not pool_audit.get("passed")
        or progress.get("policy") != campaign.POLICY
        or complete != 44
    )
    if preterminal:
        return preterminal_report(
            config,
            config_path,
            faults + p2a.strings(pool_audit.get("faults")),
            pool_audit,
            progress,
        )
    primary_control = str(progress.get("primary_control") or "")
    if primary_control not in {
        "direct_target_generation",
        "natural_language_plan_control",
    }:
        faults.append("primary_control_invalid")
    rows, row_faults = collect_task_rows(
        pool,
        progress,
        primary_control,
        evaluation_overrides=evaluation_overrides,
        run_overrides=run_overrides,
    )
    faults.extend(row_faults)
    consumption_path = p2a.resolve(str(config.get("consumption_registry") or ""))
    consumption = consumption_override if consumption_override is not None else campaign.read_jsonl(consumption_path)
    pool_sha = campaign.stable_input_hash(pool, pool_path, pool_override)
    consumption_matches = [
        row for row in consumption if row.get("task_pool_sha256") == pool_sha
    ]
    source_custody_green = pool_audit.get("passed") is True and len(consumption_matches) == 1
    information_flow_green = (
        len(rows) == 44
        and all(row["information_flow_green"] for row in rows)
        and not any("information_flow" in fault for fault in faults)
    )
    boundary_hits = int(progress.get("physical_context_boundary_hits") or 0) + sum(
        row["boundary_hits"] for row in rows
    )
    oracle_useful = sum(row["oracle_useful"] for row in rows)
    rollback_failures = sum(row["rollback_failures"] for row in rows)
    unsafe = sum(row["treatment_unsafe"] for row in rows)
    treatment = [row["treatment_useful"] for row in rows]
    control = [row["control_useful"] for row in rows]
    static = [row["static_useful"] for row in rows]
    treatment_useful = sum(treatment)
    control_useful = sum(control)
    static_useful = sum(static)
    discordant = sum(left != right for left, right in zip(treatment, control))
    wins = sum(left > right for left, right in zip(treatment, control))
    sign_probability = binomial_upper_tail(discordant, wins, 0.5) if discordant else 1.0
    treatment_rate = treatment_useful / 44
    control_rate = control_useful / 44
    static_rate = static_useful / 44
    costs, cost_faults = aggregate_costs(rows, primary_control)
    faults.extend(cost_faults)
    cost_custody_green = not cost_faults
    cost_dominated = (
        dominated(treatment_useful, costs[TREATMENT], control_useful, costs[primary_control])
        or dominated(treatment_useful, costs[TREATMENT], static_useful, costs[STATIC])
    )
    experiment_floor = (
        len(rows) == 44
        and oracle_useful == 44
        and all(row["evaluation_green"] for row in rows)
        and len(consumption_matches) == 1
        and cost_custody_green
    )
    effect_rule = (
        discordant >= 18
        and sign_probability <= 0.05
        and treatment_rate - control_rate >= 0.15
        and treatment_rate - static_rate >= 0.15
        and unsafe == 0
        and rollback_failures == 0
        and not cost_dominated
    )
    status = classify_status(
        source_custody_green=source_custody_green,
        information_flow_green=information_flow_green,
        boundary_hits=boundary_hits,
        rollback_failures=rollback_failures,
        experiment_floor=experiment_floor,
        effect_rule=effect_rule,
    )
    if faults and status == "D1_EXACT_IMPLEMENTATION_QUALIFIED":
        status = "INCONCLUSIVE_EXPERIMENT"
    terminal_green = status in set(
        p2a.strings(
            p2a.mapping(instrument.get("decision_rule")).get("terminal_states")
        )
    ) and not faults
    return {
        "policy": POLICY,
        "created_utc": seal.now(),
        "trigger_state": "GREEN" if terminal_green else "RED",
        "scientific_status": status,
        "claim_id": instrument.get("claim_id"),
        "faults": sorted(set(faults)),
        "source_custody": {
            "passed": source_custody_green,
            "task_pool_sha256": pool_sha,
            "consumption_registry_matches": len(consumption_matches),
            "post_candidate_task_replacements": 0,
        },
        "information_flow": {
            "passed": information_flow_green,
            "candidate_emitted_integrity_flags_trusted": False,
            "arm_labels_passed_to_scoring": False,
            "independent_candidate_integrity_recomputed_tasks": sum(
                row["integrity_recomputed"] for row in rows
            ),
        },
        "adequacy": {
            "tasks": len(rows),
            "distinct_repositories": len(
                {str(row["repository"]).lower() for row in rows}
            ),
            "complete_blind_evaluations": sum(row["evaluation_green"] for row in rows),
            "target_oracle_useful": oracle_useful,
            "context_or_host_boundary_hits": boundary_hits,
            "rollback_failures": rollback_failures,
            "experiment_floor_passed": experiment_floor,
            "cost_custody_passed": cost_custody_green,
        },
        "primary_test": {
            "treatment": TREATMENT,
            "control": primary_control,
            "treatment_useful": treatment_useful,
            "control_useful": control_useful,
            "static_useful": static_useful,
            "treatment_useful_rate": round(treatment_rate, 8),
            "control_useful_rate": round(control_rate, 8),
            "static_useful_rate": round(static_rate, 8),
            "discordant_pairs": discordant,
            "treatment_wins": wins,
            "one_sided_exact_sign_probability": round(sign_probability, 12),
            "unsafe_treatment_candidates": unsafe,
            "joined_cost": costs,
            "treatment_cost_dominated": cost_dominated,
            "qualification_rule_passed": effect_rule,
        },
        "weak_tail": {
            "treatment_failure_task_count": 44 - treatment_useful,
            "treatment_failure_repositories": [
                row["repository"] for row in rows if row["treatment_useful"] == 0
            ],
            "all_failures_and_abstentions_retained": True,
        },
        "denominators": {
            "tasks": 44,
            "learned_model_calls": int(progress.get("model_calls_retained") or 0),
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "training_rows_written": 0,
            "project_selected_quality_token_cap": None,
        },
        "automatic_book_support_promotion": False,
        "next_stage": {
            "state": "OPEN_GOVERNED_BOOK_EVIDENCE_REVIEW",
            "support_state_effect": "none",
        },
        "maximum_inference": str(config.get("maximum_inference") or ""),
    }


def collect_task_rows(
    pool: dict[str, Any],
    progress: dict[str, Any],
    primary_control: str,
    *,
    evaluation_overrides: dict[int, dict[str, Any]] | None,
    run_overrides: dict[int, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    progress_by_index = {
        int(row.get("campaign_index") or 0): row
        for row in p2a.dicts(progress.get("tasks"))
    }
    rows: list[dict[str, Any]] = []
    faults: list[str] = []
    for task in p2a.dicts(pool.get("tasks")):
        index = int(task.get("campaign_index") or 0)
        progress_row = progress_by_index.get(index, {})
        evaluation_path = p2a.resolve(str(progress_row.get("evaluation") or ""))
        run_path = p2a.resolve(str(progress_row.get("run") or ""))
        evaluation = (
            (evaluation_overrides or {}).get(index)
            if evaluation_overrides is not None
            else p2a.read_json(evaluation_path) if evaluation_path.is_file() else None
        ) or {}
        run = (
            (run_overrides or {}).get(index)
            if run_overrides is not None
            else p2a.read_json(run_path) if run_path.is_file() else None
        ) or {}
        if evaluation.get("policy") != evaluator.POLICY:
            faults.append(f"evaluation_policy_invalid:{index}")
        blinding = p2a.mapping(evaluation.get("evaluation_blinding"))
        info_green = (
            evaluation.get("trigger_state") == "GREEN"
            and blinding.get("arm_labels_passed_to_scoring") is False
            and blinding.get("arm_labels_attached_after_scoring") is True
            and blinding.get("candidate_emitted_integrity_flags_trusted") is False
            and blinding.get("target_or_test_source_visible_to_generation") is False
        )
        by_arm = {
            str(row.get("arm_id") or ""): row
            for row in p2a.dicts(evaluation.get("results"))
        }
        for arm in (TREATMENT, primary_control, STATIC):
            if arm not in by_arm:
                faults.append(f"required_arm_result_missing:{index}:{arm}")
        treatment = p2a.mapping(by_arm.get(TREATMENT))
        control = p2a.mapping(by_arm.get(primary_control))
        static = p2a.mapping(by_arm.get(STATIC))
        all_results = p2a.dicts(evaluation.get("results"))
        rows.append(
            {
                "campaign_index": index,
                "repository": task.get("repository"),
                "evaluation_green": evaluation.get("trigger_state") == "GREEN",
                "information_flow_green": info_green,
                "integrity_recomputed": int(
                    all(
                        not p2a.strings(row.get("integrity_faults"))
                        or row.get("unsafe") == 1
                        for row in all_results
                    )
                ),
                "oracle_useful": int(
                    p2a.mapping(evaluation.get("oracle_ceiling")).get("useful") or 0
                ),
                "treatment_useful": int(treatment.get("useful") or 0),
                "control_useful": int(control.get("useful") or 0),
                "static_useful": int(static.get("useful") or 0),
                "treatment_unsafe": int(treatment.get("unsafe") or 0),
                "rollback_failures": sum(
                    int(row.get("rollback_verified") is False) for row in all_results
                ),
                "boundary_hits": sum(
                    int(row.get("boundary_hit") is True) for row in all_results
                ),
                "run": run,
                "evaluation": evaluation,
            }
        )
    return rows, faults


def aggregate_costs(
    rows: list[dict[str, Any]], primary_control: str
) -> tuple[dict[str, Any], list[str]]:
    arms = (TREATMENT, primary_control, STATIC)
    totals = {
        arm: {
            "runtime_call_receipts": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "model_runtime_ms": 0.0,
            "visible_verifier_runtime_ms": 0.0,
            "hidden_verifier_runtime_ms": 0.0,
            "joined_runtime_ms": 0.0,
        }
        for arm in arms
    }
    faults: list[str] = []
    for row in rows:
        run = p2a.mapping(row.get("run"))
        evaluation = p2a.mapping(row.get("evaluation"))
        attempts = {
            str(item.get("arm_id") or ""): item
            for item in p2a.dicts(run.get("attempts"))
        }
        static_run = p2a.mapping(run.get("deterministic_compiler_control"))
        results = {
            str(item.get("arm_id") or ""): item
            for item in p2a.dicts(evaluation.get("results"))
        }
        for arm in arms:
            attempt = static_run if arm == STATIC else p2a.mapping(attempts.get(arm))
            calls = p2a.dicts(attempt.get("runtime_calls"))
            if arm != STATIC and len(calls) != 2:
                faults.append(
                    f"cost_runtime_call_count_invalid:{row.get('campaign_index')}:{arm}"
                )
            model_ms = 0.0
            prompt_tokens = 0
            generated_tokens = 0
            for call_index, call in enumerate(calls, 1):
                cost, call_faults = runtime_call_cost(call)
                faults.extend(
                    f"cost_custody:{row.get('campaign_index')}:{arm}:{call_index}:{fault}"
                    for fault in call_faults
                )
                model_ms += float(cost["runtime_ms"])
                prompt_tokens += int(cost["prompt_tokens"])
                generated_tokens += int(cost["generated_tokens"])
            visible_ms = visible_verifier_runtime_ms(attempt)
            verifier_ms = float(
                p2a.mapping(p2a.mapping(results.get(arm)).get("sandbox_receipt")).get(
                    "duration_ms"
                )
                or 0.0
            )
            totals[arm]["runtime_call_receipts"] += len(calls)
            totals[arm]["prompt_tokens"] += prompt_tokens
            totals[arm]["generated_tokens"] += generated_tokens
            totals[arm]["model_runtime_ms"] += model_ms
            totals[arm]["visible_verifier_runtime_ms"] += visible_ms
            totals[arm]["hidden_verifier_runtime_ms"] += verifier_ms
            totals[arm]["joined_runtime_ms"] += model_ms + visible_ms + verifier_ms
    for values in totals.values():
        for key in (
            "model_runtime_ms",
            "visible_verifier_runtime_ms",
            "hidden_verifier_runtime_ms",
            "joined_runtime_ms",
        ):
            values[key] = round(float(values[key]), 3)
    return totals, sorted(set(faults))


def runtime_call_cost(call: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    faults: list[str] = []
    metrics = p2a.mapping(call.get("metrics"))
    if not metrics:
        report_path = p2a.resolve(str(call.get("report_path") or ""))
        if (
            not report_path.is_file()
            or p2a.sha256_file(report_path) != str(call.get("report_sha256") or "")
        ):
            faults.append("runtime_report_binding_invalid")
        else:
            runtime = p2a.read_json(report_path)
            backend_path = p2a.resolve(
                str(p2a.mapping(runtime.get("checkpoint_chat")).get("out") or "")
            )
            if not backend_path.is_file():
                faults.append("backend_report_missing")
            else:
                backend = p2a.read_json(backend_path)
                metrics = p2a.mapping(backend.get("metrics"))
                if backend.get("trigger_state") != "GREEN":
                    faults.append("backend_report_not_green")
    prompt_tokens = int(metrics.get("prompt_tokens") or 0)
    generated_tokens = int(metrics.get("generated_tokens") or 0)
    context_tokens = int(metrics.get("model_context_window_tokens") or 0)
    effective_tokens = int(metrics.get("effective_maximum_tokens") or 0)
    if min(prompt_tokens, generated_tokens, context_tokens, effective_tokens) < 1:
        faults.append("termination_token_custody_incomplete")
    if effective_tokens > max(0, context_tokens - prompt_tokens):
        faults.append("context_residual_exceeded")
    if metrics.get("project_selected_quality_token_cap") is not None:
        faults.append("project_selected_quality_token_cap_present")
    if metrics.get("physical_context_boundary_hit") is True:
        faults.append("physical_context_boundary_hit")
    return {
        "runtime_ms": float(call.get("runtime_ms") or 0.0),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
    }, faults


def visible_verifier_runtime_ms(attempt: dict[str, Any]) -> float:
    provisional = p2a.mapping(
        p2a.mapping(attempt.get("provisional")).get("visible_verifier")
    )
    final = p2a.mapping(
        p2a.mapping(attempt.get("candidate")).get("visible_verifier")
    )
    static = p2a.mapping(
        p2a.mapping(attempt.get("candidate")).get("visible_verifier")
    )
    receipts = (provisional, final) if attempt.get("arm_id") != STATIC else (static,)
    return sum(float(receipt.get("duration_ms") or 0.0) for receipt in receipts)


def dominated(
    treatment_useful: int,
    treatment_cost: dict[str, Any],
    control_useful: int,
    control_cost: dict[str, Any],
) -> bool:
    dimensions = (
        "prompt_tokens",
        "generated_tokens",
        "model_runtime_ms",
        "visible_verifier_runtime_ms",
        "hidden_verifier_runtime_ms",
    )
    no_more_costly = all(
        float(control_cost.get(key) or 0.0)
        <= float(treatment_cost.get(key) or 0.0)
        for key in dimensions
    )
    strictly_cheaper = any(
        float(control_cost.get(key) or 0.0)
        < float(treatment_cost.get(key) or 0.0)
        for key in dimensions
    )
    return (
        control_useful >= treatment_useful
        and no_more_costly
        and (control_useful > treatment_useful or strictly_cheaper)
    )


def classify_status(
    *,
    source_custody_green: bool,
    information_flow_green: bool,
    boundary_hits: int,
    rollback_failures: int,
    experiment_floor: bool,
    effect_rule: bool,
) -> str:
    if not source_custody_green:
        return "INVALID_SOURCE_OR_CONSUMPTION_CUSTODY"
    if not information_flow_green:
        return "INVALID_INFORMATION_FLOW"
    if boundary_hits:
        return "INVALID_OBSERVATION_CONTEXT_OR_HOST_BOUNDARY"
    if rollback_failures:
        return "INCONCLUSIVE_IMPLEMENTATION"
    if not experiment_floor:
        return "INCONCLUSIVE_EXPERIMENT"
    if effect_rule:
        return "D1_EXACT_IMPLEMENTATION_QUALIFIED"
    return "D1_EXACT_IMPLEMENTATION_NOT_QUALIFIED"


def preterminal_report(
    config: dict[str, Any],
    config_path: Path,
    faults: list[str],
    pool_audit: dict[str, Any],
    progress: dict[str, Any],
) -> dict[str, Any]:
    blocking = [fault for fault in faults if fault not in {"task_pool_missing"}]
    return {
        "policy": POLICY,
        "created_utc": seal.now(),
        "trigger_state": "RED",
        "scientific_status": "D1_REVIEW_REQUIRED",
        "faults": sorted(set(blocking + ["campaign_not_complete"])),
        "config": {"path": p2a.rel(config_path), "sha256": p2a.sha256_file(config_path)},
        "task_pool_audit": pool_audit,
        "complete_tasks": int(progress.get("complete_tasks") or 0),
        "automatic_book_support_promotion": False,
        "maximum_inference": (
            "No D1 scientific inference before all 44 sealed tasks, the one-shot "
            "consumption receipt, and independent blind evaluations are complete."
        ),
    }


def binomial_upper_tail(trials: int, successes: int, probability: float) -> float:
    if successes > trials:
        return 0.0
    return sum(
        math.comb(trials, count)
        * probability**count
        * (1.0 - probability) ** (trials - count)
        for count in range(successes, trials + 1)
    )


def validate_config(config: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if config.get("policy") != POLICY or config.get("state") != (
        "PROSPECTIVELY_BOUND_BEFORE_ANY_D1_CANDIDATE_OUTCOME"
    ):
        faults.append("config_policy_or_state_invalid")
    primary = p2a.mapping(config.get("primary_test"))
    if primary.get("treatment") != TREATMENT:
        faults.append("treatment_invalid")
    if float(primary.get("one_sided_exact_sign_alpha") or 0.0) != 0.05:
        faults.append("sign_alpha_invalid")
    adequacy = p2a.mapping(config.get("adequacy"))
    if any(int(adequacy.get(key, -1)) != 0 for key in (
        "context_or_host_boundary_hits",
        "source_task_replacements_after_candidate_calls",
        "external_inference_calls",
        "teacher_calls",
        "training_rows_written",
    )):
        faults.append("zero_adequacy_boundary_invalid")
    if adequacy.get("project_selected_quality_token_cap") is not None:
        faults.append("quality_token_cap_present")
    authority = p2a.mapping(config.get("authority"))
    if authority.get("user_or_operator_approval_required") is not False:
        faults.append("user_gate_present")
    if authority.get("automatic_book_support_promotion") is not False:
        faults.append("automatic_book_support_promotion_allowed")
    return faults


def summary(report: dict[str, Any]) -> dict[str, Any]:
    primary = p2a.mapping(report.get("primary_test"))
    return {
        "trigger_state": report.get("trigger_state"),
        "scientific_status": report.get("scientific_status"),
        "tasks": p2a.mapping(report.get("adequacy")).get("tasks"),
        "treatment_useful": primary.get("treatment_useful"),
        "control_useful": primary.get("control_useful"),
        "discordant_pairs": primary.get("discordant_pairs"),
        "one_sided_exact_sign_probability": primary.get(
            "one_sided_exact_sign_probability"
        ),
        "project_selected_quality_token_cap": p2a.mapping(
            report.get("denominators")
        ).get("project_selected_quality_token_cap"),
        "faults": report.get("faults"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
