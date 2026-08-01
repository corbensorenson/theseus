#!/usr/bin/env python3
"""Recompute the terminal scientific disposition of the sealed P4S campaign."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_p4_cognitive_compilation_evaluator as p4_evaluator  # noqa: E402
import theseus_p4s_campaign as campaign  # noqa: E402
import theseus_p4s_cognitive_compilation as p4s  # noqa: E402


POLICY = "project_theseus_p4s_cognitive_compilation_terminal_disposition_v1"
POOL = ROOT / "configs" / "theseus_p4s_task_pool.json"
INSTRUMENT = ROOT / "configs" / "theseus_p4s_cognitive_compilation_instrument.json"
PROGRESS = ROOT / "reports" / "theseus_p4s_campaign_attempt2_progress.json"
OUT = ROOT / "reports" / "theseus_p4s_terminal_disposition.json"
ORACLE = "deterministic_compiler_oracle_ceiling"
LEARNED_ARMS = tuple(p4.ARMS)
COMPARATORS = (p4.DIRECT, p4.PLAN, p4.STATIC)
COST_DIMENSIONS = (
    "model_calls",
    "prompt_tokens",
    "generated_tokens",
    "model_runtime_ms",
    "verifier_runtime_ms",
    "rollback_failures",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=p2a.rel(OUT))
    args = parser.parse_args()
    report = build_report()
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "scientific_status": report["scientific_status"],
        "complete_tasks": report["denominators"]["tasks"],
        "learned_model_calls": report["denominators"]["learned_model_calls"],
        "safety_ceiling_hits": report["termination_custody"]["safety_ceiling_hits"],
        "faults": report["faults"],
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_report() -> dict[str, Any]:
    faults: list[str] = []
    campaign_audit = campaign.audit_campaign()
    if campaign_audit.get("trigger_state") != "GREEN":
        faults.append("campaign_audit_red")
    if campaign_audit.get("complete_tasks") != 10 or campaign_audit.get("pending_tasks") != 0:
        faults.append("campaign_not_complete")

    pool = p2a.read_json(POOL)
    instrument = p2a.read_json(INSTRUMENT)
    pool_audit = audit_source_pool(pool)
    faults.extend(pool_audit["faults"])
    tasks = p2a.dicts(pool.get("tasks"))
    if len(tasks) != 10:
        faults.append("task_count_invalid")

    totals = {arm: empty_totals() for arm in (*LEARNED_ARMS, p4.STATIC)}
    oracle_totals = empty_totals()
    termination_counts: collections.Counter[str] = collections.Counter()
    treatment_fault_counts: collections.Counter[str] = collections.Counter()
    task_rows: list[dict[str, Any]] = []
    runtime_receipts: list[dict[str, Any]] = []
    generated_tokens: list[int] = []
    prompt_tokens: list[int] = []
    route_blind_tasks = 0
    candidate_integrity_tasks = 0
    independent_evaluator_replay_tasks = 0
    exact_treatment_headers = 0
    treatment_terminal_end = 0

    for expected, pool_row in enumerate(tasks, 1):
        stem = str(pool_row.get("stem") or "")
        if int(pool_row.get("campaign_index") or 0) != expected:
            faults.append(f"campaign_index_invalid:{stem}")
        task_path = ROOT / str(pool_row.get("task") or "")
        evaluator_path = ROOT / str(pool_row.get("evaluator") or "")
        if p2a.sha256_file(task_path) != str(pool_row.get("task_sha256") or ""):
            faults.append(f"task_digest_mismatch:{stem}")
        if p2a.sha256_file(evaluator_path) != str(pool_row.get("evaluator_sha256") or ""):
            faults.append(f"evaluator_digest_mismatch:{stem}")
        paths = campaign.result_paths(pool_row)
        if not paths["run"].is_file() or not paths["evaluation"].is_file():
            faults.append(f"result_missing:{stem}")
            continue

        run = p2a.read_json(paths["run"])
        evaluation = p2a.read_json(paths["evaluation"])
        if run.get("policy") != p4s.POLICY:
            faults.append(f"run_policy_invalid:{stem}")
        if run.get("instrument_sha256") != p2a.sha256_file(INSTRUMENT):
            faults.append(f"run_instrument_mismatch:{stem}")
        if run.get("task_sha256") != p2a.sha256_file(task_path):
            faults.append(f"run_task_mismatch:{stem}")
        if evaluation.get("candidate_report_sha256") != p2a.sha256_file(paths["run"]):
            faults.append(f"evaluation_run_mismatch:{stem}")
        if evaluation.get("evaluator_sha256") != p2a.sha256_file(evaluator_path):
            faults.append(f"evaluation_owner_mismatch:{stem}")
        if evaluation.get("trigger_state") != "GREEN":
            faults.append(f"evaluation_red:{stem}")
        try:
            replayed_evaluation = p4_evaluator.evaluate_report(
                paths["run"], evaluator_path
            )
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            replayed_evaluation = {
                "trigger_state": "RED",
                "faults": [f"independent_replay_fault:{type(exc).__name__}"],
            }
            faults.append(
                f"independent_evaluation_replay_fault:{stem}:{type(exc).__name__}"
            )
        stored_projection = stable_evaluation_projection(evaluation)
        replayed_projection = stable_evaluation_projection(replayed_evaluation)
        replay_match = stored_projection == replayed_projection
        independent_evaluator_replay_tasks += int(replay_match)
        if not replay_match:
            faults.append(f"independent_evaluation_replay_mismatch:{stem}")
        if p2a.mapping(run.get("matched_set")).get("ready") is not True:
            faults.append(f"matched_set_invalid:{stem}")
        expected_order = list(p4.arm_order(expected))
        if run.get("actual_arm_order") != expected_order:
            faults.append(f"arm_order_invalid:{stem}")

        blinding = p2a.mapping(evaluation.get("evaluation_blinding"))
        blind = (
            blinding.get("arm_labels_passed_to_scoring") is False
            and blinding.get("arm_labels_attached_after_scoring") is True
            and blinding.get("candidate_authored_integrity_flags_trusted") is False
            and blinding.get("compiler_oracle_answer_visible_to_generation") is False
            and blinding.get("deterministic_control_target_or_oracle_visibility") is False
        )
        route_blind_tasks += int(blind)
        if not blind:
            faults.append(f"information_flow_invalid:{stem}")

        scored = {
            str(row.get("arm_id") or ""): row
            for row in p2a.dicts(evaluation.get("results"))
        }
        attempts = {
            str(row.get("arm_id") or ""): row
            for row in p2a.dicts(run.get("attempts"))
        }
        telemetry = {
            (str(row.get("arm_id") or ""), int(row.get("call_number") or 0)): row
            for row in p2a.dicts(run.get("generation_termination_telemetry"))
        }
        per_arm: dict[str, Any] = {}
        task_integrity_green = True

        for arm in LEARNED_ARMS:
            attempt = p2a.mapping(attempts.get(arm))
            result = p2a.mapping(scored.get(arm))
            if not attempt:
                faults.append(f"attempt_missing:{stem}:{arm}")
                continue
            parseable = attempt.get("parseable_candidate") is True
            if parseable != bool(result):
                faults.append(f"candidate_scoring_alignment_invalid:{stem}:{arm}")
            calls = p2a.dicts(attempt.get("runtime_calls"))
            if len(calls) != 2:
                faults.append(f"arm_call_count_invalid:{stem}:{arm}")
            for call in calls:
                call_number = int(call.get("call_number") or 0)
                termination = p2a.mapping(telemetry.get((arm, call_number)))
                runtime_path = p2a.resolve(str(call.get("report_path") or ""))
                runtime_valid = (
                    runtime_path.is_file()
                    and p2a.sha256_file(runtime_path) == str(call.get("report_sha256") or "")
                    and p2a.sha256_file(runtime_path) == str(termination.get("runtime_report_sha256") or "")
                )
                if not runtime_valid:
                    faults.append(f"runtime_receipt_invalid:{stem}:{arm}:{call_number}")
                reason = str(termination.get("termination_reason") or "")
                if reason not in campaign.NORMAL_TERMINATIONS:
                    faults.append(f"termination_invalid:{stem}:{arm}:{call_number}")
                if termination.get("safety_ceiling_hit") is True:
                    faults.append(f"physical_context_boundary_hit:{stem}:{arm}:{call_number}")
                if termination.get("completion_predicate_enabled") is not True:
                    faults.append(f"completion_predicate_disabled:{stem}:{arm}:{call_number}")
                termination_counts[reason] += 1
                prompt = int(termination.get("prompt_tokens") or 0)
                generated = int(termination.get("generated_tokens") or 0)
                prompt_tokens.append(prompt)
                generated_tokens.append(generated)
                totals[arm]["model_calls"] += 1
                totals[arm]["prompt_tokens"] += prompt
                totals[arm]["generated_tokens"] += generated
                totals[arm]["model_runtime_ms"] += float(call.get("runtime_ms") or 0.0)
                runtime_receipts.append({
                    "campaign_index": expected,
                    "arm_id": arm,
                    "call_number": call_number,
                    "runtime_report": p2a.rel(runtime_path),
                    "runtime_report_sha256": p2a.sha256_file(runtime_path),
                    "backend_report": termination.get("backend_report"),
                    "backend_report_sha256": termination.get("backend_report_sha256"),
                    "termination_reason": reason,
                    "prompt_tokens": prompt,
                    "generated_tokens": generated,
                    "runtime_ms": call.get("runtime_ms"),
                    "safety_ceiling_hit": termination.get("safety_ceiling_hit"),
                })

            evaluated = int(result.get("correctness_evaluated") or 0)
            useful = int(result.get("useful") or 0)
            unsafe = int(result.get("unsafe") or 0)
            rollback = int(result.get("rollback_verified") or 0)
            add_outcome(totals[arm], parseable, evaluated, useful, unsafe, rollback)
            totals[arm]["verifier_runtime_ms"] += float(
                p2a.mapping(result.get("verification")).get("runtime_ms") or 0.0
            )
            inventory_recomputed = (not result) or result.get("candidate_inventory_recomputed") == 1
            task_integrity_green = task_integrity_green and inventory_recomputed
            parse_faults = p2a.strings(attempt.get("parse_faults"))
            if arm == p4.SEMANTIC:
                treatment_fault_counts.update(parse_faults)
                if calls:
                    final_runtime = p2a.read_json(
                        p2a.resolve(str(calls[-1].get("report_path") or ""))
                    )
                    final_text = str(final_runtime.get("assistant_text") or "").strip()
                    exact_treatment_headers += int(
                        final_text.startswith("THESEUS_SEMANTIC_IR_V2\n")
                    )
                    treatment_terminal_end += int(final_text.endswith("\nEND"))
            per_arm[arm] = {
                "parseable": parseable,
                "parse_faults": sorted(parse_faults),
                "correctness_evaluated": evaluated,
                "useful": useful,
                "unsafe": unsafe,
                "rollback_verified": rollback,
            }

        candidate_integrity_tasks += int(task_integrity_green)
        if not task_integrity_green:
            faults.append(f"candidate_integrity_recomputation_invalid:{stem}")

        static = p2a.mapping(run.get("deterministic_compiler_control"))
        static_result = p2a.mapping(scored.get(p4.STATIC))
        static_parseable = static.get("parseable_candidate") is True
        add_outcome(
            totals[p4.STATIC], static_parseable,
            int(static_result.get("correctness_evaluated") or 0),
            int(static_result.get("useful") or 0),
            int(static_result.get("unsafe") or 0),
            int(static_result.get("rollback_verified") or 0),
        )
        totals[p4.STATIC]["verifier_runtime_ms"] += float(
            p2a.mapping(static_result.get("verification")).get("runtime_ms") or 0.0
        )
        oracle = p2a.mapping(scored.get(ORACLE))
        if not oracle:
            faults.append(f"oracle_score_missing:{stem}")
        add_outcome(
            oracle_totals, bool(oracle),
            int(oracle.get("correctness_evaluated") or 0),
            int(oracle.get("useful") or 0),
            int(oracle.get("unsafe") or 0),
            int(oracle.get("rollback_verified") or 0),
        )
        task_rows.append({
            "campaign_index": expected,
            "stem": stem,
            "repository": pool_row.get("repository"),
            "run": source_identity(paths["run"]),
            "evaluation": source_identity(paths["evaluation"]),
            "independent_evaluation_replay": {
                "matched": replay_match,
                "stored_projection_sha256": p2a.stable_hash(stored_projection),
                "replayed_projection_sha256": p2a.stable_hash(
                    replayed_projection
                ),
            },
            "arms": per_arm,
            "static_compiler": {
                "parseable": static_parseable,
                "abstained": static.get("abstained"),
                "compiler_faults": p2a.strings(static.get("compiler_faults")),
                "useful": int(static_result.get("useful") or 0),
            },
            "oracle_useful": int(oracle.get("useful") or 0),
        })

    learned_calls = sum(totals[arm]["model_calls"] for arm in LEARNED_ARMS)
    ceiling_hits = sum(
        int(row.get("safety_ceiling_hit") is True) for row in runtime_receipts
    )
    oracle_green = oracle_totals["useful_candidates"] == 10
    five_corruptions_green = all(
        row.get("four_base_corruptions_rejected") is True
        and p2a.mapping(row.get("dependency_corruption")).get("rejected") is True
        for row in tasks
    )
    semantic_parseable = totals[p4.SEMANTIC]["parseable_candidates"]
    termination_green = (
        len(runtime_receipts) == 60
        and sum(termination_counts.values()) == 60
        and ceiling_hits == 0
        and not set(termination_counts).difference(campaign.NORMAL_TERMINATIONS)
    )
    mechanics_floor = (
        oracle_green
        and five_corruptions_green
        and semantic_parseable >= 8
    )
    information_flow_green = (
        route_blind_tasks == 10
        and candidate_integrity_tasks == 10
        and independent_evaluator_replay_tasks == 10
    )
    experiment_floor = (
        len(task_rows) == 10
        and termination_green
        and information_flow_green
        and pool_audit["passed"]
        and learned_calls == 60
        and campaign_audit.get("trigger_state") == "GREEN"
    )

    treatment_useful = totals[p4.SEMANTIC]["useful_candidates"]
    comparator_useful = {
        arm: totals[arm]["useful_candidates"] for arm in COMPARATORS
    }
    paired = {
        arm: paired_summary(task_rows, p4.SEMANTIC, arm) for arm in COMPARATORS
    }
    dominance = {
        arm: cost_dominance(
            treatment_useful,
            totals[p4.SEMANTIC],
            comparator_useful[arm],
            totals[arm],
        )
        for arm in COMPARATORS
    }
    effect_rule = (
        treatment_useful >= 4
        and all(treatment_useful >= value + 3 for value in comparator_useful.values())
        and all(paired[arm]["treatment_losses"] <= 1 for arm in (p4.DIRECT, p4.PLAN))
        and totals[p4.SEMANTIC]["unsafe_candidates"] == 0
        and totals[p4.SEMANTIC]["rollback_failures"] == 0
        and not any(row["comparator_dominates_treatment"] for row in dominance.values())
    )
    scientific_status = classify_status(
        information_flow_green=information_flow_green,
        boundary_hits=ceiling_hits,
        mechanics_floor=mechanics_floor,
        experiment_floor=experiment_floor,
        survivor_rule=effect_rule,
    )
    if faults and scientific_status not in {
        "INVALID_INFORMATION_FLOW",
        "INSTRUMENT_INADEQUATE_GENERATION_BOUNDARY_HIT",
    }:
        scientific_status = "P4S_REVIEW_REQUIRED"

    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "scientific_status": scientific_status,
        "claim_id": "cognitive-compilation-and-semantic-ir.core",
        "scope": (
            "Exact frozen Qwen3.5 model, completion-based P4S labeled Semantic-IR "
            "instrument, and ten licensed decision-development tasks. No D1, D2, "
            "serving, training, hosted-model, or automatic book-support authority."
        ),
        "source_identities": {
            "pool": source_identity(POOL),
            "pool_seal_commit": campaign.POOL_SEAL_COMMIT,
            "instrument": source_identity(INSTRUMENT),
            "attempt2_instrument_rebind_commit": campaign.ATTEMPT2_INSTRUMENT_REBIND_COMMIT,
            "campaign_progress": source_identity(PROGRESS),
            "invalid_attempt1": source_identity(campaign.INVALID_ATTEMPT),
            "runtime_receipts": runtime_receipts,
            "source_pool_audit": pool_audit,
        },
        "denominators": {
            "tasks": len(task_rows),
            "learned_arms": len(task_rows) * len(LEARNED_ARMS),
            "learned_model_calls": learned_calls,
            "persistent_model_loads": len(task_rows),
            "static_compiler_controls": len(task_rows),
            "oracle_mechanics_controls": len(task_rows),
            "hosted_model_calls": 0,
            "project_selected_quality_token_cap": None,
            "model_context_window_tokens": p4s.MODEL_CONTEXT_TOKENS,
        },
        "arm_totals": totals,
        "oracle_totals": oracle_totals,
        "termination_custody": {
            "normal_termination_calls": sum(termination_counts.values()),
            "termination_reason_counts": dict(sorted(termination_counts.items())),
            "safety_ceiling_hits": ceiling_hits,
            "minimum_generated_tokens": min(generated_tokens, default=0),
            "maximum_generated_tokens": max(generated_tokens, default=0),
            "total_generated_tokens": sum(generated_tokens),
            "maximum_prompt_tokens": max(prompt_tokens, default=0),
            "completion_rule": (
                "Parser-complete artifact or model EOS; the exact prompt residual of "
                "the declared model context is the sole numeric boundary. A boundary "
                "hit invalidates the observation and is never a mechanism failure."
            ),
        },
        "adequacy": {
            "compiler_oracle_useful": f"{oracle_totals['useful_candidates']}/10",
            "all_five_corruption_classes_rejected": f"{sum(row.get('four_base_corruptions_rejected') is True and p2a.mapping(row.get('dependency_corruption')).get('rejected') is True for row in tasks)}/10",
            "semantic_ir_parse_and_lower": f"{semantic_parseable}/10",
            "semantic_ir_required_floor": "8/10",
            "natural_termination_receipts": f"{sum(termination_counts.values())}/60",
            "route_blind_tasks": f"{route_blind_tasks}/10",
            "candidate_integrity_recomputed_tasks": f"{candidate_integrity_tasks}/10",
            "independent_evaluator_replay_tasks": f"{independent_evaluator_replay_tasks}/10",
            "independent_evaluator_replay_contract": {
                "excluded_as_volatile": [
                    "created_utc",
                    "runtime_ms",
                    "sandbox_temporary_root_names_in_tracebacks",
                    "evaluator_only_oracle_digest_that_includes_visible_verifier_runtime",
                ],
                "learned_candidate_digests_excluded": False,
                "static_control_candidate_digests_excluded": False,
                "correctness_safety_rollback_effect_and_verifier_fields_excluded": False,
                "oracle_digest_repair_owner": (
                    "fresh_D1_evaluator_successor_after_P4S_seals"
                ),
            },
            "mechanics_floor_passed": mechanics_floor,
            "experiment_floor_passed": experiment_floor,
            "information_flow_green": information_flow_green,
            "treatment_protocol_signature": {
                "exact_headers": f"{exact_treatment_headers}/10",
                "terminal_END": f"{treatment_terminal_end}/10",
                "parse_fault_counts": dict(sorted(treatment_fault_counts.items())),
            },
        },
        "decision_rule": {
            "predeclared": p2a.mapping(instrument.get("decision_rule")),
            "treatment_useful": treatment_useful,
            "comparator_useful": comparator_useful,
            "paired_uncertainty": paired,
            "joined_cost_dominance": dominance,
            "survivor_effect_rule_passed": effect_rule,
            "effect_decision_authorized": mechanics_floor and experiment_floor,
        },
        "weak_tail": {
            "treatment_failures": len(task_rows) - treatment_useful,
            "treatment_pairwise_losses": {
                arm: paired[arm]["treatment_losses"] for arm in COMPARATORS
            },
            "all_task_failures_retained": len(task_rows) == 10,
        },
        "task_results": task_rows,
        "consumption": {
            "all_ten_tasks_consumed": len(task_rows) == 10,
            "eligible_for_exact_rerun": False if len(task_rows) == 10 else None,
            "eligible_for_training": False,
            "eligible_for_D1": scientific_status == "P4S_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE",
            "eligible_for_D2": False,
            "all_failures_retained": len(task_rows) == 10,
        },
        "next_stage": next_stage(scientific_status),
        "maximum_inference": (
            "This report can decide only whether this exact mechanics-qualified "
            "labeled Semantic-IR implementation with the frozen local model survives "
            "this ten-task decision-development surface and may be qualified once on "
            "fresh source-disjoint D1. It cannot establish or falsify cognitive "
            "compilation generally, qualify serving or training, decide D2, or promote "
            "ASI Stack book support."
        ),
        "counters": p2a.zero_counters(),
    }


def audit_source_pool(pool: dict[str, Any]) -> dict[str, Any]:
    """Independently recompute the source/licensing/disjointness pool floor."""

    faults: list[str] = []
    if pool.get("policy") != (
        "project_theseus_p4s_cognitive_compilation_task_pool_v1"
    ):
        faults.append("source_pool_policy_invalid")
    if pool.get("state") != "SEALED_BEFORE_CANDIDATE_GENERATION":
        faults.append("source_pool_not_sealed")
    if p2a.strings(pool.get("faults")):
        faults.append("source_pool_declared_faults_present")

    registry_path = p2a.resolve(str(pool.get("source_registry") or ""))
    fetch_path = p2a.resolve(str(pool.get("source_fetch_report") or ""))
    if (
        not registry_path.is_file()
        or p2a.sha256_file(registry_path)
        != str(pool.get("source_registry_sha256") or "")
    ):
        faults.append("source_registry_binding_invalid")
    if (
        not fetch_path.is_file()
        or p2a.sha256_file(fetch_path)
        != str(pool.get("source_fetch_report_sha256") or "")
    ):
        faults.append("source_fetch_binding_invalid")
    fetch = p2a.read_json(fetch_path) if fetch_path.is_file() else {}
    if (
        fetch.get("trigger_state") != "GREEN"
        or p2a.strings(fetch.get("faults"))
        or int(fetch.get("candidate_or_control_calls") or 0) != 0
    ):
        faults.append("source_fetch_custody_invalid")

    tasks = p2a.dicts(pool.get("tasks"))
    repositories = [str(row.get("repository") or "").lower() for row in tasks]
    licenses = [str(row.get("license_spdx") or "") for row in tasks]
    revisions = [
        (
            str(row.get("repository") or "").lower(),
            str(row.get("parent_revision") or ""),
            str(row.get("target_revision") or ""),
        )
        for row in tasks
    ]
    prior_repositories = {
        str(value).lower()
        for value in p2a.strings(
            p2a.mapping(pool.get("source_disjoint_from")).get(
                "P2_P3_P4_P4R"
            )
        )
    }
    if len(tasks) != 10 or int(pool.get("task_count") or 0) != 10:
        faults.append("source_pool_task_count_invalid")
    if len(set(repositories)) != 10 or int(
        pool.get("distinct_repositories") or 0
    ) != 10:
        faults.append("source_pool_repositories_not_distinct")
    if prior_repositories.intersection(repositories):
        faults.append("source_pool_predecessor_repository_overlap")
    if any(not license_id for license_id in licenses):
        faults.append("source_pool_license_missing")
    if any(
        not repository
        or not parent
        or not target
        or parent == target
        for repository, parent, target in revisions
    ):
        faults.append("source_pool_revision_identity_invalid")
    if p2a.mapping(pool.get("source_disjoint_from")).get("training") != (
        "all_P4S_tasks_permanently_excluded"
    ):
        faults.append("source_pool_training_exclusion_missing")
    return {
        "passed": not faults,
        "faults": sorted(set(faults)),
        "task_count": len(tasks),
        "distinct_repository_count": len(set(repositories)),
        "license_spdx_ids": sorted(set(licenses)),
        "predecessor_repository_overlap": sorted(
            prior_repositories.intersection(repositories)
        ),
        "source_registry": source_identity(registry_path),
        "source_fetch_report": source_identity(fetch_path),
    }


def stable_evaluation_projection(value: Any) -> Any:
    """Remove only timing/timestamp fields before exact replay comparison."""

    projected = _stable_evaluation_value(value)
    if not isinstance(projected, dict):
        return projected
    results = p2a.dicts(projected.get("results"))
    digest_to_arm = {
        str(row.get("candidate_output_sha256") or ""): str(
            row.get("arm_id") or ""
        )
        for row in results
        if str(row.get("candidate_output_sha256") or "")
    }
    for row in results:
        if row.get("arm_id") == ORACLE:
            row["candidate_output_sha256"] = (
                "INDEPENDENTLY_REBUILT_EVALUATOR_ONLY_ORACLE"
            )
    results.sort(key=lambda row: str(row.get("arm_id") or ""))
    projected["results"] = results
    blinding = p2a.mapping(projected.get("evaluation_blinding"))
    scoring_order = p2a.strings(blinding.pop("scoring_order", []))
    blinding["scored_arm_multiset"] = sorted(
        digest_to_arm.get(digest, "") for digest in scoring_order
    )
    projected["evaluation_blinding"] = blinding
    return projected


def _stable_evaluation_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _stable_evaluation_value(item)
            for key, item in sorted(value.items())
            if key not in {"created_utc", "runtime_ms"}
        }
    if isinstance(value, list):
        return [_stable_evaluation_value(item) for item in value]
    if isinstance(value, str):
        return re.sub(
            r"/(?:private/)?var/folders/(?:[^/]+/){2}T/theseus-p4-[^/]+",
            "<EVALUATOR_TEMP_ROOT>",
            value,
        )
    return value


def empty_totals() -> dict[str, int | float]:
    return {
        "tasks": 0,
        "model_calls": 0,
        "parseable_candidates": 0,
        "malformed_candidates": 0,
        "correctness_evaluated_candidates": 0,
        "useful_candidates": 0,
        "incorrect_candidates": 0,
        "unsafe_candidates": 0,
        "rollback_verified_candidates": 0,
        "rollback_failures": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "model_runtime_ms": 0.0,
        "verifier_runtime_ms": 0.0,
    }


def add_outcome(
    totals: dict[str, int | float], parseable: bool, evaluated: int,
    useful: int, unsafe: int, rollback: int,
) -> None:
    totals["tasks"] += 1
    totals["parseable_candidates"] += int(parseable)
    totals["malformed_candidates"] += int(not parseable)
    totals["correctness_evaluated_candidates"] += evaluated
    totals["useful_candidates"] += useful
    totals["incorrect_candidates"] += int(bool(evaluated and not useful))
    totals["unsafe_candidates"] += unsafe
    totals["rollback_verified_candidates"] += rollback
    totals["rollback_failures"] += int(bool(evaluated and not rollback))


def paired_summary(
    task_rows: list[dict[str, Any]], treatment: str, comparator: str,
) -> dict[str, Any]:
    deltas: list[int] = []
    for row in task_rows:
        treatment_value = int(
            p2a.mapping(p2a.mapping(row.get("arms")).get(treatment)).get("useful") or 0
        )
        if comparator == p4.STATIC:
            comparator_value = int(
                p2a.mapping(row.get("static_compiler")).get("useful") or 0
            )
        else:
            comparator_value = int(
                p2a.mapping(p2a.mapping(row.get("arms")).get(comparator)).get("useful") or 0
            )
        deltas.append(treatment_value - comparator_value)
    wins = sum(value > 0 for value in deltas)
    losses = sum(value < 0 for value in deltas)
    ties = sum(value == 0 for value in deltas)
    non_ties = wins + losses
    return {
        "treatment_wins": wins,
        "treatment_losses": losses,
        "ties": ties,
        "non_ties": non_ties,
        "exact_one_sided_sign_probability": exact_sign_probability(wins, non_ties),
        "mean_useful_delta": round(sum(deltas) / len(deltas), 6) if deltas else 0.0,
        "task_bootstrap_95_interval": bootstrap_interval(
            deltas, seed_material=f"p4s:{treatment}:{comparator}"
        ),
    }


def exact_sign_probability(wins: int, non_ties: int) -> float:
    if non_ties == 0:
        return 1.0
    numerator = sum(math.comb(non_ties, value) for value in range(wins, non_ties + 1))
    return round(numerator / (2 ** non_ties), 8)


def bootstrap_interval(
    deltas: list[int], *, seed_material: str, samples: int = 20000,
) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    count = len(deltas)
    means = sorted(
        sum(deltas[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    lower = means[int(0.025 * (samples - 1))]
    upper = means[int(0.975 * (samples - 1))]
    return [round(lower, 6), round(upper, 6)]


def cost_dominance(
    treatment_useful: int,
    treatment: dict[str, int | float],
    comparator_useful: int,
    comparator: dict[str, int | float],
) -> dict[str, Any]:
    no_more_costly = all(
        float(comparator.get(key, 0)) <= float(treatment.get(key, 0))
        for key in COST_DIMENSIONS
    )
    strictly_cheaper = any(
        float(comparator.get(key, 0)) < float(treatment.get(key, 0))
        for key in COST_DIMENSIONS
    )
    dominates = (
        comparator_useful >= treatment_useful and no_more_costly and strictly_cheaper
    )
    return {
        "comparator_useful_not_worse": comparator_useful >= treatment_useful,
        "comparator_no_more_costly_all_dimensions": no_more_costly,
        "comparator_strictly_cheaper_any_dimension": strictly_cheaper,
        "comparator_dominates_treatment": dominates,
        "dimensions": {
            key: {
                "treatment": round(float(treatment.get(key, 0)), 3),
                "comparator": round(float(comparator.get(key, 0)), 3),
            }
            for key in COST_DIMENSIONS
        },
    }


def classify_status(
    *, information_flow_green: bool, boundary_hits: int,
    mechanics_floor: bool, experiment_floor: bool, survivor_rule: bool,
) -> str:
    if not information_flow_green:
        return "INVALID_INFORMATION_FLOW"
    if boundary_hits:
        return "INSTRUMENT_INADEQUATE_GENERATION_BOUNDARY_HIT"
    if not mechanics_floor:
        return "INCONCLUSIVE_IMPLEMENTATION"
    if not experiment_floor:
        return "INCONCLUSIVE_EXPERIMENT"
    if survivor_rule:
        return "P4S_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE"
    return "P4S_ADEQUATE_NO_SURVIVOR"


def next_stage(status: str) -> dict[str, Any]:
    if status == "P4S_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE":
        return {
            "state": "OPEN_ONE_FRESH_SOURCE_DISJOINT_D1_QUALIFICATION",
            "D1_eligible": True,
            "book_support_state_effect": "none",
        }
    return {
        "state": "D1_CLOSED_RETAIN_SCOPED_P4S_EVIDENCE",
        "D1_eligible": False,
        "book_support_state_effect": "none",
    }


def source_identity(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}


if __name__ == "__main__":
    raise SystemExit(main())
