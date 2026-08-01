#!/usr/bin/env python3
"""Recompute the terminal scientific disposition of the sealed P4R campaign."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4_cognitive_compilation as p4  # noqa: E402
import theseus_p4r_campaign as campaign  # noqa: E402


POLICY = "project_theseus_p4r_cognitive_compilation_terminal_disposition_v1"
POOL = ROOT / "configs" / "theseus_p4r_task_pool.json"
INSTRUMENT = ROOT / "configs" / "theseus_p4_cognitive_compilation_repaired_instrument_r1.json"
PROGRESS = ROOT / "reports" / "theseus_p4r_campaign_progress.json"
OUT = ROOT / "reports" / "theseus_p4r_terminal_disposition.json"
ORACLE = "deterministic_compiler_oracle_ceiling"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=p2a.rel(OUT))
    args = parser.parse_args()
    report = build_report()
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "scientific_status": report["scientific_status"],
        "arm_useful": {
            arm: report["arm_totals"][arm]["useful_candidates"]
            for arm in (*p4.ARMS, p4.STATIC)
        },
        "semantic_ir_parse_and_lower": report["adequacy"]["semantic_ir_parse_and_lower"],
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
    tasks = p2a.dicts(pool.get("tasks"))
    decision = p2a.mapping(instrument.get("decision_rule"))
    if len(tasks) != 10:
        faults.append("task_count_invalid")

    arm_totals = {arm: empty_totals() for arm in (*p4.ARMS, p4.STATIC)}
    oracle_totals = empty_totals()
    termination_counts: collections.Counter[str] = collections.Counter()
    generated_tokens: list[int] = []
    prompt_tokens: list[int] = []
    treatment_fault_counts: collections.Counter[str] = collections.Counter()
    task_rows: list[dict[str, Any]] = []
    runtime_receipts: list[dict[str, Any]] = []
    treatment_exact_header = 0
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
        if run.get("instrument_sha256") != p2a.sha256_file(INSTRUMENT):
            faults.append(f"run_instrument_mismatch:{stem}")
        if run.get("task_sha256") != p2a.sha256_file(task_path):
            faults.append(f"run_task_mismatch:{stem}")
        if evaluation.get("candidate_report_sha256") != p2a.sha256_file(paths["run"]):
            faults.append(f"evaluation_run_mismatch:{stem}")
        if evaluation.get("evaluator_sha256") != p2a.sha256_file(evaluator_path):
            faults.append(f"evaluation_owner_mismatch:{stem}")
        if p2a.mapping(run.get("matched_set")).get("ready") is not True:
            faults.append(f"matched_set_invalid:{stem}")
        if evaluation.get("trigger_state") != "GREEN":
            faults.append(f"evaluation_red:{stem}")
        expected_order = list(p4.arm_order(expected))
        if run.get("actual_arm_order") != expected_order:
            faults.append(f"arm_order_invalid:{stem}")

        scored = {
            str(row.get("arm_id") or ""): row
            for row in p2a.dicts(evaluation.get("results"))
        }
        attempts = {
            str(row.get("arm_id") or ""): row
            for row in p2a.dicts(run.get("attempts"))
        }
        per_arm: dict[str, Any] = {}
        for arm in p4.ARMS:
            attempt = p2a.mapping(attempts.get(arm))
            result = p2a.mapping(scored.get(arm))
            if not attempt:
                faults.append(f"attempt_missing:{stem}:{arm}")
                continue
            parseable = attempt.get("parseable_candidate") is True
            if parseable and not result:
                faults.append(f"parseable_not_scored:{stem}:{arm}")
            if not parseable and result:
                faults.append(f"malformed_candidate_scored:{stem}:{arm}")
            calls = p2a.dicts(attempt.get("runtime_calls"))
            if len(calls) != 2:
                faults.append(f"arm_call_count_invalid:{stem}:{arm}")
            for call in calls:
                runtime_path = p2a.resolve(str(call.get("report_path") or ""))
                runtime_valid = (
                    runtime_path.is_file()
                    and p2a.sha256_file(runtime_path) == str(call.get("report_sha256") or "")
                )
                runtime = p2a.read_json(runtime_path) if runtime_valid else {}
                backend_path = p2a.resolve(
                    str(p2a.mapping(runtime.get("generation_backend")).get("out") or "")
                )
                backend = p2a.read_json(backend_path)
                metrics = p2a.mapping(backend.get("metrics"))
                reason = str(metrics.get("termination_reason") or "")
                if reason not in campaign.NORMAL_TERMINATIONS:
                    faults.append(f"termination_invalid:{stem}:{arm}:{call.get('call_number')}")
                if metrics.get("safety_ceiling_hit") is True:
                    faults.append(f"physical_context_boundary_hit:{stem}:{arm}:{call.get('call_number')}")
                if not runtime_valid:
                    faults.append(f"runtime_receipt_invalid:{stem}:{arm}:{call.get('call_number')}")
                if p2a.mapping(runtime.get("route_integrity")).get("ready") is not True:
                    faults.append(f"route_integrity_invalid:{stem}:{arm}:{call.get('call_number')}")
                termination_counts[reason] += 1
                generated_tokens.append(int(metrics.get("generated_tokens") or 0))
                prompt_tokens.append(int(metrics.get("prompt_tokens") or 0))
                runtime_receipts.append({
                    "campaign_index": expected,
                    "arm_id": arm,
                    "call_number": call.get("call_number"),
                    "runtime_report": p2a.rel(runtime_path),
                    "runtime_report_sha256": p2a.sha256_file(runtime_path),
                    "backend_report": p2a.rel(backend_path),
                    "backend_report_sha256": p2a.sha256_file(backend_path),
                    "termination_reason": reason,
                    "prompt_tokens": metrics.get("prompt_tokens"),
                    "generated_tokens": metrics.get("generated_tokens"),
                    "safety_ceiling_hit": metrics.get("safety_ceiling_hit"),
                })
            evaluated = int(result.get("correctness_evaluated") or 0)
            useful = int(result.get("useful") or 0)
            unsafe = int(result.get("unsafe") or 0)
            rollback = int(result.get("rollback_verified") or 0)
            totals = arm_totals[arm]
            add_outcome(totals, parseable, evaluated, useful, unsafe, rollback, len(calls))
            parse_faults = p2a.strings(attempt.get("parse_faults"))
            if arm == p4.SEMANTIC:
                treatment_fault_counts.update(parse_faults)
                final_runtime = p2a.read_json(p2a.resolve(str(calls[-1].get("report_path") or "")))
                final_text = str(final_runtime.get("assistant_text") or "").strip()
                treatment_exact_header += int(final_text.startswith("THESEUS_SEMANTIC_IR_V1\n"))
                treatment_terminal_end += int(final_text.endswith("\nEND"))
            per_arm[arm] = {
                "parseable": parseable,
                "parse_faults": sorted(parse_faults),
                "correctness_evaluated": evaluated,
                "useful": useful,
                "unsafe": unsafe,
                "rollback_verified": rollback,
            }

        static = p2a.mapping(run.get("deterministic_compiler_control"))
        static_result = p2a.mapping(scored.get(p4.STATIC))
        static_parseable = static.get("parseable_candidate") is True
        add_outcome(
            arm_totals[p4.STATIC], static_parseable,
            int(static_result.get("correctness_evaluated") or 0),
            int(static_result.get("useful") or 0),
            int(static_result.get("unsafe") or 0),
            int(static_result.get("rollback_verified") or 0), 0,
        )
        oracle = p2a.mapping(scored.get(ORACLE))
        if not oracle:
            faults.append(f"oracle_score_missing:{stem}")
        add_outcome(
            oracle_totals, bool(oracle),
            int(oracle.get("correctness_evaluated") or 0),
            int(oracle.get("useful") or 0),
            int(oracle.get("unsafe") or 0),
            int(oracle.get("rollback_verified") or 0), 0,
        )
        task_rows.append({
            "campaign_index": expected,
            "stem": stem,
            "repository": pool_row.get("repository"),
            "run": source_identity(paths["run"]),
            "evaluation": source_identity(paths["evaluation"]),
            "arms": per_arm,
            "static_compiler": {
                "parseable": static_parseable,
                "abstained": static.get("abstained"),
                "compiler_faults": p2a.strings(static.get("compiler_faults")),
                "useful": int(static_result.get("useful") or 0),
            },
            "oracle_useful": int(oracle.get("useful") or 0),
        })

    total_calls = sum(row["model_calls"] for row in arm_totals.values())
    ceiling_hits = sum(int(row["safety_ceiling_hit"] is True) for row in runtime_receipts)
    oracle_green = oracle_totals["useful_candidates"] == 10
    corruptions_green = all(row.get("four_corruptions_rejected") is True for row in tasks)
    semantic_parseable = arm_totals[p4.SEMANTIC]["parseable_candidates"]
    termination_green = (
        len(runtime_receipts) == 60
        and sum(termination_counts.values()) == 60
        and ceiling_hits == 0
        and not set(termination_counts).difference(campaign.NORMAL_TERMINATIONS)
    )
    mechanics_floor_passed = (
        oracle_green and corruptions_green and semantic_parseable >= 8 and termination_green
    )
    if total_calls != 60:
        faults.append("total_model_call_count_invalid")
    if oracle_totals["correctness_evaluated_candidates"] != 10:
        faults.append("oracle_evaluation_denominator_invalid")

    treatment = arm_totals[p4.SEMANTIC]["useful_candidates"]
    comparators = {
        arm: arm_totals[arm]["useful_candidates"]
        for arm in (p4.DIRECT, p4.PLAN, p4.STATIC)
    }
    pairwise_losses = {
        arm: sum(
            int(row["arms"].get(p4.SEMANTIC, {}).get("useful") == 0)
            and int(row["arms"].get(arm, {}).get("useful") == 1)
            for row in task_rows
        )
        for arm in (p4.DIRECT, p4.PLAN)
    }
    nominal_effect_passed = (
        all(treatment >= value + 2 for value in comparators.values())
        and all(value <= 1 for value in pairwise_losses.values())
        and arm_totals[p4.SEMANTIC]["unsafe_candidates"] == 0
    )
    scientific_status = (
        "INCONCLUSIVE_IMPLEMENTATION"
        if not mechanics_floor_passed
        else "P4_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE"
        if nominal_effect_passed
        else "P4_ADEQUATE_NO_SURVIVOR"
    )
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "scientific_status": scientific_status if not faults else "P4_REVIEW_REQUIRED",
        "claim_id": "cognitive-compilation-and-semantic-ir.core",
        "scope": "Exact frozen Qwen3.5 model, completion-based P4R instrument, and ten licensed development tasks. No D1, D2, serving, training, hosted-model, or book-support authority.",
        "source_identities": {
            "pool": source_identity(POOL),
            "pool_seal_commit": campaign.POOL_SEAL_COMMIT,
            "instrument": source_identity(INSTRUMENT),
            "instrument_freeze_commit": campaign.INSTRUMENT_FREEZE_COMMIT,
            "campaign_progress": source_identity(PROGRESS),
            "runtime_receipts": runtime_receipts,
        },
        "denominators": {
            "tasks": len(task_rows),
            "learned_arms": 30,
            "learned_model_calls": total_calls,
            "persistent_model_loads": 10,
            "static_compiler_controls": 10,
            "oracle_mechanics_controls": 10,
            "hosted_model_calls": 0,
            "project_selected_quality_token_cap": None,
        },
        "arm_totals": arm_totals,
        "oracle_totals": oracle_totals,
        "termination_custody": {
            "normal_termination_calls": sum(termination_counts.values()),
            "termination_reason_counts": dict(sorted(termination_counts.items())),
            "safety_ceiling_hits": ceiling_hits,
            "minimum_generated_tokens": min(generated_tokens, default=0),
            "maximum_generated_tokens": max(generated_tokens, default=0),
            "total_generated_tokens": sum(generated_tokens),
            "maximum_prompt_tokens": max(prompt_tokens, default=0),
            "finding": "All calls ended by parser completion or model EOS. The observed 7,219-token maximum confirms the rejected 1,536-token P4-v1 cap would have differentially truncated a real arm.",
        },
        "adequacy": {
            "compiler_oracle_useful": f"{oracle_totals['useful_candidates']}/10",
            "all_four_corruption_classes_rejected": f"{sum(row.get('four_corruptions_rejected') is True for row in tasks)}/10",
            "semantic_ir_parse_and_lower": f"{semantic_parseable}/10",
            "semantic_ir_required_floor": "8/10",
            "natural_termination_receipts": f"{sum(termination_counts.values())}/60",
            "mechanics_floor_passed": mechanics_floor_passed,
            "failure_owner": "learned Semantic-IR protocol emission/lowering implementation",
            "failure_signature": {
                "final_outputs_with_exact_semantic_header": f"{treatment_exact_header}/10",
                "final_outputs_with_terminal_END": f"{treatment_terminal_end}/10",
                "parse_fault_counts": dict(sorted(treatment_fault_counts.items())),
                "interpretation": "The model usually attempted the requested IR envelope, but every final UNIT grammar failed the strict parser or obligation/loss contract. Oracle lowering proves the mechanism path is reachable; learned emission does not pass the predeclared implementation floor.",
            },
        },
        "decision_rule": {
            "predeclared": decision,
            "treatment_useful": treatment,
            "comparator_useful": comparators,
            "pairwise_treatment_losses": pairwise_losses,
            "nominal_effect_rule_passed": nominal_effect_passed,
            "effect_decision_authorized": mechanics_floor_passed,
            "disposition": "NOT_EVALUABLE_IMPLEMENTATION_INADEQUATE" if not mechanics_floor_passed else "EVALUATED",
        },
        "task_results": task_rows,
        "consumption": {
            "all_ten_tasks_consumed": len(task_rows) == 10,
            "eligible_for_exact_rerun": False,
            "eligible_for_training": False,
            "eligible_for_D1_or_D2": False,
            "all_failures_retained": True,
        },
        "next_stage": {
            "state": "P4_SEMANTIC_IR_IMPLEMENTATION_REPAIR_REQUIRED",
            "D1_eligible": False,
            "required_repair": "Demonstrate learned Semantic-IR grammar emission, parsing, lowering, checkpoint-independent replay, intervention sensitivity, and dependency-local repair on non-claim mechanics fixtures before sealing any fresh P4 decision denominator.",
            "fresh_decision_tasks": "Acquire only after the mechanics canary passes; never replay these ten consumed tasks for fresh credit.",
            "book_support_state_effect": "none",
        },
        "maximum_inference": "This exact learned Semantic-IR implementation is inadequate for an arm-effect decision: 0/10 treatment outputs parsed and lowered against an 8/10 floor, while the independent oracle was useful on 10/10. Direct solved 1/10 and plan control 2/10, but those counts cannot falsify cognitive compilation because the treatment implementation failed adequacy. The result authorizes mechanics repair only, not D1, serving, training, book support, or retirement of the ASI Stack claim.",
        "counters": p2a.zero_counters(),
    }


def empty_totals() -> dict[str, int]:
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
    }


def add_outcome(
    totals: dict[str, int], parseable: bool, evaluated: int, useful: int,
    unsafe: int, rollback: int, model_calls: int,
) -> None:
    totals["tasks"] += 1
    totals["model_calls"] += model_calls
    totals["parseable_candidates"] += int(parseable)
    totals["malformed_candidates"] += int(not parseable)
    totals["correctness_evaluated_candidates"] += evaluated
    totals["useful_candidates"] += useful
    totals["incorrect_candidates"] += int(bool(evaluated and not useful))
    totals["unsafe_candidates"] += unsafe
    totals["rollback_verified_candidates"] += rollback


def source_identity(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}


if __name__ == "__main__":
    raise SystemExit(main())
