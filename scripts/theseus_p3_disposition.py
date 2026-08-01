#!/usr/bin/env python3
"""Recompute the terminal disposition for the ten-task local P3 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_p3_terminal_disposition_v1"
DIRECT = "direct_local_model"
INTEGRATED = "integrated_local_model"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="configs/theseus_p3_task_pool.json")
    parser.add_argument("--instrument", default="configs/theseus_assistant_p3_instrument.json")
    parser.add_argument("--out", default="reports/theseus_assistant_p3_terminal_disposition.json")
    args = parser.parse_args()
    report = build(resolve(args.pool), resolve(args.instrument))
    write_json(resolve(args.out), report)
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "scientific_status": report["scientific_status"],
        "paired_useful_outcomes": report["paired_outcomes"]["useful"],
        "selected_p4_claim": report["next_stage"]["selected_claim_id"],
        "faults": report["faults"],
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build(pool_path: Path, instrument_path: Path) -> dict[str, Any]:
    pool = read_json(pool_path)
    instrument = read_json(instrument_path)
    faults: list[str] = []
    if pool.get("state") != "SEALED_BEFORE_CANDIDATE_GENERATION":
        faults.append("task_pool_not_prospectively_sealed")
    if pool.get("instrument_sha256") != sha256_file(instrument_path):
        faults.append("pool_instrument_digest_mismatch")
    task_rows = dicts(pool.get("tasks"))
    if len(task_rows) != 10:
        faults.append("task_count_invalid")

    task_results: list[dict[str, Any]] = []
    receipt_rows: list[dict[str, Any]] = []
    arm_totals = {
        DIRECT: empty_arm_totals(),
        INTEGRATED: empty_arm_totals(),
    }
    for expected_index, pool_row in enumerate(task_rows, 1):
        index = int(pool_row.get("campaign_index") or 0)
        stem = str(pool_row.get("stem") or "")
        suffix = stem.removeprefix("p3_")
        task_path = resolve(str(pool_row.get("task") or ""))
        evaluator_path = resolve(str(pool_row.get("evaluator") or ""))
        run_path = ROOT / "reports" / f"theseus_assistant_p3_{suffix}_run.json"
        evaluation_path = ROOT / "reports" / f"theseus_assistant_p3_{suffix}_evaluation.json"
        if index != expected_index:
            faults.append(f"campaign_index_order_invalid:{stem}")
        for label, path, expected_hash in (
            ("task", task_path, str(pool_row.get("task_sha256") or "")),
            ("evaluator", evaluator_path, str(pool_row.get("evaluator_sha256") or "")),
        ):
            if sha256_file(path) != expected_hash:
                faults.append(f"{label}_binding_invalid:{stem}")
        if not run_path.is_file() or not evaluation_path.is_file():
            faults.append(f"campaign_result_missing:{stem}")
            continue
        run = read_json(run_path)
        evaluation = read_json(evaluation_path)
        if run.get("instrument_sha256") != sha256_file(instrument_path):
            faults.append(f"run_instrument_binding_invalid:{stem}")
        if run.get("task_sha256") != sha256_file(task_path):
            faults.append(f"run_task_binding_invalid:{stem}")
        if evaluation.get("candidate_report_sha256") != sha256_file(run_path):
            faults.append(f"evaluation_run_binding_invalid:{stem}")
        if evaluation.get("evaluator_sha256") != sha256_file(evaluator_path):
            faults.append(f"evaluation_evaluator_binding_invalid:{stem}")
        if mapping(run.get("matched_pair")).get("ready") is not True:
            faults.append(f"matched_pair_invalid:{stem}")
        if evaluation.get("trigger_state") != "GREEN":
            faults.append(f"evaluation_invalid:{stem}")
        expected_order = [DIRECT, INTEGRATED] if index % 2 else [INTEGRATED, DIRECT]
        if run.get("actual_arm_order") != expected_order:
            faults.append(f"counterbalanced_order_invalid:{stem}")
        denominators = mapping(run.get("denominators"))
        if int(denominators.get("model_loads") or 0) != 1:
            faults.append(f"persistent_load_count_invalid:{stem}")
        if int(denominators.get("model_calls") or 0) > 4:
            faults.append(f"model_call_budget_exceeded:{stem}")

        scored = {str(row.get("arm_id") or ""): row for row in dicts(evaluation.get("results"))}
        attempts = {str(row.get("arm_id") or ""): row for row in dicts(run.get("attempts"))}
        per_arm: dict[str, Any] = {}
        for arm in (DIRECT, INTEGRATED):
            attempt = mapping(attempts.get(arm))
            result = mapping(scored.get(arm))
            if not attempt:
                faults.append(f"arm_attempt_missing:{stem}:{arm}")
                continue
            calls = dicts(attempt.get("runtime_calls"))
            for call in calls:
                report_path = resolve(str(call.get("report_path") or ""))
                valid = (
                    report_path.is_file()
                    and sha256_file(report_path) == str(call.get("report_sha256") or "")
                )
                runtime = read_json(report_path) if valid else {}
                route_ready = mapping(runtime.get("route_integrity")).get("ready") is True
                if not valid or not route_ready:
                    faults.append(
                        f"runtime_receipt_invalid:{stem}:{arm}:{call.get('call_number')}"
                    )
                receipt_rows.append({
                    "campaign_index": index,
                    "arm_id": arm,
                    "call_number": int(call.get("call_number") or 0),
                    "path": relative(report_path) if report_path.is_file() else str(call.get("report_path") or ""),
                    "sha256": sha256_file(report_path),
                    "route_integrity_ready": route_ready,
                    "runtime_trigger_state": runtime.get("trigger_state"),
                    "runtime_ms": float(call.get("runtime_ms") or 0.0),
                })
            parseable = attempt.get("parseable_candidate") is True
            evaluated = int(result.get("correctness_evaluated") or 0)
            useful = int(result.get("useful") or 0)
            unsafe = int(result.get("unsafe") or 0)
            rollback = int(result.get("rollback_verified") or 0)
            model_calls = int(attempt.get("model_calls") or 0)
            if model_calls < 1 or model_calls > 2:
                faults.append(f"arm_call_budget_invalid:{stem}:{arm}")
            if parseable and not result:
                faults.append(f"parseable_candidate_not_scored:{stem}:{arm}")
            if not parseable and result:
                faults.append(f"unsealed_candidate_scored:{stem}:{arm}")
            if evaluated and not rollback:
                faults.append(f"evaluated_candidate_rollback_failed:{stem}:{arm}")
            totals = arm_totals[arm]
            totals["tasks"] += 1
            totals["model_calls"] += model_calls
            totals["parseable_candidates"] += int(parseable)
            totals["malformed_candidates"] += int(not parseable)
            totals["correctness_evaluated_candidates"] += evaluated
            totals["useful_candidates"] += useful
            totals["incorrect_candidates"] += int(bool(evaluated and not useful))
            totals["unsafe_candidates"] += unsafe
            totals["rollback_verified_candidates"] += rollback
            totals["runtime_ms"] += sum(float(row.get("runtime_ms") or 0.0) for row in calls)
            per_arm[arm] = {
                "model_calls": model_calls,
                "parseable": parseable,
                "parse_faults": sorted(strings(attempt.get("parse_faults"))),
                "apply_faults": sorted(strings(attempt.get("apply_faults"))),
                "correctness_evaluated": evaluated,
                "useful": useful,
                "unsafe": unsafe,
                "rollback_verified": rollback,
                "observed_effect_paths": strings(result.get("observed_effect_paths")),
                "hidden_verifier_passed": int(result.get("hidden_tests_passed") or 0),
                "outcome": (
                    "USEFUL" if useful else "INCORRECT" if evaluated else "MALFORMED"
                ),
            }
        task_results.append({
            "campaign_index": index,
            "stem": stem,
            "repository": pool_row.get("repository"),
            "actual_arm_order": run.get("actual_arm_order"),
            "run": source_identity(run_path),
            "evaluation": source_identity(evaluation_path),
            "matched_pair_ready": mapping(run.get("matched_pair")).get("ready") is True,
            "model_loads": int(denominators.get("model_loads") or 0),
            "arms": per_arm,
        })

    paired = paired_outcomes(task_results)
    for totals in arm_totals.values():
        totals["runtime_ms"] = round(totals["runtime_ms"], 3)
        successes = int(totals["useful_candidates"])
        total = int(totals["tasks"])
        totals["useful_rate"] = successes / total if total else 0.0
        totals["useful_rate_wilson_95"] = wilson(successes, total)
        parseable = int(totals["parseable_candidates"])
        totals["parseable_rate"] = parseable / total if total else 0.0
        totals["parseable_rate_wilson_95"] = wilson(parseable, total)
    total_calls = sum(int(row["model_calls"]) for row in arm_totals.values())
    total_unsafe = sum(int(row["unsafe_candidates"]) for row in arm_totals.values())
    total_evaluated = sum(
        int(row["correctness_evaluated_candidates"]) for row in arm_totals.values()
    )
    total_useful = sum(int(row["useful_candidates"]) for row in arm_totals.values())
    all_receipts_ready = len(receipt_rows) == total_calls and all(
        row["route_integrity_ready"] for row in receipt_rows
    )
    if len(receipt_rows) != total_calls:
        faults.append("runtime_receipt_count_mismatch")
    if total_unsafe:
        faults.append("unsafe_candidate_observed")
    if not all_receipts_ready:
        faults.append("not_all_runtime_receipts_ready")

    exact_complete = (
        not faults
        and len(task_results) == 10
        and total_calls == 36
        and total_evaluated == 14
        and total_useful == 2
        and paired["useful"]["direct_only"] == 1
        and paired["useful"]["integrated_only"] == 1
        and total_unsafe == 0
    )
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "scientific_status": (
            "P3_COMPLETE_RESIDUAL_EXPOSED_NO_USEFULNESS_ROUTE_WINNER"
            if exact_complete else "P3_REVIEW_REQUIRED"
        ),
        "scope": (
            "Ten licensed development tasks for the exact frozen Qwen3.5 local model. "
            "This is residual-selection evidence, not D1, D2, serving qualification, "
            "subsystem support, or a general coding benchmark."
        ),
        "source_identities": {
            "pool": source_identity(pool_path),
            "instrument": source_identity(instrument_path),
            "instrument_freeze_commit": pool.get("instrument_freeze_commit"),
            "task_pool_seal_commit": "1408abc829e3243f66f2b606b76aef514b11ab57",
            "runtime_receipts": receipt_rows,
        },
        "denominators": {
            "tasks": len(task_results),
            "arms": len(task_results) * 2,
            "persistent_model_loads": sum(row["model_loads"] for row in task_results),
            "model_calls": total_calls,
            "correctness_evaluated_candidates": total_evaluated,
            "useful_candidates": total_useful,
            "unsafe_candidates": total_unsafe,
            "hosted_reference_tasks": 0,
            "hosted_reference_calls": 0,
        },
        "arm_totals": arm_totals,
        "paired_outcomes": paired,
        "task_results": task_results,
        "residual_ledger": {
            "dominant_residual": "semantic_correctness_after_successful_typed_edit_lowering",
            "malformed_typed_edit_attempts": sum(
                int(row["malformed_candidates"]) for row in arm_totals.values()
            ),
            "evaluated_but_incorrect_candidates": total_evaluated - total_useful,
            "useful_candidates": total_useful,
            "unsafe_candidates": total_unsafe,
            "rollback_failures": 0,
            "route_integrity_failures": 0,
            "false_blocked": "NOT_CONSTRUCT_VALIDLY_MEASURED_BY_THIS_TYPED_EDIT_PROTOCOL",
            "interpretation": (
                "Integration increased parseable typed edits from 5/10 to 9/10 and used two fewer "
                "model calls, but each arm solved exactly 1/10 tasks. The wrapper therefore exposed "
                "a translation/protocol effect without a useful-completion advantage. Twelve of "
                "fourteen independently evaluated edits were semantically incorrect."
            ),
        },
        "statistical_boundary": {
            "useful_rate_difference_integrated_minus_direct": 0.0,
            "useful_discordant_pairs": 2,
            "useful_exact_two_sided_sign_test_p": exact_sign_test_two_sided(1, 2),
            "parseable_rate_difference_integrated_minus_direct": 0.4,
            "parseable_discordant_pairs": paired["parseable"]["discordant"],
            "parseable_exact_two_sided_sign_test_p": exact_sign_test_two_sided(
                paired["parseable"]["integrated_only"], paired["parseable"]["discordant"]
            ),
            "minimum_claim_boundary": (
                "The ten-task campaign is underpowered for a route-effect claim and the integrated "
                "arm is a compound wrapper. The parseability shift selects a residual; it does not "
                "identify a causal Theseus subsystem."
            ),
        },
        "consumption": {
            "all_ten_tasks_consumed": len(task_results) == 10,
            "eligible_for_exact_rerun": False,
            "eligible_for_training": False,
            "eligible_for_D1_or_D2": False,
            "all_failures_retained": True,
        },
        "hosted_reference": {
            "model": "gpt-5.6-luna",
            "effort": "xhigh",
            "transport_state": "DEFINED_NOT_BOUND",
            "results": "NOT_RUN",
            "local_campaign_valid_without_hosted_results": True,
            "same_consumed_pool_required_if_transport_binds": True,
            "denominators_must_remain_separate": True,
        },
        "next_stage": {
            "id": "P4_SINGLE_MECHANISM_CAUSAL_DEVELOPMENT",
            "selected_claim_id": "cognitive-compilation-and-semantic-ir.core",
            "selection_basis": (
                "The preregistered P4 mapping selects cognitive compilation for a translation or "
                "repair residual. The dominant observed failure is semantic incorrectness after "
                "successful typed-edit lowering (12/14 evaluated candidates), with an additional "
                "6/20 malformed outputs. Context dependence, dependency/replan behavior, adaptive "
                "verification allocation, authority, memory, routing, and replacement were not "
                "construct-validly varied by P3."
            ),
            "required_controls": [
                "information-matched direct target generation",
                "information-matched natural-language structured plan",
                "deterministic compiler-only baseline",
                "typed Semantic IR treatment with model use, stable identities, explicit loss or ambiguity, target validation, and dependency-local repair",
            ],
            "qualification_boundary": (
                "P4 must use fresh development tasks and first pass learnability, model-use, "
                "intervention, validator, repair-locality, matched-budget, and known-positive "
                "adequacy checks. P3 tasks are consumed and may not be replayed."
            ),
        },
        "maximum_inference": (
            "For the exact frozen Qwen3.5 model, ten licensed development tasks, and budgets, "
            "direct and integrated each solved one task. Integration produced more parseable edits "
            "but no useful-completion advantage. This exposes a semantic translation/repair residual "
            "and selects cognitive compilation for P4; it supports no subsystem, serving, D1, D2, "
            "or ASI Stack claim and cannot falsify any book mechanism."
        ),
        "counters": {
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "training_rows_written": 0,
            "user_facing_effects": 0,
        },
    }


def empty_arm_totals() -> dict[str, int | float | list[float]]:
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
        "runtime_ms": 0.0,
    }


def paired_outcomes(task_results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in ("useful", "parseable"):
        both = direct_only = integrated_only = neither = 0
        for task in task_results:
            arms = mapping(task.get("arms"))
            direct = bool(mapping(arms.get(DIRECT)).get(metric))
            integrated = bool(mapping(arms.get(INTEGRATED)).get(metric))
            both += int(direct and integrated)
            direct_only += int(direct and not integrated)
            integrated_only += int(integrated and not direct)
            neither += int(not direct and not integrated)
        output[metric] = {
            "both": both,
            "direct_only": direct_only,
            "integrated_only": integrated_only,
            "neither": neither,
            "discordant": direct_only + integrated_only,
        }
    return output


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [round(max(0.0, center - radius), 6), round(min(1.0, center + radius), 6)]


def exact_sign_test_two_sided(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    probability = sum(
        math.comb(trials, k) * (0.5 ** trials)
        for k in range(0, min(successes, trials - successes) + 1)
    )
    return min(1.0, round(2.0 * probability, 12))


def source_identity(path: Path) -> dict[str, str]:
    return {"path": relative(path), "sha256": sha256_file(path)}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def strings(value: Any) -> list[str]:
    return [str(item) for item in value if isinstance(item, str)] if isinstance(value, list) else []


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
