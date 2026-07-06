"""Real Code Benchmark Graduation v1.

This lane turns programming benchmarks into a governed ratchet instead of a
catalog entry. It runs the same public/local task cases through:

task -> local candidate generation -> sandbox tests -> critic/patch streams ->
official-ish scoring -> residual class -> transfer artifact -> retry.

The report is intentionally strict about semantics. HumanEval-style local public
tasks count as public calibration/regression evidence, not private mastery and
not a student-model score unless a student checkpoint generator is explicitly
wired in. BigCodeBench/LiveCodeBench are public-task evidence only when their
governed dataset payloads are staged locally; otherwise their loader manifests
remain non-promotable operational checks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from real_code_benchmark_constants import *  # noqa: F403
from public_code_case_manifest import (  # noqa: E402
    filter_tasks_for_card,
    load_case_manifest,
    manifest_context,
    manifest_pool_size,
)

if hasattr(sys, "set_int_max_str_digits"):
    try:
        current_limit = sys.get_int_max_str_digits()
        if current_limit and current_limit < 100000:
            sys.set_int_max_str_digits(100000)
    except (ValueError, AttributeError):
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", default=",".join(DEFAULT_CARDS))
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--max-cases-per-card", type=int, default=8)
    parser.add_argument("--code-transfer-artifacts", default="reports/code_transfer_artifacts.json")
    parser.add_argument("--student-candidate-manifest", default=DEFAULT_STUDENT_CANDIDATE_MANIFEST)
    parser.add_argument(
        "--case-manifest",
        default="",
        help="Optional public calibration selector manifest. Contains task IDs only, never public tests or solutions.",
    )
    parser.add_argument(
        "--student-candidate-generator",
        choices=["token", "program_prior"],
        default="token",
        help="Generator used when candidates are not skipped. token is the honest learned Rust token lane; program_prior is legacy private pressure only.",
    )
    parser.add_argument("--skip-student-candidate-generation", action="store_true")
    parser.add_argument(
        "--verification-workers",
        type=int,
        default=0,
        help="Parallel task-level sandbox workers. Default uses a bounded CPU-aware value.",
    )
    parser.add_argument("--out", default="reports/real_code_benchmark_graduation.json")
    parser.add_argument("--trace-out", default="reports/real_code_benchmark_traces.jsonl")
    parser.add_argument("--transfer-artifact-out", default="reports/transfer_artifacts/code/real_code_benchmark_graduation_transfer_artifact.json")
    args = parser.parse_args()

    started = time.perf_counter()
    verification_workers = bounded_verification_workers(args.verification_workers)
    requested_cards = [card.strip() for card in args.cards.split(",") if card.strip()]
    cards = expand_requested_cards(requested_cards)
    case_manifest_by_card = load_case_manifest(args.case_manifest)
    transfer = load_transfer(resolve(args.code_transfer_artifacts))
    student_candidate_generation = (
        {"skipped": True, "reason": "--skip-student-candidate-generation"}
        if args.skip_student_candidate_generation
        else ensure_student_candidates(
            cards=cards,
            seed=args.seed,
            max_cases_per_card=max(1, args.max_cases_per_card),
            manifest=resolve(args.student_candidate_manifest),
            candidate_generator=args.student_candidate_generator,
            case_manifest=args.case_manifest,
        )
    )
    student_candidates = load_student_candidates(resolve(args.student_candidate_manifest))
    suites = [
        run_suite_for_card(
            card_id=card_id,
            seed=args.seed,
            max_cases=max(1, args.max_cases_per_card),
            transfer=transfer,
            student_candidates=student_candidates,
            verification_workers=verification_workers,
            case_manifest_rows=case_manifest_by_card.get(card_id, []),
        )
        for card_id in cards
    ]
    traces = [trace for suite in suites for trace in suite.get("traces", [])]
    write_jsonl(resolve(args.trace_out), traces)
    transfer_artifact = write_transfer_artifact(
        resolve(args.transfer_artifact_out),
        suites=suites,
        transfer=transfer,
        trace_path=resolve(args.trace_out),
    )
    merge_transfer_index(resolve(args.code_transfer_artifacts), transfer_artifact)

    real_task_suites = [suite for suite in suites if suite.get("benchmark_evidence_level") == "public_benchmark_task_regression"]
    loader_suites = [suite for suite in suites if suite.get("benchmark_evidence_level") == "public_loader_regression"]
    runnable_suites = [suite for suite in suites if suite.get("case_count", 0) > 0]
    public_tasks = sum(int(suite.get("case_count") or 0) for suite in real_task_suites)
    loader_cases = sum(int(suite.get("case_count") or 0) for suite in loader_suites)
    single_passed = sum(int(suite.get("single_stream_passed") or 0) for suite in runnable_suites)
    multi_passed = sum(int(suite.get("multi_stream_passed") or 0) for suite in runnable_suites)
    total_cases = sum(int(suite.get("case_count") or 0) for suite in runnable_suites)
    real_total = sum(int(suite.get("case_count") or 0) for suite in real_task_suites)
    real_passed = sum(int(suite.get("multi_stream_passed") or 0) for suite in real_task_suites)
    regressions = sum(int(suite.get("task_level_regressions") or 0) for suite in runnable_suites)
    improvements = sum(int(suite.get("task_level_improvements") or 0) for suite in runnable_suites)
    multi_rate = ratio(multi_passed, total_cases)
    single_rate = ratio(single_passed, total_cases)
    real_public_rate = ratio(real_passed, real_total)
    transfer_delta = round(multi_rate - single_rate, 6)
    pass_origin_counts = candidate_pass_origin_counts(traces, mode="multi_stream")
    pass_family_counts = candidate_pass_family_counts(traces, mode="multi_stream")
    quality_summary = summarize_candidate_quality(traces)
    verification_cascade_summary = summarize_verification_cascade(traces)
    multi_candidate_rows = [
        row
        for row in traces
        if row.get("event") == "real_code_candidate_test" and row.get("mode") == "multi_stream"
    ]
    functional_promotion_rows = [
        row
        for row in multi_candidate_rows
        if bool(row.get("candidate_integrity_verified")) and bool(row.get("intended_behavior_passed"))
    ]
    functional_promotion_count = len(functional_promotion_rows)
    functional_promotion_by_family: dict[str, int] = {}
    for row in functional_promotion_rows:
        family = str(row.get("recomputed_candidate_family") or "unknown")
        functional_promotion_by_family[family] = functional_promotion_by_family.get(family, 0) + 1
    full_body_public_pass_count = count_pass_origins(
        pass_origin_counts,
        [
            "symliquid_recurrent_state_decoder",
            "sparse_state_sequence_decoder",
            "full_body_token_beam",
            "private_body_ngram_token_decoder",
            "greedy_body_token_decoder",
        ],
    )
    expression_fallback_public_pass_count = count_pass_origins(
        pass_origin_counts,
        ["private_expression_memory_fallback"],
    )
    blocked_cards = [suite for suite in suites if suite.get("status") == "blocked"]
    residual_rows = [row for suite in suites for row in suite.get("residuals", []) if isinstance(row, dict)]
    student_candidate_count = int(student_candidates["valid_candidate_count"] or 0)
    student_manifest_ready = (
        student_candidates["manifest_exists"]
        and student_candidate_count > 0
        and student_candidates["provenance_valid"]
    )
    student_benchmark_integrity_ready = bool(student_candidates.get("candidate_integrity_ready", student_candidates.get("benchmark_promotion_integrity_valid")))
    functional_promotion_ready = functional_promotion_count > 0 and real_public_rate > 0.0
    candidate_source = candidate_source_label(student_candidates, student_manifest_ready)
    public_benchmark_score_claim = (
        STUDENT_PUBLIC_SCORE_CLAIMS.get(candidate_source, "student_checkpoint_public_task_calibration_only")
        if student_manifest_ready and student_benchmark_integrity_ready
        else "forbidden_without_student_checkpoint_generator"
    )
    if student_manifest_ready and not student_benchmark_integrity_ready:
        public_benchmark_score_claim = FORBIDDEN_NON_LEARNED_SCORE_CLAIM
    promotion_allowed = False

    residuals_exported = bool(residual_rows) or (
        total_cases > 0
        and multi_passed == total_cases
        and not blocked_cards
    )
    residual_evidence = (
        f"residuals={len(residual_rows)} blocked={len(blocked_cards)} "
        f"multi_passed={multi_passed}/{total_cases}"
    )
    gates = [
        gate("real_public_task_regression_present", public_tasks > 0, f"public_tasks={public_tasks}"),
        gate(
            "case_manifest_selector_clean",
            not args.case_manifest or bool(case_manifest_by_card),
            manifest_context(args.case_manifest, case_manifest_by_card),
        ),
        gate("loader_regression_or_blockers_recorded", public_tasks > 0 or loader_cases > 0 or bool(blocked_cards), f"public_tasks={public_tasks} loader_cases={loader_cases} blocked={len(blocked_cards)}"),
        gate("same_cases_compared", all(bool(suite.get("same_cases_compared")) for suite in runnable_suites), f"suites={len(runnable_suites)}"),
        gate("multi_stream_no_task_regressions", regressions == 0, f"regressions={regressions}"),
        gate("transfer_heredity_measured", bool(transfer["artifacts"]) and all("transfer_behavior_changed" in suite for suite in runnable_suites), f"artifacts={len(transfer['artifacts'])}"),
        gate("residuals_exported", residuals_exported, residual_evidence),
        gate("student_checkpoint_candidate_generator_present", student_manifest_ready, f"manifest={args.student_candidate_manifest} candidates={student_candidate_count} provenance_valid={student_candidates['provenance_valid']}"),
        gate("no_hardcoded_solver_candidates", student_candidates["hardcoded_candidate_count"] == 0, f"hardcoded_candidate_count={student_candidates['hardcoded_candidate_count']}"),
        gate(
            "no_template_or_loop_distilled_benchmark_candidates",
            int(student_candidates.get("template_like_candidate_count") or 0) == 0
            and int(student_candidates.get("loop_closure_candidate_count") or 0) == 0,
            (
                f"template_like={student_candidates.get('template_like_candidate_count')} "
                f"loop_closure={student_candidates.get('loop_closure_candidate_count')} "
                "benchmark tasks may not be solved by tool/templates"
            ),
        ),
        gate(
            "token_level_student_code_generation_present",
            bool(student_candidates.get("token_level_code_generation_learned"))
            and int(student_candidates.get("benchmark_promotion_eligible_candidate_count") or 0) > 0,
            (
                f"token_level_learned={student_candidates.get('token_level_code_generation_learned')} "
                f"eligible_candidates={student_candidates.get('benchmark_promotion_eligible_candidate_count')}"
            ),
        ),
        gate(
            "recomputed_candidate_integrity_clean",
            int(student_candidates.get("candidate_integrity_mismatch_count") or 0) == 0
            and int(student_candidates.get("integrity_verified_candidate_count") or 0) > 0,
            {
                "policy": student_candidates.get("candidate_integrity_policy"),
                "mismatch_count": student_candidates.get("candidate_integrity_mismatch_count"),
                "mismatch_counts": student_candidates.get("candidate_integrity_mismatch_counts"),
                "integrity_verified_candidate_count": student_candidates.get("integrity_verified_candidate_count"),
                "family_counts": student_candidates.get("candidate_family_counts"),
            },
        ),
        gate(
            "functional_promotion_requires_behavioral_pass",
            functional_promotion_ready,
            {
                "functional_promotion_count": functional_promotion_count,
                "multi_candidate_attempt_count": len(multi_candidate_rows),
                "real_public_passed": real_passed,
                "real_public_total": real_total,
                "functional_promotion_by_family": functional_promotion_by_family,
            },
        ),
        gate(
            "compositional_token_generation_present",
            int(student_candidates.get("compositional_token_candidate_count") or 0) > 0,
            (
                f"compositional_candidates={student_candidates.get('compositional_token_candidate_count')} "
                f"expression_memory_fallbacks={student_candidates.get('expression_memory_fallback_count')}"
            ),
        ),
        gate(
            "full_body_token_generation_present",
            int(student_candidates.get("full_body_token_candidate_count") or 0) > 0,
            (
                f"full_body_candidates={student_candidates.get('full_body_token_candidate_count')} "
                f"return_expression_fallbacks={student_candidates.get('expression_memory_fallback_count')}"
            ),
        ),
        gate(
            "grammar_masked_learned_token_generation_present",
            int(student_candidates.get("grammar_masked_learned_token_candidate_count") or 0) > 0
            and int(student_candidates.get("benchmark_promotion_eligible_candidate_count") or 0) > 0,
            (
                "grammar_masked_learned_token_candidates="
                f"{student_candidates.get('grammar_masked_learned_token_candidate_count')} "
                f"eligible_candidates={student_candidates.get('benchmark_promotion_eligible_candidate_count')}"
            ),
        ),
        gate("public_score_claim_quarantined", True, public_benchmark_score_claim),
        gate("staged_verification_contract_present", len(VERIFICATION_STAGE_CONTRACT) >= 6, [row["stage"] for row in VERIFICATION_STAGE_CONTRACT]),
        gate("parallel_verification_policy_present", bool(PARALLEL_VERIFICATION_POLICY.get("default_worker_rule")), PARALLEL_VERIFICATION_POLICY),
        gate("external_inference_zero", True, "local student candidates only; no teacher/provider inference"),
    ]
    trigger_state = "GREEN" if public_tasks > 0 and all(row["passed"] for row in gates) else "YELLOW"
    report = {
        "policy": "project_theseus_real_code_benchmark_graduation_v1",
        "created_utc": now(),
        "frontier_family": "coding_local_sandbox",
        "runner_family": "real_code_benchmark_graduation",
        "seed": args.seed,
        "requested_cards": requested_cards,
        "cards": cards,
        "case_manifest": manifest_context(args.case_manifest, case_manifest_by_card),
        "candidate_source": candidate_source,
        "score": real_public_rate,
        "score_semantics": (
            score_semantics_for_candidate_source(candidate_source)
            if student_manifest_ready and student_benchmark_integrity_ready
            else "nonpromotable_candidate_calibration_not_student_learning_or_public_mastery"
        ),
        "benchmark_evidence_level": "mixed_public_task_and_loader_regression",
        "public_benchmark_score_claim": public_benchmark_score_claim,
        "promotion_allowed": promotion_allowed,
        "promotion_rule": "requires token-level learned student code generation, no template/tool-distilled benchmark candidates, and public/private regression gates",
        "trigger_state": trigger_state,
        "summary": {
            "suite_count": len(suites),
            "runnable_suite_count": len(runnable_suites),
            "blocked_suite_count": len(blocked_cards),
            "public_task_count": public_tasks,
            "loader_regression_case_count": loader_cases,
            "total_case_count": total_cases,
            "case_manifest_enabled": bool(args.case_manifest),
            "case_manifest_selected_count": sum(len(rows) for rows in case_manifest_by_card.values()),
            "case_manifest_card_counts": {card_id: len(rows) for card_id, rows in sorted(case_manifest_by_card.items())},
            "single_stream_pass_rate": single_rate,
            "multi_stream_pass_rate": multi_rate,
            "real_public_task_pass_rate": real_public_rate,
            "real_public_task_pass_fraction": fraction(real_passed, real_total),
            "real_public_task_pass_rate_ci95": wilson_ci(real_passed, real_total),
            "pass_rate_delta": transfer_delta,
            "task_level_improvements_over_single_stream": improvements,
            "task_level_regressions_vs_single_stream": regressions,
            "transfer_artifacts_loaded": len(transfer["artifacts"]),
            "transfer_behavior_changed_suites": len([suite for suite in suites if suite.get("transfer_behavior_changed")]),
            "student_candidate_count": student_candidate_count,
            "student_candidate_manifest_exists": student_candidates["manifest_exists"],
            "student_candidate_provenance_valid": student_candidates["provenance_valid"],
            "student_candidate_benchmark_integrity_valid": student_benchmark_integrity_ready,
            "student_candidate_integrity_ready": student_benchmark_integrity_ready,
            "template_like_candidate_count": student_candidates.get("template_like_candidate_count", 0),
            "loop_closure_candidate_count": student_candidates.get("loop_closure_candidate_count", 0),
            "token_level_code_generation_learned": bool(student_candidates.get("token_level_code_generation_learned")),
            "token_level_learned_candidate_count": student_candidates.get("token_level_learned_candidate_count", 0),
            "compositional_token_candidate_count": student_candidates.get("compositional_token_candidate_count", 0),
            "full_body_token_candidate_count": student_candidates.get("full_body_token_candidate_count", 0),
            "grammar_masked_learned_token_candidate_count": student_candidates.get("grammar_masked_learned_token_candidate_count", 0),
            "expression_memory_fallback_count": student_candidates.get("expression_memory_fallback_count", 0),
            "deterministic_guardrail_failed_candidate_count": student_candidates.get("deterministic_guardrail_failed_candidate_count", 0),
            "multi_stream_pass_origin_counts": pass_origin_counts,
            "multi_stream_pass_family_counts": pass_family_counts,
            "candidate_quality_summary": quality_summary,
            "verification_workers": verification_workers,
            "parallel_verification_enabled": verification_workers > 1,
            "verification_stage_count": len(VERIFICATION_STAGE_CONTRACT),
            "verification_cascade_summary": verification_cascade_summary,
            "functional_promotion_count": functional_promotion_count,
            "functional_promotion_rate": ratio(functional_promotion_count, len(multi_candidate_rows)),
            "functional_promotion_fraction": fraction(functional_promotion_count, len(multi_candidate_rows)),
            "functional_promotion_rate_ci95": wilson_ci(functional_promotion_count, len(multi_candidate_rows)),
            "functional_promotion_by_family": dict(sorted(functional_promotion_by_family.items())),
            "functional_promotion_score_semantics": "requires candidate_integrity_verified=true and intended_behavior_passed=true on executed candidates",
            "full_body_public_pass_count": full_body_public_pass_count,
            "expression_fallback_public_pass_count": expression_fallback_public_pass_count,
            "benchmark_promotion_eligible_candidate_count": student_candidates.get("benchmark_promotion_eligible_candidate_count", 0),
            "candidate_integrity_policy": student_candidates.get("candidate_integrity_policy"),
            "candidate_family_counts": student_candidates.get("candidate_family_counts", {}),
            "claimed_promotion_by_family": student_candidates.get("claimed_promotion_by_family", {}),
            "integrity_verified_by_family": student_candidates.get("integrity_verified_by_family", {}),
            "candidate_integrity_mismatch_count": student_candidates.get("candidate_integrity_mismatch_count", 0),
            "candidate_integrity_mismatch_counts": student_candidates.get("candidate_integrity_mismatch_counts", {}),
            "integrity_verified_candidate_count": student_candidates.get("integrity_verified_candidate_count", 0),
            "candidate_generation_modes": student_candidates.get("candidate_generation_modes", []),
            "external_inference_calls": 0,
        },
        "parallel_verification_policy": PARALLEL_VERIFICATION_POLICY,
        "verification_stage_contract": VERIFICATION_STAGE_CONTRACT,
        "student_candidate_manifest": {
            "path": args.student_candidate_manifest,
            "exists": student_candidates["manifest_exists"],
            "candidate_count": student_candidate_count,
            "valid_candidate_count": student_candidates["valid_candidate_count"],
            "invalid_candidate_count": student_candidates["invalid_candidate_count"],
            "hardcoded_candidate_count": student_candidates["hardcoded_candidate_count"],
            "template_like_candidate_count": student_candidates["template_like_candidate_count"],
            "loop_closure_candidate_count": student_candidates["loop_closure_candidate_count"],
            "token_level_learned_candidate_count": student_candidates["token_level_learned_candidate_count"],
            "compositional_token_candidate_count": student_candidates["compositional_token_candidate_count"],
            "full_body_token_candidate_count": student_candidates["full_body_token_candidate_count"],
            "grammar_masked_learned_token_candidate_count": student_candidates["grammar_masked_learned_token_candidate_count"],
            "expression_memory_fallback_count": student_candidates["expression_memory_fallback_count"],
            "deterministic_guardrail_failed_candidate_count": student_candidates["deterministic_guardrail_failed_candidate_count"],
            "benchmark_promotion_eligible_candidate_count": student_candidates["benchmark_promotion_eligible_candidate_count"],
            "benchmark_promotion_integrity_valid": student_candidates["benchmark_promotion_integrity_valid"],
            "candidate_integrity_ready": student_candidates.get("candidate_integrity_ready", student_candidates["benchmark_promotion_integrity_valid"]),
            "candidate_integrity_policy": student_candidates.get("candidate_integrity_policy"),
            "candidate_family_counts": student_candidates.get("candidate_family_counts", {}),
            "claimed_promotion_by_family": student_candidates.get("claimed_promotion_by_family", {}),
            "integrity_verified_by_family": student_candidates.get("integrity_verified_by_family", {}),
            "candidate_integrity_mismatch_count": student_candidates.get("candidate_integrity_mismatch_count", 0),
            "candidate_integrity_mismatch_counts": student_candidates.get("candidate_integrity_mismatch_counts", {}),
            "integrity_verified_candidate_count": student_candidates.get("integrity_verified_candidate_count", 0),
            "candidate_generation_modes": student_candidates["candidate_generation_modes"],
            "checkpoint_ids": student_candidates["checkpoint_ids"],
            "candidate_sources": student_candidates["candidate_sources"],
            "blocked_reason": (
                ""
                if student_manifest_ready and student_benchmark_integrity_ready
                else (
                    "non_token_level_or_template_candidate_generation_not_valid_for_benchmark_promotion"
                    if student_manifest_ready
                    else "missing_valid_local_theseus_student_checkpoint_candidates"
                )
            ),
        },
        "student_candidate_generation": student_candidate_generation,
        "transfer_consumption": transfer,
        "suites": without_traces(suites),
        "residuals": residual_rows,
        "gates": gates,
        "artifacts": {
            "trace": rel(resolve(args.trace_out)),
            "transfer_artifact": rel(transfer_artifact),
            "transfer_index": args.code_transfer_artifacts,
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if trigger_state in {"GREEN", "YELLOW"} else 1


def expand_requested_cards(cards: list[str]) -> list[str]:
    expanded: list[str] = []
    for card_id in cards or DEFAULT_CARDS:
        card = read_json(ROOT / "benchmarks" / "cards" / f"{card_id}.json", {})
        family = str(card.get("family") or "")
        runner = str(card.get("runner_family") or "")
        is_synthetic_code_card = (
            family == "coding_local_sandbox"
            and (
                runner == "synthetic_benchmark_local"
                or card_id.startswith("synthetic_")
                or card_id.startswith("multistream_")
            )
        )
        if is_synthetic_code_card:
            expanded.extend(DEFAULT_CARDS)
        else:
            expanded.append(card_id)
    out: list[str] = []
    seen = set()
    for card_id in expanded:
        if card_id in seen:
            continue
        seen.add(card_id)
        out.append(card_id)
    return out or list(DEFAULT_CARDS)


def ensure_student_candidates(
    *,
    cards: list[str],
    seed: int,
    max_cases_per_card: int,
    manifest: Path,
    candidate_generator: str,
    case_manifest: str = "",
) -> dict[str, Any]:
    if candidate_generator == "program_prior":
        report_path = ROOT / "reports" / "student_code_candidate_generator.json"
        checkpoint_path = ROOT / "reports" / "local_theseus_student_code_checkpoint.json"
        command = [
            sys.executable,
            "scripts/local_student_code_candidate_generator.py",
            "--cards",
            ",".join(cards),
            "--seed",
            str(seed),
            "--max-cases-per-card",
            str(max(1, max_cases_per_card)),
            "--out",
            rel(manifest),
            "--checkpoint-out",
            rel(checkpoint_path),
            "--report-out",
            rel(report_path),
        ]
    else:
        report_path = ROOT / "reports" / "student_token_code_generator.json"
        checkpoint_path = ROOT / "reports" / "student_token_code_checkpoint.json"
        command = [
            sys.executable,
            "scripts/student_token_code_candidate_generator.py",
            "--cards",
            ",".join(cards),
            "--seed",
            str(seed),
            "--max-cases-per-card",
            str(max(1, max_cases_per_card)),
            "--out",
            rel(manifest),
            "--checkpoint-out",
            rel(checkpoint_path),
            "--report-out",
            rel(report_path),
        ]
    if str(case_manifest or "").strip():
        command.extend(["--case-manifest", str(case_manifest)])
    stale_removed = remove_stale_candidate_generation_artifacts(
        [manifest, checkpoint_path, report_path]
    )
    timeout_seconds = candidate_generation_timeout_seconds(
        card_count=len(cards),
        max_cases_per_card=max_cases_per_card,
        candidate_generator=candidate_generator,
    )
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        payload = read_json(report_path, {})
        return {
            "skipped": False,
            "candidate_generator": candidate_generator,
            "command": command,
            "removed_stale_artifacts": stale_removed,
            "timeout_seconds": timeout_seconds,
            "returncode": result.returncode,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "report": rel(report_path),
            "checkpoint": rel(checkpoint_path),
            "manifest": rel(manifest),
            "trigger_state": payload.get("trigger_state"),
            "candidate_count": get_path(payload, ["summary", "candidate_count"], 0),
            "checkpoint_id": get_path(payload, ["summary", "checkpoint_id"], ""),
            "stderr_tail": result.stderr[-1200:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "skipped": False,
            "candidate_generator": candidate_generator,
            "command": command,
            "removed_stale_artifacts": stale_removed,
            "timeout_seconds": timeout_seconds,
            "returncode": 124 if isinstance(exc, subprocess.TimeoutExpired) else 127,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "report": rel(report_path),
            "checkpoint": rel(checkpoint_path),
            "manifest": rel(manifest),
            "error": str(exc)[-1200:],
        }


def candidate_generation_timeout_seconds(
    *,
    card_count: int,
    max_cases_per_card: int,
    candidate_generator: str,
) -> int:
    task_budget = max(1, int(card_count)) * max(1, int(max_cases_per_card))
    if candidate_generator == "program_prior":
        return max(120, min(900, 30 + task_budget * 2))
    return max(300, min(2400, 120 + task_budget * 7))


def remove_stale_candidate_generation_artifacts(paths: list[Path]) -> list[str]:
    removed: list[str] = []
    for path in paths:
        try:
            if path.exists():
                path.unlink()
                removed.append(rel(path))
        except OSError:
            continue
    return removed


def run_suite_for_card(
    *,
    card_id: str,
    seed: int,
    max_cases: int,
    transfer: dict[str, Any],
    student_candidates: dict[str, Any],
    verification_workers: int,
    case_manifest_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    card = read_json(ROOT / "benchmarks" / "cards" / f"{card_id}.json", {})
    source_id = str(card.get("source_id") or card_id.replace("source_", ""))
    source_path = resolve_source_path(card)
    base = {
        "card_id": card_id,
        "source_id": source_id,
        "source_path": rel_or_abs(source_path),
        "license_allowed": bool(card.get("license_allowed")),
        "status": "blocked",
        "case_count": 0,
        "single_stream_passed": 0,
        "multi_stream_passed": 0,
        "task_level_improvements": 0,
        "task_level_regressions": 0,
        "residual_count": 0,
        "same_cases_compared": False,
        "external_inference_calls": 0,
    }
    if not source_path.exists():
        return {
            **base,
            "blocker": "source_not_staged",
            "residuals": [residual("source_not_staged", "coding benchmark source is not locally staged", card_id=card_id)],
            "traces": [],
        }

    case_manifest_rows = case_manifest_rows or []
    load_limit = manifest_pool_size(max_cases, {card_id: case_manifest_rows}) if case_manifest_rows else max_cases
    tasks, evidence_level, semantics = load_cases(card_id, source_id, source_path, seed, load_limit)
    if not tasks:
        return {
            **base,
            "blocker": "no_local_task_cases",
            "benchmark_evidence_level": "source_staged_no_local_task_cases",
            "score_semantics": "blocked_not_scored",
            "residuals": [residual("problem_manifest_locator", "no local task dataset or loader-regression manifest available", card_id=card_id)],
            "traces": [],
        }
    missing_case_manifest_ids: list[str] = []
    if case_manifest_rows:
        tasks, missing_case_manifest_ids = filter_tasks_for_card(tasks, case_manifest_rows)
        if missing_case_manifest_ids or not tasks:
            return {
                **base,
                "blocker": "case_manifest_missing_task_ids",
                "benchmark_evidence_level": evidence_level,
                "score_semantics": "blocked_case_manifest_selection_not_found",
                "case_manifest_enabled": True,
                "case_manifest_selected_count": len(case_manifest_rows),
                "case_manifest_missing_task_count": len(missing_case_manifest_ids),
                "case_manifest_missing_task_ids": missing_case_manifest_ids[:32],
                "residuals": [
                    residual(
                        "case_manifest_missing_task_ids",
                        "public calibration selector named tasks that are not available in the staged local benchmark payload",
                        card_id=card_id,
                        detail=f"missing={len(missing_case_manifest_ids)}",
                    )
                ],
                "traces": [],
            }

    single = run_cases(
        tasks,
        mode="single_stream",
        transfer_categories=[],
        student_candidates=student_candidates,
        verification_workers=verification_workers,
    )
    multi = run_cases(
        tasks,
        mode="multi_stream",
        transfer_categories=transfer["categories"],
        student_candidates=student_candidates,
        verification_workers=verification_workers,
    )
    suite_student_ready = (
        student_candidates.get("manifest_exists")
        and int(student_candidates.get("valid_candidate_count") or 0) > 0
        and bool(student_candidates.get("provenance_valid"))
    )
    single_by_id = {row["task_id"]: row for row in single["results"]}
    improved = []
    regressed = []
    for row in multi["results"]:
        before = single_by_id.get(row["task_id"])
        if not before:
            continue
        if row["passed"] and not before["passed"]:
            improved.append(row["task_id"])
        elif before["passed"] and not row["passed"]:
            regressed.append(row["task_id"])
    residual_rows = [
        residual(
            row.get("residual_class") or "code_repair_failure",
            "multi-stream real-code graduation task failed",
            card_id=card_id,
            task_id=row.get("task_id"),
            detail=row.get("stderr_tail") or row.get("detail") or "",
        )
        for row in multi["results"]
        if not row.get("passed")
    ]
    suite_candidate_source = candidate_source_label(student_candidates, suite_student_ready)
    suite_integrity_ready = bool(student_candidates.get("benchmark_promotion_integrity_valid"))
    return {
        **base,
        "status": "frontier_open",
        "benchmark_evidence_level": evidence_level,
        "score_semantics": semantics,
        "candidate_source": suite_candidate_source,
        "public_benchmark_score_claim": (
            STUDENT_PUBLIC_SCORE_CLAIMS.get(suite_candidate_source, "forbidden_without_student_checkpoint_generator")
            if suite_student_ready and suite_integrity_ready
            else (FORBIDDEN_NON_LEARNED_SCORE_CLAIM if suite_student_ready else "forbidden_without_student_checkpoint_generator")
        ),
        "student_candidate_benchmark_integrity_valid": suite_integrity_ready,
        "case_count": len(tasks),
        "case_manifest_enabled": bool(case_manifest_rows),
        "case_manifest_selected_count": len(case_manifest_rows),
        "case_manifest_missing_task_count": 0,
        "single_stream_passed": single["passed"],
        "multi_stream_passed": multi["passed"],
        "single_stream_pass_rate": single["pass_rate"],
        "multi_stream_pass_rate": multi["pass_rate"],
        "pass_rate_delta": round(multi["pass_rate"] - single["pass_rate"], 6),
        "task_level_improvements": len(improved),
        "task_level_regressions": len(regressed),
        "improved_task_ids": improved,
        "regressed_task_ids": regressed,
        "transfer_behavior_changed": behavior_changed(single["results"], multi["results"]),
        "same_cases_compared": same_task_ids(single["results"], multi["results"]),
        "residual_count": len(residual_rows),
        "residuals": residual_rows,
        "traces": single["traces"] + multi["traces"],
        "verification": {
            "workers": verification_workers,
            "single_stream": {
                "runtime_ms": single.get("runtime_ms"),
                "cascade": single.get("verification_cascade_summary"),
            },
            "multi_stream": {
                "runtime_ms": multi.get("runtime_ms"),
                "cascade": multi.get("verification_cascade_summary"),
            },
        },
        "case_ids": [task["task_id"] for task in tasks],
    }

from real_code_benchmark_datasets import *  # noqa: F403
from real_code_benchmark_runtime import *  # noqa: F403
from real_code_benchmark_support import *  # noqa: F403

if __name__ == "__main__":
    raise SystemExit(main())
