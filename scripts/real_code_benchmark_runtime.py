"""Staged candidate verification for real-code benchmark graduation."""

from __future__ import annotations

import ast
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from real_code_benchmark_constants import *  # noqa: F403
from real_code_benchmark_support import *  # noqa: F403
from candidate_integrity import recompute_candidate_integrity


__all__ = [
    "run_cases",
    "run_task_candidates",
    "candidates_for",
    "mode_quality_prior",
    "candidate_pass_origin_counts",
    "candidate_pass_family_counts",
    "summarize_candidate_quality",
    "summarize_verification_cascade",
    "candidate_mode_from_origin",
    "count_pass_origins",
    "load_student_candidates",
    "candidate_source_label",
    "score_semantics_for_candidate_source",
    "normalize_student_candidate",
    "student_candidate_keys",
    "student_candidates_for_task",
    "student_candidate_identity_matches_task",
    "looks_hardcoded_origin",
    "is_template_like_candidate",
    "is_grammar_masked_learned_token_candidate",
    "benchmark_candidate_eligible",
    "run_candidate",
    "build_task_verification_context",
    "cached_candidate_quality",
    "test_harness_compile_preflight",
    "candidate_static_verification",
    "candidate_stage_result",
    "sandbox_cascade_source",
    "candidate_runtime_env",
    "visible_prompt_prelude",
    "trace_event",
]

def run_cases(
    tasks: list[dict[str, Any]],
    *,
    mode: str,
    transfer_categories: list[str],
    student_candidates: dict[str, Any],
    verification_workers: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    traces: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="theseus_real_code_grad_", dir=runtime_tmp_dir()) as tmp:
        tmp_root = Path(tmp)
        worker_count = min(max(1, verification_workers), max(1, len(tasks)))
        if worker_count == 1:
            task_runs = [
                run_task_candidates(
                    tmp_root,
                    task,
                    mode=mode,
                    transfer_categories=transfer_categories,
                    student_candidates=student_candidates,
                )
                for task in tasks
            ]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        run_task_candidates,
                        tmp_root,
                        task,
                        mode=mode,
                        transfer_categories=transfer_categories,
                        student_candidates=student_candidates,
                    )
                    for task in tasks
                ]
                task_runs = [future.result() for future in futures]
        for task_run in task_runs:
            traces.extend(task_run["traces"])
            results.append(task_run["result"])
            passed = bool(task_run["result"].get("passed"))
            if not passed:
                traces.append(
                    {
                        "event": "real_code_residual_export",
                        "created_utc": now(),
                        "task_id": task_run["result"]["task_id"],
                        "mode": mode,
                        "stream": "residual_stream",
                        "residual_class": task_run["result"].get("residual_class") or "code_repair_failure",
                        "stderr_tail": str(task_run["result"].get("stderr_tail", ""))[-800:],
                    }
                )
    passed_count = sum(1 for row in results if row.get("passed"))
    return {
        "mode": mode,
        "passed": passed_count,
        "total": len(results),
        "pass_rate": ratio(passed_count, len(results)),
        "results": results,
        "traces": traces,
        "verification_workers": verification_workers,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "verification_cascade_summary": summarize_verification_cascade(traces),
    }


def run_task_candidates(
    tmp_root: Path,
    task: dict[str, Any],
    *,
    mode: str,
    transfer_categories: list[str],
    student_candidates: dict[str, Any],
) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    candidates = candidates_for(
        task,
        mode=mode,
        transfer_categories=transfer_categories,
        student_candidates=student_candidates,
    )
    final: dict[str, Any]
    if not candidates:
        final = {
            "task_id": task["task_id"],
            "mode": mode,
            "attempt_index": 0,
            "passed": False,
            "test_passed": False,
            "beautiful_code_gate_passed": False,
            "returncode": 125,
            "stderr": "missing local Theseus student checkpoint candidate for task",
            "stdout": "",
            "runtime_ms": 0,
            "candidate_sha256": "",
            "candidate_origin": "missing_student_candidate",
            "verification_stage": "no_candidate",
            "verification_reward": 0.0,
            "reward_breakdown": {},
        }
        traces.append(trace_event(task, final, mode=mode, candidate_origin="missing_student_candidate"))
    else:
        final = {}
        verification_context = build_task_verification_context(tmp_root, task)
        for index, candidate in enumerate(candidates, start=1):
            result = run_candidate(
                tmp_root,
                task,
                candidate,
                mode=mode,
                attempt_index=index,
                verification_context=verification_context,
            )
            traces.append(trace_event(task, result, mode=mode, candidate_origin=candidate["origin"]))
            final = {**result, "candidate_origin": candidate["origin"]}
            if result["passed"]:
                break
    passed = bool(final.get("passed"))
    return {
        "traces": traces,
        "result": {
            "task_id": task["task_id"],
            "mode": mode,
            "passed": passed,
            "attempts": int(final.get("attempt_index") or 0),
            "candidate_origin": final.get("candidate_origin"),
            "candidate_sha256": final.get("candidate_sha256"),
            "recomputed_candidate_family": final.get("recomputed_candidate_family"),
            "candidate_integrity_verified": final.get("candidate_integrity_verified"),
            "test_passed": bool(final.get("test_passed")),
            "beautiful_code_gate_passed": bool(final.get("beautiful_code_gate_passed")),
            "beautiful_code_score": final.get("beautiful_code_score"),
            "beautiful_code_reasons": final.get("beautiful_code_reasons"),
            "verification_stage": final.get("verification_stage"),
            "verification_reward": final.get("verification_reward"),
            "reward_breakdown": final.get("reward_breakdown"),
            "lint_passed": bool(final.get("lint_passed")),
            "compile_passed": bool(final.get("compile_passed")),
            "runtime_loaded": bool(final.get("runtime_loaded")),
            "intended_behavior_passed": bool(final.get("intended_behavior_passed")),
            "residual_class": "" if passed else classify_failure(final.get("stderr", "")),
            "stderr_tail": final.get("stderr", ""),
        },
    }


def candidates_for(
    task: dict[str, Any],
    *,
    mode: str,
    transfer_categories: list[str],
    student_candidates: dict[str, Any],
) -> list[dict[str, str]]:
    """Return only candidates emitted by a local student checkpoint.

    Public-code graduation must evaluate learning, not a hand-written solver.
    The critic/patch streams may select, order, test, and classify candidates,
    but they may not synthesize benchmark-specific implementations here.
    """
    rows = [
        row
        for row in student_candidates_for_task(task, student_candidates)
        if benchmark_candidate_eligible(row)
    ]
    candidates = [
        {
            "origin": str(row.get("origin") or "local_theseus_student_checkpoint"),
            "code": str(row.get("code") or ""),
            "candidate_generation_mode": str(row.get("candidate_generation_mode") or ""),
            "recomputed_candidate_family": str(row.get("recomputed_candidate_family") or ""),
            "candidate_integrity_verified": bool(row.get("candidate_integrity_verified")),
            "candidate_integrity_mismatches": row.get("candidate_integrity_mismatches") or [],
            "manifest_beautiful_code_score": row.get("beautiful_code_score"),
            "manifest_placeholder_scaffold_body": row.get("placeholder_scaffold_body"),
        }
        for row in rows
        if str(row.get("code") or "").strip()
    ]
    for index, candidate in enumerate(candidates):
        quality = candidate_quality(task, candidate["code"], candidate["origin"])
        candidate.update(quality)
        candidate["candidate_manifest_index"] = index
    tags = set(str(tag) for tag in task.get("tags", [])) | set(transfer_categories)
    candidates.sort(
        key=lambda row: (
            1 if tags and any(tag in str(row.get("origin") or "") for tag in tags) else 0,
            float(row.get("beautiful_code_score") or 0.0),
            mode_quality_prior(str(row.get("origin") or "")),
            -int(row.get("candidate_manifest_index") or 0),
        ),
        reverse=True,
    )
    if mode == "single_stream":
        unconditioned = [
            row
            for row in candidates
            if "_sts_conditioned" not in str(row.get("origin") or "")
        ]
        return (unconditioned or candidates)[:1]
    return dedupe_candidates(candidates)


def mode_quality_prior(origin: str) -> float:
    mode = candidate_mode_from_origin(origin)
    if "edge_exec_sparse_state_sequence_sts" in mode:
        return 1.3
    if "edge_exec_repair" in mode:
        return 1.1
    if "private_body_ngram_sts" in mode:
        return 0.9
    if "causal_contract" in mode or "contract_guided" in mode:
        return 0.8
    if "local_adapter_edge_skeleton" in mode:
        return 0.4
    if "no_admissible" in mode:
        return -10.0
    return 0.0


def candidate_pass_origin_counts(traces: list[dict[str, Any]], *, mode: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in traces:
        if row.get("event") != "real_code_candidate_test":
            continue
        if row.get("mode") != mode or not bool(row.get("passed")):
            continue
        candidate_mode = candidate_mode_from_origin(str(row.get("candidate_origin") or ""))
        counts[candidate_mode] = counts.get(candidate_mode, 0) + 1
    return dict(sorted(counts.items()))


def candidate_pass_family_counts(traces: list[dict[str, Any]], *, mode: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in traces:
        if row.get("event") != "real_code_candidate_test":
            continue
        if row.get("mode") != mode or not bool(row.get("passed")):
            continue
        family = str(row.get("recomputed_candidate_family") or "unknown")
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def summarize_candidate_quality(traces: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_rows = [
        row
        for row in traces
        if row.get("event") == "real_code_candidate_test"
        and row.get("mode") == "multi_stream"
    ]
    test_passes = [row for row in candidate_rows if bool(row.get("test_passed"))]
    quality_passes = [row for row in candidate_rows if bool(row.get("passed"))]
    blocked_reasons: dict[str, int] = {}
    scores = []
    for row in candidate_rows:
        if row.get("beautiful_code_score") is not None:
            try:
                scores.append(float(row.get("beautiful_code_score")))
            except (TypeError, ValueError):
                pass
        if bool(row.get("test_passed")) and not bool(row.get("beautiful_code_gate_passed")):
            for reason in row.get("beautiful_code_reasons") or ["quality_gate_failed"]:
                blocked_reasons[str(reason)] = blocked_reasons.get(str(reason), 0) + 1
    rank1_quality_passes = sum(
        1
        for row in quality_passes
        if int(row.get("attempt_index") or 0) == 1
    )
    return {
        "policy": "beautiful_code_quality_gate_v1",
        "candidate_attempts": len(candidate_rows),
        "raw_test_pass_count": len(test_passes),
        "quality_pass_count": len(quality_passes),
        "quality_blocked_test_pass_count": len(test_passes) - len(quality_passes),
        "rank1_quality_pass_count": rank1_quality_passes,
        "mean_quality_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
        "quality_blocked_reasons": dict(sorted(blocked_reasons.items())),
        "score_semantics": "public calibration pass now requires tests plus exact-signature non-vacuous beautiful-code gate",
    }


def summarize_verification_cascade(traces: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_rows = [row for row in traces if row.get("event") == "real_code_candidate_test"]
    if not candidate_rows:
        return {
            "candidate_attempt_count": 0,
            "lint_pass_rate": 0.0,
            "compile_pass_rate": 0.0,
            "runtime_load_rate": 0.0,
            "intended_behavior_pass_rate": 0.0,
            "mean_verification_reward": 0.0,
            "sandbox_skipped_before_runtime_count": 0,
        }
    rewards = []
    stage_counts: dict[str, int] = {}
    for row in candidate_rows:
        try:
            rewards.append(float(row.get("verification_reward") or 0.0))
        except (TypeError, ValueError):
            rewards.append(0.0)
        stage = str(row.get("verification_stage") or "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    return {
        "candidate_attempt_count": len(candidate_rows),
        "lint_pass_rate": ratio(sum(1 for row in candidate_rows if bool(row.get("lint_passed"))), len(candidate_rows)),
        "compile_pass_rate": ratio(sum(1 for row in candidate_rows if bool(row.get("compile_passed"))), len(candidate_rows)),
        "runtime_load_rate": ratio(sum(1 for row in candidate_rows if bool(row.get("runtime_loaded"))), len(candidate_rows)),
        "intended_behavior_pass_rate": ratio(sum(1 for row in candidate_rows if bool(row.get("intended_behavior_passed"))), len(candidate_rows)),
        "mean_verification_reward": round(sum(rewards) / len(rewards), 6),
        "sandbox_skipped_before_runtime_count": sum(1 for row in candidate_rows if not bool(row.get("compile_passed"))),
        "verification_stage_counts": dict(sorted(stage_counts.items())),
        "reward_semantics": "dense public-calibration diagnostics only; do not train on public tests or public rewards",
    }


def candidate_mode_from_origin(origin: str) -> str:
    parts = origin.split(":")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return origin or "unknown_candidate_origin"


def count_pass_origins(counts: dict[str, int], needles: list[str]) -> int:
    return sum(
        count
        for origin, count in counts.items()
        if any(needle in origin for needle in needles)
    )


def load_student_candidates(path: Path) -> dict[str, Any]:
    by_key: dict[str, list[dict[str, Any]]] = {}
    checkpoint_ids: list[str] = []
    candidate_sources: list[str] = []
    candidate_generation_modes: list[str] = []
    invalid_count = 0
    hardcoded_count = 0
    template_like_count = 0
    loop_closure_count = 0
    token_level_count = 0
    compositional_token_count = 0
    full_body_token_count = 0
    grammar_masked_learned_token_count = 0
    expression_memory_fallback_count = 0
    guardrail_failed_count = 0
    benchmark_eligible_count = 0
    candidate_family_counts: dict[str, int] = {}
    claimed_promotion_by_family: dict[str, int] = {}
    integrity_verified_by_family: dict[str, int] = {}
    integrity_mismatch_counts: dict[str, int] = {}
    integrity_mismatch_count = 0
    integrity_verified_count = 0
    valid_rows = 0
    manifest_exists = path.exists()
    if not manifest_exists:
        return {
            "manifest_exists": False,
            "by_key": by_key,
            "valid_candidate_count": 0,
            "invalid_candidate_count": 0,
            "hardcoded_candidate_count": 0,
            "template_like_candidate_count": 0,
            "loop_closure_candidate_count": 0,
            "token_level_learned_candidate_count": 0,
            "compositional_token_candidate_count": 0,
            "full_body_token_candidate_count": 0,
            "grammar_masked_learned_token_candidate_count": 0,
            "expression_memory_fallback_count": 0,
            "deterministic_guardrail_failed_candidate_count": 0,
            "benchmark_promotion_eligible_candidate_count": 0,
            "benchmark_promotion_integrity_valid": False,
            "candidate_integrity_ready": False,
            "token_level_code_generation_learned": False,
            "provenance_valid": False,
            "checkpoint_ids": [],
            "candidate_sources": [],
            "candidate_generation_modes": [],
            "candidate_integrity_policy": "project_theseus_recomputed_candidate_integrity_v1",
            "candidate_family_counts": {},
            "claimed_promotion_by_family": {},
            "integrity_verified_by_family": {},
            "candidate_integrity_mismatch_count": 0,
            "candidate_integrity_mismatch_counts": {},
            "integrity_verified_candidate_count": 0,
        }
    for raw in read_jsonl(path):
        row = normalize_student_candidate(raw)
        if not row:
            invalid_count += 1
            continue
        if looks_hardcoded_origin(str(row.get("origin") or "")):
            hardcoded_count += 1
            invalid_count += 1
            continue
        if is_template_like_candidate(row):
            template_like_count += 1
        if bool(row.get("loop_closure_generated")):
            loop_closure_count += 1
        if bool(row.get("token_level_code_generation_learned")):
            token_level_count += 1
        if bool(row.get("compositional_token_candidate")):
            compositional_token_count += 1
        if bool(row.get("full_body_token_candidate")):
            full_body_token_count += 1
        if bool(row.get("grammar_masked_learned_token_candidate")):
            grammar_masked_learned_token_count += 1
        if bool(row.get("expression_memory_fallback")):
            expression_memory_fallback_count += 1
        if row.get("deterministic_guardrail_passed") is False:
            guardrail_failed_count += 1
        family = str(row.get("recomputed_candidate_family") or "unknown")
        candidate_family_counts[family] = candidate_family_counts.get(family, 0) + 1
        if row.get("self_declared_benchmark_promotion_eligible") is True:
            claimed_promotion_by_family[family] = claimed_promotion_by_family.get(family, 0) + 1
        if row.get("candidate_integrity_verified") is True:
            integrity_verified_count += 1
            integrity_verified_by_family[family] = integrity_verified_by_family.get(family, 0) + 1
        for mismatch in row.get("candidate_integrity_mismatches") or []:
            mismatch = str(mismatch)
            integrity_mismatch_count += 1
            integrity_mismatch_counts[mismatch] = integrity_mismatch_counts.get(mismatch, 0) + 1
        if benchmark_candidate_eligible(row):
            benchmark_eligible_count += 1
        source = str(row.get("candidate_source") or "")
        checkpoint_id = str(row.get("checkpoint_id") or "")
        generation_mode = str(row.get("candidate_generation_mode") or "")
        if source not in STUDENT_CANDIDATE_SOURCES or not checkpoint_id:
            invalid_count += 1
            continue
        if source not in candidate_sources:
            candidate_sources.append(source)
        if checkpoint_id not in checkpoint_ids:
            checkpoint_ids.append(checkpoint_id)
        if generation_mode and generation_mode not in candidate_generation_modes:
            candidate_generation_modes.append(generation_mode)
        valid_rows += 1
        for key in student_candidate_keys(row):
            by_key.setdefault(key, []).append(row)
    candidate_integrity_ready = (
        valid_rows > 0
        and invalid_count == 0
        and hardcoded_count == 0
        and template_like_count == 0
        and loop_closure_count == 0
        and token_level_count > 0
        and compositional_token_count > 0
        and full_body_token_count > 0
        and grammar_masked_learned_token_count > 0
        and benchmark_eligible_count > 0
        and integrity_verified_count > 0
        and integrity_mismatch_count == 0
    )
    return {
        "manifest_exists": True,
        "by_key": by_key,
        "valid_candidate_count": valid_rows,
        "invalid_candidate_count": invalid_count,
        "hardcoded_candidate_count": hardcoded_count,
        "template_like_candidate_count": template_like_count,
        "loop_closure_candidate_count": loop_closure_count,
        "token_level_learned_candidate_count": token_level_count,
        "compositional_token_candidate_count": compositional_token_count,
        "full_body_token_candidate_count": full_body_token_count,
        "grammar_masked_learned_token_candidate_count": grammar_masked_learned_token_count,
        "expression_memory_fallback_count": expression_memory_fallback_count,
        "deterministic_guardrail_failed_candidate_count": guardrail_failed_count,
        "benchmark_promotion_eligible_candidate_count": benchmark_eligible_count,
        "benchmark_promotion_integrity_valid": candidate_integrity_ready,
        "candidate_integrity_ready": candidate_integrity_ready,
        "token_level_code_generation_learned": token_level_count > 0,
        "provenance_valid": valid_rows > 0 and invalid_count == 0 and hardcoded_count == 0,
        "checkpoint_ids": checkpoint_ids,
        "candidate_sources": candidate_sources,
        "candidate_generation_modes": candidate_generation_modes,
        "candidate_integrity_policy": "project_theseus_recomputed_candidate_integrity_v1",
        "candidate_family_counts": dict(sorted(candidate_family_counts.items())),
        "claimed_promotion_by_family": dict(sorted(claimed_promotion_by_family.items())),
        "integrity_verified_by_family": dict(sorted(integrity_verified_by_family.items())),
        "candidate_integrity_mismatch_count": integrity_mismatch_count,
        "candidate_integrity_mismatch_counts": dict(sorted(integrity_mismatch_counts.items())),
        "integrity_verified_candidate_count": integrity_verified_count,
    }


def candidate_source_label(student_candidates: dict[str, Any], ready: bool) -> str:
    if not ready:
        return "missing_local_theseus_student_checkpoint_generator"
    sources = [
        str(source)
        for source in student_candidates.get("candidate_sources", [])
        if str(source)
    ]
    if len(sources) == 1:
        return sources[0]
    if "student_neural_checkpoint_v1" in sources:
        return "mixed_student_checkpoint_sources_with_student_neural"
    if "student_code_lm_checkpoint_v1" in sources:
        return "mixed_student_checkpoint_sources_with_code_lm"
    if "student_token_generator_checkpoint_v1" in sources:
        return "mixed_student_checkpoint_sources_with_student_token_generator"
    if "student_learning_checkpoint_v1" in sources:
        return "mixed_student_checkpoint_sources_with_student_learning"
    if "local_theseus_student_checkpoint" in sources:
        return "mixed_student_checkpoint_sources_with_local_student"
    return "mixed_student_checkpoint_sources"


def score_semantics_for_candidate_source(candidate_source: str) -> str:
    if candidate_source == "student_code_lm_checkpoint_v1":
        return "student_code_lm_checkpoint_public_code_calibration_pass_rate"
    if candidate_source == "student_token_generator_checkpoint_v1":
        return "student_token_generator_checkpoint_public_code_calibration_pass_rate"
    if candidate_source == "student_neural_checkpoint_v1":
        return "student_neural_checkpoint_public_code_calibration_pass_rate"
    if candidate_source == "student_learning_checkpoint_v1":
        return "student_learning_checkpoint_public_code_calibration_pass_rate"
    if candidate_source == "local_theseus_student_checkpoint":
        return "student_checkpoint_public_code_calibration_pass_rate"
    return "mixed_student_checkpoint_public_code_calibration_pass_rate"


def normalize_student_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    code = str(raw.get("code") or raw.get("candidate_code") or "")
    completion = str(raw.get("completion") or "")
    prompt = str(raw.get("prompt") or "")
    if not code and completion and prompt:
        code = prompt + completion
    if not code.strip():
        return {}
    canonical_seen = bool(raw.get("canonical_solution_seen_by_solver"))
    if canonical_seen:
        return {}
    candidate_source = str(raw.get("candidate_source") or raw.get("source") or "")
    checkpoint_id = str(raw.get("checkpoint_id") or raw.get("model_checkpoint") or "")
    origin = str(raw.get("origin") or candidate_source or "local_theseus_student_checkpoint")
    provenance = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {}
    generation_mode = str(
        raw.get("candidate_generation_mode")
        or get_path(provenance, ["candidate_generation_mode"], "")
        or ""
    )
    program_synthesis_loop = (
        raw.get("program_synthesis_loop_v1")
        if isinstance(raw.get("program_synthesis_loop_v1"), dict)
        else get_path(provenance, ["program_synthesis_loop_v1"], {})
    )
    if not isinstance(program_synthesis_loop, dict):
        program_synthesis_loop = {}
    candidate_program_scope = str(
        raw.get("candidate_program_scope")
        or get_path(provenance, ["candidate_program_scope"], "")
        or ("full_function_body" if truthy(raw.get("full_body_token_candidate")) else "")
    )
    loop_closure_generated = truthy(
        raw.get("loop_closure_generated")
        if "loop_closure_generated" in raw
        else get_path(provenance, ["loop_closure_generated"], False)
    )
    template_like = truthy(raw.get("template_like_candidate")) or generation_mode.lower() in NON_PROMOTABLE_CANDIDATE_MODES
    compositional_token_candidate = truthy(
        raw.get("compositional_token_candidate")
        if "compositional_token_candidate" in raw
        else get_path(provenance, ["compositional_token_candidate"], False)
    )
    full_body_token_candidate = truthy(
        raw.get("full_body_token_candidate")
        if "full_body_token_candidate" in raw
        else get_path(provenance, ["full_body_token_candidate"], False)
    )
    expression_memory_fallback = truthy(
        raw.get("expression_memory_fallback")
        if "expression_memory_fallback" in raw
        else get_path(provenance, ["expression_memory_fallback"], False)
    )
    deterministic_guardrail_passed = truthy(
        raw.get("deterministic_guardrail_passed")
        if "deterministic_guardrail_passed" in raw
        else get_path(provenance, ["deterministic_guardrail_passed"], True)
    )
    integrity_input = {
        **raw,
        "candidate_generation_mode": generation_mode,
        "candidate_program_scope": candidate_program_scope,
        "program_synthesis_loop_v1": program_synthesis_loop,
        "compositional_token_candidate": compositional_token_candidate,
        "full_body_token_candidate": full_body_token_candidate,
        "expression_memory_fallback": expression_memory_fallback,
        "deterministic_guardrail_passed": deterministic_guardrail_passed,
        "template_like_candidate": template_like,
    }
    candidate_integrity = recompute_candidate_integrity(integrity_input)
    grammar_masked_learned_token_candidate = is_grammar_masked_learned_token_candidate(
        {
            **integrity_input,
            "candidate_generation_mode": generation_mode,
            "candidate_program_scope": candidate_program_scope,
            "program_synthesis_loop_v1": program_synthesis_loop,
            "compositional_token_candidate": compositional_token_candidate,
            "full_body_token_candidate": full_body_token_candidate,
            "expression_memory_fallback": expression_memory_fallback,
            "deterministic_guardrail_passed": deterministic_guardrail_passed,
            "template_like_candidate": template_like,
        }
    )
    token_level_learned = (
        truthy(
            raw.get("token_level_code_generation_learned")
            if "token_level_code_generation_learned" in raw
            else get_path(provenance, ["token_level_code_generation_learned"], False)
        )
        and compositional_token_candidate
        and not expression_memory_fallback
        and deterministic_guardrail_passed
        and grammar_masked_learned_token_candidate
        and bool(candidate_integrity.get("pure_learned_generation"))
    )
    benchmark_eligible = (
        truthy(
            raw.get("benchmark_promotion_eligible")
            if "benchmark_promotion_eligible" in raw
            else get_path(provenance, ["benchmark_promotion_eligible"], False)
        )
        and token_level_learned
        and full_body_token_candidate
        and grammar_masked_learned_token_candidate
        and deterministic_guardrail_passed
        and integrity_verified(candidate_integrity)
    )
    return {
        "task_id": str(raw.get("task_id") or ""),
        "source_task_id": str(raw.get("source_task_id") or ""),
        "entry_point": str(raw.get("entry_point") or ""),
        "category": str(raw.get("category") or get_path(provenance, ["visible_task", "category"], "")),
        "candidate_source": candidate_source,
        "checkpoint_id": checkpoint_id,
        "origin": origin,
        "code": code,
        "candidate_generation_mode": generation_mode,
        "candidate_program_scope": candidate_program_scope,
        "token_level_code_generation_learned": token_level_learned,
        "benchmark_promotion_eligible": benchmark_eligible,
        "loop_closure_generated": loop_closure_generated,
        "template_like_candidate": template_like,
        "compositional_token_candidate": compositional_token_candidate,
        "full_body_token_candidate": full_body_token_candidate,
        "grammar_masked_learned_token_candidate": grammar_masked_learned_token_candidate,
        "self_declared_token_level_code_generation_learned": candidate_integrity["self_declared_flags"]["token_level_code_generation_learned"],
        "self_declared_benchmark_promotion_eligible": candidate_integrity["self_declared_flags"]["benchmark_promotion_eligible"],
        "candidate_integrity": candidate_integrity,
        "recomputed_candidate_family": candidate_integrity["recomputed_candidate_family"],
        "candidate_integrity_verified": integrity_verified(candidate_integrity),
        "candidate_integrity_mismatches": candidate_integrity["integrity_mismatches"],
        "program_synthesis_loop_v1": program_synthesis_loop,
        "sts_candidate_expression_used": raw.get("sts_candidate_expression_used") if "sts_candidate_expression_used" in raw else get_path(provenance, ["sts_candidate_expression_used"], False),
        "same_seed_non_sts_comparator": raw.get("same_seed_non_sts_comparator") if "same_seed_non_sts_comparator" in raw else get_path(provenance, ["same_seed_non_sts_comparator"], False),
        "expression_memory_fallback": expression_memory_fallback,
        "deterministic_guardrail_passed": deterministic_guardrail_passed,
        "deterministic_guardrail_reasons": raw.get("deterministic_guardrail_reasons") or get_path(provenance, ["deterministic_guardrail_reasons"], []),
        "decoder_contract_verifier_v1_passed": raw.get("decoder_contract_verifier_v1_passed") if "decoder_contract_verifier_v1_passed" in raw else get_path(provenance, ["decoder_contract_verifier_v1_passed"], None),
        "beautiful_code_score": raw.get("beautiful_code_score") if "beautiful_code_score" in raw else get_path(provenance, ["beautiful_code_score"], None),
        "placeholder_scaffold_body": raw.get("placeholder_scaffold_body") if "placeholder_scaffold_body" in raw else get_path(provenance, ["placeholder_scaffold_body"], False),
    }


def integrity_verified(integrity: dict[str, Any]) -> bool:
    return bool(integrity.get("integrity_verified", integrity.get("promotion_verified", False)))


GENERIC_ENTRY_POINT_KEYS = {
    "answer",
    "func",
    "function",
    "main",
    "solution",
    "solve",
    "task_func",
}


def student_candidate_keys(row: dict[str, Any]) -> list[str]:
    keys = []
    for field in ["task_id", "source_task_id", "entry_point"]:
        value = str(row.get(field) or "")
        if value:
            keys.append(value)
    return list(dict.fromkeys(keys))


def student_candidates_for_task(task: dict[str, Any], student_candidates: dict[str, Any]) -> list[dict[str, Any]]:
    by_key = student_candidates.get("by_key") if isinstance(student_candidates.get("by_key"), dict) else {}
    rows: list[dict[str, Any]] = []
    exact_keys = [str(task.get("task_id") or ""), str(task.get("source_task_id") or "")]
    for key in exact_keys:
        if key and key in by_key:
            rows.extend(row for row in by_key[key] if isinstance(row, dict))
    entry = str(task.get("entry_point") or "")
    if not rows and entry and entry not in GENERIC_ENTRY_POINT_KEYS and entry in by_key:
        rows.extend(row for row in by_key[entry] if isinstance(row, dict))
    deduped_by_digest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not student_candidate_identity_matches_task(task, row):
            continue
        digest = sha256_text(str(row.get("code") or ""))
        current = deduped_by_digest.get(digest)
        if current is not None and candidate_identity_preference(current) >= candidate_identity_preference(row):
            continue
        deduped_by_digest[digest] = row
    return list(deduped_by_digest.values())


def candidate_identity_preference(row: dict[str, Any]) -> tuple[float, ...]:
    """Prefer the strongest provenance when duplicate code bodies collide.

    Public calibration often carries both a same-seed comparator row and an
    STS-conditioned row with identical code. The code hash is still useful for
    avoiding repeated sandbox work, but dedupe must keep the row that remains
    eligible for benchmark calibration instead of whichever one arrived first.
    """

    return (
        1.0 if benchmark_candidate_eligible(row) else 0.0,
        1.0 if truthy(row.get("benchmark_promotion_eligible")) else 0.0,
        1.0 if truthy(row.get("grammar_masked_learned_token_candidate")) else 0.0,
        1.0 if truthy(row.get("token_level_code_generation_learned")) else 0.0,
        1.0 if truthy(row.get("full_body_token_candidate")) else 0.0,
        mode_quality_prior(str(row.get("origin") or "")),
        numeric_preference(row.get("beautiful_code_score")),
    )


def numeric_preference(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def student_candidate_identity_matches_task(task: dict[str, Any], row: dict[str, Any]) -> bool:
    """Prevent generic entry-point collisions from mixing unrelated tasks.

    BigCodeBench and several generated adapters use names like ``task_func`` or
    ``solve`` across many independent tasks. Matching on those names alone lets
    an unrelated candidate body with a different signature run against the
    current scorer. Exact task/source IDs remain the trusted route; entry-point
    fallback is allowed only when the candidate carries no conflicting identity.
    """

    task_id = str(task.get("task_id") or "")
    source_task_id = str(task.get("source_task_id") or "")
    row_task_id = str(row.get("task_id") or "")
    row_source_task_id = str(row.get("source_task_id") or "")
    if task_id and row_task_id:
        return row_task_id == task_id
    if source_task_id and row_source_task_id:
        return row_source_task_id == source_task_id
    return not row_task_id and not row_source_task_id


def looks_hardcoded_origin(origin: str) -> bool:
    lowered = origin.lower()
    blocked_tokens = [
        "canonical",
        "hardcoded",
        "public_task_pattern",
        "baseline_prompt_stub",
    ]
    return any(token in lowered for token in blocked_tokens)


def is_template_like_candidate(row: dict[str, Any]) -> bool:
    origin = str(row.get("origin") or "").lower()
    generation_mode = str(row.get("candidate_generation_mode") or "").lower()
    if truthy(row.get("template_like_candidate")):
        return True
    if generation_mode in NON_PROMOTABLE_CANDIDATE_MODES:
        return True
    return any(token in origin for token in TEMPLATE_OR_RULE_TOKENS)


def is_grammar_masked_learned_token_candidate(row: dict[str, Any]) -> bool:
    generation_mode = str(row.get("candidate_generation_mode") or "").lower()
    program_loop = row.get("program_synthesis_loop_v1") if isinstance(row.get("program_synthesis_loop_v1"), dict) else {}
    decode_control = get_path(program_loop, ["decode_control"], {})
    if not isinstance(decode_control, dict):
        decode_control = {}
    loop_shape = get_path(program_loop, ["loop_shape"], [])
    if not isinstance(loop_shape, list):
        loop_shape = []
    has_learned_mode = (
        generation_mode in LEARNED_PRIVATE_BODY_NGRAM_CANDIDATE_MODES
        or any(token in generation_mode for token in LEARNED_TOKEN_CANDIDATE_MODE_TOKENS)
    )
    has_non_learned_mode = (
        generation_mode not in LEARNED_PRIVATE_BODY_NGRAM_CANDIDATE_MODES
        and any(token in generation_mode for token in NON_LEARNED_TOKEN_CANDIDATE_MODE_TOKENS)
    )
    return bool(
        has_learned_mode
        and not has_non_learned_mode
        and not is_template_like_candidate(row)
        and str(row.get("candidate_program_scope") or "") == "full_function_body"
        and truthy(row.get("compositional_token_candidate"))
        and truthy(row.get("full_body_token_candidate"))
        and not truthy(row.get("expression_memory_fallback"))
        and not truthy(row.get("sts_candidate_expression_used"))
        and not truthy(row.get("same_seed_non_sts_comparator"))
        and truthy(row.get("deterministic_guardrail_passed"))
        and row.get("decoder_contract_verifier_v1_passed") is not False
        and get_path(program_loop, ["policy"], "") == "project_theseus_program_synthesis_loop_v1"
        and truthy(decode_control.get("constrained_token_decode"))
        and truthy(decode_control.get("parser_contract_mask"))
        and truthy(decode_control.get("exact_interface_claim"))
        and not truthy(decode_control.get("template_or_memory_fallback"))
        and (
            truthy(decode_control.get("grammar_masked_learned_token_candidate"))
            or {
                "constrained_token_decode",
                "parser_contract_mask",
            }.issubset({str(item) for item in loop_shape})
        )
    )


def benchmark_candidate_eligible(row: dict[str, Any]) -> bool:
    integrity = row.get("candidate_integrity") if isinstance(row.get("candidate_integrity"), dict) else recompute_candidate_integrity(row)
    return bool(
        truthy(row.get("benchmark_promotion_eligible"))
        and truthy(row.get("token_level_code_generation_learned"))
        and truthy(row.get("full_body_token_candidate"))
        and truthy(row.get("grammar_masked_learned_token_candidate"))
        and str(row.get("candidate_program_scope") or "") == "full_function_body"
        and truthy(row.get("deterministic_guardrail_passed"))
        and row.get("decoder_contract_verifier_v1_passed") is not False
        and not truthy(row.get("placeholder_scaffold_body"))
        and not bogus_return_attribute_body(str(row.get("code") or ""))
        and not bogus_return_local_callable_body(str(row.get("code") or ""))
        and not truthy(row.get("expression_memory_fallback"))
        and not truthy(row.get("loop_closure_generated"))
        and not is_template_like_candidate(row)
        and integrity_verified(integrity)
        and not integrity.get("integrity_mismatches")
    )


PUBLIC_TEST_RUNTIME_PRELUDE = "import math\nimport itertools\nimport functools\nimport collections\n\n"


def run_candidate(
    root: Path,
    task: dict[str, Any],
    candidate: dict[str, str],
    *,
    mode: str,
    attempt_index: int,
    verification_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    script = root / f"{safe_name(task['task_id'])}_{mode}_{attempt_index}.py"
    context = verification_context or build_task_verification_context(root, task)
    quality = cached_candidate_quality(task, candidate)
    candidate_source = PUBLIC_TEST_RUNTIME_PRELUDE + str(context.get("visible_prompt_prelude") or "") + candidate["code"] + "\n"
    tests_source = str(context.get("tests_source") or "")
    preflight = candidate_static_verification(
        candidate_source,
        tests_source,
        quality,
        test_harness_preflight=context.get("test_harness_preflight"),
    )
    if not preflight["continue"]:
        return candidate_stage_result(
            task,
            candidate,
            mode=mode,
            attempt_index=attempt_index,
            quality=quality,
            preflight=preflight,
            started=time.perf_counter(),
        )
    script.write_text(sandbox_cascade_source(candidate_source, tests_source), encoding="utf-8")
    env = context.get("runtime_env") if isinstance(context.get("runtime_env"), dict) else candidate_runtime_env(root)
    started = time.perf_counter()
    try:
        result = subprocess.run([sys.executable, str(script)], cwd=root, env=env, text=True, capture_output=True, timeout=8)
        runtime_loaded = "__THESEUS_STAGE__:runtime_loaded" in result.stdout
        intended_behavior_passed = result.returncode == 0 and "__THESEUS_STAGE__:intended_behavior_passed" in result.stdout
        test_passed = intended_behavior_passed
        passed = test_passed and bool(quality.get("beautiful_code_gate_passed"))
        reward_breakdown = dict(preflight["reward_breakdown"])
        reward_breakdown["runtime_load"] = 0.2 if runtime_loaded else 0.0
        reward_breakdown["intended_behavior"] = 0.3 if intended_behavior_passed else 0.0
        stderr_tail = result.stderr[-1200:]
        if test_passed and not passed:
            stderr_tail = (
                "beautiful_code_quality_gate_failed: "
                + ",".join(str(reason) for reason in quality.get("beautiful_code_reasons", []))
            )[-1200:]
        return {
            "task_id": task["task_id"],
            "mode": mode,
            "attempt_index": attempt_index,
            "passed": passed,
            "test_passed": test_passed,
            "beautiful_code_gate_passed": bool(quality.get("beautiful_code_gate_passed")),
            "beautiful_code_score": quality.get("beautiful_code_score"),
            "beautiful_code_reasons": quality.get("beautiful_code_reasons"),
            "lint_passed": bool(preflight.get("lint_passed")),
            "compile_passed": bool(preflight.get("compile_passed")),
            "runtime_loaded": runtime_loaded,
            "intended_behavior_passed": intended_behavior_passed,
            "verification_stage": "intended_behavior_passed" if intended_behavior_passed else ("runtime_loaded" if runtime_loaded else "runtime_failed"),
            "verification_reward": round(sum(float(value) for value in reward_breakdown.values()), 6),
            "reward_breakdown": reward_breakdown,
            "returncode": result.returncode,
            "stderr": stderr_tail,
            "stdout": result.stdout[-400:],
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "candidate_sha256": sha256_text(candidate["code"]),
            **candidate_integrity_result_fields(candidate),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "task_id": task["task_id"],
            "mode": mode,
            "attempt_index": attempt_index,
            "passed": False,
            "test_passed": False,
            "beautiful_code_gate_passed": bool(quality.get("beautiful_code_gate_passed")),
            "beautiful_code_score": quality.get("beautiful_code_score"),
            "beautiful_code_reasons": quality.get("beautiful_code_reasons"),
            "lint_passed": bool(preflight.get("lint_passed")),
            "compile_passed": bool(preflight.get("compile_passed")),
            "runtime_loaded": False,
            "intended_behavior_passed": False,
            "verification_stage": "timeout",
            "verification_reward": round(sum(float(value) for value in preflight["reward_breakdown"].values()), 6),
            "reward_breakdown": preflight["reward_breakdown"],
            "returncode": 124,
            "stderr": str(exc),
            "stdout": "",
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "candidate_sha256": sha256_text(candidate["code"]),
            **candidate_integrity_result_fields(candidate),
        }
    except OSError as exc:
        reward_breakdown = dict(preflight["reward_breakdown"])
        return {
            "task_id": task["task_id"],
            "mode": mode,
            "attempt_index": attempt_index,
            "passed": False,
            "test_passed": False,
            "beautiful_code_gate_passed": bool(quality.get("beautiful_code_gate_passed")),
            "beautiful_code_score": quality.get("beautiful_code_score"),
            "beautiful_code_reasons": quality.get("beautiful_code_reasons"),
            "lint_passed": bool(preflight.get("lint_passed")),
            "compile_passed": bool(preflight.get("compile_passed")),
            "runtime_loaded": False,
            "intended_behavior_passed": False,
            "verification_stage": "sandbox_launch_failed",
            "verification_reward": round(sum(float(value) for value in reward_breakdown.values()), 6),
            "reward_breakdown": reward_breakdown,
            "returncode": 127,
            "stderr": str(exc)[-1200:],
            "stdout": "",
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "candidate_sha256": sha256_text(candidate["code"]),
            **candidate_integrity_result_fields(candidate),
        }


def build_task_verification_context(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    tests_source = str(task.get("tests") or "")
    return {
        "tests_source": tests_source,
        "visible_prompt_prelude": visible_prompt_prelude(task),
        "test_harness_preflight": test_harness_compile_preflight(tests_source),
        "runtime_env": candidate_runtime_env(root),
    }


def cached_candidate_quality(task: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    required = {
        "beautiful_code_gate_passed",
        "beautiful_code_score",
        "beautiful_code_reasons",
    }
    if required.issubset(candidate.keys()):
        return {
            "beautiful_code_gate_passed": bool(candidate.get("beautiful_code_gate_passed")),
            "beautiful_code_score": candidate.get("beautiful_code_score"),
            "beautiful_code_reasons": candidate.get("beautiful_code_reasons") or [],
        }
    return candidate_quality(task, str(candidate.get("code") or ""), str(candidate.get("origin") or ""))


def test_harness_compile_preflight(tests_source: str) -> dict[str, Any]:
    try:
        compile(tests_source, "<tests>", "exec")
    except SyntaxError as exc:
        return {
            "compile_passed": False,
            "stage": "test_harness_compile_failed",
            "stderr": f"test_harness_compile_failed: {exc.__class__.__name__}: {exc.msg}",
            "returncode": 123,
        }
    return {
        "compile_passed": True,
        "stage": "test_harness_compile_passed",
        "stderr": "",
        "returncode": 0,
    }


def candidate_static_verification(
    candidate_source: str,
    tests_source: str,
    quality: dict[str, Any],
    *,
    test_harness_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reward_breakdown = {
        "lint_parse": 0.0,
        "beautiful_code_lint": 0.0,
        "candidate_compile": 0.0,
        "test_harness_compile": 0.0,
    }
    try:
        ast.parse(candidate_source)
    except SyntaxError as exc:
        return {
            "continue": False,
            "stage": "lint_parse_failed",
            "stderr": f"lint_parse_failed: {exc.__class__.__name__}: {exc.msg}",
            "returncode": 120,
            "reward_breakdown": reward_breakdown,
            "lint_passed": False,
            "compile_passed": False,
        }
    reward_breakdown["lint_parse"] = 0.1
    if not bool(quality.get("beautiful_code_gate_passed")):
        return {
            "continue": False,
            "stage": "beautiful_code_lint_failed",
            "stderr": "beautiful_code_lint_failed: " + ",".join(str(reason) for reason in quality.get("beautiful_code_reasons", [])),
            "returncode": 121,
            "reward_breakdown": reward_breakdown,
            "lint_passed": False,
            "compile_passed": False,
        }
    reward_breakdown["beautiful_code_lint"] = 0.15
    try:
        compile(candidate_source, "<candidate>", "exec")
    except SyntaxError as exc:
        return {
            "continue": False,
            "stage": "candidate_compile_failed",
            "stderr": f"candidate_compile_failed: {exc.__class__.__name__}: {exc.msg}",
            "returncode": 122,
            "reward_breakdown": reward_breakdown,
            "lint_passed": True,
            "compile_passed": False,
        }
    reward_breakdown["candidate_compile"] = 0.2
    missing_imports = unavailable_external_imports(candidate_source)
    if missing_imports:
        return {
            "continue": False,
            "stage": "candidate_dependency_unavailable",
            "stderr": "candidate_dependency_unavailable: " + ",".join(missing_imports[:8]),
            "returncode": 126,
            "reward_breakdown": reward_breakdown,
            "lint_passed": True,
            "compile_passed": True,
        }
    if test_harness_preflight is None:
        test_harness_preflight = test_harness_compile_preflight(tests_source)
    if not bool(test_harness_preflight.get("compile_passed")):
        return {
            "continue": False,
            "stage": "test_harness_compile_failed",
            "stderr": str(test_harness_preflight.get("stderr") or "test_harness_compile_failed"),
            "returncode": int(test_harness_preflight.get("returncode") or 123),
            "reward_breakdown": reward_breakdown,
            "lint_passed": True,
            "compile_passed": False,
        }
    reward_breakdown["test_harness_compile"] = 0.05
    return {
        "continue": True,
        "stage": "compile_passed",
        "stderr": "",
        "returncode": 0,
        "reward_breakdown": reward_breakdown,
        "lint_passed": True,
        "compile_passed": True,
    }


def candidate_stage_result(
    task: dict[str, Any],
    candidate: dict[str, str],
    *,
    mode: str,
    attempt_index: int,
    quality: dict[str, Any],
    preflight: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "mode": mode,
        "attempt_index": attempt_index,
        "passed": False,
        "test_passed": False,
        "beautiful_code_gate_passed": bool(quality.get("beautiful_code_gate_passed")),
        "beautiful_code_score": quality.get("beautiful_code_score"),
        "beautiful_code_reasons": quality.get("beautiful_code_reasons"),
        "lint_passed": bool(preflight.get("lint_passed")),
        "compile_passed": bool(preflight.get("compile_passed")),
        "runtime_loaded": False,
        "intended_behavior_passed": False,
        "verification_stage": preflight.get("stage"),
        "verification_reward": round(sum(float(value) for value in preflight["reward_breakdown"].values()), 6),
        "reward_breakdown": preflight["reward_breakdown"],
        "returncode": int(preflight.get("returncode") or 120),
        "stderr": str(preflight.get("stderr") or "")[-1200:],
        "stdout": "",
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "candidate_sha256": sha256_text(candidate["code"]),
        **candidate_integrity_result_fields(candidate),
    }


def candidate_integrity_result_fields(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "recomputed_candidate_family": candidate.get("recomputed_candidate_family") or "unknown",
        "candidate_integrity_verified": bool(candidate.get("candidate_integrity_verified")),
        "candidate_integrity_mismatches": candidate.get("candidate_integrity_mismatches") or [],
    }


def sandbox_cascade_source(candidate_source: str, tests_source: str) -> str:
    return (
        "import sys\n"
        "import traceback\n"
        f"candidate_source = {candidate_source!r}\n"
        f"tests_source = {tests_source!r}\n"
        "namespace = {}\n"
        "try:\n"
        "    exec(compile(candidate_source, '<candidate>', 'exec'), namespace)\n"
        "    print('__THESEUS_STAGE__:runtime_loaded', flush=True)\n"
        "except BaseException:\n"
        "    print('__THESEUS_STAGE__:runtime_failed', file=sys.stderr, flush=True)\n"
        "    traceback.print_exc()\n"
        "    sys.exit(20)\n"
        "try:\n"
        "    exec(compile(tests_source, '<tests>', 'exec'), namespace)\n"
        "    print('__THESEUS_STAGE__:intended_behavior_passed', flush=True)\n"
        "except BaseException:\n"
        "    print('__THESEUS_STAGE__:intended_behavior_failed', file=sys.stderr, flush=True)\n"
        "    traceback.print_exc()\n"
        "    sys.exit(30)\n"
    )


def candidate_runtime_env(root: Path) -> dict[str, str]:
    """Prepare a portable local runtime envelope for public calibration tests.

    Some benchmark tasks use POSIX-style temporary paths such as ``/tmp/foo``
    even when the calibration runner is executing on Windows. Python resolves
    those paths against the active drive, so a missing drive-root ``tmp``
    directory can fail the test harness before the candidate behavior is
    evaluated. Creating the directory is a sandbox compatibility fix; it does
    not expose public answers or change candidate code.
    """

    env = os.environ.copy()
    local_tmp = root / "tmp"
    local_tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(local_tmp)
    env["TMP"] = str(local_tmp)
    env["TEMP"] = str(local_tmp)
    if sys.platform == "win32":
        drive = root.drive or Path.cwd().drive
        if drive:
            try:
                posix_tmp = Path(drive + "\\tmp")
                posix_tmp.mkdir(parents=True, exist_ok=True)
                env["THESEUS_WINDOWS_POSIX_TMP"] = str(posix_tmp)
            except OSError:
                env["THESEUS_WINDOWS_POSIX_TMP"] = ""
    return env


def visible_prompt_prelude(task: dict[str, Any]) -> str:
    """Keep public prompt helper definitions visible without exposing solutions.

    HumanEval prompts sometimes define helper functions before the target
    function. The student still supplies the target body; this only preserves
    visible prompt context that official-ish tests may reference.
    """

    prompt = str(task.get("prompt") or "")
    entry = str(task.get("entry_point") or "")
    if not prompt.strip() or not entry.strip():
        return ""
    target_def = re.compile(rf"^\s*def\s+{re.escape(entry)}\s*\(")
    lines: list[str] = []
    for line in prompt.splitlines():
        if target_def.match(line):
            break
        lines.append(line)
    prelude = "\n".join(lines).strip()
    if not prelude:
        return ""
    return prelude + "\n\n"


def trace_event(task: dict[str, Any], result: dict[str, Any], *, mode: str, candidate_origin: str) -> dict[str, Any]:
    return {
        "event": "real_code_candidate_test",
        "created_utc": now(),
        "task_id": task["task_id"],
        "source_task_id": task.get("source_task_id"),
        "case_type": task.get("case_type"),
        "mode": mode,
        "streams": STREAMS if mode == "multi_stream" else ["single_sequence"],
        "attempt_index": result.get("attempt_index"),
        "candidate_origin": candidate_origin,
        "passed": result.get("passed"),
        "test_passed": result.get("test_passed"),
        "beautiful_code_gate_passed": result.get("beautiful_code_gate_passed"),
        "beautiful_code_score": result.get("beautiful_code_score"),
        "beautiful_code_reasons": result.get("beautiful_code_reasons"),
        "lint_passed": result.get("lint_passed"),
        "compile_passed": result.get("compile_passed"),
        "runtime_loaded": result.get("runtime_loaded"),
        "intended_behavior_passed": result.get("intended_behavior_passed"),
        "verification_stage": result.get("verification_stage"),
        "verification_reward": result.get("verification_reward"),
        "reward_breakdown": result.get("reward_breakdown"),
        "reward_semantics": "public_calibration_reward_not_training_row",
        "returncode": result.get("returncode"),
        "runtime_ms": result.get("runtime_ms"),
        "residual_class": "" if result.get("passed") else classify_failure(result.get("stderr", "")),
        "candidate_sha256": result.get("candidate_sha256"),
        "recomputed_candidate_family": result.get("recomputed_candidate_family"),
        "candidate_integrity_verified": result.get("candidate_integrity_verified"),
        "candidate_integrity_mismatches": result.get("candidate_integrity_mismatches") or [],
        "canonical_solution_seen_by_solver": False,
    }
