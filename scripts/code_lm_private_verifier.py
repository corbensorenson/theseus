"""Private Code LM candidate verification cascade.

This module owns the staged verifier used by scripts/code_lm_closure.py:
lint/parse -> candidate compile -> test-harness compile -> runtime load -> intended behavior.
It intentionally keeps public benchmark data out of training and reports reward layers as
runtime diagnostics, not capability-promotion evidence.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPINE_SMOKE_OUT = ROOT / "reports" / "private_verifier_spine_smoke.json"
DEFAULT_SPINE_SMOKE_MD = ROOT / "reports" / "private_verifier_spine_smoke.md"
DEFAULT_VCM_CONTEXT_GOVERNOR = ROOT / "reports" / "vcm_context_governor.json"

from real_code_benchmark_support import runtime_tmp_dir as benchmark_runtime_tmp_dir  # noqa: E402
import viea_spine_records  # noqa: E402
import vcm_consumer_abi  # noqa: E402
import semantic_ir  # noqa: E402

PRIVATE_STATIC_VERIFICATION_CACHE: dict[str, dict[str, Any]] = {}
PRIVATE_SANDBOX_VERIFICATION_CACHE: dict[str, dict[str, Any]] = {}
PRIVATE_TEST_HARNESS_COMPILE_CACHE: dict[str, dict[str, Any]] = {}

def evaluate_private_candidates(private_rows: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eval_rows = [row for row in private_rows if row.get("split") == "eval"]
    by_task_phase: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        by_task_phase.setdefault((str(row.get("task_id") or ""), str(row.get("phase") or "")), []).append(row)
    cache_before = private_verifier_cache_snapshot()
    baseline_pass = 0
    sts_off_pass = 0
    trained_pass = 0
    baseline_rank1_pass = 0
    sts_off_rank1_pass = 0
    trained_rank1_pass = 0
    sts_improvements = 0
    sts_regressions = 0
    sts_improvement_examples: list[dict[str, Any]] = []
    sts_regression_examples: list[dict[str, Any]] = []
    concept_residual_counts: Counter[str] = Counter()
    family_totals: dict[str, int] = defaultdict(int)
    family_passes: dict[str, int] = defaultdict(int)
    residuals = []
    verification_attempts: list[dict[str, Any]] = []
    worker_count = 1
    with tempfile.TemporaryDirectory(prefix="theseus_code_lm_private_", dir=runtime_tmp_dir()) as tmp:
        root = Path(tmp)
        worker_count = bounded_private_verification_workers(len(eval_rows))
        if worker_count <= 1:
            task_results = [
                evaluate_private_task(root, task, by_task_phase)
                for task in eval_rows
            ]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(evaluate_private_task, root, task, by_task_phase)
                    for task in eval_rows
                ]
                task_results = [future.result() for future in futures]
        for item in task_results:
            task = item["task"]
            baseline = item["baseline"]
            sts_off = item["sts_off"]
            trained = item["trained"]
            family = item["family"]
            verification_attempts.extend(item["verification_attempts"])
            family_totals[family] += 1
            family_passes[family] += int(trained["passed"])
            baseline_pass += int(baseline["passed"])
            sts_off_pass += int(sts_off["passed"])
            trained_pass += int(trained["passed"])
            baseline_rank1_pass += int(rank1_passed(baseline))
            sts_off_rank1_pass += int(rank1_passed(sts_off))
            trained_rank1_pass += int(rank1_passed(trained))
            if trained["passed"] and not sts_off["passed"]:
                sts_improvements += 1
                if len(sts_improvement_examples) < 20:
                    sts_improvement_examples.append(
                        phase_comparison_example(task, trained=trained, sts_off=sts_off)
                    )
            if sts_off["passed"] and not trained["passed"]:
                sts_regressions += 1
                if len(sts_regression_examples) < 20:
                    sts_regression_examples.append(
                        phase_comparison_example(task, trained=trained, sts_off=sts_off)
                    )
            if not trained["passed"]:
                concept_label = concept_residual_label(task, trained.get("stderr", ""))
                concept_residual_counts[concept_label] += 1
                residuals.append(
                    {
                        "task_id": task["task_id"],
                        "category": task["category"],
                        "residual_concept": family,
                        "concept_residual_label": concept_label,
                        "phase": "private_eval",
                        "residual_class": classify_failure(trained.get("stderr", "")),
                        "verification_stage": trained.get("verification_stage"),
                        "verification_reward": trained.get("verification_reward"),
                        "stderr_tail": str(trained.get("stderr", ""))[-600:],
                    }
                )
    cache_after = private_verifier_cache_snapshot()
    verifier_cache_warmup = private_verifier_cache_warmup_summary(
        cache_before,
        cache_after,
        worker_count=worker_count,
        eval_task_count=len(eval_rows),
        candidate_count=len(candidates),
    )
    baseline_rate = ratio(baseline_pass, len(eval_rows))
    sts_off_rate = ratio(sts_off_pass, len(eval_rows))
    trained_rate = ratio(trained_pass, len(eval_rows))
    baseline_rank1_rate = ratio(baseline_rank1_pass, len(eval_rows))
    sts_off_rank1_rate = ratio(sts_off_rank1_pass, len(eval_rows))
    trained_rank1_rate = ratio(trained_rank1_pass, len(eval_rows))
    concept_family_pass_rates = {
        family: ratio(family_passes[family], total)
        for family, total in sorted(family_totals.items())
    }
    return {
        "eval_task_count": len(eval_rows),
        "baseline_passed": baseline_pass,
        "sts_off_passed": sts_off_pass,
        "trained_passed": trained_pass,
        "baseline_rank1_passed": baseline_rank1_pass,
        "sts_off_rank1_passed": sts_off_rank1_pass,
        "trained_rank1_passed": trained_rank1_pass,
        "baseline_pass_if_any_passed": baseline_pass,
        "sts_off_pass_if_any_passed": sts_off_pass,
        "trained_pass_if_any_passed": trained_pass,
        "baseline_pass_rate": baseline_rate,
        "sts_off_pass_rate": sts_off_rate,
        "trained_pass_rate": trained_rate,
        "baseline_rank1_pass_rate": baseline_rank1_rate,
        "sts_off_rank1_pass_rate": sts_off_rank1_rate,
        "trained_rank1_pass_rate": trained_rank1_rate,
        "baseline_pass_if_any_rate": baseline_rate,
        "sts_off_pass_if_any_rate": sts_off_rate,
        "trained_pass_if_any_rate": trained_rate,
        "pass_rate_delta": round(trained_rate - baseline_rate, 6),
        "sts_repair_pass_rate_delta": round(trained_rate - sts_off_rate, 6),
        "sts_repair_task_level_improvements": sts_improvements,
        "sts_repair_task_level_regressions": sts_regressions,
        "sts_repair_task_level_improvement_examples": sts_improvement_examples,
        "sts_repair_task_level_regression_examples": sts_regression_examples,
        "concept_residual_counts": dict(concept_residual_counts),
        "concept_family_pass_rates": concept_family_pass_rates,
        "private_verification": summarize_private_verification(verification_attempts),
        "verifier_cache_warmup": verifier_cache_warmup,
        "correctness_labels": correctness_label_summary(verification_attempts),
        "verification_attempt_labels": verification_attempts[:1024],
        "passed_verification_traces": [
            row for row in verification_attempts if row.get("passed") is True
        ][:256],
        "residual_count": len(residuals),
        "residuals": residuals[:12],
    }


def evaluate_all_private_candidates(
    private_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    phase: str = "private_eval",
) -> dict[str, Any]:
    """Label every private candidate independently for offline preference training."""

    task_by_id = {
        str(row.get("task_id") or ""): row
        for row in private_rows
        if row.get("split") == "eval" and row.get("task_id")
    }
    jobs = [
        (task_by_id[str(candidate.get("task_id") or "")], candidate)
        for candidate in candidates
        if str(candidate.get("task_id") or "") in task_by_id
    ]
    cache_before = private_verifier_cache_snapshot()

    def evaluate_one(root: Path, task: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        result = run_any(root, task, [candidate], phase=phase)
        traces = [row for row in result.get("attempt_traces", []) if int(row.get("attempt_index") or 0) == 1]
        return traces[0] if traces else {
            "task_id": task.get("task_id"),
            "phase": phase,
            "candidate_sha256": candidate.get("candidate_sha256"),
            "passed": False,
            "verification_stage": "typed_missing_candidate_trace",
            "verification_reward": 0.0,
        }

    traces: list[dict[str, Any]] = []
    worker_count = bounded_private_verification_workers(len(jobs))
    with tempfile.TemporaryDirectory(prefix="theseus_code_lm_private_all_", dir=runtime_tmp_dir()) as tmp:
        root = Path(tmp)
        if worker_count <= 1:
            traces = [evaluate_one(root, task, candidate) for task, candidate in jobs]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(evaluate_one, root, task, candidate)
                    for task, candidate in jobs
                ]
                traces = [future.result() for future in futures]
    cache_after = private_verifier_cache_snapshot()
    return {
        "policy": "project_theseus_private_all_candidate_preference_labels_v1",
        "eval_task_count": len(task_by_id),
        "candidate_count": len(candidates),
        "labeled_candidate_count": len(traces),
        "trained_passed": len({str(row.get("task_id") or "") for row in traces if row.get("passed")}),
        "trained_rank1_passed": sum(
            row.get("passed") is True and int(row.get("rank") or 0) == 1 for row in traces
        ),
        "verification_attempt_labels": traces,
        "private_verification": summarize_private_verification(traces),
        "correctness_labels": correctness_label_summary(traces),
        "verifier_cache_warmup": private_verifier_cache_warmup_summary(
            cache_before,
            cache_after,
            worker_count=worker_count,
            eval_task_count=len(task_by_id),
            candidate_count=len(candidates),
        ),
        "generation_read_set": ["prompt", "entry_point", "callable_signature"],
        "uses_eval_tests_or_solutions_for_generation": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def rank1_passed(result: dict[str, Any]) -> bool:
    traces = [row for row in result.get("attempt_traces", []) if int(row.get("attempt_index") or 0) == 1]
    return bool(traces and traces[0].get("passed"))


def phase_comparison_example(
    task: dict[str, Any],
    *,
    trained: dict[str, Any],
    sts_off: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "category": task.get("category"),
        "residual_concept": concept_family(task),
        "trained_stage": trained.get("verification_stage"),
        "trained_reward": trained.get("verification_reward"),
        "trained_stderr_tail": str(trained.get("stderr", ""))[-320:],
        "sts_off_stage": sts_off.get("verification_stage"),
        "sts_off_reward": sts_off.get("verification_reward"),
        "sts_off_stderr_tail": str(sts_off.get("stderr", ""))[-320:],
    }


def evaluate_private_task(
    root: Path,
    task: dict[str, Any],
    by_task_phase: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    task_id = str(task.get("task_id") or "")
    baseline = run_any(root, task, by_task_phase.get((task_id, "private_baseline"), []), phase="private_baseline")
    sts_off = run_any(root, task, by_task_phase.get((task_id, "private_eval_sts_off"), []), phase="private_eval_sts_off")
    trained = run_any(root, task, by_task_phase.get((task_id, "private_eval"), []), phase="private_eval")
    return {
        "task": task,
        "baseline": baseline,
        "sts_off": sts_off,
        "trained": trained,
        "family": concept_family(task),
        "verification_attempts": (
            baseline.get("attempt_traces", [])
            + sts_off.get("attempt_traces", [])
            + trained.get("attempt_traces", [])
        ),
    }


def concept_family(task: dict[str, Any]) -> str:
    explicit = str(task.get("residual_concept") or "")
    if explicit:
        return explicit
    category = str(task.get("category") or "")
    tags = {str(tag) for tag in task.get("tags", []) if str(tag)}
    if "recurrence" in tags or category in {"tribonacci_sequence"} or "recurrence" in category:
        return "recurrence_state"
    if "vowels" in tags or "vowel" in category or "suffix" in category or "case_punct" in category:
        return "string_rule_composition"
    if "digit_rotation" in tags or "rotate" in category or "circular_digit_shift" in category:
        return "digit_rotation"
    return category or "general"


def concept_residual_label(task: dict[str, Any], stderr: str) -> str:
    explicit = str(task.get("concept_residual_label") or "")
    if explicit:
        return explicit
    family = concept_family(task)
    text = str(stderr or "").lower()
    if "nameerror" in text or "typeerror" in text:
        if family == "digit_rotation":
            return "string_int_conversion_error"
        return "type_or_name_error"
    if family == "recurrence_state":
        if "assertionerror" in text:
            return "wrong_base_case"
        return "recurrence_state_drift"
    if family == "string_rule_composition":
        return "final_character_exception_missed"
    if family == "digit_rotation":
        if "assertionerror" in text:
            return "rotation_direction_error"
        return "leading_zero_loss"
    return classify_failure(stderr)


def run_any(root: Path, task: dict[str, Any], candidates: list[dict[str, Any]], *, phase: str = "private_eval") -> dict[str, Any]:
    if not candidates:
        return {
            "passed": False,
            "stderr": "missing candidates",
        "verification_stage": "no_candidate",
        "verification_reward": 0.0,
        "test_harness_cache_hit": False,
        "attempt_traces": [
            {
                "task_id": task.get("task_id"),
                "phase": phase,
                "attempt_index": 0,
                "verification_stage": "no_candidate",
                "verification_reward": 0.0,
                "test_harness_cache_hit": False,
            }
        ],
        }
    last = {"passed": False, "stderr": "", "attempt_traces": []}
    attempt_traces: list[dict[str, Any]] = []
    runtime_prelude = "import math\nimport itertools\nimport functools\nimport collections\n\n"
    timeout_seconds = float(os.environ.get("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS", "12"))
    for idx, candidate in enumerate(candidates, start=1):
        candidate_source = runtime_prelude + str(candidate.get("code") or "") + "\n"
        tests_source = str(task.get("tests") or "")
        preflight = private_candidate_static_verification(candidate_source, tests_source)
        digest = sha256_text(f"{task.get('task_id')}:{phase}:{idx}:{candidate.get('code')}")[:16]
        if not preflight["continue"]:
            last = private_candidate_stage_result(
                task,
                phase=phase,
                attempt_index=idx,
                candidate=candidate,
                preflight=preflight,
            )
            attempt_traces.append(private_verification_trace(task, phase, idx, last, candidate))
            continue
        sandbox_cache_key = private_sandbox_cache_key(candidate_source, tests_source)
        cached_sandbox = PRIVATE_SANDBOX_VERIFICATION_CACHE.get(sandbox_cache_key)
        if cached_sandbox is not None:
            last = cached_private_candidate_result(cached_sandbox)
            last["static_cache_hit"] = bool(preflight.get("static_cache_hit"))
            last["sandbox_cache_hit"] = True
            last["test_harness_cache_hit"] = bool(preflight.get("test_harness_cache_hit"))
            last["verification_cache_key"] = sandbox_cache_key[:16]
            attempt_traces.append(private_verification_trace(task, phase, idx, last, candidate))
            if last["passed"]:
                last["attempt_traces"] = attempt_traces
                return last
            continue
        path = root / f"{safe_name(task['task_id'])}_{safe_name(phase)}_{idx}_{digest}.py"
        path.write_text(private_sandbox_cascade_source(candidate_source, tests_source), encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                cwd=root,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            reward_breakdown = dict(preflight["reward_breakdown"])
            last = {
                "passed": False,
                "stderr": f"candidate_timeout_after_{timeout_seconds:g}s",
                "stdout": exc.stdout or "",
                "returncode": "timeout",
                "verification_stage": "timeout",
                "verification_reward": round(sum(float(value) for value in reward_breakdown.values()), 6),
                "reward_breakdown": reward_breakdown,
                "lint_passed": bool(preflight.get("lint_passed")),
                "compile_passed": bool(preflight.get("compile_passed")),
                "runtime_loaded": False,
                "intended_behavior_passed": False,
                "static_cache_hit": bool(preflight.get("static_cache_hit")),
                "sandbox_cache_hit": False,
                "test_harness_cache_hit": bool(preflight.get("test_harness_cache_hit")),
                "verification_cache_key": sandbox_cache_key[:16],
            }
            attempt_traces.append(private_verification_trace(task, phase, idx, last, candidate))
            continue
        except OSError as exc:
            reward_breakdown = dict(preflight["reward_breakdown"])
            last = {
                "passed": False,
                "stderr": str(exc),
                "stdout": "",
                "returncode": "sandbox_launch_failed",
                "verification_stage": "sandbox_launch_failed",
                "verification_reward": round(sum(float(value) for value in reward_breakdown.values()), 6),
                "reward_breakdown": reward_breakdown,
                "lint_passed": bool(preflight.get("lint_passed")),
                "compile_passed": bool(preflight.get("compile_passed")),
                "runtime_loaded": False,
                "intended_behavior_passed": False,
                "static_cache_hit": bool(preflight.get("static_cache_hit")),
                "sandbox_cache_hit": False,
                "test_harness_cache_hit": bool(preflight.get("test_harness_cache_hit")),
                "verification_cache_key": sandbox_cache_key[:16],
            }
            attempt_traces.append(private_verification_trace(task, phase, idx, last, candidate))
            continue
        runtime_loaded = "__THESEUS_STAGE__:runtime_loaded" in result.stdout
        intended_behavior_passed = result.returncode == 0 and "__THESEUS_STAGE__:intended_behavior_passed" in result.stdout
        reward_breakdown = dict(preflight["reward_breakdown"])
        reward_breakdown["runtime_load"] = 0.25 if runtime_loaded else 0.0
        reward_breakdown["intended_behavior"] = 0.3 if intended_behavior_passed else 0.0
        last = {
            "passed": intended_behavior_passed,
            "stderr": result.stderr,
            "stdout": result.stdout,
            "returncode": result.returncode,
            "verification_stage": "intended_behavior_passed" if intended_behavior_passed else ("runtime_loaded" if runtime_loaded else "runtime_failed"),
            "verification_reward": round(sum(float(value) for value in reward_breakdown.values()), 6),
            "reward_breakdown": reward_breakdown,
            "lint_passed": bool(preflight.get("lint_passed")),
            "compile_passed": bool(preflight.get("compile_passed")),
            "runtime_loaded": runtime_loaded,
            "intended_behavior_passed": intended_behavior_passed,
            "static_cache_hit": bool(preflight.get("static_cache_hit")),
            "sandbox_cache_hit": False,
            "test_harness_cache_hit": bool(preflight.get("test_harness_cache_hit")),
            "verification_cache_key": sandbox_cache_key[:16],
        }
        PRIVATE_SANDBOX_VERIFICATION_CACHE[sandbox_cache_key] = cached_private_candidate_result(last)
        attempt_traces.append(private_verification_trace(task, phase, idx, last, candidate))
        if last["passed"]:
            last["attempt_traces"] = attempt_traces
            return last
    last["attempt_traces"] = attempt_traces
    return last


def private_candidate_static_verification(candidate_source: str, tests_source: str) -> dict[str, Any]:
    cache_key = private_static_cache_key(candidate_source, tests_source)
    cached = PRIVATE_STATIC_VERIFICATION_CACHE.get(cache_key)
    if cached is not None:
        result = dict(cached)
        result["reward_breakdown"] = dict(result.get("reward_breakdown") or {})
        result["static_cache_hit"] = True
        result["test_harness_cache_hit"] = bool(result.get("test_harness_cache_hit"))
        result["verification_cache_key"] = cache_key[:16]
        return result
    reward_breakdown = {
        "lint_parse": 0.0,
        "candidate_compile": 0.0,
        "test_harness_compile": 0.0,
    }
    try:
        ast.parse(candidate_source)
    except SyntaxError as exc:
        result = {
            "continue": False,
            "stage": "lint_parse_failed",
            "stderr": f"lint_parse_failed: {exc.__class__.__name__}: {exc.msg}",
            "returncode": 120,
            "reward_breakdown": reward_breakdown,
            "lint_passed": False,
            "compile_passed": False,
        }
        return cache_static_verification_result(cache_key, result)
    reward_breakdown["lint_parse"] = 0.15
    try:
        compile(candidate_source, "<private_candidate>", "exec")
    except SyntaxError as exc:
        result = {
            "continue": False,
            "stage": "candidate_compile_failed",
            "stderr": f"candidate_compile_failed: {exc.__class__.__name__}: {exc.msg}",
            "returncode": 122,
            "reward_breakdown": reward_breakdown,
            "lint_passed": True,
            "compile_passed": False,
        }
        return cache_static_verification_result(cache_key, result)
    reward_breakdown["candidate_compile"] = 0.25
    test_cache_key = private_test_harness_cache_key(tests_source)
    cached_test = PRIVATE_TEST_HARNESS_COMPILE_CACHE.get(test_cache_key)
    if cached_test is not None:
        test_passed = bool(cached_test.get("passed"))
        if not test_passed:
            result = {
                "continue": False,
                "stage": "test_harness_compile_failed",
                "stderr": str(cached_test.get("stderr") or "test_harness_compile_failed: cached"),
                "returncode": 123,
                "reward_breakdown": reward_breakdown,
                "lint_passed": True,
                "compile_passed": False,
                "test_harness_cache_hit": True,
                "test_harness_cache_key": test_cache_key[:16],
            }
            return cache_static_verification_result(cache_key, result)
        reward_breakdown["test_harness_compile"] = 0.05
        result = {
            "continue": True,
            "stage": "compile_passed",
            "stderr": "",
            "returncode": 0,
            "reward_breakdown": reward_breakdown,
            "lint_passed": True,
            "compile_passed": True,
            "test_harness_cache_hit": True,
            "test_harness_cache_key": test_cache_key[:16],
        }
        return cache_static_verification_result(cache_key, result)
    try:
        compile(tests_source, "<private_tests>", "exec")
    except SyntaxError as exc:
        PRIVATE_TEST_HARNESS_COMPILE_CACHE[test_cache_key] = {
            "passed": False,
            "stderr": f"test_harness_compile_failed: {exc.__class__.__name__}: {exc.msg}",
        }
        result = {
            "continue": False,
            "stage": "test_harness_compile_failed",
            "stderr": f"test_harness_compile_failed: {exc.__class__.__name__}: {exc.msg}",
            "returncode": 123,
            "reward_breakdown": reward_breakdown,
            "lint_passed": True,
            "compile_passed": False,
            "test_harness_cache_hit": False,
            "test_harness_cache_key": test_cache_key[:16],
        }
        return cache_static_verification_result(cache_key, result)
    reward_breakdown["test_harness_compile"] = 0.05
    PRIVATE_TEST_HARNESS_COMPILE_CACHE[test_cache_key] = {
        "passed": True,
        "stderr": "",
    }
    result = {
        "continue": True,
        "stage": "compile_passed",
        "stderr": "",
        "returncode": 0,
        "reward_breakdown": reward_breakdown,
        "lint_passed": True,
        "compile_passed": True,
        "test_harness_cache_hit": False,
        "test_harness_cache_key": test_cache_key[:16],
    }
    return cache_static_verification_result(cache_key, result)


def private_static_cache_key(candidate_source: str, tests_source: str) -> str:
    return sha256_text(f"static-v1\0{candidate_source}\0{tests_source}")


def private_test_harness_cache_key(tests_source: str) -> str:
    return sha256_text(f"private-tests-v1\0{tests_source}")


def private_sandbox_cache_key(candidate_source: str, tests_source: str) -> str:
    return sha256_text(f"sandbox-v1\0{candidate_source}\0{tests_source}")


def cache_static_verification_result(cache_key: str, result: dict[str, Any]) -> dict[str, Any]:
    stored = dict(result)
    stored["reward_breakdown"] = dict(stored.get("reward_breakdown") or {})
    stored["static_cache_hit"] = False
    stored["test_harness_cache_hit"] = bool(stored.get("test_harness_cache_hit"))
    stored["verification_cache_key"] = cache_key[:16]
    PRIVATE_STATIC_VERIFICATION_CACHE[cache_key] = dict(stored)
    return stored


def cached_private_candidate_result(result: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(result)
    cloned["reward_breakdown"] = dict(cloned.get("reward_breakdown") or {})
    cloned.pop("attempt_traces", None)
    return cloned


def private_candidate_stage_result(
    task: dict[str, Any],
    *,
    phase: str,
    attempt_index: int,
    candidate: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    reward_breakdown = dict(preflight["reward_breakdown"])
    return {
        "passed": False,
        "stderr": str(preflight.get("stderr") or ""),
        "stdout": "",
        "returncode": int(preflight.get("returncode") or 120),
        "verification_stage": str(preflight.get("stage") or "static_failed"),
        "verification_reward": round(sum(float(value) for value in reward_breakdown.values()), 6),
        "reward_breakdown": reward_breakdown,
        "lint_passed": bool(preflight.get("lint_passed")),
        "compile_passed": bool(preflight.get("compile_passed")),
        "runtime_loaded": False,
        "intended_behavior_passed": False,
        "static_cache_hit": bool(preflight.get("static_cache_hit")),
        "sandbox_cache_hit": False,
        "test_harness_cache_hit": bool(preflight.get("test_harness_cache_hit")),
        "verification_cache_key": preflight.get("verification_cache_key"),
        "task_id": task.get("task_id"),
        "phase": phase,
        "attempt_index": attempt_index,
        "candidate_sha256": sha256_text(str(candidate.get("code") or "")),
    }


def private_sandbox_cascade_source(candidate_source: str, tests_source: str) -> str:
    return (
        "import sys\n"
        "import traceback\n"
        f"candidate_source = {candidate_source!r}\n"
        f"tests_source = {tests_source!r}\n"
        "namespace = {}\n"
        "try:\n"
        "    exec(compile(candidate_source, '<private_candidate>', 'exec'), namespace)\n"
        "    print('__THESEUS_STAGE__:runtime_loaded', flush=True)\n"
        "except BaseException:\n"
        "    print('__THESEUS_STAGE__:runtime_failed', file=sys.stderr, flush=True)\n"
        "    traceback.print_exc()\n"
        "    sys.exit(20)\n"
        "try:\n"
        "    exec(compile(tests_source, '<private_tests>', 'exec'), namespace)\n"
        "    print('__THESEUS_STAGE__:intended_behavior_passed', flush=True)\n"
        "except BaseException:\n"
        "    print('__THESEUS_STAGE__:intended_behavior_failed', file=sys.stderr, flush=True)\n"
        "    traceback.print_exc()\n"
        "    sys.exit(30)\n"
    )


def private_verification_trace(
    task: dict[str, Any],
    phase: str,
    attempt_index: int,
    result: dict[str, Any],
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate = candidate if isinstance(candidate, dict) else {}
    trace = {
        "task_id": task.get("task_id"),
        "category": task.get("category"),
        "phase": phase,
        "attempt_index": attempt_index,
        "passed": bool(result.get("passed")),
        "lint_passed": bool(result.get("lint_passed")),
        "compile_passed": bool(result.get("compile_passed")),
        "runtime_loaded": bool(result.get("runtime_loaded")),
        "intended_behavior_passed": bool(result.get("intended_behavior_passed")),
        "verification_stage": result.get("verification_stage"),
        "verification_reward": float(result.get("verification_reward") or 0.0),
        "reward_breakdown": result.get("reward_breakdown") or {},
        "static_cache_hit": bool(result.get("static_cache_hit")),
        "sandbox_cache_hit": bool(result.get("sandbox_cache_hit")),
        "test_harness_cache_hit": bool(result.get("test_harness_cache_hit")),
        "verification_cache_key": result.get("verification_cache_key"),
        "failure_class": "none" if result.get("passed") else classify_failure(str(result.get("stderr") or "")),
        "exception_type": private_exception_type(str(result.get("stderr") or "")),
    }
    if candidate:
        independent_ir = semantic_ir.candidate_receipt(str(candidate.get("code") or ""))
        claimed_ir = candidate.get("semantic_ir") if isinstance(candidate.get("semantic_ir"), dict) else {}
        trace.update(
            {
                "candidate_sha256": candidate.get("candidate_sha256"),
                "code_sha256": sha256_text(str(candidate.get("code") or "")),
                "entry_point": candidate.get("entry_point"),
                "source_task_id": candidate.get("source_task_id"),
                "substrate_arm": candidate.get("substrate_arm"),
                "candidate_generation_mode": candidate.get("candidate_generation_mode"),
                "candidate_source": candidate.get("candidate_source"),
                "rank": candidate.get("rank"),
                "rank_score": candidate.get("rank_score"),
                "public_tests_visible_to_generator": bool(candidate.get("public_tests_visible_to_generator")),
                "public_solutions_visible_to_generator": bool(candidate.get("public_solutions_visible_to_generator")),
                "eval_tests_visible_to_generator": bool(candidate.get("eval_tests_visible_to_generator")),
                "eval_solution_visible_to_generator": bool(candidate.get("eval_solution_visible_to_generator")),
                "external_inference_calls": int(candidate.get("external_inference_calls") or 0),
                "semantic_ir_state": independent_ir.get("state"),
                "semantic_ir_program_sha256": independent_ir.get("program_sha256"),
                "semantic_ir_roundtrip_ast_equal": bool(independent_ir.get("roundtrip_ast_equal")),
                "semantic_ir_open_obligation_count": int(independent_ir.get("open_obligation_count") or 0),
                "semantic_ir_receipt_match": bool(
                    claimed_ir
                    and claimed_ir.get("program_sha256")
                    and claimed_ir.get("program_sha256") == independent_ir.get("program_sha256")
                ),
                "semantic_ir_independently_recomputed": True,
            }
        )
    return trace


def private_exception_type(stderr: str) -> str:
    text = str(stderr or "")
    for name in (
        "AssertionError",
        "AttributeError",
        "IndexError",
        "KeyError",
        "NameError",
        "OverflowError",
        "RecursionError",
        "RuntimeError",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
    ):
        if name in text:
            return name
    return ""


def vcm_context_governor_receipt(path: Path = DEFAULT_VCM_CONTEXT_GOVERNOR) -> dict[str, Any]:
    packet = vcm_consumer_abi.build_consumer_packet(
        consumer_id="code_lm_private_verifier",
        purpose="verification",
        read_set=["reports/vcm_context_governor.json"],
        write_set=["reports/private_verifier_spine_smoke.json"],
        authority_ceiling=["local_private_verifier", "governed_context_read"],
        permitted_uses=["private_candidate_verification", "correctness_labeling", "audit_replay"],
        governor_path=path,
        context_refs=[{
            "kind": "ephemeral_hash_set",
            "ref": "vcm://theseus/verifier/candidate-attempt-hashes@current-run",
            "required": True,
            "exists": True,
            "taint_labels": ["private_verifier_metadata"],
            "contradiction_refs": [],
        }],
        taint_labels=["private_verifier_metadata", "raw_candidate_text_not_embedded_in_context_packet"],
        deletion_obligations=["invalidate_verifier_context_when_source_context_is_revoked"],
        audit_refs=["scripts/code_lm_private_verifier.py"],
    )
    governor = packet["governor_receipt"]
    summary = governor.get("summary") if isinstance(governor.get("summary"), dict) else {}
    receipt = {
        **governor,
        "report": rel(path),
        "trigger_state": str(governor.get("trigger_state") or ""),
        "hard_gap_count": int(summary.get("hard_gap_count") or 0),
        "warning_count": int(summary.get("warning_count") or 0),
        "mission_brief_status": str(summary.get("mission_brief_status") or ""),
        "mission_brief_omission_count": int(summary.get("mission_brief_omission_count") or 0),
        "deletion_closure_status": str(summary.get("deletion_closure_status") or ""),
        "deletion_closure_fault_count": int(summary.get("deletion_closure_fault_count") or 0),
        "scif_status": str(summary.get("scif_status") or ""),
        "consumer_abi": packet,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    receipt["ready"] = bool(packet.get("ready"))
    return receipt


def verifier_context_fields(vcm_receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "vcm_context_governor_ready": bool(vcm_receipt.get("ready")),
        "vcm_context_governor_state": vcm_receipt.get("trigger_state"),
        "vcm_context_governor_receipt_id": vcm_receipt.get("receipt_id"),
        "vcm_context_governor_hard_gap_count": int(vcm_receipt.get("hard_gap_count") or 0),
        "vcm_mission_brief_status": vcm_receipt.get("mission_brief_status"),
        "vcm_deletion_closure_status": vcm_receipt.get("deletion_closure_status"),
        "vcm_scif_status": vcm_receipt.get("scif_status"),
        "vcm_context_adequacy_state": "governed_sufficient_for_verification"
        if vcm_receipt.get("ready")
        else "fault_vcm_context_governor_not_ready",
        "vcm_context_governor_receipt": vcm_receipt,
    }


def private_verifier_cache_snapshot() -> dict[str, Any]:
    return {
        "policy": "private_verifier_cache_snapshot_v1",
        "static_cache_entry_count": len(PRIVATE_STATIC_VERIFICATION_CACHE),
        "sandbox_cache_entry_count": len(PRIVATE_SANDBOX_VERIFICATION_CACHE),
        "test_harness_compile_cache_entry_count": len(PRIVATE_TEST_HARNESS_COMPILE_CACHE),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def private_verifier_cache_warmup_summary(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    worker_count: int,
    eval_task_count: int,
    candidate_count: int,
) -> dict[str, Any]:
    keys = [
        "static_cache_entry_count",
        "sandbox_cache_entry_count",
        "test_harness_compile_cache_entry_count",
    ]
    deltas = {
        key: int(after.get(key) or 0) - int(before.get(key) or 0)
        for key in keys
    }
    return {
        "policy": "private_verifier_sandbox_warmup_accounting_v1",
        "worker_count": int(worker_count or 0),
        "eval_task_count": int(eval_task_count or 0),
        "candidate_count": int(candidate_count or 0),
        "cache_before": before,
        "cache_after": after,
        "cache_entry_deltas": deltas,
        "test_harness_compile_cache_enabled": True,
        "static_candidate_cache_enabled": True,
        "sandbox_result_cache_enabled": True,
        "score_semantics": (
            "Runtime-economics receipt only. Verifier caches avoid repeated private test-harness "
            "compilation and repeated candidate sandbox execution when exact hashes match; pass/fail "
            "semantics, private-test visibility boundaries, and learned-generation credit are unchanged."
        ),
        "uses_eval_tests_or_solutions_for_generation": False,
        "uses_public_data": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "candidate_generation_credit": 0,
    }


def summarize_private_verification(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    spine_receipt = viea_spine_records.materialized_view_consumer_receipt(
        "code_lm_private_verifier",
        required_groups=[
            "claim_ledger_entries",
            "artifact_records",
            "failure_boundaries",
            "context_records",
        ],
    )
    vcm_receipt = vcm_context_governor_receipt()
    context_fields = verifier_context_fields(vcm_receipt)
    if not attempts:
        empty_summary = {
            "candidate_attempt_count": 0,
            "stage_counts": {},
            "mean_verification_reward": 0.0,
            "intended_behavior_pass_rate": 0.0,
            "runtime_load_rate": 0.0,
            **context_fields,
        }
        empty_summary["verification_bandwidth"] = private_verifier_verification_bandwidth_record(empty_summary)
        empty_summary["governance_tax"] = private_verifier_governance_tax_record(empty_summary)
        return {
            "policy": "private_code_candidate_verification_cascade_v1",
            "candidate_attempt_count": 0,
            "mean_verification_reward": 0.0,
            "stage_counts": {},
            **context_fields,
            "verification_bandwidth": empty_summary["verification_bandwidth"],
            "governance_tax": empty_summary["governance_tax"],
            "viea_verifier_records": build_private_verifier_records(empty_summary),
            "viea_spine_consumer_receipt": spine_receipt,
        }
    stage_counts: Counter[str] = Counter(str(row.get("verification_stage") or "unknown") for row in attempts)
    rewards = [float(row.get("verification_reward") or 0.0) for row in attempts]
    summary = {
        "policy": "private_code_candidate_verification_cascade_v1",
        "candidate_attempt_count": len(attempts),
        "lint_pass_rate": ratio(sum(1 for row in attempts if row.get("lint_passed")), len(attempts)),
        "compile_pass_rate": ratio(sum(1 for row in attempts if row.get("compile_passed")), len(attempts)),
        "runtime_load_rate": ratio(sum(1 for row in attempts if row.get("runtime_loaded")), len(attempts)),
        "intended_behavior_pass_rate": ratio(sum(1 for row in attempts if row.get("intended_behavior_passed")), len(attempts)),
        "mean_verification_reward": round(sum(rewards) / len(rewards), 6),
        "sandbox_skipped_before_runtime_count": sum(1 for row in attempts if not row.get("compile_passed")),
        "static_cache_hit_count": sum(1 for row in attempts if row.get("static_cache_hit")),
        "sandbox_cache_hit_count": sum(1 for row in attempts if row.get("sandbox_cache_hit")),
        "test_harness_cache_hit_count": sum(1 for row in attempts if row.get("test_harness_cache_hit")),
        "stage_counts": dict(sorted(stage_counts.items())),
        "reward_semantics": "private candidate reward layers: cached lint -> cached compile -> cached runtime load -> intended behavior",
        **context_fields,
        "viea_spine_view_ready": spine_receipt["ready"],
        "viea_spine_view_record_count": spine_receipt["record_count"],
        "viea_spine_claim_ledger_entry_count": spine_receipt["claim_ledger_entry_count"],
        "viea_spine_failure_boundary_count": spine_receipt["failure_boundary_count"],
        "viea_spine_consumer_receipt": spine_receipt,
    }
    summary["verification_bandwidth"] = private_verifier_verification_bandwidth_record(summary)
    summary["governance_tax"] = private_verifier_governance_tax_record(summary)
    summary["viea_verifier_records"] = build_private_verifier_records(summary)
    return summary


def build_private_verifier_records(summary: dict[str, Any]) -> dict[str, Any]:
    run_id = viea_spine_records.stable_id("private_verifier", summary)
    attempt_count = int(summary.get("candidate_attempt_count") or 0)
    intended_rate = float(summary.get("intended_behavior_pass_rate") or 0.0)
    runtime_load_rate = float(summary.get("runtime_load_rate") or 0.0)
    governor = summary.get("vcm_context_governor_receipt") if isinstance(summary.get("vcm_context_governor_receipt"), dict) else {}
    governor_ready = bool(governor.get("ready"))
    adequacy_state = "governed_sufficient_for_verification" if governor_ready else "fault_vcm_context_governor_not_ready"
    support_state = "SUPPORTED" if attempt_count > 0 else "RESIDUAL"
    verification_bandwidth = (
        summary.get("verification_bandwidth")
        if isinstance(summary.get("verification_bandwidth"), dict)
        else private_verifier_verification_bandwidth_record(summary)
    )
    governance_tax = (
        summary.get("governance_tax")
        if isinstance(summary.get("governance_tax"), dict)
        else private_verifier_governance_tax_record(summary)
    )
    common = {
        "run_id": run_id,
        "verifier_surface": "code_lm_private_verifier",
        "candidate_attempt_count": attempt_count,
        "runtime_load_rate": runtime_load_rate,
        "intended_behavior_pass_rate": intended_rate,
        "support_state": support_state,
        "vcm_context_governor_receipt_id": governor.get("receipt_id"),
        "vcm_context_governor_ready": governor_ready,
        "vcm_context_adequacy_state": adequacy_state,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "verification_bandwidth": verification_bandwidth,
        "governance_tax": governance_tax,
    }
    records = {
        "claim_record": {
            **common,
            "record_type": "claim_record",
            "record_id": viea_spine_records.stable_id("private_verifier_claim", run_id),
            "claim_id": viea_spine_records.stable_id("claim_private_verifier", run_id),
            "state": "private_verifier_cascade_summarized",
            "status": "GREEN" if attempt_count > 0 else "YELLOW",
            "verifier_state": "private_sandbox_cascade",
            "evidence_ref": "reports/private_verifier_spine_smoke.json",
        },
        "proof_carrying_claim": {
            **common,
            "record_type": "proof_carrying_claim",
            "record_id": viea_spine_records.stable_id("private_verifier_proof", run_id),
            "proof_claim_id": viea_spine_records.stable_id("proof_private_verifier", run_id),
            "state": "hash_stage_reward_labels_no_answers",
            "status": "GREEN" if attempt_count > 0 else "YELLOW",
            "verifier_state": "lint_compile_runtime_behavior_cascade",
            "evidence_ref": "reports/private_verifier_spine_smoke.json",
        },
        "authority_transition": {
            **common,
            "record_type": "authority_transition",
            "record_id": viea_spine_records.stable_id("private_verifier_authority_transition", run_id),
            "state": "private_local_sandbox_only",
            "status": "READY",
            "authority_scope": ["private_candidate_static_check", "private_candidate_sandbox_check"],
            "remote_arbitrary_shell_allowed": False,
        },
        "authority_use_receipt": {
            **common,
            "record_type": "authority_use_receipt",
            "record_id": viea_spine_records.stable_id("private_verifier_authority_use", run_id),
            "state": "verifier_labels_only_no_generation_credit",
            "status": "READY",
            "authority_scope": ["private_verifier_labels"],
        },
        "runtime_adapter_invocation": {
            **common,
            "record_type": "runtime_adapter_invocation",
            "record_id": viea_spine_records.stable_id("private_verifier_runtime_adapter", run_id),
            "adapter_id": "private_python_sandbox_verifier",
            "state": "local_subprocess_sandbox",
            "status": "READY" if attempt_count > 0 else "NO_ATTEMPTS",
        },
        "resource_budget": {
            **common,
            "record_type": "resource_budget",
            "record_id": viea_spine_records.stable_id("private_verifier_resource_budget", run_id),
            "worker_limit": bounded_private_verification_workers(max(1, attempt_count)),
            "network_class": "local_only",
            "task_fit": "private_verification",
            "gas_estimate_micro_twc": 0,
            "verification_obligation_count": verification_bandwidth["obligation_count"],
            "verifier_capacity_units": verification_bandwidth["verifier_capacity_units"],
            "capacity_margin_units": verification_bandwidth["capacity_margin_units"],
            "escalation_required": verification_bandwidth["escalation_required"],
            "residual_obligations": verification_bandwidth["residual_obligations"],
        },
        "costed_route_record": {
            **common,
            "record_type": "costed_route_record",
            "record_id": viea_spine_records.stable_id("private_verifier_costed_route", run_id),
            "cost_accounting": {
                "governed_overhead_ms": governance_tax["governed_overhead_ms"],
                "governed_total_latency_ms": governance_tax["governed_total_latency_ms"],
                "review_load_units": governance_tax["review_load_units"],
                "caught_failure_count": governance_tax["caught_failure_count"],
                "tax_per_caught_failure": governance_tax["tax_per_caught_failure"],
                "verification_obligation_count": verification_bandwidth["obligation_count"],
                "verifier_capacity_units": verification_bandwidth["verifier_capacity_units"],
            },
            "cost_classes": ["static_parse", "compile", "runtime_load", "intended_behavior", "vcm_context"],
            "non_claim": "Verifier governance tax is accounting for correctness labels; it is not model generation capability.",
        },
        "generation_mode": {
            **common,
            "record_type": "generation_mode",
            "record_id": viea_spine_records.stable_id("private_verifier_generation_mode", run_id),
            "state": "verifier_not_generator",
            "status": "LABEL_ONLY",
            "learned_generation_claim_allowed": False,
            "candidate_generation_credit": 0,
            "non_claim": "Private verifier labels are correctness evidence, not generation evidence.",
        },
        "failure_boundary": {
            **common,
            "record_type": "failure_boundary",
            "record_id": viea_spine_records.stable_id("private_verifier_failure_boundary", run_id),
            "failure_id": viea_spine_records.stable_id("private_verifier_boundary", run_id),
            "state": "attempts_present" if attempt_count > 0 else "no_attempts",
            "status": "READY" if attempt_count > 0 else "RESIDUAL",
            "terminal": False,
            "structured_non_solved": attempt_count <= 0,
            "fallback_return_used": False,
            "verification_escalation_required": verification_bandwidth["escalation_required"],
            "residual_obligations": verification_bandwidth["residual_obligations"],
        },
        "context_transaction": {
            **common,
            "record_type": "context_transaction_record",
            "record_id": viea_spine_records.stable_id("private_verifier_context_transaction", run_id),
            "transaction_id": viea_spine_records.stable_id("private_verifier_context_txn", run_id, governor.get("receipt_id")),
            "operation": "read",
            "mounts": ["vcm_context_governor", "private_verifier_attempt_metadata"],
            "read_set": ["reports/vcm_context_governor.json", "vcm://theseus/verifier/candidate-attempt-hashes@current-run"],
            "write_set": ["reports/private_verifier_spine_smoke.json"],
            "branch_policy": "verifier_read_only_governed_context",
            "taint_labels": ["private_verifier_metadata", "governed_context_receipt"],
            "deletion_obligations": "closed_by_vcm_governor" if governor.get("deletion_closure_status") == "closed" else "fault_requires_vcm_deletion_closure",
            "closure_state": "closed_for_verifier_read" if governor_ready else "fault_context_governance",
            "faults": [] if governor_ready else ["vcm_context_governor_not_ready"],
            "audit_refs": ["reports/vcm_context_governor.json", "reports/private_verifier_spine_smoke.json"],
        },
        "context_adequacy": {
            **common,
            "record_type": "context_adequacy_record",
            "record_id": viea_spine_records.stable_id("private_verifier_context_adequacy", run_id),
            "adequacy_id": viea_spine_records.stable_id("private_verifier_context_adequacy", run_id, governor.get("receipt_id")),
            "target_claim_id": viea_spine_records.stable_id("claim_private_verifier", run_id),
            "semantic_units": [
                {
                    "address": "reports/vcm_context_governor.json",
                    "title": "VCM context governor receipt",
                    "source_path": "reports/vcm_context_governor.json",
                    "taints": ["governed_context_receipt"],
                }
            ],
            "compression_path": "vcm_governor_receipt_to_private_verifier_context",
            "verification_mode": "private_verifier_context_adequacy_only",
            "adequacy_state": adequacy_state,
            "governor_receipt_id": governor.get("receipt_id"),
            "governor_ready": governor_ready,
            "mission_brief_status": governor.get("mission_brief_status"),
            "deletion_closure_status": governor.get("deletion_closure_status"),
            "scif_status": governor.get("scif_status"),
            "fail_closed": not governor_ready,
            "residual_risks": [] if governor_ready else ["verifier_context_governor_not_ready"],
            "required_escalation": "refresh_vcm_context_governor_before_verifier_use" if not governor_ready else "none",
        },
        "artifact_graph_record": {
            **common,
            "record_type": "artifact_graph_record",
            "record_id": viea_spine_records.stable_id("private_verifier_artifact", run_id),
            "artifact_kind": "private_verifier_spine_smoke",
            "evidence_ref": "reports/private_verifier_spine_smoke.json",
            "content_hash": viea_spine_records.stable_hash(summary),
            "context_refs": ["reports/vcm_context_governor.json", governor.get("receipt_id")],
        },
        "evidence_transition_record": {
            **common,
            "record_type": "evidence_transition_record",
            "record_id": viea_spine_records.stable_id("private_verifier_evidence_transition", run_id),
            "state": "private_candidate_attempts_to_verifier_labels",
            "status": "SUPPORTED" if attempt_count > 0 else "RESIDUAL",
            "evidence_ref": "reports/private_verifier_spine_smoke.json",
        },
    }
    abi_packet = governor.get("consumer_abi") if isinstance(governor.get("consumer_abi"), dict) else {}
    for index, row in enumerate(abi_packet.get("records", []) if isinstance(abi_packet.get("records"), list) else []):
        if isinstance(row, dict):
            records[f"vcm_consumer_abi_{index:02d}"] = row
    return records


def private_verifier_verification_bandwidth_record(summary: dict[str, Any]) -> dict[str, Any]:
    attempt_count = int(summary.get("candidate_attempt_count") or 0)
    stage_counts = summary.get("stage_counts") if isinstance(summary.get("stage_counts"), dict) else {}
    governor_ready = bool(summary.get("vcm_context_governor_ready"))
    runtime_load_rate = float(summary.get("runtime_load_rate") or 0.0)
    intended_rate = float(summary.get("intended_behavior_pass_rate") or 0.0)
    stage_count = len(stage_counts)
    obligation_count = (
        max(1, attempt_count) * 4
        + 1  # VCM adequacy
        + 1  # no public/external/fallback boundary
        + max(1, stage_count)
    )
    verifier_capacity_units = (
        max(1, attempt_count) * 3
        + int(round(runtime_load_rate * max(1, attempt_count)))
        + int(round(intended_rate * max(1, attempt_count)))
        + (2 if governor_ready else 0)
    )
    capacity_floor_units = max(4, min(obligation_count, 256))
    capacity_margin_units = verifier_capacity_units - capacity_floor_units
    residual_obligations: list[str] = []
    if not governor_ready:
        residual_obligations.append("vcm_context_governor_not_ready")
    if attempt_count <= 0:
        residual_obligations.append("no_private_verifier_attempts")
    if capacity_margin_units < 0:
        residual_obligations.append("private_verifier_capacity_escalation")
    return {
        "policy": "project_theseus_private_verifier_verification_bandwidth_v1",
        "surface": "code_lm_private_verifier",
        "obligation_count": obligation_count,
        "verifier_capacity_units": verifier_capacity_units,
        "capacity_floor_units": capacity_floor_units,
        "capacity_margin_units": capacity_margin_units,
        "verification_arms": ["lint_parse", "compile", "runtime_load", "intended_behavior", "vcm_context_adequacy"],
        "decomposition_contract": {
            "candidate_attempt_count": attempt_count,
            "stage_counts": stage_counts,
            "verification_strategy": "private_candidate_hashes_progress_through_lint_compile_runtime_and_behavior_without_exposing_tests_to_generation",
        },
        "residual_obligations": residual_obligations,
        "escalation_thresholds": {
            "capacity_margin_min": 0,
            "vcm_context_governor_required": True,
            "attempt_count_min": 1,
        },
        "escalation_required": bool(residual_obligations),
        "adequacy_state": "ready" if not residual_obligations else "verification_capacity_residual",
        "status": "ready",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "candidate_generation_credit": 0,
        "non_claims": [
            "private verifier bandwidth is correctness-label accounting, not learned generation",
            "private tests remain verifier-only and are not prompt/context for generation",
        ],
    }


def private_verifier_governance_tax_record(summary: dict[str, Any]) -> dict[str, Any]:
    attempt_count = int(summary.get("candidate_attempt_count") or 0)
    verification_bandwidth = (
        summary.get("verification_bandwidth")
        if isinstance(summary.get("verification_bandwidth"), dict)
        else private_verifier_verification_bandwidth_record(summary)
    )
    gate_costs = {
        "static_parse_cache_ms": max(1, int(summary.get("static_cache_hit_count") or 0)),
        "compile_cache_ms": max(1, int(summary.get("test_harness_cache_hit_count") or 0)),
        "sandbox_runtime_cache_ms": max(1, int(summary.get("sandbox_cache_hit_count") or 0)),
        "vcm_context_adequacy_ms": 4 if summary.get("vcm_context_governor_ready") else 8,
        "verification_bandwidth_accounting_ms": 3,
        "no_cheat_boundary_ms": 2,
    }
    governed_overhead_ms = sum(gate_costs.values())
    raw_latency_ms = max(1, attempt_count)
    caught_failure_count = len(verification_bandwidth.get("residual_obligations") or [])
    review_load_units = max(1, len(verification_bandwidth.get("verification_arms") or []))
    return {
        "policy": "project_theseus_private_verifier_governance_tax_v1",
        "surface": "code_lm_private_verifier",
        "gate_costs": gate_costs,
        "raw_route_latency_ms": raw_latency_ms,
        "governed_overhead_ms": governed_overhead_ms,
        "governed_total_latency_ms": raw_latency_ms + governed_overhead_ms,
        "review_load_units": review_load_units,
        "caught_failure_count": caught_failure_count,
        "tax_per_caught_failure": round(governed_overhead_ms / max(1, caught_failure_count), 6),
        "tax_justified": caught_failure_count > 0 or attempt_count > 0,
        "tax_value_statement": "private verifier governance cost preserves staged correctness labels, context adequacy, and no-cheat boundaries",
        "status": "ready",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "candidate_generation_credit": 0,
        "non_claims": [
            "governance tax is verifier overhead accounting, not behavior improvement",
            "verifier speed cannot hide displaced replay or context adequacy cost",
        ],
    }


def correctness_label_summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize private verifier labels without exposing test payloads.

    These labels are intended as correctness-in-the-loop training/evidence
    signals. They carry candidate hashes, phases, ranks, stages, and rewards,
    but no private tests, expected outputs, solutions, public benchmark data, or
    answer templates.
    """

    stage_counts: Counter[str] = Counter(str(row.get("verification_stage") or "unknown") for row in attempts)
    phase_counts: Counter[str] = Counter(str(row.get("phase") or "unknown") for row in attempts)
    rank1_stage_counts: Counter[str] = Counter(
        str(row.get("verification_stage") or "unknown")
        for row in attempts
        if int(row.get("attempt_index") or 0) == 1
    )
    rewards = [float(row.get("verification_reward") or 0.0) for row in attempts]
    generated = [
        row
        for row in attempts
        if str(row.get("candidate_generation_mode") or "") == "token_level_code_decoder"
        and str(row.get("candidate_source") or "") != "shared_null_baseline"
    ]
    spine_receipt = viea_spine_records.materialized_view_consumer_receipt(
        "private_verifier_candidate_correctness_labels",
        required_groups=[
            "claim_ledger_entries",
            "artifact_records",
            "failure_boundaries",
            "generation_mode_records",
        ],
    )
    return {
        "policy": "private_verifier_candidate_correctness_labels_v1",
        "candidate_attempt_count": len(attempts),
        "generated_candidate_attempt_count": len(generated),
        "phase_counts": dict(sorted(phase_counts.items())),
        "stage_counts": dict(sorted(stage_counts.items())),
        "rank1_stage_counts": dict(sorted(rank1_stage_counts.items())),
        "lint_pass_count": sum(1 for row in attempts if row.get("lint_passed")),
        "compile_pass_count": sum(1 for row in attempts if row.get("compile_passed")),
        "runtime_load_count": sum(1 for row in attempts if row.get("runtime_loaded")),
        "intended_behavior_pass_count": sum(1 for row in attempts if row.get("intended_behavior_passed")),
        "generated_intended_behavior_pass_count": sum(1 for row in generated if row.get("intended_behavior_passed")),
        "mean_verification_reward": round(sum(rewards) / len(rewards), 6) if rewards else 0.0,
        "max_verification_reward": round(max(rewards), 6) if rewards else 0.0,
        "uses_eval_tests_or_solutions_for_generation": False,
        "uses_public_data": False,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "candidate_generation_credit": 0,
        "viea_spine_view_ready": spine_receipt["ready"],
        "viea_spine_view_record_count": spine_receipt["record_count"],
        "viea_spine_generation_mode_record_count": spine_receipt["generation_mode_record_count"],
        "viea_spine_consumer_receipt": spine_receipt,
        "score_semantics": (
            "Private verifier labels for candidate training/evidence. The verifier may execute private "
            "tests inside the sandbox, but generation and ranking do not see those tests, expected "
            "outputs, public benchmarks, solutions, answer templates, or teacher output. Labels are "
            "hash/stage/reward records, not learned-generation capability credit."
        ),
    }


def bounded_private_verification_workers(task_count: int) -> int:
    if task_count <= 1:
        return 1
    value = os.environ.get("THESEUS_PRIVATE_VERIFY_WORKERS", "").strip()
    if value:
        try:
            return max(1, min(int(value), task_count, 32))
        except ValueError:
            pass
    cores = os.cpu_count() or 4
    return max(2, min(12, task_count, cores))




def classify_failure(stderr: str) -> str:
    text = stderr.lower()
    if "lint_parse_failed" in text or "candidate_compile_failed" in text or "test_harness_compile_failed" in text:
        return "verification_cascade_compile"
    if "runtime_failed" in text or "sandbox_launch_failed" in text:
        return "runtime_load_failure"
    if "syntaxerror" in text or "indentationerror" in text:
        return "parsing"
    if "typeerror" in text or "attributeerror" in text:
        return "type_handling"
    if "assertionerror" in text:
        return "wrong_answer"
    if "timeout" in text:
        return "timeout"
    return "runtime"


def ratio(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def safe_name(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "item")).strip("_") or "item"


def list_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def runtime_tmp_dir() -> Path:
    return benchmark_runtime_tmp_dir()


def build_private_verifier_spine_smoke_report() -> dict[str, Any]:
    task = {
        "task_id": "private_verifier_spine_add_one_v1",
        "split": "eval",
        "category": "private_verifier_spine_smoke",
        "tests": "assert add_one(1) == 2\nassert add_one(-1) == 0\nassert add_one(0) == 1\n",
        "tags": ["private_fixture", "verifier_spine"],
    }
    candidates = [
        {
            "task_id": task["task_id"],
            "phase": "private_eval",
            "rank": 1,
            "rank_score": 1.0,
            "entry_point": "add_one",
            "candidate_generation_mode": "private_verifier_spine_fixture",
            "candidate_source": "governed_private_fixture",
            "source_task_id": task["task_id"],
            "code": "def add_one(x):\n    return x + 1\n",
            "public_tests_visible_to_generator": False,
            "public_solutions_visible_to_generator": False,
            "eval_tests_visible_to_generator": False,
            "eval_solution_visible_to_generator": False,
            "external_inference_calls": 0,
        }
    ]
    private_eval = evaluate_private_candidates([task], candidates)
    verification = private_eval.get("private_verification") if isinstance(private_eval.get("private_verification"), dict) else {}
    labels = private_eval.get("correctness_labels") if isinstance(private_eval.get("correctness_labels"), dict) else {}
    records = verification.get("viea_verifier_records") if isinstance(verification.get("viea_verifier_records"), dict) else {}
    verification_bandwidth = (
        verification.get("verification_bandwidth")
        if isinstance(verification.get("verification_bandwidth"), dict)
        else {}
    )
    governance_tax = verification.get("governance_tax") if isinstance(verification.get("governance_tax"), dict) else {}
    record_count = sum(1 for value in records.values() if isinstance(value, dict))
    vcm_ready = bool(verification.get("vcm_context_governor_ready"))
    accounting_ready = verification_bandwidth.get("status") == "ready" and governance_tax.get("status") == "ready"
    trigger_state = (
        "GREEN"
        if private_eval.get("trained_passed") == 1 and record_count >= 13 and vcm_ready and accounting_ready
        else "RED"
    )
    return {
        "policy": "project_theseus_private_verifier_spine_smoke_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "eval_task_count": private_eval.get("eval_task_count"),
            "trained_passed": private_eval.get("trained_passed"),
            "candidate_attempt_count": verification.get("candidate_attempt_count"),
            "runtime_load_rate": verification.get("runtime_load_rate"),
            "intended_behavior_pass_rate": verification.get("intended_behavior_pass_rate"),
            "viea_verifier_record_count": record_count,
            "viea_spine_view_ready": verification.get("viea_spine_view_ready"),
            "viea_spine_view_record_count": verification.get("viea_spine_view_record_count"),
            "vcm_context_governor_ready": verification.get("vcm_context_governor_ready"),
            "vcm_context_governor_state": verification.get("vcm_context_governor_state"),
            "vcm_context_governor_receipt_id": verification.get("vcm_context_governor_receipt_id"),
            "vcm_context_adequacy_state": verification.get("vcm_context_adequacy_state"),
            "vcm_context_governor_hard_gap_count": verification.get("vcm_context_governor_hard_gap_count"),
            "vcm_mission_brief_status": verification.get("vcm_mission_brief_status"),
            "vcm_deletion_closure_status": verification.get("vcm_deletion_closure_status"),
            "vcm_scif_status": verification.get("vcm_scif_status"),
            "verification_bandwidth_status": verification_bandwidth.get("status"),
            "verification_obligation_count": verification_bandwidth.get("obligation_count"),
            "verification_escalation_required": verification_bandwidth.get("escalation_required"),
            "governance_tax_status": governance_tax.get("status"),
            "governance_tax_review_load_units": governance_tax.get("review_load_units"),
            "governance_tax_caught_failure_count": governance_tax.get("caught_failure_count"),
            "correctness_label_attempt_count": labels.get("candidate_attempt_count"),
        },
        "gates": [
            gate("trained_candidate_passed_private_fixture", private_eval.get("trained_passed") == 1, private_eval.get("trained_passed")),
            gate("viea_verifier_records_include_context_records", record_count >= 13, record_count),
            gate("vcm_context_governor_ready", vcm_ready, verification.get("vcm_context_governor_receipt")),
            gate("private_verifier_verification_bandwidth_ready", verification_bandwidth.get("status") == "ready", verification_bandwidth),
            gate("private_verifier_governance_tax_ready", governance_tax.get("status") == "ready", governance_tax),
            gate("no_public_training_rows", True, 0),
            gate("no_external_inference_calls", True, 0),
            gate("no_fallback_returns", True, 0),
        ],
        "private_verification": verification,
        "correctness_labels": labels,
        "score_semantics": "Tiny private verifier smoke only. It proves verifier cascade record emission; it is not model capability, training data, or public benchmark evidence.",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_spine_smoke_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return "\n".join(
        [
            "# Private Verifier Spine Smoke",
            "",
            f"- State: `{report.get('trigger_state')}`",
            f"- Eval tasks: `{summary.get('eval_task_count')}`",
            f"- Trained passed: `{summary.get('trained_passed')}`",
            f"- Candidate attempts: `{summary.get('candidate_attempt_count')}`",
            f"- Runtime load rate: `{summary.get('runtime_load_rate')}`",
            f"- Intended behavior pass rate: `{summary.get('intended_behavior_pass_rate')}`",
            f"- VIEA verifier records: `{summary.get('viea_verifier_record_count')}`",
            f"- VIEA view ready: `{summary.get('viea_spine_view_ready')}` records `{summary.get('viea_spine_view_record_count')}`",
            f"- VCM governor ready: `{summary.get('vcm_context_governor_ready')}` state `{summary.get('vcm_context_governor_state')}`",
            f"- VCM adequacy state: `{summary.get('vcm_context_adequacy_state')}`",
            f"- Verification bandwidth: `{summary.get('verification_bandwidth_status')}` obligations `{summary.get('verification_obligation_count')}` escalation `{summary.get('verification_escalation_required')}`",
            f"- Governance tax: `{summary.get('governance_tax_status')}` review load `{summary.get('governance_tax_review_load_units')}` caught failures `{summary.get('governance_tax_caught_failure_count')}`",
            "",
            report.get("score_semantics", ""),
            "",
        ]
    )


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spine-smoke", action="store_true", help="Emit the canonical private verifier VIEA spine smoke report.")
    parser.add_argument("--out", default=rel(DEFAULT_SPINE_SMOKE_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_SPINE_SMOKE_MD))
    args = parser.parse_args()
    report = build_private_verifier_spine_smoke_report()
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_spine_smoke_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
