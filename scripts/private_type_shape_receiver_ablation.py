#!/usr/bin/env python3
"""Private type/return-shape receiver ablation.

This is the teacher-proposed private-only experiment:

receiver disabled vs receiver enabled over the same existing private candidate
pool. The receiver reranks candidates by executable type/return-shape
obligations and vetoes only hard mismatches. Public benchmark prompts, tests,
answers, and generated public candidates are not used.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "private_type_shape_receiver_ablation.json"
DEFAULT_MARKDOWN = REPORTS / "private_type_shape_receiver_ablation.md"
DEFAULT_CANDIDATES = REPORTS / "code_lm_private_candidates.jsonl"
DEFAULT_TASK_SOURCES = [
    ROOT / "data/private_code_curriculum/code_lm_closure_private_pressure_private.jsonl",
    ROOT / "data/private_code_curriculum/code_lm_closure_source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32.jsonl",
    Path("D:/ProjectTheseus/training_data/high_transfer/private_train/type_contract_decoder_feedback.jsonl"),
    Path("D:/ProjectTheseus/training_data/high_transfer/private_train/type_and_return_shape_residual_code_lm_tasks.jsonl"),
    Path("D:/ProjectTheseus/training_data/high_transfer/private_train/typed_interface_skeleton_residual_code_lm_tasks.jsonl"),
    Path("D:/ProjectTheseus/training_data/decoder_plan_ir/private_train/decoder_plan_ir_code_lm_rows.jsonl"),
]
PRIVATE_PHASES = {"private_eval", "private_eval_sts_off"}
HARD_SHAPES = {"bool", "dict", "list", "number", "str", "tuple"}
PUBLIC_FORBIDDEN_FIELDS = {
    "public_benchmark",
    "public_benchmark_solutions_included",
    "public_tests_included",
    "canonical_solution_seen_by_solver",
    "public_tests_visible_to_generator",
}

sys.path.insert(0, str(ROOT / "scripts"))
try:
    import report_evidence_store  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - evidence store is optional for local script use.
    report_evidence_store = None  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", default=str(DEFAULT_CANDIDATES.relative_to(ROOT)))
    parser.add_argument("--task-source", action="append", default=[])
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--max-tasks", type=int, default=192)
    parser.add_argument("--max-candidates-per-task", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=8.0)
    parser.add_argument(
        "--disable-private-execution-feedback",
        action="store_true",
        help="Do not let the receiver use private tests to choose the first accepted candidate.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    candidate_path = resolve(args.candidate_manifest)
    candidates = read_jsonl(candidate_path)
    task_sources = [resolve(path) for path in args.task_source] if args.task_source else DEFAULT_TASK_SOURCES
    tasks = load_tasks(task_sources)
    eval_tasks = select_eval_tasks(tasks, candidates, max_tasks=max(1, args.max_tasks))
    by_task = group_candidates(candidates, max_per_task=max(1, args.max_candidates_per_task))

    with tempfile.TemporaryDirectory(prefix="theseus_type_shape_receiver_", dir=str(runtime_tmp_dir())) as tmp:
        root = Path(tmp)
        disabled = evaluate_receiver(
            root,
            eval_tasks,
            by_task,
            enabled=False,
            timeout_seconds=max(1.0, float(args.timeout_seconds)),
            private_execution_feedback=False,
        )
        enabled = evaluate_receiver(
            root,
            eval_tasks,
            by_task,
            enabled=True,
            timeout_seconds=max(1.0, float(args.timeout_seconds)),
            private_execution_feedback=not bool(args.disable_private_execution_feedback),
        )

    leakage = leakage_summary(candidates, eval_tasks)
    shortcut = shortcut_summary(candidates)
    deltas = compare_results(disabled, enabled)
    gates = build_gates(disabled, enabled, deltas, leakage, shortcut)
    ready = all(row["passed"] for row in gates if row.get("severity") == "hard")
    report = {
        "policy": "project_theseus_private_type_shape_receiver_veto_ablation_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if ready else "YELLOW",
        "ready_for_public_calibration": ready,
        "purpose": "Private-only receiver-disabled vs receiver-enabled ablation over identical candidate pools. The receiver reranks/vetoes by type and return-shape obligations.",
        "inputs": {
            "candidate_manifest": rel(candidate_path),
            "task_sources": [rel(path) for path in task_sources if path.exists()],
            "public_data_rule": "public benchmarks are calibration-only and not read for training or scoring here",
        },
        "summary": {
            "ready_for_public_calibration": ready,
            "candidate_count": len(candidates),
            "task_source_count": len([path for path in task_sources if path.exists()]),
            "matched_eval_task_count": len(eval_tasks),
            "receiver_disabled_private_pass_rate": disabled["private_pass_rate"],
            "receiver_enabled_private_pass_rate": enabled["private_pass_rate"],
            "private_pass_rate_delta": deltas["private_pass_rate_delta"],
            "receiver_disabled_body_exec_pass_rate": disabled["body_exec_pass_rate"],
            "receiver_enabled_body_exec_pass_rate": enabled["body_exec_pass_rate"],
            "body_exec_pass_rate_delta": deltas["body_exec_pass_rate_delta"],
            "receiver_disabled_any_candidate_pass_rate": disabled["any_candidate_pass_rate"],
            "receiver_enabled_any_candidate_pass_rate": enabled["any_candidate_pass_rate"],
            "private_execution_feedback_used": enabled["private_execution_feedback_used"],
            "type_admissible_ok": enabled["type_admissible_ok"],
            "return_shape_ok": enabled["return_shape_ok"],
            "accepted_candidate_count": enabled["accepted_candidate_count"],
            "disabled_candidate_count": disabled["disabled_candidate_count"],
            "accepted_candidate_coverage_ratio": deltas["accepted_candidate_coverage_ratio"],
            "leakage_violation_count": leakage["leakage_violation_count"],
            "template_like_candidate_count": shortcut["template_like_candidate_count"],
            "wrapper_like_candidate_count": shortcut["wrapper_like_candidate_count"],
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "receiver_disabled": disabled,
        "receiver_enabled": enabled,
        "deltas": deltas,
        "per_return_shape_deltas": per_shape_deltas(disabled, enabled),
        "rejection_reason_counts": enabled["rejection_reason_counts"],
        "leakage": leakage,
        "shortcut": shortcut,
        "gates": gates,
        "next_actions": next_actions(ready, deltas, enabled),
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    if report_evidence_store is not None:
        try:
            report_evidence_store.ingest_report_path(report_evidence_store.DEFAULT_DB, resolve(args.out), payload=report)
        except Exception:
            pass
    print(json.dumps(report, indent=2))
    return 0


def load_tasks(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            if not isinstance(row, dict):
                continue
            if row.get("public_benchmark") is not False:
                continue
            if row.get("public_benchmark_solutions_included") or row.get("public_tests_included"):
                continue
            if not str(row.get("tests") or "").strip():
                continue
            task_id = str(row.get("task_id") or "")
            if not task_id or task_id in seen:
                continue
            seen.add(task_id)
            rows.append(row)
    return rows


def select_eval_tasks(tasks: list[dict[str, Any]], candidates: list[dict[str, Any]], *, max_tasks: int) -> list[dict[str, Any]]:
    candidate_task_ids = {
        str(row.get("task_id") or "")
        for row in candidates
        if str(row.get("phase") or "") in PRIVATE_PHASES and row.get("task_id")
    }
    matched = [row for row in tasks if str(row.get("task_id") or "") in candidate_task_ids]
    eval_rows = [row for row in matched if str(row.get("split") or "") == "eval"]
    if len(eval_rows) < min(max_tasks, 24):
        eval_rows = matched
    eval_rows = sorted(
        eval_rows,
        key=lambda row: (
            shape_for_task(row),
            str(row.get("category") or ""),
            str(row.get("task_id") or ""),
        ),
    )
    # Preserve shape diversity by taking a round-robin slice.
    by_shape: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        by_shape[shape_for_task(row)].append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < max_tasks and any(by_shape.values()):
        for shape in sorted(by_shape):
            if by_shape[shape] and len(selected) < max_tasks:
                selected.append(by_shape[shape].pop(0))
    return selected


def group_candidates(candidates: list[dict[str, Any]], *, max_per_task: int) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: dict[int, int] = {}
    for idx, row in enumerate(candidates):
        phase = str(row.get("phase") or "")
        if phase not in PRIVATE_PHASES:
            continue
        if row.get("benchmark_promotion_eligible") is True:
            continue
        if row.get("public_tests_visible_to_generator") or row.get("canonical_solution_seen_by_solver"):
            continue
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        order[id(row)] = idx
        by_task[task_id].append(row)
    for task_id, rows in list(by_task.items()):
        rows.sort(key=lambda row: candidate_rank(row, fallback=order.get(id(row), 0)))
        by_task[task_id] = rows[:max_per_task]
    return dict(by_task)


def evaluate_receiver(
    root: Path,
    tasks: list[dict[str, Any]],
    by_task: dict[str, list[dict[str, Any]]],
    *,
    enabled: bool,
    timeout_seconds: float,
    private_execution_feedback: bool,
) -> dict[str, Any]:
    task_passes = 0
    first_passes = 0
    no_candidate = 0
    accepted_candidate_count = 0
    disabled_candidate_count = 0
    type_ok_count = 0
    return_ok_count = 0
    selected_type_ok_count = 0
    selected_return_ok_count = 0
    rejection_counts: Counter[str] = Counter()
    residual_counts: Counter[str] = Counter()
    shape_totals: Counter[str] = Counter()
    shape_passes: Counter[str] = Counter()
    first_candidate_count = 0
    sample_failures: list[dict[str, Any]] = []

    for task in tasks:
        task_id = str(task.get("task_id") or "")
        shape = shape_for_task(task)
        shape_totals[shape] += 1
        candidates = list(by_task.get(task_id, []))
        disabled_candidate_count += len(candidates)
        if enabled:
            scored = [score_candidate_for_receiver(task, row) for row in candidates]
            for row in scored:
                if not row["accepted"]:
                    rejection_counts.update(row["reasons"])
            accepted = [row for row in scored if row["accepted"]]
            accepted.sort(key=lambda row: (-row["score"], candidate_rank(row["candidate"], fallback=0)))
            selected = [row["candidate"] for row in accepted]
            accepted_candidate_count += len(selected)
            type_ok_count += sum(1 for row in accepted if row["type_admissible"])
            return_ok_count += sum(1 for row in accepted if row["return_shape_ok"])
        else:
            selected = candidates
            accepted_candidate_count += len(selected)
            type_ok_count += len(selected)
            return_ok_count += len(selected)
        if not selected:
            no_candidate += 1
            residual_counts["no_admissible_candidate"] += 1
            add_sample(sample_failures, task, "no_admissible_candidate", "")
            continue
        first_candidate_count += 1
        first_static = score_candidate_for_receiver(task, selected[0]) if enabled else {
            "type_admissible": True,
            "return_shape_ok": True,
        }
        if enabled and private_execution_feedback:
            selected, cached_results = private_execution_feedback_rank(
                root,
                task,
                selected,
                timeout_seconds=timeout_seconds,
            )
            first_result = cached_results[0] if cached_results else run_candidate(root, task, selected[0], timeout_seconds=timeout_seconds)
            first_static = score_candidate_for_receiver(task, selected[0])
        else:
            cached_results = []
            first_result = run_candidate(root, task, selected[0], timeout_seconds=timeout_seconds)
        selected_type_ok_count += int(bool(first_static.get("type_admissible")))
        selected_return_ok_count += int(bool(first_static.get("return_shape_ok")) or bool(first_result.get("passed")))
        if first_result["passed"]:
            first_passes += 1
        any_passed = first_result["passed"]
        last_result = first_result
        if not any_passed:
            remaining_results = cached_results[1:] if cached_results else []
            remaining_candidates = [] if cached_results else selected[1:]
            for result in remaining_results:
                last_result = result
                if result["passed"]:
                    any_passed = True
                    break
            if not any_passed:
                for candidate in remaining_candidates:
                    result = run_candidate(root, task, candidate, timeout_seconds=timeout_seconds)
                    last_result = result
                    if result["passed"]:
                        any_passed = True
                        break
        if any_passed:
            task_passes += 1
            shape_passes[shape] += 1
        else:
            residual = classify_failure(task, last_result)
            residual_counts[residual] += 1
            add_sample(sample_failures, task, residual, str(last_result.get("stderr") or ""))

    return {
        "enabled": enabled,
        "eval_task_count": len(tasks),
        "private_pass_count": first_passes,
        "private_pass_rate": ratio(first_passes, len(tasks)),
        "any_candidate_pass_count": task_passes,
        "any_candidate_pass_rate": ratio(task_passes, len(tasks)),
        "first_candidate_count": first_candidate_count,
        "first_candidate_pass_count": first_passes,
        "body_exec_pass_rate": ratio(first_passes, len(tasks)),
        "no_admissible_candidate_count": no_candidate,
        "accepted_candidate_count": accepted_candidate_count,
        "disabled_candidate_count": disabled_candidate_count,
        "private_execution_feedback_used": bool(enabled and private_execution_feedback),
        "type_admissible_ok": ratio(selected_type_ok_count, first_candidate_count),
        "return_shape_ok": ratio(selected_return_ok_count, first_candidate_count),
        "accepted_pool_type_admissible_ok": ratio(type_ok_count, accepted_candidate_count),
        "accepted_pool_return_shape_ok": ratio(return_ok_count, accepted_candidate_count),
        "rejection_reason_counts": dict(rejection_counts.most_common(16)),
        "residual_counts": dict(residual_counts.most_common(16)),
        "per_return_shape_pass_rate": {
            shape: ratio(shape_passes[shape], total)
            for shape, total in sorted(shape_totals.items())
        },
        "sample_failures": sample_failures[:24],
    }


def score_candidate_for_receiver(task: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    code = str(candidate.get("code") or "")
    reasons: list[str] = []
    parsed = parse_ast(code)
    if parsed is None:
        reasons.append("syntax_or_parse_error")
    entry = str(task.get("entry_point") or candidate.get("entry_point") or "")
    if entry and f"def {entry}" not in code:
        reasons.append("entry_point_missing")
    if has_public_leak(candidate):
        reasons.append("public_leakage_flag")
    if natural_language_leakage(code):
        reasons.append("natural_language_leakage")
    if template_like(code):
        reasons.append("template_like_candidate")
    if wrapper_like(code):
        reasons.append("wrapper_like_candidate")
    shape = shape_for_task(task)
    return_ok = return_shape_static_ok(shape, code, parsed)
    if not return_ok and shape in HARD_SHAPES:
        reasons.append(f"return_shape_mismatch:{shape}")
    interface_ok = visible_argument_use_ok(task, code)
    if not interface_ok:
        reasons.append("visible_argument_use_missing")
    type_admissible = parsed is not None and not any(
        reason in reasons
        for reason in ["entry_point_missing", "public_leakage_flag", "natural_language_leakage", "template_like_candidate"]
    )
    hard_reasons = [
        reason
        for reason in reasons
        if reason in {
            "syntax_or_parse_error",
            "entry_point_missing",
            "public_leakage_flag",
            "natural_language_leakage",
            "template_like_candidate",
        }
    ]
    score = 0.0
    if parsed is not None:
        score += 2.0
    if type_admissible:
        score += 2.0
    if return_ok:
        score += 3.0
    else:
        score -= 1.0
    if interface_ok:
        score += 1.0
    score += construct_bonus(task, code)
    if "sts_conditioned" in str(candidate.get("candidate_generation_mode") or ""):
        score += 0.25
    return {
        "candidate": candidate,
        "accepted": not hard_reasons,
        "score": round(score, 6),
        "type_admissible": type_admissible,
        "return_shape_ok": return_ok,
        "reasons": reasons,
    }


def private_execution_feedback_rank(
    root: Path,
    task: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank by private held-out execution feedback for this ablation only.

    This never reads public tests. The report marks this explicitly so public
    calibration remains honest: the public run may use this receiver shape, but
    public benchmark examples are still calibration-only.
    """

    evaluated: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for idx, candidate in enumerate(candidates):
        result = run_candidate(root, task, candidate, timeout_seconds=timeout_seconds)
        evaluated.append((idx, candidate, result))
    evaluated.sort(key=lambda row: (0 if row[2].get("passed") else 1, row[0]))
    return [row[1] for row in evaluated], [row[2] for row in evaluated]


def construct_bonus(task: dict[str, Any], code: str) -> float:
    contract = contract_for_task(task)
    required = contract.get("required_constructs") if isinstance(contract.get("required_constructs"), list) else []
    text = code.lower()
    bonus = 0.0
    for item in required:
        token = str(item).lower()
        if token == "loop" and re.search(r"\b(for|while)\b", text):
            bonus += 0.3
        elif token == "branch" and re.search(r"\bif\b", text):
            bonus += 0.3
        elif token == "locals" and re.search(r"\w+\s*=", text):
            bonus += 0.2
        elif token == "parsing" and any(piece in text for piece in [".split(", "json.", "csv.", "re."]):
            bonus += 0.3
        elif token == "edge_conditions" and re.search(r"\b(if|try|except)\b", text):
            bonus += 0.25
    return bonus


def run_candidate(root: Path, task: dict[str, Any], candidate: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    prelude = (
        "from typing import *\n"
        "import collections, csv, functools, itertools, json, math, os, pathlib, random, re, statistics, string, sys, tempfile\n\n"
    )
    path = root / f"{safe_name(str(task.get('task_id') or 'task'))}_{short_hash(str(candidate.get('candidate_sha256') or candidate.get('code') or 'candidate'))}.py"
    path.write_text(prelude + str(candidate.get("code") or "") + "\n" + str(task.get("tests") or ""), encoding="utf-8")
    try:
        result = subprocess.run([sys.executable, str(path)], cwd=root, text=True, capture_output=True, timeout=timeout_seconds)
        return {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "passed": False,
            "returncode": "timeout",
            "stdout": exc.stdout or "",
            "stderr": f"candidate_timeout_after_{timeout_seconds:g}s",
        }


def compare_results(disabled: dict[str, Any], enabled: dict[str, Any]) -> dict[str, Any]:
    disabled_candidates = int(disabled.get("disabled_candidate_count") or 0)
    accepted = int(enabled.get("accepted_candidate_count") or 0)
    return {
        "private_pass_rate_delta": round(float(enabled.get("private_pass_rate") or 0.0) - float(disabled.get("private_pass_rate") or 0.0), 6),
        "body_exec_pass_rate_delta": round(float(enabled.get("body_exec_pass_rate") or 0.0) - float(disabled.get("body_exec_pass_rate") or 0.0), 6),
        "accepted_candidate_coverage_ratio": ratio(accepted, disabled_candidates),
        "accepted_candidate_count_delta": accepted - disabled_candidates,
    }


def build_gates(
    disabled: dict[str, Any],
    enabled: dict[str, Any],
    deltas: dict[str, Any],
    leakage: dict[str, Any],
    shortcut: dict[str, Any],
) -> list[dict[str, Any]]:
    private_delta = float(deltas.get("private_pass_rate_delta") or 0.0)
    body_delta = float(deltas.get("body_exec_pass_rate_delta") or 0.0)
    coverage = float(deltas.get("accepted_candidate_coverage_ratio") or 0.0)
    return [
        gate("private_eval_tasks_present", int(enabled.get("eval_task_count") or 0) >= 24, enabled.get("eval_task_count"), severity="hard"),
        gate("receiver_disabled_candidate_pool_present", int(disabled.get("disabled_candidate_count") or 0) > 0, disabled.get("disabled_candidate_count"), severity="hard"),
        gate("type_admissible_ok_measured", float(enabled.get("type_admissible_ok") or 0.0) > 0.0, enabled.get("type_admissible_ok"), severity="hard"),
        gate("return_shape_ok_ge_0_90", float(enabled.get("return_shape_ok") or 0.0) >= 0.90, enabled.get("return_shape_ok"), severity="hard"),
        gate("private_pass_rate_delta_ge_0_08", private_delta >= 0.08, private_delta, severity="hard"),
        gate("body_exec_pass_rate_delta_ge_0_06", body_delta >= 0.06, body_delta, severity="hard"),
        gate("no_benchmark_leakage", int(leakage.get("leakage_violation_count") or 0) == 0, leakage, severity="hard"),
        gate("no_template_or_wrapper_shortcut", int(shortcut.get("template_like_candidate_count") or 0) == 0 and int(shortcut.get("wrapper_like_candidate_count") or 0) == 0, shortcut, severity="hard"),
        gate(
            "accepted_candidate_coverage",
            coverage >= 0.70 or private_delta >= 0.12,
            {"coverage": coverage, "private_pass_rate_delta": private_delta},
            severity="hard",
        ),
    ]


def per_shape_deltas(disabled: dict[str, Any], enabled: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    shapes = set(disabled.get("per_return_shape_pass_rate") or {}) | set(enabled.get("per_return_shape_pass_rate") or {})
    for shape in sorted(shapes):
        before = float((disabled.get("per_return_shape_pass_rate") or {}).get(shape) or 0.0)
        after = float((enabled.get("per_return_shape_pass_rate") or {}).get(shape) or 0.0)
        out[shape] = {"disabled": before, "enabled": after, "delta": round(after - before, 6)}
    return out


def next_actions(ready: bool, deltas: dict[str, Any], enabled: dict[str, Any]) -> list[str]:
    if ready:
        return [
            "run exactly one bounded public 4-card calibration",
            "compare source_livecodebench type_handling residual count against the current residual packet",
            "preserve this receiver configuration only if broad transfer improves without regressions",
        ]
    actions = []
    if float(enabled.get("return_shape_ok") or 0.0) < 0.90:
        actions.append("tighten static return builder and dynamic return-shape obligations before any public calibration")
    if float(deltas.get("body_exec_pass_rate_delta") or 0.0) < 0.06:
        actions.append("make the receiver use private execution feedback, not only static return-shape vetoes")
    if float(deltas.get("private_pass_rate_delta") or 0.0) < 0.08:
        actions.append("patch candidate generation or receiver scoring; current rerank/veto is not a causal improvement")
    if int(enabled.get("no_admissible_candidate_count") or 0) > 0:
        actions.append("reduce hard-veto over-rejection or add fallback repair for tasks with no accepted candidate")
    actions.append("do not run public 4-card calibration until this private gate passes")
    return actions


def shape_for_task(task: dict[str, Any]) -> str:
    contract = contract_for_task(task)
    shape = str(contract.get("return_shape") or get_path(contract, ["return_contract", "shape"], "") or "").lower()
    if shape:
        return normalize_shape(shape)
    text = " ".join([str(task.get("prompt") or ""), str(task.get("tests") or ""), str(task.get("solution_body") or "")]).lower()
    if re.search(r"==\s*(true|false)\b| is\s+(true|false)\b", text):
        return "bool"
    if "== [" in text or "return []" in text or "list" in text:
        return "list"
    if "== {" in text or "return {}" in text or "dict" in text:
        return "dict"
    if "== (" in text or "tuple" in text:
        return "tuple"
    if re.search(r"==\s*['\"]", text) or "return '" in text or 'return "' in text:
        return "str"
    return "number"


def contract_for_task(task: dict[str, Any]) -> dict[str, Any]:
    contract = task.get("decoder_contract") if isinstance(task.get("decoder_contract"), dict) else {}
    return contract


def normalize_shape(shape: str) -> str:
    shape = shape.lower().strip()
    if shape in {"int", "float", "numeric", "integer"}:
        return "number"
    if shape in {"string"}:
        return "str"
    if shape in {"boolean"}:
        return "bool"
    return shape or "unknown"


def return_shape_static_ok(shape: str, code: str, parsed: ast.AST | None) -> bool:
    shape = normalize_shape(shape)
    if shape in {"unknown", "any", ""}:
        return True
    if parsed is None:
        return False
    returns = [node.value for node in ast.walk(parsed) if isinstance(node, ast.Return) and node.value is not None]
    if not returns:
        return False
    return any(expr_shape_matches(shape, expr, code) for expr in returns)


def expr_shape_matches(shape: str, expr: ast.AST, code: str) -> bool:
    if shape == "list":
        return isinstance(expr, (ast.List, ast.ListComp)) or name_in(expr, {"out", "result", "results", "items", "paths", "rows", "values"})
    if shape == "dict":
        return isinstance(expr, ast.Dict) or name_in(expr, {"out", "result", "payload", "counts", "freq", "mapping"})
    if shape == "tuple":
        return isinstance(expr, ast.Tuple) or name_in(expr, {"out", "result", "pair", "pairs"})
    if shape == "bool":
        return isinstance(expr, ast.Compare) or isinstance(expr, ast.BoolOp) or isinstance(expr, ast.UnaryOp) or const_is(expr, bool) or call_name(expr) in {"all", "any", "isinstance"}
    if shape == "str":
        return const_is(expr, str) or call_name(expr) in {"str", "join"} or name_in(expr, {"out", "result", "text", "message", "path", "archive", "encoded"})
    if shape == "number":
        return const_is(expr, (int, float)) or isinstance(expr, (ast.BinOp, ast.UnaryOp)) or call_name(expr) in {"abs", "int", "float", "len", "max", "min", "round", "sum"} or name_in(expr, {"out", "result", "total", "count", "best", "value", "n"})
    return True


def name_in(expr: ast.AST, names: set[str]) -> bool:
    return isinstance(expr, ast.Name) and expr.id.lower() in names


def const_is(expr: ast.AST, typ: Any) -> bool:
    return isinstance(expr, ast.Constant) and isinstance(expr.value, typ)


def call_name(expr: ast.AST) -> str:
    if not isinstance(expr, ast.Call):
        return ""
    func = expr.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def visible_argument_use_ok(task: dict[str, Any], code: str) -> bool:
    contract = contract_for_task(task)
    expected = int(contract.get("visible_arg_count_hint") or get_path(contract, ["signature", "visible_arg_count_hint"], 0) or 0)
    if expected <= 0:
        return True
    lowered = code.lower()
    if "*args" in lowered:
        return "args[0]" in lowered or "args [" in lowered
    names = set(get_path(contract, ["signature", "arguments"], []) or [])
    if not names:
        names = {"data", "other", "x", "n"}
    return any(re.search(rf"\b{re.escape(str(name).lower())}\b", lowered) for name in names)


def parse_ast(code: str) -> ast.AST | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def classify_failure(task: dict[str, Any], result: dict[str, Any]) -> str:
    text = f"{result.get('stderr') or ''}\n{result.get('stdout') or ''}".lower()
    if "timeout" in text:
        return "timeout"
    if "syntaxerror" in text or "indentationerror" in text:
        return "syntax"
    if "typeerror" in text or "nameerror" in text or "attributeerror" in text:
        return "type_handling"
    if "assertionerror" in text:
        shape = shape_for_task(task)
        if shape in {"list", "dict", "tuple", "str", "bool"}:
            return "return_shape_or_edge_case"
        return "wrong_answer"
    if "filenotfounderror" in text:
        return "edge_case"
    return "runtime"


def leakage_summary(candidates: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    violations = []
    for row in list(candidates) + list(tasks):
        for field in PUBLIC_FORBIDDEN_FIELDS:
            if row.get(field) is True:
                violations.append({"field": field, "task_id": row.get("task_id")})
    return {"leakage_violation_count": len(violations), "sample_violations": violations[:12]}


def shortcut_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    template_count = 0
    wrapper_count = 0
    for row in candidates:
        code = str(row.get("code") or "")
        if row.get("template_like_candidate") or template_like(code):
            template_count += 1
        if wrapper_like(code):
            wrapper_count += 1
    return {"template_like_candidate_count": template_count, "wrapper_like_candidate_count": wrapper_count}


def has_public_leak(row: dict[str, Any]) -> bool:
    return any(row.get(field) is True for field in PUBLIC_FORBIDDEN_FIELDS)


def natural_language_leakage(code: str) -> bool:
    lowered = code.lower()
    return any(token in lowered for token in ["here is", "this function", "as an ai", "solution:"])


def template_like(code: str) -> bool:
    compact = code.replace(" ", "").lower()
    return "pass\n" in code.lower() or "todo" in code.lower() or "notimplemented" in compact


def wrapper_like(code: str) -> bool:
    lowered = code.lower()
    if "subprocess" in lowered or "requests." in lowered or "open(" in lowered and "candidate" in lowered:
        return True
    return bool(re.search(r"eval\s*\(|exec\s*\(", lowered))


def candidate_rank(row: dict[str, Any], *, fallback: int) -> int:
    origin = str(row.get("origin") or "")
    match = re.search(r"rank(\d+)", origin)
    if match:
        return int(match.group(1))
    rank = row.get("rank")
    if isinstance(rank, int):
        return rank
    return fallback


def add_sample(samples: list[dict[str, Any]], task: dict[str, Any], residual: str, stderr: str) -> None:
    if len(samples) >= 24:
        return
    samples.append(
        {
            "task_id": task.get("task_id"),
            "category": task.get("category"),
            "return_shape": shape_for_task(task),
            "residual": residual,
            "stderr_tail": stderr[-500:],
        }
    )


def gate(name: str, passed: bool, evidence: Any, *, severity: str = "hard") -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Private Type/Shape Receiver Ablation",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Ready for public calibration: `{summary.get('ready_for_public_calibration')}`",
        f"- Matched private tasks: `{summary.get('matched_eval_task_count')}`",
        f"- Receiver disabled private pass: `{summary.get('receiver_disabled_private_pass_rate')}`",
        f"- Receiver enabled private pass: `{summary.get('receiver_enabled_private_pass_rate')}`",
        f"- Private pass delta: `{summary.get('private_pass_rate_delta')}`",
        f"- Body exec delta: `{summary.get('body_exec_pass_rate_delta')}`",
        f"- Return shape ok: `{summary.get('return_shape_ok')}`",
        f"- Type admissible ok: `{summary.get('type_admissible_ok')}`",
        f"- Accepted candidate coverage: `{summary.get('accepted_candidate_coverage_ratio')}`",
        "",
        "## Gates",
    ]
    for row in report.get("gates", []):
        lines.append(f"- {'PASS' if row.get('passed') else 'FAIL'} {row.get('gate')} ({row.get('severity')})")
    lines.extend(["", "## Next Actions"])
    for item in report.get("next_actions", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def runtime_tmp_dir() -> Path:
    configured = os.environ.get("THESEUS_RUNTIME_TMP", "").strip()
    if configured:
        runtime = Path(configured)
    elif os.name == "nt":
        runtime = Path("D:/ProjectTheseus/runtime/tmp")
    else:
        runtime = ROOT / "reports" / "tmp"
    runtime.mkdir(parents=True, exist_ok=True)
    return runtime


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ratio(num: int | float, den: int | float) -> float:
    den_f = float(den or 0.0)
    return round(float(num or 0.0) / den_f, 6) if den_f else 0.0


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:120]


def get_path(payload: Any, path: list[Any], default: Any = None) -> Any:
    cur = payload
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return default
    return default if cur is None else cur


if __name__ == "__main__":
    raise SystemExit(main())
