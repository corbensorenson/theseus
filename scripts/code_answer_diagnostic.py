"""Diagnose public code benchmark answers at the candidate-body level.

This report is diagnostic-only. It reads public calibration traces and student
candidate bodies to explain which generated answers pass, which fail, and what
decoder behavior separates them. It does not train, patch, or export public
solutions/tests into any private curriculum.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = "reports/real_code_benchmark_graduation_source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32.json"
DEFAULT_TRACE = "reports/real_code_benchmark_traces_source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32.jsonl"
DEFAULT_CANDIDATES = "reports/student_code_candidates_source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32.jsonl"
DEFAULT_TASKS = "reports/code_lm_public_tasks_source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32.jsonl"


FORMULA_CATEGORIES = {
    "ascii_mod_char",
    "bell_number_sequence",
    "centered_hexagonal_number",
    "cube_lateral_surface_area",
    "cube_number",
    "cube_volume",
    "cylinder_lateral_surface_area",
    "difference_of_squares_check",
    "divisible_by_11",
    "largest_divisor",
    "largest_prime_factor",
    "modular_power_two",
    "newman_conway_sequence",
    "polygonal_centered_hexagonal_number",
    "polygonal_octagonal_number",
    "polygonal_tetrahedral_number",
    "sphere_surface_area",
    "sphere_volume",
    "triangle_area_product",
    "triangle_area_sides",
    "woodall_number_check",
}

COUNT_CATEGORIES = {
    "count_digit_under_divisibility",
    "count_distinct_characters",
    "count_integer_items",
    "count_primes_below",
    "count_truthy",
    "frequency_dict",
    "negative_count",
    "overlapping_substring_count",
    "positive_count",
    "string_char_count",
    "substring_count",
    "tuple_frequency_dict",
    "word_count",
}

NESTED_OR_SHAPE_CATEGORIES = {
    "flatten_once",
    "list_chunks_every_n",
    "matrix_diagonal",
    "nested_flat_sum",
    "nested_sum",
    "transpose_matrix",
    "tuple_nested_elementwise_max",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-code-report", default=DEFAULT_REPORT)
    parser.add_argument("--trace-in", default=DEFAULT_TRACE)
    parser.add_argument("--candidate-manifest", default=DEFAULT_CANDIDATES)
    parser.add_argument("--public-task-manifest", default=DEFAULT_TASKS)
    parser.add_argument("--out", default="reports/code_answer_diagnostic.json")
    parser.add_argument("--markdown-out", default="reports/code_answer_diagnostic.md")
    parser.add_argument("--max-examples-per-bucket", type=int, default=5)
    args = parser.parse_args()

    started = time.perf_counter()
    real_code_path = resolve(args.real_code_report)
    trace_path = resolve(args.trace_in)
    candidate_path = resolve(args.candidate_manifest)
    task_path = resolve(args.public_task_manifest)

    real_code = read_json(real_code_path, {})
    traces = read_jsonl(trace_path)
    candidates = read_jsonl(candidate_path)
    public_tasks = read_jsonl(task_path)

    task_context = build_task_context(public_tasks, candidates, real_code)
    candidate_index = build_candidate_index(candidates)
    task_runs = aggregate_task_runs(traces, candidate_index, task_context)

    candidate_rows = [
        analyze_attempt(attempt, task_context.get(attempt["task_id"], {}), candidate_index)
        for run in task_runs.values()
        for attempt in run["attempts"]
        if attempt.get("event") == "real_code_candidate_test"
    ]
    final_rows = [
        analyze_final_run(run, task_context.get(task_id, {}), candidate_index)
        for task_id, run in sorted(task_runs.items())
    ]

    summary = summarize(real_code, task_runs, candidate_rows, final_rows, started)
    pass_examples = examples_for(final_rows, passed=True, limit=args.max_examples_per_bucket)
    fail_examples = examples_for(final_rows, passed=False, limit=args.max_examples_per_bucket)
    brittle_pass_examples = brittle_examples(final_rows, limit=args.max_examples_per_bucket)
    missing_candidate_tasks = [
        row for row in final_rows if "missing_student_candidate" in row.get("final_origin", "")
    ][: args.max_examples_per_bucket]

    payload = {
        "policy": "project_theseus_code_answer_diagnostic_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if final_rows else "YELLOW",
        "score_semantics": "diagnostic_only_public_calibration_analysis_not_training_input",
        "source": {
            "real_code_report": rel(real_code_path),
            "trace": rel(trace_path),
            "candidate_manifest": rel(candidate_path),
            "public_task_manifest": rel(task_path),
        },
        "summary": summary,
        "card_summary": summarize_by(final_rows, "card_id"),
        "residual_summary": summarize_by(final_rows, "final_residual_class"),
        "winner_mode_counts": Counter(
            row.get("final_mode", "unknown")
            for row in final_rows
            if row.get("multi_stream_passed")
        ),
        "attempt_mode_summary": attempt_mode_summary(candidate_rows),
        "pass_feature_counts": feature_counts(final_rows, passed=True),
        "fail_feature_counts": feature_counts(final_rows, passed=False),
        "attempt_pass_feature_counts": attempt_feature_counts(candidate_rows, passed=True),
        "attempt_fail_feature_counts": attempt_feature_counts(candidate_rows, passed=False),
        "feature_lift_fail_minus_pass": feature_lift(final_rows),
        "rank_summary": rank_summary(final_rows),
        "task_diagnostics": final_rows,
        "examples": {
            "passing_answers": pass_examples,
            "failing_answers": fail_examples,
            "brittle_passing_answers": brittle_pass_examples,
            "missing_candidate_tasks": missing_candidate_tasks,
        },
        "recommended_architecture_work": recommended_work(summary, final_rows),
        "gates": [
            gate("traces_loaded", bool(traces), f"rows={len(traces)}"),
            gate("candidate_manifest_loaded", bool(candidates), f"rows={len(candidates)}"),
            gate("public_tests_not_exported_to_training", True, "report is diagnostic-only and does not write curriculum rows"),
            gate("candidate_code_joined_to_trace", summary["joined_attempt_count"] > 0, f"joined={summary['joined_attempt_count']}"),
        ],
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_task_context(public_tasks: list[dict[str, Any]], candidates: list[dict[str, Any]], real_code: dict[str, Any]) -> dict[str, dict[str, Any]]:
    context: dict[str, dict[str, Any]] = {}
    for row in public_tasks:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
        context[task_id] = {
            "task_id": task_id,
            "card_id": str(row.get("card_id") or card_from_task_id(task_id)),
            "category": str(row.get("category") or ""),
            "entry_point": str(row.get("entry_point") or ""),
            "required_constructs": list(contract.get("required_constructs") or []),
            "return_shape": str(contract.get("return_shape") or get_path(contract, ["return_contract", "shape"], "")),
            "type_family": str(contract.get("type_family") or ""),
            "prompt_excerpt": prompt_excerpt(str(row.get("prompt") or "")),
        }
    for row in candidates:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        item = context.setdefault(
            task_id,
            {
                "task_id": task_id,
                "card_id": str(get_path(row, ["provenance", "card_id"], "") or card_from_task_id(task_id)),
                "category": str(row.get("category") or get_path(row, ["provenance", "category"], "")),
                "entry_point": str(row.get("entry_point") or ""),
                "required_constructs": [],
                "return_shape": "",
                "type_family": "",
                "prompt_excerpt": "",
            },
        )
        for key in ["category", "entry_point"]:
            if not item.get(key) and row.get(key):
                item[key] = str(row.get(key) or "")
        if not item.get("card_id"):
            item["card_id"] = str(get_path(row, ["provenance", "card_id"], "") or card_from_task_id(task_id))
    for suite in real_code.get("suites", []) if isinstance(real_code.get("suites"), list) else []:
        for task_id in suite.get("case_ids", []) if isinstance(suite.get("case_ids"), list) else []:
            item = context.setdefault(str(task_id), {"task_id": str(task_id)})
            item.setdefault("card_id", str(suite.get("card_id") or card_from_task_id(str(task_id))))
    return context


def build_candidate_index(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_task_sha: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_task_origin_sha: dict[tuple[str, str, str], dict[str, Any]] = {}
    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        code = str(row.get("code") or "")
        digest = sha256_text(code)
        row["_full_sha256"] = digest
        task_id = str(row.get("task_id") or "")
        origin = str(row.get("origin") or "")
        by_sha[digest].append(row)
        if task_id:
            by_task[task_id].append(row)
            by_task_sha[(task_id, digest)].append(row)
            if origin:
                by_task_origin_sha[(task_id, origin, digest)] = row
    return {
        "by_task_sha": by_task_sha,
        "by_task_origin_sha": by_task_origin_sha,
        "by_sha": by_sha,
        "by_task": by_task,
    }


def aggregate_task_runs(traces: list[dict[str, Any]], candidate_index: dict[str, Any], task_context: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for row in traces:
        if not isinstance(row, dict) or row.get("event") != "real_code_candidate_test":
            continue
        if row.get("mode") != "multi_stream":
            continue
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        run = runs.setdefault(
            task_id,
            {
                "task_id": task_id,
                "card_id": str(task_context.get(task_id, {}).get("card_id") or card_from_task_id(task_id)),
                "attempts": [],
            },
        )
        run["attempts"].append(row)
    for task_id, run in runs.items():
        attempts = sorted(run["attempts"], key=lambda row: int(row.get("attempt_index") or 0))
        run["attempts"] = attempts
        passed_attempt = next((row for row in attempts if bool(row.get("passed"))), None)
        final = passed_attempt or (attempts[-1] if attempts else {})
        run["multi_stream_passed"] = bool(passed_attempt)
        run["final_attempt"] = final
        run["attempt_count"] = len(attempts)
        run["candidate_count_for_task"] = len(candidate_index["by_task"].get(task_id, []))
    return runs


def analyze_attempt(attempt: dict[str, Any], context: dict[str, Any], candidate_index: dict[str, Any]) -> dict[str, Any]:
    row = candidate_for_attempt(attempt, candidate_index)
    code = str(row.get("code") or "") if row else ""
    features = code_features(code, context)
    return {
        "task_id": str(attempt.get("task_id") or ""),
        "card_id": str(context.get("card_id") or card_from_task_id(str(attempt.get("task_id") or ""))),
        "category": str(context.get("category") or row.get("category") if row else context.get("category") or ""),
        "attempt_index": int(attempt.get("attempt_index") or 0),
        "passed": bool(attempt.get("passed")),
        "residual_class": str(attempt.get("residual_class") or ""),
        "candidate_origin": str(attempt.get("candidate_origin") or ""),
        "candidate_mode": candidate_mode(str(attempt.get("candidate_origin") or "")),
        "candidate_sha256": str(attempt.get("candidate_sha256") or ""),
        "joined_candidate": bool(row),
        "code_features": features["flags"],
        "code_metrics": features["metrics"],
        "code_excerpt": code_excerpt(code),
    }


def analyze_final_run(run: dict[str, Any], context: dict[str, Any], candidate_index: dict[str, Any]) -> dict[str, Any]:
    final = run.get("final_attempt") if isinstance(run.get("final_attempt"), dict) else {}
    row = candidate_for_attempt(final, candidate_index)
    code = str(row.get("code") or "") if row else ""
    features = code_features(code, context)
    attempts = run.get("attempts") if isinstance(run.get("attempts"), list) else []
    pass_attempt = next((row for row in attempts if bool(row.get("passed"))), None)
    all_modes = [candidate_mode(str(row.get("candidate_origin") or "")) for row in attempts]
    inventory = candidate_inventory(str(run.get("task_id") or ""), candidate_index)
    return {
        "task_id": str(run.get("task_id") or ""),
        "card_id": str(context.get("card_id") or run.get("card_id") or ""),
        "category": str(context.get("category") or (row.get("category") if row else "")),
        "entry_point": str(context.get("entry_point") or (row.get("entry_point") if row else "")),
        "prompt_excerpt": str(context.get("prompt_excerpt") or ""),
        "required_constructs": list(context.get("required_constructs") or []),
        "return_shape": str(context.get("return_shape") or ""),
        "multi_stream_passed": bool(run.get("multi_stream_passed")),
        "attempt_count": int(run.get("attempt_count") or 0),
        "candidate_count_for_task": int(run.get("candidate_count_for_task") or 0),
        "candidate_inventory": inventory,
        "pass_attempt_index": int(pass_attempt.get("attempt_index") or 0) if pass_attempt else 0,
        "final_attempt_index": int(final.get("attempt_index") or 0),
        "final_origin": str(final.get("candidate_origin") or ""),
        "final_mode": candidate_mode(str(final.get("candidate_origin") or "")),
        "final_residual_class": "" if run.get("multi_stream_passed") else str(final.get("residual_class") or "unknown_failure"),
        "final_exception_class": exception_class_from_stderr(str(final.get("stderr_tail") or final.get("stderr") or "")),
        "all_attempt_modes": all_modes,
        "joined_candidate": bool(row),
        "code_features": features["flags"],
        "code_metrics": features["metrics"],
        "code_excerpt": code_excerpt(code),
    }


def candidate_for_attempt(attempt: dict[str, Any], candidate_index: dict[str, Any]) -> dict[str, Any]:
    task_id = str(attempt.get("task_id") or "")
    origin = str(attempt.get("candidate_origin") or "")
    digest = str(attempt.get("candidate_sha256") or "")
    if task_id and origin and digest:
        row = candidate_index["by_task_origin_sha"].get((task_id, origin, digest))
        if row:
            return row
    if task_id and digest:
        rows = candidate_index["by_task_sha"].get((task_id, digest), [])
        if rows:
            if origin:
                for row in rows:
                    if str(row.get("origin") or "") == origin:
                        return row
            return rows[0]
    if digest:
        rows = candidate_index["by_sha"].get(digest, [])
        if rows:
            return rows[0]
    return {}


def code_features(code: str, context: dict[str, Any]) -> dict[str, Any]:
    flags: list[str] = []
    metrics: dict[str, Any] = {
        "code_lines": len([line for line in code.splitlines() if line.strip()]),
        "parse_ok": False,
        "function_count": 0,
        "return_count": 0,
        "loop_count": 0,
        "if_count": 0,
        "assign_count": 0,
        "call_count": 0,
        "binop_count": 0,
        "compare_count": 0,
    }
    if not code.strip():
        return {"flags": ["missing_candidate_code"], "metrics": metrics}
    if re.search(r"def\s+\w+\s*\([^)]*\*args", code):
        flags.append("signature_erased_varargs")
    if "args[0] if len(args)" in code or "extra = args" in code:
        flags.append("generic_args_adapter_scaffold")
    if "other = None" in code or "extra = ()" in code:
        flags.append("placeholder_locals_scaffold")
    if re.search(r"result\s*=\s*(False|True|0|None|\[\]|\{\}|\"\"|'')", code) and re.search(r"return\s+result\b", code):
        flags.append("placeholder_result_return")
    if re.search(r"return\s+bool\((data|x|n|num|value)\)", code) or re.search(r"return\s+(data|x|n|num|value)\s*$", code, re.MULTILINE):
        flags.append("truthiness_or_identity_return")

    try:
        tree = ast.parse(code)
        metrics["parse_ok"] = True
    except SyntaxError:
        flags.append("python_parse_error")
        return {"flags": sorted(set(flags)), "metrics": metrics}

    function_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    metrics["function_count"] = len(function_defs)
    metrics["return_count"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Return))
    metrics["loop_count"] = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While, ast.comprehension)))
    metrics["if_count"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.If))
    metrics["assign_count"] = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)))
    metrics["call_count"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call))
    metrics["binop_count"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.BinOp))
    metrics["compare_count"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Compare))
    metrics["augassign_count"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.AugAssign))

    returns = [node.value for node in ast.walk(tree) if isinstance(node, ast.Return)]
    if returns and all(is_constantish_return(value) for value in returns):
        flags.append("constantish_return_only")
    if metrics["loop_count"] == 0 and "loop" in set(context.get("required_constructs") or []):
        flags.append("missing_required_loop")
    if metrics["assign_count"] == 0 and "locals" in set(context.get("required_constructs") or []):
        flags.append("missing_required_local_state")

    category = str(context.get("category") or "").lower()
    required = set(str(item) for item in context.get("required_constructs") or [])
    if ("arithmetic_formula" in required or category in FORMULA_CATEGORIES) and not has_arithmetic_signal(tree, code):
        flags.append("missing_arithmetic_obligation")
    if category in NESTED_OR_SHAPE_CATEGORIES and not has_nested_signal(tree, code):
        flags.append("missing_nested_or_shape_strategy")
    if category not in COUNT_CATEGORIES and re.search(r"\bcount\s*\+=\s*1\b", code):
        flags.append("counter_pattern_on_non_count_task")
    if category in NESTED_OR_SHAPE_CATEGORIES and re.search(r"sorted\s*\(\s*data\s*\)", code):
        flags.append("sorts_potentially_heterogeneous_input")
    if metrics["loop_count"] > 0 and metrics["augassign_count"] == 0 and category in COUNT_CATEGORIES | NESTED_OR_SHAPE_CATEGORIES:
        flags.append("loop_without_state_update")
    return {"flags": sorted(set(flags)), "metrics": metrics}


def candidate_inventory(task_id: str, candidate_index: dict[str, Any]) -> dict[str, Any]:
    rows = candidate_index["by_task"].get(task_id, [])
    reasons: Counter[str] = Counter()
    eligible = 0
    generation_modes: Counter[str] = Counter()
    parse_errors = 0
    for row in rows:
        generation_modes.update([str(row.get("candidate_generation_mode") or "unknown")])
        row_reasons = ineligible_reasons(row)
        if row_reasons:
            reasons.update(row_reasons)
        else:
            eligible += 1
        if row.get("python_parse_error"):
            parse_errors += 1
    return {
        "candidate_rows": len(rows),
        "eligible_rows": eligible,
        "ineligible_rows": len(rows) - eligible,
        "top_ineligible_reasons": dict(reasons.most_common(8)),
        "generation_modes": dict(generation_modes.most_common(8)),
        "python_parse_errors": parse_errors,
    }


def ineligible_reasons(row: dict[str, Any]) -> list[str]:
    reasons = []
    if not truthy(row.get("benchmark_promotion_eligible")):
        reasons.append(str(row.get("promotion_ineligible_reason") or "benchmark_promotion_not_eligible"))
    if not truthy(row.get("token_level_code_generation_learned")):
        reasons.append("not_token_level_learned")
    if not truthy(row.get("full_body_token_candidate")):
        reasons.append("not_full_body_candidate")
    if row.get("deterministic_guardrail_passed") is False:
        reasons.append("deterministic_guardrail_failed")
    if truthy(row.get("expression_memory_fallback")):
        reasons.append("expression_memory_fallback")
    if truthy(row.get("loop_closure_generated")):
        reasons.append("loop_closure_generated")
    if truthy(row.get("template_like_candidate")):
        reasons.append("template_like_candidate")
    return sorted(set(reason for reason in reasons if reason))


def is_constantish_return(value: ast.AST | None) -> bool:
    if value is None:
        return True
    if isinstance(value, ast.Constant):
        return True
    if isinstance(value, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
        return True
    if isinstance(value, ast.NameConstant):
        return True
    if isinstance(value, ast.Name) and value.id in {"result", "total", "count", "answer"}:
        return False
    return False


def has_arithmetic_signal(tree: ast.AST, code: str) -> bool:
    if any(isinstance(node, (ast.BinOp, ast.AugAssign)) for node in ast.walk(tree)):
        return True
    lowered = code.lower()
    return any(token in lowered for token in ["math.", "pow(", "**", "sqrt", "pi"])


def has_nested_signal(tree: ast.AST, code: str) -> bool:
    lowered = code.lower()
    if any(token in lowered for token in ["isinstance", "recursive", "stack", "extend", "yield from"]):
        return True
    return sum(1 for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))) >= 2


def summarize(real_code: dict[str, Any], task_runs: dict[str, dict[str, Any]], candidate_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]], started: float) -> dict[str, Any]:
    total = len(final_rows)
    passed = sum(1 for row in final_rows if row.get("multi_stream_passed"))
    joined = sum(1 for row in candidate_rows if row.get("joined_candidate"))
    return {
        "diagnosed_task_count": total,
        "multi_stream_passed": passed,
        "multi_stream_failed": total - passed,
        "multi_stream_pass_rate": round(passed / total, 6) if total else 0.0,
        "single_stream_pass_rate_from_report": get_path(real_code, ["summary", "single_stream_pass_rate"], None),
        "multi_stream_pass_rate_from_report": get_path(real_code, ["summary", "multi_stream_pass_rate"], None),
        "attempt_count": len(candidate_rows),
        "joined_attempt_count": joined,
        "candidate_join_rate": round(joined / len(candidate_rows), 6) if candidate_rows else 0.0,
        "tasks_missing_candidate": sum(1 for row in final_rows if "missing_student_candidate" in row.get("final_origin", "")),
        "tasks_saved_by_later_candidate": sum(1 for row in final_rows if row.get("multi_stream_passed") and int(row.get("pass_attempt_index") or 0) > 1),
        "tasks_passed_rank1": sum(1 for row in final_rows if row.get("multi_stream_passed") and int(row.get("pass_attempt_index") or 0) == 1),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def summarize_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get(key) or "none")
        bucket = buckets.setdefault(name, {"total": 0, "passed": 0, "failed": 0, "top_failure_features": Counter(), "top_modes": Counter()})
        bucket["total"] += 1
        if row.get("multi_stream_passed"):
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
            bucket["top_failure_features"].update(row.get("code_features") or [])
        bucket["top_modes"].update([str(row.get("final_mode") or "unknown")])
    out: dict[str, Any] = {}
    for name, bucket in sorted(buckets.items()):
        total = bucket["total"]
        out[name] = {
            "total": total,
            "passed": bucket["passed"],
            "failed": bucket["failed"],
            "pass_rate": round(bucket["passed"] / total, 6) if total else 0.0,
            "top_failure_features": dict(bucket["top_failure_features"].most_common(8)),
            "top_modes": dict(bucket["top_modes"].most_common(8)),
        }
    return out


def attempt_mode_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        mode = str(row.get("candidate_mode") or "unknown")
        bucket = buckets.setdefault(mode, {"attempts": 0, "passed_attempts": 0, "feature_counts": Counter()})
        bucket["attempts"] += 1
        if row.get("passed"):
            bucket["passed_attempts"] += 1
        else:
            bucket["feature_counts"].update(row.get("code_features") or [])
    out: dict[str, Any] = {}
    for mode, bucket in sorted(buckets.items()):
        attempts = int(bucket["attempts"] or 0)
        out[mode] = {
            "attempts": attempts,
            "passed_attempts": int(bucket["passed_attempts"] or 0),
            "observed_pass_rate": round(float(bucket["passed_attempts"] or 0) / attempts, 6) if attempts else 0.0,
            "top_failed_attempt_features": dict(bucket["feature_counts"].most_common(8)),
        }
    return out


def feature_counts(rows: list[dict[str, Any]], *, passed: bool) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if bool(row.get("multi_stream_passed")) == passed:
            counts.update(row.get("code_features") or [])
    return dict(counts.most_common())


def attempt_feature_counts(rows: list[dict[str, Any]], *, passed: bool) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if bool(row.get("passed")) == passed:
            counts.update(row.get("code_features") or [])
    return dict(counts.most_common())


def feature_lift(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pass_rows = [row for row in rows if row.get("multi_stream_passed")]
    fail_rows = [row for row in rows if not row.get("multi_stream_passed")]
    features = sorted({flag for row in rows for flag in row.get("code_features") or []})
    out = []
    for flag in features:
        p_rate = sum(1 for row in pass_rows if flag in (row.get("code_features") or [])) / len(pass_rows) if pass_rows else 0.0
        f_rate = sum(1 for row in fail_rows if flag in (row.get("code_features") or [])) / len(fail_rows) if fail_rows else 0.0
        out.append({"feature": flag, "pass_rate": round(p_rate, 6), "fail_rate": round(f_rate, 6), "fail_minus_pass": round(f_rate - p_rate, 6)})
    return {row["feature"]: row for row in sorted(out, key=lambda item: item["fail_minus_pass"], reverse=True)}


def rank_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempts_to_pass = Counter()
    for row in rows:
        if row.get("multi_stream_passed"):
            attempts_to_pass[str(row.get("pass_attempt_index") or 0)] += 1
    return {
        "attempt_index_for_passing_tasks": dict(sorted(attempts_to_pass.items(), key=lambda item: int(item[0]))),
        "average_attempts_all_tasks": round(sum(int(row.get("attempt_count") or 0) for row in rows) / len(rows), 3) if rows else 0.0,
        "average_pass_attempt": round(sum(int(row.get("pass_attempt_index") or 0) for row in rows if row.get("multi_stream_passed")) / max(1, sum(1 for row in rows if row.get("multi_stream_passed"))), 3),
    }


def examples_for(rows: list[dict[str, Any]], *, passed: bool, limit: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if bool(row.get("multi_stream_passed")) == passed]
    selected.sort(key=lambda row: (str(row.get("card_id")), int(row.get("pass_attempt_index") or row.get("final_attempt_index") or 999), str(row.get("task_id"))))
    return [example_row(row) for row in selected[:limit]]


def brittle_examples(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    brittle = [
        row for row in rows
        if row.get("multi_stream_passed")
        and (
            int(row.get("pass_attempt_index") or 0) > 3
            or any(flag in (row.get("code_features") or []) for flag in ["signature_erased_varargs", "generic_args_adapter_scaffold", "placeholder_locals_scaffold"])
        )
    ]
    brittle.sort(key=lambda row: (-int(row.get("pass_attempt_index") or 0), str(row.get("card_id")), str(row.get("task_id"))))
    return [example_row(row) for row in brittle[:limit]]


def example_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": row.get("task_id"),
        "card_id": row.get("card_id"),
        "category": row.get("category"),
        "entry_point": row.get("entry_point"),
        "passed": row.get("multi_stream_passed"),
        "attempt_count": row.get("attempt_count"),
        "pass_attempt_index": row.get("pass_attempt_index"),
        "final_mode": row.get("final_mode"),
        "final_residual_class": row.get("final_residual_class"),
        "final_exception_class": row.get("final_exception_class"),
        "code_features": row.get("code_features"),
        "candidate_inventory": row.get("candidate_inventory"),
        "code_excerpt": row.get("code_excerpt"),
    }


def recommended_work(summary: dict[str, Any], final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fail_features = Counter()
    for row in final_rows:
        if not row.get("multi_stream_passed"):
            fail_features.update(row.get("code_features") or [])
    items = []
    if summary.get("tasks_missing_candidate"):
        items.append(
            {
                "priority": "critical",
                "target": "candidate_inventory",
                "reason": f"{summary['tasks_missing_candidate']} tasks had no eligible local student candidate.",
                "implementation_hint": "Fix admissibility/local adapter generation before any more training; every public task should receive at least one structurally eligible candidate.",
            }
        )
    for feature, count in fail_features.most_common(8):
        if feature == "signature_erased_varargs":
            items.append(
                {
                    "priority": "critical",
                    "target": "signature_grounding",
                    "reason": f"{count} failed final answers still used erased *args signatures.",
                    "implementation_hint": "Make exact visible signatures a hard generation invariant and invalidate candidates that wrap arguments through *args.",
                }
            )
        elif feature == "missing_arithmetic_obligation":
            items.append(
                {
                    "priority": "high",
                    "target": "formula_obligation_decoder",
                    "reason": f"{count} failed final answers lacked arithmetic structure for formula-like tasks.",
                    "implementation_hint": "Add formula-plan slots: constants, operators, recurrence/index bounds, and final expression verifier before token decode.",
                }
            )
        elif feature == "placeholder_result_return":
            items.append(
                {
                    "priority": "high",
                    "target": "vacuous_body_rejection",
                    "reason": f"{count} failed final answers returned untouched placeholder state.",
                    "implementation_hint": "Reject result/count/total returns unless the variable is updated along a required path.",
                }
            )
        elif feature == "missing_nested_or_shape_strategy":
            items.append(
                {
                    "priority": "high",
                    "target": "nested_shape_planner",
                    "reason": f"{count} failed final answers lacked recursion/stack/shape logic for nested or shape-preserving tasks.",
                    "implementation_hint": "Add explicit recursive/stack skeleton choices for nested containers and matrix-like contracts.",
                }
            )
    if summary.get("tasks_saved_by_later_candidate", 0) > summary.get("tasks_passed_rank1", 0):
        items.append(
            {
                "priority": "high",
                "target": "candidate_ranking",
                "reason": f"{summary['tasks_saved_by_later_candidate']} tasks passed only because a later candidate was tried.",
                "implementation_hint": "Use the contract verifier as a scorer: prefer exact signature, required constructs, non-vacuous state update, and semantic-family-specific obligations.",
            }
        )
    items.append(
        {
            "priority": "medium",
            "target": "public_diagnostic_policy",
            "reason": "Keep this report as calibration-only evidence.",
            "implementation_hint": "Use labels and code morphology to design private source-agnostic curricula; do not train on public answers/tests.",
        }
    )
    return dedupe_recommendations(items)


def dedupe_recommendations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for item in items:
        key = str(item.get("target"))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Code Answer Diagnostic",
        "",
        f"Created: `{payload['created_utc']}`",
        "",
        "This is diagnostic-only public calibration analysis. It does not create training rows and does not use public solutions as generation input.",
        "",
        "## Summary",
        "",
        f"- Tasks diagnosed: `{summary['diagnosed_task_count']}`",
        f"- Multi-stream pass rate: `{summary['multi_stream_pass_rate']}` ({summary['multi_stream_passed']} pass / {summary['multi_stream_failed']} fail)",
        f"- Single-stream pass rate from source report: `{summary['single_stream_pass_rate_from_report']}`",
        f"- Joined trace attempts to candidate code: `{summary['joined_attempt_count']}` / `{summary['attempt_count']}`",
        f"- Missing eligible candidate tasks: `{summary['tasks_missing_candidate']}`",
        f"- Tasks saved by later candidate: `{summary['tasks_saved_by_later_candidate']}`",
        f"- Tasks passed at rank 1: `{summary['tasks_passed_rank1']}`",
        "",
        "## Card Results",
        "",
        "| Card | Pass / Total | Rate | Top Failure Features |",
        "| --- | ---: | ---: | --- |",
    ]
    for card, row in payload["card_summary"].items():
        features = ", ".join(f"{k}={v}" for k, v in list(row.get("top_failure_features", {}).items())[:4])
        lines.append(f"| {card} | {row['passed']} / {row['total']} | {row['pass_rate']} | {features} |")
    lines.extend(["", "## Failure Features", ""])
    for feature, row in list(payload["feature_lift_fail_minus_pass"].items())[:12]:
        lines.append(f"- `{feature}`: fail rate `{row['fail_rate']}`, pass rate `{row['pass_rate']}`, lift `{row['fail_minus_pass']}`")
    lines.extend(["", "## Candidate Modes", "", "| Mode | Attempts | Passed Attempts | Observed Rate | Top Failed-Attempt Features |", "| --- | ---: | ---: | ---: | --- |"])
    for mode, row in sorted(payload["attempt_mode_summary"].items(), key=lambda item: (-item[1]["attempts"], item[0]))[:12]:
        features = ", ".join(f"{k}={v}" for k, v in list(row.get("top_failed_attempt_features", {}).items())[:4])
        lines.append(f"| {mode} | {row['attempts']} | {row['passed_attempts']} | {row['observed_pass_rate']} | {features} |")
    lines.extend(["", "## What Actually Passes", ""])
    for row in payload["examples"]["passing_answers"]:
        lines.extend(example_markdown(row))
    lines.extend(["", "## What Actually Fails", ""])
    for row in payload["examples"]["failing_answers"]:
        lines.extend(example_markdown(row))
    lines.extend(["", "## Brittle Passing Answers", ""])
    for row in payload["examples"]["brittle_passing_answers"]:
        lines.extend(example_markdown(row))
    lines.extend(["", "## Recommended Work", ""])
    for item in payload["recommended_architecture_work"]:
        lines.append(f"- **{item['target']}** ({item['priority']}): {item['reason']} {item['implementation_hint']}")
    lines.append("")
    return "\n".join(lines)


def example_markdown(row: dict[str, Any]) -> list[str]:
    code = str(row.get("code_excerpt") or "").strip()
    inventory = row.get("candidate_inventory") if isinstance(row.get("candidate_inventory"), dict) else {}
    inventory_text = ""
    if inventory:
        reasons = ", ".join(f"{k}={v}" for k, v in list(inventory.get("top_ineligible_reasons", {}).items())[:3])
        inventory_text = f"- Candidate inventory: `{inventory.get('eligible_rows')}` eligible / `{inventory.get('candidate_rows')}` rows; ineligible: `{reasons}`"
    return [
        f"### `{row.get('task_id')}`",
        "",
        f"- Card/category: `{row.get('card_id')}` / `{row.get('category')}`",
        f"- Passed: `{row.get('passed')}`; attempt count: `{row.get('attempt_count')}`; pass attempt: `{row.get('pass_attempt_index')}`",
        f"- Mode: `{row.get('final_mode')}`; residual: `{row.get('final_residual_class')}`; exception: `{row.get('final_exception_class')}`",
        f"- Features: `{', '.join(row.get('code_features') or [])}`",
        inventory_text,
        "",
        "```python",
        code,
        "```",
        "",
    ]


def candidate_mode(origin: str) -> str:
    parts = origin.split(":")
    return parts[1] if len(parts) >= 2 and parts[1] else (origin or "unknown")


def exception_class_from_stderr(stderr: str) -> str:
    if not stderr:
        return ""
    matches = re.findall(r"^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|TimeoutExpired|AssertionError))(?::|$)", stderr, re.MULTILINE)
    if matches:
        return matches[-1]
    if "AssertionError" in stderr:
        return "AssertionError"
    if "TimeoutExpired" in stderr or "timed out" in stderr.lower():
        return "TimeoutExpired"
    return ""


def prompt_excerpt(prompt: str) -> str:
    prompt = re.sub(r"\s+", " ", prompt).strip()
    return prompt[:220]


def code_excerpt(code: str, *, max_lines: int = 18, max_chars: int = 1600) -> str:
    lines = code.strip().splitlines()[:max_lines]
    text = "\n".join(lines)
    return text[:max_chars]


def card_from_task_id(task_id: str) -> str:
    for card in ["source_bigcodebench", "source_livecodebench", "source_evalplus", "source_mbpp", "source_human_eval"]:
        if task_id.startswith(card):
            return card
    parts = task_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else "unknown"


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(raw, dict):
                    rows.append(raw)
    except OSError:
        return []
    return rows


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def get_path(payload: Any, path: list[Any], default: Any = None) -> Any:
    cur = payload
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif isinstance(cur, list) and isinstance(key, int) and 0 <= key < len(cur):
            cur = cur[key]
        else:
            return default
    return cur if cur is not None else default


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
