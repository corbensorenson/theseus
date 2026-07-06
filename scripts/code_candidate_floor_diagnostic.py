#!/usr/bin/env python3
"""Diagnose why public code transfer is floor-stuck.

This reads generated candidates and public calibration traces only as evidence.
It does not use public tests or reference solutions for training or generation.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLUG = "source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def body_from_code(code: str) -> str:
    lines = code.splitlines()
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            return "\n".join(lines[idx + 1 :])
    return code


def first_return_expr(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            try:
                return ast.unparse(node.value)
            except Exception:
                return ""
    return ""


def top_level_function_uses_varargs(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "*args" in code or "**kwargs" in code
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.args.vararg is not None or node.args.kwarg is not None
    return "*args" in code or "**kwargs" in code


def top_level_function_arg_count(code: str) -> int | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return len(node.args.args)
    return None


def has_ast_node(code: str, node_type: type[ast.AST]) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, node_type) for node in ast.walk(tree))


def assignment_count(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return len(re.findall(r"(?<![<>=!])=(?!=)", code))
    return sum(isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) for node in ast.walk(tree))


def is_trivial_return_expr(expr: str) -> bool:
    compact = re.sub(r"\s+", "", expr).lower()
    if compact in {
        "",
        "data",
        "other",
        "args",
        "extra",
        "result",
        "out",
        "values",
        "true",
        "false",
        "none",
        "0",
        "1",
        "[]",
        "{}",
        "()",
        "''",
        '""',
        "bool(data)",
        "len(data)",
    }:
        return True
    if compact.startswith(("bool(", "list(", "tuple(", "dict(", "str(", "int(")) and compact.endswith("data)"):
        return True
    return False


def candidate_features(row: dict[str, Any]) -> dict[str, Any]:
    code = str(row.get("code") or "")
    body = body_from_code(code)
    return_expr = str(row.get("candidate_return_expr") or "") or first_return_expr(code)
    lowered = code.lower()
    return {
        "varargs": top_level_function_uses_varargs(code),
        "arg_count": top_level_function_arg_count(code),
        "body_lines": len([line for line in body.splitlines() if line.strip()]),
        "loop": has_ast_node(code, (ast.For, ast.While, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)),
        "branch": has_ast_node(code, (ast.If, ast.IfExp, ast.Try, ast.BoolOp)),
        "assignments": assignment_count(code),
        "uses_import": bool(re.search(r"^\s*(from|import)\s+", code, re.MULTILINE)),
        "trivial_return": is_trivial_return_expr(return_expr),
        "direct_constant_return": is_trivial_return_expr(return_expr)
        and re.sub(r"\s+", "", return_expr).lower()
        in {"true", "false", "none", "0", "1", "[]", "{}", "()", "''", '""'},
        "uses_generic_alias_boilerplate": all(token in lowered for token in ["data = args", "other = args", "extra = args"]),
        "return_expr": return_expr,
    }


def task_card(row: dict[str, Any]) -> str:
    provenance = row.get("provenance")
    if isinstance(provenance, dict):
        card = provenance.get("card_id")
        if card:
            return str(card)
    task_id = str(row.get("task_id") or "")
    parts = task_id.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return str(row.get("card_id") or row.get("source_id") or "unknown")


def task_id(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("source_task_id") or "")


def trace_task_id(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("source_task_id") or "")


def summarize_bool_rate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    total = len(rows)
    count = sum(1 for row in rows if row.get(key))
    return {"count": count, "total": total, "rate": round(count / total, 6) if total else 0.0}


def build_report(candidates: list[dict[str, Any]], traces: list[dict[str, Any]]) -> dict[str, Any]:
    features_by_candidate = []
    candidates_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        features = candidate_features(row)
        enriched = dict(row)
        enriched["_features"] = features
        features_by_candidate.append(features)
        candidates_by_task[task_id(row)].append(enriched)

    passed_tasks: set[str] = set()
    failed_tasks: set[str] = set()
    residual_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    trace_stderr_by_task: dict[str, list[str]] = defaultdict(list)
    for row in traces:
        tid = trace_task_id(row)
        if not tid:
            continue
        if row.get("passed") is True:
            passed_tasks.add(tid)
        if row.get("passed") is False:
            failed_tasks.add(tid)
        residual = str(row.get("residual_class") or "")
        if residual:
            residual_by_task[tid][residual] += 1
        stderr_tail = str(row.get("stderr_tail") or "")
        if stderr_tail:
            trace_stderr_by_task[tid].append(stderr_tail[-800:])

    all_tasks = set(candidates_by_task) | passed_tasks | failed_tasks
    residual_counts = Counter()
    for counter in residual_by_task.values():
        residual_counts.update(counter)

    card_tasks: dict[str, set[str]] = defaultdict(set)
    task_to_card: dict[str, str] = {}
    for row in candidates:
        tid = task_id(row)
        card = task_card(row)
        if tid:
            card_tasks[card].add(tid)
            task_to_card[tid] = card

    card_summaries: dict[str, Any] = {}
    for card, tids in sorted(card_tasks.items()):
        fail_tids = sorted(tid for tid in tids if tid not in passed_tasks)
        card_candidates = [row for tid in tids for row in candidates_by_task.get(tid, [])]
        fail_candidates = [row for tid in fail_tids for row in candidates_by_task.get(tid, [])]
        fail_flags = {
            "all_candidates_trivial": sum(
                1 for tid in fail_tids if candidates_by_task.get(tid) and all(row["_features"]["trivial_return"] for row in candidates_by_task[tid])
            ),
            "all_candidates_varargs": sum(
                1 for tid in fail_tids if candidates_by_task.get(tid) and all(row["_features"]["varargs"] for row in candidates_by_task[tid])
            ),
            "no_loop_candidate": sum(
                1 for tid in fail_tids if candidates_by_task.get(tid) and not any(row["_features"]["loop"] for row in candidates_by_task[tid])
            ),
            "no_branch_candidate": sum(
                1 for tid in fail_tids if candidates_by_task.get(tid) and not any(row["_features"]["branch"] for row in candidates_by_task[tid])
            ),
            "generic_alias_boilerplate_present": sum(
                1 for tid in fail_tids if any(row["_features"]["uses_generic_alias_boilerplate"] for row in candidates_by_task.get(tid, []))
            ),
            "no_admissible_candidate_sentinel": sum(
                1
                for tid in fail_tids
                if any("no_admissible_student_candidate" in str(row.get("decoder_contract_verifier_v1_reasons") or []) for row in candidates_by_task.get(tid, []))
            ),
        }
        fail_residuals = Counter()
        for tid in fail_tids:
            fail_residuals.update(residual_by_task.get(tid, Counter()))
        card_summaries[card] = {
            "task_count": len(tids),
            "passed_task_count": sum(1 for tid in tids if tid in passed_tasks),
            "pass_rate": round(sum(1 for tid in tids if tid in passed_tasks) / len(tids), 6) if tids else 0.0,
            "candidate_count": len(card_candidates),
            "candidate_feature_rates": {
                "varargs": summarize_bool_rate([row["_features"] for row in card_candidates], "varargs"),
                "trivial_return": summarize_bool_rate([row["_features"] for row in card_candidates], "trivial_return"),
                "loop": summarize_bool_rate([row["_features"] for row in card_candidates], "loop"),
                "branch": summarize_bool_rate([row["_features"] for row in card_candidates], "branch"),
                "generic_alias_boilerplate": summarize_bool_rate([row["_features"] for row in card_candidates], "uses_generic_alias_boilerplate"),
            },
            "failed_task_feature_flags": fail_flags,
            "failed_residual_counts": dict(fail_residuals.most_common(12)),
        }

    stderr_text = "\n".join("\n".join(items) for items in trace_stderr_by_task.values()).lower()
    findings: list[str] = []
    recommendations: list[str] = []
    total_candidates = len(features_by_candidate)
    varargs_rate = sum(1 for row in features_by_candidate if row["varargs"]) / total_candidates if total_candidates else 0.0
    trivial_rate = sum(1 for row in features_by_candidate if row["trivial_return"]) / total_candidates if total_candidates else 0.0
    verifier_passed_trivial = sum(
        1
        for row in candidates
        if bool(row.get("decoder_contract_verifier_v1_passed")) and candidate_features(row)["trivial_return"]
    )
    if varargs_rate >= 0.20:
        findings.append("visible_prompt_signature_erasure_or_varargs_pressure")
        recommendations.append("prefer concrete visible signatures or synthetic argument aliases over *args for public task manifests")
    if trivial_rate >= 0.20 or verifier_passed_trivial >= 20:
        findings.append("contract_verifier_too_permissive_for_vacuous_bodies")
        recommendations.append("reject vacuous return-only bodies when prompt/category imply algorithmic, collection, predicate, or execution-shaped work")
    if any(summary["failed_task_feature_flags"]["no_loop_candidate"] for summary in card_summaries.values()):
        findings.append("skeleton_generation_underuses_loop_branch_local_plans")
        recommendations.append("infer required loop/branch/local constructs from visible prompt/category and use them before token decoding")
    if "no module named 'seaborn'" in stderr_text or "truth value of" in stderr_text:
        findings.append("adapter_or_environment_semantics_gap")
        recommendations.append("separate optional dependency gaps and pandas/numpy truth-value mistakes from pure model residuals")
    if residual_counts.get("local_code_generation_adapter_needed", 0) > 0:
        findings.append("local_adapter_generation_gap")
        recommendations.append("add local adapter generation for BigCodeBench/LiveCodeBench style interfaces before scoring as model failure")

    trigger_state = "GREEN"
    if findings:
        trigger_state = "YELLOW"
    if trivial_rate >= 0.35 and varargs_rate >= 0.25:
        trigger_state = "RED"

    return {
        "policy": "project_theseus_code_candidate_floor_diagnostic_v1",
        "created_utc": now_utc(),
        "trigger_state": trigger_state,
        "inputs": {
            "candidate_count": len(candidates),
            "trace_count": len(traces),
            "public_solutions_used": False,
            "public_tests_used_for_training_or_generation": False,
            "score_semantics": "diagnostic-only; reads generated candidates and trace residual metadata",
        },
        "summary": {
            "task_count": len(all_tasks),
            "passed_task_count": len(passed_tasks),
            "pass_rate": round(len(passed_tasks) / len(all_tasks), 6) if all_tasks else 0.0,
            "residual_counts": dict(residual_counts.most_common(16)),
            "candidate_feature_rates": {
                "varargs": summarize_bool_rate(features_by_candidate, "varargs"),
                "trivial_return": summarize_bool_rate(features_by_candidate, "trivial_return"),
                "loop": summarize_bool_rate(features_by_candidate, "loop"),
                "branch": summarize_bool_rate(features_by_candidate, "branch"),
                "generic_alias_boilerplate": summarize_bool_rate(features_by_candidate, "uses_generic_alias_boilerplate"),
                "verifier_passed_trivial_candidates": verifier_passed_trivial,
            },
            "findings": findings,
            "recommendations": recommendations,
        },
        "cards": card_summaries,
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Code Candidate Floor Diagnostic",
        "",
        f"- Trigger: {report.get('trigger_state')}",
        f"- Public pass rate in traces: {summary.get('pass_rate')} ({summary.get('passed_task_count')}/{summary.get('task_count')})",
        f"- Candidate feature rates: {json.dumps(summary.get('candidate_feature_rates', {}), sort_keys=True)}",
        f"- Residual counts: {json.dumps(summary.get('residual_counts', {}), sort_keys=True)}",
        "",
        "## Findings",
    ]
    findings = summary.get("findings") or []
    if findings:
        lines.extend(f"- {item}" for item in findings)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Recommendations")
    recommendations = summary.get("recommendations") or []
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Cards")
    for card, card_summary in sorted((report.get("cards") or {}).items()):
        lines.append(
            f"- {card}: pass_rate={card_summary.get('pass_rate')} "
            f"passed={card_summary.get('passed_task_count')}/{card_summary.get('task_count')} "
            f"failed_flags={json.dumps(card_summary.get('failed_task_feature_flags', {}), sort_keys=True)}"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        default=str(ROOT / "reports" / f"student_code_candidates_{DEFAULT_SLUG}.jsonl"),
        help="Generated candidate JSONL to diagnose.",
    )
    parser.add_argument(
        "--traces",
        default=str(ROOT / "reports" / f"real_code_benchmark_traces_{DEFAULT_SLUG}.jsonl"),
        help="Public calibration trace JSONL to summarize.",
    )
    parser.add_argument("--out", default=str(ROOT / "reports" / "code_candidate_floor_diagnostic.json"))
    parser.add_argument(
        "--markdown-out",
        default=str(ROOT / "reports" / "code_candidate_floor_diagnostic.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_path = Path(args.candidates)
    trace_path = Path(args.traces)
    report = build_report(read_jsonl(candidate_path), read_jsonl(trace_path))
    report["paths"] = {
        "candidates": str(candidate_path),
        "traces": str(trace_path),
    }
    out = Path(args.out)
    write_json(out, report)
    markdown_out = Path(args.markdown_out)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"trigger_state": report["trigger_state"], "out": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
