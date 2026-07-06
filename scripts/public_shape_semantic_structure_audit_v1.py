#!/usr/bin/env python3
"""Prompt-only semantic structure audit for public-shaped candidates.

This is not public calibration. It does not execute public tests, read
canonical solutions, write training rows, or emit public prompts/candidate code.
It asks a narrower static question: when a visible prompt implies iteration,
conditionals, collection construction, string handling, or numeric aggregation,
do the generated candidates contain compatible AST structure?
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", default="reports/student_token_code_tasks_capability_transfer_closure_v1.jsonl")
    parser.add_argument("--candidate-manifest", default="reports/student_code_candidates_capability_transfer_closure_v1_public_shape.jsonl")
    parser.add_argument("--out", default="reports/public_shape_semantic_structure_audit_v1.json")
    parser.add_argument("--markdown-out", default="reports/public_shape_semantic_structure_audit_v1.md")
    parser.add_argument("--min-selected-obligation-rate", type=float, default=0.60)
    parser.add_argument("--min-any-obligation-rate", type=float, default=0.75)
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started=started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace, *, started: float) -> dict[str, Any]:
    task_path = resolve(args.task_manifest)
    candidate_path = resolve(args.candidate_manifest)
    tasks = load_jsonl(task_path)
    candidates = load_jsonl(candidate_path)
    task_by_id = {str(row.get("task_id") or ""): row for row in tasks if isinstance(row, dict)}
    candidates_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if isinstance(row, dict):
            candidates_by_task[str(row.get("task_id") or "")].append(row)

    cheat_counts = Counter()
    task_count = len(task_by_id)
    candidate_count = 0
    ast_parseable = 0
    selected_satisfied = 0
    any_satisfied = 0
    obligated_tasks = 0
    total_required_obligations = 0
    total_selected_satisfied_obligations = 0
    total_any_satisfied_obligations = 0
    selected_expression_wrapped = 0
    expression_wrapped_candidates = 0
    multi_statement_candidates = 0
    required_counts = Counter()
    selected_missing_counts = Counter()
    any_single_missing_counts = Counter()
    any_missing_counts = Counter()
    fragmented_coverage_tasks = 0
    samples = []

    for task_id, task in task_by_id.items():
        prompt = str(task.get("prompt") or "")
        entry_point = str(task.get("entry_point") or "")
        required = infer_required_structures(prompt, entry_point)
        for item in required:
            required_counts[item] += 1
        rows = candidates_by_task.get(task_id, [])
        infos = []
        for idx, row in enumerate(rows):
            candidate_count += 1
            update_cheat_counts(cheat_counts, row)
            if row.get("candidate_body_structure_kind") == "learned_expression_wrapped_body":
                expression_wrapped_candidates += 1
                if idx == 0:
                    selected_expression_wrapped += 1
            if bool(row.get("multi_statement_generated_body")):
                multi_statement_candidates += 1
            info = inspect_candidate(row, entry_point, required)
            infos.append(info)
            if info["ast_parseable"]:
                ast_parseable += 1

        if not required:
            continue
        obligated_tasks += 1
        selected = infos[0] if infos else empty_info()
        selected_ok = required.issubset(selected["structures"])
        any_ok = any(required.issubset(info["structures"]) for info in infos)
        selected_satisfied += int(selected_ok)
        any_satisfied += int(any_ok)
        selected_missing = required - selected["structures"]
        best_single_missing: set[str] = set(required)
        if infos:
            best_single_missing = min(
                (required - info["structures"] for info in infos),
                key=lambda missing: (len(missing), sorted(missing)),
            )
        any_union: set[str] = set()
        for info in infos:
            any_union.update(info["structures"])
        any_missing = required - any_union
        if best_single_missing and not any_missing:
            fragmented_coverage_tasks += 1
        total_required_obligations += len(required)
        total_selected_satisfied_obligations += len(required - selected_missing)
        total_any_satisfied_obligations += len(required - best_single_missing)
        for item in selected_missing:
            selected_missing_counts[item] += 1
        for item in best_single_missing:
            any_single_missing_counts[item] += 1
        for item in any_missing:
            any_missing_counts[item] += 1
        if (selected_missing or best_single_missing or any_missing) and len(samples) < 24:
            samples.append(
                {
                    "task_id": task_id,
                    "entry_point": entry_point,
                    "prompt_sha256": sha256(prompt),
                    "required_structures": sorted(required),
                    "selected_structures": sorted(selected["structures"]),
                    "selected_missing": sorted(selected_missing),
                    "best_single_candidate_missing": sorted(best_single_missing),
                    "candidate_union_missing": sorted(any_missing),
                    "fragmented_candidate_union_only": bool(best_single_missing and not any_missing),
                    "candidate_count": len(rows),
                }
            )

    selected_task_rate = ratio(selected_satisfied, obligated_tasks)
    any_task_rate = ratio(any_satisfied, obligated_tasks)
    selected_obligation_rate = ratio(total_selected_satisfied_obligations, total_required_obligations)
    any_obligation_rate = ratio(total_any_satisfied_obligations, total_required_obligations)
    ast_parse_rate = ratio(ast_parseable, candidate_count)
    expression_wrapped_rate = ratio(expression_wrapped_candidates, candidate_count)
    selected_expression_wrapped_rate = ratio(selected_expression_wrapped, task_count)
    multi_statement_rate = ratio(multi_statement_candidates, candidate_count)
    no_cheat_clean = all(value == 0 for value in cheat_counts.values())

    gates = [
        gate("task_manifest_loaded", task_count > 0, {"task_count": task_count}, "hard"),
        gate("candidate_manifest_loaded", candidate_count > 0, {"candidate_count": candidate_count}, "hard"),
        gate("public_tests_and_solutions_not_visible", no_public_payload_flags(tasks), "task exporter flags only", "hard"),
        gate("candidate_no_cheat_clean", no_cheat_clean, dict(cheat_counts), "hard"),
        gate("candidate_ast_parse_rate_ok", ast_parse_rate >= 0.95, ast_parse_rate, "hard"),
        gate(
            "selected_semantic_structure_obligation_rate_ok",
            selected_obligation_rate >= max(0.0, float(args.min_selected_obligation_rate)),
            selected_obligation_rate,
            "warning",
        ),
        gate(
            "any_semantic_structure_obligation_rate_ok",
            any_obligation_rate >= max(0.0, float(args.min_any_obligation_rate)),
            any_obligation_rate,
            "warning",
        ),
        gate(
            "real_multi_statement_generated_body_present",
            multi_statement_candidates > 0,
            {"multi_statement_candidates": multi_statement_candidates, "candidate_count": candidate_count},
            "warning",
        ),
        gate("report_does_not_emit_public_prompts_or_candidate_code", True, "hashes and aggregate AST labels only", "hard"),
        gate("external_inference_zero", True, 0, "hard"),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    warning_failed = [row for row in gates if row["severity"] == "warning" and not row["passed"]]
    trigger_state = "RED" if hard_failed else "YELLOW" if warning_failed else "GREEN"

    return {
        "policy": "project_theseus_public_shape_semantic_structure_audit_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {
            "task_manifest": rel(task_path),
            "candidate_manifest": rel(candidate_path),
        },
        "contract": {
            "public_tests_read": False,
            "canonical_solutions_read": False,
            "public_prompts_emitted_in_report": False,
            "candidate_code_emitted_in_report": False,
            "training_rows_written": False,
            "external_inference_calls": 0,
        },
        "summary": {
            "task_count": task_count,
            "candidate_count": candidate_count,
            "obligated_task_count": obligated_tasks,
            "candidate_ast_parse_rate": ast_parse_rate,
            "selected_task_full_obligation_rate": selected_task_rate,
            "any_task_full_obligation_rate": any_task_rate,
            "selected_obligation_satisfaction_rate": selected_obligation_rate,
            "any_obligation_satisfaction_rate": any_obligation_rate,
            "required_structure_counts": dict(sorted(required_counts.items())),
            "selected_missing_structure_counts": dict(sorted(selected_missing_counts.items())),
            "any_single_candidate_missing_structure_counts": dict(sorted(any_single_missing_counts.items())),
            "candidate_union_missing_structure_counts": dict(sorted(any_missing_counts.items())),
            "fragmented_candidate_union_only_task_count": fragmented_coverage_tasks,
            "fragmented_candidate_union_only_task_rate": ratio(fragmented_coverage_tasks, obligated_tasks),
            "any_missing_structure_counts": dict(sorted(any_missing_counts.items())),
            "expression_wrapped_body_candidate_count": expression_wrapped_candidates,
            "expression_wrapped_body_candidate_rate": expression_wrapped_rate,
            "selected_expression_wrapped_task_rate": selected_expression_wrapped_rate,
            "multi_statement_generated_body_candidate_count": multi_statement_candidates,
            "multi_statement_generated_body_candidate_rate": multi_statement_rate,
            "cheat_counts": dict(sorted(cheat_counts.items())),
            "external_inference_calls": 0,
        },
        "gates": gates,
        "sample_structure_gaps": samples,
        "recommendation": recommendation(trigger_state, warning_failed),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def infer_required_structures(prompt: str, entry_point: str) -> set[str]:
    text = normalize_words(f"{entry_point} {prompt}")
    required: set[str] = set()
    if has_any_phrase(
        text,
        [
            "list",
            "array",
            "tuple",
            "each",
            "every",
            "all elements",
            "elements",
            "filter",
            "sort",
            "remove",
            "find all",
            "strings",
            "numbers",
            "items",
            "pairs",
            "subsequence",
            "subarray",
        ],
    ):
        required.add("iteration")
    if has_any_phrase(
        text,
        [
            "if",
            "whether",
            "check",
            "valid",
            "invalid",
            "positive",
            "negative",
            "empty",
            "non empty",
            "divisible",
            "greater than",
            "less than",
            "same",
            "different",
        ],
    ) or entry_point.lower().startswith(("is_", "has_", "can_", "are_")):
        required.add("conditional")
    if has_any_phrase(
        text,
        [
            "string",
            "substring",
            "word",
            "character",
            "text",
            "sentence",
            "roman",
            "binary string",
        ],
    ):
        required.add("string_processing")
    if has_any_phrase(
        text,
        [
            "return a list",
            "return list",
            "list of",
            "dictionary",
            "dict",
            "mapping",
            "frequency",
            "return a set",
            "unique",
            "return tuple",
            "tuple of",
        ],
    ):
        required.add("collection_build")
    if has_any_phrase(
        text,
        [
            "count",
            "number of",
            "sum",
            "maximum",
            "minimum",
            "largest",
            "smallest",
            "product",
            "average",
            "mean",
            "gcd",
            "lcm",
            "difference",
            "distance",
        ],
    ):
        required.add("numeric_aggregation")
    return required


def inspect_candidate(candidate: dict[str, Any], entry_point: str, required: set[str]) -> dict[str, Any]:
    code = str(candidate.get("code") or candidate.get("candidate_code") or "")
    if not code.strip():
        return empty_info()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"ast_parseable": False, "structures": set()}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    target = next((node for node in functions if node.name == entry_point), functions[0] if functions else None)
    if target is None:
        return {"ast_parseable": True, "structures": set()}
    structures: set[str] = set()
    for node in ast.walk(target):
        if isinstance(node, (ast.For, ast.While, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            structures.add("iteration")
        if isinstance(node, (ast.If, ast.IfExp, ast.Compare, ast.BoolOp)):
            structures.add("conditional")
        if isinstance(node, (ast.List, ast.Dict, ast.Set, ast.Tuple, ast.ListComp, ast.DictComp, ast.SetComp)):
            structures.add("collection_build")
        if isinstance(node, (ast.BinOp, ast.AugAssign)):
            structures.add("numeric_aggregation")
        if "string_processing" in required and isinstance(node, ast.Subscript):
            structures.add("string_processing")
        if isinstance(node, ast.Call):
            name = call_name(node).lower()
            if name in {"sum", "max", "min", "abs", "round", "len"}:
                structures.add("numeric_aggregation")
            if name in {"list", "dict", "set", "tuple", "sorted"}:
                structures.add("collection_build")
            if name in {
                "str",
                "chr",
                "ord",
                "split",
                "join",
                "strip",
                "lstrip",
                "rstrip",
                "lower",
                "upper",
                "replace",
                "find",
                "count",
                "startswith",
                "endswith",
                "isdigit",
                "isalpha",
                "isalnum",
            } or name.endswith(
                (
                    ".split",
                    ".join",
                    ".strip",
                    ".lstrip",
                    ".rstrip",
                    ".lower",
                    ".upper",
                    ".replace",
                    ".find",
                    ".count",
                    ".startswith",
                    ".endswith",
                    ".isdigit",
                    ".isalpha",
                    ".isalnum",
                )
            ):
                structures.add("string_processing")
    return {"ast_parseable": True, "structures": structures}


def call_name(expr: ast.Call) -> str:
    func = expr.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        value = func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return ""


def update_cheat_counts(counter: Counter[str], row: dict[str, Any]) -> None:
    for key in [
        "expression_memory_fallback",
        "template_like_candidate",
        "loop_closure_generated",
        "public_tests_visible_to_generator",
        "canonical_solution_seen_by_solver",
    ]:
        if bool(row.get(key)):
            counter[key] += 1
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    if bool(provenance.get("tests_used")):
        counter["provenance_tests_used"] += 1
    if bool(provenance.get("canonical_solution_used")):
        counter["provenance_canonical_solution_used"] += 1
    counter["external_inference_calls"] += int_number(row.get("external_inference_calls"))
    counter["external_inference_calls"] += int_number(provenance.get("external_inference_calls"))


def no_public_payload_flags(tasks: list[dict[str, Any]]) -> bool:
    return all(not row.get("tests_exported") and not row.get("canonical_solution_exported") for row in tasks)


def empty_info() -> dict[str, Any]:
    return {"ast_parseable": False, "structures": set()}


def recommendation(trigger_state: str, warning_failed: list[dict[str, Any]]) -> str:
    if trigger_state == "RED":
        return "Fix hard integrity/load failures before using this evidence."
    failed_names = {row["gate"] for row in warning_failed}
    if "real_multi_statement_generated_body_present" in failed_names:
        return (
            "Return shape and AST parseability are clean, but the current promotion-shaped path is still "
            "expression-wrapped. Next work should build a real learned multi-statement structural body decoder "
            "and rerun this audit before public calibration."
        )
    if warning_failed:
        return "Mine private residuals for the missing static semantic obligations before public calibration."
    fragmented = "fragmented_candidate_union_only_task_count"
    return (
        "Static semantic structure audit is green; semantic adequacy still requires private heldout "
        f"or governed calibration evidence. Check `{fragmented}` before treating union coverage as "
        "single-candidate readiness."
    )


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence, "severity": severity}


def normalize_words(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.replace("_", " ").lower())).strip()


def has_any_phrase(text: str, phrases: list[str]) -> bool:
    padded = f" {text} "
    for phrase in phrases:
        normalized = normalize_words(phrase)
        if normalized and f" {normalized} " in padded:
            return True
    return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Public Shape Semantic Structure Audit v1",
        "",
        f"- State: `{report['trigger_state']}`",
        f"- Tasks/candidates: `{summary['task_count']}` / `{summary['candidate_count']}`",
        f"- Obligated tasks: `{summary['obligated_task_count']}`",
        f"- Selected obligation satisfaction: `{summary['selected_obligation_satisfaction_rate']}`",
        f"- Best single-candidate obligation satisfaction: `{summary['any_obligation_satisfaction_rate']}`",
        f"- Fragmented union-only tasks: `{summary['fragmented_candidate_union_only_task_count']}`",
        f"- Expression-wrapped body rate: `{summary['expression_wrapped_body_candidate_rate']}`",
        f"- Multi-statement generated body rate: `{summary['multi_statement_generated_body_candidate_rate']}`",
        f"- Cheat counts: `{summary['cheat_counts']}`",
        "",
        "## Failed Gates",
    ]
    failed = [row for row in report["gates"] if not row["passed"]]
    if not failed:
        lines.append("- none")
    else:
        for row in failed:
            lines.append(f"- `{row['gate']}` ({row['severity']}): `{row['evidence']}`")
    return "\n".join(lines) + "\n"


def ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def int_number(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
