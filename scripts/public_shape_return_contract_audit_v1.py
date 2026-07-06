#!/usr/bin/env python3
"""Static return-contract audit for prompt-only public-shaped candidates.

This script does not execute public tests, read canonical solutions, or write
training rows. It uses visible task prompts and candidate ASTs to catch obvious
return-shape mismatch before any governed public calibration is considered.
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
    parser.add_argument("--out", default="reports/public_shape_return_contract_audit_v1.json")
    parser.add_argument("--markdown-out", default="reports/public_shape_return_contract_audit_v1.md")
    parser.add_argument("--min-expected-shape-coverage", type=float, default=0.70)
    parser.add_argument("--min-selected-compatible-task-rate", type=float, default=0.70)
    parser.add_argument("--min-any-compatible-task-rate", type=float, default=0.85)
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

    task_summaries = []
    shape_counter: Counter[str] = Counter()
    expected_counter: Counter[str] = Counter()
    incompatible_counter: Counter[str] = Counter()
    selected_compatible = 0
    any_compatible = 0
    expected_shape_tasks = 0
    ast_parseable = 0
    selected_ast_parseable = 0
    candidate_count = 0
    selected_candidate_count = 0
    cheat_counts = Counter()

    for task_id, task in task_by_id.items():
        prompt = str(task.get("prompt") or "")
        entry_point = str(task.get("entry_point") or "")
        expected = infer_expected_shapes(prompt, entry_point)
        if expected:
            expected_shape_tasks += 1
            for shape in expected:
                expected_counter[shape] += 1
        rows = candidates_by_task.get(task_id, [])
        candidate_infos = []
        for idx, candidate in enumerate(rows):
            candidate_count += 1
            if idx == 0:
                selected_candidate_count += 1
            update_cheat_counts(cheat_counts, candidate)
            info = inspect_candidate(candidate, entry_point)
            candidate_infos.append(info)
            if info["ast_parseable"]:
                ast_parseable += 1
                if idx == 0:
                    selected_ast_parseable += 1
            for shape in info["return_shapes"]:
                shape_counter[shape] += 1
        selected_info = candidate_infos[0] if candidate_infos else empty_candidate_info()
        selected_match = bool(expected and compatible(expected, selected_info["return_shapes"]))
        any_match = bool(expected and any(compatible(expected, info["return_shapes"]) for info in candidate_infos))
        if selected_match:
            selected_compatible += 1
        if any_match:
            any_compatible += 1
        if expected and not selected_match:
            for shape in selected_info["return_shapes"] or ["unknown"]:
                incompatible_counter[shape] += 1
        task_summaries.append(
            {
                "task_id": task_id,
                "entry_point": entry_point,
                "prompt_sha256": sha256(prompt),
                "expected_return_shapes": sorted(expected),
                "candidate_count": len(rows),
                "selected_return_shapes": sorted(selected_info["return_shapes"]),
                "selected_ast_parseable": selected_info["ast_parseable"],
                "selected_shape_compatible": selected_match,
                "any_shape_compatible": any_match,
            }
        )

    task_count = len(task_by_id)
    expected_shape_coverage_rate = ratio(expected_shape_tasks, task_count)
    selected_rate = ratio(selected_compatible, expected_shape_tasks)
    any_rate = ratio(any_compatible, expected_shape_tasks)
    ast_parse_rate = ratio(ast_parseable, candidate_count)
    selected_ast_parse_rate = ratio(selected_ast_parseable, selected_candidate_count)
    no_cheat_clean = all(value == 0 for value in cheat_counts.values())
    gates = [
        gate("task_manifest_loaded", task_count > 0, {"task_count": task_count}),
        gate("candidate_manifest_loaded", candidate_count > 0, {"candidate_count": candidate_count}),
        gate(
            "public_tests_and_solutions_not_visible",
            all(not row.get("tests_exported") and not row.get("canonical_solution_exported") for row in tasks),
            "task exporter flags only; no public tests or canonical solutions read",
        ),
        gate("candidate_no_cheat_clean", no_cheat_clean, dict(cheat_counts)),
        gate("candidate_ast_parse_rate_ok", ast_parse_rate >= 0.95, ast_parse_rate),
        gate("selected_ast_parse_rate_ok", selected_ast_parse_rate >= 0.95, selected_ast_parse_rate),
        gate(
            "expected_shape_coverage_ok",
            expected_shape_coverage_rate >= max(0.0, float(args.min_expected_shape_coverage)),
            expected_shape_coverage_rate,
        ),
        gate(
            "selected_shape_compatible_rate_ok",
            selected_rate >= max(0.0, float(args.min_selected_compatible_task_rate)),
            selected_rate,
        ),
        gate(
            "any_shape_compatible_rate_ok",
            any_rate >= max(0.0, float(args.min_any_compatible_task_rate)),
            any_rate,
        ),
        gate("report_does_not_emit_public_prompts_or_candidate_code", True, "hashes and aggregate shape labels only"),
        gate("external_inference_zero", True, 0),
    ]
    hard_no_cheat_failed = not no_cheat_clean
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "YELLOW"
    if hard_no_cheat_failed or task_count == 0 or candidate_count == 0:
        trigger_state = "RED"
    worst = [
        row
        for row in task_summaries
        if row["expected_return_shapes"] and not row["selected_shape_compatible"]
    ][:24]
    return {
        "policy": "project_theseus_public_shape_return_contract_audit_v1",
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
            "expected_shape_task_count": expected_shape_tasks,
            "expected_shape_coverage_rate": expected_shape_coverage_rate,
            "candidate_ast_parse_rate": ast_parse_rate,
            "selected_ast_parse_rate": selected_ast_parse_rate,
            "selected_shape_compatible_task_count": selected_compatible,
            "selected_shape_compatible_task_rate": selected_rate,
            "any_shape_compatible_task_count": any_compatible,
            "any_shape_compatible_task_rate": any_rate,
            "expected_shape_counts": dict(sorted(expected_counter.items())),
            "candidate_return_shape_counts": dict(sorted(shape_counter.items())),
            "selected_incompatible_shape_counts": dict(sorted(incompatible_counter.items())),
            "cheat_counts": dict(sorted(cheat_counts.items())),
            "external_inference_calls": 0,
        },
        "gates": gates,
        "sample_mismatches": worst,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def infer_expected_shapes(prompt: str, entry_point: str) -> set[str]:
    annotated = infer_annotation_shapes(prompt)
    if annotated:
        return annotated
    text = normalize_words(f"{entry_point} {prompt}")
    if entry_point == "solve":
        return {"str"}
    shapes: set[str] = set()
    bool_from_name = entry_point.lower().startswith(("is_", "has_", "can_", "are_"))
    bool_from_text = any(phrase in text for phrase in [
        "check whether",
        "check if",
        "whether",
        "true or false",
        "return true",
        "return false",
        "is valid",
        "are valid",
    ])
    if bool_from_text or bool_from_name:
        shapes.add("bool")
    if any(phrase in text for phrase in [
        "return a list",
        "return list",
        "returns list",
        "returns a list",
        "returns an empty list",
        "return only positive numbers",
        "paths to the split files",
        "find all",
        "all elements",
        "list of",
        "array",
        "remove",
        "sort",
        "filter",
        "tuples which",
        "given list",
        "from the list",
    ]):
        shapes.add("list")
    if any(phrase in text for phrase in ["return a tuple", "return tuple", "find tuples", "tuple of"]):
        shapes.add("tuple")
    if any(phrase in text for phrase in ["return a dictionary", "return dictionary", "dict", "frequency", "mapping"]):
        shapes.add("dict")
    if any(phrase in text for phrase in ["return a set", "return set", "unique elements"]):
        shapes.add("set")
    if any(phrase in text for phrase in [
        "return a string",
        "return string",
        "returns string",
        "returns a string",
        "returns decoded string",
        "returns encoded string",
        "returns str",
        "return str",
        "returns the path",
        "returns path",
        "path to the backup file",
        "path to the generated zip file",
        "base64 encoded encrypted message",
        "encrypted message",
        "character made",
        "ascii value",
        "text",
        "sentence",
        "word",
        "substring",
        "binary string",
        "roman",
    ]):
        shapes.add("str")
    if any(phrase in text for phrase in [
        "count",
        "number of",
        "sum",
        "maximum",
        "minimum",
        "largest",
        "smallest",
        "area",
        "volume",
        "product",
        "gcd",
        "lcm",
        "index",
        "nth",
        "average",
        "mean",
        "difference",
        "distance",
        "length of given string",
        "return length",
        "closest smaller number",
        "calculate the value",
        "value of",
        "multiply all the numbers",
        "divide with the length",
        "n-th number",
        "n th number",
    ]):
        shapes.add("number")
    if (
        "bool" in shapes
        and "number" in shapes
        and not bool_from_name
        and numeric_output_contract_overrides_weak_bool(text)
    ):
        shapes.remove("bool")
    if "str" in shapes and "number" in shapes and string_output_contract_overrides_numeric(text):
        shapes.remove("number")
    if "bool" in shapes:
        return {"bool"}
    if shapes:
        return shapes
    return set()


def infer_annotation_shapes(prompt: str) -> set[str]:
    shapes: set[str] = set()
    for match in re.finditer(r"->\s*([^:\n]+)\s*:", prompt):
        shapes.update(type_annotation_shapes(match.group(1)))
    if shapes:
        return shapes
    returns_block = returns_block_shapes(prompt)
    if returns_block:
        return returns_block
    return shapes


def returns_block_shapes(prompt: str) -> set[str]:
    lines = prompt.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"\s*returns?\s*:\s*$", line, flags=re.IGNORECASE):
            for candidate in lines[index + 1 : index + 8]:
                cleaned = candidate.strip()
                if not cleaned:
                    continue
                cleaned = re.sub(r"^[-*]\s*", "", cleaned)
                typed_prefix = cleaned.split(":", 1)[0]
                shapes = type_annotation_shapes(typed_prefix)
                if shapes:
                    return shapes
                shapes = type_annotation_shapes(cleaned)
                if shapes:
                    return shapes
            return set()
    match = re.search(r"returns?\s*:\s*-?\s*([A-Za-z_][\w\[\], .]*)", prompt, flags=re.IGNORECASE)
    if match:
        return type_annotation_shapes(match.group(1))
    return set()


def type_annotation_shapes(annotation: str) -> set[str]:
    text = annotation.strip().lower()
    if not text:
        return set()
    # Keep this intentionally lexical. It is prompt-only contract evidence, not
    # a type checker and not public-test-derived information.
    if any(token in text for token in ["bool", "boolean"]):
        return {"bool"}
    if any(token in text for token in ["list", "array", "sequence"]):
        return {"list"}
    if "tuple" in text:
        return {"tuple"}
    if any(token in text for token in ["dict", "dictionary", "mapping"]):
        return {"dict"}
    if "set" in text:
        return {"set"}
    if any(token in text for token in ["str", "string", "path", "message"]):
        return {"str"}
    if any(token in text for token in ["int", "float", "number", "numeric"]):
        return {"number"}
    return set()


def numeric_output_contract_overrides_weak_bool(text: str) -> bool:
    return any(phrase in text for phrase in [
        "print the maximum",
        "print maximum",
        "print the minimum",
        "print minimum",
        "print the largest",
        "print largest",
        "print the smallest",
        "print smallest",
        "print the number of",
        "print number of",
        "print the count",
        "print count",
        "find the maximum",
        "find maximum",
        "find the minimum",
        "find minimum",
        "maximum number",
        "minimum number",
        "maximum value",
        "minimum value",
        "output the maximum",
        "output maximum",
        "output the minimum",
        "output minimum",
        "otherwise print -1",
        "print -1",
    ])


def string_output_contract_overrides_numeric(text: str) -> bool:
    return any(phrase in text for phrase in [
        "character made",
        "ascii value",
        "return character",
        "return a character",
    ])


def inspect_candidate(candidate: dict[str, Any], entry_point: str) -> dict[str, Any]:
    code = str(candidate.get("code") or candidate.get("candidate_code") or "")
    if not code.strip():
        return {"ast_parseable": False, "return_shapes": {"unknown"}}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"ast_parseable": False, "return_shapes": {"syntax_error"}}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    target = next((node for node in functions if node.name == entry_point), functions[0] if functions else None)
    if target is None:
        return {"ast_parseable": True, "return_shapes": {"unknown"}}
    assignments: dict[str, ast.AST] = {}
    shapes: set[str] = set()
    for node in ast.walk(target):
        if isinstance(node, ast.Assign):
            for target_node in node.targets:
                if isinstance(target_node, ast.Name):
                    assignments[target_node.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assignments[node.target.id] = node.value or ast.Constant(None)
        elif isinstance(node, ast.Return):
            shapes.update(expr_shapes(node.value, assignments))
    return {"ast_parseable": True, "return_shapes": shapes or {"unknown"}}


def expr_shapes(expr: ast.AST | None, assignments: dict[str, ast.AST]) -> set[str]:
    if expr is None:
        return {"none"}
    if isinstance(expr, ast.Name) and expr.id in assignments:
        return expr_shapes(assignments[expr.id], assignments)
    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, bool):
            return {"bool"}
        if isinstance(expr.value, (int, float, complex)):
            return {"number"}
        if isinstance(expr.value, str):
            return {"str"}
        if expr.value is None:
            return {"none"}
    if isinstance(expr, (ast.List, ast.ListComp)):
        return {"list"}
    if isinstance(expr, ast.Tuple):
        return {"tuple"}
    if isinstance(expr, (ast.Dict, ast.DictComp)):
        return {"dict"}
    if isinstance(expr, (ast.Set, ast.SetComp)):
        return {"set"}
    if isinstance(expr, ast.Compare):
        return {"bool"}
    if isinstance(expr, ast.BoolOp):
        shapes: set[str] = set()
        for value in expr.values:
            shapes.update(expr_shapes(value, assignments))
        concrete = {shape for shape in shapes if shape != "unknown"}
        if concrete:
            return concrete
        return shapes or {"bool"}
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return {"bool"}
    if isinstance(expr, ast.IfExp):
        return expr_shapes(expr.body, assignments) | expr_shapes(expr.orelse, assignments)
    if isinstance(expr, ast.BinOp):
        left = expr_shapes(expr.left, assignments)
        right = expr_shapes(expr.right, assignments)
        if "str" in left or "str" in right:
            return {"str"}
        if "list" in left or "list" in right:
            return {"list"}
        return {"number"}
    if isinstance(expr, ast.Call):
        name = call_name(expr).lower()
        if name in {"bool", "any", "all", "isinstance"} or name.endswith((".endswith", ".startswith", ".isdigit", ".isalpha", ".isalnum")):
            return {"bool"}
        if name in {"int", "float", "len", "sum", "max", "min", "abs", "round", "ord"}:
            return {"number"}
        if name in {
            "str",
            "chr",
            "join",
            "strip",
            "lstrip",
            "rstrip",
            "lower",
            "upper",
            "replace",
            "format",
        } or name.endswith(
            (
                ".join",
                ".strip",
                ".lstrip",
                ".rstrip",
                ".lower",
                ".upper",
                ".replace",
                ".format",
            )
        ):
            return {"str"}
        if name in {"list", "sorted", "split"} or name.endswith(".split"):
            return {"list"}
        if name in {"tuple"}:
            return {"tuple"}
        if name in {"dict"}:
            return {"dict"}
        if name in {"set"}:
            return {"set"}
    return {"unknown"}


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


def compatible(expected: set[str], actual: set[str]) -> bool:
    if not expected or not actual:
        return False
    if "unknown" in actual or "syntax_error" in actual:
        return False
    if expected.intersection(actual):
        return True
    if "number" in expected and "bool" in actual:
        return False
    if "list" in expected and "tuple" in actual:
        return True
    if "tuple" in expected and "list" in actual:
        return True
    return False


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


def empty_candidate_info() -> dict[str, Any]:
    return {"ast_parseable": False, "return_shapes": {"unknown"}}


def normalize_words(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").lower()).strip()


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


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
        "# Public Shape Return Contract Audit v1",
        "",
        f"- State: `{report['trigger_state']}`",
        f"- Tasks/candidates: `{summary['task_count']}` / `{summary['candidate_count']}`",
        f"- Expected-shape coverage: `{summary['expected_shape_coverage_rate']}`",
        f"- Selected shape-compatible rate: `{summary['selected_shape_compatible_task_rate']}`",
        f"- Any-candidate shape-compatible rate: `{summary['any_shape_compatible_task_rate']}`",
        f"- Candidate AST parse rate: `{summary['candidate_ast_parse_rate']}`",
        f"- Selected AST parse rate: `{summary['selected_ast_parse_rate']}`",
        f"- Cheat counts: `{summary['cheat_counts']}`",
        "",
        "## Failed Gates",
    ]
    failed = [row for row in report["gates"] if not row["passed"]]
    if not failed:
        lines.append("- none")
    else:
        for row in failed:
            lines.append(f"- `{row['gate']}`: `{row['evidence']}`")
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
