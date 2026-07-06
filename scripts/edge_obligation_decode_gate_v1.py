#!/usr/bin/env python3
"""Private-only edge-obligation decode gate.

This is the teacher-recommended "edge obligation" gate made operational. It
does not run public benchmarks and it does not read public tests or solutions.
It checks whether private candidate bodies satisfy the obligations that should
make Decoder V2 causal rather than merely rejective:

signature -> argument roles -> return contract -> semantic family ->
state variables -> branch/loop skeleton -> executable body.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_PRIVATE_CURRICULUM = ROOT / "data" / "private_code_curriculum" / "code_lm_closure_private_pressure_private.jsonl"
DEFAULT_PRIVATE_CANDIDATES = REPORTS / "code_lm_private_candidates_private_pressure_private.jsonl"
DEFAULT_CLOSURE_REPORT = REPORTS / "code_lm_closure_private_pressure_private.json"
DEFAULT_OUT = REPORTS / "edge_obligation_decode_gate_v1_private.json"
DEFAULT_MARKDOWN = REPORTS / "edge_obligation_decode_gate_v1_private.md"

sys.path.insert(0, str(ROOT / "scripts"))
import report_evidence_store  # noqa: E402
from code_lm_decoder_contracts import merged_decoder_contract  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-curriculum", default=str(DEFAULT_PRIVATE_CURRICULUM.relative_to(ROOT)))
    parser.add_argument("--private-candidates", default=str(DEFAULT_PRIVATE_CANDIDATES.relative_to(ROOT)))
    parser.add_argument("--closure-report", default=str(DEFAULT_CLOSURE_REPORT.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--min-heldout-tasks", type=int, default=64)
    parser.add_argument("--min-private-pass-rate", type=float, default=0.12)
    parser.add_argument("--max-exec-tasks", type=int, default=256)
    parser.add_argument("--exec-timeout-seconds", type=int, default=6)
    args = parser.parse_args()

    started = time.perf_counter()
    curriculum_path = resolve(args.private_curriculum)
    candidates_path = resolve(args.private_candidates)
    closure_path = resolve(args.closure_report)
    curriculum = read_jsonl(curriculum_path)
    candidates = read_jsonl(candidates_path)
    closure = read_json(closure_path, {})

    eval_tasks = {
        str(row.get("task_id") or row.get("source_task_id") or ""): row
        for row in curriculum
        if str(row.get("split") or "") == "eval" and not row.get("public_benchmark")
    }
    first_candidates = first_candidate_by_task(candidates, eval_tasks)
    ranked_candidates = ranked_candidate_by_task(candidates, eval_tasks)
    changed_by_ranker = [
        task_id
        for task_id, row in ranked_candidates.items()
        if first_candidates.get(task_id) is not row
    ]
    static_rows = [static_obligations(row, eval_tasks.get(task_id, {})) for task_id, row in ranked_candidates.items()]
    exec_rows = execute_private_tests(
        ranked_candidates,
        eval_tasks,
        max_tasks=max(0, int(args.max_exec_tasks)),
        timeout=max(1, int(args.exec_timeout_seconds)),
    )
    exec_by_task = {row["task_id"]: row for row in exec_rows}
    merged_rows = []
    for row in static_rows:
        exec_row = exec_by_task.get(row["task_id"], {})
        body_exec_ok = bool(exec_row.get("body_exec_ok")) if exec_row else False
        edge_obligation_ok = bool(
            row["return_shape_ok"]
            and row["type_admissible_ok"]
            and row["branch_loop_obligation_ok"]
            and row["token_level_student_generation_valid"]
            and body_exec_ok
        )
        merged_rows.append({**row, **exec_row, "edge_obligation_ok": edge_obligation_ok})

    origin_counts = Counter(candidate_mode(row) for row in ranked_candidates.values())
    rejection_reasons = Counter(
        reason
        for row in merged_rows
        for reason in row.get("failed_obligations", [])
    )
    leak_rows = [
        str(row.get("task_id") or row.get("source_task_id") or "unknown")
        for row in ranked_candidates.values()
        if row.get("canonical_solution_seen_by_solver") or row.get("public_tests_visible_to_generator")
    ]
    template_like = [row for row in merged_rows if row.get("template_like_candidate")]
    wrapper_like = [row for row in merged_rows if row.get("wrapper_like_candidate")]

    heldout_count = len(eval_tasks)
    checked_count = len(merged_rows)
    private_pass_rate = ratio(sum(1 for row in merged_rows if row.get("edge_obligation_ok")), checked_count)
    body_exec_rate = ratio(sum(1 for row in merged_rows if row.get("body_exec_ok")), len(exec_rows))
    closure_delta = number(get_path(closure, ["summary", "private_pass_rate_delta"], 0.0))
    ready = bool(
        heldout_count >= int(args.min_heldout_tasks)
        and checked_count >= int(args.min_heldout_tasks)
        and private_pass_rate >= float(args.min_private_pass_rate)
        and not leak_rows
        and not template_like
        and not wrapper_like
    )
    gates = [
        gate("private_curriculum_present", len(curriculum) > 0, {"path": rel_or_abs(curriculum_path), "rows": len(curriculum)}),
        gate("heldout_private_task_floor", heldout_count >= int(args.min_heldout_tasks), {"heldout": heldout_count, "floor": int(args.min_heldout_tasks)}),
        gate("private_candidates_present", len(candidates) > 0, {"path": rel_or_abs(candidates_path), "rows": len(candidates)}),
        gate("ranked_candidate_coverage_floor", checked_count >= int(args.min_heldout_tasks), {"checked": checked_count, "floor": int(args.min_heldout_tasks)}),
        gate("private_edge_obligation_pass_rate_floor", private_pass_rate >= float(args.min_private_pass_rate), {"pass_rate": private_pass_rate, "floor": float(args.min_private_pass_rate)}),
        gate("body_exec_private_tests_ran", len(exec_rows) > 0, {"executed": len(exec_rows), "body_exec_rate": body_exec_rate}),
        gate("no_public_leakage", not leak_rows, leak_rows[:20]),
        gate("no_template_candidates", not template_like, [row["task_id"] for row in template_like[:20]]),
        gate("no_wrapper_candidates", not wrapper_like, [row["task_id"] for row in wrapper_like[:20]]),
    ]
    report = {
        "policy": "project_theseus_edge_obligation_decode_gate_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if ready else "YELLOW",
        "ready_for_public_calibration": ready,
        "summary": {
            "private_task_count": len(curriculum),
            "heldout_private_task_count": heldout_count,
            "candidate_count": len(candidates),
            "first_candidate_count": checked_count,
            "ranked_candidate_count": len(ranked_candidates),
            "ranker_changed_task_count": len(changed_by_ranker),
            "candidate_selection_policy": "static_contract_ranker_then_private_execution_measurement",
            "executed_private_task_count": len(exec_rows),
            "private_pass_rate": private_pass_rate,
            "body_exec_pass_rate": body_exec_rate,
            "accepted_candidate_count": sum(1 for row in merged_rows if row.get("edge_obligation_ok")),
            "rejected_candidate_count": sum(1 for row in merged_rows if not row.get("edge_obligation_ok")),
            "visible_argument_count_ok": rate_field(merged_rows, "visible_argument_count_ok"),
            "return_shape_ok": rate_field(merged_rows, "return_shape_ok"),
            "type_admissible_ok": rate_field(merged_rows, "type_admissible_ok"),
            "branch_loop_obligation_ok": rate_field(merged_rows, "branch_loop_obligation_ok"),
            "token_level_student_generation_valid": rate_field(merged_rows, "token_level_student_generation_valid"),
            "body_exec_ok": body_exec_rate,
            "edge_obligation_ok": private_pass_rate,
            "candidate_origin_counts": dict(origin_counts.most_common(12)),
            "rejection_reason_counts": dict(rejection_reasons.most_common(16)),
            "leakage_violation_count": len(leak_rows),
            "template_like_candidate_count": len(template_like),
            "wrapper_like_candidate_count": len(wrapper_like),
            "closure_private_delta": closure_delta,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "inputs": {
            "private_curriculum": rel_or_abs(curriculum_path),
            "private_candidates": rel_or_abs(candidates_path),
            "closure_report": rel_or_abs(closure_path),
        },
        "gates": gates,
        "samples": {
            "failed": [compact_obligation_row(row) for row in merged_rows if not row.get("edge_obligation_ok")][:24],
            "passed": [compact_obligation_row(row) for row in merged_rows if row.get("edge_obligation_ok")][:12],
        },
        "rules": {
            "public_data": "private-only gate; public prompts/tests/solutions are not read or executed",
            "promotion": "only permits a later single public 4-card calibration; it is not itself public evidence",
            "decoder_policy": "verifier obligations must be satisfied by generated bodies, not only reported after rejection",
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    report_evidence_store.ingest_report_path(report_evidence_store.DEFAULT_DB, resolve(args.out), payload=report)
    print(json.dumps(report, indent=2))
    return 0


def first_candidate_by_task(candidates: list[dict[str, Any]], eval_tasks: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    eval_ids = set(eval_tasks)
    for row in candidates:
        if row.get("canonical_solution_seen_by_solver") or row.get("public_tests_visible_to_generator"):
            continue
        task_id = str(row.get("task_id") or row.get("source_task_id") or "")
        if task_id not in eval_ids:
            continue
        if task_id not in selected:
            selected[task_id] = row
    return selected


def ranked_candidate_by_task(candidates: list[dict[str, Any]], eval_tasks: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    eval_ids = set(eval_tasks)
    for row in candidates:
        if row.get("canonical_solution_seen_by_solver") or row.get("public_tests_visible_to_generator"):
            continue
        task_id = str(row.get("task_id") or row.get("source_task_id") or "")
        if task_id not in eval_ids:
            continue
        grouped.setdefault(task_id, []).append(row)
    return {
        task_id: max(rows, key=lambda row: static_rank_key(row, eval_tasks.get(task_id, {})))
        for task_id, rows in grouped.items()
        if rows
    }


def static_rank_key(candidate: dict[str, Any], task: dict[str, Any]) -> tuple[int, int, int, int, int]:
    obligations = static_obligations(candidate, task)
    obligation_score = sum(
        int(bool(obligations.get(name)))
        for name in [
            "visible_argument_count_ok",
            "return_shape_ok",
            "type_admissible_ok",
            "branch_loop_obligation_ok",
            "token_level_student_generation_valid",
        ]
    )
    mode = candidate_mode(candidate)
    return (
        obligation_score,
        int("contract_guided" in mode),
        int("sts_conditioned" in mode),
        int("parser_ast_completion" in mode),
        -len(str(candidate.get("code") or "")),
    )


def static_obligations(candidate: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    task_id = str(candidate.get("task_id") or candidate.get("source_task_id") or task.get("task_id") or "")
    code = str(candidate.get("code") or "")
    contract = current_decoder_contract(task)
    tree, syntax_error = parse_ast(code)
    return_shape = str(contract.get("return_shape") or get_path(contract, ["return_contract", "shape"], "") or "")
    visible_argument_count_ok = bool(tree and static_visible_arg_count_ok(tree, task, contract))
    return_shape_ok = bool(tree and static_return_shape_ok(tree, return_shape))
    type_admissible_ok = bool(candidate.get("decoder_contract_verifier_v1_passed") and candidate.get("deterministic_guardrail_passed", True))
    branch_loop_obligation_ok = bool(tree and static_branch_loop_obligation_ok(tree, task, contract))
    token_valid = bool(
        candidate.get("token_level_code_generation_learned", True)
        and not template_like_code(code)
        and not wrapper_like_code(code)
        and not syntax_error
    )
    failed = []
    for name, value in [
        ("visible_argument_count", visible_argument_count_ok),
        ("return_shape", return_shape_ok),
        ("type_admissible", type_admissible_ok),
        ("branch_loop_obligation", branch_loop_obligation_ok),
        ("token_level_generation", token_valid),
    ]:
        if not value:
            failed.append(name)
    return {
        "task_id": task_id,
        "category": str(candidate.get("category") or task.get("category") or ""),
        "candidate_generation_mode": str(candidate.get("candidate_generation_mode") or ""),
        "candidate_origin": str(candidate.get("origin") or ""),
        "return_shape": return_shape or "unknown",
        "visible_argument_count_ok": visible_argument_count_ok,
        "return_shape_ok": return_shape_ok,
        "type_admissible_ok": type_admissible_ok,
        "branch_loop_obligation_ok": branch_loop_obligation_ok,
        "token_level_student_generation_valid": token_valid,
        "template_like_candidate": template_like_code(code),
        "wrapper_like_candidate": wrapper_like_code(code),
        "syntax_error": syntax_error,
        "failed_obligations": failed,
    }


def current_decoder_contract(task: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(task, dict):
        return {}
    return merged_decoder_contract(task)


def execute_private_tests(
    candidates: dict[str, dict[str, Any]],
    eval_tasks: dict[str, dict[str, Any]],
    *,
    max_tasks: int,
    timeout: int,
) -> list[dict[str, Any]]:
    rows = []
    if max_tasks <= 0:
        return rows
    with tempfile.TemporaryDirectory(prefix="theseus_edge_obligation_") as tmp:
        root = Path(tmp)
        for task_id, candidate in list(candidates.items())[:max_tasks]:
            task = eval_tasks.get(task_id, {})
            code = str(candidate.get("code") or "")
            tests = str(task.get("tests") or "")
            if not code.strip() or not tests.strip():
                rows.append({"task_id": task_id, "body_exec_ok": False, "body_exec_error": "missing_code_or_tests"})
                continue
            path = root / f"case_{len(rows):04d}.py"
            path.write_text(code + "\n\n" + tests + "\n", encoding="utf-8")
            try:
                result = subprocess.run([sys.executable, str(path)], cwd=root, text=True, capture_output=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                rows.append({"task_id": task_id, "body_exec_ok": False, "body_exec_error": "timeout"})
                continue
            rows.append(
                {
                    "task_id": task_id,
                    "body_exec_ok": result.returncode == 0,
                    "body_exec_error": "" if result.returncode == 0 else (result.stderr or result.stdout)[-600:],
                }
            )
    return rows


def static_return_shape_ok(tree: ast.AST, shape: str) -> bool:
    shape = shape.lower()
    returns = [node.value for node in ast.walk(tree) if isinstance(node, ast.Return) and node.value is not None]
    if not returns:
        return False
    if not shape or shape in {"unknown", "any"}:
        return True
    inferred_shapes = inferred_local_return_shapes(tree)
    return any(return_expr_matches_shape(expr, shape, inferred_shapes) for expr in returns)


def return_expr_matches_shape(expr: ast.AST, shape: str, inferred_shapes: dict[str, set[str]] | None = None) -> bool:
    text = ast.unparse(expr).lower() if hasattr(ast, "unparse") else ""
    if isinstance(expr, ast.Name) and inferred_shapes and shape in inferred_shapes.get(expr.id, set()):
        return True
    if shape in {"number", "int", "float", "scalar"}:
        return isinstance(expr, (ast.Constant, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name)) and not isinstance(expr, (ast.List, ast.Dict, ast.Tuple, ast.Set))
    if shape in {"bool", "boolean"}:
        return isinstance(expr, (ast.Compare, ast.BoolOp, ast.UnaryOp)) or text in {"true", "false"} or text.startswith(("all(", "any("))
    if shape in {"string", "str"}:
        return isinstance(expr, ast.JoinedStr) or (isinstance(expr, ast.Constant) and isinstance(expr.value, str)) or any(token in text for token in [".join", ".strip", ".lower", ".upper", ".replace", "str("])
    if shape in {"list", "array", "sequence"}:
        return isinstance(expr, ast.List) or text.startswith(("list(", "sorted(")) or ".split(" in text
    if shape in {"dict", "mapping"}:
        return isinstance(expr, ast.Dict) or text.startswith("dict(")
    if shape in {"tuple", "pair"}:
        return isinstance(expr, ast.Tuple) or text.startswith("tuple(")
    if shape == "set":
        return isinstance(expr, ast.Set) or text.startswith("set(")
    return True


def inferred_local_return_shapes(tree: ast.AST) -> dict[str, set[str]]:
    shapes: dict[str, set[str]] = {}

    def add(name: str, shape: str) -> None:
        if name and shape:
            shapes.setdefault(name, set()).add(shape)

    def expr_shape(expr: ast.AST) -> str:
        text = ast.unparse(expr).lower() if hasattr(ast, "unparse") else ""
        if isinstance(expr, (ast.List, ast.ListComp)):
            return "list"
        if isinstance(expr, ast.Dict):
            return "dict"
        if isinstance(expr, ast.Set):
            return "set"
        if isinstance(expr, ast.Tuple):
            return "tuple"
        if isinstance(expr, ast.JoinedStr) or (isinstance(expr, ast.Constant) and isinstance(expr.value, str)):
            return "str"
        if isinstance(expr, ast.Constant) and isinstance(expr.value, bool):
            return "bool"
        if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)):
            return "number"
        if isinstance(expr, (ast.BinOp, ast.UnaryOp)):
            return "number"
        if isinstance(expr, ast.Call):
            if text.startswith(("list(", "sorted(")):
                return "list"
            if text.startswith("dict("):
                return "dict"
            if text.startswith("set("):
                return "set"
            if text.startswith("tuple("):
                return "tuple"
            if text.startswith("str(") or ".join(" in text:
                return "str"
        return ""

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if value is None:
                continue
            shape = expr_shape(value)
            for target in targets:
                if isinstance(target, ast.Name):
                    add(target.id, shape)
                elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    add(target.value.id, "dict")
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            add(node.target.id, "number")
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                name = call.func.value.id
                if call.func.attr in {"append", "extend", "insert", "sort", "reverse"}:
                    add(name, "list")
                elif call.func.attr == "add":
                    add(name, "set")
                elif call.func.attr in {"update", "setdefault"}:
                    add(name, "dict")
    return shapes


def static_visible_arg_count_ok(tree: ast.AST, task: dict[str, Any], contract: dict[str, Any]) -> bool:
    expected = int_or_none(contract.get("visible_arg_count_hint"))
    if expected is None or expected <= 0:
        return True
    entry = str(task.get("entry_point") or "")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not entry or node.name == entry):
            positional = len(node.args.posonlyargs) + len(node.args.args)
            return bool(node.args.vararg or positional >= expected)
    return False


def static_branch_loop_obligation_ok(tree: ast.AST, task: dict[str, Any], contract: dict[str, Any]) -> bool:
    has_return = any(isinstance(node, ast.Return) for node in ast.walk(tree))
    if not has_return:
        return False
    required = {str(item).lower() for item in contract.get("required_constructs", []) if item}
    tags = {str(item).lower() for item in task.get("tags", []) if item}
    category = str(task.get("category") or "").lower()
    structural_required = required & {
        "loop",
        "branch",
        "locals",
        "collection_ops",
        "index_or_string_ops",
        "frequency",
        "selection",
        "algorithmic_planning",
        "execution_shaped_program",
        "edge_conditions",
        "file_path",
        "csv",
        "structured_parsing",
        "system_api",
    }
    needs_structure = bool(
        structural_required
        or "branch_loop_skeleton" in tags
        or "local_state" in tags
        or any(token in category for token in ["count", "list", "string", "parser", "nested", "edge", "dict", "sequence"])
    )
    if not needs_structure:
        return True
    has_loop = any(isinstance(node, (ast.For, ast.While, ast.comprehension)) for node in ast.walk(tree))
    has_branch = any(isinstance(node, (ast.If, ast.IfExp, ast.BoolOp, ast.Compare)) for node in ast.walk(tree))
    has_local = any(isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)) for node in ast.walk(tree))
    return has_loop or (has_branch and has_local)


def parse_ast(code: str) -> tuple[ast.AST | None, str]:
    try:
        return ast.parse(code), ""
    except SyntaxError as exc:
        return None, f"{exc.__class__.__name__}: {exc.msg}"


def template_like_code(code: str) -> bool:
    lowered = code.lower()
    if "todo" in lowered or "notimplemented" in lowered or "raise notimplemented" in lowered:
        return True
    tree, _ = parse_ast(code)
    if tree is None:
        return False

    def non_docstring_body(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
        body = list(fn.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        return body

    def trivial_stmt(stmt: ast.stmt) -> bool:
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
            return True
        if isinstance(stmt, ast.Raise):
            text = ast.unparse(stmt).lower() if hasattr(ast, "unparse") else ""
            return "notimplemented" in text
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = non_docstring_body(node)
            if not body or all(trivial_stmt(stmt) for stmt in body):
                return True
    return False


def wrapper_like_code(code: str) -> bool:
    lowered = code.lower()
    blocked = ["subprocess", "os.system", "eval(", "exec(", "__import__", "requests.", "urllib."]
    return any(token in lowered for token in blocked)


def rate_field(rows: list[dict[str, Any]], field: str) -> float:
    return ratio(sum(1 for row in rows if row.get(field)), len(rows))


def ratio(num: int, den: int) -> float:
    return round(float(num) / float(den), 6) if den else 0.0


def compact_obligation_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": row.get("task_id"),
        "category": row.get("category"),
        "mode": row.get("candidate_generation_mode") or row.get("candidate_origin"),
        "return_shape": row.get("return_shape"),
        "visible_argument_count_ok": row.get("visible_argument_count_ok"),
        "body_exec_ok": row.get("body_exec_ok"),
        "failed_obligations": row.get("failed_obligations"),
        "body_exec_error": str(row.get("body_exec_error") or "")[:240],
    }


def candidate_mode(row: dict[str, Any]) -> str:
    return str(row.get("candidate_generation_mode") or row.get("origin") or "unknown")


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": "hard", "evidence": evidence}


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Edge Obligation Decode Gate V1",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Ready for public calibration: `{report.get('ready_for_public_calibration')}`",
        f"- Held-out private tasks: `{summary.get('heldout_private_task_count')}`",
        f"- Private pass rate: `{summary.get('private_pass_rate')}`",
        f"- Body exec pass rate: `{summary.get('body_exec_pass_rate')}`",
        f"- Edge obligation ok: `{summary.get('edge_obligation_ok')}`",
        "",
        "## Gates",
        "",
    ]
    for row in report.get("gates", []):
        marker = "PASS" if row.get("passed") else "FAIL"
        lines.append(f"- {marker}: `{row.get('gate')}`")
    lines.extend(["", "## Top Rejections", ""])
    for name, count in (summary.get("rejection_reason_counts") or {}).items():
        lines.append(f"- `{name}` x{count}")
    lines.append("")
    return "\n".join(lines)


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
