"""Build private Decoder Plan IR pressure rows.

This is a private-only bridge between residual evidence and causal code
generation. It extracts a compact plan intermediate representation from
private generated tasks and observed candidates:

signature -> argument roles -> return contract -> semantic family -> state
variables -> branch/loop skeleton -> library/API plan -> repair policy.

Public benchmark prompts/tests/answers are not emitted. Public candidate
manifests may be inspected only for residual category pressure, never for
training rows.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_PRIVATE_CURRICULUM = ROOT / "data/private_code_curriculum/code_lm_closure_private_pressure_private.jsonl"
DEFAULT_PRIVATE_CANDIDATES = REPORTS / "code_lm_private_candidates_private_pressure_private.jsonl"
DEFAULT_PUBLIC_CANDIDATES = REPORTS / "student_code_candidates_private_pressure_private.jsonl"
DEFAULT_OUT = REPORTS / "decoder_plan_ir_private_pressure.json"
DEFAULT_MARKDOWN = REPORTS / "decoder_plan_ir_private_pressure.md"
DEFAULT_ROWS_OUT = Path("D:/ProjectTheseus/training_data/decoder_plan_ir/private_train/decoder_plan_ir_rows.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-curriculum", default=str(DEFAULT_PRIVATE_CURRICULUM.relative_to(ROOT)))
    parser.add_argument("--private-candidates", default=str(DEFAULT_PRIVATE_CANDIDATES.relative_to(ROOT)))
    parser.add_argument("--public-candidates", default=str(DEFAULT_PUBLIC_CANDIDATES.relative_to(ROOT)))
    parser.add_argument("--rows-out", default=str(DEFAULT_ROWS_OUT))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--max-rows", type=int, default=2400)
    args = parser.parse_args()

    started = time.perf_counter()
    private_tasks = read_jsonl(resolve(args.private_curriculum))
    private_candidates = read_jsonl(resolve(args.private_candidates))
    public_candidates = read_jsonl(resolve(args.public_candidates))
    private_by_task = group_candidates(private_candidates)
    public_pressure = public_residual_pressure(public_candidates)
    rows = build_plan_rows(private_tasks, private_by_task, public_pressure, max_rows=max(1, args.max_rows))
    write_jsonl(resolve(args.rows_out), rows)

    coverage = coverage_summary(rows)
    gates = [
        gate("private_curriculum_present", len(private_tasks) > 0, len(private_tasks)),
        gate("private_plan_rows_written", len(rows) > 0, len(rows)),
        gate("public_rows_not_emitted", all(not row.get("public_benchmark") for row in rows), "private-only plan rows"),
        gate("public_candidate_answers_not_copied", public_answer_leak_count(rows) == 0, "no public code/tests/prompts in rows"),
        gate("plan_order_complete", coverage["complete_plan_order_rate"] >= 0.98, coverage),
        gate("return_contract_complete", coverage["return_contract_rate"] >= 0.98, coverage),
        gate("skeleton_obligations_present", coverage["skeleton_obligation_rate"] >= 0.95, coverage),
        gate("repair_policy_present", coverage["repair_policy_rate"] >= 0.98, coverage),
    ]
    report = {
        "policy": "project_theseus_decoder_plan_ir_private_pressure_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(item["passed"] for item in gates) else "YELLOW",
        "purpose": "Create source-agnostic private Plan IR pressure so Decoder V3 can choose skeletons before token decoding.",
        "inputs": {
            "private_curriculum": display(resolve(args.private_curriculum)),
            "private_candidates": display(resolve(args.private_candidates)),
            "public_candidates": display(resolve(args.public_candidates)),
        },
        "outputs": {
            "rows_out": display(resolve(args.rows_out)),
            "report": display(resolve(args.out)),
            "markdown": display(resolve(args.markdown_out)),
        },
        "summary": {
            "private_task_count": len(private_tasks),
            "private_candidate_count": len(private_candidates),
            "public_candidate_count_inspected_for_pressure_only": len(public_candidates),
            "plan_ir_row_count": len(rows),
            "return_shape_counts": dict(Counter(str(get_path(row, ["return_contract", "shape"], "unknown")) for row in rows)),
            "semantic_family_counts": dict(Counter(str(row.get("semantic_family") or "unknown") for row in rows)),
            "skeleton_kind_counts": dict(Counter(kind for row in rows for kind in row.get("branch_loop_skeleton", []))),
            "repair_signal_counts": dict(Counter(sig for row in rows for sig in row.get("repair_signals", []))),
            "public_pressure_residual_counts": public_pressure["residual_counts"],
            "coverage": coverage,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "external_inference_calls": 0,
        },
        "sample_rows": rows[:5],
        "gates": gates,
        "next_actions": [
            "feed decoder_plan_ir_rows.jsonl into the next Decoder V3 skeleton planner after the current closure/gate chain completes",
            "use repair_signals to bias skeleton choice before body generation",
            "keep public candidate artifacts pressure-only and calibration-only",
        ],
        "rules": {
            "training_surface": "private generated/local traces only",
            "public_benchmarks": "pressure categories only; no public prompts/tests/answers emitted",
            "decoder_contract": "Plan IR is a causal generation contract, not promotion evidence",
        },
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0


def build_plan_rows(
    tasks: list[dict[str, Any]],
    candidates_by_task: dict[str, list[dict[str, Any]]],
    public_pressure: dict[str, Any],
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        if len(rows) >= max_rows:
            break
        if bool(task.get("public_benchmark")):
            continue
        evidence = str(task.get("benchmark_evidence_level") or "")
        if evidence and "private" not in evidence:
            continue
        task_id = str(task.get("task_id") or "")
        contract = task.get("decoder_contract") if isinstance(task.get("decoder_contract"), dict) else {}
        candidate_rows = candidates_by_task.get(task_id, [])
        category = str(task.get("category") or contract.get("category") or "unknown")
        semantic_family = str(contract.get("type_family") or infer_semantic_family(category, task.get("prompt")))
        return_contract = contract.get("return_contract") if isinstance(contract.get("return_contract"), dict) else {}
        return_shape = str(contract.get("return_shape") or return_contract.get("shape") or infer_return_shape(category, task.get("solution_body")))
        required = [str(item) for item in contract.get("required_constructs", []) if str(item)]
        plan = {
            "policy": "project_theseus_decoder_plan_ir_row_v1",
            "row_id": "decoder_plan_ir_" + short_hash([task_id, category, return_shape])[:16],
            "created_utc": now(),
            "dataset_id": "dataset.decoder_plan_ir.private.v1",
            "license_spdx": "local-generated-provenance-only",
            "public_benchmark": False,
            "public_solutions_used": False,
            "public_tests_used": False,
            "source_task_hash": short_hash(str(task.get("source_task_id") or task_id)),
            "task_id": task_id,
            "category": category,
            "entry_point": str(task.get("entry_point") or ""),
            "signature": signature_ir(task),
            "argument_roles": contract.get("argument_roles") if isinstance(contract.get("argument_roles"), dict) else {"data": "primary_visible_input"},
            "return_contract": {
                "shape": return_shape,
                "must_preserve_container_shape": bool(return_contract.get("must_preserve_container_shape")),
                "empty_or_invalid_behavior": str(return_contract.get("empty_or_invalid_behavior") or infer_empty_behavior(return_shape, category)),
                "source": "private_contract_metadata_only",
            },
            "semantic_family": semantic_family,
            "state_variables": state_variables_for(category, return_shape, semantic_family),
            "branch_loop_skeleton": branch_loop_skeleton_for(category, return_shape, semantic_family, required),
            "library_api_plan": library_api_plan_for(category, task.get("prompt")),
            "edge_case_obligations": edge_case_obligations_for(category, return_shape, semantic_family),
            "repair_signals": repair_signals_for(candidate_rows, public_pressure, category, return_shape, required),
            "sts_conditioning_hints": sts_hints_for(candidate_rows, public_pressure, category, semantic_family),
            "decoder_feedback": {
                "candidate_count": len(candidate_rows),
                "candidate_modes": dict(Counter(str(row.get("candidate_generation_mode") or "unknown") for row in candidate_rows)),
                "candidate_feature_gaps": candidate_feature_gaps(candidate_rows, category, return_shape, required),
            },
            "causal_order": [
                "signature",
                "argument_roles",
                "return_contract",
                "semantic_family",
                "state_variables",
                "branch_loop_skeleton",
                "library_api_plan",
                "body",
                "private_execution_repair",
            ],
            "training_role": "skeleton_choice_and_repair_bias",
            "score_semantics": "private Plan IR pressure only; not benchmark evidence",
        }
        rows.append(plan)
    return rows


def signature_ir(task: dict[str, Any]) -> dict[str, Any]:
    contract = task.get("decoder_contract") if isinstance(task.get("decoder_contract"), dict) else {}
    roles = contract.get("argument_roles") if isinstance(contract.get("argument_roles"), dict) else {}
    args = list(roles.keys()) or ["data"]
    return {
        "entry_point": str(task.get("entry_point") or ""),
        "arguments": args,
        "visible_arg_count_hint": int(contract.get("visible_arg_count_hint") or len(args) or 1),
    }


def state_variables_for(category: str, return_shape: str, semantic_family: str) -> list[dict[str, str]]:
    lowered = f"{category} {return_shape} {semantic_family}".lower()
    variables: list[dict[str, str]] = []
    if "count" in lowered or return_shape in {"number", "int", "float"}:
        variables.append({"name": "total", "role": "accumulator"})
    if "max" in lowered or "largest" in lowered:
        variables.append({"name": "best", "role": "best_so_far"})
    if "min" in lowered or "smallest" in lowered:
        variables.append({"name": "best", "role": "best_so_far"})
    if return_shape == "list" or any(token in lowered for token in ["filter", "factors", "chunks", "indices", "matrix"]):
        variables.append({"name": "out", "role": "return_builder"})
    if return_shape == "dict" or "frequency" in lowered:
        variables.append({"name": "counts", "role": "mapping_accumulator"})
    if any(token in lowered for token in ["parse", "csv", "json", "archive", "path", "file"]):
        variables.append({"name": "records", "role": "structured_input_buffer"})
    if not variables:
        variables.append({"name": "result", "role": "return_value"})
    return dedupe_dicts(variables, "name")


def branch_loop_skeleton_for(category: str, return_shape: str, semantic_family: str, required: list[str]) -> list[str]:
    lowered = f"{category} {return_shape} {semantic_family} {' '.join(required)}".lower()
    skeleton: list[str] = []
    if any(token in lowered for token in ["list", "string", "collection", "frequency", "matrix", "tuple", "dict", "parse", "csv", "json"]):
        skeleton.append("loop_over_primary_input")
    if any(token in lowered for token in ["nested", "matrix", "flatten", "recursive"]):
        skeleton.append("nested_or_recursive_traversal")
    if any(token in lowered for token in ["edge", "empty", "safe", "invalid", "threshold", "if", "bool", "predicate", "filter", "positive", "negative"]):
        skeleton.append("guard_branch")
    if any(token in lowered for token in ["prime", "factor", "gcd", "divisor", "number_theory", "recurrence", "sequence"]):
        skeleton.append("bounded_numeric_loop")
    if any(token in lowered for token in ["sort", "top_k", "median"]):
        skeleton.append("ordering_step")
    if "locals" in required:
        skeleton.append("named_local_state")
    if "loop" in required and not any("loop" in item for item in skeleton):
        skeleton.append("loop_over_primary_input")
    if "branch" in required and "guard_branch" not in skeleton:
        skeleton.append("guard_branch")
    return sorted(set(skeleton)) or ["minimal_return_builder"]


def library_api_plan_for(category: str, prompt: Any) -> list[str]:
    lowered = f"{category} {prompt or ''}".lower()
    plan = []
    for token, api in [
        ("csv", "csv_module_or_split_lines"),
        ("json", "json_module"),
        ("path", "pathlib_or_os_path"),
        ("file", "pathlib_read_text"),
        ("archive", "zipfile_or_tarfile"),
        ("regex", "re_module"),
        ("dict", "dict_methods"),
        ("frequency", "dict_or_counter"),
    ]:
        if token in lowered:
            plan.append(api)
    return sorted(set(plan))


def edge_case_obligations_for(category: str, return_shape: str, semantic_family: str) -> list[str]:
    lowered = f"{category} {return_shape} {semantic_family}".lower()
    obligations = ["empty_input", "singleton_input"]
    if any(token in lowered for token in ["number", "prime", "factor", "divisor", "count", "threshold"]):
        obligations.extend(["zero_or_one_boundary", "negative_or_nonpositive_boundary"])
    if any(token in lowered for token in ["string", "parse", "csv", "json"]):
        obligations.extend(["blank_string", "punctuation_or_whitespace"])
    if any(token in lowered for token in ["list", "tuple", "matrix", "collection"]):
        obligations.extend(["preserve_order", "mixed_or_nested_shape"])
    if return_shape in {"bool", "number"}:
        obligations.append("exact_scalar_return")
    if return_shape in {"list", "dict", "str"}:
        obligations.append("exact_container_return")
    return sorted(set(obligations))


def repair_signals_for(
    candidates: list[dict[str, Any]],
    public_pressure: dict[str, Any],
    category: str,
    return_shape: str,
    required: list[str],
) -> list[str]:
    signals = set()
    gaps = candidate_feature_gaps(candidates, category, return_shape, required)
    for key, count in gaps.items():
        if count:
            signals.add(key)
    residual_counts = public_pressure.get("residual_counts", {})
    for residual, count in residual_counts.items():
        if int(count or 0) > 0:
            signals.add(f"public_pressure:{residual}")
    if not signals:
        signals.add("private_execution_repair_after_first_failure")
    return sorted(signals)


def sts_hints_for(
    candidates: list[dict[str, Any]],
    public_pressure: dict[str, Any],
    category: str,
    semantic_family: str,
) -> list[str]:
    hints = {category, semantic_family}
    for row in candidates:
        for hint in get_path(row, ["semantic_decoder_v2_plan", "sts_hints"], []) or []:
            hints.add(str(hint))
        if bool(row.get("sts_stream_conditioned")):
            hints.add("sts_stream_conditioned_candidate_seen")
    for residual in (public_pressure.get("residual_counts") or {}).keys():
        hints.add(f"residual:{residual}")
    return sorted(item for item in hints if item and item != "unknown")


def candidate_feature_gaps(
    candidates: list[dict[str, Any]],
    category: str,
    return_shape: str,
    required: list[str],
) -> dict[str, int]:
    gaps: Counter[str] = Counter()
    if not candidates:
        return {"no_candidate_observed": 1}
    for row in candidates[:12]:
        code = str(row.get("code") or "")
        features = code_features(code)
        lowered = f"{category} {' '.join(required)}".lower()
        if ("loop" in required or any(token in lowered for token in ["list", "string", "collection", "count", "filter"])) and not features["has_loop"]:
            gaps["missing_loop"] += 1
        if ("branch" in required or any(token in lowered for token in ["edge", "filter", "positive", "negative", "safe"])) and not features["has_branch"]:
            gaps["missing_branch"] += 1
        if return_shape in {"list", "dict", "str"} and features["return_count"] <= 0:
            gaps["missing_return_builder"] += 1
        if bool(row.get("template_like_candidate")):
            gaps["template_like_candidate"] += 1
        if not bool(row.get("decoder_contract_verifier_v1_passed", True)):
            for reason in row.get("decoder_contract_verifier_v1_reasons") or ["contract_verifier_failed"]:
                gaps[f"verifier:{reason}"] += 1
    return dict(gaps)


def code_features(code: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"parse_ok": False, "has_loop": False, "has_branch": False, "return_count": 0}
    return {
        "parse_ok": True,
        "has_loop": any(isinstance(node, (ast.For, ast.While, ast.AsyncFor)) for node in ast.walk(tree)),
        "has_branch": any(isinstance(node, ast.If) for node in ast.walk(tree)),
        "return_count": sum(1 for node in ast.walk(tree) if isinstance(node, ast.Return)),
    }


def group_candidates(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row.get("task_id") or get_path(row, ["provenance", "visible_task", "task_id"], ""))
        if task_id:
            grouped[task_id].append(row)
    return dict(grouped)


def public_residual_pressure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    residuals: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    for row in rows:
        if bool(row.get("public_benchmark")) is False and row.get("phase") != "public_calibration":
            continue
        category = str(row.get("category") or "unknown")
        categories[category] += 1
        for reason in row.get("decoder_contract_verifier_v1_reasons") or []:
            residuals[str(reason)] += 1
        if bool(row.get("template_like_candidate")):
            residuals["template_like_candidate"] += 1
        if not row.get("code"):
            residuals["no_admissible_candidate"] += 1
    return {"residual_counts": dict(residuals), "category_counts": dict(categories)}


def coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(1, len(rows))
    return {
        "complete_plan_order_rate": round(sum(1 for row in rows if len(row.get("causal_order") or []) >= 8) / total, 6),
        "return_contract_rate": round(sum(1 for row in rows if get_path(row, ["return_contract", "shape"], "")) / total, 6),
        "skeleton_obligation_rate": round(sum(1 for row in rows if row.get("branch_loop_skeleton")) / total, 6),
        "repair_policy_rate": round(sum(1 for row in rows if row.get("repair_signals")) / total, 6),
    }


def infer_semantic_family(category: str, prompt: Any) -> str:
    lowered = f"{category} {prompt or ''}".lower()
    if any(token in lowered for token in ["prime", "factor", "gcd", "fibonacci", "sequence", "divisor"]):
        return "number_theory_or_recurrence"
    if any(token in lowered for token in ["string", "word", "char", "substring", "parse"]):
        return "string_indexing"
    if any(token in lowered for token in ["list", "tuple", "dict", "matrix", "count", "frequency"]):
        return "collection_logic"
    if any(token in lowered for token in ["csv", "json", "file", "path", "archive"]):
        return "execution_shaped_program"
    if any(token in lowered for token in ["check", "is_", "bool", "predicate"]):
        return "predicate_logic"
    return "general_semantics"


def infer_return_shape(category: str, solution_body: Any) -> str:
    lowered = f"{category} {solution_body or ''}".lower()
    if "return [" in lowered or ".append(" in lowered:
        return "list"
    if "return {" in lowered or "counts" in lowered:
        return "dict"
    if "return true" in lowered or "return false" in lowered or category.startswith("is_"):
        return "bool"
    if "return '" in lowered or 'return "' in lowered:
        return "str"
    return "number"


def infer_empty_behavior(return_shape: str, category: str) -> str:
    if return_shape == "list":
        return "return_empty_list"
    if return_shape == "dict":
        return "return_empty_dict"
    if return_shape == "str":
        return "return_empty_string"
    if return_shape == "bool":
        return "return_false_unless_condition_met"
    if "max" in category or "min" in category:
        return "return_default_or_guarded_boundary"
    return "return_zero_or_neutral_scalar"


def public_answer_leak_count(rows: list[dict[str, Any]]) -> int:
    leak_keys = {"public_prompt", "public_tests", "canonical_solution", "solution_body", "solution_expr", "tests", "code"}
    return sum(1 for row in rows for key in leak_keys if key in row)


def dedupe_dicts(rows: list[dict[str, str]], key: str) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        value = str(row.get(key) or "")
        if value in seen:
            continue
        seen.add(value)
        out.append(row)
    return out


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def get_path(row: Any, path: list[Any], default: Any = None) -> Any:
    cur = row
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def short_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    coverage = summary.get("coverage") if isinstance(summary.get("coverage"), dict) else {}
    lines = [
        "# Decoder Plan IR Private Pressure",
        "",
        f"Generated: {report.get('created_utc')}",
        f"Trigger: **{report.get('trigger_state')}**",
        "",
        f"- Private tasks: `{summary.get('private_task_count')}`",
        f"- Plan IR rows: `{summary.get('plan_ir_row_count')}`",
        f"- Complete plan order rate: `{coverage.get('complete_plan_order_rate')}`",
        f"- Return contract rate: `{coverage.get('return_contract_rate')}`",
        f"- Skeleton obligation rate: `{coverage.get('skeleton_obligation_rate')}`",
        f"- Rows out: `{get_path(report, ['outputs', 'rows_out'], '')}`",
        "",
        "Public benchmark candidates were inspected only for residual category pressure. No public prompts, tests, answers, or code are emitted as training rows.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
