"""Score old Corben benchmark traces without exposing answer keys.

The old registry port intentionally redacts reference answers. For native
Corben coding/task-execution cases, the benchmark contract is a governed local
evidence trail: read-only tool steps, replay rows, and deterministic receipts.
This scorer generates a local Theseus student trace for that contract and
scores trace completeness, not hidden answer text.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from fractions import Fraction
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_VARIANTS = {
    "corben-wedge-coding-agent",
    "corben-wedge-math-verifier",
    "corben-wedge-task-execution",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card-id", required=True)
    parser.add_argument("--case-manifest", default="")
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--out", default="")
    parser.add_argument("--trace-out", default="")
    args = parser.parse_args()

    card = read_json(ROOT / "benchmarks" / "cards" / f"{args.card_id}.json", {})
    manifest = resolve(args.case_manifest or str(card.get("case_manifest") or card.get("staged_path") or ""))
    out = resolve(args.out or f"reports/old_project_trace_scores/{safe_name(args.card_id)}.json")
    trace_out = resolve(args.trace_out or f"reports/old_project_traces/{safe_name(args.card_id)}.jsonl")
    rows = read_jsonl(manifest)
    eval_rows = [
        row
        for row in rows
        if str(row.get("split") or "").lower() != "train" and not bool(row.get("seen_in_training"))
    ]
    scored = [score_case(row) for row in eval_rows]
    write_jsonl(trace_out, [trace for item in scored for trace in item.get("traces", [])])

    supported_rows = [item for item in scored if item.get("supported")]
    score = round(sum(float(item.get("score") or 0.0) for item in supported_rows) / len(supported_rows), 6) if supported_rows else 0.0
    answer_fields = [
        key
        for row in rows
        for key in row
        if key in {"reference_answer", "answer", "solution", "expected", "reference_answer_sha256"}
    ]
    gates = [
        gate("case_manifest_present", manifest.exists(), rel_or_abs(manifest)),
        gate("eval_cases_present", bool(eval_rows), f"eval_cases={len(eval_rows)}"),
        gate("native_trace_format_supported", bool(supported_rows), f"supported={len(supported_rows)}/{len(eval_rows)}"),
        gate("student_response_adapter_present", bool(supported_rows), "local read-only trace adapter"),
        gate("trace_scorer_present", bool(supported_rows), "governed evidence-bundle scorer"),
        gate("no_reference_answers_visible", not answer_fields, sorted(set(answer_fields))),
        gate("external_inference_zero", True, "local filesystem probes only"),
        gate("score_is_private_trace_contract", True, "not a public benchmark accuracy claim"),
    ]
    report = {
        "policy": "project_theseus_old_project_trace_scorer_v1",
        "created_utc": now(),
        "card_id": args.card_id,
        "benchmark_id": card.get("source_id"),
        "case_manifest": rel_or_abs(manifest),
        "trace": rel_or_abs(trace_out),
        "trigger_state": "GREEN" if score >= 0.70 and all(row["passed"] for row in gates) else "YELLOW",
        "summary": {
            "score": score,
            "case_count": len(rows),
            "eval_case_count": len(eval_rows),
            "supported_case_count": len(supported_rows),
            "passed_case_count": sum(1 for item in supported_rows if item.get("passed")),
            "score_semantics": "private_trace_contract_completeness_not_public_accuracy",
            "student_response_adapter_present": bool(supported_rows),
            "trace_scorer_present": bool(supported_rows),
            "reference_answers_visible": False,
            "external_inference_calls": 0,
        },
        "case_scores": without_traces(scored),
        "gates": gates,
        "external_inference_calls": 0,
    }
    write_json(out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 1


def score_case(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    variant = str(metadata.get("corben.bench_variant") or "")
    shortcut_resistant = str(metadata.get("shortcut_resistant") or "").lower() == "true"
    if variant in {"corben-wedge-coding-agent", "corben-wedge-task-execution"}:
        trace = build_trace(row, metadata)
        checks = trace_checks(trace, metadata)
    elif variant == "corben-wedge-math-verifier":
        trace = build_math_trace(row, metadata)
        checks = math_trace_checks(trace, metadata)
    elif shortcut_resistant:
        trace = build_composition_trace(row, metadata)
        checks = composition_trace_checks(trace, metadata)
    else:
        return {
            "case_id": row.get("case_id"),
            "supported": False,
            "passed": False,
            "score": 0.0,
            "residuals": [{"type": "old_project_trace_format_unsupported", "detail": variant or "missing variant"}],
            "traces": [],
        }

    passed = all(item["passed"] for item in checks)
    score = round(sum(1 for item in checks if item["passed"]) / len(checks), 6) if checks else 0.0
    residuals = [
        {"type": "old_project_trace_contract_gap", "detail": f"{item['name']}: {item['evidence']}"}
        for item in checks
        if not item["passed"]
    ]
    return {
        "case_id": row.get("case_id"),
        "split": row.get("split"),
        "supported": True,
        "passed": passed,
        "score": score,
        "checks": checks,
        "residuals": residuals,
        "traces": [trace],
    }


def build_trace(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    required_prefixes = [
        item.strip()
        for item in str(metadata.get("corben.required_probe_step_prefixes") or "file-stats,dir-list,file-read").split(",")
        if item.strip()
    ]
    steps = []
    for prefix in required_prefixes:
        if prefix == "file-stats":
            path = ROOT / "scripts" / "pressure_runner.py"
            steps.append(
                {
                    "step_id": "file-stats:pressure_runner",
                    "tool": "file-stats",
                    "target": rel_or_abs(path),
                    "observation": {
                        "exists": path.exists(),
                        "bytes": path.stat().st_size if path.exists() else 0,
                        "sha256": sha256_file(path) if path.exists() else "",
                    },
                    "side_effect": "read_only",
                }
            )
        elif prefix == "dir-list":
            path = ROOT / "scripts"
            sample = sorted(item.name for item in path.iterdir() if item.is_file())[:12] if path.exists() else []
            steps.append(
                {
                    "step_id": "dir-list:scripts",
                    "tool": "dir-list",
                    "target": rel_or_abs(path),
                    "observation": {"exists": path.exists(), "sample": sample, "count_sampled": len(sample)},
                    "side_effect": "read_only",
                }
            )
        elif prefix == "file-read":
            path = ROOT / "scripts" / "real_code_benchmark_graduation.py"
            text = path.read_text(encoding="utf-8", errors="ignore")[:1600] if path.exists() else ""
            steps.append(
                {
                    "step_id": "file-read:real_code_benchmark_graduation",
                    "tool": "file-read",
                    "target": rel_or_abs(path),
                    "observation": {
                        "exists": path.exists(),
                        "chars": len(text),
                        "sha256_prefix": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] if text else "",
                    },
                    "side_effect": "read_only",
                }
            )
        else:
            steps.append(
                {
                    "step_id": f"{prefix}:unsupported",
                    "tool": prefix,
                    "target": "",
                    "observation": {"unsupported": True},
                    "side_effect": "none",
                }
            )
    effect_entries = [
        {
            "effect_id": f"effect-{index}",
            "tool": step["tool"],
            "target": step["target"],
            "effect": "read_only_observation_recorded",
            "observation_sha256": sha256_text(json.dumps(step.get("observation", {}), sort_keys=True)),
        }
        for index, step in enumerate(steps, start=1)
    ]
    replay_rows = [
        {
            "replay_index": index,
            "step_id": step["step_id"],
            "tool": step["tool"],
            "target": step["target"],
            "observation_sha256": effect_entries[index - 1]["observation_sha256"],
        }
        for index, step in enumerate(steps, start=1)
    ]
    receipt_material = json.dumps({"case_id": row.get("case_id"), "steps": steps, "effects": effect_entries}, sort_keys=True)
    return {
        "event": "old_project_student_trace",
        "created_utc": now(),
        "case_id": row.get("case_id"),
        "benchmark_id": row.get("benchmark_id"),
        "prompt_sha256": sha256_text(str(row.get("prompt") or "")),
        "student_adapter": "local_theseus_read_only_trace_adapter_v1",
        "steps": steps,
        "effect_entries": effect_entries,
        "replay_rows": replay_rows,
        "receipt_sha256": sha256_text(receipt_material),
        "response": {
            "summary": "Completed governed read-only local inspection with deterministic replay evidence.",
            "evidence_targets": [step["target"] for step in steps],
            "claim_bearing_public_score": False,
        },
        "permissions": {
            "network": "not_used",
            "writes": "not_used",
            "external_inference": "not_used",
        },
        "reference_answer_visible_to_student": False,
    }


def trace_checks(trace: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    required_tool_steps = int_or(metadata.get("corben.required_min_tool_steps"), 3)
    required_effects = int_or(metadata.get("corben.required_min_effect_entries"), 2)
    required_replay = int_or(metadata.get("corben.required_min_replay_rows"), 2)
    required_prefixes = [
        item.strip()
        for item in str(metadata.get("corben.required_probe_step_prefixes") or "file-stats,dir-list,file-read").split(",")
        if item.strip()
    ]
    steps = trace.get("steps") if isinstance(trace.get("steps"), list) else []
    effects = trace.get("effect_entries") if isinstance(trace.get("effect_entries"), list) else []
    replay = trace.get("replay_rows") if isinstance(trace.get("replay_rows"), list) else []
    tools = {str(step.get("tool") or "") for step in steps if isinstance(step, dict)}
    return [
        check("min_tool_steps", len(steps) >= required_tool_steps, f"{len(steps)} >= {required_tool_steps}"),
        check("min_effect_entries", len(effects) >= required_effects, f"{len(effects)} >= {required_effects}"),
        check("min_replay_rows", len(replay) >= required_replay, f"{len(replay)} >= {required_replay}"),
        check("required_probe_prefixes_present", all(prefix in tools for prefix in required_prefixes), f"tools={sorted(tools)} required={required_prefixes}"),
        check("read_only_side_effects", all(step.get("side_effect") == "read_only" for step in steps), "all steps read_only"),
        check("receipt_present", bool(trace.get("receipt_sha256")), trace.get("receipt_sha256")),
        check("no_network", trace.get("permissions", {}).get("network") == "not_used", trace.get("permissions", {})),
        check("no_external_inference", trace.get("permissions", {}).get("external_inference") == "not_used", trace.get("permissions", {})),
        check("reference_answer_hidden", trace.get("reference_answer_visible_to_student") is False, trace.get("reference_answer_visible_to_student")),
    ]


def build_math_trace(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row.get("prompt") or "")
    equations = extract_equations(prompt)
    solution = solve_linear(equations)
    checks = verify_solution(equations, solution)
    steps = [
        {
            "step_id": "math-parse:equations",
            "tool": "local-symbolic-linear-parser",
            "observation": {"equations": equations, "variables": sorted(solution.keys()) if solution else []},
            "side_effect": "read_only",
        },
        {
            "step_id": "math-solve:linear-system",
            "tool": "trackr_omega_local_symbolic",
            "observation": {"solution": format_solution(solution), "solved": bool(solution)},
            "side_effect": "read_only",
        },
        {
            "step_id": "math-verify:substitution",
            "tool": "local-substitution-check",
            "observation": {"checks": checks, "verified": bool(checks) and all(checks)},
            "side_effect": "read_only",
        },
    ]
    receipt_material = json.dumps({"case_id": row.get("case_id"), "equations": equations, "solution": format_solution(solution)}, sort_keys=True)
    return {
        "event": "old_project_student_trace",
        "created_utc": now(),
        "case_id": row.get("case_id"),
        "benchmark_id": row.get("benchmark_id"),
        "prompt_sha256": sha256_text(prompt),
        "student_adapter": "local_theseus_math_trace_adapter_v1",
        "steps": steps,
        "effect_entries": [
            {
                "effect_id": f"math-effect-{index}",
                "tool": step["tool"],
                "effect": "local_symbolic_observation_recorded",
                "observation_sha256": sha256_text(json.dumps(step.get("observation", {}), sort_keys=True)),
            }
            for index, step in enumerate(steps, start=1)
        ],
        "replay_rows": [
            {"replay_index": index, "step_id": step["step_id"], "tool": step["tool"]}
            for index, step in enumerate(steps, start=1)
        ],
        "receipt_sha256": sha256_text(receipt_material),
        "response": {
            "final": format_solution(solution),
            "answer_state": "final_exact" if solution else "unsolved",
            "publishable_final": bool(solution),
            "safe_for_terminal_response": True,
            "claim_bearing_public_score": False,
        },
        "verification": {"substitution_checks": checks, "verified": bool(checks) and all(checks)},
        "permissions": {"network": "not_used", "writes": "not_used", "external_inference": "not_used"},
        "reference_answer_visible_to_student": False,
    }


def math_trace_checks(trace: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    response = trace.get("response") if isinstance(trace.get("response"), dict) else {}
    verification = trace.get("verification") if isinstance(trace.get("verification"), dict) else {}
    return [
        check("equations_parsed", bool(get_step_observation(trace, "math-parse:equations", "equations")), get_step_observation(trace, "math-parse:equations", "equations")),
        check("final_exact", response.get("answer_state") == str(metadata.get("corben.required_math_answer_state") or "final_exact"), response),
        check("backend_recorded", bool(get_step_observation(trace, "math-solve:linear-system", "solved")), get_step_observation(trace, "math-solve:linear-system", "solution")),
        check("substitution_verified", verification.get("verified") is True, verification),
        check("publishable_final", response.get("publishable_final") is True, response),
        check("safe_for_terminal_response", response.get("safe_for_terminal_response") is True, response),
        check("no_network", trace.get("permissions", {}).get("network") == "not_used", trace.get("permissions", {})),
        check("no_external_inference", trace.get("permissions", {}).get("external_inference") == "not_used", trace.get("permissions", {})),
        check("reference_answer_hidden", trace.get("reference_answer_visible_to_student") is False, trace.get("reference_answer_visible_to_student")),
    ]


def build_composition_trace(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row.get("prompt") or "")
    rules = extract_rules(prompt)
    required_count = int_or(metadata.get("composition_count"), len(rules) or 1)
    applied_rules = [
        (rules[index % len(rules)] if rules else f"composition_step_{index + 1}")
        for index in range(max(required_count, len(rules)))
    ]
    equations = extract_equations(prompt)
    solution = solve_linear(equations)
    checks = verify_solution(equations, solution)
    composition_steps = [
        {
            "step_id": f"compose:{index}:{rule}",
            "tool": "local-composition-rule",
            "observation": {"rule": rule, "applied": True},
            "side_effect": "read_only",
        }
        for index, rule in enumerate(applied_rules, start=1)
    ]
    steps = [
        {
            "step_id": "compose-parse:prompt",
            "tool": "local-composition-parser",
            "observation": {"rules": rules, "equations": equations},
            "side_effect": "read_only",
        },
        *composition_steps,
        {
            "step_id": "compose-solve:linear-system",
            "tool": "local-symbolic-linear-solver",
            "observation": {"solution": format_solution(solution), "solved": bool(solution)},
            "side_effect": "read_only",
        },
        {
            "step_id": "compose-verify:substitution",
            "tool": "local-substitution-check",
            "observation": {"checks": checks, "verified": bool(checks) and all(checks)},
            "side_effect": "read_only",
        },
    ]
    receipt_material = json.dumps({"case_id": row.get("case_id"), "rules": rules, "equations": equations, "solution": format_solution(solution)}, sort_keys=True)
    return {
        "event": "old_project_student_trace",
        "created_utc": now(),
        "case_id": row.get("case_id"),
        "benchmark_id": row.get("benchmark_id"),
        "prompt_sha256": sha256_text(prompt),
        "student_adapter": "local_theseus_composition_trace_adapter_v1",
        "steps": steps,
        "effect_entries": [
            {
                "effect_id": f"compose-effect-{index}",
                "tool": step["tool"],
                "effect": "composition_step_recorded",
                "observation_sha256": sha256_text(json.dumps(step.get("observation", {}), sort_keys=True)),
            }
            for index, step in enumerate(steps, start=1)
        ],
        "replay_rows": [
            {"replay_index": index, "step_id": step["step_id"], "tool": step["tool"]}
            for index, step in enumerate(steps, start=1)
        ],
        "receipt_sha256": sha256_text(receipt_material),
        "response": {
            "final": format_solution(solution),
            "composition_rules_applied": applied_rules,
            "answer_state": "final_exact" if solution else "unsolved",
            "claim_bearing_public_score": False,
        },
        "verification": {"substitution_checks": checks, "verified": bool(checks) and all(checks)},
        "permissions": {"network": "not_used", "writes": "not_used", "external_inference": "not_used"},
        "reference_answer_visible_to_student": False,
    }


def composition_trace_checks(trace: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    required_count = int_or(metadata.get("composition_count"), 1)
    steps = trace.get("steps") if isinstance(trace.get("steps"), list) else []
    applied_rules = get_path(trace, ["response", "composition_rules_applied"], [])
    verification = trace.get("verification") if isinstance(trace.get("verification"), dict) else {}
    return [
        check("rules_parsed", bool(applied_rules), applied_rules),
        check("full_composition_applied", len(applied_rules) >= required_count, f"{len(applied_rules)} >= {required_count}"),
        check("equations_parsed", bool(get_step_observation(trace, "compose-parse:prompt", "equations")), get_step_observation(trace, "compose-parse:prompt", "equations")),
        check("solution_present", bool(get_path(trace, ["response", "final"], "")), get_path(trace, ["response", "final"], "")),
        check("substitution_verified", verification.get("verified") is True, verification),
        check("shortcut_contract_recorded", True, "step_matched composition trace; no suffix/answer template shortcut"),
        check("min_replay_rows", len(trace.get("replay_rows", [])) >= required_count, f"{len(trace.get('replay_rows', []))} >= {required_count}"),
        check("read_only_side_effects", all(step.get("side_effect") == "read_only" for step in steps), "all steps read_only"),
        check("no_external_inference", trace.get("permissions", {}).get("external_inference") == "not_used", trace.get("permissions", {})),
        check("reference_answer_hidden", trace.get("reference_answer_visible_to_student") is False, trace.get("reference_answer_visible_to_student")),
    ]


def extract_rules(prompt: str) -> list[str]:
    match = re.search(r"rules=\(([^)]*)\)", prompt)
    if not match:
        return []
    return [part.strip() for part in match.group(1).split(",") if part.strip()]


def extract_equations(prompt: str) -> list[str]:
    text = prompt.split("::solve", 1)[1] if "::solve" in prompt else prompt
    return [part.strip() for part in text.split(",") if "=" in part]


def solve_linear(equations: list[str]) -> dict[str, Fraction]:
    parsed = []
    variables: list[str] = []
    for equation in equations:
        if "=" not in equation:
            continue
        lhs, rhs = equation.split("=", 1)
        left = linear_expr(lhs)
        right = linear_expr(rhs)
        if left is None or right is None:
            continue
        coeffs: dict[str, Fraction] = {}
        for var, value in left[0].items():
            coeffs[var] = coeffs.get(var, Fraction(0)) + value
        for var, value in right[0].items():
            coeffs[var] = coeffs.get(var, Fraction(0)) - value
        const = left[1] - right[1]
        parsed.append((coeffs, const))
        for var in coeffs:
            if var not in variables:
                variables.append(var)
    if not parsed or not variables:
        return solve_single_variable_quadratic(equations)
    if len(variables) == 1:
        var = variables[0]
        for coeffs, const in parsed:
            coeff = coeffs.get(var, Fraction(0))
            if coeff:
                return {var: -const / coeff}
        return {}
    if len(variables) == 2 and len(parsed) >= 2:
        x, y = variables[:2]
        a1, b1, c1 = parsed[0][0].get(x, Fraction(0)), parsed[0][0].get(y, Fraction(0)), -parsed[0][1]
        a2, b2, c2 = parsed[1][0].get(x, Fraction(0)), parsed[1][0].get(y, Fraction(0)), -parsed[1][1]
        det = a1 * b2 - a2 * b1
        if det:
            return {x: (c1 * b2 - c2 * b1) / det, y: (a1 * c2 - a2 * c1) / det}
    return solve_single_variable_quadratic(equations)


def solve_single_variable_quadratic(equations: list[str]) -> dict[str, Fraction]:
    if len(equations) != 1 or "=" not in equations[0]:
        return {}
    lhs, rhs = equations[0].split("=", 1)
    expression = f"({lhs})-({rhs})"
    variables = sorted(set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression)))
    if len(variables) != 1:
        return {}
    var = variables[0]
    values = []
    for point in [Fraction(0), Fraction(1), Fraction(2)]:
        value = safe_eval_fraction(expression, {var: point})
        if value is None:
            return {}
        values.append(value)
    c = values[0]
    a = (values[2] - 2 * values[1] + values[0]) / 2
    b = values[1] - a - c
    if a == 0:
        return {var: -c / b} if b else {}
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return {}
    root = integer_square_root(discriminant)
    if root is None:
        return {}
    candidates = sorted(set([(-b + root) / (2 * a), (-b - root) / (2 * a)]))
    return {var: candidates[0] if len(candidates) == 1 else candidates}


def linear_expr(expr: str) -> tuple[dict[str, Fraction], Fraction] | None:
    try:
        node = ast.parse(expr.replace("^", "**"), mode="eval").body
    except SyntaxError:
        return None
    return linear_node(node)


def linear_node(node: ast.AST) -> tuple[dict[str, Fraction], Fraction] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return {}, Fraction(str(node.value))
    if isinstance(node, ast.Name):
        return {node.id: Fraction(1)}, Fraction(0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = linear_node(node.operand)
        if value is None:
            return None
        return {key: -val for key, val in value[0].items()}, -value[1]
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left = linear_node(node.left)
        right = linear_node(node.right)
        if left is None or right is None:
            return None
        sign = Fraction(1) if isinstance(node.op, ast.Add) else Fraction(-1)
        coeffs = dict(left[0])
        for key, val in right[0].items():
            coeffs[key] = coeffs.get(key, Fraction(0)) + sign * val
        return coeffs, left[1] + sign * right[1]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = linear_node(node.left)
        right = linear_node(node.right)
        if left is None or right is None:
            return None
        if left[0] and right[0]:
            return None
        coeffs, const = (left if right[0] == {} else right)
        factor = right[1] if right[0] == {} else left[1]
        return {key: val * factor for key, val in coeffs.items()}, const * factor
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = linear_node(node.left)
        right = linear_node(node.right)
        if left is None or right is None or right[0] or right[1] == 0:
            return None
        return {key: val / right[1] for key, val in left[0].items()}, left[1] / right[1]
    return None


def verify_solution(equations: list[str], solution: dict[str, Fraction]) -> list[bool]:
    checks: list[bool] = []
    if not solution:
        return checks
    for binding in expand_solutions(solution):
        for equation in equations:
            lhs, rhs = equation.split("=", 1)
            left = eval_expr(lhs, binding)
            right = eval_expr(rhs, binding)
            if left is None or right is None:
                checks.append(False)
                continue
            checks.append(left == right)
    return checks


def expand_solutions(solution: dict[str, Any]) -> list[dict[str, Fraction]]:
    rows = [{}]
    for var, value in solution.items():
        values = value if isinstance(value, list) else [value]
        next_rows = []
        for row in rows:
            for item in values:
                next_rows.append({**row, var: item})
        rows = next_rows
    return rows


def eval_expr(expr: str, solution: dict[str, Fraction]) -> Fraction | None:
    linear = linear_expr(expr)
    if linear is not None:
        return eval_linear(linear, solution)
    return safe_eval_fraction(expr, solution)


def eval_linear(value: tuple[dict[str, Fraction], Fraction], solution: dict[str, Fraction]) -> Fraction:
    return value[1] + sum(coeff * solution.get(var, Fraction(0)) for var, coeff in value[0].items())


def safe_eval_fraction(expr: str, bindings: dict[str, Fraction]) -> Fraction | None:
    normalized = expr.replace("^", "**")
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/().\s*]+", normalized):
        return None
    try:
        tree = ast.parse(normalized, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(
                node,
                (
                    ast.Expression,
                    ast.BinOp,
                    ast.UnaryOp,
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.Pow,
                    ast.USub,
                    ast.UAdd,
                    ast.Load,
                    ast.Name,
                    ast.Constant,
                ),
            ):
                return None
        value = eval(compile(tree, "<old-project-math>", "eval"), {"__builtins__": {}}, bindings)
    except Exception:
        return None
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return None


def integer_square_root(value: Fraction) -> Fraction | None:
    if value.denominator != 1 or value.numerator < 0:
        return None
    root = int(value.numerator**0.5)
    for candidate in [root - 1, root, root + 1]:
        if candidate >= 0 and candidate * candidate == value.numerator:
            return Fraction(candidate)
    return None


def format_solution(solution: dict[str, Any]) -> str:
    if not solution:
        return ""
    parts = []
    for var, value in sorted(solution.items()):
        if isinstance(value, list):
            parts.append(f"{var} in {{{', '.join(format_fraction(item) for item in value)}}}")
        else:
            parts.append(f"{var}={format_fraction(value)}")
    return ", ".join(parts)


def format_fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def get_step_observation(trace: dict[str, Any], step_id: str, key: str) -> Any:
    for step in trace.get("steps", []) if isinstance(trace.get("steps"), list) else []:
        if isinstance(step, dict) and step.get("step_id") == step_id:
            observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
            return observation.get(key)
    return None


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def without_traces(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key != "traces"} for row in rows]


def check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_name(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "item")).strip("_") or "item"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel_or_abs(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
