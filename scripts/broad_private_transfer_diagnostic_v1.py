#!/usr/bin/env python3
"""Diagnose Broad Private Generalization Ladder v1 transfer failures.

The report intentionally avoids copying heldout tests or solution bodies. It
summarizes candidate shapes, modes, arity mismatches, and scorer errors so the
next repair target is smaller than "0/1008 failed".
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from theseus_archive_resolver import read_jsonl_follow_pointer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HELDOUT = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "broad_private_generalization_ladder_v1_heldout_code_lm_tasks.jsonl"
DEFAULT_SCORE = ROOT / "reports" / "broad_private_generalization_score_v1.json"
DEFAULT_CANDIDATES = ROOT / "reports" / "code_lm_private_candidates_broad_private_generalization_ladder_v1_heldout.jsonl"
DEFAULT_CONTROL = ROOT / "reports" / "code_lm_private_candidates_broad_private_generalization_ladder_v1_heldout_sts_off.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", default=rel(DEFAULT_HELDOUT))
    parser.add_argument("--score", default=rel(DEFAULT_SCORE))
    parser.add_argument("--candidates", default=rel(DEFAULT_CANDIDATES))
    parser.add_argument("--control-candidates", default=rel(DEFAULT_CONTROL))
    parser.add_argument("--out", default="reports/broad_private_transfer_diagnostic_v1.json")
    parser.add_argument("--markdown-out", default="reports/broad_private_transfer_diagnostic_v1.md")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    heldout = read_jsonl(resolve(args.heldout))
    score = read_json(resolve(args.score), {})
    candidates = read_jsonl(resolve(args.candidates))
    controls = read_jsonl(resolve(args.control_candidates))
    heldout_by_task = {str(row.get("task_id") or ""): row for row in heldout}
    candidates_by_task = group_by_task(candidates)
    controls_by_task = group_by_task(controls)
    score_results = score.get("results") if isinstance(score.get("results"), list) else []

    category_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    arity_mismatch_tasks: set[str] = set()
    no_candidate_tasks: set[str] = set()
    no_admissible_runtime_tasks: set[str] = set()
    category_errors: dict[str, Counter[str]] = defaultdict(Counter)
    category_modes: dict[str, Counter[str]] = defaultdict(Counter)
    category_arity: dict[str, Counter[str]] = defaultdict(Counter)

    for result in score_results:
        task_id = str(result.get("task_id") or "")
        task = heldout_by_task.get(task_id, {})
        category = str(result.get("category") or task.get("category") or "unknown")
        family = str(result.get("family") or task.get("broad_private_family_v1") or "unknown")
        category_counts[category] += 1
        family_counts[family] += 1
        errors = result.get("sample_errors") if isinstance(result.get("sample_errors"), list) else []
        reason = str(errors[0] if errors else result.get("status") or "unknown").splitlines()[0][:120]
        error_counts[reason] += 1
        category_errors[category][reason] += 1
        if str(result.get("status") or "") == "no_candidate":
            no_candidate_tasks.add(task_id)
        if "no admissible candidate" in reason.lower():
            no_admissible_runtime_tasks.add(task_id)
        expected_arity = expected_arg_count(task)
        task_candidates = candidates_by_task.get(task_id, [])
        if not task_candidates:
            no_candidate_tasks.add(task_id)
        for candidate in task_candidates[:8]:
            mode = str(candidate.get("candidate_generation_mode") or "unknown")
            category_modes[category][mode] += 1
            shape = candidate_signature_shape(str(candidate.get("code") or ""))
            category_arity[category][shape["label"]] += 1
            if expected_arity > 0 and not shape["uses_varargs"] and shape["arg_count"] < expected_arity:
                arity_mismatch_tasks.add(task_id)

    scorer_summary = score.get("summary") if isinstance(score.get("summary"), dict) else {}
    diagnosis = {
        "candidate_manifest_present": bool(candidates),
        "control_manifest_present": bool(controls),
        "candidate_tasks": len(candidates_by_task),
        "control_tasks": len(controls_by_task),
        "heldout_tasks": len(heldout),
        "score_pass_rate": scorer_summary.get("pass_rate"),
        "score_no_admissible_rate": scorer_summary.get("no_admissible_task_rate"),
        "sts_delta": scorer_summary.get("sts_delta"),
        "sts_regressions": scorer_summary.get("sts_regressions"),
        "wrong_or_narrow_signature_task_count": len(arity_mismatch_tasks),
        "no_candidate_task_count": len(no_candidate_tasks),
        "runtime_no_admissible_task_count": len(no_admissible_runtime_tasks),
        "primary_blocker": primary_blocker(
            pass_rate=float(scorer_summary.get("pass_rate") or 0.0),
            arity_mismatch_count=len(arity_mismatch_tasks),
            no_candidate_count=len(no_candidate_tasks),
            candidate_task_count=len(candidates_by_task),
            heldout_count=len(heldout),
        ),
    }
    return {
        "policy": "project_theseus_broad_private_transfer_diagnostic_v1",
        "created_utc": now(),
        "trigger_state": "GREEN",
        "inputs": {
            "heldout": rel(resolve(args.heldout)),
            "score": rel(resolve(args.score)),
            "candidates": rel(resolve(args.candidates)),
            "control_candidates": rel(resolve(args.control_candidates)),
            "heldout_solution_bodies_included_in_report": False,
            "heldout_tests_included_in_report": False,
            "public_tests_used": False,
            "public_solutions_used": False,
        },
        "summary": diagnosis,
        "top_error_clusters": [{"reason": key, "count": value} for key, value in error_counts.most_common(20)],
        "families": [
            {"family": key, "task_count": family_counts[key]}
            for key in sorted(family_counts)
        ],
        "categories": category_rows(category_counts, category_errors, category_modes, category_arity),
        "next_actions": next_actions(diagnosis, error_counts),
        "external_inference_calls": 0,
    }


def primary_blocker(
    *,
    pass_rate: float,
    arity_mismatch_count: int,
    no_candidate_count: int,
    candidate_task_count: int,
    heldout_count: int,
) -> str:
    if no_candidate_count:
        return "candidate_manifest_or_task_id_coverage"
    if candidate_task_count < heldout_count:
        return "candidate_task_coverage_gap"
    if pass_rate == 0.0 and arity_mismatch_count > 0:
        return "semantic_body_failure_plus_signature_arity_contract"
    if pass_rate == 0.0:
        return "semantic_body_failure"
    return "partial_transfer_quality"


def next_actions(diagnosis: dict[str, Any], error_counts: Counter[str]) -> list[str]:
    actions = []
    if diagnosis["wrong_or_narrow_signature_task_count"]:
        actions.append("Patch broad private signature/arity fidelity so multi-argument rows render with data/other/extra or correct explicit args.")
    if diagnosis["score_pass_rate"] == 0.0:
        actions.append("Add a reusable broad-private semantic decoder route for generated private contract families, marked diagnostic/non-promotion.")
    if error_counts:
        actions.append(f"Start with the dominant runtime cluster `{error_counts.most_common(1)[0][0]}` and verify per-family rates after patch.")
    return actions or ["Rerun broad private fanout and scorer."]


def category_rows(
    category_counts: Counter[str],
    category_errors: dict[str, Counter[str]],
    category_modes: dict[str, Counter[str]],
    category_arity: dict[str, Counter[str]],
) -> list[dict[str, Any]]:
    rows = []
    for category in sorted(category_counts):
        rows.append(
            {
                "category": category,
                "task_count": category_counts[category],
                "top_errors": [
                    {"reason": key, "count": value}
                    for key, value in category_errors[category].most_common(3)
                ],
                "top_candidate_modes": [
                    {"mode": key, "count": value}
                    for key, value in category_modes[category].most_common(5)
                ],
                "signature_shapes": [
                    {"shape": key, "count": value}
                    for key, value in category_arity[category].most_common(5)
                ],
            }
        )
    return rows


def expected_arg_count(row: dict[str, Any]) -> int:
    contract = row.get("decoder_contract") if isinstance(row.get("decoder_contract"), dict) else {}
    value = contract.get("visible_arg_count_hint")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def candidate_signature_shape(code: str) -> dict[str, Any]:
    try:
        module = ast.parse(code)
    except SyntaxError:
        return {"label": "syntax_error", "arg_count": 0, "uses_varargs": False}
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            arg_count = len(node.args.args)
            uses_varargs = node.args.vararg is not None
            return {
                "label": f"{'varargs' if uses_varargs else 'fixed'}_{arg_count}",
                "arg_count": arg_count,
                "uses_varargs": uses_varargs,
            }
    return {"label": "no_function", "arg_count": 0, "uses_varargs": False}


def group_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if task_id:
            out[task_id].append(row)
    return out


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return [row for row in read_jsonl_follow_pointer(path) if isinstance(row, dict)]
    except (OSError, json.JSONDecodeError):
        return []


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Broad Private Transfer Diagnostic V1",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Primary blocker: `{summary.get('primary_blocker')}`",
        f"- Score pass rate: {summary.get('score_pass_rate')}",
        f"- Candidate tasks: {summary.get('candidate_tasks')}/{summary.get('heldout_tasks')}",
        f"- Wrong/narrow signature tasks: {summary.get('wrong_or_narrow_signature_task_count')}",
        f"- Runtime no-admissible tasks: {summary.get('runtime_no_admissible_task_count')}",
        f"- STS delta: {summary.get('sts_delta')}",
        "",
        "## Top Errors",
    ]
    for row in report.get("top_error_clusters", [])[:10]:
        lines.append(f"- `{row.get('reason')}`: {row.get('count')}")
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
