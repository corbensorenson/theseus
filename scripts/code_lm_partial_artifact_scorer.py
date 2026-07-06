#!/usr/bin/env python3
"""Score timed-out Code LM closure artifacts as diagnostic-only evidence.

This script intentionally does not promote, calibrate, or train from public
benchmark answers. It answers one operational question: when a long closure
times out, did it leave enough clean candidate evidence to guide the next
bounded run?
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "code_lm_partial_artifact_score.json"
DEFAULT_MARKDOWN = REPORTS / "code_lm_partial_artifact_score.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure-report", default="reports/code_lm_closure_private_pressure_private.json")
    parser.add_argument("--rust-report", default="reports/code_lm_closure_rust_private_pressure_private.json")
    parser.add_argument("--heartbeat", default="reports/code_lm_closure_rust_private_pressure_private.heartbeat.json")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    closure_path = resolve(args.closure_report)
    rust_path = resolve(args.rust_report)
    heartbeat_path = resolve(args.heartbeat)
    closure = read_json(closure_path, {})
    rust = read_json(rust_path, {})
    heartbeat = read_json(heartbeat_path, {})

    private_manifest = resolve(
        first_string(
            [
                rust.get("private_candidate_manifest"),
                closure.get("private_candidate_manifest"),
                "reports/code_lm_private_candidates_private_pressure_private.jsonl",
            ]
        )
    )
    public_manifest = resolve(
        first_string(
            [
                rust.get("public_candidate_manifest"),
                closure.get("public_candidate_manifest"),
                "reports/student_code_candidates_private_pressure_private.jsonl",
            ]
        )
    )

    private_score = score_manifest(private_manifest, scope="private")
    public_score = score_manifest(public_manifest, scope="public_calibration_metadata_only")
    rust_summary = object_field(rust, "summary")
    heartbeat_progress = object_field(heartbeat, "progress")
    timed_out = (
        rust.get("run_status") == "timed_out_process_tree_killed"
        or closure.get("run_status") == "failed"
        and "timed_out" in json.dumps(rust, sort_keys=True)
    )
    partial_exists = private_score["row_count"] > 0 or public_score["row_count"] > 0
    gates = [
        gate("closure_report_present", bool(closure), rel_or_abs(closure_path)),
        gate("rust_report_present", bool(rust), rel_or_abs(rust_path)),
        gate("timeout_or_interruption_detected", timed_out, {"closure": closure.get("run_status"), "rust": rust.get("run_status")}),
        gate("partial_candidate_artifacts_present", partial_exists, {"private": private_score["row_count"], "public": public_score["row_count"]}),
        gate("public_artifacts_are_calibration_metadata_only", public_score["public_metadata_only"], public_score["safety"]),
    ]
    report = {
        "policy": "project_theseus_code_lm_partial_artifact_score_v1",
        "created_utc": now(),
        "trigger_state": "YELLOW" if partial_exists else "RED",
        "run_status": "diagnostic_only_partial_artifacts",
        "promotion_allowed": False,
        "public_calibration_allowed": False,
        "training_use_state": "diagnostic_only_timeout_partial_artifacts",
        "summary": {
            "timed_out_or_interrupted": timed_out,
            "partial_checkpoint_exists": bool(rust_summary.get("partial_checkpoint_exists")),
            "private_candidate_rows": private_score["row_count"],
            "public_candidate_rows": public_score["row_count"],
            "public_task_count": public_score["task_count"],
            "public_actual_token_task_coverage": public_score["actual_token_task_coverage"],
            "public_eligible_task_coverage": public_score["eligible_task_coverage"],
            "public_no_eligible_task_rate": public_score["no_eligible_task_rate"],
            "public_bogus_return_count": public_score["bogus_return_count"],
            "private_actual_token_task_coverage": private_score["actual_token_task_coverage"],
            "heartbeat_phase": heartbeat.get("phase") or heartbeat.get("stage"),
            "heartbeat_progress_ratio": heartbeat_progress.get("progress_ratio"),
            "heartbeat_completed_tasks": heartbeat_progress.get("completed_tasks"),
            "heartbeat_total_tasks": heartbeat_progress.get("total_tasks"),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "sources": {
            "closure_report": rel_or_abs(closure_path),
            "rust_report": rel_or_abs(rust_path),
            "heartbeat": rel_or_abs(heartbeat_path),
            "private_candidate_manifest": rel_or_abs(private_manifest),
            "public_candidate_manifest": rel_or_abs(public_manifest),
        },
        "private_candidate_diagnostics": private_score,
        "public_candidate_diagnostics": public_score,
        "rust_timeout_summary": rust_summary,
        "heartbeat": compact_heartbeat(heartbeat),
        "gates": gates,
        "rules": {
            "promotion": "timed-out artifacts never unlock public calibration",
            "public_data": "public candidates are inspected as generated metadata only; tests and solutions are not read",
            "next_use": "use rejection/capability counts to choose a smaller resumable closure budget",
        },
        "next_actions": next_actions(partial_exists, public_score, private_score),
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if partial_exists else 2


def score_manifest(path: Path, *, scope: str) -> dict[str, Any]:
    rows = 0
    decode_errors = 0
    tasks: set[str] = set()
    token_tasks: set[str] = set()
    eligible_tasks: set[str] = set()
    token_rows = 0
    full_body_rows = 0
    eligible_rows = 0
    verifier_pass_rows = 0
    guardrail_pass_rows = 0
    template_rows = 0
    placeholder_rows = 0
    bogus_return_rows = 0
    unsafe_public_rows = 0
    reasons: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    decode_errors += 1
                    continue
                if not isinstance(row, dict):
                    continue
                rows += 1
                task = task_id(row, rows)
                tasks.add(task)
                category = str(row.get("category") or nested(row, ["visible_task", "category"]) or "unknown")
                categories[category] += 1
                mode = str(row.get("candidate_generation_mode") or row.get("generation_mode") or "unknown")
                modes[mode] += 1
                code = str(row.get("code") or "")
                no_admissible_residual = "no_admissible" in mode.lower() or "emitted no admissible candidate" in code
                learned = (
                    not no_admissible_residual
                    and (
                        truthy(row.get("token_level_code_generation_learned"))
                        or "code_lm" in str(row.get("candidate_source") or "")
                    )
                )
                full_body = truthy(row.get("full_body_token_candidate")) or row.get("candidate_program_scope") == "full_function_body"
                eligible = truthy(row.get("benchmark_promotion_eligible"))
                if learned:
                    token_rows += 1
                    token_tasks.add(task)
                if full_body:
                    full_body_rows += 1
                if eligible:
                    eligible_rows += 1
                    eligible_tasks.add(task)
                if truthy(row.get("decoder_contract_verifier_v1_passed")):
                    verifier_pass_rows += 1
                if truthy(row.get("deterministic_guardrail_passed")):
                    guardrail_pass_rows += 1
                if truthy(row.get("template_like_candidate")):
                    template_rows += 1
                if truthy(row.get("placeholder_scaffold_body")):
                    placeholder_rows += 1
                reason = str(row.get("promotion_ineligible_reason") or "")
                if reason:
                    reasons[reason] += 1
                for item in list_field(row, "deterministic_guardrail_reasons") + list_field(row, "decoder_contract_verifier_v1_reasons"):
                    reasons[str(item)] += 1
                if reason in {"bogus_return_local_callable", "bogus_return_attribute"} or bogus_return_callable(code):
                    bogus_return_rows += 1
                if scope.startswith("public") and (
                    truthy(row.get("public_tests_visible_to_generator"))
                    or truthy(row.get("canonical_solution_seen_by_solver"))
                    or truthy(row.get("tests_used"))
                ):
                    unsafe_public_rows += 1
                if len(samples) < 12 and (reason or not eligible or bogus_return_callable(code)):
                    samples.append(
                        {
                            "task_id": task,
                            "entry_point": row.get("entry_point"),
                            "category": category,
                            "mode": mode,
                            "eligible": eligible,
                            "reason": reason or "not_eligible",
                            "code_excerpt": code[:700],
                        }
                    )
    task_count = len(tasks)
    return {
        "path": rel_or_abs(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "row_count": rows,
        "decode_errors": decode_errors,
        "task_count": task_count,
        "actual_token_row_count": token_rows,
        "full_body_row_count": full_body_rows,
        "eligible_row_count": eligible_rows,
        "verifier_pass_row_count": verifier_pass_rows,
        "guardrail_pass_row_count": guardrail_pass_rows,
        "template_like_candidate_count": template_rows,
        "placeholder_scaffold_count": placeholder_rows,
        "bogus_return_count": bogus_return_rows,
        "actual_token_task_count": len(token_tasks),
        "eligible_task_count": len(eligible_tasks),
        "actual_token_task_coverage": ratio(len(token_tasks), task_count),
        "eligible_task_coverage": ratio(len(eligible_tasks), task_count),
        "no_eligible_task_rate": ratio(task_count - len(eligible_tasks), task_count),
        "verifier_pass_rate": ratio(verifier_pass_rows, rows),
        "guardrail_pass_rate": ratio(guardrail_pass_rows, rows),
        "top_rejection_reasons": dict(reasons.most_common(16)),
        "top_candidate_modes": dict(modes.most_common(12)),
        "top_categories": dict(categories.most_common(16)),
        "sample_rejected_or_blocked_bodies": samples,
        "public_metadata_only": unsafe_public_rows == 0 if scope.startswith("public") else True,
        "safety": {
            "scope": scope,
            "unsafe_public_rows": unsafe_public_rows,
            "public_tests_or_solutions_used": unsafe_public_rows > 0,
        },
        "score_semantics": "diagnostic_candidate_inventory_only_not_promotion_evidence",
    }


def bogus_return_callable(code: str) -> bool:
    if not code.strip():
        return False
    allowed = {"len", "sum", "min", "max", "sorted", "list", "tuple", "set", "dict", "str", "int", "float", "bool", "range", "enumerate", "zip", "abs", "round"}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for fn in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
        assigned: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            elif isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.target, ast.Name):
                assigned.add(node.target.id)
            elif isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.optional_vars, ast.Name):
                        assigned.add(item.optional_vars.id)
        for node in ast.walk(fn):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                name = node.value.func.id
                if name in assigned and name not in allowed:
                    return True
    return False


def next_actions(partial_exists: bool, public_score: dict[str, Any], private_score: dict[str, Any]) -> list[str]:
    if not partial_exists:
        return ["No partial candidate artifacts were found; rerun a bounded private closure after resource policy passes."]
    actions = [
        "Preserve these artifacts as diagnostic-only evidence; do not unlock public calibration from a timed-out run.",
        "Resume with a smaller bounded private closure so the decoder gate can consume a completed report.",
    ]
    if public_score["no_eligible_task_rate"] > 0.25:
        actions.append("Prioritize public receiver candidate coverage: exact interface, AST-valid bodies, no bogus returns, and eligible learned-token candidates.")
    if private_score["bogus_return_count"] or public_score["bogus_return_count"]:
        actions.append("Keep the bogus local-call return rejection active and feed those failures into no-admissible/control rows.")
    return actions


def compact_heartbeat(payload: dict[str, Any]) -> dict[str, Any]:
    progress = object_field(payload, "progress")
    return {
        "path": "reports/code_lm_closure_rust_private_pressure_private.heartbeat.json",
        "stage": payload.get("stage"),
        "phase": payload.get("phase"),
        "run_status": payload.get("run_status"),
        "runtime_ms": payload.get("runtime_ms"),
        "trained": payload.get("trained"),
        "progress": {
            "completed_tasks": progress.get("completed_tasks"),
            "total_tasks": progress.get("total_tasks"),
            "progress_ratio": progress.get("progress_ratio"),
            "emitted_rows_so_far": progress.get("emitted_rows_so_far"),
            "current_task": object_field(progress, "current_task"),
            "rejection_counts_for_current_task": object_field(progress, "rejection_counts_for_current_task"),
        },
    }


def task_id(row: dict[str, Any], fallback: int) -> str:
    return first_string(
        [
            row.get("task_id"),
            nested(row, ["visible_task", "task_id"]),
            f"{row.get('entry_point') or 'unknown'}:{row.get('source_task_id') or fallback}",
        ]
    )


def gate(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def ratio(num: int | float, den: int | float) -> float:
    return round(float(num) / float(den), 6) if den else 0.0


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            return payload if isinstance(payload, dict) else default
    except Exception:
        return default
    return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Code LM Partial Artifact Score",
        "",
        f"- Status: **{report['trigger_state']}**",
        f"- Promotion allowed: `{report['promotion_allowed']}`",
        f"- Public calibration allowed: `{report['public_calibration_allowed']}`",
        f"- Private candidate rows: `{summary['private_candidate_rows']}`",
        f"- Public candidate rows: `{summary['public_candidate_rows']}`",
        f"- Public eligible task coverage: `{summary['public_eligible_task_coverage']}`",
        f"- Public no-eligible task rate: `{summary['public_no_eligible_task_rate']}`",
        f"- Heartbeat phase: `{summary['heartbeat_phase']}`",
        "",
        "## Gates",
        "",
    ]
    for row in report["gates"]:
        marker = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- {marker}: `{row['name']}`")
    lines.extend(["", "## Next Actions", ""])
    for action in report["next_actions"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def first_string(values: Iterable[Any]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def nested(payload: Any, path: list[str]) -> Any:
    cur = payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def object_field(payload: Any, key: str) -> dict[str, Any]:
    value = payload.get(key) if isinstance(payload, dict) else None
    return value if isinstance(value, dict) else {}


def list_field(payload: Any, key: str) -> list[Any]:
    value = payload.get(key) if isinstance(payload, dict) else None
    return value if isinstance(value, list) else []


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
