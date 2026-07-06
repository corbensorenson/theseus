"""Audit student candidate lookup coverage without executing benchmark tests.

This catches verifier/manifest contract bugs where generated candidates exist
but the scoring lane treats them as missing. It intentionally reports only
aggregate counts and task identifiers, never benchmark prompts, tests, reference
solutions, or candidate code.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import real_code_benchmark_runtime as runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-manifest",
        default="reports/student_code_candidates_public_transfer_lift_v2_seed41_5x64.jsonl",
    )
    parser.add_argument("--trace", default="")
    parser.add_argument("--out", default="reports/student_candidate_lookup_contract_audit.json")
    parser.add_argument("--markdown-out", default="reports/student_candidate_lookup_contract_audit.md")
    args = parser.parse_args()

    started = time.perf_counter()
    candidate_path = resolve(args.candidate_manifest)
    trace_path = resolve(args.trace) if args.trace else None
    raw_rows = read_jsonl(candidate_path)
    manifest = runtime.load_student_candidates(candidate_path)

    raw_task_counts: collections.Counter[str] = collections.Counter()
    raw_mode_counts: collections.Counter[str] = collections.Counter()
    raw_benchmark_eligible_by_mode: collections.Counter[str] = collections.Counter()
    normalized_eligible_by_mode: collections.Counter[str] = collections.Counter()
    exclusion_reasons: collections.Counter[str] = collections.Counter()
    task_eligible_counts: collections.Counter[str] = collections.Counter()
    task_raw_benchmark_eligible_counts: collections.Counter[str] = collections.Counter()
    task_modes: dict[str, set[str]] = collections.defaultdict(set)

    for raw in raw_rows:
        task_id = str(raw.get("task_id") or "")
        mode = str(raw.get("candidate_generation_mode") or "")
        if task_id:
            raw_task_counts[task_id] += 1
            task_modes[task_id].add(mode)
        raw_mode_counts[mode] += 1
        if truthy(raw.get("benchmark_promotion_eligible")):
            raw_benchmark_eligible_by_mode[mode] += 1
            if task_id:
                task_raw_benchmark_eligible_counts[task_id] += 1
        row = runtime.normalize_student_candidate(raw)
        if not row:
            exclusion_reasons["normalize_empty"] += 1
            continue
        eligible = runtime.benchmark_candidate_eligible(row)
        if eligible:
            normalized_eligible_by_mode[str(row.get("candidate_generation_mode") or "")] += 1
            if str(row.get("task_id") or ""):
                task_eligible_counts[str(row.get("task_id") or "")] += 1
            continue
        for reason in normalized_exclusion_reasons(row):
            exclusion_reasons[reason] += 1

    all_task_ids = sorted(raw_task_counts)
    tasks_without_raw_candidates = [task_id for task_id in all_task_ids if raw_task_counts[task_id] == 0]
    tasks_without_raw_benchmark_eligible = [
        task_id for task_id in all_task_ids if task_raw_benchmark_eligible_counts[task_id] == 0
    ]
    tasks_without_normalized_eligible = [
        task_id for task_id in all_task_ids if task_eligible_counts[task_id] == 0
    ]
    historical = trace_mismatch_summary(trace_path, task_eligible_counts) if trace_path else {}

    gates = [
        gate("candidate_manifest_exists", candidate_path.exists(), rel(candidate_path)),
        gate("candidate_manifest_loaded", len(raw_rows) > 0, {"row_count": len(raw_rows)}),
        gate(
            "provenance_valid",
            bool(manifest.get("provenance_valid")),
            {
                "valid_candidate_count": manifest.get("valid_candidate_count"),
                "invalid_candidate_count": manifest.get("invalid_candidate_count"),
            },
        ),
        gate("raw_task_candidate_coverage_complete", not tasks_without_raw_candidates, len(tasks_without_raw_candidates)),
        gate(
            "raw_benchmark_eligible_task_coverage_complete",
            not tasks_without_raw_benchmark_eligible,
            len(tasks_without_raw_benchmark_eligible),
        ),
        gate(
            "normalized_eligible_task_coverage_complete",
            not tasks_without_normalized_eligible,
            len(tasks_without_normalized_eligible),
        ),
        gate("no_candidate_code_or_prompts_emitted", True, "aggregate IDs/counts only"),
        gate("public_training_rows_written_zero", True, 0),
        gate("external_inference_zero", True, 0),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "YELLOW"
    payload = {
        "policy": "project_theseus_student_candidate_lookup_contract_audit_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {
            "candidate_manifest": rel(candidate_path),
            "trace": rel(trace_path) if trace_path else "",
        },
        "summary": {
            "raw_candidate_count": len(raw_rows),
            "raw_task_count": len(all_task_ids),
            "valid_candidate_count": manifest.get("valid_candidate_count"),
            "benchmark_promotion_eligible_candidate_count": manifest.get("benchmark_promotion_eligible_candidate_count"),
            "tasks_without_raw_candidates": len(tasks_without_raw_candidates),
            "tasks_without_raw_benchmark_eligible_candidates": len(tasks_without_raw_benchmark_eligible),
            "tasks_without_normalized_eligible_candidates": len(tasks_without_normalized_eligible),
            "normalized_eligible_task_coverage_rate": ratio(
                len(all_task_ids) - len(tasks_without_normalized_eligible),
                len(all_task_ids),
            ),
            "raw_mode_counts": dict(sorted(raw_mode_counts.items())),
            "raw_benchmark_eligible_by_mode": dict(sorted(raw_benchmark_eligible_by_mode.items())),
            "normalized_eligible_by_mode": dict(sorted(normalized_eligible_by_mode.items())),
            "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
            "historical_trace_no_candidate_rows": historical.get("no_candidate_rows", 0),
            "historical_trace_no_candidate_tasks": historical.get("no_candidate_tasks", 0),
            "historical_no_candidate_tasks_now_have_normalized_eligible_candidates": historical.get(
                "no_candidate_tasks_now_have_normalized_eligible_candidates",
                0,
            ),
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
        },
        "samples": {
            "tasks_without_normalized_eligible_candidates": tasks_without_normalized_eligible[:32],
            "task_modes_without_normalized_eligible_candidates": {
                task_id: sorted(task_modes.get(task_id, set()))
                for task_id in tasks_without_normalized_eligible[:12]
            },
        },
        "gates": gates,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }
    write_json(resolve(args.out), payload)
    write_markdown(resolve(args.markdown_out), payload)
    print(json.dumps(payload, indent=2))
    return 0 if trigger_state == "GREEN" else 1


def normalized_exclusion_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not truthy(row.get("benchmark_promotion_eligible")):
        reasons.append("not_benchmark_promotion_eligible_after_normalization")
    if not truthy(row.get("token_level_code_generation_learned")):
        reasons.append("not_token_level_code_generation_learned")
    if not truthy(row.get("full_body_token_candidate")):
        reasons.append("not_full_body_token_candidate")
    if not truthy(row.get("grammar_masked_learned_token_candidate")):
        reasons.append("not_grammar_masked_learned_token_candidate")
    if str(row.get("candidate_program_scope") or "") != "full_function_body":
        reasons.append("not_full_function_body_scope")
    if not truthy(row.get("deterministic_guardrail_passed")):
        reasons.append("deterministic_guardrail_failed")
    if row.get("decoder_contract_verifier_v1_passed") is False:
        reasons.append("decoder_contract_failed")
    if truthy(row.get("placeholder_scaffold_body")):
        reasons.append("placeholder_scaffold_body")
    if runtime.bogus_return_attribute_body(str(row.get("code") or "")):
        reasons.append("bogus_return_attribute_body")
    if runtime.bogus_return_local_callable_body(str(row.get("code") or "")):
        reasons.append("bogus_return_local_callable_body")
    if truthy(row.get("expression_memory_fallback")):
        reasons.append("expression_memory_fallback")
    if truthy(row.get("loop_closure_generated")):
        reasons.append("loop_closure_generated")
    if runtime.is_template_like_candidate(row):
        reasons.append("template_like_candidate")
    return reasons or ["unknown_normalized_exclusion"]


def trace_mismatch_summary(trace_path: Path | None, task_eligible_counts: collections.Counter[str]) -> dict[str, Any]:
    if trace_path is None or not trace_path.exists():
        return {}
    no_candidate_tasks: set[str] = set()
    no_candidate_rows = 0
    for row in read_jsonl(trace_path):
        if str(row.get("verification_stage") or "") != "no_candidate":
            continue
        no_candidate_rows += 1
        task_id = str(row.get("task_id") or "")
        if task_id:
            no_candidate_tasks.add(task_id)
    now_covered = sum(1 for task_id in no_candidate_tasks if task_eligible_counts.get(task_id, 0) > 0)
    return {
        "no_candidate_rows": no_candidate_rows,
        "no_candidate_tasks": len(no_candidate_tasks),
        "no_candidate_tasks_now_have_normalized_eligible_candidates": now_covered,
    }


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def ratio(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}
    return bool(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path | None) -> str:
    if path is None:
        return ""
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    lines = [
        "# Student Candidate Lookup Contract Audit",
        "",
        f"- State: `{payload.get('trigger_state')}`",
        f"- Raw candidates: `{summary.get('raw_candidate_count')}`",
        f"- Tasks: `{summary.get('raw_task_count')}`",
        f"- Benchmark-eligible candidates: `{summary.get('benchmark_promotion_eligible_candidate_count')}`",
        f"- Tasks without normalized eligible candidates: `{summary.get('tasks_without_normalized_eligible_candidates')}`",
        f"- Normalized eligible coverage: `{summary.get('normalized_eligible_task_coverage_rate')}`",
        f"- Historical no-candidate tasks now covered: `{summary.get('historical_no_candidate_tasks_now_have_normalized_eligible_candidates')}`",
        "",
        "No prompts, tests, reference solutions, candidate code, or public training rows are emitted.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
