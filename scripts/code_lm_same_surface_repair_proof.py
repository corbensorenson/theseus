#!/usr/bin/env python3
"""Same-surface candidate repair proof for Code LM receiver manifests.

This proof compares a pre-repair public-candidate manifest with its repaired
successor. It does not execute public tests, read public solutions, or claim a
benchmark score. Its only job is to prove that a manifest-level repair preserved
the receiver surface while shrinking no-admissible/interface failures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_BASELINE = (
    REPORTS
    / "student_code_candidates_private_pressure_private_recovery_train_once_fanout_v1.pre_no_admissible_repair_backup.jsonl"
)
DEFAULT_CURRENT = REPORTS / "student_code_candidates_private_pressure_private_recovery_train_once_fanout_v1.jsonl"
DEFAULT_MERGE_REPORT = REPORTS / "code_lm_manifest_no_admissible_repair_merge_train_once_v1.json"
DEFAULT_OUT = REPORTS / "code_lm_same_surface_repair_proof_train_once_v1.json"
DEFAULT_MARKDOWN = REPORTS / "code_lm_same_surface_repair_proof_train_once_v1.md"
MIN_ACTUAL_COVERAGE_LIFT = 0.05
MIN_ELIGIBLE_COVERAGE_LIFT = 0.03
MIN_NO_ADMISSIBLE_SHRINK = 0.05

sys.path.insert(0, str(ROOT / "scripts"))
import decoder_v2_private_ablation_gate as decoder_gate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-manifest", default=str(DEFAULT_BASELINE.relative_to(ROOT)))
    parser.add_argument("--current-manifest", default=str(DEFAULT_CURRENT.relative_to(ROOT)))
    parser.add_argument("--merge-report", default=str(DEFAULT_MERGE_REPORT.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    baseline_path = resolve(args.baseline_manifest)
    current_path = resolve(args.current_manifest)
    merge_path = resolve(args.merge_report)
    out_path = resolve(args.out)
    markdown_path = resolve(args.markdown_out)

    baseline_rows = read_jsonl(baseline_path)
    current_rows = read_jsonl(current_path)
    merge_report = read_json(merge_path, {})
    baseline_snapshot = manifest_snapshot(baseline_rows, baseline_path)
    current_snapshot = manifest_snapshot(current_rows, current_path)
    deltas = compare_snapshots(baseline_snapshot, current_snapshot)
    same_seed_private_ablation = private_same_seed_ablation_snapshot(merge_report)
    receiver_eligible_delta = max(
        deltas["eligible_task_coverage_delta"],
        float(same_seed_private_ablation.get("private_receiver_eligible_task_rate_delta") or 0.0),
    )

    baseline_surface = task_surface(baseline_rows)
    current_surface = task_surface(current_rows)
    same_task_surface = baseline_surface == current_surface and bool(current_surface)
    current_clean = current_snapshot["public_tests_visible_count"] == 0 and current_snapshot["canonical_solution_seen_count"] == 0
    baseline_clean = baseline_snapshot["public_tests_visible_count"] == 0 and baseline_snapshot["canonical_solution_seen_count"] == 0
    merge_replaced_count = int(merge_report.get("replaced_count") or 0)
    baseline_no_admissible = int(baseline_snapshot["no_admissible_row_count"])

    gates = [
        gate("baseline_manifest_present", bool(baseline_rows), rel_or_abs(baseline_path)),
        gate("current_manifest_present", bool(current_rows), rel_or_abs(current_path)),
        gate(
            "same_task_surface",
            same_task_surface,
            {
                "baseline_task_count": baseline_snapshot["task_count"],
                "current_task_count": current_snapshot["task_count"],
                "baseline_only": sorted(set(baseline_surface) - set(current_surface))[:20],
                "current_only": sorted(set(current_surface) - set(baseline_surface))[:20],
            },
        ),
        gate(
            "public_boundary_clean",
            baseline_clean and current_clean,
            {
                "baseline_public_tests_visible": baseline_snapshot["public_tests_visible_count"],
                "baseline_canonical_solution_seen": baseline_snapshot["canonical_solution_seen_count"],
                "current_public_tests_visible": current_snapshot["public_tests_visible_count"],
                "current_canonical_solution_seen": current_snapshot["canonical_solution_seen_count"],
            },
        ),
        gate(
            "actual_token_coverage_lift",
            deltas["actual_token_task_coverage_delta"] >= MIN_ACTUAL_COVERAGE_LIFT,
            {
                "delta": deltas["actual_token_task_coverage_delta"],
                "baseline": baseline_snapshot["actual_token_task_coverage"],
                "current": current_snapshot["actual_token_task_coverage"],
                "minimum": MIN_ACTUAL_COVERAGE_LIFT,
            },
        ),
        gate(
            "eligible_coverage_lift",
            receiver_eligible_delta >= MIN_ELIGIBLE_COVERAGE_LIFT,
            {
                "delta": deltas["eligible_task_coverage_delta"],
                "effective_receiver_delta": receiver_eligible_delta,
                "private_same_seed_ablation": same_seed_private_ablation,
                "baseline": baseline_snapshot["eligible_task_coverage"],
                "current": current_snapshot["eligible_task_coverage"],
                "minimum": MIN_ELIGIBLE_COVERAGE_LIFT,
            },
        ),
        gate(
            "no_admissible_shrunk",
            deltas["no_admissible_task_rate_delta"] <= -MIN_NO_ADMISSIBLE_SHRINK
            and current_snapshot["no_admissible_task_rate"] == 0.0,
            {
                "delta": deltas["no_admissible_task_rate_delta"],
                "baseline": baseline_snapshot["no_admissible_task_rate"],
                "current": current_snapshot["no_admissible_task_rate"],
                "maximum": -MIN_NO_ADMISSIBLE_SHRINK,
            },
        ),
        gate(
            "current_verifier_and_guardrail_clean",
            current_snapshot["verifier_failed_count"] == 0 and current_snapshot["guardrail_failed_count"] == 0,
            {
                "verifier_failed_count": current_snapshot["verifier_failed_count"],
                "guardrail_failed_count": current_snapshot["guardrail_failed_count"],
            },
        ),
        gate(
            "merge_replaced_no_admissible_rows",
            (
                merge_replaced_count >= baseline_no_admissible
                and int(merge_report.get("remaining_no_admissible_count") or 0) == 0
                and merge_report.get("public_tests_or_solutions_used") is False
            )
            or (
                same_seed_private_ablation.get("trigger_state") == "GREEN"
                and int(same_seed_private_ablation.get("bridge_shadow_task_delta") or 0) > 0
                and float(same_seed_private_ablation.get("no_admissible_rate_delta") or 0.0) <= -MIN_NO_ADMISSIBLE_SHRINK
                and float(same_seed_private_ablation.get("semantic_test_passed_task_rate_delta") or 0.0) > 0.0
                and not bool(same_seed_private_ablation.get("public_tests_or_solutions_used"))
            ),
            {
                "merge_report": rel_or_abs(merge_path),
                "replaced_count": merge_replaced_count,
                "baseline_no_admissible_row_count": baseline_no_admissible,
                "remaining_no_admissible_count": merge_report.get("remaining_no_admissible_count"),
                "public_tests_or_solutions_used": merge_report.get("public_tests_or_solutions_used"),
                "private_same_seed_ablation": same_seed_private_ablation,
            },
        ),
        gate(
            "program_synthesis_evidence_preserved",
            current_snapshot["program_synthesis_loop_present_rate"] >= 0.90
            and current_snapshot["program_synthesis_promotion_ready_rate"] >= 0.70,
            {
                "loop_present_rate": current_snapshot["program_synthesis_loop_present_rate"],
                "promotion_ready_rate": current_snapshot["program_synthesis_promotion_ready_rate"],
            },
        ),
    ]
    ready = all(row["passed"] for row in gates)
    report = {
        "policy": "project_theseus_code_lm_same_surface_repair_proof_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if ready else "YELLOW",
        "ready_for_transfer_proof_receiver_surface": ready,
        "summary": {
            "ready_for_transfer_proof_receiver_surface": ready,
            "baseline_manifest": rel_or_abs(baseline_path),
            "current_manifest": rel_or_abs(current_path),
            "merge_report": rel_or_abs(merge_path),
            "same_task_surface": same_task_surface,
            "task_count": current_snapshot["task_count"],
            "actual_token_task_coverage_delta": deltas["actual_token_task_coverage_delta"],
            "eligible_task_coverage_delta": deltas["eligible_task_coverage_delta"],
            "receiver_eligible_coverage_delta": receiver_eligible_delta,
            "no_admissible_task_rate_delta": deltas["no_admissible_task_rate_delta"],
            "current_no_admissible_task_rate": current_snapshot["no_admissible_task_rate"],
            "current_verifier_failed_count": current_snapshot["verifier_failed_count"],
            "current_guardrail_failed_count": current_snapshot["guardrail_failed_count"],
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "baseline": baseline_snapshot,
        "current": current_snapshot,
        "deltas": deltas,
        "same_seed_private_ablation": same_seed_private_ablation,
        "gates": gates,
        "rules": {
            "public_boundary": "public prompts/signatures may appear as metadata, but public tests and solutions must not be visible",
            "score_semantics": "same-surface receiver repair evidence only; this is not a public benchmark score",
            "promotion": "may satisfy transfer-proof receiver-surface compatibility only when paired with decoder gate and transfer deltas",
        },
        "external_inference_calls": 0,
    }
    write_json(out_path, report)
    write_text(markdown_path, render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0


def manifest_snapshot(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    generation = decoder_gate.public_candidate_generation_quality(rows)
    provenance = decoder_gate.candidate_provenance_quality(rows)
    no_admissible = decoder_gate.public_no_admissible_diagnostics(rows)
    verifier_failed = sum(1 for row in rows if row.get("decoder_contract_verifier_v1_passed") is False)
    guardrail_failed = sum(1 for row in rows if row.get("deterministic_guardrail_passed") is False)
    return {
        "path": rel_or_abs(path),
        "mtime": path.stat().st_mtime if path.exists() else 0.0,
        "row_count": len(rows),
        "task_count": generation["task_count"],
        "candidate_count": len(rows),
        "actual_token_task_count": generation["actual_token_task_count"],
        "eligible_task_count": generation["eligible_task_count"],
        "no_admissible_task_count": generation["no_admissible_task_count"],
        "actual_token_task_coverage": generation["actual_token_task_coverage"],
        "eligible_task_coverage": generation["eligible_task_coverage"],
        "no_admissible_task_rate": generation["no_admissible_task_rate"],
        "no_admissible_row_count": sum(
            1 for row in rows if "no_admissible" in decoder_gate.candidate_mode(row).lower()
        ),
        "verifier_failed_count": verifier_failed,
        "guardrail_failed_count": guardrail_failed,
        "public_tests_visible_count": sum(1 for row in rows if row.get("public_tests_visible_to_generator") is True),
        "canonical_solution_seen_count": sum(1 for row in rows if row.get("canonical_solution_seen_by_solver") is True),
        "template_like_count": sum(1 for row in rows if row.get("template_like_candidate") is True),
        "program_synthesis_loop_present_rate": provenance["program_synthesis_loop_present_rate"],
        "program_synthesis_promotion_ready_rate": provenance["program_synthesis_promotion_ready_rate"],
        "quality_gate_pass_rate": provenance["quality_gate_pass_rate"],
        "top_modes": generation["top_modes"],
        "top_no_admissible_reasons": no_admissible["top_rejection_reasons"],
    }


def compare_snapshots(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, float]:
    keys = [
        "actual_token_task_coverage",
        "eligible_task_coverage",
        "no_admissible_task_rate",
        "quality_gate_pass_rate",
        "program_synthesis_loop_present_rate",
        "program_synthesis_promotion_ready_rate",
    ]
    return {
        f"{key}_delta": round(float(current.get(key) or 0.0) - float(baseline.get(key) or 0.0), 6)
        for key in keys
    }


def private_same_seed_ablation_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("policy") != "project_theseus_broad_transfer_residual_decoder_ablation_v1":
        return {}
    delta = report.get("delta") if isinstance(report.get("delta"), dict) else {}
    manifest = report.get("manifest") if isinstance(report.get("manifest"), dict) else {}
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "status": report.get("status"),
        "private_receiver_eligible_task_rate_delta": float(
            delta.get("private_receiver_eligible_task_rate_delta") or 0.0
        ),
        "bridge_shadow_task_delta": int(
            delta.get("private_to_public_receiver_bridge_shadow_task_count_delta") or 0
        ),
        "no_admissible_rate_delta": float(delta.get("no_admissible_rate_delta") or 0.0),
        "semantic_test_passed_task_rate_delta": float(delta.get("semantic_test_passed_task_rate_delta") or 0.0),
        "semantic_test_passed_task_count_delta": int(delta.get("semantic_test_passed_task_count_delta") or 0),
        "public_task_count": int(manifest.get("public_task_count") or 0),
        "public_tests_or_solutions_used": bool(
            manifest.get("public_tests_used")
            or manifest.get("public_solutions_used")
            or report.get("public_tests_or_solutions_used")
        ),
        "rule": (
            "same-seed private receiver/bridge ablation may satisfy receiver-surface compatibility "
            "only when it is GREEN, private-only, semantic-positive, and public-boundary clean"
        ),
    }


def task_surface(rows: list[dict[str, Any]]) -> dict[str, tuple[str, str, str]]:
    surface: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or row.get("source_task_id") or "")
        if not task_id:
            continue
        source_task_id = str(row.get("source_task_id") or decoder_gate.nested_visible_task_field(row, "source_task_id") or "")
        entry_point = str(row.get("entry_point") or decoder_gate.nested_visible_task_field(row, "entry_point") or "")
        prompt_sha = str(decoder_gate.get_path(row, ["provenance", "visible_task", "prompt_sha256"], "") or "")
        surface[task_id] = (source_task_id, entry_point, prompt_sha)
    return surface


def gate(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            return payload if isinstance(payload, dict) else default
    except Exception:
        return default
    return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Code LM Same-Surface Repair Proof",
        "",
        f"- Status: **{report['trigger_state']}**",
        f"- Ready for transfer-proof receiver surface: `{report['ready_for_transfer_proof_receiver_surface']}`",
        f"- Same task surface: `{report['summary']['same_task_surface']}`",
        f"- Actual coverage delta: `{report['summary']['actual_token_task_coverage_delta']}`",
        f"- Eligible coverage delta: `{report['summary']['eligible_task_coverage_delta']}`",
        f"- No-admissible delta: `{report['summary']['no_admissible_task_rate_delta']}`",
        "",
        "## Gates",
        "",
    ]
    for row in report["gates"]:
        lines.append(f"- {'PASS' if row['passed'] else 'FAIL'}: `{row['name']}`")
    return "\n".join(lines) + "\n"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
