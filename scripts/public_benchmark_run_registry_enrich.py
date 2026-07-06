#!/usr/bin/env python3
"""Enrich consumed public benchmark registry rows after score/residual reports exist.

The guarded calibration runner appends the consumption row immediately after the
one-shot run. Score and residual reports may be produced afterward, so this tool
adds those audit fields without rerunning the public surface or embedding public
payload content.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "reports" / "public_benchmark_run_registry.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=rel(DEFAULT_REGISTRY))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--residual-report", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    registry_path = resolve(args.registry)
    calibration_path = resolve(args.calibration_report)
    residual_path = resolve(args.residual_report)
    output_path = resolve(args.out) if args.out else registry_path

    rows = read_jsonl(registry_path)
    calibration = read_json(calibration_path)
    residual = read_json(residual_path)
    summary = object_field(calibration, "summary")
    residual_summary = object_field(residual, "summary")
    dominant_categories = residual_summary.get("dominant_categories") or dominant_residual_categories(residual_summary)
    next_private_fix = residual_summary.get("recommended_private_fix_family") or private_fix_family_from_dominant(
        dominant_categories
    )

    updated = 0
    for row in rows:
        if row.get("run_id") != args.run_id:
            continue
        row.update(
            {
                "score_recorded": bool(summary),
                "score_report_path": rel_or_abs(calibration_path),
                "public_task_count": int(summary.get("public_task_count") or 0),
                "real_public_task_pass_rate": safe_float(summary.get("real_public_task_pass_rate")),
                "multi_stream_passed": multi_stream_passed(calibration, summary),
                "student_candidate_count": int(summary.get("student_candidate_count") or 0),
                "template_like_candidate_count": int(summary.get("template_like_candidate_count") or 0),
                "loop_closure_candidate_count": int(summary.get("loop_closure_candidate_count") or 0),
                "expression_memory_fallback_count": int(summary.get("expression_memory_fallback_count") or 0),
                "residual_categories_recorded": bool(residual_summary),
                "residual_report_path": rel_or_abs(residual_path),
                "dominant_residual_categories": dominant_categories,
                "next_private_fix_family": next_private_fix,
                "next_private_fix_family_after_current_adapter": residual_summary.get(
                    "recommended_private_fix_family_after_current_adapter"
                )
                or next_private_fix,
                "registry_enriched_utc": now(),
            }
        )
        updated += 1

    payload = {
        "policy": "project_theseus_public_benchmark_run_registry_enrich_v1",
        "created_utc": now(),
        "run_id": args.run_id,
        "registry": rel_or_abs(registry_path),
        "out": rel_or_abs(output_path),
        "rows_read": len(rows),
        "rows_updated": updated,
        "score_recorded": bool(summary) and updated > 0,
        "residual_categories_recorded": bool(residual_summary) and updated > 0,
        "external_inference_calls": 0,
    }
    write_jsonl(output_path, rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if updated == 1 and payload["score_recorded"] and payload["residual_categories_recorded"] else 2


def multi_stream_passed(calibration: dict[str, Any], summary: dict[str, Any]) -> int:
    suites = calibration.get("suites")
    if isinstance(suites, list):
        return sum(int(row.get("multi_stream_passed") or 0) for row in suites if isinstance(row, dict))
    return int(round(int(summary.get("public_task_count") or 0) * safe_float(summary.get("multi_stream_pass_rate"))))


def dominant_residual_categories(summary: dict[str, Any]) -> list[list[Any]]:
    dominant = summary.get("dominant_current_failure")
    if isinstance(dominant, list) and dominant:
        return [dominant]
    counts = summary.get("residual_category_counts")
    if not isinstance(counts, dict):
        return []
    rows = []
    for key, value in counts.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            rows.append([str(key), count])
    return sorted(rows, key=lambda row: (-int(row[1]), str(row[0])))[:5]


def private_fix_family_from_dominant(dominant_categories: list[list[Any]]) -> str | None:
    if not dominant_categories:
        return None
    category = str(dominant_categories[0][0] if dominant_categories[0] else "")
    mapping = {
        "algorithm_choice": "public_transfer_algorithm_planning_private_repair",
        "return_type_shape": "public_transfer_return_shape_private_repair",
        "edge_cases": "public_transfer_edge_case_private_repair",
        "external_dependency_missing": "public_transfer_dependency_boundary_private_repair",
        "selector_ranking_miss": "public_transfer_selector_ranking_private_repair",
        "verifier_mismatch": "public_transfer_verifier_contract_private_repair",
        "no_admissible_interface_coverage": "public_transfer_candidate_coverage_private_repair",
    }
    return mapping.get(category, f"public_transfer_{category}_private_repair" if category else None)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def object_field(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    return item if isinstance(item, dict) else {}


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def rel_or_abs(path: Path) -> str:
    return rel(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
