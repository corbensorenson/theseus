"""Create a tiny governed sample from admitted legacy training sources.

This is the first step after metadata admission. It reads only sources that
``legacy_training_source_audit.py`` admitted for tiny dry-runs, verifies the
source hash, rejects benchmark/holdout-overlap rows, preserves lane metadata,
and writes a bounded JSONL sample for local training experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADMISSIONS = ROOT / "data" / "training_sources" / "legacy_training_admissions.json"
DEFAULT_OUT = ROOT / "reports" / "legacy_training_source_sample.json"
DEFAULT_SAMPLE_OUT = ROOT / "data" / "training_sources" / "legacy_tiny_dry_run_sample.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admissions", default=str(DEFAULT_ADMISSIONS.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT.relative_to(ROOT)))
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--max-rows-per-lane", type=int, default=0)
    parser.add_argument("--include-contract-seeds", action="store_true")
    args = parser.parse_args()

    admissions_path = resolve(args.admissions)
    admissions = read_json(admissions_path)
    report = build_report(
        admissions=admissions,
        admissions_path=admissions_path,
        sample_out=resolve(args.sample_out),
        max_rows=max(1, args.max_rows),
        max_rows_per_lane=args.max_rows_per_lane,
        include_contract_seeds=args.include_contract_seeds,
    )
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(
    *,
    admissions: dict[str, Any],
    admissions_path: Path,
    sample_out: Path,
    max_rows: int,
    max_rows_per_lane: int,
    include_contract_seeds: bool,
) -> dict[str, Any]:
    primary = [row for row in admissions.get("admit_for_tiny_dry_run", []) if isinstance(row, dict)]
    seeds = [row for row in admissions.get("seed_for_contract_tests", []) if isinstance(row, dict)]
    selected_sources = primary + (seeds if include_contract_seeds else [])
    source_reports: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    all_rejections: Counter[str] = Counter()

    per_source_limit = max(1, math.ceil(max_rows / max(1, len(selected_sources))))
    for source in selected_sources:
        source_report, samples, rejections = sample_source(
            source,
            per_source_limit=per_source_limit,
            max_rows_per_lane=max_rows_per_lane,
        )
        source_reports.append(source_report)
        all_samples.extend(samples)
        all_rejections.update(rejections)

    all_samples = deterministic_trim(all_samples, max_rows)
    write_jsonl(sample_out, all_samples)
    lane_counts = Counter(str(row.get("lane") or "unknown") for row in all_samples)
    family_counts = Counter(str(row.get("task_family") or "unknown") for row in all_samples)
    gates = [
        gate("admissions_present", bool(admissions), rel_or_abs(admissions_path)),
        gate("tiny_dry_run_source_present", bool(primary), [row.get("dataset_id") for row in primary]),
        gate("sample_rows_written", len(all_samples) > 0, f"rows={len(all_samples)} path={rel(sample_out)}"),
        gate("bounded_row_count", len(all_samples) <= max_rows, f"rows={len(all_samples)} max={max_rows}"),
        gate("source_hashes_verified", all(row.get("sha256_verified") for row in source_reports), source_reports),
        gate("no_protected_holdout_overlap", not any(row.get("protected_holdout_overlap") for row in all_samples), "all sampled rows false"),
        gate("no_benchmark_ids", not any(row.get("benchmark_ids") for row in all_samples), "benchmark_ids empty"),
        gate("no_claim_bearing_path", not any(row.get("claim_bearing_path") for row in all_samples), "claim_bearing_path false"),
        gate("lane_metadata_present", all(row.get("lane") for row in all_samples), dict(lane_counts)),
        gate("external_inference_zero", True, "local file sampling only"),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "RED"
    if trigger_state == "GREEN" and len(lane_counts) < 2:
        trigger_state = "YELLOW"
    return {
        "policy": "project_theseus_legacy_training_source_sampler_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "admissions": rel_or_abs(admissions_path),
        "sample_path": rel(sample_out),
        "summary": {
            "selected_sources": len(selected_sources),
            "primary_sources": len(primary),
            "contract_seed_sources": len(seeds) if include_contract_seeds else 0,
            "sample_rows": len(all_samples),
            "max_rows": max_rows,
            "lane_counts": dict(lane_counts),
            "task_family_counts": dict(family_counts),
            "rejections": dict(all_rejections),
            "external_inference_calls": 0,
        },
        "source_reports": source_reports,
        "sample_preview": preview_rows(all_samples, 12),
        "gates": gates,
        "usage_policy": {
            "internal_training_only": True,
            "not_public_benchmark_claim_evidence": True,
            "bulk_copy": False,
            "max_rows_first_pass": max_rows,
            "requires_reaudit_if_source_manifest_changes": True,
        },
        "external_inference_calls": 0,
    }


def sample_source(
    source: dict[str, Any],
    *,
    per_source_limit: int,
    max_rows_per_lane: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], Counter[str]]:
    source_path = Path(str(source.get("local_path") or ""))
    expected_sha = str(source.get("sha256") or "")
    actual_sha = sha256_file(source_path) if source_path.exists() else ""
    sha_ok = bool(expected_sha and actual_sha and expected_sha == actual_sha)
    rows_by_lane: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejections: Counter[str] = Counter()
    total_rows = 0
    if source_path.exists() and sha_ok:
        with source_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                total_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    rejections["invalid_json"] += 1
                    continue
                reason = reject_reason(row)
                if reason:
                    rejections[reason] += 1
                    continue
                lane = str(row.get("lane") or row.get("task_family") or "unknown")
                rows_by_lane[lane].append(normalize_row(source, row, line_number))
    lane_limit = max_rows_per_lane or max(1, math.ceil(per_source_limit / max(1, len(rows_by_lane))))
    samples: list[dict[str, Any]] = []
    for lane, rows in sorted(rows_by_lane.items()):
        ranked = sorted(rows, key=lambda row: stable_sort_key(row))
        samples.extend(ranked[:lane_limit])
    samples = deterministic_trim(samples, per_source_limit)
    report = {
        "dataset_id": source.get("dataset_id"),
        "use_state": source.get("use_state"),
        "local_path": str(source_path),
        "source_exists": source_path.exists(),
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "sha256_verified": sha_ok,
        "source_rows_seen": total_rows,
        "eligible_lanes": {lane: len(rows) for lane, rows in sorted(rows_by_lane.items())},
        "sampled_rows": len(samples),
        "rejections": dict(rejections),
    }
    return report, samples, rejections


def reject_reason(row: dict[str, Any]) -> str:
    if str(row.get("split") or "train").lower() != "train":
        return "non_train_split"
    if bool(row.get("protected_holdout_overlap")):
        return "protected_holdout_overlap"
    if bool(row.get("claim_bearing_path")):
        return "claim_bearing_path"
    if row.get("benchmark_ids"):
        return "benchmark_ids_present"
    if not str(row.get("prompt") or "").strip() or not str(row.get("answer") or "").strip():
        return "missing_prompt_or_answer"
    if len(str(row.get("prompt") or "")) > 12000 or len(str(row.get("answer") or "")) > 12000:
        return "row_too_long_for_tiny_dry_run"
    return ""


def normalize_row(source: dict[str, Any], row: dict[str, Any], line_number: int) -> dict[str, Any]:
    prompt = str(row.get("prompt") or "").strip()
    answer = str(row.get("answer") or "").strip()
    sample_id = str(row.get("source_id") or stable_id(f"{source.get('dataset_id')}:{line_number}:{prompt}:{answer}"))
    return {
        "sample_id": sample_id,
        "dataset_id": source.get("dataset_id"),
        "source_line": line_number,
        "lane": row.get("lane") or "unknown",
        "task_family": row.get("task_family") or "unknown",
        "split": row.get("split") or "train",
        "prompt": prompt,
        "answer": answer,
        "sample_weight": row.get("sample_weight", 1.0),
        "benchmark_ids": row.get("benchmark_ids") or [],
        "protected_holdout_overlap": bool(row.get("protected_holdout_overlap")),
        "claim_bearing_path": bool(row.get("claim_bearing_path")),
        "contamination_tags": row.get("contamination_tags") or [],
        "provenance": {
            "legacy_source_id": row.get("source_id"),
            "legacy_source_kind": row.get("source_kind"),
            "legacy_source_path": row.get("source_path"),
            "admission_sha256": source.get("sha256"),
        },
        "governance": {
            "internal_training_only": True,
            "tiny_dry_run": True,
            "not_public_benchmark_claim_evidence": True,
            "bulk_copy": False,
            "external_inference_calls": 0,
        },
    }


def deterministic_trim(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(rows, key=stable_sort_key)[:limit]


def stable_sort_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "dataset_id": row.get("dataset_id"),
                "sample_id": row.get("sample_id"),
                "lane": row.get("lane"),
                "prompt": row.get("prompt"),
            },
            sort_keys=True,
        ).encode("utf-8", errors="replace")
    ).hexdigest()


def preview_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    preview = []
    for row in rows[:limit]:
        preview.append(
            {
                "sample_id": row.get("sample_id"),
                "dataset_id": row.get("dataset_id"),
                "lane": row.get("lane"),
                "task_family": row.get("task_family"),
                "prompt_chars": len(str(row.get("prompt") or "")),
                "answer_chars": len(str(row.get("answer") or "")),
            }
        )
    return preview


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
