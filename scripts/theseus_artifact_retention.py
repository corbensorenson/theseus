#!/usr/bin/env python3
"""Manifest-backed retention/archive service for bulky Theseus artifacts.

This service moves heavyweight historical checkpoint JSONs out of ``reports/``
latest-view space while leaving pointer metadata at the original paths. It does
not delete artifacts.
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_evidence_store  # noqa: E402
from theseus_archive_resolver import ARCHIVE_POINTER_POLICY  # noqa: E402


REPORTS = ROOT / "reports"
ARCHIVE_ROOT = ROOT / "archive" / "report_artifacts"
DEFAULT_MANIFEST = REPORTS / "theseus_artifact_retention_manifest.json"
DEFAULT_OUT = REPORTS / "theseus_artifact_retention.json"
DEFAULT_MARKDOWN = REPORTS / "theseus_artifact_retention.md"
DEFAULT_BUDGET_POLICY = ROOT / "configs" / "artifact_retention_budget_policy.json"
DEFAULT_BUDGET_OUT = REPORTS / "theseus_artifact_budget_gate.json"
DEFAULT_BUDGET_MARKDOWN = REPORTS / "theseus_artifact_budget_gate.md"
DEFAULT_REGISTRY = ROOT / "configs" / "project_manifest_registry.json"
MIN_BYTES = 256 * 1024 * 1024
ARCHIVEABLE_PREFIXES = (
    "student_code_lm_checkpoint_fanout_speed_",
    "student_code_lm_checkpoint_frontier_private_only_train_once_smoke_",
    "student_code_lm_checkpoint_frontier_private_transfer_private_only_train_once_",
    "student_code_lm_checkpoint_frontier_private_transfer_private_repair_",
    "student_code_lm_checkpoint_open_code_transfer_expansion_",
    "student_code_lm_checkpoint_broad_private_generalization_ladder_v1",
    "student_code_lm_checkpoint_edge_contract_",
    "student_code_lm_checkpoint_private_residual_repair_v3_",
    "student_code_lm_checkpoint_private_broad_floor_transfer_repair_closure_",
    "student_code_lm_checkpoint_private_pressure_private_recovery_",
    "student_code_lm_checkpoint_runtime_compact_sts_",
    "student_code_lm_checkpoint_train_once_fanout_",
    "student_code_lm_checkpoint_wide_public_seed23_5x32_",
)
RETAIN_LIVE_NAMES = {
    "report_evidence_store.sqlite",
    "hive_work_board.sqlite",
    "context_packets.jsonl",
    "hive_artifact_sync_ledger.jsonl",
    "world_adapter_job_control_ledger.jsonl",
}
ARCHIVEABLE_REPORT_DIRS = {
    "post_v4_generalization_autopilot_v1_archive",
    "overnight_self_improvement_v1",
    "private_residual_frontier_v1_shards",
}
ARCHIVEABLE_SUFFIXES = {".json", ".jsonl", ".dmg", ".zip", ".pkg"}
INLINE_POINTER_SUFFIXES = {".json", ".jsonl"}
BINARY_SIDECAR_SUFFIXES = {".dmg", ".zip", ".pkg"}
MAX_POINTER_BYTES = 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-files", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--min-bytes", type=int, default=MIN_BYTES)
    parser.add_argument("--include-jsonl", action="store_true")
    parser.add_argument("--include-archived-report-dirs", action="store_true")
    parser.add_argument("--include-report-snapshots", action="store_true")
    parser.add_argument("--include-runtime-replay-mirrors", action="store_true")
    parser.add_argument("--include-vcm-payloads", action="store_true")
    parser.add_argument("--include-dist-artifacts", action="store_true")
    parser.add_argument("--allow-non-json-pointer", action="store_true")
    parser.add_argument("--allow-binary-sidecar", action="store_true")
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--archive-root", default=str(ARCHIVE_ROOT.relative_to(ROOT)))
    parser.add_argument("--manifest-out", default=str(DEFAULT_MANIFEST.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--budget-gate", action="store_true", help="Audit live generated report/checkpoint budgets and registry ownership.")
    parser.add_argument("--budget-policy", default=str(DEFAULT_BUDGET_POLICY.relative_to(ROOT)))
    parser.add_argument("--budget-out", default=str(DEFAULT_BUDGET_OUT.relative_to(ROOT)))
    parser.add_argument("--budget-markdown-out", default=str(DEFAULT_BUDGET_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    if args.budget_gate:
        payload = build_budget_gate_report(
            resolve(args.budget_policy),
            resolve(args.registry),
            started=started,
        )
        report_evidence_store.write_json_report(
            resolve(args.budget_out),
            payload,
            markdown_path=resolve(args.budget_markdown_out),
            markdown_text=render_budget_markdown(payload),
        )
        print(json.dumps(budget_gate_view(payload), indent=2, sort_keys=True))
        return 0 if payload["trigger_state"] == "GREEN" else 2

    archive_root = resolve(args.archive_root)
    manifest_path = resolve(args.manifest_out)
    existing_manifest = read_json(manifest_path, {})
    previous_entries = existing_manifest.get("entries") if isinstance(existing_manifest.get("entries"), list) else []
    candidates = discover_candidates(
        min_bytes=max(1, int(args.min_bytes)),
        include_jsonl=bool(args.include_jsonl),
        include_archived_report_dirs=bool(args.include_archived_report_dirs),
        include_report_snapshots=bool(args.include_report_snapshots),
        include_runtime_replay_mirrors=bool(args.include_runtime_replay_mirrors),
        include_vcm_payloads=bool(args.include_vcm_payloads),
        include_dist_artifacts=bool(args.include_dist_artifacts),
    )
    if args.max_files > 0:
        candidates = candidates[: int(args.max_files)]

    actions = []
    for candidate in candidates:
        if args.execute:
            actions.append(
                archive_candidate(
                    candidate,
                    archive_root,
                    compress=not args.no_compress,
                    allow_non_json_pointer=bool(args.allow_non_json_pointer),
                    allow_binary_sidecar=bool(args.allow_binary_sidecar),
                )
            )
        else:
            actions.append(
                {
                    **candidate,
                    "status": "dry_run",
                    "archive_path": rel(
                        archive_path_for(
                            candidate["path"],
                            archive_root,
                            compress=candidate_compress_enabled(candidate, global_compress=not args.no_compress),
                        )
                    ),
                }
            )

    entries_by_original = {str(row.get("original_path")): row for row in previous_entries if isinstance(row, dict)}
    for row in actions:
        if row.get("status") in {"archived", "already_archived", "dry_run"}:
            entries_by_original[str(row.get("original_path") or row.get("path"))] = manifest_entry(row)
    entries = sorted(entries_by_original.values(), key=lambda row: str(row.get("original_path") or ""))
    compression_records = [record for record in (retention_compression_record(row, compress=not args.no_compress) for row in actions) if record]
    compressed_artifact_records = [
        report_evidence_store.compressed_artifact_record_from_compression_record(row, source_system="artifact_retention")
        for row in compression_records
    ]
    compression_receipts = [
        report_evidence_store.compression_receipt_from_compression_record(row, source_system="artifact_retention")
        for row in compression_records
    ]
    defeater_records = [record for record in (retention_defeater_record(row) for row in actions) if record]
    manifest = {
        "policy": "project_theseus_artifact_retention_manifest_v1",
        "created_utc": now(),
        "archive_root": rel(archive_root),
        "entry_count": len(entries),
        "entries": entries,
        "compression_record_count": len(compression_records),
        "compressed_artifact_record_count": len(compressed_artifact_records),
        "compression_receipt_count": len(compression_receipts),
        "defeater_record_count": len(defeater_records),
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }
    if args.execute or actions:
        write_json(manifest_path, manifest)

    moved_bytes = sum(int(row.get("bytes") or 0) for row in actions if row.get("status") in {"archived", "already_archived"})
    reclaimed_bytes = sum(int(row.get("reclaimed_bytes") or 0) for row in actions if row.get("status") in {"archived", "already_archived"})
    dry_run_candidate_bytes = sum(int(row.get("bytes") or 0) for row in actions if row.get("status") == "dry_run")
    dry_run_compressible_bytes = sum(
        int(row.get("bytes") or 0)
        for row in actions
        if row.get("status") == "dry_run" and should_compress(resolve(str(row.get("path") or "")), compress=not args.no_compress)
    )
    payload = {
        "policy": "project_theseus_artifact_retention_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not any(row.get("status") == "failed" for row in actions) else "YELLOW",
        "summary": {
            "execute": bool(args.execute),
            "candidate_count": len(candidates),
            "action_count": len(actions),
            "archived_count": sum(1 for row in actions if row.get("status") == "archived"),
            "already_archived_count": sum(1 for row in actions if row.get("status") == "already_archived"),
            "failed_count": sum(1 for row in actions if row.get("status") == "failed"),
            "moved_gib": round(moved_bytes / (1024**3), 3),
            "estimated_reclaimed_gib": round(reclaimed_bytes / (1024**3), 3),
            "dry_run_candidate_gib": round(dry_run_candidate_bytes / (1024**3), 3),
            "dry_run_compressible_candidate_gib": round(dry_run_compressible_bytes / (1024**3), 3),
            "manifest": rel(manifest_path),
            "compression_record_count": len(compression_records),
            "compressed_artifact_record_count": len(compressed_artifact_records),
            "compression_receipt_count": len(compression_receipts),
            "defeater_record_count": len(defeater_records),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "actions": actions,
        "compression_records": compression_records,
        "compressed_artifact_records": compressed_artifact_records,
        "compression_receipts": compression_receipts,
        "defeater_records": defeater_records,
        "retention_rules": {
            "delete": False,
            "compress": not args.no_compress,
            "allow_non_json_pointer": bool(args.allow_non_json_pointer),
            "allow_binary_sidecar": bool(args.allow_binary_sidecar),
            "pointer_policy": ARCHIVE_POINTER_POLICY,
            "archiveable_prefixes": list(ARCHIVEABLE_PREFIXES),
            "retained_live_names": sorted(RETAIN_LIVE_NAMES),
        },
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }
    report_evidence_store.write_json_report(
        resolve(args.out),
        payload,
        markdown_path=resolve(args.markdown_out),
        markdown_text=render_markdown(payload),
    )
    print(json.dumps(retention_gate_view(payload), indent=2, sort_keys=True))
    return 0 if payload["trigger_state"] == "GREEN" else 2


def discover_candidates(
    *,
    min_bytes: int,
    include_jsonl: bool,
    include_archived_report_dirs: bool,
    include_report_snapshots: bool,
    include_runtime_replay_mirrors: bool,
    include_vcm_payloads: bool,
    include_dist_artifacts: bool,
) -> list[dict[str, Any]]:
    rows = []
    for path in REPORTS.iterdir() if REPORTS.exists() else []:
        if not path.is_file():
            continue
        if path.name in RETAIN_LIVE_NAMES:
            continue
        if path.suffix.lower() != ".json":
            continue
        if not path.name.startswith(ARCHIVEABLE_PREFIXES):
            continue
        if path.stat().st_size < min_bytes:
            continue
        rows.append(candidate_for_path(path))
    if include_jsonl:
        for path in REPORTS.glob("code_lm_private_candidates*.jsonl"):
            if archiveable_file(path, min_bytes=min_bytes):
                rows.append(candidate_for_path(path, reason="large_historical_candidate_jsonl"))
    if include_archived_report_dirs:
        for dirname in ARCHIVEABLE_REPORT_DIRS:
            directory = REPORTS / dirname
            if not directory.exists():
                continue
            for path in directory.rglob("*"):
                if archiveable_file(path, min_bytes=min_bytes):
                    rows.append(candidate_for_path(path, reason=f"large_historical_report_archive:{dirname}"))
    if include_report_snapshots:
        for path in (REPORTS / "report_snapshots").rglob("*.json"):
            if archiveable_file(path, min_bytes=min_bytes):
                rows.append(candidate_for_path(path, reason="large_historical_report_snapshot"))
    if include_runtime_replay_mirrors:
        replay_dir = ROOT / "runtime" / "candidate_replay_contract_v1"
        for path in replay_dir.glob("*_resolved.json"):
            if archiveable_file(path, min_bytes=min_bytes):
                rows.append(candidate_for_path(path, reason="large_runtime_candidate_replay_checkpoint_mirror"))
    if include_vcm_payloads:
        for path in (ROOT / "data" / "public_benchmarks" / "vcm_memory_quarantine").glob("*/payloads.jsonl"):
            if archiveable_file(path, min_bytes=min_bytes):
                rows.append(candidate_for_path(path, reason="large_quarantined_public_memory_payload"))
    if include_dist_artifacts:
        for path in (ROOT / "dist" / "macos").glob("ProjectTheseusHive.*"):
            if archiveable_file(path, min_bytes=min_bytes):
                rows.append(candidate_for_path(path, reason="large_stale_macos_installer_artifact"))
    return sorted(rows, key=lambda row: (-int(row["bytes"]), row["path"]))


def archiveable_file(path: Path, *, min_bytes: int) -> bool:
    if not path.is_file():
        return False
    if path.name in RETAIN_LIVE_NAMES:
        return False
    if path.suffix.lower() not in ARCHIVEABLE_SUFFIXES:
        return False
    if path.name.endswith(".gz"):
        return False
    return path.stat().st_size >= min_bytes


def candidate_for_path(path: Path, *, reason: str = "large_historical_code_lm_checkpoint_json") -> dict[str, Any]:
    stat = path.stat()
    return {
        "record_type": "artifact_retention_candidate",
        "path": rel(path),
        "original_path": rel(path),
        "bytes": int(stat.st_size),
        "gib": round(stat.st_size / (1024**3), 3),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "family": artifact_family(path.name),
        "reason": reason,
    }


def archive_candidate(
    candidate: dict[str, Any],
    archive_root: Path,
    *,
    compress: bool,
    allow_non_json_pointer: bool,
    allow_binary_sidecar: bool,
) -> dict[str, Any]:
    source = resolve(candidate["path"])
    compressed = should_compress(source, compress=candidate_compress_enabled(candidate, global_compress=compress))
    target = archive_path_for(rel(source), archive_root, compress=compressed)
    try:
        suffix = source.suffix.lower()
        if suffix in BINARY_SIDECAR_SUFFIXES and not allow_binary_sidecar:
            return {
                **candidate,
                "status": "skipped_requires_binary_sidecar_unlock",
                "archive_path": rel(target),
                "error": "binary_original_paths_require_--allow-binary-sidecar",
            }
        if suffix not in INLINE_POINTER_SUFFIXES | BINARY_SIDECAR_SUFFIXES:
            return {
                **candidate,
                "status": "skipped_unsupported_pointer_suffix",
                "archive_path": rel(target),
                "error": f"unsupported_archive_pointer_suffix:{suffix}",
            }
        if suffix == ".jsonl" and not allow_non_json_pointer:
            return {
                **candidate,
                "status": "skipped_requires_non_json_pointer_unlock",
                "archive_path": rel(target),
                "error": "non_json_original_paths_require_--allow-non-json-pointer",
            }
        pointer_path = pointer_path_for(source)
        if is_pointer(pointer_path):
            pointer = read_pointer(pointer_path)
            existing = resolve(str(pointer.get("archive_path") or ""))
            return {
                **candidate,
                "status": "already_archived",
                "archive_path": rel(existing) if existing.exists() else str(pointer.get("archive_path") or ""),
                "pointer_path": rel(pointer_path),
            }
        source_sha256 = sha256_file(source)
        if target.exists():
            verify = verify_archive_payload(target, source_sha256, compressed=compressed)
            if not verify["verified"]:
                return {**candidate, "status": "failed", "archive_path": rel(target), **verify}
            source.unlink()
            write_pointer(pointer_path, target, {**candidate, "sha256": source_sha256})
            archived_bytes = int(target.stat().st_size)
            return {
                **candidate,
                "status": "already_archived",
                "archive_path": rel(target),
                "pointer_path": rel(pointer_path),
                "sha256": source_sha256,
                "archived_bytes": archived_bytes,
                "reclaimed_bytes": max(0, int(candidate.get("bytes") or 0) - archived_bytes),
            }
        target.parent.mkdir(parents=True, exist_ok=True)
        if compressed:
            gzip_copy(source, target)
            verify = verify_archive_payload(target, source_sha256, compressed=True)
            if not verify["verified"]:
                target.unlink(missing_ok=True)
                return {**candidate, "status": "failed", "archive_path": rel(target), **verify}
            source.unlink()
        else:
            shutil.move(str(source), str(target))
            verify = verify_archive_payload(target, source_sha256, compressed=False)
            if not verify["verified"]:
                return {**candidate, "status": "failed", "archive_path": rel(target), **verify}
        write_pointer(pointer_path, target, {**candidate, "sha256": source_sha256})
        archived_bytes = int(target.stat().st_size)
        return {
            **candidate,
            "status": "archived",
            "archive_path": rel(target),
            "pointer_path": rel(pointer_path),
            "sha256": source_sha256,
            "archived_bytes": archived_bytes,
            "archive_sha256": sha256_file(target),
            "reclaimed_bytes": max(0, int(candidate.get("bytes") or 0) - archived_bytes),
        }
    except Exception as exc:
        return {**candidate, "status": "failed", "archive_path": rel(target), "error": repr(exc)}


def write_pointer(path: Path, target: Path, candidate: dict[str, Any]) -> None:
    payload = {
        "policy": ARCHIVE_POINTER_POLICY,
        "created_utc": now(),
        "trigger_state": "GREEN",
        "original_path": str(candidate.get("original_path") or candidate.get("path") or rel(path)),
        "archive_path": rel(target),
        "original_bytes": int(candidate.get("bytes") or 0),
        "original_sha256": candidate.get("sha256"),
        "original_mtime_utc": candidate.get("mtime_utc"),
        "archive_family": candidate.get("family"),
        "resolver": "scripts/theseus_archive_resolver.py",
        "score_semantics": "pointer only; not model evidence",
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }
    if path.suffix.lower() == ".jsonl":
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    else:
        write_json(path, payload)


def archive_path_for(path: str | Path, archive_root: Path, *, compress: bool = True) -> Path:
    source = resolve(path)
    family = artifact_family(source.name)
    archive_name = source.name + ".gz" if should_compress(source, compress=compress) else source.name
    return archive_root / family / path_scope(source.parent) / archive_name


def pointer_path_for(source: Path) -> Path:
    if source.suffix.lower() in BINARY_SIDECAR_SUFFIXES:
        return source.with_name(source.name + ".archive-pointer.json")
    return source


def should_compress(source: Path, *, compress: bool) -> bool:
    return bool(compress and source.suffix.lower() in {".json", ".jsonl"})


def path_scope(path: Path) -> str:
    rel_parent = rel(path)
    digest = hashlib.sha256(rel_parent.encode("utf-8")).hexdigest()[:12]
    return f"{safe(rel_parent)[:120]}_{digest}"


def gzip_copy(source: Path, target: Path) -> None:
    with source.open("rb") as src, gzip.open(target, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def verify_archive_payload(target: Path, expected_sha256: str, *, compressed: bool) -> dict[str, Any]:
    actual = sha256_gzip_payload(target) if compressed else sha256_file(target)
    return {
        "verified": actual == expected_sha256,
        "payload_sha256": actual,
        "expected_sha256": expected_sha256,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_payload(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_entry(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "original_path": str(row.get("original_path") or row.get("path") or ""),
        "archive_path": str(row.get("archive_path") or ""),
        "pointer_path": str(row.get("pointer_path") or row.get("original_path") or row.get("path") or ""),
        "status": str(row.get("status") or ""),
        "bytes": int(row.get("bytes") or 0),
        "archived_bytes": int(row.get("archived_bytes") or 0),
        "reclaimed_bytes": int(row.get("reclaimed_bytes") or 0),
        "sha256": row.get("sha256"),
        "archive_sha256": row.get("archive_sha256"),
        "gib": row.get("gib"),
        "mtime_utc": row.get("mtime_utc"),
        "family": row.get("family"),
        "reason": row.get("reason"),
        "updated_utc": now(),
    }


def retention_compression_record(row: dict[str, Any], *, compress: bool) -> dict[str, Any]:
    status = str(row.get("status") or "")
    if status not in {"dry_run", "archived", "already_archived"}:
        return {}
    archive_path = str(row.get("archive_path") or "")
    if not archive_path:
        return {}
    original_path = str(row.get("original_path") or row.get("path") or "")
    planned = status == "dry_run"
    archive_compressed = archive_path.endswith(".gz") or should_compress(resolve(original_path), compress=compress)
    return {
        "record_type": "compression_record",
        "record_id": stable_id("artifact_retention_compression", original_path, archive_path, status, row.get("sha256") or ""),
        "family": str(row.get("family") or ""),
        "report_path": "reports/theseus_artifact_retention.json",
        "original_path": original_path,
        "archive_path": archive_path,
        "pointer_path": str(row.get("pointer_path") or original_path),
        "content_hash": str(row.get("sha256") or row.get("archive_sha256") or ""),
        "payload_bytes": int(row.get("bytes") or 0),
        "archived_bytes": int(row.get("archived_bytes") or 0),
        "codec": "gzip" if archive_compressed else "move_or_pointer",
        "compression_scope": "artifact_retention_archive",
        "reconstruction_contract": "archive_path plus pointer_path must restore the original payload hash before the artifact can support downstream claims",
        "status": "PLANNED" if planned else "SUPPORTED",
        "support_state": "PLANNED" if planned else "SUPPORTED",
        "evidence_ref": "reports/theseus_artifact_retention.json",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "non_claim": "Retention compression record proves artifact archive traceability; it is not model capability evidence.",
    }


def retention_defeater_record(row: dict[str, Any]) -> dict[str, Any]:
    status = str(row.get("status") or "")
    if status not in {"archived", "already_archived"}:
        return {}
    original_path = str(row.get("original_path") or row.get("path") or "")
    pointer_path = str(row.get("pointer_path") or original_path)
    archive_path = str(row.get("archive_path") or "")
    return {
        "record_type": "defeater_record",
        "record_id": stable_id("artifact_retention_defeater", original_path, pointer_path, archive_path, row.get("sha256") or ""),
        "family": str(row.get("family") or ""),
        "report_path": "reports/theseus_artifact_retention.json",
        "defeater_type": "latest_path_replaced_by_archive_pointer",
        "defeated_run_id": "",
        "defeating_run_id": stable_id("artifact_retention_action", original_path, archive_path, status),
        "previous_content_hash": str(row.get("sha256") or ""),
        "current_content_hash": str(row.get("archive_sha256") or row.get("sha256") or ""),
        "previous_support_state": "SUPPORTED",
        "current_support_state": "ARCHIVED_POINTER",
        "previous_trigger_state": "",
        "current_trigger_state": "GREEN",
        "original_path": original_path,
        "archive_path": archive_path,
        "pointer_path": pointer_path,
        "support_state": "SUPPORTED",
        "status": "SUPPORTED",
        "evidence_ref": "reports/theseus_artifact_retention.json",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "non_claim": "Retention defeater records that the mutable original path now resolves through an archive pointer; the archived payload remains retained.",
    }


def artifact_family(name: str) -> str:
    stem = Path(name).stem
    for marker in ["_seed", "_v", "_train_once", "_broad_floor", "_private_pressure"]:
        if marker in stem:
            return safe(stem.split(marker, 1)[0] + marker.rstrip("_"))
    return safe(stem)


def candidate_compress_enabled(candidate: dict[str, Any], *, global_compress: bool) -> bool:
    if not global_compress:
        return False
    return not bool(candidate.get("archive_uncompressed"))


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Theseus Artifact Retention",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- execute: `{summary.get('execute')}`",
        f"- candidates: `{summary.get('candidate_count')}`",
        f"- archived: `{summary.get('archived_count')}` already: `{summary.get('already_archived_count')}` failed: `{summary.get('failed_count')}`",
        f"- moved_gib: `{summary.get('moved_gib')}`",
        f"- estimated_reclaimed_gib: `{summary.get('estimated_reclaimed_gib')}`",
        f"- dry_run_candidate_gib: `{summary.get('dry_run_candidate_gib')}`",
        f"- dry_run_compressible_candidate_gib: `{summary.get('dry_run_compressible_candidate_gib')}`",
        f"- manifest: `{summary.get('manifest')}`",
        "",
        "## Actions",
        "",
    ]
    for row in payload.get("actions", [])[:40]:
        lines.append(f"- `{row.get('status')}` `{row.get('path')}` -> `{row.get('archive_path')}`")
    return "\n".join(lines) + "\n"


def retention_gate_view(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": payload.get("policy"),
        "trigger_state": payload.get("trigger_state"),
        "summary": payload.get("summary", {}),
        "failed_actions": [row for row in payload.get("actions", []) if row.get("status") == "failed"][:10],
        "sample_actions": payload.get("actions", [])[:10],
    }


def build_budget_gate_report(policy_path: Path, registry_path: Path, *, started: float) -> dict[str, Any]:
    policy = read_json(policy_path, {})
    registry = read_json(registry_path, {})
    rows = discover_budget_files(policy)
    surfaces = list_dicts(registry.get("surfaces"))
    owner_index = build_owner_index(surfaces)
    for row in rows:
        owner = owner_for_path(str(row["path"]), owner_index)
        row["registry_owner_surface_id"] = owner.get("id", "")
        row["registry_owner_role"] = owner.get("role", "")
        row["registry_owned"] = bool(owner)
        row["retention_class"] = retention_class_for_path(str(row["path"]), policy)
        row["retention_class_declared"] = bool(row["retention_class"])
        row["budget_scope"] = budget_scope_for_row(row, policy)

    summary = budget_summary(rows, policy)
    hard_gaps = budget_hard_gaps(summary, rows, policy)
    warnings = budget_warnings(summary, rows, policy)
    family_records = budget_family_records(rows)
    trigger_state = "GREEN" if not hard_gaps else "RED"
    return {
        "policy": "project_theseus_artifact_budget_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            **summary,
            "hard_gap_count": len(hard_gaps),
            "warning_count": len(warnings),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "inputs": {
            "budget_policy": rel(policy_path),
            "registry": rel(registry_path),
        },
        "budget_policy": policy,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "largest_files": sorted(rows, key=lambda row: -int(row["bytes"]))[:50],
        "largest_families": sorted(family_records, key=lambda row: -int(row["bytes"]))[:50],
        "report_family_budget_records": family_records,
        "rules": {
            "ownership": "Every generated report/checkpoint file must resolve to a registry surface owner.",
            "retention_class": "Every generated report/checkpoint file must match a declared retention class.",
            "hot_reports": "Mutable latest-view reports and unarchived snapshots must stay under the live hot-report byte/file budget.",
            "active_indexes": "Stateful indexes such as report_evidence_store.sqlite have their own cap and do not hide hot report sprawl.",
            "archive_pointers": "Archive pointers are retained as replay metadata and counted separately from hot report payload bytes.",
        },
        "non_claims": [
            "artifact budget compliance is repository hygiene evidence, not model capability evidence",
            "archived pointers do not delete evidence; replay gates must verify exact payload hashes",
            "checkpoint byte budgets do not prove checkpoint quality or training progress",
        ],
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }


def discover_budget_files(policy: dict[str, Any]) -> list[dict[str, Any]]:
    roots = [str(item) for item in list_values(policy.get("scan_roots"))] or ["reports", "checkpoints"]
    rows: list[dict[str, Any]] = []
    for raw_root in roots:
        root = resolve(raw_root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rows.append(budget_file_row(path))
    return rows


def budget_file_row(path: Path) -> dict[str, Any]:
    stat = path.stat()
    rel_path = rel(path)
    pointer = read_pointer(path)
    pointer_policy = pointer.get("policy") if isinstance(pointer, dict) else ""
    return {
        "record_type": "artifact_budget_file",
        "path": rel_path,
        "bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "family": budget_family(path),
        "is_archive_pointer": pointer_policy == ARCHIVE_POINTER_POLICY,
        "suffix": path.suffix.lower(),
    }


def budget_family(path: Path) -> str:
    rel_path = rel(path)
    parts = Path(rel_path).parts
    if len(parts) >= 3 and parts[0] == "reports" and parts[1] == "report_snapshots":
        return f"report_snapshots/{parts[2]}"
    if parts and parts[0] == "checkpoints":
        return f"checkpoint/{parts[1] if len(parts) > 1 else 'root'}"
    return artifact_family(path.name)


def build_owner_index(surfaces: list[dict[str, Any]]) -> dict[str, Any]:
    exact: dict[str, dict[str, Any]] = {}
    fallback = next((surface for surface in surfaces if str(surface.get("id") or "") == "runtime_generated_state"), {})
    for surface in surfaces:
        outputs = [str(item) for item in list_values(surface.get("report_outputs"))]
        for output in outputs:
            if not any(char in output for char in "*?[]"):
                exact.setdefault(output, surface)
    return {"exact": exact, "fallback": fallback}


def owner_for_path(rel_path: str, owner_index: dict[str, Any]) -> dict[str, Any]:
    exact = owner_index.get("exact") if isinstance(owner_index.get("exact"), dict) else {}
    if rel_path in exact:
        return exact[rel_path]
    if rel_path.startswith(("reports/", "checkpoints/")):
        fallback = owner_index.get("fallback")
        return fallback if isinstance(fallback, dict) else {}
    return {}


def retention_class_for_path(rel_path: str, policy: dict[str, Any]) -> str:
    for row in list_dicts(policy.get("retention_classes")):
        patterns = [str(item) for item in list_values(row.get("patterns"))]
        if any(fnmatch.fnmatch(rel_path, pattern) for pattern in patterns):
            return str(row.get("id") or "")
    return ""


def budget_scope_for_row(row: dict[str, Any], policy: dict[str, Any]) -> str:
    rel_path = str(row.get("path") or "")
    if bool(row.get("is_archive_pointer")):
        return "archive_pointer"
    for pattern in [str(item) for item in list_values(policy.get("active_index_patterns"))]:
        if fnmatch.fnmatch(rel_path, pattern):
            return "active_index"
    if rel_path.startswith("checkpoints/"):
        return "checkpoint"
    if rel_path.startswith("reports/"):
        return "hot_report"
    return "other_generated"


def budget_summary(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    def scope_rows(scope: str) -> list[dict[str, Any]]:
        return [row for row in rows if row.get("budget_scope") == scope]

    hot = scope_rows("hot_report")
    active = scope_rows("active_index")
    pointers = scope_rows("archive_pointer")
    checkpoints = scope_rows("checkpoint")
    other = scope_rows("other_generated")
    unowned = [row for row in rows if not row.get("registry_owned")]
    no_class = [row for row in rows if not row.get("retention_class_declared")]
    families = budget_family_records(rows)
    max_family = int(policy.get("max_single_family_bytes") or 0)
    over_family = [row for row in families if max_family and int(row.get("bytes") or 0) > max_family and row.get("budget_scope") == "hot_report"]
    return {
        "scanned_file_count": len(rows),
        "scanned_bytes": sum(int(row["bytes"]) for row in rows),
        "hot_report_file_count": len(hot),
        "hot_report_bytes": sum(int(row["bytes"]) for row in hot),
        "active_index_file_count": len(active),
        "active_index_bytes": sum(int(row["bytes"]) for row in active),
        "archive_pointer_file_count": len(pointers),
        "archive_pointer_bytes": sum(int(row["bytes"]) for row in pointers),
        "checkpoint_file_count": len(checkpoints),
        "checkpoint_bytes": sum(int(row["bytes"]) for row in checkpoints),
        "other_generated_file_count": len(other),
        "other_generated_bytes": sum(int(row["bytes"]) for row in other),
        "unowned_file_count": len(unowned),
        "missing_retention_class_count": len(no_class),
        "family_count": len(families),
        "hot_families_over_single_family_cap": len(over_family),
        "max_hot_report_bytes": int(policy.get("max_hot_report_bytes") or 0),
        "max_hot_report_files": int(policy.get("max_hot_report_files") or 0),
        "max_active_index_bytes": int(policy.get("max_active_index_bytes") or 0),
        "max_checkpoint_bytes": int(policy.get("max_checkpoint_bytes") or 0),
        "max_checkpoint_files": int(policy.get("max_checkpoint_files") or 0),
        "max_archive_pointer_files": int(policy.get("max_archive_pointer_files") or 0),
        "max_single_family_bytes": max_family,
    }


def budget_family_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("family") or ""), str(row.get("budget_scope") or ""))
        current = grouped.setdefault(
            key,
            {
                "record_type": "report_family_budget_record",
                "family": key[0],
                "budget_scope": key[1],
                "file_count": 0,
                "bytes": 0,
                "registry_owner_surface_ids": [],
                "retention_classes": [],
                "largest_file": "",
                "largest_file_bytes": 0,
                "public_training_rows_written": 0,
                "external_inference_calls": 0,
                "fallback_return_count": 0,
            },
        )
        current["file_count"] += 1
        current["bytes"] += int(row.get("bytes") or 0)
        owner = str(row.get("registry_owner_surface_id") or "")
        if owner and owner not in current["registry_owner_surface_ids"]:
            current["registry_owner_surface_ids"].append(owner)
        klass = str(row.get("retention_class") or "")
        if klass and klass not in current["retention_classes"]:
            current["retention_classes"].append(klass)
        if int(row.get("bytes") or 0) > int(current.get("largest_file_bytes") or 0):
            current["largest_file"] = str(row.get("path") or "")
            current["largest_file_bytes"] = int(row.get("bytes") or 0)
    return list(grouped.values())


def budget_hard_gaps(summary: dict[str, Any], rows: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    checks = [
        ("hot_report_bytes_over_budget", "hot_report_bytes", "max_hot_report_bytes"),
        ("hot_report_file_count_over_budget", "hot_report_file_count", "max_hot_report_files"),
        ("active_index_bytes_over_budget", "active_index_bytes", "max_active_index_bytes"),
        ("checkpoint_bytes_over_budget", "checkpoint_bytes", "max_checkpoint_bytes"),
        ("checkpoint_file_count_over_budget", "checkpoint_file_count", "max_checkpoint_files"),
        ("archive_pointer_file_count_over_budget", "archive_pointer_file_count", "max_archive_pointer_files"),
    ]
    for reason, actual_key, max_key in checks:
        maximum = int(summary.get(max_key) or 0)
        if maximum and int(summary.get(actual_key) or 0) > maximum:
            gaps.append(gap(reason, {"actual": summary.get(actual_key), "max": maximum}))
    if int(summary.get("unowned_file_count") or 0):
        gaps.append(gap("generated_artifacts_without_registry_owner", {"count": summary["unowned_file_count"], "examples": rows_with_flag(rows, "registry_owned", False)[:20]}))
    if int(summary.get("missing_retention_class_count") or 0):
        gaps.append(gap("generated_artifacts_without_retention_class", {"count": summary["missing_retention_class_count"], "examples": rows_with_flag(rows, "retention_class_declared", False)[:20]}))
    if int(summary.get("hot_families_over_single_family_cap") or 0):
        max_family = int(policy.get("max_single_family_bytes") or 0)
        offenders = [
            row
            for row in budget_family_records(rows)
            if row.get("budget_scope") == "hot_report" and int(row.get("bytes") or 0) > max_family
        ]
        gaps.append(gap("hot_report_family_over_budget", {"max_single_family_bytes": max_family, "families": offenders[:20]}))
    return gaps


def budget_warnings(summary: dict[str, Any], rows: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    checkpoint_warn = int(policy.get("checkpoint_warning_bytes") or 0)
    if checkpoint_warn and int(summary.get("checkpoint_bytes") or 0) > checkpoint_warn:
        warnings.append(gap("checkpoint_bytes_above_warning_target", {"actual": summary.get("checkpoint_bytes"), "warning": checkpoint_warn}, severity="warning"))
    return warnings


def rows_with_flag(rows: list[dict[str, Any]], key: str, expected: Any) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get(key) == expected:
            out.append({k: row.get(k) for k in ["path", "bytes", "family", "budget_scope", "registry_owner_surface_id", "retention_class"]})
    return out


def render_budget_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Theseus Artifact Budget Gate",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- hot_report_bytes: `{summary.get('hot_report_bytes')}` / `{summary.get('max_hot_report_bytes')}`",
        f"- hot_report_file_count: `{summary.get('hot_report_file_count')}` / `{summary.get('max_hot_report_files')}`",
        f"- active_index_bytes: `{summary.get('active_index_bytes')}` / `{summary.get('max_active_index_bytes')}`",
        f"- checkpoint_bytes: `{summary.get('checkpoint_bytes')}` / `{summary.get('max_checkpoint_bytes')}`",
        f"- checkpoint_file_count: `{summary.get('checkpoint_file_count')}` / `{summary.get('max_checkpoint_files')}`",
        f"- archive_pointer_file_count: `{summary.get('archive_pointer_file_count')}` / `{summary.get('max_archive_pointer_files')}`",
        f"- unowned_file_count: `{summary.get('unowned_file_count')}`",
        f"- missing_retention_class_count: `{summary.get('missing_retention_class_count')}`",
        f"- hard_gap_count: `{summary.get('hard_gap_count')}`",
        "",
        "## Largest Families",
        "",
    ]
    for row in payload.get("largest_families", [])[:20]:
        lines.append(
            f"- `{row.get('budget_scope')}` `{row.get('family')}`: `{row.get('file_count')}` files, `{row.get('bytes')}` bytes, owners `{row.get('registry_owner_surface_ids')}`, classes `{row.get('retention_classes')}`"
        )
    if payload.get("hard_gaps"):
        lines.extend(["", "## Hard Gaps", ""])
        for row in payload.get("hard_gaps", [])[:20]:
            lines.append(f"- `{row.get('reason')}` {json.dumps(row.get('detail', {}), sort_keys=True)[:1000]}")
    return "\n".join(lines) + "\n"


def budget_gate_view(payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(payload.get("summary") or {})
    return {
        "policy": payload.get("policy"),
        "trigger_state": payload.get("trigger_state"),
        "summary": summary,
        "hard_gaps": payload.get("hard_gaps", [])[:10],
        "warnings": payload.get("warnings", [])[:10],
        "largest_families": payload.get("largest_families", [])[:10],
    }


def is_pointer(path: Path) -> bool:
    return read_pointer(path).get("policy") == ARCHIVE_POINTER_POLICY


def read_pointer(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() not in INLINE_POINTER_SUFFIXES and not path.name.endswith(".archive-pointer.json"):
            return {}
        if not path.exists() or path.stat().st_size > MAX_POINTER_BYTES:
            return {}
        return read_json(path, {})
    except Exception:
        return {}


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value).strip("._")[:160] or "artifact"


def stable_id(*parts: Any) -> str:
    return hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:24]


def gap(reason: str, detail: Any, *, severity: str = "hard") -> dict[str, Any]:
    return {"reason": reason, "severity": severity, "detail": detail}


def list_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in list_values(value) if isinstance(row, dict)]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
