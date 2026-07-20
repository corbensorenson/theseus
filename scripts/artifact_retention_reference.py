#!/usr/bin/env python3
"""Current-reference and deduplication kernel for checkpoint retention.

Operational configs, route-required evidence, active manifests, and explicit
pins protect checkpoint payloads. Historical reports may cite archived
checkpoints through exact pointers, but they do not keep every old weight file
hot forever.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "checkpoints"
REPORTS_ROOT = ROOT / "reports"
MODEL_SUFFIXES = {".npz", ".pt", ".pth", ".bin", ".safetensors"}
REFERENCE_PATTERN = re.compile(r"checkpoints/[A-Za-z0-9_.@+\-/]+")
REPORT_REFERENCE_PATTERN = re.compile(r"reports/[A-Za-z0-9_.@+\-/]+")
CONFIG_EXCLUSIONS = {
    "artifact_retention_budget_policy.json",
    "roadmap_implementation_matrix.json",
}


def build_checkpoint_reference_index(
    registry: dict[str, Any],
    *,
    explicit_pins: Iterable[str] = (),
) -> dict[str, Any]:
    sources = operational_reference_sources(registry)
    references: dict[str, set[str]] = defaultdict(set)
    queue = list(sources)
    visited: set[Path] = set()
    while queue:
        source = queue.pop(0)
        if source in visited or not source.is_file() or source.stat().st_size > 32 * 1024 * 1024:
            continue
        visited.add(source)
        for ref in checkpoint_refs_in_text(source.read_text(encoding="utf-8", errors="ignore")):
            references[ref].add(rel(source))
            target = resolve(ref)
            if target.is_file() and target.suffix.lower() in {".json", ".jsonl", ".txt"}:
                queue.append(target)
    for pin in explicit_pins:
        ref = normalize_checkpoint_ref(pin)
        if ref:
            references[ref].add("explicit_cli_pin")

    protected: dict[str, set[str]] = defaultdict(set)
    for ref, source_refs in references.items():
        target = resolve(ref)
        if target.is_dir():
            for path in target.rglob("*"):
                if path.is_file():
                    protected[rel(path)].update(f"directory_ref:{source}" for source in source_refs)
        elif target.is_file():
            protected[rel(target)].update(f"file_ref:{source}" for source in source_refs)
        if "hive_promoted" in target.parts:
            bundle = target if target.is_dir() else target.parent
            for path in bundle.rglob("*"):
                if path.is_file():
                    protected[rel(path)].add("active_promoted_bundle")

    all_files = [path for path in CHECKPOINT_ROOT.rglob("*") if path.is_file()]
    records = [
        {
            "path": rel(path),
            "bytes": int(path.stat().st_size),
            "protected": rel(path) in protected,
            "protection_reasons": sorted(protected.get(rel(path), set())),
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        for path in all_files
    ]
    return {
        "policy": "project_theseus_current_checkpoint_reference_index_v1",
        "state": "GREEN",
        "reference_source_count": len(visited),
        "declared_reference_count": len(references),
        "checkpoint_file_count": len(records),
        "protected_checkpoint_file_count": sum(1 for row in records if row["protected"]),
        "protected_checkpoint_bytes": sum(int(row["bytes"]) for row in records if row["protected"]),
        "unprotected_checkpoint_file_count": sum(1 for row in records if not row["protected"]),
        "unprotected_checkpoint_bytes": sum(int(row["bytes"]) for row in records if not row["protected"]),
        "reference_sources": sorted(rel(path) for path in visited),
        "references": [
            {"checkpoint_ref": ref, "source_refs": sorted(source_refs), "exists": resolve(ref).exists()}
            for ref, source_refs in sorted(references.items())
        ],
        "file_records": records,
        "rules": {
            "operational_configs": "Current configs protect referenced checkpoints; roadmap/history matrices do not pin weights.",
            "route_required_evidence": "Registry route-required reports protect checkpoint payloads needed by a current route.",
            "active_manifests": "Active promoted manifests and their recursively referenced bundles remain hot.",
            "historical_evidence": "Historical reports keep exact archive pointers and replay receipts rather than pinning all weight payloads hot.",
        },
        "non_claim": "Reference reachability is a retention decision, not checkpoint quality evidence.",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def checkpoint_archive_candidates(
    reference_index: dict[str, Any],
    *,
    min_bytes: int,
    min_age_hours: float,
    target_hot_bytes: int,
    now_timestamp: float | None = None,
) -> list[dict[str, Any]]:
    now_value = float(now_timestamp if now_timestamp is not None else datetime.now(timezone.utc).timestamp())
    rows = []
    total_hot_bytes = sum(int(row.get("bytes") or 0) for row in list_dicts(reference_index.get("file_records")))
    required_reduction = max(0, total_hot_bytes - max(0, int(target_hot_bytes))) if target_hot_bytes else total_hot_bytes
    reduced = 0
    eligible = []
    for row in list_dicts(reference_index.get("file_records")):
        path = resolve(str(row.get("path") or ""))
        if row.get("protected") or not path.is_file() or path.suffix.lower() not in MODEL_SUFFIXES:
            continue
        age_hours = max(0.0, (now_value - path.stat().st_mtime) / 3600.0)
        if int(row.get("bytes") or 0) < max(1, int(min_bytes)) or age_hours < max(0.0, float(min_age_hours)):
            continue
        eligible.append(
            {
                "record_type": "artifact_retention_candidate",
                "path": rel(path),
                "original_path": rel(path),
                "bytes": int(row.get("bytes") or 0),
                "gib": round(int(row.get("bytes") or 0) / (1024**3), 3),
                "mtime_utc": row.get("mtime_utc"),
                "family": checkpoint_family(path),
                "reason": "unreferenced_historical_checkpoint_payload",
                "current_reference_state": "unprotected",
                "age_hours": round(age_hours, 3),
                "archive_uncompressed": True,
            }
        )
    for row in sorted(eligible, key=lambda item: (-int(item["bytes"]), str(item["path"]))):
        if reduced >= required_reduction:
            break
        rows.append(row)
        reduced += int(row["bytes"])
    return rows


def deduplicate_checkpoint_payloads(
    reference_index: dict[str, Any],
    *,
    execute: bool,
    min_bytes: int = 1024 * 1024,
) -> dict[str, Any]:
    protected = {
        str(row.get("path") or "")
        for row in list_dicts(reference_index.get("file_records"))
        if row.get("protected")
    }
    by_size: dict[int, list[Path]] = defaultdict(list)
    for path in CHECKPOINT_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES and path.stat().st_size >= max(1, int(min_bytes)):
            by_size[int(path.stat().st_size)].append(path)
    digest_groups: dict[tuple[int, str], list[Path]] = defaultdict(list)
    hashed_bytes = 0
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        for path in paths:
            digest_groups[(size, sha256_file(path))].append(path)
            hashed_bytes += size
    duplicate_groups = [paths for paths in digest_groups.values() if len(paths) > 1]
    actions: list[dict[str, Any]] = []
    for paths in duplicate_groups:
        ordered = sorted(paths, key=lambda path: (0 if rel(path) in protected else 1, rel(path)))
        canonical = ordered[0]
        canonical_digest = sha256_file(canonical)
        for duplicate in ordered[1:]:
            action = {
                "record_type": "checkpoint_deduplication_record",
                "canonical_path": rel(canonical),
                "duplicate_path": rel(duplicate),
                "content_sha256": canonical_digest,
                "bytes": int(duplicate.stat().st_size),
                "status": "dry_run",
                "same_inode_before": same_inode(canonical, duplicate),
            }
            if execute and not action["same_inode_before"]:
                action.update(hardlink_replace(canonical, duplicate, canonical_digest))
            elif execute:
                action["status"] = "already_deduplicated"
                action["same_inode_after"] = True
            actions.append(action)
    reclaimed = sum(
        int(row.get("bytes") or 0)
        for row in actions
        if row.get("status") == "deduplicated"
    )
    failures = [row for row in actions if row.get("status") == "failed"]
    return {
        "policy": "project_theseus_checkpoint_hardlink_deduplication_v1",
        "state": "GREEN" if not failures else "RED",
        "execute": bool(execute),
        "hashed_candidate_bytes": hashed_bytes,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_file_count": len(actions),
        "deduplicated_file_count": sum(1 for row in actions if row.get("status") == "deduplicated"),
        "already_deduplicated_file_count": sum(1 for row in actions if row.get("same_inode_before")),
        "physical_bytes_reclaimed": reclaimed,
        "failed_count": len(failures),
        "actions": actions,
        "replay_contract": "duplicate path retains identical sha256 and shares the canonical inode after atomic replacement",
        "non_claim": "Checkpoint deduplication changes storage allocation only; it does not change or improve model weights.",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_hot_report_reference_index(
    registry: dict[str, Any],
    roadmap_matrix: dict[str, Any],
) -> dict[str, Any]:
    references: dict[str, set[str]] = defaultdict(set)
    for contract in list_dicts(registry.get("route_evidence_contracts")):
        for requirement in list_dicts(contract.get("requirements")):
            add_report_reference(references, requirement.get("path"), "registry_route_evidence")
    for collection in (registry.get("surfaces"), registry.get("implementations")):
        for row in list_dicts(collection):
            for key in ("report_outputs", "evidence_outputs"):
                for value in list_values(row.get(key)):
                    add_report_reference(references, value, f"registry_{key}")
    for phase in list_dicts(roadmap_matrix.get("phases")):
        for value in list_values(phase.get("current_evidence")):
            add_report_reference(references, value, f"roadmap_phase_{phase.get('phase')}_current_evidence")
    config_root = ROOT / "configs"
    for source in config_root.iterdir() if config_root.exists() else []:
        if (
            not source.is_file()
            or source.suffix.lower() not in {".json", ".toml", ".yaml", ".yml"}
            or source.stat().st_size > 32 * 1024 * 1024
        ):
            continue
        for report_ref in report_refs_in_text(
            source.read_text(encoding="utf-8", errors="ignore")
        ):
            references[report_ref].add(f"operational_config:{rel(source)}")

    records = []
    for path in REPORTS_ROOT.rglob("*") if REPORTS_ROOT.exists() else []:
        if not path.is_file():
            continue
        ref = rel(path)
        protected_reasons = set(references.get(ref, set()))
        if "ledger" in path.stem.lower() or path.suffix.lower() in {".sqlite", ".db"} or ".sqlite-" in path.name:
            protected_reasons.add("mutable_ledger_or_index")
        records.append(
            {
                "path": ref,
                "bytes": int(path.stat().st_size),
                "protected": bool(protected_reasons),
                "protection_reasons": sorted(protected_reasons),
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        )
    return {
        "policy": "project_theseus_current_hot_report_reference_index_v1",
        "state": "GREEN",
        "declared_reference_count": len(references),
        "report_file_count": len(records),
        "protected_report_file_count": sum(1 for row in records if row["protected"]),
        "protected_report_bytes": sum(int(row["bytes"]) for row in records if row["protected"]),
        "unprotected_report_file_count": sum(1 for row in records if not row["protected"]),
        "unprotected_report_bytes": sum(int(row["bytes"]) for row in records if not row["protected"]),
        "file_records": records,
        "rules": {
            "exact_current_references": "Registry routes/outputs and roadmap current evidence remain hot.",
            "ledgers": "Mutable ledger/index paths remain hot even when not cited by a single gate.",
            "historical_views": "Old unreferenced views retain exact archive pointers and cumulative replay evidence.",
        },
        "non_claim": "Report reachability is retention evidence, not capability evidence.",
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def hot_report_archive_candidates(
    reference_index: dict[str, Any],
    *,
    min_bytes: int,
    min_age_hours: float,
    target_hot_bytes: int,
    now_timestamp: float | None = None,
) -> list[dict[str, Any]]:
    now_value = float(now_timestamp if now_timestamp is not None else datetime.now(timezone.utc).timestamp())
    records = list_dicts(reference_index.get("file_records"))
    total_bytes = sum(int(row.get("bytes") or 0) for row in records)
    required_reduction = max(0, total_bytes - max(0, int(target_hot_bytes))) if target_hot_bytes else total_bytes
    reduced = 0
    candidates: list[dict[str, Any]] = []
    eligible = []
    for row in records:
        path = resolve(str(row.get("path") or ""))
        if row.get("protected") or not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        if archive_pointer_payload(path):
            continue
        age_hours = max(0.0, (now_value - path.stat().st_mtime) / 3600.0)
        if int(row.get("bytes") or 0) < max(1, int(min_bytes)) or age_hours < max(0.0, float(min_age_hours)):
            continue
        eligible.append(
            {
                "record_type": "artifact_retention_candidate",
                "path": rel(path),
                "original_path": rel(path),
                "bytes": int(row.get("bytes") or 0),
                "gib": round(int(row.get("bytes") or 0) / (1024**3), 3),
                "mtime_utc": row.get("mtime_utc"),
                "family": report_family(path),
                "reason": "unreferenced_historical_hot_report",
                "current_reference_state": "unprotected",
                "age_hours": round(age_hours, 3),
            }
        )
    for row in sorted(eligible, key=lambda item: (-int(item["bytes"]), str(item["path"]))):
        if reduced >= required_reduction:
            break
        candidates.append(row)
        reduced += int(row["bytes"])
    return candidates


def operational_reference_sources(registry: dict[str, Any]) -> list[Path]:
    sources: set[Path] = set()
    configs = ROOT / "configs"
    for path in configs.iterdir() if configs.exists() else []:
        if path.is_file() and path.suffix.lower() in {".json", ".toml", ".yaml", ".yml"} and path.name not in CONFIG_EXCLUSIONS:
            sources.add(path)
    for path in CHECKPOINT_ROOT.rglob("active_manifest*.json") if CHECKPOINT_ROOT.exists() else []:
        sources.add(path)
    for contract in list_dicts(registry.get("route_evidence_contracts")):
        for requirement in list_dicts(contract.get("requirements")):
            path = resolve(str(requirement.get("path") or ""))
            if path.is_file():
                sources.add(path)
    return sorted(sources)


def add_report_reference(references: dict[str, set[str]], value: Any, reason: str) -> None:
    text = str(value or "").strip()
    if text.startswith("reports/") and not any(char in text for char in "*?[]"):
        references[text].add(reason)


def archive_pointer_payload(path: Path) -> bool:
    if path.stat().st_size > 1024 * 1024:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("policy") == "project_theseus_archived_artifact_pointer_v1"


def report_family(path: Path) -> str:
    return f"hot_report/{path.stem[:120]}"


def checkpoint_refs_in_text(text: str) -> list[str]:
    return sorted(
        {
            normalize_checkpoint_ref(match.group(0))
            for match in REFERENCE_PATTERN.finditer(text)
            if normalize_checkpoint_ref(match.group(0))
        }
    )


def report_refs_in_text(text: str) -> list[str]:
    return sorted(
        {
            match.group(0).rstrip(".,:;\"'")
            for match in REPORT_REFERENCE_PATTERN.finditer(text)
            if ".." not in Path(match.group(0)).parts
        }
    )


def normalize_checkpoint_ref(value: Any) -> str:
    text = str(value or "").strip().rstrip(".,:;\"'")
    if not text.startswith("checkpoints/") or ".." in Path(text).parts:
        return ""
    return text


def hardlink_replace(canonical: Path, duplicate: Path, expected_digest: str) -> dict[str, Any]:
    temporary = duplicate.with_name(f".{duplicate.name}.dedupe-{os.getpid()}.tmp")
    try:
        temporary.unlink(missing_ok=True)
        os.link(canonical, temporary)
        if sha256_file(temporary) != expected_digest:
            temporary.unlink(missing_ok=True)
            return {"status": "failed", "error": "temporary_hardlink_hash_mismatch"}
        os.replace(temporary, duplicate)
        passed = same_inode(canonical, duplicate) and sha256_file(duplicate) == expected_digest
        return {
            "status": "deduplicated" if passed else "failed",
            "same_inode_after": same_inode(canonical, duplicate),
            "error": "" if passed else "post_replace_replay_failed",
        }
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        return {"status": "failed", "error": repr(exc), "same_inode_after": False}


def same_inode(left: Path, right: Path) -> bool:
    try:
        left_stat = left.stat()
        right_stat = right.stat()
        return (left_stat.st_dev, left_stat.st_ino) == (right_stat.st_dev, right_stat.st_ino)
    except OSError:
        return False


def checkpoint_family(path: Path) -> str:
    try:
        relative = path.relative_to(CHECKPOINT_ROOT)
        return f"checkpoint/{relative.parts[0] if relative.parts else 'root'}"
    except ValueError:
        return "checkpoint/external"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in list_values(value) if isinstance(item, dict)]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)
