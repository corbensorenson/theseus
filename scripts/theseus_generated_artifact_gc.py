#!/usr/bin/env python3
"""Generated-artifact garbage collection service for Theseus.

Default mode is scan-only. Execute mode quarantines safe generated artifacts
under ``archive/gc_quarantine`` with a manifest. It does not delete by default.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
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


REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "theseus_generated_artifact_gc.json"
DEFAULT_MARKDOWN = REPORTS / "theseus_generated_artifact_gc.md"
DEFAULT_MANIFEST = REPORTS / "theseus_generated_artifact_gc_manifest.json"
QUARANTINE_ROOT = ROOT / "archive" / "gc_quarantine"
RUNTIME_REHYDRATED_ARTIFACTS = ROOT / "runtime" / "rehydrated_artifacts"
SCAN_DIRS = [ROOT / "tmp", ROOT / ".attd_tmp", ROOT / "logs"]
SAFE_SUFFIXES = {".tmp", ".temp", ".log", ".trace", ".stdout", ".stderr"}
COMPRESSIBLE_SUFFIXES = {".json", ".jsonl"}
SAFE_NAME_TOKENS = ("tmp", "temp", "smoke", "scratch", "sandbox", "pytest", "candidate_sandbox")
NEVER_TOUCH_DIRS = {".git", "target", "vendor", "data", "checkpoints", "reports", "archive"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--delete", action="store_true", help="Delete quarantined candidates instead of moving. Off by default.")
    parser.add_argument("--compress-json", action="store_true", help="Compress JSON/JSONL candidates when quarantining them.")
    parser.add_argument("--min-age-hours", type=float, default=24.0)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--include-runtime-rehydrated-artifacts", action="store_true")
    parser.add_argument("--quarantine-root", default=str(QUARANTINE_ROOT.relative_to(ROOT)))
    parser.add_argument("--manifest-out", default=str(DEFAULT_MANIFEST.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    candidates = discover_candidates(
        min_age_hours=max(0.0, float(args.min_age_hours)),
        include_runtime_rehydrated_artifacts=bool(args.include_runtime_rehydrated_artifacts),
    )[: max(0, int(args.max_candidates))]
    actions = apply_actions(candidates, args) if args.execute else [{**row, "status": "dry_run"} for row in candidates]
    manifest = {
        "policy": "project_theseus_generated_artifact_gc_manifest_v1",
        "created_utc": now(),
        "quarantine_root": rel(resolve(args.quarantine_root)),
        "entry_count": len(actions),
        "entries": actions,
        "external_inference_calls": 0,
    }
    if args.execute:
        write_json(resolve(args.manifest_out), manifest)

    reclaimed = sum(
        int(row.get("bytes") or 0)
        for row in actions
        if row.get("status") in {"quarantined", "already_quarantined", "deleted"}
    )
    storage_saved = sum(int(row.get("storage_saved_bytes") or 0) for row in actions)
    payload = {
        "policy": "project_theseus_generated_artifact_gc_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not any(row.get("status") == "failed" for row in actions) else "YELLOW",
        "summary": {
            "execute": bool(args.execute),
            "delete": bool(args.delete),
            "candidate_count": len(candidates),
            "quarantined_count": sum(1 for row in actions if row.get("status") == "quarantined"),
            "deleted_count": sum(1 for row in actions if row.get("status") == "deleted"),
            "failed_count": sum(1 for row in actions if row.get("status") == "failed"),
            "reclaimed_mib": round(reclaimed / (1024**2), 3),
            "storage_saved_mib": round(storage_saved / (1024**2), 3),
            "min_age_hours": float(args.min_age_hours),
            "manifest": rel(resolve(args.manifest_out)),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "candidates": candidates,
        "actions": actions,
        "rules": {
            "default": "scan_only",
            "execute_without_delete": "move safe candidates to archive/gc_quarantine",
            "delete": "requires explicit --delete and only affects discovered safe generated candidates",
            "compress_json": bool(args.compress_json),
            "include_runtime_rehydrated_artifacts": bool(args.include_runtime_rehydrated_artifacts),
            "never_touch_dirs": sorted(NEVER_TOUCH_DIRS),
        },
        "external_inference_calls": 0,
    }
    report_evidence_store.write_json_report(
        resolve(args.out),
        payload,
        markdown_path=resolve(args.markdown_out),
        markdown_text=render_markdown(payload),
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] == "GREEN" else 2


def discover_candidates(
    *,
    min_age_hours: float,
    include_runtime_rehydrated_artifacts: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    tracked = git_tracked_paths()
    for base in SCAN_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if rel(path) in tracked:
                continue
            if any(part in NEVER_TOUCH_DIRS for part in path.parts):
                continue
            if not safe_generated_path(path):
                continue
            stat = path.stat()
            age = max(0.0, (time.time() - stat.st_mtime) / 3600.0)
            if age < min_age_hours:
                continue
            rows.append(
                {
                    "record_type": "generated_artifact_gc_candidate",
                    "path": rel(path),
                    "bytes": int(stat.st_size),
                    "mib": round(stat.st_size / (1024**2), 3),
                    "age_hours": round(age, 3),
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                    "reason": reason_for(path),
                }
            )
    if include_runtime_rehydrated_artifacts and RUNTIME_REHYDRATED_ARTIFACTS.exists():
        for path in RUNTIME_REHYDRATED_ARTIFACTS.rglob("*"):
            if not path.is_file():
                continue
            if rel(path) in tracked:
                continue
            stat = path.stat()
            age = max(0.0, (time.time() - stat.st_mtime) / 3600.0)
            if age < min_age_hours:
                continue
            rows.append(
                {
                    "record_type": "generated_artifact_gc_candidate",
                    "path": rel(path),
                    "bytes": int(stat.st_size),
                    "mib": round(stat.st_size / (1024**2), 3),
                    "age_hours": round(age, 3),
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                    "reason": "runtime_rehydrated_artifact_cache_mirror",
                }
            )
    return sorted(rows, key=lambda row: (-float(row["age_hours"]), -int(row["bytes"]), row["path"]))


def git_tracked_paths() -> set[str]:
    try:
        result = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return {line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()}


def safe_generated_path(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() in SAFE_SUFFIXES:
        return True
    return any(token in name for token in SAFE_NAME_TOKENS)


def reason_for(path: Path) -> str:
    if path.suffix.lower() in SAFE_SUFFIXES:
        return f"safe_generated_suffix_{path.suffix.lower().lstrip('.')}"
    for token in SAFE_NAME_TOKENS:
        if token in path.name.lower():
            return f"safe_generated_name_token_{token}"
    return "safe_generated_candidate"


def apply_actions(candidates: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    actions = []
    quarantine_root = resolve(args.quarantine_root)
    for row in candidates:
        source = resolve(row["path"])
        target = quarantine_root / quarantine_relative_path(source, str(row.get("path") or ""))
        try:
            if args.delete:
                source.unlink()
                actions.append({**row, "status": "deleted", "storage_saved_bytes": int(row.get("bytes") or 0)})
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            compress = bool(args.compress_json and source.suffix.lower() in COMPRESSIBLE_SUFFIXES)
            if compress:
                target = target.with_name(target.name + ".gz")
            if compress:
                source_sha256 = sha256_file(source)
                if target.exists():
                    verify = verify_gzip_payload(target, source_sha256)
                    if verify["verified"]:
                        source.unlink()
                        quarantine_bytes = target.stat().st_size
                        actions.append(
                            {
                                **row,
                                "status": "already_quarantined",
                                "quarantine_path": rel(target),
                                "compressed": True,
                                "sha256": source_sha256,
                                "quarantine_bytes": quarantine_bytes,
                                "storage_saved_bytes": max(0, int(row.get("bytes") or 0) - quarantine_bytes),
                            }
                        )
                        continue
                    target = target.with_name(target.name + f".{int(time.time())}.quarantine")
                gzip_copy(source, target)
                verify = verify_gzip_payload(target, source_sha256)
                if not verify["verified"]:
                    target.unlink(missing_ok=True)
                    actions.append({**row, "status": "failed", "quarantine_path": rel(target), **verify})
                    continue
                source.unlink()
                quarantine_bytes = target.stat().st_size
                actions.append(
                    {
                        **row,
                        "status": "quarantined",
                        "quarantine_path": rel(target),
                        "compressed": True,
                        "sha256": source_sha256,
                        "quarantine_bytes": quarantine_bytes,
                        "storage_saved_bytes": max(0, int(row.get("bytes") or 0) - quarantine_bytes),
                    }
                )
            else:
                if target.exists():
                    target = target.with_name(target.name + f".{int(time.time())}.quarantine")
                shutil.move(str(source), str(target))
                actions.append(
                    {
                        **row,
                        "status": "quarantined",
                        "quarantine_path": rel(target),
                        "compressed": False,
                        "quarantine_bytes": target.stat().st_size,
                        "storage_saved_bytes": 0,
                    }
                )
        except Exception as exc:
            actions.append({**row, "status": "failed", "error": repr(exc)})
    return actions


def gzip_copy(source: Path, target: Path) -> None:
    with source.open("rb") as src, gzip.open(target, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def verify_gzip_payload(target: Path, expected_sha256: str) -> dict[str, Any]:
    actual = sha256_gzip_payload(target)
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


def quarantine_relative_path(source: Path, raw_path: str) -> Path:
    raw = Path(raw_path)
    if not raw.is_absolute():
        return raw
    try:
        return source.relative_to(ROOT)
    except ValueError:
        return Path(source.name)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Theseus Generated Artifact GC",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- execute: `{summary.get('execute')}` delete: `{summary.get('delete')}`",
        f"- candidates: `{summary.get('candidate_count')}`",
        f"- quarantined: `{summary.get('quarantined_count')}` deleted: `{summary.get('deleted_count')}` failed: `{summary.get('failed_count')}`",
        f"- reclaimable/reclaimed MiB: `{summary.get('reclaimed_mib')}`",
        f"- storage_saved MiB: `{summary.get('storage_saved_mib')}`",
        "",
        "## Top Candidates",
        "",
    ]
    for row in payload.get("actions", payload.get("candidates", []))[:40]:
        lines.append(f"- `{row.get('status', 'candidate')}` `{row.get('path')}` {row.get('mib')} MiB age={row.get('age_hours')}h")
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
