#!/usr/bin/env python3
"""Retire stale, regenerable T0A candidate-initialization caches.

The matched-candidate trainer writes one ``.npy`` file per parameter so the
second candidate can receive the exact common initialization.  Those files are
process-local transfer caches, not checkpoints or evidence, and historically
were left behind after the paired process exited.

This command is scan-only by default.  Execute mode requires an explicit
acknowledgement, writes a prepared manifest before removing anything, and
fingerprints the active production checkpoint before and after the operation.
It never removes checkpoints, receipts, reports, or any directory whose exact
basename is not ``candidate_common_initialization``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANARY_ROOT = ROOT / "runtime" / "t0a_canaries"
DEFAULT_PROTECTED_ROOT = (
    ROOT / "checkpoints" / "moecot_mlx_57m_active_preregistered_v1"
)
DEFAULT_OUT = ROOT / "reports" / "t0a_canary_storage_retention.json"
CACHE_BASENAME = "candidate_common_initialization"
ACKNOWLEDGEMENT = "delete-regenerable-candidate-initialization-caches"
NO_CHEAT = {
    "public_training_rows_written": 0,
    "external_inference_calls": 0,
    "fallback_return_count": 0,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_CANARY_ROOT))
    parser.add_argument("--protected-root", default=str(DEFAULT_PROTECTED_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--min-age-hours", type=float, default=24.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge", default="")
    args = parser.parse_args()

    root = safe_root(Path(args.root))
    protected_root = Path(args.protected_root).resolve()
    out = Path(args.out).resolve()
    if args.execute and args.acknowledge != ACKNOWLEDGEMENT:
        parser.error(f"--execute requires --acknowledge {ACKNOWLEDGEMENT}")

    started = time.perf_counter()
    tracked = tracked_paths()
    before_protected = fingerprint_tree(protected_root)
    candidates, rejected = discover(
        root,
        min_age_hours=max(0.0, float(args.min_age_hours)),
        tracked=tracked,
    )
    payload = build_payload(
        root=root,
        protected_root=protected_root,
        candidates=candidates,
        rejected=rejected,
        before_protected=before_protected,
        after_protected=before_protected,
        execute=bool(args.execute),
        phase="PREPARED" if args.execute else "DRY_RUN",
        started=started,
    )
    if args.execute:
        write_json_atomic(out, payload)

    actions = []
    if args.execute:
        for candidate in candidates:
            actions.append(remove_cache(root, candidate))
        after_protected = fingerprint_tree(protected_root)
        protected_unchanged = before_protected == after_protected
        payload = build_payload(
            root=root,
            protected_root=protected_root,
            candidates=candidates,
            rejected=rejected,
            before_protected=before_protected,
            after_protected=after_protected,
            execute=True,
            phase="COMPLETE",
            started=started,
            actions=actions,
        )
        if not protected_unchanged:
            payload["trigger_state"] = "RED"
            payload["hard_gaps"].append(
                {
                    "gap": "protected_production_tree_changed",
                    "before_manifest_sha256": before_protected["manifest_sha256"],
                    "after_manifest_sha256": after_protected["manifest_sha256"],
                }
            )
    write_json_atomic(out, payload)
    print(json.dumps(compact_view(payload), indent=2, sort_keys=True))
    return 0 if payload["trigger_state"] == "GREEN" else 2


def discover(
    root: Path,
    *,
    min_age_hours: float,
    tracked: set[str],
    now_timestamp: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    now_value = float(now_timestamp if now_timestamp is not None else time.time())
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    if not root.is_dir():
        return candidates, [{"path": relative(root), "reason": "root_missing"}]
    for path in sorted(root.rglob(CACHE_BASENAME)):
        if path.name != CACHE_BASENAME or not path.is_dir():
            continue
        try:
            record = inspect_cache(
                root,
                path,
                now_timestamp=now_value,
                tracked=tracked,
            )
        except ValueError as exc:
            rejected.append({"path": relative(path), "reason": str(exc)})
            continue
        if float(record["age_hours"]) < min_age_hours:
            rejected.append(
                {
                    "path": record["path"],
                    "reason": "younger_than_minimum_age",
                    "age_hours": record["age_hours"],
                    "minimum_age_hours": min_age_hours,
                }
            )
            continue
        candidates.append(record)
    return candidates, rejected


def inspect_cache(
    root: Path,
    path: Path,
    *,
    now_timestamp: float,
    tracked: set[str],
) -> dict[str, Any]:
    resolved = path.resolve()
    require_descendant(root, resolved)
    if resolved.name != CACHE_BASENAME:
        raise ValueError("basename_not_allowlisted")
    if path.is_symlink():
        raise ValueError("cache_directory_is_symlink")
    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    directories = [item for item in resolved.rglob("*") if item.is_dir()]
    symlinks = [item for item in resolved.rglob("*") if item.is_symlink()]
    if symlinks:
        raise ValueError("cache_contains_symlink")
    if directories:
        raise ValueError("cache_contains_nested_directory")
    if any(item.suffix != ".npy" for item in files):
        raise ValueError("cache_contains_non_npy_file")
    if any(relative(item) in tracked for item in files):
        raise ValueError("cache_contains_git_tracked_file")

    manifest = hashlib.sha256()
    logical_bytes = 0
    allocated_bytes = 0
    newest_mtime = resolved.stat().st_mtime
    metadata_rows = []
    for item in files:
        stat = item.stat()
        newest_mtime = max(newest_mtime, stat.st_mtime)
        logical_bytes += int(stat.st_size)
        allocated_bytes += int(getattr(stat, "st_blocks", 0)) * 512
        digest = sha256_file(item)
        name = item.relative_to(resolved).as_posix()
        row = f"{name}\0{stat.st_size}\0{digest}\n".encode("utf-8")
        manifest.update(row)
        metadata_rows.append((name, int(stat.st_size), int(stat.st_mtime_ns)))
    return {
        "record_type": "t0a_regenerable_cache_retention_candidate",
        "path": relative(resolved),
        "file_count": len(files),
        "logical_bytes": logical_bytes,
        "allocated_bytes": allocated_bytes,
        "logical_gib": round(logical_bytes / (1024**3), 6),
        "allocated_gib": round(allocated_bytes / (1024**3), 6),
        "newest_mtime_utc": utc_from_timestamp(newest_mtime),
        "age_hours": round(max(0.0, (now_timestamp - newest_mtime) / 3600.0), 3),
        "content_manifest_sha256": manifest.hexdigest(),
        "metadata_manifest_sha256": digest_json(metadata_rows),
        "classification": "regenerable_process_local_candidate_alignment_cache",
        "recovery": "rerun_the_exact_matched_candidate_pair",
    }


def remove_cache(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    path = (ROOT / str(candidate["path"])).resolve()
    if not path.exists() and Path(str(candidate["path"])).is_absolute():
        path = Path(str(candidate["path"])).resolve()
    try:
        require_descendant(root, path)
        if path.name != CACHE_BASENAME or path.is_symlink() or not path.is_dir():
            raise ValueError("target_no_longer_matches_allowlist")
        current = metadata_manifest(path)
        if current != candidate["metadata_manifest_sha256"]:
            raise ValueError("cache_changed_after_manifest")
        shutil.rmtree(path)
        if path.exists():
            raise ValueError("cache_directory_still_exists")
        return {
            **candidate,
            "status": "deleted",
            "deleted_utc": now(),
        }
    except Exception as exc:
        return {
            **candidate,
            "status": "failed",
            "error": repr(exc),
        }


def metadata_manifest(path: Path) -> str:
    rows = []
    for item in sorted(entry for entry in path.rglob("*") if entry.is_file()):
        stat = item.stat()
        rows.append(
            (
                item.relative_to(path).as_posix(),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            )
        )
    return digest_json(rows)


def fingerprint_tree(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {
            "path": relative(path),
            "exists": False,
            "file_count": 0,
            "logical_bytes": 0,
            "manifest_sha256": "",
        }
    manifest = hashlib.sha256()
    logical_bytes = 0
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        size = item.stat().st_size
        digest = sha256_file(item)
        logical_bytes += int(size)
        manifest.update(
            f"{item.relative_to(path).as_posix()}\0{size}\0{digest}\n".encode(
                "utf-8"
            )
        )
    return {
        "path": relative(path),
        "exists": True,
        "file_count": len(files),
        "logical_bytes": logical_bytes,
        "manifest_sha256": manifest.hexdigest(),
    }


def build_payload(
    *,
    root: Path,
    protected_root: Path,
    candidates: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    before_protected: dict[str, Any],
    after_protected: dict[str, Any],
    execute: bool,
    phase: str,
    started: float,
    actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    action_rows = actions or [
        {**candidate, "status": "prepared" if execute else "dry_run"}
        for candidate in candidates
    ]
    failures = [row for row in action_rows if row.get("status") == "failed"]
    deleted = [row for row in action_rows if row.get("status") == "deleted"]
    hard_gaps = [
        {"gap": "candidate_rejected", **row}
        for row in rejected
        if row.get("reason") not in {"younger_than_minimum_age"}
    ]
    protected_unchanged = before_protected == after_protected
    if phase == "COMPLETE" and not protected_unchanged:
        hard_gaps.append({"gap": "protected_production_tree_changed"})
    state = "GREEN"
    if failures or hard_gaps:
        state = "RED"
    return {
        "policy": "project_theseus_t0a_regenerable_cache_retention_v1",
        "created_utc": now(),
        "trigger_state": state,
        "phase": phase,
        "scope": {
            "root": relative(root),
            "exact_deletable_basename": CACHE_BASENAME,
            "allowed_file_suffix": ".npy",
            "protected_root": relative(protected_root),
        },
        "summary": {
            "execute": execute,
            "candidate_directory_count": len(candidates),
            "candidate_file_count": sum(int(row["file_count"]) for row in candidates),
            "candidate_logical_bytes": sum(
                int(row["logical_bytes"]) for row in candidates
            ),
            "candidate_allocated_bytes": sum(
                int(row["allocated_bytes"]) for row in candidates
            ),
            "candidate_allocated_gib": round(
                sum(int(row["allocated_bytes"]) for row in candidates)
                / (1024**3),
                3,
            ),
            "deleted_directory_count": len(deleted),
            "deleted_allocated_bytes": sum(
                int(row["allocated_bytes"]) for row in deleted
            ),
            "deleted_allocated_gib": round(
                sum(int(row["allocated_bytes"]) for row in deleted) / (1024**3),
                3,
            ),
            "failed_count": len(failures),
            "rejected_count": len(rejected),
            "protected_tree_unchanged": protected_unchanged,
            "runtime_seconds": round(time.perf_counter() - started, 3),
        },
        "classification": {
            "deleted": "process-local common-initialization transfer cache",
            "preserved": [
                "all checkpoints and optimizer states",
                "all training and evaluation receipts",
                "all negative evidence and reports",
                "the complete active production checkpoint tree",
            ],
            "scientific_disposition": (
                "Storage cleanup only. It does not change a model, result, "
                "architecture disposition, or capability claim."
            ),
        },
        "protected_tree_before": before_protected,
        "protected_tree_after": after_protected,
        "actions": action_rows,
        "rejected": rejected,
        "hard_gaps": hard_gaps,
        **NO_CHEAT,
    }


def compact_view(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": payload["policy"],
        "trigger_state": payload["trigger_state"],
        "phase": payload["phase"],
        "summary": payload["summary"],
        "hard_gaps": payload["hard_gaps"],
    }


def tracked_paths() -> set[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return {
        line.strip().replace("\\", "/")
        for line in result.stdout.splitlines()
        if line.strip()
    }


def safe_root(path: Path) -> Path:
    resolved = path.resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), ROOT.resolve()}
    if resolved in forbidden or len(resolved.parts) < 3:
        raise ValueError(f"unsafe retention root: {resolved}")
    return resolved


def require_descendant(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("target_outside_retention_root") from exc


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_from_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
