#!/usr/bin/env python3
"""Retire stale, regenerable T0A storage without weakening evidence.

The matched-candidate trainer writes one ``.npy`` file per parameter so the
second candidate can receive the exact common initialization.  Those files are
process-local transfer caches, not checkpoints or evidence, and historically
were left behind after the paired process exited.

The optional checkpoint-compaction mode removes only superseded versioned
model/optimizer/RNG generations when the unversioned aliases exactly match the
terminal versioned generation.  It preserves the terminal generation, aliases,
receipts, merged diagnostic weights, reports, and active production tree.

This command is scan-only by default.  Execute mode requires an explicit
acknowledgement, writes a prepared manifest before removing anything, and
fingerprints the active production checkpoint before and after the operation.
Default mode never removes checkpoints, receipts, reports, or any directory
whose exact basename is not ``candidate_common_initialization``.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
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
DEFAULT_POLICY = ROOT / "configs" / "artifact_retention_budget_policy.json"
CACHE_BASENAME = "candidate_common_initialization"
ACKNOWLEDGEMENT = "delete-regenerable-candidate-initialization-caches"
CHECKPOINT_ACKNOWLEDGEMENT = "delete-superseded-canary-checkpoint-generations"
RUN_ACKNOWLEDGEMENT = "delete-unreferenced-closed-canary-runs"
CHECKPOINT_PATTERN = re.compile(
    r"^(?P<kind>weights|optimizer)\.step-(?P<step>[0-9]+)"
    r"(?P<rng>\.mlx-rng)?\.safetensors$"
)
ALIAS_BY_KIND = {
    "weights": "weights.safetensors",
    "optimizer": "optimizer.safetensors",
    "optimizer_rng": "optimizer.mlx-rng.safetensors",
}
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
    parser.add_argument(
        "--compact-superseded-checkpoints",
        action="store_true",
        help=(
            "Retire non-terminal versioned checkpoint generations only after "
            "terminal aliases pass exact digest parity."
        ),
    )
    parser.add_argument(
        "--retire-unreferenced-canary-runs",
        action="store_true",
        help=(
            "Retire complete closed canary runs only when no canonical config, "
            "roadmap, or project-state reference protects the run."
        ),
    )
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge", default="")
    args = parser.parse_args()

    root = safe_root(Path(args.root))
    protected_root = Path(args.protected_root).resolve()
    out = Path(args.out).resolve()
    if args.compact_superseded_checkpoints and args.retire_unreferenced_canary_runs:
        parser.error("checkpoint compaction and run retirement are separate modes")
    required_acknowledgement = (
        RUN_ACKNOWLEDGEMENT
        if args.retire_unreferenced_canary_runs
        else (
            CHECKPOINT_ACKNOWLEDGEMENT
            if args.compact_superseded_checkpoints
            else ACKNOWLEDGEMENT
        )
    )
    if args.execute and args.acknowledge != required_acknowledgement:
        parser.error(
            f"--execute requires --acknowledge {required_acknowledgement}"
        )

    started = time.perf_counter()
    tracked = tracked_paths()
    before_protected = fingerprint_tree(protected_root)
    free_before = free_bytes(root)
    if args.retire_unreferenced_canary_runs:
        policy = load_runtime_canary_policy(Path(args.policy))
        candidates, rejected, preserved = discover_unreferenced_canary_runs(
            root,
            policy=policy,
            min_age_hours=max(0.0, float(args.min_age_hours)),
            tracked=tracked,
        )
        payload = build_run_retirement_payload(
            root=root,
            protected_root=protected_root,
            policy_path=Path(args.policy).resolve(),
            candidates=candidates,
            rejected=rejected,
            preserved=preserved,
            before_protected=before_protected,
            after_protected=before_protected,
            execute=bool(args.execute),
            phase="PREPARED" if args.execute else "DRY_RUN",
            started=started,
            free_before=free_before,
            free_after=free_before,
        )
        if args.execute:
            write_json_atomic(out, payload)
        actions = []
        if args.execute:
            for candidate in candidates:
                actions.append(remove_canary_run(root, candidate))
            after_protected = fingerprint_tree(protected_root)
            payload = build_run_retirement_payload(
                root=root,
                protected_root=protected_root,
                policy_path=Path(args.policy).resolve(),
                candidates=candidates,
                rejected=rejected,
                preserved=preserved,
                before_protected=before_protected,
                after_protected=after_protected,
                execute=True,
                phase="COMPLETE",
                started=started,
                free_before=free_before,
                free_after=free_bytes(root),
                actions=actions,
            )
        write_json_atomic(out, payload)
        print(json.dumps(compact_view(payload), indent=2, sort_keys=True))
        return 0 if payload["trigger_state"] == "GREEN" else 2

    if args.compact_superseded_checkpoints:
        candidates, rejected, preserved = discover_superseded_checkpoints(
            root,
            min_age_hours=max(0.0, float(args.min_age_hours)),
            tracked=tracked,
        )
        payload = build_checkpoint_payload(
            root=root,
            protected_root=protected_root,
            candidates=candidates,
            rejected=rejected,
            preserved=preserved,
            before_protected=before_protected,
            after_protected=before_protected,
            execute=bool(args.execute),
            phase="PREPARED" if args.execute else "DRY_RUN",
            started=started,
            free_before=free_before,
            free_after=free_before,
        )
        if args.execute:
            write_json_atomic(out, payload)
        actions = []
        if args.execute:
            for candidate in candidates:
                actions.append(remove_checkpoint_generation(root, candidate))
            after_protected = fingerprint_tree(protected_root)
            payload = build_checkpoint_payload(
                root=root,
                protected_root=protected_root,
                candidates=candidates,
                rejected=rejected,
                preserved=preserved,
                before_protected=before_protected,
                after_protected=after_protected,
                execute=True,
                phase="COMPLETE",
                started=started,
                free_before=free_before,
                free_after=free_bytes(root),
                actions=actions,
            )
        write_json_atomic(out, payload)
        print(json.dumps(compact_view(payload), indent=2, sort_keys=True))
        return 0 if payload["trigger_state"] == "GREEN" else 2

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


def load_runtime_canary_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    policy = payload.get("runtime_canary_retention")
    if not isinstance(policy, dict):
        raise ValueError("runtime_canary_retention policy missing")
    families = policy.get("retirable_families")
    if not isinstance(families, list) or not families:
        raise ValueError("retirable_families policy missing")
    return policy


def discover_unreferenced_canary_runs(
    root: Path,
    *,
    policy: dict[str, Any],
    min_age_hours: float,
    tracked: set[str],
    now_timestamp: float | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Classify complete closed runs against canonical and evidence references."""
    now_value = float(now_timestamp if now_timestamp is not None else time.time())
    authority_references = runtime_reference_rows(
        list(policy.get("authority_reference_sources", []))
    )
    evidence_references = runtime_reference_rows(
        list(policy.get("evidence_reference_sources", []))
    )
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    preserved: list[dict[str, Any]] = []
    for family_name in sorted(str(row) for row in policy["retirable_families"]):
        family = (root / family_name).resolve()
        require_descendant(root, family)
        if not family.is_dir() or family.is_symlink():
            rejected.append(
                {"path": relative(family), "reason": "retirable_family_missing_or_unsafe"}
            )
            continue
        for run in sorted(path for path in family.iterdir() if path.is_dir()):
            try:
                record = inspect_canary_run(
                    root,
                    run,
                    min_age_hours=min_age_hours,
                    now_timestamp=now_value,
                    tracked=tracked,
                    authority_references=authority_references,
                    evidence_references=evidence_references,
                )
            except ValueError as exc:
                rejected.append({"path": relative(run), "reason": str(exc)})
                continue
            if record["authority_references"]:
                preserved.append(
                    {
                        **record,
                        "classification": "canonical_reference_protected_canary_run",
                    }
                )
            else:
                candidates.append(record)
    return candidates, rejected, preserved


def inspect_canary_run(
    root: Path,
    run: Path,
    *,
    min_age_hours: float,
    now_timestamp: float,
    tracked: set[str],
    authority_references: list[dict[str, str]],
    evidence_references: list[dict[str, str]],
) -> dict[str, Any]:
    resolved = run.resolve()
    require_descendant(root, resolved)
    if run.is_symlink():
        raise ValueError("canary_run_is_symlink")
    tracked_descendants = sorted(
        path for path in tracked if path == relative(resolved) or path.startswith(relative(resolved) + "/")
    )
    if tracked_descendants:
        raise ValueError("canary_run_contains_git_tracked_files")
    symlinks = [path for path in resolved.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError("canary_run_contains_symlink")
    files = sorted(path for path in resolved.rglob("*") if path.is_file())
    if not files:
        raise ValueError("canary_run_empty")
    newest_mtime = max(path.stat().st_mtime for path in files)
    age_hours = max(0.0, (now_timestamp - newest_mtime) / 3600.0)
    if age_hours < min_age_hours:
        raise ValueError("canary_run_younger_than_minimum_age")

    checkpoint_directories = sorted(
        {
            path.parent
            for path in files
            if CHECKPOINT_PATTERN.match(path.name)
        }
    )
    if not checkpoint_directories:
        raise ValueError("canary_run_has_no_complete_checkpoint")
    terminal_rows = []
    for directory in checkpoint_directories:
        _, terminal = inspect_checkpoint_directory(
            root,
            directory,
            min_age_hours=0,
            now_timestamp=now_timestamp,
            tracked=tracked,
        )
        terminal_rows.append(terminal)

    identity = tree_identity(resolved, include_files=True)
    run_path = relative(resolved)
    canonical_refs = matching_runtime_references(run_path, authority_references)
    evidence_refs = matching_runtime_references(run_path, evidence_references)
    return {
        "record_type": "t0a_closed_canary_run_retention_candidate",
        "path": run_path,
        "family": resolved.parent.name,
        "age_hours": round(age_hours, 3),
        "classification": "unreferenced_closed_canary_payload",
        "recovery": (
            "rerun_the_exact_bounded_canary_from_preserved_source_config_seed_"
            "and_result_receipts"
        ),
        "authority_references": canonical_refs,
        "evidence_reference_files": sorted(
            {row["source"] for row in evidence_refs}
        ),
        "terminal_checkpoints": terminal_rows,
        **identity,
    }


def runtime_reference_rows(sources: list[str]) -> list[dict[str, str]]:
    if not sources:
        return []
    command = [
        "git",
        "grep",
        "-n",
        "-I",
        "-e",
        "runtime/t0a_canaries/",
        "--",
        *sources,
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise ValueError("runtime reference scan failed")
    rows: list[dict[str, str]] = []
    path_pattern = re.compile(
        r"runtime/t0a_canaries/[A-Za-z0-9_./{}-]+"
    )
    for line in result.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        source, line_number, text = parts
        for match in path_pattern.finditer(text):
            rows.append(
                {
                    "source": source,
                    "line": line_number,
                    "reference": match.group(0).rstrip("./"),
                }
            )
    return rows


def matching_runtime_references(
    run_path: str,
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    matches = []
    for row in rows:
        reference = row["reference"]
        pattern = re.sub(r"\{[^{}]+\}", "*", reference)
        protects_run = (
            reference == run_path
            or reference.startswith(run_path + "/")
            or fnmatch.fnmatchcase(run_path, pattern)
        )
        if protects_run:
            matches.append(row)
    return matches


def tree_identity(path: Path, *, include_files: bool) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    inode_counts: dict[tuple[int, int], int] = {}
    for item in files:
        stat = item.stat()
        key = (int(stat.st_dev), int(stat.st_ino))
        inode_counts[key] = inode_counts.get(key, 0) + 1
    manifest = hashlib.sha256()
    file_rows = []
    logical_bytes = 0
    allocated_total = 0
    reclaimable_total = 0
    seen_inodes: set[tuple[int, int]] = set()
    for item in files:
        stat = item.stat()
        key = (int(stat.st_dev), int(stat.st_ino))
        digest = sha256_file(item)
        relative_name = item.relative_to(path).as_posix()
        manifest.update(
            f"{relative_name}\0{stat.st_size}\0{digest}\n".encode("utf-8")
        )
        logical_bytes += int(stat.st_size)
        if key not in seen_inodes:
            allocated = allocated_bytes(stat)
            allocated_total += allocated
            if int(stat.st_nlink) <= inode_counts[key]:
                reclaimable_total += allocated
            seen_inodes.add(key)
        if include_files:
            file_rows.append(
                {
                    "path": relative_name,
                    "bytes": int(stat.st_size),
                    "sha256": digest,
                    "mtime_ns": int(stat.st_mtime_ns),
                    "inode": int(stat.st_ino),
                    "link_count": int(stat.st_nlink),
                }
            )
    return {
        "file_count": len(files),
        "logical_bytes": logical_bytes,
        "allocated_bytes": allocated_total,
        "reclaimable_allocated_bytes": reclaimable_total,
        "content_manifest_sha256": manifest.hexdigest(),
        "files": file_rows if include_files else None,
    }


def remove_canary_run(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    path = (ROOT / str(candidate["path"])).resolve()
    if not path.exists() and Path(str(candidate["path"])).is_absolute():
        path = Path(str(candidate["path"])).resolve()
    try:
        require_descendant(root, path)
        if path.is_symlink() or not path.is_dir():
            raise ValueError("canary_run_no_longer_safe_directory")
        current = tree_identity(path, include_files=False)
        if (
            int(current["file_count"]) != int(candidate["file_count"])
            or int(current["logical_bytes"]) != int(candidate["logical_bytes"])
            or current["content_manifest_sha256"]
            != candidate["content_manifest_sha256"]
        ):
            raise ValueError("canary_run_changed_after_manifest")
        shutil.rmtree(path)
        if path.exists():
            raise ValueError("canary_run_still_exists")
        return {**candidate, "status": "deleted", "deleted_utc": now()}
    except Exception as exc:
        return {**candidate, "status": "failed", "error": repr(exc)}


def discover_superseded_checkpoints(
    root: Path,
    *,
    min_age_hours: float,
    tracked: set[str],
    now_timestamp: float | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Find only non-terminal generations with exact terminal alias custody."""
    now_value = float(now_timestamp if now_timestamp is not None else time.time())
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    preserved: list[dict[str, Any]] = []
    if not root.is_dir():
        return candidates, [{"path": relative(root), "reason": "root_missing"}], preserved

    checkpoint_directories = sorted(
        {
            path.parent
            for path in root.rglob("*.safetensors")
            if CHECKPOINT_PATTERN.match(path.name)
        }
    )
    for directory in checkpoint_directories:
        try:
            rows, terminal_preserved = inspect_checkpoint_directory(
                root,
                directory,
                min_age_hours=min_age_hours,
                now_timestamp=now_value,
                tracked=tracked,
            )
        except ValueError as exc:
            rejected.append({"path": relative(directory), "reason": str(exc)})
            continue
        candidates.extend(rows)
        preserved.append(terminal_preserved)
    return candidates, rejected, preserved


def inspect_checkpoint_directory(
    root: Path,
    directory: Path,
    *,
    min_age_hours: float,
    now_timestamp: float,
    tracked: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved = directory.resolve()
    require_descendant(root, resolved)
    if directory.is_symlink():
        raise ValueError("checkpoint_directory_is_symlink")
    versioned: list[tuple[Path, re.Match[str]]] = []
    for path in sorted(resolved.iterdir()):
        if path.is_symlink():
            raise ValueError("checkpoint_directory_contains_symlink")
        match = CHECKPOINT_PATTERN.match(path.name)
        if match and path.is_file():
            versioned.append((path, match))
    if not versioned:
        raise ValueError("checkpoint_directory_has_no_versioned_generations")
    if any(relative(path) in tracked for path, _ in versioned):
        raise ValueError("checkpoint_generation_is_git_tracked")

    terminal_step = max(int(match.group("step")) for _, match in versioned)
    terminal_by_kind: dict[str, Path] = {}
    for path, match in versioned:
        if int(match.group("step")) != terminal_step:
            continue
        key = checkpoint_kind(match)
        if key in terminal_by_kind:
            raise ValueError("duplicate_terminal_checkpoint_kind")
        terminal_by_kind[key] = path
    required_kinds = {"weights", "optimizer", "optimizer_rng"}
    if set(terminal_by_kind) != required_kinds:
        raise ValueError("terminal_checkpoint_generation_incomplete")

    terminal_digests: dict[str, str] = {}
    aliases: dict[str, dict[str, Any]] = {}
    for kind in sorted(required_kinds):
        terminal = terminal_by_kind[kind]
        alias = resolved / ALIAS_BY_KIND[kind]
        if not alias.is_file() or alias.is_symlink():
            raise ValueError(f"terminal_alias_missing_or_unsafe:{kind}")
        terminal_digest = sha256_file(terminal)
        alias_digest = sha256_file(alias)
        if terminal_digest != alias_digest:
            raise ValueError(f"terminal_alias_digest_mismatch:{kind}")
        terminal_digests[kind] = terminal_digest
        aliases[kind] = file_identity(alias)

    candidates = []
    for path, match in versioned:
        step = int(match.group("step"))
        if step >= terminal_step:
            continue
        stat = path.stat()
        age_hours = max(0.0, (now_timestamp - stat.st_mtime) / 3600.0)
        if age_hours < min_age_hours:
            continue
        candidates.append(
            {
                "record_type": "t0a_superseded_checkpoint_generation",
                "path": relative(path),
                "checkpoint_directory": relative(resolved),
                "kind": checkpoint_kind(match),
                "step": step,
                "terminal_step": terminal_step,
                "logical_bytes": int(stat.st_size),
                "allocated_bytes": allocated_bytes(stat),
                "reclaimable_allocated_bytes": (
                    allocated_bytes(stat) if int(stat.st_nlink) == 1 else 0
                ),
                "sha256": sha256_file(path),
                "mtime_ns": int(stat.st_mtime_ns),
                "inode": int(stat.st_ino),
                "link_count": int(stat.st_nlink),
                "age_hours": round(age_hours, 3),
                "classification": (
                    "superseded_versioned_canary_checkpoint_generation"
                ),
                "recovery": (
                    "rerun_the_exact_bounded_canary_from_its_preserved_source_"
                    "checkpoint_config_seed_and_report"
                ),
            }
        )
    return candidates, {
        "checkpoint_directory": relative(resolved),
        "terminal_step": terminal_step,
        "terminal_generation": {
            kind: file_identity(path)
            for kind, path in sorted(terminal_by_kind.items())
        },
        "aliases": aliases,
        "terminal_alias_parity": True,
        "training_receipt": (
            file_identity(resolved / "training_receipt.json")
            if (resolved / "training_receipt.json").is_file()
            else None
        ),
        "merged_diagnostic_file_count": len(
            list(resolved.glob("*.merged-fp32.safetensors"))
        ),
        "candidate_count": len(candidates),
    }


def checkpoint_kind(match: re.Match[str]) -> str:
    if match.group("kind") == "weights":
        return "weights"
    return "optimizer_rng" if match.group("rng") else "optimizer"


def remove_checkpoint_generation(
    root: Path,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    path = (ROOT / str(candidate["path"])).resolve()
    if not path.exists() and Path(str(candidate["path"])).is_absolute():
        path = Path(str(candidate["path"])).resolve()
    try:
        require_descendant(root, path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("checkpoint_generation_no_longer_regular_file")
        match = CHECKPOINT_PATTERN.match(path.name)
        if not match:
            raise ValueError("checkpoint_generation_no_longer_allowlisted")
        if int(match.group("step")) != int(candidate["step"]):
            raise ValueError("checkpoint_generation_step_changed")
        stat = path.stat()
        if (
            int(stat.st_size) != int(candidate["logical_bytes"])
            or int(stat.st_mtime_ns) != int(candidate["mtime_ns"])
            or int(stat.st_ino) != int(candidate["inode"])
            or sha256_file(path) != candidate["sha256"]
        ):
            raise ValueError("checkpoint_generation_changed_after_manifest")
        path.unlink()
        if path.exists():
            raise ValueError("checkpoint_generation_still_exists")
        return {**candidate, "status": "deleted", "deleted_utc": now()}
    except Exception as exc:
        return {**candidate, "status": "failed", "error": repr(exc)}


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


def build_run_retirement_payload(
    *,
    root: Path,
    protected_root: Path,
    policy_path: Path,
    candidates: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    preserved: list[dict[str, Any]],
    before_protected: dict[str, Any],
    after_protected: dict[str, Any],
    execute: bool,
    phase: str,
    started: float,
    free_before: int,
    free_after: int,
    actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    action_rows = actions or [
        {**candidate, "status": "prepared" if execute else "dry_run"}
        for candidate in candidates
    ]
    failures = [row for row in action_rows if row.get("status") == "failed"]
    deleted = [row for row in action_rows if row.get("status") == "deleted"]
    protected_unchanged = before_protected == after_protected
    hard_gaps: list[dict[str, Any]] = []
    if phase == "COMPLETE" and not protected_unchanged:
        hard_gaps.append({"gap": "protected_production_tree_changed"})
    return {
        "policy": "project_theseus_t0a_storage_retention_v3",
        "policy_artifact": {
            "path": relative(policy_path),
            "sha256": sha256_file(policy_path),
        },
        "created_utc": now(),
        "trigger_state": "RED" if failures or hard_gaps else "GREEN",
        "phase": phase,
        "mode": "retire_unreferenced_closed_canary_runs",
        "scope": {
            "root": relative(root),
            "protected_root": relative(protected_root),
            "authority_reference_boundary": [
                "configs",
                "docs/PROJECT_STATE.md",
                "roadmap.md",
            ],
            "evidence_boundary": "reports are always preserved",
            "terminal_alias_parity_required": True,
        },
        "summary": {
            "execute": execute,
            "candidate_run_count": len(candidates),
            "canonical_reference_protected_run_count": len(preserved),
            "rejected_run_count": len(rejected),
            "candidate_file_count": sum(int(row["file_count"]) for row in candidates),
            "candidate_logical_bytes": sum(
                int(row["logical_bytes"]) for row in candidates
            ),
            "candidate_reclaimable_allocated_bytes": sum(
                int(row["reclaimable_allocated_bytes"]) for row in candidates
            ),
            "candidate_reclaimable_allocated_gib": round(
                sum(
                    int(row["reclaimable_allocated_bytes"])
                    for row in candidates
                )
                / (1024**3),
                3,
            ),
            "deleted_run_count": len(deleted),
            "deleted_file_count": sum(int(row["file_count"]) for row in deleted),
            "deleted_reclaimable_allocated_bytes": sum(
                int(row["reclaimable_allocated_bytes"]) for row in deleted
            ),
            "failed_count": len(failures),
            "protected_tree_unchanged": protected_unchanged,
            "filesystem_free_bytes_before": int(free_before),
            "filesystem_free_bytes_after": int(free_after),
            "filesystem_free_bytes_delta": int(free_after - free_before),
            "runtime_seconds": round(time.perf_counter() - started, 3),
        },
        "classification": {
            "deleted": (
                "complete terminal-parity canary runs with no protecting "
                "canonical config, roadmap, or project-state reference"
            ),
            "preserved": [
                "all canonical-reference-protected canary runs",
                "all runs that fail retirement eligibility for any reason",
                "terminal checkpoints in every retained run",
                "all reports, receipts outside retired scratch roots, and negative evidence",
                "the complete active step-11,416 production checkpoint tree",
            ],
            "scientific_disposition": (
                "Storage retirement only. Reports and candidate dispositions "
                "remain authoritative; no scientific result is rewritten."
            ),
        },
        "protected_tree_before": before_protected,
        "protected_tree_after": after_protected,
        "canonical_reference_protected_runs": preserved,
        "actions": action_rows,
        "rejected": rejected,
        "hard_gaps": hard_gaps,
        **NO_CHEAT,
    }


def build_checkpoint_payload(
    *,
    root: Path,
    protected_root: Path,
    candidates: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    preserved: list[dict[str, Any]],
    before_protected: dict[str, Any],
    after_protected: dict[str, Any],
    execute: bool,
    phase: str,
    started: float,
    free_before: int,
    free_after: int,
    actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    action_rows = actions or [
        {**candidate, "status": "prepared" if execute else "dry_run"}
        for candidate in candidates
    ]
    failures = [row for row in action_rows if row.get("status") == "failed"]
    deleted = [row for row in action_rows if row.get("status") == "deleted"]
    protected_unchanged = before_protected == after_protected
    hard_gaps = [{"gap": "checkpoint_directory_rejected", **row} for row in rejected]
    if phase == "COMPLETE" and not protected_unchanged:
        hard_gaps.append({"gap": "protected_production_tree_changed"})
    return {
        "policy": "project_theseus_t0a_storage_retention_v2",
        "created_utc": now(),
        "trigger_state": "RED" if failures or hard_gaps else "GREEN",
        "phase": phase,
        "mode": "compact_superseded_checkpoint_generations",
        "scope": {
            "root": relative(root),
            "protected_root": relative(protected_root),
            "allowlisted_pattern": CHECKPOINT_PATTERN.pattern,
            "terminal_alias_parity_required": True,
            "minimum_age_hours": min(
                (float(row["age_hours"]) for row in candidates),
                default=None,
            ),
        },
        "summary": {
            "execute": execute,
            "checkpoint_directory_count": len(preserved),
            "candidate_file_count": len(candidates),
            "candidate_logical_bytes": sum(
                int(row["logical_bytes"]) for row in candidates
            ),
            "candidate_allocated_bytes": sum(
                int(row["allocated_bytes"]) for row in candidates
            ),
            "candidate_reclaimable_allocated_bytes": sum(
                int(row["reclaimable_allocated_bytes"]) for row in candidates
            ),
            "candidate_reclaimable_allocated_gib": round(
                sum(
                    int(row["reclaimable_allocated_bytes"])
                    for row in candidates
                )
                / (1024**3),
                3,
            ),
            "deleted_file_count": len(deleted),
            "deleted_reclaimable_allocated_bytes": sum(
                int(row["reclaimable_allocated_bytes"]) for row in deleted
            ),
            "failed_count": len(failures),
            "rejected_directory_count": len(rejected),
            "protected_tree_unchanged": protected_unchanged,
            "filesystem_free_bytes_before": int(free_before),
            "filesystem_free_bytes_after": int(free_after),
            "filesystem_free_bytes_delta": int(free_after - free_before),
            "runtime_seconds": round(time.perf_counter() - started, 3),
        },
        "classification": {
            "deleted": (
                "non-terminal versioned canary model/optimizer/RNG generations "
                "whose terminal aliases passed exact digest parity"
            ),
            "preserved": [
                "every terminal versioned checkpoint generation",
                "every exact unversioned terminal alias",
                "all training receipts and heartbeats",
                "all merged-fp32 diagnostic snapshots",
                "all reports and meaningful negative evidence",
                "the complete active step-11,416 production checkpoint tree",
            ],
            "scientific_disposition": (
                "Storage compaction only. It does not change a model result, "
                "architecture disposition, or capability claim."
            ),
        },
        "protected_tree_before": before_protected,
        "protected_tree_after": after_protected,
        "preserved_terminal_checkpoints": preserved,
        "actions": action_rows,
        "rejected": rejected,
        "hard_gaps": hard_gaps,
        **NO_CHEAT,
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


def allocated_bytes(stat: os.stat_result) -> int:
    return int(getattr(stat, "st_blocks", 0)) * 512


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": relative(path),
        "bytes": int(stat.st_size),
        "allocated_bytes": allocated_bytes(stat),
        "sha256": sha256_file(path),
        "inode": int(stat.st_ino),
        "link_count": int(stat.st_nlink),
    }


def free_bytes(path: Path) -> int:
    stat = os.statvfs(path)
    return int(stat.f_bavail) * int(stat.f_frsize)


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
