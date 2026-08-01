#!/usr/bin/env python3
"""Audit the tracked repo surface before making Project Theseus public."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "public_release_manifest.json"
DEFAULT_OUT = ROOT / "reports" / "public_release_audit.json"

SECRET_PATTERNS = [
    ("github_token", re.compile(r"\bgh[opsu]_[A-Za-z0-9_]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")),
    (
        "private_key_block",
        re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    ),
    (
        "literal_secret_json",
        re.compile(
            r'"(?:join_token|api[_-]?key|password|private_key|secret|token)"\s*:\s*"[^"\n]{16,}"',
            re.IGNORECASE,
        ),
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--gate", action="store_true")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Run tracked-file and secret checks without requiring GitHub visibility.",
    )
    parser.add_argument(
        "--scope-prefix",
        action="append",
        default=[],
        help="Restrict the audit to tracked paths under one or more prefixes.",
    )
    parser.add_argument(
        "--source-release",
        action="store_true",
        help="Audit the explicit source-only release surface, including untracked non-ignored files.",
    )
    parser.add_argument(
        "--prepare-source-tree",
        help="Copy the audited source-only surface to a new directory outside the repository.",
    )
    args = parser.parse_args()

    config_path = resolve(args.config)
    out_path = resolve(args.out)
    config = read_json(config_path)
    source_release_mode = bool(args.source_release or args.prepare_source_tree)
    tracked = (
        git_source_controlled_files() if source_release_mode else git_tracked_files()
    )
    source_release_config = mapping(config.get("source_release"))
    source_release_excluded: list[str] = []
    source_release_dirty_paths: list[str] = []
    if source_release_mode:
        tracked, source_release_excluded = select_source_release_paths(
            tracked,
            config,
        )
        selected_path_set = set(tracked)
        source_release_dirty_paths = [
            path for path in git_dirty_paths() if path in selected_path_set
        ]
    if args.scope_prefix:
        prefixes = tuple(str(prefix) for prefix in args.scope_prefix)
        scoped = {rel_path for rel_path in tracked if rel_path.startswith(prefixes)}
        for prefix in prefixes:
            candidate = ROOT / prefix
            if candidate.is_file():
                scoped.add(candidate.relative_to(ROOT).as_posix())
            elif candidate.is_dir():
                scoped.update(
                    path.relative_to(ROOT).as_posix()
                    for path in candidate.rglob("*")
                    if path.is_file()
                )
        tracked = sorted(scoped)
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    forbidden_prefixes = tuple(
        str(p) for p in config.get("forbidden_tracked_prefixes", [])
    )
    forbidden_paths = set(str(p) for p in config.get("forbidden_tracked_paths", []))
    forbidden_suffixes = tuple(str(p) for p in config.get("forbidden_suffixes", []))
    allowed_large_prefixes = tuple(
        str(p) for p in config.get("allowed_large_file_prefixes", [])
    )
    allowed_secret_prefixes = tuple(
        str(p) for p in config.get("allowed_secret_literal_prefixes", [])
    )
    tracked_root_allowlist = set(
        str(p) for p in config.get("tracked_root_allowlist", [])
    )
    large_limit = int(config.get("large_file_soft_limit_bytes", 5 * 1024 * 1024))
    total_bytes = 0
    tracked_roots = sorted(
        {rel_path.split("/", 1)[0] for rel_path in tracked if rel_path}
    )

    if tracked_root_allowlist:
        for root_entry in unregistered_tracked_roots(
            tracked_roots,
            tracked_root_allowlist,
            forbidden_prefixes,
        ):
            warnings.append(
                {"kind": "unregistered_tracked_root_entry", "path": root_entry}
            )

    for rel_path in tracked:
        if rel_path in forbidden_paths or rel_path.startswith(forbidden_prefixes):
            hard_gaps.append({"kind": "forbidden_tracked_path", "path": rel_path})
            continue
        if rel_path.endswith(forbidden_suffixes):
            hard_gaps.append({"kind": "forbidden_tracked_suffix", "path": rel_path})

        full_path = ROOT / rel_path
        if full_path.is_file():
            size = full_path.stat().st_size
            total_bytes += size
            if size > large_limit and not rel_path.startswith(allowed_large_prefixes):
                warnings.append(
                    {"kind": "large_tracked_file", "path": rel_path, "bytes": size}
                )
            if is_text_candidate(full_path):
                secret_hits = scan_secret_literals(full_path)
                for kind, line in secret_hits:
                    severity = (
                        "warning"
                        if not source_release_mode
                        and rel_path.startswith(allowed_secret_prefixes)
                        else "hard_gap"
                    )
                    row = {
                        "kind": "secret_literal_match",
                        "secret_kind": kind,
                        "path": rel_path,
                        "line": line,
                    }
                    if severity == "hard_gap":
                        hard_gaps.append(row)
                    else:
                        warnings.append(row)

    if source_release_mode:
        source_file_limit = int(
            source_release_config.get("maximum_file_bytes", large_limit)
        )
        source_total_limit = int(
            source_release_config.get("maximum_total_bytes", 128 * 1024 * 1024)
        )
        for rel_path in tracked:
            full_path = ROOT / rel_path
            if full_path.is_file() and full_path.stat().st_size > source_file_limit:
                hard_gaps.append(
                    {
                        "kind": "source_release_file_too_large",
                        "path": rel_path,
                        "bytes": full_path.stat().st_size,
                        "maximum_bytes": source_file_limit,
                    }
                )
        if total_bytes > source_total_limit:
            hard_gaps.append(
                {
                    "kind": "source_release_total_too_large",
                    "bytes": total_bytes,
                    "maximum_bytes": source_total_limit,
                }
            )
        for required_path in source_release_config.get("required_paths", []):
            required = str(required_path)
            if required not in tracked:
                hard_gaps.append(
                    {"kind": "source_release_required_path_missing", "path": required}
                )
        if source_release_dirty_paths:
            hard_gaps.append(
                {
                    "kind": "source_release_dirty_worktree",
                    "path_count": len(source_release_dirty_paths),
                    "paths": source_release_dirty_paths[:100],
                    "truncated": max(0, len(source_release_dirty_paths) - 100),
                }
            )

    visibility = gh_repo_visibility()
    visibility_state = str(visibility.get("visibility", "UNKNOWN")).upper()
    if (
        not args.local_only
        and bool(config.get("require_public_visibility", False))
        and visibility_state != "PUBLIC"
    ):
        hard_gaps.append(
            {
                "kind": "github_visibility_not_public",
                "visibility": visibility.get("visibility", "UNKNOWN"),
                "repository": visibility.get("nameWithOwner", ""),
            }
        )
    summary = {
        "tracked_file_count": len(tracked),
        "forbidden_tracked_path_count": sum(
            1 for gap in hard_gaps if gap["kind"].startswith("forbidden")
        ),
        "secret_literal_hard_gap_count": sum(
            1 for gap in hard_gaps if gap["kind"] == "secret_literal_match"
        ),
        "large_file_warning_count": sum(
            1 for warning in warnings if warning["kind"] == "large_tracked_file"
        ),
        "tracked_root_entry_count": len(tracked_roots),
        "unregistered_tracked_root_warning_count": sum(
            1
            for warning in warnings
            if warning["kind"] == "unregistered_tracked_root_entry"
        ),
        "hard_gap_count": len(hard_gaps),
        "warning_count": len(warnings),
        "github_visibility": visibility.get("visibility", "UNKNOWN"),
        "github_repository": visibility.get("nameWithOwner", ""),
        "remote_visibility_checked": not args.local_only,
        "audit_mode": "source_release" if source_release_mode else "tracked_repository",
        "source_release_file_count": len(tracked) if source_release_mode else 0,
        "source_release_bytes": total_bytes if source_release_mode else 0,
        "source_release_excluded_count": (
            len(source_release_excluded) if source_release_mode else 0
        ),
    }
    report = {
        "policy": str(
            config.get("policy", "project_theseus_public_release_manifest_v1")
        ),
        "generated_utc": now(),
        "trigger_state": "GREEN" if not hard_gaps else "RED",
        "summary": summary,
        "hard_gaps": hard_gaps[:500],
        "warnings": warnings[:500],
        "truncated": {
            "hard_gaps": max(0, len(hard_gaps) - 500),
            "warnings": max(0, len(warnings) - 500),
        },
    }
    if source_release_mode:
        report["source_release"] = {
            "policy": str(
                source_release_config.get(
                    "policy",
                    "project_theseus_source_only_release_v1",
                )
            ),
            "source_commit": git_head(),
            "source_dirty": bool(source_release_dirty_paths),
            "source_dirty_paths": source_release_dirty_paths,
            "selected_paths": tracked,
            "excluded_paths": source_release_excluded,
        }
    if args.prepare_source_tree:
        destination = Path(args.prepare_source_tree).expanduser().resolve()
        prepared = prepare_source_tree(
            destination,
            tracked,
            report,
        )
        report["source_release"]["prepared_tree"] = prepared
    write_json(out_path, report)
    print(
        json.dumps(
            {"trigger_state": report["trigger_state"], "summary": summary},
            indent=2,
            sort_keys=True,
        )
    )
    if args.gate and hard_gaps:
        return 1
    return 0


def unregistered_tracked_roots(
    tracked_roots: list[str],
    allowlist: set[str],
    forbidden_prefixes: tuple[str, ...],
) -> list[str]:
    unregistered = []
    for root_entry in tracked_roots:
        whole_root_prefix = root_entry.rstrip("/") + "/"
        root_is_forbidden = whole_root_prefix in forbidden_prefixes
        if root_entry not in allowlist and not root_is_forbidden:
            unregistered.append(root_entry)
    return unregistered


def git_tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [
        item
        for item in result.stdout.decode("utf-8", errors="replace").split("\0")
        if item
    ]


def git_source_controlled_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return sorted(
        {
            item
            for item in result.stdout.decode("utf-8", errors="replace").split("\0")
            if item
        }
    )


def git_dirty_paths() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = result.stdout.decode("utf-8", errors="replace").split("\0")
    dirty: list[str] = []
    for row in rows:
        if not row:
            continue
        path = row[3:] if len(row) >= 4 else row
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        dirty.append(path)
    return sorted(set(dirty))


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def select_source_release_paths(
    paths: list[str],
    config: dict[str, Any],
) -> tuple[list[str], list[str]]:
    release = mapping(config.get("source_release"))
    include_paths = {str(path) for path in release.get("include_paths", [])}
    include_prefixes = tuple(str(path) for path in release.get("include_prefixes", []))
    exclude_paths = {str(path) for path in config.get("forbidden_tracked_paths", [])}
    exclude_paths.update(str(path) for path in release.get("exclude_paths", []))
    exclude_prefixes = tuple(
        [
            *(str(path) for path in config.get("forbidden_tracked_prefixes", [])),
            *(str(path) for path in release.get("exclude_prefixes", [])),
        ]
    )
    forbidden_suffixes = tuple(
        str(path) for path in config.get("forbidden_suffixes", [])
    )
    source_release_suffixes = tuple(
        str(path) for path in release.get("exclude_suffixes", [])
    )
    selected: list[str] = []
    excluded: list[str] = []
    for rel_path in sorted(set(paths)):
        included = rel_path in include_paths or rel_path.startswith(include_prefixes)
        forbidden = (
            rel_path in exclude_paths
            or rel_path.startswith(exclude_prefixes)
            or rel_path.endswith(forbidden_suffixes)
            or rel_path.endswith(source_release_suffixes)
        )
        if included and not forbidden:
            selected.append(rel_path)
        else:
            excluded.append(rel_path)
    return selected, excluded


def prepare_source_tree(
    destination: Path,
    paths: list[str],
    report: dict[str, Any],
) -> dict[str, Any]:
    root = ROOT.resolve()
    if destination == root or root in destination.parents:
        raise ValueError("source release destination must be outside the repository")
    if destination.exists():
        raise FileExistsError(
            f"source release destination already exists: {destination}"
        )
    destination.mkdir(parents=True)
    inventory: list[dict[str, Any]] = []
    for rel_path in paths:
        source = ROOT / rel_path
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"source release accepts regular files only: {rel_path}")
        target = destination / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        inventory.append(
            {
                "path": rel_path,
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    manifest = {
        "policy": report["source_release"]["policy"],
        "generated_utc": now(),
        "source_commit": report["source_release"]["source_commit"],
        "source_dirty": report["source_release"]["source_dirty"],
        "publishable": report["trigger_state"] == "GREEN",
        "file_count": len(inventory),
        "total_bytes": sum(int(row["bytes"]) for row in inventory),
        "files": inventory,
    }
    manifest_path = destination / "SOURCE_RELEASE_MANIFEST.json"
    write_json(manifest_path, manifest)
    return {
        "path": str(destination),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "file_count": len(inventory),
        "total_bytes": manifest["total_bytes"],
        "publishable": manifest["publishable"],
    }


def scan_secret_literals(path: Path) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle, start=1):
                for kind, pattern in SECRET_PATTERNS:
                    if pattern.search(line):
                        hits.append((kind, index))
                        break
                if len(hits) >= 20:
                    break
    except OSError:
        return hits
    return hits


def is_text_candidate(path: Path) -> bool:
    if path.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".ttf",
        ".woff",
        ".eot",
        ".pdf",
        ".docx",
    }:
        return False
    try:
        with path.open("rb") as handle:
            sample = handle.read(2048)
    except OSError:
        return False
    return b"\0" not in sample


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gh_repo_visibility() -> dict[str, Any]:
    repository = github_repository_from_remote()
    errors: list[str] = []
    command = ["gh", "repo", "view"]
    if repository:
        command.append(repository)
    command.extend(["--json", "nameWithOwner,visibility,isPrivate,url"])
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        data = json.loads(result.stdout)
        if isinstance(data, dict):
            data["source"] = "gh_repo_view"
            return data
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        errors.append(type(exc).__name__)

    if repository:
        try:
            result = subprocess.run(
                ["gh", "api", f"repos/{repository}"],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            data = json.loads(result.stdout)
            if isinstance(data, dict):
                private = bool(data.get("private"))
                return {
                    "nameWithOwner": str(data.get("full_name") or repository),
                    "visibility": str(
                        data.get("visibility") or ("private" if private else "public")
                    ).upper(),
                    "isPrivate": private,
                    "url": str(data.get("html_url") or ""),
                    "source": "github_rest_repository",
                }
        except (
            OSError,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
        ) as exc:
            errors.append(type(exc).__name__)
    return {
        "nameWithOwner": repository,
        "visibility": "UNKNOWN",
        "source": "unavailable",
        "errors": errors,
    }


def github_repository_from_remote() -> str:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return github_repository_from_url(result.stdout.strip())


def github_repository_from_url(remote: str) -> str:
    value = remote.strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    for marker in ("github.com/", "github.com:"):
        if marker in value:
            candidate = value.split(marker, 1)[1]
            parts = candidate.split("/")
            if (
                len(parts) == 2
                and all(parts)
                and all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts)
            ):
                return "/".join(parts)
    return ""


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
