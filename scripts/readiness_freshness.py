#!/usr/bin/env python3
"""Shared freshness checks for Theseus readiness gates."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DECODER_SOURCE_ROOTS = [
    ROOT / "crates" / "symliquid-cli" / "src" / "main.rs",
    ROOT / "crates" / "symliquid-cli" / "src" / "code_lm_closure",
    ROOT / "scripts" / "code_lm_closure.py",
    ROOT / "scripts" / "code_lm_private_verifier.py",
]


def release_binary_path(root: Path = ROOT) -> Path:
    name = "symliquid-cli.exe" if sys.platform.startswith("win") else "symliquid-cli"
    return root / "target" / "release" / name


def decoder_source_paths(source_roots: list[Path] | None = None) -> list[Path]:
    paths: list[Path] = []
    for root in source_roots or DECODER_SOURCE_ROOTS:
        if root.is_file():
            paths.append(root)
        elif root.is_dir():
            paths.extend(path for path in root.rglob("*.rs") if path.is_file())
    return paths


def path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def rel_or_abs(path: Path, root: Path = ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def freshness_report(
    artifacts: dict[str, Path],
    *,
    root: Path = ROOT,
    source_roots: list[Path] | None = None,
    release_binary: Path | None = None,
    rule: str = "artifacts must be regenerated after decoder source changes or release binary rebuilds",
) -> dict[str, Any]:
    release = release_binary or release_binary_path(root)
    source_paths = decoder_source_paths(source_roots)
    newest_source_mtime = max((path_mtime(path) for path in source_paths), default=0.0)
    release_mtime = path_mtime(release)
    required_mtime = max(newest_source_mtime, release_mtime)
    artifact_rows = []
    stale_reasons: list[str] = []

    if not release.exists():
        stale_reasons.append("release_binary_missing")
    if not source_paths:
        stale_reasons.append("decoder_source_paths_missing")

    for name, path in artifacts.items():
        exists = path.exists()
        mtime = path_mtime(path)
        reasons: list[str] = []
        if not exists:
            reasons.append("artifact_missing")
        if required_mtime and mtime and mtime < required_mtime:
            reasons.append("artifact_older_than_decoder_source_or_release")
        if required_mtime and not mtime:
            reasons.append("artifact_mtime_missing")
        if reasons:
            stale_reasons.extend(f"{name}:{reason}" for reason in reasons)
        artifact_rows.append(
            {
                "name": name,
                "path": rel_or_abs(path, root),
                "exists": exists,
                "mtime": mtime or None,
                "stale_reasons": reasons,
            }
        )

    return {
        "fresh": bool(release.exists() and source_paths and artifact_rows and not stale_reasons),
        "required_mtime": required_mtime or None,
        "newest_source_mtime": newest_source_mtime or None,
        "release_binary": rel_or_abs(release, root),
        "release_binary_exists": release.exists(),
        "release_binary_mtime": release_mtime or None,
        "source_count": len(source_paths),
        "newest_sources": [
            {"path": rel_or_abs(path, root), "mtime": path_mtime(path)}
            for path in sorted(source_paths, key=path_mtime, reverse=True)[:8]
        ],
        "artifacts": artifact_rows,
        "stale_reasons": sorted(set(stale_reasons)),
        "rule": rule,
    }
