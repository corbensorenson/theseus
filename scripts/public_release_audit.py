#!/usr/bin/env python3
"""Audit the tracked repo surface before making Project Theseus public."""

from __future__ import annotations

import argparse
import json
import os
import re
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
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----")),
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
    args = parser.parse_args()

    config_path = resolve(args.config)
    out_path = resolve(args.out)
    config = read_json(config_path)
    tracked = git_tracked_files()
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    forbidden_prefixes = tuple(str(p) for p in config.get("forbidden_tracked_prefixes", []))
    forbidden_paths = set(str(p) for p in config.get("forbidden_tracked_paths", []))
    forbidden_suffixes = tuple(str(p) for p in config.get("forbidden_suffixes", []))
    allowed_large_prefixes = tuple(str(p) for p in config.get("allowed_large_file_prefixes", []))
    allowed_secret_prefixes = tuple(str(p) for p in config.get("allowed_secret_literal_prefixes", []))
    large_limit = int(config.get("large_file_soft_limit_bytes", 5 * 1024 * 1024))

    for rel_path in tracked:
        if rel_path in forbidden_paths or rel_path.startswith(forbidden_prefixes):
            hard_gaps.append({"kind": "forbidden_tracked_path", "path": rel_path})
            continue
        if rel_path.endswith(forbidden_suffixes):
            hard_gaps.append({"kind": "forbidden_tracked_suffix", "path": rel_path})

        full_path = ROOT / rel_path
        if full_path.is_file():
            size = full_path.stat().st_size
            if size > large_limit and not rel_path.startswith(allowed_large_prefixes):
                warnings.append({"kind": "large_tracked_file", "path": rel_path, "bytes": size})
            if is_text_candidate(full_path):
                secret_hits = scan_secret_literals(full_path)
                for kind, line in secret_hits:
                    severity = "warning" if rel_path.startswith(allowed_secret_prefixes) else "hard_gap"
                    row = {"kind": "secret_literal_match", "secret_kind": kind, "path": rel_path, "line": line}
                    if severity == "hard_gap":
                        hard_gaps.append(row)
                    else:
                        warnings.append(row)

    visibility = gh_repo_visibility()
    summary = {
        "tracked_file_count": len(tracked),
        "forbidden_tracked_path_count": sum(1 for gap in hard_gaps if gap["kind"].startswith("forbidden")),
        "secret_literal_hard_gap_count": sum(1 for gap in hard_gaps if gap["kind"] == "secret_literal_match"),
        "large_file_warning_count": sum(1 for warning in warnings if warning["kind"] == "large_tracked_file"),
        "hard_gap_count": len(hard_gaps),
        "warning_count": len(warnings),
        "github_visibility": visibility.get("visibility", "UNKNOWN"),
        "github_repository": visibility.get("nameWithOwner", ""),
    }
    report = {
        "policy": str(config.get("policy", "project_theseus_public_release_manifest_v1")),
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
    write_json(out_path, report)
    print(json.dumps({"trigger_state": report["trigger_state"], "summary": summary}, indent=2, sort_keys=True))
    if args.gate and hard_gaps:
        return 1
    return 0


def git_tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [item for item in result.stdout.decode("utf-8", errors="replace").split("\0") if item]


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
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".ico", ".ttf", ".woff", ".eot", ".pdf", ".docx"}:
        return False
    try:
        with path.open("rb") as handle:
            sample = handle.read(2048)
    except OSError:
        return False
    return b"\0" not in sample


def gh_repo_visibility() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["gh", "repo", "view", "--json", "nameWithOwner,visibility,isPrivate,url"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        data = json.loads(result.stdout)
        return data if isinstance(data, dict) else {}
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
