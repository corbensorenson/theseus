#!/usr/bin/env python3
"""Fork-safe chronology transport for the otherwise unchanged VCM v3 selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_vcm_source_acquisition as v1
import theseus_vcm_source_acquisition_v3 as v3


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_source_acquisition_v4.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_vcm_source_acquisition_v4.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = preflight(config_path)
    if args.execute and report["trigger_state"] == "GREEN":
        try:
            report = acquire(config_path)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
            report = {
                **report,
                "trigger_state": "PAUSED",
                "state": "PUBLIC_METADATA_TRANSPORT_PAUSED_NO_SOURCE_EXPOSURE",
                "faults": [f"{type(exc).__name__}:{exc}"[:4000]],
            }
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(v1.summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    report = v3.preflight(config_path)
    if report.get("trigger_state") == "GREEN":
        report["state"] = "METADATA_SELECTION_V4_FORK_SAFE_PREFLIGHT_GREEN"
    return report


def acquire(config_path: Path) -> dict[str, Any]:
    original = v1.qualify_metadata
    v1.qualify_metadata = qualify_metadata
    try:
        return v3.acquire(config_path)
    finally:
        v1.qualify_metadata = original


def qualify_metadata(candidate: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], list[str], int, list[str]]:
    """Resolve head chronology through the PR commit list, including fork PRs."""
    repository = candidate["repository"]
    number = candidate["pull_request"]
    digests: list[str] = []
    pr, digest = v1.api_json(f"repos/{repository}/pulls/{number}", {})
    digests.append(digest)
    repo, digest = v1.api_json(f"repos/{repository}", {})
    digests.append(digest)
    files, digest = v1.api_json(
        f"repos/{repository}/pulls/{number}/files", {"per_page": 100}
    )
    digests.append(digest)
    commit_count = max(1, int(p2a.mapping(pr).get("commits") or 0))
    commit_page = (commit_count - 1) // 100 + 1
    commits, digest = v1.api_json(
        f"repos/{repository}/pulls/{number}/commits",
        {"per_page": 100, "page": commit_page},
    )
    digests.append(digest)
    commit_rows = p2a.dicts(commits)
    head_commit = commit_rows[-1] if commit_rows else {}
    reasons = v1.metadata_rejection_reasons(
        candidate, pr, repo, p2a.dicts(files), head_commit, config
    )
    if not commit_rows:
        reasons = sorted(set([*reasons, "head_commit_metadata_unavailable"]))
    head_sha = str(p2a.mapping(pr).get("head", {}).get("sha") or "")
    row = {
        "opaque_source_id": hashlib.sha256(
            f"vcm-source:{repository}#{number}".encode()
        ).hexdigest(),
        "repository": repository,
        "pull_request": number,
        "pull_request_url": pr.get("html_url"),
        "query_language": candidate["query_language"],
        "title_sha256": hashlib.sha256(str(pr.get("title") or "").encode()).hexdigest(),
        "created_utc": pr.get("created_at"),
        "merged_utc": pr.get("merged_at"),
        "base_revision": p2a.mapping(pr.get("base")).get("sha"),
        "head_revision": head_sha,
        "merge_revision": pr.get("merge_commit_sha"),
        "head_commit_utc": p2a.mapping(
            p2a.mapping(head_commit.get("commit")).get("committer")
        ).get("date"),
        "head_chronology_source": "base_pull_request_commit_list",
        "license_spdx": p2a.mapping(repo.get("license")).get("spdx_id"),
        "repository_stars": repo.get("stargazers_count"),
        "changed_file_count": pr.get("changed_files"),
        "changed_line_count": int(pr.get("additions") or 0) + int(pr.get("deletions") or 0),
        "source_paths": sorted(
            path for path in v1.file_paths(p2a.dicts(files))
            if v1.is_source(path, candidate["query_language"]) and not v1.is_test(path)
        ),
        "verifier_paths": sorted(
            path for path in v1.file_paths(p2a.dicts(files)) if v1.is_test(path)
        ),
        "selection_rank_sha256": candidate["rank"],
        "metadata_qualified": not reasons,
        "candidate_content_retrieved": False,
        "candidate_packet_materialized": False,
    }
    return row, reasons, 4, digests


if __name__ == "__main__":
    raise SystemExit(main())
