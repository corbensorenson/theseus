#!/usr/bin/env python3
"""Deterministic local repository worker for the D1 evidence campaign.

The worker is intentionally generic and weak.  It receives one parent snapshot
and the frozen candidate-visible projection, performs bounded lexical
retrieval, and emits a patch-shaped response.  It has no git access, target
identity, evaluator labels, network path, teacher path, or learned credit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ALLOWED_SUFFIXES = {".py", ".json", ".md", ".toml", ".yml", ".yaml", ".rs", ".js", ".ts", ".qmd"}
ALLOWED_TOP_LEVEL = {"configs", "scripts", "tests", "docs", "crates", "examples"}
STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "for", "from", "in", "into", "of",
    "on", "or", "the", "to", "with", "theseus",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    snapshot_root = Path(args.snapshot_root).resolve()
    out_path = Path(args.out).resolve()
    visible = read_json(input_path)
    result = run_worker(visible, snapshot_root)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def run_worker(visible: dict[str, Any], snapshot_root: Path) -> dict[str, Any]:
    if (snapshot_root / ".git").exists():
        raise ValueError("worker snapshot may not contain git metadata")
    expected = {"natural_request", "parent_source_commit", "allowed_runtime_context", "authority_grant"}
    if set(visible) != expected:
        raise ValueError(f"worker input fields must be exactly {sorted(expected)}")
    request = str(visible["natural_request"])
    tokens = keywords(request)
    ranked = rank_files(snapshot_root, tokens)
    proposed_paths = [row["path"] for row in ranked[:12]]
    verification_commands = verification_plan(proposed_paths)
    return {
        "policy": "project_theseus_local_repository_worker_v1",
        "worker_id": "theseus_local_repository_worker_v1",
        "worker_kind": "deterministic_local_repository_retrieval_and_patch_planning_control",
        "learned_generation_credit": 0,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "public_calibration_cases_consumed": 0,
        "D2_cases_consumed": 0,
        "natural_request_sha256": sha256_text(request),
        "parent_source_commit": str(visible["parent_source_commit"]),
        "patch_unified_diff": "",
        "proposed_paths": proposed_paths,
        "verification_commands": verification_commands,
        "retrieval_evidence": ranked[:12],
        "abstained": not bool(proposed_paths),
        "residuals": [
            "generic retrieval found candidate surfaces but did not synthesize a verified patch"
            if proposed_paths
            else "no candidate repository surface found"
        ],
        "non_claims": [
            "This deterministic worker is not learned generation.",
            "A path proposal or plan is not a completed repository task.",
        ],
    }


def rank_files(root: Path, tokens: list[str]) -> list[dict[str, Any]]:
    rows = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if not relative.parts or relative.parts[0] not in ALLOWED_TOP_LEVEL:
            continue
        try:
            if path.stat().st_size > 1_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lowered_path = relative.as_posix().lower().replace("_", " ").replace("-", " ")
        lowered_text = text.lower()
        path_hits = sum(3 for token in tokens if token in lowered_path)
        text_hits = sum(min(lowered_text.count(token), 5) for token in tokens)
        score = path_hits + text_hits
        if score <= 0:
            continue
        rows.append({
            "path": relative.as_posix(),
            "score": score,
            "matched_keyword_count": sum(1 for token in tokens if token in lowered_path or token in lowered_text),
            "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return sorted(rows, key=lambda row: (-int(row["score"]), str(row["path"])))


def keywords(value: str) -> list[str]:
    return sorted({
        token for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) >= 3 and token not in STOPWORDS
    })


def verification_plan(paths: list[str]) -> list[str]:
    commands = []
    test_paths = [path for path in paths if path.startswith("tests/")]
    if test_paths:
        commands.append("python3 -m pytest -q " + " ".join(test_paths[:4]))
    if any(path.endswith(".py") for path in paths):
        commands.append("python3 -m py_compile <changed-python-files>")
    if any(path.endswith(".json") for path in paths):
        commands.append("python3 -m json.tool <changed-json-files>")
    return commands[:16]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("worker input must be an object")
    return value


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
