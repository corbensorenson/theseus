#!/usr/bin/env python3
"""Classify dirty workspace entries for Theseus cleanup governance."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_evidence_store  # noqa: E402


REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "theseus_dirty_workspace_review.json"
DEFAULT_MARKDOWN = REPORTS / "theseus_dirty_workspace_review.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()
    started = time.perf_counter()
    rows = classify_rows(git_status_rows())
    counts = Counter(row["classification"] for row in rows)
    trigger_state = "YELLOW" if rows else "GREEN"
    payload = {
        "policy": "project_theseus_dirty_workspace_review_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "dirty_count": len(rows),
            "source_change_count": int(counts.get("source_change", 0)),
            "generated_artifact_count": int(counts.get("generated_artifact", 0)),
            "control_plane_change_count": int(counts.get("control_plane_change", 0)),
            "docs_or_config_count": int(counts.get("docs_or_config", 0)),
            "archive_or_ignored_count": int(counts.get("archive_or_ignored", 0)),
            "unknown_count": int(counts.get("unknown", 0)),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "rows": rows,
        "rules": {
            "source_change": "review and keep with matching tests/smokes",
            "generated_artifact": "do not commit unless it is curated training/evidence data",
            "control_plane_change": "part of the current cleanup goal",
            "archive_or_ignored": "local retention/quarantine payload; keep ignored",
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
    return 0


def git_status_rows() -> list[str]:
    result = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, timeout=15)
    return [line for line in result.stdout.splitlines() if line.strip()]


def classify_rows(lines: list[str]) -> list[dict[str, Any]]:
    rows = []
    for line in lines:
        status = line[:2].strip() or "??"
        path = line[3:].strip().replace("\\", "/")
        classification = classify_path(path)
        rows.append(
            {
                "record_type": "dirty_workspace_entry",
                "status": status,
                "path": path,
                "classification": classification,
                "recommended_action": recommended_action(classification, path),
            }
        )
    return rows


def classify_path(path: str) -> str:
    if path.startswith("archive/") or path.startswith("reports/"):
        return "archive_or_ignored"
    if path in {".gitignore"} or path.startswith("docs/") or path.startswith("configs/"):
        return "docs_or_config"
    if path.startswith("scripts/theseus_") or path == "scripts/report_evidence_store.py":
        return "control_plane_change"
    if path.startswith("scripts/") or path.startswith("crates/") or path.startswith("src/"):
        return "source_change"
    if path.startswith("data/") or path.endswith((".jsonl", ".json", ".sqlite", ".log")):
        return "generated_artifact"
    return "unknown"


def recommended_action(classification: str, path: str) -> str:
    if classification == "source_change":
        return "verify with targeted compile/smoke before keeping"
    if classification == "generated_artifact":
        return "preserve only if curated evidence/training data; otherwise archive/quarantine"
    if classification == "control_plane_change":
        return "keep with control-plane smoke and report refresh"
    if classification == "docs_or_config":
        return "keep if it supports cleanup policy or redirect consolidation"
    if classification == "archive_or_ignored":
        return "keep ignored; do not commit payload"
    return "review manually"


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Theseus Dirty Workspace Review",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- dirty_count: `{summary.get('dirty_count')}`",
        f"- source: `{summary.get('source_change_count')}` generated: `{summary.get('generated_artifact_count')}` control-plane: `{summary.get('control_plane_change_count')}`",
        "",
        "## Rows",
        "",
    ]
    for row in payload.get("rows", [])[:80]:
        lines.append(f"- `{row.get('classification')}` `{row.get('status')}` `{row.get('path')}`")
    return "\n".join(lines) + "\n"


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
