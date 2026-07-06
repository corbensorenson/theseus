#!/usr/bin/env python3
"""Audit local Markdown links in the active and deprecated Theseus docs."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_evidence_store  # noqa: E402


REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "theseus_doc_link_audit.json"
DEFAULT_MARKDOWN = REPORTS / "theseus_doc_link_audit.md"
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
EXTERNAL_SCHEMES = (
    "http://",
    "https://",
    "mailto:",
    "tel:",
    "data:",
    "app://",
    "file://",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    docs = markdown_files()
    broken = []
    checked = 0
    ignored = 0
    for doc in docs:
        doc_broken, doc_checked, doc_ignored = audit_doc(doc)
        broken.extend(doc_broken)
        checked += doc_checked
        ignored += doc_ignored

    payload = {
        "policy": "project_theseus_doc_link_audit_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not broken else "RED",
        "summary": {
            "doc_count": len(docs),
            "checked_local_links": checked,
            "ignored_external_or_anchor_links": ignored,
            "broken_local_links": len(broken),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "broken_links": broken,
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


def markdown_files() -> list[Path]:
    files = [ROOT / "README.md"]
    for base in [ROOT / "docs", ROOT / "deprecated"]:
        if base.exists():
            files.extend(path for path in base.rglob("*.md") if path.is_file())
    return sorted(set(files))


def audit_doc(path: Path) -> tuple[list[dict[str, Any]], int, int]:
    broken = []
    checked = 0
    ignored = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        for match in LINK_RE.finditer(line):
            raw_target = match.group(1).strip()
            target = normalize_target(raw_target)
            if should_ignore(target):
                ignored += 1
                continue
            checked += 1
            resolved = resolve_link(path, target)
            if not resolved.exists():
                broken.append(
                    {
                        "record_type": "broken_doc_link",
                        "source": rel(path),
                        "line": line_no,
                        "target": raw_target,
                        "resolved": rel(resolved),
                    }
                )
    return broken, checked, ignored


def normalize_target(raw_target: str) -> str:
    target = raw_target
    if target.startswith("<"):
        end = target.find(">")
        if end != -1:
            target = target[1:end]
    else:
        target = target.split()[0] if target.split() else target
    target = unquote(target.strip())
    if "#" in target:
        target = target.split("#", 1)[0]
    if "?" in target:
        target = target.split("?", 1)[0]
    return target.strip()


def should_ignore(target: str) -> bool:
    lowered = target.lower()
    return not target or target.startswith("#") or lowered.startswith(EXTERNAL_SCHEMES)


def resolve_link(source: Path, target: str) -> Path:
    target = target.replace("\\", "/")
    path = Path(target)
    if path.is_absolute():
        return path
    return (source.parent / path).resolve()


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Theseus Doc Link Audit",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- docs: `{summary.get('doc_count')}`",
        f"- checked_local_links: `{summary.get('checked_local_links')}`",
        f"- broken_local_links: `{summary.get('broken_local_links')}`",
        "",
        "## Broken Links",
        "",
    ]
    broken = payload.get("broken_links", [])
    if not broken:
        lines.append("None.")
    else:
        for row in broken[:50]:
            lines.append(
                f"- `{row.get('source')}:{row.get('line')}` -> `{row.get('target')}` "
                f"(resolved `{row.get('resolved')}`)"
            )
    return "\n".join(lines) + "\n"


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
