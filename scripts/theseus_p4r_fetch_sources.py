#!/usr/bin/env python3
"""Fetch and normalize the three prospectively fixed P4R source pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "theseus_p4r_task_sources.json"
FIXTURES = ROOT / "tests" / "fixtures" / "theseus_p4r_online"
REPORT = ROOT / "reports" / "theseus_p4r_source_fetch.json"

import sys
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p2b_sanitize_archive as sanitizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    registry = read_json(REGISTRY)
    rows: list[dict[str, Any]] = []
    faults: list[str] = []
    for task in dictionaries(registry.get("new_tasks")):
        row: dict[str, Any] = {
            "stem": task.get("stem"),
            "repository": task.get("repository"),
            "artifacts": [],
        }
        for label, revision, expected_root in (
            ("parent", task.get("parent_revision"), task.get("source_root")),
            ("target", task.get("target_revision"), task.get("target_root")),
        ):
            stem = str(task["stem"])
            upstream = FIXTURES / f"{stem}_{label}_upstream.tar.gz"
            normalized = FIXTURES / f"{stem}_{label}.tar.gz"
            report_path = ROOT / "reports" / f"theseus_{stem}_{label}_archive_sanitization.json"
            url = f"https://codeload.github.com/{task['repository']}/tar.gz/{revision}"
            if not upstream.is_file() and args.fetch:
                download(url, upstream)
            if not upstream.is_file():
                faults.append(f"source_archive_missing:{relative(upstream)}")
                continue
            report = sanitizer.sanitize(upstream, normalized)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if report.get("trigger_state") != "GREEN":
                faults.append(f"archive_sanitization_red:{stem}:{label}")
            if report.get("source_archive_root") != expected_root:
                faults.append(f"archive_root_mismatch:{stem}:{label}")
            row["artifacts"].append({
                "label": label,
                "url": url,
                "revision": revision,
                "upstream": relative(upstream),
                "upstream_sha256": sha256_file(upstream),
                "normalized": relative(normalized),
                "normalized_sha256": sha256_file(normalized),
                "sanitization_report": relative(report_path),
                "sanitization_report_sha256": sha256_file(report_path),
                "source_archive_root": report.get("source_archive_root"),
                "omitted_members": report.get("omitted_members"),
            })
        rows.append(row)
    output = {
        "policy": "project_theseus_p4r_source_fetch_v1",
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "source_registry": relative(REGISTRY),
        "source_registry_sha256": sha256_file(REGISTRY),
        "network_use": "licensed_source_archive_acquisition_only",
        "candidate_or_control_calls": 0,
        "tasks": rows,
        "maximum_inference": "Source transport and archive normalization only; no task adequacy, model, mechanism, or book claim."
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "trigger_state": output["trigger_state"],
        "faults": output["faults"],
        "task_count": len(rows),
        "artifact_count": sum(len(row["artifacts"]) for row in rows),
    }, indent=2, sort_keys=True))
    return 0 if not faults else 2


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Project-Theseus-P4R/1"})
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed GitHub codeload URL from frozen registry.
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(destination)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dictionaries(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
