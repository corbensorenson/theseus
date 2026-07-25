#!/usr/bin/env python3
"""Source-bound qualification gate for the Phase 0 verification environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import python_environment_gate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "phase0_reproducibility_gate.json"
SOURCE_ROOTS = ("configs", "requirements", "scripts", "src", "tests", "crates")
SOURCE_SUFFIXES = {".json", ".py", ".rs", ".toml", ".txt"}
ROOT_SOURCE_FILES = ("Cargo.lock", "Cargo.toml")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def source_files(root: Path = ROOT) -> list[Path]:
    files: set[Path] = set()
    for name in ROOT_SOURCE_FILES:
        path = root / name
        if path.is_file():
            files.add(path)
    for name in SOURCE_ROOTS:
        directory = root / name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES and "__pycache__" not in path.parts:
                files.add(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def source_manifest(root: Path = ROOT) -> dict[str, Any]:
    rows = []
    aggregate = hashlib.sha256()
    for path in source_files(root):
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "algorithm": "sha256_path_and_content_v1",
        "sha256": aggregate.hexdigest(),
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def run(command: list[str], *, timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "stdout_tail": result.stdout[-8000:],
            "stderr_tail": result.stderr[-8000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "passed": False,
            "fault": type(exc).__name__,
            "message": str(exc),
            "duration_seconds": round(time.monotonic() - started, 3),
        }


def junit_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        field: sum(int(suite.attrib.get(field, "0")) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }


def execute() -> dict[str, Any]:
    manifest = source_manifest()
    environment = python_environment_gate.audit(python_environment_gate.load_contract(), "cpu")
    with tempfile.TemporaryDirectory(prefix="theseus-phase0-") as directory:
        junit = Path(directory) / "pytest.xml"
        python_tests = run(
            [sys.executable, "-m", "pytest", "-qq", "-m", "not accelerator", f"--junitxml={junit}"],
            timeout=1800,
        )
        if junit.is_file():
            python_tests["junit"] = junit_summary(junit)
    rust_tests = run(["cargo", "test", "--workspace"], timeout=1800)
    ready = (
        environment.get("trigger_state") == "GREEN"
        and python_tests.get("passed") is True
        and rust_tests.get("passed") is True
    )
    return {
        "policy": "project_theseus_phase0_reproducibility_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trigger_state": "GREEN" if ready else "RED",
        "source_manifest": manifest,
        "python_environment": environment,
        "python_non_accelerator_suite": python_tests,
        "rust_workspace_suite": rust_tests,
        "boundaries": {
            "ambient_accelerator_import_count": environment.get("summary", {}).get("ambient_accelerator_import_count"),
            "external_inference_calls": 0,
            "public_training_rows": 0,
            "fallback_or_template_credit": 0,
        },
        "non_claims": [
            "Phase 0 qualification is repository reproducibility evidence, not model capability or promotion evidence.",
            "MLX package identity is checked without ambient accelerator import; accelerator behavior requires watchdog receipts.",
        ],
    }


def verify(report: dict[str, Any]) -> dict[str, Any]:
    current = source_manifest()
    recorded = report.get("source_manifest") or {}
    faults = []
    if report.get("policy") != "project_theseus_phase0_reproducibility_gate_v1":
        faults.append("policy_invalid")
    if report.get("trigger_state") != "GREEN":
        faults.append("recorded_run_not_green")
    if recorded.get("sha256") != current.get("sha256"):
        faults.append("source_manifest_drift")
    if (report.get("python_environment") or {}).get("trigger_state") != "GREEN":
        faults.append("python_environment_not_green")
    if (report.get("python_non_accelerator_suite") or {}).get("passed") is not True:
        faults.append("python_suite_not_green")
    if (report.get("rust_workspace_suite") or {}).get("passed") is not True:
        faults.append("rust_suite_not_green")
    return {
        "policy": "project_theseus_phase0_reproducibility_verification_v1",
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": faults,
        "recorded_source_sha256": recorded.get("sha256"),
        "current_source_sha256": current.get("sha256"),
        "source_file_count": current.get("file_count"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_REPORT.relative_to(ROOT)))
    args = parser.parse_args()
    report_path = resolve(args.out)
    if args.execute:
        report = execute()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        output = verify(report)
    else:
        if not report_path.is_file():
            output = {"trigger_state": "RED", "faults": ["qualification_report_missing"]}
        else:
            output = verify(json.loads(report_path.read_text(encoding="utf-8")))
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output.get("trigger_state") == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
