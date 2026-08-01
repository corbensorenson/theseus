#!/usr/bin/env python3
"""Fetch and normalize the one prospectively sealed P4 recovery source pair."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import ssl
import sys
import tempfile
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import certifi


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "theseus_p4v2r2r1_task_sources.json"
FIXTURES = ROOT / "tests" / "fixtures" / "theseus_p4v2r2r1_online"
REPORT = ROOT / "reports" / "theseus_p4v2r2r1_source_fetch.json"
EXPECTED_REGISTRY_SHA256 = "92cd51cfe0fba5ad794527573aa6cd3d570394d7d01fabbb6a55ef2bb0e25f56"
SOURCE_SELECTION_COMMIT = "b0a43dc8aacc722a92b5520026f4e91aa6e3497c"

sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p2b_sanitize_archive as sanitizer  # noqa: E402
import theseus_p4v2r2r1_source_registry as source_registry  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def artifact(task: dict[str, Any], label: str) -> dict[str, str]:
    stem = str(task["stem"])
    revision = str(task[f"{label}_revision"])
    return {
        "label": label,
        "revision": revision,
        "url": f"https://codeload.github.com/{task['repository']}/tar.gz/{revision}",
        "expected_root": str(task["source_root" if label == "parent" else "target_root"]),
        "upstream": f"{stem}_{label}_upstream.tar.gz",
        "normalized": f"{stem}_{label}.tar.gz",
        "sanitization_report": f"theseus_{stem}_{label}_archive_sanitization.json",
    }


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "Project-Theseus-P4V2R2R1/1"}
    )
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            tls = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=180, context=tls
            ) as response:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(destination)


def project_archive(
    sanitized: Path, destination: Path, *, root: str, relative_paths: list[str]
) -> dict[str, Any]:
    """Retain only exact task inputs in a deterministic, evaluator-safe archive."""
    required = [f"{root}/{path}" for path in sorted(set(relative_paths))]
    payloads: dict[str, bytes] = {}
    with tarfile.open(sanitized, "r:gz") as source:
        by_name = {member.name.rstrip("/"): member for member in source.getmembers()}
        for name in required:
            member = by_name.get(name)
            if member is None or not member.isfile():
                raise ValueError(f"projection member missing or non-regular: {name}")
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"projection member unreadable: {name}")
            payloads[name] = stream.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as raw:
        temporary = Path(raw.name)
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as target:
                for name in required:
                    payload = payloads[name]
                    info = tarfile.TarInfo(name=name)
                    info.size = len(payload)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    target.addfile(info, io.BytesIO(payload))
    temporary.replace(destination)
    return {
        "policy": "project_theseus_exact_task_source_projection_v1",
        "retained_members": required,
        "omitted_member_count": None,
        "gzip_mtime": 0,
        "member_mtime": 0,
        "uid_gid": 0,
        "output_sha256": sha256_file(destination),
    }


def acquire(*, fetch: bool) -> dict[str, Any]:
    registry = read_json(REGISTRY)
    faults: list[str] = []
    if sha256_file(REGISTRY) != EXPECTED_REGISTRY_SHA256:
        faults.append("sealed_registry_digest_mismatch")
    registry_audit = source_registry.audit(REGISTRY)
    if registry_audit.get("trigger_state") != "GREEN":
        faults.append("sealed_registry_audit_red")
    task = registry.get("replacement_task", {})
    rows: list[dict[str, Any]] = []
    for label in ("parent", "target"):
        plan = artifact(task, label)
        upstream = FIXTURES / plan["upstream"]
        normalized = FIXTURES / plan["normalized"]
        sanitization_path = ROOT / "reports" / plan["sanitization_report"]
        if not upstream.is_file() and fetch:
            download(plan["url"], upstream)
        if not upstream.is_file():
            faults.append(f"source_archive_missing:{relative(upstream)}")
            continue
        with tempfile.NamedTemporaryFile(
            dir=FIXTURES, suffix=".full-sanitized.tar.gz", delete=False
        ) as handle:
            full_sanitized = Path(handle.name)
        try:
            sanitization = sanitizer.sanitize(upstream, full_sanitized)
            full_sanitized_sha256 = sha256_file(full_sanitized)
            projection = project_archive(
                full_sanitized,
                normalized,
                root=plan["expected_root"],
                relative_paths=list(task.get("license_paths", []))
                + list(task.get("allowed_effect_paths", [])),
            )
        finally:
            full_sanitized.unlink(missing_ok=True)
        sanitization["transport_sanitized_output_retained"] = False
        sanitization["transport_sanitized_output_sha256"] = full_sanitized_sha256
        sanitization["projected_output"] = {
            "path": relative(normalized),
            "sha256": sha256_file(normalized),
        }
        sanitization["projection"] = projection
        sanitization["output"] = {
            "path": str(normalized.resolve()),
            "sha256": sha256_file(normalized),
        }
        sanitization_path.parent.mkdir(parents=True, exist_ok=True)
        sanitization_path.write_text(
            json.dumps(sanitization, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if sanitization.get("trigger_state") != "GREEN":
            faults.append(f"archive_sanitization_red:{label}")
        if sanitization.get("source_archive_root") != plan["expected_root"]:
            faults.append(f"archive_root_mismatch:{label}")
        required = [
            f"{plan['expected_root']}/{path}"
            for path in sorted(
                set(task.get("license_paths", []))
                | set(task.get("allowed_effect_paths", []))
            )
        ]
        with tarfile.open(normalized, "r:gz") as handle:
            members = {member.name.rstrip("/") for member in handle.getmembers()}
        missing = [path for path in required if path not in members]
        if missing:
            faults.append(f"required_members_missing:{label}")
        rows.append(
            {
                **plan,
                "upstream_path": relative(upstream),
                "upstream_sha256": sha256_file(upstream),
                "normalized_path": relative(normalized),
                "normalized_sha256": sha256_file(normalized),
                "projection": projection,
                "sanitization_report_path": relative(sanitization_path),
                "sanitization_report_sha256": sha256_file(sanitization_path),
                "required_members": required,
                "missing_required_members": missing,
            }
        )
    if len(rows) != 2:
        faults.append("source_pair_incomplete")
    report = {
        "policy": "project_theseus_p4v2r2r1_source_fetch_v1",
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "source_registry": relative(REGISTRY),
        "source_registry_sha256": sha256_file(REGISTRY),
        "source_selection_commit": SOURCE_SELECTION_COMMIT,
        "repository": task.get("repository"),
        "license_spdx": task.get("license_spdx"),
        "artifacts": rows,
        "network_use": "two_immutable_licensed_github_codeload_archives_only",
        "archive_fetches_after_membership_seal": len(rows),
        "parent_target_oracle_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "hosted_model_calls": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
        "project_selected_quality_token_cap": None,
        "maximum_inference": "A GREEN receipt establishes only exact licensed archive transport, normalization, root identity, and required effect/license member presence for the sealed replacement task.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    report = acquire(fetch=args.fetch)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
