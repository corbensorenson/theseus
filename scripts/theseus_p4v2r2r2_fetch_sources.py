#!/usr/bin/env python3
"""Fetch and project the sealed P4-v2r2-r2 source pairs without execution."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import ssl
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import certifi


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p2b_sanitize_archive as sanitizer  # noqa: E402
import theseus_p4v2r2r2_source_registry as source_registry  # noqa: E402


REGISTRY = ROOT / "configs" / "theseus_p4v2r2r2_task_sources.json"
FIXTURES = ROOT / "tests" / "fixtures" / "theseus_p4v2r2r2_online"
REPORT = ROOT / "reports" / "theseus_p4v2r2r2_source_fetch.json"
SOURCE_SELECTION_COMMIT = "4d39b5f31d7b8d980b57230faa9ef3884bb9d04b"
EXPECTED_REGISTRY_SHA256 = "43ceb81b7790f07d9b60d208c947320fddc4deb511be62a781341360642ac99c"
POLICY = "project_theseus_p4v2r2r2_source_fetch_v1"


def artifact_plan(task: dict[str, Any], label: str) -> dict[str, str]:
    stem = str(task["stem"])
    revision = str(task[f"{label}_revision"])
    return {
        "label": label,
        "revision": revision,
        "url": f"https://codeload.github.com/{task['repository']}/tar.gz/{revision}",
        "expected_root": str(task["source_root" if label == "parent" else "target_root"]),
        "upstream_name": f"{stem}_{label}_upstream.tar.gz",
        "normalized_name": f"{stem}_{label}.tar.gz",
        "sanitization_report": f"theseus_{stem}_{label}_archive_sanitization.json",
    }


def required_paths(task: dict[str, Any]) -> list[str]:
    return sorted(
        set(p2a_strings(task.get("license_paths")))
        | set(p2a_strings(task.get("allowed_effect_paths")))
    )


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "Project-Theseus-P4V2R2R2/1"}
    )
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            tls = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(  # noqa: S310 -- sealed immutable URL.
                request, timeout=180, context=tls
            ) as response:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(destination)


def project_archive(
    source: Path, destination: Path, *, root: str, relative_paths: list[str]
) -> dict[str, Any]:
    required = [f"{root}/{path}" for path in sorted(set(relative_paths))]
    payloads: dict[str, bytes] = {}
    with tarfile.open(source, "r:gz") as handle:
        members = {member.name.rstrip("/"): member for member in handle.getmembers()}
        for name in required:
            member = members.get(name)
            if member is None or not member.isfile():
                raise ValueError(f"projection member missing or non-regular: {name}")
            stream = handle.extractfile(member)
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
        "gzip_mtime": 0,
        "member_mtime": 0,
        "uid_gid": 0,
        "output_sha256": sha256_file(destination),
    }


def acquire_sources(*, fetch: bool) -> dict[str, Any]:
    registry = read_json(REGISTRY)
    faults: list[str] = []
    if sha256_file(REGISTRY) != EXPECTED_REGISTRY_SHA256:
        faults.append("sealed_registry_digest_mismatch")
    registry_audit = source_registry.audit(REGISTRY)
    if registry_audit.get("trigger_state") != "GREEN":
        faults.append("sealed_registry_audit_red")

    task_rows: list[dict[str, Any]] = []
    for task in dictionaries(registry.get("tasks")):
        row: dict[str, Any] = {
            "campaign_index": task.get("campaign_index"),
            "stem": task.get("stem"),
            "repository": task.get("repository"),
            "license_spdx": task.get("license_spdx"),
            "license_paths": task.get("license_paths"),
            "allowed_effect_paths": task.get("allowed_effect_paths"),
            "artifacts": [],
        }
        for label in ("parent", "target"):
            plan = artifact_plan(task, label)
            upstream = FIXTURES / plan["upstream_name"]
            normalized = FIXTURES / plan["normalized_name"]
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
                    relative_paths=required_paths(task),
                )
            except Exception as exc:  # noqa: BLE001 -- fail closed with custody.
                faults.append(f"source_projection_failed:{task['stem']}:{label}:{exc}")
                continue
            finally:
                full_sanitized.unlink(missing_ok=True)

            sanitization["transport_sanitized_output_retained"] = False
            sanitization["transport_sanitized_output_sha256"] = full_sanitized_sha256
            sanitization["projected_output"] = {
                "path": relative(normalized),
                "sha256": sha256_file(normalized),
            }
            sanitization["projection"] = projection
            sanitization_path.parent.mkdir(parents=True, exist_ok=True)
            sanitization_path.write_text(
                json.dumps(sanitization, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if sanitization.get("trigger_state") != "GREEN":
                faults.append(f"archive_sanitization_red:{task['stem']}:{label}")
            if sanitization.get("source_archive_root") != plan["expected_root"]:
                faults.append(f"archive_root_mismatch:{task['stem']}:{label}")
            row["artifacts"].append(
                {
                    **plan,
                    "upstream": relative(upstream),
                    "upstream_sha256": sha256_file(upstream),
                    "normalized": relative(normalized),
                    "normalized_sha256": sha256_file(normalized),
                    "projection": projection,
                    "sanitization_report": relative(sanitization_path),
                    "sanitization_report_sha256": sha256_file(sanitization_path),
                    "full_sanitized_sha256": full_sanitized_sha256,
                }
            )
        task_rows.append(row)

    if len(task_rows) != 10:
        faults.append("source_task_count_invalid")
    artifact_count = sum(len(row["artifacts"]) for row in task_rows)
    if artifact_count != 20:
        faults.append("source_artifact_count_invalid")
    report = {
        "policy": POLICY,
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "source_registry": relative(REGISTRY),
        "source_registry_sha256": sha256_file(REGISTRY),
        "source_selection_commit": SOURCE_SELECTION_COMMIT,
        "source_registry_audit": registry_audit,
        "network_use": "twenty_immutable_permissively_licensed_github_codeload_archives_only",
        "archive_fetches_after_membership_freeze": artifact_count,
        "parent_target_oracle_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "local_model_calls": 0,
        "hosted_model_calls": 0,
        "teacher_calls": 0,
        "public_benchmark_cases": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
        "training_rows_written": 0,
        "project_selected_quality_token_cap": None,
        "tasks": task_rows,
        "maximum_inference": "A GREEN receipt establishes only exact licensed archive transport, deterministic sanitization/projection, root identity, and required production/license member presence for the sealed ten-task source set.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dictionaries(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def p2a_strings(value: Any) -> list[str]:
    return [str(row) for row in value if str(row)] if isinstance(value, list) else []


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    report = acquire_sources(fetch=args.fetch)
    print(
        json.dumps(
            {
                "trigger_state": report["trigger_state"],
                "faults": report["faults"],
                "task_count": len(report["tasks"]),
                "artifact_count": sum(len(row["artifacts"]) for row in report["tasks"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
