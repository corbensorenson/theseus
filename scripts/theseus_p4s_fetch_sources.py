#!/usr/bin/env python3
"""Fetch and deterministically normalize the prospectively frozen P4S sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "theseus_p4s_task_sources.json"
FIXTURES = ROOT / "tests" / "fixtures" / "theseus_p4s_online"
REPORT = ROOT / "reports" / "theseus_p4s_source_fetch.json"
SOURCE_SELECTION_COMMIT = "560df8f437d470e8eea9fbd927ee7854f1b93e74"
EXPECTED_REGISTRY_SHA256 = "4237d99b6c051d2e692c2678539183d28194b10a0a704c323b38b481eaef508f"
POLICY = "project_theseus_p4s_source_fetch_v1"

sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p2b_sanitize_archive as sanitizer  # noqa: E402


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


def acquire_sources(*, fetch: bool) -> dict[str, Any]:
    registry = read_json(REGISTRY)
    faults = audit_registry(registry)
    rows: list[dict[str, Any]] = []
    for task in dictionaries(registry.get("tasks")):
        row: dict[str, Any] = {
            "campaign_index": task.get("campaign_index"),
            "stem": task.get("stem"),
            "repository": task.get("repository"),
            "artifacts": [],
        }
        for artifact in artifact_plan(task):
            upstream = FIXTURES / artifact["upstream_name"]
            normalized = FIXTURES / artifact["normalized_name"]
            report_path = ROOT / "reports" / artifact["sanitization_report"]
            if not upstream.is_file() and fetch:
                download(artifact["url"], upstream)
            if not upstream.is_file():
                faults.append(f"source_archive_missing:{relative(upstream)}")
                continue
            sanitization = sanitizer.sanitize(upstream, normalized)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(sanitization, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if sanitization.get("trigger_state") != "GREEN":
                faults.append(
                    f"archive_sanitization_red:{task['stem']}:{artifact['label']}"
                )
            if sanitization.get("source_archive_root") != artifact["expected_root"]:
                faults.append(f"archive_root_mismatch:{task['stem']}:{artifact['label']}")
            row["artifacts"].append(
                {
                    "label": artifact["label"],
                    "url": artifact["url"],
                    "revision": artifact["revision"],
                    "upstream": relative(upstream),
                    "upstream_sha256": sha256_file(upstream),
                    "normalized": relative(normalized),
                    "normalized_sha256": sha256_file(normalized),
                    "sanitization_report": relative(report_path),
                    "sanitization_report_sha256": sha256_file(report_path),
                    "source_archive_root": sanitization.get("source_archive_root"),
                    "omitted_members": sanitization.get("omitted_members"),
                }
            )
        rows.append(row)
    if len(rows) != 10:
        faults.append("source_task_count_invalid")
    if sum(len(row["artifacts"]) for row in rows) != 20:
        faults.append("source_artifact_count_invalid")
    output = {
        "policy": POLICY,
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "source_registry": relative(REGISTRY),
        "source_registry_sha256": sha256_file(REGISTRY),
        "source_selection_commit": SOURCE_SELECTION_COMMIT,
        "network_use": "licensed_source_archive_acquisition_only",
        "candidate_or_control_calls": 0,
        "model_calls": 0,
        "teacher_calls": 0,
        "public_benchmark_cases": 0,
        "tasks": rows,
        "maximum_inference": (
            "Source transport and deterministic archive normalization only; no task "
            "adequacy, candidate, subsystem, D1, D2, serving, training, or book claim."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def audit_registry(registry: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if sha256_file(REGISTRY) != EXPECTED_REGISTRY_SHA256:
        faults.append("source_registry_digest_mismatch")
    if registry.get("policy") != "project_theseus_p4s_online_source_selection_v1":
        faults.append("source_registry_policy_invalid")
    if registry.get("state") != (
        "FIXED_BEFORE_ARCHIVE_FETCH_PARENT_TARGET_EXECUTION_OR_CANDIDATE_GENERATION"
    ):
        faults.append("source_registry_not_prospectively_fixed")
    tasks = dictionaries(registry.get("tasks"))
    if len(tasks) != 10 or registry.get("task_count") != 10:
        faults.append("source_registry_task_count_invalid")
    if [row.get("campaign_index") for row in tasks] != list(range(1, 11)):
        faults.append("source_registry_campaign_indexes_invalid")
    if len({str(row.get("repository") or "").lower() for row in tasks}) != 10:
        faults.append("source_registry_repositories_not_distinct")
    boundaries = mapping(registry.get("boundaries"))
    if boundaries.get("candidate_generation_opened") is not False:
        faults.append("candidate_generation_already_opened")
    for key in (
        "archive_fetches_after_membership_freeze",
        "parent_target_oracle_executions",
        "local_model_calls",
        "hosted_model_calls",
        "deterministic_request_compiler_calls",
        "teacher_calls",
        "public_benchmark_cases",
        "training_rows_written",
    ):
        if int(boundaries.get(key) or 0) != 0:
            faults.append(f"source_registry_boundary_nonzero:{key}")
    if boundaries.get("user_task_label_or_approval_dependency") is not False:
        faults.append("source_registry_user_dependency_present")
    return faults


def artifact_plan(task: dict[str, Any]) -> list[dict[str, str]]:
    stem = str(task["stem"])
    repository = str(task["repository"])
    return [
        {
            "label": label,
            "revision": str(task[f"{label}_revision"]),
            "expected_root": str(task["source_root" if label == "parent" else "target_root"]),
            "url": f"https://codeload.github.com/{repository}/tar.gz/{task[f'{label}_revision']}",
            "upstream_name": f"{stem}_{label}_upstream.tar.gz",
            "normalized_name": f"{stem}_{label}.tar.gz",
            "sanitization_report": f"theseus_{stem}_{label}_archive_sanitization.json",
        }
        for label in ("parent", "target")
    ]


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Project-Theseus-P4S/1"})
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 -- frozen GitHub codeload URL.
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


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
