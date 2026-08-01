#!/usr/bin/env python3
"""Fetch and deterministically normalize prospectively frozen P4-v2r2 sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "theseus_p4v2r2_task_sources.json"
FIXTURES = ROOT / "tests" / "fixtures" / "theseus_p4v2r2_online"
REPORT = ROOT / "reports" / "theseus_p4v2r2_source_fetch.json"
SOURCE_SELECTION_COMMIT = "8cebe4a65bb03965e9f62efa8249f2f9ddb8fc08"
EXPECTED_REGISTRY_SHA256 = (
    "7264e2a040092de68e98a8a91b97ec38ca9a04a442f80a3c0551e767b0c68915"
)
POLICY = "project_theseus_p4v2r2_source_fetch_v1"

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
                "artifact_count": sum(
                    len(row["artifacts"]) for row in report["tasks"]
                ),
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
            "license_spdx": task.get("license_spdx"),
            "license_paths": task.get("license_paths"),
            "allowed_effect_paths": task.get("allowed_effect_paths"),
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
            required_paths = required_archive_paths(task, artifact["expected_root"])
            missing_paths = missing_archive_paths(normalized, required_paths)
            faults.extend(
                f"required_archive_member_missing:{task['stem']}:{artifact['label']}:{path}"
                for path in missing_paths
            )
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
                    "required_members": required_paths,
                    "missing_required_members": missing_paths,
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
        "archive_fetches_after_membership_freeze": sum(
            len(row["artifacts"]) for row in rows
        ),
        "parent_target_oracle_executions": 0,
        "candidate_or_control_calls": 0,
        "model_calls": 0,
        "teacher_calls": 0,
        "public_benchmark_cases": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
        "training_rows_written": 0,
        "tasks": rows,
        "maximum_inference": (
            "Exact licensed-source transport, deterministic archive normalization, "
            "root checks, and required effect/license member presence only; no parent "
            "or target behavior, evaluator adequacy, candidate, mechanism, D1, D2, "
            "serving, training, or book claim."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def audit_registry(registry: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if sha256_file(REGISTRY) != EXPECTED_REGISTRY_SHA256:
        faults.append("source_registry_digest_mismatch")
    if registry.get("policy") != "project_theseus_p4v2r2_online_source_selection_v1":
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
    repositories = {str(row.get("repository") or "").lower() for row in tasks}
    prior = {
        str(value).lower()
        for value in registry.get("source_disjoint_from_repositories", [])
    }
    if len(repositories) != 10:
        faults.append("source_registry_repositories_not_distinct")
    if repositories.intersection(prior):
        faults.append("source_registry_repository_overlap")
    for task in tasks:
        if not strings(task.get("license_paths")):
            faults.append(f"source_registry_license_path_missing:{task.get('stem')}")
        if not strings(task.get("allowed_effect_paths")):
            faults.append(f"source_registry_effect_path_missing:{task.get('stem')}")
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
        "D1_cases_consumed",
        "D2_cases_consumed",
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
            "expected_root": str(
                task["source_root" if label == "parent" else "target_root"]
            ),
            "url": (
                f"https://codeload.github.com/{repository}/tar.gz/"
                f"{task[f'{label}_revision']}"
            ),
            "upstream_name": f"{stem}_{label}_upstream.tar.gz",
            "normalized_name": f"{stem}_{label}.tar.gz",
            "sanitization_report": (
                f"theseus_{stem}_{label}_archive_sanitization.json"
            ),
        }
        for label in ("parent", "target")
    ]


def required_archive_paths(task: dict[str, Any], root: str) -> list[str]:
    relative_paths = sorted(
        set(strings(task.get("license_paths")))
        | set(strings(task.get("allowed_effect_paths")))
    )
    return [f"{root}/{path}" for path in relative_paths]


def missing_archive_paths(archive: Path, required: list[str]) -> list[str]:
    if not archive.is_file():
        return list(required)
    with tarfile.open(archive, "r:gz") as handle:
        names = {member.name.rstrip("/") for member in handle.getmembers()}
    return [path for path in required if path not in names]


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "Project-Theseus-P4V2R2/1"}
    )
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(  # noqa: S310 -- frozen GitHub codeload URL.
                request, timeout=120
            ) as response:
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


def strings(value: Any) -> list[str]:
    return [str(row) for row in value if str(row)] if isinstance(value, list) else []


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
