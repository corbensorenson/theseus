#!/usr/bin/env python3
"""Inventory VCM manifests, locks, and independently declared runner commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import tomllib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_repository_runner_inventory_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_repository_runner_inventory.json"
MANIFESTS = {"package.json", "pyproject.toml", "setup.py", "setup.cfg", "cargo.toml", "requirements.txt", "pipfile"}
LOCKS = {"package-lock.json", "npm-shrinkwrap.json", "pnpm-lock.yaml", "yarn.lock", "bun.lock", "bun.lockb", "uv.lock", "poetry.lock", "pdm.lock", "pipfile.lock", "cargo.lock"}
RUNNER_PATTERN = re.compile(r"(?:python\s+-m\s+pytest|\bpytest\b|npm\s+(?:test|run\s+test\S*)|pnpm\s+(?:test|run\s+test\S*)|yarn\s+(?:test|run\s+test\S*)|bun\s+test|npx\s+(?:vitest|jest)|cargo\s+test)(?:[^\n\r]*)", re.I)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = audit(config_path)
    p2a.write_json(p2a.resolve(args.out or p2a.read_json(config_path)["report"]), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = p2a.read_json(config_path)
    faults = []
    if config.get("policy") != POLICY: faults.append("policy_invalid")
    closure_path = p2a.resolve(config.get("closure_report", ""))
    if not closure_path.is_file() or p2a.sha256_file(closure_path) != config.get("closure_report_sha256"):
        faults.append("closure_report_binding_invalid")
        closure = {}
    else:
        closure = p2a.read_json(closure_path)
    if closure.get("trigger_state") != "GREEN" or closure.get("archive_artifacts") != 124:
        faults.append("closure_report_not_green")
    authority = p2a.mapping(config.get("authority"))
    if authority.get("normalized_archive_static_read_authorized") is not True or any(value is not False for key, value in authority.items() if key != "normalized_archive_static_read_authorized"):
        faults.append("authority_boundary_invalid")
    rows = []
    for task in p2a.dicts(closure.get("tasks")):
        artifacts = {str(row.get("label")): row for row in p2a.dicts(task.get("artifacts"))}
        per_revision = {}
        for label in ("parent", "target"):
            artifact_row = p2a.mapping(artifacts.get(label))
            path = p2a.resolve(str(artifact_row.get("normalized") or ""))
            if not path.is_file() or p2a.sha256_file(path) != artifact_row.get("normalized_sha256"):
                faults.append(f"archive_binding_invalid:{task.get('campaign_index')}:{label}")
                per_revision[label] = empty_revision()
            else:
                per_revision[label] = inspect_archive(path)
        target = per_revision["target"]
        rows.append({
            "index": task.get("campaign_index"),
            "repository": task.get("repository"),
            "parent": per_revision["parent"],
            "target": target,
            "target_manifest_present": bool(target["root_manifests"]),
            "target_lock_present": bool(target["root_locks"]),
            "independent_runner_receipt_present": bool(target["runner_receipts"]),
            "source_closure_receipt_present": True,
            "execution_performed": False,
        })
    observations = {
        "task_count": len(rows),
        "tasks_with_parent_and_target_source_closure": sum(bool(row["parent"]["member_count"] and row["target"]["member_count"]) for row in rows),
        "tasks_with_target_manifest": sum(row["target_manifest_present"] for row in rows),
        "tasks_with_target_lock": sum(row["target_lock_present"] for row in rows),
        "tasks_with_independent_runner_receipt": sum(row["independent_runner_receipt_present"] for row in rows),
        "tasks_without_independent_runner_receipt": sum(not row["independent_runner_receipt_present"] for row in rows),
        "parent_target_or_evaluator_executions": 0,
    }
    if len(rows) != 62: faults.append("task_count_invalid")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "STATIC_REPOSITORY_MANIFEST_LOCK_AND_RUNNER_INVENTORY_COMPLETE" if not faults else "STATIC_INVENTORY_INVALID",
        "faults": sorted(set(faults)),
        "config": artifact(config_path),
        "closure_report": artifact(closure_path),
        "observations": observations,
        "rows": rows,
        "runner_derivation_policy": {
            "package_json_test_scripts": True,
            "ci_run_lines_matching_known_test_runners": True,
            "cargo_manifest_standard_test_command": True,
            "python_manifest_declares_pytest_dependency_or_config": True,
            "hand_authored_repository_specific_commands": False,
            "execution_or_dependency_installation": False,
        },
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "parent_target_or_evaluator_executions": 0,
        "maximum_inference": config.get("maximum_inference"),
    }


def inspect_archive(path: Path) -> dict[str, Any]:
    manifests, locks, ci, receipts = [], [], [], []
    with tarfile.open(path, "r:gz") as handle:
        members = [row for row in handle.getmembers() if row.isfile()]
        for member in members:
            relative = strip_root(member.name)
            if not relative: continue
            parts = PurePosixPath(relative).parts
            basename = parts[-1].lower()
            root_level = len(parts) == 1
            if root_level and (basename in MANIFESTS or basename.startswith("requirements")):
                manifests.append(relative)
            if root_level and basename in LOCKS:
                locks.append(relative)
            interesting = (root_level and basename in MANIFESTS) or (len(parts) >= 3 and parts[0] == ".github" and parts[1] == "workflows" and PurePosixPath(relative).suffix.lower() in {".yml", ".yaml"})
            if not interesting: continue
            extracted = handle.extractfile(member)
            payload = extracted.read() if extracted else b""
            text = payload.decode("utf-8", errors="replace")
            if parts[:2] == (".github", "workflows"):
                ci.append(relative)
                for match in RUNNER_PATTERN.finditer(text):
                    receipts.append(receipt("ci_run_line", relative, "", match.group(0).strip()))
            elif basename == "package.json":
                try: package = json.loads(text)
                except json.JSONDecodeError: package = {}
                for key, command in p2a.mapping(package.get("scripts")).items():
                    if str(key).lower() == "test" or str(key).lower().startswith("test:"):
                        receipts.append(receipt("package_json_script", relative, str(key), f"npm run {key}" if key != "test" else "npm test", declared=str(command)))
            elif basename == "cargo.toml":
                receipts.append(receipt("cargo_manifest_standard", relative, "cargo_test", "cargo test"))
            elif basename in {"pyproject.toml", "setup.cfg", "requirements.txt"} or basename.startswith("requirements"):
                if "pytest" in text.lower():
                    receipts.append(receipt("python_manifest_pytest", relative, "pytest", "python -m pytest"))
    receipts = unique_receipts(receipts)
    return {"archive": p2a.rel(path), "archive_sha256": p2a.sha256_file(path), "member_count": len(members), "root_manifests": sorted(manifests), "root_locks": sorted(locks), "ci_workflows_read": sorted(ci), "runner_receipts": receipts}


def receipt(kind: str, path: str, key: str, command: str, declared: str = "") -> dict[str, str]:
    row = {"kind": kind, "source_path": path, "source_key": key, "command": command, "declared_value": declared}
    row["receipt_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return row


def unique_receipts(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted({row["receipt_sha256"]: row for row in rows}.values(), key=lambda row: (row["kind"], row["source_path"], row["command"]))


def strip_root(name: str) -> str:
    parts = PurePosixPath(name).parts
    return PurePosixPath(*parts[1:]).as_posix() if len(parts) > 1 else ""


def empty_revision() -> dict[str, Any]:
    return {"archive": "", "archive_sha256": "", "member_count": 0, "root_manifests": [], "root_locks": [], "ci_workflows_read": [], "runner_receipts": []}


def artifact(path: Path) -> dict[str, str]: return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path) if path.is_file() else ""}
def summary(r: dict[str, Any]) -> dict[str, Any]: return {k:r.get(k) for k in ("trigger_state","state","observations","parent_target_or_evaluator_executions","candidate_or_control_calls","external_reference_calls","faults")}

if __name__ == "__main__": raise SystemExit(main())
