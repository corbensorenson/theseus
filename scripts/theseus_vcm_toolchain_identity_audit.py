#!/usr/bin/env python3
"""Bind exact local evaluator tool identities to VCM dependency classes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_toolchain_identity_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_toolchain_identity_audit.json"
LOCK_MANAGER = {
    "package-lock.json": "npm",
    "npm-shrinkwrap.json": "npm",
    "pnpm-lock.yaml": "pnpm",
    "yarn.lock": "yarn",
    "bun.lock": "bun",
    "bun.lockb": "bun",
    "uv.lock": "uv",
    "cargo.lock": "cargo",
}


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
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    dependency_binding = p2a.mapping(config.get("dependency_class_report"))
    dependency_path = p2a.resolve(str(dependency_binding.get("path") or ""))
    if not dependency_path.is_file() or p2a.sha256_file(dependency_path) != str(dependency_binding.get("sha256") or ""):
        faults.append("dependency_class_report_binding_invalid")
        dependency = {}
    else:
        dependency = p2a.read_json(dependency_path)
    if dependency.get("trigger_state") != "GREEN" or dependency.get("observations", {}).get("tasks_dependency_classified") != 62:
        faults.append("dependency_class_report_not_green")
    authority = p2a.mapping(config.get("authority"))
    if authority.get("trusted_tool_identity_queries_authorized") is not True or any(
        value is not False for key, value in authority.items() if key != "trusted_tool_identity_queries_authorized"
    ):
        faults.append("authority_boundary_invalid")
    tools = {}
    for name, raw in p2a.mapping(config.get("tools")).items():
        declared = p2a.mapping(raw)
        path_text = str(declared.get("path") or "")
        if not path_text:
            tools[name] = {
                "available": False,
                "path": "",
                "sha256": "",
                "version_command": p2a.strings(declared.get("version_command")),
                "version_output": "",
                "version_returncode": None,
            }
            continue
        path = p2a.resolve(path_text)
        observed_hash = p2a.sha256_file(path) if path.is_file() else ""
        if observed_hash != str(declared.get("sha256") or ""):
            faults.append(f"tool_identity_invalid:{name}")
        command = [str(path), *p2a.strings(declared.get("version_command"))]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        output = (completed.stdout or completed.stderr).strip()
        if completed.returncode != 0:
            faults.append(f"tool_version_query_failed:{name}")
        tools[name] = {
            "available": path.is_file() and observed_hash == str(declared.get("sha256") or ""),
            "path": p2a.rel(path),
            "sha256": observed_hash,
            "version_command": command,
            "version_output": output,
            "version_output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "version_returncode": completed.returncode,
        }
    rows = []
    for prior in p2a.dicts(dependency.get("rows")):
        managers = sorted({
            LOCK_MANAGER.get(PurePosixPath(str(lock.get("path") or "")).name.lower(), "unknown")
            for lock in p2a.dicts(prior.get("relevant_lock_receipts"))
        } - {"unknown"})
        missing = sorted(manager for manager in managers if not p2a.mapping(tools.get(manager)).get("available"))
        rows.append({
            "index": prior.get("index"),
            "repository": prior.get("repository"),
            "query_language": prior.get("query_language"),
            "dependency_class": prior.get("dependency_class"),
            "lock_managers": managers,
            "available_lock_managers": sorted(set(managers) - set(missing)),
            "missing_lock_managers": missing,
            "lock_manager_identity_complete": bool(managers) and not missing,
            "lock_not_required_for_scoped_evaluator": prior.get("lock_not_required_for_scoped_evaluator") is True,
            "immutable_resolution_required_before_execution": prior.get("immutable_resolution_required_before_execution") is True,
            "repository_execution_performed": False,
            "evaluator_execution_ready": False,
        })
    observations = {
        "task_count": len(rows),
        "declared_tool_count": len(tools),
        "available_tool_count": sum(bool(row.get("available")) for row in tools.values()),
        "missing_tool_count": sum(not bool(row.get("available")) for row in tools.values()),
        "tasks_with_complete_lock_manager_identity": sum(row["lock_manager_identity_complete"] for row in rows),
        "tasks_with_missing_lock_manager_identity": sum(bool(row["missing_lock_managers"]) for row in rows),
        "tasks_lock_not_required_for_scoped_evaluator": sum(row["lock_not_required_for_scoped_evaluator"] for row in rows),
        "tasks_requiring_immutable_resolution": sum(row["immutable_resolution_required_before_execution"] for row in rows),
        "tasks_evaluator_execution_ready": 0,
        "parent_target_or_evaluator_executions": 0,
    }
    expected = p2a.mapping(config.get("expected_observations"))
    for key, value in expected.items():
        if observations.get(key) != value:
            faults.append(f"expected_observation_mismatch:{key}")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "LOCAL_TOOLCHAIN_IDENTITIES_BOUND_YARN_RESIDUAL" if not faults else "TOOLCHAIN_IDENTITY_AUDIT_INVALID",
        "faults": sorted(set(faults)),
        "config": artifact(config_path),
        "dependency_class_report": artifact(dependency_path),
        "tools": tools,
        "observations": observations,
        "rows": rows,
        "missing_tool_task_indices": {
            name: [row["index"] for row in rows if name in row["missing_lock_managers"]]
            for name, tool in tools.items() if not tool.get("available")
        },
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "parent_target_or_evaluator_executions": 0,
        "maximum_inference": config.get("maximum_inference"),
    }


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path) if path.is_file() else ""}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "observations", "missing_tool_task_indices",
        "parent_target_or_evaluator_executions", "candidate_or_control_calls",
        "external_reference_calls", "faults",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
