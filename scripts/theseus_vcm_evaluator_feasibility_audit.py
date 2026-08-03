#!/usr/bin/env python3
"""Audit whether the admitted VCM source panel is executable as an evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_evaluator_feasibility_audit.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_vcm_evaluator_feasibility_audit.json"
POLICY = "project_theseus_vcm_evaluator_feasibility_audit_v1"
PRIMARY_EXECUTABLE_SUFFIXES = {
    "Python": {".py"},
    "JavaScript": {".js", ".mjs", ".cjs"},
    "TypeScript": {".ts", ".tsx", ".mts", ".cts"},
    "Rust": {".rs"},
}
ALL_EXECUTABLE_SUFFIXES = set().union(*PRIMARY_EXECUTABLE_SUFFIXES.values())
MANIFEST_NAMES = {
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "package.json", "cargo.toml",
}
LOCK_NAMES = {
    "uv.lock", "poetry.lock", "pdm.lock", "pipfile.lock",
    "package-lock.json", "npm-shrinkwrap.json", "pnpm-lock.yaml", "yarn.lock",
    "bun.lock", "bun.lockb", "cargo.lock",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    args = parser.parse_args()
    report = audit(p2a.resolve(args.config))
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = p2a.read_json(config_path)
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    reports: dict[str, dict[str, Any]] = {}
    for name, raw in p2a.mapping(config.get("reports")).items():
        binding = p2a.mapping(raw)
        path = p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != str(binding.get("sha256") or ""):
            faults.append(f"report_binding_invalid:{name}")
            reports[name] = {}
        else:
            reports[name] = p2a.read_json(path)
    panel = reports.get("source_panel", {})
    python_sandbox = reports.get("python_sandbox_qualification", {})
    if panel.get("trigger_state") != "GREEN" or panel.get("source_panel_admitted") is not True:
        faults.append("source_panel_not_admitted")
    if python_sandbox.get("trigger_state") != "GREEN" or python_sandbox.get("untrusted_execution_authorized") is not True:
        faults.append("python_sandbox_not_qualified")
    authority = p2a.mapping(config.get("authority"))
    if authority.get("trusted_toolchain_identity_queries_authorized") is not True or any(value is not False for key, value in authority.items() if key != "trusted_toolchain_identity_queries_authorized"):
        faults.append("authority_boundary_invalid")

    rows = p2a.dicts(panel.get("assembled_rows"))
    if len(rows) != 62:
        faults.append("source_panel_row_count_invalid")
    task_gaps: list[dict[str, Any]] = []
    language_counts: Counter[str] = Counter()
    primary_executable_count = 0
    auxiliary_executable_count = 0
    nonexecutable_artifact_count = 0
    tasks_with_auxiliary_artifacts = 0
    tasks_with_manifest = 0
    tasks_with_lock = 0
    for row in rows:
        index = integer(row.get("index"))
        language = str(row.get("query_language") or "")
        language_counts[language] += 1
        verifier_paths = p2a.strings(row.get("selected_verifier_paths"))
        primary = [path for path in verifier_paths if Path(path).suffix.lower() in PRIMARY_EXECUTABLE_SUFFIXES.get(language, set())]
        auxiliary = [path for path in verifier_paths if Path(path).suffix.lower() in ALL_EXECUTABLE_SUFFIXES and path not in primary]
        nonexecutable = [path for path in verifier_paths if path not in primary and path not in auxiliary]
        primary_executable_count += len(primary)
        auxiliary_executable_count += len(auxiliary)
        nonexecutable_artifact_count += len(nonexecutable)
        if auxiliary or nonexecutable:
            tasks_with_auxiliary_artifacts += 1
        archive_members = {
            str(member.get("path") or "")
            for receipt in p2a.mapping(row.get("archives")).values()
            for member in p2a.dicts(p2a.mapping(receipt).get("members"))
        }
        basenames = {Path(path).name.lower() for path in archive_members}
        manifests = sorted(name for name in basenames if name in MANIFEST_NAMES or name.startswith("requirements"))
        locks = sorted(name for name in basenames if name in LOCK_NAMES)
        tasks_with_manifest += bool(manifests)
        tasks_with_lock += bool(locks)
        gaps = []
        if not primary:
            gaps.append("primary_language_executable_verifier_absent")
        if not manifests:
            gaps.append("dependency_or_package_manifest_absent")
        if not locks:
            gaps.append("dependency_lock_receipt_absent")
        gaps.extend((
            "transitive_local_source_closure_absent",
            "independent_runner_command_receipt_absent",
            "parent_fail_target_pass_not_executed",
        ))
        if language != "Python":
            gaps.append("language_specific_sandbox_not_qualified")
        task_gaps.append({
            "index": index,
            "panel": row.get("panel"),
            "query_language": language,
            "repository": row.get("repository"),
            "primary_executable_verifier_paths": primary,
            "auxiliary_executable_verifier_paths": auxiliary,
            "nonexecutable_verifier_artifact_paths": nonexecutable,
            "manifest_members": manifests,
            "lock_members": locks,
            "evaluator_ready": False,
            "gaps": gaps,
        })

    expected = p2a.mapping(config.get("expected_observations"))
    observations = {
        "task_count": len(rows),
        "primary_executable_verifier_task_count": sum(bool(row["primary_executable_verifier_paths"]) for row in task_gaps),
        "primary_executable_verifier_path_count": primary_executable_count,
        "auxiliary_executable_verifier_path_count": auxiliary_executable_count,
        "nonexecutable_verifier_artifact_count": nonexecutable_artifact_count,
        "tasks_with_auxiliary_or_nonexecutable_verifier_artifacts": tasks_with_auxiliary_artifacts,
        "tasks_with_dependency_or_package_manifest": tasks_with_manifest,
        "tasks_with_dependency_lock": tasks_with_lock,
        "tasks_with_transitive_local_source_closure_receipt": 0,
        "tasks_with_independent_runner_command_receipt": 0,
        "tasks_with_parent_fail_target_pass_receipt": 0,
        "evaluator_ready_task_count": 0,
    }
    for key, value in expected.items():
        if integer(value) != observations.get(key):
            faults.append(f"expected_observation_mismatch:{key}")
    toolchains = {
        name: trusted_tool_identity(command)
        for name, command in {
            "python": ["python3", "--version"],
            "node": ["node", "--version"],
            "npm": ["npm", "--version"],
            "rustc": ["rustc", "--version"],
            "cargo": ["cargo", "--version"],
            "sandbox_exec": ["sandbox-exec", "-h"],
        }.items()
    }
    if any(not receipt.get("available") for receipt in toolchains.values()):
        faults.append("required_trusted_toolchain_unavailable")

    readiness_faults = [
        "zero_of_62_tasks_have_materialized_dependency_manifests",
        "zero_of_62_tasks_have_materialized_dependency_locks",
        "zero_of_62_tasks_have_transitive_local_source_closure",
        "zero_of_62_tasks_have_independent_runner_command_receipts",
        "python_sandbox_only_node_typescript_rust_unqualified",
        "zero_parent_fail_target_pass_receipts",
    ]
    faults.extend(readiness_faults)
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "RED",
        "state": "EVALUATOR_INSTRUMENT_MATERIALIZATION_REQUIRED",
        "faults": sorted(set(faults)),
        "source_panel_remains_admitted": panel.get("source_panel_admitted") is True,
        "evaluator_execution_authorized": False,
        "python_sandbox_qualified_for_pinned_python_only": python_sandbox.get("trigger_state") == "GREEN",
        "language_task_counts": dict(sorted(language_counts.items())),
        "observations": observations,
        "task_gaps": task_gaps,
        "trusted_toolchains_observed_not_execution_authorized": toolchains,
        "required_next_instrument": {
            "exact_base_and_head_repository_or_transitive_source_closure": True,
            "package_manifest_and_lock_receipts": True,
            "independent_runner_command_derivation": True,
            "dependency_prefetch_then_network_denied_execution": True,
            "install_scripts_disabled_or_sandboxed": True,
            "python_node_typescript_and_rust_sandbox_canaries": True,
            "complete_output_resource_and_cost_receipts": True,
            "parent_must_fail_and_target_must_pass": True,
            "known_good_known_bad_and_transplant_controls": True,
            "inadequate_harness_disposition": "INCONCLUSIVE_EXPERIMENT_NOT_TASK_OR_VCM_FAILURE",
        },
        "parent_target_or_evaluator_executions": 0,
        "candidate_packet_materialization_opened": False,
        "local_model_calls": 0,
        "external_reference_calls": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
        "config": artifact(config_path),
        "reports": {name: artifact(p2a.resolve(str(p2a.mapping(binding).get("path") or ""))) for name, binding in p2a.mapping(config.get("reports")).items()},
        "maximum_inference": config.get("maximum_inference"),
    }


def trusted_tool_identity(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if not executable:
        return {"available": False, "command": command, "path": "", "version_output": ""}
    completed = subprocess.run([executable, *command[1:]], cwd=ROOT, capture_output=True, text=True, check=False)
    output = (completed.stdout or completed.stderr).strip()
    path = Path(executable).resolve()
    return {
        "available": path.is_file(),
        "version_query_succeeded": completed.returncode == 0,
        "command": command,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "",
        "version_output": output,
        "returncode": completed.returncode,
    }


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path) if path.is_file() else ""}


def integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "source_panel_remains_admitted",
        "evaluator_execution_authorized", "python_sandbox_qualified_for_pinned_python_only",
        "language_task_counts", "observations", "parent_target_or_evaluator_executions",
        "local_model_calls", "external_reference_calls", "faults",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
