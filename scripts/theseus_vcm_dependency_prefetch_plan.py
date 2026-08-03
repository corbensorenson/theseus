#!/usr/bin/env python3
"""Freeze a bounded, evaluator-ecosystem VCM dependency prefetch schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_dependency_prefetch_plan_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_dependency_prefetch_plan.json"
LOCK_MANAGER = {
    "package-lock.json": "npm", "npm-shrinkwrap.json": "npm",
    "pnpm-lock.yaml": "pnpm", "yarn.lock": "yarn",
    "bun.lock": "bun", "bun.lockb": "bun", "uv.lock": "uv",
    "cargo.lock": "cargo",
}
PRIMARY_SUFFIXES = {
    "Python": {".py"}, "JavaScript": {".js", ".mjs", ".cjs"},
    "TypeScript": {".ts", ".tsx", ".mts", ".cts"}, "Rust": {".rs"},
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
    reports, report_paths = {}, {}
    for name, raw in p2a.mapping(config.get("reports")).items():
        binding = p2a.mapping(raw)
        path = p2a.resolve(str(binding.get("path") or ""))
        report_paths[name] = path
        if not path.is_file() or p2a.sha256_file(path) != str(binding.get("sha256") or ""):
            faults.append(f"report_binding_invalid:{name}")
            reports[name] = {}
        else:
            reports[name] = p2a.read_json(path)
    dependency = reports.get("dependency_classes", {})
    inventory = reports.get("runner_inventory", {})
    toolchains = reports.get("toolchains", {})
    if dependency.get("trigger_state") != "GREEN" or dependency.get("observations", {}).get("task_count") != 62:
        faults.append("dependency_classes_not_green")
    if inventory.get("trigger_state") != "GREEN" or inventory.get("observations", {}).get("task_count") != 62:
        faults.append("runner_inventory_not_green")
    if toolchains.get("trigger_state") != "GREEN" or toolchains.get("observations", {}).get("missing_tool_count") != 0:
        faults.append("toolchains_not_complete")
    authority = p2a.mapping(config.get("authority"))
    if authority.get("normalized_archive_static_read_authorized") is not True or any(
        value is not False for key, value in authority.items() if key != "normalized_archive_static_read_authorized"
    ):
        faults.append("authority_boundary_invalid")
    inventory_rows = {int(row["index"]): row for row in p2a.dicts(inventory.get("rows"))}
    rows = []
    for dependency_row in p2a.dicts(dependency.get("rows")):
        index = int(dependency_row["index"])
        inventory_row = p2a.mapping(inventory_rows.get(index))
        archive_binding = p2a.mapping(inventory_row.get("target_archive"))
        archive = p2a.resolve(str(archive_binding.get("path") or ""))
        if not archive.is_file() or p2a.sha256_file(archive) != str(archive_binding.get("sha256") or ""):
            faults.append(f"archive_binding_invalid:{index}")
            members = {}
        else:
            members = read_members(archive)
        dependency_class = str(dependency_row.get("dependency_class") or "")
        if dependency_class == "EXACT_EVALUATOR_ECOSYSTEM_LOCK_RECEIPT_PRESENT":
            lock = choose_lock(dependency_row, inventory_row, members)
            if not lock:
                faults.append(f"governing_lock_not_selected:{index}")
                manager, root, estimate, command = "", ".", 0, []
            else:
                lock_path = str(lock["path"])
                manager = LOCK_MANAGER.get(PurePosixPath(lock_path).name.lower(), "")
                root = PurePosixPath(lock_path).parent.as_posix()
                payload = members.get(lock_path, b"")
                estimate = estimate_packages(manager, payload)
                command = prefetch_command(manager, index, root, config)
                if not manager or not command:
                    faults.append(f"manager_or_command_absent:{index}")
            rows.append(base_row(dependency_row, "LOCKED_PREFETCH_PLANNED") | {
                "governing_lock": lock,
                "manager": manager,
                "working_directory": root,
                "estimated_locked_package_count": estimate["count"],
                "package_estimate_known": estimate["known"],
                "package_estimate_method": estimate["method"],
                "prefetch_command_template": command,
                "prefetch_planned": True,
                "prefetch_executed": False,
            })
        elif dependency_row.get("lock_not_required_for_scoped_evaluator") is True:
            rows.append(base_row(dependency_row, "NO_PROJECT_LOCK_REQUIRED_STATIC_CLOSURE") | {
                "governing_lock": None, "manager": "trusted_runtime_or_harness",
                "working_directory": ".", "estimated_locked_package_count": 0,
                "package_estimate_known": True, "package_estimate_method": "not_applicable",
                "prefetch_command_template": [], "prefetch_planned": False,
                "prefetch_executed": False,
            })
        else:
            rows.append(base_row(dependency_row, "IMMUTABLE_RESOLUTION_MUST_BE_FROZEN_BEFORE_PREFETCH") | {
                "governing_lock": None, "manager": "resolver_required",
                "working_directory": ".", "estimated_locked_package_count": 0,
                "package_estimate_known": True, "package_estimate_method": "not_applicable",
                "prefetch_command_template": [], "prefetch_planned": False,
                "prefetch_executed": False,
            })
    schedule = sorted(
        [row for row in rows if row["prefetch_planned"]],
        key=lambda row: (
            not bool(row["package_estimate_known"]),
            int(row["estimated_locked_package_count"]),
            int(row["index"]),
        ),
    )
    for ordinal, row in enumerate(schedule, 1):
        row["schedule_ordinal"] = ordinal
    ordinal_by_index = {row["index"]: row["schedule_ordinal"] for row in schedule}
    for row in rows:
        row["schedule_ordinal"] = ordinal_by_index.get(row["index"])
    observations = {
        "task_count": len(rows),
        "locked_prefetch_planned_task_count": sum(row["prefetch_planned"] for row in rows),
        "no_project_lock_required_task_count": sum(row["state"] == "NO_PROJECT_LOCK_REQUIRED_STATIC_CLOSURE" for row in rows),
        "immutable_resolution_required_task_count": sum(row["state"] == "IMMUTABLE_RESOLUTION_MUST_BE_FROZEN_BEFORE_PREFETCH" for row in rows),
        "tasks_with_governing_lock": sum(bool(row["governing_lock"]) for row in rows),
        "tasks_with_prefetch_command": sum(bool(row["prefetch_command_template"]) for row in rows),
        "tasks_with_known_package_estimate": sum(
            row["prefetch_planned"] and row["package_estimate_known"] for row in rows
        ),
        "tasks_with_unknown_package_estimate": sum(
            row["prefetch_planned"] and not row["package_estimate_known"] for row in rows
        ),
        "prefetch_executions": 0,
        "parent_target_or_evaluator_executions": 0,
    }
    for key, value in p2a.mapping(config.get("expected_observations")).items():
        if observations.get(key) != value:
            faults.append(f"expected_observation_mismatch:{key}")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "BOUNDED_EVALUATOR_DEPENDENCY_PREFETCH_SCHEDULE_FROZEN" if not faults else "DEPENDENCY_PREFETCH_PLAN_INVALID",
        "faults": sorted(set(faults)),
        "config": artifact(config_path),
        "reports": {name: artifact(path) for name, path in report_paths.items()},
        "observations": observations,
        "resource_policy": p2a.mapping(config.get("resource_policy")),
        "manager_policy": p2a.mapping(config.get("manager_policy")),
        "schedule": schedule,
        "rows": rows,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "parent_target_or_evaluator_executions": 0,
        "maximum_inference": config.get("maximum_inference"),
    }


def choose_lock(dependency_row: dict[str, Any], inventory_row: dict[str, Any], members: dict[str, bytes]) -> dict[str, Any] | None:
    locks = p2a.dicts(dependency_row.get("evaluator_ecosystem_lock_receipts"))
    language = str(dependency_row.get("query_language") or "")
    primary = [
        path for path in p2a.strings(dependency_row.get("selected_verifier_paths"))
        if PurePosixPath(path).suffix.lower() in PRIMARY_SUFFIXES.get(language, set())
    ]
    commands = [str(row.get("command") or "").lower() for row in p2a.dicts(inventory_row.get("independent_runner_receipts"))]
    scored = []
    for lock in locks:
        path = str(lock.get("path") or "")
        root = PurePosixPath(path).parent.as_posix()
        depth = 0 if root == "." else len(PurePosixPath(root).parts) if any(candidate.startswith(root + "/") for candidate in primary) else -1
        manager = LOCK_MANAGER.get(PurePosixPath(path).name.lower(), "")
        runner_score = sum(command.startswith(manager + " ") or f" {manager} " in command for command in commands)
        manifest = root_manifest_identity(root, manager, members)
        scored.append((depth, runner_score, bool(manifest), path, lock))
    if not scored:
        return None
    scored.sort(key=lambda row: (-row[0], -row[1], not row[2], row[3]))
    selected = dict(scored[0][4])
    selected["selection_depth"] = scored[0][0]
    selected["runner_manager_match_count"] = scored[0][1]
    selected["manifest"] = root_manifest_identity(PurePosixPath(selected["path"]).parent.as_posix(), LOCK_MANAGER.get(PurePosixPath(selected["path"]).name.lower(), ""), members)
    return selected


def root_manifest_identity(root: str, manager: str, members: dict[str, bytes]) -> dict[str, str] | None:
    name = "Cargo.toml" if manager == "cargo" else "pyproject.toml" if manager == "uv" else "package.json"
    path = name if root == "." else f"{root}/{name}"
    payload = members.get(path)
    return {"path": path, "sha256": hashlib.sha256(payload).hexdigest()} if payload is not None else None


def estimate_packages(manager: str, payload: bytes) -> dict[str, Any]:
    text = payload.decode("utf-8", "replace")
    try:
        if manager == "npm":
            return estimate(max(0, len(json.loads(text).get("packages", {})) - 1), "npm_packages_map")
        if manager in {"uv", "cargo"}:
            return estimate(len(tomllib.loads(text).get("package", [])), f"{manager}_toml_package_array")
        if manager == "yarn":
            return estimate(sum(1 for line in text.splitlines() if line.startswith("  version ")), "yarn_v1_version_entries")
        if manager == "pnpm":
            return estimate(
                sum(1 for line in text.splitlines() if re.match(r"^  ['\"]?/?[^\s].*:$", line) and "settings:" not in line),
                "pnpm_top_level_lock_entries",
            )
        if manager == "bun":
            # Bun's text lockfile is JSONC-like (not strict JSON): package rows
            # are stable four-space keys whose values are compact arrays.
            count = sum(1 for line in text.splitlines() if re.match(r'^    ".+": \[', line))
            return estimate(count, "bun_text_lock_package_arrays") if count else unknown_estimate("bun_text_lock_unparsed")
    except (json.JSONDecodeError, tomllib.TOMLDecodeError):
        return unknown_estimate(f"{manager}_lock_parse_failed")
    return unknown_estimate(f"{manager}_estimate_unsupported")


def estimate(count: int, method: str) -> dict[str, Any]:
    return {"count": count, "known": True, "method": method}


def unknown_estimate(method: str) -> dict[str, Any]:
    return {"count": 0, "known": False, "method": method}


def prefetch_command(manager: str, index: int, root: str, config: dict[str, Any]) -> list[str]:
    store = f"runtime/vcm_evaluator/dependency_store/{manager}/task-{index:02d}"
    if manager == "npm": return ["node", "npm-cli.js", "ci", "--ignore-scripts", "--no-audit", "--no-fund", "--cache", store]
    if manager == "pnpm": return ["node", "pnpm.cjs", "fetch", "--frozen-lockfile", "--store-dir", store]
    if manager == "yarn": return ["node", "yarn.js", "install", "--frozen-lockfile", "--ignore-scripts", "--non-interactive", "--cache-folder", store]
    if manager == "bun": return ["bun", "install", "--frozen-lockfile", "--ignore-scripts", "--cache-dir", store]
    if manager == "uv": return ["uv", "sync", "--frozen", "--no-install-project", "--no-install-workspace", "--cache-dir", store]
    if manager == "cargo": return ["cargo", "fetch", "--locked", "--config", f"net.git-fetch-with-cli=false"]
    return []


def base_row(row: dict[str, Any], state: str) -> dict[str, Any]:
    return {"index": row.get("index"), "repository": row.get("repository"), "query_language": row.get("query_language"), "dependency_class": row.get("dependency_class"), "state": state, "repository_execution_authorized": False, "evaluator_execution_ready": False}


def read_members(archive: Path) -> dict[str, bytes]:
    result = {}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if member.isfile() and "/" in member.name:
                extracted = handle.extractfile(member)
                result[member.name.split("/", 1)[1]] = extracted.read() if extracted else b""
    return result


def artifact(path: Path) -> dict[str, str]: return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path) if path.is_file() else ""}
def summary(report: dict[str, Any]) -> dict[str, Any]: return {key: report.get(key) for key in ("trigger_state", "state", "observations", "parent_target_or_evaluator_executions", "candidate_or_control_calls", "external_reference_calls", "faults")}


if __name__ == "__main__": raise SystemExit(main())
