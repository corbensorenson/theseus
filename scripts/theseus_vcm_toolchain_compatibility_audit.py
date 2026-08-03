#!/usr/bin/env python3
"""Audit declared manifest/runtime compatibility for all locked VCM tasks."""

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

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_toolchain_compatibility_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_toolchain_compatibility_audit.json"


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
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    owner = p2a.resolve(str(config.get("owner") or ""))
    if owner != Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner) != str(config.get("owner_sha256") or ""):
        faults.append("owner_binding_invalid")
    reports: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for name, raw in p2a.mapping(config.get("reports")).items():
        binding = p2a.mapping(raw)
        path = p2a.resolve(str(binding.get("path") or ""))
        paths[name] = path
        if not path.is_file() or p2a.sha256_file(path) != str(binding.get("sha256") or ""):
            faults.append(f"report_binding_invalid:{name}")
            reports[name] = {}
        else:
            reports[name] = p2a.read_json(path)
    plan = reports.get("prefetch_plan", {})
    inventory = reports.get("runner_inventory", {})
    toolchains = reports.get("toolchains", {})
    node = reports.get("node_runtime", {})
    if plan.get("trigger_state") != "GREEN" or len(p2a.dicts(plan.get("schedule"))) != 48:
        faults.append("prefetch_plan_invalid")
    if inventory.get("trigger_state") != "GREEN" or inventory.get("observations", {}).get("task_count") != 62:
        faults.append("runner_inventory_invalid")
    if toolchains.get("trigger_state") != "GREEN" or toolchains.get("observations", {}).get("missing_tool_count") != 0:
        faults.append("toolchain_identity_invalid")
    if node.get("trigger_state") != "GREEN":
        faults.append("node_successor_invalid")
    authority = p2a.mapping(config.get("authority"))
    for key, value in authority.items():
        if value is not (key == "normalized_archive_and_manifest_static_read_authorized"):
            faults.append(f"authority_invalid:{key}")
    profiles = p2a.dicts(config.get("tool_profiles"))
    for profile in profiles:
        if not profile.get("id") or not profile.get("manager") or not profile.get("manager_version"):
            faults.append("tool_profile_invalid")
    profile_ids = [str(row.get("id")) for row in profiles]
    if len(profile_ids) != len(set(profile_ids)):
        faults.append("tool_profile_ids_not_unique")

    inventory_rows = {int(row["index"]): row for row in p2a.dicts(inventory.get("rows"))}
    rows = []
    for planned in p2a.dicts(plan.get("schedule")):
        index = int(planned["index"])
        inventory_row = p2a.mapping(inventory_rows.get(index))
        archive_binding = p2a.mapping(inventory_row.get("target_archive"))
        archive = p2a.resolve(str(archive_binding.get("path") or ""))
        manifest_binding = p2a.mapping(p2a.mapping(planned.get("governing_lock")).get("manifest"))
        manifest_path = str(manifest_binding.get("path") or "")
        payload, manifest_faults = read_manifest(archive, str(archive_binding.get("sha256") or ""), manifest_path, str(manifest_binding.get("sha256") or ""))
        faults.extend(f"task_{index}:{fault}" for fault in manifest_faults)
        requirements, parse_faults = parse_requirements(str(planned.get("manager") or ""), manifest_path, payload)
        evaluations = [evaluate_profile(profile, requirements) for profile in profiles if profile.get("manager") == planned.get("manager")]
        compatible = [row for row in evaluations if row["compatible"] is True]
        unknown = [row for row in evaluations if row["compatible"] is None]
        declared_count = len(requirements)
        if compatible:
            state = "COMPATIBLE_DECLARED_REQUIREMENTS" if declared_count else "NO_DECLARED_VERSION_REQUIREMENTS_TOOL_AVAILABLE"
        elif unknown or parse_faults:
            state = "COMPATIBILITY_UNRESOLVED_REQUIREMENT_SYNTAX"
        else:
            state = "INCOMPATIBLE_DECLARED_REQUIREMENTS"
        rows.append({
            "index": index,
            "schedule_ordinal": planned.get("schedule_ordinal"),
            "repository": planned.get("repository"),
            "manager": planned.get("manager"),
            "manifest": {"path": manifest_path, "sha256": manifest_binding.get("sha256")},
            "requirements": requirements,
            "requirement_parse_faults": parse_faults,
            "profile_evaluations": evaluations,
            "compatible_profile_ids": [row["profile_id"] for row in compatible],
            "state": state,
            "dependency_prefetch_authorized": False,
            "repository_execution_authorized": False,
        })
    observations = {
        "locked_task_count": len(rows),
        "compatible_declared_requirement_task_count": sum(row["state"] == "COMPATIBLE_DECLARED_REQUIREMENTS" for row in rows),
        "no_declared_version_requirement_task_count": sum(row["state"] == "NO_DECLARED_VERSION_REQUIREMENTS_TOOL_AVAILABLE" for row in rows),
        "incompatible_declared_requirement_task_count": sum(row["state"] == "INCOMPATIBLE_DECLARED_REQUIREMENTS" for row in rows),
        "unresolved_requirement_syntax_task_count": sum(row["state"] == "COMPATIBILITY_UNRESOLVED_REQUIREMENT_SYNTAX" for row in rows),
        "dependency_prefetch_executions": 0,
        "repository_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
    }
    if len(rows) != 48 or len({row["index"] for row in rows}) != 48:
        faults.append("locked_task_denominator_invalid")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "FORTY_EIGHT_LOCKED_TASK_TOOLCHAIN_COMPATIBILITY_CLASSIFIED" if not faults else "TOOLCHAIN_COMPATIBILITY_AUDIT_FAILED",
        "faults": sorted(set(faults)),
        "config": artifact(config_path),
        "reports": {name: artifact(path) for name, path in paths.items()},
        "tool_profiles": profiles,
        "observations": observations,
        "rows": rows,
        "dependency_prefetch_executions": 0,
        "repository_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "maximum_inference": config.get("maximum_inference"),
    }


def read_manifest(archive: Path, archive_sha256: str, manifest_path: str, manifest_sha256: str) -> tuple[bytes, list[str]]:
    faults = []
    if not archive.is_file() or p2a.sha256_file(archive) != archive_sha256:
        return b"", ["archive_binding_invalid"]
    payload = b""
    with tarfile.open(archive, "r:gz") as handle:
        candidates = [
            member for member in handle.getmembers()
            if member.isfile() and "/" in member.name and member.name.split("/", 1)[1] == manifest_path
        ]
        if len(candidates) != 1:
            faults.append("manifest_member_not_unique")
        else:
            extracted = handle.extractfile(candidates[0])
            payload = extracted.read() if extracted else b""
            if hashlib.sha256(payload).hexdigest() != manifest_sha256:
                faults.append("manifest_digest_mismatch")
    return payload, faults


def parse_requirements(manager: str, manifest_path: str, payload: bytes) -> tuple[list[dict[str, str]], list[str]]:
    requirements: list[dict[str, str]] = []
    faults: list[str] = []
    try:
        if PurePosixPath(manifest_path).name == "package.json":
            value = json.loads(payload)
            package_manager = str(value.get("packageManager") or "")
            if package_manager:
                match = re.match(r"^([a-zA-Z0-9_-]+)@([^+]+)", package_manager)
                if not match:
                    faults.append("package_manager_field_unparsed")
                else:
                    requirements.append({"kind": "manager", "name": match.group(1), "expression": match.group(2), "source": "packageManager"})
            engines = p2a.mapping(value.get("engines"))
            if manager in engines:
                requirements.append({"kind": "manager", "name": manager, "expression": str(engines[manager]), "source": f"engines.{manager}"})
            if manager in {"npm", "pnpm", "yarn"} and engines.get("node"):
                requirements.append({"kind": "runtime", "name": "node", "expression": str(engines["node"]), "source": "engines.node"})
            volta = p2a.mapping(value.get("volta"))
            if manager in {"npm", "pnpm", "yarn"} and volta.get("node"):
                requirements.append({"kind": "runtime", "name": "node", "expression": str(volta["node"]), "source": "volta.node"})
            if volta.get(manager):
                requirements.append({"kind": "manager", "name": manager, "expression": str(volta[manager]), "source": f"volta.{manager}"})
        elif PurePosixPath(manifest_path).name == "Cargo.toml":
            value = tomllib.loads(payload.decode("utf-8"))
            package = p2a.mapping(value.get("package"))
            workspace_package = p2a.mapping(p2a.mapping(value.get("workspace")).get("package"))
            rust_version = package.get("rust-version") or workspace_package.get("rust-version")
            if rust_version:
                requirements.append({"kind": "runtime_minimum", "name": "rustc", "expression": str(rust_version), "source": "package.rust-version"})
        elif PurePosixPath(manifest_path).name == "pyproject.toml":
            value = tomllib.loads(payload.decode("utf-8"))
            project = p2a.mapping(value.get("project"))
            if project.get("requires-python"):
                requirements.append({"kind": "python", "name": "python", "expression": str(project["requires-python"]), "source": "project.requires-python"})
            uv = p2a.mapping(p2a.mapping(value.get("tool")).get("uv"))
            if uv.get("required-version"):
                requirements.append({"kind": "manager", "name": "uv", "expression": str(uv["required-version"]), "source": "tool.uv.required-version"})
    except (json.JSONDecodeError, tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        faults.append(f"manifest_parse_failed:{type(exc).__name__}")
    return requirements, faults


def evaluate_profile(profile: dict[str, Any], requirements: list[dict[str, str]]) -> dict[str, Any]:
    checks = []
    for requirement in requirements:
        name = requirement["name"]
        expression = requirement["expression"]
        if requirement["kind"] == "manager" and name != profile.get("manager"):
            result: bool | None = False
        elif requirement["kind"] == "manager":
            result = semver_satisfies(str(profile.get("manager_version")), expression)
        elif requirement["kind"] == "runtime":
            result = semver_satisfies(str(p2a.mapping(profile.get("runtime_versions")).get(name) or ""), expression)
        elif requirement["kind"] == "runtime_minimum":
            result = semver_satisfies(str(p2a.mapping(profile.get("runtime_versions")).get(name) or ""), f">={expression}")
        elif requirement["kind"] == "python":
            result = python_satisfies(str(p2a.mapping(profile.get("runtime_versions")).get(name) or ""), expression)
        else:
            result = None
        checks.append({**requirement, "satisfied": result})
    compatible: bool | None
    if any(row["satisfied"] is False for row in checks):
        compatible = False
    elif any(row["satisfied"] is None for row in checks):
        compatible = None
    else:
        compatible = True
    return {"profile_id": profile.get("id"), "compatible": compatible, "checks": checks}


def semver_satisfies(version: str, expression: str) -> bool | None:
    parsed = semver_tuple(version)
    if parsed is None:
        return None
    branches = [branch.strip() for branch in expression.strip().split("||")]
    results = [semver_branch(parsed, branch) for branch in branches]
    if any(result is True for result in results):
        return True
    if any(result is None for result in results):
        return None
    return False


def semver_branch(version: tuple[int, int, int], branch: str) -> bool | None:
    if branch in {"", "*", "x", "X"}:
        return True
    hyphen = re.fullmatch(r"\s*([vV]?\d+(?:\.\d+){0,2})\s+-\s+([vV]?\d+(?:\.\d+){0,2})\s*", branch)
    if hyphen:
        low, high = semver_tuple(hyphen.group(1)), semver_tuple(hyphen.group(2))
        return None if low is None or high is None else low <= version <= high
    tokens = branch.replace(",", " ").split()
    checks = []
    for token in tokens:
        result = semver_token(version, token)
        if result is None:
            return None
        checks.append(result)
    return all(checks)


def semver_token(version: tuple[int, int, int], token: str) -> bool | None:
    match = re.fullmatch(r"(>=|<=|>|<|=|\^|~)?[vV]?([0-9xX*]+)(?:\.([0-9xX*]+))?(?:\.([0-9xX*]+))?(?:-[0-9A-Za-z.-]+)?", token)
    if not match:
        return None
    operator = match.group(1) or ""
    raw_parts = [match.group(2), match.group(3), match.group(4)]
    wildcard = next((i for i, value in enumerate(raw_parts) if value in {None, "x", "X", "*"}), 3)
    parts = tuple(int(value) if value not in {None, "x", "X", "*"} else 0 for value in raw_parts)
    if operator == ">=": return version >= parts
    if operator == "<=": return version <= parts
    if operator == ">": return version > parts
    if operator == "<": return version < parts
    if operator == "=": return version == parts
    if operator == "^":
        upper = (parts[0] + 1, 0, 0) if parts[0] else (0, parts[1] + 1, 0) if parts[1] else (0, 0, parts[2] + 1)
        return parts <= version < upper
    if operator == "~":
        upper = (parts[0], parts[1] + 1, 0) if match.group(3) is not None else (parts[0] + 1, 0, 0)
        return parts <= version < upper
    if wildcard == 0: return True
    if wildcard == 1: return version[0] == parts[0]
    if wildcard == 2: return version[:2] == parts[:2]
    return version == parts


def semver_tuple(value: str) -> tuple[int, int, int] | None:
    match = re.search(r"[vV]?(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
    return tuple(int(match.group(i) or 0) for i in range(1, 4)) if match else None


def python_satisfies(version: str, expression: str) -> bool | None:
    parsed = semver_tuple(version)
    if parsed is None:
        return None
    try:
        return Version(".".join(map(str, parsed))) in SpecifierSet(expression)
    except (InvalidSpecifier, InvalidVersion):
        return None


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path) if path.is_file() else ""}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "observations", "dependency_prefetch_executions",
        "repository_executions", "candidate_or_control_calls", "external_reference_calls", "faults",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
