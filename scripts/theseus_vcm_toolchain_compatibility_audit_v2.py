#!/usr/bin/env python3
"""Reclassify the frozen VCM toolchain audit with exact pnpm 10.32.1."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_toolchain_compatibility_audit as base  # noqa: E402

POLICY = "project_theseus_vcm_toolchain_compatibility_audit_v2"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_toolchain_compatibility_audit_v2.json"


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

    predecessor_path, predecessor = bound_report(config, "predecessor_audit", faults)
    bootstrap_path, bootstrap = bound_report(config, "pnpm_bootstrap", faults)
    if predecessor.get("trigger_state") != "GREEN" or predecessor.get("observations", {}).get("locked_task_count") != 48:
        faults.append("predecessor_audit_invalid")
    if bootstrap.get("trigger_state") != "GREEN" or bootstrap.get("state") != "EXACT_PNPM_10_32_1_MATERIALIZED_AND_VERSION_QUALIFIED":
        faults.append("pnpm_bootstrap_invalid")
    if bootstrap.get("dependency_installations") != 0 or bootstrap.get("repository_executions") != 0:
        faults.append("pnpm_bootstrap_scope_invalid")

    profile = p2a.mapping(config.get("added_profile"))
    target = p2a.mapping(p2a.mapping(bootstrap.get("receipt")).get("target"))
    cli = p2a.mapping(target.get("pnpm_cli"))
    if (
        profile.get("id") != "node22_20_pnpm10_32_1"
        or profile.get("manager") != "pnpm"
        or profile.get("manager_version") != "10.32.1"
        or p2a.mapping(profile.get("runtime_versions")).get("node") != "22.20.0"
        or cli.get("path") != profile.get("cli_path")
        or cli.get("sha256") != profile.get("cli_sha256")
        or target.get("version") != "10.32.1"
    ):
        faults.append("added_profile_identity_invalid")

    authority = p2a.mapping(config.get("authority"))
    for key, value in authority.items():
        if value is not (key == "source_bound_static_reclassification_authorized"):
            faults.append(f"authority_invalid:{key}")

    rows = copy.deepcopy(p2a.dicts(predecessor.get("rows")))
    for row in rows:
        if row.get("manager") != "pnpm":
            continue
        evaluation = base.evaluate_profile(profile, p2a.dicts(row.get("requirements")))
        evaluations = p2a.dicts(row.get("profile_evaluations")) + [evaluation]
        compatible = [item for item in evaluations if item.get("compatible") is True]
        unknown = [item for item in evaluations if item.get("compatible") is None]
        declared_count = len(p2a.dicts(row.get("requirements")))
        if compatible:
            state = "COMPATIBLE_DECLARED_REQUIREMENTS" if declared_count else "NO_DECLARED_VERSION_REQUIREMENTS_TOOL_AVAILABLE"
        elif unknown or row.get("requirement_parse_faults"):
            state = "COMPATIBILITY_UNRESOLVED_REQUIREMENT_SYNTAX"
        else:
            state = "INCOMPATIBLE_DECLARED_REQUIREMENTS"
        row["profile_evaluations"] = evaluations
        row["compatible_profile_ids"] = [item["profile_id"] for item in compatible]
        row["state"] = state

    observations = {
        "locked_task_count": len(rows),
        "compatible_declared_requirement_task_count": sum(row["state"] == "COMPATIBLE_DECLARED_REQUIREMENTS" for row in rows),
        "no_declared_version_requirement_task_count": sum(row["state"] == "NO_DECLARED_VERSION_REQUIREMENTS_TOOL_AVAILABLE" for row in rows),
        "incompatible_declared_requirement_task_count": sum(row["state"] == "INCOMPATIBLE_DECLARED_REQUIREMENTS" for row in rows),
        "unresolved_requirement_syntax_task_count": sum(row["state"] == "COMPATIBILITY_UNRESOLVED_REQUIREMENT_SYNTAX" for row in rows),
        "changed_task_indices": changed_indices(p2a.dicts(predecessor.get("rows")), rows),
        "dependency_prefetch_executions": 0,
        "repository_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
    }
    if observations["changed_task_indices"] != [7]:
        faults.append("change_scope_invalid")
    if len(rows) != 48 or len({int(row["index"]) for row in rows}) != 48:
        faults.append("locked_task_denominator_invalid")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "FORTY_EIGHT_LOCKED_TASK_TOOLCHAIN_COMPATIBILITY_RECLASSIFIED_WITH_PNPM_10_32_1" if not faults else "TOOLCHAIN_COMPATIBILITY_RECLASSIFICATION_FAILED",
        "faults": sorted(set(faults)),
        "config": artifact(config_path),
        "predecessor_audit": artifact(predecessor_path),
        "pnpm_bootstrap": artifact(bootstrap_path),
        "added_profile": profile,
        "observations": observations,
        "rows": rows,
        "dependency_prefetch_executions": 0,
        "repository_executions": 0,
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "maximum_inference": config.get("maximum_inference"),
    }


def bound_report(config: dict[str, Any], name: str, faults: list[str]) -> tuple[Path, dict[str, Any]]:
    binding = p2a.mapping(p2a.mapping(config.get("reports")).get(name))
    path = p2a.resolve(str(binding.get("path") or ""))
    if not path.is_file() or p2a.sha256_file(path) != str(binding.get("sha256") or ""):
        faults.append(f"report_binding_invalid:{name}")
        return path, {}
    return path, p2a.read_json(path)


def changed_indices(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[int]:
    states = {int(row["index"]): row.get("state") for row in before}
    return sorted(int(row["index"]) for row in after if states.get(int(row["index"])) != row.get("state"))


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path) if path.is_file() else ""}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "faults": report.get("faults"),
        "observations": report.get("observations"),
        "dependency_prefetch_executions": report.get("dependency_prefetch_executions"),
        "repository_executions": report.get("repository_executions"),
        "candidate_or_control_calls": report.get("candidate_or_control_calls"),
        "external_reference_calls": report.get("external_reference_calls"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
