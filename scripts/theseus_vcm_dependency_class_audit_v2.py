#!/usr/bin/env python3
"""Correct VCM dependency classes with evaluator-ecosystem lock selection."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_dependency_class_audit_v2"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_dependency_class_audit_v2.json"
ECOSYSTEM_LOCKS = {
    "Python": {"uv.lock", "poetry.lock", "pdm.lock", "pipfile.lock"},
    "JavaScript": {"package-lock.json", "npm-shrinkwrap.json", "pnpm-lock.yaml", "yarn.lock", "bun.lock", "bun.lockb"},
    "TypeScript": {"package-lock.json", "npm-shrinkwrap.json", "pnpm-lock.yaml", "yarn.lock", "bun.lock", "bun.lockb"},
    "Rust": {"cargo.lock"},
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
    reports = {}
    report_paths = {}
    for name, raw in p2a.mapping(config.get("reports")).items():
        binding = p2a.mapping(raw)
        path = p2a.resolve(str(binding.get("path") or ""))
        report_paths[name] = path
        if not path.is_file() or p2a.sha256_file(path) != str(binding.get("sha256") or ""):
            faults.append(f"report_binding_invalid:{name}")
            reports[name] = {}
        else:
            reports[name] = p2a.read_json(path)
    prior = reports.get("dependency_class_v1", {})
    inventory = reports.get("runner_inventory", {})
    if prior.get("trigger_state") != "GREEN" or prior.get("observations", {}).get("task_count") != 62:
        faults.append("dependency_class_v1_not_green")
    if inventory.get("trigger_state") != "GREEN" or inventory.get("observations", {}).get("task_count") != 62:
        faults.append("runner_inventory_not_green")
    authority = p2a.mapping(config.get("authority"))
    if authority.get("normalized_archive_static_read_authorized") is not True or any(
        value is not False for key, value in authority.items() if key != "normalized_archive_static_read_authorized"
    ):
        faults.append("authority_boundary_invalid")
    inventory_rows = {int(row["index"]): row for row in p2a.dicts(inventory.get("rows"))}
    rows = []
    changed_indices = []
    for row in p2a.dicts(prior.get("rows")):
        index = int(row["index"])
        language = str(row.get("query_language") or "")
        allowed_locks = ECOSYSTEM_LOCKS.get(language, set())
        ecosystem_locks = [
            lock for lock in p2a.dicts(row.get("relevant_lock_receipts"))
            if PurePosixPath(str(lock.get("path") or "")).name.lower() in allowed_locks
        ]
        inventory_row = p2a.mapping(inventory_rows.get(index))
        archive_binding = p2a.mapping(inventory_row.get("target_archive"))
        archive = p2a.resolve(str(archive_binding.get("path") or ""))
        if not archive.is_file() or p2a.sha256_file(archive) != str(archive_binding.get("sha256") or ""):
            faults.append(f"target_archive_binding_invalid:{index}")
            local_aliases = set()
        else:
            local_aliases = local_python_aliases(archive) if language == "Python" else set()
        prior_external = set(p2a.strings(row.get("external_dependencies_excluding_harness")))
        external = sorted(prior_external - local_aliases)
        eliminated = sorted(prior_external & local_aliases)
        if ecosystem_locks:
            dependency_class = "EXACT_EVALUATOR_ECOSYSTEM_LOCK_RECEIPT_PRESENT"
            immutable = False
            waived = False
        elif external or p2a.mapping(row.get("static_evaluator_closure")).get("unsupported_static_language"):
            dependency_class = "IMMUTABLE_RESOLUTION_REQUIRED"
            immutable = True
            waived = False
        else:
            dependency_class = "LOCK_NOT_REQUIRED_FOR_STATIC_EVALUATOR_CLOSURE"
            immutable = False
            waived = True
        prior_bucket = (
            "exact_lock" if row.get("dependency_class") == "EXACT_LOCK_RECEIPT_PRESENT"
            else "immutable" if row.get("immutable_resolution_required_before_execution") is True
            else "waived"
        )
        new_bucket = "exact_lock" if ecosystem_locks else "immutable" if immutable else "waived"
        if new_bucket != prior_bucket:
            changed_indices.append(index)
        rows.append({
            **row,
            "all_relevant_lock_receipts": p2a.dicts(row.get("relevant_lock_receipts")),
            "evaluator_ecosystem_lock_receipts": ecosystem_locks,
            "relevant_lock_receipts": ecosystem_locks,
            "local_aliases_eliminated_from_external_dependencies": eliminated,
            "external_dependencies_excluding_harness": external,
            "dependency_class": dependency_class,
            "lock_not_required_for_scoped_evaluator": waived,
            "immutable_resolution_required_before_execution": immutable,
            "dependency_resolution_performed": False,
            "repository_execution_performed": False,
            "evaluator_execution_ready": False,
        })
    observations = {
        "task_count": len(rows),
        "tasks_with_exact_evaluator_ecosystem_lock_receipt": sum(row["dependency_class"] == "EXACT_EVALUATOR_ECOSYSTEM_LOCK_RECEIPT_PRESENT" for row in rows),
        "tasks_lock_not_required_for_static_evaluator_closure": sum(row["lock_not_required_for_scoped_evaluator"] for row in rows),
        "tasks_requiring_immutable_resolution": sum(row["immutable_resolution_required_before_execution"] for row in rows),
        "tasks_dependency_classified": len(rows),
        "tasks_evaluator_execution_ready": 0,
        "dependency_resolutions": 0,
        "parent_target_or_evaluator_executions": 0,
    }
    expected_changed = [int(value) for value in config.get("expected_changed_indices", []) if isinstance(value, int) and not isinstance(value, bool)]
    if sorted(changed_indices) != expected_changed:
        faults.append("changed_indices_mismatch")
    for key, value in p2a.mapping(config.get("expected_observations")).items():
        if observations.get(key) != value:
            faults.append(f"expected_observation_mismatch:{key}")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "EVALUATOR_ECOSYSTEM_DEPENDENCY_CLASSES_BOUND" if not faults else "DEPENDENCY_CLASS_SUCCESSOR_INVALID",
        "faults": sorted(set(faults)),
        "config": artifact(config_path),
        "reports": {name: artifact(path) for name, path in report_paths.items()},
        "observations": observations,
        "changed_indices": sorted(changed_indices),
        "rows": rows,
        "correction_policy": {
            "locks_must_match_evaluator_ecosystem": True,
            "repository_local_python_aliases_are_not_third_party_dependencies": True,
            "unrelated_ecosystem_locks_do_not_authorize_resolution": True,
            "dependency_or_repository_execution": False,
        },
        "candidate_or_control_calls": 0,
        "external_reference_calls": 0,
        "parent_target_or_evaluator_executions": 0,
        "maximum_inference": config.get("maximum_inference"),
    }


def local_python_aliases(archive: Path) -> set[str]:
    aliases = set()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if not member.isfile() or "/" not in member.name:
                continue
            relative = member.name.split("/", 1)[1]
            path = PurePosixPath(relative)
            if path.suffix.lower() != ".py":
                continue
            aliases.add(path.stem)
            if path.parts:
                aliases.add(path.parts[0])
            for marker in ("src", "lib"):
                if marker in path.parts:
                    position = path.parts.index(marker)
                    if position + 1 < len(path.parts):
                        aliases.add(path.parts[position + 1])
    return aliases


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path) if path.is_file() else ""}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "observations", "changed_indices",
        "parent_target_or_evaluator_executions", "candidate_or_control_calls",
        "external_reference_calls", "faults",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
