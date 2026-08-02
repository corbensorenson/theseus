#!/usr/bin/env python3
"""Audit the prospectively sealed all-new P4-v2r2-r2 source registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402


REGISTRY = ROOT / "configs" / "theseus_p4v2r2r2_task_sources.json"
ALLOWED_LICENSES = {"Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "MIT"}


def prior_repositories() -> set[str]:
    repositories: set[str] = set()
    for path in sorted((ROOT / "configs").glob("theseus_*task_sources.json")):
        if path == REGISTRY:
            continue
        value = p2a.read_json(path)
        repositories.update(
            str(item).lower()
            for item in p2a.strings(value.get("source_disjoint_from_repositories"))
        )
        for row in p2a.dicts(value.get("tasks")):
            repository = str(row.get("repository") or "").lower()
            if repository:
                repositories.add(repository)
        replacement = p2a.mapping(value.get("replacement_task"))
        repository = str(replacement.get("repository") or "").lower()
        if repository:
            repositories.add(repository)
    return repositories


def audit(path: Path = REGISTRY) -> dict[str, Any]:
    value = p2a.read_json(path)
    faults: list[str] = []
    if value.get("policy") != "project_theseus_p4v2r2r2_online_source_selection_v1":
        faults.append("registry_policy_invalid")
    if value.get("state") != "FIXED_BEFORE_ARCHIVE_FETCH_PARENT_TARGET_EXECUTION_OR_CANDIDATE_GENERATION":
        faults.append("registry_not_prospectively_fixed")

    instrument_path = p2a.resolve(str(value.get("instrument") or ""))
    if (
        not instrument_path.is_file()
        or p2a.sha256_file(instrument_path) != str(value.get("instrument_sha256") or "")
    ):
        faults.append("instrument_binding_invalid")
    predecessor_path = p2a.resolve(
        str(value.get("predecessor_terminal_disposition") or "")
    )
    if (
        not predecessor_path.is_file()
        or p2a.sha256_file(predecessor_path)
        != str(value.get("predecessor_terminal_disposition_sha256") or "")
    ):
        faults.append("predecessor_disposition_binding_invalid")
        predecessor: dict[str, Any] = {}
    else:
        predecessor = p2a.read_json(predecessor_path)
    if (
        predecessor.get("trigger_state") != "GREEN"
        or predecessor.get("scientific_status") != "INCONCLUSIVE_EXPERIMENT"
        or p2a.mapping(predecessor.get("interruption")).get(
            "same_denominator_resume_authorized"
        )
        is not False
    ):
        faults.append("predecessor_disposition_invalid")

    rows = p2a.dicts(value.get("tasks"))
    repositories = [str(row.get("repository") or "").lower() for row in rows]
    if len(rows) != 10 or int(value.get("task_count") or 0) != 10:
        faults.append("task_count_invalid")
    if len(set(repositories)) != 10 or int(value.get("distinct_repository_count") or 0) != 10:
        faults.append("distinct_repository_count_invalid")
    prior = prior_repositories()
    declared_prior = set(
        item.lower()
        for item in p2a.strings(value.get("source_disjoint_from_repositories"))
    )
    if declared_prior != prior:
        faults.append("prior_repository_set_binding_invalid")
    overlap = sorted(set(repositories).intersection(prior))
    if overlap:
        faults.append("prior_repository_overlap")

    for expected, row in enumerate(rows, 1):
        stem = str(row.get("stem") or f"index_{expected}")
        if int(row.get("campaign_index") or 0) != expected:
            faults.append(f"campaign_index_invalid:{stem}")
        if row.get("license_spdx") not in ALLOWED_LICENSES or not p2a.strings(
            row.get("license_paths")
        ):
            faults.append(f"license_invalid:{stem}")
        parent = str(row.get("parent_revision") or "")
        target = str(row.get("target_revision") or "")
        merge = str(row.get("merge_revision") or "")
        if any(len(revision) != 40 for revision in (parent, target, merge)) or parent == target:
            faults.append(f"revision_identity_invalid:{stem}")
        effects = p2a.strings(row.get("allowed_effect_paths"))
        changed = set(p2a.strings(row.get("patch_changed_files")))
        if not effects or any(path not in changed or not path.endswith(".py") for path in effects):
            faults.append(f"effect_boundary_invalid:{stem}")
        if not str(row.get("natural_request") or "").endswith("Modify only that file."):
            faults.append(f"natural_request_boundary_invalid:{stem}")
        if int(row.get("behavioral_units") or 0) not in {1, 2, 3}:
            faults.append(f"behavioral_unit_count_invalid:{stem}")

    boundaries = p2a.mapping(value.get("boundaries"))
    for key in (
        "archive_fetches_after_membership_freeze",
        "parent_target_oracle_evaluator_executions",
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
            faults.append(f"prospective_boundary_nonzero:{key}")
    if boundaries.get("candidate_generation_opened") is not False:
        faults.append("candidate_generation_already_opened")
    if boundaries.get("user_task_label_or_approval_dependency") is not False:
        faults.append("user_gate_present")
    if boundaries.get("project_selected_quality_token_cap") is not None:
        faults.append("project_selected_quality_token_cap_present")

    return {
        "policy": "project_theseus_p4v2r2r2_source_registry_audit_v1",
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "registry": {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)},
        "instrument": {
            "path": p2a.rel(instrument_path),
            "sha256": p2a.sha256_file(instrument_path),
            "freeze_commit": value.get("instrument_freeze_commit"),
        },
        "predecessor_terminal_disposition": {
            "path": p2a.rel(predecessor_path),
            "sha256": p2a.sha256_file(predecessor_path),
            "scientific_status": predecessor.get("scientific_status"),
        },
        "task_count": len(rows),
        "distinct_repository_count": len(set(repositories)),
        "prior_repository_count": len(prior),
        "prior_repository_overlap": overlap,
        "license_spdx_ids": sorted({str(row.get("license_spdx")) for row in rows}),
        "behavioral_unit_distribution": {
            str(count): sum(int(row.get("behavioral_units") or 0) == count for row in rows)
            for count in (1, 2, 3)
        },
        "candidate_or_control_calls": 0,
        "archive_fetches": 0,
        "project_selected_quality_token_cap": None,
        "maximum_inference": str(value.get("maximum_inference") or ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=p2a.rel(REGISTRY))
    parser.add_argument("--out", default="reports/theseus_p4v2r2r2_source_registry_audit.json")
    args = parser.parse_args()
    report = audit(p2a.resolve(args.registry))
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
