#!/usr/bin/env python3
"""Audit the prospectively sealed P4-v2r2 recovery successor membership."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "theseus_p4v2r2r1_task_sources.json"
SELECTION = ROOT / "reports" / "theseus_p4v2r2r1_online_selection.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def strings(value: Any) -> list[str]:
    return [str(row) for row in value if str(row)] if isinstance(value, list) else []


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def audit(registry_path: Path = REGISTRY) -> dict[str, Any]:
    registry = read_json(registry_path)
    faults: list[str] = []
    if registry.get("policy") != "project_theseus_p4v2r2r1_recovery_source_selection_v1":
        faults.append("registry_policy_invalid")
    if registry.get("state") != "SEALED_BEFORE_NEW_ARCHIVE_FETCH_OR_ANY_SUCCESSOR_CANDIDATE_CALL":
        faults.append("registry_not_prospectively_sealed")

    bound: dict[str, dict[str, str]] = {}
    for owner in (
        "predecessor_interruption",
        "predecessor_terminal_disposition",
        "predecessor_pool",
        "predecessor_registry",
    ):
        path = resolve(str(registry.get(owner) or ""))
        expected = str(registry.get(f"{owner}_sha256") or "")
        observed = sha256_file(path) if path.is_file() else ""
        bound[owner] = {"path": str(registry.get(owner) or ""), "sha256": observed}
        if observed != expected:
            faults.append(f"binding_invalid:{owner}")

    incident = read_json(resolve(str(registry["predecessor_interruption"])))
    disposition = read_json(resolve(str(registry["predecessor_terminal_disposition"])))
    pool = read_json(resolve(str(registry["predecessor_pool"])))
    predecessor_registry = read_json(resolve(str(registry["predecessor_registry"])))
    carried = strings(registry.get("carried_candidate_unseen_stems"))
    if carried != strings(incident.get("candidate_unseen_task_stems")):
        faults.append("candidate_unseen_custody_mismatch")
    if len(carried) != 9 or len(set(carried)) != 9:
        faults.append("candidate_unseen_denominator_invalid")
    if strings(registry.get("consumed_stems_excluded")) != strings(
        incident.get("consumed_task_stems")
    ):
        faults.append("consumed_task_exclusion_invalid")
    if disposition.get("trigger_state") != "GREEN" or disposition.get(
        "scientific_status"
    ) != "INCONCLUSIVE_IMPLEMENTATION":
        faults.append("predecessor_terminal_disposition_invalid")
    pool_by_stem = {
        str(row.get("stem") or ""): row
        for row in pool.get("tasks", [])
        if isinstance(row, dict)
    }
    if any(stem not in pool_by_stem for stem in carried):
        faults.append("carried_task_absent_from_qualified_pool")
    for stem in carried:
        row = mapping(pool_by_stem.get(stem))
        if row.get("evaluator_audit_trigger_state") != "GREEN":
            faults.append(f"carried_evaluator_not_green:{stem}")
        if mapping(row.get("v2r2_oracle_replay")).get("trigger_state") != "GREEN":
            faults.append(f"carried_oracle_replay_not_green:{stem}")
        if mapping(row.get("dependency_corruption")).get("rejected") is not True:
            faults.append(f"carried_dependency_corruption_not_rejected:{stem}")

    replacement = mapping(registry.get("replacement_task"))
    repository = str(replacement.get("repository") or "").lower()
    prior = {
        str(value).lower()
        for value in strings(predecessor_registry.get("source_disjoint_from_repositories"))
    }
    prior.update(
        str(row.get("repository") or "").lower()
        for row in predecessor_registry.get("tasks", [])
        if isinstance(row, dict)
    )
    if not repository or repository in prior:
        faults.append("replacement_repository_not_source_disjoint")
    if replacement.get("license_spdx") != "MIT" or replacement.get("license_paths") != ["LICENSE"]:
        faults.append("replacement_license_invalid")
    if replacement.get("parent_revision") == replacement.get("target_revision"):
        faults.append("replacement_revision_identity_invalid")
    if replacement.get("allowed_effect_paths") != [
        "pydantic_ai_slim/pydantic_ai/models/openrouter.py"
    ]:
        faults.append("replacement_effect_boundary_invalid")
    if len(replacement.get("oracle_units", [])) != 5:
        faults.append("replacement_causal_unit_count_invalid")

    boundaries = mapping(registry.get("boundaries"))
    for key in (
        "new_archive_fetches",
        "new_parent_target_oracle_or_evaluator_executions",
        "successor_local_model_calls",
        "successor_hosted_model_calls",
        "teacher_calls",
        "training_rows_written",
        "D1_cases_consumed",
        "D2_cases_consumed",
    ):
        if int(boundaries.get(key) or 0) != 0:
            faults.append(f"prospective_boundary_nonzero:{key}")
    if boundaries.get("user_or_operator_gate") is not False:
        faults.append("user_gate_present")
    if boundaries.get("project_selected_quality_token_cap") is not None:
        faults.append("quality_token_cap_present")

    selection = read_json(SELECTION) if SELECTION.is_file() else {}
    if (
        selection.get("trigger_state") != "GREEN"
        or selection.get("selected_repository") != replacement.get("repository")
        or selection.get("selected_pull_request") != replacement.get("pull_request")
        or int(selection.get("archive_fetches") or 0) != 0
        or int(selection.get("candidate_or_control_calls") or 0) != 0
    ):
        faults.append("online_selection_receipt_invalid")
    return {
        "policy": "project_theseus_p4v2r2r1_source_registry_audit_v1",
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "registry": {"path": registry_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(registry_path)},
        "bound_predecessor_evidence": bound,
        "carried_candidate_unseen_tasks": len(carried),
        "replacement_tasks": 1,
        "sealed_successor_tasks": len(carried) + 1,
        "replacement_repository": repository,
        "replacement_source_disjoint": repository not in prior,
        "new_archive_fetches": int(boundaries.get("new_archive_fetches") or 0),
        "successor_candidate_or_control_calls": int(boundaries.get("successor_local_model_calls") or 0)
        + int(boundaries.get("successor_hosted_model_calls") or 0),
        "project_selected_quality_token_cap": boundaries.get("project_selected_quality_token_cap"),
        "maximum_inference": str(registry.get("maximum_inference") or ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(REGISTRY.relative_to(ROOT)))
    parser.add_argument("--out", default="reports/theseus_p4v2r2r1_source_registry_audit.json")
    args = parser.parse_args()
    report = audit(resolve(args.registry))
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
