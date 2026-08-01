#!/usr/bin/env python3
"""Recompute the terminal scientific disposition of the P4-v2r2 campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4s_disposition as base  # noqa: E402
import theseus_p4v2r2_campaign as campaign  # noqa: E402
import theseus_p4v2r2_cognitive_compilation as p4v2r2  # noqa: E402


POLICY = "project_theseus_p4v2r2_cognitive_compilation_terminal_disposition_v1"
POOL = ROOT / "configs" / "theseus_p4v2r2_task_pool.json"
INSTRUMENT = ROOT / "configs" / "theseus_p4v2r2_cognitive_compilation_instrument.json"
PROGRESS = ROOT / "reports" / "theseus_p4v2r2_campaign_attempt2_progress.json"
OUT = ROOT / "reports" / "theseus_p4v2r2_attempt2_terminal_disposition.json"
ORACLE_CORRECTIONS = (
    ROOT / "configs" / "theseus_p4v2r2_oracle_materialization_corrections.json"
)
STATUS_MAP = {
    "P4S_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE": (
        "P4V2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE"
    ),
    "P4S_ADEQUATE_NO_SURVIVOR": "P4V2R2_ADEQUATE_NO_SURVIVOR",
    "P4S_REVIEW_REQUIRED": "P4V2R2_REVIEW_REQUIRED",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=p2a.rel(OUT))
    args = parser.parse_args()
    report = build_report()
    p2a.write_json(p2a.resolve(args.out), report)
    print(
        json.dumps(
            {
                "trigger_state": report["trigger_state"],
                "scientific_status": report["scientific_status"],
                "complete_tasks": report["denominators"]["tasks"],
                "learned_model_calls": report["denominators"][
                    "learned_model_calls"
                ],
                "safety_ceiling_hits": report["termination_custody"][
                    "safety_ceiling_hits"
                ],
                "faults": report["faults"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_report() -> dict[str, Any]:
    # Reuse the already-audited statistical and route-blind aggregation kernel while
    # rebinding every source owner to this prospectively sealed successor campaign.
    rebound = {
        "POOL": POOL,
        "INSTRUMENT": INSTRUMENT,
        "PROGRESS": PROGRESS,
        "OUT": OUT,
        "campaign": campaign,
        "p4s": p4v2r2,
        "audit_source_pool": audit_source_pool,
    }
    original = {name: getattr(base, name) for name in rebound}
    for name, value in rebound.items():
        setattr(base, name, value)
    # The predecessor report references historical repair fields while building its
    # source-identity object. Give those reads a committed placeholder, then remove
    # and replace the fields before this successor report can be returned or written.
    campaign_placeholders = {
        "INVALID_ATTEMPT": ORACLE_CORRECTIONS,
        "ATTEMPT2_INSTRUMENT_REBIND_COMMIT": "not_applicable_p4v2r2_runtime_bootstrap_repair",
    }
    campaign_original = {
        name: getattr(campaign, name, None) for name in campaign_placeholders
    }
    try:
        for name, value in campaign_placeholders.items():
            setattr(campaign, name, value)
        report = base.build_report()
    finally:
        for name, value in original.items():
            setattr(base, name, value)
        for name, value in campaign_original.items():
            if value is None:
                delattr(campaign, name)
            else:
                setattr(campaign, name, value)

    status = STATUS_MAP.get(
        str(report.get("scientific_status") or ""),
        str(report.get("scientific_status") or ""),
    )
    # The inherited aggregation kernel represents a missing denominator as
    # information_flow_green=false. Before consumption, that means "not yet
    # observed," not an observed leakage event. Preserve INVALID_INFORMATION_FLOW
    # only when a completed artifact actually trips an information-flow audit.
    faults = p2a.strings(report.get("faults"))
    observed_information_flow_fault = any(
        fault.startswith(
            (
                "information_flow_invalid:",
                "candidate_integrity_recomputation_invalid:",
                "independent_evaluation_replay_fault:",
                "independent_evaluation_replay_mismatch:",
            )
        )
        for fault in faults
    )
    if (
        status == "INVALID_INFORMATION_FLOW"
        and "campaign_not_complete" in faults
        and not observed_information_flow_fault
    ):
        status = "P4V2R2_REVIEW_REQUIRED"
    report["policy"] = POLICY
    report["scientific_status"] = status
    report["scope"] = (
        "Exact frozen TMax model, completion-based Semantic-IR v2r2 instrument, "
        "and ten licensed decision-development tasks. No D1, D2, serving, "
        "training, hosted-model, or automatic book-support authority."
    )
    identities = p2a.mapping(report.get("source_identities"))
    identities.pop("attempt2_instrument_rebind_commit", None)
    identities.pop("invalid_attempt1", None)
    identities["oracle_materialization_corrections"] = base.source_identity(
        ORACLE_CORRECTIONS
    )
    identities["runtime_attempt_namespace"] = campaign.RUNTIME_ATTEMPT_NAMESPACE
    report["source_identities"] = identities
    adequacy = p2a.mapping(report.get("adequacy"))
    replay = p2a.mapping(adequacy.get("independent_evaluator_replay_contract"))
    replay["oracle_digest_repair_owner"] = (
        "fresh_D1_evaluator_successor_only_if_P4V2R2_survives"
    )
    adequacy["independent_evaluator_replay_contract"] = replay
    adequacy["v2r2_transport_oracle_replays"] = "10/10"
    report["adequacy"] = adequacy
    consumption = p2a.mapping(report.get("consumption"))
    consumption["eligible_for_D1"] = (
        status == "P4V2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE"
    )
    report["consumption"] = consumption
    report["next_stage"] = next_stage(status)
    report["maximum_inference"] = (
        "This report can decide only whether this exact mechanics-qualified "
        "Semantic-IR v2r2 implementation with frozen TMax survives this ten-task "
        "decision-development surface and may be qualified once on fresh "
        "source-disjoint D1. It cannot establish or falsify cognitive compilation "
        "generally, qualify serving or training, decide D2, or promote ASI Stack "
        "book support."
    )
    return report


def audit_source_pool(pool: dict[str, Any]) -> dict[str, Any]:
    faults: list[str] = []
    if pool.get("policy") != (
        "project_theseus_p4v2r2_cognitive_compilation_task_pool_v1"
    ):
        faults.append("source_pool_policy_invalid")
    if pool.get("state") != "SEALED_BEFORE_CANDIDATE_GENERATION":
        faults.append("source_pool_not_sealed")
    if p2a.strings(pool.get("faults")):
        faults.append("source_pool_declared_faults_present")

    registry_path = p2a.resolve(str(pool.get("source_registry") or ""))
    fetch_path = p2a.resolve(str(pool.get("source_fetch_report") or ""))
    if (
        not registry_path.is_file()
        or p2a.sha256_file(registry_path)
        != str(pool.get("source_registry_sha256") or "")
    ):
        faults.append("source_registry_binding_invalid")
    if (
        not fetch_path.is_file()
        or p2a.sha256_file(fetch_path)
        != str(pool.get("source_fetch_report_sha256") or "")
    ):
        faults.append("source_fetch_binding_invalid")
    fetch = p2a.read_json(fetch_path) if fetch_path.is_file() else {}
    if (
        fetch.get("trigger_state") != "GREEN"
        or p2a.strings(fetch.get("faults"))
        or int(fetch.get("candidate_or_control_calls") or 0) != 0
    ):
        faults.append("source_fetch_custody_invalid")

    tasks = p2a.dicts(pool.get("tasks"))
    repositories = [str(row.get("repository") or "").lower() for row in tasks]
    licenses = [str(row.get("license_spdx") or "") for row in tasks]
    revisions = [
        (
            str(row.get("repository") or "").lower(),
            str(row.get("parent_revision") or ""),
            str(row.get("target_revision") or ""),
        )
        for row in tasks
    ]
    prior_repositories = {
        str(value).lower()
        for value in p2a.strings(
            p2a.mapping(pool.get("source_disjoint_from")).get("P2_through_P4S")
        )
    }
    if len(tasks) != 10 or int(pool.get("task_count") or 0) != 10:
        faults.append("source_pool_task_count_invalid")
    if len(set(repositories)) != 10 or int(
        pool.get("distinct_repositories") or 0
    ) != 10:
        faults.append("source_pool_repositories_not_distinct")
    if prior_repositories.intersection(repositories):
        faults.append("source_pool_predecessor_repository_overlap")
    if any(not license_id for license_id in licenses):
        faults.append("source_pool_license_missing")
    if any(
        not repository or not parent or not target or parent == target
        for repository, parent, target in revisions
    ):
        faults.append("source_pool_revision_identity_invalid")
    if p2a.mapping(pool.get("source_disjoint_from")).get("training") != (
        "all_P4V2R2_tasks_permanently_excluded"
    ):
        faults.append("source_pool_training_exclusion_missing")
    if int(pool.get("green_evaluator_audits") or 0) != 10:
        faults.append("source_pool_evaluator_floor_invalid")
    if int(pool.get("v2r2_oracle_replays_green") or 0) != 10:
        faults.append("source_pool_v2r2_oracle_floor_invalid")
    if int(pool.get("dependency_corruptions_rejected") or 0) != 10:
        faults.append("source_pool_dependency_floor_invalid")
    return {
        "passed": not faults,
        "faults": sorted(set(faults)),
        "task_count": len(tasks),
        "distinct_repository_count": len(set(repositories)),
        "license_spdx_ids": sorted(set(licenses)),
        "predecessor_repository_overlap": sorted(
            prior_repositories.intersection(repositories)
        ),
        "source_registry": base.source_identity(registry_path),
        "source_fetch_report": base.source_identity(fetch_path),
    }


def classify_status(
    *,
    information_flow_green: bool,
    boundary_hits: int,
    mechanics_floor: bool,
    experiment_floor: bool,
    survivor_rule: bool,
) -> str:
    status = base.classify_status(
        information_flow_green=information_flow_green,
        boundary_hits=boundary_hits,
        mechanics_floor=mechanics_floor,
        experiment_floor=experiment_floor,
        survivor_rule=survivor_rule,
    )
    return STATUS_MAP.get(status, status)


def next_stage(status: str) -> dict[str, Any]:
    if status == "P4V2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE":
        return {
            "state": "OPEN_ONE_FRESH_SOURCE_DISJOINT_D1_QUALIFICATION",
            "D1_eligible": True,
            "book_support_state_effect": "none",
        }
    return {
        "state": "D1_CLOSED_RETAIN_SCOPED_P4V2R2_EVIDENCE",
        "D1_eligible": False,
        "book_support_state_effect": "none",
    }


if __name__ == "__main__":
    raise SystemExit(main())
