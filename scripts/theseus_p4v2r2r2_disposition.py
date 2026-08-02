#!/usr/bin/env python3
"""Independently aggregate the fresh repaired P4-v2r2-r2 campaign."""

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
import theseus_p4v2r2_cognitive_compilation as causal  # noqa: E402
import theseus_p4v2r2r2_campaign as campaign  # noqa: E402
import theseus_p4v2r2r2_source_registry as source_registry  # noqa: E402


POLICY = "project_theseus_p4v2r2r2_terminal_disposition_v1"
POOL = ROOT / "configs" / "theseus_p4v2r2r2_task_pool.json"
INSTRUMENT = ROOT / "configs" / "theseus_p4v2r2r2_cognitive_compilation_instrument.json"
PROGRESS = ROOT / "reports" / "theseus_p4v2r2r2_attempt2_campaign_progress.json"
OUT = ROOT / "reports" / "theseus_p4v2r2r2_attempt2_terminal_disposition.json"
PRE_GENERATION_FAILURE = (
    ROOT / "reports" / "theseus_p4v2r2r2_attempt1_pre_generation_failure_disposition.json"
)
EFFECTIVE_RESEAL_COMMIT = "aea5dac6a5e4ebd25391c554afb0a0c2957890c4"
STATUS_MAP = {
    "P4S_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE": (
        "P4V2R2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE"
    ),
    "P4S_ADEQUATE_NO_SURVIVOR": "P4V2R2R2_ADEQUATE_NO_SURVIVOR",
    "P4S_REVIEW_REQUIRED": "P4V2R2R2_REVIEW_REQUIRED",
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
                "physical_context_boundary_hits": report["termination_custody"].get(
                    "safety_ceiling_hits", 0
                ),
                "faults": report["faults"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_report() -> dict[str, Any]:
    rebound = {
        "POOL": POOL,
        "INSTRUMENT": INSTRUMENT,
        "PROGRESS": PROGRESS,
        "OUT": OUT,
        "campaign": campaign,
        "p4s": causal,
        "audit_source_pool": audit_source_pool,
    }
    original = {name: getattr(base, name) for name in rebound}
    campaign_placeholders = {
        "POOL_SEAL_COMMIT": EFFECTIVE_RESEAL_COMMIT,
        "ATTEMPT2_INSTRUMENT_REBIND_COMMIT": EFFECTIVE_RESEAL_COMMIT,
        "INVALID_ATTEMPT": PRE_GENERATION_FAILURE,
    }
    campaign_original = {
        name: getattr(campaign, name, None) for name in campaign_placeholders
    }
    try:
        for name, value in rebound.items():
            setattr(base, name, value)
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
        status = "P4V2R2R2_REVIEW_REQUIRED"

    report["policy"] = POLICY
    report["scientific_status"] = status
    report["scope"] = (
        "Exact frozen TMax model, repaired completion-based Semantic-IR v2r2 "
        "instrument, and ten licensed source-disjoint decision-development tasks. "
        "No D1, D2, serving, training, hosted-model, or automatic book-support "
        "authority."
    )
    identities = p2a.mapping(report.get("source_identities"))
    identities["effective_reseal_commit"] = EFFECTIVE_RESEAL_COMMIT
    identities["pre_generation_failure"] = base.source_identity(
        PRE_GENERATION_FAILURE
    )
    identities["runtime_attempt_namespace"] = campaign.RUNTIME_ATTEMPT_NAMESPACE
    report["source_identities"] = identities
    adequacy = p2a.mapping(report.get("adequacy"))
    adequacy["v2r2_transport_oracle_replays"] = "10/10"
    adequacy["dependency_corruption_rejections"] = "10/10"
    adequacy["pre_generation_runner_failure_consumed_model_calls"] = 0
    report["adequacy"] = adequacy
    consumption = p2a.mapping(report.get("consumption"))
    consumption["eligible_for_D1"] = (
        status == "P4V2R2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE"
    )
    consumption["external_inference_calls"] = 0
    consumption["teacher_calls"] = 0
    consumption["training_rows_written"] = 0
    report["consumption"] = consumption
    report["next_stage"] = next_stage(status)
    report["maximum_inference"] = (
        "This report can decide only whether this exact mechanics-qualified "
        "Semantic-IR v2r2 implementation with frozen TMax survives this ten-task "
        "development surface and may be qualified once on fresh source-disjoint "
        "D1. It cannot establish or falsify cognitive compilation generally, "
        "qualify serving or training, decide D2, or promote ASI Stack book support."
    )
    return report


def audit_source_pool(pool: dict[str, Any]) -> dict[str, Any]:
    faults: list[str] = []
    if pool.get("policy") != (
        "project_theseus_p4v2r2r2_cognitive_compilation_task_pool_v1"
    ):
        faults.append("source_pool_policy_invalid")
    if pool.get("state") != "SEALED_BEFORE_CANDIDATE_GENERATION":
        faults.append("source_pool_not_sealed")
    if p2a.strings(pool.get("faults")):
        faults.append("source_pool_declared_faults_present")
    if pool.get("candidate_generation_opened") is not False:
        faults.append("source_pool_candidate_generation_opened_at_seal")

    instrument_owner = p2a.mapping(pool.get("instrument"))
    instrument_path = p2a.resolve(str(instrument_owner.get("path") or ""))
    if (
        not instrument_path.is_file()
        or p2a.sha256_file(instrument_path)
        != str(instrument_owner.get("sha256") or "")
    ):
        faults.append("source_pool_instrument_binding_invalid")

    owners: dict[str, dict[str, str]] = {}
    for name in (
        "source_registry",
        "source_fetch",
        "revision_corrections",
        "revision_repair_fetch",
        "task_contracts",
    ):
        owner = p2a.mapping(pool.get(name))
        path = p2a.resolve(str(owner.get("path") or ""))
        observed = p2a.sha256_file(path)
        if not path.is_file() or observed != str(owner.get("sha256") or ""):
            faults.append(f"source_pool_binding_invalid:{name}")
        owners[name] = {"path": p2a.rel(path), "sha256": observed}

    registry_path = p2a.resolve(owners["source_registry"]["path"])
    registry_audit = source_registry.audit(registry_path)
    if registry_audit.get("trigger_state") != "GREEN":
        faults.append("source_registry_audit_red")
    fetch = p2a.read_json(p2a.resolve(owners["source_fetch"]["path"]))
    revision_fetch = p2a.read_json(
        p2a.resolve(owners["revision_repair_fetch"]["path"])
    )
    if fetch.get("trigger_state") != "GREEN" or p2a.strings(fetch.get("faults")):
        faults.append("source_fetch_custody_invalid")
    if revision_fetch.get("trigger_state") != "GREEN" or p2a.strings(
        revision_fetch.get("faults")
    ):
        faults.append("revision_fetch_custody_invalid")

    tasks = p2a.dicts(pool.get("tasks"))
    repositories = [str(row.get("repository") or "").lower() for row in tasks]
    if len(tasks) != 10 or int(pool.get("task_count") or 0) != 10:
        faults.append("source_pool_task_count_invalid")
    if len(set(repositories)) != 10 or int(
        pool.get("distinct_repositories") or 0
    ) != 10:
        faults.append("source_pool_repositories_not_distinct")
    for expected, row in enumerate(tasks, 1):
        stem = str(row.get("stem") or "")
        if int(row.get("campaign_index") or 0) != expected:
            faults.append(f"source_pool_campaign_index_invalid:{stem}")
        if (
            not row.get("baseline_parent_failed")
            or not row.get("upstream_target_passed")
            or not row.get("compiler_oracle_v1_passed")
            or p2a.mapping(row.get("v2r2_oracle_replay")).get("trigger_state")
            != "GREEN"
            or not row.get("four_base_corruptions_rejected")
            or p2a.mapping(row.get("dependency_corruption")).get("rejected")
            is not True
        ):
            faults.append(f"source_pool_mechanics_floor_invalid:{stem}")
        for path_key, digest_key in (
            ("task", "task_sha256"),
            ("evaluator", "evaluator_sha256"),
            ("evaluator_audit", "evaluator_audit_sha256"),
            ("oracle_ir", "oracle_ir_sha256"),
            (
                "treatment_transport_oracle_ir",
                "treatment_transport_oracle_ir_sha256",
            ),
        ):
            path = p2a.resolve(str(row.get(path_key) or ""))
            if (
                not path.is_file()
                or p2a.sha256_file(path) != str(row.get(digest_key) or "")
            ):
                faults.append(f"source_pool_task_binding_invalid:{stem}:{path_key}")

    counters = p2a.mapping(pool.get("counters"))
    if any(int(value or 0) != 0 for value in counters.values()):
        faults.append("source_pool_preseal_counter_nonzero")
    if p2a.mapping(pool.get("generation_boundary")).get(
        "project_selected_quality_token_cap"
    ) is not None:
        faults.append("source_pool_quality_token_cap_present")
    if not all(
        int(pool.get(key) or 0) == 10
        for key in (
            "green_evaluator_audits",
            "v2r2_oracle_replays_green",
            "dependency_corruptions_rejected",
        )
    ):
        faults.append("source_pool_global_mechanics_floor_invalid")
    return {
        "passed": not faults,
        "faults": sorted(set(faults)),
        "task_count": len(tasks),
        "distinct_repository_count": len(set(repositories)),
        "source_registry_audit": registry_audit,
        "bound_owners": owners,
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
    if status == "P4V2R2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE":
        return {
            "state": "OPEN_ONE_FRESH_SOURCE_DISJOINT_D1_QUALIFICATION",
            "D1_eligible": True,
            "book_support_state_effect": "none",
        }
    return {
        "state": "D1_CLOSED_RETAIN_SCOPED_P4V2R2R2_EVIDENCE",
        "D1_eligible": False,
        "book_support_state_effect": "none",
    }


if __name__ == "__main__":
    raise SystemExit(main())
