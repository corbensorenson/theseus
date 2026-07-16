#!/usr/bin/env python3
"""Validate the binding OneCell-RWM disposition for the first language campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "onecell_rwm_pretraining_disposition.json"
DEFAULT_REPORT = ROOT / "reports" / "onecell_rwm_pretraining_disposition.json"


class OneCellDispositionFault(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_config(value)
    return value


def validate_config(config: dict[str, Any]) -> None:
    if config.get("policy") != "project_theseus_onecell_rwm_pretraining_disposition_v1":
        raise OneCellDispositionFault("policy_invalid")
    source = config.get("source") or {}
    source_path = resolve(str(source.get("path") or ""))
    if not source_path.is_file() or sha256(source_path) != source.get("sha256"):
        raise OneCellDispositionFault("source_identity_invalid")
    campaign = config.get("first_campaign") or {}
    campaign_path = resolve(str(campaign.get("config") or ""))
    if not campaign_path.is_file() or sha256(campaign_path) != campaign.get("sha256"):
        raise OneCellDispositionFault("campaign_identity_invalid")
    campaign_config = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (campaign_config.get("candidate") or {}).get("id") != campaign.get("campaign_id"):
        raise OneCellDispositionFault("campaign_id_invalid")
    if (
        campaign.get("disposition") != "retired_from_first_language_campaign"
        or campaign.get("routeable") is not False
        or int(campaign.get("optimizer_exposure_steps", -1)) != 0
        or campaign.get("checkpoint_member") is not False
    ):
        raise OneCellDispositionFault("first_campaign_retirement_invalid")
    abi = config.get("cognitive_kernel_abi") or {}
    required_methods = {
        "initialize", "propose", "accept_receipt", "checkpoint", "restore",
        "parameter_accounting", "resource_accounting",
    }
    if set(abi.get("methods") or []) != required_methods:
        raise OneCellDispositionFault("cognitive_kernel_abi_incomplete")
    if len(abi.get("proposal_fields") or []) < 10 or len(abi.get("receipt_states") or []) != 8:
        raise OneCellDispositionFault("proposal_or_receipt_contract_incomplete")
    boundary = config.get("exact_latent_boundary") or {}
    if not boundary.get("exact_channels") or not boundary.get("latent_channels") or len(boundary.get("collapse_forbidden") or []) < 5:
        raise OneCellDispositionFault("exact_latent_boundary_incomplete")
    owner_reuse = config.get("existing_owner_reuse") or {}
    if len(owner_reuse) != 8 or len(set(owner_reuse.values())) != len(owner_reuse):
        raise OneCellDispositionFault("existing_owner_reuse_invalid")
    objective = config.get("objective_contract") or {}
    if len(objective.get("required_terms") or []) != 9 or float(objective.get("longer_reasoning_intrinsic_reward", -1)) != 0.0 or int(objective.get("tool_retrieval_verifier_credit_to_core", -1)) != 0:
        raise OneCellDispositionFault("objective_contract_invalid")
    checkpoint = config.get("checkpoint_contract") or {}
    if len(checkpoint.get("required_groups") or []) != 8 or checkpoint.get("exact_state_embedded_in_latent_checkpoint") is not False or checkpoint.get("migration_rehearsal_required") is not True or checkpoint.get("rollback_required") is not True:
        raise OneCellDispositionFault("checkpoint_contract_invalid")
    reentry = config.get("successor_campaign_reentry") or {}
    if len(reentry.get("required_before_optimizer") or []) < 10 or int(reentry.get("maximum_architecture_canary_steps", 0)) > 8 or reentry.get("may_delay_first_language_campaign") is not False:
        raise OneCellDispositionFault("successor_reentry_invalid")
    boundaries = config.get("boundaries") or {}
    if any(value not in (0, False) for value in boundaries.values()):
        raise OneCellDispositionFault("claim_or_exposure_boundary_nonzero")


def mutation_controls(config: dict[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def reject(case_id: str, expected: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        candidate = copy.deepcopy(config)
        mutate(candidate)
        observed = "accepted"
        try:
            validate_config(candidate)
        except OneCellDispositionFault as exc:
            observed = str(exc)
        cases.append({"case_id": case_id, "expected": expected, "observed": observed, "passed": expected in observed})

    reject("source_tamper", "source_identity_invalid", lambda row: row["source"].update(sha256="0" * 64))
    reject("campaign_tamper", "campaign_identity_invalid", lambda row: row["first_campaign"].update(sha256="0" * 64))
    reject("route_activation", "first_campaign_retirement_invalid", lambda row: row["first_campaign"].update(routeable=True))
    reject("optimizer_exposure", "first_campaign_retirement_invalid", lambda row: row["first_campaign"].update(optimizer_exposure_steps=1))
    reject("abi_method_missing", "cognitive_kernel_abi_incomplete", lambda row: row["cognitive_kernel_abi"]["methods"].pop())
    reject("exact_latent_collapse", "exact_latent_boundary_incomplete", lambda row: row["exact_latent_boundary"].update(collapse_forbidden=[]))
    reject("objective_credit_laundering", "objective_contract_invalid", lambda row: row["objective_contract"].update(tool_retrieval_verifier_credit_to_core=1))
    reject("checkpoint_exact_state_collapse", "checkpoint_contract_invalid", lambda row: row["checkpoint_contract"].update(exact_state_embedded_in_latent_checkpoint=True))
    reject("unbounded_canary", "successor_reentry_invalid", lambda row: row["successor_campaign_reentry"].update(maximum_architecture_canary_steps=9))
    reject("capability_claim", "claim_or_exposure_boundary_nonzero", lambda row: row["boundaries"].update(onecell_learned_capability_claimed=True))
    return {"case_count": len(cases), "passed_count": sum(bool(row["passed"]) for row in cases), "results": cases}


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    controls = mutation_controls(config)
    campaign = config["first_campaign"]
    gates = {
        "source_and_campaign_content_bound": True,
        "cognitive_kernel_abi_complete": True,
        "exact_latent_boundary_complete": True,
        "existing_system_owners_reused": True,
        "objective_and_checkpoint_interfaces_frozen": True,
        "first_campaign_route_disabled": campaign["routeable"] is False,
        "first_campaign_optimizer_exposure_zero": campaign["optimizer_exposure_steps"] == 0,
        "first_campaign_checkpoint_unchanged": campaign["checkpoint_member"] is False,
        "successor_reentry_preregistered": True,
        "mutation_controls_rejected": controls["case_count"] == controls["passed_count"],
    }
    return {
        "policy": config["policy"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trigger_state": "GREEN" if all(gates.values()) else "RED",
        "support_state": "replayable-reference-backed",
        "disposition": campaign["disposition"],
        "source_artifacts": {
            "handoff": config["source"],
            "first_campaign": {"path": campaign["config"], "sha256": campaign["sha256"]},
        },
        "gates": gates,
        "summary": {
            "abi_method_count": len(config["cognitive_kernel_abi"]["methods"]),
            "objective_term_count": len(config["objective_contract"]["required_terms"]),
            "checkpoint_group_count": len(config["checkpoint_contract"]["required_groups"]),
            "successor_prerequisite_count": len(config["successor_campaign_reentry"]["required_before_optimizer"]),
            "mutation_case_count": controls["case_count"],
            "mutation_passed_count": controls["passed_count"],
            "optimizer_exposure_steps": 0,
            "route_authorized": False,
            "checkpoint_member": False,
            "public_training_rows": 0,
            "external_inference_calls": 0,
        },
        "cognitive_kernel_abi": config["cognitive_kernel_abi"],
        "successor_campaign_reentry": config["successor_campaign_reentry"],
        "mutation_controls": controls,
        "non_claims": [
            "This is a binding first-campaign architecture disposition, not a OneCell implementation or learned-capability verdict.",
            "The practical transformer language campaign does not train, route, checkpoint, or receive credit from OneCell.",
            "OneCell may re-enter only as a separate matched cognitive-kernel discovery campaign after its exact substrate exists.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_REPORT))
    args = parser.parse_args()
    report = build_report(load_config(resolve(args.config)))
    output = resolve(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"trigger_state": report["trigger_state"], "disposition": report["disposition"], "summary": report["summary"]}, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
