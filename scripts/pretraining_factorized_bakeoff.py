#!/usr/bin/env python3
"""Select a first-campaign pretraining stack from independent matched evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "pretraining_factorized_bakeoff.json"
DEFAULT_REPORT = ROOT / "reports" / "pretraining_factorized_bakeoff.json"
REQUIRED_SLOTS = {
    "surface_codec",
    "relational_ir",
    "causal_generator",
    "draft_accelerator",
    "optimizer",
    "kv_runtime",
    "semantic_verifier",
}


class BakeoffFault(ValueError):
    pass


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BakeoffFault(f"json_unavailable:{path}") from exc
    if not isinstance(value, dict):
        raise BakeoffFault(f"json_object_required:{path}")
    return value


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = read_json(path)
    if config.get("policy") != "project_theseus_pretraining_factorized_bakeoff_v1":
        raise BakeoffFault("policy_invalid")
    if set(config.get("selected_control_implementations") or {}) != REQUIRED_SLOTS:
        raise BakeoffFault("selected_slots_incomplete")
    if set(config.get("evidence") or {}) != {
        "mtp", "dynamic_patch", "optimizer", "soap", "rdc_kerc"
    }:
        raise BakeoffFault("evidence_docket_incomplete")
    hard = config.get("hard_boundaries") or {}
    for key in (
        "public_training_rows",
        "public_evaluation_rows",
        "external_inference_calls",
        "fallback_template_router_tool_credit",
    ):
        if int(hard.get(key, -1)) != 0:
            raise BakeoffFault(f"hard_boundary_nonzero:{key}")
    if hard.get("production_checkpoint_mutation") is not False:
        raise BakeoffFault("production_checkpoint_mutation_allowed")
    if hard.get("scientific_falsification_from_campaign_exclusion") is not False:
        raise BakeoffFault("scientific_falsification_boundary_invalid")
    return config


def evidence_manifest(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    reports: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    for evidence_id, contract in config["evidence"].items():
        path = resolve(contract["path"])
        report = read_json(path)
        if report.get("policy") != contract["policy"]:
            raise BakeoffFault(f"evidence_policy_mismatch:{evidence_id}")
        if report.get("trigger_state") != "GREEN":
            raise BakeoffFault(f"evidence_not_green:{evidence_id}")
        reports[evidence_id] = report
        manifest[evidence_id] = {
            "path": contract["path"],
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "policy": report["policy"],
        }
    contract_path = resolve(config["candidate_contract"])
    candidate_contract = read_json(contract_path)
    if candidate_contract.get("policy") != "project_theseus_pretraining_architecture_candidates_v1":
        raise BakeoffFault("candidate_contract_policy_mismatch")
    manifest["candidate_contract"] = {
        "path": config["candidate_contract"],
        "sha256": sha256(contract_path),
        "bytes": contract_path.stat().st_size,
        "policy": candidate_contract["policy"],
    }
    reports["candidate_contract"] = candidate_contract
    training_path = resolve(config["canonical_training_config"])
    training_config = read_json(training_path)
    if training_config.get("policy") != "project_theseus_moecot_language_arm_training_v1":
        raise BakeoffFault("canonical_training_config_policy_mismatch")
    manifest["canonical_training_config"] = {
        "path": config["canonical_training_config"],
        "sha256": sha256(training_path),
        "bytes": training_path.stat().st_size,
        "policy": training_config["policy"],
    }
    reports["canonical_training_config"] = training_config
    return reports, manifest


def assert_zero(report: dict[str, Any], key: str, evidence_id: str) -> None:
    if int(report.get(key, 0)) != 0:
        raise BakeoffFault(f"no_cheat_counter_nonzero:{evidence_id}:{key}")


def validate_non_kerc_evidence(reports: dict[str, Any]) -> dict[str, Any]:
    mtp = reports["mtp"]
    if mtp.get("architecture_disposition") != "SCOPED_DISCOVERY_EXCLUDED_FROM_FIRST_CAMPAIGN":
        raise BakeoffFault("mtp_disposition_unexpected")
    if (mtp.get("campaign_disposition") or {}).get("scientific_falsification_claimed") is not False:
        raise BakeoffFault("mtp_falsification_overclaim")
    if not all((mtp.get("gates") or {}).values()):
        raise BakeoffFault("mtp_mechanics_gate_failed")

    dynamic = reports["dynamic_patch"]
    comparison = ((dynamic.get("summary") or {}).get("dynamic_comparison") or {})
    if comparison.get("disposition") != "SCOPED_DISCOVERY_EXCLUDED_FROM_FIRST_CAMPAIGN":
        raise BakeoffFault("dynamic_patch_disposition_unexpected")
    if comparison.get("scientific_falsification_claimed") is not False:
        raise BakeoffFault("dynamic_patch_falsification_overclaim")
    integrity = dynamic.get("candidate_integrity") or {}
    if (
        int(integrity.get("deterministic_or_template_credit", -1)) != 0
        or integrity.get("hidden_target_metadata_visible") is not False
        or integrity.get("public_or_confirmation_surface_consumed") is not False
        or integrity.get("functional_claim") != "NOT_EVALUATED"
        or not str(integrity.get("learned_boundary_inputs") or "")
    ):
        raise BakeoffFault("dynamic_patch_integrity_failed")

    optimizer = reports["optimizer"]
    campaign = optimizer.get("campaign_disposition") or {}
    if campaign.get("selected_optimizer") != "adamw_mlx":
        raise BakeoffFault("adamw_not_selected")
    if campaign.get("scientific_falsification_claimed") is not False:
        raise BakeoffFault("optimizer_falsification_overclaim")
    if not all((optimizer.get("gates") or {}).values()):
        raise BakeoffFault("optimizer_gate_failed")
    cards = optimizer.get("optimizer_policy_cards") or {}
    if (
        set(cards)
        != {
            "adafactor_mlx",
            "adamw_mlx",
            "muon_mlx",
            "schedule_free_adamw_mlx",
        }
        or optimizer.get("optimizer_policy_card_faults") != []
        or (optimizer.get("width_transfer") or {}).get("trigger_state")
        != "GREEN"
        or (optimizer.get("width_transfer") or {}).get("selected_optimizer")
        != campaign.get("selected_optimizer")
        or "adafactor_mlx" not in (optimizer.get("comparisons") or {})
    ):
        raise BakeoffFault("optimizer_policy_adafactor_or_width_transfer_incomplete")

    soap = reports["soap"]
    if (
        soap.get("disposition")
        != "FORMALLY_SCOPE_REMOVED_FULL_SHAPE_SOAP_UNECONOMIC_M1_MLX"
    ):
        raise BakeoffFault("soap_resource_disposition_unexpected")
    if soap.get("scientific_optimizer_quality_claim") not in {
        False,
        "NOT_EVALUATED",
    }:
        raise BakeoffFault("soap_quality_overclaim")
    if soap.get("production_checkpoint_mutation") is not False:
        raise BakeoffFault("soap_mutated_production_checkpoint")
    if soap.get("disposition", "").startswith("FORMALLY_SCOPE_REMOVED") and (
        soap.get("finite_docket_membership")
        != "REMOVED_FROM_FIRST_CAMPAIGN_FINITE_DOCKET"
        or soap.get("engineering_scope_decision")
        != "REMOVE_FULL_SHAPE_SOAP_FROM_FIRST_CAMPAIGN"
        or not soap.get("reentry_condition")
    ):
        raise BakeoffFault("soap_scope_removal_incomplete")

    return {
        "mtp": mtp["architecture_disposition"],
        "dynamic_patch": comparison["disposition"],
        "optimizer": campaign["selected_optimizer"],
        "soap": soap["disposition"],
    }


def evaluate_kerc(config: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    disposition = str(report.get("disposition") or "")
    if disposition not in {
        "ADOPT_RDC_KERC_FIRST_CAMPAIGN",
        "SCOPED_DISCOVERY_EXCLUDED_FROM_FIRST_CAMPAIGN",
        "RESOURCE_DEFERRED_ON_THIS_HOST",
    }:
        raise BakeoffFault("kerc_disposition_invalid")
    if report.get("scientific_falsification_claimed") is not False:
        raise BakeoffFault("kerc_falsification_overclaim")
    for key in (
        "public_training_rows",
        "public_evaluation_rows",
        "external_inference_calls",
        "fallback_template_router_tool_credit",
        "production_checkpoint_mutations",
    ):
        assert_zero(report, key, "rdc_kerc")
    source_artifacts = report.get("source_artifacts") or {}
    minimum_artifacts = (
        5
        if disposition == "RESOURCE_DEFERRED_ON_THIS_HOST"
        else int(config["kerc_required_seed_count"]) + 5
    )
    if len(source_artifacts) < minimum_artifacts:
        raise BakeoffFault("kerc_source_artifacts_incomplete")
    for artifact_id, artifact in source_artifacts.items():
        path = resolve(str((artifact or {}).get("path") or ""))
        if not path.is_file() or sha256(path) != str((artifact or {}).get("sha256") or ""):
            raise BakeoffFault(f"kerc_source_artifact_stale:{artifact_id}")
    checks = report.get("checks") or {}
    if disposition == "RESOURCE_DEFERRED_ON_THIS_HOST":
        if (
            report.get("candidate_execution_authorized") is not False
            or report.get("capability_claimed") is not False
            or not all(checks.values())
            or float(report.get("safety_limit_mib") or 0) <= 0
            or float(
                (report.get("measurements") or {}).get("long_panel_peak_mib")
                or 0
            )
            <= float(report.get("safety_limit_mib") or 0)
        ):
            raise BakeoffFault("kerc_resource_disposition_invalid")
        return {
            "disposition": disposition,
            "scientific_falsification_claimed": False,
            "checks": checks,
            "metrics": report.get("measurements") or {},
            "claim_boundary": report.get("claim_boundary"),
            "reentry_condition": report.get("reentry_condition"),
        }
    if int((report.get("metrics") or {}).get("seed_count") or 0) != int(
        config["kerc_required_seed_count"]
    ):
        raise BakeoffFault("kerc_seed_count_invalid")
    mechanics = (
        "seed_count",
        "matched_parameter_count",
        "equal_optimizer_steps",
    )
    if any(checks.get(key) is not True for key in mechanics):
        raise BakeoffFault("kerc_mechanics_or_weak_tail_gate_failed")
    if disposition == "ADOPT_RDC_KERC_FIRST_CAMPAIGN" and not all(checks.values()):
        raise BakeoffFault("kerc_adoption_without_all_gates")
    return {
        "disposition": disposition,
        "scientific_falsification_claimed": False,
        "checks": checks,
        "metrics": report.get("metrics") or {},
        "claim_boundary": report.get("claim_boundary"),
        "reentry_condition": report.get("reentry_condition"),
    }


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    reports, manifest = evidence_manifest(config)
    dispositions = validate_non_kerc_evidence(reports)
    kerc = evaluate_kerc(config, reports["rdc_kerc"])
    selected = dict(config["selected_control_implementations"])
    if kerc["disposition"] == "ADOPT_RDC_KERC_FIRST_CAMPAIGN":
        selected["relational_ir"] = "impl.t0a.rdc_relational_ir.v1"
    training_config = reports["canonical_training_config"]
    kerc_training = training_config.get("kernel_english_training") or {}
    kerc_disposition = kerc_training.get("disposition") or {}
    candidate_ids = set(
        (training_config.get("comparison_contract") or {}).get(
            "first_campaign_candidate_ids"
        )
        or []
    )
    if kerc["disposition"] == "ADOPT_RDC_KERC_FIRST_CAMPAIGN":
        if (
            kerc_disposition.get("full_kerc_training_enabled") is not True
            or "english_kerc" not in candidate_ids
        ):
            raise BakeoffFault("kerc_adoption_not_applied_to_canonical_training_config")
    elif (
        kerc_disposition.get("full_kerc_training_enabled") is not False
        or int(kerc_disposition.get("first_campaign_topology_exposure", -1)) != 0
        or int(kerc_disposition.get("first_campaign_optimizer_repetitions", -1)) != 0
        or "english_kerc" in candidate_ids
    ):
        raise BakeoffFault("kerc_exclusion_not_applied_to_canonical_training_config")
    return {
        "policy": config["policy"],
        "schema_version": config["schema_version"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trigger_state": "GREEN",
        "support_state": "private-source-disjoint-factorized-evidence",
        "disposition": "factorized_architecture_selected_training_not_started",
        "campaign_id": config["campaign_id"],
        "source_artifacts": manifest,
        "selected_implementation_ids": selected,
        "topology_campaign": config["topology_campaign"],
        "candidate_dispositions": {**dispositions, "rdc_kerc": kerc},
        "hard_boundaries": config["hard_boundaries"],
        "summary": {
            "selected_slot_count": len(selected),
            "required_slot_count": len(REQUIRED_SLOTS),
            "evidence_count": len(config["evidence"]),
            "public_training_rows": 0,
            "public_evaluation_rows": 0,
            "external_inference_calls": 0,
            "fallback_template_router_tool_credit": 0,
            "production_checkpoint_mutations": 0,
            "long_training_runs": 0,
        },
        "non_claims": [
            "This selects first-campaign engineering spend; it is not a public-transfer, utility, AGI, or ASI claim.",
            "The topology comparison remains prospectively unresolved until the matched 57M candidates train and complete the frozen functional utility evaluation.",
            "Campaign exclusion is not scientific falsification; every discovery candidate retains its exact re-entry condition.",
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
    print(json.dumps({"trigger_state": report["trigger_state"], "disposition": report["disposition"], "selected_implementation_ids": report["selected_implementation_ids"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
