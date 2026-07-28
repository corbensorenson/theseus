#!/usr/bin/env python3
"""Build and replay the content-addressed pretraining architecture freeze package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import host_resource_safety


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import moecot_language_arm_training as training  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "pretraining_architecture_freeze.json"
DEFAULT_REPORT = ROOT / "reports" / "pretraining_architecture_freeze_package.json"


class ArchitectureFreezeFault(ValueError):
    pass


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def negative_disposition_ready(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "")
    if status not in {"falsified_pretraining", "retired_by_pretraining_verdict"}:
        return True
    contract = row.get("negative_disposition_contract") or {}
    kind = str(contract.get("kind") or "")
    if status == "retired_by_pretraining_verdict" and kind == "campaign_scope_only":
        return (
            contract.get("scientific_falsification_claimed") is False
            and bool(str(contract.get("exact_scope") or ""))
            and bool(str(contract.get("reentry_condition") or ""))
        )
    required = (
        "mechanism_fidelity_audited",
        "learnability_sanity_passed",
        "matched_opportunity_audited",
        "independent_construct_valid_evaluation",
        "multi_seed_uncertainty_and_power_reported",
        "replicated",
    )
    return (
        kind == "decision_grade_negative"
        and all(contract.get(key) is True for key in required)
        and bool(str(contract.get("exact_claim_scope") or ""))
    )


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("policy") != "project_theseus_pretraining_architecture_freeze_v1":
        raise ArchitectureFreezeFault("policy_invalid")
    if len(config.get("required_artifacts") or []) < 40:
        raise ArchitectureFreezeFault("artifact_closure_too_small")
    if len(config.get("replay_commands") or []) < 6:
        raise ArchitectureFreezeFault("replay_closure_too_small")
    boundaries = config.get("boundaries") or {}
    if boundaries.get("long_optimizer_run_allowed_during_freeze") is not False or boundaries.get("public_calibration_allowed_during_freeze") is not False:
        raise ArchitectureFreezeFault("freeze_boundary_invalid")
    if any(int(boundaries.get(key, -1)) != 0 for key in ("public_training_rows", "external_inference_calls", "fallback_or_template_credit")):
        raise ArchitectureFreezeFault("no_cheat_boundary_nonzero")
    replay_safety_policy(config)
    return config


def architecture_dispositions(config: dict[str, Any]) -> dict[str, Any]:
    matrix_path = resolve(config["matrix"])
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    architecture = matrix.get("pre_training_architecture_contract") or {}
    required = set(architecture.get("required_backlog_ids") or [])
    ready_statuses = set(config["required_ready_backlog_statuses"])
    backlog = {
        row.get("backlog_id"): row
        for row in matrix.get("planned_codex_test_backlog") or []
        if isinstance(row, dict) and row.get("backlog_id") in required
    }
    missing = sorted(required - set(backlog))
    unready = sorted(
        backlog_id for backlog_id, row in backlog.items()
        if row.get("status") not in ready_statuses
        or not row.get("pre_training_acceptance_boundary")
        or not negative_disposition_ready(row)
    )
    if missing or unready:
        raise ArchitectureFreezeFault(
            "architecture_disposition_incomplete:missing=" + ",".join(missing)
            + ";unready=" + ",".join(unready)
        )
    rows = {
        backlog_id: {
            "status": backlog[backlog_id]["status"],
            "acceptance_boundary_digest": digest(backlog[backlog_id]["pre_training_acceptance_boundary"]),
            "evidence": backlog[backlog_id].get("pre_training_evidence"),
            "negative_disposition": backlog[backlog_id].get(
                "negative_disposition_contract"
            ),
        }
        for backlog_id in sorted(required)
    }
    return {
        "required_count": len(required),
        "ready_count": len(rows),
        "rows": rows,
        "matrix_sha256": sha256(matrix_path),
    }


def artifact_manifest(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = {}
    for value in config["required_artifacts"]:
        path = resolve(value)
        if not path.is_file():
            raise ArchitectureFreezeFault(f"required_artifact_missing:{value}")
        manifest[value] = {"path": value, "sha256": sha256(path), "bytes": path.stat().st_size}
    return manifest


def receipt_manifest(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = {}
    for value in config.get("generated_receipts") or []:
        path = resolve(value)
        if not path.is_file():
            raise ArchitectureFreezeFault(f"generated_receipt_missing:{value}")
        manifest[value] = {
            "path": value,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return manifest


def factorized_selection(config: dict[str, Any]) -> dict[str, Any]:
    contract = config.get("factorized_bakeoff") or {}
    if not contract:
        raise ArchitectureFreezeFault("factorized_bakeoff_contract_missing")
    path = resolve(str(contract.get("path") or ""))
    if not path.is_file():
        raise ArchitectureFreezeFault("factorized_bakeoff_report_missing")
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "policy": contract.get("policy"),
        "trigger_state": contract.get("required_trigger_state", "GREEN"),
        "disposition": contract.get("required_disposition"),
        "campaign_id": config.get("campaign_id"),
    }
    actual = {key: report.get(key) for key in expected}
    if actual != expected:
        raise ArchitectureFreezeFault(
            "factorized_bakeoff_contract_mismatch:"
            + canonical({"expected": expected, "actual": actual})
        )
    selected = report.get("selected_implementation_ids") or {}
    if len(selected) != 7 or any(not str(value) for value in selected.values()):
        raise ArchitectureFreezeFault("factorized_bakeoff_selection_incomplete")
    summary = report.get("summary") or {}
    for key in (
        "public_training_rows",
        "public_evaluation_rows",
        "external_inference_calls",
        "fallback_template_router_tool_credit",
        "production_checkpoint_mutations",
        "long_training_runs",
    ):
        if int(summary.get(key, -1)) != 0:
            raise ArchitectureFreezeFault(f"factorized_bakeoff_boundary_nonzero:{key}")
    return {
        "path": str(contract["path"]),
        "sha256": sha256(path),
        "policy": report["policy"],
        "disposition": report["disposition"],
        "campaign_id": report["campaign_id"],
        "selected_implementation_ids": selected,
        "candidate_dispositions": report.get("candidate_dispositions") or {},
        "topology_campaign": report.get("topology_campaign") or {},
    }


def selected_route_execution_qualification(
    config: dict[str, Any],
) -> dict[str, Any]:
    contract = config.get("selected_route_execution_qualification") or {}
    required = {
        "policy",
        "training_config",
        "required_plan_sha256",
        "sustained_report",
        "sustained_policy",
        "fresh_process_report",
        "fresh_process_policy",
        "canonical_receipt",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ArchitectureFreezeFault(
            "selected_route_execution_contract_incomplete:" + ",".join(missing)
        )
    if contract.get("policy") != (
        "project_theseus_selected_route_execution_qualification_v1"
    ):
        raise ArchitectureFreezeFault(
            "selected_route_execution_contract_invalid"
        )

    training_config_path = resolve(str(contract["training_config"]))
    training_config = training.bind_scale_preregistration(
        training.read_json(training_config_path)
    )
    plan = training.build_plan(
        training_config,
        config_path=training_config_path,
    )
    required_plan = str(contract["required_plan_sha256"])
    if (
        plan.get("trigger_state") != "GREEN"
        or plan.get("hard_gaps")
        or plan.get("plan_sha256") != required_plan
    ):
        raise ArchitectureFreezeFault(
            "selected_route_training_plan_not_green_or_identity_mismatch"
        )

    sustained_path = resolve(str(contract["sustained_report"]))
    fresh_path = resolve(str(contract["fresh_process_report"]))
    canonical_receipt_path = resolve(str(contract["canonical_receipt"]))
    for label, path in (
        ("sustained", sustained_path),
        ("fresh_process", fresh_path),
        ("canonical_receipt", canonical_receipt_path),
    ):
        if not path.is_file():
            raise ArchitectureFreezeFault(
                f"selected_route_{label}_artifact_missing"
            )
    sustained = json.loads(sustained_path.read_text(encoding="utf-8"))
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    canonical_receipt = json.loads(
        canonical_receipt_path.read_text(encoding="utf-8")
    )
    target = plan["targets"][training.SHARED_TRUNK_ID]
    expected_receipt_path = training.resolve(str(target["receipt"]))
    if expected_receipt_path != canonical_receipt_path:
        raise ArchitectureFreezeFault(
            "selected_route_canonical_receipt_path_mismatch"
        )
    migration = None
    if canonical_receipt.get("plan_sha256") != required_plan:
        migration = training.accepted_plan_identity_migration(
            canonical_receipt,
            plan,
            target,
        )
        if migration is None:
            raise ArchitectureFreezeFault(
                "selected_route_canonical_plan_migration_missing"
            )

    canonical_paths = {
        "checkpoint": training.resolve(str(canonical_receipt["checkpoint"])),
        "optimizer_state": training.resolve(
            str(canonical_receipt["optimizer_state"])
        ),
        "mlx_rng_state": training.resolve(
            str(canonical_receipt["mlx_rng_state"])
        ),
        "receipt": canonical_receipt_path,
    }
    for label, path in canonical_paths.items():
        if not path.is_file():
            raise ArchitectureFreezeFault(
                f"selected_route_canonical_{label}_missing"
            )
    canonical_identity = {
        "checkpoint_sha256": sha256(canonical_paths["checkpoint"]),
        "optimizer_state_sha256": sha256(
            canonical_paths["optimizer_state"]
        ),
        "mlx_rng_state_sha256": sha256(
            canonical_paths["mlx_rng_state"]
        ),
        "receipt_sha256": sha256(canonical_paths["receipt"]),
    }
    receipt_identity = {
        "checkpoint_sha256": canonical_receipt.get("checkpoint_sha256"),
        "optimizer_state_sha256": canonical_receipt.get(
            "optimizer_state_sha256"
        ),
        "mlx_rng_state_sha256": canonical_receipt.get(
            "mlx_rng_state_sha256"
        ),
        "receipt_sha256": canonical_identity["receipt_sha256"],
    }
    if canonical_identity != receipt_identity:
        raise ArchitectureFreezeFault(
            "selected_route_canonical_artifact_identity_mismatch"
        )

    training_config_sha256 = sha256(training_config_path)
    sustained_training_config = sustained.get("training_config") or {}
    fresh_training_config = fresh.get("training_config") or {}
    report_training_config_sha256 = sustained_training_config.get("sha256")
    if (
        sustained_training_config.get("path") != str(contract["training_config"])
        or fresh_training_config.get("path") != str(contract["training_config"])
        or not report_training_config_sha256
        or fresh_training_config.get("sha256")
        != report_training_config_sha256
    ):
        raise ArchitectureFreezeFault(
            "selected_route_report_training_config_identity_mismatch"
        )
    sustained_thermal = sustained.get("thermal_stability") or {}
    sustained_valid = bool(
        sustained.get("policy") == contract["sustained_policy"]
        and sustained.get("trigger_state") == "GREEN"
        and sustained.get("support_state") == "SUPPORTED"
        and not sustained.get("hard_gaps")
        and sustained.get("plan_sha256") == required_plan
        and int(sustained.get("successful_segment_count") or 0) >= 6
        and sustained.get("exact_resume_each_segment") is True
        and sustained.get("canonical_lineage_unchanged") is True
        and sustained.get("all_segments_on_ac_power") is True
        and sustained.get("system_swap_growth_treatment")
        == "DIAGNOSTIC_ONLY"
        and sustained.get("arbitrary_percentage_tolerance") is None
        and sustained.get("elapsed_time_requirement") is None
        and sustained_thermal.get("state")
        == "STABLE_WITHIN_OBSERVED_REPLICATE_UNCERTAINTY"
        and sustained_thermal.get("terminal") is True
        and sustained_thermal.get("uncertainty_interval_contains_one")
        is True
        and sustained.get("canonical_before") == canonical_identity
        and sustained.get("canonical_after") == canonical_identity
    )
    if not sustained_valid:
        raise ArchitectureFreezeFault(
            "selected_route_sustained_qualification_invalid"
        )

    fresh_valid = bool(
        fresh.get("policy") == contract["fresh_process_policy"]
        and fresh.get("trigger_state") == "GREEN"
        and not fresh.get("hard_gaps")
        and fresh.get("plan_sha256") == required_plan
        and int(fresh.get("contiguous_segment_count") or 0) >= 2
        and fresh.get("canonical_lineage_unchanged") is True
        and fresh.get("exact_resume_validation") is True
        and fresh.get("independent_segmented_replay_numeric_parity")
        is True
        and fresh.get("segmented_cursor_and_position_contiguous") is True
        and fresh.get("host_resource_guard_passed") is True
        and fresh.get("swap_growth_treatment") == "DIAGNOSTIC_ONLY"
        and fresh.get("canonical_before") == canonical_identity
        and fresh.get("canonical_after") == canonical_identity
        and (fresh.get("model_tensor_comparison") or {}).get("passed")
        is True
        and (fresh.get("optimizer_tensor_comparison") or {}).get("passed")
        is True
    )
    if not fresh_valid:
        raise ArchitectureFreezeFault(
            "selected_route_fresh_process_qualification_invalid"
        )
    host_policy = training_config.get("host_resource_safety") or {}
    if (
        host_policy.get("swapout_growth_action") != "report_only"
        or float(host_policy.get("minimum_available_before_launch_mib") or 0)
        != 0.0
        or float(host_policy.get("minimum_available_during_run_mib") or 0)
        != 0.0
        or float(host_policy.get("maximum_wall_seconds") or 0) != 0.0
        or host_policy.get("memory_guard_mode") != "predicted_exhaustion"
    ):
        raise ArchitectureFreezeFault(
            "selected_route_evidence_driven_host_policy_mismatch"
        )
    return {
        "policy": contract["policy"],
        "training_config": {
            "path": str(contract["training_config"]),
            "sha256": training_config_sha256,
            "qualification_report_source_sha256": (
                report_training_config_sha256
            ),
            "semantic_plan_identity_unchanged": True,
        },
        "training_plan_sha256": required_plan,
        "canonical_receipt": {
            "path": str(contract["canonical_receipt"]),
            "sha256": canonical_identity["receipt_sha256"],
            "optimizer_steps": canonical_receipt.get("optimizer_steps"),
            "optimizer_positions": canonical_receipt.get(
                "optimizer_positions"
            ),
            "resume_plan_identity_migration": migration,
        },
        "canonical_identity": canonical_identity,
        "sustained_report": {
            "path": str(contract["sustained_report"]),
            "sha256": sha256(sustained_path),
            "successful_segment_count": sustained[
                "successful_segment_count"
            ],
            "thermal_stability": sustained_thermal,
        },
        "fresh_process_report": {
            "path": str(contract["fresh_process_report"]),
            "sha256": sha256(fresh_path),
            "contiguous_segment_count": fresh[
                "contiguous_segment_count"
            ],
            "numeric_replay_parity": fresh[
                "independent_segmented_replay_numeric_parity"
            ],
            "zero_swap_growth": fresh["zero_swap_growth"],
            "swap_growth_treatment": fresh["swap_growth_treatment"],
        },
        "capability_claim": "NONE_EXECUTION_QUALIFICATION_ONLY",
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def accelerator_shard_policy(
    contract: dict[str, Any], shard: dict[str, Any]
) -> host_resource_safety.HostSafetyPolicy:
    policy = host_resource_safety.HostSafetyPolicy(
        max_process_memory_mib=float(shard["maximum_process_memory_mib"]),
        minimum_available_before_launch_mib=float(
            shard.get(
                "minimum_available_before_launch_mib",
                contract["minimum_available_before_launch_mib"],
            )
        ),
        minimum_available_during_run_mib=float(
            shard["minimum_available_memory_mib"]
        ),
        maximum_swapout_growth_mib=float(shard["maximum_swapout_growth_mib"]),
        maximum_wall_seconds=float(shard["max_wall_seconds"]),
        poll_interval_seconds=float(shard["poll_interval_seconds"]),
        terminate_grace_seconds=float(
            contract.get("terminate_grace_seconds") or 2.0
        ),
        swapout_growth_action=str(
            shard.get("swapout_growth_action") or "hard_stop"
        ),
        memory_guard_mode=str(
            shard.get("memory_guard_mode") or "fixed_reserve"
        ),
        qualified_peak_inferred_unified_memory_mib=float(
            shard.get("qualified_peak_inferred_unified_memory_mib") or 0.0
        ),
    )
    policy.validate(
        physical_memory_mib=host_resource_safety.physical_memory_mib()
    )
    return policy


def replay_safety_policy(config: dict[str, Any]) -> host_resource_safety.HostSafetyPolicy:
    contract = config.get("replay_safety") or {}
    required = {
        "watchdog_policy",
        "receipt_directory",
        "risk_class",
        "resource_basis",
        "maximum_process_memory_mib",
        "minimum_available_before_launch_mib",
        "minimum_available_during_run_mib",
        "maximum_swapout_growth_mib",
        "maximum_wall_seconds",
        "poll_interval_seconds",
        "terminate_grace_seconds",
        "swapout_growth_action",
        "memory_guard_mode",
        "qualified_peak_inferred_unified_memory_mib",
        "accelerator_authorization_allowed",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ArchitectureFreezeFault(
            "replay_safety_contract_incomplete:" + ",".join(missing)
        )
    receipt_directory = Path(str(contract["receipt_directory"]))
    if (
        contract["watchdog_policy"] != host_resource_safety.POLICY
        or contract["accelerator_authorization_allowed"] is not False
        or not str(contract.get("risk_class") or "")
        or not str(contract.get("resource_basis") or "")
        or receipt_directory.is_absolute()
        or not receipt_directory.parts
        or receipt_directory.parts[0] != "reports"
        or ".." in receipt_directory.parts
    ):
        raise ArchitectureFreezeFault("replay_safety_contract_invalid")
    policy = host_resource_safety.HostSafetyPolicy(
        max_process_memory_mib=float(contract["maximum_process_memory_mib"]),
        minimum_available_before_launch_mib=float(
            contract["minimum_available_before_launch_mib"]
        ),
        minimum_available_during_run_mib=float(
            contract["minimum_available_during_run_mib"]
        ),
        maximum_swapout_growth_mib=float(contract["maximum_swapout_growth_mib"]),
        maximum_wall_seconds=float(contract["maximum_wall_seconds"]),
        poll_interval_seconds=float(contract["poll_interval_seconds"]),
        terminate_grace_seconds=float(contract["terminate_grace_seconds"]),
        swapout_growth_action=str(contract["swapout_growth_action"]),
        memory_guard_mode=str(contract["memory_guard_mode"]),
        qualified_peak_inferred_unified_memory_mib=float(
            contract["qualified_peak_inferred_unified_memory_mib"]
        ),
    )
    policy.validate(
        physical_memory_mib=host_resource_safety.physical_memory_mib()
    )
    return policy


def validate_accelerator_shard_contract(
    contract: dict[str, Any], shard: dict[str, Any]
) -> None:
    command = [str(value) for value in shard.get("command") or []]
    qualified_python = str(contract.get("qualified_python") or "")
    if not command or command[0] != qualified_python:
        raise ArchitectureFreezeFault(
            "accelerator_shard_qualified_python_mismatch:"
            + str(shard.get("id") or "missing")
        )
    receipt = str(shard.get("receipt") or "")
    if not receipt.startswith("reports/accelerator_replay/") or not receipt.endswith(
        ".json"
    ):
        raise ArchitectureFreezeFault(
            "accelerator_shard_receipt_path_invalid:"
            + str(shard.get("id") or "missing")
        )
    memory_guard_mode = str(
        shard.get("memory_guard_mode") or "fixed_reserve"
    )
    if float(shard.get("max_wall_seconds") or 0) < 0 or (
        memory_guard_mode == "fixed_reserve"
        and float(shard.get("max_wall_seconds") or 0) == 0
    ):
        raise ArchitectureFreezeFault(
            "accelerator_shard_wall_limit_invalid:"
            + str(shard.get("id") or "missing")
        )
    shard_ids = [str(row.get("id") or "") for row in contract.get("shards") or []]
    shard_id = str(shard.get("id") or "")
    dependencies = [str(value) for value in shard.get("depends_on_shards") or []]
    if (
        not shard_id
        or shard_id not in shard_ids
        or len(dependencies) != len(set(dependencies))
        or shard_id in dependencies
        or any(value not in shard_ids for value in dependencies)
        or any(shard_ids.index(value) >= shard_ids.index(shard_id) for value in dependencies)
    ):
        raise ArchitectureFreezeFault(
            "accelerator_shard_dependency_contract_invalid:" + shard_id
        )
    generated_artifacts = [str(value) for value in shard.get("generated_artifacts") or []]
    if len(generated_artifacts) != len(set(generated_artifacts)) or any(
        not value.startswith("reports/") for value in generated_artifacts
    ):
        raise ArchitectureFreezeFault(
            "accelerator_shard_generated_artifacts_invalid:"
            + str(shard.get("id") or "missing")
        )
    required_resource_fields = {
        "risk_class",
        "resource_basis",
        "maximum_process_memory_mib",
        "minimum_available_memory_mib",
        "maximum_swapout_growth_mib",
        "poll_interval_seconds",
    }
    missing = sorted(required_resource_fields - set(shard))
    if missing:
        raise ArchitectureFreezeFault(
            "accelerator_shard_resource_contract_incomplete:"
            + str(shard.get("id") or "missing")
            + ":"
            + ",".join(missing)
        )
    if memory_guard_mode == "fixed_reserve" and "minimum_available_before_launch_mib" in shard and float(
        shard["minimum_available_before_launch_mib"]
    ) < float(shard["minimum_available_memory_mib"]):
        raise ArchitectureFreezeFault(
            "accelerator_shard_launch_reserve_below_live_reserve:"
            + str(shard.get("id") or "missing")
        )
    calibration = shard.get("measured_launch_calibration")
    if calibration is not None:
        required_calibration_fields = {
            "source_attempt_commit",
            "source_receipt",
            "qualification_authority",
            "observed_maximum_inferred_unified_memory_mib",
            "minimum_live_reserve_mib",
            "minimum_safety_margin_mib",
            "required_launch_floor_mib",
            "selected_launch_floor_mib",
        }
        if not isinstance(calibration, dict) or (
            required_calibration_fields - set(calibration)
        ):
            raise ArchitectureFreezeFault(
                "accelerator_shard_launch_calibration_incomplete:" + shard_id
            )
        source_commit = str(calibration["source_attempt_commit"])
        if len(source_commit) != 40 or any(
            value not in "0123456789abcdef" for value in source_commit
        ):
            raise ArchitectureFreezeFault(
                "accelerator_shard_launch_calibration_commit_invalid:" + shard_id
            )
        if (
            str(calibration["source_receipt"]) != receipt
            or calibration["qualification_authority"] is not False
        ):
            raise ArchitectureFreezeFault(
                "accelerator_shard_launch_calibration_authority_invalid:" + shard_id
            )
        observed_peak = float(
            calibration["observed_maximum_inferred_unified_memory_mib"]
        )
        live_reserve = float(calibration["minimum_live_reserve_mib"])
        safety_margin = float(calibration["minimum_safety_margin_mib"])
        required_launch_floor = float(calibration["required_launch_floor_mib"])
        selected_launch_floor = float(calibration["selected_launch_floor_mib"])
        derived_launch_floor = live_reserve + observed_peak + safety_margin
        if memory_guard_mode == "predicted_exhaustion":
            if (
                float(shard["minimum_available_before_launch_mib"]) != 0.0
                or float(shard["minimum_available_memory_mib"]) != 0.0
                or float(shard.get("max_wall_seconds") or 0) != 0.0
                or shard.get("swapout_growth_action") != "report_only"
                or float(
                    shard.get(
                        "qualified_peak_inferred_unified_memory_mib"
                    )
                    or 0.0
                )
                < observed_peak
            ):
                raise ArchitectureFreezeFault(
                    "accelerator_shard_launch_calibration_weakened:"
                    + shard_id
                )
        elif (
            observed_peak <= 0
            or safety_margin <= 0
            or abs(live_reserve - float(shard["minimum_available_memory_mib"]))
            > 1e-6
            or abs(required_launch_floor - derived_launch_floor) > 1e-6
            or selected_launch_floor < required_launch_floor
            or abs(
                selected_launch_floor
                - float(shard["minimum_available_before_launch_mib"])
            )
            > 1e-6
        ):
            raise ArchitectureFreezeFault(
                "accelerator_shard_launch_calibration_weakened:" + shard_id
            )
    if (
        float(shard["maximum_process_memory_mib"])
        > float(contract["maximum_process_memory_mib"])
        or (
            memory_guard_mode == "fixed_reserve"
            and float(shard["minimum_available_memory_mib"])
            < float(contract["minimum_available_memory_mib"])
        )
        or float(shard["maximum_swapout_growth_mib"])
        > float(contract["maximum_swapout_growth_mib"])
        or float(shard["poll_interval_seconds"])
        > float(contract["poll_interval_seconds"])
    ):
        raise ArchitectureFreezeFault(
            "accelerator_shard_resource_contract_weaker_than_global:"
            + str(shard.get("id") or "missing")
        )
    accelerator_shard_policy(contract, shard)


def shard_generated_artifact_manifest(shard: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for value in shard.get("generated_artifacts") or []:
        relative = str(value)
        path = resolve(relative)
        if not path.is_file():
            raise ArchitectureFreezeFault(
                "accelerator_shard_generated_artifact_missing:"
                + str(shard.get("id") or "missing")
                + ":"
                + relative
            )
        manifest[relative] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return manifest


def shard_implementation_artifact_manifest(
    shard: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for value in shard.get("implementation_artifacts") or []:
        relative = str(value)
        path = resolve(relative)
        if not path.is_file():
            raise ArchitectureFreezeFault(
                "accelerator_shard_implementation_artifact_missing:"
                + str(shard.get("id") or "missing")
                + ":"
                + relative
            )
        manifest[relative] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return manifest


def accelerator_dependency_receipt_manifest(
    contract: dict[str, Any], shard: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    by_id = {
        str(row.get("id") or ""): row for row in contract.get("shards") or []
    }
    manifest: dict[str, dict[str, Any]] = {}
    for dependency_id in shard.get("depends_on_shards") or []:
        dependency = by_id[str(dependency_id)]
        receipt_path = resolve(str(dependency["receipt"]))
        if not receipt_path.is_file():
            raise ArchitectureFreezeFault(
                "accelerator_shard_dependency_not_ready:"
                + str(shard.get("id") or "missing")
                + ":"
                + str(dependency_id)
            )
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArchitectureFreezeFault(
                "accelerator_shard_dependency_not_ready:"
                + str(shard.get("id") or "missing")
                + ":"
                + str(dependency_id)
            ) from exc
        if not accelerator_receipt_valid(contract, dependency, receipt):
            raise ArchitectureFreezeFault(
                "accelerator_shard_dependency_not_ready:"
                + str(shard.get("id") or "missing")
                + ":"
                + str(dependency_id)
            )
        manifest[str(dependency_id)] = {
            "receipt": str(dependency["receipt"]),
            "sha256": sha256(receipt_path),
        }
    return manifest


def accelerator_receipt_valid(
    contract: dict[str, Any], shard: dict[str, Any], receipt: dict[str, Any]
) -> bool:
    limits = receipt.get("limits") or {}
    expected_limits = asdict(accelerator_shard_policy(contract, shard))

    def limit_matches(key: str, expected: Any) -> bool:
        if key not in limits:
            if key == "memory_guard_mode":
                return expected == "fixed_reserve"
            if key == "qualified_peak_inferred_unified_memory_mib":
                return float(expected) == 0.0
            return False
        actual = limits[key]
        if isinstance(expected, str):
            return actual == expected
        return float(actual) == float(expected)

    try:
        expected_artifacts = shard_generated_artifact_manifest(shard)
        expected_implementation = shard_implementation_artifact_manifest(shard)
        expected_dependencies = accelerator_dependency_receipt_manifest(contract, shard)
    except ArchitectureFreezeFault:
        return False
    return (
        receipt.get("policy") == contract.get("watchdog_policy")
        and receipt.get("passed") is True
        and receipt.get("child_started") is True
        and receipt.get("terminated_by_guard") is False
        and int(receipt.get("returncode", -1)) == 0
        and receipt.get("command") == shard.get("command")
        and isinstance(receipt.get("maximum_inferred_unified_memory_mib"), (int, float))
        and (receipt.get("generated_artifacts") or {}) == expected_artifacts
        and (receipt.get("implementation_artifacts") or {})
        == expected_implementation
        and (receipt.get("dependency_receipts") or {}) == expected_dependencies
        and all(limit_matches(key, expected) for key, expected in expected_limits.items())
    )


def run_accelerator_shards(
    config: dict[str, Any], *, selected_ids: set[str] | None = None
) -> dict[str, Any]:
    """Run exact MLX shards one at a time under the external host guard."""

    contract = config.get("accelerator_replay") or {}
    shards = contract.get("shards") or []
    known_ids = {str(row.get("id") or "") for row in shards}
    requested = known_ids if selected_ids is None else set(selected_ids)
    unknown = sorted(requested - known_ids)
    if unknown:
        raise ArchitectureFreezeFault(
            "accelerator_shard_unknown:" + ",".join(unknown)
        )
    receipts: list[dict[str, Any]] = []
    attempted = 0
    reused = 0
    for shard in shards:
        shard_id = str(shard.get("id") or "")
        if shard_id not in requested:
            continue
        validate_accelerator_shard_contract(contract, shard)
        receipt_path = resolve(str(shard["receipt"]))
        try:
            dependency_receipts = accelerator_dependency_receipt_manifest(
                contract, shard
            )
        except ArchitectureFreezeFault as exc:
            policy = accelerator_shard_policy(contract, shard)
            receipt = {
                "policy": host_resource_safety.POLICY,
                "command": [str(value) for value in shard["command"]],
                "passed": False,
                "child_started": False,
                "terminated_by_guard": False,
                "fault": str(exc),
                "returncode": None,
                "limits": asdict(policy),
                "dependency_receipts": {},
            }
            atomic_write_json(receipt_path, receipt)
            receipts.append(
                {
                    "id": shard_id,
                    "receipt": str(shard["receipt"]),
                    "passed": False,
                    "fault": str(exc),
                    "child_started": False,
                    "reused": False,
                }
            )
            break
        if receipt_path.is_file():
            try:
                existing = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                existing = {}
            if accelerator_receipt_valid(contract, shard, existing):
                reused += 1
                receipts.append(
                    {
                        "id": shard_id,
                        "receipt": str(shard["receipt"]),
                        "passed": True,
                        "fault": "",
                        "child_started": False,
                        "reused": True,
                    }
                )
                continue
        policy = accelerator_shard_policy(contract, shard)
        command = [str(value) for value in shard["command"]]
        attempted += 1
        try:
            result = host_resource_safety.run_guarded(
                command,
                cwd=ROOT,
                policy=policy,
                env={"THESEUS_GUARDED_ACCELERATOR_CHILD": "1"},
            )
            receipt = result.receipt
            receipt["stdout_tail"] = result.stdout[-16000:]
            receipt["stderr_tail"] = result.stderr[-16000:]
        except host_resource_safety.HostResourceSafetyFault as exc:
            receipt = {
                "policy": host_resource_safety.POLICY,
                "command": command,
                "passed": False,
                "child_started": False,
                "terminated_by_guard": False,
                "fault": str(exc),
                "returncode": None,
                "limits": asdict(policy),
            }
        if receipt.get("passed") is True:
            try:
                receipt["generated_artifacts"] = shard_generated_artifact_manifest(shard)
                receipt["implementation_artifacts"] = (
                    shard_implementation_artifact_manifest(shard)
                )
            except ArchitectureFreezeFault as exc:
                receipt["passed"] = False
                receipt["fault"] = str(exc)
                receipt["generated_artifacts"] = {}
        receipt["dependency_receipts"] = dependency_receipts
        atomic_write_json(receipt_path, receipt)
        receipts.append(
            {
                "id": shard_id,
                "receipt": str(shard["receipt"]),
                "passed": receipt.get("passed") is True,
                "fault": str(receipt.get("fault") or ""),
                "child_started": receipt.get("child_started", True),
                "reused": False,
            }
        )
        if receipt.get("passed") is not True:
            break
    return {
        "policy": "project_theseus_guarded_accelerator_replay_v1",
        "requested_shard_count": len(requested),
        "processed_shard_count": len(receipts),
        "attempted_shard_count": attempted,
        "reused_shard_count": reused,
        "passed_shard_count": sum(row["passed"] for row in receipts),
        "complete": len(receipts) == len(requested)
        and all(row["passed"] for row in receipts),
        "receipts": receipts,
    }
def accelerator_replay_receipts(config: dict[str, Any]) -> dict[str, Any]:
    contract = config.get("accelerator_replay") or {}
    if contract.get("required_before_freeze") is not True:
        raise ArchitectureFreezeFault("accelerator_replay_not_required")
    if contract.get("direct_mlx_import_in_ambient_python_forbidden") is not True:
        raise ArchitectureFreezeFault("ambient_mlx_import_not_forbidden")
    if contract.get("watchdog_policy") != "project_theseus_host_resource_safety_v1":
        raise ArchitectureFreezeFault("accelerator_watchdog_policy_invalid")
    if contract.get("readiness_authority") != "exact_guard_receipts_only":
        raise ArchitectureFreezeFault("accelerator_readiness_authority_invalid")
    if contract.get("state") != "RECEIPT_DERIVED":
        raise ArchitectureFreezeFault("accelerator_replay_state_not_receipt_derived")
    qualified_python = resolve(str(contract.get("qualified_python") or ""))
    if not qualified_python.is_file():
        raise ArchitectureFreezeFault("qualified_accelerator_python_missing")
    shards = contract.get("shards") or []
    shard_ids = [str(row.get("id") or "") for row in shards]
    if (
        len(shards) < 8
        or any(not value for value in shard_ids)
        or len(set(shard_ids)) != len(shard_ids)
    ):
        raise ArchitectureFreezeFault("accelerator_replay_shards_incomplete")
    manifest: dict[str, Any] = {}
    for shard in shards:
        validate_accelerator_shard_contract(contract, shard)
        value = str(shard.get("receipt") or "")
        path = resolve(str(value))
        if not path.is_file():
            raise ArchitectureFreezeFault(f"accelerator_replay_receipt_missing:{value}")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if not accelerator_receipt_valid(contract, shard, receipt):
            raise ArchitectureFreezeFault(f"accelerator_replay_receipt_invalid:{value}")
        manifest[str(shard["id"])] = {
            "path": str(value),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    if not manifest:
        raise ArchitectureFreezeFault("accelerator_replay_receipts_empty")
    return manifest


def run_replays(config: dict[str, Any]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    policy = replay_safety_policy(config)
    receipt_directory = resolve(config["replay_safety"]["receipt_directory"])
    for index, command in enumerate(config["replay_commands"]):
        exact_command = [str(value) for value in command]
        started = time.perf_counter()
        try:
            result = host_resource_safety.run_guarded(
                exact_command,
                cwd=ROOT,
                policy=policy,
                env={"THESEUS_GUARDED_REPLAY_CHILD": "1"},
            )
            stdout = result.stdout
            stderr = result.stderr
            safety_receipt = result.receipt
        except host_resource_safety.HostResourceSafetyFault as exc:
            stdout = ""
            stderr = ""
            safety_receipt = {
                "policy": host_resource_safety.POLICY,
                "command": exact_command,
                "passed": False,
                "child_started": False,
                "terminated_by_guard": False,
                "fault": str(exc),
                "returncode": None,
                "limits": asdict(policy),
            }
        receipt = {
            "index": index,
            "command": exact_command,
            "passed": safety_receipt.get("passed") is True,
            "returncode": safety_receipt.get("returncode"),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            "stdout_tail": stdout[-1200:],
            "stderr_tail": stderr[-1200:],
            "safety_receipt": safety_receipt,
        }
        receipt_relative = str(
            Path(config["replay_safety"]["receipt_directory"]) / f"{index:02d}.json"
        )
        receipt_path = receipt_directory / f"{index:02d}.json"
        atomic_write_json(receipt_path, receipt)
        receipt["receipt_path"] = receipt_relative
        receipt["receipt_sha256"] = sha256(receipt_path)
        receipts.append(receipt)
        if receipt["passed"] is not True:
            raise ArchitectureFreezeFault(
                f"replay_guard_failed:{index}:{safety_receipt.get('fault') or safety_receipt.get('returncode')}:{' '.join(exact_command)}"
            )
    return receipts


def build_report(config: dict[str, Any], *, execute_replays: bool) -> dict[str, Any]:
    dispositions = architecture_dispositions(config)
    selected_route = selected_route_execution_qualification(config)
    replays = run_replays(config) if execute_replays else []
    if not execute_replays:
        raise ArchitectureFreezeFault("independent_replay_required")
    accelerator_receipts = accelerator_replay_receipts(config)
    manifest = artifact_manifest(config)
    receipts = receipt_manifest(config)
    selection = factorized_selection(config)
    package_identity = digest({
        "campaign_id": config["campaign_id"],
        "artifacts": manifest,
        "generated_receipts": receipts,
        "accelerator_replay_receipts": accelerator_receipts,
        "dispositions": dispositions,
        "factorized_selection": selection,
        "selected_route_execution_qualification": selected_route,
        "commands": config["replay_commands"],
        "replay_safety": config["replay_safety"],
        "boundaries": config["boundaries"],
    })
    return {
        "policy": config["policy"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trigger_state": "GREEN",
        "support_state": "replayable-reference-backed",
        "disposition": "architecture_frozen_training_not_started",
        "campaign_id": config["campaign_id"],
        "package_identity": package_identity,
        "source_artifacts": manifest,
        "generated_receipts": receipts,
        "architecture_dispositions": dispositions,
        "factorized_selection": selection,
        "selected_route_execution_qualification": selected_route,
        "replay_receipts": replays,
        "replay_safety": config["replay_safety"],
        "summary": {
            "artifact_count": len(manifest),
            "generated_receipt_count": len(receipts),
            "accelerator_replay_receipt_count": len(accelerator_receipts),
            "architecture_contract_count": dispositions["required_count"],
            "ready_architecture_contract_count": dispositions["ready_count"],
            "replay_count": len(replays),
            "replay_pass_count": sum(row["returncode"] == 0 for row in replays),
            "long_optimizer_steps": 0,
            "public_calibrations": 0,
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "fallback_or_template_credit": 0,
        },
        "boundaries": config["boundaries"],
        "non_claims": [
            "Architecture freeze and bounded mechanics replay are not model training or capability evidence.",
            "The package authorizes only the exact content-addressed campaign after the external readiness gate revalidates this report.",
            "Any listed artifact change invalidates this package and requires a new replay before long training.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_REPORT))
    parser.add_argument("--execute-replays", action="store_true")
    parser.add_argument("--run-accelerator-shards", action="store_true")
    parser.add_argument("--accelerator-shard", action="append", default=[])
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = load_config(config_path)
    if args.run_accelerator_shards:
        replay = run_accelerator_shards(
            config,
            selected_ids=set(args.accelerator_shard) if args.accelerator_shard else None,
        )
        print(json.dumps(replay, indent=2, sort_keys=True))
        if not replay["complete"]:
            return 2
        if not args.execute_replays:
            return 0
    elif args.accelerator_shard:
        parser.error("--accelerator-shard requires --run-accelerator-shards")
    report = build_report(config, execute_replays=args.execute_replays)
    output = resolve(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"trigger_state": report["trigger_state"], "package_identity": report["package_identity"], "summary": report["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
