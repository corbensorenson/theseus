#!/usr/bin/env python3
"""Publish the content-addressed step-11,416 replacement freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import moecot_language_arm_training as training  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/pre_long_run_replacement_freeze.json"
POLICY = "project_theseus_pre_long_run_replacement_freeze_v1"


class ReplacementFreezeFault(ValueError):
    pass


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReplacementFreezeFault(f"json_object_required:{relative(path)}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def content_identity(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("package_identity", None)
    return "sha256:" + hashlib.sha256(canonical(payload)).hexdigest()


def verify_package_identity(value: dict[str, Any]) -> bool:
    identity = value.get("package_identity")
    return isinstance(identity, str) and identity == content_identity(value)


def git_source_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReplacementFreezeFault("git_source_state_unavailable") from exc
    return {
        "commit": commit,
        "branch": branch,
        "clean_at_generation": not porcelain,
        "dirty_path_count": len(porcelain),
        "dirty_paths": porcelain,
    }


def artifact_row(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def machine_authority_boundary(
    training_availability: dict[str, Any],
    d2_controller: dict[str, Any],
    *,
    emergency_yield_present: bool,
    active_d2_lease_present: bool,
) -> dict[str, Any]:
    faults: list[str] = []
    disk = training_availability.get("disk_reserve") or {}
    lineage = training_availability.get("lineage_custody") or {}
    segment = training_availability.get("segment_behavior") or {}
    if training_availability.get("policy") != (
        "project_theseus_resource_aware_training_segments_v2"
    ):
        faults.append("training_availability_policy_mismatch")
    if training_availability.get("enabled") is not True:
        faults.append("training_availability_disabled")
    if "launch_windows" in training_availability:
        faults.append("clock_launch_window_present")
    if "minimum_disk_free_gib" in training_availability:
        faults.append("arbitrary_disk_floor_present")
    if (
        disk.get("policy") != "two_complete_checkpoint_transactions_v1"
        or int(disk.get("complete_transactions_required") or 0) < 2
    ):
        faults.append("derived_disk_transaction_reserve_missing")
    if (
        lineage.get("policy")
        != "project_theseus_append_only_training_segment_lineage_v1"
        or any(
            lineage.get(key) is not True
            for key in (
                "archive_before_and_after_receipts",
                "archive_child_and_host_guard_receipts",
                "require_contiguous_identity_before_launch",
                "manifest_written_last",
            )
        )
    ):
        faults.append("append_only_lineage_boundary_missing")
    if any(
        segment.get(key) is not True
        for key in (
            "never_suspend_in_flight_metal_graph",
            "reevaluate_after_every_transactional_segment",
            "stop_launching_when_gate_closes",
            "atomic_checkpoint_before_yield",
        )
    ):
        faults.append("transactional_segment_boundary_missing")
    if emergency_yield_present:
        faults.append("emergency_yield_requested")

    authority = d2_controller.get("authority") or {}
    local_acquisition = d2_controller.get("local_rater_model_acquisition") or {}
    if d2_controller.get("policy") != (
        "project_theseus_neural_seed_d2_autonomous_one_shot_evaluation_v1"
    ):
        faults.append("d2_controller_policy_mismatch")
    required_true = (
        "require_clean_source",
        "require_exact_freeze_source_hashes",
        "require_all_checkpoints_complete",
        "require_no_competing_accelerator_job",
    )
    if authority.get("kind") != "machine_predicate_exclusive_one_shot_lease":
        faults.append("d2_machine_lease_kind_mismatch")
    if any(authority.get(key) is not True for key in required_true):
        faults.append("d2_required_machine_predicate_missing")
    required_false = (
        "user_or_operator_approval_required",
        "rerun_consumed_identity_allowed",
        "physical_boundary_is_negative_evidence",
        "project_selected_quality_token_cap_allowed",
        "public_calibration_authorized",
        "external_inference_authorized",
        "serving_authorized",
        "training_row_admission_authorized",
    )
    if any(authority.get(key) is not False for key in required_false):
        faults.append("d2_forbidden_authority_present")
    if (
        local_acquisition.get("automatic_only_when_every_other_gate_is_green")
        is not True
        or int(local_acquisition.get("external_inference_calls", -1)) != 0
        or int(local_acquisition.get("training_rows_written", -1)) != 0
    ):
        faults.append("d2_local_rater_acquisition_boundary_mismatch")
    if active_d2_lease_present:
        faults.append("active_d2_evaluation_lease_present")
    return {
        "passed": not faults,
        "faults": faults,
        "user_or_operator_approval_required": False,
        "emergency_yield_requested": emergency_yield_present,
        "active_d2_evaluation_lease_present": active_d2_lease_present,
        "training_segments_recheck_machine_gates": True,
        "d2_requires_exclusive_one_shot_machine_lease": True,
        "package_acquires_execution_lease": False,
    }


def selected_recipe(trainer: dict[str, Any]) -> dict[str, Any]:
    execution = trainer["training"]["execution_policy"]
    pretraining = execution["pretraining"]
    topology = trainer["topology"]
    return {
        "compute_dtype": execution["compute_dtype"],
        "fp32_master": execution["fp32_master"],
        "token_loss_compute_dtype": execution["token_loss_compute_dtype"],
        "training_step_mode": pretraining["training_step_mode"],
        "compiled_microbatch_size": int(pretraining["compiled_microbatch_size"]),
        "compile_width_quantum": int(pretraining["compile_width_quantum"]),
        "optimizer_id": trainer["training"]["optimizer_id"],
        "training_rope_kernel": trainer["training"]["training_rope_kernel"],
        "self_attention_projection": topology.get(
            "self_attention_projection", "separate"
        ),
        "feed_forward_activation": topology.get("feed_forward_activation", "swiglu"),
        "residual_policy": topology.get("residual_policy", "sequential_unscaled"),
        "per_head_muon": trainer["training"]["optimizer_id"] == "per_head_muon_mlx",
    }


def frozen_consumption_matches(
    path: Path,
    freeze: dict[str, Any],
    freeze_sha256: str,
    equivalent_case_contract_sha256s: list[str] | None = None,
) -> list[int]:
    needles = {
        freeze_sha256,
        str(freeze["candidate_id"]),
        str(freeze["candidate_packet_sha256"]),
        str(freeze["case_contract_sha256"]),
    }
    needles.update(str(value) for value in equivalent_case_contract_sha256s or [])
    matches: list[int] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        serialized = json.dumps(row, sort_keys=True)
        if any(needle in serialized for needle in needles):
            matches.append(line_number)
    return matches


def lineage_manifest(
    root: Path,
    *,
    expected_count: int,
    anchor_step: int,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    paths = sorted(root.glob("step-*_to_*/manifest.json"))
    faults: list[str] = []
    previous: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        manifest = read_json(path)
        before = manifest.get("before_identity") or {}
        after = manifest.get("after_identity") or {}
        if index == 0 and before.get("optimizer_steps") != anchor_step:
            faults.append("anchor_step_mismatch")
        if previous is not None and before != previous:
            faults.append(f"identity_discontinuity:{relative(path)}")
        artifacts = {}
        for name, entry in (manifest.get("artifacts") or {}).items():
            artifact_path = resolve(entry["path"])
            artifacts[name] = artifact_row(artifact_path)
            if artifacts[name]["sha256"] != entry["sha256"]:
                faults.append(f"artifact_hash_mismatch:{name}:{relative(path)}")
        rows.append(
            {
                "manifest": artifact_row(path),
                "before_identity": before,
                "after_identity": after,
                "artifacts": artifacts,
            }
        )
        previous = after
    if len(rows) != expected_count:
        faults.append(f"manifest_count_mismatch:{len(rows)}")
    expected_terminal = {
        "optimizer_steps": receipt["optimizer_steps"],
        "optimizer_positions": receipt["optimizer_positions"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "optimizer_state_sha256": receipt["optimizer_state_sha256"],
        "mlx_rng_state_sha256": receipt["mlx_rng_state_sha256"],
    }
    if previous is None:
        faults.append("lineage_empty")
    else:
        for key, value in expected_terminal.items():
            if previous.get(key) != value:
                faults.append(f"terminal_identity_mismatch:{key}")
    return {
        "passed": not faults,
        "faults": faults,
        "manifest_count": len(rows),
        "prospective_anchor_step": (
            rows[0]["before_identity"].get("optimizer_steps") if rows else None
        ),
        "terminal_step": (
            rows[-1]["after_identity"].get("optimizer_steps") if rows else None
        ),
        "pre_anchor_full_chain_available": False,
        "pre_anchor_boundary": (
            "The package binds the append-only prospective lineage from step "
            "9048. It makes no complete-chain claim for steps 3480 through 9048."
        ),
        "rows": rows,
    }


def execute(
    config_path: Path = DEFAULT_CONFIG,
    *,
    publish_report: bool = True,
    source_state_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("policy") != POLICY:
        raise ReplacementFreezeFault("config_policy_invalid")
    expected = config["expected"]
    source_state = (
        dict(source_state_override)
        if source_state_override is not None
        else git_source_state()
    )
    source_artifact_paths = {
        str(path): resolve(str(path)) for path in config["source_artifact_paths"]
    }

    trainer_path = resolve(config["production_trainer_config"])
    trainer = training.bind_scale_preregistration(training.read_json(trainer_path))
    plan = training.build_plan(trainer, config_path=trainer_path)
    target = plan["targets"][training.SHARED_TRUNK_ID]
    checkpoint_root = resolve(config["checkpoint_root"])
    receipt_path = checkpoint_root / "training_receipt.json"
    receipt = training.read_json(receipt_path)
    migration = training.validate_resume(
        receipt,
        plan,
        target,
        training.resolve(receipt["checkpoint"]),
        training.resolve(receipt["optimizer_state"]),
    )

    versioned_paths = {
        "model": training.resolve(receipt["checkpoint"]),
        "optimizer": training.resolve(receipt["optimizer_state"]),
        "mlx_rng": training.resolve(receipt["mlx_rng_state"]),
        "receipt": receipt_path,
    }
    versioned = {name: artifact_row(path) for name, path in versioned_paths.items()}
    receipt_hash_match = {
        "model": versioned["model"]["sha256"] == receipt["checkpoint_sha256"],
        "optimizer": (
            versioned["optimizer"]["sha256"] == receipt["optimizer_state_sha256"]
        ),
        "mlx_rng": (versioned["mlx_rng"]["sha256"] == receipt["mlx_rng_state_sha256"]),
    }
    alias_paths = {
        "model": checkpoint_root / "weights.safetensors",
        "optimizer": checkpoint_root / "optimizer.safetensors",
        "mlx_rng": checkpoint_root / "optimizer.mlx-rng.safetensors",
    }
    aliases = {name: artifact_row(path) for name, path in alias_paths.items()}
    alias_match = {
        name: aliases[name]["sha256"] == versioned[name]["sha256"]
        for name in alias_paths
    }
    lineage = lineage_manifest(
        resolve(config["lineage_root"]),
        expected_count=int(expected["lineage_manifest_count"]),
        anchor_step=int(expected["prospective_anchor_step"]),
        receipt=receipt,
    )

    residual_path = resolve(config["acceleration_residual_audit"])
    residual = read_json(residual_path)
    independent_path = resolve(config["independent_readiness_audit"])
    independent = read_json(independent_path)
    candidate_contract_path = resolve(config["candidate_contract"])
    candidate_contract = read_json(candidate_contract_path)
    review_path = resolve(config["pre_long_run_review"])
    review = read_json(review_path)
    bakeoff_path = resolve(config["factorized_bakeoff"])
    bakeoff = read_json(bakeoff_path)
    candidate_reports = {
        name: (resolve(path), read_json(resolve(path)))
        for name, path in config["candidate_reports"].items()
    }
    freeze_path = resolve(config["functional_freeze"])
    functional_freeze = read_json(freeze_path)
    registry_path = resolve(config["functional_consumption_registry"])
    surface_integrity = config["functional_surface_integrity"]
    functional_freeze_sha256 = sha256_file(freeze_path)
    consumption_matches = frozen_consumption_matches(
        registry_path,
        functional_freeze,
        functional_freeze_sha256,
        list(surface_integrity["equivalent_consumed_case_contract_sha256s"]),
    )
    machine_paths = {
        name: resolve(path) for name, path in config["machine_authority"].items()
    }
    training_availability = read_json(
        machine_paths["training_availability_config"]
    )
    d2_controller = read_json(machine_paths["d2_controller_config"])
    emergency_yield_path = resolve(
        str(training_availability["yield_after_segment_control"])
    )
    active_d2_lease_path = resolve(str(d2_controller["active_lease"]))
    machine_boundary = machine_authority_boundary(
        training_availability,
        d2_controller,
        emergency_yield_present=emergency_yield_path.is_file(),
        active_d2_lease_present=active_d2_lease_path.is_file(),
    )
    superseded_path = resolve(config["supersedes"])
    recipe = selected_recipe(trainer)

    residual_authority_current = all(
        resolve(row["path"]).is_file()
        and sha256_file(resolve(row["path"])) == row["sha256"]
        for section in ("evidence", "authority")
        for row in (residual.get(section) or {}).values()
    )
    independent_authority_current = all(
        resolve(row["path"]).is_file()
        and sha256_file(resolve(row["path"])) == row["sha256"]
        for row in (independent.get("authority") or {}).values()
    )
    per_head = candidate_reports["per_head_muon"][1]
    attnres = candidate_reports["attention_residuals"][1]
    situ = candidate_reports["situ_glu"][1]
    qkv = candidate_reports["fused_qkv"][1]
    global_safety = candidate_contract["host_safety_policy"]
    inline_reserves_zero = all(
        float((row.get("host_safety_overrides") or {}).get(key, 0)) == 0.0
        for row in candidate_contract["canaries"]
        for key in (
            "minimum_available_before_launch_mib",
            "minimum_available_during_run_mib",
        )
    )

    gates = {
        "authoritative_audits": {
            "passed": (
                residual.get("trigger_state") == "GREEN"
                and residual.get("failed_domains") == []
                and residual.get("long_training_started_or_resumed") is False
                and residual_authority_current
                and independent.get("trigger_state") == "GREEN"
                and independent.get("failed_audits") == []
                and independent.get("long_training_authorized") is False
                and independent_authority_current
            )
        },
        "semantic_plan_resume": {
            "passed": (
                plan.get("trigger_state") == "GREEN"
                and plan.get("hard_gaps") == []
                and receipt.get("plan_sha256") == expected["receipt_plan_sha256"]
                and plan.get("plan_sha256") == expected["current_plan_sha256"]
                and migration is not None
                and migration.get("migration_id") == expected["migration_id"]
                and migration.get("reset_data_cursor_phase") is None
                and migration.get("reset_data_cursor_seed") is None
            ),
            "receipt_plan_sha256": receipt.get("plan_sha256"),
            "current_plan_sha256": plan.get("plan_sha256"),
            "migration": migration,
        },
        "checkpoint_custody": {
            "passed": (
                receipt.get("optimizer_steps") == expected["optimizer_steps"]
                and receipt.get("optimizer_positions")
                == expected["optimizer_positions"]
                and receipt.get("capability_claim") == expected["capability_claim"]
                and all(receipt_hash_match.values())
                and all(alias_match.values())
            ),
            "receipt_hash_match": receipt_hash_match,
            "alias_match": alias_match,
        },
        "prospective_lineage": {"passed": lineage["passed"] is True},
        "functional_surface": {
            "passed": (
                functional_freeze.get("candidate_id") == config["campaign_id"]
                and functional_freeze.get("immutable") is True
                and functional_freeze.get("source_disjoint") is True
                and functional_freeze.get("case_count")
                == expected["functional_case_count"]
                and functional_freeze.get("consumed_case_count")
                == expected["functional_consumed_case_count"]
                and functional_freeze.get("evaluation_state") == "NOT_EVALUATED"
                and functional_freeze.get("capability_claim") == "NOT_EVALUATED"
                and functional_freeze.get("public_training_rows_written") == 0
                and functional_freeze.get("external_inference_calls") == 0
                and not consumption_matches
            ),
            "matching_consumption_registry_lines": consumption_matches,
        },
        "functional_surface_freshness": {
            "passed": (
                (
                    independent.get("audits", {})
                    .get("evaluation_surface_freshness", {})
                    .get("passed")
                    is True
                )
                and (
                    independent.get("audits", {})
                    .get("evaluation_surface_freshness", {})
                    .get("independent_recomputation", {})
                    .get("passed")
                    is True
                )
                and surface_integrity.get("state") == "VALID_FRESH_PRIVATE_SURFACE"
                and surface_integrity.get("fresh_surface") is True
                and surface_integrity.get("evaluation_authorized") is False
                and surface_integrity.get("current_case_contract_sha256")
                == functional_freeze.get("case_contract_sha256")
                and not surface_integrity.get(
                    "equivalent_consumed_case_contract_sha256s"
                )
                and not consumption_matches
            ),
            "state": surface_integrity.get("state"),
            "fresh_surface": surface_integrity.get("fresh_surface"),
            "evaluation_authorized": surface_integrity.get("evaluation_authorized"),
            "current_case_contract_sha256": functional_freeze.get(
                "case_contract_sha256"
            ),
            "equivalent_consumed_case_contract_sha256s": surface_integrity.get(
                "equivalent_consumed_case_contract_sha256s"
            ),
            "equivalence_basis": surface_integrity.get("equivalence_basis"),
            "disposition": surface_integrity.get("disposition"),
            "matching_consumption_registry_lines": consumption_matches,
            "independent_recomputation": (
                independent.get("audits", {})
                .get("evaluation_surface_freshness", {})
                .get("independent_recomputation")
            ),
        },
        "architecture_selection": {
            "passed": (
                bakeoff.get("trigger_state") == "GREEN"
                and bakeoff.get("disposition")
                == "factorized_architecture_selected_training_not_started"
                and len(bakeoff.get("selected_implementation_ids") or {}) == 7
                and recipe == expected["selected_recipe"]
            )
        },
        "candidate_dispositions": {
            "passed": (
                per_head.get("trigger_state") == "GREEN"
                and per_head["campaign_disposition"]["selected_optimizer"]
                == "adamw_mlx"
                and per_head["campaign_disposition"]["scientific_falsification_claimed"]
                is False
                and attnres.get("trigger_state") == "GREEN"
                and attnres["campaign_disposition"]["selected_architecture"]
                == "control"
                and attnres["campaign_disposition"]["scientific_falsification_claimed"]
                is False
                and situ.get("trigger_state") == "GREEN"
                and situ["campaign_disposition"]["selected_architecture"] == "control"
                and situ["campaign_disposition"]["scientific_falsification_claimed"]
                is False
                and qkv["selection"]["candidate_selected"] is False
                and qkv["selection"]["production_route_changed"] is False
                and qkv["production_authority"]["live_checkpoint_mutated"] is False
            )
        },
        "resource_policy": {
            "passed": (
                trainer["host_resource_safety"]["memory_guard_mode"]
                == "predicted_exhaustion"
                and float(
                    trainer["host_resource_safety"][
                        "minimum_available_before_launch_mib"
                    ]
                )
                == 0.0
                and float(
                    trainer["host_resource_safety"]["minimum_available_during_run_mib"]
                )
                == 0.0
                and trainer["host_resource_safety"]["swapout_growth_action"]
                == "report_only"
                and float(trainer["host_resource_safety"]["maximum_wall_seconds"])
                == 0.0
                and global_safety["memory_guard_mode"] == "predicted_exhaustion"
                and float(global_safety["minimum_available_before_launch_mib"]) == 0.0
                and float(global_safety["minimum_available_during_run_mib"]) == 0.0
                and global_safety["swapout_growth_action"] == "report_only"
                and inline_reserves_zero
            ),
            "fixed_available_memory_floor_mib": 0,
            "production_wall_deadline_seconds": 0,
        },
        "source_binding": {
            "passed": (
                bool(source_state.get("commit"))
                and source_state.get("clean_at_generation") is True
                and int(source_state.get("dirty_path_count", -1)) == 0
                and source_state.get("dirty_paths") == []
            ),
            "source_state": source_state,
            "final_generation_rule": (
                "The source-bound replacement package may be finalized exactly "
                "once only from a clean post-maintenance commit."
            ),
        },
        "machine_authority_boundary": {
            **machine_boundary,
            "passed": (
                machine_boundary["passed"]
                and review.get("state") == "HOLD_FOR_FINITE_REVIEW"
                and review["execution_hold"]["new_long_segment_authorized"] is False
                and review["execution_hold"][
                    "new_architecture_may_touch_live_checkpoint"
                ]
                is False
                and review["execution_hold"]["in_flight_transaction_interrupted"]
                is False
                and config["boundaries"]["long_training_started_or_resumed"] is False
                and config["boundaries"][
                    "machine_authority_bypassed_by_freeze"
                ]
                is False
            ),
            "historical_review_hold_preserved": True,
            "machine_authority_bypassed": False,
        },
    }
    required = list(config["required_gates"])
    missing = sorted(set(required) - set(gates))
    failed = [name for name in required if not gates.get(name, {}).get("passed")]
    trigger_state = "GREEN" if not missing and not failed else "RED"

    authority_paths = {
        "config": config_path,
        "production_trainer_config": trainer_path,
        "candidate_contract": candidate_contract_path,
        "pre_long_run_review": review_path,
        "factorized_bakeoff": bakeoff_path,
        "acceleration_residual_audit": residual_path,
        "independent_readiness_audit": independent_path,
        "functional_freeze": freeze_path,
        "functional_consumption_registry": registry_path,
        **machine_paths,
    }
    report = {
        "policy": POLICY,
        "schema_version": "1.2.0",
        "campaign_id": config["campaign_id"],
        "trigger_state": trigger_state,
        "support_state": (
            "SUPPORTED_CONTENT_ADDRESSED_REPLACEMENT_FREEZE"
            if trigger_state == "GREEN"
            else "UNSUPPORTED_FREEZE_GATE_FAULT"
        ),
        "required_gates": required,
        "missing_gates": missing,
        "failed_gates": failed,
        "gates": gates,
        "decision": (
            config["decision"]
            if trigger_state == "GREEN"
            else "BLOCK_RESUME_AND_REQUIRE_REPAIR_OR_NEW_LINEAGE"
        ),
        "resume_authority": {
            "authorized_state": (
                "EXACT_STEP_11416_UNCHANGED_THROUGH_MACHINE_PREDICATE_LEASE"
                if trigger_state == "GREEN"
                else "NONE"
            ),
            "machine_authority_boundary_present": machine_boundary["passed"],
            "machine_authority_bypassed_by_package": False,
            "user_or_operator_approval_required": False,
            "active_d2_evaluation_lease_present": active_d2_lease_path.is_file(),
            "long_training_authorized_now": False,
            "long_training_started_or_resumed": False,
            "incompatible_state_policy": (
                "Any model topology, tensor, optimizer, RNG, cursor, data, "
                "objective, schedule, or evaluation mismatch requires a new "
                "content-addressed lineage; no state reinterpretation."
            ),
        },
        "selected_architecture": {
            "implementation_ids": bakeoff["selected_implementation_ids"],
            "execution_recipe": recipe,
        },
        "checkpoint_custody": {
            "optimizer_steps": receipt["optimizer_steps"],
            "optimizer_positions": receipt["optimizer_positions"],
            "capability_claim": receipt["capability_claim"],
            "versioned": versioned,
            "aliases": aliases,
            "lineage": lineage,
        },
        "functional_surface": {
            "freeze": artifact_row(freeze_path),
            "case_count": functional_freeze["case_count"],
            "consumed_case_count": functional_freeze["consumed_case_count"],
            "evaluation_state": functional_freeze["evaluation_state"],
            "capability_claim": functional_freeze["capability_claim"],
            "source_disjoint": functional_freeze["source_disjoint"],
            "registry": artifact_row(registry_path),
            "matching_registry_lines": consumption_matches,
            "integrity": surface_integrity,
        },
        "candidate_dispositions": {
            "per_head_muon": "NOT_SELECTED_FIRST_CAMPAIGN",
            "attention_residuals": "NOT_SELECTED_FIRST_CAMPAIGN",
            "situ_glu": "NOT_SELECTED_FIRST_CAMPAIGN",
            "fused_qkv": qkv["selection"]["disposition"],
            "scientific_falsification_claimed": False,
        },
        "source_binding": source_state,
        "source_artifacts": {
            name: artifact_row(path) for name, path in source_artifact_paths.items()
        },
        "authority_manifest": {
            name: artifact_row(path) for name, path in authority_paths.items()
        },
        "candidate_report_manifest": {
            name: artifact_row(path)
            for name, (path, _report) in candidate_reports.items()
        },
        "supersession": {
            "superseded_package": artifact_row(superseded_path),
            "reason": (
                "The historical package binds step 3480. This replacement "
                "binds the exact held step-11416 state, prospective lineage, "
                "completed finite candidates, current acceleration selection, "
                "independent audits, functional-surface integrity, and the "
                "source commit. Final publication is valid only when the "
                "source-binding gate records a clean post-maintenance tree."
            ),
            "historical_package_remains_valid_for_its_exact_past_transaction": True,
            "historical_package_does_not_authorize_current_resume": True,
        },
        "boundaries": config["boundaries"],
        "capability_claim": "NONE_SELECTION_CUSTODY_AND_FREEZE_ONLY",
        "non_claims": [
            "This package does not prove physical performance optimality.",
            "This package does not evaluate model capability or utility.",
            "Scoped candidate dispositions are not broad scientific falsifications.",
            "This package does not acquire a training or D2 execution lease or start long training.",
        ],
    }
    report["package_identity"] = content_identity(report)
    if not verify_package_identity(report):
        raise ReplacementFreezeFault("package_identity_internal_fault")
    if publish_report:
        output_path = resolve(config["report"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    report = execute(resolve(args.config), publish_report=not args.no_publish)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["trigger_state"] == "GREEN" else 1)


if __name__ == "__main__":
    main()
