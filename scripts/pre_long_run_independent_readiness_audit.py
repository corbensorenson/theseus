#!/usr/bin/env python3
"""Independently audit the finite pre-long-run selection and custody claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from neural_seed_functional_cases import materialize_cases


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/pre_long_run_independent_readiness_audit.json"
POLICY = "project_theseus_pre_long_run_independent_readiness_audit_v1"


class IndependentAuditFault(ValueError):
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
        raise IndependentAuditFault(f"json_object_required:{relative(path)}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recompute_machine_authority_boundary(
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
    acquisition = d2_controller.get("local_rater_model_acquisition") or {}
    if d2_controller.get("policy") != (
        "project_theseus_neural_seed_d2_autonomous_one_shot_evaluation_v1"
    ):
        faults.append("d2_controller_policy_mismatch")
    if authority.get("kind") != "machine_predicate_exclusive_one_shot_lease":
        faults.append("d2_machine_lease_kind_mismatch")
    if any(
        authority.get(key) is not True
        for key in (
            "require_clean_source",
            "require_exact_freeze_source_hashes",
            "require_all_checkpoints_complete",
            "require_no_competing_accelerator_job",
        )
    ):
        faults.append("d2_required_machine_predicate_missing")
    if any(
        authority.get(key) is not False
        for key in (
            "user_or_operator_approval_required",
            "rerun_consumed_identity_allowed",
            "physical_boundary_is_negative_evidence",
            "project_selected_quality_token_cap_allowed",
            "public_calibration_authorized",
            "external_inference_authorized",
            "serving_authorized",
            "training_row_admission_authorized",
        )
    ):
        faults.append("d2_forbidden_authority_present")
    if (
        acquisition.get("automatic_only_when_every_other_gate_is_green")
        is not True
        or int(acquisition.get("external_inference_calls", -1)) != 0
        or int(acquisition.get("training_rows_written", -1)) != 0
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
        "audit_acquires_execution_lease": False,
    }


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def normalize_prompt(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def registry_contract_lines(path: Path, contract_sha256: str) -> list[int]:
    matches: list[int] = []
    if not path.is_file():
        return matches
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        identity = row.get("identity") or {}
        if identity.get("case_contract_sha256") == contract_sha256:
            matches.append(line_number)
    return matches


def recompute_surface_integrity(
    integrity: dict[str, Any],
    freeze: dict[str, Any],
    registry_path: Path,
) -> dict[str, Any]:
    faults: list[str] = []
    functional_config_path = resolve(integrity["current_contract_config"])
    functional_config = read_json(functional_config_path)
    cases = materialize_cases(functional_config)
    contract_rows = [
        {key: value for key, value in case.items() if key != "model_visible"}
        for case in cases
    ]
    visible_rows = [case["model_visible"] for case in cases]
    current_contract_sha256 = stable_hash(contract_rows)
    current_visible_sha256 = stable_hash(visible_rows)
    current_prompts = {
        str(row.get("prompt") or ""): str(row.get("case_id") or "")
        for row in visible_rows
    }
    current_normalized = {
        normalize_prompt(prompt): case_id for prompt, case_id in current_prompts.items()
    }
    if current_contract_sha256 != freeze.get("case_contract_sha256"):
        faults.append("current_contract_recomputation_mismatch")
    if len(cases) != freeze.get("case_count"):
        faults.append("current_case_count_mismatch")
    if integrity.get("current_case_contract_sha256") != current_contract_sha256:
        faults.append("declared_current_contract_mismatch")

    historical_packets: list[dict[str, Any]] = []
    exact_overlaps: list[dict[str, str]] = []
    normalized_overlaps: list[dict[str, str]] = []
    historical_contracts: list[str] = []
    for row in integrity.get("historical_candidate_packets") or []:
        packet_path = resolve(str(row.get("path") or ""))
        if not packet_path.is_file():
            faults.append(f"historical_packet_missing:{relative(packet_path)}")
            continue
        observed_sha256 = sha256_file(packet_path)
        expected_sha256 = str(row.get("sha256") or "")
        if observed_sha256 != expected_sha256:
            faults.append(f"historical_packet_hash_mismatch:{relative(packet_path)}")
            continue
        packet = read_json(packet_path)
        contract_sha256 = str(packet.get("contract_sha256") or "")
        historical_contracts.append(contract_sha256)
        consumed_lines = registry_contract_lines(registry_path, contract_sha256)
        if not consumed_lines:
            faults.append(f"historical_contract_not_consumed:{relative(packet_path)}")
        prior_rows = packet.get("rows") or []
        prior_visible_sha256 = stable_hash(prior_rows)
        for prior in prior_rows:
            prior_prompt = str(prior.get("prompt") or "")
            prior_case_id = str(prior.get("case_id") or "")
            current_case_id = current_prompts.get(prior_prompt)
            if current_case_id is not None:
                exact_overlaps.append(
                    {
                        "current_case_id": current_case_id,
                        "historical_case_id": prior_case_id,
                    }
                )
            normalized = normalize_prompt(prior_prompt)
            current_case_id = current_normalized.get(normalized)
            if current_case_id is not None:
                normalized_overlaps.append(
                    {
                        "current_case_id": current_case_id,
                        "historical_case_id": prior_case_id,
                    }
                )
        historical_packets.append(
            {
                "path": relative(packet_path),
                "sha256": observed_sha256,
                "contract_sha256": contract_sha256,
                "row_count": len(prior_rows),
                "model_visible_sha256": prior_visible_sha256,
                "consumption_registry_lines": consumed_lines,
            }
        )

    current_consumption_lines = registry_contract_lines(
        registry_path, current_contract_sha256
    )
    if current_consumption_lines:
        faults.append("current_contract_already_consumed")
    if current_contract_sha256 in historical_contracts:
        faults.append("current_contract_equals_historical_contract")
    if exact_overlaps:
        faults.append("exact_historical_prompt_overlap")
    if normalized_overlaps:
        faults.append("normalized_historical_prompt_overlap")
    if not historical_packets:
        faults.append("historical_packet_evidence_missing")

    return {
        "passed": not faults,
        "state": (
            "VALID_FRESH_PRIVATE_SURFACE"
            if not faults
            else "INVALID_OR_UNPROVEN_PRIVATE_SURFACE"
        ),
        "freshness_scope": (
            "Exact and whitespace/case-normalized model-visible prompts against "
            "content-addressed, consumed historical candidate packets. Reuse of "
            "task families is not treated as reuse of an exact measurement surface."
        ),
        "faults": faults,
        "current": {
            "config": relative(functional_config_path),
            "config_sha256": sha256_file(functional_config_path),
            "case_count": len(cases),
            "contract_sha256": current_contract_sha256,
            "model_visible_sha256": current_visible_sha256,
            "consumption_registry_lines": current_consumption_lines,
        },
        "historical_packets": historical_packets,
        "exact_prompt_overlaps": exact_overlaps,
        "normalized_prompt_overlaps": normalized_overlaps,
    }


def production_recipe(trainer: dict[str, Any]) -> dict[str, Any]:
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


def recompute_lineage(
    root: Path,
    *,
    expected_count: int,
    anchor_step: int,
    terminal: dict[str, Any],
) -> dict[str, Any]:
    manifests = sorted(root.glob("step-*_to_*/manifest.json"))
    faults: list[str] = []
    previous: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(manifests):
        row = read_json(path)
        before = row.get("before_identity") or {}
        after = row.get("after_identity") or {}
        if index == 0 and before.get("optimizer_steps") != anchor_step:
            faults.append("anchor_step_mismatch")
        if previous is not None and before != previous:
            faults.append(f"identity_discontinuity:{relative(path)}")
        for artifact in (row.get("artifacts") or {}).values():
            artifact_path = resolve(artifact["path"])
            if not artifact_path.is_file():
                faults.append(f"artifact_missing:{relative(artifact_path)}")
            elif sha256_file(artifact_path) != artifact["sha256"]:
                faults.append(f"artifact_hash_mismatch:{relative(artifact_path)}")
        rows.append(
            {
                "path": relative(path),
                "sha256": sha256_file(path),
                "before": before,
                "after": after,
            }
        )
        previous = after
    if len(rows) != expected_count:
        faults.append(f"manifest_count_mismatch:{len(rows)}")
    terminal_keys = (
        "optimizer_steps",
        "optimizer_positions",
        "checkpoint_sha256",
        "optimizer_state_sha256",
        "mlx_rng_state_sha256",
    )
    if previous is None:
        faults.append("lineage_empty")
    else:
        for key in terminal_keys:
            if previous.get(key) != terminal.get(key):
                faults.append(f"terminal_identity_mismatch:{key}")
    return {
        "passed": not faults,
        "faults": faults,
        "manifest_count": len(rows),
        "anchor_step": rows[0]["before"].get("optimizer_steps") if rows else None,
        "terminal_step": rows[-1]["after"].get("optimizer_steps") if rows else None,
        "pre_anchor_full_chain_available": False,
        "rows": rows,
    }


def matching_consumption_rows(
    registry_path: Path,
    freeze: dict[str, Any],
    freeze_sha256: str,
    equivalent_case_contract_sha256s: list[str] | None = None,
) -> list[dict[str, Any]]:
    needles = {
        freeze_sha256,
        str(freeze["candidate_id"]),
        str(freeze["candidate_packet_sha256"]),
        str(freeze["case_contract_sha256"]),
    }
    needles.update(str(value) for value in equivalent_case_contract_sha256s or [])
    matches: list[dict[str, Any]] = []
    if not registry_path.is_file():
        return matches
    for line_number, line in enumerate(
        registry_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        serialized = json.dumps(row, sort_keys=True)
        if any(needle in serialized for needle in needles):
            matches.append({"line": line_number, "row": row})
    return matches


def execute(
    config_path: Path = DEFAULT_CONFIG,
    *,
    publish_report: bool = True,
) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("policy") != POLICY:
        raise IndependentAuditFault("config_policy_invalid")

    residual_path = resolve(config["residual_audit"])
    residual = read_json(residual_path)
    residual_config_path = resolve(config["residual_audit_config"])
    trainer_path = resolve(config["production_trainer_config"])
    trainer = read_json(trainer_path)
    candidate_contract_path = resolve(config["candidate_contract"])
    candidate_contract = read_json(candidate_contract_path)
    review_path = resolve(config["pre_long_run_review"])
    review = read_json(review_path)
    freeze_path = resolve(config["functional_freeze"])
    freeze = read_json(freeze_path)
    registry_path = resolve(config["functional_consumption_registry"])
    surface_integrity = config["functional_surface_integrity"]
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
    machine_boundary = recompute_machine_authority_boundary(
        training_availability,
        d2_controller,
        emergency_yield_present=emergency_yield_path.is_file(),
        active_d2_lease_present=active_d2_lease_path.is_file(),
    )
    candidate_reports = {
        name: (resolve(path), read_json(resolve(path)))
        for name, path in config["candidate_reports"].items()
    }
    replay_reports = {
        name: (resolve(path), read_json(resolve(path)))
        for name, path in config["replay_reports"].items()
    }
    expected = config["expected"]

    residual_hash_faults: list[str] = []
    if residual.get("config_sha256") != sha256_file(residual_config_path):
        residual_hash_faults.append("residual_config_hash_mismatch")
    for section in ("evidence", "authority"):
        for name, row in (residual.get(section) or {}).items():
            path = resolve(row["path"])
            if not path.is_file():
                residual_hash_faults.append(
                    f"{section}_missing:{name}:{relative(path)}"
                )
            elif sha256_file(path) != row["sha256"]:
                residual_hash_faults.append(
                    f"{section}_hash_mismatch:{name}:{relative(path)}"
                )

    checkpoint_root = resolve(config["checkpoint_root"])
    receipt_path = checkpoint_root / "training_receipt.json"
    receipt = read_json(receipt_path)
    versioned = {
        "model": resolve(receipt["checkpoint"]),
        "optimizer": resolve(receipt["optimizer_state"]),
        "mlx_rng": resolve(receipt["mlx_rng_state"]),
    }
    expected_hashes = {
        "model": receipt["checkpoint_sha256"],
        "optimizer": receipt["optimizer_state_sha256"],
        "mlx_rng": receipt["mlx_rng_state_sha256"],
    }
    versioned_hashes = {name: sha256_file(path) for name, path in versioned.items()}
    aliases = {
        "model": checkpoint_root / "weights.safetensors",
        "optimizer": checkpoint_root / "optimizer.safetensors",
        "mlx_rng": checkpoint_root / "optimizer.mlx-rng.safetensors",
    }
    alias_hashes = {name: sha256_file(path) for name, path in aliases.items()}
    checkpoint_hashes_pass = all(
        versioned_hashes[name] == expected_hashes[name]
        and alias_hashes[name] == versioned_hashes[name]
        for name in versioned
    )
    terminal = {
        "optimizer_steps": receipt.get("optimizer_steps"),
        "optimizer_positions": receipt.get("optimizer_positions"),
        "checkpoint_sha256": expected_hashes["model"],
        "optimizer_state_sha256": expected_hashes["optimizer"],
        "mlx_rng_state_sha256": expected_hashes["mlx_rng"],
    }
    lineage = recompute_lineage(
        resolve(config["lineage_root"]),
        expected_count=int(expected["lineage_manifest_count"]),
        anchor_step=int(expected["prospective_anchor_step"]),
        terminal=terminal,
    )

    fresh = replay_reports["fresh_process"][1]
    sustained = replay_reports["sustained_route"][1]
    global_safety = candidate_contract["host_safety_policy"]
    inline_reserves_zero = all(
        float((row.get("host_safety_overrides") or {}).get(key, 0)) == 0.0
        for row in candidate_contract["canaries"]
        for key in (
            "minimum_available_before_launch_mib",
            "minimum_available_during_run_mib",
        )
    )
    transaction_bytes = (
        sum(path.stat().st_size for path in versioned.values())
        + receipt_path.stat().st_size
    )
    disk_free_bytes = shutil.disk_usage(checkpoint_root).free

    freeze_sha256 = sha256_file(freeze_path)
    surface_recomputation = recompute_surface_integrity(
        surface_integrity,
        freeze,
        registry_path,
    )
    consumption_matches = matching_consumption_rows(
        registry_path,
        freeze,
        freeze_sha256,
        list(surface_integrity["equivalent_consumed_case_contract_sha256s"]),
    )
    per_head = candidate_reports["per_head_muon"][1]
    attnres = candidate_reports["attention_residuals"][1]
    situ = candidate_reports["situ_glu"][1]
    qkv = candidate_reports["fused_qkv"][1]
    scoped_candidates = (per_head, attnres, situ)
    selected_recipe = production_recipe(trainer)

    audits = {
        "evidence_integrity": {
            "passed": (
                not residual_hash_faults
                and residual.get("trigger_state") == "GREEN"
                and residual.get("failed_domains") == []
                and residual.get("long_training_started_or_resumed") is False
            ),
            "faults": residual_hash_faults,
            "independence": (
                "Every path and digest asserted by the residual audit was "
                "recomputed from disk; its GREEN label alone was insufficient."
            ),
        },
        "lineage_custody": {
            "passed": (
                checkpoint_hashes_pass
                and lineage["passed"] is True
                and receipt.get("optimizer_steps") == expected["optimizer_steps"]
                and receipt.get("optimizer_positions")
                == expected["optimizer_positions"]
                and receipt.get("capability_claim") == expected["capability_claim"]
            ),
            "checkpoint_hashes_pass": checkpoint_hashes_pass,
            "versioned_hashes": versioned_hashes,
            "alias_hashes": alias_hashes,
            "lineage": lineage,
        },
        "replay_integrity": {
            "passed": (
                fresh.get("trigger_state") == "GREEN"
                and fresh.get("canonical_lineage_unchanged") is True
                and fresh.get("exact_resume_validation") is True
                and fresh.get("independent_segmented_replay_numeric_parity") is True
                and fresh.get("host_resource_guard_passed") is True
                and sustained.get("trigger_state") == "GREEN"
                and sustained.get("canonical_lineage_unchanged") is True
                and sustained.get("exact_resume_each_segment") is True
                and sustained.get("hard_gaps") == []
            ),
            "fresh_process_report_sha256": sha256_file(
                replay_reports["fresh_process"][0]
            ),
            "sustained_report_sha256": sha256_file(
                replay_reports["sustained_route"][0]
            ),
        },
        "resource_integrity": {
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
                and disk_free_bytes >= 2 * transaction_bytes
                and sustained["thermal_stability"]["terminal"] is True
                and sustained["thermal_stability"]["thermal_warning_observed"] is False
                and sustained.get("elapsed_time_requirement") is None
                and sustained.get("arbitrary_percentage_tolerance") is None
            ),
            "disk_free_bytes": disk_free_bytes,
            "checkpoint_transaction_bytes": transaction_bytes,
            "two_transaction_required_bytes": 2 * transaction_bytes,
            "fixed_available_memory_floor_mib": 0,
        },
        "evaluation_nonconsumption": {
            "passed": (
                freeze.get("candidate_id") == "moecot_mlx_57m_active_preregistered_v1"
                and freeze.get("immutable") is True
                and freeze.get("source_disjoint") is True
                and freeze.get("consumed_case_count") == 0
                and freeze.get("evaluation_state") == "NOT_EVALUATED"
                and freeze.get("capability_claim") == "NOT_EVALUATED"
                and freeze.get("public_training_rows_written") == 0
                and freeze.get("external_inference_calls") == 0
                and not consumption_matches
                and receipt.get("public_training_rows_written") == 0
                and receipt.get("external_inference_calls") == 0
            ),
            "freeze_path": relative(freeze_path),
            "freeze_sha256": freeze_sha256,
            "registry_path": relative(registry_path),
            "registry_sha256": sha256_file(registry_path),
            "matching_consumption_rows": consumption_matches,
        },
        "evaluation_surface_freshness": {
            "passed": (
                surface_recomputation["passed"] is True
                and surface_integrity.get("state") == "VALID_FRESH_PRIVATE_SURFACE"
                and surface_integrity.get("fresh_surface") is True
                and surface_integrity.get("evaluation_authorized") is False
                and surface_integrity.get("current_case_contract_sha256")
                == freeze.get("case_contract_sha256")
                and not surface_integrity.get(
                    "equivalent_consumed_case_contract_sha256s"
                )
                and not consumption_matches
            ),
            "state": surface_integrity.get("state"),
            "fresh_surface": surface_integrity.get("fresh_surface"),
            "evaluation_authorized": surface_integrity.get("evaluation_authorized"),
            "current_case_contract_sha256": freeze.get("case_contract_sha256"),
            "equivalent_consumed_case_contract_sha256s": surface_integrity.get(
                "equivalent_consumed_case_contract_sha256s"
            ),
            "equivalence_basis": surface_integrity.get("equivalence_basis"),
            "disposition": surface_integrity.get("disposition"),
            "matching_consumption_rows": consumption_matches,
            "independent_recomputation": surface_recomputation,
        },
        "negative_evidence_scope": {
            "passed": (
                all(
                    report["gates"]["source_disjoint"] is True
                    and report["gates"]["no_public_external_or_fallback"] is True
                    and report["campaign_disposition"][
                        "scientific_falsification_claimed"
                    ]
                    is False
                    for report in scoped_candidates
                )
                and per_head["campaign_disposition"]["selected_optimizer"]
                == "adamw_mlx"
                and attnres["campaign_disposition"]["selected_architecture"]
                == "control"
                and situ["campaign_disposition"]["selected_architecture"] == "control"
                and qkv["selection"]["candidate_selected"] is False
                and qkv["selection"]["production_route_changed"] is False
                and qkv["production_authority"]["live_checkpoint_mutated"] is False
            ),
            "candidate_report_sha256s": {
                name: sha256_file(path)
                for name, (path, _report) in candidate_reports.items()
            },
            "scientific_falsification_claimed": False,
        },
        "claim_boundary": {
            "passed": (
                residual.get("selection_claim")
                == (
                    "No currently source- or measured-profile-justified open "
                    "acceleration residual remains among the tested eligible routes."
                )
                and residual.get("capability_claim")
                == "NONE_ACCELERATION_READINESS_AND_CUSTODY_ONLY"
                and any(
                    "does not prove" in row.lower() and "fastest" in row.lower()
                    for row in residual.get("non_claims", [])
                )
                and selected_recipe == expected["selected_recipe"]
                and receipt.get("capability_claim") == "NOT_EVALUATED"
            ),
            "authorized_claim": residual.get("selection_claim"),
            "forbidden_claims": [
                "fastest physically possible",
                "capability or utility",
                "architecture superiority",
                "broad scientific falsification",
                "long-campaign outcome",
            ],
            "selected_recipe": selected_recipe,
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
            ),
            "historical_review_hold_preserved": True,
            "machine_authority_bypassed": False,
            "long_training_started_or_resumed": False,
        },
    }
    required = list(config["required_audits"])
    missing = sorted(set(required) - set(audits))
    failed = [name for name in required if not audits.get(name, {}).get("passed")]
    trigger_state = "GREEN" if not missing and not failed else "RED"
    report = {
        "policy": POLICY,
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "trigger_state": trigger_state,
        "support_state": (
            "SUPPORTED_INDEPENDENT_PRE_LONG_RUN_READINESS_AUDIT"
            if trigger_state == "GREEN"
            else "UNSUPPORTED_INTEGRITY_OR_CLAIM_BOUNDARY_FAULT"
        ),
        "required_audits": required,
        "missing_audits": missing,
        "failed_audits": failed,
        "audits": audits,
        "authority": {
            "residual_audit": {
                "path": relative(residual_path),
                "sha256": sha256_file(residual_path),
            },
            "production_trainer_config": {
                "path": relative(trainer_path),
                "sha256": sha256_file(trainer_path),
            },
            "candidate_contract": {
                "path": relative(candidate_contract_path),
                "sha256": sha256_file(candidate_contract_path),
            },
            "pre_long_run_review": {
                "path": relative(review_path),
                "sha256": sha256_file(review_path),
            },
            "functional_freeze": {
                "path": relative(freeze_path),
                "sha256": freeze_sha256,
            },
            **{
                name: {
                    "path": relative(path),
                    "sha256": sha256_file(path),
                }
                for name, path in machine_paths.items()
            },
        },
        "decision": (
            "INDEPENDENT_AUDITS_GREEN_READY_FOR_CONTENT_ADDRESSED_REPLACEMENT_FREEZE"
            if trigger_state == "GREEN"
            else "BLOCK_REPLACEMENT_FREEZE"
        ),
        "long_training_authorized": False,
        "capability_claim": "NONE_INDEPENDENT_READINESS_AUDIT_ONLY",
        "non_claims": [
            "This audit does not prove physical performance optimality.",
            "This audit does not evaluate model capability or utility.",
            "This audit does not authorize or start long training.",
        ],
    }
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
