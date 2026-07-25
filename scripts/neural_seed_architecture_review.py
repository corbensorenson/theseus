#!/usr/bin/env python3
"""Execute matched private-development reviews for the neural-seed campaign.

The campaign controller used to validate hypothetical receipts without owning a
producer. This module closes that contract: it creates isolated review lineages,
trains the declared matched rung, generates one direct model output per frozen
case, invokes independent verifiers, and writes controller-compatible receipts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import moecot_language_arm_training as training
import neural_seed_local_english_raters as local_raters
from neural_seed_functional_cases import ARMS, materialize_cases, stable_hash
from neural_seed_functional_utility import source_disjoint_audit
from neural_seed_functional_verifiers import score_english_judgments, verify_candidate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/neural_seed_architecture_review.json"
DEFAULT_OUT = ROOT / "reports/neural_seed_architecture_review_status.json"
POLICY = "project_theseus_neural_seed_architecture_review_v1"
FREEZE_POLICY = "project_theseus_neural_seed_architecture_review_freeze_v1"
RECEIPT_POLICY = "project_theseus_architecture_review_receipt_v1"
SYSTEM_IDS = ("moecot_system", "dense_active_parameter", "dense_total_parameter")
COMPONENT_IDS = (training.SHARED_TRUNK_ID, *training.ARM_IDS, *training.CONTROL_IDS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--review-positions", type=int, default=100_000_000)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--target", action="append", choices=COMPONENT_IDS)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Bound each selected component for a mechanics smoke; zero executes its rung allocation.",
    )
    parser.add_argument("--out", default=relative(DEFAULT_OUT))
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max-steps cannot be negative")
    if args.freeze and (args.execute_training or args.evaluate):
        parser.error("--freeze cannot be combined with execution or evaluation")

    config_path = resolve(args.config)
    config = read_json(config_path)
    contract = build_contract(config, config_path, args.review_positions)
    if args.freeze:
        report = freeze_contract(contract)
    elif args.execute_training:
        report = execute_training(
            contract,
            targets=list(dict.fromkeys(args.target or [])),
            max_steps=args.max_steps,
        )
    elif args.evaluate:
        report = evaluate_all(contract)
    else:
        report = status(contract)
    write_json(resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 2 if report.get("trigger_state") == "RED" else 0


def build_contract(
    config: dict[str, Any], config_path: Path, review_positions: int
) -> dict[str, Any]:
    gaps: list[str] = []
    if config.get("policy") != POLICY:
        gaps.append("review_policy_mismatch")
    training_path = resolve(str(config.get("training_config") or ""))
    scale_path = resolve(str(config.get("scale_config") or ""))
    functional_path = resolve(str(config.get("functional_config") or ""))
    rater_path = resolve(str(config.get("local_rater_config") or ""))
    for label, path in (
        ("training", training_path),
        ("scale", scale_path),
        ("functional", functional_path),
        ("rater", rater_path),
    ):
        if not path.is_file():
            gaps.append(f"{label}_config_missing")
    if gaps:
        return {
            "policy": POLICY,
            "created_utc": now(),
            "trigger_state": "RED",
            "hard_gaps": gaps,
        }

    training_config = training.bind_scale_preregistration(read_json(training_path))
    canonical_plan = training.build_plan(training_config, config_path=training_path)
    gaps.extend(canonical_plan.get("hard_gaps") or [])
    scale_config = read_json(scale_path)
    declared = {
        int(key): value for key, value in (config.get("candidate_budgets") or {}).items()
    }
    allocation = declared.get(review_positions)
    if not isinstance(allocation, dict):
        gaps.append("review_budget_not_declared")
        allocation = {}
    phase_allocations = {
        int(key): value
        for key, value in (config.get("component_phase_budgets") or {}).items()
    }
    phase_allocation = phase_allocations.get(review_positions)
    if not isinstance(phase_allocation, dict):
        gaps.append("review_phase_budget_not_declared")
        phase_allocation = {}
    gaps.extend(validate_allocation(allocation, review_positions, phase_allocation))

    functional_config = review_functional_config(
        read_json(functional_path), config, review_positions
    )
    cases = materialize_cases(functional_config)
    source_audit = source_disjoint_audit(functional_config, cases)
    gaps.extend(source_audit.get("hard_gaps") or [])
    case_contract = [
        {key: value for key, value in row.items() if key != "model_visible"}
        for row in cases
    ]
    candidate_packet = {
        "policy": "project_theseus_architecture_review_candidate_packet_v1",
        "generator_visible_fields": list(
            (config.get("evaluation") or {}).get("generator_visible_fields") or []
        ),
        "rows": [row["model_visible"] for row in cases],
        "training_eligible": False,
        "confirmation_surface": False,
        "public_surface": False,
    }
    if any(set(row) != {"case_id", "arm_id", "prompt"} for row in candidate_packet["rows"]):
        gaps.append("candidate_packet_contains_evaluator_metadata")

    review_plan = build_review_plan(
        canonical_plan,
        config,
        allocation,
        phase_allocation,
        review_positions,
    )
    freeze_path = resolve(str(config.get("freeze") or ""))
    freeze = read_json(freeze_path) if freeze_path.is_file() else {}
    semantic = semantic_identity(
        config,
        training_path,
        scale_path,
        functional_path,
        rater_path,
        review_plan,
        case_contract,
        candidate_packet,
    )
    if freeze:
        gaps.extend(validate_freeze(freeze, semantic, review_positions))
    else:
        gaps.append("review_freeze_missing")
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": "RED" if gaps else "GREEN",
        "config": relative(config_path),
        "config_payload": config,
        "review_optimizer_positions": review_positions,
        "allocation": allocation,
        "phase_allocation": phase_allocation,
        "training_config_path": relative(training_path),
        "training_config": training_config,
        "scale_config_path": relative(scale_path),
        "scale_config": scale_config,
        "functional_config_path": relative(functional_path),
        "functional_config": functional_config,
        "local_rater_config_path": relative(rater_path),
        "local_rater_config": read_json(rater_path),
        "canonical_plan_sha256": canonical_plan.get("plan_sha256"),
        "review_plan": review_plan,
        "case_contract": case_contract,
        "case_contract_sha256": stable_hash(case_contract),
        "candidate_packet": candidate_packet,
        "candidate_packet_sha256": stable_hash(candidate_packet),
        "visible_case_ids_sha256": stable_hash(
            [row["case_id"] for row in candidate_packet["rows"]]
        ),
        "source_disjoint_audit": source_audit,
        "semantic_identity": semantic,
        "freeze_path": relative(freeze_path),
        "freeze": freeze,
        "hard_gaps": sorted(set(gaps)),
        "boundaries": dict(config.get("boundaries") or {}),
    }


def validate_allocation(
    allocation: dict[str, Any],
    review_positions: int,
    phase_allocation: dict[str, Any] | None = None,
) -> list[str]:
    gaps = []
    if set(allocation) != set(SYSTEM_IDS):
        gaps.append("candidate_allocation_inventory_mismatch")
        return gaps
    moecot = allocation.get("moecot_system") or {}
    if set(moecot) != {training.SHARED_TRUNK_ID, *training.ARM_IDS}:
        gaps.append("moecot_component_inventory_mismatch")
    if sum(int(value) for value in moecot.values()) != review_positions:
        gaps.append("moecot_total_position_budget_mismatch")
    if len({int(moecot.get(arm) or 0) for arm in training.ARM_IDS}) != 1:
        gaps.append("moecot_weak_tail_arm_budget_mismatch")
    if int(moecot.get(training.SHARED_TRUNK_ID) or 0) <= sum(
        int(moecot.get(arm) or 0) for arm in training.ARM_IDS
    ):
        gaps.append("moecot_shared_trunk_not_majority_allocation")
    for candidate in training.CONTROL_IDS:
        row = allocation.get(candidate) or {}
        if row != {candidate: review_positions}:
            gaps.append(f"dense_control_budget_mismatch:{candidate}")
    if phase_allocation is not None:
        component_totals = {
            **dict(moecot),
            **{
                candidate: review_positions
                for candidate in training.CONTROL_IDS
            },
        }
        if set(phase_allocation) != set(component_totals):
            gaps.append("phase_budget_component_inventory_mismatch")
        for component, expected_total in component_totals.items():
            phases = phase_allocation.get(component) or {}
            required = {
                "pretraining",
                "source_conditioned_pretraining",
                "supervision",
                "total",
            }
            if set(phases) != required:
                gaps.append(f"phase_budget_inventory_mismatch:{component}")
                continue
            observed = sum(
                int(phases[name])
                for name in (
                    "pretraining",
                    "source_conditioned_pretraining",
                    "supervision",
                )
            )
            if observed != int(phases["total"]) or observed != int(expected_total):
                gaps.append(f"phase_budget_total_mismatch:{component}")
    return gaps


def build_review_plan(
    canonical: dict[str, Any],
    config: dict[str, Any],
    allocation: dict[str, Any],
    phase_allocation: dict[str, Any],
    review_positions: int,
) -> dict[str, Any]:
    plan = copy.deepcopy(canonical)
    review_root = resolve(str(config["checkpoint_root"])) / str(review_positions)
    identity = stable_hash(
        {
            "policy": "project_theseus_architecture_review_plan_v1",
            "canonical_plan_sha256": canonical.get("plan_sha256"),
            "review_id": config.get("review_id"),
            "review_optimizer_positions": review_positions,
            "allocation": allocation,
            "checkpoint_root": relative(review_root),
        }
    )
    component_budget = {
        **dict(allocation.get("moecot_system") or {}),
        **dict(allocation.get("dense_active_parameter") or {}),
        **dict(allocation.get("dense_total_parameter") or {}),
    }
    for target_id in COMPONENT_IDS:
        target = plan["targets"][target_id]
        directory = review_root / target_id
        suffix = "expert_delta.safetensors" if target_id in training.ARM_IDS else "weights.safetensors"
        total_positions = int(component_budget[target_id])
        phases = dict(phase_allocation[target_id])
        pretraining_positions = int(phases["pretraining"])
        target.update(
            {
                "optimizer_target_positions": pretraining_positions,
                "minimum_optimizer_positions": pretraining_positions,
                "optimizer_repetition_factor": round(
                    pretraining_positions / max(1, int(target["unique_target_positions"])), 8
                ),
                "optimizer_repetition_ceiling_ready": True,
                "estimated_parameter_token_product": int(target["owned_parameter_count"])
                * total_positions,
                "checkpoint": relative(directory / suffix),
                "optimizer_state": relative(directory / "optimizer.safetensors"),
                "receipt": relative(directory / "training_receipt.json"),
                "plan_sha256": identity,
                "review_only": True,
                "review_optimizer_positions": review_positions,
                "review_component_total_optimizer_positions": total_positions,
                "review_phase_optimizer_positions": phases,
            }
        )
    shared = plan["targets"][training.SHARED_TRUNK_ID]["checkpoint"]
    for arm in training.ARM_IDS:
        plan["targets"][arm]["shared_trunk_checkpoint"] = shared
    plan.update(
        {
            "plan_sha256": identity,
            "mode": "matched_private_development_architecture_review",
            "canonical_plan_sha256": canonical.get("plan_sha256"),
            "review_optimizer_positions": review_positions,
            "review_checkpoint_root": relative(review_root),
            "review_allocation": allocation,
            "review_phase_allocation": phase_allocation,
            "checkpoint_inventory": training.inspect_checkpoint_inventory(
                plan["targets"],
                identity,
                (plan.get("stage") or {}).get("stage_signature"),
            ),
        }
    )
    return plan


def review_functional_config(
    base: dict[str, Any], config: dict[str, Any], review_positions: int
) -> dict[str, Any]:
    result = copy.deepcopy(base)
    surface = config["development_surface"]
    result["seed"] = int(surface["seed"]) + int(review_positions // 100_000_000) - 1
    result["variants_per_family"] = int(surface["variants_per_family"])
    result["expected_cases_per_arm"] = int(surface["expected_cases_per_arm"])
    prior = list((result.get("source_disjoint") or {}).get("prior_candidate_packets") or [])
    confirmation = ROOT / "reports/private_functional_utility_candidate_packet.json"
    if confirmation.is_file():
        prior.append({"path": relative(confirmation), "sha256": file_sha256(confirmation)})
    result["source_disjoint"]["prior_candidate_packets"] = prior
    return result


def semantic_identity(
    config: dict[str, Any],
    training_path: Path,
    scale_path: Path,
    functional_path: Path,
    rater_path: Path,
    plan: dict[str, Any],
    case_contract: list[dict[str, Any]],
    packet: dict[str, Any],
) -> dict[str, Any]:
    paths = {
        "review_config": resolve(DEFAULT_CONFIG),
        "review_runner": Path(__file__).resolve(),
        "training_config": training_path,
        "training_runner": ROOT / "scripts/moecot_language_arm_training.py",
        "functional_config": functional_path,
        "case_compiler": ROOT / "scripts/neural_seed_functional_cases.py",
        "verifier": ROOT / "scripts/neural_seed_functional_verifiers.py",
        "local_rater_config": rater_path,
        "local_rater_runner": ROOT / "scripts/neural_seed_local_english_raters.py",
        "scale_config": scale_path,
    }
    artifacts = {
        key: {"path": relative(path), "sha256": file_sha256(path)}
        for key, path in paths.items()
    }
    return {
        "policy": "project_theseus_architecture_review_semantic_identity_v1",
        "review_id": config.get("review_id"),
        "artifacts": artifacts,
        "plan_sha256": plan.get("plan_sha256"),
        "stage_signature": (plan.get("stage") or {}).get("stage_signature"),
        "case_contract_sha256": stable_hash(case_contract),
        "candidate_packet_sha256": stable_hash(packet),
        "verifier_budget_sha256": verifier_budget_sha(config),
    }


def freeze_contract(contract: dict[str, Any]) -> dict[str, Any]:
    gaps = [gap for gap in contract.get("hard_gaps") or [] if gap != "review_freeze_missing"]
    if gaps:
        return {
            "policy": FREEZE_POLICY,
            "created_utc": now(),
            "trigger_state": "RED",
            "hard_gaps": gaps,
        }
    path = resolve(str(contract["freeze_path"]))
    if path.exists():
        existing = read_json(path)
        validation = validate_freeze(
            existing,
            contract["semantic_identity"],
            int(contract["review_optimizer_positions"]),
        )
        return {
            **existing,
            "trigger_state": "GREEN" if not validation else "RED",
            "hard_gaps": validation,
        }
    freeze = {
        "policy": FREEZE_POLICY,
        "frozen_utc": now(),
        "immutable": True,
        "review_id": contract["config_payload"]["review_id"],
        "review_optimizer_positions": [int(contract["review_optimizer_positions"])],
        "semantic_identity": contract["semantic_identity"],
        "supersedes_freeze_sha256": str(
            contract["config_payload"].get("supersedes_freeze_sha256") or ""
        ),
        "development_surface_reusable": True,
        "training_eligible": False,
        "confirmation_surface_consumed": False,
        "public_surface_consumed": False,
        "trigger_state": "GREEN",
        "hard_gaps": [],
    }
    write_json(path, freeze)
    return freeze


def validate_freeze(
    freeze: dict[str, Any], semantic: dict[str, Any], review_positions: int
) -> list[str]:
    gaps = []
    if freeze.get("policy") != FREEZE_POLICY or freeze.get("immutable") is not True:
        gaps.append("review_freeze_policy_invalid")
    if review_positions not in [int(value) for value in freeze.get("review_optimizer_positions") or []]:
        gaps.append("review_position_not_frozen")
    if freeze.get("semantic_identity") != semantic:
        gaps.append("review_semantic_identity_mismatch")
    if freeze.get("training_eligible") is not False:
        gaps.append("review_surface_training_eligibility_invalid")
    return gaps


def status(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("trigger_state") == "RED":
        return contract
    progress = component_progress(contract)
    ready = all(row["state"] == "COMPLETE" for row in progress)
    receipts = review_receipt_inventory(contract)
    return {
        "policy": "project_theseus_neural_seed_architecture_review_status_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if ready and receipts["complete"] else "READY",
        "review_optimizer_positions": contract["review_optimizer_positions"],
        "review_plan_sha256": contract["review_plan"]["plan_sha256"],
        "component_progress": progress,
        "training_complete": ready,
        "review_receipts": receipts,
        "next_action": next_action(progress, receipts),
        "hard_gaps": [],
        "boundaries": contract["boundaries"],
    }


def component_progress(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    plan = contract["review_plan"]
    for target_id in contract["config_payload"]["execution"]["dependency_order"]:
        target = plan["targets"][target_id]
        receipt_path = resolve(str(target["receipt"]))
        receipt = read_json(receipt_path) if receipt_path.is_file() else {}
        positions = int(receipt.get("optimizer_positions") or 0)
        target_positions = int(target["review_component_total_optimizer_positions"])
        faults = []
        if receipt:
            try:
                training.validate_resume(
                    receipt,
                    plan,
                    target,
                    resolve(str(receipt.get("checkpoint") or target["checkpoint"])),
                    resolve(str(receipt.get("optimizer_state") or target["optimizer_state"])),
                )
            except ValueError as exc:
                faults.append(str(exc))
        rows.append(
            {
                "target_id": target_id,
                "state": "RED" if faults else "COMPLETE" if bool(receipt.get("complete")) and positions >= target_positions else "IN_PROGRESS" if receipt else "NOT_STARTED",
                "optimizer_positions": positions,
                "target_optimizer_positions": target_positions,
                "completion_fraction": round(min(1.0, positions / max(1, target_positions)), 8),
                "receipt": relative(receipt_path),
                "faults": faults,
            }
        )
    return rows


def execute_training(
    contract: dict[str, Any], *, targets: list[str], max_steps: int
) -> dict[str, Any]:
    if contract.get("trigger_state") == "RED":
        return contract
    config = contract["training_config"]
    plan = contract["review_plan"]
    authority = training.architecture_training_authority(config, max_steps=max_steps)
    if authority.get("trigger_state") != "GREEN":
        return {
            "policy": "project_theseus_architecture_review_training_execution_v1",
            "created_utc": now(),
            "trigger_state": "RED",
            "architecture_training_authority": authority,
            "hard_gaps": ["architecture_training_authority_denied"],
        }
    order = list(contract["config_payload"]["execution"]["dependency_order"])
    selected = targets or order
    invalid_order = [item for item in selected if item not in order]
    if invalid_order:
        raise ValueError("unknown review targets: " + ",".join(invalid_order))
    reports = []
    gaps = []
    for target_id in order:
        if target_id not in selected:
            continue
        if target_id in training.ARM_IDS:
            shared = component_row(component_progress(contract), training.SHARED_TRUNK_ID)
            if shared["state"] != "COMPLETE":
                gaps.append(f"dependency_incomplete:{target_id}:shared_trunk")
                break
        current = component_row(component_progress(contract), target_id)
        if current["state"] == "COMPLETE":
            continue
        invocation_started = time.perf_counter()
        fault = ""
        report: dict[str, Any] = {}
        try:
            report = training.execute_targets(
                config,
                plan,
                targets=[target_id],
                max_steps=max_steps,
                resume=current["state"] == "IN_PROGRESS",
            )
        except BaseException as exc:
            fault = f"{type(exc).__name__}:{exc}"
            raise
        finally:
            refreshed = component_row(component_progress(contract), target_id)
            append_training_invocation(
                contract,
                target_id=target_id,
                optimizer_positions_before=int(current["optimizer_positions"]),
                optimizer_positions_after=int(refreshed["optimizer_positions"]),
                wall_seconds=time.perf_counter() - invocation_started,
                max_steps=max_steps,
                fault=fault,
            )
        reports.append(compact_training_report(report, target_id))
        if report.get("trigger_state") == "RED":
            gaps.extend(f"{target_id}:{gap}" for gap in report.get("hard_gaps") or [])
            break
    final = component_progress(contract)
    return {
        "policy": "project_theseus_architecture_review_training_execution_v1",
        "created_utc": now(),
        "trigger_state": "RED" if gaps else "GREEN",
        "review_optimizer_positions": contract["review_optimizer_positions"],
        "architecture_training_authority": authority,
        "selected_targets": selected,
        "max_steps_per_target": max_steps,
        "component_progress": final,
        "execution_reports": reports,
        "hard_gaps": gaps,
        "boundaries": contract["boundaries"],
    }


def compact_training_report(report: dict[str, Any], target_id: str) -> dict[str, Any]:
    result = next(
        (row for row in report.get("results") or [] if row.get("target_id") == target_id),
        {},
    )
    return {
        "target_id": target_id,
        "trigger_state": report.get("trigger_state"),
        "optimizer_steps": result.get("optimizer_steps"),
        "optimizer_positions": result.get("optimizer_positions"),
        "complete": result.get("complete"),
        "wall_seconds": result.get("wall_seconds"),
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "hard_gaps": result.get("hard_gaps") or report.get("hard_gaps") or [],
    }


def evaluate_all(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("trigger_state") == "RED":
        return contract
    progress = component_progress(contract)
    if any(row["state"] != "COMPLETE" for row in progress):
        return {
            "policy": "project_theseus_architecture_review_evaluation_v1",
            "created_utc": now(),
            "trigger_state": "READY",
            "component_progress": progress,
            "next_action": next_action(progress, review_receipt_inventory(contract)),
            "hard_gaps": [],
        }
    started = time.perf_counter()
    rows = []
    gaps = []
    station_timing: dict[str, Any] = {}
    try:
        generation_started = time.perf_counter()
        prepared = [prepare_candidate_generation(contract, candidate_id) for candidate_id in SYSTEM_IDS]
        station_timing["candidate_generation_wall_seconds"] = round(
            time.perf_counter() - generation_started, 6
        )

        verification = verify_candidate_bundles(contract, prepared)
        station_timing["code_verification_wall_seconds"] = verification[
            "wall_seconds"
        ]

        rater_started = time.perf_counter()
        candidate_root = resolve(
            str(contract["config_payload"]["candidate_directory"])
        ) / str(contract["review_optimizer_positions"])
        judgment_dir = candidate_root / "english_judgments"
        packet_specs = [
            (row["candidate_id"], resolve(str(row["english_packet_path"])))
            for row in prepared
        ]
        rater_receipt = local_raters.execute(
            contract["local_rater_config"],
            resolve(contract["local_rater_config_path"]),
            packet_specs,
            judgment_dir=judgment_dir,
        )
        station_timing["blind_english_scoring_wall_seconds"] = round(
            time.perf_counter() - rater_started, 6
        )
        if rater_receipt.get("trigger_state") != "GREEN":
            raise ValueError(
                "local English rater execution failed: "
                + ",".join(rater_receipt.get("hard_gaps") or [])
            )
        shared_evaluation_wall = time.perf_counter() - started
        finalization_started = time.perf_counter()
        for prepared_row in prepared:
            rows.append(
                finalize_candidate_evaluation(
                    contract,
                    prepared_row,
                    verification,
                    rater_receipt,
                    shared_evaluation_wall_seconds=shared_evaluation_wall,
                )
            )
        station_timing["decision_and_receipt_publication_wall_seconds"] = round(
            time.perf_counter() - finalization_started, 6
        )
    except BaseException as exc:
        gaps.append(f"{type(exc).__name__}:{exc}")
    station_timing["end_to_end_wall_seconds"] = round(
        time.perf_counter() - started, 6
    )
    return {
        "policy": "project_theseus_architecture_review_evaluation_v1",
        "created_utc": now(),
        "trigger_state": "RED" if gaps else "GREEN",
        "review_optimizer_positions": contract["review_optimizer_positions"],
        "candidate_receipts": rows,
        "station_timing": station_timing,
        "evaluation_execution": {
            "candidate_generation": "serial_per_resident_candidate_model_on_one_unified_memory_device",
            "code_verification": "bounded_parallel_independent_sandboxes",
            "blind_english_scoring": "one_shared_model_load_pass_across_all_opaque_candidate_packets",
        },
        "hard_gaps": gaps,
        "boundaries": contract["boundaries"],
    }


def evaluate_candidate(contract: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    """Compatibility entry point for one candidate using the same shared evaluator path."""

    started = time.perf_counter()
    prepared = prepare_candidate_generation(contract, candidate_id)
    verification = verify_candidate_bundles(contract, [prepared])
    candidate_root = resolve(
        str(contract["config_payload"]["candidate_directory"])
    ) / str(contract["review_optimizer_positions"])
    rater_receipt = local_raters.execute(
        contract["local_rater_config"],
        resolve(contract["local_rater_config_path"]),
        [(candidate_id, resolve(str(prepared["english_packet_path"])))],
        judgment_dir=candidate_root / "english_judgments",
    )
    if rater_receipt.get("trigger_state") != "GREEN":
        raise ValueError(
            "local English rater execution failed: "
            + ",".join(rater_receipt.get("hard_gaps") or [])
        )
    return finalize_candidate_evaluation(
        contract,
        prepared,
        verification,
        rater_receipt,
        shared_evaluation_wall_seconds=time.perf_counter() - started,
    )


def prepare_candidate_generation(
    contract: dict[str, Any], candidate_id: str
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn

    config = contract["training_config"]
    plan = contract["review_plan"]
    functional = contract["functional_config"]
    stage_dir = resolve(str(config["stage_dir"]))
    metadata = read_json(stage_dir / "stage_metadata_v1.json")
    base = read_json(resolve(str(config["base_config"])))
    source_vocab = dict(metadata.get("source_vocab") or {})
    target_vocab = dict(metadata.get("target_vocab") or {})
    cases = materialize_cases(functional)
    visible = {row["case_id"]: row["model_visible"] for row in cases}
    models: dict[str, Any] = {}
    outputs: dict[str, str] = {}
    generation_ms: dict[str, float] = {}
    load_ms: dict[str, float] = {}
    started = time.perf_counter()
    for case in cases:
        target_id = case["arm_id"] if candidate_id == "moecot_system" else candidate_id
        if target_id not in models:
            load_started = time.perf_counter()
            target = plan["targets"][target_id]
            model = training.build_model(
                training.CausalTransformerConfig(
                    vocab_size=int(target.get("vocab_size") or plan["models"]["vocab_size"]),
                    **target["model"],
                ),
                mx=mx,
                nn=nn,
                state_role_lookup=None,
                source_to_target_lookup=training.build_source_to_target_lookup(
                    base,
                    metadata,
                    vocab_size=int(target.get("vocab_size") or plan["models"]["vocab_size"]),
                    identity_ranges=training.target_copy_identity_ranges(target),
                ),
                rope_kernel=str(config["training"].get("inference_rope_kernel") or "manual_reference"),
            )
            if target_id in training.ARM_IDS:
                model.load_weights(str(resolve(str(target["shared_trunk_checkpoint"]))), strict=False)
                model.load_weights(str(resolve(str(target["checkpoint"]))), strict=False)
            else:
                model.load_weights(str(resolve(str(target["checkpoint"]))))
            mx.eval(model.parameters())
            model.eval()
            models[target_id] = model
            load_ms[target_id] = round((time.perf_counter() - load_started) * 1000.0, 6)
        row = visible[case["case_id"]]
        generated_at = time.perf_counter()
        output, _generation = training.generate_model_text(
            models[target_id],
            str(row["prompt"]),
            source_vocab,
            target_vocab,
            base,
            max_tokens=int(config["evaluation"]["decode_max_target_tokens"]),
            max_source_tokens=int(config["supervision"]["maximum_source_encoded_tokens"]),
            beam_width=int(config["evaluation"]["beam_width"]),
            branching_factor=int(config["evaluation"]["branching_factor"]),
            length_penalty=float(config["evaluation"]["length_penalty"]),
            mx=mx,
        )
        outputs[case["case_id"]] = output
        generation_ms[case["case_id"]] = round(
            (time.perf_counter() - generated_at) * 1000.0, 6
        )
    models.clear()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    english_cases = [row for row in cases if row["arm_id"] == "english"]
    english_packet = build_english_packet(contract, candidate_id, english_cases, outputs)
    candidate_root = resolve(str(contract["config_payload"]["candidate_directory"])) / str(contract["review_optimizer_positions"])
    packet_path = candidate_root / f"{candidate_id}_english_packet.json"
    write_json(packet_path, english_packet)
    return {
        "candidate_id": candidate_id,
        "cases": cases,
        "outputs": outputs,
        "generation_duration_ms": generation_ms,
        "checkpoint_load_duration_ms_by_target": load_ms,
        "generation_wall_seconds": round(time.perf_counter() - started, 6),
        "english_packet_path": relative(packet_path),
    }


def verify_candidate_bundles(
    contract: dict[str, Any], prepared: list[dict[str, Any]]
) -> dict[str, Any]:
    execution = (
        (contract["config_payload"].get("evaluation") or {}).get(
            "verifier_parallelism"
        )
        or {}
    )
    maximum_workers = max(1, int(execution.get("maximum_workers") or 4))
    limits = {
        "python_isolated_tests_v1": 4,
        "deno_typescript_tests_v1": 2,
        "rust_cargo_tests_v1": 1,
        "html_dom_a11y_render_v1": 1,
        **{
            str(key): max(1, int(value))
            for key, value in (execution.get("per_verifier_maximum") or {}).items()
        },
    }
    semaphores = {
        kind: threading.BoundedSemaphore(value) for kind, value in limits.items()
    }
    tasks = []
    for bundle in prepared:
        for case in bundle["cases"]:
            if case["arm_id"] == "english":
                continue
            tasks.append((bundle["candidate_id"], case, bundle["outputs"][case["case_id"]]))

    def run(task: tuple[str, dict[str, Any], str]) -> tuple[str, dict[str, Any]]:
        candidate_id, case, output = task
        kind = str(case["verifier"]["kind"])
        semaphore = semaphores.setdefault(kind, threading.BoundedSemaphore(1))
        with semaphore:
            row = verify_candidate(case, output, contract["functional_config"])
        return candidate_id, row

    started = time.perf_counter()
    rows: dict[str, dict[str, dict[str, Any]]] = {
        bundle["candidate_id"]: {} for bundle in prepared
    }
    with ThreadPoolExecutor(max_workers=maximum_workers) as pool:
        futures = {pool.submit(run, task): task for task in tasks}
        for future in as_completed(futures):
            candidate_id, row = future.result()
            rows[candidate_id][str(row["case_id"])] = row
    ordered = {
        bundle["candidate_id"]: [
            rows[bundle["candidate_id"]][case["case_id"]]
            for case in bundle["cases"]
            if case["arm_id"] != "english"
        ]
        for bundle in prepared
    }
    return {
        "policy": "project_theseus_bounded_parallel_functional_verification_v1",
        "wall_seconds": round(time.perf_counter() - started, 6),
        "candidate_count": len(prepared),
        "case_count": len(tasks),
        "maximum_workers": maximum_workers,
        "per_verifier_maximum": limits,
        "sum_case_duration_ms": round(
            sum(float(row.get("duration_ms") or 0.0) for values in ordered.values() for row in values),
            3,
        ),
        "rows": ordered,
    }


def finalize_candidate_evaluation(
    contract: dict[str, Any],
    prepared: dict[str, Any],
    verification: dict[str, Any],
    rater_receipt: dict[str, Any],
    *,
    shared_evaluation_wall_seconds: float,
) -> dict[str, Any]:
    candidate_id = str(prepared["candidate_id"])
    plan = contract["review_plan"]
    cases = prepared["cases"]
    outputs = prepared["outputs"]
    generation_ms = prepared["generation_duration_ms"]
    load_ms = prepared["checkpoint_load_duration_ms_by_target"]
    code_rows = list((verification.get("rows") or {}).get(candidate_id) or [])
    for row in code_rows:
        row["generation_duration_ms"] = generation_ms[row["case_id"]]
    judgment_file = next(
        row
        for row in rater_receipt["judgment_files"]
        if row["label"] == candidate_id
    )
    judgments = read_jsonl(resolve(str(judgment_file["path"])))
    english = score_english_judgments(
        cases, outputs, judgments, contract["functional_config"]
    )
    if not english.get("valid"):
        raise ValueError("English judgment contract failed: " + ",".join(english.get("faults") or []))

    by_arm: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        arm_rows = english["results"] if arm == "english" else [row for row in code_rows if row["arm_id"] == arm]
        by_arm[arm] = {
            "case_count": len(arm_rows),
            "passed_count": sum(bool(row.get("passed")) for row in arm_rows),
        }
    passed = sum(row["passed_count"] for row in by_arm.values())
    total = sum(row["case_count"] for row in by_arm.values())
    wall = max(1e-9, float(shared_evaluation_wall_seconds))
    checkpoints = checkpoint_artifacts(plan, candidate_id)
    actual_positions = candidate_optimizer_positions(plan, candidate_id)
    cost = training_cost_accounting(contract, candidate_id)
    receipt = {
        "policy": RECEIPT_POLICY,
        "created_utc": now(),
        "candidate_id": candidate_id,
        "review_optimizer_positions": int(contract["review_optimizer_positions"]),
        "evidence": {
            "split": "private_dev",
            "source_disjoint": True,
            "direct_model_only": True,
            "confirmation_surface_consumed": False,
            "public_surface_consumed": False,
            "fallback_return_count": 0,
            "templates_renderers_routers_tools_credit": 0,
            "case_count": total,
            "passed_count": passed,
            "by_arm": by_arm,
            "plan_sha256": plan["plan_sha256"],
            "stage_signature": plan["stage"]["stage_signature"],
            "checkpoint_sha256": stable_hash(checkpoints),
            "checkpoint_artifacts": checkpoints,
            "evaluator_sha256": evaluator_sha(contract),
            "case_contract_sha256": contract["case_contract_sha256"],
            "visible_case_ids_sha256": contract["visible_case_ids_sha256"],
            "verifier_budget_sha256": verifier_budget_sha(contract["config_payload"]),
            "optimizer_positions": actual_positions,
            "cost_accounting": cost,
            "accepted_verified_outputs_per_second": round(passed / max(1e-9, wall), 8),
            "wall_seconds": round(wall, 6),
            "wall_seconds_scope": "shared_end_to_end_review_evaluation_denominator",
            "candidate_generation_wall_seconds": prepared["generation_wall_seconds"],
            "checkpoint_load_duration_ms_by_target": load_ms,
            "generation_duration_ms_total": round(sum(generation_ms.values()), 6),
            "verification_execution": {
                key: verification[key]
                for key in (
                    "policy",
                    "wall_seconds",
                    "candidate_count",
                    "case_count",
                    "maximum_workers",
                    "per_verifier_maximum",
                    "sum_case_duration_ms",
                )
            },
            "blind_rater_execution": {
                "packet_count": int(rater_receipt.get("packet_count") or 0),
                "blind_item_count": int(rater_receipt.get("blind_item_count") or 0),
                "model_load_passes": len(rater_receipt.get("model_receipts") or []),
                "local_evaluator_inference_calls": int(
                    rater_receipt.get("local_evaluator_inference_calls") or 0
                ),
            },
            "local_evaluator_inference_calls": int(rater_receipt.get("local_evaluator_inference_calls") or 0),
            "local_evaluator_inference_calls_scope": "shared_review_total_not_additive_across_candidate_receipts",
            "candidate_budget_per_case": 1,
            "candidate_outputs_training_eligible": False,
        },
    }
    output_root = resolve(str(contract["config_payload"]["review_directory"]))
    output = output_root / f"{contract['review_optimizer_positions']}_{candidate_id}.json"
    if output.exists():
        prior = read_json(output)
        if prior != receipt:
            raise ValueError(f"review receipt already exists with different evidence: {relative(output)}")
    else:
        write_json(output, receipt)
    candidate_bundle = {
        "policy": "project_theseus_architecture_review_candidate_bundle_v1",
        "created_utc": now(),
        "candidate_id": candidate_id,
        "case_contract_sha256": contract["case_contract_sha256"],
        "generator_visible_fields": ["case_id", "arm_id", "prompt"],
        "rows": [
            {
                "case_id": case["case_id"],
                "output": outputs[case["case_id"]],
                "output_sha256": hashlib.sha256(outputs[case["case_id"]].encode()).hexdigest(),
                "generation_duration_ms": generation_ms[case["case_id"]],
            }
            for case in cases
        ],
        "training_eligible": False,
        "fallback_return_count": 0,
        "templates_renderers_routers_tools_credit": 0,
        "public_training_rows_written": 0,
    }
    candidate_root = resolve(
        str(contract["config_payload"]["candidate_directory"])
    ) / str(contract["review_optimizer_positions"])
    write_json(candidate_root / f"{candidate_id}_candidates.json", candidate_bundle)
    return {
        "candidate_id": candidate_id,
        "receipt": relative(output),
        "receipt_sha256": file_sha256(output),
        "passed_count": passed,
        "case_count": total,
        "by_arm": by_arm,
        "optimizer_positions": actual_positions,
        "wall_seconds": round(wall, 6),
    }


def build_english_packet(
    contract: dict[str, Any],
    candidate_id: str,
    cases: list[dict[str, Any]],
    outputs: dict[str, str],
) -> dict[str, Any]:
    items = []
    for case in cases:
        output = outputs[case["case_id"]]
        output_sha = hashlib.sha256(output.encode()).hexdigest()
        items.append(
            {
                "blind_item_id": stable_hash({"case_id": case["case_id"], "candidate_sha256": output_sha})[:24],
                "case_id": case["case_id"],
                "prompt": case["prompt"],
                "candidate_output": output,
                "candidate_sha256": output_sha,
                "dimensions": list(contract["functional_config"]["english_scoring"]["dimensions"]),
                "score_scale": list(contract["functional_config"]["english_scoring"]["score_scale"]),
            }
        )
    core = {
        "policy": "project_theseus_blind_english_judgment_packet_v1",
        "freeze_sha256": stable_hash(contract["freeze"]),
        "item_count": len(items),
        "items": items,
        "judgment_required_fields": ["case_id", "blind_item_id", "candidate_sha256", "rater_id", "scores"],
        "model_identity_present": False,
        "checkpoint_identity_present": False,
        "reference_answer_present": False,
    }
    return {
        **core,
        "created_utc": now(),
        "trigger_state": "GREEN" if len(items) == 32 else "RED",
        "packet_sha256": stable_hash(core),
        "opaque_label": candidate_id,
        "training_eligible": False,
    }


def checkpoint_artifacts(plan: dict[str, Any], candidate_id: str) -> list[dict[str, Any]]:
    targets = [*training.ARM_IDS, training.SHARED_TRUNK_ID] if candidate_id == "moecot_system" else [candidate_id]
    rows = []
    for target_id in sorted(targets):
        target = plan["targets"][target_id]
        receipt = read_json(resolve(str(target["receipt"])))
        checkpoint = resolve(str(receipt["checkpoint"]))
        if not receipt.get("complete") or file_sha256(checkpoint) != receipt.get("checkpoint_sha256"):
            raise ValueError(f"review checkpoint incomplete or invalid: {target_id}")
        rows.append(
            {
                "target_id": target_id,
                "path": relative(checkpoint),
                "sha256": file_sha256(checkpoint),
                "optimizer_positions": int(receipt.get("optimizer_positions") or 0),
            }
        )
    return rows


def candidate_optimizer_positions(plan: dict[str, Any], candidate_id: str) -> int:
    targets = [training.SHARED_TRUNK_ID, *training.ARM_IDS] if candidate_id == "moecot_system" else [candidate_id]
    return sum(
        int(read_json(resolve(str(plan["targets"][target]["receipt"]))).get("optimizer_positions") or 0)
        for target in targets
    )


def training_cost_accounting(
    contract: dict[str, Any], candidate_id: str
) -> dict[str, Any]:
    plan = contract["review_plan"]
    targets = (
        [training.SHARED_TRUNK_ID, *training.ARM_IDS]
        if candidate_id == "moecot_system"
        else [candidate_id]
    )
    ledger = read_training_invocations(contract)
    components = []
    total_positions = 0
    active_forward_ptp = 0
    trainable_update_ptp = 0
    wall_seconds = 0.0
    for target_id in targets:
        target = plan["targets"][target_id]
        receipt = read_json(resolve(str(target["receipt"])))
        positions = int(receipt.get("optimizer_positions") or 0)
        parameter_count = int(receipt.get("parameter_count") or target["parameter_count"])
        trainable_count = int(
            receipt.get("trainable_parameter_count") or target["owned_parameter_count"]
        )
        target_ledger = [row for row in ledger if row.get("target_id") == target_id]
        component_wall = sum(float(row["wall_seconds"]) for row in target_ledger)
        components.append(
            {
                "target_id": target_id,
                "optimizer_positions": positions,
                "active_forward_parameter_count": parameter_count,
                "trainable_update_parameter_count": trainable_count,
                "active_forward_parameter_token_product": parameter_count * positions,
                "trainable_update_parameter_token_product": trainable_count * positions,
                "measured_training_wall_seconds": round(component_wall, 6),
                "invocation_count": len(target_ledger),
            }
        )
        total_positions += positions
        active_forward_ptp += parameter_count * positions
        trainable_update_ptp += trainable_count * positions
        wall_seconds += component_wall
    gaps = validate_training_invocations(contract, targets, ledger)
    if gaps:
        raise ValueError("review training cost ledger invalid: " + ",".join(gaps))
    return {
        "policy": "project_theseus_architecture_review_training_cost_v1",
        "components": components,
        "optimizer_positions": total_positions,
        "active_forward_parameter_token_product": active_forward_ptp,
        "trainable_update_parameter_token_product": trainable_update_ptp,
        "measured_training_wall_seconds": round(wall_seconds, 6),
        "invocation_count": sum(row["invocation_count"] for row in components),
        "transfer_and_merge_wall_seconds": 0.0,
        "energy_joules": None,
        "energy_measurement_state": "UNAVAILABLE_ON_CURRENT_MACOS_ROUTE",
        "hard_gaps": [],
    }


def training_ledger_path(contract: dict[str, Any]) -> Path:
    return (
        resolve(str(contract["config_payload"]["checkpoint_root"]))
        / str(contract["review_optimizer_positions"])
        / "training_invocations.jsonl"
    )


def append_training_invocation(
    contract: dict[str, Any],
    *,
    target_id: str,
    optimizer_positions_before: int,
    optimizer_positions_after: int,
    wall_seconds: float,
    max_steps: int,
    fault: str,
) -> None:
    path = training_ledger_path(contract)
    rows = read_jsonl(path) if path.is_file() else []
    row = {
        "policy": "project_theseus_architecture_review_training_invocation_v1",
        "created_utc": now(),
        "review_plan_sha256": contract["review_plan"]["plan_sha256"],
        "target_id": target_id,
        "optimizer_positions_before": optimizer_positions_before,
        "optimizer_positions_after": optimizer_positions_after,
        "optimizer_positions_added": max(
            0, optimizer_positions_after - optimizer_positions_before
        ),
        "wall_seconds": round(max(0.0, wall_seconds), 6),
        "max_steps": max_steps,
        "fault": fault,
    }
    rows.append(row)
    write_jsonl_atomic(path, rows)


def read_training_invocations(contract: dict[str, Any]) -> list[dict[str, Any]]:
    path = training_ledger_path(contract)
    return read_jsonl(path) if path.is_file() else []


def validate_training_invocations(
    contract: dict[str, Any], targets: list[str], rows: list[dict[str, Any]]
) -> list[str]:
    gaps = []
    plan_sha = contract["review_plan"]["plan_sha256"]
    for target_id in targets:
        target_rows = [row for row in rows if row.get("target_id") == target_id]
        expected_before = 0
        for row in target_rows:
            if row.get("policy") != "project_theseus_architecture_review_training_invocation_v1":
                gaps.append(f"training_ledger_policy_mismatch:{target_id}")
            if row.get("review_plan_sha256") != plan_sha:
                gaps.append(f"training_ledger_plan_mismatch:{target_id}")
            before = int(row.get("optimizer_positions_before") or 0)
            after = int(row.get("optimizer_positions_after") or 0)
            if before != expected_before or after < before:
                gaps.append(f"training_ledger_interval_invalid:{target_id}")
            if float(row.get("wall_seconds") or 0.0) <= 0.0:
                gaps.append(f"training_ledger_wall_invalid:{target_id}")
            if row.get("fault"):
                gaps.append(f"training_ledger_fault:{target_id}")
            expected_before = after
        receipt = read_json(resolve(str(contract["review_plan"]["targets"][target_id]["receipt"])))
        if not target_rows or expected_before != int(receipt.get("optimizer_positions") or 0):
            gaps.append(f"training_ledger_receipt_position_mismatch:{target_id}")
    return sorted(set(gaps))


def evaluator_sha(contract: dict[str, Any]) -> str:
    semantic = contract["semantic_identity"]
    names = ("review_runner", "case_compiler", "verifier", "local_rater_config", "local_rater_runner")
    return stable_hash({name: semantic["artifacts"][name] for name in names})


def verifier_budget_sha(config: dict[str, Any]) -> str:
    return stable_hash(
        {
            "evaluation": config.get("evaluation") or {},
            "development_surface": config.get("development_surface") or {},
        }
    )


def review_receipt_inventory(contract: dict[str, Any]) -> dict[str, Any]:
    root = resolve(str(contract["config_payload"]["review_directory"]))
    rows = []
    for candidate in SYSTEM_IDS:
        path = root / f"{contract['review_optimizer_positions']}_{candidate}.json"
        rows.append(
            {
                "candidate_id": candidate,
                "present": path.is_file(),
                "path": relative(path),
                "sha256": file_sha256(path) if path.is_file() else "",
            }
        )
    return {"complete": all(row["present"] for row in rows), "rows": rows}


def next_action(progress: list[dict[str, Any]], receipts: dict[str, Any]) -> dict[str, Any]:
    fault = next((row for row in progress if row["state"] == "RED"), None)
    if fault:
        return {"kind": "repair_review_lineage", "target_id": fault["target_id"], "faults": fault["faults"]}
    pending = next((row for row in progress if row["state"] != "COMPLETE"), None)
    if pending:
        return {"kind": "train_review_component", "target_id": pending["target_id"], "remaining_optimizer_positions": pending["target_optimizer_positions"] - pending["optimizer_positions"]}
    if not receipts["complete"]:
        return {"kind": "evaluate_matched_candidates", "candidate_ids": list(SYSTEM_IDS)}
    return {"kind": "run_campaign_controller"}


def component_row(rows: list[dict[str, Any]], target_id: str) -> dict[str, Any]:
    return next(row for row in rows if row["target_id"] == target_id)


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "policy",
            "created_utc",
            "trigger_state",
            "review_optimizer_positions",
            "training_complete",
            "next_action",
            "hard_gaps",
        )
        if key in report
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
