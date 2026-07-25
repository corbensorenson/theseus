#!/usr/bin/env python3
"""Content-bound T0A implementation cards and candidate-specific canary leases."""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import host_resource_safety
import pretraining_optimizers


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs" / "pretraining_architecture_candidates.json"
POLICY = "project_theseus_pretraining_architecture_candidates_v1"


class CandidateCanaryFault(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode()).hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def resolved_execution_policy(
    candidate: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    """Merge the contract-owned execution override for one candidate."""

    candidate_id = str(candidate.get("candidate_id") or "")
    overrides = contract.get("execution_policy_overrides") or {}
    return {
        **dict(candidate.get("execution_policy") or {}),
        **dict(overrides.get(candidate_id) or {}),
    }


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "policy", "schema_version", "owner", "scratch_root",
        "host_safety_policy",
        "immutable_control_roots", "stable_slots", "implementation_card_required_fields",
        "implementation_cards", "compatibility_rules", "canary_required_fields",
        "canaries", "hard_boundaries",
    }
    missing = sorted(required.difference(contract))
    if missing or contract.get("policy") != POLICY:
        raise CandidateCanaryFault("contract_invalid:" + ",".join(missing))
    cards = contract["implementation_cards"]
    safety = contract["host_safety_policy"]
    if safety.get("required_for_accelerator_subprocesses") is not True:
        raise CandidateCanaryFault("host_safety_policy_not_required")
    host_resource_safety.policy_from_mapping(
        safety,
        maximum_wall_seconds=max(
            float(row.get("max_wall_seconds") or 0)
            for row in contract["canaries"]
        ),
    )
    card_required = set(contract["implementation_card_required_fields"])
    card_ids: set[str] = set()
    covered_slots: set[str] = set()
    for card in cards:
        absent = sorted(field for field in card_required if not card.get(field))
        if absent:
            raise CandidateCanaryFault(f"implementation_card_incomplete:{card.get('id')}:{','.join(absent)}")
        if card["id"] in card_ids:
            raise CandidateCanaryFault(f"implementation_card_duplicate:{card['id']}")
        card_ids.add(card["id"])
        covered_slots.add(card["slot"])
        if card["slot"] not in contract["stable_slots"]:
            raise CandidateCanaryFault(f"implementation_card_slot_unknown:{card['id']}")
        missing_sources = [path for path in card["source_paths"] if not resolve(path).is_file()]
        if missing_sources:
            raise CandidateCanaryFault(f"implementation_card_source_missing:{card['id']}:{','.join(missing_sources)}")
    if covered_slots != set(contract["stable_slots"]):
        raise CandidateCanaryFault("stable_slot_coverage_incomplete")
    canary_required = set(contract["canary_required_fields"])
    canary_ids: set[str] = set()
    for row in contract["canaries"]:
        absent = sorted(field for field in canary_required if row.get(field) in (None, "", [], {}))
        if absent:
            raise CandidateCanaryFault(f"canary_incomplete:{row.get('candidate_id')}:{','.join(absent)}")
        candidate_id = str(row["candidate_id"])
        if candidate_id in canary_ids:
            raise CandidateCanaryFault(f"canary_duplicate:{candidate_id}")
        canary_ids.add(candidate_id)
        unknown = sorted(set(row["implementation_ids"]) - card_ids)
        if unknown:
            raise CandidateCanaryFault(f"canary_implementation_unknown:{candidate_id}:{','.join(unknown)}")
        for key in ("max_steps", "max_positions", "max_wall_seconds", "max_peak_memory_mib", "max_disk_mib"):
            if int(row[key]) <= 0:
                raise CandidateCanaryFault(f"canary_budget_invalid:{candidate_id}:{key}")
        if len(set(row["seeds"])) != len(row["seeds"]):
            raise CandidateCanaryFault(f"canary_seed_duplicate:{candidate_id}")
        raw_execution = dict(row.get("execution_policy") or {})
        execution = resolved_execution_policy(row, contract)
        selective_fp32_policy = (
            execution.get("selective_fp32_trainables") is True
            and candidate_id
            in {"rdc_kerc_k5_adequacy", "rdc_kerc_k5_overfit"}
            and execution.get("compute_dtype") == "bfloat16"
            and execution.get("fp32_master") is False
            and execution.get("parameter_initialization_dtype") == "bfloat16"
            and execution.get(
                "exact_checkpoint_placeholder_initialization"
            )
            is True
            and execution.get("freeze_warm_trunk_train_kerc_delta") is True
            and int(execution.get("kerc_delta_stage_only") or -1) == 1
            and (
                (
                    execution.get("kerc_stage_train_stage_embedding") is True
                    and execution.get("kerc_stage_detach_frozen_trunk") is False
                )
                or (
                    execution.get("kerc_stage_train_stage_embedding") is False
                    and execution.get("kerc_stage_detach_frozen_trunk") is True
                )
            )
            and execution.get(
                "continuation_optimizer_state_projection_policy"
            )
            == "project_theseus_exact_kerc_stage_optimizer_projection_v1"
            and bool(execution.get("continuation_source_checkpoint_sha256"))
            and bool(execution.get("continuation_source_report"))
        )
        if execution and (
            any(
                key in raw_execution and int(raw_execution.get(key) or 0) < 0
                    for key in (
                        "attention_query_chunk_size",
                        "attention_key_chunk_size",
                        "compact_partition_width_quantum",
                        "fresh_process_step_segment",
                        "optimizer_state_offload_minimum_target_positions",
                    )
            )
            or (
                "token_loss_position_chunk_size" in raw_execution
                and int(raw_execution.get("token_loss_position_chunk_size") or 0)
                <= 0
            )
            or (
                "maximum_training_sequence_tokens" in raw_execution
                and int(raw_execution.get("maximum_training_sequence_tokens") or 0)
                <= 0
            )
            or int(execution.get("batch_size") or 0) <= 0
            or int(execution.get("mlx_cache_limit_mib") or 0) <= 0
            or int(execution.get("attention_query_chunk_size") or 0) < 0
            or int(execution.get("attention_key_chunk_size") or 0) < 0
            or (
                "maximum_training_sequence_tokens" in execution
                and int(execution.get("maximum_training_sequence_tokens") or 0)
                <= 0
            )
            or any(
                target not in set(row.get("allowed_targets") or [])
                or int(limit) <= 0
                or int(limit)
                > int(execution.get("maximum_training_sequence_tokens") or 0)
                for target, limit in dict(
                    execution.get(
                        "maximum_supervised_training_sequence_tokens_by_target"
                    )
                    or {}
                ).items()
            )
            or not isinstance(execution.get("clear_mlx_cache_before_step"), bool)
            or (
                "clear_mlx_cache_after_backward" in execution
                and not isinstance(
                    execution.get("clear_mlx_cache_after_backward"), bool
                )
            )
            or not isinstance(execution.get("clear_mlx_cache_after_step"), bool)
            or not isinstance(execution.get("gradient_checkpointing"), bool)
            or not isinstance(execution.get("objective_gradient_checkpointing"), bool)
            or (
                "objective_gradient_decomposition" in execution
                and not isinstance(
                    execution.get("objective_gradient_decomposition"), bool
                )
            )
            or (
                "token_loss_position_chunk_size" in execution
                and int(execution.get("token_loss_position_chunk_size") or 0)
                <= 0
            )
            or not isinstance(execution.get("transactional_eager_step"), bool)
                or not isinstance(
                    execution.get("compact_encoder_decoder_partitions"), bool
                )
                or int(execution.get("compact_partition_width_quantum") or 0) < 0
                or int(execution.get("fresh_process_step_segment") or 0) < 0
                or (
                    int(execution.get("fresh_process_step_segment") or 0)
                    and execution.get("candidate_scratch_resume_policy")
                    != "exact_fresh_process_segment_v1"
                )
                or (
                    int(execution.get("compact_partition_width_quantum") or 0)
                    and execution.get("compact_encoder_decoder_partitions")
                    is not True
                )
            or execution.get("compute_dtype") not in {"float32", "bfloat16"}
            or not isinstance(execution.get("fp32_master"), bool)
            or (
                "selective_fp32_trainables" in execution
                and not isinstance(
                    execution.get("selective_fp32_trainables"), bool
                )
            )
            or (
                "kerc_stage_train_stage_embedding" in execution
                and not isinstance(
                    execution.get("kerc_stage_train_stage_embedding"), bool
                )
            )
            or (
                "kerc_stage_detach_frozen_trunk" in execution
                and not isinstance(
                    execution.get("kerc_stage_detach_frozen_trunk"), bool
                )
            )
            or execution.get("parameter_initialization_dtype", "float32")
            not in {"float32", "bfloat16"}
            or (
                "exact_checkpoint_placeholder_initialization" in execution
                and not isinstance(
                    execution.get(
                        "exact_checkpoint_placeholder_initialization"
                    ),
                    bool,
                )
            )
            or not isinstance(execution.get("require_external_watchdog"), bool)
            or execution.get("optimizer_id") not in pretraining_optimizers.OPTIMIZER_IDS
            or not 0.0
            <= float(execution.get("target_token_frequency_balance_power") or 0.0)
            <= 1.0
            or (
                "initialization_policy" in execution
                and execution.get("initialization_policy")
                != "registered_shared_trunk_progress_checkpoint_common_subspace_v1"
            )
            or (
                execution.get("compute_dtype") == "bfloat16"
                and execution.get("fp32_master") is not True
                and not selective_fp32_policy
            )
            or (
                execution.get("selective_fp32_trainables") is True
                and not selective_fp32_policy
            )
        ):
            raise CandidateCanaryFault(
                f"canary_execution_policy_invalid:{candidate_id}"
            )
    overrides = contract.get("execution_policy_overrides") or {}
    if not isinstance(overrides, dict):
        raise CandidateCanaryFault("execution_policy_overrides_invalid")
    unknown_overrides = sorted(set(overrides) - canary_ids)
    if unknown_overrides:
        raise CandidateCanaryFault(
            "execution_policy_override_candidate_unknown:"
            + ",".join(unknown_overrides)
        )
    kerc_ids = {
        candidate_id
        for candidate_id in canary_ids
        if candidate_id.startswith("rdc_kerc_")
    }
    if set(overrides) != kerc_ids:
        raise CandidateCanaryFault(
            "kerc_execution_policy_override_coverage_incomplete"
        )
    host_overrides = contract.get("host_safety_policy_overrides") or {}
    if (
        not isinstance(host_overrides, dict)
        or set(host_overrides) != {"rdc_kerc_k5_adequacy"}
        or host_overrides["rdc_kerc_k5_adequacy"]
        != {"maximum_swapout_growth_mib": 320}
    ):
        raise CandidateCanaryFault("host_safety_policy_overrides_invalid")
    for candidate_id, override in overrides.items():
        required_override = {
            "objective_gradient_decomposition": True,
            "token_loss_position_chunk_size": 128,
        }
        if candidate_id == "rdc_kerc_adequacy":
            allowed_override = {
                **required_override,
                "token_loss_position_chunk_size": 64,
                "attention_query_chunk_size": 32,
                "attention_key_chunk_size": 32,
            }
        elif candidate_id == "rdc_kerc_k5_adequacy":
            allowed_override = {
                **required_override,
                "token_loss_position_chunk_size": 32,
                "maximum_training_sequence_tokens": 628,
                    "maximum_supervised_training_sequence_tokens_by_target": {
                        "english_kerc": 628
                    },
                    "compact_partition_width_quantum": 0,
                    "fresh_process_step_segment": 1,
                    "candidate_scratch_resume_policy": (
                        "exact_fresh_process_segment_v1"
                    ),
                    "retain_segment_checkpoint_generations": True,
                    "attention_query_chunk_size": 32,
                "attention_key_chunk_size": 32,
                "optimizer_state_offload_between_steps": True,
                "optimizer_state_offload_minimum_target_positions": 600,
                "kerc_resource_stress_prefix": False,
                "continuation_source_report": (
                    "reports/rdc_kerc_k5_compiler_path_qualified_semantic_pointer_"
                    "fidelity_loss_retained_generations_v4_planbb9f_regression_"
                    "search_step27_merged_fp32_seed_20260722.json"
                ),
                "continuation_source_report_sha256": (
                    "d52ff92917b5dfa7036ce06b7965208d11ca170fdba4aae082f8fb0afb208d3b"
                ),
                "continuation_source_plan_sha256": (
                    "bb9fe9a5d3f6651a23780dde996e3fe49b0784cc9900c580af42f1a878e4a4d7"
                ),
                "continuation_source_checkpoint_sha256": (
                    "e1c76ec1276dc3e32ffa4dca3bd1c649ca49c633d0afcd4691a413a8a09ac691"
                ),
                "continuation_source_optimizer_state_sha256": (
                    "ff51b259c6f018fd33851665a08d3b1041a1c0a86379cb1e1540422573d76ecf"
                ),
                "continuation_source_mlx_rng_state_sha256": (
                    "f2f82eb4678013d3e8f1edaf53edfb29a0c1cdd8d6402582e1bbcc9f375d1bc9"
                ),
                "continuation_source_optimizer_steps": 4657,
                "continuation_source_optimizer_positions": 457027,
                "continuation_learning_rate": 0.00003,
                "continuation_min_learning_rate": 0.00003,
                "large_sequence_route": (
                    "compiler_full_stage_semantic_target_mass_rebalance_capped_"
                    "inverse_sqrt_lr_min_position32_q32_k32_coverage128_v19"
                ),
            }
        elif candidate_id == "rdc_kerc_k5_overfit":
            allowed_override = {
                **required_override,
                "token_loss_position_chunk_size": 32,
                "kerc_stage_minimum_coverage_rows": 8,
                "kerc_stage_coverage_multiplier": 16,
            }
        else:
            allowed_override = required_override
        if override != allowed_override:
            raise CandidateCanaryFault(
                f"kerc_execution_policy_override_invalid:{candidate_id}"
            )
        candidate = next(
            row
            for row in contract["canaries"]
            if row["candidate_id"] == candidate_id
        )
        if (
            candidate.get("execution_policy") or {}
        ).get("objective_gradient_checkpointing") is not True:
            raise CandidateCanaryFault(
                f"kerc_objective_checkpointing_required:{candidate_id}"
            )
    measured_overrides = contract.get("measured_preflight_overrides") or {}
    if not isinstance(measured_overrides, dict) or set(measured_overrides) != {
        "rdc_kerc_k5_adequacy"
    }:
        raise CandidateCanaryFault("measured_preflight_override_coverage_invalid")
    measured_override = measured_overrides["rdc_kerc_k5_adequacy"]
    if (
        not isinstance(measured_override, dict)
        or set(measured_override)
        != {
            "path",
            "sha256",
            "qualified_maximum_training_sequence_tokens",
            "qualified_maximum_token_loss_position_chunk_size",
            "qualified_maximum_attention_query_chunk_size",
            "qualified_maximum_attention_key_chunk_size",
            "command_marker",
        }
        or int(
            measured_override.get("qualified_maximum_training_sequence_tokens")
            or 0
        )
        != 1005
        or int(
            measured_override.get(
                "qualified_maximum_token_loss_position_chunk_size"
            )
            or 0
        )
        != 64
        or int(
            measured_override.get(
                "qualified_maximum_attention_query_chunk_size"
            )
            or 0
        )
        != 32
        or int(
            measured_override.get(
                "qualified_maximum_attention_key_chunk_size"
            )
            or 0
        )
        != 32
        or measured_override.get("command_marker") != "rdc_kerc_adequacy"
    ):
        raise CandidateCanaryFault("measured_preflight_override_invalid")
    boundaries = contract["hard_boundaries"]
    expected_zero = ("public_training_rows", "external_inference_calls", "fallback_template_router_tool_credit", "confirmation_surface_consumption")
    if any(boundaries.get(key) != 0 for key in expected_zero):
        raise CandidateCanaryFault("hard_boundary_nonzero")
    if boundaries.get("production_checkpoint_mutation") is not False or boundaries.get("resume_from_candidate_scratch") is not False:
        raise CandidateCanaryFault("hard_boundary_boolean_invalid")
    return contract


def candidate_host_safety_mapping(
    candidate_id: str, contract: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve the exact host guard mapping owned by one candidate."""

    contract = contract or load_contract()
    candidate = next(
        (
            row
            for row in contract["canaries"]
            if row["candidate_id"] == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise CandidateCanaryFault(f"candidate_unknown:{candidate_id}")
    mapping = {
        **contract["host_safety_policy"],
        **dict(candidate.get("host_safety_overrides") or {}),
        **dict(
            (
                contract.get("host_safety_policy_overrides")
                or {}
            ).get(candidate_id)
            or {}
        ),
    }
    measured_preflight = {
        **dict(candidate.get("measured_launch_preflight") or {}),
        **dict(
            (contract.get("measured_preflight_overrides") or {}).get(
                candidate_id
            )
            or {}
        ),
    }
    if measured_preflight:
        receipt_path = resolve(str(measured_preflight.get("path") or ""))
        expected_sha256 = str(measured_preflight.get("sha256") or "")
        if (
            not receipt_path.is_file()
            or hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            != expected_sha256
        ):
            raise CandidateCanaryFault(
                f"candidate_measured_preflight_identity_mismatch:{candidate_id}"
            )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        command = [str(value) for value in receipt.get("command") or []]
        qualified_sequence_tokens = int(
            measured_preflight.get("qualified_maximum_training_sequence_tokens")
            or 0
        )
        current_sequence_tokens = int(
            resolved_execution_policy(candidate, contract).get(
                "maximum_training_sequence_tokens"
            )
            or 0
        )
        qualified_position_chunk_size = int(
            measured_preflight.get(
                "qualified_maximum_token_loss_position_chunk_size"
            )
            or 0
        )
        current_position_chunk_size = int(
            resolved_execution_policy(candidate, contract).get(
                "token_loss_position_chunk_size"
            )
            or 0
        )
        qualified_query_chunk_size = int(
            measured_preflight.get(
                "qualified_maximum_attention_query_chunk_size"
            )
            or 0
        )
        qualified_key_chunk_size = int(
            measured_preflight.get(
                "qualified_maximum_attention_key_chunk_size"
            )
            or 0
        )
        current_execution_policy = resolved_execution_policy(candidate, contract)
        current_query_chunk_size = int(
            current_execution_policy.get("attention_query_chunk_size") or 0
        )
        current_key_chunk_size = int(
            current_execution_policy.get("attention_key_chunk_size") or 0
        )
        command_marker = str(
            measured_preflight.get("command_marker") or candidate_id
        )
        if (
            receipt.get("passed") is not True
            or receipt.get("fault") not in (None, "")
            or command_marker not in command
            or float(receipt.get("maximum_inferred_unified_memory_mib") or 0)
            <= 0.0
            or qualified_sequence_tokens <= 0
            or current_sequence_tokens <= 0
            or current_sequence_tokens > qualified_sequence_tokens
            or qualified_position_chunk_size <= 0
            or current_position_chunk_size <= 0
            or current_position_chunk_size > qualified_position_chunk_size
            or qualified_query_chunk_size <= 0
            or current_query_chunk_size <= 0
            or current_query_chunk_size > qualified_query_chunk_size
            or qualified_key_chunk_size <= 0
            or current_key_chunk_size <= 0
            or current_key_chunk_size > qualified_key_chunk_size
        ):
            raise CandidateCanaryFault(
                f"candidate_measured_preflight_invalid:{candidate_id}"
            )
        live_reserve_mib = float(mapping["minimum_available_during_run_mib"])
        measured_peak_mib = float(receipt["maximum_inferred_unified_memory_mib"])
        configured_launch_reserve_mib = max(
            float(mapping["minimum_available_before_launch_mib"]),
            live_reserve_mib,
        )
        mapping["minimum_available_before_launch_mib"] = (
            configured_launch_reserve_mib
        )
        mapping["measured_launch_preflight"] = {
            "path": relative(receipt_path),
            "sha256": expected_sha256,
            "command_marker": command_marker,
            "maximum_inferred_unified_memory_mib": measured_peak_mib,
            "qualified_maximum_training_sequence_tokens": (
                qualified_sequence_tokens
            ),
            "current_maximum_training_sequence_tokens": current_sequence_tokens,
            "qualified_maximum_token_loss_position_chunk_size": (
                qualified_position_chunk_size
            ),
            "current_token_loss_position_chunk_size": (
                current_position_chunk_size
            ),
            "qualified_maximum_attention_query_chunk_size": (
                qualified_query_chunk_size
            ),
            "current_attention_query_chunk_size": current_query_chunk_size,
            "qualified_maximum_attention_key_chunk_size": (
                qualified_key_chunk_size
            ),
            "current_attention_key_chunk_size": current_key_chunk_size,
            "required_live_reserve_mib": live_reserve_mib,
            "resolved_minimum_available_before_launch_mib": (
                configured_launch_reserve_mib
            ),
            "advisory_projected_available_mib": (
                round(measured_peak_mib + live_reserve_mib, 3)
            ),
            "measured_peak_role": (
                "advisory_capacity_projection_not_launch_gate"
            ),
            "launch_gate": "configured_live_reserve_only",
            "runtime_enforcement": (
                "external_live_reserve_and_swap_growth_watchdog"
            ),
        }
    host_resource_safety.policy_from_mapping(
        mapping,
        maximum_wall_seconds=float(candidate["max_wall_seconds"]),
    )
    return mapping


def candidate_lease(
    *,
    candidate_id: str,
    max_steps: int,
    scratch_checkpoint_root: str | Path,
    targets: list[str],
    phase: str,
    resume: bool,
    selected_seed: int | None = None,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = contract or load_contract()
    candidate = next((row for row in contract["canaries"] if row["candidate_id"] == candidate_id), None)
    faults: list[str] = []
    if candidate is None:
        faults.append("candidate_unknown")
        candidate = {}
    scratch = resolve(scratch_checkpoint_root) if str(scratch_checkpoint_root) else Path()
    expected_root = resolve(contract["scratch_root"]) / candidate_id
    if not str(scratch_checkpoint_root):
        faults.append("scratch_namespace_missing")
    else:
        try:
            scratch.resolve().relative_to(expected_root.resolve())
        except ValueError:
            faults.append("scratch_namespace_outside_candidate")
    execution_policy = resolved_execution_policy(candidate, contract)
    segmented_resume_authorized = (
        resume
        and int(execution_policy.get("fresh_process_step_segment") or 0) == 1
        and execution_policy.get("candidate_scratch_resume_policy")
        == "exact_fresh_process_segment_v1"
    )
    if resume and not segmented_resume_authorized:
        faults.append("candidate_resume_forbidden")
    if max_steps <= 0 or max_steps > int(candidate.get("max_steps") or 0):
        faults.append("step_budget_exceeded")
    if not targets or not set(targets).issubset(set(candidate.get("allowed_targets") or [])):
        faults.append("target_not_authorized")
    if phase not in set(candidate.get("allowed_phases") or []):
        faults.append("phase_not_authorized")
    allowed_seeds = [int(value) for value in candidate.get("seeds") or []]
    if selected_seed is not None and int(selected_seed) not in allowed_seeds:
        faults.append("seed_not_authorized")
    for control_root in contract["immutable_control_roots"]:
        try:
            scratch.resolve().relative_to(resolve(control_root).resolve())
            faults.append("immutable_control_namespace_overlap")
        except ValueError:
            pass
    resolved_host_safety_policy = (
        candidate_host_safety_mapping(candidate_id, contract)
        if candidate
        else dict(contract["host_safety_policy"])
    )
    external_watchdog = execution_policy.get("require_external_watchdog") is True
    peak_fraction = float(candidate.get("max_peak_memory_fraction_of_physical") or 0.0)
    if external_watchdog:
        live_reserve_mib = float(
            resolved_host_safety_policy["minimum_available_during_run_mib"]
        )
        resolved_peak_memory_mib = max(
            1.0,
            host_resource_safety.physical_memory_mib() - live_reserve_mib,
        )
        peak_memory_budget_basis = "physical_memory_minus_live_reserve"
    elif peak_fraction:
        if not 0.0 < peak_fraction <= 0.5:
            faults.append("peak_memory_fraction_invalid")
        resolved_peak_memory_mib = min(
            float(candidate.get("max_peak_memory_mib") or 0),
            host_resource_safety.physical_memory_mib() * peak_fraction,
        )
        peak_memory_budget_basis = "declared_fraction_of_physical_memory"
    else:
        resolved_peak_memory_mib = min(
            float(candidate.get("max_peak_memory_mib") or 0),
            host_resource_safety.policy_from_mapping(
                contract["host_safety_policy"],
                maximum_wall_seconds=float(
                    candidate.get("max_wall_seconds") or 1
                ),
            ).max_process_memory_mib,
        )
        peak_memory_budget_basis = "declared_peak_capped_by_process_watchdog"
    lease_payload = {
        "policy": "project_theseus_candidate_specific_canary_lease_v1",
        "contract_digest": digest(contract),
        "candidate_id": candidate_id,
        "implementation_ids": candidate.get("implementation_ids") or [],
        "scratch_checkpoint_root": relative(scratch) if str(scratch_checkpoint_root) else "",
        "targets": sorted(set(targets)),
        "phase": phase,
        "requested_steps": max_steps,
        "budgets": {
            **{
                key: candidate.get(key)
                for key in (
                    "max_steps",
                    "max_positions",
                    "max_wall_seconds",
                    "max_disk_mib",
                )
            },
            "max_peak_memory_mib": resolved_peak_memory_mib,
        },
        "declared_max_peak_memory_mib": candidate.get("max_peak_memory_mib"),
        "max_peak_memory_fraction_of_physical": peak_fraction,
        "peak_memory_budget_basis": peak_memory_budget_basis,
        "host_safety_policy": resolved_host_safety_policy,
        "seeds": allowed_seeds,
        "selected_seed": int(selected_seed) if selected_seed is not None else None,
        "seed_execution_mode": (
            "single_bound_seed" if selected_seed is not None else "aggregate_all_declared_seeds"
        ),
        "heldout_contract": candidate.get("heldout_contract"),
        "behavior_eval_rows": int(candidate.get("behavior_eval_rows") or 0),
        "execution_policy": execution_policy,
        "required_checks": candidate.get("required_checks") or [],
        "hard_boundaries": contract["hard_boundaries"],
        "faults": sorted(set(faults)),
    }
    lease_payload["lease_digest"] = digest(lease_payload)
    lease_payload["authorized"] = not faults
    return lease_payload


def validate_lease(lease: dict[str, Any], contract: dict[str, Any] | None = None) -> bool:
    contract = contract or load_contract()
    expected = candidate_lease(
        candidate_id=str(lease.get("candidate_id") or ""),
        max_steps=int(lease.get("requested_steps") or 0),
        scratch_checkpoint_root=str(lease.get("scratch_checkpoint_root") or ""),
        targets=[str(value) for value in lease.get("targets") or []],
        phase=str(lease.get("phase") or ""),
        resume=False,
        selected_seed=(
            int(lease["selected_seed"])
            if lease.get("selected_seed") is not None
            else None
        ),
        contract=contract,
    )
    return lease == expected and lease.get("authorized") is True


class CandidateCanaryMonitor:
    """Fail closed on the host-resource portion of a candidate lease."""

    def __init__(self, lease: dict[str, Any]) -> None:
        if lease.get("authorized") is not True:
            raise CandidateCanaryFault("candidate_lease_not_authorized")
        self.lease = lease
        self.root = resolve(str(lease["scratch_checkpoint_root"]))
        self.started = time.perf_counter()
        self.maximum_rss_mib = 0.0
        self.maximum_mlx_active_mib = 0.0
        self.maximum_mlx_cache_mib = 0.0
        self.maximum_mlx_peak_mib = 0.0
        self.maximum_disk_mib = 0.0
        self.maximum_step = 0
        self.observations: list[dict[str, Any]] = []
        target_suffix = "-".join(str(value) for value in lease.get("targets") or [])
        self.station_log_path = self.root / (
            f"resource_station_observations.{target_suffix}.jsonl"
        )
        self.station_log_path.unlink(missing_ok=True)

    def _rss_mib(self) -> float:
        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return raw / (1024.0 * 1024.0) if sys.platform == "darwin" else raw / 1024.0

    def _disk_mib(self) -> float:
        if not self.root.exists():
            return 0.0
        total = sum(
            path.stat().st_size
            for path in self.root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        return total / (1024.0 * 1024.0)

    def _mlx_memory_mib(self) -> dict[str, float]:
        # Importing MLX is itself a native accelerator action and can abort the
        # interpreter in a broken or sandboxed Metal environment. Telemetry may
        # observe an already-loaded runtime, but must never initialize one.
        mx = sys.modules.get("mlx.core")
        if mx is None:
            return {"active": 0.0, "cache": 0.0, "peak": 0.0}
        scale = 1024.0 * 1024.0
        return {
            "active": float(mx.get_active_memory()) / scale,
            "cache": float(mx.get_cache_memory()) / scale,
            "peak": float(mx.get_peak_memory()) / scale,
        }

    def check(self, stage: str, step: int) -> None:
        elapsed = time.perf_counter() - self.started
        self.maximum_step = max(self.maximum_step, int(step))
        self.maximum_rss_mib = max(self.maximum_rss_mib, self._rss_mib())
        mlx_memory = self._mlx_memory_mib()
        observation = {
            "stage": stage,
            "step": int(step),
            "elapsed_seconds": round(elapsed, 6),
            "rss_mib": round(self._rss_mib(), 3),
            "mlx_active_mib": round(mlx_memory["active"], 3),
            "mlx_cache_mib": round(mlx_memory["cache"], 3),
            "mlx_peak_mib": round(mlx_memory["peak"], 3),
        }
        if len(self.observations) < 32:
            self.observations.append(observation)
        self.station_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.station_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(observation, separators=(",", ":")) + "\n")
        self.maximum_mlx_active_mib = max(
            self.maximum_mlx_active_mib, mlx_memory["active"]
        )
        self.maximum_mlx_cache_mib = max(
            self.maximum_mlx_cache_mib, mlx_memory["cache"]
        )
        self.maximum_mlx_peak_mib = max(
            self.maximum_mlx_peak_mib, mlx_memory["peak"]
        )
        self.maximum_disk_mib = max(self.maximum_disk_mib, self._disk_mib())
        budgets = self.lease["budgets"]
        external_memory_guard = bool(
            (self.lease.get("execution_policy") or {}).get(
                "require_external_watchdog", False
            )
        )
        faults = []
        if elapsed > float(budgets["max_wall_seconds"]):
            faults.append("wall_budget_exceeded")
        if (
            not external_memory_guard
            and max(self.maximum_rss_mib, self.maximum_mlx_peak_mib)
            > float(budgets["max_peak_memory_mib"])
        ):
            faults.append("memory_budget_exceeded")
        if self.maximum_disk_mib > float(budgets["max_disk_mib"]):
            faults.append("disk_budget_exceeded")
        if self.maximum_step > int(budgets["max_steps"]):
            faults.append("step_budget_exceeded_runtime")
        if faults:
            raise CandidateCanaryFault(
                f"candidate_canary_budget_fault:{stage}:{','.join(faults)}:"
                f"rss_mib={self.maximum_rss_mib:.3f}:"
                f"mlx_active_mib={self.maximum_mlx_active_mib:.3f}:"
                f"mlx_cache_mib={self.maximum_mlx_cache_mib:.3f}:"
                f"mlx_peak_mib={self.maximum_mlx_peak_mib:.3f}:"
                f"observations={json.dumps(self.observations, separators=(',', ':'))}"
            )

    def finalize(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        self.check("finalize", self.maximum_step)
        positions = sum(int(row.get("optimizer_positions") or 0) for row in results)
        faults = []
        if positions > int(self.lease["budgets"]["max_positions"]):
            faults.append("position_budget_exceeded")
        return {
            "policy": "project_theseus_candidate_canary_resource_receipt_v1",
            "lease_digest": self.lease["lease_digest"],
            "candidate_id": self.lease["candidate_id"],
            "observed_optimizer_positions": positions,
            "observed_maximum_step": self.maximum_step,
            "wall_seconds": round(time.perf_counter() - self.started, 6),
            "peak_rss_mib": round(self.maximum_rss_mib, 3),
            "peak_mlx_active_mib": round(self.maximum_mlx_active_mib, 3),
            "peak_mlx_cache_mib": round(self.maximum_mlx_cache_mib, 3),
            "peak_mlx_allocator_mib": round(self.maximum_mlx_peak_mib, 3),
            "memory_budget_measurement": "max(host_rss,mlx_allocator_peak)",
            "memory_budget_enforcement": (
                "external_host_live_reserve_and_swap_watchdog"
                if (self.lease.get("execution_policy") or {}).get(
                    "require_external_watchdog", False
                )
                else "internal_peak_budget"
            ),
            "declared_peak_memory_mib": float(
                self.lease["budgets"]["max_peak_memory_mib"]
            ),
            "declared_peak_memory_exceeded": max(
                self.maximum_rss_mib, self.maximum_mlx_peak_mib
            )
            > float(self.lease["budgets"]["max_peak_memory_mib"]),
            "observation_prefix": self.observations,
            "peak_scratch_disk_mib": round(self.maximum_disk_mib, 3),
            "faults": faults,
            "passed": not faults,
        }
