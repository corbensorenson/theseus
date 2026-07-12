#!/usr/bin/env python3
"""Train a shared MoECOT trunk, language experts, and matched dense controls.

The runtime consumes the immutable canonical stage produced by the standard
transformer corpus path. It does not build another corpus, route answers, or
turn training loss into a capability claim.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from standard_causal_transformer_model import CausalTransformerConfig, build_model, parameter_count
from standard_causal_transformer_corpus import load_pretrain_memmaps, pretrain_array_paths
from standard_causal_transformer_survival import (
    GLOBAL_BOS_ID,
    SOURCE_TARGET_SEPARATOR_ID,
    build_schedule,
    causal_loss,
    model_vocab_size,
    required_steps,
    source_token_offset,
    target_token_offset,
    train_phase,
)
from moecot_language_tokenizer import exact_text_tokens
from neural_seed_open_vocab import (
    TARGET_BYTE_BEGIN,
    TARGET_BYTE_END,
    active_target_span,
    decode_target_tokens,
    encode_tokens,
    is_byte_token,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "moecot_language_arm_training.json"
ARM_IDS = ("english", "python", "javascript_typescript", "html_css", "rust")
SHARED_TRUNK_ID = "shared_trunk"
CONTROL_IDS = ("dense_total_parameter", "dense_active_parameter")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--target",
        action="append",
        choices=[SHARED_TRUNK_ID, *ARM_IDS, *CONTROL_IDS],
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if args.execute and not args.target:
        parser.error("--execute requires at least one explicit --target")
    if args.max_steps < 0:
        parser.error("--max-steps cannot be negative")

    config_path = resolve(args.config)
    config = read_json(config_path)
    plan = build_plan(config, config_path=config_path)
    if plan["trigger_state"] == "RED":
        write_json(resolve(args.out or config["report"]), plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 2
    report = plan
    if args.execute:
        report = execute_targets(
            config,
            plan,
            targets=list(dict.fromkeys(args.target or [])),
            max_steps=args.max_steps,
            resume=args.resume,
        )
    write_json(resolve(args.out or config["report"]), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_plan(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    gaps: list[str] = []
    validate_config(config)
    base_path = resolve(str(config["base_config"]))
    base = read_json(base_path)
    stage_dir = resolve(str(config["stage_dir"]))
    metadata_path = stage_dir / "stage_metadata_v1.json"
    if not metadata_path.is_file():
        gaps.append("canonical_stage_metadata_missing")
        metadata: dict[str, Any] = {}
    else:
        metadata = read_json(metadata_path)
    summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
    canonical = (
        summary.get("canonical_pretrain_stage")
        if isinstance(summary.get("canonical_pretrain_stage"), dict)
        else {}
    )
    arm_views = canonical.get("arm_views") if isinstance(canonical.get("arm_views"), dict) else {}
    range_audit = audit_arm_views(arm_views, int(canonical.get("window_count") or 0))
    gaps.extend(range_audit["hard_gaps"])
    tokenizer_audit = audit_tokenizer_stage(base, canonical)
    gaps.extend(tokenizer_audit["hard_gaps"])
    supervision_audit = audit_supervision_stage(config, config_path=config_path)
    gaps.extend(supervision_audit["hard_gaps"])
    source_conditioned_audit = audit_source_conditioned_stage(config)
    gaps.extend(source_conditioned_audit["hard_gaps"])
    stage_arrays = canonical.get("array_artifacts") if isinstance(canonical.get("array_artifacts"), dict) else {}
    for key, row in stage_arrays.items():
        path = resolve(str(row.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(row.get("sha256") or ""):
            gaps.append(f"canonical_stage_array_identity_mismatch:{key}")

    models: dict[str, Any] = {}
    if metadata:
        models = model_accounting(config, base, metadata)
        dense_total = int(models["dense_total_parameter"]["parameter_count"])
        arm_total = int(models["moecot_system"]["total_parameter_count"])
        delta = abs(arm_total - dense_total) / max(1, dense_total)
        models["moecot_system"]["total_parameter_delta_vs_dense_total"] = round(delta, 8)
        if delta > 0.10:
            gaps.append("moecot_total_parameters_outside_preregistered_tolerance")
        active_reference = int(models["moecot_system"]["active_parameter_count_per_request"])
        active_delta = abs(
            int(models["dense_active_parameter"]["parameter_count"])
            - active_reference
        ) / max(1, active_reference)
        models["dense_active_parameter"]["parameter_delta_fraction"] = round(
            active_delta, 8
        )
        if active_delta > 0.01:
            gaps.append("active_parameter_control_mismatch")
    plan_identity = plan_sha256(
        config, metadata, models, supervision_audit, source_conditioned_audit
    )
    targets = target_contracts(
        config,
        arm_views,
        models,
        plan_identity,
        supervision_audit=supervision_audit,
        source_conditioned_audit=source_conditioned_audit,
    )
    specialist_scaling = audit_specialist_data_scaling(
        base,
        targets,
        models,
    )
    gaps.extend(specialist_scaling["hard_gaps"])
    checkpoint_inventory = inspect_checkpoint_inventory(targets, plan_identity, summary.get("stage_signature"))
    gaps.extend(checkpoint_inventory["hard_gaps"])
    return {
        "policy": "project_theseus_moecot_language_arm_training_plan_v1",
        "created_utc": now(),
        "trigger_state": "RED" if gaps else "GREEN",
        "mode": "preregistered_plan",
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "base_config": relative(base_path),
        "base_config_sha256": sha256_file(base_path),
        "stage": {
            "path": relative(stage_dir),
            "metadata": relative(metadata_path),
            "metadata_sha256": sha256_file(metadata_path) if metadata_path.is_file() else "",
            "stage_signature": summary.get("stage_signature"),
            "array_artifacts": stage_arrays,
            "arm_view_policy": arm_views.get("policy"),
            "range_audit": range_audit,
            "tokenizer_audit": tokenizer_audit,
        },
        "models": models,
        "supervision": supervision_audit,
        "source_conditioned_pretraining": source_conditioned_audit,
        "targets": targets,
        "specialist_data_scaling": specialist_scaling,
        "checkpoint_inventory": checkpoint_inventory,
        "comparison_contract": config["comparison_contract"],
        "plan_sha256": plan_identity,
        "hard_gaps": sorted(set(gaps)),
        "non_claims": [
            "plan and checkpoint smoke are not learned capability",
            "training loss is not direct answer utility",
            "routing success is not answer success",
            "neither accounting view may be selected after results are known",
        ],
        **no_cheat(config),
    }


def audit_specialist_data_scaling(
    base: dict[str, Any],
    targets: dict[str, Any],
    models: dict[str, Any],
) -> dict[str, Any]:
    """Bind every trained parameter owner to enough unique model-visible data."""

    ratio = float(
        ((base.get("data_model_scaling_contract") or {}).get("planning_basis") or {}).get(
            "minimum_unique_positions_per_active_parameter"
        )
        or 0.0
    )
    expert_parameters = int(
        ((models.get("moecot_system") or {}).get("expert_parameter_count_per_arm"))
        or 0
    )
    trunk_parameters = int(
        ((models.get("moecot_system") or {}).get("shared_trunk_parameter_count"))
        or 0
    )
    rows: list[dict[str, Any]] = []
    gaps: list[str] = []
    for target_id in (SHARED_TRUNK_ID, *ARM_IDS):
        parameters = trunk_parameters if target_id == SHARED_TRUNK_ID else expert_parameters
        positions = int((targets.get(target_id) or {}).get("unique_target_positions") or 0)
        required = int(np.ceil(parameters * ratio)) if parameters and ratio else 0
        row = {
            "target_id": target_id,
            "owned_parameter_count": parameters,
            "unique_model_visible_positions": positions,
            "minimum_required_positions": required,
            "positions_per_owned_parameter": round(positions / max(1, parameters), 6),
            "meets_floor": bool(parameters > 0 and positions >= required),
        }
        if not row["meets_floor"]:
            gaps.append(f"specialist_unique_position_floor_not_met:{target_id}")
        rows.append(row)
    return {
        "policy": "project_theseus_moecot_specialist_data_scaling_v1",
        "minimum_unique_positions_per_owned_parameter": ratio,
        "state": "GREEN" if not gaps else "RED",
        "rows": rows,
        "hard_gaps": gaps,
        "optimizer_repetition_counted_as_unique_data": False,
        "capability_credit": "NONE",
    }


def inspect_checkpoint_inventory(
    targets: dict[str, Any], plan_identity: str, stage_signature: Any
) -> dict[str, Any]:
    rows = []
    gaps = []
    checkpoint_hashes: set[str] = set()
    optimizer_hashes: set[str] = set()
    for target_id, target in targets.items():
        receipt_path = resolve(str(target["receipt"]))
        if not receipt_path.is_file():
            rows.append({"target_id": target_id, "state": "NOT_RUN"})
            continue
        receipt = read_json(receipt_path)
        checkpoint = resolve(str(receipt.get("checkpoint") or target["checkpoint"]))
        optimizer = resolve(
            str(receipt.get("optimizer_state") or target["optimizer_state"])
        )
        faults = []
        try:
            validate_resume(
                receipt,
                {"plan_sha256": plan_identity, "stage": {"stage_signature": stage_signature}},
                target,
                checkpoint,
                optimizer,
            )
        except ValueError as exc:
            faults.append(str(exc))
        checkpoint_hash = str(receipt.get("checkpoint_sha256") or "")
        optimizer_hash = str(receipt.get("optimizer_state_sha256") or "")
        if checkpoint_hash in checkpoint_hashes:
            faults.append("checkpoint_digest_not_distinct")
        if optimizer_hash in optimizer_hashes:
            faults.append("optimizer_digest_not_distinct")
        checkpoint_hashes.add(checkpoint_hash)
        optimizer_hashes.add(optimizer_hash)
        if faults:
            gaps.extend(f"checkpoint_inventory:{target_id}:{fault}" for fault in faults)
        rows.append(
            {
                "target_id": target_id,
                "state": "GREEN" if not faults else "RED",
                "optimizer_steps": int(receipt.get("optimizer_steps") or 0),
                "optimizer_positions": int(receipt.get("optimizer_positions") or 0),
                "complete": bool(receipt.get("complete")),
                "checkpoint_sha256": checkpoint_hash,
                "optimizer_state_sha256": optimizer_hash,
                "capability_claim": receipt.get("capability_claim"),
                "faults": faults,
            }
        )
    completed_smokes = sum(
        row.get("state") == "GREEN" and int(row.get("optimizer_steps") or 0) > 0 for row in rows
    )
    return {
        "state": "GREEN" if completed_smokes == len(targets) and not gaps else (
            "RED" if gaps else "NOT_RUN"
        ),
        "target_count": len(targets),
        "valid_smoke_count": completed_smokes,
        "distinct_checkpoint_digest_count": len(checkpoint_hashes),
        "distinct_optimizer_digest_count": len(optimizer_hashes),
        "all_targets_smoke_ready": completed_smokes == len(targets) and not gaps,
        "rows": rows,
        "hard_gaps": gaps,
        "capability_claim": "NOT_EVALUATED",
    }


def model_accounting(
    config: dict[str, Any], base: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils as mlx_utils

    vocab_size = model_vocab_size(
        base,
        dict(metadata.get("source_vocab") or {}),
        dict(metadata.get("target_vocab") or {}),
    )
    copy_lookup = build_source_to_target_lookup(base, metadata)

    def instantiate(model_config: dict[str, Any]) -> Any:
        return build_model(
            CausalTransformerConfig(vocab_size=vocab_size, **model_config),
            mx=mx,
            nn=nn,
            state_role_lookup=None,
            source_to_target_lookup=copy_lookup,
        )

    def count(model_config: dict[str, Any]) -> int:
        return int(parameter_count(instantiate(model_config), mlx_utils))

    trunk_count = count(config["shared_trunk_model"])
    arm = instantiate(config["arm_model"])
    arm_count = int(parameter_count(arm, mlx_utils))
    expert_scope = str(config["topology"]["expert_trainable_scope"])
    arm.freeze_to_language_expert(expert_scope)
    expert_count = int(
        sum(
            value.size
            for _name, value in mlx_utils.tree_flatten(arm.trainable_parameters())
        )
    )
    if expert_count <= 0:
        raise ValueError("language expert must add parameters to the shared trunk")
    system_total = trunk_count + expert_count * len(ARM_IDS)
    dense_active_model, dense_active_count = matched_decoder_only_config(
        arm_count, config["arm_model"], count=count
    )
    dense_total_model, dense_total_count = matched_decoder_only_config(
        system_total, base["model"], count=count
    )
    return {
        "moecot_system": {
            "topology": config["topology"],
            "shared_trunk_model": config["shared_trunk_model"],
            "shared_trunk_parameter_count": trunk_count,
            "arm_model": config["arm_model"],
            "arm_parameter_count": arm_count,
            "expert_parameter_count_per_arm": expert_count,
            "expert_trainable_scope": expert_scope,
            "arm_count": len(ARM_IDS),
            "total_parameter_count": system_total,
            "active_parameter_count_per_request": arm_count,
            "router_parameter_count": 0,
            "router_accounting_state": "EXCLUDED_UNTIL_LANGUAGE_ROUTER_IS_TRAINED",
        },
        "dense_total_parameter": {
            "model": dense_total_model,
            "parameter_count": dense_total_count,
            "active_parameter_count_per_request": dense_total_count,
            "parameter_delta_vs_moecot_total": dense_total_count
            - system_total,
            "architecture": "decoder_only_prefix_lm_control",
        },
        "dense_active_parameter": {
            "model": dense_active_model,
            "parameter_count": dense_active_count,
            "active_parameter_count_per_request": dense_active_count,
            "parameter_delta_vs_active_arm": dense_active_count - arm_count,
            "architecture": "decoder_only_prefix_lm_control",
        },
        "vocab_size": vocab_size,
    }


def matched_decoder_only_config(
    reference_parameters: int,
    seed: dict[str, Any],
    *,
    count: Any,
) -> tuple[dict[str, Any], int]:
    """Mechanically width-match a prefix-LM control without copying the encoder."""

    candidate = dict(seed)
    candidate["attention_policy"] = "prefix_lm"
    candidate.pop("source_encoder_layers", None)
    candidate.pop("source_copy_mode", None)
    candidate.pop("source_copy_auxiliary_loss_weight", None)
    candidate.pop("expert_adapter_dim", None)
    candidate.pop("source_expert_adapter_dim", None)
    candidate["ff_dim"] = 1
    low_count = int(count(candidate))
    candidate["ff_dim"] = 2
    slope = int(count(candidate)) - low_count
    if slope <= 0:
        raise ValueError("decoder-only parameter matching requires positive FF slope")
    estimated = max(1, round(1 + (reference_parameters - low_count) / slope))
    choices: list[tuple[int, int, dict[str, Any]]] = []
    for width in range(max(1, estimated - 3), estimated + 4):
        model = {**candidate, "ff_dim": width}
        observed = int(count(model))
        choices.append((abs(observed - reference_parameters), observed, model))
    _delta, observed, selected = min(choices, key=lambda row: (row[0], row[1]))
    return selected, observed


def build_source_to_target_lookup(
    base: dict[str, Any], metadata: dict[str, Any]
) -> np.ndarray:
    """Map exact source token identities into target IDs for learned copying."""

    source_vocab = dict(metadata.get("source_vocab") or {})
    target_vocab = dict(metadata.get("target_vocab") or {})
    vocab_size = model_vocab_size(base, source_vocab, target_vocab)
    lookup = np.full(vocab_size, -1, dtype=np.int32)
    source_offset = source_token_offset(base, source_vocab)
    target_offset = target_token_offset(base, source_vocab)
    for token, source_id in source_vocab.items():
        target_id = target_vocab.get(token)
        if target_id is not None:
            lookup[source_offset + int(source_id)] = target_offset + int(target_id)
    return lookup


def target_contracts(
    config: dict[str, Any],
    arm_views: dict[str, Any],
    models: dict[str, Any],
    plan_identity: str,
    *,
    supervision_audit: dict[str, Any],
    source_conditioned_audit: dict[str, Any],
) -> dict[str, Any]:
    root = resolve(str(config["checkpoint_root"]))
    targets: dict[str, Any] = {}
    for target in (SHARED_TRUNK_ID, *ARM_IDS, *CONTROL_IDS):
        if target == SHARED_TRUNK_ID:
            view = arm_views.get("mixed_dense_control") or {}
            model_key = "moecot_system"
            model = (models.get(model_key) or {}).get("shared_trunk_model") or config[
                "shared_trunk_model"
            ]
            parameter_count_value = int(
                (models.get(model_key) or {}).get("shared_trunk_parameter_count") or 0
            )
            role = "shared_trunk"
        elif target in ARM_IDS:
            view = (arm_views.get("arms") or {}).get(target) or {}
            model_key = "moecot_system"
            model = (models.get(model_key) or {}).get("arm_model") or config["arm_model"]
            parameter_count_value = int((models.get(model_key) or {}).get("arm_parameter_count") or 0)
            role = "language_expert"
        else:
            view = arm_views.get("mixed_dense_control") or {}
            model = (models.get(target) or {}).get("model") or {}
            parameter_count_value = int((models.get(target) or {}).get("parameter_count") or 0)
            role = "dense_control"
        directory = root / target
        targets[target] = {
            "target_id": target,
            "role": role,
            "expert_trainable_scope": (
                str(config["topology"]["expert_trainable_scope"])
                if target in ARM_IDS
                else ""
            ),
            "row_ranges": list(view.get("row_ranges") or []),
            "row_count": sum(int(row["stop"]) - int(row["start"]) for row in view.get("row_ranges") or []),
            "unique_target_positions": int(view.get("target_positions") or 0),
            "model": model,
            "parameter_count": parameter_count_value,
            "estimated_parameter_token_product": parameter_count_value
            * int(view.get("target_positions") or 0),
            "checkpoint": relative(
                directory
                / ("expert_delta.safetensors" if target in ARM_IDS else "weights.npz")
            ),
            "shared_trunk_checkpoint": (
                relative(root / SHARED_TRUNK_ID / "weights.npz")
                if target in ARM_IDS
                else ""
            ),
            "optimizer_state": relative(directory / "optimizer.safetensors"),
            "receipt": relative(directory / "training_receipt.json"),
            "plan_sha256": plan_identity,
            "supervision_artifacts": (
                {
                    split: supervision_audit["artifacts"].get(f"{target}:{split}")
                    for split in ("private_train", "private_dev", "private_eval")
                }
                if target in ARM_IDS
                else {
                    f"{arm}:{split}": supervision_audit["artifacts"].get(f"{arm}:{split}")
                    for arm in ARM_IDS
                    for split in ("private_train", "private_dev", "private_eval")
                }
            ),
            "source_conditioned_artifacts": (
                {
                    "private_train": source_conditioned_audit["artifacts"].get(target)
                }
                if target in ARM_IDS
                and source_conditioned_audit["artifacts"].get(target)
                else {
                    f"{arm}:private_train": source_conditioned_audit["artifacts"].get(arm)
                    for arm in ARM_IDS
                    if source_conditioned_audit["artifacts"].get(arm)
                }
                if target not in ARM_IDS
                else {}
            ),
        }
    return targets


def audit_arm_views(arm_views: dict[str, Any], window_count: int) -> dict[str, Any]:
    gaps: list[str] = []
    arms = arm_views.get("arms") if isinstance(arm_views.get("arms"), dict) else {}
    if tuple(arms) != ARM_IDS:
        gaps.append("canonical_arm_set_or_order_mismatch")
    occupied: list[tuple[int, int, str]] = []
    for arm_id in ARM_IDS:
        view = arms.get(arm_id) if isinstance(arms.get(arm_id), dict) else {}
        if view.get("independent_weights_required") is not True:
            gaps.append(f"independent_weights_not_required:{arm_id}")
        for row in view.get("row_ranges") or []:
            start, stop = int(row.get("start") or 0), int(row.get("stop") or 0)
            if start < 0 or stop <= start or stop > window_count:
                gaps.append(f"invalid_row_range:{arm_id}")
            occupied.append((start, stop, arm_id))
    occupied.sort()
    cursor = 0
    for start, stop, arm_id in occupied:
        if start != cursor:
            gaps.append(f"arm_range_gap_or_overlap:{arm_id}:{cursor}:{start}")
        cursor = max(cursor, stop)
    if cursor != window_count:
        gaps.append("arm_ranges_do_not_cover_stage")
    control = arm_views.get("mixed_dense_control") or {}
    if control.get("row_ranges") != [{"start": 0, "stop": window_count}]:
        gaps.append("dense_control_not_exact_full_stage")
    if arm_views.get("hidden_generalist_fallback") != "forbidden":
        gaps.append("hidden_generalist_fallback_not_forbidden")
    return {
        "state": "GREEN" if not gaps else "RED",
        "hard_gaps": gaps,
        "window_count": window_count,
        "covered_rows": cursor,
        "non_overlapping_complete_partition": not gaps,
    }


def audit_tokenizer_stage(base: dict[str, Any], canonical: dict[str, Any]) -> dict[str, Any]:
    expected = (base.get("tokenization") or {}).get("canonical_language_profiles") or {}
    observed = canonical.get("tokenizer_audit") if isinstance(canonical.get("tokenizer_audit"), dict) else {}
    category_profiles = (
        observed.get("category_profiles_by_selected_document")
        if isinstance(observed.get("category_profiles_by_selected_document"), dict)
        else {}
    )
    gaps: list[str] = []
    if expected.get("policy") != "project_theseus_moecot_language_tokenizer_v1":
        gaps.append("canonical_language_tokenizer_policy_missing")
    for category in (
        "english_conversation_instruction",
        "english_broad",
        "python",
        "javascript_typescript",
        "html_css",
        "rust",
    ):
        profile = str(expected.get(category) or "")
        if not profile:
            gaps.append(f"canonical_language_tokenizer_profile_missing:{category}")
        elif int(category_profiles.get(f"{category}:{profile}") or 0) <= 0:
            gaps.append(f"canonical_stage_tokenizer_profile_unproven:{category}:{profile}")
    if int(observed.get("roundtrip_failure_count") or 0):
        gaps.append("canonical_stage_tokenizer_roundtrip_failure")
    if int(observed.get("admitted_unknown_token_position_count") or 0):
        gaps.append("canonical_stage_admitted_unknown_token_position")
    return {
        "state": "GREEN" if not gaps else "RED",
        "policy": expected.get("policy"),
        "expected_profiles": {
            category: expected.get(category)
            for category in (
                "english_conversation_instruction",
                "english_broad",
                "python",
                "javascript_typescript",
                "html_css",
                "rust",
            )
        },
        "observed": observed,
        "hard_gaps": gaps,
        "failure_behavior": "deny_training_until_stage_is_rebuilt",
    }


def audit_supervision_stage(
    config: dict[str, Any], *, config_path: Path
) -> dict[str, Any]:
    cfg = config.get("supervision") if isinstance(config.get("supervision"), dict) else {}
    root = resolve(str(cfg.get("stage_root") or ""))
    manifest_path = root / "manifest.json"
    gaps: list[str] = []
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    if not manifest:
        gaps.append("moecot_supervision_manifest_missing")
    if manifest.get("policy") != "project_theseus_moecot_language_supervision_v1":
        gaps.append("moecot_supervision_manifest_policy_mismatch")
    if manifest.get("trigger_state") != "GREEN":
        gaps.append("moecot_supervision_manifest_not_green")
    expected_supervision_contract = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest.get("contract_sha256") != expected_supervision_contract:
        gaps.append("moecot_supervision_contract_identity_mismatch")
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    for arm in ARM_IDS:
        for split, wanted in (
            ("private_train", int((cfg.get("train_rows_by_arm") or {}).get(arm) or 0)),
            ("private_dev", int((cfg.get("development_rows_by_arm") or {}).get(arm) or 0)),
            ("private_eval", int((cfg.get("heldout_rows_by_arm") or {}).get(arm) or 0)),
        ):
            key = f"{arm}:{split}"
            row = artifacts.get(key) if isinstance(artifacts.get(key), dict) else {}
            path = resolve(str(row.get("path") or ""))
            if not path.is_file() or sha256_file(path) != str(row.get("sha256") or ""):
                gaps.append(f"moecot_supervision_artifact_identity_mismatch:{key}")
            if int(row.get("row_count") or 0) != wanted:
                gaps.append(f"moecot_supervision_row_count_mismatch:{key}")
    overlap = manifest.get("split_overlap_audit") if isinstance(manifest.get("split_overlap_audit"), dict) else {}
    if int(overlap.get("prompt_overlap_count") or 0):
        gaps.append("moecot_supervision_prompt_overlap")
    if int(overlap.get("target_overlap_count") or 0):
        gaps.append("moecot_supervision_target_overlap")
    for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count"):
        if int(manifest.get(key) or 0):
            gaps.append(f"moecot_supervision_nonzero_boundary:{key}")
    return {
        "state": "GREEN" if not gaps else "RED",
        "manifest": relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else "",
        "artifacts": artifacts,
        "row_counts": manifest.get("row_counts") or {},
        "split_overlap_audit": overlap,
        "source_receipts": manifest.get("source_receipts") or [],
        "hard_gaps": gaps,
        "score_semantics": "frozen supervision provenance and split integrity only",
    }


def audit_source_conditioned_stage(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("source_conditioned_pretraining")
    cfg = cfg if isinstance(cfg, dict) else {}
    root = resolve(str(cfg.get("stage_root") or ""))
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    gaps: list[str] = []
    if manifest.get("policy") != "project_theseus_moecot_source_conditioned_pretraining_v1":
        gaps.append("source_conditioned_manifest_policy_mismatch")
    if manifest.get("trigger_state") != "GREEN":
        gaps.append("source_conditioned_manifest_not_green")
    expected_contract = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest.get("contract_sha256") != expected_contract:
        gaps.append("source_conditioned_contract_identity_mismatch")
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    for arm, wanted in (cfg.get("rows_by_arm") or {}).items():
        if int(wanted) <= 0:
            continue
        row = artifacts.get(arm) if isinstance(artifacts.get(arm), dict) else {}
        path = resolve(str(row.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(row.get("sha256") or ""):
            gaps.append(f"source_conditioned_artifact_identity_mismatch:{arm}")
        if int(row.get("row_count") or 0) != int(wanted):
            gaps.append(f"source_conditioned_row_count_mismatch:{arm}")
    for key in (
        "public_training_rows_written",
        "public_benchmark_payload_count",
        "external_inference_calls",
        "fallback_return_count",
    ):
        if int(manifest.get(key) or 0):
            gaps.append(f"source_conditioned_nonzero_boundary:{key}")
    return {
        "state": "GREEN" if not gaps else "RED",
        "manifest": relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else "",
        "artifacts": artifacts,
        "copy_coverage_by_arm": manifest.get("copy_coverage_by_arm") or {},
        "corruption": manifest.get("corruption") or {},
        "hard_gaps": gaps,
        "score_semantics": "source-conditioned objective readiness only",
    }


def execute_targets(
    config: dict[str, Any], plan: dict[str, Any], *, targets: list[str], max_steps: int, resume: bool
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    stage_dir = resolve(str(config["stage_dir"]))
    metadata = read_json(stage_dir / "stage_metadata_v1.json")
    base = read_json(resolve(str(config["base_config"])))
    if any(target_id == SHARED_TRUNK_ID or target_id in ARM_IDS for target_id in targets):
        ensure_shared_trunk_migration(
            config,
            plan,
            metadata=metadata,
            base=base,
            mx=mx,
            nn=nn,
            require_existing=any(target_id in ARM_IDS for target_id in targets),
        )
    canonical = metadata["summary"]["canonical_pretrain_stage"]
    shape = (int(canonical["window_count"]), int(canonical["max_sequence_tokens"]))
    arrays = load_pretrain_memmaps(
        pretrain_array_paths(stage_dir),
        shape,
        expected=canonical["array_artifacts"],
    )
    stage = SimpleNamespace(
        pretrain_inputs=arrays[0],
        pretrain_labels=arrays[1],
        pretrain_mask=arrays[2],
    )
    supervision_stages = {
        target_id: materialize_target_supervision(
            config,
            base,
            plan["targets"][target_id],
            metadata=metadata,
        )
        for target_id in targets
    }
    source_conditioned_stages = {
        target_id: materialize_target_supervision(
            config,
            base,
            plan["targets"][target_id],
            metadata=metadata,
            artifact_field="source_conditioned_artifacts",
            receipt_policy="project_theseus_moecot_source_conditioned_arrays_v1",
        )
        if (plan["targets"][target_id].get("source_conditioned_artifacts") or {})
        else None
        for target_id in targets
    }
    results = []
    for target_id in targets:
        target = plan["targets"][target_id]
        result = train_target(
            config,
            plan,
            target,
            stage=stage,
            source_conditioned_stage=source_conditioned_stages[target_id],
            supervision_stage=supervision_stages[target_id],
            max_steps=max_steps,
            resume=resume,
            mx=mx,
            nn=nn,
            optim=optim,
            mlx_utils=mlx_utils,
        )
        if result.get("complete") and should_evaluate_target(target):
            result["evaluation"] = evaluate_target(
                config,
                base,
                plan,
                target,
                metadata=metadata,
                mx=mx,
                nn=nn,
            )
        results.append(result)
    gaps = [
        f"{row['target_id']}:{gap}"
        for row in results
        for gap in row.get("hard_gaps") or []
    ]
    refreshed_inventory = inspect_checkpoint_inventory(
        plan["targets"], plan["plan_sha256"], plan["stage"]["stage_signature"]
    )
    return {
        **plan,
        "checkpoint_inventory": refreshed_inventory,
        "created_utc": now(),
        "trigger_state": "RED" if gaps else "GREEN",
        "mode": "training_execution",
        "executed_targets": targets,
        "results": results,
        "hard_gaps": gaps,
        "all_requested_targets_complete": bool(results)
        and all(row.get("complete") for row in results),
        **no_cheat(config),
    }


def ensure_shared_trunk_migration(
    config: dict[str, Any],
    plan: dict[str, Any],
    *,
    metadata: dict[str, Any],
    base: dict[str, Any],
    mx: Any,
    nn: Any,
    require_existing: bool = True,
) -> dict[str, Any]:
    """Validate, migrate, or authorize fresh initialization for the shared trunk."""

    target = plan["targets"][SHARED_TRUNK_ID]
    checkpoint = resolve(str(target["checkpoint"]))
    optimizer = resolve(str(target["optimizer_state"]))
    receipt_path = resolve(str(target["receipt"]))
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        committed_checkpoint = resolve(str(receipt.get("checkpoint") or checkpoint))
        committed_optimizer = resolve(
            str(receipt.get("optimizer_state") or optimizer)
        )
        validate_resume(
            receipt,
            plan,
            target,
            committed_checkpoint,
            committed_optimizer,
        )
        return receipt
    if any(path.exists() for path in (checkpoint, optimizer, receipt_path)):
        raise ValueError("partial shared trunk migration state requires operator cleanup")

    topology = config["topology"]
    initialization = topology.get("shared_trunk_initialization") or {}
    if initialization.get("policy") == "project_theseus_seeded_fresh_trunk_initialization_v1":
        if int(initialization.get("seed") or -1) != int(config["seed"]):
            raise ValueError("fresh shared trunk seed mismatch")
        if require_existing:
            raise ValueError("language expert requires a completed fresh shared trunk")
        return {
            "policy": initialization["policy"],
            "state": "FRESH_INITIALIZATION_AUTHORIZED",
            "seed": int(config["seed"]),
            "training_positions_added": 0,
            "capability_credit": "NONE",
        }

    bootstrap = topology.get("shared_trunk_bootstrap") or initialization
    if bootstrap.get("policy") != "project_theseus_exact_shared_trunk_migration_v1":
        raise ValueError("unsupported shared trunk initialization policy")
    source_checkpoint = resolve(str(bootstrap["checkpoint"]))
    source_optimizer = resolve(str(bootstrap["optimizer_state"]))
    source_receipt_path = resolve(str(bootstrap["receipt"]))
    for path, expected, label in (
        (source_checkpoint, bootstrap["checkpoint_sha256"], "checkpoint"),
        (source_optimizer, bootstrap["optimizer_state_sha256"], "optimizer"),
        (source_receipt_path, bootstrap["receipt_sha256"], "receipt"),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"shared trunk migration source {label} identity mismatch")
    source_receipt = read_json(source_receipt_path)
    if not bool(source_receipt.get("complete")):
        raise ValueError("shared trunk migration source is incomplete")
    if source_receipt.get("checkpoint_sha256") != bootstrap["checkpoint_sha256"]:
        raise ValueError("shared trunk source receipt checkpoint mismatch")
    if source_receipt.get("optimizer_state_sha256") != bootstrap["optimizer_state_sha256"]:
        raise ValueError("shared trunk source receipt optimizer mismatch")
    if source_receipt.get("stage_signature") != plan["stage"]["stage_signature"]:
        raise ValueError("shared trunk migration stage identity mismatch")

    copy_lookup = build_source_to_target_lookup(base, metadata)
    model = build_model(
        CausalTransformerConfig(
            vocab_size=int(plan["models"]["vocab_size"]), **target["model"]
        ),
        mx=mx,
        nn=nn,
        state_role_lookup=None,
        source_to_target_lookup=copy_lookup,
    )
    model.load_weights(str(source_checkpoint), strict=True)
    mx.eval(model.parameters())

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    atomic_copy(source_checkpoint, checkpoint)
    atomic_copy(source_optimizer, optimizer)
    receipt = {
        **source_receipt,
        "created_utc": now(),
        "target_id": SHARED_TRUNK_ID,
        "role": "shared_trunk",
        "plan_sha256": plan["plan_sha256"],
        "row_ranges": target["row_ranges"],
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "optimizer_state": relative(optimizer),
        "optimizer_state_sha256": sha256_file(optimizer),
        "resume": False,
        "resume_base_checkpoint_sha256": "",
        "migration": {
            "policy": bootstrap["policy"],
            "source_checkpoint": relative(source_checkpoint),
            "source_checkpoint_sha256": bootstrap["checkpoint_sha256"],
            "source_optimizer_state": relative(source_optimizer),
            "source_optimizer_state_sha256": bootstrap["optimizer_state_sha256"],
            "source_receipt": relative(source_receipt_path),
            "source_receipt_sha256": bootstrap["receipt_sha256"],
            "strict_model_load_proved": True,
            "model_config_sha256": hashlib.sha256(
                json.dumps(
                    target["model"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "training_positions_added": 0,
            "capability_credit": "NONE",
        },
        "capability_claim": "NOT_EVALUATED",
        "hard_gaps": [],
    }
    write_json_atomic(receipt_path, receipt)
    validate_resume(receipt, plan, target, checkpoint, optimizer)
    return receipt


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def should_evaluate_target(target: dict[str, Any]) -> bool:
    """Only executable model compositions receive direct behavior evaluation."""

    role = str(target.get("role") or "")
    return role in {"language_expert", "dense_control"}


def materialize_target_supervision(
    config: dict[str, Any],
    base: dict[str, Any],
    target: dict[str, Any],
    *,
    metadata: dict[str, Any],
    artifact_field: str = "supervision_artifacts",
    receipt_policy: str = "project_theseus_moecot_exact_supervision_arrays_v1",
) -> Any:
    """Encode the frozen train split without truncation or hidden-field routing."""

    source_vocab = dict(metadata.get("source_vocab") or {})
    target_vocab = dict(metadata.get("target_vocab") or {})
    if not source_vocab or not target_vocab:
        raise ValueError("canonical stage metadata is missing exact vocabularies")
    source_offset = source_token_offset(base, source_vocab)
    target_offset = target_token_offset(base, source_vocab)
    max_sequence = int((base.get("tokenization") or {}).get("max_sequence_tokens") or 0)
    artifacts = target.get(artifact_field) or {}
    selected = [
        (key, row)
        for key, row in artifacts.items()
        if key == "private_train" or str(key).endswith(":private_train")
    ]
    if not selected:
        raise ValueError(
            f"target has no frozen {artifact_field} train artifact: {target['target_id']}"
        )

    sequences: list[list[int]] = []
    mask_starts: list[int] = []
    row_hashes: list[str] = []
    artifact_receipts: list[dict[str, Any]] = []
    for key, artifact in selected:
        if not isinstance(artifact, dict):
            raise ValueError(f"invalid supervision artifact contract: {key}")
        path = resolve(str(artifact.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(artifact.get("sha256") or ""):
            raise ValueError(f"supervision artifact identity mismatch: {key}")
        observed_rows = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("split") != "private_train" or row.get("public_benchmark") is not False:
                    raise ValueError(f"invalid supervision boundary: {key}:{observed_rows}")
                prompt = str(row.get("prompt") or "")
                answer = str(row.get("target") or "")
                source_ids, source_receipt = encode_tokens(
                    exact_text_tokens(prompt), source_vocab, stream="source"
                )
                target_ids, target_receipt = encode_tokens(
                    exact_text_tokens(answer), target_vocab, stream="target"
                )
                if int(source_receipt.get("unknown_token_count") or 0) or int(
                    target_receipt.get("unknown_token_count") or 0
                ):
                    raise ValueError(f"frozen supervision row became unrepresentable: {key}")
                sequence = [GLOBAL_BOS_ID]
                sequence.extend(source_offset + int(value) for value in source_ids)
                sequence.append(SOURCE_TARGET_SEPARATOR_ID)
                sequence.append(target_offset + int(target_vocab["<bos>"]))
                target_start = len(sequence)
                sequence.extend(target_offset + int(value) for value in target_ids)
                sequence.append(target_offset + int(target_vocab["<eos>"]))
                if len(sequence) > max_sequence + 1:
                    raise ValueError(f"frozen supervision row requires truncation: {key}")
                sequences.append(sequence)
                mask_starts.append(target_start - 1)
                row_hashes.append(
                    hashlib.sha256((prompt + "\0" + answer).encode()).hexdigest()
                )
                observed_rows += 1
        if observed_rows != int(artifact.get("row_count") or 0):
            raise ValueError(f"supervision row count changed: {key}")
        artifact_receipts.append(
            {
                "key": key,
                "path": relative(path),
                "sha256": str(artifact["sha256"]),
                "row_count": observed_rows,
            }
        )

    inputs = np.zeros((len(sequences), max_sequence), dtype=np.int32)
    labels = np.zeros((len(sequences), max_sequence), dtype=np.int32)
    mask = np.zeros((len(sequences), max_sequence), dtype=np.uint8)
    for index, (sequence, mask_start) in enumerate(zip(sequences, mask_starts)):
        width = len(sequence) - 1
        inputs[index, :width] = sequence[:-1]
        labels[index, :width] = sequence[1:]
        mask[index, mask_start:width] = 1
    loss_mask = mask.astype(np.float32)
    termination_id = target_offset + int(target_vocab["<eos>"])
    byte_begin_id = target_offset + int(target_vocab[TARGET_BYTE_BEGIN])
    byte_end_id = target_offset + int(target_vocab[TARGET_BYTE_END])
    loss_mask[(mask == 1) & (labels == termination_id)] = float(
        config["training"]["termination_loss_weight"]
    )
    loss_mask[
        (mask == 1) & ((labels == byte_begin_id) | (labels == byte_end_id))
    ] = float(config["training"]["byte_boundary_loss_weight"])
    receipt = {
        "policy": receipt_policy,
        "target_id": target["target_id"],
        "artifacts": artifact_receipts,
        "row_count": len(sequences),
        "target_positions": int(mask.sum()),
        "weighted_loss_positions": float(loss_mask.sum()),
        "termination_loss_weight": float(config["training"]["termination_loss_weight"]),
        "byte_boundary_loss_weight": float(config["training"]["byte_boundary_loss_weight"]),
        "sequence_width": max_sequence,
        "content_digest": hashlib.sha256("\n".join(row_hashes).encode()).hexdigest(),
        "generator_visible_fields": ["prompt"],
        "evaluator_only_fields": ["target", "target_sha256", "source_identity"],
        "source_truncation_count": 0,
        "target_truncation_count": 0,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    return SimpleNamespace(
        inputs=inputs,
        labels=labels,
        mask=mask,
        loss_mask=loss_mask,
        receipt=receipt,
    )


def evaluate_target(
    config: dict[str, Any],
    base: dict[str, Any],
    plan: dict[str, Any],
    target: dict[str, Any],
    *,
    metadata: dict[str, Any],
    mx: Any,
    nn: Any,
    split: str = "private_dev",
) -> dict[str, Any]:
    """Evaluate frozen rows while keeping answers outside the generation call."""

    source_vocab = dict(metadata.get("source_vocab") or {})
    target_vocab = dict(metadata.get("target_vocab") or {})
    model = build_model(
        CausalTransformerConfig(
            vocab_size=int(plan["models"]["vocab_size"]), **target["model"]
        ),
        mx=mx,
        nn=nn,
        state_role_lookup=None,
        source_to_target_lookup=build_source_to_target_lookup(base, metadata),
    )
    checkpoint = resolve(str(target["checkpoint"]))
    if target.get("role") == "language_expert":
        shared = resolve(str(target.get("shared_trunk_checkpoint") or ""))
        if not shared.is_file():
            raise ValueError("expert evaluation requires shared trunk checkpoint")
        model.load_weights(str(shared), strict=False)
        model.load_weights(str(checkpoint), strict=False)
    else:
        model.load_weights(str(checkpoint))
    mx.eval(model.parameters())
    model.eval()
    artifacts = target.get("supervision_artifacts") or {}
    selected = [
        (key, row)
        for key, row in artifacts.items()
        if key == split or str(key).endswith(f":{split}")
    ]
    rows: list[dict[str, Any]] = []
    evaluation_artifacts: list[dict[str, Any]] = []
    for key, artifact in selected:
        path = resolve(str((artifact or {}).get("path") or ""))
        if not path.is_file() or sha256_file(path) != str((artifact or {}).get("sha256") or ""):
            raise ValueError(f"evaluation artifact identity mismatch: {key}")
        evaluation_artifacts.append(
            {
                "key": key,
                "path": relative(path),
                "sha256": str(artifact["sha256"]),
                "row_count": int(artifact["row_count"]),
            }
        )
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("split") != split or row.get("public_benchmark") is not False:
                    raise ValueError(f"invalid evaluation boundary: {key}")
                generated, generation = generate_model_text(
                    model,
                    str(row.get("prompt") or ""),
                    source_vocab,
                    target_vocab,
                    base,
                    max_tokens=int(config["evaluation"]["decode_max_target_tokens"]),
                    max_source_tokens=int(
                        config["supervision"]["maximum_source_encoded_tokens"]
                    ),
                    beam_width=int(config["evaluation"]["beam_width"]),
                    branching_factor=int(config["evaluation"]["branching_factor"]),
                    length_penalty=float(config["evaluation"]["length_penalty"]),
                    mx=mx,
                )
                expected = str(row.get("target") or "")
                arm_id = str(row.get("arm_id") or "")
                diagnostics = behavior_diagnostics(
                    generated=generated,
                    expected=expected,
                    prompt=str(row.get("prompt") or ""),
                )
                rows.append(
                    {
                        "row_id": row.get("row_id"),
                        "arm_id": arm_id,
                        "prompt_sha256": row.get("prompt_sha256"),
                        "expected_sha256": row.get("target_sha256"),
                        "generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
                        "exact_match": (
                            generated == expected
                            and generation.get("state") == "GREEN"
                            and generation.get("stop_reason") == "eos"
                        ),
                        "nonempty": bool(generated),
                        "behavior_diagnostics": diagnostics,
                        "syntax": syntax_diagnostic(generated, arm_id),
                        "generation": generation,
                    }
                )
    by_arm: dict[str, Any] = {}
    for arm_id in ARM_IDS:
        arm_rows = [row for row in rows if row["arm_id"] == arm_id]
        if arm_rows:
            by_arm[arm_id] = evaluation_summary(arm_rows)
    report = {
        "policy": config["evaluation"]["policy"],
        "created_utc": now(),
        "trigger_state": "GREEN",
        "target_id": target["target_id"],
        "split": split,
        "evaluation_contract_sha256": hashlib.sha256(
            json.dumps(config["evaluation"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "evaluation_artifacts": evaluation_artifacts,
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "row_count": len(rows),
        "summary": evaluation_summary(rows),
        "by_arm": by_arm,
        "rows": rows,
        "generator_visible_fields": ["prompt"],
        "evaluator_only_fields": ["target", "target_sha256", "source_identity"],
        "target_visible_to_generator": False,
        "candidate_family": "direct_autoregressive_model_text",
        "templates_renderers_routers_tools_credit": 0,
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "capability_claim": (
            "PRIVATE_DEVELOPMENT_DIAGNOSTIC"
            if split == "private_dev"
            else "PRIVATE_FROZEN_CONFIRMATION_ONLY"
        ),
    }
    output = resolve(str(target["receipt"])).with_name(f"evaluation_{split}_receipt.json")
    write_json_atomic(output, report)
    return {**report, "rows": {"path": relative(output), "embedded_row_count": len(rows)}}


def generate_model_text(
    model: Any,
    prompt: str,
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    base: dict[str, Any],
    *,
    max_tokens: int,
    max_source_tokens: int,
    beam_width: int,
    branching_factor: int,
    length_penalty: float,
    mx: Any,
) -> tuple[str, dict[str, Any]]:
    """Generate from prompt only; the grammar constrains byte serialization, not meaning."""

    source_ids, source_receipt = encode_tokens(
        exact_text_tokens(prompt), source_vocab, stream="source"
    )
    if int(source_receipt.get("unknown_token_count") or 0):
        return "", generation_fault("source_unrepresentable")
    if len(source_ids) > max_source_tokens:
        return "", generation_fault("source_requires_truncation")
    source_offset = source_token_offset(base, source_vocab)
    target_offset = target_token_offset(base, source_vocab)
    prompt_ids = [GLOBAL_BOS_ID]
    prompt_ids.extend(source_offset + int(value) for value in source_ids)
    prompt_ids.append(SOURCE_TARGET_SEPARATOR_ID)
    prompt_ids.append(target_offset + int(target_vocab["<bos>"]))
    logits, cache = model(mx.array([prompt_ids], dtype=mx.int32))
    mx.eval(logits)
    inverse = {int(value): str(token) for token, value in target_vocab.items()}
    beams = [{"tokens": [], "score": 0.0, "logits": logits, "cache": cache}]
    complete: list[dict[str, Any]] = []
    for _ in range(max_tokens):
        expansions: list[dict[str, Any]] = []
        for beam in beams:
            allowed = serialization_valid_local_ids(beam["tokens"], inverse)
            if not allowed:
                continue
            values = np.asarray(
                beam["logits"][0, -1, target_offset : target_offset + len(target_vocab)]
            ).astype(np.float64)
            allowed_values = np.asarray([values[index] for index in allowed], dtype=np.float64)
            maximum = float(allowed_values.max())
            normalizer = maximum + float(np.log(np.exp(allowed_values - maximum).sum()))
            ranked = sorted(allowed, key=lambda index: float(values[index]), reverse=True)[
                : max(1, branching_factor)
            ]
            for local_id in ranked:
                token = inverse[local_id]
                score = float(beam["score"]) + float(values[local_id]) - normalizer
                if token == "<eos>":
                    complete.append({"tokens": list(beam["tokens"]), "score": score})
                    continue
                next_logits, next_cache = model(
                    mx.array([[target_offset + local_id]], dtype=mx.int32), beam["cache"]
                )
                mx.eval(next_logits)
                expansions.append(
                    {
                        "tokens": [*beam["tokens"], token],
                        "score": score,
                        "logits": next_logits,
                        "cache": next_cache,
                    }
                )
        beams = sorted(
            expansions,
            key=lambda row: beam_score(row, length_penalty),
            reverse=True,
        )[: max(1, beam_width)]
        complete = sorted(
            complete,
            key=lambda row: beam_score(row, length_penalty),
            reverse=True,
        )[: max(1, beam_width)]
        if not beams or (
            complete
            and len(complete) >= beam_width
            and beam_score(complete[0], length_penalty)
            >= beam_score(beams[0], length_penalty)
        ):
            break
    if complete:
        selected = max(complete, key=lambda row: beam_score(row, length_penalty))
        generated_tokens = list(selected["tokens"])
        stop_reason = "eos"
    elif beams:
        selected = max(beams, key=lambda row: beam_score(row, length_penalty))
        generated_tokens = list(selected["tokens"])
        stop_reason = "max_tokens"
    else:
        return "", generation_fault("no_serialization_valid_sequence")
    decoded, decode_receipt = decode_target_tokens(generated_tokens)
    if decode_receipt.get("state") != "READY":
        return "", {
            **generation_fault("byte_serialization_fault"),
            "decode_receipt": decode_receipt,
        }
    text = "".join(decoded)
    return text, {
        "state": "GREEN",
        "decoder": "beam_exact_text_with_byte_span_grammar_v1",
        "beam_width": int(beam_width),
        "branching_factor": int(branching_factor),
        "stop_reason": stop_reason,
        "generated_token_count": len(generated_tokens),
        "generated_token_sha256": hashlib.sha256(
            "\n".join(generated_tokens).encode()
        ).hexdigest(),
        "byte_serialization_valid": True,
        "target_visible_to_generator": False,
        "fallback_return_count": 0,
    }


def serialization_valid_local_ids(
    generated_tokens: list[str], inverse: dict[int, str]
) -> list[int]:
    active = bool(active_target_span(generated_tokens)["active"])
    allowed: list[int] = []
    for local_id, token in inverse.items():
        if active:
            if is_byte_token(token) or token == TARGET_BYTE_END:
                allowed.append(local_id)
        elif token == "<eos>" or token == TARGET_BYTE_BEGIN or (
            token not in {"<pad>", "<unk>", "<bos>", TARGET_BYTE_END}
            and not is_byte_token(token)
        ):
            allowed.append(local_id)
    return allowed


def beam_score(row: dict[str, Any], length_penalty: float) -> float:
    length = max(1, len(row.get("tokens") or []))
    return float(row.get("score") or 0.0) / (length ** max(0.0, length_penalty))


def generation_fault(reason: str) -> dict[str, Any]:
    return {
        "state": "FAULT",
        "reason": reason,
        "target_visible_to_generator": False,
        "failure_behavior": "reject_without_fallback",
        "fallback_return_count": 0,
    }


def syntax_diagnostic(text: str, arm_id: str) -> dict[str, Any]:
    if arm_id == "python":
        try:
            ast.parse(text)
        except SyntaxError as exc:
            return {"state": "INVALID", "checker": "python_ast", "detail": str(exc)[:200]}
        return {"state": "VALID", "checker": "python_ast"}
    return {
        "state": "NOT_CLAIMED",
        "checker": "none",
        "reason": "language-native parser not yet bound into this evaluation contract",
    }


def behavior_diagnostics(*, generated: str, expected: str, prompt: str) -> dict[str, Any]:
    """Evaluator-only failure telemetry; none of these values enter generation."""

    source_excerpt = ""
    marker = "\nCurrent excerpt:\n"
    terminator = "\n\n\nReturn only the complete revised excerpt."
    if marker in prompt:
        source_excerpt = prompt.split(marker, 1)[1]
        if terminator in source_excerpt:
            source_excerpt = source_excerpt.split(terminator, 1)[0]
    generated_lines = [line for line in generated.splitlines() if line.strip()]
    return {
        "generated_character_count": len(generated),
        "expected_character_count": len(expected),
        "target_length_ratio": round(len(generated) / max(1, len(expected)), 8),
        "target_sequence_similarity": round(
            difflib.SequenceMatcher(None, generated, expected, autojunk=False).ratio(), 8
        ),
        "source_excerpt_available": bool(source_excerpt),
        "source_sequence_similarity": round(
            difflib.SequenceMatcher(
                None, generated, source_excerpt, autojunk=False
            ).ratio(),
            8,
        )
        if source_excerpt
        else None,
        "nonempty_line_count": len(generated_lines),
        "unique_nonempty_line_ratio": round(
            len(set(generated_lines)) / max(1, len(generated_lines)), 8
        ),
        "raw_generated_text_retained": False,
    }


def evaluation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    exact = sum(bool(row.get("exact_match")) for row in rows)
    nonempty = sum(bool(row.get("nonempty")) for row in rows)
    serialization_valid = sum(
        bool((row.get("generation") or {}).get("byte_serialization_valid")) for row in rows
    )
    syntax_valid = sum((row.get("syntax") or {}).get("state") == "VALID" for row in rows)
    syntax_checked = sum(
        (row.get("syntax") or {}).get("state") in {"VALID", "INVALID"} for row in rows
    )
    similarities = [
        float((row.get("behavior_diagnostics") or {}).get("target_sequence_similarity") or 0.0)
        for row in rows
    ]
    source_similarities = [
        float(value)
        for row in rows
        if (
            value := (row.get("behavior_diagnostics") or {}).get(
                "source_sequence_similarity"
            )
        )
        is not None
    ]
    length_ratios = [
        float((row.get("behavior_diagnostics") or {}).get("target_length_ratio") or 0.0)
        for row in rows
    ]
    return {
        "row_count": total,
        "exact_match_count": exact,
        "exact_target_match_rate": round(exact / max(1, total), 8),
        "nonempty_count": nonempty,
        "nonempty_rate": round(nonempty / max(1, total), 8),
        "byte_serialization_valid_count": serialization_valid,
        "byte_serialization_valid_rate": round(serialization_valid / max(1, total), 8),
        "syntax_checked_count": syntax_checked,
        "syntax_valid_count": syntax_valid,
        "syntax_valid_rate_when_checked": round(syntax_valid / max(1, syntax_checked), 8),
        "mean_target_sequence_similarity": round(sum(similarities) / max(1, total), 8),
        "mean_source_sequence_similarity": round(
            sum(source_similarities) / max(1, len(source_similarities)), 8
        ),
        "mean_target_length_ratio": round(sum(length_ratios) / max(1, total), 8),
        "raw_generated_text_retained": False,
    }


def train_target(
    config: dict[str, Any],
    plan: dict[str, Any],
    target: dict[str, Any],
    *,
    stage: Any,
    max_steps: int,
    resume: bool,
    mx: Any,
    nn: Any,
    optim: Any,
    mlx_utils: Any,
    source_conditioned_stage: Any | None = None,
    supervision_stage: Any | None = None,
) -> dict[str, Any]:
    target_id = str(target["target_id"])
    inputs = range_view(stage.pretrain_inputs, target["row_ranges"])
    labels = range_view(stage.pretrain_labels, target["row_ranges"])
    mask = range_view(stage.pretrain_mask, target["row_ranges"])
    copy_lookup = None
    if str((target.get("model") or {}).get("source_copy_mode") or "none") != "none":
        copy_lookup = build_source_to_target_lookup(
            read_json(resolve(str(config["base_config"]))),
            read_json(resolve(str(config["stage_dir"])) / "stage_metadata_v1.json"),
        )
    model = build_model(
        CausalTransformerConfig(vocab_size=int(plan["models"]["vocab_size"]), **target["model"]),
        mx=mx,
        nn=nn,
        state_role_lookup=None,
        source_to_target_lookup=copy_lookup,
    )
    expert_mode = target.get("role") == "language_expert"
    expert_scope = ""
    shared_trunk_checkpoint = resolve(str(target.get("shared_trunk_checkpoint") or ""))
    shared_trunk_checkpoint_sha256 = ""
    if expert_mode:
        if not shared_trunk_checkpoint.is_file():
            raise ValueError("language expert requires a completed shared trunk checkpoint")
        shared_receipt_path = shared_trunk_checkpoint.parent / "training_receipt.json"
        shared_receipt = read_json(shared_receipt_path)
        if not bool(shared_receipt.get("complete")):
            raise ValueError("language expert requires a complete shared trunk receipt")
        shared_trunk_checkpoint_sha256 = sha256_file(shared_trunk_checkpoint)
        if shared_trunk_checkpoint_sha256 != shared_receipt.get("checkpoint_sha256"):
            raise ValueError("shared trunk checkpoint identity mismatch")
        model.load_weights(str(shared_trunk_checkpoint), strict=False)
        expert_scope = str(
            target.get("expert_trainable_scope")
            or config["topology"]["expert_trainable_scope"]
        )
        model.freeze_to_language_expert(expert_scope)
    observed_parameters = int(parameter_count(model, mlx_utils))
    if observed_parameters != int(target["parameter_count"]):
        raise ValueError("target model parameter identity changed after preregistration")
    trainable_parameters = int(
        sum(
            value.size
            for _name, value in mlx_utils.tree_flatten(model.trainable_parameters())
        )
    )
    if expert_mode and trainable_parameters != int(
        plan["models"]["moecot_system"]["expert_parameter_count_per_arm"]
    ):
        raise ValueError("expert trainable parameter ownership mismatch")
    checkpoint = resolve(str(target["checkpoint"]))
    optimizer_path = resolve(str(target["optimizer_state"]))
    receipt_path = resolve(str(target["receipt"]))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    planned_steps = required_steps(mask, int(training["batch_size"]), int(target["unique_target_positions"]))
    unique_sft_positions = int(supervision_stage.mask.sum()) if supervision_stage is not None else 0
    sft_repetitions = int(training.get("supervision_optimizer_repetitions") or 1)
    sft_positions = unique_sft_positions * sft_repetitions
    sft_planned_steps = (
        required_steps(
            supervision_stage.mask,
            int(training["batch_size"]),
            sft_positions,
        )
        if sft_positions
        else 0
    )
    unique_source_positions = (
        int(source_conditioned_stage.mask.sum())
        if source_conditioned_stage is not None
        else 0
    )
    source_repetitions = int(training.get("source_conditioned_optimizer_repetitions") or 1)
    source_positions = unique_source_positions * source_repetitions
    source_planned_steps = (
        required_steps(
            source_conditioned_stage.mask,
            int(training["batch_size"]),
            source_positions,
        )
        if source_positions
        else 0
    )
    schedule = build_schedule(
        optim,
        mx,
        training,
        planned_steps + source_planned_steps + sft_planned_steps + 128,
    )
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=float(training["weight_decay"]))
    prior_steps = 0
    prior_pretrain_positions = 0
    prior_source_positions = 0
    prior_sft_positions = 0
    prior_checkpoint_hash = ""
    if resume:
        prior = read_json(receipt_path)
        resume_checkpoint = resolve(str(prior.get("checkpoint") or checkpoint))
        resume_optimizer = resolve(
            str(prior.get("optimizer_state") or optimizer_path)
        )
        validate_resume(
            prior,
            plan,
            target,
            resume_checkpoint,
            resume_optimizer,
        )
        model.load_weights(str(resume_checkpoint), strict=not expert_mode)
        optimizer.state = mlx_utils.tree_unflatten(
            list(mx.load(str(resume_optimizer)).items())
        )
        mx.eval(model.parameters(), optimizer.state)
        prior_steps = int(prior.get("optimizer_steps") or 0)
        prior_pretrain_positions = int(prior.get("pretrain_optimizer_positions") or 0)
        prior_source_positions = int(
            prior.get("source_conditioned_optimizer_positions") or 0
        )
        prior_sft_positions = int(prior.get("supervision_optimizer_positions") or 0)
        prior_checkpoint_hash = sha256_file(resume_checkpoint)
    remaining_positions = max(
        0, int(target["unique_target_positions"]) - prior_pretrain_positions
    )
    remaining_sft_positions = max(0, sft_positions - prior_sft_positions)
    remaining_source_positions = max(0, source_positions - prior_source_positions)
    allowed_steps = (
        max_steps
        if max_steps
        else planned_steps + source_planned_steps + sft_planned_steps + 128
    )
    temporary_checkpoint = checkpoint.with_name(
        checkpoint.stem + ".partial" + checkpoint.suffix
    )
    heartbeat = checkpoint.parent / "training_heartbeat.json"
    started = time.perf_counter()
    completed_positions = {
        "pretrain": prior_pretrain_positions,
        "source": prior_source_positions,
        "supervision": prior_sft_positions,
    }

    def commit_progress_checkpoint(progress: dict[str, Any]) -> None:
        phase = str(progress["phase"])
        positions = dict(completed_positions)
        if "source_conditioned_pretraining" in phase:
            positions["source"] = prior_source_positions + int(
                progress["target_positions_consumed"]
            )
        elif "supervision" in phase:
            positions["supervision"] = prior_sft_positions + int(
                progress["target_positions_consumed"]
            )
        else:
            positions["pretrain"] = prior_pretrain_positions + int(
                progress["target_positions_consumed"]
            )
        global_step = int(progress["global_step"])
        generation_checkpoint, generation_optimizer = checkpoint_generation_paths(
            checkpoint,
            optimizer_path,
            global_step,
        )
        previous = read_json(receipt_path) if receipt_path.is_file() else {}
        publish_model(
            model,
            generation_checkpoint,
            generation_checkpoint.with_name(
                generation_checkpoint.stem + ".partial" + generation_checkpoint.suffix
            ),
            mx=mx,
            mlx_utils=mlx_utils,
            trainable_only=expert_mode,
        )
        publish_optimizer(mx, mlx_utils, optimizer, generation_optimizer)
        progress_receipt = {
            "policy": "project_theseus_moecot_language_arm_training_receipt_v1",
            "created_utc": now(),
            "trigger_state": "GREEN",
            "target_id": target_id,
            "role": target["role"],
            "plan_sha256": plan["plan_sha256"],
            "stage_signature": plan["stage"]["stage_signature"],
            "stage_metadata_sha256": plan["stage"]["metadata_sha256"],
            "row_ranges": target["row_ranges"],
            "parameter_count": observed_parameters,
            "trainable_parameter_count": trainable_parameters,
            "expert_trainable_scope": expert_scope if expert_mode else "",
            "shared_trunk_checkpoint": (
                relative(shared_trunk_checkpoint) if expert_mode else ""
            ),
            "shared_trunk_checkpoint_sha256": shared_trunk_checkpoint_sha256,
            "optimizer_steps": global_step,
            "optimizer_positions": sum(positions.values()),
            "pretrain_optimizer_positions": positions["pretrain"],
            "source_conditioned_optimizer_positions": positions["source"],
            "supervision_optimizer_positions": positions["supervision"],
            "unique_target_positions": int(target["unique_target_positions"]),
            "checkpoint": relative(generation_checkpoint),
            "checkpoint_sha256": sha256_file(generation_checkpoint),
            "optimizer_state": relative(generation_optimizer),
            "optimizer_state_sha256": sha256_file(generation_optimizer),
            "complete": False,
            "transactional_progress": progress,
            "capability_claim": "NOT_EVALUATED",
            "hard_gaps": [],
            **no_cheat(config),
        }
        write_json_atomic(receipt_path, progress_receipt)
        cleanup_progress_generation(
            previous,
            canonical_checkpoint=checkpoint,
            canonical_optimizer=optimizer_path,
            keep={generation_checkpoint, generation_optimizer},
        )

    random.seed(int(config["seed"]) + stable_int(target_id) + prior_steps)
    mx.random.seed(int(config["seed"]) + stable_int(target_id) + prior_steps)
    loss_and_grad = nn.value_and_grad(model, causal_loss)
    pretrain_phase = train_phase(
        model,
        optimizer,
        loss_and_grad,
        inputs,
        labels,
        mask,
        progress_mask=mask,
        ordered_plan_loss_weight=1.0,
        sample_weights=None,
        plan_labels=None,
        plan_label_mode="none",
        plan_auxiliary_weight=0.0,
        plan_shuffle_seed=0,
        plan_loss_mode="binary_multilabel",
        plan_slot_count=0,
        plan_factor_group_sizes=(),
        phase_name=f"moecot_pretraining:{target_id}",
        target_positions=remaining_positions,
        batch_size=int(training["batch_size"]),
        gradient_clip=float(training["gradient_clip_norm"]),
        seed=int(config["seed"]) + stable_int(target_id) + prior_steps,
        max_steps=allowed_steps,
        checkpoint=temporary_checkpoint,
        checkpoint_every=max(1, int(training["checkpoint_every_steps"])),
        heartbeat=heartbeat,
        global_step_offset=prior_steps,
        mx=mx,
        optim=optim,
        checkpoint_callback=commit_progress_checkpoint,
    )
    completed_positions["pretrain"] = prior_pretrain_positions + int(
        pretrain_phase["target_positions_consumed"]
    )
    used_steps = int(pretrain_phase["optimizer_steps"])
    source_conditioned_phase = {
        "phase": f"moecot_source_conditioned_pretraining:{target_id}",
        "optimizer_steps": 0,
        "target_positions_consumed": 0,
        "target_positions_requested": remaining_source_positions,
        "mean_loss": None,
        "final_loss": None,
    }
    if (
        source_conditioned_stage is not None
        and remaining_source_positions > 0
        and used_steps < allowed_steps
    ):
        source_conditioned_phase = train_phase(
            model,
            optimizer,
            loss_and_grad,
            source_conditioned_stage.inputs,
            source_conditioned_stage.labels,
            source_conditioned_stage.loss_mask,
            progress_mask=source_conditioned_stage.mask,
            ordered_plan_loss_weight=1.0,
            sample_weights=None,
            plan_labels=None,
            plan_label_mode="none",
            plan_auxiliary_weight=0.0,
            plan_shuffle_seed=0,
            plan_loss_mode="binary_multilabel",
            plan_slot_count=0,
            plan_factor_group_sizes=(),
            phase_name=f"moecot_source_conditioned_pretraining:{target_id}",
            target_positions=remaining_source_positions,
            batch_size=int(training["batch_size"]),
            gradient_clip=float(training["gradient_clip_norm"]),
            seed=int(config["seed"]) + stable_int(target_id) + prior_steps + used_steps,
            max_steps=allowed_steps - used_steps,
            checkpoint=temporary_checkpoint,
            checkpoint_every=max(1, int(training["checkpoint_every_steps"])),
            heartbeat=heartbeat,
            global_step_offset=prior_steps + used_steps,
            mx=mx,
            optim=optim,
            checkpoint_callback=commit_progress_checkpoint,
        )
        used_steps += int(source_conditioned_phase["optimizer_steps"])
        completed_positions["source"] = prior_source_positions + int(
            source_conditioned_phase["target_positions_consumed"]
        )
    supervision_phase = {
        "phase": f"moecot_supervision:{target_id}",
        "optimizer_steps": 0,
        "target_positions_consumed": 0,
        "target_positions_requested": remaining_sft_positions,
        "mean_loss": None,
        "final_loss": None,
    }
    if supervision_stage is not None and remaining_sft_positions > 0 and used_steps < allowed_steps:
        supervision_phase = train_phase(
            model,
            optimizer,
            loss_and_grad,
            supervision_stage.inputs,
            supervision_stage.labels,
            supervision_stage.loss_mask,
            progress_mask=supervision_stage.mask,
            ordered_plan_loss_weight=1.0,
            sample_weights=None,
            plan_labels=None,
            plan_label_mode="none",
            plan_auxiliary_weight=0.0,
            plan_shuffle_seed=0,
            plan_loss_mode="binary_multilabel",
            plan_slot_count=0,
            plan_factor_group_sizes=(),
            phase_name=f"moecot_supervision:{target_id}",
            target_positions=remaining_sft_positions,
            batch_size=int(training["batch_size"]),
            gradient_clip=float(training["gradient_clip_norm"]),
            seed=int(config["seed"]) + stable_int(target_id) + prior_steps + used_steps,
            max_steps=allowed_steps - used_steps,
            checkpoint=temporary_checkpoint,
            checkpoint_every=max(1, int(training["checkpoint_every_steps"])),
            heartbeat=heartbeat,
            global_step_offset=prior_steps + used_steps,
            mx=mx,
            optim=optim,
            checkpoint_callback=commit_progress_checkpoint,
        )
    publish_model(
        model,
        checkpoint,
        temporary_checkpoint,
        mx=mx,
        mlx_utils=mlx_utils,
        trainable_only=expert_mode,
    )
    publish_optimizer(mx, mlx_utils, optimizer, optimizer_path)
    total_steps = prior_steps + used_steps + int(supervision_phase["optimizer_steps"])
    total_pretrain_positions = prior_pretrain_positions + int(
        pretrain_phase["target_positions_consumed"]
    )
    total_sft_positions = prior_sft_positions + int(
        supervision_phase["target_positions_consumed"]
    )
    total_source_positions = prior_source_positions + int(
        source_conditioned_phase["target_positions_consumed"]
    )
    total_positions = total_pretrain_positions + total_source_positions + total_sft_positions
    receipt = {
        "policy": "project_theseus_moecot_language_arm_training_receipt_v1",
        "created_utc": now(),
        "trigger_state": "GREEN",
        "target_id": target_id,
        "role": target["role"],
        "plan_sha256": plan["plan_sha256"],
        "stage_signature": plan["stage"]["stage_signature"],
        "stage_metadata_sha256": plan["stage"]["metadata_sha256"],
        "row_ranges": target["row_ranges"],
        "parameter_count": observed_parameters,
        "trainable_parameter_count": trainable_parameters,
        "expert_trainable_scope": (
            expert_scope if expert_mode else ""
        ),
        "shared_trunk_checkpoint": (
            relative(shared_trunk_checkpoint) if expert_mode else ""
        ),
        "shared_trunk_checkpoint_sha256": shared_trunk_checkpoint_sha256,
        "optimizer_steps": total_steps,
        "optimizer_positions": total_positions,
        "pretrain_optimizer_positions": total_pretrain_positions,
        "source_conditioned_optimizer_positions": total_source_positions,
        "supervision_optimizer_positions": total_sft_positions,
        "unique_target_positions": int(target["unique_target_positions"]),
        "unique_source_conditioned_target_positions": unique_source_positions,
        "source_conditioned_optimizer_target_positions": source_positions,
        "source_conditioned_optimizer_repetitions": source_repetitions,
        "unique_supervision_target_positions": unique_sft_positions,
        "supervision_optimizer_target_positions": sft_positions,
        "supervision_optimizer_repetitions": sft_repetitions,
        "complete": (
            total_pretrain_positions >= int(target["unique_target_positions"])
            and total_source_positions >= source_positions
            and total_sft_positions >= sft_positions
        ),
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "optimizer_state": relative(optimizer_path),
        "optimizer_state_sha256": sha256_file(optimizer_path),
        "resume": resume,
        "resume_base_checkpoint_sha256": prior_checkpoint_hash,
        "phases": {
            "pretraining": pretrain_phase,
            "source_conditioned_pretraining": source_conditioned_phase,
            "supervision": supervision_phase,
        },
        "source_conditioned_stage": (
            source_conditioned_stage.receipt
            if source_conditioned_stage is not None
            else None
        ),
        "supervision_stage": (
            supervision_stage.receipt if supervision_stage is not None else None
        ),
        "wall_seconds": round(time.perf_counter() - started, 6),
        "energy_joules": None,
        "energy_measurement_state": "NOT_AVAILABLE_FROM_MLX_RUNTIME",
        "capability_claim": "NOT_EVALUATED",
        "hard_gaps": [],
        **no_cheat(config),
    }
    previous_receipt = read_json(receipt_path) if receipt_path.is_file() else {}
    write_json_atomic(receipt_path, receipt)
    cleanup_progress_generation(
        previous_receipt,
        canonical_checkpoint=checkpoint,
        canonical_optimizer=optimizer_path,
        keep={checkpoint, optimizer_path},
    )
    return receipt


def range_view(array: np.ndarray, ranges: list[dict[str, int]]) -> np.ndarray:
    normalized = [(int(row["start"]), int(row["stop"])) for row in ranges]
    if not normalized:
        raise ValueError("training target has no stage ranges")
    if all(normalized[index][1] == normalized[index + 1][0] for index in range(len(normalized) - 1)):
        return array[normalized[0][0] : normalized[-1][1]]
    return np.concatenate([array[start:stop] for start, stop in normalized], axis=0)


def publish_model(
    model: Any,
    checkpoint: Path,
    temporary: Path,
    *,
    mx: Any,
    mlx_utils: Any,
    trainable_only: bool,
) -> None:
    temporary.unlink(missing_ok=True)
    if trainable_only:
        weights = {
            name: value
            for name, value in mlx_utils.tree_flatten(model.trainable_parameters())
        }
        mx.save_safetensors(
            str(temporary),
            weights,
            metadata={"policy": "moecot_language_expert_delta_v2"},
        )
    else:
        model.save_weights(str(temporary))
    if not temporary.is_file():
        raise ValueError("MLX model checkpoint publication failed")
    os.replace(temporary, checkpoint)


def publish_optimizer(mx: Any, mlx_utils: Any, optimizer: Any, path: Path) -> None:
    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    temporary.unlink(missing_ok=True)
    flat = {name: value for name, value in mlx_utils.tree_flatten(optimizer.state)}
    mx.save_safetensors(str(temporary), flat, metadata={"policy": "moecot_optimizer_state_v1"})
    os.replace(temporary, path)


def checkpoint_generation_paths(
    checkpoint: Path, optimizer: Path, global_step: int
) -> tuple[Path, Path]:
    if global_step <= 0:
        raise ValueError("checkpoint generation step must be positive")
    suffix = f".step-{global_step:08d}"
    return (
        checkpoint.with_name(checkpoint.stem + suffix + checkpoint.suffix),
        optimizer.with_name(optimizer.stem + suffix + optimizer.suffix),
    )


def cleanup_progress_generation(
    receipt: dict[str, Any],
    *,
    canonical_checkpoint: Path,
    canonical_optimizer: Path,
    keep: set[Path] | None = None,
) -> None:
    """Delete only superseded step generations after a newer receipt commits."""

    retained = {path.resolve() for path in (keep or set())}
    for key, canonical in (
        ("checkpoint", canonical_checkpoint),
        ("optimizer_state", canonical_optimizer),
    ):
        value = str(receipt.get(key) or "")
        if not value:
            continue
        candidate = resolve(value)
        prefix = canonical.stem + ".step-"
        if (
            candidate.resolve() not in retained
            and candidate.parent.resolve() == canonical.parent.resolve()
            and candidate.name.startswith(prefix)
            and candidate.suffix == canonical.suffix
        ):
            candidate.unlink(missing_ok=True)


def validate_resume(
    receipt: dict[str, Any], plan: dict[str, Any], target: dict[str, Any], checkpoint: Path, optimizer: Path
) -> None:
    faults = []
    if receipt.get("policy") != "project_theseus_moecot_language_arm_training_receipt_v1":
        faults.append("receipt_policy_mismatch")
    if receipt.get("target_id") != target["target_id"]:
        faults.append("target_identity_mismatch")
    if receipt.get("plan_sha256") != plan["plan_sha256"]:
        faults.append("plan_identity_mismatch")
    if receipt.get("stage_signature") != plan["stage"]["stage_signature"]:
        faults.append("stage_identity_mismatch")
    if receipt.get("row_ranges") != target["row_ranges"]:
        faults.append("stage_range_mismatch")
    if target.get("role") == "language_expert":
        shared = resolve(str(target.get("shared_trunk_checkpoint") or ""))
        if (
            not shared.is_file()
            or sha256_file(shared)
            != receipt.get("shared_trunk_checkpoint_sha256")
        ):
            faults.append("shared_trunk_checkpoint_identity_mismatch")
    if not checkpoint.is_file() or sha256_file(checkpoint) != receipt.get("checkpoint_sha256"):
        faults.append("checkpoint_identity_mismatch")
    if not optimizer.is_file() or sha256_file(optimizer) != receipt.get("optimizer_state_sha256"):
        faults.append("optimizer_identity_mismatch")
    if faults:
        raise ValueError("resume denied: " + ",".join(faults))


def plan_sha256(
    config: dict[str, Any],
    metadata: dict[str, Any],
    models: dict[str, Any],
    supervision: dict[str, Any],
    source_conditioned: dict[str, Any],
) -> str:
    training_artifacts = {
        key: value
        for key, value in (supervision.get("artifacts") or {}).items()
        if str(key).endswith(":private_train")
    }
    payload = {
        "training_contract": {
            key: config.get(key)
            for key in (
                "policy",
                "seed",
                "topology",
                "shared_trunk_model",
                "arm_model",
                "controls",
                "training",
                "boundaries",
            )
        },
        "stage_signature": (metadata.get("summary") or {}).get("stage_signature"),
        "arm_views": ((metadata.get("summary") or {}).get("canonical_pretrain_stage") or {}).get("arm_views"),
        "models": models,
        "supervision_training_artifacts": training_artifacts,
        "source_conditioned_training_artifacts": source_conditioned.get("artifacts")
        or {},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("policy") != "project_theseus_moecot_language_arm_training_v1":
        raise ValueError("unexpected MoECOT training policy")
    if config.get("comparison_contract", {}).get("preregistered_before_training") is not True:
        raise ValueError("comparison contract must be preregistered")
    topology = config.get("topology") or {}
    if topology.get("policy") not in {
        "project_theseus_moecot_shared_trunk_source_specialists_v2",
        "project_theseus_moecot_scaled_low_rank_specialists_v3",
    } or topology.get("mode") != "shared_trunk_language_experts":
        raise ValueError("unexpected MoECOT shared-trunk topology")
    arm_model = dict(config.get("arm_model") or {})
    expert_dim = int(arm_model.pop("expert_adapter_dim", 0))
    source_expert_dim = int(arm_model.pop("source_expert_adapter_dim", 0))
    if arm_model != dict(config.get("shared_trunk_model") or {}):
        raise ValueError("language expert model must exactly extend the shared trunk")
    if expert_dim != int(topology.get("expert_adapter_dim") or 0) or expert_dim <= 0:
        raise ValueError("language expert dimension must match the topology contract")
    if source_expert_dim != int(topology.get("source_expert_adapter_dim") or 0):
        raise ValueError("source expert dimension must match the topology contract")
    if topology.get("expert_trainable_scope") not in {
        "adapter_only",
        "source_conditioned_delta",
        "low_rank_source_adapters",
    }:
        raise ValueError("unsupported language expert trainable scope")
    initialization = topology.get("shared_trunk_initialization") or {}
    bootstrap = topology.get("shared_trunk_bootstrap") or initialization
    if bootstrap.get("policy") == "project_theseus_exact_shared_trunk_migration_v1":
        for key in (
            "checkpoint",
            "checkpoint_sha256",
            "optimizer_state",
            "optimizer_state_sha256",
            "receipt",
            "receipt_sha256",
        ):
            if not bootstrap.get(key):
                raise ValueError(f"shared trunk migration missing {key}")
    elif initialization.get("policy") == "project_theseus_seeded_fresh_trunk_initialization_v1":
        if int(initialization.get("seed") or -1) != int(config.get("seed") or -2):
            raise ValueError("fresh shared trunk initialization seed mismatch")
        if not str(initialization.get("reason") or "").strip():
            raise ValueError("fresh shared trunk initialization requires a reason")
    else:
        raise ValueError("shared trunk initialization contract is required")
    boundaries = config.get("boundaries") or {}
    if any(int(boundaries.get(key) or 0) for key in (
        "public_training_rows_written", "external_inference_calls", "fallback_return_count",
        "templates_renderers_routers_tools_credit",
    )):
        raise ValueError("MoECOT training no-cheat counters must remain zero")
    if boundaries.get("hidden_generalist_fallback") != "forbidden":
        raise ValueError("hidden generalist fallback must remain forbidden")
    evaluation = config.get("evaluation") or {}
    if evaluation.get("policy") != "project_theseus_moecot_direct_model_only_evaluation_v1":
        raise ValueError("unexpected MoECOT evaluation policy")
    if not 1 <= int(evaluation.get("beam_width") or 0) <= 16:
        raise ValueError("evaluation beam width must be bounded")
    if not 1 <= int(evaluation.get("branching_factor") or 0) <= 16:
        raise ValueError("evaluation branching factor must be bounded")
    if evaluation.get("target_visible_to_generator") is not False:
        raise ValueError("evaluation target must remain hidden from generation")
    if evaluation.get("templates_renderers_routers_tools_allowed") is not False:
        raise ValueError("assisted generation is forbidden in model-only evaluation")
    training = config.get("training") or {}
    repetitions = int(training.get("supervision_optimizer_repetitions") or 0)
    if not 1 <= repetitions <= int(
        training.get("maximum_supervision_optimizer_repetitions") or 0
    ):
        raise ValueError("supervision repetition must remain within the frozen maximum")
    source_repetitions = int(
        training.get("source_conditioned_optimizer_repetitions") or 1
    )
    if not 1 <= source_repetitions <= int(
        training.get("maximum_source_conditioned_optimizer_repetitions") or 1
    ):
        raise ValueError(
            "source-conditioned repetition must remain within the frozen maximum"
        )
    if not 1.0 <= float(training.get("termination_loss_weight") or 0.0) <= 8.0:
        raise ValueError("termination loss weight must remain bounded")
    if not 1.0 <= float(training.get("byte_boundary_loss_weight") or 0.0) <= 8.0:
        raise ValueError("byte-boundary loss weight must remain bounded")


def no_cheat(config: dict[str, Any]) -> dict[str, Any]:
    return {**config["boundaries"], "score_semantics": "training provenance only; direct verifier behavior is evaluated separately"}


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    write_json(temporary, payload)
    os.replace(temporary, path)


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
