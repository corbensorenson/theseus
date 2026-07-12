#!/usr/bin/env python3
"""Train independent MoECOT language arms and preregister dense controls.

The runtime consumes the immutable canonical stage produced by the standard
transformer corpus path. It does not build another corpus, route answers, or
turn training loss into a capability claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from standard_causal_transformer_model import CausalTransformerConfig, build_model, parameter_count
from standard_causal_transformer_corpus import load_pretrain_memmaps, pretrain_array_paths
from standard_causal_transformer_survival import (
    build_schedule,
    causal_loss,
    model_vocab_size,
    required_steps,
    train_phase,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "moecot_language_arm_training.json"
ARM_IDS = ("english", "python", "javascript_typescript", "html_css", "rust")
CONTROL_IDS = ("dense_total_parameter", "dense_active_parameter")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--target", action="append", choices=[*ARM_IDS, *CONTROL_IDS])
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
        if int(models["dense_active_parameter"]["parameter_count"]) != int(
            models["moecot_system"]["active_parameter_count_per_request"]
        ):
            gaps.append("active_parameter_control_mismatch")
    plan_identity = plan_sha256(config, metadata, models)
    targets = target_contracts(config, arm_views, models, plan_identity)
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
        "targets": targets,
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
        checkpoint = resolve(str(target["checkpoint"]))
        optimizer = resolve(str(target["optimizer_state"]))
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

    def count(model_config: dict[str, Any]) -> int:
        model = build_model(
            CausalTransformerConfig(vocab_size=vocab_size, **model_config),
            mx=mx,
            nn=nn,
            state_role_lookup=None,
        )
        return int(parameter_count(model, mlx_utils))

    arm_count = count(config["arm_model"])
    dense_total_count = count(base["model"])
    return {
        "moecot_system": {
            "arm_model": config["arm_model"],
            "arm_parameter_count": arm_count,
            "arm_count": len(ARM_IDS),
            "total_parameter_count": arm_count * len(ARM_IDS),
            "active_parameter_count_per_request": arm_count,
            "router_parameter_count": 0,
            "router_accounting_state": "EXCLUDED_UNTIL_LANGUAGE_ROUTER_IS_TRAINED",
        },
        "dense_total_parameter": {
            "model": base["model"],
            "parameter_count": dense_total_count,
            "active_parameter_count_per_request": dense_total_count,
        },
        "dense_active_parameter": {
            "model": config["arm_model"],
            "parameter_count": arm_count,
            "active_parameter_count_per_request": arm_count,
        },
        "vocab_size": vocab_size,
    }


def target_contracts(
    config: dict[str, Any], arm_views: dict[str, Any], models: dict[str, Any], plan_identity: str
) -> dict[str, Any]:
    root = resolve(str(config["checkpoint_root"]))
    targets: dict[str, Any] = {}
    for target in (*ARM_IDS, *CONTROL_IDS):
        if target in ARM_IDS:
            view = (arm_views.get("arms") or {}).get(target) or {}
            model_key = "moecot_system"
            model = (models.get(model_key) or {}).get("arm_model") or config["arm_model"]
            parameter_count_value = int((models.get(model_key) or {}).get("arm_parameter_count") or 0)
        else:
            view = arm_views.get("mixed_dense_control") or {}
            model = (models.get(target) or {}).get("model") or {}
            parameter_count_value = int((models.get(target) or {}).get("parameter_count") or 0)
        directory = root / target
        targets[target] = {
            "target_id": target,
            "role": "language_arm" if target in ARM_IDS else "dense_control",
            "row_ranges": list(view.get("row_ranges") or []),
            "row_count": sum(int(row["stop"]) - int(row["start"]) for row in view.get("row_ranges") or []),
            "unique_target_positions": int(view.get("target_positions") or 0),
            "model": model,
            "parameter_count": parameter_count_value,
            "estimated_parameter_token_product": parameter_count_value * int(view.get("target_positions") or 0),
            "checkpoint": relative(directory / "weights.npz"),
            "optimizer_state": relative(directory / "optimizer.safetensors"),
            "receipt": relative(directory / "training_receipt.json"),
            "plan_sha256": plan_identity,
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


def execute_targets(
    config: dict[str, Any], plan: dict[str, Any], *, targets: list[str], max_steps: int, resume: bool
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    stage_dir = resolve(str(config["stage_dir"]))
    metadata = read_json(stage_dir / "stage_metadata_v1.json")
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
    results = []
    for target_id in targets:
        target = plan["targets"][target_id]
        results.append(
            train_target(
                config,
                plan,
                target,
                stage=stage,
                max_steps=max_steps,
                resume=resume,
                mx=mx,
                nn=nn,
                optim=optim,
                mlx_utils=mlx_utils,
            )
        )
    gaps = [f"{row['target_id']}:{gap}" for row in results for gap in row.get("hard_gaps") or []]
    return {
        **plan,
        "created_utc": now(),
        "trigger_state": "RED" if gaps else "GREEN",
        "mode": "training_execution",
        "executed_targets": targets,
        "results": results,
        "hard_gaps": gaps,
        "all_requested_targets_complete": bool(results) and all(row.get("complete") for row in results),
        **no_cheat(config),
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
) -> dict[str, Any]:
    target_id = str(target["target_id"])
    inputs = range_view(stage.pretrain_inputs, target["row_ranges"])
    labels = range_view(stage.pretrain_labels, target["row_ranges"])
    mask = range_view(stage.pretrain_mask, target["row_ranges"])
    model = build_model(
        CausalTransformerConfig(vocab_size=int(plan["models"]["vocab_size"]), **target["model"]),
        mx=mx,
        nn=nn,
        state_role_lookup=None,
    )
    observed_parameters = int(parameter_count(model, mlx_utils))
    if observed_parameters != int(target["parameter_count"]):
        raise ValueError("target model parameter identity changed after preregistration")
    checkpoint = resolve(str(target["checkpoint"]))
    optimizer_path = resolve(str(target["optimizer_state"]))
    receipt_path = resolve(str(target["receipt"]))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    planned_steps = required_steps(mask, int(training["batch_size"]), int(target["unique_target_positions"]))
    schedule = build_schedule(optim, mx, training, planned_steps + 128)
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=float(training["weight_decay"]))
    prior_steps = 0
    prior_positions = 0
    prior_checkpoint_hash = ""
    if resume:
        prior = read_json(receipt_path)
        validate_resume(prior, plan, target, checkpoint, optimizer_path)
        model.load_weights(str(checkpoint))
        optimizer.state = mlx_utils.tree_unflatten(list(mx.load(str(optimizer_path)).items()))
        mx.eval(model.parameters(), optimizer.state)
        prior_steps = int(prior.get("optimizer_steps") or 0)
        prior_positions = int(prior.get("optimizer_positions") or 0)
        prior_checkpoint_hash = sha256_file(checkpoint)
    remaining_positions = max(0, int(target["unique_target_positions"]) - prior_positions)
    allowed_steps = max_steps if max_steps else planned_steps + 64
    temporary_checkpoint = checkpoint.with_name("weights.partial.npz")
    heartbeat = checkpoint.parent / "training_heartbeat.json"
    started = time.perf_counter()
    random.seed(int(config["seed"]) + stable_int(target_id) + prior_steps)
    mx.random.seed(int(config["seed"]) + stable_int(target_id) + prior_steps)
    loss_and_grad = nn.value_and_grad(model, causal_loss)
    phase = train_phase(
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
    )
    publish_model(model, checkpoint, temporary_checkpoint)
    publish_optimizer(mx, mlx_utils, optimizer, optimizer_path)
    total_steps = prior_steps + int(phase["optimizer_steps"])
    total_positions = prior_positions + int(phase["target_positions_consumed"])
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
        "optimizer_steps": total_steps,
        "optimizer_positions": total_positions,
        "unique_target_positions": int(target["unique_target_positions"]),
        "complete": total_positions >= int(target["unique_target_positions"]),
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "optimizer_state": relative(optimizer_path),
        "optimizer_state_sha256": sha256_file(optimizer_path),
        "resume": resume,
        "resume_base_checkpoint_sha256": prior_checkpoint_hash,
        "phase": phase,
        "wall_seconds": round(time.perf_counter() - started, 6),
        "energy_joules": None,
        "energy_measurement_state": "NOT_AVAILABLE_FROM_MLX_RUNTIME",
        "capability_claim": "NOT_EVALUATED",
        "hard_gaps": [],
        **no_cheat(config),
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def range_view(array: np.ndarray, ranges: list[dict[str, int]]) -> np.ndarray:
    normalized = [(int(row["start"]), int(row["stop"])) for row in ranges]
    if not normalized:
        raise ValueError("training target has no stage ranges")
    if all(normalized[index][1] == normalized[index + 1][0] for index in range(len(normalized) - 1)):
        return array[normalized[0][0] : normalized[-1][1]]
    return np.concatenate([array[start:stop] for start, stop in normalized], axis=0)


def publish_model(model: Any, checkpoint: Path, temporary: Path) -> None:
    temporary.unlink(missing_ok=True)
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
    if not checkpoint.is_file() or sha256_file(checkpoint) != receipt.get("checkpoint_sha256"):
        faults.append("checkpoint_identity_mismatch")
    if not optimizer.is_file() or sha256_file(optimizer) != receipt.get("optimizer_state_sha256"):
        faults.append("optimizer_identity_mismatch")
    if faults:
        raise ValueError("resume denied: " + ",".join(faults))


def plan_sha256(config: dict[str, Any], metadata: dict[str, Any], models: dict[str, Any]) -> str:
    payload = {
        "config": config,
        "stage_signature": (metadata.get("summary") or {}).get("stage_signature"),
        "arm_views": ((metadata.get("summary") or {}).get("canonical_pretrain_stage") or {}).get("arm_views"),
        "models": models,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("policy") != "project_theseus_moecot_language_arm_training_v1":
        raise ValueError("unexpected MoECOT training policy")
    if config.get("comparison_contract", {}).get("preregistered_before_training") is not True:
        raise ValueError("comparison contract must be preregistered")
    boundaries = config.get("boundaries") or {}
    if any(int(boundaries.get(key) or 0) for key in (
        "public_training_rows_written", "external_inference_calls", "fallback_return_count",
        "templates_renderers_routers_tools_credit",
    )):
        raise ValueError("MoECOT training no-cheat counters must remain zero")
    if boundaries.get("hidden_generalist_fallback") != "forbidden":
        raise ValueError("hidden generalist fallback must remain forbidden")


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
