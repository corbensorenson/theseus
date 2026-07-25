#!/usr/bin/env python3
"""Isolate MLX backward-memory stations on one content-bound KERC train row."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import moecot_language_arm_training as arm  # noqa: E402
import host_resource_safety  # noqa: E402
from standard_causal_transformer_model import (  # noqa: E402
    CausalTransformerConfig,
    build_model,
    parameter_count,
)
from standard_causal_transformer_objectives import (  # noqa: E402
    checkpointed_causal_loss,
    decomposed_checkpointed_causal_loss_and_grad,
)
from standard_causal_transformer_survival import coverage_first_plan  # noqa: E402


POLICY = "project_theseus_kerc_training_memory_preflight_v1"
STATIONS = (
    "decoder_core",
    "encoder_pointer_primary",
    "encoder_pointer_copy",
    "full_kerc_objective",
)
DIAGNOSTIC_STATIONS = STATIONS + (
    "encoder_cross_attention",
    "canonical_train_phase",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_bound_context(
    *,
    row_limit: int,
    coverage_step: int,
    representative_full_objective_row: bool = False,
    maximum_full_objective_row: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], Any, int]:
    if representative_full_objective_row and maximum_full_objective_row:
        raise ValueError("representative and maximum row selectors are exclusive")
    config_path = arm.resolve("configs/moecot_language_arm_training.json")
    config = arm.bind_scale_preregistration(arm.read_json(config_path))
    lease = arm.pretraining_candidate_canary.candidate_lease(
        candidate_id="rdc_kerc_adequacy",
        max_steps=9,
        scratch_checkpoint_root=(
            "runtime/t0a_canaries/rdc_kerc_adequacy/memory_preflight"
        ),
        targets=["english_kerc", "english_surface_control"],
        phase="kernel_english",
        resume=False,
        selected_seed=20260722,
    )
    if lease.get("authorized") is not True:
        raise ValueError("memory preflight candidate lease is not authorized")
    config = arm.bind_candidate_canary_overlay(config, lease)
    plan = arm.build_plan(config, config_path=config_path, candidate_lease=lease)
    if plan.get("trigger_state") == "RED":
        raise ValueError("memory preflight cannot bind the canonical training plan")
    target = plan["targets"]["english_kerc"]
    metadata = arm.read_json(arm.resolve(config["stage_dir"]) / "stage_metadata_v1.json")
    base = arm.read_json(arm.resolve(config["base_config"]))
    stage = arm.materialize_target_supervision(
        config,
        base,
        target,
        metadata=metadata,
        artifact_field="kernel_english_artifacts",
        receipt_policy="project_theseus_moecot_kernel_english_arrays_v1",
        maximum_sequence_tokens=int(
            config["kernel_english_training"]["maximum_sequence_tokens"]
        ),
        objective_filter=tuple(target.get("kernel_english_objectives") or ()),
        bounded_source_row_limit=row_limit,
    )
    coverage = coverage_first_plan(
        stage.kerc_coverage_labels,
        arm.KERC_CANARY_REQUIRED_COVERAGE,
        row_count=len(stage.inputs),
        capacity=9,
        row_costs=tuple(len(np.asarray(row)) for row in stage.inputs),
    )
    if representative_full_objective_row or maximum_full_objective_row:
        eligible = []
        for index in range(len(stage.inputs)):
            if (
                int(np.count_nonzero(np.asarray(stage.loss_mask[index]))) > 0
                and float(stage.kerc_residual_loss_mask[index]) > 0
                and stage.kerc_unit_allocator_rows[index] is not None
                and float(stage.kerc_decision_loss_mask[index]) > 0
                and bool(np.all(np.asarray(stage.kerc_verifier_labels[index]) == 1.0))
            ):
                eligible.append(
                    (
                        len(np.asarray(stage.inputs[index])),
                        index,
                    )
                )
        if not eligible:
            raise ValueError("KERC full-objective representative row is unavailable")
        # A resource-entry representative is the deterministic population
        # median, not the maximum-length stress case.  The maximum remains a
        # separate long-context qualification target and cannot silently stand
        # in for ordinary KERC training cost.
        eligible.sort(key=lambda row: (row[0], row[1]))
        row_index = int(
            eligible[-1 if maximum_full_objective_row else len(eligible) // 2][1]
        )
    else:
        if not 1 <= coverage_step <= len(coverage["selected_indices"]):
            raise ValueError("KERC preflight coverage step is outside the selected prefix")
        row_index = int(coverage["selected_indices"][coverage_step - 1])
    return config, target, stage, row_index


def execute_station(
    station: str,
    *,
    row_limit: int,
    coverage_step: int,
    query_chunk_size: int,
    key_chunk_size: int,
    compact_partitions: bool,
    progress_path: Path | None = None,
    representative_full_objective_row: bool = False,
    maximum_full_objective_row: bool = False,
    decomposed_objective_backward: bool = False,
    token_loss_position_chunk_size: int = 0,
) -> dict[str, Any]:
    if station not in DIAGNOSTIC_STATIONS:
        raise ValueError(f"unknown memory station: {station}")
    if decomposed_objective_backward and station != "full_kerc_objective":
        raise ValueError(
            "decomposed objective backward is valid only for the full KERC objective"
        )
    if token_loss_position_chunk_size and not decomposed_objective_backward:
        raise ValueError(
            "token-loss position chunking requires decomposed objective backward"
        )
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils as mlx_utils

    def progress(stage_name: str) -> None:
        if progress_path is None:
            return
        arm.write_json_atomic(
            progress_path,
            {
                "policy": POLICY,
                "created_utc": datetime.now(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                ),
                "stage": stage_name,
                "mlx_active_memory_bytes": int(mx.get_active_memory()),
                "mlx_cache_memory_bytes": int(mx.get_cache_memory()),
                "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
            },
        )

    config, target, stage, row_index = build_bound_context(
        row_limit=row_limit,
        coverage_step=coverage_step,
        representative_full_objective_row=representative_full_objective_row,
        maximum_full_objective_row=maximum_full_objective_row,
    )
    progress("bound_context_ready")
    seed = 20260722
    mx.random.seed(seed)
    trained_vocab_size = int(target["vocab_size"])
    base = arm.read_json(arm.resolve(config["base_config"]))
    metadata = arm.read_json(arm.resolve(config["stage_dir"]) / "stage_metadata_v1.json")
    copy_lookup = arm.build_source_to_target_lookup(
        base,
        metadata,
        vocab_size=trained_vocab_size,
        identity_ranges=arm.target_copy_identity_ranges(target),
    )
    model_config = CausalTransformerConfig(
        vocab_size=trained_vocab_size, **target["model"]
    )
    master_model = build_model(
        model_config,
        mx=mx,
        nn=nn,
        source_to_target_lookup=copy_lookup,
        rope_kernel=str(config["training"].get("training_rope_kernel") or "mlx_fast"),
        gradient_checkpointing=False,
    )
    progress("master_model_constructed")
    mx.eval(master_model.parameters())
    progress("master_model_materialized")
    model = build_model(
        model_config,
        mx=mx,
        nn=nn,
        source_to_target_lookup=copy_lookup,
        rope_kernel=str(config["training"].get("training_rope_kernel") or "mlx_fast"),
        gradient_checkpointing=True,
        attention_query_chunk_size=query_chunk_size,
        attention_key_chunk_size=key_chunk_size,
        compact_encoder_decoder_partitions=compact_partitions,
    )
    progress("compute_model_constructed")
    model.load_weights(list(mlx_utils.tree_flatten(master_model.parameters())), strict=True)
    progress("compute_model_weights_bound")
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters(), master_model.parameters())
    progress("compute_and_master_materialized")
    if station == "encoder_cross_attention":
        def generator_only_output(
            hidden: Any,
            _source_memory: Any,
            _source_mask: Any,
            _source_copy_ids: Any,
            generator_logits: Any | None = None,
            _pointer_access: Any | None = None,
        ) -> tuple[Any, None]:
            return (
                generator_logits
                if generator_logits is not None
                else model.token_embedding.as_linear(hidden),
                None,
            )

        model.output_logits = generator_only_output

    selected = [row_index]
    input_row = np.asarray(stage.inputs[selected])
    label_row = np.asarray(stage.labels[selected])
    mask_row = np.asarray(stage.loss_mask[selected])
    active = np.flatnonzero((input_row != 0) | (label_row != 0) | (mask_row != 0))
    width = int(active[-1] % input_row.shape[1] + 1) if len(active) else 1
    x = mx.array(input_row[:, :width], dtype=mx.int32)
    y = mx.array(label_row[:, :width], dtype=mx.int32)
    mask = mx.array(mask_row[:, :width], dtype=mx.float32)
    progress("representative_row_materialized")

    kwargs: dict[str, Any] = {
        "source_conditioning": station != "decoder_core",
    }
    if station in {
        "decoder_core",
        "encoder_cross_attention",
        "encoder_pointer_primary",
    }:
        model.copy_auxiliary_loss_weight = 0.0
    if station in {"full_kerc_objective", "canonical_train_phase"}:
        packed = arm.pack_kerc_unit_allocator_batch(
            [stage.kerc_unit_allocator_rows[row_index]]
        )
        kwargs.update(
            {
                "kerc_verifier_labels": mx.array(
                    stage.kerc_verifier_labels[row_index : row_index + 1],
                    dtype=mx.float32,
                ),
                "kerc_verifier_weight": float(
                    config["kernel_english_training"]["verifier_auxiliary_weight"]
                ),
                "kerc_decision_labels": mx.array(
                    stage.kerc_decision_labels[row_index : row_index + 1],
                    dtype=mx.int32,
                ),
                "kerc_decision_weight": float(
                    config["kernel_english_training"]["decision_auxiliary_weight"]
                ),
                "kerc_decision_loss_mask": mx.array(
                    stage.kerc_decision_loss_mask[row_index : row_index + 1],
                    dtype=mx.float32,
                ),
            }
        )
        if packed is not None:
            kwargs.update(
                {
                    "kerc_unit_residual_labels": mx.array(
                        packed["labels"], dtype=mx.int32
                    ),
                    "kerc_unit_residual_weight": float(
                        config["kernel_english_training"][
                            "unit_residual_auxiliary_weight"
                        ]
                    ),
                    "kerc_unit_residual_loss_mask": mx.array(
                        packed["loss_mask"], dtype=mx.float32
                    ),
                    "kerc_unit_confidence_targets": mx.array(
                        packed["confidence_targets"], dtype=mx.float32
                    ),
                    "kerc_unit_byte_ids": mx.array(
                        packed["byte_ids"], dtype=mx.int32
                    ),
                    "kerc_unit_byte_offsets": mx.array(
                        packed["byte_offsets"], dtype=mx.int64
                    ),
                    "kerc_unit_kind_ids": mx.array(
                        packed["kind_ids"], dtype=mx.int32
                    ),
                    "kerc_unit_candidate_features": mx.array(
                        packed["candidate_features"], dtype=mx.float32
                    ),
                    "kerc_unit_mask": mx.array(
                        packed["unit_mask"], dtype=mx.float32
                    ),
                    "kerc_unit_hard_block_mask": mx.array(
                        packed["hard_block_mask"], dtype=mx.bool_
                    ),
                }
            )

    mx.set_cache_limit(512 * 1024 * 1024)
    mx.clear_cache()
    mx.reset_peak_memory()
    baseline_active = int(mx.get_active_memory())
    if station == "canonical_train_phase":
        import mlx.optimizers as optim

        progress_row = np.asarray(stage.mask[[row_index]])[:, :width]
        phase = arm.train_phase(
            model,
            optim.AdamW(learning_rate=3e-4, weight_decay=0.01),
            nn.value_and_grad(model, checkpointed_causal_loss),
            input_row[:, :width],
            label_row[:, :width],
            mask_row[:, :width],
            progress_mask=progress_row,
            ordered_plan_loss_weight=1.0,
            sample_weights=None,
            plan_labels=None,
            plan_label_mode="none",
            plan_auxiliary_weight=0.0,
            plan_shuffle_seed=0,
            plan_loss_mode="binary_multilabel",
            plan_slot_count=0,
            plan_factor_group_sizes=(),
            kerc_unit_allocator_rows=(stage.kerc_unit_allocator_rows[row_index],),
            kerc_unit_batch_packer=arm.pack_kerc_unit_allocator_batch,
            kerc_unit_residual_weight=float(
                config["kernel_english_training"]["unit_residual_auxiliary_weight"]
            ),
            kerc_unit_require_two_classes=False,
            kerc_verifier_labels=stage.kerc_verifier_labels[[row_index]],
            kerc_verifier_weight=float(
                config["kernel_english_training"]["verifier_auxiliary_weight"]
            ),
            kerc_verifier_require_both_classes=False,
            kerc_decision_labels=stage.kerc_decision_labels[[row_index]],
            kerc_decision_weight=float(
                config["kernel_english_training"]["decision_auxiliary_weight"]
            ),
            kerc_decision_class_count=len(arm.ANSWER_DISPOSITION_ORDER),
            kerc_decision_require_two_classes=False,
            kerc_decision_loss_mask=stage.kerc_decision_loss_mask[[row_index]],
            phase_name="kerc_memory_preflight",
            target_positions=max(1, int(progress_row.sum())),
            batch_size=1,
            gradient_clip=1.0,
            seed=seed,
            max_steps=1,
            checkpoint=ROOT / "runtime/t0a_canaries/rdc_kerc_adequacy/memory_preflight/unused.safetensors",
            checkpoint_every=99,
            heartbeat=ROOT / "runtime/t0a_canaries/rdc_kerc_adequacy/memory_preflight/heartbeat.json",
            global_step_offset=0,
            mx=mx,
            optim=optim,
            training_step_mode="eager",
            master_model=master_model,
            compute_dtype_name="bfloat16",
            clear_device_cache_after_step=True,
            transactional_eager_step=True,
        )
        return {
            "policy": POLICY,
            "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "trigger_state": "GREEN",
            "station": station,
            "mlx_version": __import__("importlib.metadata").metadata.version("mlx"),
            "python_executable": sys.executable,
            "seed": seed,
            "target_id": target["target_id"],
            "parameter_count": int(parameter_count(model, mlx_utils)),
            "row_index": row_index,
            "row_sha256": sha256_bytes(input_row[:, :width].astype(np.int32).tobytes()),
            "sequence_width": width,
            "memory_execution_policy": {
                "row_limit": row_limit,
                "coverage_step": coverage_step,
                "attention_query_chunk_size": query_chunk_size,
                "attention_key_chunk_size": key_chunk_size,
                "compact_encoder_decoder_partitions": compact_partitions,
                "representative_full_objective_row": representative_full_objective_row,
                "full_objective_row_selection": (
                    "length_population_median"
                    if representative_full_objective_row
                    else "coverage_prefix_step"
                ),
            },
            "phase": phase,
            "mlx_peak_memory_bytes": int(
                phase["mlx_peak_memory_bytes_maximum"]
            ),
            "capability_credit": "NONE_RESOURCE_PREFLIGHT_ONLY",
            "public_training_rows": 0,
            "public_evaluation_rows": 0,
            "external_inference_calls": 0,
            "fallback_template_router_tool_credit": 0,
        }
    progress("backward_entry")
    if decomposed_objective_backward:
        loss, gradients = decomposed_checkpointed_causal_loss_and_grad(
            model,
            x,
            y,
            mask,
            mx,
            nn,
            token_loss_position_chunk_size=token_loss_position_chunk_size,
            **kwargs,
        )
    else:
        loss_and_grad = nn.value_and_grad(model, checkpointed_causal_loss)
        loss, gradients = loss_and_grad(model, x, y, mask, mx, nn, **kwargs)
    mx.eval(loss, gradients)
    progress("backward_materialized")
    flat_gradients = dict(mlx_utils.tree_flatten(gradients))
    gradient_mass = sum(
        float(mx.sum(mx.abs(value.astype(mx.float32))).item())
        for value in flat_gradients.values()
    )
    report = {
        "policy": POLICY,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trigger_state": "GREEN",
        "station": station,
        "mlx_version": __import__("importlib.metadata").metadata.version("mlx"),
        "python_executable": sys.executable,
        "seed": seed,
        "target_id": target["target_id"],
        "parameter_count": int(parameter_count(model, mlx_utils)),
        "row_index": row_index,
        "row_sha256": sha256_bytes(input_row[:, :width].astype(np.int32).tobytes()),
        "sequence_width": width,
        "memory_execution_policy": {
            "row_limit": row_limit,
            "coverage_step": coverage_step,
            "attention_query_chunk_size": query_chunk_size,
            "attention_key_chunk_size": key_chunk_size,
            "compact_encoder_decoder_partitions": compact_partitions,
            "representative_full_objective_row": representative_full_objective_row,
            "maximum_full_objective_row": maximum_full_objective_row,
            "full_objective_row_selection": (
                "length_population_maximum"
                if maximum_full_objective_row
                else "length_population_median"
                if representative_full_objective_row
                else "coverage_prefix_step"
            ),
            "objective_backward": (
                "serial_additive_fp32_gradient_accumulation_v1"
                if decomposed_objective_backward
                else "monolithic_checkpointed_scalar_v1"
            ),
            "token_loss_position_chunk_size": int(
                token_loss_position_chunk_size
            ),
        },
        "loss": float(loss.item()),
        "gradient_tensor_count": len(flat_gradients),
        "gradient_l1_mass": gradient_mass,
        "mlx_active_memory_bytes_before_backward": baseline_active,
        "mlx_active_memory_bytes_after_backward": int(mx.get_active_memory()),
        "mlx_cache_memory_bytes_after_backward": int(mx.get_cache_memory()),
        "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
        "objective_gradient_checkpointing": True,
        "objective_gradient_decomposition": bool(
            decomposed_objective_backward
        ),
        "objective_gradient_accumulation_dtype": (
            "float32" if decomposed_objective_backward else "native_gradient_dtype"
        ),
        "layer_gradient_checkpointing": True,
        "compute_dtype": "bfloat16",
        "fp32_master_resident": True,
        "capability_credit": "NONE_RESOURCE_PREFLIGHT_ONLY",
        "public_training_rows": 0,
        "public_evaluation_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit": 0,
    }
    return report


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if [row.get("station") for row in rows] != list(STATIONS):
        raise ValueError("memory station panel is incomplete or out of order")
    identities = {
        (
            row["row_sha256"],
            row["sequence_width"],
            row["parameter_count"],
            json.dumps(row["memory_execution_policy"], sort_keys=True),
        )
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError("memory station panel changed row or model identity")
    return {
        "policy": POLICY,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trigger_state": "GREEN",
        "support_state": "exact_row_isolated_backward_resource_diagnosis",
        "stations": rows,
        "memory_execution_policy": rows[0]["memory_execution_policy"],
        "peak_memory_mib_by_station": {
            row["station"]: round(row["mlx_peak_memory_bytes"] / (1024**2), 3)
            for row in rows
        },
        "capability_credit": "NONE_RESOURCE_PREFLIGHT_ONLY",
        "scientific_falsification_claimed": False,
        "public_training_rows": 0,
        "public_evaluation_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", choices=DIAGNOSTIC_STATIONS)
    parser.add_argument(
        "--out", default="reports/kerc_training_memory_preflight.json"
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=int(os.environ.get("THESEUS_KERC_PREFLIGHT_ROW_LIMIT", "256")),
    )
    parser.add_argument(
        "--coverage-step",
        type=int,
        default=int(os.environ.get("THESEUS_KERC_PREFLIGHT_COVERAGE_STEP", "1")),
    )
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=int(os.environ.get("THESEUS_KERC_PREFLIGHT_QUERY_CHUNK", "0")),
    )
    parser.add_argument(
        "--key-chunk-size",
        type=int,
        default=int(os.environ.get("THESEUS_KERC_PREFLIGHT_KEY_CHUNK", "0")),
    )
    parser.add_argument(
        "--compact-partitions",
        action="store_true",
        default=(
            os.environ.get("THESEUS_KERC_PREFLIGHT_COMPACT_PARTITIONS", "0") == "1"
        ),
    )
    parser.add_argument("--representative-full-objective-row", action="store_true")
    parser.add_argument("--maximum-full-objective-row", action="store_true")
    parser.add_argument("--decomposed-objective-backward", action="store_true")
    parser.add_argument("--token-loss-position-chunk-size", type=int, default=0)
    args = parser.parse_args()
    if args.row_limit <= 0 or args.coverage_step <= 0:
        parser.error("row limit and coverage step must be positive")
    if args.query_chunk_size < 0 or args.key_chunk_size < 0:
        parser.error("attention chunk sizes must be nonnegative")
    if args.token_loss_position_chunk_size < 0:
        parser.error("token-loss position chunk size must be nonnegative")
    output = arm.resolve(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.station:
        if os.environ.get("THESEUS_GUARDED_ACCELERATOR_CHILD") != "1":
            raise host_resource_safety.HostResourceSafetyFault(
                "direct MLX station execution is denied; use the guarded panel runner"
            )
        report = execute_station(
            args.station,
            row_limit=args.row_limit,
            coverage_step=args.coverage_step,
            query_chunk_size=args.query_chunk_size,
            key_chunk_size=args.key_chunk_size,
            compact_partitions=args.compact_partitions,
            progress_path=output.with_suffix(output.suffix + ".progress.json"),
            representative_full_objective_row=args.representative_full_objective_row,
            maximum_full_objective_row=args.maximum_full_objective_row,
            decomposed_objective_backward=args.decomposed_objective_backward,
            token_loss_position_chunk_size=args.token_loss_position_chunk_size,
        )
    else:
        contract = arm.pretraining_candidate_canary.load_contract()
        safety_policy = host_resource_safety.policy_from_mapping(
            contract["host_safety_policy"], maximum_wall_seconds=1800.0
        )
        rows = []
        for station in STATIONS:
            child_output = output.with_name(f"{output.stem}.{station}{output.suffix}")
            process = host_resource_safety.run_guarded(
                [
                    sys.executable,
                    __file__,
                    "--station",
                    station,
                    "--out",
                    str(child_output),
                    "--row-limit",
                    str(args.row_limit),
                    "--coverage-step",
                    str(args.coverage_step),
                    "--query-chunk-size",
                    str(args.query_chunk_size),
                    "--key-chunk-size",
                    str(args.key_chunk_size),
                    *(["--compact-partitions"] if args.compact_partitions else []),
                    *(
                        ["--maximum-full-objective-row"]
                        if args.maximum_full_objective_row
                        else []
                    ),
                ],
                cwd=ROOT,
                policy=safety_policy,
                env={"THESEUS_GUARDED_ACCELERATOR_CHILD": "1"},
            )
            guard_output = child_output.with_name(
                f"{child_output.stem}.host_resource_safety{child_output.suffix}"
            )
            arm.write_json_atomic(guard_output, process.receipt)
            if process.returncode or process.receipt.get("passed") is not True:
                raise RuntimeError(
                    f"memory station failed:{station}:{process.returncode}:"
                    f"guard={process.receipt.get('fault')}:{process.stderr[-2000:]}"
                )
            row = json.loads(child_output.read_text(encoding="utf-8"))
            row["host_resource_safety_receipt"] = process.receipt
            rows.append(row)
        report = aggregate(rows)
    arm.write_json_atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
