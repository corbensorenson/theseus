#!/usr/bin/env python3
"""Qualify capability-critical acceleration against exact reference behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import moecot_language_arm_training as training  # noqa: E402
import neural_seed_campaign_controller as campaign_controller  # noqa: E402
import neural_seed_resident_runtime as resident_runtime  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/moecot_language_arm_training.json"
DEFAULT_PACKET = ROOT / "reports/private_functional_utility_candidate_packet.json"
DEFAULT_TRAINING_REPORT = (
    ROOT / "reports/moecot_language_arm_training_acceleration_500step_qualification.json"
)
DEFAULT_LEARNING_CURVE = (
    ROOT / "reports/moecot_57m_shared_trunk_learning_curve_step3000.json"
)
DEFAULT_OUT = ROOT / "reports/resource_acceleration_qualification.json"
DEFAULT_MARKDOWN = ROOT / "reports/resource_acceleration_qualification.md"
DEFAULT_ASSISTANT_CONFIG = ROOT / "configs/theseus_assistant_runtime.json"
ACCELERATION_KEYS = {
    "beam_advance",
    "logit_filter",
    "preprune_beam_expansions",
    "prompt_prefill_seconds",
}
MAX_FINAL_LOSS_ABSOLUTE_DELTA = 2e-6
MAX_PARAMETER_ABSOLUTE_DELTA = 5e-6
MAX_PARAMETER_RELATIVE_L2_DELTA = 1e-6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--packet", default=relative(DEFAULT_PACKET))
    parser.add_argument("--training-report", default=relative(DEFAULT_TRAINING_REPORT))
    parser.add_argument("--learning-curve", default=relative(DEFAULT_LEARNING_CURVE))
    parser.add_argument("--out", default=relative(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=relative(DEFAULT_MARKDOWN))
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--training-pair-steps", type=int, default=24)
    parser.add_argument("--training-pair-repetitions", type=int, default=3)
    parser.add_argument("--compiled-microbatch-size", type=int, default=4)
    parser.add_argument("--precision-pair-steps", type=int, default=8)
    parser.add_argument("--precision-pair-repetitions", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.sample_count < 1:
        parser.error("--sample-count must be positive")
    if args.max_tokens < 0:
        parser.error("--max-tokens cannot be negative")
    if args.training_pair_steps < 2:
        parser.error("--training-pair-steps must be at least two")
    if args.training_pair_repetitions < 2:
        parser.error("--training-pair-repetitions must be at least two")
    if args.compiled_microbatch_size < 1:
        parser.error("--compiled-microbatch-size must be positive")
    if args.precision_pair_steps < 2:
        parser.error("--precision-pair-steps must be at least two")
    if args.precision_pair_repetitions < 2:
        parser.error("--precision-pair-repetitions must be at least two")

    report = qualify(
        config_path=resolve(args.config),
        packet_path=resolve(args.packet),
        training_report_path=resolve(args.training_report),
        learning_curve_path=resolve(args.learning_curve),
        sample_count=args.sample_count,
        max_tokens=args.max_tokens,
        training_pair_steps=args.training_pair_steps,
        training_pair_repetitions=args.training_pair_repetitions,
        compiled_microbatch_size=args.compiled_microbatch_size,
        precision_pair_steps=args.precision_pair_steps,
        precision_pair_repetitions=args.precision_pair_repetitions,
        execute=args.execute,
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report_summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] != "RED" else 2


def qualify(
    *,
    config_path: Path,
    packet_path: Path,
    training_report_path: Path,
    learning_curve_path: Path,
    sample_count: int,
    max_tokens: int,
    training_pair_steps: int,
    training_pair_repetitions: int,
    compiled_microbatch_size: int,
    precision_pair_steps: int,
    precision_pair_repetitions: int,
    execute: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    process_resources_before = process_resource_receipt()
    config = training.bind_scale_preregistration(read_json(config_path))
    plan = training.build_plan(config, config_path=config_path)
    decision_control = campaign_controller.build_campaign_status(
        scale_config_path=campaign_controller.DEFAULT_SCALE_CONFIG,
        training_config_path=config_path,
        review_dir=campaign_controller.DEFAULT_REVIEW_DIR,
    )
    packet = read_json(packet_path)
    packet_rows = packet.get("rows") if isinstance(packet.get("rows"), list) else []
    gaps = validate_packet(packet_rows)
    selected = select_qualification_rows(packet_rows, sample_count)
    target = (plan.get("targets") or {}).get(training.SHARED_TRUNK_ID) or {}
    receipt_path = resolve(str(target.get("receipt") or ""))
    receipt = read_json(receipt_path) if receipt_path.is_file() else {}
    checkpoint = resolve(str(receipt.get("checkpoint") or target.get("checkpoint") or ""))
    optimizer = resolve(str(receipt.get("optimizer_state") or target.get("optimizer_state") or ""))
    migration = None
    if not checkpoint.is_file():
        gaps.append("shared_trunk_checkpoint_missing")
    if not optimizer.is_file():
        gaps.append("shared_trunk_optimizer_state_missing")
    if checkpoint.is_file() and optimizer.is_file():
        try:
            migration = training.validate_resume(
                receipt,
                plan,
                target,
                checkpoint,
                optimizer,
            )
        except ValueError as exc:
            gaps.append(f"checkpoint_lineage_invalid:{exc}")
    training_evidence = training_summary(training_report_path)
    learning_evidence = learning_summary(learning_curve_path)
    if training_evidence.get("state") != "READY":
        gaps.append("training_acceleration_evidence_missing")
    if learning_evidence.get("state") != "READY":
        gaps.append("private_dev_learning_evidence_missing")

    inference = {
        "state": "NOT_EXECUTED",
        "case_count": len(selected),
        "reference_route": reference_route(),
        "optimized_route": optimized_route(),
        "minimum_uncached_decode_speedup": 2.0,
    }
    load = {"state": "NOT_EXECUTED"}
    checkpoint_storage = {"state": "NOT_EXECUTED"}
    assistant_refresh = {"state": "NOT_EXECUTED"}
    resident = {"trigger_state": "NOT_EXECUTED"}
    if execute and not gaps:
        training_pair = run_training_pair_qualification(
            config=config,
            plan=plan,
            target=target,
            checkpoint=checkpoint,
            optimizer_path=optimizer,
            steps=training_pair_steps,
            repetitions=training_pair_repetitions,
            compiled_microbatch_size=compiled_microbatch_size,
        )
        training_evidence["paired_canary"] = training_pair
        if training_pair.get("state") != "GREEN":
            gaps.append("same_semantics_training_speedup_below_2x")
        precision = run_precision_pair_qualification(
            config=config,
            plan=plan,
            target=target,
            checkpoint=checkpoint,
            optimizer_path=optimizer,
            steps=precision_pair_steps,
            repetitions=precision_pair_repetitions,
            compiled_microbatch_size=compiled_microbatch_size,
        )
        training_evidence["precision_autotune"] = precision
        if precision.get("state") == "RED":
            gaps.append("mixed_precision_qualification_fault")
        checkpoint_storage = run_checkpoint_storage_qualification(checkpoint)
        if checkpoint_storage.get("exact_tensor_parity") is not True:
            gaps.append("checkpoint_format_exact_tensor_parity_failed")
        assistant_refresh = run_assistant_refresh_qualification(DEFAULT_ASSISTANT_CONFIG)
        if assistant_refresh.get("exact_refresh_identity_parity") is not True:
            gaps.append("assistant_refresh_cache_identity_parity_failed")
        if float(assistant_refresh.get("speedup") or 0.0) < 5.0:
            gaps.append("assistant_refresh_cache_speedup_below_5x")
        load, inference = run_inference_qualification(
            config=config,
            plan=plan,
            target=target,
            checkpoint=checkpoint,
            rows=selected,
            max_tokens=max_tokens,
        )
        if inference["exact_parity_case_count"] != inference["case_count"]:
            gaps.append("optimized_decode_exact_parity_failed")
        if float(inference.get("uncached_aggregate_speedup") or 0.0) < 2.0:
            gaps.append("optimized_decode_speedup_below_2x")
        resident = resident_runtime.qualify_resident_runtime(
            config_path=config_path,
            packet_path=packet_path,
            max_tokens=max(2, min(8, max_tokens or 8)),
        )
        if resident.get("trigger_state") != "GREEN":
            gaps.append("resident_runtime_qualification_failed")
        if resident.get("exact_output_and_token_parity") is not True:
            gaps.append("resident_runtime_output_or_token_parity_failed")
        if float(resident.get("repeated_prompt_speedup") or 0.0) < 5.0:
            gaps.append("resident_repeated_prompt_speedup_below_5x")

    state = "RED" if gaps else "GREEN" if execute else "READY"
    return {
        "policy": "project_theseus_resource_acceleration_qualification_v1",
        "created_utc": now(),
        "trigger_state": state,
        "mode": "executed" if execute else "plan",
        "hardware": hardware_receipt(),
        "process_resources": process_resource_delta(
            process_resources_before, process_resource_receipt()
        ),
        "config": artifact(config_path),
        "packet": artifact(packet_path),
        "training_report": artifact(training_report_path),
        "learning_curve": artifact(learning_curve_path),
        "plan_sha256": plan.get("plan_sha256"),
        "checkpoint_lineage": {
            "receipt": relative(receipt_path),
            "receipt_plan_sha256": receipt.get("plan_sha256"),
            "checkpoint": relative(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint) if checkpoint.is_file() else "",
            "optimizer_state": relative(optimizer),
            "optimizer_state_sha256": file_sha256(optimizer) if optimizer.is_file() else "",
            "optimizer_steps": int(receipt.get("optimizer_steps") or 0),
            "optimizer_positions": int(receipt.get("optimizer_positions") or 0),
            "registered_migration": migration,
        },
        "selection": {
            "policy": "arm_cover_then_case_id_hash_v1",
            "candidate_count": len(packet_rows),
            "sample_count": len(selected),
            "case_ids": [str(row["case_id"]) for row in selected],
            "arm_counts": count_by(selected, "arm_id"),
            "prompt_or_target_text_retained": False,
        },
        "training": training_evidence,
        "private_dev_learning": learning_evidence,
        "architecture_decision_control": decision_control,
        "checkpoint_storage": checkpoint_storage,
        "assistant_context_refresh": assistant_refresh,
        "checkpoint_load": load,
        "inference": inference,
        "resident_runtime": resident,
        "adoption": {
            "mlx_compiled_fixed_width_microbatch": (
                "QUALIFIED"
                if ((training_evidence.get("paired_canary") or {}).get("state") == "GREEN")
                else "SEMANTICS_QUALIFIED_SPEED_TARGET_PENDING"
                if training_evidence.get("same_semantics") is True
                else "REVIEW"
            ),
            "bf16": (
                "QUALIFIED_BFLOAT16_COMPUTE_FP32_MASTER"
                if ((training_evidence.get("precision_autotune") or {}).get("adopt"))
                else "NOT_ADOPTED"
                if execute
                else "PENDING_MIXED_PRECISION_QUALIFICATION"
            ),
            "batched_beam_device_filter_preprune": (
                "QUALIFIED"
                if inference.get("exact_parity_case_count") == inference.get("case_count")
                and float(inference.get("uncached_aggregate_speedup") or 0.0) >= 2.0
                else "NOT_QUALIFIED"
            ),
            "kerc_batched_beam_device_filter_preprune": (
                "PARITY_QUALIFIED_FULL_PIPELINE_THROUGHPUT_PENDING"
            ),
            "wide_ragged_batching": "DEFERRED_KERC_ONLY_NOT_PRACTICAL_TRUNK_BOTTLENECK",
            "preallocated_kv_cache": "REJECTED_NO_MATERIAL_SPEEDUP",
            "model_checkpoint_format": checkpoint_storage.get(
                "adoption_recommendation", "PENDING_MEASUREMENT"
            ),
            "assistant_content_bound_refresh_cache": (
                "QUALIFIED"
                if assistant_refresh.get("exact_refresh_identity_parity") is True
                and float(assistant_refresh.get("speedup") or 0.0) >= 5.0
                else "NOT_QUALIFIED"
            ),
            "resident_model_prefix_and_completion_cache": (
                "QUALIFIED_EVALUATION_RUNTIME_SERVING_PENDING_CAPABILITY"
                if resident.get("trigger_state") == "GREEN"
                and resident.get("exact_output_and_token_parity") is True
                and float(resident.get("repeated_prompt_speedup") or 0.0) >= 5.0
                else "NOT_QUALIFIED"
            ),
            "continuous_multi_request_batching": (
                "QUALIFIED_EVALUATION_RUNTIME_SERVING_PENDING_CAPABILITY"
                if ((resident.get("continuous_batching") or {}).get("state"))
                == "QUALIFIED"
                and (resident.get("continuous_batching") or {}).get(
                    "exact_output_state_reason_and_token_parity"
                )
                is True
                else "NOT_QUALIFIED"
            ),
            "evidence_efficient_successive_halving": (
                "EMPIRICALLY_QUALIFIED"
                if decision_control.get("target_speedup_empirically_proven") is True
                else "CONTRACT_READY_REVIEW_EVIDENCE_PENDING"
                if decision_control.get("trigger_state") == "READY"
                else "NOT_QUALIFIED"
            ),
        },
        "boundaries": {
            "generator_visible_fields": ["case_id", "arm_id", "prompt"],
            "target_or_verifier_visible_to_generator": False,
            "public_benchmark_rows_read": 0,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "templates_renderers_routers_tools_credit": 0,
            "fallback_return_count": 0,
            "quality_or_verification_skipped_for_speed": False,
        },
        "hard_gaps": gaps,
        "remaining_gaps": [
            *(
                []
                if decision_control.get("target_speedup_empirically_proven") is True
                else ["first_architecture_decision_10x_empirical_proof_pending"]
            ),
            *(
                []
                if resident.get("trigger_state") == "GREEN"
                else ["resident_model_and_prefix_reuse_qualification_pending"]
            ),
            "production_serving_capability_qualification_pending",
            *(
                []
                if ((resident.get("continuous_batching") or {}).get("state"))
                == "QUALIFIED"
                else ["continuous_multi_request_batching_pending"]
            ),
            "system_energy_measurement_unavailable",
        ],
        "wall_seconds": round(time.perf_counter() - started, 6),
        "claim_scope": (
            "Same-process private prompt-only acceleration qualification; this is not a "
            "capability, public-transfer, or model-quality claim."
        ),
    }


def run_assistant_refresh_qualification(config_path: Path) -> dict[str, Any]:
    """Measure exact content-bound refresh reuse through the canonical assistant route."""

    import theseus_assistant_runtime as assistant

    config = read_json(config_path)
    with tempfile.TemporaryDirectory(prefix="theseus-assistant-refresh-") as directory:
        config["context_refresh_cache"] = str(Path(directory) / "cache.json")
        cold_started = time.perf_counter()
        cold = assistant.refresh_context(config)
        cold_seconds = time.perf_counter() - cold_started
        warm_started = time.perf_counter()
        warm = assistant.refresh_context(config)
        warm_seconds = time.perf_counter() - warm_started
    cold_ids = [str(row.get("id") or "") for row in cold]
    warm_ids = [str(row.get("id") or "") for row in warm]
    exact = bool(
        cold_ids
        and cold_ids == warm_ids
        and all(row.get("returncode") == 0 for row in cold)
        and all(row.get("returncode") == 0 for row in warm)
        and all(row.get("cache_state") == "MISS" for row in cold)
        and all(row.get("cache_state") == "HIT" for row in warm)
        and all(row.get("input_fingerprint") for row in warm)
    )
    speedup = cold_seconds / max(1e-12, warm_seconds)
    return {
        "policy": "project_theseus_assistant_refresh_acceleration_pair_v1",
        "state": "GREEN" if exact and speedup >= 5.0 else "RED",
        "config": artifact(config_path),
        "command_ids": cold_ids,
        "command_count": len(cold_ids),
        "cold_seconds": round(cold_seconds, 6),
        "warm_seconds": round(warm_seconds, 6),
        "speedup": round(speedup, 6),
        "cold_command_runtime_ms": sum(int(row.get("runtime_ms") or 0) for row in cold),
        "warm_cache_lookup_ms": sum(int(row.get("runtime_ms") or 0) for row in warm),
        "cold_commands": [
            {
                "id": row.get("id"),
                "runtime_ms": int(row.get("runtime_ms") or 0),
                "cache_state": row.get("cache_state"),
                "returncode": row.get("returncode"),
            }
            for row in cold
        ],
        "warm_commands": [
            {
                "id": row.get("id"),
                "runtime_ms": int(row.get("runtime_ms") or 0),
                "cache_state": row.get("cache_state"),
                "returncode": row.get("returncode"),
            }
            for row in warm
        ],
        "cold_miss_count": sum(row.get("cache_state") == "MISS" for row in cold),
        "warm_hit_count": sum(row.get("cache_state") == "HIT" for row in warm),
        "exact_refresh_identity_parity": exact,
        "freshness_window_seconds": sorted(
            {
                int(((item.get("cache") or {}).get("ttl_seconds") or 0))
                for item in config.get("context_refresh_commands", [])
                if isinstance(item, dict)
            }
        ),
        "fail_closed_invalidation": [
            "command_identity",
            "input_content_hashes",
            "output_content_hashes",
            "freshness_window",
            "prior_success",
        ],
        "governance_or_verification_skipped": False,
        "deterministic_tool_refresh_mode": "qualification_bound_runtime_refresh",
        "claim_scope": (
            "Repeated unchanged assistant context refresh only; generation and task-specific "
            "tool execution remain outside this cache."
        ),
    }


def run_checkpoint_storage_qualification(
    checkpoint: Path,
    *,
    load_repetitions: int = 3,
) -> dict[str, Any]:
    """Compare the live NPZ contract with safetensors without mutating durable state."""

    import mlx.core as mx

    if load_repetitions < 1:
        raise ValueError("load_repetitions must be positive")
    source = mx.load(str(checkpoint))
    mx.eval(*source.values())
    source_manifest = tensor_mapping_manifest(source)
    formats: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="theseus-checkpoint-qualification-") as directory:
        root = Path(directory)
        paths = {
            "npz": root / "weights.npz",
            "safetensors": root / "weights.safetensors",
        }
        for format_id, path in paths.items():
            temporary = path.with_name(path.stem + ".partial" + path.suffix)
            serialize_started = time.perf_counter()
            if format_id == "npz":
                mx.savez(str(temporary), **source)
            else:
                mx.save_safetensors(
                    str(temporary),
                    source,
                    metadata={"policy": "theseus_model_checkpoint_candidate_v1"},
                )
            os.replace(temporary, path)
            serialize_seconds = time.perf_counter() - serialize_started
            hash_started = time.perf_counter()
            content_sha256 = file_sha256(path)
            hash_seconds = time.perf_counter() - hash_started
            formats[format_id] = {
                "bytes": path.stat().st_size,
                "serialization_seconds": round(serialize_seconds, 6),
                "content_hash_seconds": round(hash_seconds, 6),
                "content_sha256": content_sha256,
                "atomic_file_replacement": True,
                "materialized_load_seconds": [],
                "tensor_manifest_sha256": "",
                "tensor_manifest_matches_source": False,
            }

        order = ["npz", "safetensors"]
        manifests: dict[str, list[dict[str, Any]]] = {key: [] for key in order}
        for repetition in range(load_repetitions):
            for format_id in (order if repetition % 2 == 0 else list(reversed(order))):
                load_started = time.perf_counter()
                loaded = mx.load(str(paths[format_id]))
                mx.eval(*loaded.values())
                formats[format_id]["materialized_load_seconds"].append(
                    time.perf_counter() - load_started
                )
                manifests[format_id].append(tensor_mapping_manifest(loaded))
                del loaded
        for format_id in order:
            observed = manifests[format_id]
            manifest_hashes = {str(row["sha256"]) for row in observed}
            formats[format_id]["materialized_load_seconds"] = distribution(
                [float(value) for value in formats[format_id]["materialized_load_seconds"]]
            )
            formats[format_id]["tensor_manifest_sha256"] = (
                next(iter(manifest_hashes)) if len(manifest_hashes) == 1 else ""
            )
            formats[format_id]["tensor_manifest_matches_source"] = all(
                row == source_manifest for row in observed
            )

    npz = formats["npz"]
    safe = formats["safetensors"]
    exact = bool(
        npz["tensor_manifest_matches_source"]
        and safe["tensor_manifest_matches_source"]
        and npz["tensor_manifest_sha256"] == safe["tensor_manifest_sha256"]
    )
    size_ratio = float(safe["bytes"]) / max(1.0, float(npz["bytes"]))
    load_speedup = float(npz["materialized_load_seconds"]["p50"]) / max(
        1e-12, float(safe["materialized_load_seconds"]["p50"])
    )
    save_speedup = float(npz["serialization_seconds"]) / max(
        1e-12, float(safe["serialization_seconds"])
    )
    materially_better = exact and (
        size_ratio <= 0.95 or load_speedup >= 1.2 or save_speedup >= 1.2
    )
    return {
        "policy": "project_theseus_checkpoint_format_qualification_v1",
        "state": "GREEN" if exact else "RED",
        "source_checkpoint": relative(checkpoint),
        "source_checkpoint_bytes": checkpoint.stat().st_size,
        "source_tensor_manifest": source_manifest,
        "load_repetitions_per_format": load_repetitions,
        "load_order": "alternating_npz_first_and_safetensors_first",
        "formats": formats,
        "exact_tensor_parity": exact,
        "safetensors_to_npz_size_ratio": round(size_ratio, 6),
        "safetensors_load_speedup": round(load_speedup, 6),
        "safetensors_save_speedup": round(save_speedup, 6),
        "adoption_threshold": (
            "exact parity and >=5% smaller or >=1.2x materialized load/save speedup"
        ),
        "adoption_recommendation": (
            "QUALIFIED_FOR_CONTROLLED_MIGRATION"
            if materially_better
            else "KEEP_CURRENT_NPZ"
            if exact
            else "REJECT_SAFETENSORS"
        ),
        "durable_checkpoint_mutated": False,
        "background_serialization": "NOT_ADOPTED_ON_16GB_UNIFIED_MEMORY_WITHOUT_PEAK_MEMORY_PROOF",
    }


def tensor_mapping_manifest(mapping: dict[str, Any]) -> dict[str, Any]:
    """Content-bind tensor names, shapes, dtypes, and bytes independently of file format."""

    import numpy as np

    digest = hashlib.sha256()
    total_elements = 0
    total_bytes = 0
    for name in sorted(mapping):
        array = np.asarray(mapping[name])
        descriptor = json.dumps(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        contiguous = np.ascontiguousarray(array)
        digest.update(memoryview(contiguous).cast("B"))
        total_elements += int(array.size)
        total_bytes += int(array.nbytes)
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": len(mapping),
        "element_count": total_elements,
        "payload_bytes": total_bytes,
    }


def run_training_pair_qualification(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    target: dict[str, Any],
    checkpoint: Path,
    optimizer_path: Path,
    steps: int,
    repetitions: int = 3,
    compiled_microbatch_size: int = 4,
) -> dict[str, Any]:
    """Compare repeated eager/compiled updates from identical durable state."""

    if repetitions < 2:
        raise ValueError("training pair qualification requires at least two repetitions")
    if compiled_microbatch_size < 1:
        raise ValueError("compiled microbatch size must be positive")
    route_context = build_training_route_context(
        config=config,
        plan=plan,
        target=target,
        checkpoint=checkpoint,
        optimizer_path=optimizer_path,
        steps=steps,
        compiled_microbatch_size=compiled_microbatch_size,
    )
    mx = route_context["mx"]
    trials: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        mode_reports: dict[str, dict[str, Any]] = {}
        mode_order = (
            ("eager", "compiled")
            if repetition % 2 == 0
            else ("compiled", "eager")
        )
        for mode in mode_order:
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            kernel_options = (
                {
                    "rope_kernel": "manual_reference",
                    "prune_inactive_auxiliary_outputs": False,
                }
                if mode == "eager"
                else {
                    "rope_kernel": "mlx_fast",
                    "prune_inactive_auxiliary_outputs": True,
                }
            )
            mode_reports[mode] = run_training_route(
                mode=mode,
                capture_parameter_snapshot=True,
                **kernel_options,
                **route_context,
            )
        eager = mode_reports["eager"]
        compiled = mode_reports["compiled"]
        parameter_comparison = compare_parameter_snapshots(
            eager.pop("_parameter_snapshot"),
            compiled.pop("_parameter_snapshot"),
        )
        speedup = float(compiled["warmup_excluded_positions_per_second"]) / max(
            1e-12, float(eager["warmup_excluded_positions_per_second"])
        )
        trials.append(
            {
                "repetition": repetition + 1,
                "route_order": list(mode_order),
                "eager": eager,
                "compiled": compiled,
                "speedup": round(speedup, 6),
                "final_loss_absolute_delta": round(
                    float(compiled["final_loss"]) - float(eager["final_loss"]), 8
                ),
                "parameter_comparison": parameter_comparison,
            }
        )
    eager = aggregate_training_routes([row["eager"] for row in trials])
    compiled = aggregate_training_routes([row["compiled"] for row in trials])
    speedups = [float(row["speedup"]) for row in trials]
    pooled_speedup = float(eager["warmup_excluded_seconds_total"]) / max(
        1e-12, float(compiled["warmup_excluded_seconds_total"])
    )
    median_speedup = float(statistics.median(speedups))
    bounded_loss_parity = all(
        abs(float(row["final_loss_absolute_delta"]))
        <= MAX_FINAL_LOSS_ABSOLUTE_DELTA
        for row in trials
    )
    bounded_parameter_parity = all(
        bool((row.get("parameter_comparison") or {}).get("within_tolerance"))
        for row in trials
    )
    robust = (
        bounded_loss_parity
        and bounded_parameter_parity
        and median_speedup >= 2.0
        and pooled_speedup >= 2.0
    )
    return {
        "policy": "project_theseus_same_semantics_training_acceleration_pair_v2",
        "state": "GREEN" if robust else "YELLOW",
        "starting_checkpoint_sha256": file_sha256(checkpoint),
        "starting_optimizer_state_sha256": file_sha256(optimizer_path),
        "same_starting_state": True,
        "same_data_order": True,
        "same_batch_size": True,
        "same_loss_mass_weighting": True,
        "same_gradient_clip_and_update_count": True,
        "reference_rope_kernel": "manual_reference",
        "optimized_training_rope_kernel": "mlx_fast",
        "inference_rope_kernel_unchanged": "manual_reference",
        "inactive_auxiliary_pruning_requires_zero_effective_weight": True,
        "steps_per_route_per_repetition": steps,
        "repetitions": repetitions,
        "route_order_control": "alternating eager-first and compiled-first",
        "compiled_microbatch_size": compiled_microbatch_size,
        "eager": eager,
        "compiled": compiled,
        "trials": trials,
        "trial_speedup_distribution": distribution(speedups),
        "median_speedup": round(median_speedup, 6),
        "pooled_speedup": round(pooled_speedup, 6),
        "acceptance": {
            "bounded_loss_parity_every_trial": bounded_loss_parity,
            "maximum_final_loss_absolute_delta": MAX_FINAL_LOSS_ABSOLUTE_DELTA,
            "bounded_full_parameter_parity_every_trial": bounded_parameter_parity,
            "maximum_parameter_absolute_delta": MAX_PARAMETER_ABSOLUTE_DELTA,
            "maximum_parameter_relative_l2_delta": MAX_PARAMETER_RELATIVE_L2_DELTA,
            "median_speedup_at_least_2x": median_speedup >= 2.0,
            "pooled_speedup_at_least_2x": pooled_speedup >= 2.0,
        },
        "checkpoint_or_training_state_written": False,
    }


def run_precision_pair_qualification(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    target: dict[str, Any],
    checkpoint: Path,
    optimizer_path: Path,
    steps: int,
    repetitions: int,
    compiled_microbatch_size: int,
) -> dict[str, Any]:
    """Compare fp32 compiled training with bf16 compute and fp32 master weights."""

    if repetitions < 2:
        raise ValueError("precision qualification requires at least two repetitions")
    route_context = build_training_route_context(
        config=config,
        plan=plan,
        target=target,
        checkpoint=checkpoint,
        optimizer_path=optimizer_path,
        steps=steps,
        compiled_microbatch_size=compiled_microbatch_size,
    )
    mx = route_context["mx"]
    trials: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        route_order = (
            ("float32", "bfloat16_fp32_master")
            if repetition % 2 == 0
            else ("bfloat16_fp32_master", "float32")
        )
        routes: dict[str, dict[str, Any]] = {}
        for precision_mode in route_order:
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            routes[precision_mode] = run_training_route(
                mode="compiled",
                precision_mode=precision_mode,
                **route_context,
            )
        baseline = routes["float32"]
        candidate = routes["bfloat16_fp32_master"]
        baseline_rate = float(baseline["warmup_excluded_positions_per_second"])
        candidate_rate = float(candidate["warmup_excluded_positions_per_second"])
        loss_delta = float(candidate["final_loss"]) - float(baseline["final_loss"])
        relative_loss_delta = abs(loss_delta) / max(1e-12, abs(float(baseline["final_loss"])))
        trials.append(
            {
                "repetition": repetition + 1,
                "route_order": list(route_order),
                "float32": baseline,
                "bfloat16_fp32_master": candidate,
                "speedup": round(candidate_rate / max(1e-12, baseline_rate), 6),
                "final_loss_delta": round(loss_delta, 8),
                "relative_final_loss_delta": round(relative_loss_delta, 8),
            }
        )
    baseline = aggregate_training_routes([row["float32"] for row in trials])
    candidate = aggregate_training_routes(
        [row["bfloat16_fp32_master"] for row in trials]
    )
    speedups = [float(row["speedup"]) for row in trials]
    median_speedup = float(statistics.median(speedups))
    pooled_speedup = float(baseline["warmup_excluded_seconds_total"]) / max(
        1e-12, float(candidate["warmup_excluded_seconds_total"])
    )
    maximum_relative_loss_delta = max(
        float(row["relative_final_loss_delta"]) for row in trials
    )
    numeric_integrity = all(
        route[section]["all_finite"]
        for row in trials
        for route in (row["float32"], row["bfloat16_fp32_master"])
        for section in (
            "compute_parameters",
            "authoritative_parameters",
            "optimizer_state",
        )
    )
    dtype_integrity = all(
        row["bfloat16_fp32_master"]["compute_parameters"]["dtypes"]
        == ["mlx.core.bfloat16"]
        and row["bfloat16_fp32_master"]["authoritative_parameters"]["dtypes"]
        == ["mlx.core.float32"]
        for row in trials
    )
    loss_integrity = maximum_relative_loss_delta <= 0.02
    memory_nonregressed = (
        int(candidate["peak_mlx_bytes_maximum"])
        <= int(baseline["peak_mlx_bytes_maximum"])
    )
    adopt = bool(
        numeric_integrity
        and dtype_integrity
        and loss_integrity
        and memory_nonregressed
        and median_speedup >= 1.15
        and pooled_speedup >= 1.15
    )
    fault = not numeric_integrity or not dtype_integrity or not loss_integrity
    return {
        "policy": "project_theseus_mlx_mixed_precision_master_pair_v1",
        "state": "RED" if fault else "GREEN" if adopt else "YELLOW",
        "adopt": adopt,
        "candidate": "bfloat16_compute_fp32_master_weights_and_optimizer",
        "same_starting_checkpoint_and_optimizer": True,
        "same_data_order_batch_schedule_objective_and_update_count": True,
        "steps_per_route_per_repetition": steps,
        "repetitions": repetitions,
        "route_order_control": "alternating fp32-first and bf16-first",
        "float32": baseline,
        "bfloat16_fp32_master": candidate,
        "trials": trials,
        "median_speedup": round(median_speedup, 6),
        "pooled_speedup": round(pooled_speedup, 6),
        "maximum_relative_final_loss_delta": round(maximum_relative_loss_delta, 8),
        "acceptance": {
            "all_numeric_state_finite": numeric_integrity,
            "bf16_compute_fp32_authority_dtypes_exact": dtype_integrity,
            "relative_final_loss_delta_at_most_0_02": loss_integrity,
            "peak_mlx_memory_nonregressed": memory_nonregressed,
            "median_speedup_at_least_1_15x": median_speedup >= 1.15,
            "pooled_speedup_at_least_1_15x": pooled_speedup >= 1.15,
        },
        "checkpoint_or_training_state_written": False,
    }


def build_training_route_context(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    target: dict[str, Any],
    checkpoint: Path,
    optimizer_path: Path,
    steps: int,
    compiled_microbatch_size: int,
) -> dict[str, Any]:
    """Materialize one immutable route context shared by acceleration comparisons."""

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    stage_dir = resolve(str(config["stage_dir"]))
    metadata = read_json(stage_dir / "stage_metadata_v1.json")
    base = read_json(resolve(str(config["base_config"])))
    canonical = metadata["summary"]["canonical_pretrain_stage"]
    shape = (
        int(canonical["window_count"]),
        int(canonical["max_sequence_tokens"]),
    )
    arrays = training.load_pretrain_memmaps(
        training.pretrain_array_paths(stage_dir),
        shape,
        expected=canonical["array_artifacts"],
    )
    inputs = training.range_view(arrays[0], target["row_ranges"])
    labels = training.range_view(arrays[1], target["row_ranges"])
    mask = training.range_view(arrays[2], target["row_ranges"])
    training_cfg = config["training"]
    total_schedule_steps = training.required_steps(
        mask,
        int(training_cfg["batch_size"]),
        int(target["optimizer_target_positions"]),
    ) + 128
    vocab_size = int(target.get("vocab_size") or plan["models"]["vocab_size"])
    copy_lookup = training.build_source_to_target_lookup(
        base,
        metadata,
        vocab_size=vocab_size,
        identity_ranges=training.target_copy_identity_ranges(target),
    )
    return {
        "config": config,
        "plan": plan,
        "target": target,
        "checkpoint": checkpoint,
        "optimizer_path": optimizer_path,
        "steps": steps,
        "compiled_microbatch_size": compiled_microbatch_size,
        "inputs": inputs,
        "labels": labels,
        "mask": mask,
        "copy_lookup": copy_lookup,
        "total_schedule_steps": total_schedule_steps,
        "receipt": read_json(resolve(str(target["receipt"]))),
        "mx": mx,
        "nn": nn,
        "optim": optim,
        "mlx_utils": mlx_utils,
    }


def run_training_route(
    *,
    mode: str,
    config: dict[str, Any],
    plan: dict[str, Any],
    target: dict[str, Any],
    checkpoint: Path,
    optimizer_path: Path,
    steps: int,
    compiled_microbatch_size: int,
    inputs: Any,
    labels: Any,
    mask: Any,
    copy_lookup: Any,
    total_schedule_steps: int,
    receipt: dict[str, Any],
    mx: Any,
    nn: Any,
    optim: Any,
    mlx_utils: Any,
    precision_mode: str = "float32",
    capture_parameter_snapshot: bool = False,
    rope_kernel: str = "mlx_fast",
    prune_inactive_auxiliary_outputs: bool = True,
) -> dict[str, Any]:
    """Run one non-mutating route from the exact registered checkpoint state."""

    if precision_mode not in {"float32", "bfloat16_fp32_master"}:
        raise ValueError(f"unsupported precision mode: {precision_mode}")
    training_cfg = config["training"]
    vocab_size = int(target.get("vocab_size") or plan["models"]["vocab_size"])
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    mx.random.seed(int(config["seed"]) + training.stable_int(training.SHARED_TRUNK_ID))
    model = training.build_model(
        training.CausalTransformerConfig(vocab_size=vocab_size, **target["model"]),
        mx=mx,
        nn=nn,
        state_role_lookup=None,
        source_to_target_lookup=copy_lookup,
        rope_kernel=rope_kernel,
    )
    master_model = None
    if precision_mode == "bfloat16_fp32_master":
        master_model = training.build_model(
            training.CausalTransformerConfig(vocab_size=vocab_size, **target["model"]),
            mx=mx,
            nn=nn,
            state_role_lookup=None,
            source_to_target_lookup=copy_lookup,
            rope_kernel=rope_kernel,
        )
    schedule = training.build_schedule(optim, mx, training_cfg, total_schedule_steps)
    optimizer = optim.AdamW(
        learning_rate=schedule,
        weight_decay=float(training_cfg["weight_decay"]),
    )
    model.load_weights(str(checkpoint))
    if master_model is not None:
        master_model.load_weights(str(checkpoint))
        model.set_dtype(mx.bfloat16)
    optimizer.state = mlx_utils.tree_unflatten(list(mx.load(str(optimizer_path)).items()))
    mx.eval(
        model.parameters(),
        master_model.parameters() if master_model is not None else model.parameters(),
        optimizer.state,
    )
    if master_model is not None:
        loss_function = mixed_precision_token_loss
    elif prune_inactive_auxiliary_outputs:
        def loss_function(*loss_args: Any, **loss_kwargs: Any) -> Any:
            return training.causal_loss(
                *loss_args,
                **loss_kwargs,
                prune_inactive_auxiliary_outputs=True,
            )
    else:
        loss_function = training.causal_loss
    phase = training.train_phase(
        model,
        optimizer,
        nn.value_and_grad(model, loss_function),
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
        phase_name=f"resource_acceleration_{mode}_reference",
        target_positions=10**12,
        batch_size=int(training_cfg["batch_size"]),
        gradient_clip=float(training_cfg["gradient_clip_norm"]),
        seed=(
            int(config["seed"])
            + training.stable_int(training.SHARED_TRUNK_ID)
            + int(receipt.get("optimizer_steps") or 0)
        ),
        max_steps=steps,
        checkpoint=Path("/tmp/theseus_acceleration_unused.npz"),
        checkpoint_every=10**9,
        heartbeat=Path("/tmp/theseus_acceleration_heartbeat.json"),
        global_step_offset=int(receipt.get("optimizer_steps") or 0),
        mx=mx,
        optim=optim,
        source_conditioning=False,
        training_step_mode=mode,
        compiled_microbatch_size=compiled_microbatch_size,
        master_model=master_model,
        compute_dtype_name=(
            "bfloat16" if master_model is not None else "float32"
        ),
    )
    authoritative_model = master_model if master_model is not None else model
    observed = {
        "training_step_execution": phase["training_step_execution"],
        "optimizer_steps": phase["optimizer_steps"],
        "optimizer_positions": phase["optimizer_all_target_positions_consumed"],
        "warmup_excluded_positions": phase["warmup_excluded_positions"],
        "warmup_excluded_seconds": phase["warmup_excluded_seconds"],
        "warmup_excluded_positions_per_second": phase[
            "warmup_excluded_tokens_per_second"
        ],
        "post_first_positions_per_second": phase[
            "post_first_optimizer_tokens_per_second"
        ],
        "first_optimizer_step_seconds": phase["first_optimizer_step_seconds"],
        "median_optimizer_step_seconds": phase["median_optimizer_step_seconds"],
        "mean_loss": phase["mean_loss"],
        "final_loss": phase["final_loss"],
        "optimizer_step_seconds_prefix": phase["optimizer_step_seconds_prefix"],
        "compiled_accumulation_seconds_total": phase[
            "compiled_accumulation_seconds_total"
        ],
        "compiled_update_seconds_total": phase["compiled_update_seconds_total"],
        "compiled_accumulation_seconds_prefix": phase[
            "compiled_accumulation_seconds_prefix"
        ],
        "compiled_update_seconds_prefix": phase[
            "compiled_update_seconds_prefix"
        ],
        "precision_mode": precision_mode,
        "rope_kernel": rope_kernel,
        "prune_inactive_auxiliary_outputs": prune_inactive_auxiliary_outputs,
        "compute_parameters": tree_numeric_receipt(
            model.trainable_parameters(), mx=mx, mlx_utils=mlx_utils
        ),
        "authoritative_parameters": tree_numeric_receipt(
            authoritative_model.trainable_parameters(), mx=mx, mlx_utils=mlx_utils
        ),
        "optimizer_state": tree_numeric_receipt(
            optimizer.state, mx=mx, mlx_utils=mlx_utils
        ),
        "mlx_memory": mlx_memory_receipt(mx),
    }
    if capture_parameter_snapshot:
        observed["_parameter_snapshot"] = {
            name: np.asarray(value).copy()
            for name, value in mlx_utils.tree_flatten(
                authoritative_model.trainable_parameters()
            )
        }
    del model, master_model, optimizer
    return observed


def compare_parameter_snapshots(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    names_match = set(reference) == set(candidate)
    shapes_match = names_match and all(
        np.asarray(reference[name]).shape == np.asarray(candidate[name]).shape
        for name in reference
    )
    if not shapes_match:
        return {
            "within_tolerance": False,
            "names_match": names_match,
            "shapes_match": False,
        }
    maximum_absolute_delta = 0.0
    squared_delta = 0.0
    squared_reference = 0.0
    element_count = 0
    all_finite = True
    for name in sorted(reference):
        reference_array = np.asarray(reference[name])
        candidate_array = np.asarray(candidate[name])
        delta = candidate_array - reference_array
        all_finite = bool(
            all_finite
            and np.isfinite(reference_array).all()
            and np.isfinite(candidate_array).all()
            and np.isfinite(delta).all()
        )
        maximum_absolute_delta = max(
            maximum_absolute_delta,
            float(np.max(np.abs(delta), initial=0.0)),
        )
        squared_delta += float(np.sum(np.square(delta), dtype=np.float64))
        squared_reference += float(
            np.sum(np.square(reference_array), dtype=np.float64)
        )
        element_count += int(reference_array.size)
    relative_l2_delta = math.sqrt(squared_delta) / max(
        1e-30, math.sqrt(squared_reference)
    )
    within_tolerance = bool(
        all_finite
        and maximum_absolute_delta <= MAX_PARAMETER_ABSOLUTE_DELTA
        and relative_l2_delta <= MAX_PARAMETER_RELATIVE_L2_DELTA
    )
    return {
        "within_tolerance": within_tolerance,
        "names_match": names_match,
        "shapes_match": shapes_match,
        "all_finite": all_finite,
        "tensor_count": len(reference),
        "element_count": element_count,
        "maximum_absolute_delta": round(maximum_absolute_delta, 12),
        "relative_l2_delta": round(relative_l2_delta, 12),
        "maximum_absolute_delta_allowed": MAX_PARAMETER_ABSOLUTE_DELTA,
        "maximum_relative_l2_delta_allowed": MAX_PARAMETER_RELATIVE_L2_DELTA,
    }


def mixed_precision_token_loss(
    model: Any,
    inputs: Any,
    labels: Any,
    mask: Any,
    mx: Any,
    nn: Any,
    *,
    source_conditioning: bool | None = None,
) -> Any:
    """Keep the token loss reduction in fp32 while the model computes in bf16."""

    logits, _cache = model(inputs, source_conditioning=source_conditioning)
    token_loss = nn.losses.cross_entropy(logits.astype(mx.float32), labels)
    denominator = mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mx.float32))
    return mx.sum(token_loss * mask) / denominator


def tree_numeric_receipt(tree: Any, *, mx: Any, mlx_utils: Any) -> dict[str, Any]:
    """Report dtypes and finite state without copying full tensors to the host."""

    rows = list(mlx_utils.tree_flatten(tree))
    finite_checks = [mx.all(mx.isfinite(value)) for _name, value in rows if value.dtype in {
        mx.float16,
        mx.bfloat16,
        mx.float32,
    }]
    if finite_checks:
        mx.eval(*finite_checks)
    return {
        "tensor_count": len(rows),
        "element_count": sum(int(value.size) for _name, value in rows),
        "dtypes": sorted({str(value.dtype) for _name, value in rows}),
        "all_finite": all(bool(value.item()) for value in finite_checks),
    }


def aggregate_training_routes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate repeats while retaining each underlying route receipt."""

    seconds = sum(float(row["warmup_excluded_seconds"]) for row in rows)
    positions = sum(int(row["warmup_excluded_positions"]) for row in rows)
    return {
        "repetition_count": len(rows),
        "training_step_execution": rows[0]["training_step_execution"],
        "optimizer_steps_total": sum(int(row["optimizer_steps"]) for row in rows),
        "optimizer_positions_total": sum(int(row["optimizer_positions"]) for row in rows),
        "warmup_excluded_positions_total": positions,
        "warmup_excluded_seconds_total": round(seconds, 6),
        "pooled_positions_per_second": round(positions / max(1e-12, seconds), 3),
        "compiled_accumulation_seconds_total": round(
            sum(
                float(row.get("compiled_accumulation_seconds_total") or 0.0)
                for row in rows
            ),
            6,
        ),
        "compiled_update_seconds_total": round(
            sum(
                float(row.get("compiled_update_seconds_total") or 0.0)
                for row in rows
            ),
            6,
        ),
        "positions_per_second_distribution": distribution(
            [float(row["warmup_excluded_positions_per_second"]) for row in rows]
        ),
        "peak_mlx_bytes_maximum": max(
            int((row.get("mlx_memory") or {}).get("peak_bytes") or 0) for row in rows
        ),
        "runs": rows,
    }


def run_inference_qualification(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    target: dict[str, Any],
    checkpoint: Path,
    rows: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import mlx.core as mx
    import mlx.nn as nn

    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    metadata = read_json(resolve(str(config["stage_dir"])) / "stage_metadata_v1.json")
    base = read_json(resolve(str(config["base_config"])))
    source_vocab = dict(metadata["source_vocab"])
    target_vocab = dict(metadata["target_vocab"])
    vocab_size = int(target.get("vocab_size") or plan["models"]["vocab_size"])
    construct_started = time.perf_counter()
    model = training.build_model(
        training.CausalTransformerConfig(vocab_size=vocab_size, **target["model"]),
        mx=mx,
        nn=nn,
        state_role_lookup=None,
        source_to_target_lookup=training.build_source_to_target_lookup(
            base,
            metadata,
            vocab_size=vocab_size,
            identity_ranges=training.target_copy_identity_ranges(target),
        ),
    )
    construct_seconds = time.perf_counter() - construct_started
    load_started = time.perf_counter()
    model.load_weights(str(checkpoint))
    mx.eval(model.parameters())
    model.eval()
    load_seconds = time.perf_counter() - load_started
    decode_max = max_tokens or int(config["evaluation"]["decode_max_target_tokens"])
    common = {
        "model": model,
        "source_vocab": source_vocab,
        "target_vocab": target_vocab,
        "base": base,
        "max_source_tokens": int(
            config["supervision"]["maximum_source_encoded_tokens"]
        ),
        "beam_width": int(config["evaluation"]["beam_width"]),
        "branching_factor": int(config["evaluation"]["branching_factor"]),
        "length_penalty": float(config["evaluation"]["length_penalty"]),
        "mx": mx,
    }
    # Warm only the production route; measured cases still build fresh per-request caches.
    training.generate_model_text(
        prompt=str(rows[0]["prompt"]),
        max_tokens=min(8, decode_max),
        **common,
    )
    case_reports = []
    for index, row in enumerate(rows):
        routes = (
            (("reference", reference_route()), ("optimized", optimized_route()))
            if index % 2 == 0
            else (("optimized", optimized_route()), ("reference", reference_route()))
        )
        observed: dict[str, dict[str, Any]] = {}
        for label, route in routes:
            run_started = time.perf_counter()
            output, receipt = training.generate_model_text(
                prompt=str(row["prompt"]),
                max_tokens=decode_max,
                batched_beam_advance=bool(route["batched_beam_advance"]),
                device_logit_filter=bool(route["device_logit_filter"]),
                preprune_beam_expansions=bool(route["preprune_beam_expansions"]),
                **common,
            )
            observed[label] = {
                "duration_seconds": time.perf_counter() - run_started,
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "receipt": receipt,
            }
        exact = (
            observed["reference"]["output_sha256"]
            == observed["optimized"]["output_sha256"]
            and semantic_receipt(observed["reference"]["receipt"])
            == semantic_receipt(observed["optimized"]["receipt"])
        )
        case_reports.append(
            {
                "case_id": row["case_id"],
                "arm_id": row["arm_id"],
                "exact_parity": exact,
                "output_sha256": observed["optimized"]["output_sha256"],
                "semantic_receipt_sha256": stable_hash(
                    semantic_receipt(observed["optimized"]["receipt"])
                ),
                "reference_seconds": round(
                    observed["reference"]["duration_seconds"], 6
                ),
                "optimized_seconds": round(
                    observed["optimized"]["duration_seconds"], 6
                ),
                "speedup": round(
                    observed["reference"]["duration_seconds"]
                    / max(1e-12, observed["optimized"]["duration_seconds"]),
                    6,
                ),
                "generation_state": observed["optimized"]["receipt"].get("state"),
                "generation_reason": observed["optimized"]["receipt"].get("reason"),
                "raw_prompt_or_output_retained": False,
            }
        )
    reference = [float(row["reference_seconds"]) for row in case_reports]
    optimized = [float(row["optimized_seconds"]) for row in case_reports]
    load = {
        "state": "READY",
        "model_construct_seconds": round(construct_seconds, 6),
        "weights_load_and_materialize_seconds": round(load_seconds, 6),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": file_sha256(checkpoint),
        "resident_model_reused_across_case_count": len(rows),
        "model_loads": 1,
    }
    return load, {
        "state": "GREEN" if all(row["exact_parity"] for row in case_reports) else "RED",
        "case_count": len(case_reports),
        "exact_parity_case_count": sum(row["exact_parity"] for row in case_reports),
        "uncached_aggregate_speedup": round(
            sum(reference) / max(1e-12, sum(optimized)), 6
        ),
        "measurement_scope": (
            "novel-request uncached decode; completion and prompt-prefix caches disabled"
        ),
        "reference_latency_seconds": distribution(reference),
        "optimized_latency_seconds": distribution(optimized),
        "reference_route": reference_route(),
        "optimized_route": optimized_route(),
        "max_target_tokens": decode_max,
        "order_bias_control": "alternating_reference_first_and_optimized_first",
        "warmup": "optimized_route_eight_token_compile_warmup",
        "cases": case_reports,
        "minimum_uncached_decode_speedup": 2.0,
        "minimum_uncached_decode_speedup_role": "acceptance_threshold_not_measurement",
        "quality_preservation": "exact output and normalized generation receipt parity",
        "mlx_memory": mlx_memory_receipt(mx),
        "resident_process_scope": (
            "one model load reused across all measured requests; cross-process service "
            "lifetime remains pending"
        ),
    }


def reference_route() -> dict[str, bool]:
    return {
        "batched_beam_advance": False,
        "device_logit_filter": False,
        "preprune_beam_expansions": False,
    }


def optimized_route() -> dict[str, bool]:
    return {
        "batched_beam_advance": True,
        "device_logit_filter": True,
        "preprune_beam_expansions": True,
    }


def semantic_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key not in ACCELERATION_KEYS}


def select_qualification_rows(
    rows: list[dict[str, Any]], sample_count: int
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: stable_hash({"case_id": row["case_id"]}))
    selected: list[dict[str, Any]] = []
    seen_arms: set[str] = set()
    for row in ordered:
        arm = str(row["arm_id"])
        if arm not in seen_arms:
            selected.append(row)
            seen_arms.add(arm)
            if len(selected) >= sample_count:
                return selected
    selected_ids = {str(row["case_id"]) for row in selected}
    selected.extend(
        row
        for row in ordered
        if str(row["case_id"]) not in selected_ids
    )
    return selected[:sample_count]


def validate_packet(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["private_prompt_packet_empty"]
    allowed = {"case_id", "arm_id", "prompt"}
    if any(set(row) != allowed for row in rows):
        return ["private_prompt_packet_contains_evaluator_or_target_fields"]
    return []


def training_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"state": "MISSING"}
    report = read_json(path)
    results = report.get("results") if isinstance(report.get("results"), list) else []
    result = results[0] if results else {}
    phase = ((result.get("phases") or {}).get("pretraining") or {})
    throughput = float(phase.get("warmup_excluded_tokens_per_second") or 0.0)
    return {
        "state": "READY" if throughput > 0.0 else "INVALID",
        "report_sha256": file_sha256(path),
        "optimizer_steps": int(phase.get("optimizer_steps") or 0),
        "optimizer_positions": int(phase.get("optimizer_all_target_positions_consumed") or 0),
        "warmup_excluded_positions_per_second": throughput,
        "post_first_positions_per_second": float(
            phase.get("post_first_optimizer_tokens_per_second") or 0.0
        ),
        "training_step_execution": phase.get("training_step_execution"),
        "compiled_microbatch_size": phase.get("compiled_microbatch_size"),
        "same_semantics": (
            phase.get("training_step_execution") == "mlx_compiled_shape_bucket_v1"
            and int(phase.get("compiled_microbatch_size") or 0) > 0
        ),
        "wall_seconds": float(result.get("wall_seconds") or report.get("wall_seconds") or 0.0),
    }


def learning_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"state": "MISSING"}
    report = read_json(path)
    results = report.get("results") if isinstance(report.get("results"), list) else []
    comparison = (results[0].get("comparison") or {}) if results else {}
    return {
        "state": "READY" if comparison else "INVALID",
        "report_sha256": file_sha256(path),
        "absolute_loss_delta": comparison.get("absolute_loss_delta"),
        "relative_loss_reduction": comparison.get("relative_loss_reduction"),
        "aggregate_improved": comparison.get("improved"),
        "regressed_arms": comparison.get("regressed_arms") or [],
        "loss_delta_by_arm": comparison.get("loss_delta_by_arm") or {},
        "evaluation_split": report.get("evaluation_split"),
        "confirmation_split_consumed": report.get("confirmation_split_consumed"),
        "public_calibration_consumed": report.get("public_calibration_consumed"),
    }


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "total": round(sum(ordered), 6),
        "mean": round(statistics.fmean(ordered), 6),
        "p50": round(statistics.median(ordered), 6),
        "p95": round(ordered[p95_index], 6),
        "minimum": round(ordered[0], 6),
        "maximum": round(ordered[-1], 6),
    }


def hardware_receipt() -> dict[str, Any]:
    disk = os.statvfs(ROOT)
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "cpu": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "battery": command_output(["pmset", "-g", "batt"]),
        "thermal": command_output(["pmset", "-g", "therm"]),
        "disk_free_bytes": disk.f_bavail * disk.f_frsize,
        "mlx_runtime": "required_only_for_execute_mode",
        "energy_measurement_state": "NOT_AVAILABLE_FROM_MLX_RUNTIME",
    }


def mlx_memory_receipt(mx: Any) -> dict[str, int | str]:
    """Read allocator counters without presenting them as system-wide memory."""

    return {
        "active_bytes": int(mx.get_active_memory()) if hasattr(mx, "get_active_memory") else 0,
        "cache_bytes": int(mx.get_cache_memory()) if hasattr(mx, "get_cache_memory") else 0,
        "peak_bytes": int(mx.get_peak_memory()) if hasattr(mx, "get_peak_memory") else 0,
        "scope": "mlx_allocator_only",
    }


def process_resource_receipt() -> dict[str, int | str]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    maximum_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        maximum_rss *= 1024
    return {
        "maximum_resident_set_bytes": maximum_rss,
        "block_input_operations": int(usage.ru_inblock),
        "block_output_operations": int(usage.ru_oublock),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
        "scope": "current_qualification_process_cumulative",
    }


def process_resource_delta(
    before: dict[str, int | str], after: dict[str, int | str]
) -> dict[str, int | str]:
    return {
        "maximum_resident_set_bytes": int(after["maximum_resident_set_bytes"]),
        "block_input_operations_delta": int(after["block_input_operations"])
        - int(before["block_input_operations"]),
        "block_output_operations_delta": int(after["block_output_operations"])
        - int(before["block_output_operations"]),
        "voluntary_context_switches_delta": int(after["voluntary_context_switches"])
        - int(before["voluntary_context_switches"]),
        "involuntary_context_switches_delta": int(after["involuntary_context_switches"])
        - int(before["involuntary_context_switches"]),
        "scope": "current_qualification_process",
    }


def command_output(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return (completed.stdout or completed.stderr or "").strip()[:1000]


def render_markdown(report: dict[str, Any]) -> str:
    inference = report["inference"]
    training = report["training"]
    learning = report["private_dev_learning"]
    checkpoint = report["checkpoint_storage"]
    assistant_refresh = report["assistant_context_refresh"]
    resident = report.get("resident_runtime") or {}
    decision = report["architecture_decision_control"]
    training_pair = training.get("paired_canary") or {}
    precision = training.get("precision_autotune") or {}
    return "\n".join(
        [
            "# Resource Acceleration Qualification",
            "",
            f"- State: **{report['trigger_state']}**",
            f"- Training throughput: `{training.get('warmup_excluded_positions_per_second', 0):.3f}` positions/s",
            f"- Training execution: `{training.get('training_step_execution')}`",
            f"- Repeated paired training median speedup: `{float(training_pair.get('median_speedup') or 0.0):.3f}x`",
            f"- Repeated paired training pooled speedup: `{float(training_pair.get('pooled_speedup') or 0.0):.3f}x`",
            f"- Repeated paired training state: `{training_pair.get('state')}`",
            f"- Mixed-precision candidate state: `{precision.get('state')}`",
            f"- Mixed-precision median speedup: `{float(precision.get('median_speedup') or 0.0):.3f}x`",
            f"- Mixed-precision pooled speedup: `{float(precision.get('pooled_speedup') or 0.0):.3f}x`",
            f"- Mixed-precision adopted: `{precision.get('adopt')}`",
            "- Mixed-precision timing scope: `rejected bf16-compute/fp32-master candidate; not resident-runtime overhead`",
            f"- Private-dev loss delta: `{learning.get('absolute_loss_delta')}`",
            f"- Weak-tail regressions: `{', '.join(learning.get('regressed_arms') or []) or 'none'}`",
            f"- Decode cases: `{inference.get('case_count', 0)}`",
            f"- Exact parity cases: `{inference.get('exact_parity_case_count', 0)}`",
            f"- Uncached novel-request aggregate decode speedup: `{float(inference.get('uncached_aggregate_speedup') or 0.0):.3f}x`",
            f"- Uncached decode acceptance threshold: `{float(inference.get('minimum_uncached_decode_speedup') or 0.0):.3f}x`",
            f"- Checkpoint tensor parity: `{checkpoint.get('exact_tensor_parity')}`",
            f"- Checkpoint format recommendation: `{checkpoint.get('adoption_recommendation')}`",
            f"- Warm governed assistant refresh speedup: `{float(assistant_refresh.get('speedup') or 0.0):.3f}x`",
            f"- Resident repeated-prompt speedup: `{float(resident.get('repeated_prompt_speedup') or 0.0):.3f}x`",
            f"- Resident prefix-prefill speedup: `{float(resident.get('prefix_prefill_speedup') or 0.0):.3f}x`",
            f"- Resident output/token parity: `{resident.get('exact_output_and_token_parity')}`",
            f"- Resident production serving allowed: `{((resident.get('boundaries') or {}).get('runtime_serving_allowed'))}`",
            f"- First architecture-review budget opportunity: `{float(decision.get('first_review_budget_speedup_opportunity') or 0.0):.3f}x`",
            f"- First-decision speedup empirically proven: `{decision.get('target_speedup_empirically_proven')}`",
            "",
            "The decode comparison disables completion and prompt-prefix caches on both routes. Resident cache speedups are reported separately and do not contribute to uncached decode speedup. This qualification does not claim model capability or public transfer.",
            "",
            "## Hard Gaps",
            "",
            *(f"- {gap}" for gap in report["hard_gaps"]),
            *( ["- none"] if not report["hard_gaps"] else [] ),
            "",
            "## Remaining Gaps",
            "",
            *(f"- {gap}" for gap in report.get("remaining_gaps") or []),
            "",
        ]
    )


def report_summary(report: dict[str, Any]) -> dict[str, Any]:
    training_pair = report["training"].get("paired_canary") or {}
    precision = report["training"].get("precision_autotune") or {}
    return {
        "policy": report["policy"],
        "created_utc": report["created_utc"],
        "trigger_state": report["trigger_state"],
        "mode": report["mode"],
        "training_positions_per_second": report["training"].get(
            "warmup_excluded_positions_per_second"
        ),
        "paired_training_median_speedup": training_pair.get("median_speedup"),
        "paired_training_pooled_speedup": training_pair.get("pooled_speedup"),
        "mixed_precision_median_speedup": precision.get("median_speedup"),
        "mixed_precision_pooled_speedup": precision.get("pooled_speedup"),
        "mixed_precision_adopted": precision.get("adopt"),
        "private_dev_loss_delta": report["private_dev_learning"].get(
            "absolute_loss_delta"
        ),
        "decode_case_count": report["inference"].get("case_count"),
        "decode_exact_parity_case_count": report["inference"].get(
            "exact_parity_case_count"
        ),
        "uncached_decode_aggregate_speedup": report["inference"].get(
            "uncached_aggregate_speedup"
        ),
        "uncached_decode_acceptance_threshold": report["inference"].get(
            "minimum_uncached_decode_speedup"
        ),
        "checkpoint_format_recommendation": report["checkpoint_storage"].get(
            "adoption_recommendation"
        ),
        "assistant_refresh_speedup": report["assistant_context_refresh"].get("speedup"),
        "resident_repeated_prompt_speedup": (report.get("resident_runtime") or {}).get(
            "repeated_prompt_speedup"
        ),
        "resident_prefix_prefill_speedup": (report.get("resident_runtime") or {}).get(
            "prefix_prefill_speedup"
        ),
        "resident_exact_output_and_token_parity": (
            report.get("resident_runtime") or {}
        ).get("exact_output_and_token_parity"),
        "continuous_batch_uncached_speedup": (
            (report.get("resident_runtime") or {}).get("continuous_batching") or {}
        ).get("direct_batch_speedup"),
        "continuous_batch_exact_parity": (
            (report.get("resident_runtime") or {}).get("continuous_batching") or {}
        ).get("exact_output_state_reason_and_token_parity"),
        "first_review_budget_speedup_opportunity": report[
            "architecture_decision_control"
        ].get("first_review_budget_speedup_opportunity"),
        "first_decision_speedup_empirically_proven": report[
            "architecture_decision_control"
        ].get("target_speedup_empirically_proven"),
        "hard_gaps": report["hard_gaps"],
        "remaining_gaps": report.get("remaining_gaps") or [],
    }


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "sha256": file_sha256(path) if path.is_file() else "",
        "bytes": path.stat().st_size if path.is_file() else 0,
    }


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
