#!/usr/bin/env python3
"""Qualify capability-critical acceleration against exact reference behavior."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import resource
import re
import shutil
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
from resource_acceleration_assembly_line import (  # noqa: E402
    build_assembly_line,
    generation_quality_receipt,
)


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
DEFAULT_CORPUS_REPORT = ROOT / "reports/theseus_corpus_acceleration.json"
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
MAX_RETAINED_METAL_TRACE_BYTES = 2 * 1024 * 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--packet", default=relative(DEFAULT_PACKET))
    parser.add_argument("--training-report", default=relative(DEFAULT_TRAINING_REPORT))
    parser.add_argument("--learning-curve", default=relative(DEFAULT_LEARNING_CURVE))
    parser.add_argument("--corpus-report", default=relative(DEFAULT_CORPUS_REPORT))
    parser.add_argument("--out", default=relative(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=relative(DEFAULT_MARKDOWN))
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--training-pair-steps", type=int, default=24)
    parser.add_argument("--training-pair-repetitions", type=int, default=3)
    parser.add_argument("--compiled-microbatch-size", type=int, default=4)
    parser.add_argument("--compile-width-quantum", type=int, default=64)
    parser.add_argument(
        "--materialize-compiled-state-after-update",
        action="store_true",
        help=(
            "Explicitly materialize model and optimizer state after each "
            "compiled update to test graph-chain detachment."
        ),
    )
    parser.add_argument(
        "--unmigrated-implementation-challenger",
        action="store_true",
        help=(
            "Permit a non-mutating acceleration diagnostic when the only "
            "resume fault is the expected current implementation-plan mismatch. "
            "The report receives no production authority."
        ),
    )
    parser.add_argument(
        "--diagnostic-state-root",
        default="",
        help=(
            "Optional scratch-only directory for post-timing model, optimizer, "
            "and RNG tensors from --compiled-route-only."
        ),
    )
    parser.add_argument(
        "--compact-encoder-decoder-partitions",
        action="store_true",
        help=(
            "Use the existing exact encoder/decoder partition compaction "
            "implementation as a non-production training challenger."
        ),
    )
    parser.add_argument(
        "--bf16-clear-device-cache-after-step",
        action="store_true",
        help="Clear the MLX allocator cache after each BF16 optimizer update.",
    )
    parser.add_argument(
        "--fp32-clear-device-cache-after-step",
        action="store_true",
        help="Clear the MLX allocator cache after each FP32 optimizer update.",
    )
    parser.add_argument("--precision-pair-steps", type=int, default=8)
    parser.add_argument("--precision-pair-repetitions", type=int, default=2)
    parser.add_argument(
        "--precision-resume-only",
        action="store_true",
        help="Run only the immutable compiled checkpoint/resume qualification.",
    )
    parser.add_argument(
        "--precision-pair-only",
        action="store_true",
        help="Run only alternating FP32/BF16 throughput and loss qualification.",
    )
    parser.add_argument(
        "--training-pair-only",
        action="store_true",
        help="Run only alternating eager/compiled training qualification.",
    )
    parser.add_argument(
        "--compiled-route-only",
        action="store_true",
        help=(
            "Run one isolated compiled route from the immutable checkpoint. "
            "Use matched separate processes for implementation challengers."
        ),
    )
    parser.add_argument(
        "--isolated-route-mode",
        choices=("compiled", "eager"),
        default="compiled",
        help="Execution mode for the isolated non-mutating route diagnostic.",
    )
    parser.add_argument(
        "--joined-training-only",
        action="store_true",
        help="Run one non-mutating joined campaign-path training qualification.",
    )
    parser.add_argument("--joined-pretraining-steps", type=int, default=64)
    parser.add_argument("--joined-source-steps", type=int, default=8)
    parser.add_argument("--joined-supervision-steps", type=int, default=8)
    parser.add_argument(
        "--training-phase",
        choices=(
            "pretraining",
            "source_conditioned_pretraining",
            "supervision",
        ),
        default="pretraining",
    )
    parser.add_argument(
        "--precision-mode",
        choices=(
            "float32",
            "float16_fp32_master",
            "bfloat16_fp32_master",
        ),
        default="bfloat16_fp32_master",
    )
    parser.add_argument(
        "--precision-repeatability-variant",
        choices=(
            "guarded_compiled",
            "integrated_compiled_diagnostic",
            "guarded_eager",
        ),
        default="guarded_compiled",
        help=(
            "Execution boundary for --precision-resume-only. The integrated "
            "variant disables the pre-update host finite-gradient stop only "
            "as a non-production causality diagnostic."
        ),
    )
    parser.add_argument("--metal-trace-out", default="")
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
    if args.compile_width_quantum < 1:
        parser.error("--compile-width-quantum must be positive")
    if args.precision_pair_steps < 2:
        parser.error("--precision-pair-steps must be at least two")
    if args.precision_pair_repetitions < 2:
        parser.error("--precision-pair-repetitions must be at least two")

    focused_modes = sum(
        bool(value)
        for value in (
            args.precision_resume_only,
            args.precision_pair_only,
            args.training_pair_only,
            args.compiled_route_only,
            args.joined_training_only,
        )
    )
    if focused_modes > 1:
        parser.error("choose only one focused training qualification")
    if args.precision_pair_only and args.precision_mode == "float32":
        parser.error(
            "--precision-pair-only requires a mixed --precision-mode candidate"
        )
    if any(
        value < 2
        for value in (
            args.joined_pretraining_steps,
            args.joined_source_steps,
            args.joined_supervision_steps,
        )
    ):
        parser.error("joined phase step counts must each be at least two")
    if args.joined_training_only:
        if not args.execute:
            parser.error("--joined-training-only requires --execute")
        report = run_joined_training_entry(
            config_path=resolve(args.config),
            pretraining_steps=args.joined_pretraining_steps,
            source_steps=args.joined_source_steps,
            supervision_steps=args.joined_supervision_steps,
        )
        write_json(resolve(args.out), report)
        print(
            json.dumps(
                {
                    "policy": report.get("policy"),
                    "created_utc": report.get("created_utc"),
                    "trigger_state": report.get("trigger_state"),
                    "phase_steps": report.get("phase_steps"),
                    "joined_training_positions_per_second": report.get(
                        "joined_training_positions_per_second"
                    ),
                    "joined_end_to_end_positions_per_second": report.get(
                        "joined_end_to_end_positions_per_second"
                    ),
                    "checkpoint_publication_seconds": (
                        report.get("checkpoint_publication") or {}
                    ).get("total_seconds"),
                    "hard_gaps": report.get("hard_gaps") or [],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2 if report.get("trigger_state") == "RED" else 0
    if args.compiled_route_only:
        if not args.execute:
            parser.error("--compiled-route-only requires --execute")
        report = run_compiled_route_entry(
            config_path=resolve(args.config),
            steps=args.training_pair_steps,
            compiled_microbatch_size=args.compiled_microbatch_size,
            compile_width_quantum=args.compile_width_quantum,
            materialize_compiled_state_after_update=(
                args.materialize_compiled_state_after_update
            ),
            unmigrated_implementation_challenger=(
                args.unmigrated_implementation_challenger
            ),
            training_phase=args.training_phase,
            precision_mode=args.precision_mode,
            diagnostic_state_root=(
                resolve(args.diagnostic_state_root)
                if args.diagnostic_state_root
                else None
            ),
            compact_encoder_decoder_partitions=(
                args.compact_encoder_decoder_partitions
            ),
            route_mode=args.isolated_route_mode,
        )
        write_json(resolve(args.out), report)
        print(
            json.dumps(
                {
                    "policy": report.get("policy"),
                    "created_utc": report.get("created_utc"),
                    "trigger_state": report.get("trigger_state"),
                    "optimizer_steps": report.get("optimizer_steps"),
                    "warmup_excluded_positions_per_second": report.get(
                        "warmup_excluded_positions_per_second"
                    ),
                    "mlx_active_memory_bytes_maximum": report.get(
                        "mlx_active_memory_bytes_maximum"
                    ),
                    "hard_gaps": report.get("hard_gaps") or [],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2 if report.get("trigger_state") == "RED" else 0
    if args.training_pair_only:
        if not args.execute:
            parser.error("--training-pair-only requires --execute")
        report = run_training_pair_entry(
            config_path=resolve(args.config),
            steps=args.training_pair_steps,
            repetitions=args.training_pair_repetitions,
            compiled_microbatch_size=args.compiled_microbatch_size,
            compile_width_quantum=args.compile_width_quantum,
            materialize_compiled_state_after_update=(
                args.materialize_compiled_state_after_update
            ),
            unmigrated_implementation_challenger=(
                args.unmigrated_implementation_challenger
            ),
            training_phase=args.training_phase,
            precision_mode=args.precision_mode,
        )
        write_json(resolve(args.out), report)
        print(
            json.dumps(
                {
                    "policy": report.get("policy"),
                    "created_utc": report.get("created_utc"),
                    "trigger_state": report.get("trigger_state"),
                    "training_phase": report.get("training_phase"),
                    "precision_mode": report.get("precision_mode"),
                    "compiled_microbatch_size": report.get(
                        "compiled_microbatch_size"
                    ),
                    "median_speedup": report.get("median_speedup"),
                    "pooled_speedup": report.get("pooled_speedup"),
                    "hard_gaps": report.get("hard_gaps") or [],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2 if report.get("trigger_state") == "RED" else 0

    if args.precision_pair_only:
        if not args.execute:
            parser.error("--precision-pair-only requires --execute")
        report = run_precision_pair_entry(
            config_path=resolve(args.config),
            steps=args.precision_pair_steps,
            repetitions=args.precision_pair_repetitions,
            compiled_microbatch_size=args.compiled_microbatch_size,
            compile_width_quantum=args.compile_width_quantum,
            bf16_clear_device_cache_after_step=(
                args.bf16_clear_device_cache_after_step
            ),
            fp32_clear_device_cache_after_step=(
                args.fp32_clear_device_cache_after_step
            ),
            precision_mode=args.precision_mode,
            unmigrated_implementation_challenger=(
                args.unmigrated_implementation_challenger
            ),
        )
        write_json(resolve(args.out), report)
        print(
            json.dumps(
                {
                    "policy": report.get("policy"),
                    "created_utc": report.get("created_utc"),
                    "trigger_state": report.get("trigger_state"),
                    "compiled_microbatch_size": report.get(
                        "compiled_microbatch_size"
                    ),
                    "compile_width_quantum": report.get("compile_width_quantum"),
                    "median_speedup": report.get("median_speedup"),
                    "pooled_speedup": report.get("pooled_speedup"),
                    "adopt": report.get("adopt"),
                    "hard_gaps": report.get("hard_gaps") or [],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2 if report.get("trigger_state") == "RED" else 0

    if args.precision_resume_only:
        if not args.execute:
            parser.error("--precision-resume-only requires --execute")
        report = run_precision_resume_entry(
            config_path=resolve(args.config),
            steps=args.precision_pair_steps,
            compiled_microbatch_size=args.compiled_microbatch_size,
            compile_width_quantum=args.compile_width_quantum,
            precision_mode=args.precision_mode,
            unmigrated_implementation_challenger=(
                args.unmigrated_implementation_challenger
            ),
            repeatability_variant=args.precision_repeatability_variant,
        )
        write_json(resolve(args.out), report)
        print(
            json.dumps(
                {
                    "policy": report.get("policy"),
                    "created_utc": report.get("created_utc"),
                    "trigger_state": report.get("trigger_state"),
                    "precision_mode": report.get("precision_mode"),
                    "compiled_microbatch_size": report.get(
                        "compiled_microbatch_size"
                    ),
                    "compile_width_quantum": report.get("compile_width_quantum"),
                    "data_order_exact": report.get("data_order_exact"),
                    "data_cursor_exact": report.get("data_cursor_exact"),
                    "hard_gaps": report.get("hard_gaps") or [],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2 if report.get("trigger_state") == "RED" else 0

    metal_trace_path = resolve(args.metal_trace_out) if args.metal_trace_out else None
    if metal_trace_path is not None:
        os.environ["MTL_CAPTURE_ENABLED"] = "1"
    report = qualify(
        config_path=resolve(args.config),
        packet_path=resolve(args.packet),
        training_report_path=resolve(args.training_report),
        learning_curve_path=resolve(args.learning_curve),
        corpus_report_path=resolve(args.corpus_report),
        sample_count=args.sample_count,
        max_tokens=args.max_tokens,
        training_pair_steps=args.training_pair_steps,
        training_pair_repetitions=args.training_pair_repetitions,
        compiled_microbatch_size=args.compiled_microbatch_size,
        compile_width_quantum=args.compile_width_quantum,
        precision_pair_steps=args.precision_pair_steps,
        precision_pair_repetitions=args.precision_pair_repetitions,
        metal_trace_path=metal_trace_path,
        execute=args.execute,
    )
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report_summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] != "RED" else 2


def run_precision_resume_entry(
    *,
    config_path: Path,
    steps: int,
    compiled_microbatch_size: int,
    compile_width_quantum: int,
    precision_mode: str,
    unmigrated_implementation_challenger: bool = False,
    repeatability_variant: str = "guarded_compiled",
) -> dict[str, Any]:
    """Bind the focused precision-resume probe to the canonical durable state."""

    config = training.bind_scale_preregistration(read_json(config_path))
    plan = training.build_plan(config, config_path=config_path)
    target = (plan.get("targets") or {}).get(training.SHARED_TRUNK_ID) or {}
    receipt_path = resolve(str(target.get("receipt") or ""))
    receipt = read_json(receipt_path) if receipt_path.is_file() else {}
    checkpoint = resolve(str(receipt.get("checkpoint") or target.get("checkpoint") or ""))
    optimizer_path = resolve(
        str(receipt.get("optimizer_state") or target.get("optimizer_state") or "")
    )
    gaps = []
    if not checkpoint.is_file():
        gaps.append("shared_trunk_checkpoint_missing")
    if not optimizer_path.is_file():
        gaps.append("shared_trunk_optimizer_state_missing")
    implementation_plan_mismatch = False
    if not gaps:
        try:
            training.validate_resume(
                receipt,
                plan,
                target,
                checkpoint,
                optimizer_path,
            )
        except ValueError as exc:
            if (
                unmigrated_implementation_challenger
                and str(exc) == "resume denied: plan_identity_mismatch"
            ):
                implementation_plan_mismatch = True
            else:
                gaps.append(f"checkpoint_lineage_invalid:{exc}")
    if gaps:
        return {
            "policy": "project_theseus_focused_precision_resume_qualification_v1",
            "created_utc": now(),
            "trigger_state": "RED",
            "hard_gaps": gaps,
        }
    result = run_precision_resume_qualification(
        config=config,
        plan=plan,
        target=target,
        checkpoint=checkpoint,
        optimizer_path=optimizer_path,
        steps=steps,
        compiled_microbatch_size=compiled_microbatch_size,
        compile_width_quantum=compile_width_quantum,
        precision_mode=precision_mode,
        repeatability_variant=repeatability_variant,
    )
    return {
        **result,
        "policy": "project_theseus_focused_precision_resume_qualification_v1",
        "created_utc": now(),
        "trigger_state": (
            "RED"
            if result.get("state") == "RED"
            else "YELLOW"
            if (
                implementation_plan_mismatch
                or repeatability_variant
                == "integrated_compiled_diagnostic"
            )
            else result.get("state")
        ),
        "source_precision_policy": result.get("policy"),
        "training_config": {
            "path": relative(config_path),
            "sha256": file_sha256(config_path),
        },
        "starting_checkpoint_sha256": file_sha256(checkpoint),
        "starting_optimizer_state_sha256": file_sha256(optimizer_path),
        "compiled_microbatch_size": compiled_microbatch_size,
        "compile_width_quantum": compile_width_quantum,
        "repeatability_variant": repeatability_variant,
        "implementation_authority": (
            "DIAGNOSTIC_ONLY_UNMIGRATED_CHALLENGER"
            if implementation_plan_mismatch
            else "DIAGNOSTIC_ONLY_FINITE_STOP_ABLATION"
            if repeatability_variant
            == "integrated_compiled_diagnostic"
            else "PLAN_BOUND_QUALIFICATION"
        ),
        "hard_gaps": (
            ["precision_checkpoint_reload_integrity_fault"]
            if result.get("state") == "RED"
            else []
        ),
        "open_conditions": [
            *(
                ["precision_trajectory_repeatability_not_exact"]
                if result.get("state") == "YELLOW"
                else []
            ),
            *(
                [
                    "current implementation plan migration is not authorized",
                    "production route remains unchanged",
                ]
                if implementation_plan_mismatch
                else []
            ),
            *(
                [
                    "pre-update finite-gradient stop disabled for causality diagnosis",
                    "variant cannot receive production authority",
                ]
                if repeatability_variant
                == "integrated_compiled_diagnostic"
                else []
            ),
        ],
    }


def run_precision_pair_entry(
    *,
    config_path: Path,
    steps: int,
    repetitions: int,
    compiled_microbatch_size: int,
    compile_width_quantum: int,
    bf16_clear_device_cache_after_step: bool = False,
    fp32_clear_device_cache_after_step: bool = False,
    precision_mode: str = "bfloat16_fp32_master",
    unmigrated_implementation_challenger: bool = False,
) -> dict[str, Any]:
    """Bind a focused FP32/mixed pair to the canonical immutable checkpoint."""

    config = training.bind_scale_preregistration(read_json(config_path))
    plan = training.build_plan(config, config_path=config_path)
    target = (plan.get("targets") or {}).get(training.SHARED_TRUNK_ID) or {}
    receipt_path = resolve(str(target.get("receipt") or ""))
    receipt = read_json(receipt_path) if receipt_path.is_file() else {}
    checkpoint = resolve(str(receipt.get("checkpoint") or target.get("checkpoint") or ""))
    optimizer_path = resolve(
        str(receipt.get("optimizer_state") or target.get("optimizer_state") or "")
    )
    gaps = []
    if not checkpoint.is_file():
        gaps.append("shared_trunk_checkpoint_missing")
    if not optimizer_path.is_file():
        gaps.append("shared_trunk_optimizer_state_missing")
    implementation_plan_mismatch = False
    if not gaps:
        try:
            training.validate_resume(
                receipt,
                plan,
                target,
                checkpoint,
                optimizer_path,
            )
        except ValueError as exc:
            if (
                unmigrated_implementation_challenger
                and str(exc) == "resume denied: plan_identity_mismatch"
            ):
                implementation_plan_mismatch = True
            else:
                gaps.append(f"checkpoint_lineage_invalid:{exc}")
    if gaps:
        return {
            "policy": "project_theseus_focused_precision_pair_qualification_v1",
            "created_utc": now(),
            "trigger_state": "RED",
            "hard_gaps": gaps,
        }
    result = run_precision_pair_qualification(
        config=config,
        plan=plan,
        target=target,
        checkpoint=checkpoint,
        optimizer_path=optimizer_path,
        steps=steps,
        repetitions=repetitions,
        compiled_microbatch_size=compiled_microbatch_size,
        compile_width_quantum=compile_width_quantum,
        bf16_clear_device_cache_after_step=bf16_clear_device_cache_after_step,
        fp32_clear_device_cache_after_step=fp32_clear_device_cache_after_step,
        candidate_precision_mode=precision_mode,
    )
    return {
        **result,
        "policy": "project_theseus_focused_precision_pair_qualification_v1",
        "created_utc": now(),
        "trigger_state": (
            "RED"
            if result.get("state") == "RED"
            else "YELLOW"
            if implementation_plan_mismatch
            else result.get("state")
        ),
        "source_precision_policy": result.get("policy"),
        "training_config": {
            "path": relative(config_path),
            "sha256": file_sha256(config_path),
        },
        "starting_checkpoint_sha256": file_sha256(checkpoint),
        "starting_optimizer_state_sha256": file_sha256(optimizer_path),
        "compiled_microbatch_size": compiled_microbatch_size,
        "compile_width_quantum": compile_width_quantum,
        "implementation_authority": (
            "DIAGNOSTIC_ONLY_UNMIGRATED_CHALLENGER"
            if implementation_plan_mismatch
            else "PLAN_BOUND_QUALIFICATION"
        ),
        "bf16_clear_device_cache_after_step": bool(
            bf16_clear_device_cache_after_step
        ),
        "fp32_clear_device_cache_after_step": bool(
            fp32_clear_device_cache_after_step
        ),
        "hard_gaps": (
            ["mixed_precision_numeric_or_loss_integrity_fault"]
            if result.get("state") == "RED"
            else []
        ),
        "open_conditions": (
            [
                "current implementation plan migration is not authorized",
                "production route remains unchanged",
            ]
            if implementation_plan_mismatch
            else []
        ),
    }


def run_training_pair_entry(
    *,
    config_path: Path,
    steps: int,
    repetitions: int,
    compiled_microbatch_size: int,
    compile_width_quantum: int,
    materialize_compiled_state_after_update: bool,
    unmigrated_implementation_challenger: bool,
    training_phase: str,
    precision_mode: str,
) -> dict[str, Any]:
    """Bind a focused eager/compiled phase pair to the immutable trunk."""

    config = training.bind_scale_preregistration(read_json(config_path))
    plan = training.build_plan(config, config_path=config_path)
    target = (plan.get("targets") or {}).get(training.SHARED_TRUNK_ID) or {}
    receipt_path = resolve(str(target.get("receipt") or ""))
    receipt = read_json(receipt_path) if receipt_path.is_file() else {}
    checkpoint = resolve(str(receipt.get("checkpoint") or target.get("checkpoint") or ""))
    optimizer_path = resolve(
        str(receipt.get("optimizer_state") or target.get("optimizer_state") or "")
    )
    gaps = []
    if not checkpoint.is_file():
        gaps.append("shared_trunk_checkpoint_missing")
    if not optimizer_path.is_file():
        gaps.append("shared_trunk_optimizer_state_missing")
    implementation_plan_mismatch = False
    if not gaps:
        try:
            training.validate_resume(
                receipt,
                plan,
                target,
                checkpoint,
                optimizer_path,
            )
        except ValueError as exc:
            if (
                unmigrated_implementation_challenger
                and str(exc) == "resume denied: plan_identity_mismatch"
            ):
                implementation_plan_mismatch = True
            else:
                gaps.append(f"checkpoint_lineage_invalid:{exc}")
    if gaps:
        return {
            "policy": "project_theseus_focused_training_pair_qualification_v1",
            "created_utc": now(),
            "trigger_state": "RED",
            "hard_gaps": gaps,
        }
    result = run_training_pair_qualification(
        config=config,
        plan=plan,
        target=target,
        checkpoint=checkpoint,
        optimizer_path=optimizer_path,
        steps=steps,
        repetitions=repetitions,
        compiled_microbatch_size=compiled_microbatch_size,
        compile_width_quantum=compile_width_quantum,
        materialize_compiled_state_after_update=(
            materialize_compiled_state_after_update
        ),
        training_phase=training_phase,
        precision_mode=precision_mode,
    )
    return {
        **result,
        "policy": "project_theseus_focused_training_pair_qualification_v1",
        "created_utc": now(),
        "trigger_state": (
            "YELLOW" if implementation_plan_mismatch else result.get("state")
        ),
        "source_training_policy": result.get("policy"),
        "training_phase": training_phase,
        "precision_mode": precision_mode,
        "materialize_compiled_state_after_update": bool(
            materialize_compiled_state_after_update
        ),
        "implementation_authority": (
            "DIAGNOSTIC_ONLY_UNMIGRATED_CHALLENGER"
            if implementation_plan_mismatch
            else "PLAN_BOUND_QUALIFICATION"
        ),
        "open_conditions": (
            [
                "current implementation plan migration is not authorized",
                "production route remains unchanged",
            ]
            if implementation_plan_mismatch
            else []
        ),
        "training_config": {
            "path": relative(config_path),
            "sha256": file_sha256(config_path),
        },
        "hard_gaps": [],
    }


def run_compiled_route_entry(
    *,
    config_path: Path,
    steps: int,
    compiled_microbatch_size: int,
    compile_width_quantum: int,
    materialize_compiled_state_after_update: bool,
    unmigrated_implementation_challenger: bool,
    training_phase: str,
    precision_mode: str,
    diagnostic_state_root: Path | None,
    compact_encoder_decoder_partitions: bool,
    route_mode: str,
) -> dict[str, Any]:
    """Measure one compiled implementation without an eager route in-process."""

    config = training.bind_scale_preregistration(read_json(config_path))
    plan = training.build_plan(config, config_path=config_path)
    target = (plan.get("targets") or {}).get(training.SHARED_TRUNK_ID) or {}
    receipt_path = resolve(str(target.get("receipt") or ""))
    receipt = read_json(receipt_path) if receipt_path.is_file() else {}
    checkpoint = resolve(
        str(receipt.get("checkpoint") or target.get("checkpoint") or "")
    )
    optimizer_path = resolve(
        str(
            receipt.get("optimizer_state")
            or target.get("optimizer_state")
            or ""
        )
    )
    gaps: list[str] = []
    for label, path in (
        ("shared_trunk_checkpoint", checkpoint),
        ("shared_trunk_optimizer_state", optimizer_path),
    ):
        if not path.is_file():
            gaps.append(f"{label}_missing")
    implementation_plan_mismatch = False
    if not gaps:
        try:
            training.validate_resume(
                receipt,
                plan,
                target,
                checkpoint,
                optimizer_path,
            )
        except ValueError as exc:
            if (
                unmigrated_implementation_challenger
                and str(exc) == "resume denied: plan_identity_mismatch"
            ):
                implementation_plan_mismatch = True
            else:
                gaps.append(f"checkpoint_lineage_invalid:{exc}")
    if gaps:
        return {
            "policy": "project_theseus_isolated_compiled_route_diagnostic_v1",
            "created_utc": now(),
            "trigger_state": "RED",
            "hard_gaps": gaps,
        }
    route_context = build_training_route_context(
        config=config,
        plan=plan,
        target=target,
        checkpoint=checkpoint,
        optimizer_path=optimizer_path,
        steps=steps,
        compiled_microbatch_size=compiled_microbatch_size,
        compile_width_quantum=compile_width_quantum,
        training_phase=training_phase,
    )
    route = run_training_route(
        mode=route_mode,
        precision_mode=precision_mode,
        rope_kernel="mlx_fast",
        prune_inactive_auxiliary_outputs=True,
        capture_content_digest=True,
        diagnostic_state_root=diagnostic_state_root,
        materialize_compiled_state_after_update=(
            materialize_compiled_state_after_update
        ),
        compact_encoder_decoder_partitions=(
            compact_encoder_decoder_partitions
        ),
        eager_gradient_accumulation_microbatch_size=(
            1 if route_mode == "eager" and training_phase != "pretraining" else 0
        ),
        **route_context,
    )
    return {
        **route,
        "policy": "project_theseus_isolated_compiled_route_diagnostic_v1",
        "created_utc": now(),
        "trigger_state": (
            "YELLOW" if implementation_plan_mismatch else "GREEN"
        ),
        "training_phase": training_phase,
        "precision_mode": precision_mode,
        "materialize_compiled_state_after_update": bool(
            materialize_compiled_state_after_update
        ),
        "compact_encoder_decoder_partitions": bool(
            compact_encoder_decoder_partitions
        ),
        "route_mode": route_mode,
        "implementation_authority": (
            "DIAGNOSTIC_ONLY_UNMIGRATED_CHALLENGER"
            if implementation_plan_mismatch
            else "PLAN_BOUND_QUALIFICATION"
        ),
        "starting_checkpoint_sha256": file_sha256(checkpoint),
        "starting_optimizer_state_sha256": file_sha256(optimizer_path),
        "training_config": {
            "path": relative(config_path),
            "sha256": file_sha256(config_path),
        },
        "open_conditions": (
            [
                "matched route parity and resume qualification pending",
                "current implementation plan migration is not authorized",
                "production route remains unchanged",
            ]
            if implementation_plan_mismatch
            else ["matched route parity and resume qualification pending"]
        ),
        "hard_gaps": [],
    }


def run_joined_training_entry(
    *,
    config_path: Path,
    pretraining_steps: int,
    source_steps: int,
    supervision_steps: int,
) -> dict[str, Any]:
    """Replay the registered joined curriculum in an ephemeral exact lineage."""

    started = time.perf_counter()
    config = training.bind_scale_preregistration(read_json(config_path))
    plan = training.build_plan(config, config_path=config_path)
    target = (plan.get("targets") or {}).get(training.SHARED_TRUNK_ID) or {}
    receipt_path = resolve(str(target.get("receipt") or ""))
    receipt = read_json(receipt_path) if receipt_path.is_file() else {}
    checkpoint = resolve(
        str(receipt.get("checkpoint") or target.get("checkpoint") or "")
    )
    optimizer_path = resolve(
        str(
            receipt.get("optimizer_state")
            or target.get("optimizer_state")
            or ""
        )
    )
    rng_value = str(receipt.get("mlx_rng_state") or "")
    rng_path = resolve(rng_value) if rng_value else None
    gaps: list[str] = []
    for label, path in (
        ("shared_trunk_checkpoint", checkpoint),
        ("shared_trunk_optimizer_state", optimizer_path),
    ):
        if not path.is_file():
            gaps.append(f"{label}_missing")
    if rng_path is not None and not rng_path.is_file():
        gaps.append("shared_trunk_mlx_rng_state_missing")
    if not gaps:
        try:
            training.validate_resume(
                receipt,
                plan,
                target,
                checkpoint,
                optimizer_path,
            )
        except ValueError as exc:
            gaps.append(f"checkpoint_lineage_invalid:{exc}")
    execution_policy = dict(plan.get("execution_policy") or {})
    expected_phase_policy = {
        "compute_dtype": "float32",
        "fp32_master": False,
        "pretraining": ("compiled", 4),
        "source_conditioned_pretraining": ("eager", 1),
        "supervision": ("eager", 1),
    }
    if (
        execution_policy.get("compute_dtype")
        != expected_phase_policy["compute_dtype"]
        or execution_policy.get("fp32_master")
        is not expected_phase_policy["fp32_master"]
    ):
        gaps.append("registered_joined_precision_policy_mismatch")
    for phase in (
        "pretraining",
        "source_conditioned_pretraining",
        "supervision",
    ):
        row = dict(execution_policy.get(phase) or {})
        expected_mode, expected_microbatch = expected_phase_policy[phase]
        if (
            row.get("training_step_mode") != expected_mode
            or int(row.get("compiled_microbatch_size") or 0)
            != expected_microbatch
        ):
            gaps.append(f"registered_joined_phase_policy_mismatch:{phase}")
    if gaps:
        return {
            "policy": "project_theseus_joined_training_acceleration_qualification_v1",
            "created_utc": now(),
            "trigger_state": "RED",
            "hard_gaps": gaps,
        }

    canonical_before = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "optimizer_state_sha256": file_sha256(optimizer_path),
        "mlx_rng_state_sha256": (
            file_sha256(rng_path) if rng_path is not None else ""
        ),
        "receipt_sha256": file_sha256(receipt_path),
    }
    stage_started = time.perf_counter()
    stage_dir = resolve(str(config["stage_dir"]))
    metadata = read_json(stage_dir / "stage_metadata_v1.json")
    base = read_json(resolve(str(config["base_config"])))
    canonical = metadata["summary"]["canonical_pretrain_stage"]
    stage = training.canonical_pretraining_execution_stage(
        stage_dir,
        canonical,
        active=True,
    )
    cache_prepare_started = time.perf_counter()
    cache_paths: dict[str, Path] = {}
    for artifact_field in (
        "source_conditioned_artifacts",
        "supervision_artifacts",
    ):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "moecot_auxiliary_stage_cache.py"),
                "--config",
                str(config_path),
                "--target",
                training.SHARED_TRUNK_ID,
                "--artifact-field",
                artifact_field,
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        receipt_policy = (
            "project_theseus_moecot_source_conditioned_arrays_v1"
            if artifact_field == "source_conditioned_artifacts"
            else "project_theseus_moecot_exact_supervision_arrays_v1"
        )
        cache_paths[artifact_field] = training.auxiliary_stage_cache_path(
            config,
            base,
            target,
            metadata=metadata,
            artifact_field=artifact_field,
            receipt_policy=receipt_policy,
        )
    cache_prepare_seconds = time.perf_counter() - cache_prepare_started
    source_stage = training.defer_target_supervision(
        config,
        base,
        target,
        metadata=metadata,
        artifact_field="source_conditioned_artifacts",
        receipt_policy="project_theseus_moecot_source_conditioned_arrays_v1",
        cache_path=cache_paths["source_conditioned_artifacts"],
    )
    supervision_stage = training.defer_target_supervision(
        config,
        base,
        target,
        metadata=metadata,
        cache_path=cache_paths["supervision_artifacts"],
    )
    stage_seconds = time.perf_counter() - stage_started
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils
    phase_steps = {
        "pretraining": int(pretraining_steps),
        "source_conditioned_pretraining": int(source_steps),
        "supervision": int(supervision_steps),
    }
    training_receipt: dict[str, Any] = {}
    final_validation: dict[str, Any] | None = None
    final_validation_completed = False
    independent_reload: dict[str, Any] = {}
    clone_seconds = 0.0
    with tempfile.TemporaryDirectory(
        prefix="theseus-joined-training-", dir="/private/tmp"
    ) as temporary_root:
        scratch_root = Path(temporary_root)
        scratch_target = training.scratch_target_contract(
            target, scratch_root
        )
        scratch_checkpoint = Path(str(scratch_target["checkpoint"]))
        scratch_optimizer = Path(str(scratch_target["optimizer_state"]))
        scratch_receipt_path = Path(str(scratch_target["receipt"]))
        scratch_rng = training.rng_state_path(scratch_optimizer)
        scratch_checkpoint.parent.mkdir(parents=True, exist_ok=True)

        def clone_exact(source: Path, destination: Path) -> None:
            destination.unlink(missing_ok=True)
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)

        clone_started = time.perf_counter()
        clone_exact(checkpoint, scratch_checkpoint)
        clone_exact(optimizer_path, scratch_optimizer)
        if rng_path is not None:
            clone_exact(rng_path, scratch_rng)
        scratch_receipt = dict(receipt)
        scratch_receipt.update(
            {
                "checkpoint": str(scratch_checkpoint),
                "optimizer_state": str(scratch_optimizer),
            }
        )
        if rng_path is not None:
            scratch_receipt["mlx_rng_state"] = str(scratch_rng)
        else:
            scratch_receipt.pop("mlx_rng_state", None)
            scratch_receipt.pop("mlx_rng_state_sha256", None)
        training.write_json_atomic(scratch_receipt_path, scratch_receipt)
        clone_seconds = time.perf_counter() - clone_started
        training_receipt = training.train_target(
            config,
            plan,
            scratch_target,
            stage=stage,
            source_conditioned_stage=source_stage,
            kernel_english_stage=None,
            supervision_stage=supervision_stage,
            max_steps=sum(phase_steps.values()),
            resume=True,
            training_phase="all",
            mx=mx,
            nn=nn,
            optim=optim,
            mlx_utils=mlx_utils,
            qualification_phase_step_limits=phase_steps,
        )
        final_checkpoint = Path(str(training_receipt["checkpoint"]))
        final_optimizer = Path(str(training_receipt["optimizer_state"]))
        final_plan_migration = training.validate_resume(
            training_receipt,
            plan,
            scratch_target,
            final_checkpoint,
            final_optimizer,
        )
        final_validation = {
            "state": "GREEN",
            "plan_identity_migration": final_plan_migration,
        }
        final_validation_completed = True
        release_accelerator_route_state(mx)
        reload_started = time.perf_counter()
        reloaded_model = mx.load(str(final_checkpoint))
        reloaded_optimizer = mx.load(str(final_optimizer))
        reloaded_rng = mx.load(str(training_receipt["mlx_rng_state"]))
        independent_reload = {
            "seconds": round(time.perf_counter() - reload_started, 6),
            "model": tree_numeric_receipt(
                reloaded_model, mx=mx, mlx_utils=mlx_utils
            ),
            "optimizer": tree_numeric_receipt(
                reloaded_optimizer, mx=mx, mlx_utils=mlx_utils
            ),
            "rng": tree_numeric_receipt(
                reloaded_rng, mx=mx, mlx_utils=mlx_utils
            ),
            "checkpoint_sha256": file_sha256(final_checkpoint),
            "optimizer_state_sha256": file_sha256(final_optimizer),
            "mlx_rng_state_sha256": file_sha256(
                Path(str(training_receipt["mlx_rng_state"]))
            ),
        }
        del reloaded_model, reloaded_optimizer, reloaded_rng
        release_accelerator_route_state(mx)

    canonical_after = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "optimizer_state_sha256": file_sha256(optimizer_path),
        "mlx_rng_state_sha256": (
            file_sha256(rng_path) if rng_path is not None else ""
        ),
        "receipt_sha256": file_sha256(receipt_path),
    }
    phase_reports = dict(training_receipt.get("phases") or {})
    selected_phases = {
        phase: phase_reports[phase]
        for phase in phase_steps
    }
    phase_positions = {
        phase: int(row.get("target_positions_consumed") or 0)
        for phase, row in selected_phases.items()
    }
    phase_device_seconds = {
        phase: float(row.get("device_step_seconds_total") or 0.0)
        for phase, row in selected_phases.items()
    }
    total_positions = sum(phase_positions.values())
    total_device_seconds = sum(phase_device_seconds.values())
    elapsed_seconds = time.perf_counter() - started
    publication = dict(
        training_receipt.get("checkpoint_publication") or {}
    )
    phase_counts_exact = all(
        int(selected_phases[phase].get("optimizer_steps") or 0)
        == int(expected)
        for phase, expected in phase_steps.items()
    )
    canonical_unchanged = canonical_before == canonical_after
    reload_exact = bool(
        independent_reload.get("checkpoint_sha256")
        == training_receipt.get("checkpoint_sha256")
        and independent_reload.get("optimizer_state_sha256")
        == training_receipt.get("optimizer_state_sha256")
        and independent_reload.get("mlx_rng_state_sha256")
        == training_receipt.get("mlx_rng_state_sha256")
        and all(
            bool((independent_reload.get(key) or {}).get("all_finite"))
            for key in ("model", "optimizer", "rng")
        )
    )
    hard_gaps = []
    if not phase_counts_exact:
        hard_gaps.append("joined_phase_step_count_mismatch")
    if not canonical_unchanged:
        hard_gaps.append("canonical_lineage_mutated_by_qualification")
    if not reload_exact:
        hard_gaps.append("joined_checkpoint_independent_reload_failed")
    if not final_validation_completed:
        hard_gaps.append("joined_checkpoint_resume_validation_missing")
    return {
        "policy": "project_theseus_joined_training_acceleration_qualification_v1",
        "created_utc": now(),
        "trigger_state": "RED" if hard_gaps else "GREEN",
        "training_config": {
            "path": relative(config_path),
            "sha256": file_sha256(config_path),
        },
        "plan_sha256": plan["plan_sha256"],
        "starting_optimizer_step": int(receipt.get("optimizer_steps") or 0),
        "starting_optimizer_positions": int(
            receipt.get("optimizer_positions") or 0
        ),
        "starting_rng_policy": (
            "durable_artifact"
            if rng_path is not None
            else "legacy_exact_seed_plus_optimizer_step_reconstruction"
        ),
        "execution_policy": execution_policy,
        "auxiliary_cache_prepare_seconds": round(
            cache_prepare_seconds, 6
        ),
        "auxiliary_cache_paths": {
            key: relative(path) for key, path in cache_paths.items()
        },
        "phase_steps": phase_steps,
        "phase_positions": phase_positions,
        "phase_device_seconds": {
            key: round(value, 6)
            for key, value in phase_device_seconds.items()
        },
        "phase_positions_per_second": {
            phase: round(
                phase_positions[phase]
                / max(1e-12, phase_device_seconds[phase]),
                3,
            )
            for phase in phase_steps
        },
        "joined_training_positions": total_positions,
        "joined_training_device_seconds": round(
            total_device_seconds, 6
        ),
        "joined_training_positions_per_second": round(
            total_positions / max(1e-12, total_device_seconds), 3
        ),
        "joined_end_to_end_seconds": round(elapsed_seconds, 6),
        "joined_end_to_end_positions_per_second": round(
            total_positions / max(1e-12, elapsed_seconds), 3
        ),
        "stage_materialization_seconds": round(stage_seconds, 6),
        "exact_lineage_clone_seconds": round(clone_seconds, 6),
        "checkpoint_publication": publication,
        "checkpoint_publication_fraction_of_end_to_end": round(
            float(publication.get("total_seconds") or 0.0)
            / max(1e-12, elapsed_seconds),
            8,
        ),
        "phase_reports": selected_phases,
        "phase_step_counts_exact": phase_counts_exact,
        "final_resume_validation": final_validation,
        "independent_reload": independent_reload,
        "canonical_before": canonical_before,
        "canonical_after": canonical_after,
        "canonical_lineage_unchanged": canonical_unchanged,
        "ephemeral_checkpoint_namespace_removed": True,
        "checkpoint_or_registered_training_state_written": False,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "capability_claim": "NONE_ACCELERATION_AND_LIFECYCLE_ONLY",
        "hard_gaps": hard_gaps,
    }


def qualify(
    *,
    config_path: Path,
    packet_path: Path,
    training_report_path: Path,
    learning_curve_path: Path,
    corpus_report_path: Path,
    sample_count: int,
    max_tokens: int,
    training_pair_steps: int,
    training_pair_repetitions: int,
    compiled_microbatch_size: int,
    compile_width_quantum: int,
    precision_pair_steps: int,
    precision_pair_repetitions: int,
    metal_trace_path: Path | None,
    execute: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    process_resources_before = process_resource_receipt()
    system_memory_before = system_memory_receipt()
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
    performance_shortfalls: list[str] = []
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
    corpus_evidence = (
        read_json(corpus_report_path)
        if corpus_report_path.is_file()
        else {"state": "MISSING", "path": relative(corpus_report_path)}
    )
    if training_evidence.get("state") != "READY":
        gaps.append("training_acceleration_evidence_missing")
    if learning_evidence.get("state") != "READY":
        gaps.append("private_dev_learning_evidence_missing")
    corpus_acceleration_ready = bool(
        corpus_evidence.get("state") == "GREEN"
        and corpus_evidence.get("route_adoption_ready")
    )
    if execute and not corpus_acceleration_ready:
        gaps.append("corpus_to_tensor_acceleration_evidence_missing_or_unqualified")

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
    metal_trace = {"state": "NOT_REQUESTED"}
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
            compile_width_quantum=compile_width_quantum,
        )
        training_evidence["paired_canary"] = training_pair
        pair_acceptance = training_pair.get("acceptance") or {}
        if not (
            pair_acceptance.get("bounded_loss_parity_every_trial") is True
            and pair_acceptance.get("bounded_full_parameter_parity_every_trial")
            is True
        ):
            gaps.append("same_semantics_training_parity_failed")
        if not (
            pair_acceptance.get("median_speedup_at_least_2x") is True
            and pair_acceptance.get("pooled_speedup_at_least_2x") is True
        ):
            performance_shortfalls.append("same_semantics_training_speedup_below_2x")
        if metal_trace_path is not None:
            metal_trace = run_metal_trace_qualification(
                config=config,
                plan=plan,
                target=target,
                checkpoint=checkpoint,
                optimizer_path=optimizer,
                compiled_microbatch_size=compiled_microbatch_size,
                compile_width_quantum=compile_width_quantum,
                output_path=metal_trace_path,
            )
            if metal_trace.get("state") != "GREEN":
                gaps.append("mlx_metal_trace_capture_failed")
        precision = run_precision_pair_qualification(
            config=config,
            plan=plan,
            target=target,
            checkpoint=checkpoint,
            optimizer_path=optimizer,
            steps=precision_pair_steps,
            repetitions=precision_pair_repetitions,
            compiled_microbatch_size=compiled_microbatch_size,
            compile_width_quantum=compile_width_quantum,
        )
        training_evidence["precision_autotune"] = precision
        if precision.get("state") == "RED":
            gaps.append("mixed_precision_qualification_fault")
        precision_resume = run_precision_resume_qualification(
            config=config,
            plan=plan,
            target=target,
            checkpoint=checkpoint,
            optimizer_path=optimizer,
            steps=precision_pair_steps,
            compiled_microbatch_size=compiled_microbatch_size,
            compile_width_quantum=compile_width_quantum,
        )
        training_evidence["bf16_checkpoint_resume"] = precision_resume
        fp32_resume = run_precision_resume_qualification(
            config=config,
            plan=plan,
            target=target,
            checkpoint=checkpoint,
            optimizer_path=optimizer,
            steps=precision_pair_steps,
            compiled_microbatch_size=min(4, compiled_microbatch_size),
            compile_width_quantum=compile_width_quantum,
            precision_mode="float32",
        )
        training_evidence["fp32_checkpoint_resume"] = fp32_resume
        if fp32_resume.get("state") != "GREEN":
            gaps.append("canonical_fp32_exact_resume_failed")
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
        inference["quality_denominator"] = generation_quality_receipt(inference)
        if not inference["quality_denominator"]["capability_grade_speed_evidence"]:
            gaps.append("optimized_decode_successful_nonempty_quality_floor_failed")
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

    state = (
        "RED"
        if gaps
        else "YELLOW"
        if execute and performance_shortfalls
        else "GREEN"
        if execute
        else "READY"
    )
    report = {
        "policy": "project_theseus_resource_acceleration_qualification_v1",
        "created_utc": now(),
        "trigger_state": state,
        "mode": "executed" if execute else "plan",
        "hardware": hardware_receipt(),
        "process_resources": process_resource_delta(
            process_resources_before, process_resource_receipt()
        ),
        "system_memory_pressure": system_memory_delta(
            system_memory_before, system_memory_receipt()
        ),
        "config": artifact(config_path),
        "packet": artifact(packet_path),
        "training_report": artifact(training_report_path),
        "learning_curve": artifact(learning_curve_path),
        "corpus_report": artifact(corpus_report_path),
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
        "metal_trace": metal_trace,
        "private_dev_learning": learning_evidence,
        "corpus_to_tensor": corpus_evidence,
        "architecture_decision_control": decision_control,
        "checkpoint_storage": checkpoint_storage,
        "assistant_context_refresh": assistant_refresh,
        "checkpoint_load": load,
        "inference": inference,
        "resident_runtime": resident,
        "adoption": {
            "rust_exact_corpus_to_tensor": (
                "QUALIFIED_FROZEN_LINEAGE"
                if corpus_evidence.get("state") == "GREEN"
                and bool(corpus_evidence.get("route_adoption_ready"))
                else "NOT_QUALIFIED"
            ),
            "rust_kerc_dual_space_encoding": (
                "QUALIFIED_TYPED_PREPROCESSING"
                if ((corpus_evidence.get("kerc_dual_space") or {}).get("state"))
                == "GREEN"
                and bool(
                    (corpus_evidence.get("kerc_dual_space") or {}).get(
                        "route_adoption_ready"
                    )
                )
                else "NOT_QUALIFIED"
            ),
            "mlx_compiled_fixed_width_microbatch": (
                "QUALIFIED"
                if ((training_evidence.get("paired_canary") or {}).get("state") == "GREEN")
                else "SEMANTICS_QUALIFIED_SPEED_TARGET_PENDING"
                if training_evidence.get("same_semantics") is True
                else "REVIEW"
            ),
            "bf16": (
                "QUALIFIED_BFLOAT16_COMPUTE_FP32_MASTER"
                if (
                    (training_evidence.get("precision_autotune") or {}).get("adopt")
                    and (
                        training_evidence.get("bf16_checkpoint_resume") or {}
                    ).get("state")
                    == "GREEN"
                )
                else "NOT_ADOPTED"
                if execute
                else "PENDING_MIXED_PRECISION_QUALIFICATION"
            ),
            "canonical_fp32_checkpoint_resume": (
                "QUALIFIED"
                if (training_evidence.get("fp32_checkpoint_resume") or {}).get(
                    "state"
                )
                == "GREEN"
                else "NOT_QUALIFIED"
            ),
            "batched_beam_device_filter_preprune": (
                "QUALIFIED"
                if inference.get("exact_parity_case_count") == inference.get("case_count")
                and float(inference.get("uncached_aggregate_speedup") or 0.0) >= 2.0
                and bool(
                    (inference.get("quality_denominator") or {}).get(
                        "capability_grade_speed_evidence"
                    )
                )
                else "MECHANICS_PARITY_ONLY_CAPABILITY_BLOCKED"
                if inference.get("exact_parity_case_count") == inference.get("case_count")
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
        "performance_shortfalls": performance_shortfalls,
        "remaining_gaps": [
            *performance_shortfalls,
            *(
                []
                if metal_trace.get("state") == "GREEN"
                else ["canonical_mlx_metal_trace_pending"]
            ),
            *(
                []
                if corpus_acceleration_ready
                else ["corpus_to_tensor_acceleration_qualification_pending"]
            ),
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
    report["assembly_line"] = build_assembly_line(report)
    return report


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
    compile_width_quantum: int = 64,
    materialize_compiled_state_after_update: bool = False,
    training_phase: str = "pretraining",
    precision_mode: str = "float32",
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
        compile_width_quantum=compile_width_quantum,
        training_phase=training_phase,
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
                precision_mode=precision_mode,
                capture_parameter_snapshot=True,
                materialize_compiled_state_after_update=(
                    bool(materialize_compiled_state_after_update)
                    if mode == "compiled"
                    else False
                ),
                eager_gradient_accumulation_microbatch_size=(
                    compiled_microbatch_size
                    if route_context["source_conditioning"]
                    else 0
                ),
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
        "compile_width_quantum": compile_width_quantum,
        "training_phase": training_phase,
        "precision_mode": precision_mode,
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
    compile_width_quantum: int = 64,
    fp32_compiled_microbatch_size: int = 4,
    bf16_clear_device_cache_after_step: bool = False,
    fp32_clear_device_cache_after_step: bool = False,
    candidate_precision_mode: str = "bfloat16_fp32_master",
) -> dict[str, Any]:
    """Compare FP32 compiled training with mixed compute and FP32 master weights."""

    if repetitions < 2:
        raise ValueError("precision qualification requires at least two repetitions")
    if fp32_compiled_microbatch_size < 1:
        raise ValueError("FP32 compiled microbatch size must be positive")
    if candidate_precision_mode not in {
        "float16_fp32_master",
        "bfloat16_fp32_master",
    }:
        raise ValueError(
            f"unsupported mixed precision candidate: {candidate_precision_mode}"
        )
    candidate_compute_dtype = candidate_precision_mode.removesuffix(
        "_fp32_master"
    )
    route_context = build_training_route_context(
        config=config,
        plan=plan,
        target=target,
        checkpoint=checkpoint,
        optimizer_path=optimizer_path,
        steps=steps,
        compiled_microbatch_size=compiled_microbatch_size,
        compile_width_quantum=compile_width_quantum,
    )
    mx = route_context["mx"]
    trials: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        route_order = (
            ("float32", candidate_precision_mode)
            if repetition % 2 == 0
            else (candidate_precision_mode, "float32")
        )
        routes: dict[str, dict[str, Any]] = {}
        for precision_mode in route_order:
            release_accelerator_route_state(mx)
            route_microbatch_size = (
                fp32_compiled_microbatch_size
                if precision_mode == "float32"
                else compiled_microbatch_size
            )
            routes[precision_mode] = run_training_route(
                mode="compiled",
                precision_mode=precision_mode,
                **{
                    **route_context,
                    "compiled_microbatch_size": route_microbatch_size,
                    "clear_device_cache_after_step": (
                        bool(bf16_clear_device_cache_after_step)
                        if precision_mode == candidate_precision_mode
                        else bool(fp32_clear_device_cache_after_step)
                    ),
                },
            )
            release_accelerator_route_state(mx)
        baseline = routes["float32"]
        candidate = routes[candidate_precision_mode]
        baseline_rate = float(baseline["warmup_excluded_positions_per_second"])
        candidate_rate = float(candidate["warmup_excluded_positions_per_second"])
        loss_delta = float(candidate["final_loss"]) - float(baseline["final_loss"])
        relative_loss_delta = abs(loss_delta) / max(1e-12, abs(float(baseline["final_loss"])))
        trials.append(
            {
                "repetition": repetition + 1,
                "route_order": list(route_order),
                "float32": baseline,
                candidate_precision_mode: candidate,
                "speedup": round(candidate_rate / max(1e-12, baseline_rate), 6),
                "final_loss_delta": round(loss_delta, 8),
                "relative_final_loss_delta": round(relative_loss_delta, 8),
            }
        )
    baseline = aggregate_training_routes([row["float32"] for row in trials])
    candidate = aggregate_training_routes(
        [row[candidate_precision_mode] for row in trials]
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
        for route in (row["float32"], row[candidate_precision_mode])
        for section in (
            "compute_parameters",
            "authoritative_parameters",
            "optimizer_state",
        )
    )
    dtype_integrity = all(
        row[candidate_precision_mode]["compute_parameters"]["dtypes"]
        == [f"mlx.core.{candidate_compute_dtype}"]
        and row[candidate_precision_mode]["authoritative_parameters"]["dtypes"]
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
        "candidate": (
            f"{candidate_compute_dtype}_compute_fp32_master_weights_and_optimizer"
        ),
        "candidate_precision_mode": candidate_precision_mode,
        "same_starting_checkpoint_and_optimizer": True,
        "same_data_order_batch_schedule_objective_and_update_count": True,
        "fp32_compiled_microbatch_size": fp32_compiled_microbatch_size,
        "candidate_compiled_microbatch_size": compiled_microbatch_size,
        "bf16_compiled_microbatch_size": (
            compiled_microbatch_size
            if candidate_precision_mode == "bfloat16_fp32_master"
            else 0
        ),
        "float16_compiled_microbatch_size": (
            compiled_microbatch_size
            if candidate_precision_mode == "float16_fp32_master"
            else 0
        ),
        "bf16_clear_device_cache_after_step": bool(
            bf16_clear_device_cache_after_step
        ),
        "fp32_clear_device_cache_after_step": bool(
            fp32_clear_device_cache_after_step
        ),
        "steps_per_route_per_repetition": steps,
        "repetitions": repetitions,
        "route_order_control": (
            f"alternating fp32-first and {candidate_compute_dtype}-first"
        ),
        "float32": baseline,
        candidate_precision_mode: candidate,
        "trials": trials,
        "median_speedup": round(median_speedup, 6),
        "pooled_speedup": round(pooled_speedup, 6),
        "maximum_relative_final_loss_delta": round(maximum_relative_loss_delta, 8),
        "acceptance": {
            "all_numeric_state_finite": numeric_integrity,
            "mixed_compute_fp32_authority_dtypes_exact": dtype_integrity,
            "relative_final_loss_delta_at_most_0_02": loss_integrity,
            "peak_mlx_memory_nonregressed": memory_nonregressed,
            "median_speedup_at_least_1_15x": median_speedup >= 1.15,
            "pooled_speedup_at_least_1_15x": pooled_speedup >= 1.15,
        },
        "checkpoint_or_training_state_written": False,
    }


def release_accelerator_route_state(mx: Any) -> None:
    """Release cyclic model objects before the next same-process Metal route."""

    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


def run_precision_resume_qualification(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    target: dict[str, Any],
    checkpoint: Path,
    optimizer_path: Path,
    steps: int,
    compiled_microbatch_size: int,
    compile_width_quantum: int = 64,
    precision_mode: str = "bfloat16_fp32_master",
    repeatability_variant: str = "guarded_compiled",
) -> dict[str, Any]:
    """Prove selected precision state and data order survive a real reload."""

    if steps < 4:
        raise ValueError("resume qualification requires at least four updates")
    variant_contract = {
        "guarded_compiled": {
            "mode": "compiled",
            "reject_nonfinite_gradients": True,
            "eager_gradient_accumulation_microbatch_size": 0,
        },
        "integrated_compiled_diagnostic": {
            "mode": "compiled",
            "reject_nonfinite_gradients": False,
            "eager_gradient_accumulation_microbatch_size": 0,
        },
        "guarded_eager": {
            "mode": "eager",
            "reject_nonfinite_gradients": True,
            "eager_gradient_accumulation_microbatch_size": (
                compiled_microbatch_size
            ),
        },
    }
    if repeatability_variant not in variant_contract:
        raise ValueError(
            f"unsupported precision repeatability variant: {repeatability_variant}"
        )
    variant = variant_contract[repeatability_variant]
    if (
        precision_mode != "float16_fp32_master"
        and repeatability_variant != "guarded_compiled"
    ):
        raise ValueError(
            "alternate repeatability variants are restricted to FP16 diagnosis"
        )
    first_steps = steps // 2
    second_steps = steps - first_steps
    context = build_training_route_context(
        config=config,
        plan=plan,
        target=target,
        checkpoint=checkpoint,
        optimizer_path=optimizer_path,
        steps=steps,
        compiled_microbatch_size=compiled_microbatch_size,
        compile_width_quantum=compile_width_quantum,
    )
    mx = context["mx"]
    release_accelerator_route_state(mx)
    uninterrupted = run_training_route(
        mode=str(variant["mode"]),
        precision_mode=precision_mode,
        reject_nonfinite_gradients_override=bool(
            variant["reject_nonfinite_gradients"]
        ),
        eager_gradient_accumulation_microbatch_size=int(
            variant["eager_gradient_accumulation_microbatch_size"]
        ),
        capture_parameter_snapshot=True,
        capture_optimizer_snapshot=True,
        capture_rng_snapshot=True,
        **context,
    )
    uninterrupted_parameters = uninterrupted.pop("_parameter_snapshot")
    uninterrupted_optimizer = uninterrupted.pop("_optimizer_snapshot")
    uninterrupted_rng = uninterrupted.pop("_rng_snapshot")

    first_context = {**context, "steps": first_steps}
    release_accelerator_route_state(mx)
    first = run_training_route(
        mode=str(variant["mode"]),
        precision_mode=precision_mode,
        reject_nonfinite_gradients_override=bool(
            variant["reject_nonfinite_gradients"]
        ),
        eager_gradient_accumulation_microbatch_size=int(
            variant["eager_gradient_accumulation_microbatch_size"]
        ),
        capture_parameter_snapshot=True,
        capture_optimizer_snapshot=True,
        capture_rng_snapshot=True,
        **first_context,
    )
    first_parameters = first.pop("_parameter_snapshot")
    first_optimizer = first.pop("_optimizer_snapshot")
    first_rng = first.pop("_rng_snapshot")
    with tempfile.TemporaryDirectory(prefix="theseus-precision-resume-") as directory:
        root = Path(directory)
        resumed_checkpoint = root / "model.safetensors"
        resumed_optimizer = root / "optimizer.safetensors"
        resumed_rng = root / "rng.safetensors"
        checkpoint_started = time.perf_counter()
        save_array_mapping_atomic(mx, first_parameters, resumed_checkpoint)
        save_array_mapping_atomic(mx, first_optimizer, resumed_optimizer)
        save_array_mapping_atomic(mx, first_rng, resumed_rng)
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        checkpoint_receipt = {
            "checkpoint_sha256": file_sha256(resumed_checkpoint),
            "optimizer_state_sha256": file_sha256(resumed_optimizer),
            "rng_state_sha256": file_sha256(resumed_rng),
            "checkpoint_bytes": resumed_checkpoint.stat().st_size,
            "optimizer_state_bytes": resumed_optimizer.stat().st_size,
            "rng_state_bytes": resumed_rng.stat().st_size,
            "publication_seconds": round(checkpoint_seconds, 6),
            "atomic_file_replacement": True,
        }
        second_context = {
            **context,
            "checkpoint": resumed_checkpoint,
            "optimizer_path": resumed_optimizer,
            "steps": second_steps,
        }
        release_accelerator_route_state(mx)
        reloaded_rng = {
            name: value for name, value in mx.load(str(resumed_rng)).items()
        }
        resumed = run_training_route(
            mode=str(variant["mode"]),
            precision_mode=precision_mode,
            reject_nonfinite_gradients_override=bool(
                variant["reject_nonfinite_gradients"]
            ),
            eager_gradient_accumulation_microbatch_size=int(
                variant["eager_gradient_accumulation_microbatch_size"]
            ),
            resume_data_cursor=first["data_cursor_next"],
            resume_rng_state=reloaded_rng,
            capture_parameter_snapshot=True,
            capture_optimizer_snapshot=True,
            capture_rng_snapshot=True,
            capture_starting_snapshot=True,
            **second_context,
        )
        resumed_parameters = resumed.pop("_parameter_snapshot")
        resumed_optimizer_state = resumed.pop("_optimizer_snapshot")
        resumed_rng_state = resumed.pop("_rng_snapshot")
        resumed_start_parameters = resumed.pop("_starting_parameter_snapshot")
        resumed_start_optimizer = resumed.pop("_starting_optimizer_snapshot")
        resumed_start_rng = resumed.pop("_starting_rng_snapshot")

    parameter_comparison = compare_parameter_snapshots(
        uninterrupted_parameters, resumed_parameters
    )
    optimizer_comparison = compare_parameter_snapshots(
        uninterrupted_optimizer, resumed_optimizer_state
    )
    rng_comparison = compare_parameter_snapshots(
        uninterrupted_rng, resumed_rng_state
    )
    reload_parameter_comparison = compare_parameter_snapshots(
        first_parameters, resumed_start_parameters
    )
    reload_optimizer_comparison = compare_parameter_snapshots(
        first_optimizer, resumed_start_optimizer
    )
    reload_rng_comparison = compare_parameter_snapshots(
        first_rng, resumed_start_rng
    )
    batch_hashes = [
        *first.get("batch_index_sha256_prefix", []),
        *resumed.get("batch_index_sha256_prefix", []),
    ]
    exact_data_order = batch_hashes == uninterrupted.get(
        "batch_index_sha256_prefix", []
    )
    exact_cursor = resumed.get("data_cursor_next") == uninterrupted.get(
        "data_cursor_next"
    )
    final_loss_delta = abs(
        float(resumed["final_loss"]) - float(uninterrupted["final_loss"])
    )
    uninterrupted_first_losses = [
        float(value)
        for value in (uninterrupted.get("loss_prefix") or [])[:first_steps]
    ]
    split_first_losses = [
        float(value) for value in (first.get("loss_prefix") or [])[:first_steps]
    ]
    first_segment_max_loss_delta = (
        max(
            abs(left - right)
            for left, right in zip(
                uninterrupted_first_losses,
                split_first_losses,
                strict=True,
            )
        )
        if uninterrupted_first_losses
        and len(uninterrupted_first_losses) == len(split_first_losses)
        else float("inf")
    )
    reload_boundary_exact = all(
        comparison.get("within_tolerance") is True
        for comparison in (
            reload_parameter_comparison,
            reload_optimizer_comparison,
            reload_rng_comparison,
        )
    )
    trajectory_repeatable = all(
        (
            parameter_comparison.get("within_tolerance") is True,
            optimizer_comparison.get("within_tolerance") is True,
            final_loss_delta <= MAX_FINAL_LOSS_ABSOLUTE_DELTA,
        )
    )
    custody_integrity = all(
        (
            reload_boundary_exact,
            rng_comparison.get("within_tolerance") is True,
            exact_data_order,
            exact_cursor,
        )
    )
    state = (
        "RED"
        if not custody_integrity
        else "GREEN"
        if trajectory_repeatable
        else "YELLOW"
    )
    return {
        "policy": "project_theseus_mlx_precision_checkpoint_resume_v1",
        "state": state,
        "precision_mode": precision_mode,
        "repeatability_variant": repeatability_variant,
        "execution_mode": variant["mode"],
        "preupdate_finite_gradient_stop": bool(
            variant["reject_nonfinite_gradients"]
        ),
        "eager_gradient_accumulation_microbatch_size": int(
            variant["eager_gradient_accumulation_microbatch_size"]
        ),
        "updates": {
            "uninterrupted": steps,
            "before_checkpoint": first_steps,
            "after_reload": second_steps,
        },
        "checkpoint_publication": checkpoint_receipt,
        "data_order_exact": exact_data_order,
        "data_cursor_exact": exact_cursor,
        "checkpoint_reload_state": "GREEN" if reload_boundary_exact else "RED",
        "trajectory_repeatability_state": (
            "GREEN" if trajectory_repeatable else "YELLOW"
        ),
        "trajectory_divergence_predates_checkpoint": (
            first_segment_max_loss_delta > MAX_FINAL_LOSS_ABSOLUTE_DELTA
        ),
        "first_segment_max_loss_absolute_delta": round(
            first_segment_max_loss_delta, 12
        ),
        "batch_index_sha256": hashlib.sha256(
            json.dumps(batch_hashes, separators=(",", ":")).encode()
        ).hexdigest(),
        "final_loss_absolute_delta": round(final_loss_delta, 12),
        "parameter_comparison": parameter_comparison,
        "optimizer_state_comparison": optimizer_comparison,
        "rng_state_comparison": rng_comparison,
        "reload_boundary": {
            "parameter_comparison": reload_parameter_comparison,
            "optimizer_state_comparison": reload_optimizer_comparison,
            "rng_state_comparison": reload_rng_comparison,
        },
        "uninterrupted": uninterrupted,
        "before_checkpoint": first,
        "after_reload": resumed,
        "durable_training_state_mutated": False,
        "scratch_artifacts_retained": False,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "claim_boundary": "bounded exact-resume mechanics only; not convergence or capability evidence",
    }


def save_array_mapping_atomic(mx: Any, mapping: dict[str, Any], path: Path) -> None:
    """Write one flat tensor mapping without exposing a partial checkpoint."""

    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    temporary.unlink(missing_ok=True)
    mx.save_safetensors(
        str(temporary),
        {name: mx.array(value) for name, value in mapping.items()},
        metadata={"policy": "project_theseus_scratch_resume_state_v1"},
    )
    os.replace(temporary, path)


def run_metal_trace_qualification(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    target: dict[str, Any],
    checkpoint: Path,
    optimizer_path: Path,
    compiled_microbatch_size: int,
    compile_width_quantum: int = 64,
    output_path: Path,
    precision_mode: str = "float32",
) -> dict[str, Any]:
    """Capture two immutable compiled updates for native Metal inspection."""

    if output_path.exists():
        return {
            "state": "FAULT",
            "reason": "trace_output_already_exists",
            "path": relative(output_path),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    context = build_training_route_context(
        config=config,
        plan=plan,
        target=target,
        checkpoint=checkpoint,
        optimizer_path=optimizer_path,
        steps=2,
        compiled_microbatch_size=compiled_microbatch_size,
        compile_width_quantum=compile_width_quantum,
    )
    mx = context["mx"]
    capture_started = False
    capture_completed = False

    def capture_boundary(boundary: str, phase_step: int) -> None:
        nonlocal capture_started, capture_completed
        if phase_step != 2:
            return
        if boundary == "before_device_step":
            mx.metal.start_capture(str(output_path.resolve()))
            capture_started = True
        elif boundary == "after_device_step" and capture_started:
            mx.metal.stop_capture()
            capture_started = False
            capture_completed = True

    try:
        route = run_training_route(
            mode="compiled",
            precision_mode=precision_mode,
            rope_kernel="mlx_fast",
            prune_inactive_auxiliary_outputs=True,
            step_boundary_callback=capture_boundary,
            **context,
        )
    except Exception as exc:
        return {
            "state": "FAULT",
            "reason": f"{type(exc).__name__}:{exc}",
            "path": relative(output_path),
        }
    finally:
        if capture_started:
            try:
                mx.metal.stop_capture()
            except Exception:
                pass
    if not capture_completed or not output_path.is_dir():
        return {
            "state": "FAULT",
            "reason": "post_compile_capture_artifact_missing",
            "path": relative(output_path),
        }
    trace = directory_artifact(output_path)
    if int(trace["bytes"]) > MAX_RETAINED_METAL_TRACE_BYTES:
        shutil.rmtree(output_path)
        return {
            "policy": "project_theseus_mlx_metal_trace_qualification_v1",
            "state": "CAPTURED_AND_EVICTED",
            "trace": {
                **trace,
                "retained": False,
                "inspectable": False,
                "eviction_reason": "raw capture exceeded the 2 GiB evidence-retention ceiling",
            },
            "route": route,
            "checkpoint_sha256": file_sha256(checkpoint),
            "optimizer_state_sha256": file_sha256(optimizer_path),
            "optimizer_steps": 2,
            "captured_optimizer_steps": [2],
            "compile_and_first_step_excluded": True,
            "compiled_microbatch_size": compiled_microbatch_size,
            "compile_width_quantum": compile_width_quantum,
            "precision_mode": precision_mode,
            "checkpoint_or_training_state_written": False,
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "adoption_ready": False,
            "smallest_next_patch": "capture one post-compile optimizer step with bounded signposts",
            "claim_boundary": "capture identity and mechanics timings only; raw trace was not retained",
        }
    return {
        "policy": "project_theseus_mlx_metal_trace_qualification_v1",
        "state": "GREEN",
        "trace": trace,
        "route": route,
        "checkpoint_sha256": file_sha256(checkpoint),
        "optimizer_state_sha256": file_sha256(optimizer_path),
        "optimizer_steps": 2,
        "captured_optimizer_steps": [2],
        "compile_and_first_step_excluded": True,
        "compiled_microbatch_size": compiled_microbatch_size,
        "compile_width_quantum": compile_width_quantum,
        "precision_mode": precision_mode,
        "checkpoint_or_training_state_written": False,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "claim_boundary": "native trace and mechanics timings only; not speed or learning evidence",
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
    compile_width_quantum: int = 64,
    training_phase: str = "pretraining",
) -> dict[str, Any]:
    """Materialize one immutable route context shared by acceleration comparisons."""

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    setup_started = time.perf_counter()
    stage_dir = resolve(str(config["stage_dir"]))
    metadata_started = time.perf_counter()
    metadata = read_json(stage_dir / "stage_metadata_v1.json")
    base = read_json(resolve(str(config["base_config"])))
    metadata_seconds = time.perf_counter() - metadata_started
    canonical = metadata["summary"]["canonical_pretrain_stage"]
    shape = (
        int(canonical["window_count"]),
        int(canonical["max_sequence_tokens"]),
    )
    open_started = time.perf_counter()
    arrays = training.load_pretrain_memmaps(
        training.pretrain_array_paths(stage_dir),
        shape,
        expected=canonical["array_artifacts"],
    )
    memmap_open_seconds = time.perf_counter() - open_started
    view_started = time.perf_counter()
    if training_phase == "pretraining":
        inputs = training.range_view(arrays[0], target["row_ranges"])
        labels = training.range_view(arrays[1], target["row_ranges"])
        mask = training.range_view(arrays[2], target["row_ranges"])
        progress_mask = mask
        source_conditioning = False
        phase_receipt = {
            "policy": "canonical_pretraining_memmap_range_view_v1",
            "row_count": len(inputs),
        }
    elif training_phase in {
        "source_conditioned_pretraining",
        "supervision",
    }:
        materialize_kwargs = {}
        if training_phase == "source_conditioned_pretraining":
            materialize_kwargs = {
                "artifact_field": "source_conditioned_artifacts",
                "receipt_policy": (
                    "project_theseus_moecot_source_conditioned_arrays_v1"
                ),
            }
        phase_stage = training.materialize_target_supervision(
            config,
            base,
            target,
            metadata=metadata,
            **materialize_kwargs,
        )
        inputs = phase_stage.inputs
        labels = phase_stage.labels
        mask = phase_stage.loss_mask
        progress_mask = phase_stage.mask
        source_conditioning = True
        phase_receipt = phase_stage.receipt
    else:
        raise ValueError(f"unsupported acceleration training phase: {training_phase}")
    range_view_seconds = time.perf_counter() - view_started
    training_cfg = config["training"]
    total_schedule_steps = training.required_steps(
        progress_mask,
        int(training_cfg["batch_size"]),
        int(target["optimizer_target_positions"]),
    ) + 128
    vocab_size = int(target.get("vocab_size") or plan["models"]["vocab_size"])
    lookup_started = time.perf_counter()
    copy_lookup = training.build_source_to_target_lookup(
        base,
        metadata,
        vocab_size=vocab_size,
        identity_ranges=training.target_copy_identity_ranges(target),
    )
    copy_lookup_seconds = time.perf_counter() - lookup_started
    return {
        "config": config,
        "plan": plan,
        "target": target,
        "checkpoint": checkpoint,
        "optimizer_path": optimizer_path,
        "steps": steps,
        "compiled_microbatch_size": compiled_microbatch_size,
        "compile_width_quantum": compile_width_quantum,
        "inputs": inputs,
        "labels": labels,
        "mask": mask,
        "progress_mask": progress_mask,
        "source_conditioning": source_conditioning,
        "training_phase": training_phase,
        "phase_receipt": phase_receipt,
        "copy_lookup": copy_lookup,
        "total_schedule_steps": total_schedule_steps,
        "receipt": read_json(resolve(str(target["receipt"]))),
        "mx": mx,
        "nn": nn,
        "optim": optim,
        "mlx_utils": mlx_utils,
        "setup_timings": {
            "metadata_and_base_config_seconds": round(metadata_seconds, 6),
            "memmap_open_and_validation_seconds": round(memmap_open_seconds, 6),
            "range_view_seconds": round(range_view_seconds, 6),
            "copy_lookup_seconds": round(copy_lookup_seconds, 6),
            "total_seconds": round(time.perf_counter() - setup_started, 6),
        },
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
    compile_width_quantum: int,
    inputs: Any,
    labels: Any,
    mask: Any,
    progress_mask: Any,
    source_conditioning: bool,
    training_phase: str,
    phase_receipt: dict[str, Any],
    copy_lookup: Any,
    total_schedule_steps: int,
    receipt: dict[str, Any],
    mx: Any,
    nn: Any,
    optim: Any,
    mlx_utils: Any,
    setup_timings: dict[str, float],
    precision_mode: str = "float32",
    capture_parameter_snapshot: bool = False,
    capture_optimizer_snapshot: bool = False,
    capture_rng_snapshot: bool = False,
    capture_content_digest: bool = False,
    capture_starting_snapshot: bool = False,
    resume_data_cursor: dict[str, Any] | None = None,
    resume_rng_state: dict[str, Any] | None = None,
    step_boundary_callback: Any = None,
    rope_kernel: str = "mlx_fast",
    prune_inactive_auxiliary_outputs: bool = True,
    clear_device_cache_after_step: bool = False,
    eager_gradient_accumulation_microbatch_size: int = 0,
    materialize_compiled_state_after_update: bool = False,
    diagnostic_state_root: Path | None = None,
    compact_encoder_decoder_partitions: bool = False,
    reject_nonfinite_gradients_override: bool | None = None,
) -> dict[str, Any]:
    """Run one non-mutating route from the exact registered checkpoint state."""

    if precision_mode not in {
        "float32",
        "float16_fp32_master",
        "bfloat16_fp32_master",
    }:
        raise ValueError(f"unsupported precision mode: {precision_mode}")
    compute_dtype_name = (
        precision_mode.removesuffix("_fp32_master")
        if precision_mode != "float32"
        else "float32"
    )
    compute_dtype = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }[compute_dtype_name]
    gradient_loss_scale = (
        128.0 if precision_mode == "float16_fp32_master" else 1.0
    )
    reject_nonfinite_gradients = (
        precision_mode == "float16_fp32_master"
        if reject_nonfinite_gradients_override is None
        else bool(reject_nonfinite_gradients_override)
    )
    training_cfg = config["training"]
    vocab_size = int(target.get("vocab_size") or plan["models"]["vocab_size"])
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    mx.random.seed(int(config["seed"]) + training.stable_int(training.SHARED_TRUNK_ID))
    construct_started = time.perf_counter()
    model = training.build_model(
        training.CausalTransformerConfig(vocab_size=vocab_size, **target["model"]),
        mx=mx,
        nn=nn,
        state_role_lookup=None,
        source_to_target_lookup=copy_lookup,
        rope_kernel=rope_kernel,
        compact_encoder_decoder_partitions=(
            compact_encoder_decoder_partitions
        ),
        compact_partition_width_quantum=(
            64 if compact_encoder_decoder_partitions else 0
        ),
    )
    master_model = None
    if precision_mode != "float32":
        master_model = training.build_model(
            training.CausalTransformerConfig(vocab_size=vocab_size, **target["model"]),
            mx=mx,
            nn=nn,
            state_role_lookup=None,
            source_to_target_lookup=copy_lookup,
            rope_kernel=rope_kernel,
            compact_encoder_decoder_partitions=(
                compact_encoder_decoder_partitions
            ),
            compact_partition_width_quantum=(
                64 if compact_encoder_decoder_partitions else 0
            ),
        )
    model_construct_seconds = time.perf_counter() - construct_started
    optimizer_construct_started = time.perf_counter()
    schedule = training.build_schedule(optim, mx, training_cfg, total_schedule_steps)
    optimizer = optim.AdamW(
        learning_rate=schedule,
        weight_decay=float(training_cfg["weight_decay"]),
    )
    optimizer_construct_seconds = time.perf_counter() - optimizer_construct_started
    restore_started = time.perf_counter()
    model.load_weights(str(checkpoint))
    if master_model is not None:
        master_model.load_weights(str(checkpoint))
        model.set_dtype(compute_dtype)
    optimizer.state = mlx_utils.tree_unflatten(list(mx.load(str(optimizer_path)).items()))
    mx.eval(
        model.parameters(),
        master_model.parameters() if master_model is not None else model.parameters(),
        optimizer.state,
    )
    checkpoint_restore_seconds = time.perf_counter() - restore_started
    if resume_rng_state is not None:
        expected_rng_names = [f"state.{index}" for index in range(len(mx.random.state))]
        if sorted(resume_rng_state) != expected_rng_names:
            raise ValueError("MLX RNG state shape contract mismatch")
        mx.random.state = [
            resume_rng_state[name] for name in expected_rng_names
        ]
        mx.eval(*mx.random.state)
    starting_parameter_snapshot = None
    starting_optimizer_snapshot = None
    starting_rng_snapshot = None
    if capture_starting_snapshot:
        authoritative_model = master_model if master_model is not None else model
        starting_parameter_snapshot = {
            name: np.asarray(value).copy()
            for name, value in mlx_utils.tree_flatten(
                authoritative_model.trainable_parameters()
            )
        }
        starting_optimizer_snapshot = {
            name: np.asarray(value).copy()
            for name, value in mlx_utils.tree_flatten(optimizer.state)
        }
        starting_rng_snapshot = {
            f"state.{index}": np.asarray(value).copy()
            for index, value in enumerate(mx.random.state)
        }
    if master_model is not None and source_conditioning is False:
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
    if gradient_loss_scale != 1.0:
        unscaled_loss_function = loss_function

        def scaled_loss_function(*loss_args: Any, **loss_kwargs: Any) -> Any:
            return (
                unscaled_loss_function(*loss_args, **loss_kwargs)
                * gradient_loss_scale
            )

        loss_function = scaled_loss_function
    phase = training.train_phase(
        model,
        optimizer,
        nn.value_and_grad(model, loss_function),
        inputs,
        labels,
        mask,
        progress_mask=progress_mask,
        ordered_plan_loss_weight=1.0,
        sample_weights=None,
        plan_labels=None,
        plan_label_mode="none",
        plan_auxiliary_weight=0.0,
        plan_shuffle_seed=0,
        plan_loss_mode="binary_multilabel",
        plan_slot_count=0,
        plan_factor_group_sizes=(),
        phase_name=f"resource_acceleration_{training_phase}_{mode}_reference",
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
        resume_data_cursor=resume_data_cursor,
        mx=mx,
        optim=optim,
        source_conditioning=source_conditioning,
        source_to_target_lookup=copy_lookup,
        training_step_mode=mode,
        compiled_microbatch_size=compiled_microbatch_size,
        compile_width_quantum=compile_width_quantum,
        materialize_compiled_state_after_update=(
            materialize_compiled_state_after_update
        ),
        eager_gradient_accumulation_microbatch_size=(
            eager_gradient_accumulation_microbatch_size
        ),
        master_model=master_model,
        compute_dtype_name=compute_dtype_name,
        gradient_loss_scale=gradient_loss_scale,
        reject_nonfinite_gradients=reject_nonfinite_gradients,
        step_boundary_callback=step_boundary_callback,
        clear_device_cache_after_step=clear_device_cache_after_step,
    )
    authoritative_model = master_model if master_model is not None else model
    observed = {
        "training_step_execution": phase["training_step_execution"],
        "setup_timings": setup_timings,
        "model_construct_seconds": round(model_construct_seconds, 6),
        "optimizer_construct_seconds": round(optimizer_construct_seconds, 6),
        "checkpoint_restore_seconds": round(checkpoint_restore_seconds, 6),
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
        "loss_prefix": phase["loss_prefix"],
        "optimizer_step_seconds_prefix": phase["optimizer_step_seconds_prefix"],
        "compiled_accumulation_seconds_total": phase[
            "compiled_accumulation_seconds_total"
        ],
        "compiled_update_seconds_total": phase["compiled_update_seconds_total"],
        "compiled_state_materialization_seconds_total": phase[
            "compiled_state_materialization_seconds_total"
        ],
        "host_batch_preparation_seconds_total": phase[
            "host_batch_preparation_seconds_total"
        ],
        "unit_allocator_pack_seconds_total": phase[
            "unit_allocator_pack_seconds_total"
        ],
        "device_step_seconds_total": phase["device_step_seconds_total"],
        "static_sequence_width": phase["static_sequence_width"],
        "maximum_dynamic_batch_width": phase["maximum_dynamic_batch_width"],
        "mean_dynamic_batch_width": phase["mean_dynamic_batch_width"],
        "maximum_execution_batch_width": phase["maximum_execution_batch_width"],
        "padded_positions_avoided": phase["padded_positions_avoided"],
        "compile_width_quantum": phase["compile_width_quantum"],
        "compiled_microbatch_size": phase["compiled_microbatch_size"],
        "compiled_accumulation_seconds_prefix": phase[
            "compiled_accumulation_seconds_prefix"
        ],
        "compiled_update_seconds_prefix": phase[
            "compiled_update_seconds_prefix"
        ],
        "compiled_state_materialization_seconds_prefix": phase[
            "compiled_state_materialization_seconds_prefix"
        ],
        "mlx_active_memory_bytes_maximum": phase[
            "mlx_active_memory_bytes_maximum"
        ],
        "mlx_active_memory_bytes_prefix": phase[
            "mlx_active_memory_bytes_prefix"
        ],
        "mlx_cache_memory_bytes_maximum": phase[
            "mlx_cache_memory_bytes_maximum"
        ],
        "mlx_peak_memory_bytes_maximum": phase[
            "mlx_peak_memory_bytes_maximum"
        ],
        "materialize_compiled_state_after_update": phase[
            "materialize_compiled_state_after_update"
        ],
        "data_cursor_start": phase["data_cursor_start"],
        "data_cursor_next": phase["data_cursor_next"],
        "batch_index_sha256_prefix": phase["batch_index_sha256_prefix"],
        "precision_mode": precision_mode,
        "compute_dtype": compute_dtype_name,
        "gradient_loss_scale": gradient_loss_scale,
        "reject_nonfinite_gradients": reject_nonfinite_gradients,
        "training_phase": training_phase,
        "source_conditioning": source_conditioning,
        "phase_receipt": phase_receipt,
        "rope_kernel": rope_kernel,
        "prune_inactive_auxiliary_outputs": prune_inactive_auxiliary_outputs,
        "compact_encoder_decoder_partitions": bool(
            compact_encoder_decoder_partitions
        ),
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
    if capture_content_digest:
        observed["parameter_content"] = tree_content_receipt(
            authoritative_model.trainable_parameters(),
            mlx_utils=mlx_utils,
        )
        observed["optimizer_content"] = tree_content_receipt(
            optimizer.state,
            mlx_utils=mlx_utils,
        )
        observed["rng_content"] = tree_content_receipt(
            {
                f"state.{index}": value
                for index, value in enumerate(mx.random.state)
            },
            mlx_utils=mlx_utils,
        )
    if diagnostic_state_root is not None:
        diagnostic_state_root.mkdir(parents=True, exist_ok=True)
        model_state_path = diagnostic_state_root / "model.safetensors"
        optimizer_state_path = diagnostic_state_root / "optimizer.safetensors"
        rng_state_path = diagnostic_state_root / "rng.safetensors"
        mx.save_safetensors(
            str(model_state_path),
            dict(mlx_utils.tree_flatten(
                authoritative_model.trainable_parameters()
            )),
            metadata={"policy": "project_theseus_scratch_state_diagnostic_v1"},
        )
        mx.save_safetensors(
            str(optimizer_state_path),
            dict(mlx_utils.tree_flatten(optimizer.state)),
            metadata={"policy": "project_theseus_scratch_state_diagnostic_v1"},
        )
        mx.save_safetensors(
            str(rng_state_path),
            {
                f"state.{index}": value
                for index, value in enumerate(mx.random.state)
            },
            metadata={"policy": "project_theseus_scratch_state_diagnostic_v1"},
        )
        observed["diagnostic_state"] = {
            "policy": "project_theseus_scratch_state_diagnostic_v1",
            "production_authority": False,
            "model": str(model_state_path),
            "model_sha256": file_sha256(model_state_path),
            "optimizer": str(optimizer_state_path),
            "optimizer_sha256": file_sha256(optimizer_state_path),
            "rng": str(rng_state_path),
            "rng_sha256": file_sha256(rng_state_path),
        }
    if capture_parameter_snapshot:
        observed["_parameter_snapshot"] = {
            name: np.asarray(value).copy()
            for name, value in mlx_utils.tree_flatten(
                authoritative_model.trainable_parameters()
            )
        }
    if capture_optimizer_snapshot:
        observed["_optimizer_snapshot"] = {
            name: np.asarray(value).copy()
            for name, value in mlx_utils.tree_flatten(optimizer.state)
        }
    if capture_rng_snapshot:
        observed["_rng_snapshot"] = {
            f"state.{index}": np.asarray(value).copy()
            for index, value in enumerate(mx.random.state)
        }
    if capture_starting_snapshot:
        observed["_starting_parameter_snapshot"] = starting_parameter_snapshot
        observed["_starting_optimizer_snapshot"] = starting_optimizer_snapshot
        observed["_starting_rng_snapshot"] = starting_rng_snapshot
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
    tensor_deltas: list[dict[str, Any]] = []
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
        tensor_maximum_absolute_delta = float(
            np.max(np.abs(delta), initial=0.0)
        )
        tensor_squared_delta = float(
            np.sum(np.square(delta), dtype=np.float64)
        )
        tensor_squared_reference = float(
            np.sum(np.square(reference_array), dtype=np.float64)
        )
        tensor_relative_l2_delta = math.sqrt(
            tensor_squared_delta
        ) / max(1e-30, math.sqrt(tensor_squared_reference))
        maximum_absolute_delta = max(
            maximum_absolute_delta, tensor_maximum_absolute_delta
        )
        squared_delta += tensor_squared_delta
        squared_reference += tensor_squared_reference
        element_count += int(reference_array.size)
        tensor_deltas.append(
            {
                "name": name,
                "shape": list(reference_array.shape),
                "dtype": str(reference_array.dtype),
                "element_count": int(reference_array.size),
                "changed_element_count": int(np.count_nonzero(delta)),
                "maximum_absolute_delta": round(
                    tensor_maximum_absolute_delta, 12
                ),
                "relative_l2_delta": round(
                    tensor_relative_l2_delta, 12
                ),
            }
        )
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
        "tensor_count_exceeding_absolute_tolerance": sum(
            row["maximum_absolute_delta"] > MAX_PARAMETER_ABSOLUTE_DELTA
            for row in tensor_deltas
        ),
        "tensor_count_exceeding_relative_l2_tolerance": sum(
            row["relative_l2_delta"] > MAX_PARAMETER_RELATIVE_L2_DELTA
            for row in tensor_deltas
        ),
        "top_tensors_by_maximum_absolute_delta": sorted(
            tensor_deltas,
            key=lambda row: (
                row["maximum_absolute_delta"],
                row["relative_l2_delta"],
                row["name"],
            ),
            reverse=True,
        )[:10],
        "top_tensors_by_relative_l2_delta": sorted(
            tensor_deltas,
            key=lambda row: (
                row["relative_l2_delta"],
                row["maximum_absolute_delta"],
                row["name"],
            ),
            reverse=True,
        )[:10],
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
    token_supervision_active: bool | None = None,
    token_denominator_override: Any | None = None,
    copy_alignment_denominator_override: Any | None = None,
    copy_gate_denominator_override: Any | None = None,
) -> Any:
    """Keep token-loss reduction in FP32 while the model uses a lower precision."""

    if token_supervision_active is not True:
        raise ValueError("mixed-precision token route requires active token supervision")
    if (
        source_conditioning is not True
        and (
            copy_alignment_denominator_override is not None
            or copy_gate_denominator_override is not None
        )
    ):
        raise ValueError(
            "plain-token mixed precision cannot receive copy-loss denominators"
        )
    logits, _cache = model(inputs, source_conditioning=source_conditioning)
    token_loss = nn.losses.cross_entropy(logits.astype(mx.float32), labels)
    denominator = (
        token_denominator_override
        if token_denominator_override is not None
        else mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mx.float32))
    )
    return mx.sum(token_loss * mask) / denominator


def tree_numeric_receipt(tree: Any, *, mx: Any, mlx_utils: Any) -> dict[str, Any]:
    """Report dtypes and finite state without copying full tensors to the host."""

    rows = list(mlx_utils.tree_flatten(tree))
    numeric_dtypes = {mx.float16, mx.bfloat16, mx.float32}
    finite_check_chunk_size = 32
    all_finite = True
    numeric_rows = [
        (name, value) for name, value in rows if value.dtype in numeric_dtypes
    ]
    for start in range(0, len(numeric_rows), finite_check_chunk_size):
        checks = [
            mx.all(mx.isfinite(value))
            for _name, value in numeric_rows[
                start : start + finite_check_chunk_size
            ]
        ]
        mx.eval(*checks)
        all_finite = all_finite and all(bool(value.item()) for value in checks)
        del checks
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    return {
        "tensor_count": len(rows),
        "element_count": sum(int(value.size) for _name, value in rows),
        "dtypes": sorted({str(value.dtype) for _name, value in rows}),
        "all_finite": all_finite,
        "finite_check_chunk_size": finite_check_chunk_size,
    }


def tree_content_receipt(tree: Any, *, mlx_utils: Any) -> dict[str, Any]:
    """Hash one materialized tensor at a time without retaining host copies."""

    digest = hashlib.sha256()
    tensor_count = 0
    element_count = 0
    payload_bytes = 0
    for name, value in sorted(mlx_utils.tree_flatten(tree)):
        array = np.ascontiguousarray(np.asarray(value))
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
        digest.update(memoryview(array).cast("B"))
        tensor_count += 1
        element_count += int(array.size)
        payload_bytes += int(array.nbytes)
        del array
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": tensor_count,
        "element_count": element_count,
        "payload_bytes": payload_bytes,
        "host_copy_policy": "one_tensor_at_a_time_after_timed_training",
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
        "host_batch_preparation_seconds_total": round(
            sum(
                float(row.get("host_batch_preparation_seconds_total") or 0.0)
                for row in rows
            ),
            6,
        ),
        "unit_allocator_pack_seconds_total": round(
            sum(
                float(row.get("unit_allocator_pack_seconds_total") or 0.0)
                for row in rows
            ),
            6,
        ),
        "device_step_seconds_total": round(
            sum(float(row.get("device_step_seconds_total") or 0.0) for row in rows),
            6,
        ),
        "padded_positions_avoided_total": sum(
            int(row.get("padded_positions_avoided") or 0) for row in rows
        ),
        "dynamic_width": {
            "static_sequence_width": max(
                int(row.get("static_sequence_width") or 0) for row in rows
            ),
            "maximum_dynamic_batch_width": max(
                int(row.get("maximum_dynamic_batch_width") or 0) for row in rows
            ),
            "maximum_execution_batch_width": max(
                int(row.get("maximum_execution_batch_width") or 0) for row in rows
            ),
            "mean_dynamic_batch_width": round(
                statistics.mean(
                    float(row.get("mean_dynamic_batch_width") or 0.0) for row in rows
                ),
                3,
            ),
            "compile_width_quanta": sorted(
                {int(row.get("compile_width_quantum") or 0) for row in rows}
            ),
            "compiled_microbatch_sizes": sorted(
                {int(row.get("compiled_microbatch_size") or 0) for row in rows}
            ),
        },
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


def parse_swap_usage(value: str) -> dict[str, int]:
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    observed: dict[str, int] = {}
    for key in ("total", "used", "free"):
        match = re.search(rf"\b{key}\s*=\s*([0-9.]+)([KMGT])", value)
        if match:
            observed[f"swap_{key}_bytes"] = int(
                float(match.group(1)) * units[match.group(2)]
            )
    return observed


def parse_vm_stat(value: str) -> dict[str, int]:
    counters: dict[str, int] = {}
    for label, key in (
        ("Pageouts", "pageouts"),
        ("Swapins", "swapins"),
        ("Swapouts", "swapouts"),
        ("Compressions", "compressions"),
        ("Decompressions", "decompressions"),
    ):
        match = re.search(rf"^{label}:\s+([0-9]+)\.", value, re.MULTILINE)
        if match:
            counters[key] = int(match.group(1))
    return counters


def system_memory_receipt() -> dict[str, int | str]:
    if platform.system() != "Darwin":
        return {"state": "UNAVAILABLE", "scope": "host_cumulative_counters"}
    return {
        "state": "READY",
        **parse_swap_usage(command_output(["sysctl", "vm.swapusage"])),
        **parse_vm_stat(
            subprocess.run(
                ["vm_stat"], capture_output=True, text=True, check=False
            ).stdout
        ),
        "scope": "host_cumulative_counters",
    }


def system_memory_delta(
    before: dict[str, int | str], after: dict[str, int | str]
) -> dict[str, int | str | bool]:
    if before.get("state") != "READY" or after.get("state") != "READY":
        return {"state": "UNAVAILABLE", "scope": "host_during_qualification"}
    result: dict[str, int | str | bool] = {
        "state": "READY",
        "scope": "host_during_qualification_not_process_attributed",
        "swap_used_bytes_start": int(before.get("swap_used_bytes") or 0),
        "swap_used_bytes_end": int(after.get("swap_used_bytes") or 0),
    }
    for key in ("pageouts", "swapins", "swapouts", "compressions", "decompressions"):
        result[f"{key}_delta"] = int(after.get(key) or 0) - int(before.get(key) or 0)
    result["no_host_swap_io_observed"] = (
        int(result["swapins_delta"]) == 0 and int(result["swapouts_delta"]) == 0
    )
    return result


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
    corpus = report.get("corpus_to_tensor") or {}
    corpus_performance = corpus.get("performance") or {}
    training_pair = training.get("paired_canary") or {}
    precision = training.get("precision_autotune") or {}
    bf16_resume = training.get("bf16_checkpoint_resume") or {}
    bf16_adopted = (report.get("adoption") or {}).get("bf16") == (
        "QUALIFIED_BFLOAT16_COMPUTE_FP32_MASTER"
    )
    quality = inference.get("quality_denominator") or generation_quality_receipt(inference)
    assembly = report.get("assembly_line") or build_assembly_line(report)
    memory = report.get("system_memory_pressure") or {}
    return "\n".join(
        [
            "# Resource Acceleration Qualification",
            "",
            f"- State: **{report['trigger_state']}**",
            f"- Exact corpus-to-tensor state: `{corpus.get('state')}`",
            f"- Exact corpus-to-tensor speedup: `{float(corpus_performance.get('corpus_to_tensor_speedup') or 0.0):.3f}x`",
            f"- Exact corpus-to-tensor route adoption ready: `{corpus.get('route_adoption_ready')}`",
            f"- Training throughput: `{training.get('warmup_excluded_positions_per_second', 0):.3f}` positions/s",
            f"- Training execution: `{training.get('training_step_execution')}`",
            f"- Repeated paired training median speedup: `{float(training_pair.get('median_speedup') or 0.0):.3f}x`",
            f"- Repeated paired training pooled speedup: `{float(training_pair.get('pooled_speedup') or 0.0):.3f}x`",
            f"- Repeated paired training state: `{training_pair.get('state')}`",
            f"- Mixed-precision candidate state: `{precision.get('state')}`",
            f"- Mixed-precision median speedup: `{float(precision.get('median_speedup') or 0.0):.3f}x`",
            f"- Mixed-precision pooled speedup: `{float(precision.get('pooled_speedup') or 0.0):.3f}x`",
            f"- Short mixed-precision speed gate passed: `{precision.get('adopt')}`",
            f"- BF16 exact-resume state: `{bf16_resume.get('state')}`",
            f"- Mixed-precision adopted after resume qualification: `{bf16_adopted}`",
            "- Mixed-precision timing scope: `bf16-compute/fp32-master candidate; final adoption also requires reproducible resume`",
            f"- Private-dev loss delta: `{learning.get('absolute_loss_delta')}`",
            f"- Weak-tail regressions: `{', '.join(learning.get('regressed_arms') or []) or 'none'}`",
            f"- Decode cases: `{inference.get('case_count', 0)}`",
            f"- Exact parity cases: `{inference.get('exact_parity_case_count', 0)}`",
            f"- Successful nonempty decode cases: `{quality.get('successful_nonempty_case_count', 0)}/{quality.get('case_count', 0)}`",
            f"- Capability-grade decode speed evidence: `{quality.get('capability_grade_speed_evidence')}`",
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
            f"- Assembly-line measurement complete: `{assembly.get('measurement_complete')}`",
            f"- Assembly-line quality complete: `{assembly.get('quality_complete')}`",
            f"- Host swap used at start/end: `{int(memory.get('swap_used_bytes_start') or 0)}` / `{int(memory.get('swap_used_bytes_end') or 0)}` bytes",
            f"- Host swap I/O observed during qualification: `{not bool(memory.get('no_host_swap_io_observed'))}`",
            "",
            "The decode comparison disables completion and prompt-prefix caches on both routes. Resident cache speedups are reported separately and do not contribute to uncached decode speedup. This qualification does not claim model capability or public transfer.",
            "Faulted or empty generations can establish route parity, but they cannot qualify capability-grade speed.",
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
    bf16_resume = report["training"].get("bf16_checkpoint_resume") or {}
    bf16_adopted = (report.get("adoption") or {}).get("bf16") == (
        "QUALIFIED_BFLOAT16_COMPUTE_FP32_MASTER"
    )
    quality = report["inference"].get("quality_denominator") or generation_quality_receipt(
        report["inference"]
    )
    assembly = report.get("assembly_line") or build_assembly_line(report)
    corpus = report.get("corpus_to_tensor") or {}
    corpus_performance = corpus.get("performance") or {}
    memory = report.get("system_memory_pressure") or {}
    return {
        "policy": report["policy"],
        "created_utc": report["created_utc"],
        "trigger_state": report["trigger_state"],
        "mode": report["mode"],
        "corpus_to_tensor_state": corpus.get("state"),
        "corpus_to_tensor_speedup": corpus_performance.get("corpus_to_tensor_speedup"),
        "corpus_to_tensor_route_adoption_ready": corpus.get("route_adoption_ready"),
        "training_positions_per_second": report["training"].get(
            "warmup_excluded_positions_per_second"
        ),
        "paired_training_median_speedup": training_pair.get("median_speedup"),
        "paired_training_pooled_speedup": training_pair.get("pooled_speedup"),
        "mixed_precision_median_speedup": precision.get("median_speedup"),
        "mixed_precision_pooled_speedup": precision.get("pooled_speedup"),
        "mixed_precision_short_speed_gate_passed": precision.get("adopt"),
        "mixed_precision_resume_state": bf16_resume.get("state"),
        "mixed_precision_adopted": bf16_adopted,
        "private_dev_loss_delta": report["private_dev_learning"].get(
            "absolute_loss_delta"
        ),
        "decode_case_count": report["inference"].get("case_count"),
        "decode_exact_parity_case_count": report["inference"].get(
            "exact_parity_case_count"
        ),
        "decode_successful_nonempty_case_count": quality.get(
            "successful_nonempty_case_count"
        ),
        "decode_capability_grade_speed_evidence": quality.get(
            "capability_grade_speed_evidence"
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
        "assembly_line_measurement_complete": assembly.get("measurement_complete"),
        "assembly_line_quality_complete": assembly.get("quality_complete"),
        "host_swap_used_bytes_start": memory.get("swap_used_bytes_start"),
        "host_swap_used_bytes_end": memory.get("swap_used_bytes_end"),
        "host_swapins_delta": memory.get("swapins_delta"),
        "host_swapouts_delta": memory.get("swapouts_delta"),
        "no_host_swap_io_observed": memory.get("no_host_swap_io_observed"),
        "assembly_line_next_measurement_count": len(
            assembly.get("next_measurements") or []
        ),
        "hard_gaps": report["hard_gaps"],
        "performance_shortfalls": report.get("performance_shortfalls") or [],
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


def directory_artifact(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    total_bytes = 0
    for candidate in files:
        relative_path = candidate.relative_to(path).as_posix()
        digest.update(relative_path.encode())
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                total_bytes += len(chunk)
    return {
        "path": relative(path),
        "sha256": digest.hexdigest(),
        "bytes": total_bytes,
        "file_count": len(files),
        "format": "apple_metal_gputrace_directory",
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
