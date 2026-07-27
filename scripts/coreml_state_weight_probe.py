#!/usr/bin/env python3
"""Probe public Core ML state as mutable FP16 training-weight transport.

This is an operator/placement experiment only.  It does not read a corpus,
mutate a checkpoint, claim transformer parity, or authorize a training route.
The graph intentionally performs a production-shape state-backed projection
and an in-graph multiplicative state update so the compute plan and runtime can
answer two bounded questions:

1. Can an FP16 weight matrix persist as MLState without recompilation?
2. Does Core ML prefer the Apple Neural Engine for the state-backed matmul?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


POLICY = "project_theseus_coreml_state_weight_probe_v1"
ROWS = 4 * 512
INPUT_CHANNELS = 512
OUTPUT_CHANNELS = 768


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def device_label(device: Any) -> str:
    for attribute in ("name", "description"):
        value = getattr(device, attribute, None)
        if value:
            return str(value)
    return type(device).__name__


def flatten_operations(block: Any) -> list[Any]:
    operations: list[Any] = []
    for operation in block.operations:
        operations.append(operation)
        for nested in operation.blocks:
            operations.extend(flatten_operations(nested))
    return operations


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_update_program(ct: Any, mb: Any, types: Any) -> Any:
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(ROWS, INPUT_CHANNELS), dtype=types.fp16),
            mb.TensorSpec(shape=(1,), dtype=types.fp16),
            mb.StateTensorSpec(
                shape=(INPUT_CHANNELS, OUTPUT_CHANNELS),
                dtype=types.fp16,
            ),
        ],
        opset_version=ct.target.macOS15,
    )
    def state_projection(x: Any, update_scale: Any, weight_state: Any) -> Any:
        weight = mb.read_state(input=weight_state, name="read_weight_state")
        output = mb.matmul(x=x, y=weight, name="state_weight_matmul")
        updated_weight = mb.mul(
            x=weight,
            y=update_scale,
            name="state_weight_scale_update",
        )
        mb.coreml_update_state(
            state=weight_state,
            value=updated_weight,
            name="publish_weight_state",
        )
        return output

    return state_projection


def build_read_only_program(ct: Any, mb: Any, types: Any) -> Any:
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(ROWS, INPUT_CHANNELS), dtype=types.fp16),
            mb.StateTensorSpec(
                shape=(INPUT_CHANNELS, OUTPUT_CHANNELS),
                dtype=types.fp16,
            ),
        ],
        opset_version=ct.target.macOS15,
    )
    def state_projection_read_only(x: Any, weight_state: Any) -> Any:
        weight = mb.read_state(
            input=weight_state,
            name="read_only_weight_state",
        )
        return mb.matmul(
            x=x,
            y=weight,
            name="read_only_state_weight_matmul",
        )

    return state_projection_read_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 1 or args.repetitions < 2:
        raise SystemExit("warmup must be positive and repetitions must be at least two")

    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil.mil import Builder as mb, types

    started = time.perf_counter()
    program = build_update_program(ct, mb, types)
    package_root = Path(tempfile.mkdtemp(prefix="theseus_coreml_state_"))
    package_path = package_root / "StateWeightProjection.mlpackage"
    conversion_started = time.perf_counter()
    model = ct.convert(
        program,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        package_dir=str(package_path),
    )
    conversion_seconds = time.perf_counter() - conversion_started
    compiled_path = model.get_compiled_model_path()
    read_only_package_path = (
        package_root / "ReadOnlyStateWeightProjection.mlpackage"
    )
    read_only_conversion_started = time.perf_counter()
    read_only_model = ct.convert(
        build_read_only_program(ct, mb, types),
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        package_dir=str(read_only_package_path),
    )
    read_only_conversion_seconds = (
        time.perf_counter() - read_only_conversion_started
    )

    compute_plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=compiled_path,
        compute_units=ct.ComputeUnit.ALL,
    )
    structure = compute_plan.model_structure
    if structure.program is None or "main" not in structure.program.functions:
        raise RuntimeError("compiled Core ML artifact has no main ML Program")
    operations = flatten_operations(structure.program.functions["main"].block)
    placement = []
    for operation in operations:
        usage = compute_plan.get_compute_device_usage_for_mlprogram_operation(
            operation
        )
        placement.append(
            {
                "operator_name": operation.operator_name,
                "preferred_compute_device": (
                    device_label(usage.preferred_compute_device)
                    if usage is not None
                    else ""
                ),
                "supported_compute_devices": (
                    [
                        device_label(device)
                        for device in usage.supported_compute_devices
                    ]
                    if usage is not None
                    else []
                ),
            }
        )

    rng = np.random.default_rng(20260727)
    x = (
        rng.standard_normal((ROWS, INPUT_CHANNELS), dtype=np.float32) * 0.125
    ).astype(np.float16)
    weight = (
        rng.standard_normal(
            (INPUT_CHANNELS, OUTPUT_CHANNELS), dtype=np.float32
        )
        * 0.03125
    ).astype(np.float16)
    identity_scale = np.asarray([1.0], dtype=np.float16)
    half_scale = np.asarray([0.5], dtype=np.float16)
    state = model.make_state()
    # Core ML's Python state bridge accepts ordinary NumPy float arrays and
    # converts them to the declared FP16 state.  Direct np.float16 state writes
    # are rejected by the macOS 15 bridge even though FP16 prediction inputs
    # are valid.
    state.write_state(name="weight_state", value=weight.astype(np.float32))
    first_update_output = model.predict(
        {"x": x, "update_scale": half_scale},
        state=state,
    )
    scaled_weight = (weight.astype(np.float32) * 0.5).astype(np.float16)
    observed_scaled_state = np.asarray(
        state.read_state(name="weight_state"),
        dtype=np.float16,
    )
    state_transition_tolerance = 0.001
    state_transition_delta = np.abs(
        observed_scaled_state.astype(np.float32)
        - scaled_weight.astype(np.float32)
    )
    nonidentity_state_bit_mismatch_count = int(
        np.count_nonzero(observed_scaled_state != scaled_weight)
    )
    nonidentity_state_mismatch_count = int(
        np.count_nonzero(state_transition_delta > state_transition_tolerance)
    )
    state_transition_maximum_absolute_delta = float(
        state_transition_delta.max(initial=0.0)
    )
    second_update_output = model.predict(
        {"x": x, "update_scale": identity_scale},
        state=state,
    )
    first_update_observed = np.asarray(
        next(iter(first_update_output.values())),
        dtype=np.float16,
    )
    second_update_observed = np.asarray(
        next(iter(second_update_output.values())),
        dtype=np.float16,
    )
    first_update_reference = (
        x.astype(np.float32) @ weight.astype(np.float32)
    ).astype(np.float16)
    second_update_reference = (
        x.astype(np.float32) @ scaled_weight.astype(np.float32)
    ).astype(np.float16)
    first_update_mismatch_count = int(
        np.count_nonzero(
            np.abs(
                first_update_observed.astype(np.float32)
                - first_update_reference.astype(np.float32)
            )
            > state_transition_tolerance
        )
    )
    second_update_mismatch_count = int(
        np.count_nonzero(
            np.abs(
                second_update_observed.astype(np.float32)
                - second_update_reference.astype(np.float32)
            )
            > state_transition_tolerance
        )
    )
    state.write_state(name="weight_state", value=weight.astype(np.float32))

    for _ in range(args.warmup):
        model.predict(
            {"x": x, "update_scale": identity_scale},
            state=state,
        )

    durations: list[float] = []
    output = None
    for _ in range(args.repetitions):
        iteration_started = time.perf_counter()
        output = model.predict(
            {"x": x, "update_scale": identity_scale},
            state=state,
        )
        durations.append(time.perf_counter() - iteration_started)
    if output is None or len(output) != 1:
        raise RuntimeError("Core ML state projection returned an invalid output")
    output_name, observed = next(iter(output.items()))
    observed = np.asarray(observed, dtype=np.float16)
    reference = (x.astype(np.float32) @ weight.astype(np.float32)).astype(
        np.float16
    )
    delta = np.abs(
        observed.astype(np.float32) - reference.astype(np.float32)
    )
    tolerance = 0.001
    mismatch_count = int(np.count_nonzero(delta > tolerance))
    maximum_absolute_delta = float(delta.max(initial=0.0))
    rmse = float(np.sqrt(np.mean(np.square(delta), dtype=np.float64)))

    read_only_state = read_only_model.make_state()
    read_only_state.write_state(
        name="weight_state",
        value=weight.astype(np.float32),
    )
    for _ in range(args.warmup):
        read_only_model.predict({"x": x}, state=read_only_state)
    read_only_durations: list[float] = []
    read_only_output = None
    for _ in range(args.repetitions):
        iteration_started = time.perf_counter()
        read_only_output = read_only_model.predict(
            {"x": x},
            state=read_only_state,
        )
        read_only_durations.append(
            time.perf_counter() - iteration_started
        )
    if read_only_output is None or len(read_only_output) != 1:
        raise RuntimeError("read-only Core ML state projection returned invalid output")
    read_only_observed = np.asarray(
        next(iter(read_only_output.values())),
        dtype=np.float16,
    )
    read_only_delta = np.abs(
        read_only_observed.astype(np.float32)
        - reference.astype(np.float32)
    )
    read_only_mismatch_count = int(
        np.count_nonzero(read_only_delta > tolerance)
    )
    read_only_mean_seconds = statistics.fmean(read_only_durations)

    state_after = np.asarray(
        state.read_state(name="weight_state"),
        dtype=np.float16,
    )
    state_mismatch_count = int(np.count_nonzero(state_after != weight))
    matmul_rows = [
        row
        for row in placement
        if row["operator_name"] == "matmul"
        or "matmul" in row["operator_name"].lower()
    ]
    matmul_prefers_ane = bool(
        matmul_rows
        and all(
            "neural" in row["preferred_compute_device"].lower()
            for row in matmul_rows
        )
    )
    matmul_supports_ane = bool(
        matmul_rows
        and all(
            any("neural" in value.lower() for value in row["supported_compute_devices"])
            for row in matmul_rows
        )
    )
    state_update_visible = (
        nonidentity_state_mismatch_count == 0
        and first_update_mismatch_count == 0
        and second_update_mismatch_count == 0
        and state_mismatch_count == 0
    )
    runtime_parity = (
        mismatch_count == 0 and read_only_mismatch_count == 0
    )
    trigger_state = (
        "GREEN_PUBLIC_ANE_STATE_WEIGHT_TRANSPORT"
        if runtime_parity
        and state_update_visible
        and matmul_prefers_ane
        else "YELLOW_PUBLIC_STATE_WITHOUT_ANE_PREFERENCE"
        if runtime_parity and state_update_visible
        else "RED"
    )
    mean_seconds = statistics.fmean(durations)
    report = {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": trigger_state,
        "claim_scope": (
            "Public Core ML production-shape state-backed FP16 projection and "
            "non-identity state-update mechanics only; no backward, optimizer, "
            "transformer, checkpoint, convergence, utility, or capability claim."
        ),
        "hardware": {
            "model": platform.machine(),
            "platform": platform.platform(),
            "coremltools_version": ct.__version__,
        },
        "shape": {
            "logical_batch": 4,
            "sequence": 512,
            "rows": ROWS,
            "input_channels": INPUT_CHANNELS,
            "output_channels": OUTPUT_CHANNELS,
            "state_elements": INPUT_CHANNELS * OUTPUT_CHANNELS,
        },
        "build": {
            "minimum_deployment_target": "macOS15",
            "compute_units": "ALL",
            "conversion_seconds": round(conversion_seconds, 6),
            "read_only_conversion_seconds": round(
                read_only_conversion_seconds, 6
            ),
            "compiled_path_identity_sha256": hashlib.sha256(
                str(compiled_path).encode()
            ).hexdigest(),
        },
        "compute_plan": {
            "operations": placement,
            "state_matmul_operation_count": len(matmul_rows),
            "state_matmul_supports_ane": matmul_supports_ane,
            "state_matmul_prefers_ane": matmul_prefers_ane,
        },
        "runtime": {
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "output_name": output_name,
            "mean_milliseconds": round(mean_seconds * 1000.0, 6),
            "median_milliseconds": round(
                statistics.median(durations) * 1000.0, 6
            ),
            "p95_milliseconds": round(
                percentile(durations, 0.95) * 1000.0, 6
            ),
            "rows_per_second": round(ROWS / mean_seconds, 6),
            "read_only_control": {
                "mean_milliseconds": round(
                    read_only_mean_seconds * 1000.0, 6
                ),
                "median_milliseconds": round(
                    statistics.median(read_only_durations) * 1000.0,
                    6,
                ),
                "p95_milliseconds": round(
                    percentile(read_only_durations, 0.95) * 1000.0,
                    6,
                ),
                "rows_per_second": round(
                    ROWS / read_only_mean_seconds, 6
                ),
                "mismatch_count": read_only_mismatch_count,
                "maximum_absolute_delta": float(
                    read_only_delta.max(initial=0.0)
                ),
            },
            "state_update_wall_ratio_over_read_only": round(
                mean_seconds / read_only_mean_seconds,
                6,
            ),
            "maximum_absolute_delta": maximum_absolute_delta,
            "rmse": rmse,
            "tolerance": tolerance,
            "mismatch_count": mismatch_count,
            "state_identity_update_mismatch_count": state_mismatch_count,
            "state_nonidentity_update_mismatch_count": (
                nonidentity_state_mismatch_count
            ),
            "state_nonidentity_update_bit_mismatch_count": (
                nonidentity_state_bit_mismatch_count
            ),
            "state_transition_maximum_absolute_delta": (
                state_transition_maximum_absolute_delta
            ),
            "state_transition_tolerance": state_transition_tolerance,
            "state_preupdate_output_mismatch_count": (
                first_update_mismatch_count
            ),
            "state_postupdate_output_mismatch_count": (
                second_update_mismatch_count
            ),
            "state_transition_scale": 0.5,
            "state_update_visible_without_recompile": state_update_visible,
        },
        "resource_custody": {
            "public_benchmark_rows_read": 0,
            "external_inference_calls": 0,
            "canonical_checkpoint_mutated": False,
            "temporary_model_only": True,
        },
        "nonclaims": [
            "No proof that a full Theseus backward or AdamW update can run on ANE.",
            "No proof that state read/write is zero-copy across Core ML and Metal.",
            "No end-to-end training or inference speedup claim.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    temporary_model_removed = True
    try:
        shutil.rmtree(package_root)
    except OSError:
        temporary_model_removed = False
    report["resource_custody"]["temporary_model_removed"] = (
        temporary_model_removed
    )
    if args.out is not None:
        atomic_json(args.out, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if trigger_state != "RED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
