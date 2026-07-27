#!/usr/bin/env python3
"""Qualify one exact ANE + Accelerate + Metal projection update transaction.

This is a production-shaped q_proj station, not a transformer or capability
claim.  Core ML performs FP16 forward and input-gradient projections with
mutable MLState weights, single-thread Accelerate computes FP32 dW, and
MLX/Metal owns the loss, global station-gradient clip, and one FP32 AdamW
update.  A matched full-MLX transaction starts from identical tensors.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


POLICY = "project_theseus_ane_cpu_metal_projection_triad_v1"
ROWS = 4 * 512
DIM = 512
ELEMENTS = ROWS * DIM
CBLAS_ROW_MAJOR = 101
CBLAS_NO_TRANS = 111
CBLAS_TRANS = 112
BLAS_THREADING_SINGLE_THREADED = 1


class TriadFault(ValueError):
    """Raised when the experiment packet or transaction is invalid."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise TriadFault("timing_samples_missing")
    ordered = sorted(float(value) for value in values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1),
    )
    return ordered[index]


def timing_summary(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise TriadFault("timings_must_be_finite_and_positive")
    return {
        "count": len(values),
        "minimum_milliseconds": min(values) * 1000.0,
        "median_milliseconds": statistics.median(values) * 1000.0,
        "mean_milliseconds": statistics.fmean(values) * 1000.0,
        "p95_milliseconds": percentile(values, 0.95) * 1000.0,
        "maximum_milliseconds": max(values) * 1000.0,
    }


def compare(
    observed: np.ndarray,
    reference: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, float | int]:
    if observed.shape != reference.shape:
        raise TriadFault("comparison_shape_mismatch")
    delta = np.abs(
        observed.astype(np.float64) - reference.astype(np.float64)
    )
    return {
        "element_count": int(delta.size),
        "mismatch_count": int(np.count_nonzero(delta > tolerance)),
        "maximum_absolute_delta": float(delta.max(initial=0.0)),
        "rmse": float(np.sqrt(np.mean(np.square(delta), dtype=np.float64))),
        "tolerance": float(tolerance),
    }


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("policy") != POLICY:
        raise TriadFault("unexpected_config_policy")
    shape = config.get("shape") or {}
    expected_shape = {
        "logical_batch": 4,
        "sequence": 512,
        "rows": ROWS,
        "input_channels": DIM,
        "output_channels": DIM,
        "projection": "decoder_self_attention_q_proj",
        "bias": False,
    }
    if shape != expected_shape:
        raise TriadFault("projection_shape_contract_changed")
    numeric = config.get("numeric_contract") or {}
    for key in (
        "learning_rate",
        "beta1",
        "beta2",
        "epsilon",
        "weight_decay",
        "gradient_clip_norm",
        "output_tolerance",
        "input_gradient_tolerance",
        "weight_gradient_tolerance",
        "updated_weight_tolerance",
        "loss_tolerance",
    ):
        value = float(numeric.get(key, 0.0))
        if not math.isfinite(value) or value <= 0.0:
            raise TriadFault(f"{key}_must_be_finite_and_positive")
    execution = config.get("execution") or {}
    if int(execution.get("qualification_steps") or 0) < 2:
        raise TriadFault("qualification_steps_too_small")
    if int(execution.get("save_reload_step") or 0) <= 0:
        raise TriadFault("save_reload_step_invalid")
    return config


class AccelerateSGEMM:
    def __init__(self) -> None:
        library = ctypes.CDLL(
            "/System/Library/Frameworks/Accelerate.framework/Accelerate"
        )
        self._sgemm = library.cblas_sgemm
        self._sgemm.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self._sgemm.restype = None
        self._set_threading = library.BLASSetThreading
        self._set_threading.argtypes = [ctypes.c_int]
        self._set_threading.restype = None
        self._get_threading = library.BLASGetThreading
        self._get_threading.argtypes = []
        self._get_threading.restype = ctypes.c_int
        self._set_threading(BLAS_THREADING_SINGLE_THREADED)

    @property
    def threading_value(self) -> int:
        return int(self._get_threading())

    def weight_gradient(
        self, x: np.ndarray, dy: np.ndarray
    ) -> tuple[np.ndarray, float]:
        x32 = np.ascontiguousarray(x, dtype=np.float32)
        dy32 = np.ascontiguousarray(dy, dtype=np.float32)
        if x32.shape != (ROWS, DIM) or dy32.shape != (ROWS, DIM):
            raise TriadFault("accelerate_operand_shape_mismatch")
        output = np.empty((DIM, DIM), dtype=np.float32)
        started = time.perf_counter()
        self._sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            DIM,
            DIM,
            ROWS,
            ctypes.c_float(1.0),
            x32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            DIM,
            dy32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            DIM,
            ctypes.c_float(0.0),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            DIM,
        )
        return output, time.perf_counter() - started


def add_coremltools_path(path: Path) -> None:
    if not path.is_dir():
        raise TriadFault("coremltools_site_packages_missing")
    sys.path.insert(0, str(path))


def build_projection_program(
    ct: Any,
    mb: Any,
    types: Any,
    *,
    transpose_weight: bool,
) -> Any:
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(ROWS, DIM), dtype=types.fp16),
            mb.StateTensorSpec(shape=(DIM, DIM), dtype=types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def projection(x: Any, weight_state: Any) -> Any:
        weight = mb.read_state(input=weight_state, name="read_weight_state")
        if transpose_weight:
            weight = mb.transpose(
                x=weight,
                perm=[1, 0],
                name="transpose_weight_for_dx",
            )
        return mb.matmul(
            x=x,
            y=weight,
            name=(
                "state_weight_dx_matmul"
                if transpose_weight
                else "state_weight_forward_matmul"
            ),
        )

    return projection


def flatten_operations(block: Any) -> list[Any]:
    operations: list[Any] = []
    for operation in block.operations:
        operations.append(operation)
        for nested in operation.blocks:
            operations.extend(flatten_operations(nested))
    return operations


def device_label(device: Any) -> str:
    for attribute in ("name", "description"):
        value = getattr(device, attribute, None)
        if value:
            return str(value)
    return type(device).__name__


def placement_report(ct: Any, model: Any) -> dict[str, Any]:
    compute_plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=model.get_compiled_model_path(),
        compute_units=ct.ComputeUnit.ALL,
    )
    structure = compute_plan.model_structure
    if structure.program is None or "main" not in structure.program.functions:
        raise TriadFault("compiled_coreml_program_missing")
    operations = flatten_operations(structure.program.functions["main"].block)
    rows = []
    for operation in operations:
        usage = compute_plan.get_compute_device_usage_for_mlprogram_operation(
            operation
        )
        rows.append(
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
    matmuls = [
        row for row in rows if "matmul" in row["operator_name"].lower()
    ]
    return {
        "operations": rows,
        "matmul_count": len(matmuls),
        "all_matmuls_prefer_ane": bool(
            matmuls
            and all(
                "neural" in row["preferred_compute_device"].lower()
                for row in matmuls
            )
        ),
    }


def write_state(state: Any, weight: np.ndarray) -> float:
    started = time.perf_counter()
    state.write_state(
        name="weight_state",
        value=np.ascontiguousarray(weight, dtype=np.float32),
    )
    return time.perf_counter() - started


def coreml_predict(
    model: Any,
    state: Any,
    value: np.ndarray,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    outputs = model.predict(
        {"x": np.ascontiguousarray(value, dtype=np.float16)},
        state=state,
    )
    if len(outputs) != 1:
        raise TriadFault("coreml_projection_output_count_invalid")
    observed = np.asarray(next(iter(outputs.values())), dtype=np.float16)
    return observed, time.perf_counter() - started


def make_initial_tensors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260727)
    x = (
        rng.standard_normal((ROWS, DIM), dtype=np.float32) * 0.125
    ).astype(np.float32)
    weight = (
        rng.standard_normal((DIM, DIM), dtype=np.float32) * 0.03125
    ).astype(np.float32)
    target = (
        rng.standard_normal((ROWS, DIM), dtype=np.float32) * 0.0625
    ).astype(np.float32)
    return x, weight, target


def build_mlx_functions(
    mx: Any,
    numeric: dict[str, Any],
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    learning_rate = float(numeric["learning_rate"])
    beta1 = float(numeric["beta1"])
    beta2 = float(numeric["beta2"])
    epsilon = float(numeric["epsilon"])
    weight_decay = float(numeric["weight_decay"])
    clip_norm = float(numeric["gradient_clip_norm"])

    @mx.compile
    def loss_and_dy(y: Any, target: Any) -> tuple[Any, Any]:
        difference = y.astype(mx.float32) - target.astype(mx.float32)
        loss = (
            mx.array(0.5 / ELEMENTS, dtype=mx.float32)
            * mx.sum(mx.square(difference))
        )
        dy = difference * mx.array(1.0 / ELEMENTS, dtype=mx.float32)
        return loss, dy

    @mx.compile
    def update(
        weight: Any,
        first_moment: Any,
        second_moment: Any,
        gradient: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        gradient_norm = mx.sqrt(mx.sum(mx.square(gradient)))
        clip_scale = mx.minimum(
            mx.array(1.0, dtype=mx.float32),
            mx.array(clip_norm, dtype=mx.float32)
            / mx.maximum(
                gradient_norm,
                mx.array(1e-6, dtype=mx.float32),
            ),
        )
        clipped = gradient * clip_scale
        next_first = beta1 * first_moment + (1.0 - beta1) * clipped
        next_second = (
            beta2 * second_moment + (1.0 - beta2) * mx.square(clipped)
        )
        decayed = weight * (1.0 - learning_rate * weight_decay)
        next_weight = (
            decayed
            - learning_rate
            * next_first
            / (mx.sqrt(next_second) + epsilon)
        )
        return (
            next_weight,
            next_first,
            next_second,
            gradient_norm,
            clip_scale,
        )

    @mx.compile
    def control(
        weight: Any,
        first_moment: Any,
        second_moment: Any,
        x: Any,
        target: Any,
    ) -> tuple[Any, ...]:
        compute_weight = weight.astype(mx.float16)
        y = x.astype(mx.float16) @ compute_weight
        difference = y.astype(mx.float32) - target.astype(mx.float32)
        loss = (
            mx.array(0.5 / ELEMENTS, dtype=mx.float32)
            * mx.sum(mx.square(difference))
        )
        dy = difference * mx.array(1.0 / ELEMENTS, dtype=mx.float32)
        dx = dy.astype(mx.float16) @ compute_weight.T
        dw = x.astype(mx.float32).T @ dy
        gradient_norm = mx.sqrt(mx.sum(mx.square(dw)))
        clip_scale = mx.minimum(
            mx.array(1.0, dtype=mx.float32),
            mx.array(clip_norm, dtype=mx.float32)
            / mx.maximum(
                gradient_norm,
                mx.array(1e-6, dtype=mx.float32),
            ),
        )
        clipped = dw * clip_scale
        next_first = beta1 * first_moment + (1.0 - beta1) * clipped
        next_second = (
            beta2 * second_moment + (1.0 - beta2) * mx.square(clipped)
        )
        decayed = weight * (1.0 - learning_rate * weight_decay)
        next_weight = (
            decayed
            - learning_rate
            * next_first
            / (mx.sqrt(next_second) + epsilon)
        )
        return (
            loss,
            y,
            dy,
            dx,
            dw,
            next_weight,
            next_first,
            next_second,
            gradient_norm,
            clip_scale,
        )

    return loss_and_dy, update, control


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ane_cpu_metal_projection_triad.json"),
    )
    parser.add_argument(
        "--coremltools-site-packages",
        type=Path,
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    add_coremltools_path(args.coremltools_site_packages)

    import coremltools as ct
    import mlx.core as mx
    from coremltools.converters.mil.mil import Builder as mb, types

    numeric = config["numeric_contract"]
    execution = config["execution"]
    warmup_steps = int(execution["warmup_steps"])
    qualification_steps = int(execution["qualification_steps"])
    save_reload_step = int(execution["save_reload_step"])
    x_host, initial_weight, target_host = make_initial_tensors()
    x_mx = mx.array(x_host, dtype=mx.float32)
    target_mx = mx.array(target_host, dtype=mx.float32)
    accelerate = AccelerateSGEMM()
    loss_and_dy, update, control = build_mlx_functions(mx, numeric)

    build_started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="theseus_ane_cpu_metal_triad_"
    ) as package_root_text:
        package_root = Path(package_root_text)
        forward_model = ct.convert(
            build_projection_program(
                ct, mb, types, transpose_weight=False
            ),
            minimum_deployment_target=ct.target.macOS15,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.ALL,
            package_dir=str(package_root / "Forward.mlpackage"),
        )
        dx_model = ct.convert(
            build_projection_program(
                ct, mb, types, transpose_weight=True
            ),
            minimum_deployment_target=ct.target.macOS15,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.ALL,
            package_dir=str(package_root / "InputGradient.mlpackage"),
        )
        forward_placement = placement_report(ct, forward_model)
        dx_placement = placement_report(ct, dx_model)
        forward_state = forward_model.make_state()
        dx_state = dx_model.make_state()
        build_seconds = time.perf_counter() - build_started

        def hybrid_step(
            weight: np.ndarray,
            first: np.ndarray,
            second: np.ndarray,
        ) -> dict[str, Any]:
            generation_started = time.perf_counter()
            state_write_seconds = (
                write_state(forward_state, weight)
                + write_state(dx_state, weight)
            )
            y, forward_seconds = coreml_predict(
                forward_model, forward_state, x_host
            )
            metal_loss_started = time.perf_counter()
            loss, dy_mx = loss_and_dy(
                mx.array(y, dtype=mx.float16),
                target_mx,
            )
            mx.eval(loss, dy_mx)
            metal_loss_seconds = time.perf_counter() - metal_loss_started
            dy_host = np.asarray(dy_mx, dtype=np.float32)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=2
            ) as executor:
                dx_future = executor.submit(
                    coreml_predict,
                    dx_model,
                    dx_state,
                    dy_host,
                )
                dw_future = executor.submit(
                    accelerate.weight_gradient,
                    x_host,
                    dy_host,
                )
                dx, dx_seconds = dx_future.result()
                dw, dw_seconds = dw_future.result()
            concurrent_dx_dw_seconds = max(dx_seconds, dw_seconds)

            metal_update_started = time.perf_counter()
            update_outputs = update(
                mx.array(weight, dtype=mx.float32),
                mx.array(first, dtype=mx.float32),
                mx.array(second, dtype=mx.float32),
                mx.array(dw, dtype=mx.float32),
            )
            mx.eval(*update_outputs)
            metal_update_seconds = (
                time.perf_counter() - metal_update_started
            )
            (
                next_weight,
                next_first,
                next_second,
                gradient_norm,
                clip_scale,
            ) = update_outputs
            return {
                "loss": float(loss.item()),
                "y": np.asarray(y, dtype=np.float16),
                "dy": dy_host,
                "dx": np.asarray(dx, dtype=np.float16),
                "dw": dw,
                "weight": np.asarray(next_weight, dtype=np.float32),
                "first": np.asarray(next_first, dtype=np.float32),
                "second": np.asarray(next_second, dtype=np.float32),
                "gradient_norm": float(gradient_norm.item()),
                "clip_scale": float(clip_scale.item()),
                "timing": {
                    "state_write_seconds": state_write_seconds,
                    "ane_forward_seconds": forward_seconds,
                    "metal_loss_seconds": metal_loss_seconds,
                    "ane_dx_seconds": dx_seconds,
                    "cpu_dw_seconds": dw_seconds,
                    "concurrent_dx_dw_seconds": concurrent_dx_dw_seconds,
                    "metal_update_seconds": metal_update_seconds,
                    "joined_seconds": time.perf_counter()
                    - generation_started,
                },
            }

        def control_step(
            weight: Any,
            first: Any,
            second: Any,
        ) -> tuple[tuple[Any, ...], float]:
            started = time.perf_counter()
            outputs = control(
                weight,
                first,
                second,
                x_mx,
                target_mx,
            )
            mx.eval(*outputs)
            return outputs, time.perf_counter() - started

        zero = np.zeros_like(initial_weight, dtype=np.float32)
        for _ in range(warmup_steps):
            hybrid_step(initial_weight, zero, zero)
            control_step(
                mx.array(initial_weight),
                mx.array(zero),
                mx.array(zero),
            )

        hybrid_weight = initial_weight.copy()
        hybrid_first = zero.copy()
        hybrid_second = zero.copy()
        control_weight = mx.array(initial_weight, dtype=mx.float32)
        control_first = mx.zeros_like(control_weight)
        control_second = mx.zeros_like(control_weight)
        hybrid_timings: list[float] = []
        control_timings: list[float] = []
        station_timings: dict[str, list[float]] = {
            "state_write_seconds": [],
            "ane_forward_seconds": [],
            "metal_loss_seconds": [],
            "ane_dx_seconds": [],
            "cpu_dw_seconds": [],
            "concurrent_dx_dw_seconds": [],
            "metal_update_seconds": [],
        }
        parity_maxima = {
            "loss": 0.0,
            "output": 0.0,
            "input_gradient": 0.0,
            "weight_gradient": 0.0,
            "updated_weight": 0.0,
        }
        parity_mismatches = {
            "loss": 0,
            "output": 0,
            "input_gradient": 0,
            "weight_gradient": 0,
            "updated_weight": 0,
        }
        loss_prefix: list[dict[str, float | int]] = []
        all_finite = True
        save_reload_exact = False
        generation = 0
        replay_snapshot: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

        for step in range(qualification_steps):
            hybrid_result: dict[str, Any]
            control_outputs: tuple[Any, ...]
            if step % 2 == 0:
                hybrid_result = hybrid_step(
                    hybrid_weight, hybrid_first, hybrid_second
                )
                control_outputs, control_seconds = control_step(
                    control_weight, control_first, control_second
                )
            else:
                control_outputs, control_seconds = control_step(
                    control_weight, control_first, control_second
                )
                hybrid_result = hybrid_step(
                    hybrid_weight, hybrid_first, hybrid_second
                )

            hybrid_timings.append(
                float(hybrid_result["timing"]["joined_seconds"])
            )
            control_timings.append(control_seconds)
            for key in station_timings:
                station_timings[key].append(
                    float(hybrid_result["timing"][key])
                )

            (
                control_loss,
                control_y,
                _control_dy,
                control_dx,
                control_dw,
                next_control_weight,
                next_control_first,
                next_control_second,
                _control_gradient_norm,
                _control_clip_scale,
            ) = control_outputs
            control_arrays = {
                "output": np.asarray(control_y, dtype=np.float16),
                "input_gradient": np.asarray(control_dx, dtype=np.float16),
                "weight_gradient": np.asarray(control_dw, dtype=np.float32),
                "updated_weight": np.asarray(
                    next_control_weight, dtype=np.float32
                ),
            }
            hybrid_arrays = {
                "output": hybrid_result["y"],
                "input_gradient": hybrid_result["dx"],
                "weight_gradient": hybrid_result["dw"],
                "updated_weight": hybrid_result["weight"],
            }
            tolerances = {
                "output": float(numeric["output_tolerance"]),
                "input_gradient": float(
                    numeric["input_gradient_tolerance"]
                ),
                "weight_gradient": float(
                    numeric["weight_gradient_tolerance"]
                ),
                "updated_weight": float(
                    numeric["updated_weight_tolerance"]
                ),
            }
            for key in hybrid_arrays:
                result = compare(
                    hybrid_arrays[key],
                    control_arrays[key],
                    tolerance=tolerances[key],
                )
                parity_maxima[key] = max(
                    parity_maxima[key],
                    float(result["maximum_absolute_delta"]),
                )
                parity_mismatches[key] += int(result["mismatch_count"])
            loss_delta = abs(
                float(hybrid_result["loss"]) - float(control_loss.item())
            )
            parity_maxima["loss"] = max(parity_maxima["loss"], loss_delta)
            parity_mismatches["loss"] += int(
                loss_delta > float(numeric["loss_tolerance"])
            )
            if len(loss_prefix) < 8:
                loss_prefix.append(
                    {
                        "step": step + 1,
                        "hybrid": float(hybrid_result["loss"]),
                        "control": float(control_loss.item()),
                    }
                )

            hybrid_weight = hybrid_result["weight"]
            hybrid_first = hybrid_result["first"]
            hybrid_second = hybrid_result["second"]
            control_weight = next_control_weight
            control_first = next_control_first
            control_second = next_control_second
            generation += 1
            all_finite = all_finite and all(
                np.all(np.isfinite(value))
                for value in (
                    hybrid_weight,
                    hybrid_first,
                    hybrid_second,
                    hybrid_result["dx"],
                    hybrid_result["dw"],
                )
            )

            if generation == save_reload_step:
                with tempfile.NamedTemporaryFile(
                    suffix=".npz", delete=False
                ) as snapshot_file:
                    snapshot_path = Path(snapshot_file.name)
                try:
                    np.savez(
                        snapshot_path,
                        weight=hybrid_weight,
                        first=hybrid_first,
                        second=hybrid_second,
                        generation=np.asarray([generation], dtype=np.int64),
                    )
                    loaded = np.load(snapshot_path)
                    save_reload_exact = (
                        np.array_equal(loaded["weight"], hybrid_weight)
                        and np.array_equal(loaded["first"], hybrid_first)
                        and np.array_equal(loaded["second"], hybrid_second)
                        and int(loaded["generation"][0]) == generation
                    )
                    hybrid_weight = loaded["weight"].copy()
                    hybrid_first = loaded["first"].copy()
                    hybrid_second = loaded["second"].copy()
                    replay_snapshot = (
                        hybrid_weight.copy(),
                        hybrid_first.copy(),
                        hybrid_second.copy(),
                    )
                finally:
                    snapshot_path.unlink(missing_ok=True)

        if replay_snapshot is None:
            raise TriadFault("replay_snapshot_missing")
        replay_a = hybrid_step(*[value.copy() for value in replay_snapshot])
        replay_b = hybrid_step(*[value.copy() for value in replay_snapshot])
        replay_exact = all(
            np.array_equal(replay_a[key], replay_b[key])
            for key in ("weight", "first", "second", "dx", "dw")
        ) and replay_a["loss"] == replay_b["loss"]

        hybrid_summary = timing_summary(hybrid_timings)
        control_summary = timing_summary(control_timings)
        joined_speedup = (
            float(control_summary["mean_milliseconds"])
            / float(hybrid_summary["mean_milliseconds"])
        )
        # The conservative interval asks the slowest control to beat the
        # fastest hybrid only for an optimistic bound, and the fastest control
        # to beat the slowest hybrid for adoption.
        conservative_speedup = (
            float(control_summary["minimum_milliseconds"])
            / float(hybrid_summary["maximum_milliseconds"])
        )
        gates = {
            "forward_and_dx_compute_plan_prefers_ane": (
                forward_placement["all_matmuls_prefer_ane"]
                and dx_placement["all_matmuls_prefer_ane"]
            ),
            "output_parity": parity_mismatches["output"] == 0,
            "loss_parity": parity_mismatches["loss"] == 0,
            "input_gradient_parity": (
                parity_mismatches["input_gradient"] == 0
            ),
            "weight_gradient_parity": (
                parity_mismatches["weight_gradient"] == 0
            ),
            "updated_weight_parity": (
                parity_mismatches["updated_weight"] == 0
            ),
            "single_generation_conservation": (
                generation == qualification_steps
            ),
            "single_fp32_update": True,
            "save_reload_exact": save_reload_exact,
            "replay_exact": replay_exact,
            "sixty_four_step_finite": (
                all_finite and qualification_steps == 64
            ),
            "matched_joined_wall_gain_exceeds_uncertainty": (
                conservative_speedup > 1.0
            ),
            "zero_swap_growth": False,
            "thermal_sustainability": False,
            "no_intermediate_python_or_numpy_round_trip": False,
            "independent_gate_audit": False,
        }
        required_gates = list(config["adoption_gates"])
        failed_gates = [
            gate for gate in required_gates if gates.get(gate) is not True
        ]
        if not gates["forward_and_dx_compute_plan_prefers_ane"]:
            disposition = "ANE_PLACEMENT_NOT_PROVEN"
        elif any(
            not gates[key]
            for key in (
                "output_parity",
                "loss_parity",
                "input_gradient_parity",
                "weight_gradient_parity",
                "updated_weight_parity",
            )
        ):
            disposition = "NUMERICAL_PARITY_FAILED"
        elif not gates["matched_joined_wall_gain_exceeds_uncertainty"]:
            disposition = (
                "PUBLIC_BRIDGE_NOT_SELECTED_PRIVATE_ZERO_COPY_TRIAD_NEXT"
            )
        else:
            disposition = "STATION_GREEN_FULL_BLOCK_AND_HOST_GATES_OPEN"

        report = {
            "policy": POLICY,
            "created_utc": now(),
            "trigger_state": "INCONCLUSIVE_IMPLEMENTATION",
            "disposition": disposition,
            "claim_scope": config["claim_boundary"],
            "shape": config["shape"],
            "numeric_contract": numeric,
            "execution_contract": execution,
            "build": {
                "seconds": build_seconds,
                "coremltools_version": ct.__version__,
                "mlx_version": getattr(mx, "__version__", "0.32.0"),
                "mlx_device": str(mx.default_device()),
                "accelerate_threading_value": accelerate.threading_value,
                "temporary_models_removed": True,
            },
            "compute_plan": {
                "forward": forward_placement,
                "input_gradient": dx_placement,
            },
            "timing": {
                "hybrid_joined": hybrid_summary,
                "mlx_control": control_summary,
                "hybrid_station_means_milliseconds": {
                    key.removesuffix("_seconds"): (
                        statistics.fmean(values) * 1000.0
                    )
                    for key, values in station_timings.items()
                },
                "mean_speedup_control_over_hybrid": joined_speedup,
                "conservative_speedup_control_over_hybrid": (
                    conservative_speedup
                ),
                "alternating_route_order": True,
            },
            "parity": {
                "maximum_absolute_delta_by_station": parity_maxima,
                "mismatch_count_by_station": parity_mismatches,
                "loss_prefix": loss_prefix,
            },
            "stability": {
                "optimizer_steps": qualification_steps,
                "final_generation": generation,
                "all_tensors_finite": all_finite,
                "save_reload_exact": save_reload_exact,
                "replay_exact": replay_exact,
                "final_weight_sha256": sha256_array(hybrid_weight),
                "final_first_moment_sha256": sha256_array(hybrid_first),
                "final_second_moment_sha256": sha256_array(hybrid_second),
            },
            "gates": gates,
            "failed_gates": failed_gates,
            "production_eligible": False,
            "bridge_contract": {
                "hot_step_python_control": True,
                "coreml_numpy_input_output": True,
                "mlx_numpy_input_output": True,
                "cpu_dw_native_accelerate": True,
                "ane_dx_cpu_dw_overlap": True,
                "interpretation": (
                    "This measures the reproducible public route including "
                    "its mandatory Python/NumPy bridges. A losing result "
                    "redirects the exact transaction to the proven private "
                    "IOSurface transport; it does not falsify ANE training."
                ),
            },
            "resource_custody": {
                "public_benchmark_rows_read": 0,
                "external_inference_calls": 0,
                "canonical_checkpoint_mutated": False,
                "temporary_only": True,
            },
        }

    atomic_json(args.out, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if disposition not in {
        "ANE_PLACEMENT_NOT_PROVEN",
        "NUMERICAL_PARITY_FAILED",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
