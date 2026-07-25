#!/usr/bin/env python3
"""Shape-exact SOAP resource and update-timing disposition for the M1 host."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import host_resource_safety


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "soap_full_shape_resource_preflight.json"
POLICY = "project_theseus_soap_full_shape_resource_preflight_v1"


class SoapPreflightFault(ValueError):
    pass


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("policy") != POLICY:
        raise SoapPreflightFault("config_policy_invalid")
    if int(config.get("dtype_bytes") or 0) not in {2, 4, 8}:
        raise SoapPreflightFault("dtype_bytes_invalid")
    if not 0 < float(config["maximum_optimizer_memory_fraction"]) <= 1:
        raise SoapPreflightFault("memory_fraction_invalid")
    if int(config["precondition_frequency_steps"]) <= 0:
        raise SoapPreflightFault("precondition_frequency_invalid")
    scope = config.get("campaign_scope_decision") or {}
    if (
        scope.get("ineligible_disposition")
        != "FORMALLY_SCOPE_REMOVED_FULL_SHAPE_SOAP_UNECONOMIC_M1_MLX"
        or scope.get("ineligible_membership")
        != "REMOVED_FROM_FIRST_CAMPAIGN_FINITE_DOCKET"
        or not scope.get("decision_authority")
        or not scope.get("reentry_requires")
    ):
        raise SoapPreflightFault("campaign_scope_decision_invalid")
    return config


def safetensor_shapes(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    rows = []
    for name, record in header.items():
        if name == "__metadata__":
            continue
        shape = [int(value) for value in record["shape"]]
        elements = math.prod(shape)
        rows.append(
            {
                "name": name,
                "shape": shape,
                "rank": len(shape),
                "elements": elements,
                "dtype": record["dtype"],
            }
        )
    return rows


def full_shape_accounting(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    matrices = [row for row in rows if row["rank"] == 2]
    parameter_elements = sum(row["elements"] for row in rows)
    factor_elements = sum(
        sum(dimension * dimension for dimension in row["shape"])
        for row in matrices
    )
    largest_dimension = max(dimension for row in matrices for dimension in row["shape"])
    largest_factor_elements = largest_dimension * largest_dimension
    state = config["state_contract"]
    dtype_bytes = int(config["dtype_bytes"])
    parameter_runtime = parameter_elements * dtype_bytes * (
        int(state["parameter_copies"])
        + int(state["gradient_copies"])
        + int(state["adam_moment_copies"])
    )
    soap_persistent = factor_elements * dtype_bytes * (
        int(state["kronecker_covariance_copies"])
        + int(state["eigenbasis_copies"])
    )
    workspace = largest_factor_elements * dtype_bytes * int(
        state["largest_factor_workspace_copies"]
    )
    total = parameter_runtime + soap_persistent + workspace
    costly = sorted(
        (
            {
                "name": row["name"],
                "shape": row["shape"],
                "factor_elements": sum(d * d for d in row["shape"]),
            }
            for row in matrices
        ),
        key=lambda row: -row["factor_elements"],
    )
    return {
        "tensor_count": len(rows),
        "matrix_count": len(matrices),
        "parameter_elements": parameter_elements,
        "full_shape_factor_elements": factor_elements,
        "largest_factor_dimension": largest_dimension,
        "parameter_gradient_moment_mib": parameter_runtime / (1024**2),
        "soap_persistent_factor_mib": soap_persistent / (1024**2),
        "largest_factor_workspace_mib": workspace / (1024**2),
        "total_lower_bound_mib": total / (1024**2),
        "largest_factor_owners": costly[:10],
        "matrix_shapes": [row["shape"] for row in matrices],
    }


def mlx_eigh_support_and_timing(config: dict[str, Any]) -> dict[str, Any]:
    if not host_resource_safety.accelerator_child_authorized():
        raise SoapPreflightFault("accelerator_watchdog_required")
    payload = {
        "dimensions": config["cpu_eigh_benchmark_dimensions"],
        "repetitions": int(config["timing_repetitions"]),
    }
    code = r'''
import json,time
import mlx.core as mx
p=json.loads(input())
gpu_supported=True;gpu_fault=""
x=mx.eye(4)
try:
    w,q=mx.linalg.eigh(x);mx.eval(w,q)
except Exception as exc:
    gpu_supported=False;gpu_fault=f"{type(exc).__name__}:{exc}"
rows=[]
for n in p["dimensions"]:
    durations=[]
    for repetition in range(p["repetitions"]+1):
        matrix=mx.random.normal((n,n),stream=mx.cpu)
        matrix=(matrix+matrix.T)*0.5;mx.eval(matrix)
        started=time.perf_counter();w,q=mx.linalg.eigh(matrix,stream=mx.cpu);mx.eval(w,q)
        if repetition: durations.append(time.perf_counter()-started)
    rows.append({"dimension":n,"median_seconds":sorted(durations)[len(durations)//2],"durations":durations})
print(json.dumps({"gpu_supported":gpu_supported,"gpu_fault":gpu_fault,"cpu_timings":rows}))
'''
    proc = subprocess.run(
        [platform.python_implementation() == "CPython" and __import__("sys").executable or "python3", "-c", code],
        input=json.dumps(payload) + "\n",
        text=True,
        capture_output=True,
        timeout=120,
    )
    if proc.returncode:
        raise SoapPreflightFault("mlx_eigh_probe_failed:" + proc.stderr[-1000:])
    return json.loads(proc.stdout)


def project_refresh_seconds(
    matrix_shapes: list[list[int]], timing: dict[str, Any], optimistic_multiplier: float
) -> dict[str, Any]:
    rates = []
    for row in timing["cpu_timings"]:
        seconds = float(row["median_seconds"])
        dimension = int(row["dimension"])
        rates.append((dimension**3) / max(seconds, 1e-12))
    measured_best_rate = max(rates)
    optimistic_rate = measured_best_rate * float(optimistic_multiplier)
    dimension_cubes = sum(sum(int(dimension) ** 3 for dimension in shape) for shape in matrix_shapes)
    projected = dimension_cubes / optimistic_rate
    return {
        "dimension_cube_sum": dimension_cubes,
        "measured_best_dimension_cube_per_second": measured_best_rate,
        "optimistic_throughput_multiplier": float(optimistic_multiplier),
        "optimistic_dimension_cube_per_second": optimistic_rate,
        "optimistic_full_refresh_seconds": projected,
    }


def execute(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    checkpoint = resolve(config["checkpoint"])
    receipt = json.loads(resolve(config["training_receipt"]).read_text(encoding="utf-8"))
    rows = safetensor_shapes(checkpoint)
    accounting = full_shape_accounting(rows, config)
    timing = mlx_eigh_support_and_timing(config)
    projection = project_refresh_seconds(
        accounting["matrix_shapes"],
        timing,
        float(config["optimistic_throughput_multiplier"]),
    )
    reference_step_seconds = float(receipt["wall_seconds"]) / int(receipt["optimizer_steps"])
    amortized = projection["optimistic_full_refresh_seconds"] / int(
        config["precondition_frequency_steps"]
    )
    allowed_overhead = reference_step_seconds * float(
        config["maximum_step_time_overhead_fraction"]
    )
    memory_cap = float(config["host_peak_memory_mib"]) * float(
        config["maximum_optimizer_memory_fraction"]
    )
    gates = {
        "full_shape_memory_lower_bound": accounting["total_lower_bound_mib"] <= memory_cap,
        "native_mlx_gpu_eigendecomposition": bool(timing["gpu_supported"]),
        "optimistic_amortized_update_timing": amortized <= allowed_overhead,
    }
    eligible = all(gates.values())
    scope = config["campaign_scope_decision"]
    disposition = (
        "ELIGIBLE_FOR_MATCHED_QUALITY_CANARY"
        if eligible
        else scope["ineligible_disposition"]
    )
    report = {
        "policy": POLICY,
        "schema_version": "1.0.0",
        "trigger_state": "GREEN",
        "checkpoint": relative(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "shape_accounting": accounting,
        "mlx_eigh": timing,
        "timing_projection": {
            **projection,
            "reference_adamw_step_seconds": reference_step_seconds,
            "precondition_frequency_steps": int(config["precondition_frequency_steps"]),
            "optimistic_amortized_seconds_per_step": amortized,
            "allowed_overhead_seconds_per_step": allowed_overhead,
        },
        "resource_gates": gates,
        "memory_cap_mib": memory_cap,
        "disposition": disposition,
        "finite_docket_membership": (
            "CONDITIONAL_MATCHED_QUALITY_CANARY"
            if eligible
            else scope["ineligible_membership"]
        ),
        "engineering_scope_decision": (
            "EXECUTE_MATCHED_QUALITY_CANARY"
            if eligible
            else "REMOVE_FULL_SHAPE_SOAP_FROM_FIRST_CAMPAIGN"
        ),
        "decision_authority": scope["decision_authority"],
        "scientific_optimizer_quality_claim": "NOT_EVALUATED",
        "reason": (
            "full-shape SOAP fits the declared memory and native timing envelope"
            if eligible
            else "full-shape SOAP is not routeable on this M1/MLX runtime without violating at least one measured resource gate"
        ),
        "reentry_condition": (
            None
            if eligible
            else scope["reentry_requires"]
        ),
        "claim_boundary": config["claim_boundary"],
        "public_training_rows": 0,
        "public_evaluation_rows": 0,
        "external_inference_calls": 0,
        "production_checkpoint_mutation": False,
    }
    output = resolve(config["report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    report = execute(resolve(args.config))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
