#!/usr/bin/env python3
"""Alternate the exact native decoder block against its compiled-MLX control."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
JOIN_MODULE_PATH = ROOT / "scripts/exact_decoder_block_join_qualification.py"
MLX_CONTROL = ROOT / "scripts/exact_decoder_block_mlx_control.py"
POLICY = "project_theseus_exact_decoder_block_backend_bakeoff_v1"


class BakeoffFault(ValueError):
    pass


def load_join_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "exact_decoder_block_join_qualification", JOIN_MODULE_PATH
    )
    if not spec or not spec.loader:
        raise BakeoffFault("join_module_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_last_json(output: bytes) -> dict[str, Any]:
    text = output.decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    start = text.find("{")
    if start >= 0:
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    raise BakeoffFault("subprocess_output_missing_json")


def validate_pair(native: dict[str, Any], mlx: dict[str, Any]) -> None:
    if native.get("trigger_state") != "GREEN":
        raise BakeoffFault("native_not_green")
    if mlx.get("state") != "GREEN_MATCHED_COMPILED_MLX_CONTROL":
        raise BakeoffFault("mlx_not_green")
    for field in ("shape", "parameter_elements", "parameter_leaf_count"):
        if native.get(field) != mlx.get(field):
            raise BakeoffFault(f"authority_mismatch:{field}")
    if float(native["objective_authority_mass"]) != float(
        mlx["objective_authority_mass"]
    ):
        raise BakeoffFault("authority_mismatch:objective_mass")
    required_native = (
        "replay_exact",
        "all_finite",
        "sixty_four_step_finite",
        "one_fp32_adamw_publication",
    )
    required_mlx = (
        "replay_exact",
        "sixty_four_step_finite",
        "one_fp32_adamw_publication",
        "matched_precision_split",
    )
    if any(native["gates"].get(name) is not True for name in required_native):
        raise BakeoffFault("native_gate_failed")
    if any(mlx["gates"].get(name) is not True for name in required_mlx):
        raise BakeoffFault("mlx_gate_failed")


def decide(
    native_rounds: list[dict[str, Any]],
    mlx_rounds: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(native_rounds) != len(mlx_rounds) or len(native_rounds) < 2:
        raise BakeoffFault("alternating_round_count_invalid")
    for native, mlx in zip(native_rounds, mlx_rounds, strict=True):
        validate_pair(native, mlx)
    native_ms = [
        float(value["timing"]["mean_joined_64_milliseconds"])
        for value in native_rounds
    ]
    mlx_ms = [
        float(value["timing"]["mean_milliseconds"])
        for value in mlx_rounds
    ]
    if any(not math.isfinite(value) or value <= 0.0 for value in native_ms + mlx_ms):
        raise BakeoffFault("invalid_timing")
    mean_native = sum(native_ms) / len(native_ms)
    mean_mlx = sum(mlx_ms) / len(mlx_ms)
    mean_control_over_candidate = mean_mlx / mean_native
    conservative_control_over_candidate = min(mlx_ms) / max(native_ms)
    native_selected = (
        mean_control_over_candidate > 1.0
        and conservative_control_over_candidate > 1.0
    )
    return {
        "policy": POLICY,
        "state": (
            "GREEN_MATCHED_BAKEOFF_SELECT_NATIVE"
            if native_selected
            else "GREEN_MATCHED_BAKEOFF_RETAIN_MLX"
        ),
        "shape": native_rounds[0]["shape"],
        "parameter_elements": native_rounds[0]["parameter_elements"],
        "parameter_leaf_count": native_rounds[0]["parameter_leaf_count"],
        "objective_authority_mass": native_rounds[0][
            "objective_authority_mass"
        ],
        "alternating_rounds": len(native_rounds),
        "native_mean_joined_milliseconds_by_round": native_ms,
        "mlx_mean_joined_milliseconds_by_round": mlx_ms,
        "native_mean_milliseconds": mean_native,
        "mlx_mean_milliseconds": mean_mlx,
        "mean_control_over_candidate_speedup": mean_control_over_candidate,
        "conservative_control_over_candidate_speedup": (
            conservative_control_over_candidate
        ),
        "first_step_numerical_delta": {
            "absolute_loss": abs(
                float(native_rounds[0]["loss"])
                - float(mlx_rounds[0]["first_loss"])
            ),
            "absolute_gradient_norm": abs(
                float(native_rounds[0]["global_gradient_norm"])
                - float(mlx_rounds[0]["first_gradient_norm"])
            ),
            "scope": (
                "Descriptive mixed-FP16 implementation delta; component-level "
                "analytical propagation and exact-ABI gates remain authoritative."
            ),
        },
        "selection": {
            "native_selected": native_selected,
            "retained_backend": "native_ane_metal_accelerate"
            if native_selected
            else "compiled_mlx",
            "rule": (
                "Select native only when both the pooled mean and the "
                "worst-native versus best-control comparison beat MLX while "
                "all authority gates remain green."
            ),
            "arbitrary_percentage_hurdle": False,
        },
        "gates": {
            "alternating_order": True,
            "matched_shape": True,
            "matched_parameter_and_objective_authority": True,
            "matched_precision_split": True,
            "replay_and_stability": True,
            "native_joined_wall_beats_mlx_mean": (
                mean_control_over_candidate > 1.0
            ),
            "native_joined_wall_beats_mlx_conservative": (
                conservative_control_over_candidate > 1.0
            ),
            "native_production_eligible": native_selected,
        },
        "canonical_backend_changed": native_selected,
        "next_gate": (
            "If native is retained, add file save/reload and sustained thermal "
            "qualification. If MLX is retained, stop this exact native route "
            "and continue the source-conditioned model on compiled MLX."
        ),
        "claim_scope": (
            "Exact batch-one sequence-128 decoder-block backend selection only. "
            "This does not falsify other ANE shapes/topologies or establish "
            "full-model, convergence, utility, or capability results."
        ),
        "capability_claim": "NONE_ENGINEERING_BACKEND_SELECTION_ONLY",
    }


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise BakeoffFault(
            f"subprocess_failed:{completed.returncode}:"
            + completed.stdout.decode("utf-8", errors="replace")[-3000:]
        )
    return parse_last_json(completed.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rounds < 2 or args.steps < 64:
        raise BakeoffFault("insufficient_bakeoff_budget")
    join = load_join_module()
    built = subprocess.run(
        join.build_command(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if built.returncode:
        raise BakeoffFault(
            "native_build_failed:"
            + built.stdout.decode("utf-8", errors="replace")[-3000:]
        )
    native_rounds: list[dict[str, Any]] = []
    mlx_rounds: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="theseus_block_bakeoff_") as directory:
        temporary = Path(directory)
        for round_index in range(args.rounds):
            mlx_command = [
                str(Path(__import__("sys").executable)),
                str(MLX_CONTROL),
                "--out",
                str(temporary / f"mlx_{round_index}.json"),
                "--steps",
                str(args.steps),
            ]
            native_command = [str(join.BINARY)]
            if round_index % 2 == 0:
                native_rounds.append(run_command(native_command))
                mlx_rounds.append(run_command(mlx_command))
            else:
                mlx_rounds.append(run_command(mlx_command))
                native_rounds.append(run_command(native_command))
    report = decide(native_rounds, mlx_rounds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
