#!/usr/bin/env python3
"""Build and qualify the exact joined ANE/Metal/Accelerate decoder block."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native/ane_metal/exact_decoder_block_join.m"
DEPENDENCIES = (
    ROOT / "native/ane_metal/ane_exact_attention_forward.m",
    ROOT / "native/ane_metal/ane_exact_attention_backward.m",
    ROOT / "native/ane_metal/exact_decoder_block_remainder.m",
    ROOT / "native/ane_metal/ane_metal_surface_contract.h",
)
BINARY = Path("/private/tmp/theseus_exact_decoder_block_join_qualification")
POLICY = "project_theseus_exact_decoder_block_join_qualification_v1"


class QualificationFault(ValueError):
    pass


def build_command() -> list[str]:
    return [
        "xcrun",
        "clang",
        "-fobjc-arc",
        "-O3",
        "-DACCELERATE_NEW_LAPACK",
        "-framework",
        "Foundation",
        "-framework",
        "IOSurface",
        "-framework",
        "Metal",
        "-framework",
        "Accelerate",
        str(SOURCE),
        "-o",
        str(BINARY),
    ]


def parse_last_json(output: bytes) -> dict[str, Any]:
    for line in reversed(output.decode("utf-8", errors="replace").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise QualificationFault("native_output_missing_json")


def validate(native: dict[str, Any]) -> dict[str, Any]:
    if (
        native.get("policy") != "project_theseus_exact_decoder_block_join_v1"
        or native.get("trigger_state") != "GREEN"
        or int(native.get("mismatch_count", -1)) != 0
    ):
        raise QualificationFault("native_join_not_green")
    required = (
        "one_process",
        "compile_once_ane_forward_backward",
        "generation_tagged_iosurface_forward_backward",
        "single_thread_fp32_accelerate_dw",
        "native_metal_remainder",
        "all_nine_parameter_leaves",
        "combined_hidden_gradient",
        "one_objective_mass_normalization",
        "one_global_norm_and_clip",
        "one_fp32_adamw_publication",
        "replay_exact",
        "all_finite",
        "sixty_four_step_finite",
    )
    missing = [name for name in required if native["gates"].get(name) is not True]
    if missing:
        raise QualificationFault("missing_gate:" + ",".join(missing))
    if (
        int(native["parameter_elements"]) != 3_015_680
        or int(native["parameter_leaf_count"]) != 9
    ):
        raise QualificationFault("parameter_authority_mismatch")
    if float(native["minimum_leaf_nonzero_gradient_fraction"]) < 0.95:
        raise QualificationFault("gradient_coverage_below_frozen_gate")
    return {
        "policy": POLICY,
        "state": "GREEN_EXACT_DECODER_BLOCK_JOIN_MLX_CONTROL_OPEN",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "dependency_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in DEPENDENCIES
        },
        "shape": native["shape"],
        "parameter_generation": native["parameter_generation"],
        "parameter_elements": native["parameter_elements"],
        "parameter_leaf_count": native["parameter_leaf_count"],
        "objective_authority_mass": native["objective_authority_mass"],
        "timing": native["timing"],
        "loss": native["loss"],
        "global_gradient_norm": native["global_gradient_norm"],
        "gradient_coverage": {
            "attention_nonzero_fraction": native[
                "attention_nonzero_gradient_fraction"
            ],
            "remainder_nonzero_fraction": native[
                "remainder_nonzero_gradient_fraction"
            ],
            "minimum_leaf_nonzero_fraction": native[
                "minimum_leaf_nonzero_gradient_fraction"
            ],
            "frozen_minimum": 0.95,
        },
        "gates": {
            **native["gates"],
            "complete_decoder_block_mechanics": True,
            "matched_mlx_wall_control": False,
            "save_reload_file_roundtrip": False,
            "sustained_thermal_qualification": False,
            "production_eligible": False,
        },
        "next_gate": (
            "Run alternating complete-block repeats against the matched "
            "compiled-MLX FP32 and FP16 controls, add file save/reload and "
            "sustained thermal qualification, and select only if joined wall "
            "wins while replay and numerical authority remain green."
        ),
        "claim_scope": (
            "One exact batch-one sequence-128 decoder-block optimizer "
            "transaction. Complete mechanics do not establish a wall-time "
            "speedup, full-model integration, convergence, or capability."
        ),
        "capability_claim": "NONE_ENGINEERING_EXACT_BLOCK_JOIN_ONLY",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    built = subprocess.run(
        build_command(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if built.returncode:
        raise QualificationFault(
            "native_build_failed:"
            + built.stdout.decode("utf-8", errors="replace")[-3000:]
        )
    completed = subprocess.run(
        [str(BINARY)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise QualificationFault(
            "native_execution_failed:"
            + completed.stdout.decode("utf-8", errors="replace")[-3000:]
        )
    report = validate(parse_last_json(completed.stdout))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
