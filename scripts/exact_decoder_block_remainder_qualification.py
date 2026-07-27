#!/usr/bin/env python3
"""Build and qualify the exact native Metal/Accelerate block remainder."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native/ane_metal/exact_decoder_block_remainder.m"
BINARY = Path("/private/tmp/theseus_exact_decoder_block_remainder_qualification")
POLICY = "project_theseus_exact_decoder_block_remainder_qualification_v1"


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
        native.get("policy")
        != "project_theseus_exact_decoder_block_remainder_v1"
        or native.get("trigger_state") != "GREEN"
        or int(native.get("mismatch_count", -1)) != 0
    ):
        raise QualificationFault("native_remainder_not_green")
    required_comparisons = (
        "swiglu_activation",
        "swiglu_gate_gradient",
        "swiglu_up_gradient",
    )
    for name in required_comparisons:
        item = native["comparisons"][name]
        if (
            int(item["mismatch_count"]) != 0
            or float(item["maximum_absolute_delta"]) > float(item["tolerance"])
        ):
            raise QualificationFault(f"comparison_failed:{name}")
    required_gates = (
        "out_projection_and_unscaled_residual",
        "second_rmsnorm_forward_backward",
        "swiglu_forward_backward",
        "down_projection",
        "masked_scalar_loss",
        "all_five_parameter_leaves",
        "attended_and_direct_hidden_gradients",
        "one_global_clip",
        "one_fp32_adamw_update",
        "replay_exact",
        "sixty_four_step_finite",
    )
    missing = [name for name in required_gates if native["gates"].get(name) is not True]
    if missing:
        raise QualificationFault("missing_gate:" + ",".join(missing))
    if int(native["parameter_elements"]) != 2_621_952:
        raise QualificationFault("parameter_count_mismatch")
    if float(native["nonzero_gradient_fraction"]) != 1.0:
        raise QualificationFault("incomplete_gradient_coverage")
    return {
        "policy": POLICY,
        "state": "GREEN_EXACT_BLOCK_REMAINDER_ATTENTION_JOIN_OPEN",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "shape": native["shape"],
        "parameter_generation": native["parameter_generation"],
        "parameter_elements": native["parameter_elements"],
        "parameter_leaves": [
            "attention.out_proj.weight",
            "ffn_norm.weight",
            "feed_forward.gate.weight",
            "feed_forward.up.weight",
            "feed_forward.down.weight",
        ],
        "objective_authority_mass": native["objective_authority_mass"],
        "timing": native["timing"],
        "comparisons": native["comparisons"],
        "loss": native["loss"],
        "gradient_norm": native["gradient_norm"],
        "nonzero_gradient_fraction": native["nonzero_gradient_fraction"],
        "gates": {
            **native["gates"],
            "native_metal_elementwise_loss_reduction_update": True,
            "single_thread_fp32_accelerate_gemms": True,
            "all_nine_block_parameter_leaves": False,
            "complete_decoder_block": False,
            "matched_mlx_wall_control": False,
            "production_eligible": False,
        },
        "next_gate": (
            "Join the exact ANE attention forward/backward tree to this "
            "Metal/Accelerate remainder over generation-tagged IOSurfaces, "
            "then compare the complete native block against compiled MLX."
        ),
        "claim_scope": (
            "Exact native decoder-block remainder mechanics from hidden and "
            "attended boundaries through one update. No joined attention "
            "tree, complete block, full model, speedup, or capability claim."
        ),
        "capability_claim": "NONE_ENGINEERING_BLOCK_REMAINDER_ONLY",
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
            + built.stdout.decode("utf-8", errors="replace")[-2000:]
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
            + completed.stdout.decode("utf-8", errors="replace")[-2000:]
        )
    report = validate(parse_last_json(completed.stdout))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
