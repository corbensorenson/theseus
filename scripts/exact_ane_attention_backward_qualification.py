#!/usr/bin/env python3
"""Build and qualify the exact native ANE causal-attention backward core."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native/ane_metal/ane_exact_attention_backward.m"
BINARY = Path("/private/tmp/theseus_exact_attention_backward_qualification")
POLICY = "project_theseus_exact_ane_attention_backward_qualification_v1"


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
        "Accelerate",
        str(SOURCE),
        "-o",
        str(BINARY),
    ]


def parse_last_json(output: bytes) -> dict[str, Any]:
    text = output.decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise QualificationFault("native_output_missing_json")


def validate(native: dict[str, Any]) -> dict[str, Any]:
    if (
        native.get("policy")
        != "project_theseus_exact_ane_attention_backward_v1"
        or native.get("trigger_state") != "GREEN"
        or int(native.get("mismatch_count", -1)) != 0
    ):
        raise QualificationFault("native_backward_core_not_green")
    for name in (
        "dq_rope",
        "dk_tiled_rope",
        "dv_tiled",
        "dq_inverse_split_half_rope",
        "dk_contiguous_reduce_inverse_split_half_rope",
        "dv_contiguous_reduce",
    ):
        item = native["comparisons"][name]
        if (
            int(item["mismatch_count"]) != 0
            or float(item["maximum_absolute_delta"]) > float(item["tolerance"])
        ):
            raise QualificationFault(f"comparison_failed:{name}")
    operator = native["comparisons"]["accelerate_projection_operator"]
    if (
        int(operator["mismatch_count"]) != 0
        or float(operator["maximum_absolute_delta"])
        > float(operator["tolerance"])
    ):
        raise QualificationFault("comparison_failed:accelerate_projection_operator")
    for name in (
        "q_proj_weight_gradient",
        "k_proj_weight_gradient",
        "v_proj_weight_gradient",
        "projection_input_gradient",
        "attention_norm_input_gradient",
        "attention_norm_scale_gradient",
    ):
        item = native["comparisons"][name]
        if (
            item.get("tolerance_policy")
            != "analytical_fp16_boundary_propagation"
            or int(item["mismatch_count"]) != 0
            or float(item["maximum_absolute_delta"])
            > float(item["maximum_allowed_delta"])
        ):
            raise QualificationFault(f"comparison_failed:{name}")
    return {
        "policy": POLICY,
        "state": "GREEN_EXACT_ATTENTION_GRADIENT_TREE_BLOCK_REMAINDER_OPEN",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "shape": native["shape"],
        "parameter_generation": native["parameter_generation"],
        "compile_milliseconds": native["compile_milliseconds"],
        "mean_evaluation_milliseconds": native["mean_evaluation_milliseconds"],
        "mean_cpu_projection_gradient_milliseconds": native[
            "mean_cpu_projection_gradient_milliseconds"
        ],
        "comparisons": native["comparisons"],
        "gates": {
            "causal_softmax_backward": True,
            "full_query_head_dq_dk_dv": True,
            "output_parity": True,
            "contiguous_gqa_kv_reduction": True,
            "inverse_split_half_rope": True,
            "attention_rmsnorm_input_gradient": True,
            "attention_rmsnorm_scale_gradient": True,
            "qkv_parameter_gradients": True,
            "fp32_gradient_accumulation": True,
            "single_thread_accelerate": True,
            "complete_attention_gradient_tree": True,
            "complete_decoder_block": False,
            "production_eligible": False,
        },
        "next_gate": (
            "Add the output projection and unscaled first residual, then the "
            "second RMSNorm, SwiGLU, down projection, unscaled residual, "
            "scalar loss, and one authoritative optimizer update."
        ),
        "claim_scope": (
            "One native causal-attention backward core with contiguous KV "
            "reduction, inverse split-half RoPE, FP32 Q/K/V parameter "
            "gradients, and attention-RMSNorm input/scale gradients. No "
            "complete decoder block, optimizer, full model, speedup, or "
            "capability claim."
        ),
        "capability_claim": "NONE_ENGINEERING_ATTENTION_GRADIENT_ONLY",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build = subprocess.run(
        build_command(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if build.returncode:
        raise QualificationFault(
            "native_build_failed:"
            + build.stdout.decode("utf-8", errors="replace")[-2000:]
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
