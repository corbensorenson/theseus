#!/usr/bin/env python3
"""Build and qualify the exact native ANE self-attention forward slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native/ane_metal/ane_exact_attention_forward.m"
POLICY = "project_theseus_exact_ane_attention_forward_qualification_v1"
STAGES = (0, 1, 2, 3, 4)


class QualificationFault(ValueError):
    pass


def binary_path(stage: int, *, aligned: bool = True) -> Path:
    suffix = "aligned" if aligned else "unaligned"
    return Path(f"/private/tmp/theseus_exact_attention_stage{stage}_{suffix}")


def build_command(stage: int, *, aligned: bool = True) -> list[str]:
    command = [
        "xcrun",
        "clang",
        "-fobjc-arc",
        "-O3",
        "-DACCELERATE_NEW_LAPACK",
        f"-DQUALIFICATION_STAGE={stage}",
    ]
    if not aligned:
        command.append("-DNORM_SCALE_SPAN=1")
    command.extend(
        [
            "-framework",
            "Foundation",
            "-framework",
            "IOSurface",
            "-framework",
            "Accelerate",
            str(SOURCE),
            "-o",
            str(binary_path(stage, aligned=aligned)),
        ]
    )
    return command


def build(stage: int, *, aligned: bool = True) -> None:
    completed = subprocess.run(
        build_command(stage, aligned=aligned),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise QualificationFault(
            f"native_build_failed_stage_{stage}:"
            + completed.stdout.decode("utf-8", errors="replace")[-2000:]
        )


def run(stage: int, *, aligned: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(binary_path(stage, aligned=aligned))],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def parse_last_json(output: bytes) -> dict[str, Any]:
    text = output.decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise QualificationFault("native_output_missing_json")


def validate(
    aligned_stages: list[dict[str, Any]],
    unaligned: subprocess.CompletedProcess[bytes],
) -> dict[str, Any]:
    if len(aligned_stages) != len(STAGES):
        raise QualificationFault("aligned_stage_count_mismatch")
    for expected, receipt in zip(STAGES, aligned_stages, strict=True):
        if expected < 4:
            if (
                receipt.get("qualification_stage") != expected
                or receipt.get("trigger_state") != "GREEN_RUNTIME_EXECUTION"
            ):
                raise QualificationFault(f"stage_{expected}_not_green")
        elif (
            receipt.get("policy")
            != "project_theseus_exact_ane_attention_forward_v1"
            or receipt.get("trigger_state") != "GREEN"
            or int(receipt.get("mismatch_count", -1)) != 0
        ):
            raise QualificationFault("full_forward_not_green")
    full = aligned_stages[-1]
    comparisons = full["comparisons"]
    for name in (
        "attended",
        "query_rope",
        "key_rope",
        "value",
        "attention_norm",
    ):
        item = comparisons[name]
        if (
            int(item["mismatch_count"]) != 0
            or float(item["maximum_absolute_delta"]) > float(item["tolerance"])
        ):
            raise QualificationFault(f"comparison_failed:{name}")
    unaligned_text = unaligned.stdout.decode("utf-8", errors="replace")
    unaligned_reproduced = (
        unaligned.returncode != 0
        and "ANEProgramProcessRequestDirect() Failed with status=0x1d"
        in unaligned_text
    )
    if not unaligned_reproduced:
        raise QualificationFault("unaligned_runtime_failure_not_reproduced")
    return {
        "policy": POLICY,
        "state": "GREEN_EXACT_ATTENTION_FORWARD_NATIVE_BACKWARD_OPEN",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "shape": full["shape"],
        "packed_surface": {
            "unaligned_spatial_extent": 897,
            "unaligned_compile_succeeded": True,
            "unaligned_evaluation_returncode": unaligned.returncode,
            "unaligned_runtime_status": "0x1d",
            "aligned_norm_scale_span": 128,
            "aligned_spatial_extent": 1024,
            "aligned_all_runtime_stages_green": True,
            "claim_scope": (
                "An exact-shape M1 runtime compatibility constraint. It does "
                "not establish a universal ANE alignment rule."
            ),
        },
        "runtime_bisect": [
            {
                "stage": stage,
                "scope": (
                    "packed_surface_passthrough"
                    if stage == 0
                    else "rmsnorm"
                    if stage == 1
                    else "rmsnorm_dynamic_qkv"
                    if stage == 2
                    else "rmsnorm_dynamic_qkv_split_half_rope"
                    if stage == 3
                    else "exact_attention_forward"
                ),
                "compile_milliseconds": float(receipt["compile_milliseconds"]),
                "mean_evaluation_milliseconds": float(
                    receipt["mean_evaluation_milliseconds"]
                ),
                "green": True,
            }
            for stage, receipt in zip(STAGES, aligned_stages, strict=True)
        ],
        "comparisons": comparisons,
        "gates": {
            "aligned_surface_runtime": True,
            "dynamic_rmsnorm_scale": True,
            "dynamic_qkv_weights": True,
            "split_half_rope": True,
            "contiguous_gqa": True,
            "causal_attention": True,
            "forward_taps_for_backward": True,
            "output_parity": True,
            "input_gradient": False,
            "every_parameter_gradient": False,
            "out_projection_and_unscaled_residual": False,
            "swiglu_and_second_residual": False,
            "scalar_loss_and_optimizer_update": False,
            "complete_decoder_block": False,
            "production_eligible": False,
        },
        "next_gate": (
            "Implement exact attention backward from the emitted Q/K/V/"
            "normalized-input taps, including split-half inverse rotation, "
            "contiguous GQA reduction, dX, attention RMSNorm-scale gradient, "
            "and Q/K/V parameter gradients; then add out_proj/residual and "
            "SwiGLU."
        ),
        "claim_scope": (
            "One native exact self-attention forward slice at batch one and "
            "sequence 128. No backward, optimizer, full block, full model, "
            "training speed, convergence, serving, or capability claim."
        ),
        "capability_claim": "NONE_ENGINEERING_FORWARD_SLICE_ONLY",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build(0, aligned=False)
    unaligned = run(0, aligned=False)
    aligned: list[dict[str, Any]] = []
    for stage in STAGES:
        build(stage, aligned=True)
        completed = run(stage, aligned=True)
        if completed.returncode:
            raise QualificationFault(
                f"aligned_stage_{stage}_failed:"
                + completed.stdout.decode("utf-8", errors="replace")[-2000:]
            )
        aligned.append(parse_last_json(completed.stdout))
    report = validate(aligned, unaligned)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
