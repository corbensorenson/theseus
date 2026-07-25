#!/usr/bin/env python3
"""Merge a governed KERC compiler delta and bind an evaluation report to it."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import moecot_language_arm_training as training


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def require_fresh(path: Path, role: str) -> None:
    if path.exists():
        raise ValueError(f"{role} must be fresh: {training.relative(path)}")


def merge_training_report(
    source_report_path: Path,
    merged_checkpoint_path: Path,
    merge_receipt_path: Path,
    output_report_path: Path,
) -> dict[str, Any]:
    """Merge and independently bind one selective KERC checkpoint generation."""

    for path, role in (
        (merged_checkpoint_path, "merged checkpoint"),
        (merge_receipt_path, "merge receipt"),
        (output_report_path, "derived evaluation report"),
    ):
        require_fresh(path, role)

    source_report = training.read_json(source_report_path)
    results = source_report.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("source training report must contain exactly one result")
    result = results[0]
    if not isinstance(result, dict):
        raise ValueError("source training result must be an object")

    delta_path = resolve(str(result.get("checkpoint") or ""))
    delta_sha256 = str(result.get("checkpoint_sha256") or "")
    if (
        not delta_path.is_file()
        or not delta_sha256
        or training.sha256_file(delta_path) != delta_sha256
    ):
        raise ValueError("source training report delta checkpoint binding failed")

    selective = result.get("selective_compute_checkpoint")
    if not isinstance(selective, dict):
        raise ValueError("selective-compute checkpoint custody is missing")
    if (
        selective.get("policy")
        != "project_theseus_kerc_stage_selective_compute_checkpoint_v1"
        or selective.get("scope") != "compiler"
        or selective.get("selected_fp32_exact") is not True
        or selective.get("merge_into_verified_fp32_source_required_for_promotion")
        is not True
    ):
        raise ValueError("selective-compute checkpoint custody is not mergeable")
    source_checkpoint = resolve(str(selective.get("source_checkpoint") or ""))
    source_checkpoint_sha256 = str(
        selective.get("source_checkpoint_sha256") or ""
    )
    if (
        not source_checkpoint.is_file()
        or not source_checkpoint_sha256
        or training.sha256_file(source_checkpoint)
        != source_checkpoint_sha256
    ):
        raise ValueError("verified FP32 source checkpoint binding failed")

    receipt = training.merge_kerc_compiler_delta_checkpoint(
        source_checkpoint,
        delta_path,
        merged_checkpoint_path,
    )
    training.write_json_atomic(merge_receipt_path, receipt)

    derived = copy.deepcopy(source_report)
    derived_result = derived["results"][0]
    derived_result["checkpoint"] = training.relative(merged_checkpoint_path)
    derived_result["checkpoint_sha256"] = receipt["merged_checkpoint_sha256"]
    derived_result["checkpoint_representation"] = (
        "full_fp32_source_with_exact_compiler_delta_merge_v1"
    )
    derived_result["selective_compute_checkpoint"] = {
        **copy.deepcopy(selective),
        "resume_representation": "full_checkpoint_no_delta_overlay",
        "merged_checkpoint": training.relative(merged_checkpoint_path),
        "merged_checkpoint_sha256": receipt["merged_checkpoint_sha256"],
        "merge_receipt": training.relative(merge_receipt_path),
        "merge_receipt_sha256": training.sha256_file(merge_receipt_path),
    }
    derived["derived_diagnostic_report"] = {
        "policy": "project_theseus_kerc_compiler_delta_evaluation_report_v1",
        "source_training_report": training.relative(source_report_path),
        "source_training_report_sha256": training.sha256_file(
            source_report_path
        ),
        "merge_receipt": training.relative(merge_receipt_path),
        "merge_receipt_sha256": training.sha256_file(merge_receipt_path),
        "capability_claim": "NOT_EVALUATED",
    }
    training.write_json_atomic(output_report_path, derived)
    return {
        "output_report": training.relative(output_report_path),
        "output_report_sha256": training.sha256_file(output_report_path),
        **receipt,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--merged-checkpoint", required=True)
    parser.add_argument("--merge-receipt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = merge_training_report(
        resolve(args.training_report),
        resolve(args.merged_checkpoint),
        resolve(args.merge_receipt),
        resolve(args.out),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
