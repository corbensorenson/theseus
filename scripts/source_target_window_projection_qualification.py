#!/usr/bin/env python3
"""Qualify target-window-only source-conditioned output projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from fresh_process_pretraining_qualification import compare_safetensors


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_source_target_window_projection_qualification_v1"


class QualificationFault(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QualificationFault(f"json_object_required:{path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def evidence(path: Path) -> dict[str, Any]:
    return {"path": relative(path), "sha256": sha256_file(path)}


def decide(
    *,
    control: dict[str, Any],
    control_replay: dict[str, Any],
    candidate: dict[str, Any],
    control_resource: dict[str, Any],
    control_replay_resource: dict[str, Any],
    candidate_resource: dict[str, Any],
    control_replay_model_comparison: dict[str, Any],
    candidate_model_comparison: dict[str, Any],
) -> dict[str, Any]:
    reports = (control, control_replay, candidate)
    if any(
        report.get("training_phase") != "source_conditioned_pretraining"
        or report.get("route_mode") != "eager"
        or report.get("precision_mode") != "float32"
        or int(report.get("optimizer_steps") or 0) != 4
        for report in reports
    ):
        raise QualificationFault("route_authority_mismatch")
    authority_fields = (
        "starting_checkpoint_sha256",
        "starting_optimizer_state_sha256",
        "batch_index_sha256_prefix",
        "optimizer_positions",
        "data_cursor_start",
        "data_cursor_next",
    )
    if any(
        report.get(field) != control.get(field)
        for report in (control_replay, candidate)
        for field in authority_fields
    ):
        raise QualificationFault("sampler_or_checkpoint_authority_mismatch")
    if control.get("compact_output_projection") is True:
        raise QualificationFault("control_compaction_enabled")
    if control_replay.get("compact_output_projection") is True:
        raise QualificationFault("control_replay_compaction_enabled")
    if candidate.get("compact_output_projection") is not True:
        raise QualificationFault("candidate_compaction_disabled")

    control_rates = [
        float(control["post_first_positions_per_second"]),
        float(control_replay["post_first_positions_per_second"]),
    ]
    candidate_rate = float(candidate["post_first_positions_per_second"])
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in [*control_rates, candidate_rate]
    ):
        raise QualificationFault("invalid_rate")
    pooled_control = sum(control_rates) / len(control_rates)
    mean_speedup = candidate_rate / pooled_control
    conservative_speedup = candidate_rate / max(control_rates)
    exact_batches = all(
        report["batch_index_sha256_prefix"]
        == control["batch_index_sha256_prefix"]
        for report in (control_replay, candidate)
    )
    loss_delta = abs(float(candidate["final_loss"]) - float(control["final_loss"]))
    rng_exact = (
        candidate.get("rng_content") == control.get("rng_content")
        == control_replay.get("rng_content")
    )
    resources = (
        control_resource,
        control_replay_resource,
        candidate_resource,
    )
    resource_receipts_complete = all(
        receipt.get("passed") is True for receipt in resources
    )
    zero_swap_growth = all(
        float(receipt.get("maximum_swapout_growth_mib") or 0.0) == 0.0
        for receipt in resources
    )
    control_replay_green = (
        control_replay_model_comparison.get("passed") is True
    )
    candidate_state_green = candidate_model_comparison.get("passed") is True
    numerical_authority_green = bool(
        exact_batches
        and rng_exact
        and loss_delta <= 2e-6
        and control_replay_green
        and candidate_state_green
    )
    wall_green = mean_speedup > 1.0 and conservative_speedup > 1.0
    selected = bool(
        numerical_authority_green
        and wall_green
        and resource_receipts_complete
        and zero_swap_growth
    )
    return {
        "policy": POLICY,
        "trigger_state": (
            "GREEN_SELECTED"
            if selected
            else "INCONCLUSIVE_IMPLEMENTATION"
            if not candidate_state_green
            else "INCONCLUSIVE_RESOURCE"
            if not zero_swap_growth
            else "NOT_SELECTED"
        ),
        "selection": {
            "candidate_selected": selected,
            "production_route_changed": selected,
            "arbitrary_percentage_hurdle": False,
            "rule": (
                "Select only when the candidate beats both the pooled and "
                "faster control wall rates while sampler, loss, RNG, model "
                "state, and zero-swap resource authority remain green."
            ),
        },
        "timing": {
            "control_positions_per_second": control_rates,
            "candidate_positions_per_second": candidate_rate,
            "mean_control_positions_per_second": pooled_control,
            "mean_control_over_candidate_speedup": mean_speedup,
            "conservative_control_over_candidate_speedup": (
                conservative_speedup
            ),
        },
        "numerical_authority": {
            "batch_identity_exact": exact_batches,
            "rng_identity_exact": rng_exact,
            "final_loss_absolute_delta": loss_delta,
            "final_loss_absolute_delta_allowed": 2e-6,
            "control_replay_model": control_replay_model_comparison,
            "candidate_model": candidate_model_comparison,
        },
        "resource_authority": {
            "receipts_complete": resource_receipts_complete,
            "zero_swap_growth": zero_swap_growth,
            "maximum_swapout_growth_mib": max(
                float(receipt.get("maximum_swapout_growth_mib") or 0.0)
                for receipt in resources
            ),
            "maximum_inferred_unified_memory_mib": max(
                float(
                    receipt.get("maximum_inferred_unified_memory_mib")
                    or 0.0
                )
                for receipt in resources
            ),
            "minimum_reclaimable_available_mib": min(
                float(receipt.get("minimum_reclaimable_available_mib") or 0.0)
                for receipt in resources
            ),
        },
        "gates": {
            "same_checkpoint_optimizer_sampler_and_cursor": exact_batches,
            "independent_control_replay_within_frozen_tolerance": (
                control_replay_green
            ),
            "candidate_model_within_frozen_tolerance": candidate_state_green,
            "loss_and_rng_authority": rng_exact and loss_delta <= 2e-6,
            "candidate_beats_mean_control": mean_speedup > 1.0,
            "candidate_beats_faster_control": conservative_speedup > 1.0,
            "zero_swap_growth": zero_swap_growth,
            "production_eligible": selected,
        },
        "next_gate": (
            "Repair the compact tied-output/pointer backward so the exact "
            "source-conditioned model state stays inside the frozen tolerance, "
            "then repeat alternating no-swap trials. Do not migrate the "
            "production plan from this result."
        ),
        "claim_scope": (
            "One scratch-only four-update source-conditioned FP32 eager route "
            "on the 54.8M-parameter checkpoint. This is implementation and "
            "resource evidence only, not convergence or capability."
        ),
        "capability_claim": "NONE_ACCELERATION_DIAGNOSTIC_ONLY",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "control",
        "control_replay",
        "candidate",
        "control_resource",
        "control_replay_resource",
        "candidate_resource",
        "control_model",
        "control_replay_model",
        "candidate_model",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    control_replay_comparison = compare_safetensors(
        args.control_model, args.control_replay_model
    )
    candidate_comparison = compare_safetensors(
        args.control_model, args.candidate_model
    )
    report = decide(
        control=read_json(args.control),
        control_replay=read_json(args.control_replay),
        candidate=read_json(args.candidate),
        control_resource=read_json(args.control_resource),
        control_replay_resource=read_json(args.control_replay_resource),
        candidate_resource=read_json(args.candidate_resource),
        control_replay_model_comparison=control_replay_comparison,
        candidate_model_comparison=candidate_comparison,
    )
    report["evidence"] = {
        name: evidence(getattr(args, name))
        for name in (
            "control",
            "control_replay",
            "candidate",
            "control_resource",
            "control_replay_resource",
            "candidate_resource",
        )
    }
    report["scratch_state_inputs"] = {
        name: {
            "path": str(getattr(args, name).resolve()),
            "sha256": sha256_file(getattr(args, name)),
            "retention": "DISPOSABLE_AFTER_COMPARISON_REPORT_PUBLICATION",
        }
        for name in (
            "control_model",
            "control_replay_model",
            "candidate_model",
        )
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "trigger_state": report["trigger_state"],
                "candidate_selected": report["selection"][
                    "candidate_selected"
                ],
                "mean_speedup": report["timing"][
                    "mean_control_over_candidate_speedup"
                ],
                "failed_gates": [
                    name
                    for name, passed in report["gates"].items()
                    if not passed
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
