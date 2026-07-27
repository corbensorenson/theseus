#!/usr/bin/env python3
"""Qualify MLX_METAL_FAST_SYNCH without an arbitrary percentage floor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from fresh_process_pretraining_qualification import compare_safetensors


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_mlx_fast_sync_qualification_v1"
CONFIG_POLICY = "project_theseus_mlx_fast_sync_qualification_config_v1"


class QualificationFault(ValueError):
    pass


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QualificationFault(f"json_object_required:{relative(path)}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def disposition(
    *,
    speedup_ratios: list[float],
    model_comparison: dict[str, Any],
    current_control: dict[str, Any],
    current_candidate: dict[str, Any],
    control_resource: dict[str, Any],
    candidate_resource: dict[str, Any],
) -> dict[str, Any]:
    if len(speedup_ratios) < 4 or any(
        not math.isfinite(value) or value <= 0.0
        for value in speedup_ratios
    ):
        raise QualificationFault("four_valid_pairs_required")
    authority_fields = (
        "starting_checkpoint_sha256",
        "starting_optimizer_state_sha256",
        "batch_index_sha256_prefix",
        "optimizer_positions",
        "data_cursor_start",
        "data_cursor_next",
    )
    if any(
        current_candidate.get(field) != current_control.get(field)
        for field in authority_fields
    ):
        raise QualificationFault("current_pair_authority_mismatch")
    model_state_green = model_comparison.get("passed") is True
    current_loss_exact = (
        current_candidate.get("final_loss")
        == current_control.get("final_loss")
    )
    candidate_wins_every_pair = min(speedup_ratios) > 1.0
    pooled_geometric = math.exp(
        sum(math.log(value) for value in speedup_ratios)
        / len(speedup_ratios)
    )
    zero_swap = all(
        float(receipt.get("maximum_swapout_growth_mib") or 0.0) == 0.0
        for receipt in (control_resource, candidate_resource)
    )
    resources_complete = all(
        receipt.get("passed") is True
        for receipt in (control_resource, candidate_resource)
    )
    selected = bool(
        model_state_green
        and current_loss_exact
        and candidate_wins_every_pair
        and zero_swap
        and resources_complete
    )
    return {
        "policy": POLICY,
        "trigger_state": (
            "GREEN_SELECTED"
            if selected
            else "INCONCLUSIVE_EXPERIMENT"
        ),
        "selection": {
            "candidate_selected": selected,
            "production_route_changed": selected,
            "arbitrary_percentage_hurdle": False,
            "rule": (
                "Select only if every matched pair wins, the current full "
                "model state remains within the frozen tolerance, loss and "
                "sampler authority match, and the current pair adds no swap."
            ),
        },
        "timing": {
            "matched_pair_speedup_ratios": speedup_ratios,
            "geometric_mean_speedup": pooled_geometric,
            "minimum_pair_speedup": min(speedup_ratios),
            "maximum_pair_speedup": max(speedup_ratios),
            "candidate_win_count": sum(
                value > 1.0 for value in speedup_ratios
            ),
            "pair_count": len(speedup_ratios),
        },
        "numerical_authority": {
            "model_state": model_comparison,
            "final_loss_exact": current_loss_exact,
        },
        "resource_authority": {
            "receipts_complete": resources_complete,
            "zero_swap_growth": zero_swap,
            "maximum_swapout_growth_mib": max(
                float(
                    control_resource.get("maximum_swapout_growth_mib")
                    or 0.0
                ),
                float(
                    candidate_resource.get("maximum_swapout_growth_mib")
                    or 0.0
                ),
            ),
        },
        "gates": {
            "matched_checkpoint_optimizer_sampler_and_cursor": True,
            "full_model_state_within_frozen_tolerance": model_state_green,
            "final_loss_exact": current_loss_exact,
            "candidate_wins_every_pair": candidate_wins_every_pair,
            "zero_swap_growth": zero_swap,
            "production_eligible": selected,
        },
        "next_gate": (
            "Retain ordinary synchronization. Reopen only with an alternating "
            "no-swap pair because the fourth matched pair reversed the prior "
            "small gain; no percentage threshold is involved."
        ),
        "claim_scope": (
            "Four matched FP32 compiled pretraining timing pairs plus one "
            "eight-update full-model state comparison on this M1. This is "
            "execution evidence only, not convergence or capability."
        ),
        "capability_claim": "NONE_ACCELERATION_DIAGNOSTIC_ONLY",
    }


def execute(
    config_path: Path,
    *,
    control_model: Path,
    candidate_model: Path,
) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("policy") != CONFIG_POLICY:
        raise QualificationFault("config_policy_invalid")
    evidence: dict[str, Any] = {}
    ratios: list[float] = []
    for index, pair in enumerate(config["historical_pairs"]):
        control_path = resolve(pair["control"])
        candidate_path = resolve(pair["candidate"])
        control = read_json(control_path)
        candidate = read_json(candidate_path)
        ratios.append(
            float(candidate["post_first_positions_per_second"])
            / float(control["post_first_positions_per_second"])
        )
        evidence[f"historical_control_{index}"] = {
            "path": relative(control_path),
            "sha256": sha256_file(control_path),
        }
        evidence[f"historical_candidate_{index}"] = {
            "path": relative(candidate_path),
            "sha256": sha256_file(candidate_path),
        }
    current = config["current_pair"]
    paths = {name: resolve(value) for name, value in current.items()}
    control = read_json(paths["control"])
    candidate = read_json(paths["candidate"])
    control_resource = read_json(paths["control_resource"])
    candidate_resource = read_json(paths["candidate_resource"])
    command = [str(value) for value in candidate_resource.get("command") or []]
    if command[:2] != ["env", "MLX_METAL_FAST_SYNCH=1"]:
        raise QualificationFault("candidate_environment_not_bound")
    if (
        [str(value) for value in control_resource.get("command") or []][:1]
        == ["env"]
    ):
        raise QualificationFault("control_environment_contaminated")
    ratios.append(
        float(candidate["post_first_positions_per_second"])
        / float(control["post_first_positions_per_second"])
    )
    model_comparison = compare_safetensors(
        control_model, candidate_model
    )
    report = disposition(
        speedup_ratios=ratios,
        model_comparison=model_comparison,
        current_control=control,
        current_candidate=candidate,
        control_resource=control_resource,
        candidate_resource=candidate_resource,
    )
    report["config"] = {
        "path": relative(config_path),
        "sha256": sha256_file(config_path),
    }
    report["evidence"] = evidence | {
        name: {
            "path": relative(path),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }
    report["scratch_state_inputs"] = {
        "control_model": {
            "path": str(control_model.resolve()),
            "sha256": sha256_file(control_model),
            "retention": "DISPOSABLE_AFTER_COMPARISON_REPORT_PUBLICATION",
        },
        "candidate_model": {
            "path": str(candidate_model.resolve()),
            "sha256": sha256_file(candidate_model),
            "retention": "DISPOSABLE_AFTER_COMPARISON_REPORT_PUBLICATION",
        },
    }
    out = resolve(config["report"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--control-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    args = parser.parse_args()
    report = execute(
        args.config.resolve(),
        control_model=args.control_model.resolve(),
        candidate_model=args.candidate_model.resolve(),
    )
    print(
        json.dumps(
            {
                "trigger_state": report["trigger_state"],
                "candidate_selected": report["selection"][
                    "candidate_selected"
                ],
                "geometric_mean_speedup": report["timing"][
                    "geometric_mean_speedup"
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
