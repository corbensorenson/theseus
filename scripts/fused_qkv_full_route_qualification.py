#!/usr/bin/env python3
"""Disposition checkpoint-compatible fused QKV on matched full-route evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from fresh_process_pretraining_qualification import compare_safetensors


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_fused_qkv_full_route_qualification_v1"
CONFIG_POLICY = (
    "project_theseus_fused_qkv_full_route_qualification_config_v1"
)
DEFAULT_CONFIG = ROOT / "configs/fused_qkv_full_route_qualification.json"


class FusedQKVQualificationFault(ValueError):
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
        raise FusedQKVQualificationFault(
            f"json_object_required:{relative(path)}"
        )
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = read_json(path)
    if config.get("policy") != CONFIG_POLICY:
        raise FusedQKVQualificationFault("config_policy_invalid")
    pairs = config.get("pairs") or []
    if len(pairs) < 3:
        raise FusedQKVQualificationFault(
            "three_alternating_pairs_required"
        )
    orders = [tuple(row.get("order") or ()) for row in pairs]
    if orders != [
        ("control", "candidate"),
        ("candidate", "control"),
        ("control", "candidate"),
    ]:
        raise FusedQKVQualificationFault(
            "alternating_order_contract_invalid"
        )
    boundaries = config.get("hard_boundaries") or {}
    for field in (
        "public_training_rows",
        "public_evaluation_rows",
        "external_inference_calls",
        "frozen_evaluation_surface_consumption",
    ):
        if boundaries.get(field) != 0:
            raise FusedQKVQualificationFault(
                f"hard_boundary_nonzero:{field}"
            )
    if (
        boundaries.get("production_checkpoint_mutation") is not False
        or boundaries.get("production_config_mutation") is not False
    ):
        raise FusedQKVQualificationFault(
            "production_mutation_boundary_invalid"
        )
    return config


def decide(
    *,
    speedup_ratios: list[float],
    loss_deltas: list[float],
    model_state_passed: bool,
    optimizer_state_passed: bool,
    rng_state_exact: bool,
    matched_authority: bool,
    resources_passed: bool,
    maximum_loss_delta: float,
) -> dict[str, Any]:
    if len(speedup_ratios) < 3 or any(
        not math.isfinite(value) or value <= 0.0
        for value in speedup_ratios
    ):
        raise FusedQKVQualificationFault("valid_speedup_pairs_required")
    every_pair_wins = all(value > 1.0 for value in speedup_ratios)
    geometric_mean = math.exp(
        sum(math.log(value) for value in speedup_ratios)
        / len(speedup_ratios)
    )
    loss_passed = all(
        abs(value) <= maximum_loss_delta for value in loss_deltas
    )
    gates = {
        "candidate_wins_every_pair": every_pair_wins,
        "pooled_direction_positive": geometric_mean > 1.0,
        "final_loss_within_frozen_tolerance": loss_passed,
        "model_state_within_frozen_tolerance": model_state_passed,
        "optimizer_state_within_frozen_tolerance": (
            optimizer_state_passed
        ),
        "rng_state_exact": rng_state_exact,
        "matched_checkpoint_optimizer_data_cursor_and_batches": (
            matched_authority
        ),
        "resource_guards_passed": resources_passed,
    }
    selected = all(gates.values())
    return {
        "gates": gates,
        "selected": selected,
        "geometric_mean_speedup": geometric_mean,
        "candidate_win_count": sum(
            value > 1.0 for value in speedup_ratios
        ),
        "pair_count": len(speedup_ratios),
        "maximum_final_loss_absolute_delta": max(
            abs(value) for value in loss_deltas
        ),
    }


def execute(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    production_config_path = resolve(config["production_config"])
    production_config = read_json(production_config_path)
    selected_projection = str(
        (production_config.get("training") or {}).get(
            "self_attention_projection"
        )
        or "separate"
    )
    if selected_projection != "separate":
        raise FusedQKVQualificationFault(
            "production_projection_already_changed"
        )
    pair_receipts: list[dict[str, Any]] = []
    ratios: list[float] = []
    loss_deltas: list[float] = []
    authority_fields = (
        "starting_checkpoint_sha256",
        "starting_optimizer_state_sha256",
        "optimizer_positions",
        "data_cursor_start",
        "data_cursor_next",
        "batch_index_sha256_prefix",
    )
    all_authority = True
    all_resources = True
    evidence: dict[str, dict[str, Any]] = {}
    total_control_seconds = 0.0
    total_candidate_seconds = 0.0
    for pair in config["pairs"]:
        paths = {
            name: resolve(pair[name])
            for name in (
                "control",
                "candidate",
                "control_guard",
                "candidate_guard",
            )
        }
        control = read_json(paths["control"])
        candidate = read_json(paths["candidate"])
        control_guard = read_json(paths["control_guard"])
        candidate_guard = read_json(paths["candidate_guard"])
        if control.get("self_attention_projection") != "separate":
            raise FusedQKVQualificationFault(
                f"control_projection_invalid:{pair['id']}"
            )
        if candidate.get("self_attention_projection") != "fused_qkv":
            raise FusedQKVQualificationFault(
                f"candidate_projection_invalid:{pair['id']}"
            )
        if (
            int(control.get("optimizer_steps") or -1)
            != int(
                config["hard_boundaries"][
                    "optimizer_steps_per_process"
                ]
            )
            or int(candidate.get("optimizer_steps") or -1)
            != int(
                config["hard_boundaries"][
                    "optimizer_steps_per_process"
                ]
            )
        ):
            raise FusedQKVQualificationFault(
                f"step_count_invalid:{pair['id']}"
            )
        matched = all(
            control.get(field) == candidate.get(field)
            for field in authority_fields
        )
        all_authority = all_authority and matched
        resources = all(
            receipt.get("passed") is True
            and receipt.get("terminated_by_guard") is not True
            for receipt in (control_guard, candidate_guard)
        )
        all_resources = all_resources and resources
        control_rate = float(
            control["warmup_excluded_positions_per_second"]
        )
        candidate_rate = float(
            candidate["warmup_excluded_positions_per_second"]
        )
        ratio = candidate_rate / control_rate
        loss_delta = float(candidate["final_loss"]) - float(
            control["final_loss"]
        )
        ratios.append(ratio)
        loss_deltas.append(loss_delta)
        total_control_seconds += float(
            control["warmup_excluded_seconds"]
        )
        total_candidate_seconds += float(
            candidate["warmup_excluded_seconds"]
        )
        pair_receipts.append(
            {
                "id": pair["id"],
                "order": pair["order"],
                "control_rate": control_rate,
                "candidate_rate": candidate_rate,
                "candidate_over_control_speedup": ratio,
                "final_loss_absolute_delta": abs(loss_delta),
                "authority_matched": matched,
                "resource_guards_passed": resources,
                "control_swap_growth_mib": float(
                    control_guard.get("maximum_swapout_growth_mib")
                    or 0.0
                ),
                "candidate_swap_growth_mib": float(
                    candidate_guard.get("maximum_swapout_growth_mib")
                    or 0.0
                ),
            }
        )
        for name, path in paths.items():
            evidence[f"{pair['id']}_{name}"] = {
                "path": relative(path),
                "sha256": sha256_file(path),
            }
    state_paths = {
        name: resolve(value)
        for name, value in config["retained_state_pair"].items()
    }
    model_comparison = compare_safetensors(
        state_paths["control_model"],
        state_paths["candidate_model"],
    )
    optimizer_comparison = compare_safetensors(
        state_paths["control_optimizer"],
        state_paths["candidate_optimizer"],
    )
    rng_comparison = compare_safetensors(
        state_paths["control_rng"],
        state_paths["candidate_rng"],
    )
    for name, path in state_paths.items():
        evidence[f"retained_{name}"] = {
            "path": relative(path),
            "sha256": sha256_file(path),
            "retention": (
                "DISPOSABLE_AFTER_QUALIFICATION_REPORT_PUBLICATION"
            ),
        }
    decision = decide(
        speedup_ratios=ratios,
        loss_deltas=loss_deltas,
        model_state_passed=model_comparison["passed"] is True,
        optimizer_state_passed=optimizer_comparison["passed"] is True,
        rng_state_exact=(
            rng_comparison["passed"] is True
            and float(rng_comparison["maximum_absolute_delta"]) == 0.0
        ),
        matched_authority=all_authority,
        resources_passed=all_resources,
        maximum_loss_delta=float(
            config["decision"]["maximum_final_loss_absolute_delta"]
        ),
    )
    selected = decision["selected"]
    return {
        "policy": POLICY,
        "trigger_state": "GREEN",
        "support_state": (
            "matched_full_route_engineering_disposition"
        ),
        "config": {
            "path": relative(config_path),
            "sha256": sha256_file(config_path),
        },
        "implementation": {
            "path": relative(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
            "model_path": "scripts/standard_causal_transformer_model.py",
            "model_sha256": sha256_file(
                ROOT / "scripts/standard_causal_transformer_model.py"
            ),
            "route_owner_path": (
                "scripts/resource_acceleration_qualification.py"
            ),
            "route_owner_sha256": sha256_file(
                ROOT / "scripts/resource_acceleration_qualification.py"
            ),
        },
        "production_authority": {
            "config": relative(production_config_path),
            "config_sha256": sha256_file(production_config_path),
            "selected_self_attention_projection": selected_projection,
            "changed": False,
            "live_checkpoint_mutated": False,
        },
        "timing": {
            "pairs": pair_receipts,
            "geometric_mean_candidate_over_control_speedup": (
                decision["geometric_mean_speedup"]
            ),
            "pooled_candidate_over_control_speedup": (
                total_control_seconds
                / max(total_candidate_seconds, 1e-12)
            ),
            "candidate_win_count": decision["candidate_win_count"],
            "pair_count": decision["pair_count"],
        },
        "numerical_authority": {
            "model_state": model_comparison,
            "optimizer_state": optimizer_comparison,
            "rng_state": rng_comparison,
            "maximum_final_loss_absolute_delta": decision[
                "maximum_final_loss_absolute_delta"
            ],
        },
        "gates": decision["gates"],
        "selection": {
            "candidate_selected": selected,
            "selected_projection": (
                "fused_qkv" if selected else "separate"
            ),
            "production_route_changed": False,
            "disposition": (
                "ADOPT_AFTER_NEW_CONTENT_ADDRESSED_FREEZE"
                if selected
                else (
                    "NOT_SELECTED_FIRST_CAMPAIGN_FULL_ROUTE_DIRECTION_"
                    "INCONSISTENT_AND_NUMERICAL_CUSTODY_FAILED"
                )
            ),
            "arbitrary_percentage_hurdle": False,
            "rule": config["decision"]["rule"],
        },
        "evidence": evidence,
        "hard_boundaries": config["hard_boundaries"],
        "claim_boundary": (
            "This rejects only the checkpoint-compatible concatenate-then-"
            "matmul fused-QKV execution graph on three alternating eight-"
            "update step-11416 full-route pairs. It does not falsify custom "
            "QKV kernels, other hardware, or projection fusion generally."
        ),
        "capability_claim": "NONE_ACCELERATION_DIAGNOSTIC_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = resolve(args.config)
    report = execute(config_path)
    output = resolve(load_config(config_path)["report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "trigger_state": report["trigger_state"],
                "selection": report["selection"],
                "timing": report["timing"],
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
    return 0 if report["trigger_state"] == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
