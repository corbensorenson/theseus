#!/usr/bin/env python3
"""Matched private learning/resource selection for canonical MLX optimizers."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

import mtp_matched_adequacy as corpus
import pretraining_candidate_canary
import pretraining_optimizers
from standard_causal_transformer_model import (
    CausalTransformerConfig,
    analytical_parameter_breakdown,
    build_model,
)
from standard_causal_transformer_survival import (
    SOURCE_TARGET_SEPARATOR_ID,
    model_vocab_size,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/optimizer_matched_adequacy.json"
POLICY = "project_theseus_optimizer_matched_adequacy_v1"


class OptimizerAdequacyFault(ValueError):
    pass


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def journal_contract(
    config_path: Path, scratch: Path
) -> tuple[Path, Path, dict[str, Any]]:
    contract_path = scratch / "run_journal.contract.json"
    journal_path = scratch / "run_journal.jsonl"
    contract = {
        "policy": "project_theseus_optimizer_run_journal_v1",
        "config_sha256": sha256_file(config_path),
        "optimizer_implementation_sha256": sha256_file(
            ROOT / "scripts/pretraining_optimizers.py"
        ),
        "adequacy_implementation_sha256": sha256_file(Path(__file__)),
    }
    return contract_path, journal_path, contract


def prepare_run_journal(
    config_path: Path, scratch: Path
) -> tuple[Path, list[dict[str, Any]], bool]:
    contract_path, journal_path, expected = journal_contract(
        config_path, scratch
    )
    resumable = False
    rows: list[dict[str, Any]] = []
    if contract_path.is_file() and journal_path.is_file():
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed == expected:
            for line_number, line in enumerate(
                journal_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise OptimizerAdequacyFault(
                        f"run_journal_invalid:{line_number}"
                    ) from exc
                if not isinstance(row, dict) or not row.get("stage"):
                    raise OptimizerAdequacyFault(
                        f"run_journal_row_invalid:{line_number}"
                    )
                rows.append(row)
            resumable = True
    if not resumable:
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True)
        contract_path.write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        journal_path.touch()
    return journal_path, rows, resumable


def append_run_journal(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("policy") != POLICY:
        raise OptimizerAdequacyFault("config_policy_invalid")
    candidates = config.get("candidates") or []
    legacy_expected = {
        "adafactor_mlx",
        "adamw_mlx",
        "muon_mlx",
        "schedule_free_adamw_mlx",
    }
    expected = set(config.get("candidate_inventory") or legacy_expected)
    observed = {row.get("id") for row in candidates}
    if (
        observed != expected
        or "adamw_mlx" not in observed
        or not observed <= pretraining_optimizers.OPTIMIZER_IDS
    ):
        raise OptimizerAdequacyFault("candidate_inventory_invalid")
    profile_counts = {len(row.get("profiles") or []) for row in candidates}
    if profile_counts != {3}:
        raise OptimizerAdequacyFault("matched_tuning_opportunity_invalid")
    profile_ids: set[str] = set()
    for candidate in candidates:
        for profile in candidate["profiles"]:
            if not profile.get("id") or float(profile.get("learning_rate") or 0.0) <= 0.0:
                raise OptimizerAdequacyFault("optimizer_profile_invalid")
            if profile["id"] in profile_ids:
                raise OptimizerAdequacyFault("optimizer_profile_duplicate")
            profile_ids.add(profile["id"])
            if candidate["id"] == "muon_mlx" and float(profile.get("muon_learning_rate") or 0.0) <= 0.0:
                raise OptimizerAdequacyFault("muon_matrix_rate_missing")
    policy_card = config.get("policy_card_contract") or {}
    for field in (
        "schema_version",
        "implementation_version",
        "gradient_accumulation_steps",
        "distributed_placement",
        "stopping_rule",
        "rescue_budget",
    ):
        if policy_card.get(field) in (None, ""):
            raise OptimizerAdequacyFault(f"policy_card_contract_missing:{field}")
    transfer = config.get("width_transfer") or {}
    target_config = json.loads(
        resolve(transfer.get("target_architecture_config") or "").read_text(
            encoding="utf-8"
        )
    )
    frozen_target_width = int(
        target_config["candidate"]["shared_trunk_model"]["d_model"]
    )
    if (
        transfer.get("policy") != "selected_recipe_proxy_to_target_width_v1"
        or int(transfer.get("source_width") or 0) != int(config["model"]["d_model"])
        or int(transfer.get("target_width") or 0) != frozen_target_width
        or int((transfer.get("model") or {}).get("d_model") or 0)
        != frozen_target_width
        or transfer.get("isolate_width_only") is not True
        or list(transfer.get("seeds") or []) != list(config["seeds"])
        or int(transfer.get("steps") or 0) <= 0
        or int(transfer.get("confirmation_surface_consumption", -1)) != 0
    ):
        raise OptimizerAdequacyFault("width_transfer_contract_invalid")
    boundaries = config.get("hard_boundaries") or {}
    for field in (
        "public_training_rows",
        "public_evaluation_rows",
        "external_inference_calls",
        "fallback_or_template_credit",
        "confirmation_surface_consumption",
    ):
        if boundaries.get(field) != 0:
            raise OptimizerAdequacyFault(f"hard_boundary_nonzero:{field}")
    for field in (
        "production_checkpoint_mutation",
        "heldout_labels_visible_to_tuning",
        "selection_from_update_throughput_alone",
    ):
        if boundaries.get(field) is not False:
            raise OptimizerAdequacyFault(f"hard_boundary_boolean_invalid:{field}")
    if int(config["training"]["final_steps"]) > 192:
        raise OptimizerAdequacyFault("candidate_step_budget_exceeded")
    if float(config["training"].get("gradient_clip_norm") or 0.0) <= 0.0:
        raise OptimizerAdequacyFault("gradient_clip_contract_invalid")
    parameter_range = config.get("model_parameter_count_range")
    if parameter_range is not None and (
        int(parameter_range.get("minimum") or 0) <= 0
        or int(parameter_range.get("maximum") or 0)
        < int(parameter_range.get("minimum") or 0)
    ):
        raise OptimizerAdequacyFault("model_parameter_count_range_invalid")
    return config


POLICY_CARD_REQUIRED_FIELDS = {
    "card_id",
    "schema_version",
    "implementation",
    "parameterization",
    "eligibility_groups_and_fallback",
    "state_tensors_and_dtypes",
    "exact_update_order",
    "learning_rate_and_warmup",
    "decay",
    "clipping",
    "epsilon_and_stabilizers",
    "approximation_cadence_and_precision",
    "accumulation",
    "distributed_placement",
    "tuning_and_rescue_budget",
    "stopping_rule",
    "full_checkpoint_state",
}


def optimizer_policy_cards(
    config: dict[str, Any],
    *,
    selected_profiles: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build executable, content-bound identity cards for compared optimizers."""

    selected_profiles = selected_profiles or {}
    candidate_profiles = {
        row["id"]: row["profiles"] for row in config["candidates"]
    }
    implementation = {
        "path": relative(ROOT / "scripts/pretraining_optimizers.py"),
        "sha256": sha256_file(ROOT / "scripts/pretraining_optimizers.py"),
        "version": config["policy_card_contract"]["implementation_version"],
    }
    common = {
        "schema_version": config["policy_card_contract"]["schema_version"],
        "implementation": implementation,
        "accumulation": {
            "gradient_accumulation_steps": int(
                config["policy_card_contract"]["gradient_accumulation_steps"]
            ),
            "gradient_reduction": "none_before_single_step",
        },
        "distributed_placement": config["policy_card_contract"][
            "distributed_placement"
        ],
        "tuning_and_rescue_budget": {
            "profiles_per_candidate": 3,
            "tuning_seeds": list(config["seeds"]),
            "tune_steps": int(config["training"]["tune_steps"]),
            "final_steps": int(config["training"]["final_steps"]),
            "rescue": config["policy_card_contract"]["rescue_budget"],
        },
        "stopping_rule": config["policy_card_contract"]["stopping_rule"],
    }
    cards: dict[str, dict[str, Any]] = {
        "adamw_mlx": {
            **common,
            "card_id": "optimizer_policy.adamw_mlx.v1",
            "parameterization": {
                "betas": [0.9, 0.999],
                "bias_correction": False,
            },
            "eligibility_groups_and_fallback": {
                "eligible": "all trainable tensors",
                "fallback": None,
            },
            "state_tensors_and_dtypes": {
                "per_parameter": ["m:same_as_parameter", "v:same_as_parameter"],
                "global": ["step:uint64", "learning_rate:float32"],
            },
            "exact_update_order": [
                "clip global gradient norm before optimizer state update",
                "increment step",
                "update first and second moments",
                "apply decoupled weight decay",
                "apply normalized moment update",
            ],
            "learning_rate_and_warmup": {
                "profiles": candidate_profiles.get("adamw_mlx", []),
                "warmup": "none in the matched fixed-rate rung",
            },
            "decay": {
                "kind": "decoupled_weight_decay",
                "weight_decay": float(config["training"]["weight_decay"]),
            },
            "clipping": {
                "global_gradient_norm_before_update": float(
                    config["training"]["gradient_clip_norm"]
                ),
                "optimizer_internal": "none",
            },
            "epsilon_and_stabilizers": {"eps": 1e-8},
            "approximation_cadence_and_precision": {
                "approximation": "none",
                "persistent_precision": "parameter_dtype",
            },
            "full_checkpoint_state": [
                "step",
                "learning_rate",
                "every parameter m",
                "every parameter v",
            ],
        },
        "adafactor_mlx": {
            **common,
            "card_id": "optimizer_policy.adafactor_mlx.v1",
            "parameterization": {
                "decay_rate": -0.8,
                "beta1": None,
                "relative_step": False,
                "parameter_scale": True,
            },
            "eligibility_groups_and_fallback": {
                "factored": "every trainable tensor with ndim >= 2",
                "fallback": "unfactored second moment for vectors and scalars",
            },
            "state_tensors_and_dtypes": {
                "matrix": [
                    "exp_avg_sq_row:parameter_dtype:shape[:-1]",
                    "exp_avg_sq_col:parameter_dtype:shape[:-2]+shape[-1:]",
                ],
                "vector_or_scalar": ["exp_avg_sq:same_shape_and_dtype"],
                "global": ["step:uint64", "learning_rate:float32"],
            },
            "exact_update_order": [
                "clip global gradient norm before optimizer state update",
                "increment step",
                "update factored or unfactored second moment",
                "form inverse-rms preconditioned update",
                "clip update rms",
                "scale by max(eps2, parameter rms) and fixed learning rate",
                "apply decoupled weight decay and parameter update",
            ],
            "learning_rate_and_warmup": {
                "profiles": candidate_profiles.get("adafactor_mlx", []),
                "relative_step": False,
                "warmup": "none",
            },
            "decay": {
                "kind": "decoupled_weight_decay",
                "weight_decay": float(config["training"]["weight_decay"]),
                "second_moment_decay_rate": -0.8,
            },
            "clipping": {
                "global_gradient_norm_before_update": float(
                    config["training"]["gradient_clip_norm"]
                ),
                "optimizer_update_rms_threshold": 1.0,
            },
            "epsilon_and_stabilizers": {
                "squared_gradient_eps1": 1e-30,
                "parameter_scale_eps2": 1e-3,
            },
            "approximation_cadence_and_precision": {
                "approximation": "row_column_factored_second_moment",
                "cadence": "every optimizer step",
                "persistent_precision": "parameter_dtype",
            },
            "full_checkpoint_state": [
                "step",
                "learning_rate",
                "every matrix row and column factor",
                "every vector or scalar unfactored second moment",
            ],
        },
        "muon_mlx": {
            **common,
            "card_id": "optimizer_policy.muon_mlx.v1",
            "parameterization": {
                "momentum": 0.95,
                "nesterov": True,
                "newton_schulz_steps": 5,
            },
            "eligibility_groups_and_fallback": {
                "muon": "hidden 2D weights excluding embeddings, readouts, classifiers, pointers, registers, norms, and biases",
                "fallback": "AdamW for every remaining trainable tensor",
            },
            "state_tensors_and_dtypes": {
                "muon_matrix": ["momentum_buffer:parameter_dtype"],
                "adamw_fallback": [
                    "m:same_as_parameter",
                    "v:same_as_parameter",
                ],
                "global": "independent Muon and AdamW steps and learning rates",
            },
            "exact_update_order": [
                "clip global gradient norm before either child update",
                "partition by content-bound path predicate",
                "Muon momentum and Newton-Schulz orthogonalization on eligible matrices",
                "AdamW update on fallback tensors",
                "merge disjoint updated trees",
            ],
            "learning_rate_and_warmup": {
                "profiles": candidate_profiles.get("muon_mlx", []),
                "warmup": "none in the matched fixed-rate rung",
            },
            "decay": {
                "kind": "optimizer_native_decoupled",
                "weight_decay": float(config["training"]["weight_decay"]),
            },
            "clipping": {
                "global_gradient_norm_before_update": float(
                    config["training"]["gradient_clip_norm"]
                ),
                "optimizer_internal": "none",
            },
            "epsilon_and_stabilizers": {
                "adamw_eps": 1e-8,
                "newton_schulz_steps": 5,
            },
            "approximation_cadence_and_precision": {
                "approximation": "Newton-Schulz orthogonalization",
                "cadence": "every eligible-matrix update",
                "precision": "parameter_dtype",
                "required_observations": [
                    "orthogonalization_residual",
                    "matrix_shape",
                    "spectral_norm_estimate",
                ],
            },
            "full_checkpoint_state": [
                "both child optimizer steps and learning rates",
                "every Muon momentum buffer",
                "every fallback AdamW m and v",
                "content-bound parameter partition",
            ],
        },
        "schedule_free_adamw_mlx": {
            **common,
            "card_id": "optimizer_policy.schedule_free_adamw_mlx.v1",
            "parameterization": {
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_lr_power": 2.0,
            },
            "eligibility_groups_and_fallback": {
                "eligible": "all trainable tensors",
                "fallback": None,
            },
            "state_tensors_and_dtypes": {
                "per_parameter": [
                    "z:same_as_parameter",
                    "x:same_as_parameter",
                    "y:same_as_parameter",
                    "exp_avg_sq:same_as_parameter",
                ],
                "global": [
                    "step",
                    "learning_rate",
                    "weight_sum",
                    "lr_max",
                    "scheduled_lr",
                    "ckp1",
                ],
            },
            "exact_update_order": [
                "clip global gradient norm before optimizer state update",
                "increment step and compute warmup rate",
                "update averaging weight",
                "update second moment and z iterate",
                "update averaged x iterate",
                "publish y training iterate",
                "publish x only for evaluation then restore y",
            ],
            "learning_rate_and_warmup": {
                "profiles": candidate_profiles.get(
                    "schedule_free_adamw_mlx", []
                ),
                "linear_warmup_steps": int(config["training"]["warmup_steps"]),
            },
            "decay": {
                "kind": "decoupled_on_z_using_y",
                "weight_decay": float(config["training"]["weight_decay"]),
            },
            "clipping": {
                "global_gradient_norm_before_update": float(
                    config["training"]["gradient_clip_norm"]
                ),
                "optimizer_internal": "none",
            },
            "epsilon_and_stabilizers": {"eps": 1e-8},
            "approximation_cadence_and_precision": {
                "approximation": "online weighted averaging",
                "cadence": "every optimizer step",
                "persistent_precision": "parameter_dtype",
            },
            "full_checkpoint_state": [
                "all global schedule and averaging state",
                "every parameter z, x, y, and exp_avg_sq",
                "publication mode restored to training y after evaluation",
            ],
        },
        "ademamix_mlx": {
            **common,
            "card_id": "optimizer_policy.ademamix_mlx.v1",
            "parameterization": {
                "betas": [0.9, 0.999, 0.9999],
                "alpha": 8.0,
                "alpha_warmup_steps": int(
                    config["training"]["final_steps"]
                ),
                "beta3_half_life_warmup_steps": int(
                    config["training"]["final_steps"]
                ),
            },
            "eligibility_groups_and_fallback": {
                "eligible": "all trainable tensors",
                "fallback": None,
            },
            "state_tensors_and_dtypes": {
                "per_parameter": [
                    "exp_avg_fast:same_as_parameter",
                    "exp_avg_slow:same_as_parameter",
                    "exp_avg_sq:same_as_parameter",
                ],
                "global": [
                    "step:uint64",
                    "learning_rate",
                    "effective_alpha:float32",
                    "effective_beta3:float32",
                ],
            },
            "exact_update_order": [
                "clip global gradient norm before optimizer state update",
                "increment step",
                "linearly warm alpha and half-life-linearly warm beta3",
                "update fast, slow, and squared exponential averages",
                "bias-correct fast and squared averages",
                "mix fast and alpha-scaled slow averages",
                "apply normalized update plus AdamW-equivalent decay",
            ],
            "learning_rate_and_warmup": {
                "profiles": candidate_profiles.get("ademamix_mlx", []),
                "slow_ema_and_alpha_warmup": "full matched final-step budget",
            },
            "decay": {
                "kind": "AdamW-equivalent parameter term in update",
                "weight_decay": float(config["training"]["weight_decay"]),
            },
            "clipping": {
                "global_gradient_norm_before_update": float(
                    config["training"]["gradient_clip_norm"]
                ),
                "optimizer_internal": "none",
            },
            "epsilon_and_stabilizers": {"eps": 1e-8},
            "approximation_cadence_and_precision": {
                "approximation": "none",
                "cadence": "all three moments every step",
                "persistent_precision": "parameter_dtype",
            },
            "full_checkpoint_state": [
                "step and scheduled learning rate",
                "effective alpha and beta3",
                "every parameter fast, slow, and squared EMA",
            ],
        },
        "adam_mini_mlx": {
            **common,
            "card_id": "optimizer_policy.adam_mini_mlx.v1",
            "parameterization": {
                "betas": [0.9, 0.999],
                "partition_version": "official_v1.1_with_theseus_names",
                "d_model": int(config["model"]["d_model"]),
                "num_heads": int(config["model"]["num_heads"]),
                "num_kv_heads": int(config["model"]["num_kv_heads"]),
            },
            "eligibility_groups_and_fallback": {
                "full_adam": "bias tensors",
                "per_head_second_moment": "query and key projections",
                "per_row_second_moment": (
                    "embedding, output, value, attention-output, and MLP matrices"
                ),
                "scalar_second_moment_no_decay": "normalization tensors",
                "scalar_second_moment_fallback": "remaining tensors",
            },
            "state_tensors_and_dtypes": {
                "all_parameters": ["m:same_shape_and_dtype"],
                "bias": ["v:same_shape_and_dtype"],
                "query_key": ["vmean:one_scalar_per_attention_head"],
                "row_partition": ["vmean:one_scalar_per_output_row"],
                "block_partition": ["vmean:scalar"],
                "global": "one step and learning-rate state per nonempty partition",
            },
            "exact_update_order": [
                "clip global gradient norm before optimizer state update",
                "partition by content-bound parameter path",
                "increment each nonempty partition step",
                "update full first moment",
                "update full, per-head, per-row, or scalar second moment",
                "bias-correct both moments",
                "apply partition-preconditioned update and decoupled decay",
            ],
            "learning_rate_and_warmup": {
                "profiles": candidate_profiles.get("adam_mini_mlx", []),
                "warmup": "none in the matched fixed-rate rung",
            },
            "decay": {
                "kind": "decoupled_weight_decay_except_norm_and_bias",
                "weight_decay": float(config["training"]["weight_decay"]),
            },
            "clipping": {
                "global_gradient_norm_before_update": float(
                    config["training"]["gradient_clip_norm"]
                ),
                "optimizer_internal": "none",
            },
            "epsilon_and_stabilizers": {"eps": 1e-8},
            "approximation_cadence_and_precision": {
                "approximation": (
                    "Hessian-informed shared second moments by head, row, or block"
                ),
                "cadence": "every optimizer step",
                "persistent_precision": "parameter_dtype",
            },
            "full_checkpoint_state": [
                "all child partition steps and learning rates",
                "every full first moment",
                "every full, per-head, per-row, or scalar second moment",
                "content-bound partition policy and model dimensions",
            ],
        },
    }
    configured = {row["id"] for row in config["candidates"]}
    cards = {
        optimizer_id: card
        for optimizer_id, card in cards.items()
        if optimizer_id in configured
    }
    for optimizer_id, card in cards.items():
        selected = selected_profiles.get(optimizer_id)
        card["selected_profile"] = selected["profile"] if selected else None
        card["card_digest"] = pretraining_optimizers.optimizer_contract_digest(
            {key: value for key, value in card.items() if key != "card_digest"}
        )
    return cards


def validate_optimizer_policy_cards(
    cards: dict[str, dict[str, Any]], config: dict[str, Any]
) -> list[str]:
    faults: list[str] = []
    expected = {row["id"] for row in config["candidates"]}
    if set(cards) != expected:
        faults.append("policy_card_inventory_mismatch")
    for optimizer_id in sorted(expected):
        card = cards.get(optimizer_id) or {}
        missing = sorted(POLICY_CARD_REQUIRED_FIELDS - set(card))
        if missing:
            faults.append(f"policy_card_fields_missing:{optimizer_id}:{','.join(missing)}")
        observed = card.get("card_digest")
        recomputed = pretraining_optimizers.optimizer_contract_digest(
            {key: value for key, value in card.items() if key != "card_digest"}
        )
        if observed != recomputed:
            faults.append(f"policy_card_digest_invalid:{optimizer_id}")
    return faults


def split_tuning_rows(
    rows_by_arm: dict[str, list[dict[str, Any]]],
    *,
    tune_rows_per_arm: int,
    namespace: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    train: dict[str, list[dict[str, Any]]] = {}
    tune: dict[str, list[dict[str, Any]]] = {}
    for arm, rows in rows_by_arm.items():
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{namespace}:{arm}:{row['source_identity']}".encode()
            ).hexdigest(),
        )
        if tune_rows_per_arm <= 0 or tune_rows_per_arm >= len(ranked):
            raise OptimizerAdequacyFault(f"tune_split_invalid:{arm}")
        tune[arm] = ranked[:tune_rows_per_arm]
        train[arm] = ranked[tune_rows_per_arm:]
    return train, tune


def source_sets(rows_by_arm: dict[str, list[dict[str, Any]]]) -> set[str]:
    return {
        str(row["source_identity"])
        for rows in rows_by_arm.values()
        for row in rows
    }


def model_config(
    config: dict[str, Any],
    vocab_size: int,
    *,
    model_override: dict[str, Any] | None = None,
) -> CausalTransformerConfig:
    row = model_override or config["model"]
    return CausalTransformerConfig(
        vocab_size=vocab_size,
        d_model=int(row["d_model"]),
        num_layers=int(row["num_layers"]),
        num_heads=int(row["num_heads"]),
        num_kv_heads=int(row["num_kv_heads"]),
        ff_dim=int(row["ff_dim"]),
        attention_policy=str(row["attention_policy"]),
        source_target_separator_token_id=SOURCE_TARGET_SEPARATOR_ID,
    )


def parameter_count_gate(
    config: dict[str, Any], vocab_size: int
) -> dict[str, Any]:
    count = sum(
        analytical_parameter_breakdown(model_config(config, vocab_size)).values()
    )
    declared = config.get("model_parameter_count_range")
    passed = (
        True
        if declared is None
        else int(declared["minimum"]) <= count <= int(declared["maximum"])
    )
    return {
        "analytical_parameter_count": count,
        "declared_range": declared,
        "passed": passed,
    }


def parameter_digest(model: Any, mlx_utils: Any) -> str:
    digest = hashlib.sha256()
    for name, value in mlx_utils.tree_flatten(model.parameters()):
        array = np.asarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def tree_array_digest(tree: Any, mlx_utils: Any) -> str:
    digest = hashlib.sha256()
    for name, value in mlx_utils.tree_flatten(tree):
        array = np.asarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def tree_max_absolute_difference(left: Any, right: Any, mlx_utils: Any) -> float:
    left_flat = mlx_utils.tree_flatten(left)
    right_flat = mlx_utils.tree_flatten(right)
    if [name for name, _value in left_flat] != [name for name, _value in right_flat]:
        return float("inf")
    return max(
        (
            float(np.max(np.abs(np.asarray(left_value) - np.asarray(right_value))))
            if int(left_value.size)
            else 0.0
        )
        for (_left_name, left_value), (_right_name, right_value) in zip(
            left_flat, right_flat
        )
    )


def optimizer_state_bytes(optimizer: Any, mlx_utils: Any) -> int:
    return sum(
        int(value.size) * int(value.itemsize)
        for _name, value in mlx_utils.tree_flatten(optimizer.state)
        if hasattr(value, "size") and hasattr(value, "itemsize")
    )


def optimizer_state_inventory(optimizer: Any, mlx_utils: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": [int(size) for size in value.shape],
            "dtype": str(value.dtype),
            "bytes": int(value.size) * int(value.itemsize),
        }
        for name, value in mlx_utils.tree_flatten(optimizer.state)
        if hasattr(value, "shape")
        and hasattr(value, "size")
        and hasattr(value, "itemsize")
    ]


def muon_matrix_diagnostics(optimizer: Any, mlx_utils: Any) -> dict[str, Any] | None:
    if optimizer.__class__.__name__ != "MultiOptimizer":
        return None
    muon = optimizer.optimizers[0]
    rows = []
    for name, value in mlx_utils.tree_flatten(muon.state):
        if not name.endswith(".v") or int(getattr(value, "ndim", 0)) != 2:
            continue
        matrix = np.asarray(value, dtype=np.float64)
        norm = np.linalg.norm(matrix)
        if norm == 0.0:
            continue
        # This calls the exact candidate orthogonalizer and audits its result
        # independently in NumPy; the diagnostic is never used by generation.
        orthogonalized = np.asarray(
            muon._zeropower_via_newtonschulz5(value, steps=muon.ns_steps),
            dtype=np.float64,
        )
        if orthogonalized.shape[0] <= orthogonalized.shape[1]:
            gram = orthogonalized @ orthogonalized.T
        else:
            gram = orthogonalized.T @ orthogonalized
        identity = np.eye(gram.shape[0], dtype=np.float64)
        rows.append(
            {
                "state": name,
                "shape": list(matrix.shape),
                "input_spectral_norm": float(np.linalg.norm(matrix, ord=2)),
                "orthogonalized_spectral_norm": float(
                    np.linalg.norm(orthogonalized, ord=2)
                ),
                "orthogonalization_residual_frobenius_per_dimension": float(
                    np.linalg.norm(gram - identity, ord="fro")
                    / max(gram.shape[0], 1)
                ),
            }
        )
    return {
        "matrix_count": len(rows),
        "matrices": rows,
        "maximum_orthogonalization_residual_frobenius_per_dimension": max(
            (
                row["orthogonalization_residual_frobenius_per_dimension"]
                for row in rows
            ),
            default=None,
        ),
    }


def build_candidate_optimizer(
    candidate_id: str,
    profile: dict[str, Any],
    config: dict[str, Any],
    *,
    mx: Any,
    optim: Any,
    local_model_config: CausalTransformerConfig | None = None,
) -> Any:
    model = local_model_config or model_config(
        config, 1, model_override=config["model"]
    )
    return pretraining_optimizers.build_optimizer(
        candidate_id,
        learning_rate=float(profile["learning_rate"]),
        muon_learning_rate=(
            float(profile["muon_learning_rate"])
            if profile.get("muon_learning_rate") is not None
            else None
        ),
        weight_decay=float(config["training"]["weight_decay"]),
        warmup_steps=int(config["training"]["warmup_steps"]),
        ademamix_alpha=float(profile.get("alpha", 8.0)),
        ademamix_beta3=float(profile.get("beta3", 0.9999)),
        ademamix_alpha_warmup_steps=int(
            profile.get(
                "alpha_warmup_steps", config["training"]["final_steps"]
            )
        ),
        ademamix_beta3_warmup_steps=int(
            profile.get(
                "beta3_warmup_steps", config["training"]["final_steps"]
            )
        ),
        adam_mini_dim=int(model.d_model),
        adam_mini_num_heads=int(model.num_heads),
        adam_mini_num_kv_heads=int(model.num_kv_heads),
        optim=optim,
        mx=mx,
    )


def evaluate(model: Any, optimizer: Any, rows: dict[str, list[dict[str, Any]]], maximum: int, mx: Any, nn: Any) -> dict[str, Any]:
    schedule_free = hasattr(optimizer, "set_evaluation_iterate")
    if schedule_free:
        optimizer.set_evaluation_iterate(model)
        mx.eval(model.parameters())
    result = corpus._evaluate(model, rows, maximum, mx, nn)
    if schedule_free:
        optimizer.set_training_iterate(model)
        mx.eval(model.parameters())
    return result


def save_optimizer_state(path: Path, optimizer: Any, mx: Any, mlx_utils: Any) -> Path:
    flat = mlx_utils.tree_flatten(optimizer.state)
    arrays = {f"a{index}": value for index, (_name, value) in enumerate(flat)}
    mx.savez(str(path), **arrays)
    names = path.with_suffix(".names.json")
    names.write_text(json.dumps([name for name, _value in flat]) + "\n", encoding="utf-8")
    return names


def load_optimizer_state(path: Path, names: Path, optimizer: Any, mx: Any, mlx_utils: Any) -> None:
    loaded = mx.load(str(path))
    paths = json.loads(names.read_text(encoding="utf-8"))
    optimizer.state = mlx_utils.tree_unflatten(
        [(name, loaded[f"a{index}"]) for index, name in enumerate(paths)]
    )


def bind_loaded_optimizer_state(optimizer: Any, parameters: dict[str, Any]) -> None:
    """Mark loaded leaves initialized without replacing any retained moment."""

    optimizer.init(parameters)


def run_profile(
    config: dict[str, Any],
    *,
    candidate_id: str,
    profile: dict[str, Any],
    seed: int,
    train_rows: dict[str, list[dict[str, Any]]],
    eval_rows: dict[str, list[dict[str, Any]]],
    steps: int,
    stage: str,
    monitor: pretraining_candidate_canary.CandidateCanaryMonitor,
    retain_checkpoint: bool,
    model_override: dict[str, Any] | None = None,
    checkpoint_namespace: str | None = None,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    metadata = json.loads(resolve(config["stage_metadata"]).read_text(encoding="utf-8"))
    base = json.loads(resolve(config["base_config"]).read_text(encoding="utf-8"))
    vocabulary = model_vocab_size(base, metadata["source_vocab"], metadata["target_vocab"])
    local_model_config = model_config(
        config, vocabulary, model_override=model_override
    )
    mx.random.seed(int(seed))
    model = build_model(local_model_config, mx=mx, nn=nn)
    initial_parameter_sha256 = parameter_digest(model, mlx_utils)
    optimizer = build_candidate_optimizer(
        candidate_id,
        profile,
        config,
        mx=mx,
        optim=optim,
        local_model_config=local_model_config,
    )
    maximum = int(config["supervision"]["maximum_sequence_tokens"])
    batches = corpus.balanced_batches(train_rows, steps=steps, seed=seed)

    def objective(local_model: Any, x: Any, y: Any, mask: Any) -> Any:
        logits, _cache = local_model(x)
        losses = nn.losses.cross_entropy(logits, y)
        denominator = mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mx.float32))
        return mx.sum(losses * mask) / denominator

    value_and_grad = nn.value_and_grad(model, objective)
    initial = evaluate(model, optimizer, eval_rows, maximum, mx, nn)
    curve = [{"step": 0, "heldout": initial}]
    positions = 0
    started = time.perf_counter()
    first_gradient_l1 = 0.0
    first_gradient_norm = 0.0
    for step, batch in enumerate(batches, 1):
        x_np, y_np, mask_np = corpus.make_batch(batch, maximum)
        x, y, mask = mx.array(x_np), mx.array(y_np), mx.array(mask_np)
        loss, gradients = value_and_grad(model, x, y, mask)
        if step == 1:
            first_gradient_l1 = sum(
                float(mx.sum(mx.abs(value)).item())
                for _name, value in mlx_utils.tree_flatten(gradients)
            )
        gradients, gradient_norm = optim.clip_grad_norm(
            gradients, float(config["training"]["gradient_clip_norm"])
        )
        if step == 1:
            first_gradient_norm = float(gradient_norm.item())
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        positions += int(mask_np.sum())
        monitor.check(f"{stage}:{candidate_id}:{profile['id']}:{seed}", step)
        if step % int(config["training"]["evaluation_interval_steps"]) == 0 or step == steps:
            curve.append({"step": step, "heldout": evaluate(model, optimizer, eval_rows, maximum, mx, nn)})
    wall = time.perf_counter() - started
    final = curve[-1]["heldout"]
    result = {
        "stage": stage,
        "candidate_id": candidate_id,
        "profile": profile,
        "seed": int(seed),
        "model_config": asdict(local_model_config),
        "initial_parameter_sha256": initial_parameter_sha256,
        "optimizer_steps": steps,
        "optimizer_positions": positions,
        "first_gradient_l1": first_gradient_l1,
        "first_gradient_norm_before_clip": first_gradient_norm,
        "global_gradient_clip_norm": float(
            config["training"]["gradient_clip_norm"]
        ),
        "training_wall_seconds": wall,
        "optimizer_state_bytes": optimizer_state_bytes(optimizer, mlx_utils),
        "optimizer_state_inventory": optimizer_state_inventory(
            optimizer, mlx_utils
        ),
        "muon_matrix_diagnostics": muon_matrix_diagnostics(
            optimizer, mlx_utils
        ),
        "initial_heldout": initial,
        "final_heldout": final,
        "learning_curve": curve,
        "checkpoint_reload_exact_state": None,
        "checkpoint_next_update_numerically_equivalent": None,
        "checkpoint_next_update_max_absolute_error": None,
        "public_training_rows": 0,
        "public_evaluation_rows": 0,
        "external_inference_calls": 0,
        "fallback_or_template_credit": 0,
    }
    if retain_checkpoint:
        checkpoint_dir = resolve(config["scratch_root"])
        if checkpoint_namespace:
            checkpoint_dir = checkpoint_dir / checkpoint_namespace
        checkpoint_dir = checkpoint_dir / candidate_id / profile["id"] / str(seed)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        weights = checkpoint_dir / "weights.npz"
        state = checkpoint_dir / "optimizer.npz"
        model.save_weights(str(weights))
        state_names = save_optimizer_state(state, optimizer, mx, mlx_utils)
        reloaded = build_model(local_model_config, mx=mx, nn=nn)
        reloaded.load_weights(str(weights))
        resumed = build_candidate_optimizer(
            candidate_id,
            profile,
            config,
            mx=mx,
            optim=optim,
            local_model_config=local_model_config,
        )
        load_optimizer_state(state, state_names, resumed, mx, mlx_utils)
        bind_loaded_optimizer_state(resumed, reloaded.trainable_parameters())
        if parameter_digest(model, mlx_utils) != parameter_digest(reloaded, mlx_utils):
            raise OptimizerAdequacyFault(
                f"checkpoint_parameter_mismatch:{candidate_id}:{profile['id']}:{seed}"
            )
        if tree_array_digest(optimizer.state, mlx_utils) != tree_array_digest(
            resumed.state, mlx_utils
        ):
            raise OptimizerAdequacyFault(
                f"checkpoint_optimizer_state_mismatch:{candidate_id}:{profile['id']}:{seed}"
            )
        batch = batches[-1]
        x_np, y_np, mask_np = corpus.make_batch(batch, maximum)
        x, y, mask = mx.array(x_np), mx.array(y_np), mx.array(mask_np)
        left_loss, left_gradients = value_and_grad(model, x, y, mask)
        right_value_and_grad = nn.value_and_grad(reloaded, objective)
        right_loss, right_gradients = right_value_and_grad(reloaded, x, y, mask)
        gradient_error = tree_max_absolute_difference(
            left_gradients, right_gradients, mlx_utils
        )
        tolerance = float(
            config["prospective_decision"]["maximum_next_update_absolute_error"]
        )
        if gradient_error > tolerance:
            raise OptimizerAdequacyFault(
                f"checkpoint_gradient_mismatch:{candidate_id}:{profile['id']}:{seed}:"
                f"max_abs={gradient_error}"
            )
        left_gradients, left_gradient_norm = optim.clip_grad_norm(
            left_gradients, float(config["training"]["gradient_clip_norm"])
        )
        right_gradients, right_gradient_norm = optim.clip_grad_norm(
            right_gradients, float(config["training"]["gradient_clip_norm"])
        )
        mx.eval(left_gradient_norm, right_gradient_norm)
        optimizer.update(model, left_gradients)
        resumed.update(reloaded, right_gradients)
        mx.eval(model.parameters(), reloaded.parameters(), left_loss, right_loss)
        next_update_error = tree_max_absolute_difference(
            model.parameters(), reloaded.parameters(), mlx_utils
        )
        if next_update_error > tolerance:
            raise OptimizerAdequacyFault(
                f"checkpoint_next_update_mismatch:{candidate_id}:{profile['id']}:{seed}"
            )
        result.update(
            {
                "checkpoint": relative(weights),
                "checkpoint_sha256": sha256_file(weights),
                "optimizer_checkpoint": relative(state),
                "optimizer_checkpoint_sha256": sha256_file(state),
                "checkpoint_reload_exact_state": True,
                "checkpoint_next_update_numerically_equivalent": True,
                "checkpoint_next_update_max_absolute_error": next_update_error,
                "checkpoint_gradient_max_absolute_error": gradient_error,
            }
        )
    return result


def select_tuned_profiles(tune_runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in tune_runs:
        grouped.setdefault((row["candidate_id"], row["profile"]["id"]), []).append(row)
    selected: dict[str, dict[str, Any]] = {}
    for (candidate_id, profile_id), rows in grouped.items():
        score = statistics.fmean(float(row["final_heldout"]["ntp_loss"]) for row in rows)
        candidate = {
            "candidate_id": candidate_id,
            "profile": rows[0]["profile"],
            "mean_tune_ntp_loss": score,
            "seed_count": len(rows),
        }
        prior = selected.get(candidate_id)
        if prior is None or (score, profile_id) < (prior["mean_tune_ntp_loss"], prior["profile"]["id"]):
            selected[candidate_id] = candidate
    return selected


def compare_final(final_runs: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in final_runs:
        by_candidate.setdefault(row["candidate_id"], []).append(row)
    reference_id = config["prospective_decision"]["reference_optimizer"]
    reference = {row["seed"]: row for row in by_candidate[reference_id]}
    comparisons: dict[str, Any] = {}
    eligible: list[tuple[float, str]] = []
    for candidate_id, rows in by_candidate.items():
        if candidate_id == reference_id:
            continue
        paired = []
        for row in rows:
            control = reference[row["seed"]]
            control_loss = float(control["final_heldout"]["ntp_loss"])
            candidate_loss = float(row["final_heldout"]["ntp_loss"])

            def first_quality_step(run: dict[str, Any]) -> int | None:
                for point in run.get("learning_curve") or []:
                    if (
                        float(point["heldout"]["ntp_loss"])
                        <= control_loss
                    ):
                        return int(point["step"])
                return None

            control_quality_step = first_quality_step(control)
            candidate_quality_step = first_quality_step(row)
            control_step_seconds = float(control["training_wall_seconds"]) / max(
                int(control.get("optimizer_steps") or 1), 1
            )
            candidate_step_seconds = float(row["training_wall_seconds"]) / max(
                int(row.get("optimizer_steps") or 1), 1
            )
            time_to_quality_ratio = (
                candidate_step_seconds * candidate_quality_step
                / max(
                    control_step_seconds * int(control_quality_step or 0),
                    1e-12,
                )
                if candidate_quality_step is not None
                and control_quality_step is not None
                else None
            )
            arm_regressions = {
                arm: (
                    float(row["final_heldout"]["by_arm"][arm]["ntp_loss"])
                    - float(control["final_heldout"]["by_arm"][arm]["ntp_loss"])
                ) / max(float(control["final_heldout"]["by_arm"][arm]["ntp_loss"]), 1e-12)
                for arm in config["scoped_arms"]
            }
            paired.append(
                {
                    "seed": row["seed"],
                    "relative_loss_improvement": (control_loss - candidate_loss) / max(control_loss, 1e-12),
                    "maximum_weak_arm_relative_loss_regression": max(arm_regressions.values()),
                    "arm_relative_loss_regressions": arm_regressions,
                    "wall_time_ratio": float(row["training_wall_seconds"]) / max(float(control["training_wall_seconds"]), 1e-12),
                    "optimizer_state_ratio": float(row["optimizer_state_bytes"]) / max(float(control["optimizer_state_bytes"]), 1.0),
                    "reference_quality_threshold": control_loss,
                    "control_first_quality_step": control_quality_step,
                    "candidate_first_quality_step": candidate_quality_step,
                    "time_to_reference_quality_ratio": time_to_quality_ratio,
                }
            )
        decision = config["prospective_decision"]
        mean_gain = statistics.fmean(row["relative_loss_improvement"] for row in paired)
        gates = {
            "loss_improvement": mean_gain >= float(decision["minimum_relative_heldout_loss_improvement"]),
            "weak_arm": max(row["maximum_weak_arm_relative_loss_regression"] for row in paired) <= float(decision["maximum_weak_arm_relative_loss_regression"]),
            "seed_wins": sum(row["relative_loss_improvement"] > 0.0 for row in paired) / len(paired) >= float(decision["minimum_seed_win_fraction"]),
            "wall_time": statistics.fmean(row["wall_time_ratio"] for row in paired) <= float(decision["maximum_wall_time_ratio"]),
            "optimizer_state": statistics.fmean(row["optimizer_state_ratio"] for row in paired) <= float(decision["maximum_optimizer_state_ratio"]),
        }
        joined_limit = decision.get(
            "maximum_mean_time_to_reference_quality_ratio"
        )
        if joined_limit is not None:
            joined = [
                row["time_to_reference_quality_ratio"] for row in paired
            ]
            gates["time_to_reference_quality"] = all(
                value is not None for value in joined
            ) and statistics.fmean(float(value) for value in joined) <= float(
                joined_limit
            )
        adopted = all(gates.values())
        comparisons[candidate_id] = {
            "control_id": reference_id,
            "paired_runs": paired,
            "mean_relative_loss_improvement": mean_gain,
            "mean_time_to_reference_quality_ratio": (
                statistics.fmean(
                    float(row["time_to_reference_quality_ratio"])
                    for row in paired
                )
                if all(
                    row["time_to_reference_quality_ratio"] is not None
                    for row in paired
                )
                else None
            ),
            "gates": gates,
            "disposition": "ADOPTED" if adopted else "NOT_SELECTED_FIRST_CAMPAIGN",
            "scientific_falsification_claimed": False,
        }
        if adopted:
            eligible.append((mean_gain, candidate_id))
    selected = max(eligible)[1] if eligible else reference_id
    disposition = {
        "selected_optimizer": selected,
        "kind": "CHALLENGER_ADOPTED" if eligible else "REFERENCE_RETAINED_NO_CHALLENGER_PARETO_GAIN",
        "scientific_falsification_claimed": False,
        "reentry_condition": (
            None if eligible else "a prospectively matched larger rung with direct functional progress and the same source-disjoint custody"
        ),
    }
    return comparisons, disposition


def assess_width_transfer(
    runs: list[dict[str, Any]], config: dict[str, Any], selected_optimizer: str
) -> dict[str, Any]:
    contract = config["width_transfer"]
    improvements = [
        (
            float(row["initial_heldout"]["ntp_loss"])
            - float(row["final_heldout"]["ntp_loss"])
        )
        / max(float(row["initial_heldout"]["ntp_loss"]), 1e-12)
        for row in runs
    ]
    seed_progress_fraction = (
        sum(value > 0.0 for value in improvements) / len(improvements)
        if improvements
        else 0.0
    )
    gates = {
        "selected_recipe_only": bool(runs)
        and {row["candidate_id"] for row in runs} == {selected_optimizer},
        "target_width_exact": bool(runs)
        and {
            int(row["model_config"]["d_model"])
            for row in runs
        }
        == {int(contract["target_width"])},
        "all_preregistered_seeds": {
            int(row["seed"]) for row in runs
        }
        == {int(seed) for seed in contract["seeds"]},
        "loss_progress": seed_progress_fraction
        >= float(contract["minimum_seed_progress_fraction"])
        and statistics.fmean(improvements)
        >= -float(contract["maximum_mean_loss_regression_from_initial"]),
        "gradient_flow": bool(runs)
        and all(float(row["first_gradient_l1"]) > 0.0 for row in runs),
        "checkpoint_exact_state": bool(runs)
        and all(row["checkpoint_reload_exact_state"] is True for row in runs),
        "checkpoint_next_update_numerically_equivalent": bool(runs)
        and all(
            row["checkpoint_next_update_numerically_equivalent"] is True
            and float(row["checkpoint_next_update_max_absolute_error"])
            <= float(
                config["prospective_decision"][
                    "maximum_next_update_absolute_error"
                ]
            )
            for row in runs
        ),
        "confirmation_surface_unconsumed": int(
            contract["confirmation_surface_consumption"]
        )
        == 0,
    }
    target_path = resolve(contract["target_architecture_config"])
    return {
        "policy": contract["policy"],
        "trigger_state": "GREEN" if all(gates.values()) else "RED",
        "selected_optimizer": selected_optimizer,
        "source_width": int(contract["source_width"]),
        "target_width": int(contract["target_width"]),
        "isolate_width_only": bool(contract["isolate_width_only"]),
        "target_architecture_config": relative(target_path),
        "target_architecture_config_sha256": sha256_file(target_path),
        "seed_relative_initial_to_final_loss_improvements": improvements,
        "seed_progress_fraction": seed_progress_fraction,
        "mean_relative_initial_to_final_loss_improvement": (
            statistics.fmean(improvements) if improvements else None
        ),
        "gates": gates,
        "disposition": (
            "SELECTED_RECIPE_TRANSFERS_TO_TARGET_WIDTH"
            if all(gates.values())
            else "TRANSFER_FAILED_RETUNE_OR_ISOLATE_PARAMETERIZATION_CHALLENGER"
        ),
        "non_claim": (
            "target-width private loss progress is a recipe-transfer check, "
            "not full-depth capability or long-campaign evidence"
        ),
    }


def execute(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    metadata = json.loads(
        resolve(config["stage_metadata"]).read_text(encoding="utf-8")
    )
    base = json.loads(resolve(config["base_config"]).read_text(encoding="utf-8"))
    vocabulary = model_vocab_size(
        base, metadata["source_vocab"], metadata["target_vocab"]
    )
    parameter_count = parameter_count_gate(config, vocabulary)
    if not parameter_count["passed"]:
        raise OptimizerAdequacyFault("model_parameter_count_out_of_range")
    all_train = corpus.load_governed_rows(config, split="train")
    heldout = corpus.load_governed_rows(config, split="heldout")
    tune_count = int(config["supervision"]["tune_rows_per_arm"])
    tune_train, tune = split_tuning_rows(
        all_train,
        tune_rows_per_arm=tune_count,
        namespace=config["supervision"]["selection_namespace"],
    )
    train_ids, tune_ids, heldout_ids = source_sets(tune_train), source_sets(tune), source_sets(heldout)
    if train_ids & tune_ids or train_ids & heldout_ids or tune_ids & heldout_ids:
        raise OptimizerAdequacyFault("source_disjointness_failed")
    scratch = resolve(config["scratch_root"])
    journal_path, journal_rows, resumed_from_journal = prepare_run_journal(
        config_path, scratch
    )
    journal_index = {
        (
            str(row["stage"]),
            str(row["candidate_id"]),
            str(row["profile"]["id"]),
            int(row["seed"]),
        ): row
        for row in journal_rows
    }
    lease = pretraining_candidate_canary.candidate_lease(
        candidate_id=config["candidate_lease_id"],
        max_steps=int(config["training"]["final_steps"]),
        scratch_checkpoint_root=scratch,
        targets=["shared_trunk"],
        phase="pretraining",
        resume=False,
    )
    if not lease["authorized"]:
        raise OptimizerAdequacyFault("candidate_lease_denied:" + ",".join(lease["faults"]))
    monitor = pretraining_candidate_canary.CandidateCanaryMonitor(lease)
    tune_runs = []
    for candidate in config["candidates"]:
        for profile in candidate["profiles"]:
            for seed in config["seeds"]:
                key = (
                    "tune",
                    candidate["id"],
                    profile["id"],
                    int(seed),
                )
                row = journal_index.get(key)
                if row is None:
                    row = run_profile(
                        config,
                        candidate_id=candidate["id"],
                        profile=profile,
                        seed=int(seed),
                        train_rows=tune_train,
                        eval_rows=tune,
                        steps=int(config["training"]["tune_steps"]),
                        stage="tune",
                        monitor=monitor,
                        retain_checkpoint=False,
                    )
                    append_run_journal(journal_path, row)
                    journal_index[key] = row
                tune_runs.append(row)
    selected_profiles = select_tuned_profiles(tune_runs)
    final_runs = []
    for candidate in config["candidates"]:
        profile = selected_profiles[candidate["id"]]["profile"]
        for seed in config["seeds"]:
            key = (
                "final",
                candidate["id"],
                profile["id"],
                int(seed),
            )
            row = journal_index.get(key)
            if row is None:
                row = run_profile(
                    config,
                    candidate_id=candidate["id"],
                    profile=profile,
                    seed=int(seed),
                    train_rows=all_train,
                    eval_rows=heldout,
                    steps=int(config["training"]["final_steps"]),
                    stage="final",
                    monitor=monitor,
                    retain_checkpoint=True,
                )
                append_run_journal(journal_path, row)
                journal_index[key] = row
            final_runs.append(row)
    for seed in config["seeds"]:
        identities = {
            row["initial_parameter_sha256"]
            for row in final_runs
            if row["seed"] == seed
        }
        if len(identities) != 1:
            raise OptimizerAdequacyFault(f"matched_initialization_failed:{seed}")
    comparisons, disposition = compare_final(final_runs, config)
    selected_optimizer = disposition["selected_optimizer"]
    selected_profile = selected_profiles[selected_optimizer]["profile"]
    width_transfer_runs = []
    for seed in config["width_transfer"]["seeds"]:
        key = (
            "width_transfer",
            selected_optimizer,
            selected_profile["id"],
            int(seed),
        )
        row = journal_index.get(key)
        if row is None:
            row = run_profile(
                config,
                candidate_id=selected_optimizer,
                profile=selected_profile,
                seed=int(seed),
                train_rows=all_train,
                eval_rows=heldout,
                steps=int(config["width_transfer"]["steps"]),
                stage="width_transfer",
                monitor=monitor,
                retain_checkpoint=True,
                model_override=config["width_transfer"]["model"],
                checkpoint_namespace="width_transfer",
            )
            append_run_journal(journal_path, row)
            journal_index[key] = row
        width_transfer_runs.append(row)
    width_transfer = assess_width_transfer(
        width_transfer_runs, config, selected_optimizer
    )
    cards = optimizer_policy_cards(
        config, selected_profiles=selected_profiles
    )
    card_faults = validate_optimizer_policy_cards(cards, config)
    resource = monitor.finalize(
        tune_runs + final_runs + width_transfer_runs
    )
    gates = {
        "source_disjoint": not (train_ids & tune_ids or train_ids & heldout_ids or tune_ids & heldout_ids),
        "matched_tuning_opportunity": len({len(row["profiles"]) for row in config["candidates"]}) == 1,
        "matched_initialization": True,
        "gradient_flow": all(row["first_gradient_l1"] > 0.0 for row in tune_runs + final_runs),
        "checkpoint_exact_state": all(row["checkpoint_reload_exact_state"] is True for row in final_runs),
        "checkpoint_next_update_numerically_equivalent": all(
            row["checkpoint_next_update_numerically_equivalent"] is True
            and float(row["checkpoint_next_update_max_absolute_error"])
            <= float(config["prospective_decision"]["maximum_next_update_absolute_error"])
            for row in final_runs
        ),
        "optimizer_policy_cards": not card_faults,
        "selected_recipe_target_width_transfer": width_transfer[
            "trigger_state"
        ]
        == "GREEN",
        "resource_bounds": resource["passed"],
        "model_parameter_count": parameter_count["passed"],
        "no_public_external_or_fallback": all(
            row["public_training_rows"] == 0
            and row["public_evaluation_rows"] == 0
            and row["external_inference_calls"] == 0
            and row["fallback_or_template_credit"] == 0
            for row in tune_runs + final_runs + width_transfer_runs
        ),
    }
    report = {
        "policy": POLICY,
        "trigger_state": "GREEN" if all(gates.values()) else "RED",
        "support_state": "private-source-disjoint-matched-optimizer-experiment",
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "source_disjointness": {
            "train": len(train_ids),
            "tune": len(tune_ids),
            "heldout": len(heldout_ids),
            "overlap_count": len((train_ids & tune_ids) | (train_ids & heldout_ids) | (tune_ids & heldout_ids)),
        },
        "candidate_lease": lease,
        "run_journal": {
            "path": relative(journal_path),
            "resumed": resumed_from_journal,
            "reused_run_count": len(journal_rows),
            "completed_run_count": len(
                tune_runs + final_runs + width_transfer_runs
            ),
        },
        "model_parameter_count": parameter_count,
        "resource_receipt": resource,
        "selected_profiles": selected_profiles,
        "optimizer_policy_cards": cards,
        "optimizer_policy_card_faults": card_faults,
        "tune_runs": tune_runs,
        "final_runs": final_runs,
        "width_transfer_runs": width_transfer_runs,
        "width_transfer": width_transfer,
        "comparisons": comparisons,
        "campaign_disposition": disposition,
        "gates": gates,
        "non_claims": [
            "private language-model loss is not direct assistant utility",
            "a first-campaign scope decision is not scientific falsification",
            "this run does not consume the frozen functional confirmation surface or public calibration",
        ],
    }
    output = resolve(config["report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def preflight(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    train = corpus.load_governed_rows(config, split="train")
    heldout = corpus.load_governed_rows(config, split="heldout")
    tune_train, tune = split_tuning_rows(
        train,
        tune_rows_per_arm=int(config["supervision"]["tune_rows_per_arm"]),
        namespace=config["supervision"]["selection_namespace"],
    )
    ids = [source_sets(rows) for rows in (tune_train, tune, heldout)]
    overlap = len((ids[0] & ids[1]) | (ids[0] & ids[2]) | (ids[1] & ids[2]))
    cards = optimizer_policy_cards(config)
    card_faults = validate_optimizer_policy_cards(cards, config)
    metadata = json.loads(
        resolve(config["stage_metadata"]).read_text(encoding="utf-8")
    )
    base = json.loads(resolve(config["base_config"]).read_text(encoding="utf-8"))
    vocabulary = model_vocab_size(
        base, metadata["source_vocab"], metadata["target_vocab"]
    )
    parameter_count = parameter_count_gate(config, vocabulary)
    return {
        "policy": POLICY,
        "trigger_state": (
            "GREEN"
            if overlap == 0 and not card_faults and parameter_count["passed"]
            else "RED"
        ),
        "candidate_count": len(config["candidates"]),
        "profile_count_per_candidate": len(config["candidates"][0]["profiles"]),
        "seed_count": len(config["seeds"]),
        "source_overlap_count": overlap,
        "optimizer_policy_cards": cards,
        "optimizer_policy_card_faults": card_faults,
        "model_parameter_count": parameter_count,
        "width_transfer": {
            "source_width": config["width_transfer"]["source_width"],
            "target_width": config["width_transfer"]["target_width"],
            "seeds": config["width_transfer"]["seeds"],
            "steps": config["width_transfer"]["steps"],
            "execution_required": True,
        },
        "execution_required": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    path = resolve(args.config)
    report = execute(path) if args.execute else preflight(path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
