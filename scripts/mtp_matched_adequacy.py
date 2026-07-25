#!/usr/bin/env python3
"""Source-disjoint matched AR-versus-MTP adequacy experiment.

The AR controls retain and train the same future heads as their MTP pair, but
stop the auxiliary gradient at the shared hidden state. This preserves the
topology, head compute, data order, and optimizer exposure while isolating the
effect of future-token shaping on the causal trunk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

import pretraining_candidate_canary
from moecot_language_tokenizer import exact_text_tokens
from neural_seed_open_vocab import encode_tokens
from standard_causal_transformer_model import CausalTransformerConfig, build_model
from standard_causal_transformer_objectives import mtp_auxiliary_loss, mtp_curriculum_scale
from standard_causal_transformer_survival import (
    GLOBAL_BOS_ID,
    SOURCE_TARGET_SEPARATOR_ID,
    model_vocab_size,
    source_token_offset,
    target_token_offset,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "mtp_matched_adequacy.json"
POLICY = "project_theseus_mtp_matched_adequacy_v1"


class MtpAdequacyFault(ValueError):
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


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("policy") != POLICY:
        raise MtpAdequacyFault("config_policy_invalid")
    arms = tuple(config.get("scoped_arms") or ())
    if arms != ("english", "python", "javascript_typescript", "html_css", "rust"):
        raise MtpAdequacyFault("scoped_arm_contract_invalid")
    candidates = config.get("candidates") or []
    expected = {
        "ar_independent_sham",
        "mtp_conventional_independent",
        "mtp_curriculum_independent",
        "ar_register_sham",
        "mtp_register_curriculum",
    }
    if {row.get("id") for row in candidates} != expected:
        raise MtpAdequacyFault("candidate_inventory_invalid")
    for row in candidates:
        if row.get("head_mode") not in {"independent_mlp", "register_conditioned"}:
            raise MtpAdequacyFault(f"candidate_head_mode_invalid:{row.get('id')}")
        if row.get("gradient_route") not in {
            "future_heads_only_stop_gradient_at_trunk",
            "future_heads_and_shared_trunk",
        }:
            raise MtpAdequacyFault(f"candidate_gradient_route_invalid:{row.get('id')}")
        if row.get("schedule") not in {"fixed", "curriculum"}:
            raise MtpAdequacyFault(f"candidate_schedule_invalid:{row.get('id')}")
    boundaries = config.get("hard_boundaries") or {}
    zero_fields = (
        "public_training_rows",
        "public_evaluation_rows",
        "external_inference_calls",
        "fallback_or_template_credit",
        "confirmation_surface_consumption",
    )
    if any(boundaries.get(field) != 0 for field in zero_fields):
        raise MtpAdequacyFault("hard_boundary_nonzero")
    if boundaries.get("production_checkpoint_mutation") is not False:
        raise MtpAdequacyFault("production_mutation_boundary_invalid")
    if boundaries.get("heldout_labels_visible_to_optimizer") is not False:
        raise MtpAdequacyFault("heldout_visibility_boundary_invalid")
    if boundaries.get("architecture_selection_from_auxiliary_loss_alone") is not False:
        raise MtpAdequacyFault("selection_boundary_invalid")
    if int(config["training"]["steps"]) > 96:
        raise MtpAdequacyFault("candidate_step_budget_exceeded")
    return config


def _rank(row: dict[str, Any], namespace: str) -> str:
    identity = str(row.get("source_identity") or "")
    return hashlib.sha256(f"{namespace}:{identity}".encode()).hexdigest()


def load_governed_rows(
    config: dict[str, Any],
    *,
    split: str,
) -> dict[str, list[dict[str, Any]]]:
    supervision = config["supervision"]
    root = resolve(supervision["root"])
    expected_split = (
        supervision["train_split"]
        if split == "train"
        else supervision["heldout_split"]
    )
    limit = int(
        supervision["train_rows_per_arm"]
        if split == "train"
        else supervision["heldout_rows_per_arm"]
    )
    maximum = int(supervision["maximum_sequence_tokens"])
    metadata = json.loads(resolve(config["stage_metadata"]).read_text(encoding="utf-8"))
    base = json.loads(resolve(config["base_config"]).read_text(encoding="utf-8"))
    source_vocab = metadata["source_vocab"]
    target_vocab = metadata["target_vocab"]
    source_offset = source_token_offset(base, source_vocab)
    target_offset = target_token_offset(base, source_vocab)
    selected: dict[str, list[dict[str, Any]]] = {}
    for arm in config["scoped_arms"]:
        path = root / expected_split / f"{arm}.jsonl"
        eligible: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                if (
                    row.get("split") != expected_split
                    or row.get("arm_id") != arm
                    or row.get("public_benchmark") is not False
                    or row.get("public_benchmark_payload") is not False
                    or int(row.get("external_inference_calls") or 0) != 0
                    or int(row.get("fallback_return_count") or 0) != 0
                ):
                    raise MtpAdequacyFault(
                        f"governed_row_boundary_invalid:{arm}:{expected_split}:{line_number}"
                    )
                if not row.get("license_spdx") or not row.get("source_identity"):
                    raise MtpAdequacyFault(
                        f"governed_row_provenance_missing:{arm}:{expected_split}:{line_number}"
                    )
                source_ids, source_receipt = encode_tokens(
                    exact_text_tokens(str(row["prompt"])), source_vocab, stream="source"
                )
                target_ids, target_receipt = encode_tokens(
                    exact_text_tokens(str(row["target"])), target_vocab, stream="target"
                )
                if source_receipt["unknown_token_count"] or target_receipt["unknown_token_count"]:
                    raise MtpAdequacyFault(
                        f"canonical_tokenization_unknown:{arm}:{expected_split}:{line_number}"
                    )
                sequence = [GLOBAL_BOS_ID]
                sequence.extend(source_offset + int(value) for value in source_ids)
                sequence.append(SOURCE_TARGET_SEPARATOR_ID)
                sequence.append(target_offset + int(target_vocab["<bos>"]))
                target_start = len(sequence)
                sequence.extend(target_offset + int(value) for value in target_ids)
                sequence.append(target_offset + int(target_vocab["<eos>"]))
                if len(sequence) - 1 > maximum:
                    continue
                eligible.append(
                    {
                        "arm_id": arm,
                        "row_id": str(row["row_id"]),
                        "source_identity": str(row["source_identity"]),
                        "dataset_id": str(row["dataset_id"]),
                        "license_spdx": str(row["license_spdx"]),
                        "sequence": sequence,
                        "target_mask_start": target_start - 1,
                        "target_byte_count": len(str(row["target"]).encode("utf-8")),
                    }
                )
        namespace = f"{supervision['selection_namespace']}:{expected_split}:{arm}"
        eligible.sort(key=lambda row: _rank(row, namespace))
        if len(eligible) < limit:
            raise MtpAdequacyFault(
                f"source_disjoint_rows_insufficient:{arm}:{expected_split}:{len(eligible)}<{limit}"
            )
        selected[arm] = eligible[:limit]
    return selected


def source_disjoint_receipt(
    train: dict[str, list[dict[str, Any]]],
    heldout: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    train_ids = {row["source_identity"] for rows in train.values() for row in rows}
    heldout_ids = {row["source_identity"] for rows in heldout.values() for row in rows}
    overlap = sorted(train_ids & heldout_ids)
    by_arm = {}
    for arm in train:
        train_arm = {row["source_identity"] for row in train[arm]}
        heldout_arm = {row["source_identity"] for row in heldout[arm]}
        by_arm[arm] = {
            "train_rows": len(train[arm]),
            "heldout_rows": len(heldout[arm]),
            "source_identity_overlap": len(train_arm & heldout_arm),
        }
    return {
        "policy": "project_theseus_source_disjoint_mtp_split_v1",
        "train_source_count": len(train_ids),
        "heldout_source_count": len(heldout_ids),
        "cross_split_overlap_count": len(overlap),
        "cross_split_overlap_sample": overlap[:10],
        "by_arm": by_arm,
        "passed": not overlap,
        "selection_digest": digest(
            {
                "train": sorted(train_ids),
                "heldout": sorted(heldout_ids),
            }
        ),
    }


def make_batch(
    rows: list[dict[str, Any]], maximum: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.zeros((len(rows), maximum), dtype=np.int32)
    y = np.zeros((len(rows), maximum), dtype=np.int32)
    mask = np.zeros((len(rows), maximum), dtype=np.float32)
    for index, row in enumerate(rows):
        sequence = np.asarray(row["sequence"], dtype=np.int32)
        length = len(sequence) - 1
        x[index, :length] = sequence[:-1]
        y[index, :length] = sequence[1:]
        mask[index, int(row["target_mask_start"]):length] = 1.0
    return x, y, mask


def balanced_batches(
    rows_by_arm: dict[str, list[dict[str, Any]]],
    *,
    steps: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    orders = {arm: rng.permutation(len(rows)).tolist() for arm, rows in rows_by_arm.items()}
    batches = []
    for step in range(steps):
        batch = []
        for arm, rows in rows_by_arm.items():
            order = orders[arm]
            batch.append(rows[order[step % len(order)]])
        batches.append(batch)
    return batches


def candidate_scale(candidate: dict[str, Any], step: int, training: dict[str, Any]) -> float:
    maximum = float(training["maximum_mtp_loss_scale"])
    if candidate["schedule"] == "fixed":
        return maximum
    return mtp_curriculum_scale(
        step,
        warmup_steps=int(training["warmup_steps"]),
        ramp_steps=int(training["ramp_steps"]),
        maximum=maximum,
    )


def paired_controls(config: dict[str, Any]) -> dict[str, str]:
    return {
        "mtp_conventional_independent": "ar_independent_sham",
        "mtp_curriculum_independent": "ar_independent_sham",
        "mtp_register_curriculum": "ar_register_sham",
    }


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def summarize_pair(
    candidate_runs: list[dict[str, Any]],
    control_runs: list[dict[str, Any]],
    decision: dict[str, Any],
) -> dict[str, Any]:
    controls = {int(row["seed"]): row for row in control_runs}
    paired = []
    for row in candidate_runs:
        control = controls[int(row["seed"])]
        candidate_loss = float(row["final_heldout"]["ntp_loss"])
        control_loss = float(control["final_heldout"]["ntp_loss"])
        relative_loss = (control_loss - candidate_loss) / max(control_loss, 1e-12)
        candidate_accuracy = float(row["final_heldout"]["greedy_token_accuracy"])
        control_accuracy = float(control["final_heldout"]["greedy_token_accuracy"])
        arm_regressions = {}
        for arm, candidate_arm in row["final_heldout"]["by_arm"].items():
            candidate_arm_loss = float(candidate_arm["ntp_loss"])
            control_arm_loss = float(control["final_heldout"]["by_arm"][arm]["ntp_loss"])
            arm_regressions[arm] = (
                candidate_arm_loss - control_arm_loss
            ) / max(control_arm_loss, 1e-12)
        paired.append(
            {
                "seed": int(row["seed"]),
                "relative_ntp_loss_improvement": relative_loss,
                "greedy_token_accuracy_delta": candidate_accuracy - control_accuracy,
                "maximum_arm_relative_loss_regression": max(arm_regressions.values()),
                "arm_relative_loss_regressions": arm_regressions,
                "wall_time_ratio": float(row["training_wall_seconds"])
                / max(float(control["training_wall_seconds"]), 1e-12),
            }
        )
    loss_improvements = [row["relative_ntp_loss_improvement"] for row in paired]
    accuracy_deltas = [row["greedy_token_accuracy_delta"] for row in paired]
    weak_regressions = [row["maximum_arm_relative_loss_regression"] for row in paired]
    wall_ratios = [row["wall_time_ratio"] for row in paired]
    seed_win_fraction = sum(value > 0.0 for value in loss_improvements) / max(1, len(paired))
    statistical_power_adequate = len(paired) >= 3
    functional_verifier_available = all(
        bool(row["final_heldout"].get("functional_verifier_available"))
        for row in candidate_runs
    )
    gates = {
        "mean_loss_improvement": _mean(loss_improvements)
        >= float(decision["minimum_relative_heldout_ntp_loss_improvement"]),
        "weak_arm_regression": max(weak_regressions)
        <= float(decision["maximum_weak_arm_relative_loss_regression"]),
        "seed_win_fraction": seed_win_fraction
        >= float(decision["minimum_seed_win_fraction"]),
        "greedy_token_accuracy": _mean(accuracy_deltas)
        >= float(decision["minimum_relative_greedy_token_accuracy_improvement"]),
        "wall_time": _mean(wall_ratios) <= float(decision["maximum_wall_time_ratio"]),
        "statistical_power": statistical_power_adequate,
        "functional_verifier": functional_verifier_available
        or not bool(decision["functional_verifier_required_for_adoption"]),
    }
    adopted = all(gates.values())
    return {
        "paired_runs": paired,
        "mean_relative_ntp_loss_improvement": _mean(loss_improvements),
        "mean_greedy_token_accuracy_delta": _mean(accuracy_deltas),
        "maximum_weak_arm_relative_loss_regression": max(weak_regressions),
        "mean_wall_time_ratio": _mean(wall_ratios),
        "seed_win_fraction": seed_win_fraction,
        "gates": gates,
        "disposition": "ADOPTED" if adopted else "INCONCLUSIVE_EXPERIMENT",
        "reason": (
            "prospective quality, weak-tail, seed, cost, and functional gates passed"
            if adopted
            else "bounded private evidence does not satisfy every prospective adoption gate"
        ),
    }


def campaign_disposition(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Choose first-campaign scope without converting a bounded loss into falsification."""

    if any(row["disposition"] == "ADOPTED" for row in comparisons.values()):
        return {
            "kind": "INCLUDE_IN_T0A_BAKEOFF",
            "scientific_falsification_claimed": False,
            "reason": "at least one candidate cleared every prospective adoption gate",
            "reentry_condition": None,
        }
    no_loss_signal = all(
        float(row["mean_relative_ntp_loss_improvement"]) <= 0.0
        for row in comparisons.values()
    )
    functional_missing = all(
        not bool(row["gates"]["functional_verifier"])
        for row in comparisons.values()
    )
    if no_loss_signal and functional_missing:
        return {
            "kind": "SCOPED_DISCOVERY_EXCLUDED_FROM_FIRST_CAMPAIGN",
            "scientific_falsification_claimed": False,
            "reason": (
                "all adequate MTP candidates missed the preregistered bounded loss floor and "
                "the admitted corpus cannot measure functional utility; optimized AR therefore "
                "receives the first-campaign budget"
            ),
            "reentry_condition": (
                "a source-disjoint verifier-bearing private workload, frozen before execution, "
                "with accepted target spans and matched total-system cost, or a materially larger "
                "data/scale rung justified independently of this result"
            ),
        }
    return {
        "kind": "INCONCLUSIVE_REQUIRES_FOLLOWUP_BEFORE_CAMPAIGN_SCOPE",
        "scientific_falsification_claimed": False,
        "reason": "the bounded evidence is mixed and cannot support either adoption or exclusion",
        "reentry_condition": "repair the exact missing prospective gate without consuming confirmation data",
    }


def _model_config(
    config: dict[str, Any], candidate: dict[str, Any], vocab_size: int
) -> CausalTransformerConfig:
    model = config["model"]
    register = len(model["future_offsets"]) if candidate["head_mode"] == "register_conditioned" else 0
    return CausalTransformerConfig(
        vocab_size=vocab_size,
        d_model=int(model["d_model"]),
        num_layers=int(model["num_layers"]),
        num_heads=int(model["num_heads"]),
        num_kv_heads=int(model["num_kv_heads"]),
        ff_dim=int(model["ff_dim"]),
        attention_policy=str(model["attention_policy"]),
        source_target_separator_token_id=SOURCE_TARGET_SEPARATOR_ID,
        mtp_future_offsets=tuple(int(value) for value in model["future_offsets"]),
        mtp_head_mode=str(candidate["head_mode"]),
        mtp_hidden_dim=int(model["hidden_dim"]),
        mtp_register_count=register,
        mtp_loss_weights=tuple(float(value) for value in model["loss_weights"]),
        mtp_loss_scale=float(config["training"]["maximum_mtp_loss_scale"]),
        mtp_maximum_head_parameter_overhead_ratio=float(
            model["maximum_parameter_overhead_ratio"]
        ),
    )


def _evaluate(model: Any, rows_by_arm: dict[str, list[dict[str, Any]]], maximum: int, mx: Any, nn: Any) -> dict[str, Any]:
    by_arm = {}
    total_loss_mass = 0.0
    total_tokens = 0.0
    total_correct = 0
    total_rows = 0
    exact_rows = 0
    total_initial_run = 0
    for arm, rows in rows_by_arm.items():
        x_np, y_np, mask_np = make_batch(rows, maximum)
        x, y, mask = mx.array(x_np), mx.array(y_np), mx.array(mask_np)
        logits, _cache = model(x)
        losses = nn.losses.cross_entropy(logits, y)
        predictions = mx.argmax(logits, axis=-1)
        mx.eval(losses, predictions)
        losses_np = np.asarray(losses)
        predictions_np = np.asarray(predictions)
        token_count = int(mask_np.sum())
        correct = int(((predictions_np == y_np) * mask_np).sum())
        row_exact = 0
        initial_runs = []
        for row_index in range(len(rows)):
            positions = np.flatnonzero(mask_np[row_index] > 0)
            matches = predictions_np[row_index, positions] == y_np[row_index, positions]
            row_exact += int(bool(len(matches)) and bool(np.all(matches)))
            run = 0
            for value in matches:
                if not value:
                    break
                run += 1
            initial_runs.append(run)
        loss_mass = float((losses_np * mask_np).sum())
        by_arm[arm] = {
            "row_count": len(rows),
            "target_token_count": token_count,
            "ntp_loss": loss_mass / max(1, token_count),
            "greedy_token_accuracy": correct / max(1, token_count),
            "exact_teacher_forced_target_rows": row_exact,
            "mean_initial_correct_token_run": _mean(initial_runs),
            "target_byte_count": sum(int(row["target_byte_count"]) for row in rows),
        }
        total_loss_mass += loss_mass
        total_tokens += token_count
        total_correct += correct
        total_rows += len(rows)
        exact_rows += row_exact
        total_initial_run += sum(initial_runs)
    return {
        "ntp_loss": total_loss_mass / max(1.0, total_tokens),
        "greedy_token_accuracy": total_correct / max(1.0, total_tokens),
        "exact_teacher_forced_target_rows": exact_rows,
        "row_count": total_rows,
        "mean_initial_correct_token_run": total_initial_run / max(1, total_rows),
        "by_arm": by_arm,
        "functional_verifier_available": False,
        "functional_verifier_state": "NOT_MEASURED_CORPUS_HAS_NO_EXECUTABLE_VERIFIER_CONTRACT",
        "capability_claim": "NOT_EVALUATED",
    }


def run_candidate(
    config: dict[str, Any],
    candidate: dict[str, Any],
    seed: int,
    train_rows: dict[str, list[dict[str, Any]]],
    heldout_rows: dict[str, list[dict[str, Any]]],
    monitor: pretraining_candidate_canary.CandidateCanaryMonitor,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    metadata = json.loads(resolve(config["stage_metadata"]).read_text(encoding="utf-8"))
    base = json.loads(resolve(config["base_config"]).read_text(encoding="utf-8"))
    vocab_size = model_vocab_size(base, metadata["source_vocab"], metadata["target_vocab"])
    model_config = _model_config(config, candidate, vocab_size)
    mx.random.seed(int(seed))
    model = build_model(model_config, mx=mx, nn=nn)
    maximum = int(config["supervision"]["maximum_sequence_tokens"])
    training = config["training"]
    batches = balanced_batches(train_rows, steps=int(training["steps"]), seed=seed)

    def objective(local_model: Any, x: Any, y: Any, mask: Any, scale: float) -> Any:
        logits, _cache, aux = local_model(x, return_training_aux=True)
        losses = nn.losses.cross_entropy(logits, y)
        denominator = mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mx.float32))
        ntp = mx.sum(losses * mask) / denominator
        hidden = aux["final_hidden"]
        if candidate["gradient_route"] == "future_heads_only_stop_gradient_at_trunk":
            hidden = mx.stop_gradient(hidden)
        future_logits = local_model.mtp_logits(hidden)
        future = mtp_auxiliary_loss(
            list(future_logits),
            y,
            mask,
            local_model.mtp_future_offsets,
            local_model.mtp_loss_weights,
            mx,
            nn,
        )
        return ntp + float(scale) * future

    loss_and_grad = nn.value_and_grad(model, objective)
    optimizer = optim.AdamW(
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    initial = _evaluate(model, heldout_rows, maximum, mx, nn)
    curve = [{"step": 0, "heldout": initial}]
    started = time.perf_counter()
    optimizer_positions = 0
    first_active_head_gradient_l1 = 0.0
    first_active_core_gradient_l1 = 0.0
    first_active_gradient_step = 0
    for step, batch in enumerate(batches, 1):
        x_np, y_np, mask_np = make_batch(batch, maximum)
        x, y, mask = mx.array(x_np), mx.array(y_np), mx.array(mask_np)
        scale = candidate_scale(candidate, step - 1, training)
        loss, gradients = loss_and_grad(model, x, y, mask, scale)
        if scale > 0.0 and first_active_gradient_step == 0:
            first_active_gradient_step = step
            for name, value in mlx_utils.tree_flatten(gradients):
                amount = float(mx.sum(mx.abs(value)).item())
                if name.startswith("mtp_"):
                    first_active_head_gradient_l1 += amount
                else:
                    first_active_core_gradient_l1 += amount
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        optimizer_positions += int(mask_np.sum())
        monitor.check(f"{candidate['id']}:{seed}:train", step)
        if step % int(training["evaluation_interval_steps"]) == 0 or step == len(batches):
            curve.append(
                {
                    "step": step,
                    "heldout": _evaluate(model, heldout_rows, maximum, mx, nn),
                }
            )
    wall = time.perf_counter() - started
    final = curve[-1]["heldout"]
    flat = mlx_utils.tree_flatten(model.parameters())
    total_parameters = sum(int(value.size) for _name, value in flat)
    head_parameters = sum(int(value.size) for name, value in flat if name.startswith("mtp_"))
    checkpoint_dir = resolve(config["scratch_root"]) / candidate["id"] / str(seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "weights.npz"
    model.save_weights(str(checkpoint))
    reloaded = build_model(model_config, mx=mx, nn=nn)
    reloaded.load_weights(str(checkpoint))
    replay = _evaluate(reloaded, heldout_rows, maximum, mx, nn)
    reload_exact = canonical(final) == canonical(replay)
    if not reload_exact:
        raise MtpAdequacyFault(f"checkpoint_reload_not_exact:{candidate['id']}:{seed}")
    return {
        "candidate_id": candidate["id"],
        "seed": int(seed),
        "head_mode": candidate["head_mode"],
        "gradient_route": candidate["gradient_route"],
        "schedule": candidate["schedule"],
        "model_config": asdict(model_config),
        "total_parameter_count": total_parameters,
        "future_head_parameter_count": head_parameters,
        "shared_core_parameter_count": total_parameters - head_parameters,
        "optimizer_steps": len(batches),
        "optimizer_positions": optimizer_positions,
        "first_active_gradient_step": first_active_gradient_step,
        "first_active_head_gradient_l1": first_active_head_gradient_l1,
        "first_active_core_gradient_l1": first_active_core_gradient_l1,
        "training_wall_seconds": wall,
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_reload_exact": reload_exact,
        "initial_heldout": initial,
        "final_heldout": final,
        "learning_curve": curve,
        "public_training_rows": 0,
        "public_evaluation_rows": 0,
        "external_inference_calls": 0,
        "fallback_or_template_credit": 0,
        "capability_claim": "NOT_EVALUATED",
    }


def audit_matches(runs: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {(row["candidate_id"], int(row["seed"])): row for row in runs}
    pairs = paired_controls({})
    checks = []
    for candidate_id, control_id in pairs.items():
        for seed in sorted({int(row["seed"]) for row in runs}):
            candidate = indexed[(candidate_id, seed)]
            control = indexed[(control_id, seed)]
            checks.append(
                {
                    "candidate_id": candidate_id,
                    "control_id": control_id,
                    "seed": seed,
                    "same_shared_core_parameter_count": candidate["shared_core_parameter_count"]
                    == control["shared_core_parameter_count"],
                    "same_total_parameter_count": candidate["total_parameter_count"]
                    == control["total_parameter_count"],
                    "same_optimizer_steps": candidate["optimizer_steps"] == control["optimizer_steps"],
                    "same_optimizer_positions": candidate["optimizer_positions"]
                    == control["optimizer_positions"],
                    "same_head_mode": candidate["head_mode"] == control["head_mode"],
                    "same_checkpoint_bytes": candidate["checkpoint_bytes"]
                    == control["checkpoint_bytes"],
                }
            )
    passed = all(all(value for key, value in row.items() if key.startswith("same_")) for row in checks)
    return {"passed": passed, "pairs": checks}


def execute(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    train_rows = load_governed_rows(config, split="train")
    heldout_rows = load_governed_rows(config, split="heldout")
    disjoint = source_disjoint_receipt(train_rows, heldout_rows)
    if not disjoint["passed"]:
        raise MtpAdequacyFault("source_disjointness_failed")
    scratch = resolve(config["scratch_root"])
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    lease = pretraining_candidate_canary.candidate_lease(
        candidate_id=config["candidate_lease_id"],
        max_steps=int(config["training"]["steps"]),
        scratch_checkpoint_root=scratch,
        targets=["shared_trunk"],
        phase="pretraining",
        resume=False,
    )
    if not lease["authorized"]:
        raise MtpAdequacyFault("candidate_lease_denied:" + ",".join(lease["faults"]))
    monitor = pretraining_candidate_canary.CandidateCanaryMonitor(lease)
    runs = []
    for candidate in config["candidates"]:
        for seed in config["seeds"]:
            runs.append(
                run_candidate(
                    config,
                    candidate,
                    int(seed),
                    train_rows,
                    heldout_rows,
                    monitor,
                )
            )
    match_audit = audit_matches(runs)
    if not match_audit["passed"]:
        raise MtpAdequacyFault("matched_control_audit_failed")
    indexed = {}
    for row in runs:
        indexed.setdefault(row["candidate_id"], []).append(row)
    comparisons = {}
    for candidate_id, control_id in paired_controls(config).items():
        comparisons[candidate_id] = {
            "control_id": control_id,
            **summarize_pair(
                indexed[candidate_id],
                indexed[control_id],
                config["prospective_decision"],
            ),
        }
    disposition = campaign_disposition(comparisons)
    resource = monitor.finalize(runs)
    gates = {
        "source_disjoint": disjoint["passed"],
        "matched_controls": match_audit["passed"],
        "checkpoint_reload": all(row["checkpoint_reload_exact"] for row in runs),
        "head_gradient_flow": all(
            row["first_active_gradient_step"] > 0
            and row["first_active_head_gradient_l1"] > 0.0
            for row in runs
        ),
        "resource_bounds": resource["passed"],
        "no_public_or_external_data": all(
            row["public_training_rows"] == 0
            and row["public_evaluation_rows"] == 0
            and row["external_inference_calls"] == 0
            for row in runs
        ),
        "no_fallback_or_template_credit": all(
            row["fallback_or_template_credit"] == 0 for row in runs
        ),
    }
    report = {
        "policy": POLICY,
        "trigger_state": "GREEN" if all(gates.values()) else "RED",
        "support_state": "private-source-disjoint-bounded-experiment",
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "candidate_lease": lease,
        "source_disjointness": disjoint,
        "matched_control_audit": match_audit,
        "resource_receipt": resource,
        "gates": gates,
        "runs": runs,
        "comparisons": comparisons,
        "architecture_disposition": disposition["kind"],
        "campaign_disposition": disposition,
        "non_claims": [
            "teacher-forced token loss and correctness are not direct assistant capability",
            "this corpus has no executable functional verifier contract",
            "an auxiliary-loss improvement cannot by itself authorize architecture adoption",
            "this bounded adequacy run does not consume or predict a public benchmark",
        ],
    }
    output = resolve(config["report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def preflight(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    train_rows = load_governed_rows(config, split="train")
    heldout_rows = load_governed_rows(config, split="heldout")
    disjoint = source_disjoint_receipt(train_rows, heldout_rows)
    return {
        "policy": POLICY,
        "trigger_state": "GREEN" if disjoint["passed"] else "RED",
        "config": relative(config_path),
        "source_disjointness": disjoint,
        "selected_rows": {
            "train": sum(len(rows) for rows in train_rows.values()),
            "heldout": sum(len(rows) for rows in heldout_rows.values()),
        },
        "execution_required": True,
        "capability_claim": "NOT_EVALUATED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    report = execute(config_path) if args.execute else preflight(config_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
