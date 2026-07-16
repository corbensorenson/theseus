#!/usr/bin/env python3
"""Preregister and resource-canary the first post-v8 neural-seed scale rung."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from moecot_language_arm_training import (
    ARM_IDS,
    build_source_to_target_lookup,
    matched_decoder_only_config,
)
from moecot_language_tokenizer import exact_text_tokens
from neural_seed_open_vocab import encode_tokens
from standard_causal_transformer_model import (
    CausalTransformerConfig,
    build_model,
    parameter_count,
)
from standard_causal_transformer_survival import (
    GLOBAL_BOS_ID,
    SOURCE_TARGET_SEPARATOR_ID,
    causal_loss,
    model_vocab_size,
    source_token_offset,
    target_token_offset,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "neural_seed_50m_scale_preregistration.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--execute-canaries", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    report = build_report(config, config_path=config_path, execute_canaries=args.execute_canaries)
    output = resolve(args.out or config["outputs"]["report"])
    write_json(output, report)
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "contract_state": report["contract_state"],
        "proposal_state": report["proposal_state"],
        "architecture": report["architecture"],
        "data_support": report["data_support"],
        "resource_canaries": report["resource_canaries"],
        "hard_gaps": report["hard_gaps"],
    }, indent=2, sort_keys=True))
    return 2 if report["contract_state"] == "RED" else 0


def build_report(
    config: dict[str, Any], *, config_path: Path, execute_canaries: bool
) -> dict[str, Any]:
    started = time.perf_counter()
    validate_config(config)
    paths = {key: resolve(value) for key, value in config["inputs"].items()}
    missing = [key for key, path in paths.items() if not path.is_file()]
    hard_gaps: list[dict[str, Any]] = []
    if missing:
        hard_gaps.append({"kind": "required_input_missing", "inputs": missing})
        return terminal_report(config, config_path, hard_gaps, started)

    verdict = read_json(paths["falsification_verdict"])
    task_report = read_json(paths["task_complete_report"])
    training_admission = read_json(paths["training_admission_report"])
    admission_tcb = read_json(paths["training_admission_epistemic_tcb"])
    capacity_report = read_json(paths["canonical_capacity_report"])
    vocabulary = read_json(paths["vocabulary"])
    generation_contract = read_json(paths["generation_architecture_contract"])
    generation_alignment = generation_architecture_alignment(config, generation_contract)
    if not generation_alignment["aligned"]:
        hard_gaps.append({
            "kind": "generation_architecture_contract_mismatch",
            "evidence": generation_alignment,
        })
    if verdict.get("decision") != "FALSIFY_10_8M_ACTIVE_SCALE_RUNG":
        hard_gaps.append({"kind": "required_scale_falsification_missing"})
    if verdict.get("confirmation_surface_spent") is not False:
        hard_gaps.append({"kind": "confirmation_surface_not_reserved"})
    for key, value in config["boundaries"].items():
        if isinstance(value, bool):
            violated = value
        elif isinstance(value, (int, float)):
            violated = value != 0
        else:
            hard_gaps.append({
                "kind": "boundary_value_not_boolean_or_numeric",
                "field": key,
            })
            continue
        if violated:
            hard_gaps.append({"kind": "boundary_nonzero", "field": key})

    task_contract = task_complete_contract(task_report)
    if not task_contract["contract_ready"]:
        hard_gaps.append({"kind": "task_complete_contract_not_replayable", "evidence": task_contract})
    admission_contract = training_admission_contract(
        training_admission,
        admission_tcb,
        task_report_path=paths["task_complete_report"],
        tcb_path=paths["training_admission_epistemic_tcb"],
    )
    if not admission_contract["contract_ready"]:
        hard_gaps.append({
            "kind": "training_admission_epistemic_tcb_not_qualified",
            "evidence": admission_contract,
        })
    capacity = canonical_capacity(capacity_report)
    if not capacity["receipt_valid"]:
        hard_gaps.append({"kind": "canonical_capacity_receipt_invalid", "evidence": capacity})

    architecture: dict[str, Any] = {}
    mlx_fault = ""
    try:
        architecture = architecture_contract(config, vocabulary)
    except Exception as exc:  # noqa: BLE001
        mlx_fault = f"{type(exc).__name__}: {exc}"
        hard_gaps.append({"kind": "mlx_architecture_instantiation_failed", "error": mlx_fault})
    scale = config["scaling_contract"]
    required_positions = int(math.ceil(
        int(architecture.get("active_parameter_count_per_request") or 0)
        * float(scale["minimum_unique_positions_per_active_parameter"])
    ))
    observed_positions = int(capacity["unique_model_visible_positions"])
    unique_position_ready = bool(required_positions and observed_positions >= required_positions)
    specialist_support = specialist_data_support(
        architecture,
        capacity,
        minimum_ratio=float(scale["minimum_unique_positions_per_active_parameter"]),
    )
    if not specialist_support["ready"]:
        hard_gaps.append({
            "kind": "specialist_unique_position_floor_not_met",
            "arms": specialist_support["shortfall_arms"],
        })
    coverage_ready = bool(task_contract["coverage_ready"])
    data_support = {
        "canonical_unique_model_visible_positions": observed_positions,
        "required_unique_positions": required_positions,
        "unique_position_shortfall": max(0, required_positions - observed_positions),
        "unique_positions_per_active_parameter": round(
            observed_positions / max(1, int(architecture.get("active_parameter_count_per_request") or 0)), 6
        ),
        "minimum_unique_positions_per_active_parameter": float(
            scale["minimum_unique_positions_per_active_parameter"]
        ),
        "unique_position_floor_ready": unique_position_ready,
        "specialist_unique_position_floor_ready": specialist_support["ready"],
        "specialist_unique_position_support": specialist_support["arms"],
        "task_complete_contract_ready": task_contract["contract_ready"],
        "task_complete_coverage_ready": coverage_ready,
        "task_complete_coverage": task_contract["coverage"],
        "training_admission_contract_ready": admission_contract["contract_ready"],
        "training_admission_epistemic_tcb_qualified": admission_contract["tcb_qualified"],
        "training_admission_epistemic_tcb_surviving_mutants": admission_contract["surviving_mutant_count"],
        "training_data_supported": (
            unique_position_ready and specialist_support["ready"] and coverage_ready
        ),
        "optimizer_repetition_counted_as_unique_data": False,
    }

    canary_report: dict[str, Any] = {
        "state": "NOT_RUN",
        "reason": "pass --execute-canaries to run one-step checkpoint/resume proofs",
        "models": [],
    }
    if execute_canaries and not hard_gaps:
        try:
            canary_report = run_resource_canaries(config, architecture, vocabulary, task_report)
        except Exception as exc:  # noqa: BLE001
            canary_report = {
                "state": "RED",
                "reason": f"{type(exc).__name__}: {exc}"[:2000],
                "models": [],
            }
    canary_ready = canary_report.get("state") == "GREEN"
    proposal_ready = not hard_gaps and data_support["training_data_supported"] and canary_ready
    if proposal_ready:
        proposal_state = "AUTHORIZED_FOR_FROZEN_TRAINING_PLAN"
    elif hard_gaps:
        proposal_state = "DENIED_CONTRACT_FAILURE"
    elif not data_support["training_data_supported"]:
        proposal_state = "DENIED_INSUFFICIENT_DATA_SUPPORT"
    else:
        proposal_state = "DENIED_RESOURCE_CANARY_NOT_GREEN"
    contract_state = "RED" if hard_gaps else "GREEN"
    trigger_state = "GREEN" if proposal_ready else ("RED" if hard_gaps else "YELLOW")
    return {
        "policy": config["policy"],
        "created_utc": now(),
        "trigger_state": trigger_state,
        "contract_state": contract_state,
        "proposal_state": proposal_state,
        "training_authorized": proposal_ready,
        "config": artifact_ref(config_path),
        "input_artifacts": {key: artifact_ref(path) for key, path in paths.items()},
        "falsified_rung_disposition": verdict.get("decision"),
        "confirmation_surface_spent": verdict.get("confirmation_surface_spent"),
        "architecture": architecture,
        "generation_architecture": generation_alignment,
        "data_support": data_support,
        "resource_canaries": canary_report,
        "training_stop_contract": config["training_stop_contract"],
        "heldout_utility_contract": config["heldout_utility_contract"],
        "hard_gaps": hard_gaps,
        "boundaries": config["boundaries"],
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "score_semantics": (
            "This report preregisters scale and proves bounded MLX mechanics. It is not training, "
            "model utility, public calibration, architecture promotion, or route authority."
        ),
        "non_claims": [
            "A successful optimizer/checkpoint canary is not evidence that the model can answer a task.",
            "Canonical bulk positions do not replace per-arm task-complete coverage.",
            "A preregistered architecture is not authorized while any data or resource floor is unmet.",
            "The consumed 10.8M rung remains immutable and is not restarted by this proposal.",
        ],
    }


def terminal_report(
    config: dict[str, Any], config_path: Path, hard_gaps: list[dict[str, Any]], started: float
) -> dict[str, Any]:
    return {
        "policy": config.get("policy"),
        "created_utc": now(),
        "trigger_state": "RED",
        "contract_state": "RED",
        "proposal_state": "DENIED_CONTRACT_FAILURE",
        "training_authorized": False,
        "config": artifact_ref(config_path),
        "architecture": {},
        "data_support": {},
        "resource_canaries": {"state": "NOT_RUN", "models": []},
        "hard_gaps": hard_gaps,
        "boundaries": config.get("boundaries") or {},
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def architecture_contract(config: dict[str, Any], vocabulary: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils as mlx_utils

    source_vocab = dict(vocabulary.get("source_vocab") or {})
    target_vocab = dict(vocabulary.get("target_vocab") or {})
    if not source_vocab or not target_vocab:
        raise ValueError("canonical exact vocabulary is empty")
    base: dict[str, Any] = {"tokenization": {"shared_source_target_vocabulary": False}}
    vocab_size = model_vocab_size(base, source_vocab, target_vocab)
    metadata = {"source_vocab": source_vocab, "target_vocab": target_vocab}
    copy_lookup = build_source_to_target_lookup(base, metadata)

    def instantiate(model_config: dict[str, Any]) -> Any:
        return build_model(
            CausalTransformerConfig(vocab_size=vocab_size, **model_config),
            mx=mx,
            nn=nn,
            source_to_target_lookup=(
                mx.array(copy_lookup, dtype=mx.int32)
                if model_config.get("attention_policy") == "encoder_decoder"
                else None
            ),
        )

    def count(model_config: dict[str, Any]) -> int:
        model = instantiate(model_config)
        observed = int(parameter_count(model, mlx_utils))
        del model
        return observed

    candidate = config["candidate"]
    trunk_count = count(candidate["shared_trunk_model"])
    arm = instantiate(candidate["arm_model"])
    arm_count = int(parameter_count(arm, mlx_utils))
    arm.freeze_to_language_expert(candidate["expert_trainable_scope"])
    expert_count = int(sum(
        value.size for _name, value in mlx_utils.tree_flatten(arm.trainable_parameters())
    ))
    del arm
    total_count = trunk_count + expert_count * int(candidate["arm_count"])
    dense_active_model, dense_active_count = matched_decoder_only_config(
        arm_count, candidate["arm_model"], count=count
    )
    dense_total_model, dense_total_count = matched_decoder_only_config(
        total_count, candidate["arm_model"], count=count
    )
    active_increment_model = dict(dense_active_model)
    active_increment_model["ff_dim"] = int(active_increment_model["ff_dim"]) + 1
    total_increment_model = dict(dense_total_model)
    total_increment_model["ff_dim"] = int(total_increment_model["ff_dim"]) + 1
    active_ff_increment = abs(count(active_increment_model) - dense_active_count)
    total_ff_increment = abs(count(total_increment_model) - dense_total_count)
    floor = int(config["scaling_contract"]["minimum_active_parameters"])
    ceiling = int(config["scaling_contract"]["maximum_active_parameters"])
    if not floor <= arm_count <= ceiling:
        raise ValueError(f"candidate active count outside preregistered band: {arm_count}")
    active_matched = abs(dense_active_count - arm_count) <= active_ff_increment
    total_matched = abs(dense_total_count - total_count) <= total_ff_increment
    if not active_matched or not total_matched:
        raise ValueError("matched dense control exceeds one indivisible FF-width increment")
    return {
        "candidate_id": candidate["id"],
        "vocab_size": vocab_size,
        "shared_trunk_model": candidate["shared_trunk_model"],
        "arm_model": candidate["arm_model"],
        "shared_trunk_parameter_count": trunk_count,
        "expert_parameter_count_per_arm": expert_count,
        "active_parameter_count_per_request": arm_count,
        "total_parameter_count": total_count,
        "arm_count": int(candidate["arm_count"]),
        "dense_active_parameter": {
            "model": dense_active_model,
            "parameter_count": dense_active_count,
            "delta": dense_active_count - arm_count,
            "ff_width_parameter_increment": active_ff_increment,
        },
        "dense_total_parameter": {
            "model": dense_total_model,
            "parameter_count": dense_total_count,
            "delta": dense_total_count - total_count,
            "ff_width_parameter_increment": total_ff_increment,
        },
        "matched_active_within_one_ff_increment": active_matched,
        "matched_total_within_one_ff_increment": total_matched,
        "router_parameter_count": 0,
        "router_state": "NOT_TRAINED_OR_CREDITED",
    }


def run_resource_canaries(
    config: dict[str, Any],
    architecture: dict[str, Any],
    vocabulary: dict[str, Any],
    task_report: dict[str, Any],
) -> dict[str, Any]:
    ledger = resolve(str((task_report.get("ledger_receipt") or {}).get("path") or ""))
    ledger_receipt = task_report.get("ledger_receipt") or {}
    if not ledger.is_file() or file_sha256(ledger) != ledger_receipt.get("sha256"):
        raise ValueError("task-complete ledger identity changed before resource canary")
    unit = first_canary_unit(ledger)
    canary = config["resource_canary"]
    batch = encode_canary_batch(
        unit,
        vocabulary,
        sequence_length=int(canary["sequence_length"]),
    )
    models = [
        ("moecot_active_arm", architecture["arm_model"]),
        ("dense_active_parameter", architecture["dense_active_parameter"]["model"]),
        ("dense_total_parameter", architecture["dense_total_parameter"]["model"]),
    ]
    rows = [
        run_one_model_canary(
            model_id,
            model_config,
            vocabulary=vocabulary,
            batch=batch,
            seed=int(config["seed"]) + index,
            learning_rate=float(canary["learning_rate"]),
        )
        for index, (model_id, model_config) in enumerate(models)
    ]
    required = (
        "finite_loss",
        "parameter_updated",
        "checkpoint_reload_equivalent",
        "optimizer_resume_succeeded",
    )
    state = "GREEN" if all(all(row["checks"].get(key) for key in required) for row in rows) else "RED"
    return {
        "state": state,
        "models": rows,
        "source_unit_id": unit["unit_id"],
        "source_unit_arm": unit["arm_id"],
        "source_payload_retained_in_report": False,
        "sequence_length": int(canary["sequence_length"]),
        "batch_size": int(canary["batch_size"]),
        "temporary_artifacts_retained": False,
    }


def run_one_model_canary(
    model_id: str,
    model_config: dict[str, Any],
    *,
    vocabulary: dict[str, Any],
    batch: dict[str, np.ndarray],
    seed: int,
    learning_rate: float,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    source_vocab = dict(vocabulary["source_vocab"])
    target_vocab = dict(vocabulary["target_vocab"])
    base = {"tokenization": {"shared_source_target_vocabulary": False}}
    vocab_size = model_vocab_size(base, source_vocab, target_vocab)
    copy_lookup = build_source_to_target_lookup(base, vocabulary)

    def instantiate() -> Any:
        return build_model(
            CausalTransformerConfig(vocab_size=vocab_size, **model_config),
            mx=mx,
            nn=nn,
            source_to_target_lookup=(
                mx.array(copy_lookup, dtype=mx.int32)
                if model_config.get("attention_policy") == "encoder_decoder"
                else None
            ),
        )

    mx.random.seed(seed)
    mx.clear_cache()
    mx.reset_peak_memory()
    model = instantiate()
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    inputs = mx.array(batch["inputs"], dtype=mx.int32)
    labels = mx.array(batch["labels"], dtype=mx.int32)
    mask = mx.array(batch["mask"], dtype=mx.float32)

    def loss_fn(active_model: Any) -> Any:
        return causal_loss(active_model, inputs, labels, mask, mx, nn)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    flat_before = mlx_utils.tree_flatten(model.trainable_parameters())
    first_name, first_value = flat_before[0]
    before = np.array(first_value)
    started = time.perf_counter()
    first_loss, grads = loss_and_grad(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, first_loss)
    after_first = float(loss_fn(model).item())
    after = np.array(dict(mlx_utils.tree_flatten(model.trainable_parameters()))[first_name])
    parameter_updated = not np.array_equal(before, after)

    with tempfile.TemporaryDirectory(prefix="theseus-50m-canary-") as raw:
        root = Path(raw)
        checkpoint = root / "weights.safetensors"
        optimizer_path = root / "optimizer.safetensors"
        model.save_weights(str(checkpoint))
        mx.save_safetensors(
            str(optimizer_path),
            {name: value for name, value in mlx_utils.tree_flatten(optimizer.state)},
            metadata={"policy": "project_theseus_50m_resource_canary_optimizer_v1"},
        )
        checkpoint_ref = artifact_ref(checkpoint)
        optimizer_ref = artifact_ref(optimizer_path)
        resumed_model = instantiate()
        resumed_model.load_weights(str(checkpoint))
        resumed_optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.01)
        resumed_optimizer.state = mlx_utils.tree_unflatten(list(mx.load(str(optimizer_path)).items()))
        mx.eval(resumed_model.parameters(), resumed_optimizer.state)
        reloaded_loss = float(loss_fn(resumed_model).item())
        resumed_loss_and_grad = nn.value_and_grad(resumed_model, loss_fn)
        second_loss, second_grads = resumed_loss_and_grad(resumed_model)
        resumed_optimizer.update(resumed_model, second_grads)
        mx.eval(resumed_model.parameters(), resumed_optimizer.state, second_loss)
        after_resume = float(loss_fn(resumed_model).item())
    duration = time.perf_counter() - started
    losses = [float(first_loss.item()), after_first, reloaded_loss, float(second_loss.item()), after_resume]
    checks = {
        "finite_loss": all(math.isfinite(value) for value in losses),
        "parameter_updated": parameter_updated,
        "checkpoint_reload_equivalent": abs(after_first - reloaded_loss) <= 1e-5,
        "optimizer_resume_succeeded": math.isfinite(after_resume),
        "checkpoint_and_optimizer_written": checkpoint_ref["bytes"] > 0 and optimizer_ref["bytes"] > 0,
    }
    row = {
        "model_id": model_id,
        "model_config": model_config,
        "parameter_count": int(parameter_count(model, mlx_utils)),
        "checks": checks,
        "losses": {
            "first_step_before_update": round(losses[0], 8),
            "after_first_update": round(after_first, 8),
            "after_checkpoint_reload": round(reloaded_loss, 8),
            "resume_step_before_update": round(losses[3], 8),
            "after_resume_update": round(after_resume, 8),
        },
        "checkpoint_receipt": checkpoint_ref,
        "optimizer_receipt": optimizer_ref,
        "temporary_artifacts_deleted_after_replay": True,
        "metal_peak_memory_bytes": int(mx.get_peak_memory()),
        "duration_ms": int(duration * 1000),
        "optimizer_steps": 2,
        "optimizer_positions": int(batch["mask"].sum()) * 2,
        "accepted_verified_output_count": 0,
        "capability_claim": "NOT_EVALUATED",
    }
    del model, optimizer, resumed_model, resumed_optimizer, grads, second_grads
    gc.collect()
    mx.clear_cache()
    return row


def encode_canary_batch(
    unit: dict[str, Any], vocabulary: dict[str, Any], *, sequence_length: int
) -> dict[str, np.ndarray]:
    source_vocab = dict(vocabulary["source_vocab"])
    target_vocab = dict(vocabulary["target_vocab"])
    base = {"tokenization": {"shared_source_target_vocabulary": False}}
    source_ids, _source_receipt = encode_tokens(
        exact_text_tokens(str(unit["visible_context"])), source_vocab, stream="source"
    )
    target_ids, _target_receipt = encode_tokens(
        exact_text_tokens(str(unit["target"])), target_vocab, stream="target"
    )
    source_budget = min(len(source_ids), max(8, sequence_length // 2))
    target_budget = max(1, sequence_length - source_budget - 4)
    source_offset = source_token_offset(base, source_vocab)
    target_offset = target_token_offset(base, source_vocab)
    sequence = [GLOBAL_BOS_ID]
    sequence.extend(source_offset + int(value) for value in source_ids[:source_budget])
    sequence.append(SOURCE_TARGET_SEPARATOR_ID)
    sequence.append(target_offset + int(target_vocab["<bos>"]))
    target_start = len(sequence)
    sequence.extend(target_offset + int(value) for value in target_ids[:target_budget])
    sequence.append(target_offset + int(target_vocab["<eos>"]))
    sequence = sequence[: sequence_length + 1]
    if len(sequence) < 2 or target_start >= len(sequence):
        raise ValueError("task-complete canary unit has no representable target span")
    inputs = np.zeros((1, sequence_length), dtype=np.int32)
    labels = np.zeros((1, sequence_length), dtype=np.int32)
    mask = np.zeros((1, sequence_length), dtype=np.float32)
    active_inputs = sequence[:-1]
    active_labels = sequence[1:]
    inputs[0, : len(active_inputs)] = active_inputs
    labels[0, : len(active_labels)] = active_labels
    mask_start = max(0, target_start - 1)
    mask[0, mask_start : len(active_labels)] = 1.0
    if not mask.any():
        raise ValueError("task-complete canary target mask is empty")
    return {"inputs": inputs, "labels": labels, "mask": mask}


def first_canary_unit(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("decision") == "admit" and row.get("split") == "train":
                return row
    raise ValueError("task-complete ledger has no admitted train unit")


def task_complete_contract(report: dict[str, Any]) -> dict[str, Any]:
    ledger = report.get("ledger_receipt") if isinstance(report.get("ledger_receipt"), dict) else {}
    path = resolve(str(ledger.get("path") or "")) if ledger.get("path") else None
    replay = bool(
        path
        and path.is_file()
        and ledger.get("replay_valid") is True
        and file_sha256(path) == ledger.get("sha256")
    )
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    coverage = report.get("coverage") if isinstance(report.get("coverage"), dict) else {}
    return {
        "contract_ready": bool(
            report.get("policy") == "project_theseus_task_complete_training_units_v1"
            and report.get("contract_state") == "GREEN"
            and replay
            and int(summary.get("contract_hard_gap_count") or 0) == 0
        ),
        "coverage_ready": report.get("coverage_state") == "GREEN",
        "coverage": coverage,
        "ledger_replay_valid": replay,
    }


def training_admission_contract(
    admission: dict[str, Any],
    tcb: dict[str, Any],
    *,
    task_report_path: Path,
    tcb_path: Path,
) -> dict[str, Any]:
    summary = admission.get("summary") if isinstance(admission.get("summary"), dict) else {}
    embedded = (
        admission.get("training_admission_epistemic_tcb")
        if isinstance(admission.get("training_admission_epistemic_tcb"), dict)
        else {}
    )
    tcb_summary = tcb.get("summary") if isinstance(tcb.get("summary"), dict) else {}
    artifacts = tcb.get("input_artifacts") if isinstance(tcb.get("input_artifacts"), dict) else {}
    task_ref = artifacts.get("task_report") if isinstance(artifacts.get("task_report"), dict) else {}
    task_hash_matches = bool(
        task_report_path.is_file()
        and task_ref.get("sha256") == file_sha256(task_report_path)
        and task_ref.get("path") == relative(task_report_path)
    )
    tcb_hash_matches = bool(
        tcb_path.is_file()
        and embedded.get("sha256") == file_sha256(tcb_path)
        and embedded.get("path") == relative(tcb_path)
    )
    tcb_qualified = bool(
        tcb.get("policy") == "project_theseus_training_admission_epistemic_tcb_v1"
        and tcb.get("trigger_state") == "GREEN"
        and not tcb.get("hard_gaps")
        and int(tcb_summary.get("mutation_count") or 0) > 0
        and int(tcb_summary.get("surviving_mutant_count") or 0) == 0
        and embedded.get("qualified") is True
        and summary.get("training_admission_epistemic_tcb_qualified") is True
        and task_hash_matches
        and tcb_hash_matches
    )
    return {
        "contract_ready": bool(
            admission.get("policy") == "project_theseus_training_data_admission_v1"
            and admission.get("trigger_state") in {"GREEN", "YELLOW"}
            and not [
                row for row in admission.get("gates", [])
                if isinstance(row, dict)
                and row.get("severity") == "hard"
                and row.get("passed") is not True
            ]
            and tcb_qualified
        ),
        "tcb_qualified": tcb_qualified,
        "task_report_hash_matches": task_hash_matches,
        "tcb_report_hash_matches": tcb_hash_matches,
        "mutation_count": int(tcb_summary.get("mutation_count") or 0),
        "surviving_mutant_count": int(tcb_summary.get("surviving_mutant_count") or 0),
        "non_claim": "Training admission trust qualification is not data sufficiency or learned capability.",
    }


def canonical_capacity(report: dict[str, Any]) -> dict[str, Any]:
    contract = report.get("data_model_scaling_contract") if isinstance(report.get("data_model_scaling_contract"), dict) else {}
    receipt = contract.get("canonical_corpus_receipt") if isinstance(contract.get("canonical_corpus_receipt"), dict) else {}
    return {
        "receipt_valid": bool(receipt.get("valid") and receipt.get("content_bound") and not receipt.get("hard_gaps")),
        "unique_model_visible_positions": int(receipt.get("unique_model_visible_positions") or 0),
        "domain_unique_positions": dict(receipt.get("domain_unique_positions") or {}),
        "code_language_unique_positions": dict(
            receipt.get("code_language_unique_positions") or {}
        ),
        "optimizer_repetition_counted_as_unique_data": False,
    }


def specialist_data_support(
    architecture: dict[str, Any],
    capacity: dict[str, Any],
    *,
    minimum_ratio: float,
) -> dict[str, Any]:
    """Bind every independently trained expert to unique in-scope source data."""

    expert_parameters = int(architecture.get("expert_parameter_count_per_arm") or 0)
    required = int(math.ceil(expert_parameters * minimum_ratio)) if expert_parameters else 0
    domains = dict(capacity.get("domain_unique_positions") or {})
    languages = dict(capacity.get("code_language_unique_positions") or {})
    observed = {
        "english": int(domains.get("english_natural_language_total") or 0),
        "python": int(languages.get("python") or 0),
        "javascript_typescript": int(languages.get("javascript_typescript") or 0),
        "html_css": int(languages.get("html_css") or 0),
        "rust": int(languages.get("rust") or 0),
    }
    rows: dict[str, Any] = {}
    shortfall_arms: list[str] = []
    for arm_id in ARM_IDS:
        positions = observed[arm_id]
        ready = bool(required and positions >= required)
        if not ready:
            shortfall_arms.append(arm_id)
        rows[arm_id] = {
            "owned_parameter_count": expert_parameters,
            "unique_model_visible_positions": positions,
            "minimum_required_positions": required,
            "shortfall_positions": max(0, required - positions),
            "positions_per_owned_parameter": round(
                positions / max(1, expert_parameters), 6
            ),
            "meets_floor": ready,
        }
    return {
        "policy": "project_theseus_neural_seed_specialist_data_support_v1",
        "minimum_unique_positions_per_owned_parameter": minimum_ratio,
        "ready": not shortfall_arms,
        "shortfall_arms": shortfall_arms,
        "arms": rows,
        "optimizer_repetition_counted_as_unique_data": False,
    }


def validate_config(config: dict[str, Any]) -> None:
    if config.get("policy") != "project_theseus_neural_seed_50m_scale_preregistration_v1":
        raise ValueError("unexpected 50M scale preregistration policy")
    if tuple(config["candidate"]["arms"]) != ARM_IDS:
        raise ValueError("candidate arm order must match canonical MoECOT arm order")
    if config["resource_canary"].get("models") != [
        "moecot_active_arm", "dense_active_parameter", "dense_total_parameter"
    ]:
        raise ValueError("resource canary must cover candidate and both matched controls")
    if config["boundaries"].get("falsified_10_8m_rung_restart_allowed") is not False:
        raise ValueError("falsified rung restart must remain forbidden")
    if "generation_architecture_contract" not in config.get("inputs", {}):
        raise ValueError("generation architecture contract input is required")


def generation_architecture_alignment(
    config: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    mtp = dict((contract.get("modes") or {}).get("mtp") or {})
    shape = dict(contract.get("mtp_shape_contract") or {})
    expected = {
        "mtp_future_offsets": list(shape.get("future_offsets") or []),
        "mtp_low_rank": int(mtp.get("low_rank") or 0),
        "mtp_loss_weights": list(mtp.get("loss_weights") or []),
        "mtp_loss_scale": 0.0,
        "mtp_maximum_head_parameter_overhead_ratio": float(
            shape.get("maximum_parameter_overhead_ratio") or 0.0
        ),
    }
    candidate = config.get("candidate") or {}
    observed = {
        model_id: {key: (candidate.get(model_id) or {}).get(key) for key in expected}
        for model_id in ("shared_trunk_model", "arm_model")
    }
    mismatches = [
        model_id
        for model_id, fields in observed.items()
        if fields != expected
    ]
    policy_valid = (
        contract.get("policy") == "project_theseus_generation_architecture_contracts_v1"
        and contract.get("first_campaign_base") == "autoregressive"
        and mtp.get("first_campaign_disposition")
        == "included_disabled_weight_zero_until_preregistered_schedule"
    )
    return {
        "aligned": policy_valid and not mismatches,
        "contract_policy": contract.get("policy"),
        "base_mode": contract.get("first_campaign_base"),
        "checkpoint_shaping_auxiliary": "mtp",
        "expected_model_fields": expected,
        "observed_model_fields": observed,
        "mismatched_models": mismatches,
        "initial_optimizer_exposure": 0,
        "behavior_or_speed_claim": "NOT_CLAIMED",
    }


def artifact_ref(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "sha256": file_sha256(path) if path.is_file() else None,
        "bytes": path.stat().st_size if path.is_file() else 0,
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
