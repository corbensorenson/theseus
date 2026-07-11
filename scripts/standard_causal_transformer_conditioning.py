#!/usr/bin/env python3
"""Increase prompt/body dependence of the standard causal survival model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from standard_causal_transformer_model import CausalTransformerConfig, build_model  # noqa: E402
from standard_causal_transformer_survival import (  # noqa: E402
    SOURCE_TARGET_SEPARATOR_ID,
    build_schedule,
    encode_sft_examples,
    eval_example,
    evaluate_loss,
    model_vocab_size,
    visible_eval_source,
)


DEFAULT_CONFIG = ROOT / "configs" / "standard_causal_transformer_conditioning.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = read_json(resolve(args.config))
    validate_config(config)
    if not args.execute:
        print(json.dumps({"policy": config["policy"], "trigger_state": "PLANNED"}, indent=2))
        return 0
    report = run(config)
    write_json(resolve(config["canonical_report"]), report)
    print(json.dumps({"trigger_state": report["trigger_state"], "conditioning": report["conditioning"]}, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def run(config: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    base_cfg = read_json(resolve(config["base_config"]))
    base_checkpoint = resolve(config["base_checkpoint"])
    if not base_checkpoint.exists():
        raise FileNotFoundError(f"base checkpoint missing: {base_checkpoint}")
    stage_dir = ROOT / "runtime" / "standard_causal_transformer_survival_v1"
    arrays = np.load(stage_dir / "stage_arrays_v1.npz")
    metadata = read_json(stage_dir / "stage_metadata_v1.json")
    vocab = read_json(resolve(base_cfg["tokenization"]["source_vocab"]))
    model_cfg = CausalTransformerConfig(
        vocab_size=model_vocab_size(base_cfg, vocab["source_vocab"], vocab["target_vocab"]),
        **base_cfg["model"],
    )
    model = build_model(model_cfg, mx=mx, nn=nn)
    model.load_weights(str(base_checkpoint))
    mx.eval(model.parameters())

    positive_inputs = arrays["sft_inputs"]
    positive_labels = arrays["sft_labels"]
    positive_mask = arrays["sft_mask"]
    negative_inputs, negative_labels, negative_mask = deranged_source_arrays(
        positive_inputs,
        positive_labels,
        positive_mask,
        seed=int(config["seed"]),
    )
    eval_rows = list(metadata["eval_rows"])
    matched_eval = encode_sft_examples(
        base_cfg,
        [eval_example(row) for row in eval_rows],
        vocab["source_vocab"],
        vocab["target_vocab"],
    )
    mismatched_eval = encode_sft_examples(
        base_cfg,
        [
            {
                "source_text": visible_eval_source(eval_rows[(index + 7) % len(eval_rows)]),
                "body": str(row.get("solution_body") or "").strip(),
                "source": "deterministic_eval_derangement",
            }
            for index, row in enumerate(eval_rows)
        ],
        vocab["source_vocab"],
        vocab["target_vocab"],
    )
    matched_before = evaluate_loss(
        model, *matched_eval, batch_size=int(config["training"]["batch_size"]), mx=mx, nn=nn
    )
    mismatched_before = evaluate_loss(
        model, *mismatched_eval, batch_size=int(config["training"]["batch_size"]), mx=mx, nn=nn
    )

    training = config["training"]
    mean_positions = max(1.0, float(positive_mask.sum(axis=1).mean()))
    planned_steps = max(
        1,
        math.ceil(
            int(training["target_token_positions"])
            / (mean_positions * int(training["batch_size"]))
        ),
    )
    schedule = build_schedule(optim, mx, training, planned_steps)
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=float(training["weight_decay"]))
    loss_and_grad = nn.value_and_grad(model, pairwise_conditioning_loss)
    matrices = [
        mx.array(value, dtype=mx.float32 if value.dtype.kind == "f" else mx.int32)
        for value in (
            positive_inputs,
            positive_labels,
            positive_mask,
            negative_inputs,
            negative_labels,
            negative_mask,
        )
    ]
    mx.eval(*matrices)
    order = list(range(len(positive_inputs)))
    consumed = 0
    steps = 0
    losses: list[float] = []
    pairwise_terms: list[float] = []
    started = time.perf_counter()
    epoch = 0
    checkpoint_dir = resolve(config["conditioned_checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "standard_causal_transformer_survival_v1.npz"
    model.train()
    while consumed < int(training["target_token_positions"]):
        random.Random(int(config["seed"]) + epoch).shuffle(order)
        for start in range(0, len(order), int(training["batch_size"])):
            if consumed >= int(training["target_token_positions"]):
                break
            indices = order[start : start + int(training["batch_size"])]
            take = mx.array(indices, dtype=mx.int32)
            values = [matrix[take] for matrix in matrices]
            (loss, diagnostics), grads = loss_and_grad(
                model,
                *values,
                float(training["pairwise_margin"]),
                float(training["pairwise_weight"]),
                mx,
                nn,
            )
            grads, grad_norm = optim.clip_grad_norm(grads, float(training["gradient_clip_norm"]))
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss, diagnostics, grad_norm)
            losses.append(float(loss.item()))
            pairwise_terms.append(float(diagnostics.item()))
            consumed += int(positive_mask[indices].sum())
            steps += 1
            if steps % 25 == 0:
                print(
                    f"phase=source_conditioning step={steps}/{planned_steps} "
                    f"positions={consumed}/{training['target_token_positions']} "
                    f"loss={losses[-1]:.4f} pairwise={pairwise_terms[-1]:.4f}",
                    flush=True,
                )
            if steps % int(training["checkpoint_every_steps"]) == 0:
                model.save_weights(str(checkpoint))
        epoch += 1
    model.save_weights(str(checkpoint))
    mx.eval(model.parameters())
    matched_after = evaluate_loss(
        model, *matched_eval, batch_size=int(training["batch_size"]), mx=mx, nn=nn
    )
    mismatched_after = evaluate_loss(
        model, *mismatched_eval, batch_size=int(training["batch_size"]), mx=mx, nn=nn
    )

    base_report = read_json(resolve(config["base_report"]))
    base_training = base_report.get("training") if isinstance(base_report.get("training"), dict) else {}
    phase = {
        "phase": "contrastive_prompt_body_source_conditioning",
        "optimizer_steps": steps,
        "epochs_touched": epoch,
        "target_positions_consumed": consumed,
        "target_positions_requested": int(training["target_token_positions"]),
        "mean_loss": round(sum(losses) / max(1, len(losses)), 6),
        "final_loss": round(losses[-1], 6),
        "mean_pairwise_term": round(sum(pairwise_terms) / max(1, len(pairwise_terms)), 6),
        "tokens_per_second": round(consumed / max(1e-9, time.perf_counter() - started), 3),
        "external_inference_calls": 0,
    }
    gap_before = mismatched_before - matched_before
    gap_after = mismatched_after - matched_after
    conditioning = {
        "policy": config["policy"],
        "base_checkpoint": rel(base_checkpoint),
        "base_checkpoint_sha256": file_sha256(base_checkpoint),
        "conditioned_checkpoint": rel(checkpoint),
        "conditioned_checkpoint_sha256": file_sha256(checkpoint),
        "negative_pair_policy": config["boundaries"]["negative_pair_policy"],
        "matched_loss_before": matched_before,
        "mismatched_loss_before": mismatched_before,
        "conditioning_gap_before": round(gap_before, 6),
        "matched_loss_after": matched_after,
        "mismatched_loss_after": mismatched_after,
        "conditioning_gap_after": round(gap_after, 6),
        "conditioning_gap_improved": gap_after > gap_before,
        "training_phase": phase,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    report = dict(base_report)
    report["created_utc"] = now()
    report["trigger_state"] = "GREEN" if conditioning["conditioning_gap_improved"] else "YELLOW"
    report["conditioning"] = conditioning
    report.setdefault("artifacts", {})["checkpoint"] = rel(checkpoint)
    report["training"] = {
        **base_training,
        "optimizer_steps": int(base_training.get("optimizer_steps") or 0) + steps,
        "phases": [*list(base_training.get("phases") or []), phase],
        "conditioning_complete": True,
        "evaluation_only_replay": False,
    }
    return report


def pairwise_conditioning_loss(
    model: Any,
    positive_inputs: Any,
    positive_labels: Any,
    positive_mask: Any,
    negative_inputs: Any,
    negative_labels: Any,
    negative_mask: Any,
    margin: float,
    pairwise_weight: float,
    mx: Any,
    nn: Any,
) -> tuple[Any, Any]:
    positive = row_nll(model, positive_inputs, positive_labels, positive_mask, mx, nn)
    negative = row_nll(model, negative_inputs, negative_labels, negative_mask, mx, nn)
    pairwise = mx.mean(mx.logaddexp(mx.array(0.0), margin + positive - negative))
    return mx.mean(positive) + pairwise_weight * pairwise, pairwise


def row_nll(model: Any, inputs: Any, labels: Any, mask: Any, mx: Any, nn: Any) -> Any:
    logits, _cache = model(inputs)
    token_loss = nn.losses.cross_entropy(logits, labels)
    denominator = mx.maximum(mx.sum(mask, axis=1), mx.ones((mask.shape[0],)))
    return mx.sum(token_loss * mask, axis=1) / denominator


def deranged_source_arrays(
    inputs: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count, width = inputs.shape
    if count < 2:
        raise ValueError("source conditioning requires at least two rows")
    shift = 1 + seed % (count - 1)
    negative_inputs = np.zeros_like(inputs)
    negative_labels = np.zeros_like(labels)
    negative_mask = np.zeros_like(mask)
    for index in range(count):
        source_index = (index + shift) % count
        source_separator = separator_index(inputs[source_index])
        target_separator = separator_index(inputs[index])
        target_positions = np.flatnonzero(mask[index] > 0)
        if not len(target_positions):
            continue
        target_end = int(target_positions[-1]) + 1
        source = list(inputs[source_index, 1:source_separator])
        target_inputs = list(inputs[index, target_separator + 1 : target_end])
        final_label = int(labels[index, target_end - 1])
        sequence = [int(inputs[index, 0]), *source, SOURCE_TARGET_SEPARATOR_ID, *target_inputs, final_label]
        packed_width = min(width, len(sequence) - 1)
        negative_inputs[index, :packed_width] = sequence[:packed_width]
        negative_labels[index, :packed_width] = sequence[1 : packed_width + 1]
        mask_start = 1 + len(source) + 1
        negative_mask[index, mask_start:packed_width] = 1.0
    return negative_inputs, negative_labels, negative_mask


def separator_index(row: np.ndarray) -> int:
    found = np.flatnonzero(row == SOURCE_TARGET_SEPARATOR_ID)
    if not len(found):
        raise ValueError("SFT row is missing source/target separator")
    return int(found[0])


def validate_config(config: dict[str, Any]) -> None:
    if config.get("policy") != "project_theseus_standard_causal_transformer_conditioning_v1":
        raise ValueError("unexpected conditioning policy")
    boundary = config.get("boundaries") or {}
    if int(boundary.get("public_training_rows") or 0) != 0:
        raise ValueError("public training rows are forbidden")
    if int(boundary.get("external_inference_calls") or 0) != 0:
        raise ValueError("external inference is forbidden")
    if boundary.get("fallback_returns_allowed") is not False:
        raise ValueError("fallback returns must remain forbidden")
    if boundary.get("hidden_eval_fields_used") is not False:
        raise ValueError("hidden eval fields cannot enter source conditioning")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(ROOT))
    except ValueError:
        return str(value)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
