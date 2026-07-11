#!/usr/bin/env python3
"""Private verifier preference updates for the canonical causal transformer."""

from __future__ import annotations

import ast
import hashlib
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class PreferenceArrays:
    chosen_inputs: np.ndarray
    chosen_labels: np.ndarray
    chosen_mask: np.ndarray
    rejected_inputs: np.ndarray
    rejected_labels: np.ndarray
    rejected_mask: np.ndarray

    @property
    def pair_count(self) -> int:
        return int(len(self.chosen_inputs))


def build_preference_pairs(
    tasks: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    verifier: dict[str, Any],
    *,
    max_pairs: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair exact verifier passes with hard model-generated negatives from train tasks."""

    task_by_id = {str(row.get("task_id") or ""): row for row in tasks}
    candidate_by_hash = {
        str(row.get("candidate_sha256") or ""): row
        for row in candidates
        if row.get("candidate_sha256")
    }
    traces_by_task: dict[str, list[dict[str, Any]]] = {}
    for trace in verifier.get("verification_attempt_labels", []):
        if not isinstance(trace, dict) or trace.get("phase") != "private_eval":
            continue
        candidate_hash = str(trace.get("candidate_sha256") or "")
        if candidate_hash not in candidate_by_hash:
            continue
        traces_by_task.setdefault(str(trace.get("task_id") or ""), []).append(trace)

    pairs: list[dict[str, Any]] = []
    pass_task_count = 0
    for task_id, traces in traces_by_task.items():
        accepted = [row for row in traces if row.get("passed") is True]
        rejected = [row for row in traces if row.get("passed") is not True]
        if not accepted:
            continue
        pass_task_count += 1
        if not rejected or task_id not in task_by_id:
            continue
        chosen_trace = max(
            accepted,
            key=lambda row: (
                float(row.get("verification_reward") or 0.0),
                float(row.get("rank_score") or float("-inf")),
            ),
        )
        rejected_trace = max(
            rejected,
            key=lambda row: (
                float(row.get("verification_reward") or 0.0),
                float(row.get("rank_score") or float("-inf")),
            ),
        )
        chosen = candidate_by_hash[str(chosen_trace["candidate_sha256"])]
        rejected_candidate = candidate_by_hash[str(rejected_trace["candidate_sha256"])]
        pairs.append(
            {
                "pair_id": stable_hash(
                    f"{task_id}:{chosen.get('candidate_sha256')}:{rejected_candidate.get('candidate_sha256')}"
                )[:20],
                "task_id": task_id,
                "task": task_by_id[task_id],
                "chosen": chosen,
                "rejected": rejected_candidate,
                "chosen_reward": float(chosen_trace.get("verification_reward") or 0.0),
                "rejected_reward": float(rejected_trace.get("verification_reward") or 0.0),
                "chosen_stage": str(chosen_trace.get("verification_stage") or ""),
                "rejected_stage": str(rejected_trace.get("verification_stage") or ""),
                "chosen_semantic_ir_state": str(chosen_trace.get("semantic_ir_state") or ""),
                "rejected_semantic_ir_state": str(rejected_trace.get("semantic_ir_state") or ""),
                "public_training_rows_written": 0,
                "external_inference_calls": 0,
                "fallback_return_count": 0,
            }
        )
    pairs.sort(key=lambda row: stable_hash(f"{seed}:{row['pair_id']}"))
    selected = pairs[: max(0, max_pairs)]
    return selected, {
        "task_count": len(tasks),
        "candidate_count": len(candidates),
        "verifier_pass_task_count": pass_task_count,
        "eligible_pair_count": len(pairs),
        "selected_pair_count": len(selected),
        "mean_reward_gap": round(
            sum(row["chosen_reward"] - row["rejected_reward"] for row in selected)
            / max(1, len(selected)),
            6,
        ),
        "semantic_ir_ready_pair_count": sum(
            row["chosen_semantic_ir_state"] == "READY"
            and row["rejected_semantic_ir_state"] == "READY"
            for row in selected
        ),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def reward_removed_pairs(
    pairs: list[dict[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove verifier direction by comparing every candidate with itself."""

    ordered = sorted(pairs, key=lambda row: stable_hash(f"{seed}:control:{row['pair_id']}"))
    control: list[dict[str, Any]] = []
    for row in ordered:
        item = dict(row)
        item["rejected"] = item["chosen"]
        item["rejected_reward"] = item["chosen_reward"]
        item["rejected_stage"] = item["chosen_stage"]
        control.append(item)
    return control, {
        "pair_count": len(control),
        "zero_reward_pair_count": len(control),
        "all_pair_margins_identically_zero": True,
        "verifier_direction_available_to_control": False,
    }


def encode_preference_arrays(
    pairs: list[dict[str, Any]],
    *,
    encode_examples: Callable[[list[dict[str, Any]]], tuple[np.ndarray, np.ndarray, np.ndarray]],
    visible_source: Callable[[dict[str, Any]], str],
) -> PreferenceArrays:
    chosen_examples = []
    rejected_examples = []
    for pair in pairs:
        task = pair["task"]
        source = visible_source(task)
        chosen_examples.append({"source_text": source, "body": function_body(pair["chosen"]["code"])})
        rejected_examples.append({"source_text": source, "body": function_body(pair["rejected"]["code"])})
    chosen = encode_examples(chosen_examples)
    rejected = encode_examples(rejected_examples)
    if len(chosen[0]) != len(pairs) or len(rejected[0]) != len(pairs):
        raise ValueError("preference candidates must encode without truncation or unknown tokens")
    return PreferenceArrays(
        chosen_inputs=chosen[0],
        chosen_labels=chosen[1],
        chosen_mask=chosen[2],
        rejected_inputs=rejected[0],
        rejected_labels=rejected[1],
        rejected_mask=rejected[2],
    )


def train_dpo(
    policy_model: Any,
    reference_model: Any,
    arrays: PreferenceArrays,
    *,
    optimizer_steps: int,
    batch_size: int,
    learning_rate: float,
    beta: float,
    gradient_clip_norm: float,
    seed: int,
    mx: Any,
    nn: Any,
    optim: Any,
) -> dict[str, Any]:
    if arrays.pair_count <= 0:
        return {
            "state": "TYPED_NO_REWARD_PAIRS",
            "optimizer_steps": 0,
            "pair_count": 0,
            "typed_failure": "no_private_verifier_preference_pairs",
        }
    matrices = [
        mx.array(value, dtype=mx.float32 if "mask" in name else mx.int32)
        for name, value in (
            ("chosen_inputs", arrays.chosen_inputs),
            ("chosen_labels", arrays.chosen_labels),
            ("chosen_mask", arrays.chosen_mask),
            ("rejected_inputs", arrays.rejected_inputs),
            ("rejected_labels", arrays.rejected_labels),
            ("rejected_mask", arrays.rejected_mask),
        )
    ]
    mx.eval(*matrices)
    reference_model.eval()
    policy_model.train()
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.0)

    def objective(model: Any, *batch: Any) -> Any:
        return dpo_loss(model, reference_model, *batch, beta=beta, mx=mx, nn=nn)

    loss_and_grad = nn.value_and_grad(policy_model, objective)
    initial_margin = mean_dpo_margin(policy_model, reference_model, *matrices, mx=mx, nn=nn)
    losses: list[float] = []
    started = time.perf_counter()
    order = list(range(arrays.pair_count))
    steps = 0
    epoch = 0
    while steps < optimizer_steps:
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), max(1, batch_size)):
            if steps >= optimizer_steps:
                break
            indices = mx.array(order[start : start + max(1, batch_size)], dtype=mx.int32)
            batch = [matrix[indices] for matrix in matrices]
            loss, grads = loss_and_grad(policy_model, *batch)
            grads, grad_norm = optim.clip_grad_norm(grads, gradient_clip_norm)
            optimizer.update(policy_model, grads)
            mx.eval(policy_model.parameters(), optimizer.state, loss, grad_norm)
            losses.append(float(loss.item()))
            steps += 1
        epoch += 1
    policy_model.eval()
    final_margin = mean_dpo_margin(policy_model, reference_model, *matrices, mx=mx, nn=nn)
    return {
        "state": "TRAINED",
        "objective": "length_normalized_dpo",
        "pair_count": arrays.pair_count,
        "optimizer_steps": steps,
        "beta": beta,
        "learning_rate": learning_rate,
        "mean_loss": round(sum(losses) / max(1, len(losses)), 6),
        "final_loss": round(losses[-1], 6) if losses else None,
        "initial_mean_preference_margin": round(initial_margin, 6),
        "final_mean_preference_margin": round(final_margin, 6),
        "preference_margin_delta": round(final_margin - initial_margin, 6),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def dpo_loss(
    policy_model: Any,
    reference_model: Any,
    chosen_inputs: Any,
    chosen_labels: Any,
    chosen_mask: Any,
    rejected_inputs: Any,
    rejected_labels: Any,
    rejected_mask: Any,
    *,
    beta: float,
    mx: Any,
    nn: Any,
) -> Any:
    margin = dpo_margin(
        policy_model,
        reference_model,
        chosen_inputs,
        chosen_labels,
        chosen_mask,
        rejected_inputs,
        rejected_labels,
        rejected_mask,
        mx=mx,
        nn=nn,
    )
    return mx.mean(mx.logaddexp(mx.array(0.0, dtype=mx.float32), -float(beta) * margin))


def mean_dpo_margin(
    policy_model: Any,
    reference_model: Any,
    chosen_inputs: Any,
    chosen_labels: Any,
    chosen_mask: Any,
    rejected_inputs: Any,
    rejected_labels: Any,
    rejected_mask: Any,
    *,
    mx: Any,
    nn: Any,
) -> float:
    margin = dpo_margin(
        policy_model,
        reference_model,
        chosen_inputs,
        chosen_labels,
        chosen_mask,
        rejected_inputs,
        rejected_labels,
        rejected_mask,
        mx=mx,
        nn=nn,
    )
    mx.eval(margin)
    return float(mx.mean(margin).item())


def dpo_margin(
    policy_model: Any,
    reference_model: Any,
    chosen_inputs: Any,
    chosen_labels: Any,
    chosen_mask: Any,
    rejected_inputs: Any,
    rejected_labels: Any,
    rejected_mask: Any,
    *,
    mx: Any,
    nn: Any,
) -> Any:
    policy_chosen = sequence_log_probability(
        policy_model, chosen_inputs, chosen_labels, chosen_mask, mx=mx, nn=nn
    )
    policy_rejected = sequence_log_probability(
        policy_model, rejected_inputs, rejected_labels, rejected_mask, mx=mx, nn=nn
    )
    reference_chosen = mx.stop_gradient(
        sequence_log_probability(
            reference_model, chosen_inputs, chosen_labels, chosen_mask, mx=mx, nn=nn
        )
    )
    reference_rejected = mx.stop_gradient(
        sequence_log_probability(
            reference_model, rejected_inputs, rejected_labels, rejected_mask, mx=mx, nn=nn
        )
    )
    return (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)


def sequence_log_probability(
    model: Any, inputs: Any, labels: Any, mask: Any, *, mx: Any, nn: Any
) -> Any:
    logits, _cache = model(inputs)
    log_probability = nn.log_softmax(logits, axis=-1)
    selected = mx.take_along_axis(log_probability, labels[..., None], axis=-1).squeeze(-1)
    denominator = mx.maximum(mx.sum(mask, axis=-1), mx.array(1.0, dtype=mx.float32))
    return mx.sum(selected * mask, axis=-1) / denominator


def function_body(code: str) -> str:
    tree = ast.parse(str(code))
    function = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if function is None or not function.body:
        raise ValueError("preference candidate must contain a non-empty function body")
    return "\n".join(ast.unparse(statement) for statement in function.body)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()
