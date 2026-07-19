"""Autoregressive execution qualification for the canonical KERC candidate.

This is a learnability and integration canary, not a utility benchmark. It
selects one coherent compiler/core/renderer chain from governed source-bound
training rows, trains the production-shape model on those rows, and requires
the real no-fallback execution route to replay the chain after checkpoint load.
"""

from __future__ import annotations

import copy
import json
import os
import resource
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from kernel_english_protocol import (
    KERC_HIERARCHICAL_COMPILER_POLICY,
    KernelProtocolFault,
    canonical_json,
    compiler_input_from_source,
    learned_answer_packet_view,
    learned_prior_claim_context_view,
    materialize_learned_answer_packet,
    merge_hierarchical_answer_packets,
    stable_hash,
)
from moecot_language_arm_training import (
    build_source_to_target_lookup,
    generate_kerc_pipeline_text,
    materialize_target_supervision,
    publish_optimizer,
    sha256_file,
    target_copy_identity_ranges,
)
from standard_causal_transformer_model import CausalTransformerConfig, build_model
from standard_causal_transformer_survival import causal_loss
import vcm_semantic_memory


PIPELINE_OBJECTIVES = (
    "surface_to_kernel_program_v1",
    "kernel_program_to_answer_packet_v1",
    "answer_packet_to_surface_v1",
)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _artifact_path(target: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    artifact = (target.get("kernel_english_artifacts") or {}).get("private_train") or {}
    path = Path(str(artifact.get("path") or ""))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.is_file() or sha256_file(path) != str(artifact.get("sha256") or ""):
        raise ValueError("KERC autoregressive qualification artifact identity mismatch")
    return path, artifact


def _chunk_index(row: dict[str, Any]) -> int:
    transport = row.get("kerc_hierarchical_transport") or {}
    return int(transport.get("chunk_index") or 0)


def _select_chain(
    target: dict[str, Any], *, profile: str = "protected_single"
) -> dict[str, Any]:
    path, artifact = _artifact_path(target)

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    observed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            observed += 1
            objective = str(row.get("objective") or "")
            source_id = str(row.get("source_record_sha256") or "")
            if objective not in PIPELINE_OBJECTIVES or not source_id:
                continue
            grouped.setdefault(source_id, {}).setdefault(objective, []).append(row)
    if observed != int(artifact.get("row_count") or 0):
        raise ValueError("KERC autoregressive qualification source row count changed")

    candidates: list[tuple[int, str, str, list[dict[str, Any]], dict[str, Any]]] = []
    for source_id, objective_rows in grouped.items():
        if set(objective_rows) != set(PIPELINE_OBJECTIVES):
            continue
        compiler_rows = sorted(
            objective_rows[PIPELINE_OBJECTIVES[0]],
            key=lambda row: (_chunk_index(row), str(row.get("row_id") or "")),
        )
        core_rows = sorted(
            objective_rows[PIPELINE_OBJECTIVES[1]],
            key=lambda row: (_chunk_index(row), str(row.get("row_id") or "")),
        )
        renderer_rows = objective_rows[PIPELINE_OBJECTIVES[2]]
        compiler_prompt = json.loads(str(compiler_rows[0]["prompt"]))
        compiler_target = json.loads(str(compiler_rows[0]["target"]))
        source = str(compiler_prompt.get("source_surface") or "")
        protected_spans = compiler_target.get("protected_objects")
        if (
            not source
            or compiler_prompt.get("protected_objects")
            or compiler_prompt.get("interaction")
            or not isinstance(protected_spans, list)
            or not protected_spans
        ):
            continue
        hrl_state = vcm_semantic_memory.create_hierarchical_residual_state(
            "kerc-autoregressive-qualification",
            scope={
                "user": "local-evaluation",
                "project": "theseus",
                "conversation": "kerc-autoregressive-qualification",
                "privacy": "private_local",
            },
        )
        first_replay_prompt = canonical_json(
            {
                **compiler_input_from_source(source, hrl_state=hrl_state),
                "hierarchical_compiler": {
                    "policy": KERC_HIERARCHICAL_COMPILER_POLICY,
                    "chunk_index": 0,
                    "prior_node_count": 0,
                    "previous_program": None,
                    "accumulated_program_sha256": stable_hash([]),
                },
            }
        )
        if first_replay_prompt != str(compiler_rows[0]["prompt"]):
            continue

        if profile == "protected_single":
            terminal_compilers = [
                row
                for row in compiler_rows
                if (json.loads(str(row["target"])).get("hierarchical_compiler") or {}).get(
                    "continuation"
                )
                is False
                and _chunk_index(row) == 0
            ]
            if not terminal_compilers:
                continue
            compiler = min(
                terminal_compilers,
                key=lambda row: (
                    len(str(row["prompt"])) + len(str(row["target"])),
                    str(row["row_id"]),
                ),
            )
            core = min(
                core_rows,
                key=lambda row: (
                    len(str(row["prompt"])) + len(str(row["target"])),
                    str(row["row_id"]),
                ),
            )
            renderer = min(
                (row for row in renderer_rows if str(row.get("target") or "") == source),
                key=lambda row: (
                    len(str(row["prompt"])) + len(str(row["target"])),
                    str(row["row_id"]),
                ),
                default=None,
            )
            if renderer is None:
                continue
            rows = [compiler, core, renderer]
            properties = {
                "compiler_chunk_count": 1,
                "core_chunk_count": 1,
                "compiler_continuation_prompt_count": 0,
                "dependency_claim_count": 0,
            }
        elif profile == "hierarchical_stateful":
            compiler_count = int(
                (compiler_rows[0].get("kerc_hierarchical_transport") or {}).get(
                    "chunk_count"
                )
                or 0
            )
            core_count = int(
                (core_rows[0].get("kerc_hierarchical_transport") or {}).get(
                    "chunk_count"
                )
                or 0
            )
            if (
                compiler_count <= 1
                or core_count <= 1
                or len(compiler_rows) != compiler_count
                or len(core_rows) != core_count
                or [_chunk_index(row) for row in compiler_rows]
                != list(range(compiler_count))
                or [_chunk_index(row) for row in core_rows] != list(range(core_count))
            ):
                continue
            compiler_targets = [json.loads(str(row["target"])) for row in compiler_rows]
            if (
                any(
                    target_row.get("protected_objects") != protected_spans
                    for target_row in compiler_targets[1:]
                )
                or any(
                    (json.loads(str(row["prompt"])).get("hierarchical_compiler") or {}).get(
                        "previous_program"
                    )
                    != compiler_targets[index - 1].get("program")
                    for index, row in enumerate(compiler_rows[1:], start=1)
                )
                or any(
                    (target_row.get("hierarchical_compiler") or {}).get("continuation")
                    is not (index < compiler_count - 1)
                    for index, target_row in enumerate(compiler_targets)
                )
            ):
                continue
            dependency_claim_count = max(
                int(
                    (
                        json.loads(str(row["prompt"])).get("prior_claims") or {}
                    ).get("claim_count")
                    or 0
                )
                for row in core_rows
            )
            if dependency_claim_count <= 0:
                continue
            renderer = min(
                (row for row in renderer_rows if str(row.get("target") or "") == source),
                key=lambda row: (
                    len(str(row["prompt"])) + len(str(row["target"])),
                    str(row["row_id"]),
                ),
                default=None,
            )
            if renderer is None:
                continue
            renderer_packet = materialize_learned_answer_packet(
                json.loads(str(renderer["prompt"]))["answer_packet"]
            )
            partials = [
                materialize_learned_answer_packet(json.loads(str(row["target"])))
                for row in core_rows
            ]
            try:
                merged = merge_hierarchical_answer_packets(
                    partials,
                    expected_chunk_count=core_count,
                    claim_order=[
                        str(claim["claim_id"])
                        for claim in renderer_packet["claims"]
                    ],
                )
            except KernelProtocolFault:
                continue
            if learned_answer_packet_view(merged) != json.loads(str(renderer["prompt"]))[
                "answer_packet"
            ]:
                continue
            rows = [*compiler_rows, *core_rows, renderer]
            properties = {
                "compiler_chunk_count": compiler_count,
                "core_chunk_count": core_count,
                "compiler_continuation_prompt_count": compiler_count - 1,
                "dependency_claim_count": dependency_claim_count,
            }
        else:
            raise ValueError(f"unknown KERC autoregressive profile: {profile}")
        total_size = sum(
            len(str(row["prompt"])) + len(str(row["target"])) for row in rows
        )
        candidates.append((total_size, source_id, source, rows, properties))
    if not candidates:
        raise ValueError("no coherent source-bound KERC execution chain is available")
    total_size, source_id, source, rows, properties = min(
        candidates, key=lambda row: (row[0], row[1])
    )
    return {
        "profile": profile,
        "source_record_sha256": source_id,
        "source": source,
        "rows": rows,
        "protected_span_count": len(
            json.loads(str(rows[0]["target"]))["protected_objects"]
        ),
        "protected_object_types": sorted(
            str(row["object_type"])
            for row in json.loads(str(rows[0]["target"]))["protected_objects"]
        ),
        "candidate_count": len(candidates),
        "selected_total_prompt_target_characters": total_size,
        **properties,
    }


def _dense_positive_stage(
    stage: Any, objectives: list[str], row_ids: list[str]
) -> Any:
    indices = [
        index for index in range(len(stage.mask)) if int(stage.mask[index].sum()) > 0
    ]
    if len(indices) != len(objectives) or len(objectives) != len(row_ids):
        raise ValueError("KERC chain positive-row identity changed during materialization")
    width = max(len(stage.inputs[index]) for index in indices)

    def padded(rows: Any, dtype: Any) -> np.ndarray:
        result = np.zeros((len(indices), width), dtype=dtype)
        for local, index in enumerate(indices):
            row = np.asarray(rows[index], dtype=dtype)
            result[local, : len(row)] = row
        return result

    return SimpleNamespace(
        inputs=padded(stage.inputs, np.int32),
        labels=padded(stage.labels, np.int32),
        loss_mask=padded(stage.loss_mask, np.float32),
        objectives=tuple(objectives),
        row_ids=tuple(row_ids),
    )


def _build_model(
    target: dict[str, Any], copy_lookup: np.ndarray, *, mx: Any, nn: Any
) -> Any:
    return build_model(
        CausalTransformerConfig(
            vocab_size=int(target["vocab_size"]), **target["model"]
        ),
        mx=mx,
        nn=nn,
        state_role_lookup=None,
        source_to_target_lookup=copy_lookup,
    )


def _token_metrics(
    model: Any, stage: Any, *, mx: Any
) -> tuple[dict[str, Any], list[float], list[list[float]]]:
    correct = 0
    count = 0
    by_objective: dict[str, Any] = {}
    by_row: dict[str, Any] = {}
    expected_logits: list[float] = []
    row_expected_logits: list[list[float]] = []
    model.eval()
    for index, objective in enumerate(stage.objectives):
        active = np.asarray(stage.loss_mask[index]).astype(bool)
        width = int(np.flatnonzero(active)[-1] + 1)
        active = active[:width]
        x = mx.array(stage.inputs[index : index + 1, :width], dtype=mx.int32)
        logits, _cache = model(x)
        mx.eval(logits)
        values = np.asarray(logits[0])
        labels = np.asarray(stage.labels[index, :width])
        predicted = values.argmax(axis=-1)
        row_correct = int((predicted[active] == labels[active]).sum())
        row_count = int(active.sum())
        correct += row_correct
        count += row_count
        selected_logits = (
            values[np.arange(width)[active], labels[active]].astype(float).tolist()
        )
        expected_logits.extend(selected_logits)
        row_expected_logits.append(selected_logits)
        objective_receipt = by_objective.setdefault(
            objective,
            {"target_position_count": 0, "correct_position_count": 0},
        )
        objective_receipt["target_position_count"] += row_count
        objective_receipt["correct_position_count"] += row_correct
        by_row[str(stage.row_ids[index])] = {
            "objective": objective,
            "target_position_count": row_count,
            "teacher_forced_token_accuracy": row_correct / max(1, row_count),
        }
    for receipt in by_objective.values():
        receipt["teacher_forced_token_accuracy"] = receipt[
            "correct_position_count"
        ] / max(1, receipt["target_position_count"])
        del receipt["correct_position_count"]
    return {
        "target_position_count": count,
        "teacher_forced_token_accuracy": correct / max(1, count),
        "by_objective": by_objective,
        "by_row": by_row,
    }, expected_logits, row_expected_logits


def _gradient_receipt(
    gradients: Any, objective: str, *, mlx_utils: Any, mx: Any
) -> dict[str, Any]:
    stage_index = PIPELINE_OBJECTIVES.index(objective) + 1
    groups = {
        "shared_trunk": ("token_embedding", "source_layers", "layers"),
        "stage_adapter": (f"kerc_stage_adapters.{stage_index}",),
        "stage_output": (
            "kerc_surface_output"
            if objective == "answer_packet_to_surface_v1"
            else "kerc_kernel_output",
        ),
    }
    flat = list(mlx_utils.tree_flatten(gradients))
    mx.eval(*(value for _name, value in flat))
    receipt: dict[str, Any] = {}
    for group, fragments in groups.items():
        selected = [
            np.asarray(value, dtype=np.float64)
            for name, value in flat
            if any(fragment in name for fragment in fragments)
        ]
        receipt[group] = {
            "tensor_count": len(selected),
            "l2_norm": float(
                np.sqrt(sum(float(np.square(value).sum()) for value in selected))
            ),
        }
    return receipt


def _max_delta(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return float("inf")
    return max((abs(a - b) for a, b in zip(left, right)), default=0.0)


def _mean_delta(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return float("inf")
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


def _materialize_rows(
    rows: list[dict[str, Any]],
    *,
    label: str,
    output_root: Path,
    training: dict[str, Any],
    target: dict[str, Any],
    base: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[Any, Any, Path]:
    path = output_root / f"{label}.private.jsonl"
    _atomic_jsonl(path, rows)
    candidate = copy.deepcopy(target)
    candidate["kernel_english_artifacts"] = {
        "private_train": {
            "path": str(path),
            "sha256": sha256_file(path),
            "row_count": len(rows),
            "unique_record_count": 1,
        }
    }
    materialized = materialize_target_supervision(
        training,
        base,
        candidate,
        metadata=metadata,
        artifact_field="kernel_english_artifacts",
        receipt_policy="project_theseus_kerc_autoregressive_chain_arrays_v2",
        maximum_sequence_tokens=int(
            training["kernel_english_training"]["maximum_sequence_tokens"]
        ),
        objective_filter=PIPELINE_OBJECTIVES,
    )
    stage = _dense_positive_stage(
        materialized,
        [str(row["objective"]) for row in rows],
        [str(row["row_id"]) for row in rows],
    )
    return materialized, stage, path


def _state_interventions(
    model: Any,
    chain: dict[str, Any],
    reference_row_logits: list[list[float]],
    *,
    output_root: Path,
    training: dict[str, Any],
    target: dict[str, Any],
    base: dict[str, Any],
    metadata: dict[str, Any],
    minimum_delta: float,
    mx: Any,
) -> dict[str, Any]:
    specifications = (
        (
            "compiler_prior_program_removed",
            PIPELINE_OBJECTIVES[0],
            lambda prompt: int(
                (prompt.get("hierarchical_compiler") or {}).get("chunk_index") or 0
            )
            > 0,
        ),
        (
            "core_prior_claims_removed",
            PIPELINE_OBJECTIVES[1],
            lambda prompt: int((prompt.get("prior_claims") or {}).get("claim_count") or 0)
            > 0,
        ),
    )
    receipts: dict[str, Any] = {}
    for name, objective, predicate in specifications:
        rows = copy.deepcopy(chain["rows"])
        selected_index = -1
        original_prompt = ""
        intervened_prompt = ""
        for index, row in enumerate(rows):
            if row["objective"] != objective:
                continue
            prompt = json.loads(str(row["prompt"]))
            if not predicate(prompt):
                continue
            selected_index = index
            original_prompt = str(row["prompt"])
            if name == "compiler_prior_program_removed":
                contract = prompt["hierarchical_compiler"]
                contract["previous_program"] = None
                contract["prior_node_count"] = 0
                contract["accumulated_program_sha256"] = stable_hash([])
            else:
                prompt["prior_claims"] = learned_prior_claim_context_view([])
            intervened_prompt = canonical_json(prompt)
            row["prompt"] = intervened_prompt
            row["prompt_sha256"] = stable_hash(intervened_prompt.encode("utf-8"))
            break
        if selected_index < 0:
            receipts[name] = {
                "passed": False,
                "reason": "required_stateful_row_missing",
            }
            continue
        _materialized, stage, artifact = _materialize_rows(
            rows,
            label=f"intervention_{name}",
            output_root=output_root,
            training=training,
            target=target,
            base=base,
            metadata=metadata,
        )
        _metrics, _flat_logits, row_logits = _token_metrics(model, stage, mx=mx)
        maximum_delta = _max_delta(
            reference_row_logits[selected_index], row_logits[selected_index]
        )
        mean_delta = _mean_delta(
            reference_row_logits[selected_index], row_logits[selected_index]
        )
        receipts[name] = {
            "passed": bool(maximum_delta > minimum_delta),
            "objective": objective,
            "row_id": str(rows[selected_index]["row_id"]),
            "original_prompt_sha256": stable_hash(original_prompt.encode("utf-8")),
            "intervened_prompt_sha256": stable_hash(
                intervened_prompt.encode("utf-8")
            ),
            "target_expected_logit_maximum_delta": maximum_delta,
            "target_expected_logit_mean_absolute_delta": mean_delta,
            "minimum_required_delta": minimum_delta,
            "intervention_artifact": str(artifact),
            "claim_scope": "causal_use_of_generated_prior_state_not_utility",
        }
    return receipts


def _run_profile(
    profile: str,
    qualification: dict[str, Any],
    config: dict[str, Any],
    training: dict[str, Any],
    target: dict[str, Any],
    base: dict[str, Any],
    metadata: dict[str, Any],
    output_root: Path,
    *,
    mx: Any,
    nn: Any,
    optim: Any,
    mlx_utils: Any,
) -> dict[str, Any]:
    output_root = output_root / profile
    output_root.mkdir(parents=True, exist_ok=True)
    chain = _select_chain(target, profile=profile)
    materialized, stage, chain_path = _materialize_rows(
        chain["rows"],
        label="autoregressive_pipeline_chain",
        output_root=output_root,
        training=training,
        target=target,
        base=base,
        metadata=metadata,
    )
    copy_lookup = build_source_to_target_lookup(
        base,
        metadata,
        vocab_size=int(target["vocab_size"]),
        identity_ranges=target_copy_identity_ranges(target),
    )
    mx.random.seed(int(config["seed"]) + 17)
    model = _build_model(target, copy_lookup, mx=mx, nn=nn)
    optimizer = optim.AdamW(
        learning_rate=float(qualification["learning_rate"]),
        weight_decay=float(qualification["weight_decay"]),
    )
    loss_and_grad = nn.value_and_grad(model, causal_loss)
    before, _before_logits, _before_row_logits = _token_metrics(model, stage, mx=mx)
    gradients_by_stage: dict[str, Any] = {}
    for index, objective in enumerate(stage.objectives):
        active = np.asarray(stage.loss_mask[index]).astype(bool)
        width = int(np.flatnonzero(active)[-1] + 1)
        _loss, gradients = loss_and_grad(
            model,
            mx.array(stage.inputs[index : index + 1, :width], dtype=mx.int32),
            mx.array(stage.labels[index : index + 1, :width], dtype=mx.int32),
            mx.array(stage.loss_mask[index : index + 1, :width], dtype=mx.float32),
            mx,
            nn,
        )
        gradients_by_stage[str(stage.row_ids[index])] = {
            "objective": objective,
            "receipt": _gradient_receipt(
                gradients, objective, mlx_utils=mlx_utils, mx=mx
            ),
        }

    curve = [{"step": 0, **before}]
    progress_path = output_root / "kerc_autoregressive_qualification_progress.json"
    _atomic_json(
        progress_path,
        {
            "policy": "project_theseus_kerc_autoregressive_qualification_progress_v1",
            "state": "RUNNING",
            "step": 0,
            "maximum_steps": int(qualification["maximum_steps"]),
            "metrics": before,
        },
    )
    losses: list[float] = []
    target_positions = 0
    started = time.perf_counter()
    completed_steps = 0
    model.train()
    for step in range(1, int(qualification["maximum_steps"]) + 1):
        index = (step - 1) % len(stage.inputs)
        active = np.asarray(stage.loss_mask[index]).astype(bool)
        width = int(np.flatnonzero(active)[-1] + 1)
        loss, gradients = loss_and_grad(
            model,
            mx.array(stage.inputs[index : index + 1, :width], dtype=mx.int32),
            mx.array(stage.labels[index : index + 1, :width], dtype=mx.int32),
            mx.array(stage.loss_mask[index : index + 1, :width], dtype=mx.float32),
            mx,
            nn,
        )
        gradients, norm = optim.clip_grad_norm(
            gradients, float(qualification["gradient_clip_norm"])
        )
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss, norm)
        losses.append(float(loss.item()))
        target_positions += int(active.sum())
        completed_steps = step
        if step % int(qualification["evaluation_interval"]) == 0:
            metrics, _logits, _row_logits = _token_metrics(model, stage, mx=mx)
            curve.append({"step": step, **metrics})
            _atomic_json(
                progress_path,
                {
                    "policy": "project_theseus_kerc_autoregressive_qualification_progress_v1",
                    "state": "RUNNING",
                    "step": step,
                    "maximum_steps": int(qualification["maximum_steps"]),
                    "latest_loss": losses[-1],
                    "elapsed_seconds": time.perf_counter() - started,
                    "target_tokens_per_second": target_positions
                    / max(time.perf_counter() - started, 1e-9),
                    "peak_rss_bytes": int(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    ),
                    "metrics": metrics,
                },
            )
            model.train()
            if step >= int(qualification["minimum_steps"]) and all(
                row["teacher_forced_token_accuracy"]
                >= float(qualification["minimum_stage_token_accuracy"])
                for row in metrics["by_row"].values()
            ):
                break
    elapsed = time.perf_counter() - started
    after, after_logits, after_row_logits = _token_metrics(model, stage, mx=mx)
    state_interventions = (
        _state_interventions(
            model,
            chain,
            after_row_logits,
            output_root=output_root,
            training=training,
            target=target,
            base=base,
            metadata=metadata,
            minimum_delta=float(
                qualification.get("minimum_state_intervention_logit_delta") or 1e-7
            ),
            mx=mx,
        )
        if profile == "hierarchical_stateful"
        else {}
    )
    checkpoint = output_root / "kerc_autoregressive_qualification_weights.safetensors"
    optimizer_path = (
        output_root / "kerc_autoregressive_qualification_optimizer.safetensors"
    )
    model.save_weights(str(checkpoint))
    publish_optimizer(mx, mlx_utils, optimizer, optimizer_path)

    generation_args = {
        "target": target,
        "max_tokens": int(qualification["decode_max_target_tokens"]),
        "max_source_tokens": int(
            training["kernel_english_training"]["maximum_sequence_tokens"]
        ),
        "beam_width": int(qualification["beam_width"]),
        "branching_factor": int(qualification["branching_factor"]),
        "length_penalty": 1.0,
        "interaction_id": f"kerc-autoregressive-qualification-{profile}",
        "mx": mx,
    }
    source_vocab = dict(metadata["source_vocab"])
    target_vocab = dict(metadata["target_vocab"])
    generated, execution = generate_kerc_pipeline_text(
        model, chain["source"], source_vocab, target_vocab, base, **generation_args
    )
    reloaded = _build_model(target, copy_lookup, mx=mx, nn=nn)
    reloaded.load_weights(str(checkpoint), strict=True)
    mx.eval(reloaded.parameters())
    reload_metrics, reload_logits, _reload_row_logits = _token_metrics(
        reloaded, stage, mx=mx
    )
    reloaded_generated, reloaded_execution = generate_kerc_pipeline_text(
        reloaded, chain["source"], source_vocab, target_vocab, base, **generation_args
    )

    gradient_floor = float(qualification["minimum_parameter_family_gradient_l2"])
    gradients_pass = all(
        int(family["tensor_count"]) > 0 and float(family["l2_norm"]) > gradient_floor
        for stage_receipt in gradients_by_stage.values()
        for family in stage_receipt["receipt"].values()
    )
    teacher_forced_pass = all(
        row["teacher_forced_token_accuracy"]
        >= float(qualification["minimum_stage_token_accuracy"])
        for row in after["by_row"].values()
    )
    expected_stage_objectives = [
        *([PIPELINE_OBJECTIVES[0]] * int(chain["compiler_chunk_count"])),
        *([PIPELINE_OBJECTIVES[1]] * int(chain["core_chunk_count"])),
        PIPELINE_OBJECTIVES[2],
        *([PIPELINE_OBJECTIVES[0]] * int(chain["compiler_chunk_count"])),
        *([PIPELINE_OBJECTIVES[1]] * int(chain["core_chunk_count"])),
    ]
    execution_pass = bool(
        execution.get("state") == "GREEN"
        and generated == chain["source"]
        and int(execution.get("fallback_return_count") or 0) == 0
        and execution.get("stage_objectives") == expected_stage_objectives
    )
    protected_span_pass = bool(
        execution_pass
        and int(chain["protected_span_count"]) > 0
        and int(execution.get("learned_protected_object_count") or 0)
        == int(chain["protected_span_count"])
        and int(execution.get("recompiled_protected_object_count") or 0)
        == int(chain["protected_span_count"])
        and execution.get("protected_span_route")
        == "learned_compiler_span_output_then_exact_source_materialization"
    )
    reload_delta = _max_delta(after_logits, reload_logits)
    reload_pass = bool(
        reload_delta <= float(qualification["maximum_checkpoint_reload_logit_delta"])
        and reloaded_generated == generated
        and reloaded_execution.get("state") == execution.get("state")
    )
    hierarchy_pass = bool(
        profile != "hierarchical_stateful"
        or (
            int(chain["compiler_chunk_count"]) > 1
            and int(chain["core_chunk_count"]) > 1
            and int(chain["compiler_continuation_prompt_count"]) > 0
            and int(chain["dependency_claim_count"]) > 0
            and len(state_interventions) == 2
            and all(row.get("passed") is True for row in state_interventions.values())
        )
    )
    passed = (
        gradients_pass
        and teacher_forced_pass
        and execution_pass
        and protected_span_pass
        and reload_pass
        and hierarchy_pass
    )
    _atomic_json(
        progress_path,
        {
            "policy": "project_theseus_kerc_autoregressive_qualification_progress_v1",
            "state": "GREEN" if passed else "INCONCLUSIVE_IMPLEMENTATION",
            "step": completed_steps,
            "maximum_steps": int(qualification["maximum_steps"]),
            "metrics": after,
            "gates": {
                "stage_gradient_paths": gradients_pass,
                "stage_teacher_forced_overfit": teacher_forced_pass,
                "real_no_fallback_autoregressive_route": execution_pass,
                "learned_protected_span_route": protected_span_pass,
                "checkpoint_reload_same_route": reload_pass,
                "hierarchical_generated_state_is_causal": hierarchy_pass,
            },
        },
    )
    return {
        "policy": "project_theseus_kerc_autoregressive_qualification_v1",
        "state": "GREEN" if passed else "INCONCLUSIVE_IMPLEMENTATION",
        "claim_scope": "production_shape_autoregressive_learnability_and_route_integration_only",
        "negative_verdict_authority": "NONE",
        "selection": {key: value for key, value in chain.items() if key != "rows"}
        | {
            "row_ids": [str(row["row_id"]) for row in chain["rows"]],
            "chain_artifact": str(chain_path),
            "chain_artifact_sha256": sha256_file(chain_path),
        },
        "materialization": materialized.receipt,
        "training": {
            "optimizer_steps": completed_steps,
            "maximum_steps": int(qualification["maximum_steps"]),
            "target_positions": target_positions,
            "wall_seconds": elapsed,
            "target_tokens_per_second": target_positions / max(elapsed, 1e-9),
            "first_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "curve": curve,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "gradients_by_stage": gradients_by_stage,
        "teacher_forced_before": before,
        "teacher_forced_after": after,
        "autoregressive_execution": {
            "generated_sha256": stable_hash(generated.encode("utf-8")),
            "exact_surface_recovery": generated == chain["source"],
            "receipt": execution,
        },
        "checkpoint_reload": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "optimizer": str(optimizer_path),
            "optimizer_sha256": sha256_file(optimizer_path),
            "maximum_expected_logit_delta": reload_delta,
            "teacher_forced_metrics_equal": reload_metrics == after,
            "autoregressive_output_equal": reloaded_generated == generated,
            "execution_state_equal": reloaded_execution.get("state")
            == execution.get("state"),
        },
        "state_interventions": state_interventions,
        "gates": {
            "stage_gradient_paths": gradients_pass,
            "stage_teacher_forced_overfit": teacher_forced_pass,
            "real_no_fallback_autoregressive_route": execution_pass,
            "learned_protected_span_route": protected_span_pass,
            "checkpoint_reload_same_route": reload_pass,
            "hierarchical_generated_state_is_causal": hierarchy_pass,
        },
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "template_credit": 0,
        "learned_capability_claimed": False,
    }


def run_autoregressive_qualification(
    config: dict[str, Any],
    training: dict[str, Any],
    target: dict[str, Any],
    base: dict[str, Any],
    metadata: dict[str, Any],
    output_root: Path,
    *,
    mx: Any,
    nn: Any,
    optim: Any,
    mlx_utils: Any,
) -> dict[str, Any]:
    """Qualify simple and stateful hierarchical production routes independently."""

    qualification = config.get("autoregressive_pipeline") or {}
    if qualification.get("enabled") is not True:
        return {
            "state": "INCONCLUSIVE_IMPLEMENTATION",
            "reason": "qualification_disabled",
        }
    profile_overrides = qualification.get("profiles") or {}
    profiles: dict[str, Any] = {}
    for profile in ("protected_single", "hierarchical_stateful"):
        profile_config = {
            key: copy.deepcopy(value)
            for key, value in qualification.items()
            if key != "profiles"
        }
        profile_config.update(copy.deepcopy(profile_overrides.get(profile) or {}))
        profiles[profile] = _run_profile(
            profile,
            profile_config,
            config,
            training,
            target,
            base,
            metadata,
            output_root,
            mx=mx,
            nn=nn,
            optim=optim,
            mlx_utils=mlx_utils,
        )
        mx.clear_cache()
    passed = all(row.get("state") == "GREEN" for row in profiles.values())
    return {
        "policy": "project_theseus_kerc_autoregressive_qualification_suite_v2",
        "state": "GREEN" if passed else "INCONCLUSIVE_IMPLEMENTATION",
        "claim_scope": "production_shape_autoregressive_learnability_state_causality_and_route_integration_only",
        "negative_verdict_authority": "NONE",
        "profiles": profiles,
        "gates": {
            "protected_single_route": profiles["protected_single"].get("state")
            == "GREEN",
            "hierarchical_stateful_route": profiles["hierarchical_stateful"].get(
                "state"
            )
            == "GREEN",
        },
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "template_credit": 0,
        "learned_capability_claimed": False,
    }
