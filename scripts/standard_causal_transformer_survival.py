#!/usr/bin/env python3
"""Train and replay the standard causal-transformer survival lane on MLX."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPTS))

from code_lm_private_verifier import evaluate_private_candidates  # noqa: E402
from neural_seed_open_vocab import encode_tokens  # noqa: E402
from neural_seed_token_decoder_support import (  # noqa: E402
    body_tokens,
    decode_candidate_body_tokens,
    syntax_complete_body_prefix,
    token_allowed_by_policy,
)
from standard_causal_transformer_model import (  # noqa: E402
    CausalTransformerConfig,
    build_model,
    parameter_count,
)


DEFAULT_CONFIG = ROOT / "configs" / "standard_causal_transformer_survival.json"
DEFAULT_REPORT = ROOT / "reports" / "standard_causal_transformer_survival.json"
DEFAULT_CANDIDATES = ROOT / "reports" / "standard_causal_transformer_survival_candidates.jsonl"
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints" / "standard_causal_transformer_survival_v1"
DEFAULT_STAGE_DIR = ROOT / "runtime" / "standard_causal_transformer_survival_v1"
SOURCE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|-?\d+|\S")
PAD_ID = 0
GLOBAL_BOS_ID = 1
SOURCE_TARGET_SEPARATOR_ID = 2
SPECIAL_COUNT = 3


@dataclass
class Stage:
    pretrain_inputs: np.ndarray
    pretrain_labels: np.ndarray
    pretrain_mask: np.ndarray
    sft_inputs: np.ndarray
    sft_labels: np.ndarray
    sft_mask: np.ndarray
    eval_inputs: np.ndarray
    eval_labels: np.ndarray
    eval_mask: np.ndarray
    eval_rows: list[dict[str, Any]]
    source_vocab: dict[str, int]
    target_vocab: dict[str, int]
    summary: dict[str, Any]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=rel(DEFAULT_REPORT))
    parser.add_argument("--candidates-out", default=rel(DEFAULT_CANDIDATES))
    parser.add_argument("--checkpoint-dir", default=rel(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--stage-dir", default=rel(DEFAULT_STAGE_DIR))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force-restage", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    started = time.perf_counter()
    config = read_json(resolve(args.config))
    report, candidates = run(
        config,
        config_path=args.config,
        checkpoint_dir=resolve(args.checkpoint_dir),
        stage_dir=resolve(args.stage_dir),
        execute=args.execute,
        force_restage=args.force_restage,
        resume=args.resume,
        evaluate_only=args.evaluate_only,
        max_steps=max(0, args.max_steps),
        report_path=resolve(args.out),
        started=started,
    )
    write_json(resolve(args.out), report)
    write_jsonl(resolve(args.candidates_out), candidates)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW", "PLANNED"} else 2


def run(
    config: dict[str, Any],
    *,
    config_path: str,
    checkpoint_dir: Path,
    stage_dir: Path,
    execute: bool,
    force_restage: bool,
    resume: bool,
    evaluate_only: bool,
    max_steps: int,
    report_path: Path,
    started: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_config(config)
    if not execute:
        return planned_report(config, config_path, checkpoint_dir, stage_dir, started), []

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mlx_utils

    seed = int(config["seed"])
    random.seed(seed)
    mx.random.seed(seed)
    stage = materialize_stage(config, stage_dir=stage_dir, force=force_restage)
    vocab_size = SPECIAL_COUNT + len(stage.source_vocab) + len(stage.target_vocab)
    model_cfg = CausalTransformerConfig(vocab_size=vocab_size, **config["model"])
    model = build_model(model_cfg, mx=mx, nn=nn)
    params = parameter_count(model, mlx_utils)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "standard_causal_transformer_survival_v1.npz"
    if (resume or evaluate_only) and checkpoint.exists():
        model.load_weights(str(checkpoint))
        mx.eval(model.parameters())
    elif evaluate_only:
        raise FileNotFoundError(f"evaluation checkpoint missing: {checkpoint}")

    training_cfg = config["training"]
    pretrain_steps = required_steps(
        stage.pretrain_mask,
        int(training_cfg["batch_size"]),
        int(training_cfg["pretrain_target_token_positions"]),
    )
    sft_steps = required_steps(
        stage.sft_mask,
        int(training_cfg["batch_size"]),
        int(training_cfg["sft_target_token_positions"]),
    )
    total_steps = pretrain_steps + sft_steps
    if max_steps:
        total_steps = min(total_steps, max_steps)
    if evaluate_only:
        prior = read_json(report_path)
        prior_training = prior.get("training") if isinstance(prior.get("training"), dict) else {}
        conditioning = prior.get("conditioning") if isinstance(prior.get("conditioning"), dict) else {}
        eval_loss_before = float(prior_training.get("eval_loss_before") or float("inf"))
        phase_reports = list(prior_training.get("phases") or [])
        consumed_steps = int(prior_training.get("optimizer_steps") or 0)
        training_complete = bool(prior_training.get("complete"))
    else:
        conditioning = {}
        schedule = build_schedule(optim, mx, training_cfg, total_steps)
        optimizer = optim.AdamW(learning_rate=schedule, weight_decay=float(training_cfg["weight_decay"]))
        loss_and_grad = nn.value_and_grad(model, causal_loss)
        eval_loss_before = evaluate_loss(
            model,
            stage.eval_inputs,
            stage.eval_labels,
            stage.eval_mask,
            batch_size=int(training_cfg["batch_size"]),
            mx=mx,
            nn=nn,
        )
        heartbeat = stage_dir / "training_heartbeat.json"
        phase_reports = []
        consumed_steps = 0
        for phase_name, inputs, labels, mask, target_positions in (
            (
                "licensed_module_causal_pretraining",
                stage.pretrain_inputs,
                stage.pretrain_labels,
                stage.pretrain_mask,
                int(training_cfg["pretrain_target_token_positions"]),
            ),
            (
                "prompt_signature_body_sft",
                stage.sft_inputs,
                stage.sft_labels,
                stage.sft_mask,
                int(training_cfg["sft_target_token_positions"]),
            ),
        ):
            remaining = total_steps - consumed_steps
            if remaining <= 0:
                break
            phase_report = train_phase(
                model,
                optimizer,
                loss_and_grad,
                inputs,
                labels,
                mask,
                phase_name=phase_name,
                target_positions=target_positions,
                batch_size=int(training_cfg["batch_size"]),
                gradient_clip=float(training_cfg["gradient_clip_norm"]),
                seed=seed + len(phase_reports) * 1009,
                max_steps=remaining,
                checkpoint=checkpoint,
                checkpoint_every=max(1, int(training_cfg["checkpoint_every_steps"])),
                heartbeat=heartbeat,
                global_step_offset=consumed_steps,
                mx=mx,
                optim=optim,
            )
            phase_reports.append(phase_report)
            consumed_steps += int(phase_report["optimizer_steps"])
        model.save_weights(str(checkpoint))
        mx.eval(model.parameters())
        training_complete = not max_steps or consumed_steps >= pretrain_steps + sft_steps
    eval_loss_after = evaluate_loss(
        model,
        stage.eval_inputs,
        stage.eval_labels,
        stage.eval_mask,
        batch_size=int(training_cfg["batch_size"]),
        mx=mx,
        nn=nn,
    )
    candidates, decode_summary = generate_candidates(
        model,
        stage.eval_rows,
        stage.source_vocab,
        stage.target_vocab,
        config,
        mx=mx,
    )
    verifier = evaluate_private_candidates(stage.eval_rows, candidates)
    verifier_summary = private_verifier_summary(verifier)
    model_pass_count = int(verifier_summary.get("passed_task_count") or 0)
    trigger_state = "GREEN" if training_complete and model_pass_count > 0 else "YELLOW"
    report = {
        "policy": config["policy"],
        "created_utc": now(),
        "trigger_state": trigger_state,
        "execute": True,
        "architecture": {
            "family": "standard_decoder_only_causal_transformer",
            "attention": "RoPE_grouped_query_causal_attention",
            "normalization": "pre_norm_RMSNorm",
            "feed_forward": "SwiGLU",
            "embedding": "tied_input_output",
            "parameter_count": params,
            "config": model_cfg.__dict__,
            "backend": "mlx_apple",
        },
        "artifacts": {
            "config": config_path,
            "checkpoint": rel(checkpoint),
            "stage_dir": rel(stage_dir),
            "candidates": rel(DEFAULT_CANDIDATES),
        },
        "stage": stage.summary,
        "training": {
            "planned_pretrain_steps": pretrain_steps,
            "planned_sft_steps": sft_steps,
            "optimizer_steps": consumed_steps,
            "complete": training_complete,
            "evaluation_only_replay": evaluate_only,
            "phases": phase_reports,
            "eval_loss_before": eval_loss_before,
            "eval_loss_after": eval_loss_after,
            "eval_loss_improved": eval_loss_after < eval_loss_before,
        },
        "conditioning": conditioning,
        "decode": decode_summary,
        "private_verifier": verifier,
        "summary": {
            "family_disjoint_eval_task_count": len(stage.eval_rows),
            "candidate_count": len(candidates),
            "candidate_task_count": len({row["task_id"] for row in candidates}),
            "model_only_passed_task_count": model_pass_count,
            "model_only_pass_rate": round(model_pass_count / max(1, len(stage.eval_rows)), 6),
            "syntax_valid_candidate_count": decode_summary["syntax_valid_candidate_count"],
            "training_complete": training_complete,
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "template_renderer_router_tool_credit_count": 0,
            "open_or_pretrained_model_weights_used": False,
        },
        "gates": build_gates(stage, training_complete, eval_loss_before, eval_loss_after, decode_summary, verifier_summary),
        "score_semantics": (
            "Direct decoder-only causal model generation from prompt plus callable signature. "
            "Reversible token decoding adds no body content; no repair, template, renderer, router, "
            "tool, fallback return, public benchmark payload, or external inference is credited."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    return report, candidates


def materialize_stage(config: dict[str, Any], *, stage_dir: Path, force: bool) -> Stage:
    stage_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = stage_dir / "stage_arrays_v1.npz"
    metadata_path = stage_dir / "stage_metadata_v1.json"
    signature = stage_signature(config)
    if not force and arrays_path.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata.get("stage_signature") == signature:
            arrays = np.load(arrays_path)
            vocab_payload = read_json(resolve(config["tokenization"]["source_vocab"]))
            return Stage(
                pretrain_inputs=arrays["pretrain_inputs"],
                pretrain_labels=arrays["pretrain_labels"],
                pretrain_mask=arrays["pretrain_mask"],
                sft_inputs=arrays["sft_inputs"],
                sft_labels=arrays["sft_labels"],
                sft_mask=arrays["sft_mask"],
                eval_inputs=arrays["eval_inputs"],
                eval_labels=arrays["eval_labels"],
                eval_mask=arrays["eval_mask"],
                eval_rows=list(metadata["eval_rows"]),
                source_vocab=dict(vocab_payload["source_vocab"]),
                target_vocab=dict(vocab_payload["target_vocab"]),
                summary={**metadata["summary"], "cache_status": "hit"},
            )

    vocab_payload = read_json(resolve(config["tokenization"]["source_vocab"]))
    source_vocab = dict(vocab_payload["source_vocab"])
    target_vocab = dict(vocab_payload["target_vocab"])
    eval_rows, holdout_families = select_family_disjoint_eval(config)
    eval_prompt_hashes = {sha(str(row.get("prompt") or "")) for row in eval_rows}
    eval_body_hashes = {sha(str(row.get("solution_body") or "")) for row in eval_rows}
    eval_body_token_sequences = {
        tuple(body_tokens(str(row.get("solution_body") or ""))) for row in eval_rows
    }
    sft_examples, sft_audit = load_sft_examples(
        config,
        holdout_families=holdout_families,
        eval_prompt_hashes=eval_prompt_hashes,
        eval_body_hashes=eval_body_hashes,
    )
    pretrain, pretrain_audit = build_pretrain_windows(
        config,
        target_vocab,
        eval_body_token_sequences=eval_body_token_sequences,
    )
    sft = encode_sft_examples(config, sft_examples, source_vocab, target_vocab)
    eval_examples = [eval_example(row) for row in eval_rows]
    eval_arrays = encode_sft_examples(config, eval_examples, source_vocab, target_vocab)
    summary = {
        "stage_signature": signature,
        "licensed_pretrain_window_count": int(pretrain[0].shape[0]),
        "licensed_pretrain_target_positions": int(pretrain[2].sum()),
        "sft_example_count": int(sft[0].shape[0]),
        "sft_target_positions": int(sft[2].sum()),
        "family_disjoint_eval_task_count": len(eval_rows),
        "unique_semantic_eval_task_count": len(
            {
                (str(row.get("prompt") or ""), str(row.get("solution_body") or ""))
                for row in eval_rows
            }
        ),
        "holdout_families": holdout_families,
        "train_holdout_family_overlap_count": sft_audit["train_holdout_family_overlap_count"],
        "train_eval_prompt_overlap_count": sft_audit["train_eval_prompt_overlap_count"],
        "train_eval_body_overlap_count": sft_audit["train_eval_body_overlap_count"],
        "unique_sft_body_count": sft_audit["unique_sft_body_count"],
        "unique_sft_pair_count": sft_audit["unique_sft_pair_count"],
        "licensed_function_example_count": sft_audit["licensed_function_example_count"],
        "governed_private_unique_body_count": sft_audit["governed_private_unique_body_count"],
        "rejected_placeholder_source_count": sft_audit["rejected_placeholder_source_count"],
        "rejected_short_source_count": sft_audit["rejected_short_source_count"],
        "metadata_tagged_source_count": sft_audit["metadata_tagged_source_count"],
        "licensed_pretrain_eval_body_overlap_source_detected_count": pretrain_audit[
            "eval_body_overlap_source_detected_count"
        ],
        "licensed_pretrain_eval_body_overlap_sources_excluded": pretrain_audit[
            "eval_body_overlap_sources_excluded"
        ],
        "licensed_pretrain_eval_body_overlap_source_surviving_count": 0,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "cache_status": "miss_rebuilt",
    }
    np.savez(
        arrays_path,
        pretrain_inputs=pretrain[0],
        pretrain_labels=pretrain[1],
        pretrain_mask=pretrain[2],
        sft_inputs=sft[0],
        sft_labels=sft[1],
        sft_mask=sft[2],
        eval_inputs=eval_arrays[0],
        eval_labels=eval_arrays[1],
        eval_mask=eval_arrays[2],
    )
    write_json(metadata_path, {"stage_signature": signature, "summary": summary, "eval_rows": eval_rows})
    return Stage(*pretrain, *sft, *eval_arrays, eval_rows, source_vocab, target_vocab, summary)


def build_pretrain_windows(
    config: dict[str, Any],
    target_vocab: dict[str, int],
    *,
    eval_body_token_sequences: set[tuple[str, ...]],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    manifest = read_json(resolve(config["sources"]["licensed_code_manifest"]))
    token_cfg = config["tokenization"]
    max_seq = int(token_cfg["max_sequence_tokens"])
    stride = int(token_cfg["raw_code_window_stride"])
    target_offset = SPECIAL_COUNT + len(read_json(resolve(token_cfg["source_vocab"]))["source_vocab"])
    wanted = int(config["training"]["pretrain_target_token_positions"])
    windows: list[list[int]] = []
    positions = 0
    overlap_sources: list[str] = []
    eval_body_patterns = [" ".join(sequence) for sequence in eval_body_token_sequences if sequence]
    admitted = [row for row in manifest.get("sources", []) if row.get("admitted") is True]
    admitted.sort(key=lambda row: sha(str(row.get("path") or "")))
    for source in admitted:
        if source.get("public_benchmark_payload_detected") is True or source.get("eval_overlap_detected") is True:
            continue
        path = Path(str(source.get("path") or ""))
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        module_tokens = body_tokens(text)
        normalized_module = " ".join(module_tokens)
        if any(pattern in normalized_module for pattern in eval_body_patterns):
            overlap_sources.append(rel(path))
            continue
        payload, receipt = encode_tokens(module_tokens, target_vocab, stream="target")
        if receipt.get("unknown_token_count"):
            continue
        ids = [target_offset + int(value) for value in payload]
        for start in range(0, len(ids), stride):
            chunk = ids[start : start + max_seq + 1]
            if len(chunk) < 33:
                continue
            windows.append(chunk)
            positions += len(chunk) - 1
            if positions >= wanted:
                return window_arrays(windows, max_seq), {
                    "eval_body_overlap_source_detected_count": len(overlap_sources),
                    "eval_body_overlap_sources_excluded": overlap_sources,
                }
    return window_arrays(windows, max_seq), {
        "eval_body_overlap_source_detected_count": len(overlap_sources),
        "eval_body_overlap_sources_excluded": overlap_sources,
    }


def window_arrays(windows: list[list[int]], max_seq: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = np.zeros((len(windows), max_seq), dtype=np.int32)
    labels = np.zeros((len(windows), max_seq), dtype=np.int32)
    mask = np.zeros((len(windows), max_seq), dtype=np.float32)
    for index, row in enumerate(windows):
        width = min(max_seq, len(row) - 1)
        inputs[index, :width] = row[:width]
        labels[index, :width] = row[1 : width + 1]
        mask[index, :width] = 1.0
    return inputs, labels, mask


def load_sft_examples(
    config: dict[str, Any],
    *,
    holdout_families: list[str],
    eval_prompt_hashes: set[str],
    eval_body_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = read_json(resolve(config["sources"]["function_stage_report"]))
    cache_path = resolve(str(report.get("summary", {}).get("row_summary", {}).get("cache_path") or ""))
    cache = read_json(cache_path)
    examples: list[dict[str, Any]] = []
    rejected_prompt = 0
    rejected_body = 0
    rejected_placeholder_source = 0
    rejected_short_source = 0
    metadata_tagged_source_count = 0
    seen_pairs: set[str] = set()
    unique_bodies: set[str] = set()
    for row in cache.get("examples", []):
        body = str(row.get("body") or "").strip()
        source_text, source_audit = semantic_stage_source(row)
        metadata_tagged_source_count += int(source_audit["metadata_tagged"])
        if source_audit["placeholder"]:
            rejected_placeholder_source += 1
            continue
        if source_audit["too_short"]:
            rejected_short_source += 1
            continue
        if not source_text:
            continue
        prompt_hash = sha(source_text.partition("\nsignature ")[0])
        body_hash = sha(body)
        if prompt_hash in eval_prompt_hashes:
            rejected_prompt += 1
            continue
        if body_hash in eval_body_hashes:
            rejected_body += 1
            continue
        pair_hash = sha(f"{source_text}\n{body}")
        if not body or pair_hash in seen_pairs:
            continue
        seen_pairs.add(pair_hash)
        unique_bodies.add(body_hash)
        examples.append({"source_text": source_text, "body": body, "source": "licensed_function"})
    licensed_count = len(examples)

    admission = read_json(resolve(config["sources"]["training_admission"]))
    private_added = 0
    observed_holdout_families: set[str] = set()
    for source in admission.get("train_admitted_sources", []):
        path_text = str(source.get("path") or "")
        if "conversation" in path_text or not path_text.endswith(".jsonl"):
            continue
        path = resolve(path_text)
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row.get("public_benchmark") is True or any(
                row.get(key) is True
                for key in (
                    "public_prompts_included",
                    "public_tests_included",
                    "public_benchmark_solutions_included",
                    "public_score_labels_included",
                )
            ):
                continue
            family = str(row.get("concept_residual_label") or "")
            if family in holdout_families:
                observed_holdout_families.add(family)
                continue
            prompt = str(row.get("prompt") or "").strip()
            body = str(row.get("solution_body") or "").strip()
            if not prompt or not body:
                continue
            prompt_hash = sha(prompt)
            body_hash = sha(body)
            if prompt_hash in eval_prompt_hashes:
                rejected_prompt += 1
                continue
            if body_hash in eval_body_hashes:
                rejected_body += 1
                continue
            signature = callable_signature(row)
            source_text = f"{prompt}\nsignature {signature}"
            pair_hash = sha(f"{source_text}\n{body}")
            if pair_hash in seen_pairs:
                continue
            seen_pairs.add(pair_hash)
            unique_bodies.add(body_hash)
            examples.append({"source_text": source_text, "body": body, "source": "governed_private"})
            private_added += 1
    examples.sort(key=lambda row: sha(row["source_text"] + "\n" + row["body"]))
    return examples, {
        "unique_sft_body_count": len(unique_bodies),
        "unique_sft_pair_count": len(seen_pairs),
        "licensed_function_example_count": licensed_count,
        "governed_private_unique_body_count": private_added,
        "train_holdout_family_overlap_count": 0,
        "excluded_holdout_family_count": len(observed_holdout_families),
        "train_eval_prompt_overlap_count": 0,
        "train_eval_body_overlap_count": 0,
        "rejected_prompt_overlap_count": rejected_prompt,
        "rejected_body_overlap_count": rejected_body,
        "rejected_placeholder_source_count": rejected_placeholder_source,
        "rejected_short_source_count": rejected_short_source,
        "metadata_tagged_source_count": metadata_tagged_source_count,
    }


def encode_sft_examples(
    config: dict[str, Any],
    examples: list[dict[str, Any]],
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    token_cfg = config["tokenization"]
    max_seq = int(token_cfg["max_sequence_tokens"])
    max_source = int(token_cfg["max_source_tokens"])
    max_target = int(token_cfg["max_target_tokens"])
    source_offset = SPECIAL_COUNT
    target_offset = SPECIAL_COUNT + len(source_vocab)
    rows: list[tuple[list[int], int]] = []
    for row in examples:
        source_ids, source_receipt = encode_tokens(source_tokens(row["source_text"]), source_vocab, stream="source")
        target_ids, target_receipt = encode_tokens(body_tokens(row["body"]), target_vocab, stream="target")
        if source_receipt.get("unknown_token_count") or target_receipt.get("unknown_token_count"):
            continue
        if len(target_ids) > max_target - 2:
            continue
        source_ids = head_tail(source_ids, max_source)
        sequence = [GLOBAL_BOS_ID]
        sequence.extend(source_offset + int(value) for value in source_ids)
        sequence.append(SOURCE_TARGET_SEPARATOR_ID)
        sequence.append(target_offset + int(target_vocab["<bos>"]))
        target_start = len(sequence)
        sequence.extend(target_offset + int(value) for value in target_ids)
        sequence.append(target_offset + int(target_vocab["<eos>"]))
        if len(sequence) > max_seq + 1:
            continue
        rows.append((sequence, target_start - 1))
    inputs = np.zeros((len(rows), max_seq), dtype=np.int32)
    labels = np.zeros((len(rows), max_seq), dtype=np.int32)
    mask = np.zeros((len(rows), max_seq), dtype=np.float32)
    for index, (sequence, mask_start) in enumerate(rows):
        width = len(sequence) - 1
        inputs[index, :width] = sequence[:-1]
        labels[index, :width] = sequence[1:]
        mask[index, mask_start:width] = 1.0
    return inputs, labels, mask


def select_family_disjoint_eval(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows = read_jsonl(resolve(config["sources"]["private_eval"]))
    families = sorted({str(row.get("concept_residual_label") or "") for row in rows if row.get("concept_residual_label")})
    family_count = int(config["evaluation"]["holdout_family_count"])
    seed = int(config["seed"])
    selected_families = sorted(families, key=lambda value: sha(f"{seed}:{value}"))[:family_count]
    per_family = int(config["evaluation"]["rows_per_family"])
    selected: list[dict[str, Any]] = []
    for family in selected_families:
        candidates = [row for row in rows if str(row.get("concept_residual_label") or "") == family]
        candidates.sort(key=lambda row: sha(f"{seed}:{row.get('task_id')}:{row.get('prompt')}"))
        selected.extend(candidates[:per_family])
    return selected, selected_families


def eval_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_text": visible_eval_source(row),
        "body": str(row.get("solution_body") or "").strip(),
        "source": "family_disjoint_private_eval",
    }


def visible_eval_source(row: dict[str, Any]) -> str:
    """Compile the only generator-visible eval fields without touching answers/tests."""
    return f"{str(row.get('prompt') or '').strip()}\nsignature {callable_signature(row)}"


def minimal_stage_source(row: dict[str, Any]) -> str:
    text = str(row.get("source_text") or "")
    function = str(row.get("function") or "")
    marker = f"\n{function}\n" if function else ""
    prompt = text.split(marker, 1)[0].strip() if marker and marker in text else text.split("\nsignature ", 1)[0].strip()
    signature_line = next((line for line in text.splitlines() if line.startswith("signature def ")), "")
    if not signature_line:
        signature_line = f"signature def {function or 'solve'}(data):"
    return f"{prompt}\n{signature_line}"


def semantic_stage_source(row: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    """Retain real prose descriptions and strip train-only metadata from licensed pairs."""
    text = str(row.get("source_text") or "")
    function = str(row.get("function") or "")
    lines = text.splitlines()
    signature_line = next((line for line in lines if line.startswith("signature def ")), "")
    if not signature_line:
        signature_line = f"signature def {function or 'solve'}(data):"
    semantic_lines: list[str] = []
    metadata_tagged = False
    metadata_prefixes = (
        "visible_intent_tags ",
        "prompt_operation_hints ",
        "visible_type_shape_tags ",
        "entry_point_parts ",
        "argument_parts ",
        "visible_subwords ",
        "arguments ",
    )
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("signature def "):
            break
        if stripped.startswith(metadata_prefixes):
            metadata_tagged = True
            continue
        if stripped == function:
            continue
        if stripped:
            semantic_lines.append(stripped)
    description = " ".join(semantic_lines).strip()
    placeholder = description.startswith("Implement Python function ")
    too_short = len(source_tokens(description)) < 12
    if placeholder or too_short:
        return "", {"placeholder": placeholder, "too_short": too_short, "metadata_tagged": metadata_tagged}
    return f"{description}\n{signature_line}", {
        "placeholder": False,
        "too_short": False,
        "metadata_tagged": metadata_tagged,
    }


def callable_signature(row: dict[str, Any]) -> str:
    entry = str(row.get("entry_point") or "solve")
    count = int((row.get("decoder_contract") or {}).get("visible_arg_count_hint") or 1)
    if count <= 1:
        return f"def {entry}(data):"
    if count == 2:
        return f"def {entry}(data, other):"
    return f"def {entry}(data, other=None, *extra):"


def source_tokens(text: str) -> list[str]:
    return SOURCE_TOKEN_RE.findall(str(text))


def head_tail(values: list[int], limit: int) -> list[int]:
    if len(values) <= limit:
        return list(values)
    head = max(1, (limit * 3) // 4)
    return [*values[:head], *values[-(limit - head) :]]


def required_steps(mask: np.ndarray, batch_size: int, target_positions: int) -> int:
    if not len(mask) or target_positions <= 0:
        return 0
    mean_positions = max(1.0, float(mask.sum(axis=1).mean()))
    return max(1, math.ceil(target_positions / (mean_positions * batch_size)))


def build_schedule(optim: Any, mx: Any, cfg: dict[str, Any], total_steps: int) -> Any:
    warmup = min(int(cfg["warmup_steps"]), max(0, total_steps - 1))
    peak = float(cfg["learning_rate"])
    floor = float(cfg["min_learning_rate"])
    if warmup <= 0:
        return optim.cosine_decay(peak, max(1, total_steps), end=floor)
    return optim.join_schedules(
        [
            optim.linear_schedule(floor, peak, warmup),
            optim.cosine_decay(peak, max(1, total_steps - warmup), end=floor),
        ],
        [warmup],
    )


def causal_loss(model: Any, inputs: Any, labels: Any, mask: Any, mx: Any, nn: Any) -> Any:
    logits, _cache = model(inputs)
    token_loss = nn.losses.cross_entropy(logits, labels)
    denominator = mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mx.float32))
    return mx.sum(token_loss * mask) / denominator


def train_phase(
    model: Any,
    optimizer: Any,
    loss_and_grad: Any,
    inputs: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    phase_name: str,
    target_positions: int,
    batch_size: int,
    gradient_clip: float,
    seed: int,
    max_steps: int,
    checkpoint: Path,
    checkpoint_every: int,
    heartbeat: Path,
    global_step_offset: int,
    mx: Any,
    optim: Any,
) -> dict[str, Any]:
    if not len(inputs) or max_steps <= 0:
        return {"phase": phase_name, "optimizer_steps": 0, "target_positions_consumed": 0, "losses": []}
    matrix_x = mx.array(inputs, dtype=mx.int32)
    matrix_y = mx.array(labels, dtype=mx.int32)
    matrix_mask = mx.array(mask, dtype=mx.float32)
    mx.eval(matrix_x, matrix_y, matrix_mask)
    order = list(range(len(inputs)))
    consumed = 0
    steps = 0
    losses: list[float] = []
    started = time.perf_counter()
    epoch = 0
    model.train()
    while consumed < target_positions and steps < max_steps:
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), batch_size):
            if consumed >= target_positions or steps >= max_steps:
                break
            indices = order[start : start + batch_size]
            take = mx.array(indices, dtype=mx.int32)
            x = matrix_x[take]
            y = matrix_y[take]
            m = matrix_mask[take]
            loss, grads = loss_and_grad(model, x, y, m, mx, __import__("mlx.nn", fromlist=["nn"]))
            grads, grad_norm = optim.clip_grad_norm(grads, gradient_clip)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss, grad_norm)
            loss_value = float(loss.item())
            losses.append(loss_value)
            consumed += int(mask[indices].sum())
            steps += 1
            global_step = global_step_offset + steps
            if global_step % 25 == 0:
                write_json(
                    heartbeat,
                    {
                        "policy": "standard_causal_transformer_training_heartbeat_v1",
                        "created_utc": now(),
                        "phase": phase_name,
                        "global_step": global_step,
                        "phase_step": steps,
                        "target_positions_consumed": consumed,
                        "target_positions_requested": target_positions,
                        "latest_loss": round(loss_value, 6),
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "external_inference_calls": 0,
                    },
                )
                print(
                    f"phase={phase_name} step={steps} global={global_step} "
                    f"positions={consumed}/{target_positions} loss={loss_value:.4f}",
                    flush=True,
                )
            if global_step % checkpoint_every == 0:
                model.save_weights(str(checkpoint))
        epoch += 1
    return {
        "phase": phase_name,
        "optimizer_steps": steps,
        "epochs_touched": epoch,
        "target_positions_consumed": consumed,
        "target_positions_requested": target_positions,
        "mean_loss": round(sum(losses) / max(1, len(losses)), 6),
        "final_loss": round(losses[-1], 6) if losses else None,
        "tokens_per_second": round(consumed / max(1e-9, time.perf_counter() - started), 3),
        "external_inference_calls": 0,
    }


def evaluate_loss(
    model: Any,
    inputs: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    batch_size: int,
    mx: Any,
    nn: Any,
) -> float:
    if not len(inputs):
        return float("inf")
    model.eval()
    weighted = 0.0
    positions = 0
    for start in range(0, len(inputs), batch_size):
        x = mx.array(inputs[start : start + batch_size], dtype=mx.int32)
        y = mx.array(labels[start : start + batch_size], dtype=mx.int32)
        m = mx.array(mask[start : start + batch_size], dtype=mx.float32)
        loss = causal_loss(model, x, y, m, mx, nn)
        mx.eval(loss)
        count = int(mask[start : start + batch_size].sum())
        weighted += float(loss.item()) * count
        positions += count
    return round(weighted / max(1, positions), 6)


def generate_candidates(
    model: Any,
    eval_rows: list[dict[str, Any]],
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    config: dict[str, Any],
    *,
    mx: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    candidates: list[dict[str, Any]] = []
    syntax_valid = 0
    empty_rejected = 0
    decode_faults = 0
    timed_out = 0
    for task in eval_rows:
        task_started = time.perf_counter()
        visible = visible_eval_source(task)
        source_ids, receipt = encode_tokens(source_tokens(visible), source_vocab, stream="source")
        if receipt.get("unknown_token_count"):
            decode_faults += 1
            continue
        source_ids = head_tail(source_ids, int(config["tokenization"]["max_source_tokens"]))
        prompt_ids = [GLOBAL_BOS_ID]
        prompt_ids.extend(SPECIAL_COUNT + int(value) for value in source_ids)
        prompt_ids.append(SOURCE_TARGET_SEPARATOR_ID)
        target_offset = SPECIAL_COUNT + len(source_vocab)
        prompt_ids.append(target_offset + int(target_vocab["<bos>"]))
        beams = decode_beams(
            model,
            prompt_ids,
            target_vocab,
            target_offset=target_offset,
            allowed_names=signature_names(callable_signature(task)),
            config=config["evaluation"],
            started=task_started,
            mx=mx,
        )
        if time.perf_counter() - task_started >= float(config["evaluation"]["timeout_seconds_per_task"]):
            timed_out += 1
        seen_code: set[str] = set()
        for rank, beam in enumerate(beams, start=1):
            body, meta = decode_candidate_body_tokens(beam["tokens"], {}, target_mode="body_tokens")
            if not body:
                empty_rejected += 1
                continue
            code = render_visible_signature(callable_signature(task), body)
            try:
                ast.parse(code)
            except SyntaxError:
                continue
            code_hash = sha(code)
            if code_hash in seen_code:
                continue
            seen_code.add(code_hash)
            syntax_valid += 1
            candidates.append(
                {
                    "task_id": str(task.get("task_id") or ""),
                    "source_task_id": str(task.get("source_task_id") or ""),
                    "entry_point": str(task.get("entry_point") or "solve"),
                    "phase": "private_eval",
                    "candidate_source": "standard_causal_transformer_survival",
                    "candidate_generation_mode": "direct_decoder_only_causal_body_tokens",
                    "code": code,
                    "candidate_sha256": code_hash,
                    "substrate_arm": "transformer_hybrid_survival",
                    "substrate_adapter": "mlx_standard_causal_transformer",
                    "rank": rank,
                    "rank_score": round(float(beam["score"]), 8),
                    "decoded_token_count": len(beam["tokens"]),
                    "decoded_token_sha256": sha(" ".join(beam["tokens"])),
                    "body_structure_decode": meta,
                    "template_id": "",
                    "template_sha256": "",
                    "body_template_selected": False,
                    "fallback_return_used": False,
                    "candidate_generation_credit": 1,
                    "public_tests_visible_to_generator": False,
                    "public_solutions_visible_to_generator": False,
                    "eval_tests_visible_to_generator": False,
                    "eval_solution_visible_to_generator": False,
                    "generation_read_set": ["prompt", "entry_point", "callable_signature"],
                    "external_inference_calls": 0,
                    "public_training_rows_written": 0,
                    "fallback_return_count": 0,
                    "provenance": {
                        "policy": config["policy"],
                        "candidate_family": "learned_full_body_token",
                        "model_family": "standard_decoder_only_causal_transformer",
                        "tests_used_for_generation": False,
                        "solutions_used_for_generation": False,
                        "body_template_selected": False,
                        "renderer_used": False,
                        "grammar_constraint_only": True,
                    },
                }
            )
            if len(seen_code) >= int(config["evaluation"]["fanout"]):
                break
    return candidates, {
        "task_count": len(eval_rows),
        "candidate_count": len(candidates),
        "syntax_valid_candidate_count": syntax_valid,
        "empty_body_rejected_count": empty_rejected,
        "decode_fault_count": decode_faults,
        "timed_out_task_count": timed_out,
        "fallback_return_count": 0,
        "template_renderer_router_tool_credit_count": 0,
    }


def decode_beams(
    model: Any,
    prompt_ids: list[int],
    target_vocab: dict[str, int],
    *,
    target_offset: int,
    allowed_names: set[str],
    config: dict[str, Any],
    started: float,
    mx: Any,
) -> list[dict[str, Any]]:
    inverse = {int(value): key for key, value in target_vocab.items()}
    eos_local = int(target_vocab["<eos>"])
    logits, cache = model(mx.array([prompt_ids], dtype=mx.int32))
    mx.eval(logits, *[value for pair in cache for value in pair])
    beams = [{"tokens": [], "score": 0.0, "cache": cache, "logits": logits[0, -1]}]
    complete: list[dict[str, Any]] = []
    for _step in range(int(config["decode_max_target_tokens"])):
        if time.perf_counter() - started >= float(config["timeout_seconds_per_task"]):
            break
        expanded: list[dict[str, Any]] = []
        for beam in beams:
            choices = grammar_choices(
                beam["logits"],
                beam["tokens"],
                target_vocab,
                inverse,
                target_offset=target_offset,
                eos_local=eos_local,
                top_k=int(config["grammar_top_k"]),
                branching=int(config["branching_factor"]),
                allowed_names=allowed_names,
            )
            for local_id, token, log_probability in choices:
                if local_id == eos_local:
                    complete.append({"tokens": list(beam["tokens"]), "score": beam["score"] + log_probability})
                    continue
                next_logits, next_cache = model(
                    mx.array([[target_offset + local_id]], dtype=mx.int32),
                    beam["cache"],
                )
                mx.eval(next_logits, *[value for pair in next_cache for value in pair])
                expanded.append(
                    {
                        "tokens": [*beam["tokens"], token],
                        "score": beam["score"] + log_probability,
                        "cache": next_cache,
                        "logits": next_logits[0, -1],
                    }
                )
        if len(complete) >= int(config["fanout"]):
            break
        dedup: dict[tuple[str, ...], dict[str, Any]] = {}
        for row in expanded:
            key = tuple(row["tokens"])
            if key not in dedup or row["score"] > dedup[key]["score"]:
                dedup[key] = row
        beams = sorted(
            dedup.values(),
            key=lambda row: row["score"] / max(1, len(row["tokens"])),
            reverse=True,
        )[: int(config["beam_width"])]
        if not beams:
            break
    for beam in beams:
        if syntax_complete_body_prefix(beam["tokens"]):
            complete.append({"tokens": beam["tokens"], "score": beam["score"]})
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in complete:
        key = tuple(row["tokens"])
        if key not in unique or row["score"] > unique[key]["score"]:
            unique[key] = row
    return sorted(
        unique.values(),
        key=lambda row: row["score"] / max(1, len(row["tokens"])),
        reverse=True,
    )[: int(config["fanout"])]


def grammar_choices(
    logits: Any,
    prefix: list[str],
    target_vocab: dict[str, int],
    inverse: dict[int, str],
    *,
    target_offset: int,
    eos_local: int,
    top_k: int,
    branching: int,
    allowed_names: set[str],
) -> list[tuple[int, str, float]]:
    values = np.asarray(logits[target_offset : target_offset + len(target_vocab)], dtype=np.float64)
    maximum = float(values.max())
    log_probs = values - maximum - math.log(float(np.exp(values - maximum).sum()))
    ranked = np.argsort(log_probs)[::-1][: max(top_k, branching)]
    choices: list[tuple[int, str, float]] = []
    for raw_id in ranked:
        local_id = int(raw_id)
        token = inverse.get(local_id, "<unk>")
        if local_id == eos_local:
            if syntax_complete_body_prefix(prefix):
                choices.append((local_id, token, float(log_probs[local_id])))
        elif token not in {"<pad>", "<bos>", "<eos>", "<unk>"} and token_allowed_by_policy(
            prefix,
            token,
            policy="strict_body_token_legality_v1",
            allowed_names=allowed_names,
        ):
            choices.append((local_id, token, float(log_probs[local_id])))
        if len(choices) >= branching:
            break
    return choices


def render_visible_signature(signature: str, body: str) -> str:
    if not body.strip():
        raise ValueError("empty body cannot be rendered")
    return signature.rstrip() + "\n" + "\n".join(f"    {line}" if line else "" for line in body.splitlines()) + "\n"


def signature_names(signature: str) -> set[str]:
    try:
        parsed = ast.parse(signature + "\n    pass\n")
        fn = parsed.body[0]
        if not isinstance(fn, ast.FunctionDef):
            return set()
        return {arg.arg for arg in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]}
    except SyntaxError:
        return set()


def private_verifier_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    passed = int(
        summary.get("passed_task_count")
        or summary.get("tasks_passed")
        or summary.get("selected_pass_count")
        or report.get("passed_task_count")
        or report.get("trained_passed")
        or 0
    )
    return {"passed_task_count": passed, "raw_summary": summary}


def build_gates(
    stage: Stage,
    training_complete: bool,
    loss_before: float,
    loss_after: float,
    decode: dict[str, Any],
    verifier: dict[str, Any],
) -> list[dict[str, Any]]:
    checks = [
        ("licensed_pretraining_positions_present", stage.summary["licensed_pretrain_target_positions"] > 0),
        ("deduplicated_sft_rows_present", stage.summary["sft_example_count"] > 0),
        ("family_disjoint_split_clean", stage.summary["train_holdout_family_overlap_count"] == 0),
        (
            "distinct_family_disjoint_eval_tasks",
            stage.summary["unique_semantic_eval_task_count"]
            == stage.summary["family_disjoint_eval_task_count"],
        ),
        ("prompt_overlap_zero", stage.summary["train_eval_prompt_overlap_count"] == 0),
        ("body_overlap_zero", stage.summary["train_eval_body_overlap_count"] == 0),
        (
            "licensed_pretrain_body_overlap_zero",
            stage.summary["licensed_pretrain_eval_body_overlap_source_surviving_count"] == 0,
        ),
        ("training_complete", training_complete),
        ("heldout_lm_loss_improved", loss_after < loss_before),
        ("syntax_valid_candidate_present", decode["syntax_valid_candidate_count"] > 0),
        ("model_only_private_behavior_above_zero", int(verifier.get("passed_task_count") or 0) > 0),
        ("fallback_returns_zero", decode["fallback_return_count"] == 0),
        ("public_training_rows_zero", stage.summary["public_training_rows"] == 0),
        ("external_inference_zero", stage.summary["external_inference_calls"] == 0),
    ]
    return [{"gate": name, "passed": bool(passed), "status": "PASSED" if passed else "FAILED"} for name, passed in checks]


def validate_config(config: dict[str, Any]) -> None:
    if config.get("policy") != "project_theseus_standard_causal_transformer_survival_v1":
        raise ValueError("unexpected standard causal transformer policy")
    boundary = config.get("boundaries") or {}
    if int(boundary.get("public_training_rows") or 0) != 0:
        raise ValueError("public training is forbidden")
    if int(boundary.get("external_inference_calls") or 0) != 0:
        raise ValueError("external inference is forbidden")
    if boundary.get("fallback_returns_allowed") is not False:
        raise ValueError("fallback returns must remain forbidden")
    evaluation = config.get("evaluation") or {}
    if int(evaluation.get("holdout_family_count") or 0) < 24:
        raise ValueError("family-disjoint evaluation requires at least 24 distinct families")
    if int(evaluation.get("rows_per_family") or 0) != 1:
        raise ValueError("evaluation must use one unique semantic row per holdout family")


def planned_report(
    config: dict[str, Any], config_path: str, checkpoint_dir: Path, stage_dir: Path, started: float
) -> dict[str, Any]:
    return {
        "policy": config["policy"],
        "created_utc": now(),
        "trigger_state": "PLANNED",
        "execute": False,
        "artifacts": {"config": config_path, "checkpoint_dir": rel(checkpoint_dir), "stage_dir": rel(stage_dir)},
        "architecture": {"family": "standard_decoder_only_causal_transformer", **config["model"]},
        "boundaries": config["boundaries"],
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def stage_signature(config: dict[str, Any]) -> str:
    paths = [
        Path(__file__),
        resolve(config["sources"]["licensed_code_manifest"]),
        resolve(config["sources"]["function_stage_report"]),
        resolve(config["sources"]["training_admission"]),
        resolve(config["sources"]["private_eval"]),
        resolve(config["tokenization"]["source_vocab"]),
    ]
    payload = json.dumps(config, sort_keys=True) + "|" + "|".join(
        f"{rel(path)}:{path.stat().st_size if path.exists() else -1}:{path.stat().st_mtime_ns if path.exists() else -1}"
        for path in paths
    )
    return sha(payload)


def sha(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


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
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
