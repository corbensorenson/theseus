#!/usr/bin/env python3
"""Train and replay the standard causal-transformer survival lane on MLX."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
import math
import os
import random
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPTS))

from code_lm_private_verifier import (  # noqa: E402
    evaluate_all_private_candidates,
    evaluate_private_candidates,
)
import semantic_ir  # noqa: E402
from neural_seed_open_vocab import encode_tokens  # noqa: E402
from neural_seed_token_decoder_support import (  # noqa: E402
    body_tokens,
    decode_candidate_body_tokens,
    syntax_complete_body_prefix,
    token_allowed_by_policy,
)
from neural_seed_token_decoder_rendering import (  # noqa: E402
    PLAN_BODY_START_TOKEN,
    learned_semantic_ir_plan_body_target_mode,
    split_learned_plan_prefix_tokens,
)
from standard_causal_transformer_model import (  # noqa: E402
    CausalTransformerConfig,
    build_model,
    parameter_count,
)
from standard_causal_transformer_preference import (  # noqa: E402
    build_preference_pairs,
    encode_preference_arrays,
    reward_removed_pairs,
    train_dpo,
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
CANONICAL_MODEL_SIGNATURE_NAME = "solve"


@dataclass
class Stage:
    pretrain_inputs: np.ndarray
    pretrain_labels: np.ndarray
    pretrain_mask: np.ndarray
    sft_inputs: np.ndarray
    sft_labels: np.ndarray
    sft_mask: np.ndarray
    sft_sampling_weights: np.ndarray
    eval_inputs: np.ndarray
    eval_labels: np.ndarray
    eval_mask: np.ndarray
    eval_rows: list[dict[str, Any]]
    preference_rows: list[dict[str, Any]]
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
    parser.add_argument(
        "--prior-report",
        default="",
        help="Completed training receipt to preserve during resume/evaluation-only replay.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force-restage", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--preference-canary", action="store_true")
    parser.add_argument("--generation-mode-canary", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    if (args.resume or args.evaluate_only) and not args.execute:
        parser.error("--resume and --evaluate-only require --execute")
    prior_report_path = resolve(args.prior_report) if args.prior_report else resolve(args.out)
    if (args.resume or args.evaluate_only) and not prior_report_path.exists():
        parser.error(f"prior training receipt missing: {prior_report_path}")

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
        preference_canary=args.preference_canary,
        generation_mode_canary=args.generation_mode_canary,
        max_steps=max(0, args.max_steps),
        prior_report_path=prior_report_path,
        candidates_path=resolve(args.candidates_out),
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
    preference_canary: bool,
    generation_mode_canary: bool,
    max_steps: int,
    prior_report_path: Path,
    candidates_path: Path,
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
    vocab_size = model_vocab_size(config, stage.source_vocab, stage.target_vocab)
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
        sample_weights=stage.sft_sampling_weights,
    )
    total_steps = pretrain_steps + sft_steps
    if max_steps:
        total_steps = min(total_steps, max_steps)
    if evaluate_only:
        prior = read_json(prior_report_path)
        prior_training = prior.get("training") if isinstance(prior.get("training"), dict) else {}
        conditioning = prior.get("conditioning") if isinstance(prior.get("conditioning"), dict) else {}
        eval_loss_before = float(prior_training.get("eval_loss_before") or float("inf"))
        phase_reports = list(prior_training.get("phases") or [])
        consumed_steps = int(prior_training.get("optimizer_steps") or 0)
        training_complete = bool(prior_training.get("complete"))
    elif resume:
        prior = read_json(prior_report_path)
        prior_training = prior.get("training") if isinstance(prior.get("training"), dict) else {}
        prior_conditioning = (
            prior.get("conditioning") if isinstance(prior.get("conditioning"), dict) else {}
        )
        resume_base_checkpoint_sha256 = file_content_sha256(checkpoint)
        eval_loss_before = evaluate_loss(
            model,
            stage.eval_inputs,
            stage.eval_labels,
            stage.eval_mask,
            batch_size=int(training_cfg["batch_size"]),
            mx=mx,
            nn=nn,
        )
        conditioning = {
            **prior_conditioning,
            "resume_base_checkpoint_sha256": resume_base_checkpoint_sha256,
            "resume_eval_loss_before": eval_loss_before,
            "resume_stage_signature": stage.summary["stage_signature"],
        }
        phase_reports = list(prior_training.get("phases") or [])
        consumed_steps = sum(int(row.get("optimizer_steps") or 0) for row in phase_reports)
        optimizer = optim.AdamW(
            learning_rate=float(training_cfg["min_learning_rate"]),
            weight_decay=float(training_cfg["weight_decay"]),
        )
        loss_and_grad = nn.value_and_grad(model, causal_loss)
        heartbeat = stage_dir / "training_heartbeat.json"
        prior_sft_positions = phase_target_positions(phase_reports, "prompt_signature_body_sft")
        remaining_sft_positions = max(
            0, int(training_cfg["sft_target_token_positions"]) - prior_sft_positions
        )
        if remaining_sft_positions:
            continuation_steps = required_steps(
                stage.sft_mask,
                int(training_cfg["batch_size"]),
                remaining_sft_positions,
                sample_weights=stage.sft_sampling_weights,
            )
            phase_report = train_phase(
                model,
                optimizer,
                loss_and_grad,
                stage.sft_inputs,
                stage.sft_labels,
                stage.sft_mask,
                sample_weights=stage.sft_sampling_weights,
                phase_name="prompt_signature_body_sft_continuation",
                target_positions=remaining_sft_positions,
                batch_size=int(training_cfg["batch_size"]),
                gradient_clip=float(training_cfg["gradient_clip_norm"]),
                seed=seed + 7919,
                max_steps=continuation_steps + 64,
                checkpoint=checkpoint,
                checkpoint_every=max(1, int(training_cfg["checkpoint_every_steps"])),
                heartbeat=heartbeat,
                global_step_offset=consumed_steps,
                mx=mx,
                optim=optim,
            )
            phase_report["optimizer_state_restored"] = False
            phase_report["continuation_learning_rate"] = float(training_cfg["min_learning_rate"])
            phase_reports.append(phase_report)
            consumed_steps += int(phase_report["optimizer_steps"])
        model.save_weights(str(checkpoint))
        mx.eval(model.parameters())
        training_complete = training_targets_complete(phase_reports, training_cfg)
    else:
        conditioning = {}
        schedule = build_schedule(optim, mx, training_cfg, total_steps + 128)
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
        for phase_name, inputs, labels, mask, sample_weights, target_positions, planned_steps in (
            (
                "licensed_module_causal_pretraining",
                stage.pretrain_inputs,
                stage.pretrain_labels,
                stage.pretrain_mask,
                None,
                int(training_cfg["pretrain_target_token_positions"]),
                pretrain_steps,
            ),
            (
                "prompt_signature_body_sft",
                stage.sft_inputs,
                stage.sft_labels,
                stage.sft_mask,
                stage.sft_sampling_weights,
                int(training_cfg["sft_target_token_positions"]),
                sft_steps,
            ),
        ):
            remaining = (total_steps - consumed_steps) if max_steps else (planned_steps + 64)
            if remaining <= 0:
                break
            phase_report = train_phase(
                model,
                optimizer,
                loss_and_grad,
                inputs,
                labels,
                mask,
                sample_weights=sample_weights,
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
        training_complete = training_targets_complete(phase_reports, training_cfg)
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
    decode_runtime_seconds = float(decode_summary.get("runtime_ms") or 0) / 1000.0
    decode_summary["accepted_verified_output_per_second"] = (
        round(model_pass_count / max(1e-9, decode_runtime_seconds), 8)
        if decode_runtime_seconds > 0
        else None
    )
    preference_report: dict[str, Any] = {
        "state": "NOT_RUN",
        "reason": "enable --preference-canary for the bounded private reward-present/control comparison",
    }
    if preference_canary:
        preference_report = run_preference_canary(
            reference_model=model,
            model_cfg=model_cfg,
            checkpoint=checkpoint,
            checkpoint_dir=checkpoint_dir,
            stage=stage,
            config=config,
            base_verifier=verifier,
            base_candidates=candidates,
            base_decode=decode_summary,
            mx=mx,
            nn=nn,
            optim=optim,
        )
    generation_mode_report: dict[str, Any] = {
        "state": "NOT_RUN",
        "reason": "enable --generation-mode-canary for matched serial-versus-batched beam evidence",
    }
    if generation_mode_canary:
        generation_mode_report = run_generation_mode_canary(
            reference_model=model,
            stage=stage,
            config=config,
            batched_candidates=candidates,
            batched_decode=decode_summary,
            batched_verifier=verifier,
            checkpoint_dir=checkpoint_dir,
            mx=mx,
        )
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
            "candidates": rel(candidates_path),
            "prior_training_receipt": rel(prior_report_path) if (resume or evaluate_only) else "",
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
        "preference_canary": preference_report,
        "generation_mode_canary": generation_mode_report,
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
            "An optional learned semantic-plan prefix is emitted by the same model and stripped before "
            "the directly generated body; it does not render or repair code. Reversible token decoding "
            "adds no body content, and no template, renderer, router, tool, fallback return, public "
            "benchmark payload, or external inference is credited."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    return report, candidates


def materialize_stage(config: dict[str, Any], *, stage_dir: Path, force: bool) -> Stage:
    stage_dir.mkdir(parents=True, exist_ok=True)
    with stage_materialization_lock(stage_dir):
        return _materialize_stage_unlocked(config, stage_dir=stage_dir, force=force)


def _materialize_stage_unlocked(config: dict[str, Any], *, stage_dir: Path, force: bool) -> Stage:
    stage_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = stage_dir / "stage_arrays_v1.npz"
    metadata_path = stage_dir / "stage_metadata_v1.json"
    signature = stage_signature(config)
    if not force and arrays_path.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata.get("stage_signature") == signature:
            arrays = np.load(arrays_path)
            vocab_payload = read_json(resolve(config["tokenization"]["source_vocab"]))
            source_vocab = dict(metadata.get("source_vocab") or vocab_payload["source_vocab"])
            target_vocab = dict(metadata.get("target_vocab") or vocab_payload["target_vocab"])
            return Stage(
                pretrain_inputs=arrays["pretrain_inputs"],
                pretrain_labels=arrays["pretrain_labels"],
                pretrain_mask=arrays["pretrain_mask"],
                sft_inputs=arrays["sft_inputs"],
                sft_labels=arrays["sft_labels"],
                sft_mask=arrays["sft_mask"],
                sft_sampling_weights=(
                    arrays["sft_sampling_weights"]
                    if "sft_sampling_weights" in arrays.files
                    else np.ones((len(arrays["sft_inputs"]),), dtype=np.float32)
                ),
                eval_inputs=arrays["eval_inputs"],
                eval_labels=arrays["eval_labels"],
                eval_mask=arrays["eval_mask"],
                eval_rows=list(metadata["eval_rows"]),
                preference_rows=list(metadata.get("preference_rows") or []),
                source_vocab=source_vocab,
                target_vocab=target_vocab,
                summary={**metadata["summary"], "cache_status": "hit"},
            )

    vocab_payload = read_json(resolve(config["tokenization"]["source_vocab"]))
    source_vocab = dict(vocab_payload["source_vocab"])
    target_vocab = dict(vocab_payload["target_vocab"])
    target_vocab_extension = extend_target_vocab_for_mode(config, target_vocab)
    eval_rows, holdout_families = select_family_disjoint_eval(config)
    eval_prompt_hashes = {sha(model_prompt(row)) for row in eval_rows}
    eval_body_hashes = {sha(str(row.get("solution_body") or "")) for row in eval_rows}
    eval_body_token_sequences = {
        tuple(body_tokens(str(row.get("solution_body") or ""))) for row in eval_rows
    }
    preference_rows, preference_audit = select_preference_train_rows(
        config,
        holdout_families=holdout_families,
        eval_prompt_hashes=eval_prompt_hashes,
        eval_body_hashes=eval_body_hashes,
    )
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
    sft = encode_sft_training_examples(config, sft_examples, source_vocab, target_vocab)
    eval_examples = [eval_example(row) for row in eval_rows]
    eval_arrays = encode_sft_examples(config, eval_examples, source_vocab, target_vocab)
    summary = {
        "stage_signature": signature,
        "licensed_pretrain_window_count": int(pretrain[0].shape[0]),
        "licensed_pretrain_target_positions": int(pretrain[2].sum()),
        "sft_example_count": int(sft[0].shape[0]),
        "sft_target_positions": int(sft[2].sum()),
        "sft_sampling_weight_sum": round(float(sft[3].sum()), 6),
        "family_disjoint_eval_task_count": len(eval_rows),
        "encoded_family_disjoint_eval_task_count": int(eval_arrays[0].shape[0]),
        "eval_target_overflow_count": len(eval_rows) - int(eval_arrays[0].shape[0]),
        "preference_train_task_count": len(preference_rows),
        "preference_train_family_count": len(
            {str(row.get("concept_residual_label") or "") for row in preference_rows}
        ),
        "preference_train_eval_family_overlap_count": preference_audit[
            "train_eval_family_overlap_count"
        ],
        "preference_train_eval_prompt_overlap_count": preference_audit[
            "train_eval_prompt_overlap_count"
        ],
        "preference_train_eval_body_overlap_count": preference_audit[
            "train_eval_body_overlap_count"
        ],
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
        "governed_private_prompt_pair_count": sft_audit["governed_private_prompt_pair_count"],
        "private_explicit_signature_count": sft_audit["private_explicit_signature_count"],
        "private_generic_prompt_only_signature_count": sft_audit[
            "private_generic_prompt_only_signature_count"
        ],
        "private_hidden_derived_signature_count": sft_audit[
            "private_hidden_derived_signature_count"
        ],
        "private_sampling_probability": sft_audit["private_sampling_probability"],
        "shared_source_target_vocabulary": shared_vocabulary_enabled(config),
        "target_mode": str(config["tokenization"]["target_mode"]),
        "target_vocab_extension": target_vocab_extension,
        "model_vocabulary_size": model_vocab_size(config, source_vocab, target_vocab),
        "eval_hidden_derived_signature_count": sum(
            int(
                str((row.get("callable_signature_receipt") or {}).get("source") or "")
                not in {"explicit_callable_signature", "generic_prompt_only_interface"}
            )
            for row in eval_rows
        ),
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
    arrays_temporary = arrays_path.with_suffix(arrays_path.suffix + f".{os.getpid()}.tmp.npz")
    try:
        np.savez(
            arrays_temporary,
            pretrain_inputs=pretrain[0],
            pretrain_labels=pretrain[1],
            pretrain_mask=pretrain[2],
            sft_inputs=sft[0],
            sft_labels=sft[1],
            sft_mask=sft[2],
            sft_sampling_weights=sft[3],
            eval_inputs=eval_arrays[0],
            eval_labels=eval_arrays[1],
            eval_mask=eval_arrays[2],
        )
        arrays_temporary.replace(arrays_path)
    finally:
        arrays_temporary.unlink(missing_ok=True)
    write_json(
        metadata_path,
        {
            "stage_signature": signature,
            "summary": summary,
            "eval_rows": eval_rows,
            "preference_rows": preference_rows,
            "source_vocab": source_vocab,
            "target_vocab": target_vocab,
        },
    )
    return Stage(
        pretrain_inputs=pretrain[0],
        pretrain_labels=pretrain[1],
        pretrain_mask=pretrain[2],
        sft_inputs=sft[0],
        sft_labels=sft[1],
        sft_mask=sft[2],
        sft_sampling_weights=sft[3],
        eval_inputs=eval_arrays[0],
        eval_labels=eval_arrays[1],
        eval_mask=eval_arrays[2],
        eval_rows=eval_rows,
        preference_rows=preference_rows,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        summary=summary,
    )


@contextmanager
def stage_materialization_lock(
    stage_dir: Path, *, timeout_seconds: float = 600.0, stale_seconds: float = 14_400.0
) -> Any:
    """Serialize stage writes and recover only locks whose owner is gone or stale."""

    lock_path = stage_dir / ".materialize.lock"
    owner_token = f"{os.getpid()}:{time.time_ns()}"
    started = time.monotonic()
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            owner_pid, created = read_stage_lock_owner(lock_path)
            stale = (created is not None and time.time() - created > stale_seconds) or (
                owner_pid is not None and not process_is_alive(owner_pid)
            )
            if stale:
                lock_path.unlink(missing_ok=True)
                continue
            if time.monotonic() - started >= timeout_seconds:
                raise TimeoutError(
                    f"stage materialization lock timed out: {lock_path} owner_pid={owner_pid}"
                )
            time.sleep(0.25)
            continue
        else:
            payload = json.dumps(
                {"owner_token": owner_token, "pid": os.getpid(), "created_epoch": time.time()}
            ).encode("utf-8")
            os.write(descriptor, payload)
            os.close(descriptor)
            break
    try:
        yield
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            current = {}
        if current.get("owner_token") == owner_token:
            lock_path.unlink(missing_ok=True)


def read_stage_lock_owner(lock_path: Path) -> tuple[int | None, float | None]:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        return int(payload.get("pid")), float(payload.get("created_epoch"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        try:
            return None, lock_path.stat().st_mtime
        except OSError:
            return None, None


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
    source_vocab = read_json(resolve(token_cfg["source_vocab"]))["source_vocab"]
    target_offset = target_token_offset(config, source_vocab)
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
    private_body_hashes: set[str] = set()
    private_explicit_signatures = 0
    private_generic_signatures = 0
    private_hidden_derived_signatures = 0
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
            prompt = model_prompt(row)
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
            signature, signature_receipt = training_callable_signature(row)
            if not signature:
                continue
            source_text = f"{prompt}\nsignature {canonical_model_signature(signature, config)}"
            pair_hash = sha(f"{source_text}\n{body}")
            if pair_hash in seen_pairs:
                continue
            private_explicit_signatures += int(
                signature_receipt["source"] == "explicit_callable_signature"
            )
            private_generic_signatures += int(
                signature_receipt["source"] == "generic_prompt_only_interface"
            )
            private_hidden_derived_signatures += int(
                signature_receipt["source"]
                not in {"explicit_callable_signature", "generic_prompt_only_interface"}
            )
            seen_pairs.add(pair_hash)
            unique_bodies.add(body_hash)
            private_body_hashes.add(body_hash)
            examples.append({"source_text": source_text, "body": body, "source": "governed_private"})
            private_added += 1
    examples, sampling = assign_body_balanced_sampling_weights(
        examples,
        private_body_weight=float(config["training"]["private_body_sampling_weight"]),
    )
    examples.sort(key=lambda row: sha(row["source_text"] + "\n" + row["body"]))
    return examples, {
        "unique_sft_body_count": len(unique_bodies),
        "unique_sft_pair_count": len(seen_pairs),
        "licensed_function_example_count": licensed_count,
        "governed_private_unique_body_count": len(private_body_hashes),
        "governed_private_prompt_pair_count": private_added,
        "private_explicit_signature_count": private_explicit_signatures,
        "private_generic_prompt_only_signature_count": private_generic_signatures,
        "private_hidden_derived_signature_count": private_hidden_derived_signatures,
        **sampling,
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
    inputs, labels, mask, _weights = encode_sft_training_examples(
        config, examples, source_vocab, target_vocab
    )
    return inputs, labels, mask


def encode_sft_training_examples(
    config: dict[str, Any],
    examples: list[dict[str, Any]],
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    token_cfg = config["tokenization"]
    max_seq = int(token_cfg["max_sequence_tokens"])
    max_source = int(token_cfg["max_source_tokens"])
    max_target = int(token_cfg["max_target_tokens"])
    source_offset = source_token_offset(config, source_vocab)
    target_offset = target_token_offset(config, source_vocab)
    rows: list[tuple[list[int], int, float]] = []
    for row in examples:
        source_ids, source_receipt = encode_model_source(
            row["source_text"], source_vocab, target_vocab, config
        )
        target_ids, target_receipt = encode_tokens(
            training_target_tokens(str(row["body"]), config),
            target_vocab,
            stream="target",
        )
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
        rows.append((sequence, target_start - 1, float(row.get("sampling_weight") or 1.0)))
    inputs = np.zeros((len(rows), max_seq), dtype=np.int32)
    labels = np.zeros((len(rows), max_seq), dtype=np.int32)
    mask = np.zeros((len(rows), max_seq), dtype=np.float32)
    weights = np.ones((len(rows),), dtype=np.float32)
    for index, (sequence, mask_start, sampling_weight) in enumerate(rows):
        width = len(sequence) - 1
        inputs[index, :width] = sequence[:-1]
        labels[index, :width] = sequence[1:]
        mask[index, mask_start:width] = 1.0
        weights[index] = max(0.0, sampling_weight)
    return inputs, labels, mask, weights


def training_target_tokens(body: str, config: dict[str, Any]) -> list[str]:
    tokenization = config.get("tokenization") if isinstance(config.get("tokenization"), dict) else {}
    target_mode = str(tokenization.get("target_mode") or "body_tokens")
    if learned_semantic_ir_plan_body_target_mode(target_mode):
        max_plan_tokens = int(tokenization.get("semantic_plan_max_tokens") or semantic_ir.PLAN_MAX_TOKENS)
        return [
            *semantic_ir.body_to_plan_tokens(body, max_tokens=max_plan_tokens),
            PLAN_BODY_START_TOKEN,
            *body_tokens(body),
        ]
    return body_tokens(body)


def extend_target_vocab_for_mode(config: dict[str, Any], target_vocab: dict[str, int]) -> dict[str, Any]:
    target_mode = str(config.get("tokenization", {}).get("target_mode") or "body_tokens")
    before = len(target_vocab)
    added: list[str] = []
    if learned_semantic_ir_plan_body_target_mode(target_mode):
        for token in (*semantic_ir.plan_protocol_tokens(), PLAN_BODY_START_TOKEN):
            if token in target_vocab:
                continue
            target_vocab[token] = len(target_vocab)
            added.append(token)
    return {
        "policy": "project_theseus_standard_causal_target_vocab_extension_v1",
        "target_mode": target_mode,
        "size_before": before,
        "size_after": len(target_vocab),
        "added_token_count": len(added),
        "added_token_sha256": sha("\n".join(added)),
        "target_independent_closed_protocol": learned_semantic_ir_plan_body_target_mode(target_mode),
    }


def assign_body_balanced_sampling_weights(
    examples: list[dict[str, Any]], *, private_body_weight: float
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    private_counts: dict[str, int] = {}
    for row in examples:
        if row.get("source") != "governed_private":
            continue
        body_hash = sha(str(row.get("body") or ""))
        private_counts[body_hash] = private_counts.get(body_hash, 0) + 1
    weighted: list[dict[str, Any]] = []
    licensed_mass = 0.0
    private_mass = 0.0
    for row in examples:
        item = dict(row)
        if item.get("source") == "governed_private":
            body_hash = sha(str(item.get("body") or ""))
            weight = max(0.0, private_body_weight) / max(1, private_counts.get(body_hash, 1))
            private_mass += weight
        else:
            weight = 1.0
            licensed_mass += weight
        item["sampling_weight"] = weight
        weighted.append(item)
    total = licensed_mass + private_mass
    return weighted, {
        "licensed_sampling_mass": round(licensed_mass, 6),
        "private_sampling_mass": round(private_mass, 6),
        "private_sampling_probability": round(private_mass / total, 6) if total else 0.0,
        "private_body_sampling_weight": float(private_body_weight),
    }


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
        for row in candidates[:per_family]:
            item = dict(row)
            signature, receipt = training_callable_signature(item)
            item["callable_signature"] = signature
            item["callable_signature_receipt"] = receipt
            selected.append(item)
    return selected, selected_families


def select_preference_train_rows(
    config: dict[str, Any],
    *,
    holdout_families: list[str],
    eval_prompt_hashes: set[str],
    eval_body_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select verifier-bearing private train tasks without exposing their tests to generation."""

    preference = config.get("preference") if isinstance(config.get("preference"), dict) else {}
    limit = max(0, int(preference.get("max_train_tasks") or 0))
    admission = read_json(resolve(config["sources"]["training_admission"]))
    holdouts = set(holdout_families)
    eligible: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    excluded_family = 0
    excluded_prompt = 0
    excluded_body = 0
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
            task_id = str(row.get("task_id") or "")
            family = str(row.get("concept_residual_label") or "")
            prompt = str(row.get("prompt") or "").strip()
            body = str(row.get("solution_body") or "").strip()
            tests = str(row.get("tests") or "").strip()
            if not task_id or task_id in seen_tasks or not prompt or not body or not tests:
                continue
            if family in holdouts:
                excluded_family += 1
                continue
            if sha(prompt) in eval_prompt_hashes:
                excluded_prompt += 1
                continue
            if sha(body) in eval_body_hashes:
                excluded_body += 1
                continue
            signature, receipt = training_callable_signature(row)
            item = dict(row)
            item["split"] = "eval"
            item["callable_signature"] = signature
            item["callable_signature_receipt"] = receipt
            item["preference_source_path"] = path_text
            eligible.append(item)
            seen_tasks.add(task_id)
    seed = int(config["seed"])
    eligible.sort(
        key=lambda row: sha(
            f"{seed}:preference:{row.get('concept_residual_label')}:{row.get('task_id')}:{row.get('prompt')}"
        )
    )
    selected: list[dict[str, Any]] = []
    selected_families: set[str] = set()
    for row in eligible:
        family = str(row.get("concept_residual_label") or "")
        if family and family not in selected_families:
            selected.append(row)
            selected_families.add(family)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        selected_ids = {str(row.get("task_id") or "") for row in selected}
        selected.extend(
            row for row in eligible if str(row.get("task_id") or "") not in selected_ids
        )
        selected = selected[:limit]
    return selected, {
        "eligible_task_count": len(eligible),
        "selected_task_count": len(selected),
        "train_eval_family_overlap_count": sum(
            str(row.get("concept_residual_label") or "") in holdouts for row in selected
        ),
        "train_eval_prompt_overlap_count": sum(
            sha(str(row.get("prompt") or "")) in eval_prompt_hashes for row in selected
        ),
        "train_eval_body_overlap_count": sum(
            sha(str(row.get("solution_body") or "")) in eval_body_hashes for row in selected
        ),
        "excluded_holdout_family_row_count": excluded_family,
        "excluded_eval_prompt_row_count": excluded_prompt,
        "excluded_eval_body_row_count": excluded_body,
    }


def eval_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_text": visible_eval_source(row),
        "body": str(row.get("solution_body") or "").strip(),
        "source": "family_disjoint_private_eval",
    }


def visible_eval_source(row: dict[str, Any]) -> str:
    """Compile the only generator-visible eval fields without touching answers/tests."""
    return f"{model_prompt(row)}\nsignature {canonical_model_signature(callable_signature(row))}"


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
    signature = signature_line.removeprefix("signature ")
    return f"{description}\nsignature {canonical_model_signature(signature)}", {
        "placeholder": False,
        "too_short": False,
        "metadata_tagged": metadata_tagged,
    }


def callable_signature(row: dict[str, Any]) -> str:
    entry = str(row.get("entry_point") or "solve")
    explicit = str(row.get("callable_signature") or "").strip()
    if explicit:
        return explicit
    return f"def {entry}(data=None, other=None, *extra):"


def model_prompt(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or "").strip()
    entry = str(row.get("entry_point") or "").strip()
    named_prefix = f"Write a Python function named {entry}." if entry else ""
    if named_prefix and prompt.startswith(named_prefix):
        return prompt[len(named_prefix) :].strip()
    return prompt


def canonical_model_signature(
    signature: str, config: dict[str, Any] | None = None
) -> str:
    tokenization = config.get("tokenization", {}) if isinstance(config, dict) else {}
    name = str(tokenization.get("canonical_model_signature_name") or CANONICAL_MODEL_SIGNATURE_NAME)
    try:
        parsed = ast.parse(signature + "\n    pass\n")
    except SyntaxError:
        return signature
    function = parsed.body[0] if parsed.body else None
    if not isinstance(function, ast.FunctionDef):
        return signature
    opening = signature.find("(")
    return f"def {name}{signature[opening:]}" if opening >= 0 else signature


def shared_vocabulary_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("tokenization", {}).get("shared_source_target_vocabulary"))


def target_token_offset(config: dict[str, Any], source_vocab: dict[str, int]) -> int:
    return SPECIAL_COUNT if shared_vocabulary_enabled(config) else SPECIAL_COUNT + len(source_vocab)


def source_token_offset(config: dict[str, Any], source_vocab: dict[str, int]) -> int:
    """Map encoded source IDs into the embedding segment that owns their vocabulary."""

    return target_token_offset(config, source_vocab) if shared_vocabulary_enabled(config) else SPECIAL_COUNT


def model_vocab_size(
    config: dict[str, Any], source_vocab: dict[str, int], target_vocab: dict[str, int]
) -> int:
    return target_token_offset(config, source_vocab) + len(target_vocab)


def encode_model_source(
    text: str,
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    config: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    if shared_vocabulary_enabled(config):
        ids, receipt = encode_tokens(source_tokens(text), target_vocab, stream="target")
        return ids, {**receipt, "model_stream": "shared_source_target"}
    ids, receipt = encode_tokens(source_tokens(text), source_vocab, stream="source")
    return ids, {**receipt, "model_stream": "split_source_target"}


def training_callable_signature(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return only a declared signature or a target-independent generic interface."""

    entry = str(row.get("entry_point") or "solve")
    explicit = str(row.get("callable_signature") or "").strip()
    if explicit:
        return explicit, {"source": "explicit_callable_signature", "arity": signature_arity(explicit)}
    signature = f"def {entry}(data=None, other=None, *extra):"
    return signature, {"source": "generic_prompt_only_interface", "arity": "variable"}


def signature_for_arity(entry: str, count: int) -> str:
    if count <= 0:
        return f"def {entry}():"
    if count <= 1:
        return f"def {entry}(data):"
    if count == 2:
        return f"def {entry}(data, other):"
    return f"def {entry}(data, other, *extra):"


def signature_arity(signature: str) -> int | None:
    try:
        tree = ast.parse(signature + "\n    pass\n")
    except SyntaxError:
        return None
    function = tree.body[0] if tree.body else None
    if not isinstance(function, ast.FunctionDef):
        return None
    if function.args.vararg is not None:
        return max(3, len(function.args.posonlyargs) + len(function.args.args))
    return len(function.args.posonlyargs) + len(function.args.args) + len(function.args.kwonlyargs)


def source_tokens(text: str) -> list[str]:
    return SOURCE_TOKEN_RE.findall(str(text))


def head_tail(values: list[int], limit: int) -> list[int]:
    if len(values) <= limit:
        return list(values)
    head = max(1, (limit * 3) // 4)
    return [*values[:head], *values[-(limit - head) :]]


def required_steps(
    mask: np.ndarray,
    batch_size: int,
    target_positions: int,
    *,
    sample_weights: np.ndarray | None = None,
) -> int:
    if not len(mask) or target_positions <= 0:
        return 0
    row_positions = mask.sum(axis=1)
    if sample_weights is not None and len(sample_weights) == len(row_positions) and float(sample_weights.sum()) > 0:
        mean_positions = float(np.average(row_positions, weights=sample_weights))
    else:
        mean_positions = float(row_positions.mean())
    mean_positions = max(1.0, mean_positions)
    return max(1, math.ceil(target_positions / (mean_positions * batch_size)))


def phase_target_positions(phase_reports: list[dict[str, Any]], phase_prefix: str) -> int:
    return sum(
        int(row.get("target_positions_consumed") or 0)
        for row in phase_reports
        if str(row.get("phase") or "").startswith(phase_prefix)
    )


def training_targets_complete(
    phase_reports: list[dict[str, Any]], training_config: dict[str, Any]
) -> bool:
    return bool(
        phase_target_positions(phase_reports, "licensed_module_causal_pretraining")
        >= int(training_config["pretrain_target_token_positions"])
        and phase_target_positions(phase_reports, "prompt_signature_body_sft")
        >= int(training_config["sft_target_token_positions"])
    )


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
    sample_weights: np.ndarray | None,
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
    probabilities = normalized_sampling_probabilities(sample_weights, len(inputs))
    consumed = 0
    steps = 0
    losses: list[float] = []
    started = time.perf_counter()
    epoch = 0
    model.train()
    while consumed < target_positions and steps < max_steps:
        if probabilities is None:
            random.Random(seed + epoch).shuffle(order)
        else:
            order = np.random.default_rng(seed + epoch).choice(
                len(inputs), size=len(inputs), replace=True, p=probabilities
            ).tolist()
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
        "weighted_sampling": probabilities is not None,
        "sampling_effective_size": (
            round(float(1.0 / np.square(probabilities).sum()), 3)
            if probabilities is not None
            else len(inputs)
        ),
        "external_inference_calls": 0,
    }


def normalized_sampling_probabilities(
    sample_weights: np.ndarray | None, row_count: int
) -> np.ndarray | None:
    if sample_weights is None:
        return None
    weights = np.asarray(sample_weights, dtype=np.float64)
    if len(weights) != row_count or row_count == 0 or np.any(weights < 0):
        raise ValueError("sampling weights must be non-negative and match the training row count")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("sampling weights must contain positive mass")
    return weights / total


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
    generation_started = time.perf_counter()
    model.eval()
    candidates: list[dict[str, Any]] = []
    syntax_valid = 0
    empty_rejected = 0
    decode_faults = 0
    timed_out = 0
    cross_task_duplicate_body_rejected = 0
    global_body_hashes: set[str] = set()
    target_mode = str(config["tokenization"]["target_mode"])
    for task in eval_rows:
        task_started = time.perf_counter()
        visible = visible_eval_source(task)
        source_ids, receipt = encode_model_source(visible, source_vocab, target_vocab, config)
        if receipt.get("unknown_token_count"):
            decode_faults += 1
            continue
        source_ids = head_tail(source_ids, int(config["tokenization"]["max_source_tokens"]))
        prompt_ids = [GLOBAL_BOS_ID]
        source_offset = source_token_offset(config, source_vocab)
        target_offset = target_token_offset(config, source_vocab)
        prompt_ids.extend(source_offset + int(value) for value in source_ids)
        prompt_ids.append(SOURCE_TARGET_SEPARATOR_ID)
        prompt_ids.append(target_offset + int(target_vocab["<bos>"]))
        beams = decode_beams(
            model,
            prompt_ids,
            target_vocab,
            target_offset=target_offset,
            allowed_names=signature_names(callable_signature(task)),
            config=config["evaluation"],
            target_mode=target_mode,
            started=task_started,
            mx=mx,
        )
        if time.perf_counter() - task_started >= float(config["evaluation"]["timeout_seconds_per_task"]):
            timed_out += 1
        seen_code: set[str] = set()
        for rank, beam in enumerate(beams, start=1):
            body, meta = decode_candidate_body_tokens(beam["tokens"], {}, target_mode=target_mode)
            if not body:
                empty_rejected += 1
                continue
            code = render_visible_signature(callable_signature(task), body)
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            code_hash = sha(code)
            if code_hash in seen_code:
                continue
            body_hash = normalized_function_body_hash(tree)
            if body_hash in global_body_hashes:
                cross_task_duplicate_body_rejected += 1
                continue
            global_body_hashes.add(body_hash)
            seen_code.add(code_hash)
            syntax_valid += 1
            candidates.append(
                candidate_row := {
                    "task_id": str(task.get("task_id") or ""),
                    "source_task_id": str(task.get("source_task_id") or ""),
                    "entry_point": str(task.get("entry_point") or "solve"),
                    "phase": "private_eval",
                    "candidate_source": "standard_causal_transformer_survival",
                    "candidate_generation_mode": (
                        "direct_decoder_only_causal_semantic_plan_body_tokens"
                        if learned_semantic_ir_plan_body_target_mode(target_mode)
                        else "direct_decoder_only_causal_body_tokens"
                    ),
                    "code": code,
                    "candidate_sha256": code_hash,
                    "substrate_arm": "transformer_hybrid_survival",
                    "substrate_adapter": "mlx_standard_causal_transformer",
                    "rank": rank,
                    "rank_score": round(float(beam["score"]), 8),
                    "decoded_token_count": len(beam["tokens"]),
                    "decoded_token_sha256": sha(" ".join(beam["tokens"])),
                    "decoded_target_tokens": list(beam["tokens"]),
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
                        "learned_semantic_plan_prefix": learned_semantic_ir_plan_body_target_mode(
                            target_mode
                        ),
                    },
                }
            )
            candidate_row["semantic_ir"] = semantic_ir.candidate_receipt(
                code,
                prompt=str(task.get("prompt") or ""),
                callable_signature=callable_signature(task),
                learned_prefix_tokens=list(meta.get("learned_plan_prefix_tokens") or []),
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
        "cross_task_duplicate_body_rejected_count": cross_task_duplicate_body_rejected,
        "fallback_return_count": 0,
        "template_renderer_router_tool_credit_count": 0,
        "runtime_ms": int((time.perf_counter() - generation_started) * 1000),
        "accepted_verified_output_per_second": None,
    }


def run_preference_canary(
    *,
    reference_model: Any,
    model_cfg: CausalTransformerConfig,
    checkpoint: Path,
    checkpoint_dir: Path,
    stage: Stage,
    config: dict[str, Any],
    base_verifier: dict[str, Any],
    base_candidates: list[dict[str, Any]],
    base_decode: dict[str, Any],
    mx: Any,
    nn: Any,
    optim: Any,
) -> dict[str, Any]:
    preference_cfg = config["preference"]
    canary_started = time.perf_counter()
    train_generation_config = copy.deepcopy(config)
    train_generation_config["evaluation"].update(
        {
            "fanout": int(preference_cfg["candidate_fanout"]),
            "beam_width": int(preference_cfg["beam_width"]),
            "branching_factor": int(preference_cfg["branching_factor"]),
            "completion_pool_multiplier": int(preference_cfg["completion_pool_multiplier"]),
            "timeout_seconds_per_task": float(preference_cfg["timeout_seconds_per_task"]),
        }
    )
    train_candidates, train_decode = generate_candidates(
        reference_model,
        stage.preference_rows,
        stage.source_vocab,
        stage.target_vocab,
        train_generation_config,
        mx=mx,
    )
    train_verifier = evaluate_all_private_candidates(stage.preference_rows, train_candidates)
    pairs, pair_summary = build_preference_pairs(
        stage.preference_rows,
        train_candidates,
        train_verifier,
        max_pairs=int(preference_cfg["max_pairs"]),
        seed=int(config["seed"]),
    )
    runtime_dir = checkpoint_dir.parent.parent / "runtime" / "standard_causal_transformer_survival_v1"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    train_candidates_path = runtime_dir / "preference_train_candidates.jsonl"
    write_jsonl(train_candidates_path, train_candidates)
    if not pairs:
        return {
            "state": "TYPED_NO_REWARD_PAIRS",
            "typed_failure": "no_private_verifier_preference_pairs",
            "preference_train_decode": train_decode,
            "preference_pair_summary": pair_summary,
            "artifacts": {"preference_train_candidates": rel(train_candidates_path)},
            "runtime_ms": int((time.perf_counter() - canary_started) * 1000),
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }

    def encode_examples(examples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return encode_sft_examples(config, examples, stage.source_vocab, stage.target_vocab)

    reward_arrays = encode_preference_arrays(
        pairs,
        encode_examples=encode_examples,
        visible_source=visible_eval_source,
    )
    control_pairs, control_pair_summary = reward_removed_pairs(
        pairs, seed=int(config["seed"]) + 104729
    )
    control_arrays = encode_preference_arrays(
        control_pairs,
        encode_examples=encode_examples,
        visible_source=visible_eval_source,
    )
    reward_model = build_model(model_cfg, mx=mx, nn=nn)
    reward_model.load_weights(str(checkpoint))
    reward_training = train_dpo(
        reward_model,
        reference_model,
        reward_arrays,
        optimizer_steps=int(preference_cfg["optimizer_steps"]),
        batch_size=int(preference_cfg["batch_size"]),
        learning_rate=float(preference_cfg["learning_rate"]),
        beta=float(preference_cfg["beta"]),
        gradient_clip_norm=float(preference_cfg["gradient_clip_norm"]),
        seed=int(config["seed"]) + 17,
        mx=mx,
        nn=nn,
        optim=optim,
    )
    reward_checkpoint = checkpoint_dir / "standard_causal_transformer_preference_reward_v1.npz"
    reward_model.save_weights(str(reward_checkpoint))
    mx.eval(reward_model.parameters())
    reward_candidates, reward_decode = generate_candidates(
        reward_model,
        stage.eval_rows,
        stage.source_vocab,
        stage.target_vocab,
        config,
        mx=mx,
    )
    reward_verifier = evaluate_private_candidates(stage.eval_rows, reward_candidates)
    reward_summary = verifier_generation_summary(reward_verifier, reward_decode, reward_candidates)
    reward_candidates_path = runtime_dir / "preference_reward_eval_candidates.jsonl"
    write_jsonl(reward_candidates_path, reward_candidates)
    del reward_model
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()

    control_model = build_model(model_cfg, mx=mx, nn=nn)
    control_model.load_weights(str(checkpoint))
    control_training = train_dpo(
        control_model,
        reference_model,
        control_arrays,
        optimizer_steps=int(preference_cfg["optimizer_steps"]),
        batch_size=int(preference_cfg["batch_size"]),
        learning_rate=float(preference_cfg["learning_rate"]),
        beta=float(preference_cfg["beta"]),
        gradient_clip_norm=float(preference_cfg["gradient_clip_norm"]),
        seed=int(config["seed"]) + 17,
        mx=mx,
        nn=nn,
        optim=optim,
    )
    control_checkpoint = checkpoint_dir / "standard_causal_transformer_preference_control_v1.npz"
    control_model.save_weights(str(control_checkpoint))
    mx.eval(control_model.parameters())
    control_candidates, control_decode = generate_candidates(
        control_model,
        stage.eval_rows,
        stage.source_vocab,
        stage.target_vocab,
        config,
        mx=mx,
    )
    control_verifier = evaluate_private_candidates(stage.eval_rows, control_candidates)
    control_summary = verifier_generation_summary(control_verifier, control_decode, control_candidates)
    control_candidates_path = runtime_dir / "preference_control_eval_candidates.jsonl"
    write_jsonl(control_candidates_path, control_candidates)

    base_summary = verifier_generation_summary(base_verifier, base_decode, base_candidates)
    reward_improves_behavior = (
        reward_summary["passed_task_count"] > base_summary["passed_task_count"]
        and reward_summary["passed_task_count"] > control_summary["passed_task_count"]
    ) or (
        reward_summary["passed_task_count"] >= base_summary["passed_task_count"]
        and reward_summary["passed_task_count"] >= control_summary["passed_task_count"]
        and reward_summary["rank1_passed_task_count"]
        > max(base_summary["rank1_passed_task_count"], control_summary["rank1_passed_task_count"])
    )
    adoption_state = "QUALIFIED_SHADOW" if reward_improves_behavior else "NOT_ADOPTED"
    return {
        "state": "GREEN",
        "adoption_state": adoption_state,
        "preference_train_decode": train_decode,
        "preference_pair_summary": pair_summary,
        "reward_removed_control": control_pair_summary,
        "reward_present_training": reward_training,
        "reward_removed_training": control_training,
        "base_heldout": base_summary,
        "reward_present_heldout": reward_summary,
        "reward_removed_heldout": control_summary,
        "reward_improves_behavior": reward_improves_behavior,
        "artifacts": {
            "preference_train_candidates": rel(train_candidates_path),
            "reward_checkpoint": rel(reward_checkpoint),
            "reward_eval_candidates": rel(reward_candidates_path),
            "control_checkpoint": rel(control_checkpoint),
            "control_eval_candidates": rel(control_candidates_path),
        },
        "claim_boundary": {
            "deterministic_semantic_ir_repair_credit": 0,
            "learned_generation_credit": "shadow only until independent audit and route replacement",
            "non_claims": [
                "private preference canary is not public transfer",
                "shadow checkpoint is not runtime serving",
                "semantic-IR receipts and verifier labels are not candidate generation",
            ],
        },
        "runtime_ms": int((time.perf_counter() - canary_started) * 1000),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def run_generation_mode_canary(
    *,
    reference_model: Any,
    stage: Stage,
    config: dict[str, Any],
    batched_candidates: list[dict[str, Any]],
    batched_decode: dict[str, Any],
    batched_verifier: dict[str, Any],
    checkpoint_dir: Path,
    mx: Any,
) -> dict[str, Any]:
    serial_config = copy.deepcopy(config)
    serial_config["evaluation"]["batched_beam_advance"] = False
    serial_candidates, serial_decode = generate_candidates(
        reference_model,
        stage.eval_rows,
        stage.source_vocab,
        stage.target_vocab,
        serial_config,
        mx=mx,
    )
    serial_verifier = evaluate_private_candidates(stage.eval_rows, serial_candidates)
    batched_summary = verifier_generation_summary(
        batched_verifier, batched_decode, batched_candidates
    )
    serial_summary = verifier_generation_summary(serial_verifier, serial_decode, serial_candidates)
    batched_hashes = [str(row.get("candidate_sha256") or "") for row in batched_candidates]
    serial_hashes = [str(row.get("candidate_sha256") or "") for row in serial_candidates]
    candidate_manifest_equal = batched_hashes == serial_hashes
    behavior_non_regression = (
        batched_summary["passed_task_count"] >= serial_summary["passed_task_count"]
        and batched_summary["rank1_passed_task_count"] >= serial_summary["rank1_passed_task_count"]
    )
    integrity_non_regression = (
        batched_summary["integrity_mismatch_count"]
        <= serial_summary["integrity_mismatch_count"]
    )
    serial_runtime = int(serial_summary["generation_runtime_ms"] or 0)
    batched_runtime = int(batched_summary["generation_runtime_ms"] or 0)
    speedup = serial_runtime / max(1, batched_runtime)
    runtime_qualified = (
        candidate_manifest_equal
        and behavior_non_regression
        and integrity_non_regression
        and speedup > 1.0
    )
    behavior_qualified = (
        runtime_qualified
        and batched_summary["passed_task_count"] > 0
    )
    runtime_dir = checkpoint_dir.parent.parent / "runtime" / "standard_causal_transformer_survival_v1"
    serial_candidates_path = runtime_dir / "serial_beam_eval_candidates.jsonl"
    write_jsonl(serial_candidates_path, serial_candidates)
    return {
        "state": "GREEN",
        "adoption_state": (
            "BATCHED_DEFAULT"
            if behavior_qualified
            else "BATCHED_RUNTIME_ONLY"
            if runtime_qualified
            else "NOT_ADOPTED"
        ),
        "serial": serial_summary,
        "batched": batched_summary,
        "candidate_manifest_equal": candidate_manifest_equal,
        "behavior_non_regression": behavior_non_regression,
        "integrity_non_regression": integrity_non_regression,
        "generation_speedup": round(speedup, 6),
        "artifacts": {"serial_candidates": rel(serial_candidates_path)},
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "non_claims": [
            "beam batching is a runtime optimization, not learned capability",
            "speedup does not imply public transfer or runtime-serving readiness",
        ],
    }


def verifier_generation_summary(
    verifier: dict[str, Any], decode: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    passed = int(verifier.get("trained_passed") or 0)
    rank1 = int(verifier.get("trained_rank1_passed") or 0)
    runtime_seconds = float(decode.get("runtime_ms") or 0) / 1000.0
    from candidate_integrity import recompute_candidate_integrity

    integrity = [recompute_candidate_integrity(row) for row in candidates]
    verified = sum(bool(row.get("integrity_verified")) for row in integrity)
    mismatches = sum(bool(row.get("integrity_mismatch")) for row in integrity)
    return {
        "passed_task_count": passed,
        "rank1_passed_task_count": rank1,
        "candidate_count": len(candidates),
        "integrity_verified_candidate_count": verified,
        "integrity_mismatch_count": mismatches,
        "generation_runtime_ms": int(decode.get("runtime_ms") or 0),
        "accepted_verified_output_per_second": round(passed / max(1e-9, runtime_seconds), 8)
        if runtime_seconds > 0
        else None,
    }


def normalized_function_body_hash(tree: ast.Module) -> str:
    function = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    body = function.body if function is not None else tree.body
    payload = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
    return sha(payload)


def decode_beams(
    model: Any,
    prompt_ids: list[int],
    target_vocab: dict[str, int],
    *,
    target_offset: int,
    allowed_names: set[str],
    config: dict[str, Any],
    target_mode: str,
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
        expansion_specs: list[dict[str, Any]] = []
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
                target_mode=target_mode,
            )
            for local_id, token, log_probability in choices:
                if local_id == eos_local:
                    complete.append({"tokens": list(beam["tokens"]), "score": beam["score"] + log_probability})
                    continue
                expansion_specs.append(
                    {
                        "beam": beam,
                        "local_id": local_id,
                        "token": token,
                        "log_probability": log_probability,
                    }
                )
        expanded = (
            batched_beam_advance(
                model,
                expansion_specs,
                target_offset=target_offset,
                mx=mx,
            )
            if config.get("batched_beam_advance", True)
            else serial_beam_advance(
                model,
                expansion_specs,
                target_offset=target_offset,
                mx=mx,
            )
        )
        dedup: dict[tuple[str, ...], dict[str, Any]] = {}
        for row in expanded:
            key = tuple(row["tokens"])
            if key not in dedup or row["score"] > dedup[key]["score"]:
                dedup[key] = row
        beams = sorted(
            dedup.values(),
            key=lambda row: beam_rank_score(row, float(config["length_penalty"])),
            reverse=True,
        )[: int(config["beam_width"])]
        complete = prune_complete_beams(
            complete,
            limit=completion_pool_target(config),
            length_penalty=float(config["length_penalty"]),
        )
        if len(complete) >= completion_pool_target(config):
            break
        if not beams:
            break
    for beam in beams:
        if generation_prefix_complete(beam["tokens"], target_mode=target_mode):
            complete.append({"tokens": beam["tokens"], "score": beam["score"]})
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in complete:
        key = tuple(row["tokens"])
        if key not in unique or row["score"] > unique[key]["score"]:
            unique[key] = row
    return sorted(
        unique.values(),
        key=lambda row: beam_rank_score(row, float(config["length_penalty"])),
        reverse=True,
    )[: int(config["fanout"])]


def batched_beam_advance(
    model: Any,
    expansion_specs: list[dict[str, Any]],
    *,
    target_offset: int,
    mx: Any,
) -> list[dict[str, Any]]:
    """Advance equal-length beam branches in one MLX call while preserving per-branch caches."""

    if not expansion_specs:
        return []
    tokens = mx.array(
        [[target_offset + int(spec["local_id"])] for spec in expansion_specs],
        dtype=mx.int32,
    )
    layer_count = len(expansion_specs[0]["beam"]["cache"])
    batched_cache = []
    for layer_index in range(layer_count):
        keys = [spec["beam"]["cache"][layer_index][0] for spec in expansion_specs]
        values = [spec["beam"]["cache"][layer_index][1] for spec in expansion_specs]
        batched_cache.append(
            (
                mx.contiguous(mx.concatenate(keys, axis=0)),
                mx.contiguous(mx.concatenate(values, axis=0)),
            )
        )
    logits, next_cache = model(tokens, batched_cache)
    mx.eval(logits, *[value for pair in next_cache for value in pair])
    rows = []
    for index, spec in enumerate(expansion_specs):
        branch_cache = [(key[index : index + 1], value[index : index + 1]) for key, value in next_cache]
        beam = spec["beam"]
        rows.append(
            {
                "tokens": [*beam["tokens"], str(spec["token"])],
                "score": float(beam["score"]) + float(spec["log_probability"]),
                "cache": branch_cache,
                "logits": logits[index, -1],
            }
        )
    return rows


def serial_beam_advance(
    model: Any,
    expansion_specs: list[dict[str, Any]],
    *,
    target_offset: int,
    mx: Any,
) -> list[dict[str, Any]]:
    rows = []
    for spec in expansion_specs:
        beam = spec["beam"]
        next_logits, next_cache = model(
            mx.array([[target_offset + int(spec["local_id"])]], dtype=mx.int32),
            beam["cache"],
        )
        mx.eval(next_logits, *[value for pair in next_cache for value in pair])
        rows.append(
            {
                "tokens": [*beam["tokens"], str(spec["token"])],
                "score": float(beam["score"]) + float(spec["log_probability"]),
                "cache": next_cache,
                "logits": next_logits[0, -1],
            }
        )
    return rows


def completion_pool_target(config: dict[str, Any]) -> int:
    return max(
        int(config["fanout"]),
        int(config["fanout"]) * max(1, int(config.get("completion_pool_multiplier") or 1)),
    )


def beam_rank_score(row: dict[str, Any], length_penalty: float) -> float:
    length = max(1, len(row.get("tokens") or []))
    return float(row.get("score") or 0.0) / (length**max(0.0, length_penalty))


def prune_complete_beams(
    rows: list[dict[str, Any]], *, limit: int, length_penalty: float
) -> list[dict[str, Any]]:
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get("tokens") or [])
        if key not in unique or float(row.get("score") or 0.0) > float(unique[key].get("score") or 0.0):
            unique[key] = row
    return sorted(
        unique.values(),
        key=lambda row: beam_rank_score(row, length_penalty),
        reverse=True,
    )[: max(1, limit)]


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
    target_mode: str,
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
            if generation_prefix_complete(prefix, target_mode=target_mode):
                choices.append((local_id, token, float(log_probs[local_id])))
        elif token not in {"<pad>", "<bos>", "<eos>", "<unk>"}:
            body_prefix, prefix_meta = split_generation_prefix(prefix, target_mode=target_mode)
            if prefix_meta["phase"] == "plan":
                allowed = semantic_ir.plan_prefix_token_allowed(
                    prefix,
                    token,
                    body_start_token=PLAN_BODY_START_TOKEN,
                )
            else:
                allowed = token_allowed_by_policy(
                    body_prefix,
                    token,
                    policy="strict_body_token_legality_v1",
                    allowed_names=allowed_names,
                )
            if allowed:
                choices.append((local_id, token, float(log_probs[local_id])))
        if len(choices) >= branching:
            break
    return choices


def split_generation_prefix(prefix: list[str], *, target_mode: str) -> tuple[list[str], dict[str, Any]]:
    if not learned_semantic_ir_plan_body_target_mode(target_mode):
        return list(prefix), {"phase": "body", "plan_complete": False, "body_started": True}
    body, metadata = split_learned_plan_prefix_tokens(prefix)
    body_started = PLAN_BODY_START_TOKEN in prefix
    return body, {
        "phase": "body" if body_started else "plan",
        "plan_complete": bool(metadata.get("semantic_ir_plan_complete")),
        "body_started": body_started,
    }


def generation_prefix_complete(prefix: list[str], *, target_mode: str) -> bool:
    body, metadata = split_generation_prefix(prefix, target_mode=target_mode)
    if learned_semantic_ir_plan_body_target_mode(target_mode) and not (
        metadata["body_started"] and metadata["plan_complete"]
    ):
        return False
    return syntax_complete_body_prefix(body)


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
        (
            "all_family_disjoint_eval_targets_encoded",
            stage.summary["encoded_family_disjoint_eval_task_count"]
            == stage.summary["family_disjoint_eval_task_count"]
            and stage.summary["eval_target_overflow_count"] == 0,
        ),
        ("prompt_overlap_zero", stage.summary["train_eval_prompt_overlap_count"] == 0),
        ("body_overlap_zero", stage.summary["train_eval_body_overlap_count"] == 0),
        (
            "private_signature_not_derived_from_hidden_targets",
            stage.summary["private_hidden_derived_signature_count"] == 0,
        ),
        (
            "eval_signature_not_derived_from_hidden_targets",
            stage.summary["eval_hidden_derived_signature_count"] == 0,
        ),
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
    tokenization = config.get("tokenization") or {}
    if not isinstance(tokenization.get("shared_source_target_vocabulary"), bool):
        raise ValueError("source/target vocabulary mode must be explicitly boolean")
    if tokenization.get("canonical_model_signature_name") != CANONICAL_MODEL_SIGNATURE_NAME:
        raise ValueError("canonical model signature name must remain solve")
    target_mode = str(tokenization.get("target_mode") or "")
    if target_mode not in {"body_tokens", semantic_ir.PLAN_BODY_TARGET_MODE}:
        raise ValueError("target mode must be direct body tokens or learned semantic-plan plus body tokens")
    plan_budget = int(tokenization.get("semantic_plan_max_tokens") or 0)
    if learned_semantic_ir_plan_body_target_mode(target_mode) and not 8 <= plan_budget <= semantic_ir.PLAN_MAX_TOKENS:
        raise ValueError("semantic plan token budget must be between 8 and the protocol maximum")
    evaluation = config.get("evaluation") or {}
    if int(evaluation.get("holdout_family_count") or 0) < 24:
        raise ValueError("family-disjoint evaluation requires at least 24 distinct families")
    if int(evaluation.get("rows_per_family") or 0) != 1:
        raise ValueError("evaluation must use one unique semantic row per holdout family")
    preference = config.get("preference") if isinstance(config.get("preference"), dict) else {}
    if int(preference.get("max_train_tasks") or 0) < 8:
        raise ValueError("preference canary requires at least eight private training tasks")
    if not 0 < int(preference.get("max_pairs") or 0) <= int(preference["max_train_tasks"]):
        raise ValueError("preference pair budget must be positive and bounded by training tasks")
    if not 0 < int(preference.get("optimizer_steps") or 0) <= 128:
        raise ValueError("preference optimizer steps must stay within the bounded canary budget")
    if float(preference.get("beta") or 0.0) <= 0:
        raise ValueError("preference beta must be positive")


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


def compare_target_mode_canaries(
    plan_report: dict[str, Any],
    control_report: dict[str, Any],
    *,
    plan_config: dict[str, Any],
    control_config: dict[str, Any],
    plan_integrity: dict[str, Any],
    control_integrity: dict[str, Any],
) -> dict[str, Any]:
    """Compare a learned-plan target against body-only under equal exposure."""

    def matched_tokenization(config: dict[str, Any]) -> dict[str, Any]:
        tokenization = dict(config.get("tokenization") or {})
        tokenization.pop("target_mode", None)
        tokenization.pop("semantic_plan_max_tokens", None)
        return tokenization

    def metric(report: dict[str, Any]) -> dict[str, Any]:
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        training = report.get("training") if isinstance(report.get("training"), dict) else {}
        verifier = private_verifier_summary(report.get("private_verifier") or {})
        decode = report.get("decode") if isinstance(report.get("decode"), dict) else {}
        return {
            "target_mode": str(report.get("stage", {}).get("target_mode") or ""),
            "parameter_count": int(report.get("architecture", {}).get("parameter_count") or 0),
            "training_complete": training.get("complete") is True,
            "eval_loss_after": float(training.get("eval_loss_after") or float("inf")),
            "candidate_count": int(summary.get("candidate_count") or 0),
            "candidate_task_count": int(summary.get("candidate_task_count") or 0),
            "syntax_valid_candidate_count": int(summary.get("syntax_valid_candidate_count") or 0),
            "passed_task_count": int(verifier.get("passed_task_count") or 0),
            "mean_verification_reward": float(
                (report.get("private_verifier", {}).get("private_verification") or {}).get(
                    "mean_verification_reward"
                )
                or 0.0
            ),
            "generation_runtime_ms": int(decode.get("runtime_ms") or 0),
            "total_runtime_ms": int(report.get("runtime_ms") or 0),
            "integrity_mismatch_count": int(
                (plan_integrity if report is plan_report else control_integrity)
                .get("summary", {})
                .get("integrity_mismatch_count")
                or 0
            ),
            "integrity_verified_candidate_count": int(
                (plan_integrity if report is plan_report else control_integrity)
                .get("summary", {})
                .get("integrity_verified_candidate_count")
                or 0
            ),
        }

    config_pairs = (
        ("seed", plan_config.get("seed"), control_config.get("seed")),
        ("sources", plan_config.get("sources"), control_config.get("sources")),
        ("tokenization_except_target_mode", matched_tokenization(plan_config), matched_tokenization(control_config)),
        ("model", plan_config.get("model"), control_config.get("model")),
        ("training", plan_config.get("training"), control_config.get("training")),
        ("evaluation", plan_config.get("evaluation"), control_config.get("evaluation")),
        ("preference", plan_config.get("preference"), control_config.get("preference")),
        ("boundaries", plan_config.get("boundaries"), control_config.get("boundaries")),
    )
    matched_checks = {name: left == right for name, left, right in config_pairs}
    plan = metric(plan_report)
    control = metric(control_report)
    matched_checks["target_modes_expected"] = (
        plan["target_mode"] == semantic_ir.PLAN_BODY_TARGET_MODE
        and str((plan_config.get("tokenization") or {}).get("target_mode") or "")
        == semantic_ir.PLAN_BODY_TARGET_MODE
        and control["target_mode"] == "body_tokens"
        and str((control_config.get("tokenization") or {}).get("target_mode") or "") == "body_tokens"
    )
    boundaries_clean = all(
        int(report.get(key) or 0) == 0
        for report in (plan_report, control_report)
        for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count")
    )
    behavior_better = plan["passed_task_count"] > control["passed_task_count"]
    coverage_non_regression = plan["candidate_task_count"] >= control["candidate_task_count"]
    reward_non_regression = plan["mean_verification_reward"] >= control["mean_verification_reward"]
    integrity_clean = plan["integrity_mismatch_count"] == control["integrity_mismatch_count"] == 0
    adopted = (
        all(matched_checks.values())
        and boundaries_clean
        and integrity_clean
        and behavior_better
        and coverage_non_regression
        and reward_non_regression
    )
    rejection_reasons: list[str] = []
    if not all(matched_checks.values()):
        rejection_reasons.append("matched_configuration_failed")
    if not boundaries_clean:
        rejection_reasons.append("no_cheat_boundary_failed")
    if not integrity_clean:
        rejection_reasons.append("candidate_integrity_failed")
    if not behavior_better:
        rejection_reasons.append("no_verifier_pass_gain")
    if not coverage_non_regression:
        rejection_reasons.append("candidate_task_coverage_regressed")
    if not reward_non_regression:
        rejection_reasons.append("mean_verification_reward_regressed")
    return {
        "policy": "project_theseus_standard_causal_target_mode_matched_comparison_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(matched_checks.values()) and boundaries_clean else "RED",
        "adoption_state": "ADOPTED" if adopted else "NOT_ADOPTED",
        "adoption_rejection_reasons": rejection_reasons,
        "matched_checks": matched_checks,
        "boundaries_clean": boundaries_clean,
        "plan": plan,
        "control": control,
        "deltas": {
            "passed_task_count": plan["passed_task_count"] - control["passed_task_count"],
            "candidate_task_count": plan["candidate_task_count"] - control["candidate_task_count"],
            "candidate_count": plan["candidate_count"] - control["candidate_count"],
            "mean_verification_reward": round(
                plan["mean_verification_reward"] - control["mean_verification_reward"], 6
            ),
            "generation_runtime_ms": plan["generation_runtime_ms"] - control["generation_runtime_ms"],
            "total_runtime_ms": plan["total_runtime_ms"] - control["total_runtime_ms"],
        },
        "non_claims": [
            "lower language-model loss is not behavior improvement",
            "learned plan tokens do not receive body-generation credit",
            "deterministic Semantic IR compilation or repair is not used",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def stage_signature(config: dict[str, Any]) -> str:
    paths = [
        resolve(config["sources"]["licensed_code_manifest"]),
        resolve(config["sources"]["function_stage_report"]),
        resolve(config["sources"]["training_admission"]),
        resolve(config["sources"]["private_eval"]),
        resolve(config["tokenization"]["source_vocab"]),
        SCRIPTS / "neural_seed_open_vocab.py",
        SCRIPTS / "neural_seed_token_decoder_support.py",
        SCRIPTS / "neural_seed_token_decoder_rendering.py",
        SCRIPTS / "semantic_ir.py",
    ]
    paths.extend(admitted_training_source_paths(config))
    stage_config = {
        "seed": config["seed"],
        "sources": config["sources"],
        "tokenization": config["tokenization"],
        "training": {
            "pretrain_target_token_positions": config["training"]["pretrain_target_token_positions"],
            "private_body_sampling_weight": config["training"]["private_body_sampling_weight"],
        },
        "evaluation": {
            "holdout_family_count": config["evaluation"]["holdout_family_count"],
            "rows_per_family": config["evaluation"]["rows_per_family"],
        },
        "preference": {"max_train_tasks": config["preference"]["max_train_tasks"]},
    }
    stage_functions = (
        build_pretrain_windows,
        window_arrays,
        load_sft_examples,
        encode_sft_examples,
        encode_sft_training_examples,
        training_target_tokens,
        extend_target_vocab_for_mode,
        assign_body_balanced_sampling_weights,
        select_family_disjoint_eval,
        select_preference_train_rows,
        eval_example,
        semantic_stage_source,
        training_callable_signature,
        model_prompt,
        canonical_model_signature,
        shared_vocabulary_enabled,
        source_token_offset,
        target_token_offset,
        model_vocab_size,
        encode_model_source,
        admitted_training_source_paths,
        file_content_sha256,
    )
    stage_logic_sha256 = sha("\n".join(inspect.getsource(function) for function in stage_functions))
    payload = json.dumps(stage_config, sort_keys=True) + f"|stage_logic:{stage_logic_sha256}|" + "|".join(
        f"{rel(path)}:{file_content_sha256(path) if path.exists() and path.is_file() else 'missing'}"
        for path in paths
    )
    return sha(payload)


def admitted_training_source_paths(config: dict[str, Any]) -> list[Path]:
    admission_path = resolve(config["sources"]["training_admission"])
    if not admission_path.exists():
        return []
    admission = read_json(admission_path)
    paths = {
        resolve(str(row.get("path") or ""))
        for row in admission.get("train_admitted_sources", [])
        if isinstance(row, dict) and str(row.get("path") or "")
    }
    return sorted(paths, key=rel)


def file_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
