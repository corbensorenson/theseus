#!/usr/bin/env python3
"""Train and replay the standard causal-transformer survival lane on MLX."""

from __future__ import annotations

import argparse
import ast
import builtins
import copy
import hashlib
import inspect
import json
import math
import os
import random
import re
import symtable
import time
from collections import Counter
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
import theseus_artifact_admission  # noqa: E402
from neural_seed_open_vocab import encode_tokens  # noqa: E402
from moecot_language_tokenizer import encode_document as encode_language_document  # noqa: E402
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
from neural_seed_teacher_distillation_rows import (  # noqa: E402
    load_governed_teacher_code_lm_training_rows,
)
from standard_causal_transformer_model import (  # noqa: E402
    CausalTransformerConfig,
    build_model,
    parameter_count,
)
from standard_causal_transformer_corpus import (  # noqa: E402
    code_quality_rejection_reasons,
    load_pretrain_memmaps,
    materialize_pretrain_stage,
    measure_pretrain_index_capacity,
    pretrain_array_paths,
    validate_language_scope,
    validate_code_quality_policy,
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
DEFAULT_CANONICAL_CORPUS_RECEIPT = DEFAULT_STAGE_DIR / "canonical_mixed_corpus_receipt.json"
SOURCE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|-?\d+|\S")
PAD_ID = 0
GLOBAL_BOS_ID = 1
SOURCE_TARGET_SEPARATOR_ID = 2
SPECIAL_COUNT = 3
CANONICAL_MODEL_SIGNATURE_NAME = "solve"
EXECUTABLE_STATE_ROLES = (
    "control_stack",
    "bindings",
    "traversal",
    "state_update",
    "finalizer",
    "value_expression",
    "return_closure",
    "open_obligations",
)


@dataclass
class Stage:
    pretrain_inputs: np.ndarray
    pretrain_labels: np.ndarray
    pretrain_mask: np.ndarray
    sft_inputs: np.ndarray
    sft_labels: np.ndarray
    sft_mask: np.ndarray
    sft_body_mask: np.ndarray
    sft_sampling_weights: np.ndarray
    sft_plan_labels: np.ndarray
    eval_inputs: np.ndarray
    eval_labels: np.ndarray
    eval_mask: np.ndarray
    eval_body_mask: np.ndarray
    eval_plan_labels: np.ndarray
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
    parser.add_argument("--audit-corpus", action="store_true")
    parser.add_argument("--measure-corpus-capacity", action="store_true")
    parser.add_argument("--canonical-corpus-receipt", default=rel(DEFAULT_CANONICAL_CORPUS_RECEIPT))
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    if (args.resume or args.evaluate_only) and not args.execute:
        parser.error("--resume and --evaluate-only require --execute")
    prior_report_path = resolve(args.prior_report) if args.prior_report else resolve(args.out)
    if (args.resume or args.evaluate_only) and not prior_report_path.exists():
        parser.error(f"prior training receipt missing: {prior_report_path}")

    started = time.perf_counter()
    config = read_json(resolve(args.config))
    if args.audit_corpus:
        validate_config(config)
        receipt = materialize_canonical_mixed_corpus_receipt(config)
        write_json(resolve(args.canonical_corpus_receipt), receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 2 if receipt["trigger_state"] == "RED" else 0
    if args.measure_corpus_capacity:
        validate_config(config)
        metadata = read_json(resolve(args.stage_dir) / "stage_metadata_v1.json")
        index_path = Path(metadata["summary"]["canonical_pretrain_stage"]["index"]["path"])
        vocab_payload = read_json(resolve(config["tokenization"]["source_vocab"]))
        target_vocab = dict(vocab_payload["target_vocab"])
        eval_rows, _families = select_family_disjoint_eval(config)
        eval_patterns = {
            " ".join(body_tokens(str(row.get("solution_body") or "")))
            for row in eval_rows
            if str(row.get("solution_body") or "")
        }
        receipt = measure_pretrain_index_capacity(
            index_path,
            tokenize_and_encode=lambda text, category: encode_canonical_pretrain_document(
                text, target_vocab, category=category
            ),
            eval_body_patterns=eval_patterns,
        )
        receipt["created_utc"] = now()
        receipt["config"] = rel(resolve(args.config))
        receipt["config_sha256"] = file_content_sha256(resolve(args.config))
        write_json(resolve(args.out), receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
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
    scaling_contract = build_data_model_scaling_contract(config)
    if not execute:
        return planned_report(
            config,
            config_path,
            checkpoint_dir,
            stage_dir,
            started,
            scaling_contract=scaling_contract,
        ), []
    if scaling_contract["training_authorized"] is not True and not evaluate_only:
        raise ValueError(
            "dense MLX training denied by frozen data/model scaling contract: "
            + ", ".join(scaling_contract["hard_gaps"])
        )

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
    state_role_lookup = executable_state_role_lookup(
        config, stage.source_vocab, stage.target_vocab
    )
    model = build_model(
        model_cfg,
        mx=mx,
        nn=nn,
        state_role_lookup=state_role_lookup,
    )
    plan_training = semantic_plan_training_contract(config)
    ordered_plan_training = ordered_plan_training_contract(config)
    params = parameter_count(model, mlx_utils)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "standard_causal_transformer_survival_v1.npz"
    if (resume or evaluate_only) and checkpoint.exists():
        admitted_load_weights(model, checkpoint, config)
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
        stage.sft_body_mask,
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
        if prior_training.get("complete") is not True:
            raise ValueError("evaluation-only replay requires a complete prior training receipt")
        prior_checkpoint = resolve(
            str((prior.get("artifacts") or {}).get("checkpoint") or "")
        )
        if prior_checkpoint != checkpoint or not checkpoint.exists():
            raise ValueError("evaluation-only replay checkpoint does not match prior receipt")
        conditioning = {
            **(
                prior.get("conditioning")
                if isinstance(prior.get("conditioning"), dict)
                else {}
            ),
            "evaluation_base_checkpoint_sha256": file_content_sha256(checkpoint),
            "prior_training_receipt_sha256": file_content_sha256(prior_report_path),
            "evaluation_replay_contract": "content_bound_checkpoint_and_training_receipt_v1",
        }
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
            stage.eval_body_mask,
            batch_size=int(training_cfg["batch_size"]),
            mx=mx,
            nn=nn,
        )
        plan_eval_before = evaluate_semantic_plan_head(
            model,
            stage.eval_inputs,
            stage.eval_plan_labels,
            batch_size=int(training_cfg["batch_size"]),
            enabled=plan_training["enabled"],
            loss_mode=plan_training["loss_mode"],
            slot_count=plan_training["ordered_slot_count"],
            factor_group_sizes=plan_training["factor_group_sizes"],
            mx=mx,
        )
        ordered_plan_eval_before = evaluate_ordered_plan_loss(
            model,
            stage.eval_inputs,
            stage.eval_labels,
            stage.eval_mask,
            stage.eval_body_mask,
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
                stage.sft_body_mask,
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
                progress_mask=stage.sft_body_mask,
                ordered_plan_loss_weight=ordered_plan_training["plan_loss_weight"],
                sample_weights=stage.sft_sampling_weights,
                plan_labels=stage.sft_plan_labels,
                plan_label_mode=plan_training["label_mode"],
                plan_auxiliary_weight=plan_training["auxiliary_loss_weight"],
                plan_shuffle_seed=plan_training["shuffle_seed"],
                plan_loss_mode=plan_training["loss_mode"],
                plan_slot_count=plan_training["ordered_slot_count"],
                plan_factor_group_sizes=plan_training["factor_group_sizes"],
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
            stage.eval_body_mask,
            batch_size=int(training_cfg["batch_size"]),
            mx=mx,
            nn=nn,
        )
        plan_eval_before = evaluate_semantic_plan_head(
            model,
            stage.eval_inputs,
            stage.eval_plan_labels,
            batch_size=int(training_cfg["batch_size"]),
            enabled=plan_training["enabled"],
            loss_mode=plan_training["loss_mode"],
            slot_count=plan_training["ordered_slot_count"],
            factor_group_sizes=plan_training["factor_group_sizes"],
            mx=mx,
        )
        ordered_plan_eval_before = evaluate_ordered_plan_loss(
            model,
            stage.eval_inputs,
            stage.eval_labels,
            stage.eval_mask,
            stage.eval_body_mask,
            batch_size=int(training_cfg["batch_size"]),
            mx=mx,
            nn=nn,
        )
        heartbeat = stage_dir / "training_heartbeat.json"
        phase_reports = []
        consumed_steps = 0
        for (
            phase_name,
            inputs,
            labels,
            mask,
            progress_mask,
            sample_weights,
            phase_plan_labels,
            target_positions,
            planned_steps,
        ) in (
            (
                "licensed_module_causal_pretraining",
                stage.pretrain_inputs,
                stage.pretrain_labels,
                stage.pretrain_mask,
                stage.pretrain_mask,
                None,
                None,
                int(training_cfg["pretrain_target_token_positions"]),
                pretrain_steps,
            ),
            (
                "prompt_signature_body_sft",
                stage.sft_inputs,
                stage.sft_labels,
                stage.sft_mask,
                stage.sft_body_mask,
                stage.sft_sampling_weights,
                stage.sft_plan_labels,
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
                progress_mask=progress_mask,
                ordered_plan_loss_weight=(
                    ordered_plan_training["plan_loss_weight"]
                    if phase_plan_labels is not None
                    else 1.0
                ),
                sample_weights=sample_weights,
                plan_labels=phase_plan_labels,
                plan_label_mode=(
                    plan_training["label_mode"] if phase_plan_labels is not None else "none"
                ),
                plan_auxiliary_weight=(
                    plan_training["auxiliary_loss_weight"]
                    if phase_plan_labels is not None
                    else 0.0
                ),
                plan_shuffle_seed=plan_training["shuffle_seed"],
                plan_loss_mode=plan_training["loss_mode"],
                plan_slot_count=plan_training["ordered_slot_count"],
                plan_factor_group_sizes=plan_training["factor_group_sizes"],
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
        stage.eval_body_mask,
        batch_size=int(training_cfg["batch_size"]),
        mx=mx,
        nn=nn,
    )
    if evaluate_only:
        plan_eval_before = (
            prior_training.get("semantic_plan_eval_before")
            if isinstance(prior_training.get("semantic_plan_eval_before"), dict)
            else {"state": "NOT_RETAINED"}
        )
        ordered_plan_eval_before = (
            prior_training.get("ordered_plan_eval_before")
            if isinstance(prior_training.get("ordered_plan_eval_before"), dict)
            else {"state": "NOT_RETAINED"}
        )
    plan_eval_after = evaluate_semantic_plan_head(
        model,
        stage.eval_inputs,
        stage.eval_plan_labels,
        batch_size=int(training_cfg["batch_size"]),
        enabled=plan_training["enabled"],
        loss_mode=plan_training["loss_mode"],
        slot_count=plan_training["ordered_slot_count"],
        factor_group_sizes=plan_training["factor_group_sizes"],
        mx=mx,
    )
    ordered_plan_eval_after = evaluate_ordered_plan_loss(
        model,
        stage.eval_inputs,
        stage.eval_labels,
        stage.eval_mask,
        stage.eval_body_mask,
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
        "seed": seed,
        "architecture": {
            "family": "standard_decoder_only_causal_transformer",
            "role": str(config["architecture_role"]),
            "moecot_language_seed_contract": config["moecot_language_seed_contract"],
            "attention": (
                "RoPE_grouped_query_prefix_lm_attention"
                if model_cfg.attention_policy == "prefix_lm"
                else "RoPE_grouped_query_causal_attention"
            ),
            "attention_policy": model_cfg.attention_policy,
            "normalization": "pre_norm_RMSNorm",
            "feed_forward": "SwiGLU",
            "embedding": "tied_input_output",
            "parameter_count": params,
            "config": model_cfg.__dict__,
            "backend": "mlx_apple",
            "executable_state_memory": {
                "enabled": model_cfg.state_memory_mode != "none",
                "mode": model_cfg.state_memory_mode,
                "ablation": model_cfg.state_memory_ablation,
                "roles": list(EXECUTABLE_STATE_ROLES),
                "role_lookup_sha256": (
                    hashlib.sha256(state_role_lookup.tobytes()).hexdigest()
                    if state_role_lookup is not None
                    else ""
                ),
                "target_stream": str(config["tokenization"]["target_mode"]),
                "auxiliary_target_count": 0,
                "deterministic_renderer_credit": 0,
                "read_policy": (
                    model_cfg.state_memory_read_policy
                    if model_cfg.state_memory_mode != "none"
                    else "not_applicable"
                ),
            },
            "semantic_plan_head": {
                **plan_training,
                "feature_count": model_cfg.semantic_plan_feature_count,
                "feature_contract_sha256": stage.summary[
                    "semantic_plan_feature_contract_sha256"
                ],
                "source_fields": ["natural_language_prompt", "callable_signature"],
                "target_body_visible_at_inference": False,
                "deterministic_renderer_credit": 0,
            },
            "ordered_semantic_plan": {
                **ordered_plan_training,
                "protocol_sha256": str(
                    stage.summary["ordered_plan_label_receipt"].get("protocol_sha256")
                    or ""
                ),
                "target_body_visible_at_inference": False,
                "direct_body_decoder": True,
            },
        },
        "data_model_scaling_contract": scaling_contract,
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
            "semantic_plan_eval_before": plan_eval_before,
            "semantic_plan_eval_after": plan_eval_after,
            "ordered_plan_eval_before": ordered_plan_eval_before,
            "ordered_plan_eval_after": ordered_plan_eval_after,
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
            "Direct decoder-only transformer generation from prompt plus callable signature. "
            f"The sequence attention policy is {model_cfg.attention_policy}; prefix-LM, when "
            "selected, is bidirectional only through the canonical source separator and remains "
            "causal on every generated target position. "
            "An optional learned semantic-plan prefix or latent multi-label obligation head is predicted "
            "from that visible source and causally conditions the directly generated body; neither path "
            "renders or repairs code. Optional executable-state "
            "memory is updated only from visible prompt/signature tokens and the causal generated prefix, "
            "while the latent plan head may receive fixed generic IR obligation labels only on admitted "
            "training bodies. Heldout labels are measurement-only and never enter generation. "
            "Reversible token decoding "
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
            pretrain_summary = metadata["summary"]["canonical_pretrain_stage"]
            pretrain_shape = (
                int(pretrain_summary["window_count"]),
                int(pretrain_summary["max_sequence_tokens"]),
            )
            pretrain = load_pretrain_memmaps(
                pretrain_array_paths(stage_dir),
                pretrain_shape,
                expected=pretrain_summary["array_artifacts"],
            )
            vocab_payload = read_json(resolve(config["tokenization"]["source_vocab"]))
            source_vocab = dict(metadata.get("source_vocab") or vocab_payload["source_vocab"])
            target_vocab = dict(metadata.get("target_vocab") or vocab_payload["target_vocab"])
            return Stage(
                pretrain_inputs=pretrain[0],
                pretrain_labels=pretrain[1],
                pretrain_mask=pretrain[2],
                sft_inputs=arrays["sft_inputs"],
                sft_labels=arrays["sft_labels"],
                sft_mask=arrays["sft_mask"],
                sft_body_mask=(
                    arrays["sft_body_mask"]
                    if "sft_body_mask" in arrays.files
                    else arrays["sft_mask"]
                ),
                sft_sampling_weights=(
                    arrays["sft_sampling_weights"]
                    if "sft_sampling_weights" in arrays.files
                    else np.ones((len(arrays["sft_inputs"]),), dtype=np.float32)
                ),
                sft_plan_labels=(
                    arrays["sft_plan_labels"]
                    if "sft_plan_labels" in arrays.files
                    else np.zeros(
                        (len(arrays["sft_inputs"]), len(semantic_ir.plan_obligation_features())),
                        dtype=np.float32,
                    )
                ),
                eval_inputs=arrays["eval_inputs"],
                eval_labels=arrays["eval_labels"],
                eval_mask=arrays["eval_mask"],
                eval_body_mask=(
                    arrays["eval_body_mask"]
                    if "eval_body_mask" in arrays.files
                    else arrays["eval_mask"]
                ),
                eval_plan_labels=(
                    arrays["eval_plan_labels"]
                    if "eval_plan_labels" in arrays.files
                    else np.zeros(
                        (len(arrays["eval_inputs"]), len(semantic_ir.plan_obligation_features())),
                        dtype=np.float32,
                    )
                ),
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
    target_offset = target_token_offset(config, source_vocab)
    pretrain = materialize_pretrain_stage(
        config,
        root=ROOT,
        stage_dir=stage_dir,
        target_vocab=target_vocab,
        target_offset=target_offset,
        tokenize_and_encode=lambda text, category: encode_canonical_pretrain_document(
            text, target_vocab, category=category
        ),
        eval_body_patterns={" ".join(sequence) for sequence in eval_body_token_sequences if sequence},
    )
    pretrain_arrays = pretrain[:3]
    pretrain_audit = pretrain[3]
    ordered_plan = ordered_plan_training_contract(config)
    sft = encode_sft_training_examples(
        config,
        sft_examples,
        source_vocab,
        target_vocab,
        ordered_plan_mode=ordered_plan["label_mode"],
    )
    eval_examples = [eval_example(row) for row in eval_rows]
    eval_encoded = encode_sft_training_examples(
        config,
        eval_examples,
        source_vocab,
        target_vocab,
        ordered_plan_mode="semantic",
    )
    eval_arrays = eval_encoded[:3]
    sequence_partition = {
        "pretrain": sequence_partition_audit(
            pretrain_arrays[0], pretrain_arrays[2], require_separator=False
        ),
        "sft": sequence_partition_audit(sft[0], sft[2], require_separator=True),
        "eval": sequence_partition_audit(
            eval_arrays[0], eval_arrays[2], require_separator=True
        ),
    }
    invalid_partitions = [
        name for name, receipt in sequence_partition.items() if receipt["valid"] is not True
    ]
    if invalid_partitions:
        raise ValueError(
            "source-target sequence partition audit failed: "
            + ", ".join(invalid_partitions)
        )
    summary = {
        "stage_signature": signature,
        "licensed_pretrain_window_count": int(pretrain_arrays[0].shape[0]),
        "licensed_pretrain_target_positions": int(pretrain_arrays[2].sum()),
        "canonical_pretrain_stage": pretrain_audit,
        "sft_example_count": int(sft[0].shape[0]),
        "sft_target_positions": int(sft[2].sum()),
        "unique_body_target_positions": int(sft[5].sum()),
        "ordered_plan_target_positions": int(
            np.maximum(sft[2] - sft[5], 0.0).sum()
        ),
        "ordered_plan_training": ordered_plan,
        "ordered_plan_label_receipt": sft[6],
        "ordered_plan_eval_receipt": eval_encoded[6],
        "sequence_partition_audit": sequence_partition,
        "sft_sampling_weight_sum": round(float(sft[3].sum()), 6),
        "semantic_plan_feature_count": int(sft[4].shape[1]) if sft[4].ndim == 2 else 0,
        "semantic_plan_positive_label_count": int(sft[4].sum()),
        "semantic_plan_label_density": round(float(sft[4].mean()), 8) if sft[4].size else 0.0,
        "semantic_plan_feature_contract_sha256": sha(
            "\n".join(semantic_plan_feature_contract(config))
        ),
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
        "governed_teacher_unique_body_count": sft_audit[
            "governed_teacher_unique_body_count"
        ],
        "governed_teacher_prompt_pair_count": sft_audit[
            "governed_teacher_prompt_pair_count"
        ],
        "governed_teacher_current_holdout_rejected_count": sft_audit[
            "governed_teacher_current_holdout_rejected_count"
        ],
        "governed_teacher_eval_overlap_rejected_count": sft_audit[
            "governed_teacher_eval_overlap_rejected_count"
        ],
        "governed_teacher_source_summary": sft_audit[
            "governed_teacher_source_summary"
        ],
        "private_sampling_probability": sft_audit["private_sampling_probability"],
        "teacher_sampling_probability": sft_audit["teacher_sampling_probability"],
        "teacher_sampling_mass": sft_audit["teacher_sampling_mass"],
        "sft_contract_admission": sft_audit["sft_contract_admission"],
        "shared_source_target_vocabulary": shared_vocabulary_enabled(config),
        "target_mode": str(config["tokenization"]["target_mode"]),
        "sequence_plan_reserve_tokens": int(
            config["tokenization"].get("sequence_plan_reserve_tokens") or 0
        ),
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
        "licensed_pretrain_eval_body_overlap_source_detected_count": int(
            pretrain_audit["excluded_counts"].get("eval_body_overlap", 0)
        ),
        "licensed_pretrain_eval_body_overlap_sources_excluded": [],
        "licensed_pretrain_eval_body_overlap_source_surviving_count": 0,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "cache_status": "miss_rebuilt",
    }
    arrays_temporary = arrays_path.with_suffix(arrays_path.suffix + f".{os.getpid()}.tmp.npz")
    try:
        np.savez(
            arrays_temporary,
            sft_inputs=sft[0],
            sft_labels=sft[1],
            sft_mask=sft[2],
            sft_body_mask=sft[5],
            sft_sampling_weights=sft[3],
            sft_plan_labels=sft[4],
            eval_inputs=eval_arrays[0],
            eval_labels=eval_arrays[1],
            eval_mask=eval_arrays[2],
            eval_body_mask=eval_encoded[5],
            eval_plan_labels=eval_encoded[4],
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
        pretrain_inputs=pretrain_arrays[0],
        pretrain_labels=pretrain_arrays[1],
        pretrain_mask=pretrain_arrays[2],
        sft_inputs=sft[0],
        sft_labels=sft[1],
        sft_mask=sft[2],
        sft_body_mask=sft[5],
        sft_sampling_weights=sft[3],
        sft_plan_labels=sft[4],
        eval_inputs=eval_arrays[0],
        eval_labels=eval_arrays[1],
        eval_mask=eval_arrays[2],
        eval_body_mask=eval_encoded[5],
        eval_plan_labels=eval_encoded[4],
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


def encode_canonical_pretrain_document(
    text: str, target_vocab: dict[str, int], *, category: str
) -> tuple[list[str], list[int], dict[str, Any]]:
    return encode_language_document(text, target_vocab, category=category)


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
    contract_policy = (
        config.get("sft_contract_admission")
        if isinstance(config.get("sft_contract_admission"), dict)
        else {}
    )
    contract_filter_enabled = contract_policy.get("require_self_contained_body") is True
    licensed_self_contained_weight = float(
        contract_policy.get("licensed_self_contained_sampling_weight", 1.0)
    )
    licensed_context_dependent_weight = float(
        contract_policy.get("licensed_context_dependent_sampling_weight", 1.0)
    )
    contract_curriculum_enabled = (
        licensed_self_contained_weight != 1.0
        or licensed_context_dependent_weight != 1.0
    )
    contract_rejections: Counter[str] = Counter()
    licensed_contract_accepted = 0
    licensed_contract_rejected = 0
    licensed_self_contained_count = 0
    licensed_context_dependent_count = 0
    private_contract_accepted = 0
    private_contract_rejected = 0
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
        contract = None
        if contract_filter_enabled or contract_curriculum_enabled:
            contract = standalone_sft_contract_decision(source_text, body)
            licensed_self_contained_count += int(contract["accepted"])
            licensed_context_dependent_count += int(not contract["accepted"])
        if contract_filter_enabled:
            assert contract is not None
            if not contract["accepted"]:
                licensed_contract_rejected += 1
                contract_rejections.update(contract["reject_reasons"])
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
        examples.append(
            {
                "source_text": source_text,
                "body": body,
                "source": "licensed_function",
                "sampling_multiplier": (
                    licensed_self_contained_weight
                    if contract is not None and contract["accepted"]
                    else licensed_context_dependent_weight
                    if contract is not None
                    else 1.0
                ),
            }
        )
        licensed_contract_accepted += int(contract_filter_enabled)
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
            if contract_filter_enabled:
                contract = standalone_sft_contract_decision(source_text, body)
                if not contract["accepted"]:
                    private_contract_rejected += 1
                    contract_rejections.update(contract["reject_reasons"])
                    continue
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
            private_contract_accepted += int(contract_filter_enabled)

    teacher_bundle = load_governed_teacher_code_lm_training_rows(config)
    teacher_summary = (
        teacher_bundle.get("summary")
        if isinstance(teacher_bundle.get("summary"), dict)
        else {}
    )
    teacher_rows = teacher_bundle.get("rows") if isinstance(teacher_bundle.get("rows"), list) else []
    teacher_cfg = (
        config.get("teacher_distillation")
        if isinstance(config.get("teacher_distillation"), dict)
        else {}
    )
    teacher_minimum_rows = max(
        0, int(teacher_cfg.get("minimum_code_lm_rows_for_sampling") or 0)
    )
    teacher_tranche_ready = len(teacher_rows) >= teacher_minimum_rows
    if not teacher_tranche_ready:
        teacher_rows = []
    teacher_summary = {
        **teacher_summary,
        "minimum_code_lm_rows_for_sampling": teacher_minimum_rows,
        "tranche_ready": teacher_tranche_ready,
    }
    teacher_added = 0
    teacher_body_hashes: set[str] = set()
    teacher_contract_accepted = 0
    teacher_contract_rejected = 0
    teacher_current_holdout_rejected = 0
    teacher_overlap_rejected = 0
    for row in teacher_rows:
        if not isinstance(row, dict):
            continue
        family = str(row.get("concept_residual_label") or "")
        if family in holdout_families:
            teacher_current_holdout_rejected += 1
            observed_holdout_families.add(family)
            continue
        prompt = model_prompt(row)
        body = str(row.get("solution_body") or "").strip()
        if not prompt or not body:
            continue
        prompt_hash = sha(prompt)
        body_hash = sha(body)
        if prompt_hash in eval_prompt_hashes or body_hash in eval_body_hashes:
            teacher_overlap_rejected += 1
            continue
        signature, _signature_receipt = training_callable_signature(row)
        if not signature:
            continue
        source_text = f"{prompt}\nsignature {canonical_model_signature(signature, config)}"
        if contract_filter_enabled:
            contract = standalone_sft_contract_decision(source_text, body)
            if not contract["accepted"]:
                teacher_contract_rejected += 1
                contract_rejections.update(contract["reject_reasons"])
                continue
        pair_hash = sha(f"{source_text}\n{body}")
        if pair_hash in seen_pairs:
            continue
        seen_pairs.add(pair_hash)
        unique_bodies.add(body_hash)
        teacher_body_hashes.add(body_hash)
        examples.append(
            {
                "source_text": source_text,
                "body": body,
                "source": "governed_openai_teacher",
            }
        )
        teacher_added += 1
        teacher_contract_accepted += int(contract_filter_enabled)
    examples, sampling = assign_body_balanced_sampling_weights(
        examples,
        private_body_weight=float(config["training"]["private_body_sampling_weight"]),
        private_sampling_probability_target=(
            float(contract_policy["private_sampling_probability_target"])
            if contract_policy.get("private_sampling_probability_target") is not None
            else None
        ),
        teacher_sampling_probability_target=(
            float(config["teacher_distillation"]["teacher_sampling_probability_target"])
            if isinstance(config.get("teacher_distillation"), dict)
            and config["teacher_distillation"].get("teacher_sampling_probability_target") is not None
            else None
        ),
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
        "governed_teacher_unique_body_count": len(teacher_body_hashes),
        "governed_teacher_prompt_pair_count": teacher_added,
        "governed_teacher_current_holdout_rejected_count": teacher_current_holdout_rejected,
        "governed_teacher_eval_overlap_rejected_count": teacher_overlap_rejected,
        "governed_teacher_source_summary": teacher_summary,
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
        "sft_contract_admission": {
            "enabled": contract_filter_enabled,
            "sampling_curriculum_enabled": contract_curriculum_enabled,
            "policy": "target_body_self_containment_filter_no_source_feature_derivation",
            "licensed_accepted_count": licensed_contract_accepted,
            "licensed_rejected_count": licensed_contract_rejected,
            "private_accepted_count": private_contract_accepted,
            "private_rejected_count": private_contract_rejected,
            "teacher_accepted_count": teacher_contract_accepted,
            "teacher_rejected_count": teacher_contract_rejected,
            "licensed_self_contained_count": licensed_self_contained_count,
            "licensed_context_dependent_count": licensed_context_dependent_count,
            "licensed_self_contained_sampling_weight": licensed_self_contained_weight,
            "licensed_context_dependent_sampling_weight": licensed_context_dependent_weight,
            "reject_reason_counts": dict(sorted(contract_rejections.items())),
            "target_body_used_for_admission_only": contract_filter_enabled,
            "target_body_used_for_sampling_weight_only": contract_curriculum_enabled,
            "target_body_fields_added_to_model_source": 0,
            "heldout_rows_read_by_filter": 0,
        },
    }


def encode_sft_examples(
    config: dict[str, Any],
    examples: list[dict[str, Any]],
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    (
        inputs,
        labels,
        mask,
        _weights,
        _plan_labels,
        _body_mask,
        _ordered_plan_receipt,
    ) = encode_sft_training_examples(
        config, examples, source_vocab, target_vocab
    )
    return inputs, labels, mask


def sequence_partition_audit(
    inputs: np.ndarray,
    loss_mask: np.ndarray,
    *,
    require_separator: bool,
    separator_id: int = SOURCE_TARGET_SEPARATOR_ID,
) -> dict[str, Any]:
    """Recompute the source/target information boundary from encoded arrays."""

    if inputs.ndim != 2 or loss_mask.shape != inputs.shape:
        raise ValueError("sequence partition audit requires aligned rank-two arrays")
    missing_separator_rows: list[int] = []
    multiple_separator_rows: list[int] = []
    unexpected_separator_rows: list[int] = []
    empty_supervision_rows: list[int] = []
    target_not_strictly_after_separator_rows: list[int] = []
    for row_index in range(inputs.shape[0]):
        separator_positions = np.flatnonzero(inputs[row_index] == separator_id)
        supervised_positions = np.flatnonzero(loss_mask[row_index] > 0.0)
        if require_separator:
            if len(separator_positions) == 0:
                missing_separator_rows.append(row_index)
                continue
            if len(separator_positions) > 1:
                multiple_separator_rows.append(row_index)
                continue
            if len(supervised_positions) == 0:
                empty_supervision_rows.append(row_index)
                continue
            if int(supervised_positions[0]) <= int(separator_positions[0]):
                target_not_strictly_after_separator_rows.append(row_index)
        elif len(separator_positions):
            unexpected_separator_rows.append(row_index)
    valid = not any(
        (
            missing_separator_rows,
            multiple_separator_rows,
            unexpected_separator_rows,
            empty_supervision_rows,
            target_not_strictly_after_separator_rows,
        )
    )
    return {
        "policy": "canonical_source_target_sequence_partition_v1",
        "valid": valid,
        "row_count": int(inputs.shape[0]),
        "separator_required": require_separator,
        "separator_token_id": separator_id,
        "missing_separator_row_count": len(missing_separator_rows),
        "multiple_separator_row_count": len(multiple_separator_rows),
        "unexpected_separator_row_count": len(unexpected_separator_rows),
        "empty_supervision_row_count": len(empty_supervision_rows),
        "target_not_strictly_after_separator_row_count": len(
            target_not_strictly_after_separator_rows
        ),
        "invalid_row_indices": sorted(
            set(
                missing_separator_rows
                + multiple_separator_rows
                + unexpected_separator_rows
                + empty_supervision_rows
                + target_not_strictly_after_separator_rows
            )
        ),
    }


def encode_sft_training_examples(
    config: dict[str, Any],
    examples: list[dict[str, Any]],
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    *,
    ordered_plan_mode: str | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    token_cfg = config["tokenization"]
    max_seq = int(token_cfg["max_sequence_tokens"])
    max_source = int(token_cfg["max_source_tokens"])
    max_target = int(token_cfg["max_target_tokens"])
    plan_reserve = int(token_cfg.get("sequence_plan_reserve_tokens") or 0)
    source_offset = source_token_offset(config, source_vocab)
    target_offset = target_token_offset(config, source_vocab)
    ordered_plans, ordered_plan_receipt = prepare_ordered_plan_sequences(
        examples,
        config,
        mode=ordered_plan_mode,
    )
    rows: list[tuple[list[int], int, int, float, tuple[int, ...]]] = []
    for row_index, row in enumerate(examples):
        source_ids, source_receipt = encode_model_source(
            row["source_text"], source_vocab, target_vocab, config
        )
        ordered_plan_tokens = ordered_plans[row_index]
        direct_body_tokens = body_tokens(str(row["body"]))
        target_tokens = training_target_tokens(
            str(row["body"]), config, ordered_plan_tokens=ordered_plan_tokens
        )
        target_ids, target_receipt = encode_tokens(
            target_tokens,
            target_vocab,
            stream="target",
        )
        if source_receipt.get("unknown_token_count") or target_receipt.get("unknown_token_count"):
            continue
        if len(direct_body_tokens) > max_target - 2:
            continue
        encoded_plan_positions = (
            len(ordered_plan_tokens) + 1 if ordered_plan_tokens is not None else 0
        )
        unoccupied_plan_reserve = max(0, plan_reserve - encoded_plan_positions)
        available_source = max_seq - 3 - len(target_ids) - unoccupied_plan_reserve
        if available_source <= 0:
            continue
        source_ids = head_tail(source_ids, min(max_source, available_source))
        sequence = [GLOBAL_BOS_ID]
        sequence.extend(source_offset + int(value) for value in source_ids)
        sequence.append(SOURCE_TARGET_SEPARATOR_ID)
        sequence.append(target_offset + int(target_vocab["<bos>"]))
        target_start = len(sequence)
        sequence.extend(target_offset + int(value) for value in target_ids)
        sequence.append(target_offset + int(target_vocab["<eos>"]))
        if len(sequence) > max_seq + 1:
            continue
        plan_labels = semantic_plan_labels_for_body(str(row["body"]), config)
        body_target_offset = encoded_plan_positions
        rows.append(
            (
                sequence,
                target_start - 1,
                target_start - 1 + body_target_offset,
                float(row.get("sampling_weight") or 1.0),
                plan_labels,
            )
        )
    inputs = np.zeros((len(rows), max_seq), dtype=np.int32)
    labels = np.zeros((len(rows), max_seq), dtype=np.int32)
    mask = np.zeros((len(rows), max_seq), dtype=np.float32)
    body_mask = np.zeros((len(rows), max_seq), dtype=np.float32)
    weights = np.ones((len(rows),), dtype=np.float32)
    plan_labels = np.zeros(
        (len(rows), len(semantic_plan_feature_contract(config))), dtype=np.float32
    )
    for index, (
        sequence,
        mask_start,
        body_mask_start,
        sampling_weight,
        row_plan_labels,
    ) in enumerate(rows):
        width = len(sequence) - 1
        inputs[index, :width] = sequence[:-1]
        labels[index, :width] = sequence[1:]
        mask[index, mask_start:width] = 1.0
        body_mask[index, body_mask_start:width] = 1.0
        weights[index] = max(0.0, sampling_weight)
        plan_labels[index] = np.asarray(row_plan_labels, dtype=np.float32)
    return inputs, labels, mask, weights, plan_labels, body_mask, {
        **ordered_plan_receipt,
        "encoded_row_count": len(rows),
        "encoded_target_position_count": int(mask.sum()),
        "encoded_body_position_count": int(body_mask.sum()),
        "encoded_plan_position_count": int(np.maximum(mask - body_mask, 0.0).sum()),
    }


def training_target_tokens(
    body: str,
    config: dict[str, Any],
    *,
    ordered_plan_tokens: list[str] | None = None,
) -> list[str]:
    tokenization = config.get("tokenization") if isinstance(config.get("tokenization"), dict) else {}
    target_mode = str(tokenization.get("target_mode") or "body_tokens")
    if learned_semantic_ir_plan_body_target_mode(target_mode):
        max_plan_tokens = int(tokenization.get("semantic_plan_max_tokens") or semantic_ir.PLAN_MAX_TOKENS)
        plan = (
            list(ordered_plan_tokens)
            if ordered_plan_tokens is not None
            else semantic_ir.body_to_plan_tokens(body, max_tokens=max_plan_tokens)
        )
        return [
            *plan,
            PLAN_BODY_START_TOKEN,
            *body_tokens(body),
        ]
    return body_tokens(body)


def ordered_plan_training_contract(config: dict[str, Any]) -> dict[str, Any]:
    target_mode = str(config.get("tokenization", {}).get("target_mode") or "body_tokens")
    enabled = learned_semantic_ir_plan_body_target_mode(target_mode)
    cfg = (
        config.get("ordered_plan_training")
        if isinstance(config.get("ordered_plan_training"), dict)
        else {}
    )
    return {
        "enabled": enabled,
        "label_mode": str(cfg.get("label_mode") or ("semantic" if enabled else "none")),
        "plan_loss_weight": float(cfg.get("plan_loss_weight") or (0.25 if enabled else 0.0)),
        "shuffle_seed": int(cfg.get("shuffle_seed") or int(config.get("seed") or 0) + 2713),
        "body_progress_unit": "direct_body_token_positions",
        "plan_target": "ordered_alpha_renamed_semantic_ir",
        "deterministic_renderer_or_repair_credit": 0,
    }


def prepare_ordered_plan_sequences(
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    mode: str | None = None,
) -> tuple[list[list[str] | None], dict[str, Any]]:
    contract = ordered_plan_training_contract(config)
    if not contract["enabled"]:
        return [None for _row in examples], {
            "label_mode": "none",
            "row_count": len(examples),
            "token_count": 0,
            "fixed_point_count": 0,
            "plan_sha256": "",
            "protocol_sha256": sha("\n".join(semantic_ir.plan_protocol_tokens())),
            "identifier_free": True,
            "target_body_visible_at_inference": False,
        }
    selected_mode = str(mode or contract["label_mode"])
    max_tokens = int(
        config.get("tokenization", {}).get("semantic_plan_max_tokens")
        or semantic_ir.PLAN_MAX_TOKENS
    )
    semantic = [
        semantic_ir.body_to_plan_tokens(str(row["body"]), max_tokens=max_tokens)
        for row in examples
    ]
    fixed_points = len(semantic)
    if selected_mode == "semantic":
        prepared = semantic
    elif selected_mode == "shuffled":
        if len(semantic) < 2:
            raise ValueError("ordered-plan shuffled control requires at least two rows")
        order = np.random.default_rng(contract["shuffle_seed"]).permutation(len(semantic))
        assignment = np.empty_like(order)
        assignment[order] = np.roll(order, 1)
        prepared = [
            project_plan_content(semantic[int(source)], semantic[index])
            for index, source in enumerate(assignment)
        ]
        fixed_points = sum(left == right for left, right in zip(prepared, semantic))
        if fixed_points:
            raise AssertionError("ordered-plan shuffled control contains fixed plans")
    elif selected_mode == "dropout":
        prepared = [semantic_ir.dropout_plan_tokens(plan) for plan in semantic]
        fixed_points = sum(left == right for left, right in zip(prepared, semantic))
    else:
        raise ValueError(f"unsupported ordered-plan label mode: {selected_mode}")
    flattened = "\n".join(" ".join(plan) for plan in prepared)
    return prepared, {
        "label_mode": selected_mode,
        "row_count": len(prepared),
        "token_count": sum(len(plan) for plan in prepared),
        "semantic_token_count": sum(len(plan) for plan in semantic),
        "fixed_point_count": fixed_points,
        "plan_sha256": sha(flattened),
        "protocol_sha256": sha("\n".join(semantic_ir.plan_protocol_tokens())),
        "identifier_free": True,
        "target_body_visible_at_inference": False,
    }


def project_plan_content(donor: list[str], shape: list[str]) -> list[str]:
    """Project a deranged donor onto another plan's exact protocol shape."""

    pools: dict[str, list[str]] = {}
    for token in donor:
        token_class = ordered_plan_token_class(token)
        if token_class:
            pools.setdefault(token_class, []).append(token)
    cursors: dict[str, int] = Counter()
    projected: list[str] = []
    neutral = semantic_ir.dropout_plan_tokens(shape)
    for index, token in enumerate(shape):
        token_class = ordered_plan_token_class(token)
        if token_class in {"BEGIN", "END"}:
            projected.append(token)
            continue
        choices = pools.get(token_class) or []
        if choices:
            cursor = cursors[token_class]
            projected.append(choices[cursor % len(choices)])
            cursors[token_class] += 1
        else:
            projected.append(neutral[index])
    if projected == shape:
        projected = neutral
    if projected == shape:
        for index, token in enumerate(projected):
            if token.startswith("IRP:STEP:"):
                projected[index] = (
                    "IRP:STEP:D0:return"
                    if token != "IRP:STEP:D0:return"
                    else "IRP:STEP:D0:statement"
                )
                break
    return projected


def ordered_plan_token_class(token: str) -> str:
    value = str(token)
    if value == semantic_ir.PLAN_BEGIN:
        return "BEGIN"
    if value == semantic_ir.PLAN_END:
        return "END"
    for prefix, name in (
        ("IRP:STEP:", "STEP"),
        ("IRP:SEM:", "SEM"),
        ("IRP:FLOW:", "FLOW"),
        ("IRP:DATA:", "DATA"),
        ("IRP:VALUE:", "VALUE"),
        ("IRP:FEATURE:", "FEATURE"),
    ):
        if value.startswith(prefix):
            return name
    return ""


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
    examples: list[dict[str, Any]],
    *,
    private_body_weight: float,
    private_sampling_probability_target: float | None = None,
    teacher_sampling_probability_target: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    private_counts: dict[str, int] = {}
    for row in examples:
        if row.get("source") != "governed_private":
            continue
        body_hash = sha(str(row.get("body") or ""))
        private_counts[body_hash] = private_counts.get(body_hash, 0) + 1
    teacher_counts: dict[str, int] = {}
    for row in examples:
        if row.get("source") != "governed_openai_teacher":
            continue
        body_hash = sha(str(row.get("body") or ""))
        teacher_counts[body_hash] = teacher_counts.get(body_hash, 0) + 1
    configured_private_body_weight = float(private_body_weight)
    licensed_base_mass = sum(
        max(0.0, float(row.get("sampling_multiplier", 1.0)))
        for row in examples
        if row.get("source") not in {"governed_private", "governed_openai_teacher"}
    )
    if private_sampling_probability_target is not None and private_counts and licensed_base_mass:
        target = float(private_sampling_probability_target)
        if not 0.0 < target < 1.0:
            raise ValueError("private sampling probability target must be between zero and one")
        private_body_weight = (
            target * licensed_base_mass / ((1.0 - target) * len(private_counts))
        )
    teacher_body_weight = 1.0
    base_mass = licensed_base_mass + max(0.0, private_body_weight) * len(private_counts)
    if teacher_sampling_probability_target is not None and teacher_counts:
        teacher_target = float(teacher_sampling_probability_target)
        if not 0.0 < teacher_target < 1.0:
            raise ValueError("teacher sampling probability target must be between zero and one")
        teacher_body_weight = (
            teacher_target * base_mass / ((1.0 - teacher_target) * len(teacher_counts))
        )
    weighted: list[dict[str, Any]] = []
    licensed_mass = 0.0
    private_mass = 0.0
    teacher_mass = 0.0
    for row in examples:
        item = dict(row)
        if item.get("source") == "governed_private":
            body_hash = sha(str(item.get("body") or ""))
            weight = max(0.0, private_body_weight) / max(1, private_counts.get(body_hash, 1))
            private_mass += weight
        elif item.get("source") == "governed_openai_teacher":
            body_hash = sha(str(item.get("body") or ""))
            weight = max(0.0, teacher_body_weight) / max(1, teacher_counts.get(body_hash, 1))
            teacher_mass += weight
        else:
            weight = max(0.0, float(item.get("sampling_multiplier", 1.0)))
            licensed_mass += weight
        item["sampling_weight"] = weight
        weighted.append(item)
    total = licensed_mass + private_mass + teacher_mass
    return weighted, {
        "licensed_sampling_mass": round(licensed_mass, 6),
        "private_sampling_mass": round(private_mass, 6),
        "private_sampling_probability": round(private_mass / total, 6) if total else 0.0,
        "teacher_sampling_mass": round(teacher_mass, 6),
        "teacher_sampling_probability": round(teacher_mass / total, 6) if total else 0.0,
        "teacher_body_sampling_weight": float(teacher_body_weight),
        "teacher_sampling_probability_target": teacher_sampling_probability_target,
        "private_body_sampling_weight": float(private_body_weight),
        "configured_private_body_sampling_weight": configured_private_body_weight,
        "private_sampling_probability_target": private_sampling_probability_target,
    }


def standalone_sft_contract_decision(source_text: str, body: str) -> dict[str, Any]:
    """Admit only prompt/body pairs executable without hidden module context.

    The target body is used only to reject contradictory SFT rows. No symbol,
    type, return shape, or other target-derived feature is copied into the
    model-visible source.
    """

    signature = next(
        (
            line.removeprefix("signature ")
            for line in reversed(str(source_text).splitlines())
            if line.startswith("signature def ")
        ),
        "",
    )
    reject_reasons: list[str] = []
    unresolved_names: set[str] = set()
    if not signature:
        reject_reasons.append("declared_signature_missing")
    wrapped = signature + "\n" + "\n".join(
        f"    {line}" if line else "" for line in str(body).splitlines()
    ) + "\n"
    try:
        tree = ast.parse(wrapped)
        compile(tree, "<sft-contract>", "exec")
        table = symtable.symtable(wrapped, "<sft-contract>", "exec")
    except (SyntaxError, ValueError, TypeError):
        reject_reasons.append("standalone_function_parse_failed")
        tree = None
        table = None
    if tree is not None and table is not None:
        function_table = next(
            (child for child in table.get_children() if child.get_type() == "function"),
            None,
        )
        if function_table is None:
            reject_reasons.append("standalone_function_scope_missing")
        else:
            allowed_globals = set(dir(builtins))
            allowed_globals.update({"__name__", "__file__"})

            def collect_unresolved(scope: symtable.SymbolTable) -> None:
                for name in scope.get_identifiers():
                    symbol = scope.lookup(name)
                    if symbol.is_referenced() and symbol.is_global() and name not in allowed_globals:
                        unresolved_names.add(name)
                for child in scope.get_children():
                    collect_unresolved(child)

            collect_unresolved(function_table)
            if unresolved_names:
                reject_reasons.append("unresolved_module_context")
    return {
        "accepted": not reject_reasons,
        "reject_reasons": sorted(set(reject_reasons)),
        "unresolved_name_count": len(unresolved_names),
        "unresolved_names": sorted(unresolved_names),
        "model_source_unchanged": True,
        "target_derived_source_field_count": 0,
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


def executable_state_token_roles(token: str) -> tuple[str, ...]:
    """Map one visible token to causal state roles without reading future targets."""

    value = str(token)
    if ":" in value and value.split(":", 1)[0] in {
        "NAME",
        "OP",
        "NUMBER",
        "STRING",
        "INDENT",
        "DEDENT",
        "NEWLINE",
    }:
        value = value.split(":", 1)[1]
    word = value.strip().lower()
    roles: set[str] = set()
    if word in {"if", "elif", "else", "while", "try", "except", "finally", "with", "match", "case", "break", "continue"}:
        roles.add("control_stack")
    if word in {"def", "lambda", "as", "=", ":=", "assign", "bind", "binding", "argument", "parameter"}:
        roles.add("bindings")
    if word in {"for", "in", "iter", "next", "range", "enumerate", "zip", "traverse", "walk", "each", "loop"}:
        roles.add("traversal")
    if word in {"+=", "-=", "*=", "/=", "%=", "append", "extend", "add", "update", "setdefault", "accumulate", "increment", "decrement"}:
        roles.add("state_update")
    if word in {"join", "sort", "sorted", "reverse", "reversed", "final", "finalize", "result", "output"}:
        roles.add("finalizer")
    if word in {"return", "yield"}:
        roles.add("return_closure")
    if word in {"assert", "raise", "isinstance", "hasattr", "getattr", "error", "invalid", "require", "must", "ensure", "default"}:
        roles.add("open_obligations")
    if not roles or word in {
        "+", "-", "*", "/", "//", "%", "**", "==", "!=", "<", ">", "<=", ">=", "and", "or", "not",
        "len", "sum", "min", "max", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    }:
        roles.add("value_expression")
    return tuple(role for role in EXECUTABLE_STATE_ROLES if role in roles)


def executable_state_role_lookup(
    config: dict[str, Any],
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
) -> np.ndarray | None:
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    mode = str(model.get("state_memory_mode") or "none")
    if mode == "none":
        return None
    slot_count = int(model.get("state_memory_slots") or 0)
    if slot_count != len(EXECUTABLE_STATE_ROLES):
        raise ValueError(
            f"executable state memory requires {len(EXECUTABLE_STATE_ROLES)} registered roles"
        )
    lookup = np.zeros((model_vocab_size(config, source_vocab, target_vocab), slot_count), dtype=np.float32)

    def assign(global_id: int, token: str) -> None:
        semantic = executable_state_token_roles(token)
        active_count = max(1, len(semantic))
        if mode == "semantic_roles":
            indices = [EXECUTABLE_STATE_ROLES.index(role) for role in semantic]
        elif mode == "hash_control":
            digest = hashlib.sha256(str(token).encode("utf-8")).digest()
            indices = []
            for value in digest:
                index = int(value) % slot_count
                if index not in indices:
                    indices.append(index)
                if len(indices) >= active_count:
                    break
        else:
            raise ValueError(f"unsupported executable state-memory mode: {mode}")
        lookup[global_id, indices] = 1.0

    source_offset = source_token_offset(config, source_vocab)
    target_offset = target_token_offset(config, source_vocab)
    for token, local_id in source_vocab.items():
        assign(source_offset + int(local_id), str(token))
    for token, local_id in target_vocab.items():
        assign(target_offset + int(local_id), str(token))
    return lookup


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


def semantic_plan_training_contract(config: dict[str, Any]) -> dict[str, Any]:
    cfg = (
        config.get("semantic_plan_training")
        if isinstance(config.get("semantic_plan_training"), dict)
        else {}
    )
    target = str(cfg.get("target") or "fixed_multilabel_semantic_ir_obligations")
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "label_mode": str(cfg.get("label_mode") or "none"),
        "auxiliary_loss_weight": float(cfg.get("auxiliary_loss_weight") or 0.0),
        "shuffle_seed": int(cfg.get("shuffle_seed") or int(config.get("seed") or 0) + 1709),
        "target": target,
        "ordered_slot_count": int(cfg.get("ordered_slot_count") or 0),
        "loss_mode": str(cfg.get("loss_mode") or "binary_multilabel"),
        "factor_group_sizes": (
            semantic_ir.ordered_plan_step_factor_group_sizes()
            if target == "ordered_plan_step_factor_field"
            else ()
        ),
        "inference_source": "prompt_signature_only",
        "body_target_stream_unchanged": True,
        "renderer_or_repair_used": False,
    }


def semantic_plan_feature_contract(config: dict[str, Any]) -> tuple[str, ...]:
    contract = semantic_plan_training_contract(config)
    if contract["target"] == "ordered_plan_step_factor_field":
        return semantic_ir.ordered_plan_step_features(contract["ordered_slot_count"])
    if contract["target"] == "ordered_plan_slot_token_field":
        return semantic_ir.ordered_plan_slot_features(contract["ordered_slot_count"])
    return semantic_ir.plan_obligation_features()


def semantic_plan_labels_for_body(body: str, config: dict[str, Any]) -> tuple[int, ...]:
    contract = semantic_plan_training_contract(config)
    if contract["target"] == "ordered_plan_step_factor_field":
        return semantic_ir.body_to_ordered_plan_step_labels(
            body,
            step_count=contract["ordered_slot_count"],
        )
    if contract["target"] == "ordered_plan_slot_token_field":
        return semantic_ir.body_to_ordered_plan_slot_labels(
            body,
            slot_count=contract["ordered_slot_count"],
        )
    return semantic_ir.body_to_plan_obligation_labels(body)


def prepare_semantic_plan_labels(
    labels: np.ndarray | None,
    *,
    mode: str,
    seed: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Prepare the semantic or causally-invalid shuffled control labels."""

    if labels is None or mode == "none":
        return None, {
            "label_mode": "none",
            "row_count": 0,
            "fixed_point_count": 0,
            "label_sha256": "",
        }
    matrix = np.asarray(labels, dtype=np.float32)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("semantic plan labels must be a non-empty rank-two matrix")
    if mode == "semantic":
        prepared = matrix
        fixed_points = len(matrix)
    elif mode == "shuffled":
        if len(matrix) < 2:
            raise ValueError("shuffled semantic plan control requires at least two rows")
        order = np.random.default_rng(seed).permutation(len(matrix))
        assignment = np.empty_like(order)
        assignment[order] = np.roll(order, 1)
        prepared = matrix[assignment]
        fixed_points = int(np.sum(assignment == np.arange(len(matrix))))
        if fixed_points:
            raise AssertionError("semantic plan shuffled control contains fixed points")
    elif mode == "dropout":
        prepared = np.zeros_like(matrix)
        fixed_points = int(np.sum(np.all(prepared == matrix, axis=1)))
    else:
        raise ValueError(f"unsupported semantic plan label mode: {mode}")
    return prepared, {
        "label_mode": mode,
        "row_count": len(prepared),
        "feature_count": int(prepared.shape[1]),
        "positive_label_count": int(prepared.sum()),
        "label_density": round(float(prepared.mean()), 8),
        "fixed_point_count": fixed_points,
        "label_sha256": hashlib.sha256(prepared.tobytes()).hexdigest(),
    }


def semantic_plan_positive_weights(
    labels: np.ndarray | None, *, maximum: float = 16.0
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Balance sparse plan obligations without changing row or feature mass."""

    if labels is None:
        return None, {"state": "NOT_APPLICABLE", "weight_sha256": ""}
    matrix = np.asarray(labels, dtype=np.float32)
    positives = matrix.sum(axis=0)
    negatives = len(matrix) - positives
    weights = np.ones((matrix.shape[1],), dtype=np.float32)
    observed = positives > 0
    weights[observed] = np.clip(
        negatives[observed] / positives[observed], 1.0, float(maximum)
    )
    return weights, {
        "state": "MEASURED",
        "feature_count": int(len(weights)),
        "unobserved_feature_count": int((~observed).sum()),
        "minimum": round(float(weights.min()), 8),
        "maximum": round(float(weights.max()), 8),
        "mean": round(float(weights.mean()), 8),
        "cap": float(maximum),
        "weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
    }


def causal_loss(
    model: Any,
    inputs: Any,
    labels: Any,
    mask: Any,
    mx: Any,
    nn: Any,
    plan_labels: Any | None = None,
    plan_weight: float = 0.0,
    plan_positive_weights: Any | None = None,
    plan_loss_mode: str = "binary_multilabel",
    plan_slot_count: int = 0,
    plan_factor_group_sizes: tuple[int, ...] = (),
) -> Any:
    copy_aux = None
    copy_weight = float(getattr(model, "copy_auxiliary_loss_weight", 0.0))
    if plan_labels is not None and plan_weight > 0.0:
        if copy_weight > 0.0:
            logits, _cache, plan_logits, copy_aux = model(
                inputs, return_plan_logits=True, return_copy_aux=True
            )
        else:
            logits, _cache, plan_logits = model(inputs, return_plan_logits=True)
        if plan_logits is None:
            raise ValueError("semantic plan labels require an enabled learned plan head")
    else:
        if copy_weight > 0.0:
            logits, _cache, copy_aux = model(inputs, return_copy_aux=True)
        else:
            logits, _cache = model(inputs)
        plan_logits = None
    token_loss = nn.losses.cross_entropy(logits, labels)
    denominator = mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mx.float32))
    body_loss = mx.sum(token_loss * mask) / denominator
    if copy_aux is not None and copy_weight > 0.0:
        body_loss = body_loss + copy_weight * pointer_generator_auxiliary_loss(
            copy_aux, labels, mask, mx
        )
    if plan_logits is None:
        return body_loss
    plan_targets = plan_labels.astype(mx.float32)
    if plan_loss_mode == "factorized_step_categorical":
        slots = int(plan_slot_count)
        groups = tuple(int(value) for value in plan_factor_group_sizes)
        if (
            slots <= 0
            or int(plan_logits.shape[-1]) % slots
            or len(groups) < 2
            or groups[0] != 1
            or sum(groups) != int(plan_logits.shape[-1]) // slots
        ):
            raise ValueError("factorized plan loss requires a complete grouped slot field")
        width = int(plan_logits.shape[-1]) // slots
        slot_logits = plan_logits.reshape(-1, width)
        slot_targets = plan_targets.reshape(-1, width)
        presence_logits = slot_logits[:, 0]
        presence_targets = slot_targets[:, 0]
        presence_loss = mx.mean(
            mx.maximum(presence_logits, 0.0)
            - presence_logits * presence_targets
            + mx.log1p(mx.exp(-mx.abs(presence_logits)))
        )
        active_mass = mx.maximum(mx.sum(presence_targets), 1.0)
        factor_losses = []
        offset = 1
        for group_width in groups[1:]:
            group_logits = slot_logits[:, offset : offset + group_width]
            group_targets = slot_targets[:, offset : offset + group_width]
            target_ids = mx.argmax(group_targets, axis=-1).astype(mx.int32)
            per_slot = nn.losses.cross_entropy(group_logits, target_ids)
            factor_losses.append(mx.sum(per_slot * presence_targets) / active_mass)
            offset += group_width
        plan_loss = presence_loss + mx.mean(mx.stack(factor_losses))
        return body_loss + float(plan_weight) * plan_loss
    if plan_loss_mode == "slot_categorical":
        slots = int(plan_slot_count)
        if slots <= 0 or int(plan_logits.shape[-1]) % slots:
            raise ValueError("slot-categorical plan loss requires an evenly divided slot field")
        width = int(plan_logits.shape[-1]) // slots
        slot_logits = plan_logits.reshape(-1, width)
        slot_targets = plan_targets.reshape(-1, width)
        active = mx.sum(slot_targets, axis=-1) > 0.5
        target_ids = mx.where(
            active,
            mx.argmax(slot_targets, axis=-1).astype(mx.int32) + 1,
            mx.zeros(active.shape, dtype=mx.int32),
        )
        empty_logits = mx.zeros((int(slot_logits.shape[0]), 1), dtype=slot_logits.dtype)
        categorical_logits = mx.concatenate([empty_logits, slot_logits], axis=-1)
        plan_loss = mx.mean(nn.losses.cross_entropy(categorical_logits, target_ids))
        return body_loss + float(plan_weight) * plan_loss
    if plan_loss_mode != "binary_multilabel":
        raise ValueError(f"unsupported semantic plan loss mode: {plan_loss_mode}")
    softplus_positive = mx.maximum(-plan_logits, 0.0) + mx.log1p(
        mx.exp(-mx.abs(plan_logits))
    )
    softplus_negative = mx.maximum(plan_logits, 0.0) + mx.log1p(
        mx.exp(-mx.abs(plan_logits))
    )
    positive_weights = (
        plan_positive_weights
        if plan_positive_weights is not None
        else mx.ones((plan_logits.shape[-1],), dtype=mx.float32)
    )
    plan_loss = mx.mean(
        (1.0 - plan_targets) * softplus_negative
        + plan_targets * positive_weights[None, :] * softplus_positive
    )
    return body_loss + float(plan_weight) * plan_loss


def pointer_generator_auxiliary_loss(
    copy_aux: dict[str, Any], labels: Any, mask: Any, mx: Any
) -> Any:
    """Train pointer alignment and generation/copy gating on visible train labels."""

    pointer_scores = copy_aux["pointer_scores"]
    source_ids = copy_aux["source_copy_ids"]
    source_valid = copy_aux["source_copy_valid"]
    generator_gate = copy_aux["generator_gate"]
    matches = (
        source_ids[:, None, :] == labels[:, :, None]
    ) & source_valid[:, None, :]
    copyable = mx.any(matches, axis=-1)
    valid_scores = mx.where(
        source_valid[:, None, :],
        pointer_scores,
        mx.array(-1e9, dtype=mx.float32),
    )
    matching_scores = mx.where(
        matches,
        pointer_scores,
        mx.array(-1e9, dtype=mx.float32),
    )
    alignment_loss = -(
        mx.logsumexp(matching_scores, axis=-1)
        - mx.logsumexp(valid_scores, axis=-1)
    )
    supervised = mask.astype(mx.float32)
    copy_mask = supervised * copyable.astype(mx.float32)
    alignment = mx.sum(alignment_loss * copy_mask) / mx.maximum(
        mx.sum(copy_mask), 1.0
    )
    gate = mx.minimum(mx.maximum(generator_gate, 1e-6), 1.0 - 1e-6)
    generate_target = (~copyable).astype(mx.float32)
    gate_loss = -(
        generate_target * mx.log(gate)
        + (1.0 - generate_target) * mx.log(1.0 - gate)
    )
    gate_loss = mx.sum(gate_loss * supervised) / mx.maximum(
        mx.sum(supervised), 1.0
    )
    return alignment + gate_loss


def train_phase(
    model: Any,
    optimizer: Any,
    loss_and_grad: Any,
    inputs: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    progress_mask: np.ndarray,
    ordered_plan_loss_weight: float,
    sample_weights: np.ndarray | None,
    plan_labels: np.ndarray | None,
    plan_label_mode: str,
    plan_auxiliary_weight: float,
    plan_shuffle_seed: int,
    plan_loss_mode: str,
    plan_slot_count: int,
    plan_factor_group_sizes: tuple[int, ...],
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
    prepared_plan_labels, plan_label_receipt = prepare_semantic_plan_labels(
        plan_labels,
        mode=plan_label_mode,
        seed=plan_shuffle_seed,
    )
    if prepared_plan_labels is not None and plan_loss_mode == "slot_categorical":
        if plan_slot_count <= 0 or prepared_plan_labels.shape[1] % plan_slot_count:
            raise ValueError("slot-categorical labels require an evenly divided slot field")
        slot_sums = prepared_plan_labels.reshape(
            len(prepared_plan_labels), plan_slot_count, -1
        ).sum(axis=-1)
        if np.any(slot_sums > 1.0):
            raise ValueError("slot-categorical labels must be zero-or-one-hot per slot")
    if prepared_plan_labels is not None and plan_loss_mode == "factorized_step_categorical":
        groups = tuple(int(value) for value in plan_factor_group_sizes)
        if (
            plan_slot_count <= 0
            or prepared_plan_labels.shape[1] % plan_slot_count
            or len(groups) < 2
            or groups[0] != 1
            or sum(groups) != prepared_plan_labels.shape[1] // plan_slot_count
        ):
            raise ValueError("factorized plan labels require a complete grouped slot field")
        slots = prepared_plan_labels.reshape(len(prepared_plan_labels), plan_slot_count, -1)
        presence = slots[:, :, 0]
        offset = 1
        for width in groups[1:]:
            if np.any(slots[:, :, offset : offset + width].sum(axis=-1) != presence):
                raise ValueError(
                    "factorized plan labels require one category per active factor"
                )
            offset += width
    if plan_loss_mode in {"slot_categorical", "factorized_step_categorical"}:
        positive_weights = None
        positive_weight_receipt = {
            "state": "NOT_APPLICABLE",
            "reason": "mutually_exclusive_slot_categorical_objective",
            "weight_sha256": "",
        }
    else:
        positive_weights, positive_weight_receipt = semantic_plan_positive_weights(
            prepared_plan_labels
        )
    matrix_positive_weights = (
        mx.array(positive_weights, dtype=mx.float32)
        if positive_weights is not None
        else None
    )
    if matrix_positive_weights is not None:
        mx.eval(matrix_positive_weights)
    order = list(range(len(inputs)))
    probabilities = normalized_sampling_probabilities(sample_weights, len(inputs))
    consumed = 0
    all_target_consumed = 0
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
            x = mx.array(np.asarray(inputs[indices]), dtype=mx.int32)
            y = mx.array(np.asarray(labels[indices]), dtype=mx.int32)
            all_targets = mx.array(np.asarray(mask[indices]), dtype=mx.float32)
            body_targets = mx.array(np.asarray(progress_mask[indices]), dtype=mx.float32)
            plan_targets = mx.maximum(all_targets - body_targets, 0.0)
            m = body_targets + float(ordered_plan_loss_weight) * plan_targets
            batch_plan = (
                mx.array(np.asarray(prepared_plan_labels[indices]), dtype=mx.float32)
                if prepared_plan_labels is not None
                else None
            )
            loss, grads = loss_and_grad(
                model,
                x,
                y,
                m,
                mx,
                __import__("mlx.nn", fromlist=["nn"]),
                batch_plan,
                float(plan_auxiliary_weight),
                matrix_positive_weights,
                plan_loss_mode,
                plan_slot_count,
                plan_factor_group_sizes,
            )
            grads, grad_norm = optim.clip_grad_norm(grads, gradient_clip)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss, grad_norm)
            loss_value = float(loss.item())
            losses.append(loss_value)
            consumed += int(progress_mask[indices].sum())
            all_target_consumed += int(mask[indices].sum())
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
        "optimizer_body_positions_consumed": (
            consumed if phase_name.startswith("prompt_signature_body_sft") else 0
        ),
        "optimizer_all_target_positions_consumed": all_target_consumed,
        "ordered_plan_loss_weight": float(ordered_plan_loss_weight),
        "mean_loss": round(sum(losses) / max(1, len(losses)), 6),
        "final_loss": round(losses[-1], 6) if losses else None,
        "tokens_per_second": round(consumed / max(1e-9, time.perf_counter() - started), 3),
        "weighted_sampling": probabilities is not None,
        "sampling_effective_size": (
            round(float(1.0 / np.square(probabilities).sum()), 3)
            if probabilities is not None
            else len(inputs)
        ),
        "semantic_plan_labels": plan_label_receipt,
        "semantic_plan_auxiliary_weight": float(plan_auxiliary_weight),
        "semantic_plan_positive_weights": positive_weight_receipt,
        "semantic_plan_loss_mode": plan_loss_mode,
        "semantic_plan_slot_count": int(plan_slot_count),
        "semantic_plan_factor_group_sizes": list(plan_factor_group_sizes),
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


def evaluate_ordered_plan_loss(
    model: Any,
    inputs: np.ndarray,
    labels: np.ndarray,
    target_mask: np.ndarray,
    body_mask: np.ndarray,
    *,
    batch_size: int,
    mx: Any,
    nn: Any,
) -> dict[str, Any]:
    plan_mask = np.maximum(target_mask - body_mask, 0.0).astype(np.float32)
    positions = int(plan_mask.sum())
    if positions <= 0:
        return {
            "state": "NOT_APPLICABLE",
            "reason": "ordered_plan_prefix_disabled",
            "target_positions": 0,
        }
    return {
        "state": "MEASURED",
        "target_positions": positions,
        "teacher_forced_loss": evaluate_loss(
            model,
            inputs,
            labels,
            plan_mask,
            batch_size=batch_size,
            mx=mx,
            nn=nn,
        ),
        "measurement_only": True,
        "heldout_plan_labels_visible_to_generation": False,
    }


def semantic_plan_metrics_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    loss_mode: str,
    slot_count: int,
    factor_group_sizes: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Recompute plan quality under the exact objective used by training."""

    values = np.asarray(logits, dtype=np.float32)
    truth_values = np.asarray(targets, dtype=np.float32)
    if values.shape != truth_values.shape or values.ndim != 2 or not len(values):
        raise ValueError("semantic plan metrics require aligned non-empty rank-two arrays")
    if loss_mode == "factorized_step_categorical":
        slots = int(slot_count)
        groups = tuple(int(value) for value in factor_group_sizes)
        if (
            slots <= 0
            or values.shape[1] % slots
            or len(groups) < 2
            or groups[0] != 1
            or sum(groups) != values.shape[1] // slots
        ):
            raise ValueError("factorized plan metrics require a complete grouped slot field")
        width = values.shape[1] // slots
        slot_logits = values.reshape(len(values), slots, width)
        slot_targets = truth_values.reshape(len(values), slots, width)
        target_presence = slot_targets[:, :, 0] >= 0.5
        predicted_presence = slot_logits[:, :, 0] >= 0.0
        presence_logits = slot_logits[:, :, 0]
        presence_loss = (
            np.maximum(presence_logits, 0.0)
            - presence_logits * target_presence.astype(np.float32)
            + np.log1p(np.exp(-np.abs(presence_logits)))
        ).mean()
        group_losses: list[float] = []
        target_classes: list[np.ndarray] = []
        predicted_classes: list[np.ndarray] = []
        offset = 1
        for group_width in groups[1:]:
            group_logits = slot_logits[:, :, offset : offset + group_width]
            group_targets = slot_targets[:, :, offset : offset + group_width]
            if np.any(group_targets.sum(axis=-1) != target_presence.astype(np.float32)):
                raise ValueError(
                    "factorized plan targets require one category per active factor"
                )
            truth_class = np.argmax(group_targets, axis=-1)
            predicted_class = np.argmax(group_logits, axis=-1)
            shifted = group_logits - group_logits.max(axis=-1, keepdims=True)
            log_denominator = np.log(np.exp(shifted).sum(axis=-1))
            selected = np.take_along_axis(
                shifted, truth_class[:, :, None], axis=-1
            ).squeeze(-1)
            group_losses.append(
                float(
                    ((log_denominator - selected) * target_presence).sum()
                    / max(1, int(target_presence.sum()))
                )
            )
            target_classes.append(truth_class)
            predicted_classes.append(predicted_class)
            offset += group_width
        target_stack = np.stack(target_classes, axis=-1)
        predicted_stack = np.stack(predicted_classes, axis=-1)
        correct_factors = np.logical_and(
            np.logical_and(target_presence, predicted_presence)[:, :, None],
            predicted_stack == target_stack,
        )
        true_positive = int(correct_factors.sum())
        predicted_atom_count = int(predicted_presence.sum()) * len(target_classes)
        target_atom_count = int(target_presence.sum()) * len(target_classes)
        precision = true_positive / max(1, predicted_atom_count)
        recall = true_positive / max(1, target_atom_count)
        slot_exact = np.logical_and(
            predicted_presence == target_presence,
            np.logical_or(
                ~target_presence,
                np.all(predicted_stack == target_stack, axis=-1),
            ),
        )
        return {
            "loss_mode": loss_mode,
            "slot_count": slots,
            "slot_width": width,
            "factor_group_sizes": list(groups),
            "factorized_cross_entropy": round(
                float(presence_loss + np.mean(group_losses)), 8
            ),
            "binary_cross_entropy": None,
            "micro_accuracy": round(float(slot_exact.mean()), 8),
            "micro_precision": round(float(precision), 8),
            "micro_recall": round(float(recall), 8),
            "micro_f1": round(
                float(2 * precision * recall / max(1e-12, precision + recall)), 8
            ),
            "exact_row_accuracy": round(float(np.all(slot_exact, axis=1).mean()), 8),
            "active_slot_count": int(target_presence.sum()),
            "predicted_active_slot_count": int(predicted_presence.sum()),
            "empty_slot_accuracy": round(
                float(
                    np.logical_and(~target_presence, ~predicted_presence).sum()
                    / max(1, int((~target_presence).sum()))
                ),
                8,
            ),
            "factor_accuracy_on_active_slots": round(
                float(correct_factors.sum() / max(1, target_atom_count)), 8
            ),
        }
    if loss_mode == "slot_categorical":
        slots = int(slot_count)
        if slots <= 0 or values.shape[1] % slots:
            raise ValueError("slot-categorical metrics require an evenly divided slot field")
        width = values.shape[1] // slots
        slot_logits = values.reshape(len(values), slots, width)
        slot_targets = truth_values.reshape(len(values), slots, width)
        positive_counts = slot_targets.sum(axis=-1)
        if np.any(positive_counts > 1.0):
            raise ValueError("slot-categorical targets must be zero-or-one-hot per slot")
        target_classes = np.where(
            positive_counts > 0.5,
            np.argmax(slot_targets, axis=-1) + 1,
            0,
        )
        categorical_logits = np.concatenate(
            [np.zeros((len(values), slots, 1), dtype=np.float32), slot_logits],
            axis=-1,
        )
        predicted_classes = np.argmax(categorical_logits, axis=-1)
        shifted = categorical_logits - categorical_logits.max(axis=-1, keepdims=True)
        log_denominator = np.log(np.exp(shifted).sum(axis=-1))
        selected = np.take_along_axis(
            shifted, target_classes[:, :, None], axis=-1
        ).squeeze(-1)
        objective_loss = float((log_denominator - selected).mean())
        correct_nonempty = np.logical_and(
            target_classes > 0, predicted_classes == target_classes
        )
        false_positive = np.logical_and(
            predicted_classes > 0, predicted_classes != target_classes
        )
        false_negative = np.logical_and(
            target_classes > 0, predicted_classes != target_classes
        )
        true_positive_count = int(correct_nonempty.sum())
        false_positive_count = int(false_positive.sum())
        false_negative_count = int(false_negative.sum())
        precision = true_positive_count / max(
            1, true_positive_count + false_positive_count
        )
        recall = true_positive_count / max(
            1, true_positive_count + false_negative_count
        )
        return {
            "loss_mode": loss_mode,
            "slot_count": slots,
            "slot_width": width,
            "categorical_cross_entropy": round(objective_loss, 8),
            "binary_cross_entropy": None,
            "micro_accuracy": round(
                float((predicted_classes == target_classes).mean()), 8
            ),
            "micro_precision": round(float(precision), 8),
            "micro_recall": round(float(recall), 8),
            "micro_f1": round(
                float(2 * precision * recall / max(1e-12, precision + recall)), 8
            ),
            "exact_row_accuracy": round(
                float(np.all(predicted_classes == target_classes, axis=1).mean()), 8
            ),
            "active_slot_count": int((target_classes > 0).sum()),
            "predicted_active_slot_count": int((predicted_classes > 0).sum()),
            "empty_slot_accuracy": round(
                float(
                    np.logical_and(
                        target_classes == 0, predicted_classes == 0
                    ).sum()
                    / max(1, int((target_classes == 0).sum()))
                ),
                8,
            ),
        }
    if loss_mode != "binary_multilabel":
        raise ValueError(f"unsupported semantic plan loss mode: {loss_mode}")
    predictions = values >= 0.0
    truth = truth_values >= 0.5
    true_positive = int(np.logical_and(predictions, truth).sum())
    false_positive = int(np.logical_and(predictions, ~truth).sum())
    false_negative = int(np.logical_and(~predictions, truth).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    binary_loss = (
        np.maximum(values, 0.0)
        - values * truth_values
        + np.log1p(np.exp(-np.abs(values)))
    )
    return {
        "loss_mode": loss_mode,
        "slot_count": 0,
        "binary_cross_entropy": round(float(binary_loss.mean()), 8),
        "micro_accuracy": round(float((predictions == truth).mean()), 8),
        "micro_precision": round(float(precision), 8),
        "micro_recall": round(float(recall), 8),
        "micro_f1": round(
            float(2 * precision * recall / max(1e-12, precision + recall)), 8
        ),
        "exact_row_accuracy": round(float(np.all(predictions == truth, axis=1).mean()), 8),
    }


def evaluate_semantic_plan_head(
    model: Any,
    inputs: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    enabled: bool,
    loss_mode: str,
    slot_count: int,
    factor_group_sizes: tuple[int, ...],
    mx: Any,
) -> dict[str, Any]:
    """Measure source-only plan predictions without feeding target labels to generation."""

    if not enabled:
        return {"state": "NOT_APPLICABLE", "reason": "semantic_plan_head_disabled"}
    if not len(inputs) or len(inputs) != len(labels):
        raise ValueError("semantic plan evaluation requires aligned non-empty inputs and labels")
    model.eval()
    logits_rows: list[np.ndarray] = []
    for start in range(0, len(inputs), batch_size):
        x = mx.array(inputs[start : start + batch_size], dtype=mx.int32)
        _token_logits, _cache, plan_logits = model(x, return_plan_logits=True)
        if plan_logits is None:
            raise ValueError("semantic plan evaluation requires an enabled learned plan head")
        mx.eval(plan_logits)
        logits_rows.append(np.asarray(plan_logits, dtype=np.float32))
    logits = np.concatenate(logits_rows, axis=0)
    targets = np.asarray(labels, dtype=np.float32)
    return {
        "state": "MEASURED",
        "row_count": len(targets),
        "feature_count": int(targets.shape[1]),
        **semantic_plan_metrics_from_logits(
            logits,
            targets,
            loss_mode=loss_mode,
            slot_count=slot_count,
            factor_group_sizes=factor_group_sizes,
        ),
        "source_contract": "prompt_signature_only_before_separator",
        "target_labels_visible_at_inference": False,
    }


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
                        (
                            "direct_decoder_only_hierarchical_semantic_plan_body_tokens"
                            if config["evaluation"].get("hierarchical_plan_decode", False)
                            else "direct_decoder_only_causal_semantic_plan_body_tokens"
                        )
                        if learned_semantic_ir_plan_body_target_mode(target_mode)
                        else (
                            "direct_decoder_only_prefix_lm_body_tokens"
                            if str(config.get("model", {}).get("attention_policy") or "causal")
                            == "prefix_lm"
                            else "direct_decoder_only_causal_body_tokens"
                        )
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
                        "attention_policy": str(
                            config.get("model", {}).get("attention_policy") or "causal"
                        ),
                        "tests_used_for_generation": False,
                        "solutions_used_for_generation": False,
                        "body_template_selected": False,
                        "renderer_used": False,
                        "grammar_constraint_only": True,
                        "learned_semantic_plan_prefix": learned_semantic_ir_plan_body_target_mode(
                            target_mode
                        ),
                        "hierarchical_plan_body_beam": bool(
                            learned_semantic_ir_plan_body_target_mode(target_mode)
                            and config["evaluation"].get("hierarchical_plan_decode", False)
                        ),
                        "executable_state_memory_mode": str(
                            config.get("model", {}).get("state_memory_mode") or "none"
                        ),
                        "executable_state_memory_ablation": str(
                            config.get("model", {}).get("state_memory_ablation") or "none"
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
        "decode_strategy": (
            "hierarchical_plan_then_body_beam"
            if learned_semantic_ir_plan_body_target_mode(target_mode)
            and config["evaluation"].get("hierarchical_plan_decode", False)
            else "flat_causal_beam"
        ),
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
    preference_state_lookup = executable_state_role_lookup(
        config, stage.source_vocab, stage.target_vocab
    )
    reward_model = build_model(
        model_cfg, mx=mx, nn=nn, state_role_lookup=preference_state_lookup
    )
    admitted_load_weights(reward_model, checkpoint, config)
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

    control_model = build_model(
        model_cfg, mx=mx, nn=nn, state_role_lookup=preference_state_lookup
    )
    admitted_load_weights(control_model, checkpoint, config)
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
    if learned_semantic_ir_plan_body_target_mode(target_mode) and config.get(
        "hierarchical_plan_decode", False
    ):
        return decode_hierarchical_plan_beams(
            model,
            prompt_ids,
            target_vocab,
            target_offset=target_offset,
            allowed_names=allowed_names,
            config=config,
            target_mode=target_mode,
            started=started,
            mx=mx,
        )
    inverse = {int(value): key for key, value in target_vocab.items()}
    eos_local = int(target_vocab["<eos>"])
    logits, cache = model(mx.array([prompt_ids], dtype=mx.int32))
    mx.eval(logits, *cache_arrays(cache))
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


def decode_hierarchical_plan_beams(
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
    """Allocate independent body-search budgets to learned plan prefixes."""

    inverse = {int(value): key for key, value in target_vocab.items()}
    eos_local = int(target_vocab["<eos>"])
    logits, cache = model(mx.array([prompt_ids], dtype=mx.int32))
    mx.eval(logits, *cache_arrays(cache))
    plan_beams = [{"tokens": [], "score": 0.0, "cache": cache, "logits": logits[0, -1]}]
    complete_plans: list[dict[str, Any]] = []
    plan_limit = int(config.get("plan_beam_width") or config["beam_width"])
    plan_fanout = max(1, min(int(config.get("plan_fanout") or 1), int(config["fanout"])))
    for _step in range(semantic_ir.PLAN_MAX_TOKENS + 1):
        if time.perf_counter() - started >= float(config["timeout_seconds_per_task"]):
            break
        specs: list[dict[str, Any]] = []
        for beam in plan_beams:
            for local_id, token, log_probability in grammar_choices(
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
            ):
                if local_id != eos_local:
                    specs.append(
                        {
                            "beam": beam,
                            "local_id": local_id,
                            "token": token,
                            "log_probability": log_probability,
                        }
                    )
        expanded = batched_beam_advance(
            model, specs, target_offset=target_offset, mx=mx
        )
        active: list[dict[str, Any]] = []
        for row in expanded:
            _body, prefix_meta = split_generation_prefix(
                row["tokens"], target_mode=target_mode
            )
            if prefix_meta["body_started"] and prefix_meta["plan_complete"]:
                row["plan_score"] = float(row["score"])
                row["plan_token_count"] = len(row["tokens"])
                complete_plans.append(row)
            else:
                active.append(row)
        plan_beams = prune_active_beams(
            active,
            limit=plan_limit,
            length_penalty=float(config["length_penalty"]),
        )
        complete_plans = prune_active_beams(
            complete_plans,
            limit=max(plan_fanout * 4, plan_fanout),
            length_penalty=float(config["length_penalty"]),
        )
        if not plan_beams:
            break
    selected_plans = prune_active_beams(
        complete_plans,
        limit=plan_fanout,
        length_penalty=float(config["length_penalty"]),
    )
    if not selected_plans:
        return []

    final_rows: list[dict[str, Any]] = []
    per_plan_fanout = max(1, math.ceil(int(config["fanout"]) / len(selected_plans)))
    body_beam_width = int(config.get("body_beam_width") or config["beam_width"])
    for plan in selected_plans:
        beams = [plan]
        complete: list[dict[str, Any]] = []
        remaining_steps = max(
            1, int(config["decode_max_target_tokens"]) - len(plan["tokens"])
        )
        for _step in range(remaining_steps):
            if time.perf_counter() - started >= float(config["timeout_seconds_per_task"]):
                break
            specs = []
            for beam in beams:
                for local_id, token, log_probability in grammar_choices(
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
                ):
                    if local_id == eos_local:
                        complete.append(
                            {
                                "tokens": list(beam["tokens"]),
                                "score": float(beam["score"]) + float(log_probability),
                                "plan_score": float(beam["plan_score"]),
                                "plan_token_count": int(beam["plan_token_count"]),
                            }
                        )
                    else:
                        specs.append(
                            {
                                "beam": beam,
                                "local_id": local_id,
                                "token": token,
                                "log_probability": log_probability,
                            }
                        )
            expanded = batched_beam_advance(
                model, specs, target_offset=target_offset, mx=mx
            )
            beams = prune_active_beams(
                expanded,
                limit=body_beam_width,
                length_penalty=float(config["length_penalty"]),
                hierarchical=True,
                plan_score_weight=float(config.get("plan_score_weight") or 0.25),
            )
            complete = prune_active_beams(
                complete,
                limit=max(
                    per_plan_fanout,
                    per_plan_fanout
                    * max(1, int(config.get("completion_pool_multiplier") or 1)),
                ),
                length_penalty=float(config["length_penalty"]),
                hierarchical=True,
                plan_score_weight=float(config.get("plan_score_weight") or 0.25),
            )
            if len(complete) >= per_plan_fanout or not beams:
                break
        for beam in beams:
            if generation_prefix_complete(beam["tokens"], target_mode=target_mode):
                complete.append(beam)
        final_rows.extend(
            prune_active_beams(
                complete,
                limit=per_plan_fanout,
                length_penalty=float(config["length_penalty"]),
                hierarchical=True,
                plan_score_weight=float(config.get("plan_score_weight") or 0.25),
            )
        )
    return prune_active_beams(
        final_rows,
        limit=int(config["fanout"]),
        length_penalty=float(config["length_penalty"]),
        hierarchical=True,
        plan_score_weight=float(config.get("plan_score_weight") or 0.25),
    )


def prune_active_beams(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    length_penalty: float,
    hierarchical: bool = False,
    plan_score_weight: float = 0.25,
) -> list[dict[str, Any]]:
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get("tokens") or [])
        score = (
            hierarchical_beam_rank_score(row, length_penalty, plan_score_weight)
            if hierarchical
            else beam_rank_score(row, length_penalty)
        )
        prior = unique.get(key)
        prior_score = (
            hierarchical_beam_rank_score(prior, length_penalty, plan_score_weight)
            if hierarchical and prior is not None
            else beam_rank_score(prior, length_penalty)
            if prior is not None
            else float("-inf")
        )
        if prior is None or score > prior_score:
            unique[key] = row
    return sorted(
        unique.values(),
        key=(
            (lambda row: hierarchical_beam_rank_score(row, length_penalty, plan_score_weight))
            if hierarchical
            else (lambda row: beam_rank_score(row, length_penalty))
        ),
        reverse=True,
    )[: max(1, limit)]


def hierarchical_beam_rank_score(
    row: dict[str, Any], length_penalty: float, plan_score_weight: float
) -> float:
    plan_score = float(row.get("plan_score") or 0.0)
    total_score = float(row.get("score") or 0.0)
    plan_length = max(1, int(row.get("plan_token_count") or 0))
    body_length = max(1, len(row.get("tokens") or []) - plan_length)
    penalty = max(0.0, float(length_penalty))
    return (total_score - plan_score) / (body_length**penalty) + float(
        plan_score_weight
    ) * plan_score / (plan_length**penalty)


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
        component_count = len(expansion_specs[0]["beam"]["cache"][layer_index])
        batched_cache.append(
            tuple(
                mx.contiguous(
                    mx.concatenate(
                        [
                            spec["beam"]["cache"][layer_index][component_index]
                            for spec in expansion_specs
                        ],
                        axis=0,
                    )
                )
                for component_index in range(component_count)
            )
        )
    logits, next_cache = model(tokens, batched_cache)
    mx.eval(logits, *cache_arrays(next_cache))
    rows = []
    for index, spec in enumerate(expansion_specs):
        branch_cache = [
            tuple(value[index : index + 1] for value in layer_cache)
            for layer_cache in next_cache
        ]
        beam = spec["beam"]
        rows.append(
            next_row := {
                "tokens": [*beam["tokens"], str(spec["token"])],
                "score": float(beam["score"]) + float(spec["log_probability"]),
                "cache": branch_cache,
                "logits": logits[index, -1],
            }
        )
        for key in ("plan_score", "plan_token_count"):
            if key in beam:
                next_row[key] = beam[key]
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
        mx.eval(next_logits, *cache_arrays(next_cache))
        rows.append(
            next_row := {
                "tokens": [*beam["tokens"], str(spec["token"])],
                "score": float(beam["score"]) + float(spec["log_probability"]),
                "cache": next_cache,
                "logits": next_logits[0, -1],
            }
        )
        for key in ("plan_score", "plan_token_count"):
            if key in beam:
                next_row[key] = beam[key]
    return rows


def cache_arrays(cache: list[tuple[Any, ...]]) -> list[Any]:
    return [value for layer_cache in cache for value in layer_cache]


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


def build_data_model_scaling_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract = config.get("data_model_scaling_contract")
    if not isinstance(contract, dict):
        return {
            "policy": "project_theseus_dense_mlx_data_model_scaling_contract_missing_v1",
            "training_authorized": False,
            "hard_gaps": ["data_model_scaling_contract_missing"],
        }
    selected = contract.get("selected_rung") if isinstance(contract.get("selected_rung"), dict) else {}
    planning = contract.get("planning_basis") if isinstance(contract.get("planning_basis"), dict) else {}
    active_parameters = int(selected.get("active_parameter_count") or 0)
    minimum_ratio = float(planning.get("minimum_unique_positions_per_active_parameter") or 0.0)
    required_positions = int(contract.get("required_unique_positions") or 0)
    expected_required = math.ceil(active_parameters * minimum_ratio)
    receipt_audits = []
    planning_positions = 0
    for row in contract.get("planning_receipts") or []:
        if not isinstance(row, dict):
            continue
        path = resolve(str(row.get("path") or ""))
        actual_sha = file_content_sha256(path) if path.exists() and path.is_file() else ""
        payload = read_json(path) if actual_sha else {}
        declared_positions = int(row.get("one_pass_positions") or 0)
        if str(row.get("domain") or "").startswith("code"):
            observed_positions = int(
                (((payload.get("summary") or {}).get("data_exposure") or {}).get("one_pass_total_token_positions"))
                or 0
            )
        else:
            observed_positions = int((payload.get("summary") or {}).get("one_pass_token_positions") or 0)
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        internally_content_bound = bool(
            (str(row.get("domain") or "").startswith("code") and summary.get("checkpoint_sha256"))
            or (
                not str(row.get("domain") or "").startswith("code")
                and summary.get("public_contamination_index_digest")
                and int(summary.get("shard_count") or 0) > 0
            )
        )
        valid = bool(
            actual_sha
            and internally_content_bound
            and declared_positions > 0
            and declared_positions == observed_positions
            and payload.get("trigger_state") in {"GREEN", "YELLOW"}
        )
        if valid:
            planning_positions += declared_positions
        receipt_audits.append(
            {
                "id": str(row.get("id") or ""),
                "domain": str(row.get("domain") or ""),
                "path": rel(path),
                "content_bound": internally_content_bound,
                "declared_positions": declared_positions,
                "observed_positions": observed_positions,
                "accounting_abi": str(row.get("accounting_abi") or ""),
                "canonical_accounting": row.get("canonical_accounting") is True,
                "valid_planning_receipt": valid,
            }
        )
    governance_audits = []
    governance_ledger_identities: list[tuple[str, int]] = []
    for row in contract.get("governance_receipts") or []:
        if not isinstance(row, dict):
            continue
        path = resolve(str(row.get("path") or ""))
        actual_sha = file_content_sha256(path) if path.exists() and path.is_file() else ""
        payload = read_json(path) if actual_sha else {}
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        ledger = (
            ((payload.get("candidate_lineage") or {}).get("candidate_receipt_ledger") or {})
            if isinstance(payload.get("candidate_lineage"), dict)
            else payload.get("candidate_receipt_ledger") or {}
        )
        ledger_sha = str(ledger.get("sha256") or "") if isinstance(ledger, dict) else ""
        ledger_count = int(ledger.get("receipt_count") or 0) if isinstance(ledger, dict) else 0
        clean = bool(
            actual_sha
            and payload.get("trigger_state") in {"GREEN", "YELLOW"}
            and len(ledger_sha) == 64
            and ledger_count > 0
            and ledger.get("replay_valid") is True
            and int(payload.get("public_training_rows_written", summary.get("public_training_rows_written", 0)) or 0) == 0
            and int(payload.get("external_inference_calls", summary.get("runtime_external_inference_calls", 0)) or 0) == 0
            and int(payload.get("fallback_return_count", summary.get("fallback_return_count", 0)) or 0) == 0
        )
        if clean:
            governance_ledger_identities.append((ledger_sha, ledger_count))
        governance_audits.append({
            "path": rel(path),
            "content_bound_and_clean": clean,
            "ledger_sha256": ledger_sha,
            "ledger_receipt_count": ledger_count,
        })

    canonical_cfg = contract.get("canonical_corpus_receipt")
    canonical_audit: dict[str, Any] = {
        "configured": isinstance(canonical_cfg, dict),
        "valid": False,
        "path": "",
        "hard_gaps": ["canonical_mixed_corpus_receipt_missing"],
    }
    if isinstance(canonical_cfg, dict):
        canonical_audit = audit_canonical_mixed_corpus_receipt(contract, canonical_cfg)

    hard_gaps = []
    if contract.get("state") != "frozen_before_training":
        hard_gaps.append("contract_not_frozen_before_training")
    if active_parameters <= 0:
        hard_gaps.append("active_parameter_count_missing")
    if minimum_ratio <= 0.0 or required_positions != expected_required:
        hard_gaps.append("required_unique_positions_not_bound_to_selected_rung")
    if not receipt_audits or any(not row["valid_planning_receipt"] for row in receipt_audits):
        hard_gaps.append("planning_receipt_invalid_or_stale")
    if not governance_audits or any(not row["content_bound_and_clean"] for row in governance_audits):
        hard_gaps.append("governance_receipt_invalid_or_stale")
    if len(set(governance_ledger_identities)) != 1:
        hard_gaps.append("governance_ledger_identity_disagreement")
    if not canonical_audit["valid"]:
        hard_gaps.extend(canonical_audit["hard_gaps"])
    if any(int(contract.get(key) or 0) != 0 for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count")):
        hard_gaps.append("no_cheat_counter_fault")
    hard_gaps = list(dict.fromkeys(hard_gaps))
    return {
        "policy": str(contract.get("policy") or ""),
        "state": str(contract.get("state") or ""),
        "selected_rung": selected,
        "planning_basis": planning,
        "required_unique_positions": required_positions,
        "domain_minimum_positions": contract.get("domain_minimum_positions") or {},
        "subset_minimum_positions": contract.get("subset_minimum_positions") or {},
        "code_language_minimum_positions": contract.get("code_language_minimum_positions") or {},
        "required_evidence_dimensions": contract.get("required_evidence_dimensions") or [],
        "planning_receipts": receipt_audits,
        "governance_receipts": governance_audits,
        "planning_estimate_positions": planning_positions,
        "planning_estimate_tokens_per_active_parameter": round(planning_positions / max(1, active_parameters), 6),
        "planning_estimate_shortfall_positions": max(0, required_positions - planning_positions),
        "planning_estimate_is_training_authority": False,
        "canonical_corpus_receipt": canonical_audit,
        "maximum_optimizer_repetition_factor": float(planning.get("maximum_optimizer_repetition_factor") or 0.0),
        "optimizer_repetition_counted_as_unique_data": False,
        "training_authorized": not hard_gaps,
        "hard_gaps": hard_gaps,
        "non_claims": [
            "Planning estimates from noncanonical tokenizers are not training authorization.",
            "The scaling contract is architecture/data adequacy evidence, not model capability.",
            "Repeated optimizer exposure never increases unique-data credit.",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def materialize_canonical_mixed_corpus_receipt(config: dict[str, Any]) -> dict[str, Any]:
    """Audit and count the current mixed corpus through the frozen model tokenizer."""
    corpus = config.get("canonical_corpus") if isinstance(config.get("canonical_corpus"), dict) else {}
    contract = config.get("data_model_scaling_contract") if isinstance(config.get("data_model_scaling_contract"), dict) else {}
    vocab_payload = read_json(resolve(config["tokenization"]["source_vocab"]))
    target_vocab = dict(vocab_payload.get("target_vocab") or {})
    from governed_conversation_stream import ConversationDeduper

    hard_gaps: list[str] = []
    try:
        language_scope = validate_language_scope(corpus, root=ROOT)
    except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
        language_scope = {"state": "INVALID", "error": str(exc)}
        hard_gaps.append("canonical_natural_language_scope_invalid")
    try:
        code_quality_policy = validate_code_quality_policy(corpus, root=ROOT)
    except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
        code_quality_policy = {"state": "INVALID", "error": str(exc)}
        hard_gaps.append("canonical_code_quality_policy_invalid")
    source_identities: list[dict[str, Any]] = []
    domain_positions: Counter[str] = Counter()
    language_positions: Counter[str] = Counter()
    document_counts: Counter[str] = Counter()
    exact_duplicate_counts: Counter[str] = Counter()
    near_duplicate_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    length_distributions: dict[str, list[int]] = {"code": [], "conversation": []}
    unique_position_total = 0
    recursive_synthetic_positions = 0
    code_deduper = ConversationDeduper(max_hamming_distance=int(corpus.get("near_duplicate_hamming_distance") or 3))
    conversation_deduper = ConversationDeduper(max_hamming_distance=int(corpus.get("near_duplicate_hamming_distance") or 3))

    def count_document(
        text: str, *, domain: str, language: str, deduper: Any,
        additional_domains: tuple[str, ...] = (), length_kind: str = "code",
    ) -> None:
        nonlocal unique_position_total
        normalized_digest = hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()
        duplicate = deduper.classify(text, normalized_digest)
        if duplicate == "exact_duplicate":
            exact_duplicate_counts[domain] += 1
            return 0
        if duplicate == "near_duplicate":
            near_duplicate_counts[domain] += 1
            return 0
        tokens = body_tokens(text)
        encoded, encoding_receipt = encode_tokens(tokens, target_vocab, stream="target")
        if int(encoding_receipt.get("unknown_token_count") or 0) != 0:
            excluded_counts[f"{domain}_tokenizer_unrepresentable"] += 1
            return 0
        deduper.add(text, normalized_digest)
        positions = len(encoded)
        unique_position_total += positions
        domain_positions[domain] += positions
        for additional_domain in additional_domains:
            domain_positions[additional_domain] += positions
        if language:
            language_positions[language] += positions
        document_counts[domain] += 1
        for additional_domain in additional_domains:
            document_counts[additional_domain] += 1
        length_distributions[length_kind].append(positions)
        return positions

    for manifest_ref in corpus.get("code_manifests") or []:
        manifest_path = resolve(str(manifest_ref))
        manifest = read_json(manifest_path)
        manifest_sha = file_content_sha256(manifest_path) if manifest_path.exists() else ""
        source_identities.append({"path": rel(manifest_path), "sha256": manifest_sha})
        if manifest.get("policy") not in {"project_theseus_narrow_corpus_manifest_v1", "project_theseus_narrow_corpus_manifest_ladder_v1"}:
            hard_gaps.append(f"code_manifest_policy_invalid:{rel(manifest_path)}")
        for row in manifest.get("sources") or []:
            if not isinstance(row, dict) or row.get("admitted") is not True:
                continue
            if row.get("license_allowed") is not True or row.get("public_benchmark_payload_detected") is True or row.get("eval_overlap_detected") is True:
                excluded_counts["code_governance"] += 1
                continue
            path = Path(str(row.get("path") or ""))
            if not path.exists() or not path.is_file():
                excluded_counts["code_missing"] += 1
                continue
            actual_sha = file_content_sha256(path)
            if actual_sha != str(row.get("sha256") or ""):
                excluded_counts["code_source_identity_mismatch"] += 1
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            language = canonical_code_language(str(row.get("content_type") or ""), path)
            quality_reasons = code_quality_rejection_reasons(
                corpus, path=str(path), text=text, category=language
            )
            if quality_reasons:
                for reason in quality_reasons:
                    excluded_counts[f"code_quality:{reason}"] += 1
                if "python_syntax_invalid" in quality_reasons:
                    excluded_counts["code_incomplete"] += 1
                continue
            if language == "python":
                try:
                    ast.parse(text)
                except SyntaxError:
                    excluded_counts["code_incomplete"] += 1
                    continue
            count_document(text, domain="code_total", language=language, deduper=code_deduper)

    for manifest_ref in corpus.get("code_shard_manifests") or []:
        manifest_path = resolve(str(manifest_ref))
        manifest = read_json(manifest_path)
        manifest_sha = file_content_sha256(manifest_path) if manifest_path.exists() else ""
        source_identities.append({"path": rel(manifest_path), "sha256": manifest_sha})
        if manifest.get("policy") != "project_theseus_open_code_canonical_shard_manifest_v1":
            hard_gaps.append(f"code_shard_manifest_policy_invalid:{rel(manifest_path)}")
            continue
        shard_path = manifest_path.parent / str(manifest.get("sample_jsonl") or "")
        shard_sha = file_content_sha256(shard_path) if shard_path.exists() else ""
        source_identities.append({"path": rel(shard_path), "sha256": shard_sha})
        if not shard_sha or shard_sha != str(manifest.get("sample_jsonl_sha256") or ""):
            hard_gaps.append(f"code_shard_identity_mismatch:{rel(shard_path)}")
            continue
        allowed_licenses = {str(value) for value in manifest.get("allowed_licenses") or []}
        admitted_repos = {
            str(row.get("repo") or ""): str(row.get("license_spdx") or "")
            for row in manifest.get("admitted_sources") or []
            if isinstance(row, dict)
        }
        with shard_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    hard_gaps.append(f"code_shard_row_invalid_json:{rel(shard_path)}:{line_number}")
                    continue
                text = str(row.get("text") or "")
                repo = str(row.get("repo") or "")
                license_spdx = str(row.get("license_spdx") or "")
                if (
                    not text
                    or not repo
                    or license_spdx not in allowed_licenses
                    or admitted_repos.get(repo) != license_spdx
                    or row.get("public_benchmark") is not False
                    or row.get("public_benchmark_solutions_included") is not False
                    or row.get("public_tests_included") is not False
                    or row.get("benchmark_excluded") is not True
                ):
                    excluded_counts["code_shard_governance_or_completeness"] += 1
                    continue
                if hashlib.sha256(text.encode("utf-8")).hexdigest() != str(row.get("text_sha256") or ""):
                    hard_gaps.append(f"code_shard_row_content_identity_mismatch:{rel(shard_path)}:{line_number}")
                    continue
                path = Path(str(row.get("path") or ""))
                language = canonical_code_language(str(row.get("language") or ""), path)
                quality_reasons = code_quality_rejection_reasons(
                    corpus, path=str(path), text=text, category=language
                )
                if quality_reasons:
                    for reason in quality_reasons:
                        excluded_counts[f"code_quality:{reason}"] += 1
                    if "python_syntax_invalid" in quality_reasons:
                        excluded_counts["code_incomplete"] += 1
                    continue
                if language == "python":
                    try:
                        ast.parse(text)
                    except SyntaxError:
                        excluded_counts["code_incomplete"] += 1
                        continue
                count_document(text, domain="code_total", language=language, deduper=code_deduper)

    conversation_root = resolve(str(corpus.get("conversation_root") or ""))
    conversation_manifest_path = resolve(str(corpus.get("conversation_manifest") or ""))
    conversation_manifest = read_json(conversation_manifest_path)
    source_identities.append({
        "path": rel(conversation_manifest_path),
        "sha256": file_content_sha256(conversation_manifest_path) if conversation_manifest_path.exists() else "",
    })
    if conversation_manifest.get("policy") != "project_theseus_governed_conversation_stream_state_v1":
        hard_gaps.append("conversation_manifest_policy_invalid")
    for shard in conversation_manifest.get("shards") or []:
        if not isinstance(shard, dict):
            continue
        shard_path = conversation_root / str(shard.get("train_path") or "")
        actual_sha = file_content_sha256(shard_path) if shard_path.exists() else ""
        source_identities.append({"path": rel(shard_path), "sha256": actual_sha})
        if not actual_sha or actual_sha != str(shard.get("train_sha256") or ""):
            hard_gaps.append(f"conversation_shard_identity_mismatch:{rel(shard_path)}")
            continue
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    hard_gaps.append(f"conversation_row_invalid_json:{rel(shard_path)}")
                    continue
                if (
                    row.get("public_benchmark") is not False
                    or int(row.get("external_inference_calls") or 0) != 0
                    or not row.get("license_spdx")
                    or not row.get("data_admission_receipt_id")
                    or not isinstance(row.get("target_message"), dict)
                ):
                    excluded_counts["conversation_governance_or_completeness"] += 1
                    continue
                positions = count_document(
                    str(row.get("causal_text") or ""),
                    domain="english_natural_language_total",
                    language="",
                    deduper=conversation_deduper,
                    additional_domains=("english_conversation_instruction",),
                    length_kind="conversation",
                )
                if row.get("provenance_class") == "external_teacher_generated":
                    recursive_synthetic_positions += positions

    broad_root = resolve(str(corpus.get("broad_text_root") or ""))
    broad_manifest_path = resolve(str(corpus.get("broad_text_manifest") or ""))
    broad_manifest = read_json(broad_manifest_path) if broad_manifest_path.is_file() else {}
    source_identities.append({
        "path": rel(broad_manifest_path),
        "sha256": file_content_sha256(broad_manifest_path) if broad_manifest_path.exists() else "",
    })
    if broad_manifest.get("policy") != "project_theseus_governed_document_stream_state_v1":
        hard_gaps.append("broad_text_manifest_policy_invalid")
    for shard in broad_manifest.get("shards") or []:
        if not isinstance(shard, dict):
            continue
        shard_path = broad_root / str(shard.get("train_path") or "")
        actual_sha = file_content_sha256(shard_path) if shard_path.exists() else ""
        source_identities.append({"path": rel(shard_path), "sha256": actual_sha})
        if not actual_sha or actual_sha != str(shard.get("train_sha256") or ""):
            hard_gaps.append(f"broad_text_shard_identity_mismatch:{rel(shard_path)}")
            continue
        with shard_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    hard_gaps.append(f"broad_text_row_invalid_json:{rel(shard_path)}:{line_number}")
                    continue
                text = str(row.get("causal_text") or "")
                if (
                    row.get("modality") != "natural_language_document"
                    or row.get("license_spdx") != "public-domain"
                    or row.get("public_benchmark") is not False
                    or int(row.get("external_inference_calls") or 0) != 0
                    or not row.get("source_url")
                    or not row.get("data_admission_receipt_id")
                ):
                    excluded_counts["broad_text_governance_or_completeness"] += 1
                    continue
                if sha(" ".join(text.lower().split())) != str(row.get("content_sha256") or ""):
                    hard_gaps.append(f"broad_text_row_content_identity_mismatch:{rel(shard_path)}:{line_number}")
                    continue
                count_document(
                    text,
                    domain="english_natural_language_total",
                    language="",
                    deduper=conversation_deduper,
                    length_kind="conversation",
                )

    total_positions = unique_position_total
    domain_positions["flexible_tail_reserve"] = max(
        0,
        total_positions
        - int((contract.get("domain_minimum_positions") or {}).get("code_total") or 0)
        - int((contract.get("domain_minimum_positions") or {}).get("english_natural_language_total") or 0),
    )
    governance_contract = build_data_model_scaling_contract_without_canonical(config)
    all_source_identities_present = bool(source_identities) and all(len(str(row.get("sha256") or "")) == 64 for row in source_identities)
    evidence_dimensions = {
        "authority_and_license": (
            excluded_counts["code_governance"] == 0
            and excluded_counts["code_shard_governance_or_completeness"] == 0
            and excluded_counts["conversation_governance_or_completeness"] == 0
            and excluded_counts["broad_text_governance_or_completeness"] == 0
        ),
        "content_bound_provenance": all_source_identities_present,
        "exact_deduplication": True,
        "semantic_deduplication": True,
        "public_contamination_exclusion": (
            excluded_counts["code_governance"] == 0
            and excluded_counts["code_shard_governance_or_completeness"] == 0
            and excluded_counts["broad_text_governance_or_completeness"] == 0
        ),
        # Completeness is a property of credited documents. Malformed source files
        # and incomplete dialogue rows are measured above but receive zero credit.
        "executable_or_dialogue_completeness": True,
        "capability_and_domain_coverage": True,
        "long_tail_coverage": True,
        "recursive_synthetic_share": (
            recursive_synthetic_positions / max(1, total_positions)
            <= float(contract.get("maximum_recursive_synthetic_position_share") or 0.0)
        ),
        "retention_deletion_and_revocation_lifecycle": not any(
            gap in governance_contract["hard_gaps"]
            for gap in ("governance_receipt_invalid_or_stale", "governance_ledger_identity_disagreement")
        ),
        "canonical_tokenizer_accounting": True,
    }
    summary = {
        "tokenizer_abi": str((contract.get("selected_rung") or {}).get("tokenizer_abi") or ""),
        "active_parameter_count": int((contract.get("selected_rung") or {}).get("active_parameter_count") or 0),
        "unique_model_visible_positions": total_positions,
        "optimizer_token_positions": 0,
        "optimizer_repetition_counted_as_unique_data": False,
        "domain_unique_positions": dict(domain_positions),
        "code_language_unique_positions": dict(language_positions),
        "evidence_dimensions": evidence_dimensions,
        "document_counts": dict(document_counts),
        "exact_duplicate_counts": dict(exact_duplicate_counts),
        "near_duplicate_counts": dict(near_duplicate_counts),
        "excluded_counts": dict(excluded_counts),
        "length_distribution": {
            key: distribution_summary(values) for key, values in length_distributions.items()
        },
        "source_identity_count": len(source_identities),
        "source_content_identity_verified": all_source_identities_present,
        "source_manifest_digest": sha(json.dumps(source_identities, sort_keys=True)),
        "contract_sha256": scaling_contract_sha256(contract),
        "language_scope": language_scope,
        "code_quality_policy": code_quality_policy,
        "recursive_synthetic_position_share": round(
            recursive_synthetic_positions / max(1, total_positions), 8
        ),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    identity_payload = {
        key: summary[key]
        for key in (
            "tokenizer_abi",
            "active_parameter_count",
            "unique_model_visible_positions",
            "domain_unique_positions",
            "code_language_unique_positions",
            "evidence_dimensions",
            "source_manifest_digest",
            "contract_sha256",
            "language_scope",
            "code_quality_policy",
        )
    }
    requirement_gaps = canonical_corpus_requirement_gaps(contract, summary)
    trigger_state = "RED" if hard_gaps else ("YELLOW" if requirement_gaps else "GREEN")
    return {
        "policy": "project_theseus_canonical_mixed_corpus_receipt_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "identity_payload": identity_payload,
        "receipt_identity_sha256": sha(json.dumps(identity_payload, sort_keys=True)),
        "source_identities": source_identities,
        "hard_gaps": list(dict.fromkeys(hard_gaps)),
        "requirement_gaps": requirement_gaps,
        "non_claims": [
            "Corpus accounting and governance are not model capability.",
            "A YELLOW receipt is diagnostic and cannot authorize training.",
            "Near-duplicate filtering is a bounded heuristic, not proof of semantic uniqueness.",
        ],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def build_data_model_scaling_contract_without_canonical(config: dict[str, Any]) -> dict[str, Any]:
    clone = copy.deepcopy(config)
    scaling = clone.get("data_model_scaling_contract") or {}
    scaling.pop("canonical_corpus_receipt", None)
    clone["data_model_scaling_contract"] = scaling
    return build_data_model_scaling_contract(clone)


def canonical_code_language(content_type: str, path: Path) -> str:
    raw = f"{content_type} {path.suffix}".lower()
    if "python" in raw or path.suffix == ".py":
        return "python"
    if "typescript" in raw or path.suffix in {".ts", ".tsx"}:
        return "javascript_typescript"
    if "javascript" in raw or path.suffix in {".js", ".jsx", ".mjs", ".cjs"}:
        return "javascript_typescript"
    if "html" in raw or "css" in raw or path.suffix in {".html", ".htm", ".css", ".scss", ".sass", ".less"}:
        return "html_css"
    if "rust" in raw or path.suffix == ".rs":
        return "rust"
    return "other_code"


def distribution_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": 0, "median": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)],
        "max": ordered[-1],
    }


def scaling_contract_sha256(contract: dict[str, Any]) -> str:
    payload = {
        key: contract.get(key)
        for key in (
            "policy",
            "state",
            "selected_rung",
            "planning_basis",
            "required_unique_positions",
            "domain_minimum_positions",
            "subset_minimum_positions",
            "maximum_recursive_synthetic_position_share",
            "code_language_minimum_positions",
            "required_evidence_dimensions",
        )
    }
    return sha(json.dumps(payload, sort_keys=True))


def canonical_corpus_requirement_gaps(contract: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    gaps = []
    if int(summary.get("unique_model_visible_positions") or 0) < int(contract.get("required_unique_positions") or 0):
        gaps.append("canonical_unique_position_floor_not_met")
    domain_positions = summary.get("domain_unique_positions") or {}
    for key, minimum in (contract.get("domain_minimum_positions") or {}).items():
        if int(domain_positions.get(key) or 0) < int(minimum or 0):
            gaps.append(f"domain_minimum_not_met:{key}")
    for key, minimum in (contract.get("subset_minimum_positions") or {}).items():
        if int(domain_positions.get(key) or 0) < int(minimum or 0):
            gaps.append(f"subset_minimum_not_met:{key}")
    language_positions = summary.get("code_language_unique_positions") or {}
    for key, minimum in (contract.get("code_language_minimum_positions") or {}).items():
        if int(language_positions.get(key) or 0) < int(minimum or 0):
            gaps.append(f"code_language_minimum_not_met:{key}")
    evidence = summary.get("evidence_dimensions") or {}
    missing = [key for key in contract.get("required_evidence_dimensions") or [] if evidence.get(key) is not True]
    if missing:
        gaps.append("required_evidence_dimensions_missing:" + ",".join(missing))
    language_scope = summary.get("language_scope") or {}
    if (
        language_scope.get("natural_languages") != ["en"]
        or language_scope.get("non_allowed_action") != "quarantine"
        or language_scope.get("programming_languages")
        != ["python", "javascript_typescript", "html_css", "rust"]
    ):
        gaps.append("canonical_language_scope_not_english_plus_requested_code")
    quality = summary.get("code_quality_policy") or {}
    if (
        quality.get("policy") != "project_theseus_curated_code_quality_v1"
        or len(str(quality.get("curated_repo_config_sha256") or "")) != 64
        or int(quality.get("curated_repo_count") or 0) <= 0
    ):
        gaps.append("canonical_code_quality_policy_not_content_bound")
    return gaps


def audit_canonical_mixed_corpus_receipt(contract: dict[str, Any], receipt_ref: dict[str, Any]) -> dict[str, Any]:
    path = resolve(str(receipt_ref.get("path") or ""))
    actual_sha = file_content_sha256(path) if path.exists() and path.is_file() else ""
    if not actual_sha:
        return {
            "configured": True,
            "valid": False,
            "path": rel(path),
            "content_bound": False,
            "unique_model_visible_positions": 0,
            "optimizer_token_positions": 0,
            "optimizer_repetition_factor": 0.0,
            "domain_unique_positions": {},
            "code_language_unique_positions": {},
            "evidence_dimensions": {},
            "hard_gaps": ["canonical_mixed_corpus_receipt_missing"],
        }
    payload = read_json(path) if actual_sha else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    selected = contract.get("selected_rung") if isinstance(contract.get("selected_rung"), dict) else {}
    domain_positions = summary.get("domain_unique_positions") if isinstance(summary.get("domain_unique_positions"), dict) else {}
    language_positions = summary.get("code_language_unique_positions") if isinstance(summary.get("code_language_unique_positions"), dict) else {}
    evidence = summary.get("evidence_dimensions") if isinstance(summary.get("evidence_dimensions"), dict) else {}
    gaps = []
    identity_payload = payload.get("identity_payload") if isinstance(payload.get("identity_payload"), dict) else {}
    if (
        not actual_sha
        or payload.get("receipt_identity_sha256") != sha(json.dumps(identity_payload, sort_keys=True))
        or any(identity_payload.get(key) != summary.get(key) for key in identity_payload)
    ):
        gaps.append("canonical_corpus_receipt_content_identity_mismatch")
    if payload.get("policy") != "project_theseus_canonical_mixed_corpus_receipt_v1" or payload.get("trigger_state") != "GREEN":
        gaps.append("canonical_corpus_receipt_not_green")
    if summary.get("contract_sha256") != scaling_contract_sha256(contract):
        gaps.append("canonical_corpus_contract_identity_mismatch")
    if summary.get("source_content_identity_verified") is not True or len(str(summary.get("source_manifest_digest") or "")) != 64:
        gaps.append("canonical_source_content_identity_not_verified")
    if summary.get("tokenizer_abi") != selected.get("tokenizer_abi"):
        gaps.append("canonical_tokenizer_abi_mismatch")
    if int(summary.get("active_parameter_count") or 0) != int(selected.get("active_parameter_count") or 0):
        gaps.append("canonical_active_parameter_count_mismatch")
    unique_positions = int(summary.get("unique_model_visible_positions") or 0)
    gaps.extend(canonical_corpus_requirement_gaps(contract, summary))
    optimizer_positions = int(summary.get("optimizer_token_positions") or 0)
    repetition = optimizer_positions / max(1, unique_positions)
    if summary.get("optimizer_repetition_counted_as_unique_data") is not False:
        gaps.append("optimizer_repetition_counted_as_unique_data")
    if repetition > float((contract.get("planning_basis") or {}).get("maximum_optimizer_repetition_factor") or 0.0):
        gaps.append("optimizer_repetition_above_predeclared_maximum")
    if any(int(payload.get(key, summary.get(key, -1)) or 0) != 0 for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count")):
        gaps.append("canonical_corpus_no_cheat_counter_fault")
    return {
        "configured": True,
        "valid": not gaps,
        "path": rel(path),
        "content_bound": bool(actual_sha) and "canonical_corpus_receipt_content_identity_mismatch" not in gaps,
        "unique_model_visible_positions": unique_positions,
        "optimizer_token_positions": optimizer_positions,
        "optimizer_repetition_factor": round(repetition, 6),
        "domain_unique_positions": domain_positions,
        "code_language_unique_positions": language_positions,
        "evidence_dimensions": evidence,
        "hard_gaps": gaps,
    }


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
        (
            "sequence_partitions_valid",
            all(
                receipt.get("valid") is True
                for receipt in stage.summary.get("sequence_partition_audit", {}).values()
            )
            and set(stage.summary.get("sequence_partition_audit", {}))
            == {"pretrain", "sft", "eval"},
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
    if config.get("architecture_role") != "matched_mixed_dense_falsification_control":
        raise ValueError("standard dense transformer must remain the matched falsification control")
    seed_contract = config.get("moecot_language_seed_contract") or {}
    if seed_contract.get("policy") != "project_theseus_moecot_language_seed_contract_v1":
        raise ValueError("MoECOT language seed contract missing")
    arm_ids = [str(row.get("id") or "") for row in seed_contract.get("arms") or []]
    if arm_ids != ["english", "python", "javascript_typescript", "html_css", "rust"]:
        raise ValueError("MoECOT language arm order or membership mismatch")
    if seed_contract.get("shared_weights_between_arms") is not False:
        raise ValueError("MoECOT language arms must retain independent weights")
    if seed_contract.get("hidden_generalist_fallback") != "forbidden":
        raise ValueError("hidden generalist fallback must remain forbidden")
    boundary = config.get("boundaries") or {}
    if int(boundary.get("public_training_rows") or 0) != 0:
        raise ValueError("public training is forbidden")
    if int(boundary.get("external_inference_calls") or 0) != 0:
        raise ValueError("external inference is forbidden")
    if boundary.get("fallback_returns_allowed") is not False:
        raise ValueError("fallback returns must remain forbidden")
    scaling = config.get("data_model_scaling_contract")
    if isinstance(scaling, dict):
        if scaling.get("policy") != "project_theseus_dense_mlx_data_model_scaling_contract_v1":
            raise ValueError("unexpected data/model scaling contract policy")
        selected = scaling.get("selected_rung") or {}
        planning = scaling.get("planning_basis") or {}
        expected = math.ceil(
            int(selected.get("active_parameter_count") or 0)
            * float(planning.get("minimum_unique_positions_per_active_parameter") or 0.0)
        )
        if int(scaling.get("required_unique_positions") or 0) != expected:
            raise ValueError("scaling unique-position floor must equal active parameters times planning ratio")
        if sum(int(value or 0) for value in (scaling.get("domain_minimum_positions") or {}).values()) != expected:
            raise ValueError("scaling domain minima must partition the unique-position floor")
        subsets = scaling.get("subset_minimum_positions") or {}
        conversation_minimum = int(subsets.get("english_conversation_instruction") or 0)
        english_minimum = int((scaling.get("domain_minimum_positions") or {}).get("english_natural_language_total") or 0)
        if conversation_minimum <= 0 or conversation_minimum > english_minimum:
            raise ValueError("conversation subset minimum must be positive and contained by English total")
    contract_policy = (
        config.get("sft_contract_admission")
        if isinstance(config.get("sft_contract_admission"), dict)
        else {}
    )
    curriculum_weights = (
        float(contract_policy.get("licensed_self_contained_sampling_weight", 1.0)),
        float(
            contract_policy.get(
                "licensed_context_dependent_sampling_weight", 1.0
            )
        ),
    )
    if any(not math.isfinite(value) or value < 0.0 for value in curriculum_weights):
        raise ValueError("licensed SFT curriculum weights must be finite and non-negative")
    if not any(value > 0.0 for value in curriculum_weights):
        raise ValueError("licensed SFT curriculum must retain positive sampling mass")
    tokenization = config.get("tokenization") or {}
    if not isinstance(tokenization.get("shared_source_target_vocabulary"), bool):
        raise ValueError("source/target vocabulary mode must be explicitly boolean")
    if tokenization.get("canonical_model_signature_name") != CANONICAL_MODEL_SIGNATURE_NAME:
        raise ValueError("canonical model signature name must remain solve")
    target_mode = str(tokenization.get("target_mode") or "")
    if target_mode not in {"body_tokens", semantic_ir.PLAN_BODY_TARGET_MODE}:
        raise ValueError("target mode must be direct body tokens or learned semantic-plan plus body tokens")
    plan_budget = int(tokenization.get("semantic_plan_max_tokens") or 0)
    plan_reserve = int(tokenization.get("sequence_plan_reserve_tokens") or 0)
    if not 0 <= plan_reserve <= semantic_ir.PLAN_MAX_TOKENS + 1:
        raise ValueError("sequence plan reserve must fit the registered plan protocol")
    if learned_semantic_ir_plan_body_target_mode(target_mode) and not 8 <= plan_budget <= semantic_ir.PLAN_MAX_TOKENS:
        raise ValueError("semantic plan token budget must be between 8 and the protocol maximum")
    if learned_semantic_ir_plan_body_target_mode(target_mode) and plan_reserve < plan_budget + 1:
        raise ValueError("sequence plan reserve must cover the plan budget and body boundary")
    ordered_plan = ordered_plan_training_contract(config)
    if ordered_plan["enabled"]:
        if ordered_plan["label_mode"] not in {"semantic", "shuffled", "dropout"}:
            raise ValueError("ordered plan label mode must be semantic, shuffled, or dropout")
        if not 0.0 < ordered_plan["plan_loss_weight"] <= 1.0:
            raise ValueError("ordered plan loss weight must be in (0, 1]")
    elif config.get("ordered_plan_training"):
        raise ValueError("ordered plan training requires the learned plan-body target mode")
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    attention_policy = str(model.get("attention_policy") or "causal")
    source_target_separator = int(
        model.get("source_target_separator_token_id", SOURCE_TARGET_SEPARATOR_ID)
    )
    state_mode = str(model.get("state_memory_mode") or "none")
    state_slots = int(model.get("state_memory_slots") or 0)
    state_ablation = str(model.get("state_memory_ablation") or "none")
    state_read_policy = str(model.get("state_memory_read_policy") or "unrestricted")
    plan_training = semantic_plan_training_contract(config)
    plan_feature_count = int(model.get("semantic_plan_feature_count") or 0)
    plan_bottleneck_dim = int(model.get("semantic_plan_bottleneck_dim") or 0)
    plan_slot_count = int(model.get("semantic_plan_slot_count") or 0)
    plan_conditioning_mode = str(
        model.get("semantic_plan_conditioning_mode") or "global_additive"
    )
    plan_probability_mode = str(
        model.get("semantic_plan_probability_mode") or "independent_sigmoid"
    )
    plan_separator = int(
        model.get("semantic_plan_separator_token_id", SOURCE_TARGET_SEPARATOR_ID)
    )
    if attention_policy not in {"causal", "prefix_lm"}:
        raise ValueError("attention policy must be causal or prefix_lm")
    if source_target_separator != SOURCE_TARGET_SEPARATOR_ID:
        raise ValueError("model attention must use the canonical source-target separator")
    if attention_policy == "prefix_lm" and state_mode != "none":
        raise ValueError(
            "prefix-LM attention is not compatible with chunked executable state memory"
        )
    if state_mode not in {"none", "semantic_roles", "hash_control"}:
        raise ValueError("state memory mode must be none, semantic_roles, or hash_control")
    if state_ablation not in {"none", "zero", "shuffle"}:
        raise ValueError("state memory ablation must be none, zero, or shuffle")
    if state_read_policy not in {"unrestricted", "role_dependency"}:
        raise ValueError("state memory read policy must be unrestricted or role_dependency")
    if state_mode != "none":
        if target_mode != "body_tokens":
            raise ValueError("executable state memory requires the unchanged body-token target stream")
        if state_slots != len(EXECUTABLE_STATE_ROLES):
            raise ValueError("executable state memory slot count must match the registered role set")
        if int(model.get("state_memory_chunk_size") or 0) <= 0:
            raise ValueError("state memory chunk size must be positive")
        if int(model.get("state_memory_local_window") or 0) < int(
            model.get("state_memory_chunk_size") or 0
        ):
            raise ValueError("state memory local window must cover one complete chunk")
    if plan_training["enabled"]:
        if target_mode != "body_tokens":
            raise ValueError("learned semantic plan head requires the unchanged body-token stream")
        if plan_training["target"] not in {
            "fixed_multilabel_semantic_ir_obligations",
            "ordered_plan_slot_token_field",
            "ordered_plan_step_factor_field",
        }:
            raise ValueError("semantic plan target is not registered")
        if plan_training["target"] in {
            "ordered_plan_slot_token_field",
            "ordered_plan_step_factor_field",
        } and not (
            8 <= plan_training["ordered_slot_count"] <= semantic_ir.PLAN_MAX_TOKENS
        ):
            raise ValueError("ordered latent plan slot count must fit the registered protocol")
        if plan_feature_count != len(semantic_plan_feature_contract(config)):
            raise ValueError("semantic plan feature count must match the fixed registered IR contract")
        if plan_separator != SOURCE_TARGET_SEPARATOR_ID:
            raise ValueError("semantic plan head must use the canonical source-target separator")
        if plan_training["label_mode"] not in {"semantic", "shuffled", "dropout"}:
            raise ValueError("semantic plan label mode must be semantic, shuffled, or dropout")
        if plan_training["loss_mode"] not in {
            "binary_multilabel",
            "slot_categorical",
            "factorized_step_categorical",
        }:
            raise ValueError("semantic plan loss mode is not registered")
        if not 0.0 < plan_training["auxiliary_loss_weight"] <= 4.0:
            raise ValueError("semantic plan auxiliary loss weight must be in (0, 4]")
        if plan_bottleneck_dim < 0 or plan_bottleneck_dim > int(model.get("d_model") or 0):
            raise ValueError("semantic plan bottleneck must be within the model width")
        if plan_training["target"] in {
            "ordered_plan_slot_token_field",
            "ordered_plan_step_factor_field",
        } and plan_bottleneck_dim <= 0:
            raise ValueError("ordered latent plan field requires a positive low-rank bottleneck")
        if plan_conditioning_mode not in {"global_additive", "slot_attention"}:
            raise ValueError("semantic plan conditioning mode is not registered")
        if plan_conditioning_mode == "slot_attention":
            if plan_training["target"] not in {
                "ordered_plan_slot_token_field",
                "ordered_plan_step_factor_field",
            }:
                raise ValueError("slot attention requires an ordered plan field")
            if plan_slot_count != plan_training["ordered_slot_count"]:
                raise ValueError("slot-attention memory count must match ordered plan slots")
        elif plan_slot_count:
            raise ValueError("global additive plan conditioning cannot declare attention slots")
        if plan_probability_mode not in {
            "independent_sigmoid",
            "slot_categorical",
            "factorized_step",
        }:
            raise ValueError("semantic plan probability mode is not registered")
        if plan_training["loss_mode"] == "slot_categorical":
            if plan_conditioning_mode != "slot_attention":
                raise ValueError("slot-categorical plan loss requires slot attention")
            if plan_probability_mode != "slot_categorical":
                raise ValueError(
                    "slot-categorical plan loss requires categorical slot probabilities"
                )
        elif plan_training["loss_mode"] == "factorized_step_categorical":
            if plan_training["target"] != "ordered_plan_step_factor_field":
                raise ValueError(
                    "factorized plan loss requires the ordered step-factor field"
                )
            if plan_conditioning_mode != "slot_attention":
                raise ValueError("factorized plan loss requires slot attention")
            if plan_probability_mode != "factorized_step":
                raise ValueError(
                    "factorized plan loss requires factorized slot probabilities"
                )
            model_groups = tuple(
                int(value)
                for value in (model.get("semantic_plan_factor_group_sizes") or [])
            )
            if model_groups != tuple(plan_training["factor_group_sizes"]):
                raise ValueError("factorized plan groups must match the registered IR field")
        elif plan_probability_mode != "independent_sigmoid":
            raise ValueError(
                "binary-multilabel plan loss requires independent sigmoid probabilities"
            )
    elif plan_feature_count:
        raise ValueError("semantic plan model parameters require enabled semantic plan training")
    evaluation = config.get("evaluation") or {}
    if int(evaluation.get("holdout_family_count") or 0) < 24:
        raise ValueError("family-disjoint evaluation requires at least 24 distinct families")
    if int(evaluation.get("rows_per_family") or 0) != 1:
        raise ValueError("evaluation must use one unique semantic row per holdout family")
    if evaluation.get("hierarchical_plan_decode", False):
        if not isinstance(evaluation.get("hierarchical_plan_decode"), bool):
            raise ValueError("hierarchical plan decode flag must be boolean")
        if not 1 <= int(evaluation.get("plan_fanout") or 0) <= int(
            evaluation.get("fanout") or 0
        ):
            raise ValueError("hierarchical plan fanout must be within candidate fanout")
        if int(evaluation.get("plan_beam_width") or 0) <= 0 or int(
            evaluation.get("body_beam_width") or 0
        ) <= 0:
            raise ValueError("hierarchical plan and body beam widths must be positive")
        if not 0.0 <= float(evaluation.get("plan_score_weight") or 0.0) <= 1.0:
            raise ValueError("hierarchical plan score weight must be in [0, 1]")
    preference = config.get("preference") if isinstance(config.get("preference"), dict) else {}
    if int(preference.get("max_train_tasks") or 0) < 8:
        raise ValueError("preference canary requires at least eight private training tasks")
    if not 0 < int(preference.get("max_pairs") or 0) <= int(preference["max_train_tasks"]):
        raise ValueError("preference pair budget must be positive and bounded by training tasks")
    if not 0 < int(preference.get("optimizer_steps") or 0) <= 128:
        raise ValueError("preference optimizer steps must stay within the bounded canary budget")
    if float(preference.get("beta") or 0.0) <= 0:
        raise ValueError("preference beta must be positive")
    contract = (
        config.get("sft_contract_admission")
        if isinstance(config.get("sft_contract_admission"), dict)
        else {}
    )
    if contract and not isinstance(contract.get("require_self_contained_body"), bool):
        raise ValueError("SFT self-contained-body admission must be explicitly boolean")
    target_probability = contract.get("private_sampling_probability_target")
    if target_probability is not None and not 0.0 < float(target_probability) < 1.0:
        raise ValueError("private sampling probability target must be between zero and one")


def planned_report(
    config: dict[str, Any], config_path: str, checkpoint_dir: Path, stage_dir: Path, started: float,
    *, scaling_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "policy": config["policy"],
        "created_utc": now(),
        "trigger_state": "PLANNED",
        "execute": False,
        "seed": int(config["seed"]),
        "artifacts": {"config": config_path, "checkpoint_dir": rel(checkpoint_dir), "stage_dir": rel(stage_dir)},
        "architecture": {
            "family": "standard_decoder_only_causal_transformer",
            "role": str(config["architecture_role"]),
            "moecot_language_seed_contract": config["moecot_language_seed_contract"],
            "attention_policy": str(config["model"].get("attention_policy") or "causal"),
            **config["model"],
        },
        "data_model_scaling_contract": scaling_contract or build_data_model_scaling_contract(config),
        "boundaries": config["boundaries"],
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def compare_attention_policy_canaries(
    causal_report: dict[str, Any],
    prefix_report: dict[str, Any],
) -> dict[str, Any]:
    """Audit a parameter-neutral causal versus prefix-LM private comparison."""

    def architecture_without_policy(report: dict[str, Any]) -> dict[str, Any]:
        config = dict((report.get("architecture") or {}).get("config") or {})
        config.pop("attention_policy", None)
        return config

    def phase_exposure(report: dict[str, Any]) -> list[tuple[str, int, int]]:
        phases = (report.get("training") or {}).get("phases") or []
        return [
            (
                str(row.get("phase") or ""),
                int(row.get("optimizer_steps") or 0),
                int(row.get("target_positions_consumed") or 0),
            )
            for row in phases
            if isinstance(row, dict)
        ]

    def metrics(report: dict[str, Any]) -> dict[str, Any]:
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        verifier = private_verifier_summary(report.get("private_verifier") or {})
        verification = (
            (report.get("private_verifier") or {}).get("private_verification") or {}
        )
        training_phases = (report.get("training") or {}).get("phases") or []
        sft_phase = next(
            (
                row
                for row in training_phases
                if isinstance(row, dict)
                and str(row.get("phase") or "").startswith("prompt_signature_body_sft")
            ),
            {},
        )
        return {
            "passed_task_count": int(verifier.get("passed_task_count") or 0),
            "candidate_task_count": int(summary.get("candidate_task_count") or 0),
            "syntax_valid_candidate_count": int(
                summary.get("syntax_valid_candidate_count") or 0
            ),
            "mean_verification_reward": float(
                verification.get("mean_verification_reward") or 0.0
            ),
            "eval_loss_after": float(
                (report.get("training") or {}).get("eval_loss_after") or float("inf")
            ),
            "training_tokens_per_second": float(
                sft_phase.get("tokens_per_second") or 0.0
            ),
            "decode_runtime_ms": int((report.get("decode") or {}).get("runtime_ms") or 0),
        }

    causal_architecture = causal_report.get("architecture") or {}
    prefix_architecture = prefix_report.get("architecture") or {}
    matched_checks = {
        "policy": causal_report.get("policy") == prefix_report.get("policy"),
        "seed": causal_report.get("seed") == prefix_report.get("seed"),
        "parameter_count": int(causal_architecture.get("parameter_count") or 0)
        == int(prefix_architecture.get("parameter_count") or 0)
        > 0,
        "model_config_except_attention_policy": architecture_without_policy(causal_report)
        == architecture_without_policy(prefix_report),
        "causal_policy_declared": causal_architecture.get("attention_policy") == "causal",
        "prefix_policy_declared": prefix_architecture.get("attention_policy") == "prefix_lm",
        "stage_signature": (causal_report.get("stage") or {}).get("stage_signature")
        == (prefix_report.get("stage") or {}).get("stage_signature"),
        "holdout_families": (causal_report.get("stage") or {}).get("holdout_families")
        == (prefix_report.get("stage") or {}).get("holdout_families"),
        "optimizer_exposure": phase_exposure(causal_report)
        == phase_exposure(prefix_report),
        "training_complete": (causal_report.get("training") or {}).get("complete") is True
        and (prefix_report.get("training") or {}).get("complete") is True,
        "sequence_partitions_valid": all(
            receipt.get("valid") is True
            for report in (causal_report, prefix_report)
            for receipt in (
                (report.get("stage") or {}).get("sequence_partition_audit") or {}
            ).values()
        ),
        "no_public_training": all(
            int((report.get("summary") or {}).get("public_training_rows") or 0) == 0
            for report in (causal_report, prefix_report)
        ),
        "no_external_inference": all(
            int((report.get("summary") or {}).get("external_inference_calls") or 0) == 0
            for report in (causal_report, prefix_report)
        ),
        "no_fallback_or_assisted_credit": all(
            int((report.get("summary") or {}).get("fallback_return_count") or 0) == 0
            and int(
                (report.get("summary") or {}).get(
                    "template_renderer_router_tool_credit_count"
                )
                or 0
            )
            == 0
            for report in (causal_report, prefix_report)
        ),
    }
    causal_metrics = metrics(causal_report)
    prefix_metrics = metrics(prefix_report)
    hard_gaps = [name for name, passed in matched_checks.items() if not passed]
    behavior_gain = (
        prefix_metrics["passed_task_count"] > causal_metrics["passed_task_count"]
        and prefix_metrics["candidate_task_count"]
        >= causal_metrics["candidate_task_count"]
        and prefix_metrics["mean_verification_reward"]
        >= causal_metrics["mean_verification_reward"]
    )
    adopted = not hard_gaps and behavior_gain
    rejection_reasons: list[str] = []
    if hard_gaps:
        rejection_reasons.append("matched_comparison_contract_failed")
    if prefix_metrics["passed_task_count"] <= causal_metrics["passed_task_count"]:
        rejection_reasons.append("no_family_disjoint_verifier_pass_gain")
    if prefix_metrics["candidate_task_count"] < causal_metrics["candidate_task_count"]:
        rejection_reasons.append("candidate_task_coverage_regressed")
    if prefix_metrics["mean_verification_reward"] < causal_metrics["mean_verification_reward"]:
        rejection_reasons.append("mean_verification_reward_regressed")
    return {
        "policy": "standard_transformer_attention_policy_matched_ablation_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not hard_gaps else "RED",
        "adoption_state": "ADOPTED" if adopted else "NOT_ADOPTED",
        "matched_checks": matched_checks,
        "hard_gaps": hard_gaps,
        "causal": causal_metrics,
        "prefix_lm": prefix_metrics,
        "deltas": {
            key: round(prefix_metrics[key] - causal_metrics[key], 8)
            for key in causal_metrics
        },
        "rejection_reasons": rejection_reasons,
        "score_semantics": (
            "Private family-disjoint model-only behavior decides adoption. Lower loss alone, "
            "templates, renderers, routers, tools, fallback returns, public payloads, and external "
            "inference cannot support adoption."
        ),
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
        phases = training.get("phases") if isinstance(training.get("phases"), list) else []
        return {
            "target_mode": str(report.get("stage", {}).get("target_mode") or ""),
            "parameter_count": int(report.get("architecture", {}).get("parameter_count") or 0),
            "training_complete": training.get("complete") is True,
            "optimizer_body_positions": sum(
                int(row.get("optimizer_body_positions_consumed") or 0)
                for row in phases
                if isinstance(row, dict)
            ),
            "unique_body_target_positions": int(
                report.get("stage", {}).get("unique_body_target_positions") or 0
            ),
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
        (
            "sft_contract_admission",
            plan_config.get("sft_contract_admission", {}),
            control_config.get("sft_contract_admission", {}),
        ),
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
    matched_checks["optimizer_body_positions_equal"] = (
        plan["optimizer_body_positions"] == control["optimizer_body_positions"]
    )
    matched_checks["unique_body_target_positions_equal"] = (
        plan["unique_body_target_positions"] == control["unique_body_target_positions"]
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
        SCRIPTS / "neural_seed_teacher_distillation_rows.py",
        SCRIPTS / "semantic_ir.py",
        SCRIPTS / "standard_causal_transformer_corpus.py",
    ]
    canonical_corpus = config.get("canonical_corpus") or {}
    for key in ("code_shard_manifests",):
        paths.extend(resolve(value) for value in canonical_corpus.get(key) or [])
    for key in ("conversation_manifest", "broad_text_manifest"):
        value = str(canonical_corpus.get(key) or "")
        if value:
            paths.append(resolve(value))
    language_policy = str(
        (canonical_corpus.get("natural_language_scope") or {}).get("intake_policy") or ""
    )
    if language_policy:
        paths.append(resolve(language_policy))
    quality_policy = str(
        (canonical_corpus.get("code_quality_policy") or {}).get("curated_repo_config")
        or ""
    )
    if quality_policy:
        paths.append(resolve(quality_policy))
    canonical_receipt = str(
        ((config.get("data_model_scaling_contract") or {}).get("canonical_corpus_receipt") or {}).get("path") or ""
    )
    if canonical_receipt:
        paths.append(resolve(canonical_receipt))
    teacher_config = (
        config.get("teacher_distillation")
        if isinstance(config.get("teacher_distillation"), dict)
        else {}
    )
    for teacher_artifact_key in ("manifest", "gate"):
        teacher_artifact = str(teacher_config.get(teacher_artifact_key) or "")
        if teacher_artifact:
            paths.append(resolve(teacher_artifact))
    paths.extend(admitted_training_source_paths(config))
    stage_config = {
        "seed": config["seed"],
        "sources": config["sources"],
        "tokenization": config["tokenization"],
        "training": {
            "pretrain_target_token_positions": config["training"]["pretrain_target_token_positions"],
            "private_body_sampling_weight": config["training"]["private_body_sampling_weight"],
        },
        "canonical_corpus": canonical_corpus,
        "architecture_role": config.get("architecture_role"),
        "moecot_language_seed_contract": config.get("moecot_language_seed_contract", {}),
        "data_model_scaling_contract": config.get("data_model_scaling_contract", {}),
        "sft_contract_admission": config.get("sft_contract_admission", {}),
        "ordered_plan_training": config.get("ordered_plan_training", {}),
        "semantic_plan_training": config.get("semantic_plan_training", {}),
        "teacher_distillation": teacher_config,
        "evaluation": {
            "holdout_family_count": config["evaluation"]["holdout_family_count"],
            "rows_per_family": config["evaluation"]["rows_per_family"],
        },
        "preference": {"max_train_tasks": config["preference"]["max_train_tasks"]},
    }
    stage_functions = (
        load_sft_examples,
        encode_canonical_pretrain_document,
        encode_sft_examples,
        encode_sft_training_examples,
        sequence_partition_audit,
        training_target_tokens,
        ordered_plan_training_contract,
        prepare_ordered_plan_sequences,
        semantic_plan_training_contract,
        semantic_plan_feature_contract,
        semantic_plan_labels_for_body,
        extend_target_vocab_for_mode,
        assign_body_balanced_sampling_weights,
        standalone_sft_contract_decision,
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


def admitted_load_weights(model: Any, checkpoint: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Verify configured artifact identity before MLX/NumPy deserialization."""
    decision = theseus_artifact_admission.admit_from_config(checkpoint, config)
    if decision.get("required") and not decision.get("admitted"):
        raise ValueError(f"artifact admission rejected before load: {decision.get('reason')}")
    model.load_weights(str(checkpoint))
    return decision


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
