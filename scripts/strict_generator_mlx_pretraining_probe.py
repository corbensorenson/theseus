#!/usr/bin/env python3
"""MLX high-level strict-generator pretraining probe.

This is a bounded acceleration contract for the same private
prompt/signature-to-body corpus used by ``strict_generator_pretraining_spine``.
It does not emit candidates, does not claim checkpoint parity with the Torch
strict generator, and does not credit templates/tools/renderers as learned
generation. Its job is to answer one question: can the strict-generator
training objective run faster through MLX high-level transformer ops on this
Mac?
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neural_seed_code_proposer_comparator import (  # noqa: E402
    build_vocab,
    deterministic_sample,
    dict_or_empty,
    get_path,
    load_private_rows,
    rel,
    resolve,
    stable_hash,
)
from neural_seed_token_decoder_comparator import (  # noqa: E402
    build_target_vocab,
    decoder_source_text,
    encode_target_rows,
    full_state_pretraining_config,
    full_state_source_vocab_extension_summary,
    full_state_target_vocab_extension_summary,
)
from neural_seed_token_decoder_support import PLAN_BODY_START_TOKEN, semantic_plan_from_body, target_tokens  # noqa: E402
from strict_generator_pretraining_spine import (  # noqa: E402
    config_with_budget_overrides as strict_spine_config_with_budget_overrides,
    encode_staged_full_state_rows,
    safe_slug,
    stage_full_state_examples,
    transformer_dims_with_budget,
)


DEFAULT_CONFIG = ROOT / "configs" / "neural_seed_token_decoder_comparator.json"
DEFAULT_OUT = ROOT / "reports" / "strict_generator_mlx_pretraining_probe.json"
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints" / "strict_generator_mlx_probe"
SEMANTIC_SLOT_ROLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("return_shape", ("SLOT:RETURN_SHAPE_",)),
    ("loop_source", ("SLOT:LOOP_SOURCE_",)),
    ("init", ("SLOT:INIT_",)),
    ("update", ("SLOT:UPDATE_",)),
    ("state_finalizer", ("SLOT:STATE_FINALIZER_",)),
    ("state_update", ("SLOT:STATE_UPDATE_",)),
    ("binding_loop", ("SLOT:BIND_LOOP_", "SLOT:BIND_BRANCH_")),
    ("binding_update", ("SLOT:BIND_UPDATE_",)),
    ("binding_finalizer", ("SLOT:BIND_FINALIZER_", "SLOT:BIND_RETURN_")),
    ("finalizer", ("SLOT:FINALIZER_",)),
    ("statement_sequence", ("SLOT:STMT_SEQ_",)),
    ("statement_loop", ("SLOT:STMT_LOOP_",)),
    ("statement_final", ("SLOT:STMT_FINAL_",)),
    ("condition", ("SLOT:COND_",)),
    ("guarded_return", ("SLOT:RETURN_GUARDED_",)),
    ("default_return", ("SLOT:RETURN_DEFAULT_",)),
    ("expression_call", ("SLOT:EXPR_CALL_",)),
    ("expression_binop", ("SLOT:EXPR_BINOP_",)),
    ("expression_return", ("SLOT:EXPR_RETURN_",)),
    ("expression_update", ("SLOT:EXPR_LOOP_UPDATE_", "SLOT:EXPR_TOP_ASSIGN_")),
    ("expression_branch", ("SLOT:EXPR_LOOP_BRANCH_", "SLOT:EXPR_COMPARE_", "SLOT:EXPR_BOOLOP_")),
    ("expression_structure", ("SLOT:EXPR_INDEXING", "SLOT:EXPR_COMPREHENSION_")),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--checkpoint-dir", default=rel(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--budget-id", default="strict_generator_smoke")
    parser.add_argument("--batch-size-override", type=int, default=0)
    parser.add_argument("--target-mode-override", default="")
    parser.add_argument("--target-token-positions-override", type=int, default=0)
    parser.add_argument("--max-source-tokens-override", type=int, default=0)
    parser.add_argument("--max-target-tokens-override", type=int, default=0)
    parser.add_argument(
        "--rung-token-positions",
        default="",
        help="Comma/space separated token-position milestones where replayable intermediate checkpoints are saved.",
    )
    parser.add_argument("--semantic-plan-loss-weight", type=float, default=-1.0)
    parser.add_argument("--semantic-slot-loss-weight", type=float, default=-1.0)
    parser.add_argument("--source-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--source-contrastive-margin", type=float, default=0.25)
    parser.add_argument("--source-contrastive-prefix-tokens", type=int, default=0)
    parser.add_argument(
        "--source-contrastive-span-mode",
        choices=("prefix", "after_body_start"),
        default="prefix",
        help="Which target span the source mismatch margin should compare. Default preserves legacy prefix behavior.",
    )
    parser.add_argument("--enable-primary-dataflow-weights", action="store_true")
    parser.add_argument("--primary-dataflow-weight-scale", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = read_json(resolve(args.config))
    report = run_probe(
        config,
        config_path=args.config,
        checkpoint_dir=resolve(args.checkpoint_dir),
        budget_id=str(args.budget_id or "strict_generator_smoke"),
        batch_size_override=max(0, int(args.batch_size_override or 0)),
        target_mode_override=str(args.target_mode_override or ""),
        target_token_positions_override=max(0, int(args.target_token_positions_override or 0)),
        max_source_tokens_override=max(0, int(args.max_source_tokens_override or 0)),
        max_target_tokens_override=max(0, int(args.max_target_tokens_override or 0)),
        rung_token_positions=parse_token_position_milestones(str(args.rung_token_positions or "")),
        semantic_plan_loss_weight=float(args.semantic_plan_loss_weight),
        semantic_slot_loss_weight=float(args.semantic_slot_loss_weight),
        source_contrastive_loss_weight=max(0.0, float(args.source_contrastive_loss_weight or 0.0)),
        source_contrastive_margin=max(0.0, float(args.source_contrastive_margin or 0.0)),
        source_contrastive_prefix_tokens=max(0, int(args.source_contrastive_prefix_tokens or 0)),
        source_contrastive_span_mode=str(args.source_contrastive_span_mode or "prefix"),
        enable_primary_dataflow_weights=bool(args.enable_primary_dataflow_weights),
        primary_dataflow_weight_scale=max(0.0, float(args.primary_dataflow_weight_scale or 1.0)),
        execute=bool(args.execute),
    )
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW", "PLANNED"} else 2


def run_probe(
    config: dict[str, Any],
    *,
    config_path: str,
    checkpoint_dir: Path,
    budget_id: str,
    batch_size_override: int = 0,
    target_mode_override: str = "",
    target_token_positions_override: int = 0,
    max_source_tokens_override: int = 0,
    max_target_tokens_override: int = 0,
    rung_token_positions: list[int] | None = None,
    semantic_plan_loss_weight: float = -1.0,
    semantic_slot_loss_weight: float = -1.0,
    source_contrastive_loss_weight: float = 0.0,
    source_contrastive_margin: float = 0.25,
    source_contrastive_prefix_tokens: int = 0,
    source_contrastive_span_mode: str = "prefix",
    enable_primary_dataflow_weights: bool = False,
    primary_dataflow_weight_scale: float = 1.0,
    execute: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = dict_or_empty(config.get("strict_generator_pretraining"))
    budget = dict(selected_budget(cfg, budget_id))
    runtime_overrides: dict[str, Any] = {}
    if batch_size_override > 0:
        budget["batch_size"] = int(batch_size_override)
        runtime_overrides["batch_size"] = int(batch_size_override)
    if str(target_mode_override or "").strip():
        budget["target_mode"] = str(target_mode_override).strip()
        runtime_overrides["target_mode"] = str(target_mode_override).strip()
    if target_token_positions_override > 0:
        budget["target_token_positions"] = int(target_token_positions_override)
        runtime_overrides["target_token_positions"] = int(target_token_positions_override)
    if max_source_tokens_override > 0:
        budget["max_source_tokens"] = int(max_source_tokens_override)
        runtime_overrides["max_source_tokens"] = int(max_source_tokens_override)
    if max_target_tokens_override > 0:
        budget["max_target_tokens"] = int(max_target_tokens_override)
        runtime_overrides["max_target_tokens"] = int(max_target_tokens_override)
    rung_token_positions = sorted({int(value) for value in (rung_token_positions or []) if int(value) > 0})
    if rung_token_positions:
        runtime_overrides["rung_token_positions"] = rung_token_positions
    if semantic_plan_loss_weight >= 0.0:
        semantic_cfg = dict_or_empty(budget.get("semantic_plan_auxiliary_loss"))
        semantic_cfg.update(
            {
                "enabled": semantic_plan_loss_weight > 0.0,
                "policy": str(semantic_cfg.get("policy") or "private_body_derived_source_plan_auxiliary_v1"),
                "weight": float(semantic_plan_loss_weight),
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "uses_answer_metadata": False,
                "served_at_runtime": False,
                "candidate_generation_credit": 0,
            }
        )
        budget["semantic_plan_auxiliary_loss"] = semantic_cfg
        runtime_overrides["semantic_plan_auxiliary_loss_weight"] = float(semantic_plan_loss_weight)
    if semantic_slot_loss_weight >= 0.0:
        slot_cfg = dict_or_empty(budget.get("semantic_slot_auxiliary_loss"))
        slot_cfg.update(
            {
                "enabled": semantic_slot_loss_weight > 0.0,
                "policy": str(slot_cfg.get("policy") or "private_body_derived_source_slot_auxiliary_v1"),
                "weight": float(semantic_slot_loss_weight),
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "uses_answer_metadata": False,
                "served_at_runtime": False,
                "candidate_generation_credit": 0,
            }
        )
        budget["semantic_slot_auxiliary_loss"] = slot_cfg
        runtime_overrides["semantic_slot_auxiliary_loss_weight"] = float(semantic_slot_loss_weight)
    if source_contrastive_loss_weight > 0.0:
        budget["source_contrastive_loss_weight"] = float(source_contrastive_loss_weight)
        budget["source_contrastive_margin"] = float(source_contrastive_margin)
        budget["source_contrastive_prefix_tokens"] = int(source_contrastive_prefix_tokens)
        budget["source_contrastive_span_mode"] = (
            str(source_contrastive_span_mode or "prefix")
            if str(source_contrastive_span_mode or "prefix") in {"prefix", "after_body_start"}
            else "prefix"
        )
        runtime_overrides["source_contrastive_loss_weight"] = float(source_contrastive_loss_weight)
        runtime_overrides["source_contrastive_margin"] = float(source_contrastive_margin)
        runtime_overrides["source_contrastive_prefix_tokens"] = int(source_contrastive_prefix_tokens)
        runtime_overrides["source_contrastive_span_mode"] = str(budget["source_contrastive_span_mode"])
    if enable_primary_dataflow_weights:
        apply_primary_dataflow_weight_override(budget, scale=primary_dataflow_weight_scale)
        runtime_overrides["primary_dataflow_weights_enabled"] = True
        runtime_overrides["primary_dataflow_weight_scale"] = float(primary_dataflow_weight_scale)
    if not execute:
        return {
            "policy": "project_theseus_strict_generator_mlx_pretraining_probe_v1",
            "created_utc": now(),
            "config": config_path,
            "execute": False,
            "trigger_state": "PLANNED",
            "summary": {
                "budget_id": budget_id,
                "checkpoint_dir": rel(checkpoint_dir),
                "runtime_overrides": runtime_overrides,
                "public_training_rows": 0,
                "external_inference_calls": 0,
            },
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        }

    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        import mlx.utils as mlx_utils
    except BaseException as exc:
        return {
            "policy": "project_theseus_strict_generator_mlx_pretraining_probe_v1",
            "created_utc": now(),
            "config": config_path,
            "execute": True,
            "trigger_state": "RED",
            "summary": {
                "mlx_available": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:600],
                "public_training_rows": 0,
                "external_inference_calls": 0,
            },
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        }

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = train_budget_mlx(
        config,
        budget,
        budget_id=budget_id,
        checkpoint_dir=checkpoint_dir,
        rung_token_positions=rung_token_positions,
        mx=mx,
        nn=nn,
        optim=optim,
        mlx_utils=mlx_utils,
    )
    gates = build_gates(payload)
    hard_pass = all(row["passed"] for row in gates if row["severity"] == "hard")
    trigger_state = "GREEN" if hard_pass else "RED"
    if trigger_state == "GREEN" and any(not row["passed"] for row in gates):
        trigger_state = "YELLOW"
    return {
        "policy": "project_theseus_strict_generator_mlx_pretraining_probe_v1",
        "created_utc": now(),
        "config": config_path,
        "execute": True,
        "trigger_state": trigger_state,
        "summary": {
            "budget_id": budget_id,
            "backend": payload.get("backend"),
            "device": str(payload.get("device")),
            "runtime_overrides": runtime_overrides,
            "parameter_count": payload.get("parameter_count"),
            "trainable_parameter_count": payload.get("trainable_parameter_count"),
            "optimizer_step_count": payload.get("optimizer_step_count"),
            "optimizer_token_positions_consumed": payload.get("optimizer_token_positions_consumed"),
            "training_plan": payload.get("training_plan"),
            "training_wall_time_ms": payload.get("training_wall_time_ms"),
            "training_tokens_per_second": payload.get("training_tokens_per_second"),
            "optimizer_steps_per_second": payload.get("optimizer_steps_per_second"),
            "heldout_lm_improved": payload.get("heldout_lm_improved"),
            "heldout_lm_loss_before": payload.get("heldout_lm_loss_before"),
            "heldout_lm_loss_after": payload.get("heldout_lm_loss_after"),
            "heldout_lm_perplexity_before": payload.get("heldout_lm_perplexity_before"),
            "heldout_lm_perplexity_after": payload.get("heldout_lm_perplexity_after"),
            "parameter_update_fraction": payload.get("parameter_update_fraction"),
            "parameter_tensor_update_fraction": payload.get("parameter_tensor_update_fraction"),
            "row_summary": payload.get("row_summary"),
            "checkpoint": payload.get("checkpoint"),
            "checkpoint_sha256": payload.get("checkpoint_sha256"),
            "vocab": payload.get("vocab"),
            "vocab_sha256": payload.get("vocab_sha256"),
            "rung_checkpoints": payload.get("rung_checkpoints"),
            "target_mode": payload.get("target_mode"),
            "source_vocab_size": payload.get("source_vocab_size"),
            "target_vocab_size": payload.get("target_vocab_size"),
            "max_source": payload.get("max_source"),
            "max_target": payload.get("max_target"),
            "loss_weighting": payload.get("loss_weighting"),
            "semantic_plan_auxiliary": payload.get("semantic_plan_auxiliary"),
            "semantic_slot_auxiliary": payload.get("semantic_slot_auxiliary"),
            "semantic_plan_loss_before": get_path(payload, ["semantic_plan_auxiliary", "heldout_plan_loss_before"]),
            "semantic_plan_loss_after": get_path(payload, ["semantic_plan_auxiliary", "heldout_plan_loss_after"]),
            "semantic_plan_accuracy_before": get_path(payload, ["semantic_plan_auxiliary", "heldout_plan_accuracy_before"]),
            "semantic_plan_accuracy_after": get_path(payload, ["semantic_plan_auxiliary", "heldout_plan_accuracy_after"]),
            "source_contrastive_loss": payload.get("source_contrastive_loss"),
            "source_contrastive_gap_before": get_path(payload, ["source_contrastive_loss", "heldout_source_loss_gap_before"]),
            "source_contrastive_gap_after": get_path(payload, ["source_contrastive_loss", "heldout_source_loss_gap_after"]),
            "public_training_rows": 0,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "open_or_pretrained_model_weights_used": False,
            "fallback_template_router_tool_credit_count": 0,
        },
        "budget": payload,
        "gates": gates,
        "score_semantics": (
            "MLX acceleration probe for strict-generator pretraining only. It emits no candidates, "
            "uses no public benchmark payloads, calls no external inference, and credits no templates, "
            "routers, tools, structural adapters, or fallback returns as learned generation."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


class MlxStrictGenerator:
    def __init__(
        self,
        *,
        source_vocab_size: int,
        target_vocab_size: int,
        max_source_len: int,
        max_target_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        mx: Any,
        nn: Any,
    ) -> None:
        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.source_embedding = nn.Embedding(source_vocab_size, d_model)
                self.target_embedding = nn.Embedding(target_vocab_size, d_model)
                self.source_position = mx.zeros((1, max_source_len, d_model), dtype=mx.float32)
                self.target_position = mx.zeros((1, max_target_len, d_model), dtype=mx.float32)
                self.encoder = nn.TransformerEncoder(
                    num_layers,
                    d_model,
                    nhead,
                    dim_feedforward,
                    0.0,
                    nn.gelu,
                    True,
                )
                self.decoder = nn.TransformerDecoder(
                    num_layers,
                    d_model,
                    nhead,
                    dim_feedforward,
                    0.0,
                    nn.gelu,
                    True,
                )
                self.output = nn.Linear(d_model, target_vocab_size)
                self.plan_router = nn.Linear(d_model, target_vocab_size)
                self.slot_router = nn.Linear(d_model, target_vocab_size * len(SEMANTIC_SLOT_ROLES))

            def encode_source(self, src: Any) -> tuple[Any, Any]:
                src_mask = additive_padding_mask(src, mx)
                valid = (src != 0).astype(mx.float32)[:, :, None]
                source = self.source_embedding(src) + self.source_position[:, : src.shape[1], :]
                memory = self.encoder(source, src_mask)
                pooled = mx.sum(memory * valid, axis=1) / mx.maximum(
                    mx.sum(valid, axis=1),
                    mx.array(1.0, dtype=mx.float32),
                )
                return memory, pooled

            def semantic_plan_logits(self, src: Any) -> Any:
                _memory, pooled = self.encode_source(src)
                return self.plan_router(pooled)

            def semantic_slot_logits(self, src: Any) -> Any:
                _memory, pooled = self.encode_source(src)
                flat = self.slot_router(pooled)
                return flat.reshape((flat.shape[0], len(SEMANTIC_SLOT_ROLES), target_vocab_size))

            def __call__(self, src: Any, tgt_in: Any) -> Any:
                src_mask = additive_padding_mask(src, mx)
                tgt_mask = nn.MultiHeadAttention.create_additive_causal_mask(tgt_in.shape[1], mx.float32)
                target = self.target_embedding(tgt_in) + self.target_position[:, : tgt_in.shape[1], :]
                memory, _pooled = self.encode_source(src)
                decoded = self.decoder(target, memory, tgt_mask, src_mask)
                return self.output(decoded)

        self.model = _Model()


def train_budget_mlx(
    config: dict[str, Any],
    budget: dict[str, Any],
    *,
    budget_id: str,
    checkpoint_dir: Path,
    rung_token_positions: list[int] | None = None,
    mx: Any,
    nn: Any,
    optim: Any,
    mlx_utils: Any,
) -> dict[str, Any]:
    seed = int(budget.get("seed") or get_path(config, ["matched_budget", "seeds"], [23])[0] or 23)
    random.seed(seed)
    mx.random.seed(seed)
    working_config = strict_spine_config_with_budget_overrides(config, budget)
    data_cfg = dict_or_empty(working_config.get("data"))
    text_views = dict_or_empty(working_config.get("text_views"))
    matched_budget = dict_or_empty(working_config.get("matched_budget"))
    full_state_cfg = full_state_pretraining_config(working_config)
    target_mode = str(get_path(working_config, ["body_structure_decoder", "target_mode"], "body_tokens"))
    train_rows_all = load_private_rows(resolve(str(data_cfg.get("train_jsonl") or "")), data_cfg)
    train_rows = deterministic_sample(train_rows_all, int(data_cfg.get("max_train_rows") or 512), seed)
    staged = stage_full_state_examples(
        working_config,
        budget,
        budget_id=budget_id,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )
    source_vocab_extension_texts = list(staged.get("source_vocab_extension_texts") or [])
    target_vocab_extension_bodies = list(staged.get("target_vocab_extension_bodies") or [])
    source_vocab = build_vocab(
        [
            decoder_source_text(row, text_views.get(view, []))
            for view in ["sts_off", "sts_on"]
            for row in train_rows
        ]
        + source_vocab_extension_texts,
        max_vocab=int(matched_budget.get("max_source_vocab") or 4096),
    )
    target_vocab = build_target_vocab(
        [str(row.get("solution_body") or "") for row in train_rows] + target_vocab_extension_bodies,
        max_vocab=int(matched_budget.get("max_target_vocab") or 4096),
        target_mode=target_mode,
    )
    semantic_plan_cfg = semantic_plan_auxiliary_config(budget, matched_budget)
    semantic_slot_cfg = semantic_slot_auxiliary_config(budget, matched_budget)
    semantic_plan_vocab_summary = extend_target_vocab_with_semantic_plan_tokens(
        target_vocab,
        [str(row.get("body") or "") for row in list(staged.get("examples") or [])] + target_vocab_extension_bodies,
        enabled=bool(semantic_plan_cfg.get("enabled")),
    )
    semantic_slot_vocab_summary = extend_target_vocab_with_semantic_slot_tokens(
        target_vocab,
        [str(row.get("body") or "") for row in list(staged.get("examples") or [])] + target_vocab_extension_bodies,
        target_mode=target_mode,
        enabled=bool(semantic_slot_cfg.get("enabled")),
    )
    max_source = int(matched_budget.get("max_source_tokens") or 96)
    max_target = int(matched_budget.get("max_target_tokens") or 160)
    rows = encode_staged_full_state_rows(
        staged,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        max_source=max_source,
        max_target=max_target,
        target_mode=target_mode,
    )
    if not bool(rows.get("active")):
        return {
            "id": budget_id,
            "active": False,
            "reason": "no_active_full_state_rows",
            "row_summary": dict_or_empty(rows.get("summary")),
            "public_training_rows": 0,
            "external_inference_calls": 0,
        }
    dims = transformer_dims_with_budget(working_config, budget)
    model = MlxStrictGenerator(
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        max_source_len=max_source,
        max_target_len=max_target,
        mx=mx,
        nn=nn,
        **dims,
    ).model
    source_rows = list(rows.get("source_rows") or [])
    target_rows = list(rows.get("target_rows") or [])
    eval_source_rows = list(rows.get("eval_source_rows") or [])
    eval_target_rows = list(rows.get("eval_target_rows") or [])
    train_examples = training_examples_for_encoded_rows(staged, rows)
    eval_examples = eval_examples_for_encoded_rows(staged, rows)
    loss_weight_rows, loss_weight_summary = target_loss_weight_rows(
        target_rows,
        train_examples=train_examples,
        target_vocab=target_vocab,
        budget=budget,
        matched_budget=matched_budget,
    )
    pad_id = int(target_vocab.get("<pad>", 0))
    plan_target_rows, semantic_plan_target_summary = semantic_plan_target_ids(
        train_examples,
        target_vocab=target_vocab,
        pad_id=pad_id,
        enabled=bool(semantic_plan_cfg.get("enabled")),
    )
    eval_plan_target_rows, semantic_plan_eval_target_summary = semantic_plan_target_ids(
        eval_examples,
        target_vocab=target_vocab,
        pad_id=pad_id,
        enabled=bool(semantic_plan_cfg.get("enabled")),
    )
    plan_sample_weight_rows, semantic_plan_balance_summary = semantic_plan_sample_weights(
        plan_target_rows,
        target_summary=semantic_plan_target_summary,
        pad_id=pad_id,
        config=semantic_plan_cfg,
    )
    slot_target_rows, semantic_slot_target_summary = semantic_slot_target_ids(
        train_examples,
        target_vocab=target_vocab,
        pad_id=pad_id,
        target_mode=target_mode,
        enabled=bool(semantic_slot_cfg.get("enabled")),
    )
    eval_slot_target_rows, semantic_slot_eval_target_summary = semantic_slot_target_ids(
        eval_examples,
        target_vocab=target_vocab,
        pad_id=pad_id,
        target_mode=target_mode,
        enabled=bool(semantic_slot_cfg.get("enabled")),
    )
    slot_sample_weight_rows, semantic_slot_balance_summary = semantic_slot_sample_weights(
        slot_target_rows,
        target_summary=semantic_slot_target_summary,
        pad_id=pad_id,
        config=semantic_slot_cfg,
    )
    semantic_slot_class_summary = semantic_slot_role_class_summary(target_vocab)
    batch_size = int(budget.get("batch_size") or matched_budget.get("batch_size") or 64)
    configured_epochs = max(1, int(budget.get("epochs") or 1))
    configured_step_limit = max(0, int(budget.get("steps") or matched_budget.get("steps") or 0))
    target_token_positions = max(
        0,
        int(budget.get("target_token_positions") or matched_budget.get("target_token_positions") or 0),
    )
    lr = float(budget.get("learning_rate") or matched_budget.get("learning_rate") or 0.0008)
    weight_decay = float(budget.get("weight_decay") or matched_budget.get("weight_decay") or 0.0001)
    source_contrastive_weight = float(budget.get("source_contrastive_loss_weight") or 0.0)
    source_contrastive_margin = float(budget.get("source_contrastive_margin") or 0.25)
    source_contrastive_prefix_tokens = max(0, int(budget.get("source_contrastive_prefix_tokens") or 0))
    source_contrastive_span_mode = str(budget.get("source_contrastive_span_mode") or "prefix")
    if source_contrastive_span_mode not in {"prefix", "after_body_start"}:
        source_contrastive_span_mode = "prefix"
    source_contrastive_body_start_id = int(target_vocab.get(PLAN_BODY_START_TOKEN, -1))
    semantic_plan_weight = float(semantic_plan_cfg.get("weight") or 0.0)
    semantic_plan_active = (
        bool(semantic_plan_cfg.get("enabled"))
        and semantic_plan_weight > 0.0
        and bool(semantic_plan_target_summary.get("active_target_count"))
    )
    semantic_slot_weight = float(semantic_slot_cfg.get("weight") or 0.0)
    semantic_slot_active = (
        bool(semantic_slot_cfg.get("enabled"))
        and semantic_slot_weight > 0.0
        and bool(semantic_slot_target_summary.get("active_target_count"))
        and hasattr(model, "semantic_slot_logits")
    )
    slot_role_class_ids = [
        mx.array(list(row.get("class_ids") or []), dtype=mx.int32)
        for row in semantic_slot_class_summary
    ]
    if slot_role_class_ids:
        mx.eval(*slot_role_class_ids)
    before = parameter_snapshot(model, mlx_utils, mx)
    heldout_before = evaluate_loss_mlx(model, eval_source_rows, eval_target_rows, batch_size=batch_size, pad_id=pad_id, mx=mx, nn=nn)
    contrast_before = evaluate_source_contrast_mlx(
        model,
        eval_source_rows,
        eval_target_rows,
        batch_size=batch_size,
        pad_id=pad_id,
        prefix_token_count=source_contrastive_prefix_tokens,
        span_mode=source_contrastive_span_mode,
        body_start_token_id=source_contrastive_body_start_id,
        mx=mx,
        nn=nn,
    )
    semantic_plan_before = evaluate_semantic_plan_mlx(
        model,
        eval_source_rows,
        eval_plan_target_rows,
        batch_size=batch_size,
        pad_id=pad_id,
        mx=mx,
        nn=nn,
    )
    semantic_slot_before = evaluate_semantic_slot_mlx(
        model,
        eval_source_rows,
        eval_slot_target_rows,
        batch_size=batch_size,
        pad_id=pad_id,
        enabled=semantic_slot_active,
        mx=mx,
        nn=nn,
        slot_role_class_ids=slot_role_class_ids,
    )
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
    loss_and_grad_plain = nn.value_and_grad(model, weighted_loss_fn_mlx)
    loss_and_grad_semantic_aux = (
        nn.value_and_grad(model, semantic_aux_weighted_loss_fn_mlx)
        if semantic_plan_active or semantic_slot_active
        else None
    )
    loss_and_grad_contrast = (
        nn.value_and_grad(model, source_contrastive_weighted_loss_fn_mlx)
        if source_contrastive_weight > 0.0
        else None
    )
    loss_and_grad_semantic_aux_contrast = (
        nn.value_and_grad(model, semantic_aux_source_contrastive_weighted_loss_fn_mlx)
        if source_contrastive_weight > 0.0 and (semantic_plan_active or semantic_slot_active)
        else None
    )
    order = list(range(len(source_rows)))
    losses: list[float] = []
    optimizer_steps = 0
    optimizer_token_positions = 0
    optimizer_windows_consumed = 0
    source_nonpad_by_row = [sum(1 for value in row if int(value) != 0) for row in source_rows]
    target_nonpad_by_row = [sum(1 for value in row if int(value) != pad_id) for row in target_rows]
    epoch_token_positions = int(sum(source_nonpad_by_row) + sum(target_nonpad_by_row))
    batches_per_epoch = max(1, ((len(order) + batch_size - 1) // batch_size))
    estimated_token_positions_per_step = max(1, (epoch_token_positions + batches_per_epoch - 1) // batches_per_epoch)
    target_derived_step_limit = (
        (target_token_positions + estimated_token_positions_per_step - 1) // estimated_token_positions_per_step
        if target_token_positions > 0
        else 0
    )
    target_step_limit_safety_steps = (
        max(2, (target_derived_step_limit + 49) // 50)
        if target_derived_step_limit > 0
        else 0
    )
    source_matrix = mx.array(source_rows, dtype=mx.int32)
    target_matrix = mx.array(target_rows, dtype=mx.int32)
    weight_matrix = mx.array(loss_weight_rows, dtype=mx.float32)
    plan_target_vector = mx.array(plan_target_rows, dtype=mx.int32)
    plan_weight_vector = mx.array(plan_sample_weight_rows, dtype=mx.float32)
    slot_target_matrix = mx.array(slot_target_rows, dtype=mx.int32)
    slot_weight_matrix = mx.array(slot_sample_weight_rows, dtype=mx.float32)
    mx.eval(
        source_matrix,
        target_matrix,
        weight_matrix,
        plan_target_vector,
        plan_weight_vector,
        slot_target_matrix,
        slot_weight_matrix,
        *slot_role_class_ids,
    )
    checkpoint_slug = safe_slug(budget_id)
    checkpoint_path = checkpoint_dir / f"strict_generator_mlx_{checkpoint_slug}.npz"
    vocab_path = checkpoint_dir / f"strict_generator_mlx_{checkpoint_slug}_vocab.json"
    vocab_payload = {
        "policy": "project_theseus_strict_generator_mlx_vocab_v1",
        "created_utc": now(),
        "budget_id": budget_id,
        "source_vocab": source_vocab,
        "target_vocab": target_vocab,
        "source_vocab_sha256": stable_hash(json.dumps(source_vocab, sort_keys=True)),
        "target_vocab_sha256": stable_hash(json.dumps(target_vocab, sort_keys=True)),
        "dims": dims,
        "max_source": max_source,
        "max_target": max_target,
        "target_mode": target_mode,
        "source_text_style": str(full_state_cfg.get("source_text_style") or "prompt_signature_metadata_v2"),
        "semantic_plan_auxiliary": {
            **semantic_plan_cfg,
            "active": semantic_plan_active,
            "vocab_extension": semantic_plan_vocab_summary,
            "train_targets": semantic_plan_target_summary,
            "eval_targets": semantic_plan_eval_target_summary,
        },
        "semantic_slot_auxiliary": {
            **semantic_slot_cfg,
            "active": semantic_slot_active,
            "vocab_extension": semantic_slot_vocab_summary,
            "class_summary": semantic_slot_class_summary,
            "class_balance": semantic_slot_balance_summary,
            "roles": semantic_slot_role_summary(),
            "train_targets": semantic_slot_target_summary,
            "eval_targets": semantic_slot_eval_target_summary,
        },
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "public_training_rows": 0,
        "external_inference_calls": 0,
    }
    # Write the vocab before training so any intermediate rung checkpoint is replayable
    # even if a long run is interrupted before the final checkpoint is emitted.
    write_json(vocab_path, vocab_payload)
    rung_milestones = [
        int(value)
        for value in sorted({int(value) for value in (rung_token_positions or []) if int(value) > 0})
        if target_token_positions <= 0 or int(value) <= target_token_positions
    ]
    next_rung_index = 0
    rung_checkpoints: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    model.train()
    epoch = 0
    fallback_step_limit = max(1, configured_epochs * batches_per_epoch)
    if target_token_positions > 0:
        step_limit = max(
            configured_step_limit,
            target_derived_step_limit + target_step_limit_safety_steps,
            fallback_step_limit,
        )
    else:
        step_limit = configured_step_limit or fallback_step_limit
    stop_reason = "step_limit_reached"
    while optimizer_steps < step_limit:
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), batch_size):
            if optimizer_steps >= step_limit:
                break
            indices = order[start : start + batch_size]
            batch_token_positions = sum(source_nonpad_by_row[index] + target_nonpad_by_row[index] for index in indices)
            batch_indices = mx.array(indices, dtype=mx.int32)
            src = source_matrix[batch_indices]
            tgt = target_matrix[batch_indices]
            token_weights = weight_matrix[batch_indices]
            plan_targets = plan_target_vector[batch_indices]
            plan_weights = plan_weight_vector[batch_indices]
            slot_targets = slot_target_matrix[batch_indices]
            slot_weights = slot_weight_matrix[batch_indices]
            if source_contrastive_weight > 0.0 and (semantic_plan_active or semantic_slot_active) and loss_and_grad_semantic_aux_contrast is not None:
                shifted_indices = indices[1:] + indices[:1]
                mismatched_src = source_matrix[mx.array(shifted_indices, dtype=mx.int32)]
                effective_contrastive_weight = source_contrastive_weight if len(indices) > 1 else 0.0
                loss, grads = loss_and_grad_semantic_aux_contrast(
                    model,
                    src,
                    mismatched_src,
                    tgt,
                    pad_id,
                    token_weights,
                    effective_contrastive_weight,
                    source_contrastive_margin,
                    source_contrastive_prefix_tokens,
                    source_contrastive_span_mode,
                    source_contrastive_body_start_id,
                    plan_targets,
                    plan_weights,
                    semantic_plan_weight,
                    slot_targets,
                    slot_weights,
                    semantic_slot_weight,
                    mx,
                    nn,
                    slot_role_class_ids,
                )
            elif (semantic_plan_active or semantic_slot_active) and loss_and_grad_semantic_aux is not None:
                loss, grads = loss_and_grad_semantic_aux(
                    model,
                    src,
                    tgt,
                    pad_id,
                    token_weights,
                    plan_targets,
                    plan_weights,
                    semantic_plan_weight,
                    slot_targets,
                    slot_weights,
                    semantic_slot_weight,
                    mx,
                    nn,
                    slot_role_class_ids,
                )
            elif source_contrastive_weight > 0.0 and loss_and_grad_contrast is not None:
                shifted_indices = indices[1:] + indices[:1]
                mismatched_src = source_matrix[mx.array(shifted_indices, dtype=mx.int32)]
                effective_contrastive_weight = source_contrastive_weight if len(indices) > 1 else 0.0
                loss, grads = loss_and_grad_contrast(
                    model,
                    src,
                    mismatched_src,
                    tgt,
                    pad_id,
                    token_weights,
                    effective_contrastive_weight,
                    source_contrastive_margin,
                    source_contrastive_prefix_tokens,
                    source_contrastive_span_mode,
                    source_contrastive_body_start_id,
                    mx,
                    nn,
                )
            else:
                loss, grads = loss_and_grad_plain(model, src, tgt, pad_id, token_weights, mx, nn)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            losses.append(round(float(loss.item()), 6))
            optimizer_steps += 1
            optimizer_windows_consumed += len(indices)
            optimizer_token_positions += int(batch_token_positions)
            while next_rung_index < len(rung_milestones) and optimizer_token_positions >= rung_milestones[next_rung_index]:
                milestone = int(rung_milestones[next_rung_index])
                rung_path = checkpoint_dir / f"strict_generator_mlx_{checkpoint_slug}_rung_{milestone}.npz"
                model.save_weights(str(rung_path))
                mx.eval(model.parameters())
                rung_elapsed_ms = int((time.perf_counter() - training_started) * 1000)
                rung_lm = evaluate_loss_mlx(
                    model,
                    eval_source_rows,
                    eval_target_rows,
                    batch_size=batch_size,
                    pad_id=pad_id,
                    mx=mx,
                    nn=nn,
                )
                rung_contrast = evaluate_source_contrast_mlx(
                    model,
                    eval_source_rows,
                    eval_target_rows,
                    batch_size=batch_size,
                    pad_id=pad_id,
                    prefix_token_count=source_contrastive_prefix_tokens,
                    span_mode=source_contrastive_span_mode,
                    body_start_token_id=source_contrastive_body_start_id,
                    mx=mx,
                    nn=nn,
                )
                rung_plan = evaluate_semantic_plan_mlx(
                    model,
                    eval_source_rows,
                    eval_plan_target_rows,
                    batch_size=batch_size,
                    pad_id=pad_id,
                    mx=mx,
                    nn=nn,
                )
                rung_slot = evaluate_semantic_slot_mlx(
                    model,
                    eval_source_rows,
                    eval_slot_target_rows,
                    batch_size=batch_size,
                    pad_id=pad_id,
                    enabled=semantic_slot_active,
                    mx=mx,
                    nn=nn,
                    slot_role_class_ids=slot_role_class_ids,
                )
                rung_seconds = max(rung_elapsed_ms / 1000.0, 1e-9)
                rung_record = {
                    "policy": "project_theseus_strict_generator_mlx_rung_checkpoint_v1",
                    "milestone_token_positions": milestone,
                    "optimizer_token_positions_consumed": optimizer_token_positions,
                    "optimizer_step_count": optimizer_steps,
                    "optimizer_windows_consumed": optimizer_windows_consumed,
                    "training_wall_time_ms": rung_elapsed_ms,
                    "training_tokens_per_second_so_far": round(optimizer_token_positions / rung_seconds, 3),
                    "checkpoint": rel(rung_path),
                    "checkpoint_sha256": stable_hash_file(rung_path),
                    "vocab": rel(vocab_path),
                    "vocab_sha256": stable_hash_file(vocab_path),
                    "heldout_lm_loss": rung_lm.get("loss"),
                    "heldout_lm_perplexity": rung_lm.get("perplexity"),
                    "heldout_source_loss_gap": rung_contrast.get("loss_gap"),
                    "heldout_plan_loss": rung_plan.get("loss"),
                    "heldout_plan_accuracy": rung_plan.get("accuracy"),
                    "heldout_slot_loss": rung_slot.get("loss"),
                    "heldout_slot_accuracy": rung_slot.get("accuracy"),
                    "uses_eval_tests_or_solutions": False,
                    "uses_public_data": False,
                    "public_training_rows": 0,
                    "external_inference_calls": 0,
                    "fallback_template_router_tool_credit_count": 0,
                    "score_semantics": (
                        "Replayable intermediate MLX checkpoint selected only by token-position milestone "
                        "and private heldout source/target losses. It emits no candidates and does not use "
                        "public benchmarks, eval tests, eval solutions, teacher output, tools, templates, "
                        "or fallback returns."
                    ),
                }
                rung_checkpoints.append(rung_record)
                print(
                    "[strict-generator-mlx] "
                    f"rung={milestone} tokens={optimizer_token_positions} "
                    f"checkpoint={rel(rung_path)} heldout_loss={rung_lm.get('loss')}",
                    flush=True,
                )
                next_rung_index += 1
            if optimizer_steps == 1 or optimizer_steps % 50 == 0:
                print(
                    "[strict-generator-mlx] "
                    f"budget={budget_id} epoch={epoch + 1} "
                    f"step={optimizer_steps}/{step_limit} "
                    f"tokens={optimizer_token_positions} loss={float(loss.item()):.6f}",
                    flush=True,
                )
            if target_token_positions > 0 and optimizer_token_positions >= target_token_positions:
                stop_reason = "target_token_positions_reached"
                break
        if stop_reason == "target_token_positions_reached":
            break
        epoch += 1
    training_wall_ms = int((time.perf_counter() - training_started) * 1000)
    heldout_after = evaluate_loss_mlx(model, eval_source_rows, eval_target_rows, batch_size=batch_size, pad_id=pad_id, mx=mx, nn=nn)
    contrast_after = evaluate_source_contrast_mlx(
        model,
        eval_source_rows,
        eval_target_rows,
        batch_size=batch_size,
        pad_id=pad_id,
        prefix_token_count=source_contrastive_prefix_tokens,
        span_mode=source_contrastive_span_mode,
        body_start_token_id=source_contrastive_body_start_id,
        mx=mx,
        nn=nn,
    )
    semantic_plan_after = evaluate_semantic_plan_mlx(
        model,
        eval_source_rows,
        eval_plan_target_rows,
        batch_size=batch_size,
        pad_id=pad_id,
        mx=mx,
        nn=nn,
    )
    semantic_slot_after = evaluate_semantic_slot_mlx(
        model,
        eval_source_rows,
        eval_slot_target_rows,
        batch_size=batch_size,
        pad_id=pad_id,
        enabled=semantic_slot_active,
        mx=mx,
        nn=nn,
        slot_role_class_ids=slot_role_class_ids,
    )
    update_summary = parameter_update_summary(model, before, mlx_utils, mx)
    model.save_weights(str(checkpoint_path))
    write_json(vocab_path, vocab_payload)
    mx.eval(model.parameters())
    source_nonpad = sum(source_nonpad_by_row)
    target_nonpad = sum(target_nonpad_by_row)
    seconds = max(training_wall_ms / 1000.0, 1e-9)
    return {
        "id": budget_id,
        "active": True,
        "backend": "mlx_high_level_transformer",
        "device": str(mx.default_device()),
        "checkpoint": rel(checkpoint_path),
        "checkpoint_sha256": stable_hash_file(checkpoint_path),
        "vocab": rel(vocab_path),
        "vocab_sha256": stable_hash_file(vocab_path),
        "dims": dims,
        "source_vocab_size": len(source_vocab),
        "target_vocab_size": len(target_vocab),
        "max_source": max_source,
        "max_target": max_target,
        "source_vocab_sha256": stable_hash(json.dumps(source_vocab, sort_keys=True)),
        "target_vocab_sha256": stable_hash(json.dumps(target_vocab, sort_keys=True)),
        "target_mode": target_mode,
        "source_vocab_extension": full_state_source_vocab_extension_summary(source_vocab_extension_texts),
        "target_vocab_extension": full_state_target_vocab_extension_summary(target_vocab_extension_bodies),
        "row_summary": dict_or_empty(rows.get("summary")),
        "loss_weighting": loss_weight_summary,
        "semantic_plan_auxiliary": {
            **semantic_plan_cfg,
            "active": semantic_plan_active,
            "vocab_extension": semantic_plan_vocab_summary,
            "train_targets": semantic_plan_target_summary,
            "eval_targets": semantic_plan_eval_target_summary,
            "class_balance": semantic_plan_balance_summary,
            "heldout_plan_loss_before": semantic_plan_before.get("loss"),
            "heldout_plan_loss_after": semantic_plan_after.get("loss"),
            "heldout_plan_accuracy_before": semantic_plan_before.get("accuracy"),
            "heldout_plan_accuracy_after": semantic_plan_after.get("accuracy"),
            "heldout_plan_improved": (
                semantic_plan_before.get("loss") is not None
                and semantic_plan_after.get("loss") is not None
                and float(semantic_plan_after["loss"]) < float(semantic_plan_before["loss"])
            ),
            "score_semantics": (
                "Auxiliary source-only semantic plan classification over admitted private/licensed "
                "corpus bodies. Labels are AST/body-derived from training rows; generation inputs "
                "still see only prompt/signature source text and allowed visible runtime context. "
                "The plan head emits no candidates and grants no learned-generation credit."
            ),
        },
        "semantic_slot_auxiliary": {
            **semantic_slot_cfg,
            "active": semantic_slot_active,
            "vocab_extension": semantic_slot_vocab_summary,
            "class_summary": semantic_slot_class_summary,
            "class_balance": semantic_slot_balance_summary,
            "roles": semantic_slot_role_summary(),
            "train_targets": semantic_slot_target_summary,
            "eval_targets": semantic_slot_eval_target_summary,
            "heldout_slot_loss_before": semantic_slot_before.get("loss"),
            "heldout_slot_loss_after": semantic_slot_after.get("loss"),
            "heldout_slot_accuracy_before": semantic_slot_before.get("accuracy"),
            "heldout_slot_accuracy_after": semantic_slot_after.get("accuracy"),
            "heldout_role_accuracy_before": semantic_slot_before.get("role_accuracy"),
            "heldout_role_accuracy_after": semantic_slot_after.get("role_accuracy"),
            "heldout_slot_improved": (
                semantic_slot_before.get("loss") is not None
                and semantic_slot_after.get("loss") is not None
                and float(semantic_slot_after["loss"]) < float(semantic_slot_before["loss"])
            ),
            "score_semantics": (
                "Auxiliary source-only semantic slot classification over admitted private/licensed "
                "corpus target prefixes. Labels are AST/body-derived from training rows; the head "
                "does not render code, inspect tests/solutions, use public data, or grant "
                "learned-generation promotion credit."
            ),
        },
        "source_contrastive_loss": {
            "enabled": source_contrastive_weight > 0.0,
            "policy": "private_source_mismatch_margin_loss_v1",
            "weight": source_contrastive_weight,
            "margin": source_contrastive_margin,
            "prefix_token_count": source_contrastive_prefix_tokens,
            "span_mode": source_contrastive_span_mode,
            "body_start_token_id": source_contrastive_body_start_id,
            "heldout_matched_loss_before": contrast_before.get("matched_loss"),
            "heldout_mismatched_loss_before": contrast_before.get("mismatched_loss"),
            "heldout_source_loss_gap_before": contrast_before.get("loss_gap"),
            "heldout_matched_loss_after": contrast_after.get("matched_loss"),
            "heldout_mismatched_loss_after": contrast_after.get("mismatched_loss"),
            "heldout_source_loss_gap_after": contrast_after.get("loss_gap"),
            "loss_gap_improved": (
                contrast_before.get("loss_gap") is not None
                and contrast_after.get("loss_gap") is not None
                and float(contrast_after["loss_gap"]) > float(contrast_before["loss_gap"])
            ),
            "score_semantics": (
                "Uses only admitted private source/target rows. It compares correct-prompt body loss "
                "against deterministic in-batch mismatched-prompt body loss to make source conditioning "
                "measurable; it does not inspect eval tests, eval solutions, public benchmarks, answer "
                "metadata, or teacher output."
            ),
        },
        "parameter_count": update_summary["parameter_count"],
        "trainable_parameter_count": update_summary["parameter_count"],
        "parameter_update_fraction": update_summary["parameter_update_fraction"],
        "parameter_tensor_update_fraction": update_summary["parameter_tensor_update_fraction"],
        "optimizer_step_count": optimizer_steps,
        "optimizer_token_positions_consumed": optimizer_token_positions,
        "optimizer_windows_consumed": optimizer_windows_consumed,
        "rung_checkpoints": rung_checkpoints,
        "checkpoint_selection": {
            "policy": "final_checkpoint_not_assumed_best_v1",
            "rung_token_positions_requested": list(rung_milestones),
            "rung_checkpoint_count": len(rung_checkpoints),
            "selection_basis": (
                "Intermediate checkpoints are replay artifacts for later private decode/verifier comparison. "
                "The final checkpoint is not treated as best merely because LM loss is lower."
            ),
        },
        "training_plan": {
            "policy": "repeat_shuffled_batches_until_token_target_or_step_limit_v1",
            "configured_epochs": configured_epochs,
            "configured_step_limit": configured_step_limit,
            "fallback_step_limit": fallback_step_limit,
            "target_derived_step_limit": target_derived_step_limit,
            "target_step_limit_safety_steps": target_step_limit_safety_steps,
            "effective_step_limit": step_limit,
            "rung_token_positions_requested": list(rung_milestones),
            "estimated_token_positions_per_step": estimated_token_positions_per_step,
            "epoch_token_positions": epoch_token_positions,
            "target_token_positions": target_token_positions,
            "stop_reason": stop_reason,
            "score_semantics": (
                "The MLX trainer repeats shuffled private/licensed batches until the configured "
                "target token-position budget is reached, or until the explicit step limit fires. "
                "Consumed token positions are counted from non-padding source and target tokens in "
                "the actual batches updated by the optimizer."
            ),
        },
        "training_wall_time_ms": training_wall_ms,
        "training_batch_materialization": "preloaded_mx_arrays_indexed_by_shuffled_batch_v1",
        "batch_size": batch_size,
        "training_tokens_per_second": round(optimizer_token_positions / seconds, 3),
        "optimizer_steps_per_second": round(optimizer_steps / seconds, 6),
        "source_train_token_count": source_nonpad,
        "target_train_token_count": target_nonpad,
        "eval_source_token_count": sum(1 for row in eval_source_rows for value in row if int(value) != 0),
        "eval_target_token_count": sum(1 for row in eval_target_rows for value in row if int(value) != pad_id),
        "heldout_lm_loss_before": heldout_before.get("loss"),
        "heldout_lm_loss_after": heldout_after.get("loss"),
        "heldout_lm_perplexity_before": heldout_before.get("perplexity"),
        "heldout_lm_perplexity_after": heldout_after.get("perplexity"),
        "heldout_lm_loss_curve": [heldout_before.get("loss"), *losses, heldout_after.get("loss")],
        "heldout_lm_improved": bool(
            heldout_before.get("loss") is not None
            and heldout_after.get("loss") is not None
            and float(heldout_after["loss"]) < float(heldout_before["loss"])
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "open_or_pretrained_model_weights_used": False,
        "fallback_template_router_tool_credit_count": 0,
        "score_semantics": "MLX native training proof only; no candidate generation or promotion claim.",
    }


def loss_fn_mlx(model: Any, src: Any, tgt: Any, pad_id: int, mx: Any, nn: Any) -> Any:
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    logits = model(src, tgt_in)
    losses = nn.losses.cross_entropy(logits, tgt_out, reduction="none")
    valid = (tgt_out != pad_id).astype(mx.float32)
    return mx.sum(losses * valid) / mx.maximum(mx.sum(valid), mx.array(1.0, dtype=mx.float32))


def weighted_loss_fn_mlx(model: Any, src: Any, tgt: Any, pad_id: int, token_weights: Any, mx: Any, nn: Any) -> Any:
    return weighted_loss_with_prefix_mlx(model, src, tgt, pad_id, token_weights, mx, nn, prefix_token_count=0)


def weighted_loss_with_prefix_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    token_weights: Any,
    mx: Any,
    nn: Any,
    *,
    prefix_token_count: int = 0,
    span_mode: str = "prefix",
    body_start_token_id: int = -1,
) -> Any:
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    logits = model(src, tgt_in)
    losses = nn.losses.cross_entropy(logits, tgt_out, reduction="none")
    valid = (tgt_out != pad_id).astype(mx.float32)
    if str(span_mode or "prefix") == "after_body_start" and int(body_start_token_id) >= 0:
        valid = valid * body_start_span_mask_mlx(tgt, tgt_out, int(body_start_token_id), mx)
    elif int(prefix_token_count or 0) > 0:
        positions = mx.arange(tgt_out.shape[1])[None, :]
        valid = valid * (positions < int(prefix_token_count)).astype(mx.float32)
    weights = token_weights[:, 1:]
    weighted_valid = valid * weights
    return mx.sum(losses * weighted_valid) / mx.maximum(mx.sum(weighted_valid), mx.array(1.0, dtype=mx.float32))


def body_start_span_mask_mlx(tgt: Any, tgt_out: Any, body_start_token_id: int, mx: Any) -> Any:
    matches = (tgt == int(body_start_token_id)).astype(mx.int32)
    has_body_start = mx.max(matches, axis=1).astype(mx.float32)
    start_positions = mx.argmax(matches, axis=1)
    positions = mx.arange(tgt_out.shape[1])[None, :]
    return (positions >= start_positions[:, None]).astype(mx.float32) * has_body_start[:, None]


def source_contrastive_weighted_loss_fn_mlx(
    model: Any,
    src: Any,
    mismatched_src: Any,
    tgt: Any,
    pad_id: int,
    token_weights: Any,
    source_contrastive_weight: float,
    source_contrastive_margin: float,
    source_contrastive_prefix_tokens: int,
    source_contrastive_span_mode: str,
    source_contrastive_body_start_id: int,
    mx: Any,
    nn: Any,
) -> Any:
    matched = weighted_loss_fn_mlx(model, src, tgt, pad_id, token_weights, mx, nn)
    matched_for_contrast = weighted_loss_with_prefix_mlx(
        model,
        src,
        tgt,
        pad_id,
        token_weights,
        mx,
        nn,
        prefix_token_count=int(source_contrastive_prefix_tokens or 0),
        span_mode=source_contrastive_span_mode,
        body_start_token_id=int(source_contrastive_body_start_id),
    )
    mismatched = weighted_loss_with_prefix_mlx(
        model,
        mismatched_src,
        tgt,
        pad_id,
        token_weights,
        mx,
        nn,
        prefix_token_count=int(source_contrastive_prefix_tokens or 0),
        span_mode=source_contrastive_span_mode,
        body_start_token_id=int(source_contrastive_body_start_id),
    )
    contrast = mx.maximum(
        mx.array(0.0, dtype=mx.float32),
        mx.array(float(source_contrastive_margin), dtype=mx.float32) + matched_for_contrast - mismatched,
    )
    return matched + (float(source_contrastive_weight) * contrast)


def semantic_plan_weighted_loss_fn_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    token_weights: Any,
    plan_targets: Any,
    plan_sample_weights: Any,
    semantic_plan_weight: float,
    mx: Any,
    nn: Any,
    plan_class_ids: Any | None = None,
) -> Any:
    token_loss = weighted_loss_fn_mlx(model, src, tgt, pad_id, token_weights, mx, nn)
    plan_loss = semantic_plan_loss_mlx(
        model,
        src,
        plan_targets,
        pad_id,
        mx,
        nn,
        plan_sample_weights=plan_sample_weights,
        plan_class_ids=plan_class_ids,
    )
    return token_loss + (float(semantic_plan_weight) * plan_loss)


def semantic_plan_source_contrastive_weighted_loss_fn_mlx(
    model: Any,
    src: Any,
    mismatched_src: Any,
    tgt: Any,
    pad_id: int,
    token_weights: Any,
    source_contrastive_weight: float,
    source_contrastive_margin: float,
    source_contrastive_prefix_tokens: int,
    source_contrastive_span_mode: str,
    source_contrastive_body_start_id: int,
    plan_targets: Any,
    plan_sample_weights: Any,
    semantic_plan_weight: float,
    mx: Any,
    nn: Any,
    plan_class_ids: Any | None = None,
) -> Any:
    base = source_contrastive_weighted_loss_fn_mlx(
        model,
        src,
        mismatched_src,
        tgt,
        pad_id,
        token_weights,
        source_contrastive_weight,
        source_contrastive_margin,
        source_contrastive_prefix_tokens,
        source_contrastive_span_mode,
        source_contrastive_body_start_id,
        mx,
        nn,
    )
    plan_loss = semantic_plan_loss_mlx(
        model,
        src,
        plan_targets,
        pad_id,
        mx,
        nn,
        plan_sample_weights=plan_sample_weights,
        plan_class_ids=plan_class_ids,
    )
    return base + (float(semantic_plan_weight) * plan_loss)


def semantic_aux_weighted_loss_fn_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    token_weights: Any,
    plan_targets: Any,
    plan_sample_weights: Any,
    semantic_plan_weight: float,
    slot_targets: Any,
    slot_sample_weights: Any,
    semantic_slot_weight: float,
    mx: Any,
    nn: Any,
    slot_role_class_ids: list[Any] | None = None,
) -> Any:
    loss = weighted_loss_fn_mlx(model, src, tgt, pad_id, token_weights, mx, nn)
    if float(semantic_plan_weight or 0.0) > 0.0 and hasattr(model, "semantic_plan_logits"):
        loss = loss + (
            float(semantic_plan_weight)
            * semantic_plan_loss_mlx(
                model,
                src,
                plan_targets,
                pad_id,
                mx,
                nn,
                plan_sample_weights=plan_sample_weights,
            )
        )
    if float(semantic_slot_weight or 0.0) > 0.0 and hasattr(model, "semantic_slot_logits"):
        loss = loss + (
            float(semantic_slot_weight)
            * semantic_slot_loss_mlx(
                model,
                src,
                slot_targets,
                pad_id,
                mx,
                nn,
                slot_sample_weights=slot_sample_weights,
                slot_role_class_ids=slot_role_class_ids,
            )
        )
    return loss


def semantic_aux_source_contrastive_weighted_loss_fn_mlx(
    model: Any,
    src: Any,
    mismatched_src: Any,
    tgt: Any,
    pad_id: int,
    token_weights: Any,
    source_contrastive_weight: float,
    source_contrastive_margin: float,
    source_contrastive_prefix_tokens: int,
    source_contrastive_span_mode: str,
    source_contrastive_body_start_id: int,
    plan_targets: Any,
    plan_sample_weights: Any,
    semantic_plan_weight: float,
    slot_targets: Any,
    slot_sample_weights: Any,
    semantic_slot_weight: float,
    mx: Any,
    nn: Any,
    slot_role_class_ids: list[Any] | None = None,
) -> Any:
    loss = source_contrastive_weighted_loss_fn_mlx(
        model,
        src,
        mismatched_src,
        tgt,
        pad_id,
        token_weights,
        source_contrastive_weight,
        source_contrastive_margin,
        source_contrastive_prefix_tokens,
        source_contrastive_span_mode,
        source_contrastive_body_start_id,
        mx,
        nn,
    )
    if float(semantic_plan_weight or 0.0) > 0.0 and hasattr(model, "semantic_plan_logits"):
        loss = loss + (
            float(semantic_plan_weight)
            * semantic_plan_loss_mlx(
                model,
                src,
                plan_targets,
                pad_id,
                mx,
                nn,
                plan_sample_weights=plan_sample_weights,
            )
        )
    if float(semantic_slot_weight or 0.0) > 0.0 and hasattr(model, "semantic_slot_logits"):
        loss = loss + (
            float(semantic_slot_weight)
            * semantic_slot_loss_mlx(
                model,
                src,
                slot_targets,
                pad_id,
                mx,
                nn,
                slot_sample_weights=slot_sample_weights,
                slot_role_class_ids=slot_role_class_ids,
            )
        )
    return loss


def semantic_plan_loss_mlx(
    model: Any,
    src: Any,
    plan_targets: Any,
    pad_id: int,
    mx: Any,
    nn: Any,
    *,
    plan_sample_weights: Any | None = None,
    plan_class_ids: Any | None = None,
) -> Any:
    logits = model.semantic_plan_logits(src)
    class_ids = plan_class_ids
    if class_ids is not None and int(class_ids.shape[0]) > 0:
        logits = mx.take(logits, class_ids, axis=1)
        matches = (plan_targets[:, None] == class_ids[None, :])
        class_targets = mx.argmax(matches.astype(mx.int32), axis=1)
        valid = mx.max(matches.astype(mx.float32), axis=1)
        valid = valid * (plan_targets != pad_id).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, class_targets, reduction="none")
    else:
        losses = nn.losses.cross_entropy(logits, plan_targets, reduction="none")
        valid = (plan_targets != pad_id).astype(mx.float32)
    if plan_sample_weights is not None:
        valid = valid * plan_sample_weights
    return mx.sum(losses * valid) / mx.maximum(mx.sum(valid), mx.array(1.0, dtype=mx.float32))


def semantic_slot_loss_mlx(
    model: Any,
    src: Any,
    slot_targets: Any,
    pad_id: int,
    mx: Any,
    nn: Any,
    *,
    slot_sample_weights: Any | None = None,
    slot_role_class_ids: list[Any] | None = None,
) -> Any:
    logits = model.semantic_slot_logits(src)
    total = mx.array(0.0, dtype=mx.float32)
    valid_total = mx.array(0.0, dtype=mx.float32)
    for role_index in range(len(SEMANTIC_SLOT_ROLES)):
        targets = slot_targets[:, role_index]
        role_logits = logits[:, role_index, :]
        class_ids = None
        if slot_role_class_ids is not None and role_index < len(slot_role_class_ids):
            class_ids = slot_role_class_ids[role_index]
        if class_ids is not None and int(class_ids.shape[0]) > 0:
            role_logits = mx.take(role_logits, class_ids, axis=1)
            matches = targets[:, None] == class_ids[None, :]
            class_targets = mx.argmax(matches.astype(mx.int32), axis=1)
            valid = mx.max(matches.astype(mx.float32), axis=1)
            valid = valid * (targets != pad_id).astype(mx.float32)
            losses = nn.losses.cross_entropy(role_logits, class_targets, reduction="none")
        else:
            losses = nn.losses.cross_entropy(role_logits, targets, reduction="none")
            valid = (targets != pad_id).astype(mx.float32)
        if slot_sample_weights is not None:
            valid = valid * slot_sample_weights[:, role_index]
        total = total + mx.sum(losses * valid)
        valid_total = valid_total + mx.sum(valid)
    return total / mx.maximum(valid_total, mx.array(1.0, dtype=mx.float32))


def training_examples_for_encoded_rows(staged: dict[str, Any], rows: dict[str, Any]) -> list[dict[str, Any]]:
    examples = list(staged.get("examples") or [])
    summary = dict_or_empty(rows.get("summary"))
    eval_count = int(summary.get("eval_example_count") or 0)
    train_count = int(summary.get("train_example_count") or 0)
    train_examples = examples[eval_count : eval_count + train_count]
    if not train_examples and examples:
        train_examples = examples
    return [row if isinstance(row, dict) else {} for row in train_examples]


def eval_examples_for_encoded_rows(staged: dict[str, Any], rows: dict[str, Any]) -> list[dict[str, Any]]:
    examples = list(staged.get("examples") or [])
    summary = dict_or_empty(rows.get("summary"))
    eval_count = int(summary.get("eval_example_count") or 0)
    eval_examples = examples[:eval_count]
    return [row if isinstance(row, dict) else {} for row in eval_examples]


def semantic_plan_auxiliary_config(budget: dict[str, Any], matched_budget: dict[str, Any]) -> dict[str, Any]:
    cfg = dict_or_empty(budget.get("semantic_plan_auxiliary_loss") or matched_budget.get("semantic_plan_auxiliary_loss"))
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "policy": str(cfg.get("policy") or "private_body_derived_source_plan_auxiliary_v1"),
        "weight": float(cfg.get("weight") or 0.0),
        "class_balance_enabled": bool(cfg.get("class_balance_enabled", True)),
        "class_balance_policy": str(cfg.get("class_balance_policy") or "inverse_sqrt_plan_frequency_v1"),
        "class_balance_min_weight": float(cfg.get("class_balance_min_weight") or 0.35),
        "class_balance_max_weight": float(cfg.get("class_balance_max_weight") or 4.0),
        "label_source": "admitted_private_or_licensed_training_body_ast_only",
        "served_at_runtime": False,
        "candidate_generation_credit": 0,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
    }


def semantic_slot_auxiliary_config(budget: dict[str, Any], matched_budget: dict[str, Any]) -> dict[str, Any]:
    cfg = dict_or_empty(budget.get("semantic_slot_auxiliary_loss") or matched_budget.get("semantic_slot_auxiliary_loss"))
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "policy": str(cfg.get("policy") or "private_body_derived_source_slot_auxiliary_v1"),
        "weight": float(cfg.get("weight") or 0.0),
        "class_balance_enabled": bool(cfg.get("class_balance_enabled", True)),
        "class_balance_policy": str(cfg.get("class_balance_policy") or "inverse_sqrt_slot_frequency_by_role_v1"),
        "class_balance_min_weight": float(cfg.get("class_balance_min_weight") or 0.25),
        "class_balance_max_weight": float(cfg.get("class_balance_max_weight") or 5.0),
        "expression_role_weight": float(cfg.get("expression_role_weight") or 1.75),
        "role_count": len(SEMANTIC_SLOT_ROLES),
        "label_source": "admitted_private_or_licensed_training_body_ast_only",
        "served_at_runtime": False,
        "candidate_generation_credit": 0,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
    }


def semantic_slot_role_summary() -> list[dict[str, Any]]:
    return [
        {"index": index, "role": role, "prefixes": list(prefixes)}
        for index, (role, prefixes) in enumerate(SEMANTIC_SLOT_ROLES)
    ]


def semantic_plan_token_for_body(body: str) -> str:
    return f"SLOT:PLAN_{semantic_plan_from_body(str(body or ''))}"


def extend_target_vocab_with_semantic_plan_tokens(
    target_vocab: dict[str, int],
    bodies: list[str],
    *,
    enabled: bool,
) -> dict[str, Any]:
    before_size = len(target_vocab)
    if not enabled:
        return {
            "enabled": False,
            "policy": "not_enabled",
            "target_vocab_size_before": before_size,
            "target_vocab_size_after": before_size,
            "plan_token_count": 0,
            "added_plan_token_count": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    tokens = sorted({semantic_plan_token_for_body(body) for body in bodies if str(body or "").strip()})
    added = 0
    for token in tokens:
        if token not in target_vocab:
            target_vocab[token] = len(target_vocab)
            added += 1
    return {
        "enabled": True,
        "policy": "append_private_body_semantic_plan_tokens_to_target_vocab_v1",
        "target_vocab_size_before": before_size,
        "target_vocab_size_after": len(target_vocab),
        "plan_token_count": len(tokens),
        "added_plan_token_count": added,
        "plan_tokens": tokens[:128],
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "score_semantics": (
            "Plan tokens are auxiliary classifier labels only. They are derived from admitted private/licensed "
            "corpus bodies and are not valid body-token decoder outputs or learned-generation evidence."
        ),
    }


def extend_target_vocab_with_semantic_slot_tokens(
    target_vocab: dict[str, int],
    bodies: list[str],
    *,
    target_mode: str,
    enabled: bool,
) -> dict[str, Any]:
    before_size = len(target_vocab)
    if not enabled:
        return {
            "enabled": False,
            "policy": "not_enabled",
            "target_vocab_size_before": before_size,
            "target_vocab_size_after": before_size,
            "slot_token_count": 0,
            "added_slot_token_count": 0,
            "missing_role_prefixes": [],
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }

    role_tokens: dict[str, set[str]] = {role: set() for role, _prefixes in SEMANTIC_SLOT_ROLES}
    for body in bodies:
        for token in semantic_slot_prefix_tokens_for_body(str(body or ""), target_mode=target_mode):
            for role, prefixes in SEMANTIC_SLOT_ROLES:
                if any(token.startswith(prefix) for prefix in prefixes):
                    role_tokens.setdefault(role, set()).add(token)
                    break

    added = 0
    ordered_tokens = [
        token
        for role, _prefixes in SEMANTIC_SLOT_ROLES
        for token in sorted(role_tokens.get(role, set()))
    ]
    for token in ordered_tokens:
        if token not in target_vocab:
            target_vocab[token] = len(target_vocab)
            added += 1

    missing_role_prefixes = [
        {"role": role, "prefixes": list(prefixes)}
        for role, prefixes in SEMANTIC_SLOT_ROLES
        if not role_tokens.get(role)
    ]
    return {
        "enabled": True,
        "policy": "append_private_body_semantic_slot_tokens_to_target_vocab_v1",
        "target_vocab_size_before": before_size,
        "target_vocab_size_after": len(target_vocab),
        "slot_token_count": len(ordered_tokens),
        "added_slot_token_count": added,
        "role_token_counts": {
            role: len(role_tokens.get(role, set()))
            for role, _prefixes in SEMANTIC_SLOT_ROLES
        },
        "missing_role_prefixes": missing_role_prefixes,
        "slot_tokens": ordered_tokens[:192],
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "score_semantics": (
            "Slot tokens are auxiliary classifier labels derived only from admitted "
            "private/licensed target bodies already staged for this checkpoint. They "
            "are appended to keep the slot-head ABI closed under train/eval labels. "
            "They are not rendered into code, do not inspect tests/solutions/public "
            "payloads, and grant no learned-generation promotion credit."
        ),
    }


def semantic_plan_target_ids(
    examples: list[dict[str, Any]],
    *,
    target_vocab: dict[str, int],
    pad_id: int,
    enabled: bool,
) -> tuple[list[int], dict[str, Any]]:
    if not enabled:
        return [pad_id for _row in examples], {
            "enabled": False,
            "rows": len(examples),
            "active_target_count": 0,
            "missing_plan_token_count": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    ids: list[int] = []
    counts: dict[str, int] = {}
    missing = 0
    for row in examples:
        token = semantic_plan_token_for_body(str(row.get("body") or ""))
        counts[token] = counts.get(token, 0) + 1
        token_id = target_vocab.get(token)
        if token_id is None:
            missing += 1
            ids.append(pad_id)
        else:
            ids.append(int(token_id))
    active = sum(1 for value in ids if int(value) != int(pad_id))
    return ids, {
        "enabled": True,
        "rows": len(examples),
        "active_target_count": active,
        "missing_plan_token_count": missing,
        "unique_plan_count": len(counts),
        "top_plans": sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:24],
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
    }


def semantic_slot_target_ids(
    examples: list[dict[str, Any]],
    *,
    target_vocab: dict[str, int],
    pad_id: int,
    target_mode: str,
    enabled: bool,
) -> tuple[list[list[int]], dict[str, Any]]:
    if not enabled:
        return [[pad_id for _role in SEMANTIC_SLOT_ROLES] for _row in examples], {
            "enabled": False,
            "rows": len(examples),
            "role_count": len(SEMANTIC_SLOT_ROLES),
            "active_target_count": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    rows: list[list[int]] = []
    role_counts: dict[str, int] = {role: 0 for role, _prefixes in SEMANTIC_SLOT_ROLES}
    missing = 0
    token_counts: dict[str, int] = {}
    for row in examples:
        body = str(row.get("body") or "")
        prefix_tokens = semantic_slot_prefix_tokens_for_body(body, target_mode=target_mode)
        encoded: list[int] = []
        for role, prefixes in SEMANTIC_SLOT_ROLES:
            token = next((tok for tok in prefix_tokens if any(tok.startswith(prefix) for prefix in prefixes)), "")
            if not token:
                encoded.append(pad_id)
                continue
            token_id = target_vocab.get(token)
            if token_id is None:
                missing += 1
                encoded.append(pad_id)
                continue
            encoded.append(int(token_id))
            role_counts[role] = role_counts.get(role, 0) + 1
            token_counts[token] = token_counts.get(token, 0) + 1
        rows.append(encoded)
    active = sum(1 for row in rows for value in row if int(value) != int(pad_id))
    return rows, {
        "enabled": True,
        "rows": len(examples),
        "role_count": len(SEMANTIC_SLOT_ROLES),
        "active_target_count": active,
        "missing_slot_token_count": missing,
        "role_active_counts": dict(sorted(role_counts.items())),
        "top_slot_targets": sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))[:32],
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "score_semantics": (
            "Slot targets are extracted from admitted private/licensed target-body prefix tokens "
            "for supervised source-conditioned classification. They are not rendered into code, "
            "do not inspect eval tests/solutions, and do not grant learned-generation credit."
        ),
    }


def semantic_slot_prefix_tokens_for_body(body: str, *, target_mode: str) -> list[str]:
    tokens = target_tokens(str(body or ""), target_mode=target_mode)
    prefix: list[str] = []
    for token in tokens:
        if token == PLAN_BODY_START_TOKEN:
            break
        if str(token).startswith("SLOT:"):
            prefix.append(str(token))
    return prefix


def semantic_plan_sample_weights(
    plan_target_rows: list[int],
    *,
    target_summary: dict[str, Any],
    pad_id: int,
    config: dict[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    enabled = bool(config.get("enabled")) and bool(config.get("class_balance_enabled", True))
    if not enabled:
        return [1.0 for _row in plan_target_rows], {
            "enabled": False,
            "policy": "not_enabled",
            "rows": len(plan_target_rows),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    counts: dict[int, int] = {}
    for value in plan_target_rows:
        target_id = int(value)
        if target_id == int(pad_id):
            continue
        counts[target_id] = counts.get(target_id, 0) + 1
    active = sum(counts.values())
    if active <= 0 or not counts:
        return [1.0 for _row in plan_target_rows], {
            "enabled": False,
            "policy": "no_active_plan_targets",
            "rows": len(plan_target_rows),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    raw = {target_id: 1.0 / (count ** 0.5) for target_id, count in counts.items()}
    weighted_mean = sum(raw[target_id] * count for target_id, count in counts.items()) / max(1, active)
    min_weight = max(0.0, float(config.get("class_balance_min_weight") or 0.35))
    max_weight = max(min_weight, float(config.get("class_balance_max_weight") or 4.0))
    normalized = {
        target_id: min(max_weight, max(min_weight, raw[target_id] / max(weighted_mean, 1e-9)))
        for target_id in counts
    }
    weights = [float(normalized.get(int(value), 1.0)) if int(value) != int(pad_id) else 0.0 for value in plan_target_rows]
    weight_values = [weights[index] for index, value in enumerate(plan_target_rows) if int(value) != int(pad_id)]
    return weights, {
        "enabled": True,
        "policy": str(config.get("class_balance_policy") or "inverse_sqrt_plan_frequency_v1"),
        "rows": len(plan_target_rows),
        "active_target_count": active,
        "unique_plan_id_count": len(counts),
        "min_weight": round(min(weight_values), 6) if weight_values else None,
        "max_weight": round(max(weight_values), 6) if weight_values else None,
        "mean_weight": round(sum(weight_values) / max(1, len(weight_values)), 6) if weight_values else None,
        "top_plan_targets": dict_or_empty(target_summary).get("top_plans", [])[:24],
        "score_semantics": (
            "Plan class weights are derived only from admitted training-label frequency. "
            "They rebalance the auxiliary classifier loss; they do not alter candidate generation, "
            "inspect tests/solutions, use public data, or grant learned-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
    }


def semantic_slot_role_class_summary(target_vocab: dict[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, prefixes in SEMANTIC_SLOT_ROLES:
        class_ids = sorted(
            int(idx)
            for token, idx in target_vocab.items()
            if any(str(token).startswith(prefix) for prefix in prefixes)
        )
        rows.append(
            {
                "role": role,
                "prefixes": list(prefixes),
                "class_ids": class_ids,
                "class_count": len(class_ids),
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "uses_answer_metadata": False,
            }
        )
    return rows


def semantic_slot_sample_weights(
    slot_target_rows: list[list[int]],
    *,
    target_summary: dict[str, Any],
    pad_id: int,
    config: dict[str, Any],
) -> tuple[list[list[float]], dict[str, Any]]:
    width = len(SEMANTIC_SLOT_ROLES)
    enabled = bool(config.get("enabled")) and bool(config.get("class_balance_enabled", True))
    if not enabled:
        return [[1.0 for _role in range(width)] for _row in slot_target_rows], {
            "enabled": False,
            "policy": "not_enabled",
            "rows": len(slot_target_rows),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    role_counts: list[dict[int, int]] = [{} for _role in range(width)]
    for row in slot_target_rows:
        for role_index, value in enumerate(row[:width]):
            target_id = int(value)
            if target_id == int(pad_id):
                continue
            counts = role_counts[role_index]
            counts[target_id] = counts.get(target_id, 0) + 1
    min_weight = max(0.0, float(config.get("class_balance_min_weight") or 0.25))
    max_weight = max(min_weight, float(config.get("class_balance_max_weight") or 5.0))
    expression_role_weight = max(1.0, float(config.get("expression_role_weight") or 1.75))
    role_weight_maps: list[dict[int, float]] = []
    role_summaries: dict[str, Any] = {}
    all_weights: list[float] = []
    for role_index, (role, _prefixes) in enumerate(SEMANTIC_SLOT_ROLES):
        counts = role_counts[role_index]
        active = sum(counts.values())
        if active <= 0:
            role_weight_maps.append({})
            role_summaries[role] = {"active_target_count": 0, "unique_target_count": 0}
            continue
        raw = {target_id: 1.0 / (count ** 0.5) for target_id, count in counts.items()}
        weighted_mean = sum(raw[target_id] * count for target_id, count in counts.items()) / max(1, active)
        role_multiplier = expression_role_weight if role.startswith("expression_") else 1.0
        weights = {
            target_id: min(
                max_weight,
                max(min_weight, (raw[target_id] / max(weighted_mean, 1e-9)) * role_multiplier),
            )
            for target_id in counts
        }
        role_weight_maps.append(weights)
        values = [weights[target_id] for target_id in counts for _ in range(counts[target_id])]
        all_weights.extend(values)
        role_summaries[role] = {
            "active_target_count": active,
            "unique_target_count": len(counts),
            "min_weight": round(min(values), 6) if values else None,
            "max_weight": round(max(values), 6) if values else None,
            "mean_weight": round(sum(values) / max(1, len(values)), 6) if values else None,
            "expression_role_multiplier": expression_role_weight if role.startswith("expression_") else 1.0,
        }
    matrix: list[list[float]] = []
    for row in slot_target_rows:
        weights: list[float] = []
        for role_index, value in enumerate(row[:width]):
            target_id = int(value)
            if target_id == int(pad_id):
                weights.append(0.0)
            else:
                weights.append(float(role_weight_maps[role_index].get(target_id, 1.0)))
        if len(weights) < width:
            weights.extend([0.0] * (width - len(weights)))
        matrix.append(weights)
    return matrix, {
        "enabled": True,
        "policy": str(config.get("class_balance_policy") or "inverse_sqrt_slot_frequency_by_role_v1"),
        "rows": len(slot_target_rows),
        "active_target_count": sum(sum(counts.values()) for counts in role_counts),
        "role_count": width,
        "min_weight": round(min(all_weights), 6) if all_weights else None,
        "max_weight": round(max(all_weights), 6) if all_weights else None,
        "mean_weight": round(sum(all_weights) / max(1, len(all_weights)), 6) if all_weights else None,
        "expression_role_weight": expression_role_weight,
        "role_summaries": role_summaries,
        "top_slot_targets": dict_or_empty(target_summary).get("top_slot_targets", [])[:32],
        "score_semantics": (
            "Slot class weights are derived only from admitted private/licensed training-label "
            "frequency within each semantic slot role. Expression roles receive a bounded "
            "multiplier because they are the current private heldout bottleneck. These weights "
            "rebalance an auxiliary classifier only; they do not alter candidate generation, "
            "inspect tests/solutions, use public data, or grant learned-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
    }


def target_loss_weight_rows(
    target_rows: list[list[int]],
    *,
    train_examples: list[dict[str, Any]],
    target_vocab: dict[str, int],
    budget: dict[str, Any],
    matched_budget: dict[str, Any],
) -> tuple[list[list[float]], dict[str, Any]]:
    first_token_weight = float(budget.get("first_target_token_loss_weight") or matched_budget.get("first_target_token_loss_weight") or 1.0)
    return_token_weight = float(budget.get("return_token_loss_weight") or matched_budget.get("return_token_loss_weight") or 1.0)
    semantic_cfg = dict_or_empty(matched_budget.get("semantic_token_loss_weights"))
    semantic_cfg.update(dict_or_empty(budget.get("semantic_token_loss_weights")))
    semantic_enabled = bool(semantic_cfg.get("enabled", False))
    parameter_weight = float(semantic_cfg.get("parameter_identifier_weight") or 1.0)
    update_weight = float(semantic_cfg.get("update_call_name_weight") or 1.0)
    operator_weight = float(semantic_cfg.get("operator_token_weight") or 1.0)
    control_weight = float(semantic_cfg.get("control_token_weight") or 1.0)
    structural_enabled = bool(semantic_cfg.get("structural_trajectory_weights_enabled", False))
    line_boundary_weight = float(semantic_cfg.get("line_boundary_weight") or 1.15)
    loop_header_weight = float(semantic_cfg.get("loop_header_weight") or 1.35)
    update_line_weight = float(semantic_cfg.get("accumulator_update_line_weight") or 1.65)
    top_level_return_weight = float(semantic_cfg.get("top_level_return_weight") or 2.25)
    final_dedent_weight = float(semantic_cfg.get("final_dedent_weight") or 1.8)
    builtin_call_enabled = bool(semantic_cfg.get("builtin_call_trajectory_weights_enabled", False))
    builtin_call_name_weight = float(semantic_cfg.get("builtin_call_name_weight") or 1.6)
    builtin_call_argument_weight = float(semantic_cfg.get("builtin_call_argument_weight") or 1.2)
    builtin_call_punctuation_weight = float(semantic_cfg.get("builtin_call_punctuation_weight") or 1.35)
    returned_local_enabled = bool(semantic_cfg.get("returned_local_dependency_weights_enabled", False))
    returned_local_update_weight = float(semantic_cfg.get("returned_local_update_weight") or 2.4)
    returned_local_return_weight = float(semantic_cfg.get("returned_local_return_weight") or 2.8)
    primary_dataflow_enabled = bool(semantic_cfg.get("primary_dataflow_weights_enabled", False))
    primary_parameter_weight = float(semantic_cfg.get("primary_parameter_identifier_weight") or parameter_weight)
    primary_dataflow_update_weight = float(semantic_cfg.get("primary_dataflow_update_weight") or 2.7)
    primary_dataflow_return_weight = float(semantic_cfg.get("primary_dataflow_return_weight") or 3.4)
    primary_loop_header_weight = float(semantic_cfg.get("primary_loop_header_weight") or 2.0)
    max_weight = max(1.0, float(semantic_cfg.get("max_weight") or 4.0))
    update_names = set(
        semantic_cfg.get("update_call_names")
        or [
            "append",
            "add",
            "extend",
            "get",
            "items",
            "join",
            "setdefault",
            "split",
            "splitlines",
            "update",
        ]
    )
    operator_values = set(semantic_cfg.get("operator_tokens") or ["=", ".", "[", "]", "+", "-", "%", "==", "!=", "<", ">"])
    control_names = set(semantic_cfg.get("control_names") or ["for", "in", "while", "if", "return"])
    inverse = {idx: tok for tok, idx in target_vocab.items()}
    return_token_id = int(target_vocab.get("NAME:return", -1))
    weights: list[list[float]] = []
    weighted_count = 0
    first_count = 0
    return_count = 0
    parameter_count = 0
    update_count = 0
    operator_count = 0
    control_count = 0
    line_boundary_count = 0
    loop_header_count = 0
    update_line_count = 0
    top_level_return_count = 0
    final_dedent_count = 0
    builtin_call_name_count = 0
    builtin_call_argument_count = 0
    builtin_call_punctuation_count = 0
    returned_local_update_count = 0
    returned_local_return_count = 0
    primary_parameter_count = 0
    primary_dataflow_update_count = 0
    primary_dataflow_return_count = 0
    primary_loop_header_count = 0
    row_parameter_lists: list[list[str]] = [
        visible_parameter_names_from_source_text(str(row.get("source_text") or ""))
        for row in train_examples
    ]
    for row_index, row in enumerate(target_rows):
        parameter_names = row_parameter_lists[row_index] if row_index < len(row_parameter_lists) else []
        allowed = set(parameter_names)
        primary_parameter = parameter_names[0] if parameter_names else ""
        row_weights = [1.0 for _token in row]
        structural_classes = structural_trajectory_classes_for_row(row, inverse) if structural_enabled else {}
        builtin_call_classes = builtin_call_trajectory_classes_for_row(row, inverse) if builtin_call_enabled else {}
        returned_local_classes = returned_local_dependency_classes_for_row(row, inverse) if returned_local_enabled else {}
        primary_dataflow_classes = (
            primary_dataflow_classes_for_row(row, inverse, primary_parameter=primary_parameter)
            if primary_dataflow_enabled and primary_parameter
            else {}
        )
        if len(row_weights) > 1 and first_token_weight != 1.0:
            row_weights[1] = max(row_weights[1], first_token_weight)
            first_count += 1
        for pos, token_id_raw in enumerate(row):
            token_id = int(token_id_raw)
            token_weight = row_weights[pos]
            if return_token_weight != 1.0 and token_id == return_token_id:
                token_weight = max(token_weight, return_token_weight)
                return_count += 1
            if semantic_enabled:
                tok = str(inverse.get(token_id, ""))
                kind, _, value = tok.partition(":")
                token_class = ""
                if kind == "NAME" and value in allowed and parameter_weight > token_weight:
                    token_weight = parameter_weight
                    token_class = "parameter"
                elif kind == "NAME" and value in update_names and update_weight > token_weight:
                    token_weight = update_weight
                    token_class = "update"
                elif kind == "OP" and value in operator_values and operator_weight > token_weight:
                    token_weight = operator_weight
                    token_class = "operator"
                elif kind == "NAME" and value in control_names and control_weight > token_weight:
                    token_weight = control_weight
                    token_class = "control"
                if token_class == "parameter":
                    parameter_count += 1
                elif token_class == "update":
                    update_count += 1
                elif token_class == "operator":
                    operator_count += 1
                elif token_class == "control":
                    control_count += 1
            if primary_dataflow_enabled:
                primary_class = str(primary_dataflow_classes.get(pos) or "")
                primary_weight = float(
                    {
                        "primary_parameter": primary_parameter_weight,
                        "primary_dataflow_update": primary_dataflow_update_weight,
                        "primary_dataflow_return": primary_dataflow_return_weight,
                        "primary_loop_header": primary_loop_header_weight,
                    }.get(primary_class, 1.0)
                )
                if primary_weight > token_weight:
                    token_weight = primary_weight
                if primary_class == "primary_parameter":
                    primary_parameter_count += 1
                elif primary_class == "primary_dataflow_update":
                    primary_dataflow_update_count += 1
                elif primary_class == "primary_dataflow_return":
                    primary_dataflow_return_count += 1
                elif primary_class == "primary_loop_header":
                    primary_loop_header_count += 1
            if structural_enabled:
                structural_class = str(structural_classes.get(pos) or "")
                structural_weight = float(
                    {
                        "line_boundary": line_boundary_weight,
                        "loop_header": loop_header_weight,
                        "update_line": update_line_weight,
                        "top_level_return": top_level_return_weight,
                        "final_dedent": final_dedent_weight,
                    }.get(structural_class, 1.0)
                )
                if structural_weight > token_weight:
                    token_weight = structural_weight
                if structural_class == "line_boundary":
                    line_boundary_count += 1
                elif structural_class == "loop_header":
                    loop_header_count += 1
                elif structural_class == "update_line":
                    update_line_count += 1
                elif structural_class == "top_level_return":
                    top_level_return_count += 1
                elif structural_class == "final_dedent":
                    final_dedent_count += 1
            if builtin_call_enabled:
                builtin_class = str(builtin_call_classes.get(pos) or "")
                builtin_weight = float(
                    {
                        "builtin_call_name": builtin_call_name_weight,
                        "builtin_call_argument": builtin_call_argument_weight,
                        "builtin_call_punctuation": builtin_call_punctuation_weight,
                    }.get(builtin_class, 1.0)
                )
                if builtin_weight > token_weight:
                    token_weight = builtin_weight
                if builtin_class == "builtin_call_name":
                    builtin_call_name_count += 1
                elif builtin_class == "builtin_call_argument":
                    builtin_call_argument_count += 1
                elif builtin_class == "builtin_call_punctuation":
                    builtin_call_punctuation_count += 1
            if returned_local_enabled:
                returned_class = str(returned_local_classes.get(pos) or "")
                returned_weight = float(
                    {
                        "returned_local_update": returned_local_update_weight,
                        "returned_local_return": returned_local_return_weight,
                    }.get(returned_class, 1.0)
                )
                if returned_weight > token_weight:
                    token_weight = returned_weight
                if returned_class == "returned_local_update":
                    returned_local_update_count += 1
                elif returned_class == "returned_local_return":
                    returned_local_return_count += 1
            token_weight = min(max_weight, max(1.0, float(token_weight)))
            if token_weight > 1.0:
                weighted_count += 1
            row_weights[pos] = token_weight
        weights.append(row_weights)
    return weights, {
        "enabled": bool(
            first_token_weight != 1.0
            or return_token_weight != 1.0
            or semantic_enabled
            or structural_enabled
            or primary_dataflow_enabled
        ),
        "policy": str(semantic_cfg.get("policy") or "visible_parameter_and_expression_token_weighting_v1"),
        "first_target_token_loss_weight": first_token_weight,
        "return_token_loss_weight": return_token_weight,
        "semantic_token_loss_weights_enabled": semantic_enabled,
        "parameter_identifier_weight": parameter_weight,
        "update_call_name_weight": update_weight,
        "operator_token_weight": operator_weight,
        "control_token_weight": control_weight,
        "structural_trajectory_weights_enabled": structural_enabled,
        "builtin_call_trajectory_weights_enabled": builtin_call_enabled,
        "line_boundary_weight": line_boundary_weight,
        "loop_header_weight": loop_header_weight,
        "accumulator_update_line_weight": update_line_weight,
        "top_level_return_weight": top_level_return_weight,
        "final_dedent_weight": final_dedent_weight,
        "builtin_call_name_weight": builtin_call_name_weight,
        "builtin_call_argument_weight": builtin_call_argument_weight,
        "builtin_call_punctuation_weight": builtin_call_punctuation_weight,
        "returned_local_dependency_weights_enabled": returned_local_enabled,
        "returned_local_update_weight": returned_local_update_weight,
        "returned_local_return_weight": returned_local_return_weight,
        "primary_dataflow_weights_enabled": primary_dataflow_enabled,
        "primary_parameter_identifier_weight": primary_parameter_weight,
        "primary_dataflow_update_weight": primary_dataflow_update_weight,
        "primary_dataflow_return_weight": primary_dataflow_return_weight,
        "primary_loop_header_weight": primary_loop_header_weight,
        "max_weight": max_weight,
        "weighted_token_count": weighted_count,
        "first_token_weighted_count": first_count,
        "return_token_count": return_count,
        "parameter_token_count": parameter_count,
        "update_token_count": update_count,
        "operator_token_count": operator_count,
        "control_token_count": control_count,
        "line_boundary_token_count": line_boundary_count,
        "loop_header_token_count": loop_header_count,
        "accumulator_update_line_token_count": update_line_count,
        "top_level_return_token_count": top_level_return_count,
        "final_dedent_token_count": final_dedent_count,
        "builtin_call_name_token_count": builtin_call_name_count,
        "builtin_call_argument_token_count": builtin_call_argument_count,
        "builtin_call_punctuation_token_count": builtin_call_punctuation_count,
        "returned_local_update_token_count": returned_local_update_count,
        "returned_local_return_token_count": returned_local_return_count,
        "primary_parameter_token_count": primary_parameter_count,
        "primary_dataflow_update_token_count": primary_dataflow_update_count,
        "primary_dataflow_return_token_count": primary_dataflow_return_count,
        "primary_loop_header_token_count": primary_loop_header_count,
        "rows": len(target_rows),
        "rows_with_visible_parameters": sum(1 for row in row_parameter_lists if row),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "score_semantics": (
            "MLX supervised loss weighting over admitted private/licensed target tokens only. "
            "It favors visible signature parameters, coherent builtin-call spans, and "
            "returned-local update/return trajectories; it does not "
            "synthesize code, inspect eval tests/solutions, use public benchmarks, or affect verifier scoring."
        ),
    }


def apply_primary_dataflow_weight_override(budget: dict[str, Any], *, scale: float = 1.0) -> None:
    """Enable first-argument dataflow weighting for a controlled ablation.

    The override only changes supervised private/licensed target-token loss
    weights. It does not change candidate generation, verifier scoring, public
    benchmark policy, or learned-generation credit.
    """

    factor = max(0.0, float(scale or 1.0))
    semantic_cfg = dict_or_empty(budget.get("semantic_token_loss_weights"))
    semantic_cfg.update(
        {
            "enabled": True,
            "primary_dataflow_weights_enabled": True,
            "primary_parameter_identifier_weight": round(3.7 * factor, 6),
            "primary_dataflow_update_weight": round(2.7 * factor, 6),
            "primary_dataflow_return_weight": round(3.6 * factor, 6),
            "primary_loop_header_weight": round(2.1 * factor, 6),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    )
    budget["semantic_token_loss_weights"] = semantic_cfg


def builtin_call_trajectory_classes_for_row(row: list[int], inverse: dict[int, str]) -> dict[int, str]:
    """Classify token positions inside known builtin calls for supervised weighting.

    This is AST-free and target-local: it sees only admitted private/licensed
    target tokens. It does not use eval tests, solutions from heldout rows,
    public benchmark data, or verifier outcomes.
    """

    known = {
        "abs",
        "all",
        "any",
        "bool",
        "bytes",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
    tokens = [str(inverse.get(int(token_id), "")) for token_id in row]
    values = [tok.partition(":")[2] for tok in tokens]
    classes: dict[int, str] = {}
    index = 0
    while index + 1 < len(tokens):
        kind, _, value = tokens[index].partition(":")
        next_kind, _, next_value = tokens[index + 1].partition(":")
        if kind != "NAME" or value not in known or next_kind != "OP" or next_value != "(":
            index += 1
            continue
        close = matching_close_paren_index(values, index + 1)
        if close <= index + 1:
            index += 1
            continue
        classes[index] = "builtin_call_name"
        for pos in range(index + 1, close + 1):
            pos_kind, _, pos_value = tokens[pos].partition(":")
            if pos_kind == "OP" and pos_value in {"(", ")", ","}:
                classes[pos] = "builtin_call_punctuation"
            elif pos_kind not in {"NEWLINE", "INDENT", "DEDENT"} and tokens[pos] not in {"<pad>", "<bos>", "<eos>"}:
                classes[pos] = "builtin_call_argument"
        index = close + 1
    return classes


def matching_close_paren_index(values: list[str], open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(values)):
        value = values[index]
        if value == "(":
            depth += 1
        elif value == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


def returned_local_dependency_classes_for_row(row: list[int], inverse: dict[int, str]) -> dict[int, str]:
    tokens = [str(inverse.get(int(token_id), "")) for token_id in row]
    lines = token_lines_with_depth(tokens)
    returned_locals = {
        values[1]
        for depth, _positions, values in lines
        if depth == 0 and len(values) >= 2 and values[0] == "return" and values[1].isidentifier()
    }
    if not returned_locals:
        return {}
    classes: dict[int, str] = {}
    for depth, positions, values in lines:
        if not values:
            continue
        if values[0] == "return" and len(values) >= 2 and values[1] in returned_locals:
            for pos in positions:
                classes[pos] = "returned_local_return"
            continue
        if line_updates_returned_local(values, returned_locals):
            for pos in positions:
                classes[pos] = "returned_local_update"
    return classes


def token_lines_with_depth(tokens: list[str]) -> list[tuple[int, list[int], list[str]]]:
    lines: list[tuple[int, list[int], list[str]]] = []
    depth = 0
    positions: list[int] = []
    values: list[str] = []
    line_depth = 0
    for index, tok in enumerate(tokens):
        kind, _, value = tok.partition(":")
        if tok in {"<pad>", "<bos>", "<eos>"} or kind == "SLOT":
            continue
        if kind == "DEDENT":
            if values:
                lines.append((line_depth, positions, values))
                positions = []
                values = []
            depth = max(0, depth - 1)
            continue
        if kind == "INDENT":
            if values:
                lines.append((line_depth, positions, values))
                positions = []
                values = []
            depth += 1
            continue
        if kind == "NEWLINE":
            if values:
                lines.append((line_depth, positions, values))
                positions = []
                values = []
            continue
        if not values:
            line_depth = depth
        positions.append(index)
        values.append(value)
    if values:
        lines.append((line_depth, positions, values))
    return lines


def line_updates_returned_local(values: list[str], returned_locals: set[str]) -> bool:
    if not values:
        return False
    if values[0] in returned_locals and "=" in values:
        equals = values.index("=")
        if equals > 0 and any(value in values[equals + 1 :] for value in returned_locals):
            return True
        # Initial empty-literal setup is not a dependency update by itself.
        rhs = values[equals + 1 :]
        return bool(rhs and rhs[:1] not in (["["], ["{"], ["("], ["''"], ['""']))
    for name in returned_locals:
        if name in values and "." in values:
            dot_positions = [idx for idx, value in enumerate(values) if value == "."]
            if any(idx > 0 and values[idx - 1] == name for idx in dot_positions):
                return True
        if name in values and "[" in values and "=" in values:
            name_index = values.index(name)
            equals = values.index("=")
            bracket_index = values.index("[")
            if name_index < bracket_index < equals:
                return True
    return False


def primary_dataflow_classes_for_row(
    row: list[int],
    inverse: dict[int, str],
    *,
    primary_parameter: str,
) -> dict[int, str]:
    """Classify private target tokens that carry first-argument dataflow.

    This sees only admitted private/licensed target tokens plus the visible
    callable parameter name. It does not inspect tests, eval rows, public
    benchmark payloads, verifier results, task ids, or solution metadata beyond
    the training body itself.
    """

    if not primary_parameter:
        return {}
    tokens = [str(inverse.get(int(token_id), "")) for token_id in row]
    lines = token_lines_with_depth(tokens)
    derived = {primary_parameter}
    classes: dict[int, str] = {}
    for _depth, positions, values in lines:
        if not values:
            continue
        if values[:1] == ["for"]:
            target_names = names_before_in(values)
            iter_values = values_after_in(values)
            if target_names and any(value in derived for value in iter_values):
                for pos in positions:
                    classes[pos] = "primary_loop_header"
                derived.update(target_names)
        assignment_targets = assignment_targets_for_values(values)
        if assignment_targets:
            rhs = values_after_assignment_operator(values)
            if any(value in derived for value in rhs):
                for pos in positions:
                    classes[pos] = "primary_dataflow_update"
                derived.update(assignment_targets)
        method_update_targets = method_update_targets_from_derived_args(values, derived)
        if method_update_targets:
            for pos in positions:
                classes[pos] = "primary_dataflow_update"
            derived.update(method_update_targets)
        if values[0] == "return" and any(value in derived for value in values[1:]):
            for pos in positions:
                classes[pos] = "primary_dataflow_return"
        for pos, value in zip(positions, values):
            if value == primary_parameter:
                classes[pos] = "primary_parameter"
    return classes


def names_before_in(values: list[str]) -> set[str]:
    if "in" not in values:
        return set()
    index = values.index("in")
    return {value for value in values[1:index] if value.isidentifier() and value not in {",", "(", ")"}}


def values_after_in(values: list[str]) -> list[str]:
    if "in" not in values:
        return []
    return values[values.index("in") + 1 :]


def assignment_targets_for_values(values: list[str]) -> set[str]:
    if "=" not in values:
        return set()
    equals = values.index("=")
    if equals <= 0:
        return set()
    return {value for value in values[:equals] if value.isidentifier()}


def values_after_assignment_operator(values: list[str]) -> list[str]:
    if "=" not in values:
        return []
    return values[values.index("=") + 1 :]


def method_update_targets_from_derived_args(values: list[str], derived: set[str]) -> set[str]:
    if not values:
        return set()
    update_methods = {"append", "add", "extend", "insert", "setdefault", "update"}
    targets: set[str] = set()
    for index, value in enumerate(values):
        if value != "." or index <= 0 or index + 1 >= len(values):
            continue
        receiver = values[index - 1]
        method = values[index + 1]
        if method not in update_methods:
            continue
        if receiver in derived or any(arg in derived for arg in values[index + 2 :]):
            targets.add(receiver)
    return targets


def visible_parameter_names_from_source_text(text: str) -> list[str]:
    names: list[str] = []

    def add(name: str) -> None:
        cleaned = str(name or "").strip("* ,:()")
        if cleaned.isidentifier() and cleaned not in {"self", "cls"} and cleaned not in names:
            names.append(cleaned)

    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("arguments "):
            for raw in stripped.removeprefix("arguments ").replace(",", " ").split():
                add(raw)
        elif stripped.startswith("signature "):
            for name in parameter_names_from_signature_text(stripped.removeprefix("signature ").strip()):
                add(name)
    return names


def allowed_parameter_names_from_source_text(text: str) -> set[str]:
    names: set[str] = set()
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("arguments "):
            for raw in stripped.removeprefix("arguments ").replace(",", " ").split():
                cleaned = raw.strip("* ,:()")
                if cleaned.isidentifier() and cleaned not in {"self", "cls"}:
                    names.add(cleaned)
        if stripped.startswith("signature "):
            names.update(parameter_names_from_signature_text(stripped.removeprefix("signature ").strip()))
    return names


def parameter_names_from_signature_text(signature: str) -> list[str]:
    names: list[str] = []

    def add(name: str) -> None:
        cleaned = str(name or "").strip().split(":")[0].split("=")[0].strip("* /")
        if cleaned.isidentifier() and cleaned not in {"self", "cls"} and cleaned not in names:
            names.append(cleaned)

    try:
        parsed = ast.parse(signature + "\n    pass\n")
        function = parsed.body[0] if parsed.body and isinstance(parsed.body[0], ast.FunctionDef) else None
        if function is not None:
            args = function.args
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                add(arg.arg)
            if args.vararg:
                add(args.vararg.arg)
            if args.kwarg:
                add(args.kwarg.arg)
    except SyntaxError:
        match = re.search(r"\((.*?)\)", signature)
        if match:
            for raw in match.group(1).split(","):
                add(raw)
    return names


def structural_trajectory_classes_for_row(row: list[int], inverse: dict[int, str]) -> dict[int, str]:
    tokens = [str(inverse.get(int(token_id), "")) for token_id in row]
    classes: dict[int, str] = {}
    depth = 0
    current_positions: list[int] = []
    current_values: list[str] = []
    current_depth = 0

    def classify_line() -> None:
        if not current_positions or not current_values:
            return
        first = current_values[0]
        line_class = ""
        if first == "return" and current_depth == 0:
            line_class = "top_level_return"
        elif first in {"for", "while"}:
            line_class = "loop_header"
        elif target_line_is_accumulator_update(current_values):
            line_class = "update_line"
        if line_class:
            for index in current_positions:
                classes[index] = line_class

    for index, tok in enumerate(tokens):
        if tok in {"<pad>", "<bos>", "<eos>", ""}:
            continue
        kind, _, value = tok.partition(":")
        if kind == "INDENT":
            classes.setdefault(index, "line_boundary")
            depth += 1
            continue
        if kind == "DEDENT":
            depth = max(0, depth - 1)
            classes.setdefault(index, "line_boundary")
            if next_significant_values_start_with_return(tokens, index + 1) and depth == 0:
                classes[index] = "final_dedent"
            continue
        if kind == "NEWLINE":
            classes.setdefault(index, "line_boundary")
            classify_line()
            current_positions = []
            current_values = []
            continue
        if not current_positions:
            current_depth = depth
        current_positions.append(index)
        current_values.append(value)
    classify_line()
    return classes


def next_significant_values_start_with_return(tokens: list[str], start: int) -> bool:
    for tok in tokens[start:]:
        if tok in {"<pad>", "<bos>", "<eos>", ""}:
            continue
        kind, _, value = tok.partition(":")
        if kind in {"DEDENT", "INDENT", "NEWLINE"}:
            continue
        return kind == "NAME" and value == "return"
    return False


def target_line_is_accumulator_update(values: list[str]) -> bool:
    if not values:
        return False
    if values[0] in {"return", "if", "elif", "while", "for"}:
        return False
    if "." in values and any(
        name in values
        for name in {"add", "append", "extend", "setdefault", "update"}
    ):
        return True
    if "=" in values and values.index("=") > 0:
        return True
    return False


def evaluate_loss_mlx(
    model: Any,
    source_rows: list[list[int]],
    target_rows: list[list[int]],
    *,
    batch_size: int,
    pad_id: int,
    mx: Any,
    nn: Any,
) -> dict[str, Any]:
    if not source_rows or not target_rows:
        return {"loss": None, "perplexity": None}
    losses: list[float] = []
    model.eval()
    for start in range(0, len(source_rows), batch_size):
        src = mx.array(source_rows[start : start + batch_size], dtype=mx.int32)
        tgt = mx.array(target_rows[start : start + batch_size], dtype=mx.int32)
        loss = loss_fn_mlx(model, src, tgt, pad_id, mx, nn)
        mx.eval(loss)
        losses.append(float(loss.item()))
    model.train()
    loss = sum(losses) / max(1, len(losses))
    return {"loss": round(loss, 6), "perplexity": round(min(1e12, pow(2.718281828459045, loss)), 6)}


def evaluate_source_contrast_mlx(
    model: Any,
    source_rows: list[list[int]],
    target_rows: list[list[int]],
    *,
    batch_size: int,
    pad_id: int,
    prefix_token_count: int,
    span_mode: str = "prefix",
    body_start_token_id: int = -1,
    mx: Any,
    nn: Any,
) -> dict[str, Any]:
    if len(source_rows) < 2 or len(target_rows) < 2:
        return {"matched_loss": None, "mismatched_loss": None, "loss_gap": None}
    mismatched_source_rows = [list(row) for row in source_rows[1:]] + [list(source_rows[0])]
    matched = evaluate_prefix_loss_mlx(
        model,
        source_rows,
        target_rows,
        batch_size=batch_size,
        pad_id=pad_id,
        prefix_token_count=prefix_token_count,
        span_mode=span_mode,
        body_start_token_id=body_start_token_id,
        mx=mx,
        nn=nn,
    )
    mismatched = evaluate_prefix_loss_mlx(
        model,
        mismatched_source_rows,
        target_rows,
        batch_size=batch_size,
        pad_id=pad_id,
        prefix_token_count=prefix_token_count,
        span_mode=span_mode,
        body_start_token_id=body_start_token_id,
        mx=mx,
        nn=nn,
    )
    if matched.get("loss") is None or mismatched.get("loss") is None:
        return {
            "matched_loss": matched.get("loss"),
            "mismatched_loss": mismatched.get("loss"),
            "loss_gap": None,
        }
    return {
        "matched_loss": matched.get("loss"),
        "mismatched_loss": mismatched.get("loss"),
        "loss_gap": round(float(mismatched["loss"]) - float(matched["loss"]), 6),
    }


def evaluate_prefix_loss_mlx(
    model: Any,
    source_rows: list[list[int]],
    target_rows: list[list[int]],
    *,
    batch_size: int,
    pad_id: int,
    prefix_token_count: int,
    span_mode: str = "prefix",
    body_start_token_id: int = -1,
    mx: Any,
    nn: Any,
) -> dict[str, Any]:
    if int(prefix_token_count or 0) <= 0:
        return evaluate_loss_mlx(model, source_rows, target_rows, batch_size=batch_size, pad_id=pad_id, mx=mx, nn=nn)
    if not source_rows or not target_rows:
        return {"loss": None, "perplexity": None}
    losses: list[float] = []
    model.eval()
    for start in range(0, len(source_rows), batch_size):
        src = mx.array(source_rows[start : start + batch_size], dtype=mx.int32)
        tgt = mx.array(target_rows[start : start + batch_size], dtype=mx.int32)
        weights = mx.ones(tgt.shape, dtype=mx.float32)
        loss = weighted_loss_with_prefix_mlx(
            model,
            src,
            tgt,
            pad_id,
            weights,
            mx,
            nn,
            prefix_token_count=int(prefix_token_count),
            span_mode=span_mode,
            body_start_token_id=int(body_start_token_id),
        )
        mx.eval(loss)
        losses.append(float(loss.item()))
    model.train()
    loss = sum(losses) / max(1, len(losses))
    return {"loss": round(loss, 6), "perplexity": round(min(1e12, pow(2.718281828459045, loss)), 6)}


def evaluate_semantic_plan_mlx(
    model: Any,
    source_rows: list[list[int]],
    plan_targets: list[int],
    *,
    batch_size: int,
    pad_id: int,
    mx: Any,
    nn: Any,
    plan_class_ids: Any | None = None,
) -> dict[str, Any]:
    if not source_rows or not plan_targets:
        return {"loss": None, "accuracy": None, "active_target_count": 0}
    active_total = sum(1 for value in plan_targets if int(value) != int(pad_id))
    if active_total <= 0:
        return {"loss": None, "accuracy": None, "active_target_count": 0}
    model.eval()
    losses: list[float] = []
    correct = 0
    counted = 0
    for start in range(0, len(source_rows), batch_size):
        src = mx.array(source_rows[start : start + batch_size], dtype=mx.int32)
        targets = mx.array(plan_targets[start : start + batch_size], dtype=mx.int32)
        logits = model.semantic_plan_logits(src)
        class_ids = plan_class_ids
        if class_ids is not None and int(class_ids.shape[0]) > 0:
            logits = mx.take(logits, class_ids, axis=1)
            matches = (targets[:, None] == class_ids[None, :])
            pred_targets = mx.argmax(matches.astype(mx.int32), axis=1)
            valid = mx.max(matches.astype(mx.float32), axis=1)
            valid = valid * (targets != pad_id).astype(mx.float32)
            losses_row = nn.losses.cross_entropy(logits, pred_targets, reduction="none")
            pred = mx.argmax(logits, axis=-1)
            hit = pred == pred_targets
        else:
            losses_row = nn.losses.cross_entropy(logits, targets, reduction="none")
            valid = (targets != pad_id).astype(mx.float32)
            pred = mx.argmax(logits, axis=-1)
            hit = pred == targets
        loss = mx.sum(losses_row * valid) / mx.maximum(mx.sum(valid), mx.array(1.0, dtype=mx.float32))
        batch_correct = mx.sum((hit.astype(mx.float32) * valid)).astype(mx.float32)
        batch_count = mx.sum(valid)
        mx.eval(loss, batch_correct, batch_count)
        if float(batch_count.item()) > 0.0:
            losses.append(float(loss.item()))
            correct += int(batch_correct.item())
            counted += int(batch_count.item())
    model.train()
    loss_value = sum(losses) / max(1, len(losses)) if losses else None
    return {
        "loss": round(loss_value, 6) if loss_value is not None else None,
        "accuracy": round(correct / counted, 6) if counted else None,
        "correct_count": correct,
        "active_target_count": counted,
        "label_space": (
            "semantic_plan_token_subspace_v1"
            if plan_class_ids is not None and int(plan_class_ids.shape[0]) > 0
            else "full_target_vocab_v1"
        ),
        "plan_class_count": (
            int(plan_class_ids.shape[0])
            if plan_class_ids is not None and int(plan_class_ids.shape[0]) > 0
            else None
        ),
    }


def evaluate_semantic_slot_mlx(
    model: Any,
    source_rows: list[list[int]],
    slot_targets: list[list[int]],
    *,
    batch_size: int,
    pad_id: int,
    enabled: bool,
    mx: Any,
    nn: Any,
    slot_role_class_ids: list[Any] | None = None,
) -> dict[str, Any]:
    if not enabled or not source_rows or not slot_targets or not hasattr(model, "semantic_slot_logits"):
        return {"loss": None, "accuracy": None, "active_target_count": 0, "enabled": bool(enabled)}
    active_total = sum(
        1
        for row in slot_targets
        for value in row[: len(SEMANTIC_SLOT_ROLES)]
        if int(value) != int(pad_id)
    )
    if active_total <= 0:
        return {"loss": None, "accuracy": None, "active_target_count": 0, "enabled": bool(enabled)}
    model.eval()
    losses: list[float] = []
    correct = 0
    counted = 0
    role_correct = {role: 0 for role, _prefixes in SEMANTIC_SLOT_ROLES}
    role_count = {role: 0 for role, _prefixes in SEMANTIC_SLOT_ROLES}
    for start in range(0, len(source_rows), batch_size):
        src = mx.array(source_rows[start : start + batch_size], dtype=mx.int32)
        targets = mx.array(slot_targets[start : start + batch_size], dtype=mx.int32)
        logits = model.semantic_slot_logits(src)
        loss = semantic_slot_loss_mlx(
            model,
            src,
            targets,
            pad_id,
            mx,
            nn,
            slot_role_class_ids=slot_role_class_ids,
        )
        pred_columns: list[Any] = []
        for role_index in range(len(SEMANTIC_SLOT_ROLES)):
            role_logits = logits[:, role_index, :]
            class_ids = None
            if slot_role_class_ids is not None and role_index < len(slot_role_class_ids):
                class_ids = slot_role_class_ids[role_index]
            if class_ids is not None and int(class_ids.shape[0]) > 0:
                role_logits = mx.take(role_logits, class_ids, axis=1)
                sub_pred = mx.argmax(role_logits, axis=-1)
                pred_columns.append(mx.take(class_ids, sub_pred, axis=0))
            else:
                pred_columns.append(mx.argmax(role_logits, axis=-1))
        pred = mx.stack(pred_columns, axis=1)
        valid = (targets != pad_id).astype(mx.float32)
        hit = (pred == targets).astype(mx.float32) * valid
        batch_correct = mx.sum(hit).astype(mx.float32)
        batch_count = mx.sum(valid)
        mx.eval(loss, pred, targets, valid, hit, batch_correct, batch_count)
        if float(batch_count.item()) > 0.0:
            losses.append(float(loss.item()))
            correct += int(batch_correct.item())
            counted += int(batch_count.item())
            pred_rows = pred.tolist()
            target_rows = targets.tolist()
            valid_rows = valid.tolist()
            for row_index, target_row in enumerate(target_rows):
                for role_index, target_id in enumerate(target_row[: len(SEMANTIC_SLOT_ROLES)]):
                    if float(valid_rows[row_index][role_index]) <= 0.0:
                        continue
                    role = SEMANTIC_SLOT_ROLES[role_index][0]
                    role_count[role] += 1
                    if int(pred_rows[row_index][role_index]) == int(target_id):
                        role_correct[role] += 1
    model.train()
    loss_value = sum(losses) / max(1, len(losses)) if losses else None
    return {
        "enabled": bool(enabled),
        "loss": round(loss_value, 6) if loss_value is not None else None,
        "accuracy": round(correct / counted, 6) if counted else None,
        "correct_count": correct,
        "active_target_count": counted,
        "role_accuracy": {
            role: {
                "accuracy": round(role_correct[role] / role_count[role], 6) if role_count[role] else None,
                "correct_count": role_correct[role],
                "active_target_count": role_count[role],
            }
            for role, _prefixes in SEMANTIC_SLOT_ROLES
        },
        "label_space": (
            "role_prefix_subspace_v1"
            if slot_role_class_ids is not None and any(int(row.shape[0]) > 0 for row in slot_role_class_ids)
            else "role_conditioned_full_target_vocab_v1"
        ),
    }


def additive_padding_mask(tokens: Any, mx: Any) -> Any:
    pad = (tokens == 0).astype(mx.float32) * mx.finfo(mx.float32).min
    return pad[:, None, None, :]


def selected_budget(cfg: dict[str, Any], budget_id: str) -> dict[str, Any]:
    budgets = [dict_or_empty(row) for row in cfg.get("budgets", []) if isinstance(row, dict)]
    for row in budgets:
        if str(row.get("id") or row.get("budget_id") or "") == budget_id:
            return row
    return {
        "id": budget_id,
        "max_files": 64,
        "max_examples": 256,
        "target_vocab_max_files": 64,
        "target_vocab_max_examples": 512,
        "source_vocab_max_files": 64,
        "source_vocab_max_examples": 512,
        "max_eval_examples": 64,
        "epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.0008,
    }


def parameter_snapshot(model: Any, mlx_utils: Any, mx: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name, value in mlx_utils.tree_flatten(model.trainable_parameters()):
        snapshot[name] = mx.array(value)
    mx.eval(*snapshot.values())
    return snapshot


def parameter_update_summary(model: Any, before: dict[str, Any], mlx_utils: Any, mx: Any) -> dict[str, Any]:
    total = 0
    changed = 0
    tensor_total = 0
    tensor_changed = 0
    for name, value in mlx_utils.tree_flatten(model.trainable_parameters()):
        tensor_total += 1
        total += int(value.size)
        old = before.get(name)
        if old is None:
            changed += int(value.size)
            tensor_changed += 1
            continue
        delta = mx.abs(value - old)
        element_changed = int(mx.sum((delta > 1e-10).astype(mx.int32)).item())
        changed += element_changed
        if element_changed:
            tensor_changed += 1
    return {
        "parameter_count": total,
        "updated_parameter_count": changed,
        "parameter_update_fraction": round(changed / total, 6) if total else 0.0,
        "parameter_tensor_count": tensor_total,
        "updated_parameter_tensor_count": tensor_changed,
        "parameter_tensor_update_fraction": round(tensor_changed / tensor_total, 6) if tensor_total else 0.0,
    }


def build_gates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    source_contrast = dict_or_empty(payload.get("source_contrastive_loss"))
    semantic_plan = dict_or_empty(payload.get("semantic_plan_auxiliary"))
    semantic_slot = dict_or_empty(payload.get("semantic_slot_auxiliary"))
    training_plan = dict_or_empty(payload.get("training_plan"))
    target_token_positions = int(training_plan.get("target_token_positions") or 0)
    consumed_token_positions = int(payload.get("optimizer_token_positions_consumed") or 0)
    requested_rungs = [int(value) for value in training_plan.get("rung_token_positions_requested") or []]
    written_rungs = list(payload.get("rung_checkpoints") or [])
    checkpoint_path = resolve(str(payload.get("checkpoint") or ""))
    vocab_path = resolve(str(payload.get("vocab") or ""))
    device_text = str(payload.get("device") or "").lower()
    return [
        gate("active_training_rows_present", bool(payload.get("active")), "hard", payload.get("row_summary", {})),
        gate("mlx_high_level_backend", str(payload.get("backend") or "") == "mlx_high_level_transformer", "hard", payload.get("backend")),
        gate("mlx_gpu_device_used", "gpu" in device_text, "hard", payload.get("device")),
        gate("parameter_count_recorded", int(payload.get("parameter_count") or 0) > 0, "hard", payload.get("parameter_count")),
        gate("checkpoint_artifacts_written", checkpoint_path.exists() and vocab_path.exists(), "hard", {"checkpoint": payload.get("checkpoint"), "vocab": payload.get("vocab")}),
        gate("public_training_rows_zero", int(payload.get("public_training_rows") or 0) == 0, "hard", payload.get("public_training_rows")),
        gate("external_inference_zero", int(payload.get("external_inference_calls") or 0) == 0, "hard", payload.get("external_inference_calls")),
        gate("no_open_or_pretrained_weights", not bool(payload.get("open_or_pretrained_model_weights_used")), "hard", "from_scratch_only"),
        gate("no_template_router_tool_credit", int(payload.get("fallback_template_router_tool_credit_count") or 0) == 0, "hard", payload.get("fallback_template_router_tool_credit_count")),
        gate(
            "target_token_positions_reached_when_declared",
            target_token_positions <= 0 or consumed_token_positions >= target_token_positions,
            "hard",
            {
                "target_token_positions": target_token_positions,
                "optimizer_token_positions_consumed": consumed_token_positions,
                "training_plan": training_plan,
            },
        ),
        gate("parameter_update_recorded", float(payload.get("parameter_update_fraction") or 0.0) >= 0.95, "hard", payload.get("parameter_update_fraction")),
        gate("heldout_lm_improved", bool(payload.get("heldout_lm_improved")), "hard", [payload.get("heldout_lm_loss_before"), payload.get("heldout_lm_loss_after")]),
        gate("throughput_recorded", float(payload.get("training_tokens_per_second") or 0.0) > 0.0, "hard", payload.get("training_tokens_per_second")),
        gate(
            "requested_rung_checkpoints_written",
            not requested_rungs or len(written_rungs) == len(requested_rungs),
            "hard",
            {
                "requested": requested_rungs,
                "written": [
                    {
                        "milestone_token_positions": row.get("milestone_token_positions"),
                        "checkpoint": row.get("checkpoint"),
                    }
                    for row in written_rungs
                ],
            },
        ),
        gate(
            "source_contrastive_gap_improved_when_enabled",
            not bool(source_contrast.get("enabled")) or bool(source_contrast.get("loss_gap_improved")),
            "soft",
            source_contrast,
        ),
        gate(
            "semantic_plan_loss_improved_when_enabled",
            not bool(semantic_plan.get("enabled")) or bool(semantic_plan.get("heldout_plan_improved")),
            "soft",
            semantic_plan,
        ),
        gate(
            "semantic_slot_loss_improved_when_active",
            not bool(semantic_slot.get("active")) or bool(semantic_slot.get("heldout_slot_improved")),
            "soft",
            semantic_slot,
        ),
    ]


def gate(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def stable_hash_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_token_position_milestones(text: str) -> list[int]:
    values: list[int] = []
    for part in re.split(r"[\s,]+", str(text or "").strip()):
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            raise SystemExit(f"invalid --rung-token-positions value: {part!r}") from None
        if value > 0:
            values.append(value)
    return sorted(set(values))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
