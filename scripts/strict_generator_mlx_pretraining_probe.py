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
import math
import random
import re
import sys
import time
from collections import Counter
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
from neural_seed_teacher_distillation_rows import load_governed_teacher_code_lm_training_rows  # noqa: E402
from strict_generator_pretraining_spine import (  # noqa: E402
    config_with_budget_overrides as strict_spine_config_with_budget_overrides,
    encode_staged_full_state_rows,
    safe_slug,
    stage_full_state_examples,
    transformer_dims_with_budget,
)
from strict_generator_mlx_model import (  # noqa: E402
    MlxStrictGenerator as SharedMlxStrictGenerator,
    normalize_specialist_core_config,
    specialist_core_parameter_estimate,
)
from strict_generator_mlx_replay_selection import (  # noqa: E402
    exclude_configured_holdout_rows_for_replay,
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

BODY_ACTION_ROLES: tuple[str, ...] = (
    "ignore",
    "return",
    "block_exit",
    "block_enter",
    "loop",
    "branch",
    "assignment",
    "update_operator",
    "comparison",
    "bool_op",
    "call_or_method",
    "attribute",
    "open_expr",
    "close_expr",
    "identifier",
    "literal",
    "line_boundary",
    "eos",
    "other",
)
BODY_ACTION_ROLE_TO_ID = {role: index for index, role in enumerate(BODY_ACTION_ROLES)}

BODY_OPERAND_ROLES: tuple[str, ...] = (
    "ignore",
    "visible_parameter",
    "loop_variable",
    "local_state",
    "builtin_function",
    "method_name",
    "attribute_name",
    "literal_value",
    "assignment_operator",
    "arithmetic_operator",
    "comparison_operator",
    "boolean_operator",
    "call_delimiter",
    "index_delimiter",
    "punctuation",
    "statement_boundary",
    "return_keyword",
    "control_keyword",
    "eos",
    "other",
)
BODY_OPERAND_ROLE_TO_ID = {role: index for index, role in enumerate(BODY_OPERAND_ROLES)}

BODY_STATE_EVENT_ROLES: tuple[str, ...] = (
    "none",
    "traversal_or_call",
    "state_update",
    "control_transition",
    "return_finalizer",
    "value_expression",
    "statement_boundary",
)
BODY_STATE_EVENT_ROLE_TO_ID = {role: index for index, role in enumerate(BODY_STATE_EVENT_ROLES)}

ACTION_STATE_EVENT_ROLE_BY_ACTION_ROLE: dict[str, str] = {
    "return": "return_finalizer",
    "block_exit": "statement_boundary",
    "block_enter": "control_transition",
    "loop": "control_transition",
    "branch": "control_transition",
    "assignment": "state_update",
    "update_operator": "state_update",
    "comparison": "control_transition",
    "bool_op": "control_transition",
    "call_or_method": "traversal_or_call",
    "attribute": "traversal_or_call",
    "line_boundary": "statement_boundary",
    "eos": "return_finalizer",
}

OPERAND_STATE_EVENT_ROLE_BY_OPERAND_ROLE: dict[str, str] = {
    "builtin_function": "traversal_or_call",
    "method_name": "traversal_or_call",
    "attribute_name": "traversal_or_call",
    "assignment_operator": "state_update",
    "arithmetic_operator": "value_expression",
    "comparison_operator": "control_transition",
    "boolean_operator": "control_transition",
    "return_keyword": "return_finalizer",
    "control_keyword": "control_transition",
    "eos": "return_finalizer",
}

BODY_STATE_EVENT_ACTION_COMPATIBILITY_MASK: tuple[tuple[float, ...], ...] = tuple(
    tuple(
        1.0 if ACTION_STATE_EVENT_ROLE_BY_ACTION_ROLE.get(action_role) == event_role else 0.0
        for action_role in BODY_ACTION_ROLES
    )
    for event_role in BODY_STATE_EVENT_ROLES
)

BODY_STATE_EVENT_OPERAND_COMPATIBILITY_MASK: tuple[tuple[float, ...], ...] = tuple(
    tuple(
        1.0 if OPERAND_STATE_EVENT_ROLE_BY_OPERAND_ROLE.get(operand_role) == event_role else 0.0
        for operand_role in BODY_OPERAND_ROLES
    )
    for event_role in BODY_STATE_EVENT_ROLES
)

BODY_EXECUTABLE_SPAN_ROLES: tuple[str, ...] = (
    "ignore",
    "guard_control_span",
    "traversal_call_span",
    "state_update_span",
    "value_expression_span",
    "return_finalizer_span",
    "state_reference_span",
    "literal_span",
    "delimiter_span",
    "statement_boundary_span",
    "other_span",
)
BODY_EXECUTABLE_SPAN_ROLE_TO_ID = {role: index for index, role in enumerate(BODY_EXECUTABLE_SPAN_ROLES)}


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
            "specialist_core": payload.get("specialist_core"),
            "specialist_router_supervision": payload.get("specialist_router_supervision"),
            "specialist_routing_before": payload.get("specialist_routing_before"),
            "specialist_routing": payload.get("specialist_routing"),
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
            "family_disjoint_evidence": payload.get("family_disjoint_evidence"),
            "family_disjoint_holdout_exclusion": payload.get("family_disjoint_holdout_exclusion"),
            "checkpoint_training_lineage": payload.get("checkpoint_training_lineage"),
            "teacher_training": payload.get("teacher_training"),
            "data_exposure": payload.get("data_exposure"),
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


MlxStrictGenerator = SharedMlxStrictGenerator


def inject_governed_teacher_examples(
    staged: dict[str, Any],
    *,
    teacher_rows: list[dict[str, Any]],
    teacher_summary: dict[str, Any],
    text_views: dict[str, Any],
) -> dict[str, Any]:
    """Reserve staged train positions for manifest-admitted teacher rows."""
    result = dict(staged)
    examples = [dict(row) for row in list(staged.get("examples") or []) if isinstance(row, dict)]
    teacher_examples = []
    source_fields = list(text_views.get("sts_on") or text_views.get("sts_off") or ["prompt", "entry_point"])
    for row in teacher_rows:
        body = str(row.get("solution_body") or "").strip()
        prompt = str(row.get("prompt") or "").strip()
        entry_point = str(row.get("entry_point") or "").strip()
        if not body or not prompt or not entry_point:
            continue
        teacher_examples.append(
            {
                "path": f"teacher_manifest://{row.get('teacher_manifest_row_id') or row.get('task_id')}",
                "function": entry_point,
                "source_text": decoder_source_text(row, source_fields),
                "source_text_style": "prompt_signature_operation_metadata_v3",
                "body": body,
                "source_kind": "teacher_distillation",
                "teacher_generated": True,
                "teacher_manifest_row_id": row.get("teacher_manifest_row_id"),
                "quality": {
                    "policy": "manifest_execution_verified_teacher_row_v1",
                    "uses_eval_tests_or_solutions": False,
                    "uses_public_data": False,
                },
            }
        )
    limit = max(len(examples), len(teacher_examples))
    result["examples"] = [*examples[: max(0, limit - len(teacher_examples))], *teacher_examples]
    result["source_vocab_extension_texts"] = [
        *list(staged.get("source_vocab_extension_texts") or []),
        *(str(row["source_text"]) for row in teacher_examples),
    ]
    result["target_vocab_extension_bodies"] = [
        *list(staged.get("target_vocab_extension_bodies") or []),
        *(str(row["body"]) for row in teacher_examples),
    ]
    summary = dict_or_empty(staged.get("summary"))
    summary.update(
        {
            "governed_teacher_training_enabled": bool(teacher_summary.get("enabled")),
            "governed_teacher_gate_green": bool(teacher_summary.get("gate_green")),
            "governed_teacher_available_row_count": int(teacher_summary.get("available_code_lm_training_rows") or 0),
            "governed_teacher_injected_row_count": len(teacher_examples),
            "governed_teacher_holdout_family_row_count": int(teacher_summary.get("holdout_family_code_lm_training_rows") or 0),
            "governed_teacher_manifest": teacher_summary.get("manifest"),
            "teacher_source_external_inference_calls": int(teacher_summary.get("external_inference_calls") or 0),
            "runtime_external_inference_calls": 0,
            "public_training_rows": 0,
        }
    )
    result["summary"] = summary
    result["teacher_training"] = {
        **teacher_summary,
        "injected_row_count": len(teacher_examples),
        "runtime_external_inference_calls": 0,
        "public_training_rows": 0,
    }
    return result


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
    train_rows_pool, family_holdout_exclusion = exclude_configured_holdout_rows_for_replay(
        working_config,
        train_rows_all,
    )
    teacher_training = load_governed_teacher_code_lm_training_rows(working_config)
    teacher_rows = list(teacher_training.get("rows") or [])
    family_disjoint_claim_required = bool(
        dict_or_empty(data_cfg.get("family_disjoint_eval")).get("enabled")
    )
    family_disjoint_evidence = bool(
        family_holdout_exclusion.get("enabled")
        and family_holdout_exclusion.get("clean")
        and int(family_holdout_exclusion.get("excluded_row_count") or 0) > 0
        and not teacher_rows
    )
    family_disjoint_lineage = {
        "policy": "strict_generator_from_scratch_family_disjoint_lineage_v1",
        "family_disjoint_claim_required": family_disjoint_claim_required,
        "family_disjoint_evidence": family_disjoint_evidence,
        "family_disjoint_claim_state": "VERIFIED" if family_disjoint_evidence else "UNVERIFIED",
        "holdout_exclusion": family_holdout_exclusion,
        "private_vocab_seed_rows_before_exclusion": len(train_rows_all),
        "private_vocab_seed_rows_after_exclusion": len(train_rows_pool),
        "teacher_training_row_count": len(teacher_rows),
        "model_initialized_from_scratch": True,
        "staged_training_data_role": "admitted_private_or_licensed_source_corpus",
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
        "score_semantics": (
            "Base-checkpoint lineage for configured private family-disjoint evaluation. Heldout task "
            "families are removed before private prompt/body rows can seed source or target vocabulary. "
            "Teacher rows make this claim unverified until separately audited."
        ),
    }
    max_train_rows = int(data_cfg.get("max_train_rows") or 512)
    train_rows = [
        *deterministic_sample(train_rows_pool, max(0, max_train_rows - len(teacher_rows)), seed),
        *teacher_rows,
    ]
    staged = stage_full_state_examples(
        working_config,
        budget,
        budget_id=budget_id,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )
    staged = inject_governed_teacher_examples(
        staged,
        teacher_rows=teacher_rows,
        teacher_summary=dict_or_empty(teacher_training.get("summary")),
        text_views=text_views,
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
        byte_fallback=True,
    )
    target_vocab = build_target_vocab(
        [str(row.get("solution_body") or "") for row in train_rows] + target_vocab_extension_bodies,
        max_vocab=int(matched_budget.get("max_target_vocab") or 4096),
        target_mode=target_mode,
    )
    semantic_plan_cfg = semantic_plan_auxiliary_config(budget, matched_budget)
    semantic_slot_cfg = semantic_slot_auxiliary_config(budget, matched_budget)
    private_vocab_seed_bodies = [
        str(row.get("solution_body") or "")
        for row in train_rows
        if str(row.get("solution_body") or "").strip()
    ]
    semantic_plan_vocab_summary = extend_target_vocab_with_semantic_plan_tokens(
        target_vocab,
        [str(row.get("body") or "") for row in list(staged.get("examples") or [])]
        + target_vocab_extension_bodies
        + private_vocab_seed_bodies,
        enabled=bool(semantic_plan_cfg.get("enabled")),
    )
    semantic_slot_vocab_summary = extend_target_vocab_with_semantic_slot_tokens(
        target_vocab,
        [str(row.get("body") or "") for row in list(staged.get("examples") or [])]
        + target_vocab_extension_bodies
        + private_vocab_seed_bodies,
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
    coupled_constructor_cfg = dict_or_empty(budget.get("coupled_state_body_constructor"))
    coupled_state_body_constructor = bool(coupled_constructor_cfg.get("enabled"))
    coupled_state_body_constructor_scale = max(
        0.0,
        float(coupled_constructor_cfg.get("scale") if coupled_constructor_cfg.get("scale") is not None else 0.35),
    )
    executable_span_cfg = dict_or_empty(budget.get("body_executable_span_auxiliary"))
    executable_constructor_cfg = dict_or_empty(budget.get("executable_span_body_constructor"))
    body_executable_span_head = bool(executable_span_cfg.get("enabled")) or bool(executable_constructor_cfg.get("enabled"))
    executable_span_body_constructor = bool(executable_constructor_cfg.get("enabled"))
    executable_span_body_constructor_scale = max(
        0.0,
        float(
            executable_constructor_cfg.get("scale")
            if executable_constructor_cfg.get("scale") is not None
            else 0.25
        ),
    )
    specialist_core_cfg = normalize_specialist_core_config(budget.get("specialist_core"))
    specialist_core_estimate = specialist_core_parameter_estimate(
        int(dims.get("d_model") or 1), specialist_core_cfg
    )
    specialist_token_expert_ids, specialist_router_supervision = specialist_token_expert_map(
        target_vocab, specialist_core_cfg
    )
    auxiliary_head_policy = str(
        budget.get("auxiliary_head_policy") or "legacy_materialized_v1"
    )
    output_projection_policy = str(
        budget.get("output_projection_policy") or "independent_output_v1"
    )
    model = MlxStrictGenerator(
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        max_source_len=max_source,
        max_target_len=max_target,
        coupled_state_body_constructor=coupled_state_body_constructor,
        coupled_state_body_constructor_scale=coupled_state_body_constructor_scale,
        body_executable_span_head=body_executable_span_head,
        executable_span_body_constructor=executable_span_body_constructor,
        executable_span_body_constructor_scale=executable_span_body_constructor_scale,
        semantic_slot_role_count=len(SEMANTIC_SLOT_ROLES),
        semantic_slot_head=bool(semantic_slot_cfg.get("enabled")),
        body_action_role_count=len(BODY_ACTION_ROLES),
        body_operand_role_count=len(BODY_OPERAND_ROLES),
        body_state_event_role_count=len(BODY_STATE_EVENT_ROLES),
        body_executable_span_role_count=len(BODY_EXECUTABLE_SPAN_ROLES),
        auxiliary_head_policy=auxiliary_head_policy,
        output_projection_policy=output_projection_policy,
        specialist_core=specialist_core_cfg,
        specialist_token_expert_ids=specialist_token_expert_ids,
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
    specialist_routing_before = evaluate_specialist_routing_mlx(
        model,
        eval_source_rows,
        eval_target_rows,
        target_vocab=target_vocab,
        batch_size=batch_size,
        pad_id=pad_id,
        mx=mx,
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
        "auxiliary_head_policy": auxiliary_head_policy,
        "output_projection_policy": output_projection_policy,
        "semantic_slot_head_materialized": bool(semantic_slot_cfg.get("enabled")),
        "specialist_core": specialist_core_estimate,
        "specialist_router_supervision": {
            **specialist_router_supervision,
            "token_expert_ids": specialist_token_expert_ids,
        },
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
        "coupled_state_body_constructor": {
            "enabled": coupled_state_body_constructor,
            "policy": (
                "predicted_state_event_conditioned_body_constructor_v1"
                if coupled_state_body_constructor
                else "not_enabled"
            ),
            "scale": coupled_state_body_constructor_scale if coupled_state_body_constructor else 0.0,
            "event_role_count": len(BODY_STATE_EVENT_ROLES),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
            "score_semantics": (
                "Opt-in architecture coupling: the model's own predicted state-event distribution "
                "projects back into the body hidden state before token/action/operand/transition heads. "
                "It does not receive target event labels during generation and grants no credit without "
                "strict decode/verifier behavior."
            ),
        },
        "body_executable_span_auxiliary": {
            "enabled": body_executable_span_head,
            "policy": "private_executable_span_head_requested_v1" if body_executable_span_head else "not_enabled",
            "role_count": len(BODY_EXECUTABLE_SPAN_ROLES),
            "roles": list(BODY_EXECUTABLE_SPAN_ROLES),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
            "score_semantics": (
                "Optional private executable-span role head over target-body positions. It is constructed "
                "only when requested by a governed budget or adaptation run and emits no candidate by itself."
            ),
        },
        "executable_span_body_constructor": {
            "enabled": executable_span_body_constructor,
            "policy": (
                "predicted_executable_span_conditioned_body_constructor_v1"
                if executable_span_body_constructor
                else "not_enabled"
            ),
            "scale": executable_span_body_constructor_scale if executable_span_body_constructor else 0.0,
            "span_role_count": len(BODY_EXECUTABLE_SPAN_ROLES),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
            "score_semantics": (
                "Opt-in architecture coupling: the model's own predicted executable-span distribution "
                "projects into the body hidden state before body token/action/operand/transition logits. "
                "It receives no target span labels at generation time and grants no credit without strict replay."
            ),
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
    specialist_routing = evaluate_specialist_routing_mlx(
        model,
        eval_source_rows,
        eval_target_rows,
        target_vocab=target_vocab,
        batch_size=batch_size,
        pad_id=pad_id,
        mx=mx,
    )
    update_summary = parameter_update_summary(model, before, mlx_utils, mx)
    active_optional_roots: set[str] = set()
    if semantic_plan_active and auxiliary_head_policy == "legacy_materialized_v1":
        active_optional_roots.add("plan_router")
    if semantic_slot_active:
        active_optional_roots.add(
            "slot_role_router"
            if auxiliary_head_policy == "shared_factorized_on_demand_v1"
            else "slot_router"
        )
    if coupled_state_body_constructor:
        active_optional_roots.update({"body_state_event_router", "body_state_event_to_hidden"})
    if body_executable_span_head:
        active_optional_roots.add("body_executable_span_router")
    if executable_span_body_constructor:
        active_optional_roots.add("body_executable_span_to_hidden")
    active_accounting = active_parameter_accounting(
        update_summary,
        specialist_core_estimate,
        active_optional_roots=active_optional_roots,
    )
    model_active_parameters = int(active_accounting["model_active_parameter_count_per_token"])
    specialist_core_estimate.update(active_accounting)
    model.save_weights(str(checkpoint_path))
    write_json(vocab_path, vocab_payload)
    mx.eval(model.parameters())
    source_nonpad = sum(source_nonpad_by_row)
    target_nonpad = sum(target_nonpad_by_row)
    data_exposure = model_data_exposure_summary(
        one_pass_source_token_positions=source_nonpad,
        one_pass_target_token_positions=target_nonpad,
        optimizer_token_positions=optimizer_token_positions,
        active_parameter_count=model_active_parameters,
    )
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
        "auxiliary_head_policy": auxiliary_head_policy,
        "output_projection_policy": output_projection_policy,
        "semantic_slot_head_materialized": bool(semantic_slot_cfg.get("enabled")),
        "specialist_core": specialist_core_estimate,
        "specialist_router_supervision": specialist_router_supervision,
        "specialist_routing_before": specialist_routing_before,
        "specialist_routing": specialist_routing,
        "source_vocab_size": len(source_vocab),
        "target_vocab_size": len(target_vocab),
        "max_source": max_source,
        "max_target": max_target,
        "source_vocab_sha256": stable_hash(json.dumps(source_vocab, sort_keys=True)),
        "target_vocab_sha256": stable_hash(json.dumps(target_vocab, sort_keys=True)),
        "target_mode": target_mode,
        "family_disjoint_evidence": family_disjoint_evidence,
        "family_disjoint_holdout_exclusion": family_holdout_exclusion,
        "checkpoint_training_lineage": family_disjoint_lineage,
        "source_vocab_extension": full_state_source_vocab_extension_summary(source_vocab_extension_texts),
        "target_vocab_extension": full_state_target_vocab_extension_summary(target_vocab_extension_bodies),
        "row_summary": dict_or_empty(rows.get("summary")),
        "teacher_training": dict_or_empty(staged.get("teacher_training")),
        "data_exposure": data_exposure,
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
        "coupled_state_body_constructor": {
            "enabled": coupled_state_body_constructor,
            "policy": (
                "predicted_state_event_conditioned_body_constructor_v1"
                if coupled_state_body_constructor
                else "not_enabled"
            ),
            "scale": coupled_state_body_constructor_scale if coupled_state_body_constructor else 0.0,
            "event_role_count": len(BODY_STATE_EVENT_ROLES),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
            "score_semantics": (
                "Opt-in architecture coupling: predicted state-event probabilities are projected into "
                "a shared body-state representation before token/action/operand/transition logits. "
                "This emits no candidate by itself and does not inspect tests, solutions, public "
                "benchmark payloads, tools, templates, or fallback bodies."
            ),
        },
        "body_executable_span_auxiliary": {
            "enabled": body_executable_span_head,
            "policy": "private_executable_span_head_requested_v1" if body_executable_span_head else "not_enabled",
            "role_count": len(BODY_EXECUTABLE_SPAN_ROLES),
            "roles": list(BODY_EXECUTABLE_SPAN_ROLES),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
            "score_semantics": (
                "Optional private executable-span role head over target-body positions. It is constructed "
                "only when requested by a governed budget or adaptation run and emits no candidate by itself."
            ),
        },
        "executable_span_body_constructor": {
            "enabled": executable_span_body_constructor,
            "policy": (
                "predicted_executable_span_conditioned_body_constructor_v1"
                if executable_span_body_constructor
                else "not_enabled"
            ),
            "scale": executable_span_body_constructor_scale if executable_span_body_constructor else 0.0,
            "span_role_count": len(BODY_EXECUTABLE_SPAN_ROLES),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
            "score_semantics": (
                "Opt-in architecture coupling: predicted executable-span probabilities are projected into "
                "a shared body-state representation before token/action/operand/transition logits. "
                "This emits no candidate by itself and does not inspect tests, solutions, public "
                "benchmark payloads, tools, templates, or fallback bodies."
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
        "core_parameter_update_fraction": update_summary["core_parameter_update_fraction"],
        "core_parameter_tensor_update_fraction": update_summary["core_parameter_tensor_update_fraction"],
        "parameter_update_summary": update_summary,
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


def model_data_exposure_summary(
    *,
    one_pass_source_token_positions: int,
    one_pass_target_token_positions: int,
    optimizer_token_positions: int,
    active_parameter_count: int,
) -> dict[str, Any]:
    one_pass_total = max(0, int(one_pass_source_token_positions)) + max(0, int(one_pass_target_token_positions))
    ratio_to_active = one_pass_total / max(1, int(active_parameter_count))
    repetition = max(0, int(optimizer_token_positions)) / max(1, one_pass_total)
    if ratio_to_active >= 10.0:
        state = "scaling_runway"
    elif ratio_to_active >= 1.0:
        state = "minimum_unique_exposure"
    else:
        state = "underdata"
    return {
        "policy": "strict_generator_unique_data_exposure_v1",
        "one_pass_source_token_positions": int(one_pass_source_token_positions),
        "one_pass_target_token_positions": int(one_pass_target_token_positions),
        "one_pass_total_token_positions": one_pass_total,
        "active_parameter_count": int(active_parameter_count),
        "one_pass_tokens_per_active_parameter": round(ratio_to_active, 6),
        "optimizer_token_positions": int(optimizer_token_positions),
        "optimizer_repetition_factor": round(repetition, 6),
        "data_scale_state": state,
        "optimizer_repetition_counted_as_unique_data": False,
        "score_semantics": (
            "One-pass positions count each selected source/target row once after corpus deduplication. "
            "Optimizer epochs and repeated shuffled windows are reported only as repetition and never increase data-scale credit."
        ),
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


def body_transition_loss_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    transition_weights: Any,
    mx: Any,
    nn: Any,
) -> Any:
    if not hasattr(model, "body_transition_logits"):
        return mx.array(0.0, dtype=mx.float32)
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    logits = model.body_transition_logits(src, tgt_in)
    losses = nn.losses.cross_entropy(logits, tgt_out, reduction="none")
    valid = (tgt_out != pad_id).astype(mx.float32) * transition_weights[:, 1:]
    return mx.sum(losses * valid) / mx.maximum(mx.sum(valid), mx.array(1.0, dtype=mx.float32))


def body_action_loss_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    action_targets: Any,
    action_weights: Any,
    mx: Any,
    nn: Any,
) -> Any:
    if not hasattr(model, "body_action_logits"):
        return mx.array(0.0, dtype=mx.float32)
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    logits = model.body_action_logits(src, tgt_in)
    targets = action_targets[:, 1:]
    losses = nn.losses.cross_entropy(logits, targets, reduction="none")
    valid = (tgt_out != pad_id).astype(mx.float32) * action_weights[:, 1:]
    return mx.sum(losses * valid) / mx.maximum(mx.sum(valid), mx.array(1.0, dtype=mx.float32))


def body_operand_loss_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    operand_targets: Any,
    operand_weights: Any,
    mx: Any,
    nn: Any,
) -> Any:
    if not hasattr(model, "body_operand_logits"):
        return mx.array(0.0, dtype=mx.float32)
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    logits = model.body_operand_logits(src, tgt_in)
    targets = operand_targets[:, 1:]
    losses = nn.losses.cross_entropy(logits, targets, reduction="none")
    valid = (tgt_out != pad_id).astype(mx.float32) * operand_weights[:, 1:]
    return mx.sum(losses * valid) / mx.maximum(mx.sum(valid), mx.array(1.0, dtype=mx.float32))


def body_state_event_role_id_for_roles(action_role: str, operand_role: str) -> int:
    event_role = (
        ACTION_STATE_EVENT_ROLE_BY_ACTION_ROLE.get(str(action_role or ""))
        or OPERAND_STATE_EVENT_ROLE_BY_OPERAND_ROLE.get(str(operand_role or ""))
        or "none"
    )
    return int(BODY_STATE_EVENT_ROLE_TO_ID.get(event_role, 0))


def body_state_event_role_id_for_token(
    token_text: str,
    *,
    allowed_names: set[str] | None = None,
    generated_tokens: list[str] | None = None,
    prefix_context: dict[str, set[str]] | None = None,
) -> int:
    """Map a candidate token to the coarse state-machine event role.

    This is used only for task-blind probability biasing from a learned
    auxiliary head. It derives the role from the same token-local action and
    operand classifiers used by the training targets; it does not inspect
    solutions, tests, verifier labels, or benchmark metadata.
    """

    action_role_id = body_action_role_id_for_token(str(token_text or ""))
    action_role = BODY_ACTION_ROLES[action_role_id] if 0 <= int(action_role_id) < len(BODY_ACTION_ROLES) else "other"
    operand_role_id = body_operand_role_id_for_token(
        str(token_text or ""),
        allowed_names=allowed_names,
        generated_tokens=generated_tokens,
        prefix_context=prefix_context,
    )
    operand_role = (
        BODY_OPERAND_ROLES[operand_role_id]
        if 0 <= int(operand_role_id) < len(BODY_OPERAND_ROLES)
        else "other_operand"
    )
    return body_state_event_role_id_for_roles(action_role, operand_role)


def body_state_event_target_rows(
    action_target_rows: list[list[int]],
    operand_target_rows: list[list[int]],
    action_weight_rows: list[list[float]],
    operand_weight_rows: list[list[float]],
    *,
    event_weight: float = 1.0,
    none_weight: float = 0.20,
) -> tuple[list[list[int]], list[list[float]], dict[str, Any]]:
    event_weight = max(0.0, float(event_weight if event_weight is not None else 1.0))
    none_weight = max(0.0, float(none_weight if none_weight is not None else 0.20))
    target_rows: list[list[int]] = []
    weight_rows: list[list[float]] = []
    active_positions = 0
    event_positions = 0
    none_positions = 0
    role_counts = {role: 0 for role in BODY_STATE_EVENT_ROLES}
    row_count = max(len(action_target_rows), len(operand_target_rows), len(action_weight_rows), len(operand_weight_rows))
    for row_index in range(row_count):
        width = max(
            len(action_target_rows[row_index]) if row_index < len(action_target_rows) else 0,
            len(operand_target_rows[row_index]) if row_index < len(operand_target_rows) else 0,
            len(action_weight_rows[row_index]) if row_index < len(action_weight_rows) else 0,
            len(operand_weight_rows[row_index]) if row_index < len(operand_weight_rows) else 0,
        )
        target_row: list[int] = []
        weight_row: list[float] = []
        for pos in range(width):
            action_id = (
                int(action_target_rows[row_index][pos])
                if row_index < len(action_target_rows) and pos < len(action_target_rows[row_index])
                else 0
            )
            operand_id = (
                int(operand_target_rows[row_index][pos])
                if row_index < len(operand_target_rows) and pos < len(operand_target_rows[row_index])
                else 0
            )
            action_role = BODY_ACTION_ROLES[action_id] if 0 <= action_id < len(BODY_ACTION_ROLES) else "other"
            operand_role = BODY_OPERAND_ROLES[operand_id] if 0 <= operand_id < len(BODY_OPERAND_ROLES) else "other"
            event_id = body_state_event_role_id_for_roles(action_role, operand_role)
            action_weight = (
                float(action_weight_rows[row_index][pos] or 0.0)
                if row_index < len(action_weight_rows) and pos < len(action_weight_rows[row_index])
                else 0.0
            )
            operand_weight = (
                float(operand_weight_rows[row_index][pos] or 0.0)
                if row_index < len(operand_weight_rows) and pos < len(operand_weight_rows[row_index])
                else 0.0
            )
            base_weight = max(action_weight, operand_weight)
            if base_weight > 0.0:
                active_positions += 1
                role = BODY_STATE_EVENT_ROLES[event_id] if 0 <= event_id < len(BODY_STATE_EVENT_ROLES) else "none"
                role_counts[role] = role_counts.get(role, 0) + 1
                if event_id == 0:
                    none_positions += 1
                    weight = base_weight * none_weight
                else:
                    event_positions += 1
                    weight = base_weight * event_weight
            else:
                weight = 0.0
            target_row.append(event_id)
            weight_row.append(round(float(weight), 6))
        target_rows.append(target_row)
        weight_rows.append(weight_row)
    return target_rows, weight_rows, {
        "enabled": active_positions > 0,
        "policy": "private_body_state_event_target_rows_v1",
        "active_positions": active_positions,
        "event_positions": event_positions,
        "none_positions": none_positions,
        "event_position_rate": round(event_positions / max(1, active_positions), 6),
        "event_weight": event_weight,
        "none_weight": none_weight,
        "role_counts": {role: count for role, count in role_counts.items() if count},
        "candidate_generation_credit": 0,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
    }


def body_state_event_loss_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    event_targets: Any,
    event_weights: Any,
    mx: Any,
    nn: Any,
) -> Any:
    if not hasattr(model, "body_state_event_logits"):
        return mx.array(0.0, dtype=mx.float32)
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    logits = model.body_state_event_logits(src, tgt_in)
    targets = event_targets[:, 1:]
    losses = nn.losses.cross_entropy(logits, targets, reduction="none")
    valid = (tgt_out != pad_id).astype(mx.float32) * event_weights[:, 1:]
    return mx.sum(losses * valid) / mx.maximum(mx.sum(valid), mx.array(1.0, dtype=mx.float32))


def body_executable_span_role_for_roles(action_role: str, operand_role: str, event_role: str) -> str:
    action = str(action_role or "")
    operand = str(operand_role or "")
    event = str(event_role or "")
    if action == "ignore" and operand == "ignore" and event == "none":
        return "ignore"
    if event == "return_finalizer" or action in {"return", "eos"} or operand in {"return_keyword", "eos"}:
        return "return_finalizer_span"
    if event == "control_transition" or action in {"loop", "branch", "block_enter", "comparison", "bool_op"}:
        return "guard_control_span"
    if event == "traversal_or_call" or action in {"call_or_method", "attribute"} or operand in {
        "builtin_function",
        "method_name",
        "attribute_name",
    }:
        return "traversal_call_span"
    if event == "state_update" or action in {"assignment", "update_operator"} or operand == "assignment_operator":
        return "state_update_span"
    if event == "value_expression" or action in {"open_expr", "close_expr"} or operand in {
        "arithmetic_operator",
        "comparison_operator",
        "boolean_operator",
    }:
        return "value_expression_span"
    if operand in {"visible_parameter", "loop_variable", "local_state"}:
        return "state_reference_span"
    if operand == "literal_value" or action == "literal":
        return "literal_span"
    if operand in {"call_delimiter", "index_delimiter", "punctuation"}:
        return "delimiter_span"
    if event == "statement_boundary" or action in {"block_exit", "line_boundary"} or operand == "statement_boundary":
        return "statement_boundary_span"
    return "other_span"


def body_executable_span_target_rows(
    action_target_rows: list[list[int]],
    operand_target_rows: list[list[int]],
    event_target_rows: list[list[int]],
    action_weight_rows: list[list[float]],
    operand_weight_rows: list[list[float]],
    event_weight_rows: list[list[float]],
) -> tuple[list[list[int]], list[list[float]], dict[str, Any]]:
    target_rows: list[list[int]] = []
    weight_rows: list[list[float]] = []
    active_positions = 0
    role_counts = {role: 0 for role in BODY_EXECUTABLE_SPAN_ROLES}
    row_count = max(
        len(action_target_rows),
        len(operand_target_rows),
        len(event_target_rows),
        len(action_weight_rows),
        len(operand_weight_rows),
        len(event_weight_rows),
    )
    for row_index in range(row_count):
        width = max(
            len(action_target_rows[row_index]) if row_index < len(action_target_rows) else 0,
            len(operand_target_rows[row_index]) if row_index < len(operand_target_rows) else 0,
            len(event_target_rows[row_index]) if row_index < len(event_target_rows) else 0,
            len(action_weight_rows[row_index]) if row_index < len(action_weight_rows) else 0,
            len(operand_weight_rows[row_index]) if row_index < len(operand_weight_rows) else 0,
            len(event_weight_rows[row_index]) if row_index < len(event_weight_rows) else 0,
        )
        target_row: list[int] = []
        weight_row: list[float] = []
        for pos in range(width):
            action_id = (
                int(action_target_rows[row_index][pos])
                if row_index < len(action_target_rows) and pos < len(action_target_rows[row_index])
                else 0
            )
            operand_id = (
                int(operand_target_rows[row_index][pos])
                if row_index < len(operand_target_rows) and pos < len(operand_target_rows[row_index])
                else 0
            )
            event_id = (
                int(event_target_rows[row_index][pos])
                if row_index < len(event_target_rows) and pos < len(event_target_rows[row_index])
                else 0
            )
            action_role = BODY_ACTION_ROLES[action_id] if 0 <= action_id < len(BODY_ACTION_ROLES) else "other"
            operand_role = BODY_OPERAND_ROLES[operand_id] if 0 <= operand_id < len(BODY_OPERAND_ROLES) else "other"
            event_role = BODY_STATE_EVENT_ROLES[event_id] if 0 <= event_id < len(BODY_STATE_EVENT_ROLES) else "none"
            role = body_executable_span_role_for_roles(action_role, operand_role, event_role)
            role_id = BODY_EXECUTABLE_SPAN_ROLE_TO_ID.get(role, BODY_EXECUTABLE_SPAN_ROLE_TO_ID["other_span"])
            base_weight = max(
                float(action_weight_rows[row_index][pos] or 0.0)
                if row_index < len(action_weight_rows) and pos < len(action_weight_rows[row_index])
                else 0.0,
                float(operand_weight_rows[row_index][pos] or 0.0)
                if row_index < len(operand_weight_rows) and pos < len(operand_weight_rows[row_index])
                else 0.0,
                float(event_weight_rows[row_index][pos] or 0.0)
                if row_index < len(event_weight_rows) and pos < len(event_weight_rows[row_index])
                else 0.0,
            )
            if base_weight > 0.0 and role != "ignore":
                active_positions += 1
                role_counts[role] = role_counts.get(role, 0) + 1
                weight = base_weight
            else:
                role_id = BODY_EXECUTABLE_SPAN_ROLE_TO_ID["ignore"]
                weight = 0.0
            target_row.append(role_id)
            weight_row.append(round(float(weight), 6))
        target_rows.append(target_row)
        weight_rows.append(weight_row)
    return target_rows, weight_rows, {
        "enabled": active_positions > 0,
        "policy": "private_executable_span_target_rows_v1",
        "rows": row_count,
        "role_count": len(BODY_EXECUTABLE_SPAN_ROLES),
        "roles": list(BODY_EXECUTABLE_SPAN_ROLES),
        "active_positions": active_positions,
        "role_counts": {role: count for role, count in role_counts.items() if count},
        "label_source": "admitted_private_body_action_operand_event_roles_only",
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
        "score_semantics": (
            "Maps admitted private/licensed body tokens into executable span roles such as guard/control, "
            "traversal/call, state update, value expression, return/finalizer, and state reference. "
            "Labels are derived from existing private action/operand/event role targets; no tests, "
            "solutions, public benchmark payloads, tools, templates, or fallback bodies are consulted."
        ),
    }


def body_executable_span_loss_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    span_targets: Any,
    span_weights: Any,
    mx: Any,
    nn: Any,
) -> Any:
    if not hasattr(model, "body_executable_span_logits"):
        return mx.array(0.0, dtype=mx.float32)
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    logits = model.body_executable_span_logits(src, tgt_in)
    targets = span_targets[:, 1:]
    losses = nn.losses.cross_entropy(logits, targets, reduction="none")
    valid = (tgt_out != pad_id).astype(mx.float32) * span_weights[:, 1:]
    return mx.sum(losses * valid) / mx.maximum(mx.sum(valid), mx.array(1.0, dtype=mx.float32))


def body_state_event_consistency_loss_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    event_targets: Any,
    event_weights: Any,
    action_consistency_weight: float,
    operand_consistency_weight: float,
    mx: Any,
) -> Any:
    """Softly couple action/operand heads to the state-event target.

    For an event such as ``state_update`` or ``return_finalizer``, this loss
    rewards probability mass on action/operand roles compatible with that event
    instead of forcing a single exact role. It is a private target-side
    auxiliary objective only; it emits no candidates and cannot count as
    learned-generation evidence by itself.
    """

    if not hasattr(model, "body_action_logits") or not hasattr(model, "body_operand_logits"):
        return mx.array(0.0, dtype=mx.float32)
    action_weight = max(0.0, float(action_consistency_weight or 0.0))
    operand_weight = max(0.0, float(operand_consistency_weight or 0.0))
    if action_weight <= 0.0 and operand_weight <= 0.0:
        return mx.array(0.0, dtype=mx.float32)
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    targets = event_targets[:, 1:]
    base_valid = (tgt_out != pad_id).astype(mx.float32) * event_weights[:, 1:]
    eps = mx.array(1e-9, dtype=mx.float32)
    total = mx.array(0.0, dtype=mx.float32)
    if action_weight > 0.0:
        action_masks = mx.array(BODY_STATE_EVENT_ACTION_COMPATIBILITY_MASK, dtype=mx.float32)
        action_compat = mx.take(action_masks, targets, axis=0)
        action_valid = base_valid * (mx.sum(action_compat, axis=-1) > 0.0).astype(mx.float32)
        action_probs = mx.softmax(model.body_action_logits(src, tgt_in), axis=-1)
        action_mass = mx.sum(action_probs * action_compat, axis=-1)
        action_losses = -mx.log(mx.maximum(action_mass, eps))
        total = total + (
            action_weight
            * mx.sum(action_losses * action_valid)
            / mx.maximum(mx.sum(action_valid), mx.array(1.0, dtype=mx.float32))
        )
    if operand_weight > 0.0:
        operand_masks = mx.array(BODY_STATE_EVENT_OPERAND_COMPATIBILITY_MASK, dtype=mx.float32)
        operand_compat = mx.take(operand_masks, targets, axis=0)
        operand_valid = base_valid * (mx.sum(operand_compat, axis=-1) > 0.0).astype(mx.float32)
        operand_probs = mx.softmax(model.body_operand_logits(src, tgt_in), axis=-1)
        operand_mass = mx.sum(operand_probs * operand_compat, axis=-1)
        operand_losses = -mx.log(mx.maximum(operand_mass, eps))
        total = total + (
            operand_weight
            * mx.sum(operand_losses * operand_valid)
            / mx.maximum(mx.sum(operand_valid), mx.array(1.0, dtype=mx.float32))
        )
    return total


def body_transition_aux_weighted_loss_fn_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    token_weights: Any,
    transition_weights: Any,
    body_transition_weight: float,
    mx: Any,
    nn: Any,
) -> Any:
    loss = weighted_loss_fn_mlx(model, src, tgt, pad_id, token_weights, mx, nn)
    if float(body_transition_weight or 0.0) > 0.0:
        loss = loss + (
            float(body_transition_weight)
            * body_transition_loss_mlx(model, src, tgt, pad_id, transition_weights, mx, nn)
        )
    return loss


def body_action_transition_aux_weighted_loss_fn_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    token_weights: Any,
    transition_weights: Any,
    body_transition_weight: float,
    action_targets: Any,
    action_weights: Any,
    body_action_weight: float,
    mx: Any,
    nn: Any,
) -> Any:
    loss = body_transition_aux_weighted_loss_fn_mlx(
        model,
        src,
        tgt,
        pad_id,
        token_weights,
        transition_weights,
        body_transition_weight,
        mx,
        nn,
    )
    if float(body_action_weight or 0.0) > 0.0:
        loss = loss + (
            float(body_action_weight)
            * body_action_loss_mlx(model, src, tgt, pad_id, action_targets, action_weights, mx, nn)
        )
    return loss


def body_action_operand_transition_aux_weighted_loss_fn_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    token_weights: Any,
    transition_weights: Any,
    body_transition_weight: float,
    action_targets: Any,
    action_weights: Any,
    body_action_weight: float,
    operand_targets: Any,
    operand_weights: Any,
    body_operand_weight: float,
    mx: Any,
    nn: Any,
) -> Any:
    loss = body_action_transition_aux_weighted_loss_fn_mlx(
        model,
        src,
        tgt,
        pad_id,
        token_weights,
        transition_weights,
        body_transition_weight,
        action_targets,
        action_weights,
        body_action_weight,
        mx,
        nn,
    )
    if float(body_operand_weight or 0.0) > 0.0:
        loss = loss + (
            float(body_operand_weight)
            * body_operand_loss_mlx(model, src, tgt, pad_id, operand_targets, operand_weights, mx, nn)
        )
    return loss


def body_action_operand_transition_event_aux_weighted_loss_fn_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    token_weights: Any,
    transition_weights: Any,
    body_transition_weight: float,
    action_targets: Any,
    action_weights: Any,
    body_action_weight: float,
    operand_targets: Any,
    operand_weights: Any,
    body_operand_weight: float,
    event_targets: Any,
    event_weights: Any,
    body_state_event_weight: float,
    mx: Any,
    nn: Any,
    body_state_event_action_consistency_weight: float = 0.0,
    body_state_event_operand_consistency_weight: float = 0.0,
) -> Any:
    loss = body_action_operand_transition_aux_weighted_loss_fn_mlx(
        model,
        src,
        tgt,
        pad_id,
        token_weights,
        transition_weights,
        body_transition_weight,
        action_targets,
        action_weights,
        body_action_weight,
        operand_targets,
        operand_weights,
        body_operand_weight,
        mx,
        nn,
    )
    if float(body_state_event_weight or 0.0) > 0.0:
        loss = loss + (
            float(body_state_event_weight)
            * body_state_event_loss_mlx(model, src, tgt, pad_id, event_targets, event_weights, mx, nn)
        )
    if float(body_state_event_action_consistency_weight or 0.0) > 0.0 or float(body_state_event_operand_consistency_weight or 0.0) > 0.0:
        loss = loss + body_state_event_consistency_loss_mlx(
            model,
            src,
            tgt,
            pad_id,
            event_targets,
            event_weights,
            float(body_state_event_action_consistency_weight),
            float(body_state_event_operand_consistency_weight),
            mx,
        )
    return loss


def body_action_operand_transition_event_span_aux_weighted_loss_fn_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    token_weights: Any,
    transition_weights: Any,
    body_transition_weight: float,
    action_targets: Any,
    action_weights: Any,
    body_action_weight: float,
    operand_targets: Any,
    operand_weights: Any,
    body_operand_weight: float,
    event_targets: Any,
    event_weights: Any,
    body_state_event_weight: float,
    span_targets: Any,
    span_weights: Any,
    body_executable_span_weight: float,
    mx: Any,
    nn: Any,
    body_state_event_action_consistency_weight: float = 0.0,
    body_state_event_operand_consistency_weight: float = 0.0,
) -> Any:
    loss = body_action_operand_transition_event_aux_weighted_loss_fn_mlx(
        model,
        src,
        tgt,
        pad_id,
        token_weights,
        transition_weights,
        body_transition_weight,
        action_targets,
        action_weights,
        body_action_weight,
        operand_targets,
        operand_weights,
        body_operand_weight,
        event_targets,
        event_weights,
        body_state_event_weight,
        mx,
        nn,
        body_state_event_action_consistency_weight,
        body_state_event_operand_consistency_weight,
    )
    if float(body_executable_span_weight or 0.0) > 0.0:
        loss = loss + (
            float(body_executable_span_weight)
            * body_executable_span_loss_mlx(model, src, tgt, pad_id, span_targets, span_weights, mx, nn)
        )
    return loss


def semantic_body_transition_aux_weighted_loss_fn_mlx(
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
    transition_weights: Any,
    body_transition_weight: float,
    mx: Any,
    nn: Any,
    slot_role_class_ids: list[Any] | None = None,
    body_state_event_action_consistency_weight: float = 0.0,
    body_state_event_operand_consistency_weight: float = 0.0,
) -> Any:
    loss = semantic_aux_weighted_loss_fn_mlx(
        model,
        src,
        tgt,
        pad_id,
        token_weights,
        plan_targets,
        plan_sample_weights,
        semantic_plan_weight,
        slot_targets,
        slot_sample_weights,
        semantic_slot_weight,
        mx,
        nn,
        slot_role_class_ids=slot_role_class_ids,
    )
    if float(body_transition_weight or 0.0) > 0.0:
        loss = loss + (
            float(body_transition_weight)
            * body_transition_loss_mlx(model, src, tgt, pad_id, transition_weights, mx, nn)
        )
    return loss


def semantic_body_action_transition_aux_weighted_loss_fn_mlx(
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
    transition_weights: Any,
    body_transition_weight: float,
    action_targets: Any,
    action_weights: Any,
    body_action_weight: float,
    mx: Any,
    nn: Any,
    slot_role_class_ids: list[Any] | None = None,
) -> Any:
    loss = semantic_body_transition_aux_weighted_loss_fn_mlx(
        model,
        src,
        tgt,
        pad_id,
        token_weights,
        plan_targets,
        plan_sample_weights,
        semantic_plan_weight,
        slot_targets,
        slot_sample_weights,
        semantic_slot_weight,
        transition_weights,
        body_transition_weight,
        mx,
        nn,
        slot_role_class_ids=slot_role_class_ids,
    )
    if float(body_action_weight or 0.0) > 0.0:
        loss = loss + (
            float(body_action_weight)
            * body_action_loss_mlx(model, src, tgt, pad_id, action_targets, action_weights, mx, nn)
        )
    return loss


def semantic_body_action_operand_transition_aux_weighted_loss_fn_mlx(
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
    transition_weights: Any,
    body_transition_weight: float,
    action_targets: Any,
    action_weights: Any,
    body_action_weight: float,
    operand_targets: Any,
    operand_weights: Any,
    body_operand_weight: float,
    mx: Any,
    nn: Any,
    slot_role_class_ids: list[Any] | None = None,
) -> Any:
    loss = semantic_body_action_transition_aux_weighted_loss_fn_mlx(
        model,
        src,
        tgt,
        pad_id,
        token_weights,
        plan_targets,
        plan_sample_weights,
        semantic_plan_weight,
        slot_targets,
        slot_sample_weights,
        semantic_slot_weight,
        transition_weights,
        body_transition_weight,
        action_targets,
        action_weights,
        body_action_weight,
        mx,
        nn,
        slot_role_class_ids=slot_role_class_ids,
    )
    if float(body_operand_weight or 0.0) > 0.0:
        loss = loss + (
            float(body_operand_weight)
            * body_operand_loss_mlx(model, src, tgt, pad_id, operand_targets, operand_weights, mx, nn)
        )
    return loss


def semantic_body_action_operand_transition_event_aux_weighted_loss_fn_mlx(
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
    transition_weights: Any,
    body_transition_weight: float,
    action_targets: Any,
    action_weights: Any,
    body_action_weight: float,
    operand_targets: Any,
    operand_weights: Any,
    body_operand_weight: float,
    event_targets: Any,
    event_weights: Any,
    body_state_event_weight: float,
    mx: Any,
    nn: Any,
    slot_role_class_ids: list[Any] | None = None,
    body_state_event_action_consistency_weight: float = 0.0,
    body_state_event_operand_consistency_weight: float = 0.0,
) -> Any:
    loss = semantic_body_action_operand_transition_aux_weighted_loss_fn_mlx(
        model,
        src,
        tgt,
        pad_id,
        token_weights,
        plan_targets,
        plan_sample_weights,
        semantic_plan_weight,
        slot_targets,
        slot_sample_weights,
        semantic_slot_weight,
        transition_weights,
        body_transition_weight,
        action_targets,
        action_weights,
        body_action_weight,
        operand_targets,
        operand_weights,
        body_operand_weight,
        mx,
        nn,
        slot_role_class_ids=slot_role_class_ids,
    )
    if float(body_state_event_weight or 0.0) > 0.0:
        loss = loss + (
            float(body_state_event_weight)
            * body_state_event_loss_mlx(model, src, tgt, pad_id, event_targets, event_weights, mx, nn)
        )
    if float(body_state_event_action_consistency_weight or 0.0) > 0.0 or float(body_state_event_operand_consistency_weight or 0.0) > 0.0:
        loss = loss + body_state_event_consistency_loss_mlx(
            model,
            src,
            tgt,
            pad_id,
            event_targets,
            event_weights,
            float(body_state_event_action_consistency_weight),
            float(body_state_event_operand_consistency_weight),
            mx,
        )
    return loss


def semantic_body_action_operand_transition_event_span_aux_weighted_loss_fn_mlx(
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
    transition_weights: Any,
    body_transition_weight: float,
    action_targets: Any,
    action_weights: Any,
    body_action_weight: float,
    operand_targets: Any,
    operand_weights: Any,
    body_operand_weight: float,
    event_targets: Any,
    event_weights: Any,
    body_state_event_weight: float,
    span_targets: Any,
    span_weights: Any,
    body_executable_span_weight: float,
    mx: Any,
    nn: Any,
    slot_role_class_ids: list[Any] | None = None,
    body_state_event_action_consistency_weight: float = 0.0,
    body_state_event_operand_consistency_weight: float = 0.0,
) -> Any:
    loss = semantic_body_action_operand_transition_event_aux_weighted_loss_fn_mlx(
        model,
        src,
        tgt,
        pad_id,
        token_weights,
        plan_targets,
        plan_sample_weights,
        semantic_plan_weight,
        slot_targets,
        slot_sample_weights,
        semantic_slot_weight,
        transition_weights,
        body_transition_weight,
        action_targets,
        action_weights,
        body_action_weight,
        operand_targets,
        operand_weights,
        body_operand_weight,
        event_targets,
        event_weights,
        body_state_event_weight,
        mx,
        nn,
        slot_role_class_ids=slot_role_class_ids,
        body_state_event_action_consistency_weight=body_state_event_action_consistency_weight,
        body_state_event_operand_consistency_weight=body_state_event_operand_consistency_weight,
    )
    if float(body_executable_span_weight or 0.0) > 0.0:
        loss = loss + (
            float(body_executable_span_weight)
            * body_executable_span_loss_mlx(model, src, tgt, pad_id, span_targets, span_weights, mx, nn)
        )
    return loss


def semantic_body_transition_aux_source_contrastive_weighted_loss_fn_mlx(
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
    transition_weights: Any,
    body_transition_weight: float,
    mx: Any,
    nn: Any,
    slot_role_class_ids: list[Any] | None = None,
    body_state_event_action_consistency_weight: float = 0.0,
    body_state_event_operand_consistency_weight: float = 0.0,
) -> Any:
    loss = semantic_aux_source_contrastive_weighted_loss_fn_mlx(
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
        plan_targets,
        plan_sample_weights,
        semantic_plan_weight,
        slot_targets,
        slot_sample_weights,
        semantic_slot_weight,
        mx,
        nn,
        slot_role_class_ids=slot_role_class_ids,
    )
    if float(body_transition_weight or 0.0) > 0.0:
        loss = loss + (
            float(body_transition_weight)
            * body_transition_loss_mlx(model, src, tgt, pad_id, transition_weights, mx, nn)
        )
    return loss


def semantic_body_action_transition_aux_source_contrastive_weighted_loss_fn_mlx(
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
    transition_weights: Any,
    body_transition_weight: float,
    action_targets: Any,
    action_weights: Any,
    body_action_weight: float,
    mx: Any,
    nn: Any,
    slot_role_class_ids: list[Any] | None = None,
) -> Any:
    loss = semantic_body_transition_aux_source_contrastive_weighted_loss_fn_mlx(
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
        plan_targets,
        plan_sample_weights,
        semantic_plan_weight,
        slot_targets,
        slot_sample_weights,
        semantic_slot_weight,
        transition_weights,
        body_transition_weight,
        mx,
        nn,
        slot_role_class_ids=slot_role_class_ids,
    )
    if float(body_action_weight or 0.0) > 0.0:
        loss = loss + (
            float(body_action_weight)
            * body_action_loss_mlx(model, src, tgt, pad_id, action_targets, action_weights, mx, nn)
        )
    return loss


def semantic_body_action_operand_transition_aux_source_contrastive_weighted_loss_fn_mlx(
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
    transition_weights: Any,
    body_transition_weight: float,
    action_targets: Any,
    action_weights: Any,
    body_action_weight: float,
    operand_targets: Any,
    operand_weights: Any,
    body_operand_weight: float,
    mx: Any,
    nn: Any,
    slot_role_class_ids: list[Any] | None = None,
) -> Any:
    loss = semantic_body_action_transition_aux_source_contrastive_weighted_loss_fn_mlx(
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
        plan_targets,
        plan_sample_weights,
        semantic_plan_weight,
        slot_targets,
        slot_sample_weights,
        semantic_slot_weight,
        transition_weights,
        body_transition_weight,
        action_targets,
        action_weights,
        body_action_weight,
        mx,
        nn,
        slot_role_class_ids=slot_role_class_ids,
    )
    if float(body_operand_weight or 0.0) > 0.0:
        loss = loss + (
            float(body_operand_weight)
            * body_operand_loss_mlx(model, src, tgt, pad_id, operand_targets, operand_weights, mx, nn)
        )
    return loss


def semantic_body_action_operand_transition_event_aux_source_contrastive_weighted_loss_fn_mlx(
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
    transition_weights: Any,
    body_transition_weight: float,
    action_targets: Any,
    action_weights: Any,
    body_action_weight: float,
    operand_targets: Any,
    operand_weights: Any,
    body_operand_weight: float,
    event_targets: Any,
    event_weights: Any,
    body_state_event_weight: float,
    mx: Any,
    nn: Any,
    slot_role_class_ids: list[Any] | None = None,
    body_state_event_action_consistency_weight: float = 0.0,
    body_state_event_operand_consistency_weight: float = 0.0,
) -> Any:
    loss = semantic_body_action_operand_transition_aux_source_contrastive_weighted_loss_fn_mlx(
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
        plan_targets,
        plan_sample_weights,
        semantic_plan_weight,
        slot_targets,
        slot_sample_weights,
        semantic_slot_weight,
        transition_weights,
        body_transition_weight,
        action_targets,
        action_weights,
        body_action_weight,
        operand_targets,
        operand_weights,
        body_operand_weight,
        mx,
        nn,
        slot_role_class_ids=slot_role_class_ids,
    )
    if float(body_state_event_weight or 0.0) > 0.0:
        loss = loss + (
            float(body_state_event_weight)
            * body_state_event_loss_mlx(model, src, tgt, pad_id, event_targets, event_weights, mx, nn)
        )
    if float(body_state_event_action_consistency_weight or 0.0) > 0.0 or float(body_state_event_operand_consistency_weight or 0.0) > 0.0:
        loss = loss + body_state_event_consistency_loss_mlx(
            model,
            src,
            tgt,
            pad_id,
            event_targets,
            event_weights,
            float(body_state_event_action_consistency_weight),
            float(body_state_event_operand_consistency_weight),
            mx,
        )
    return loss


def semantic_body_action_operand_transition_event_span_aux_source_contrastive_weighted_loss_fn_mlx(
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
    transition_weights: Any,
    body_transition_weight: float,
    action_targets: Any,
    action_weights: Any,
    body_action_weight: float,
    operand_targets: Any,
    operand_weights: Any,
    body_operand_weight: float,
    event_targets: Any,
    event_weights: Any,
    body_state_event_weight: float,
    span_targets: Any,
    span_weights: Any,
    body_executable_span_weight: float,
    mx: Any,
    nn: Any,
    slot_role_class_ids: list[Any] | None = None,
    body_state_event_action_consistency_weight: float = 0.0,
    body_state_event_operand_consistency_weight: float = 0.0,
) -> Any:
    loss = semantic_body_action_operand_transition_event_aux_source_contrastive_weighted_loss_fn_mlx(
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
        plan_targets,
        plan_sample_weights,
        semantic_plan_weight,
        slot_targets,
        slot_sample_weights,
        semantic_slot_weight,
        transition_weights,
        body_transition_weight,
        action_targets,
        action_weights,
        body_action_weight,
        operand_targets,
        operand_weights,
        body_operand_weight,
        event_targets,
        event_weights,
        body_state_event_weight,
        mx,
        nn,
        slot_role_class_ids=slot_role_class_ids,
        body_state_event_action_consistency_weight=body_state_event_action_consistency_weight,
        body_state_event_operand_consistency_weight=body_state_event_operand_consistency_weight,
    )
    if float(body_executable_span_weight or 0.0) > 0.0:
        loss = loss + (
            float(body_executable_span_weight)
            * body_executable_span_loss_mlx(model, src, tgt, pad_id, span_targets, span_weights, mx, nn)
        )
    return loss


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
    router_aux_loss = mx.array(0.0, dtype=mx.float32)
    if hasattr(model, "forward_with_router_loss"):
        logits, router_aux_loss = model.forward_with_router_loss(src, tgt_in, tgt_out)
    else:
        logits = model(src, tgt_in)
    losses = nn.losses.cross_entropy(logits, tgt_out, reduction="none")
    valid = (tgt_out != pad_id).astype(mx.float32)
    if str(span_mode or "prefix") == "after_body_start" and int(body_start_token_id) >= 0:
        valid = valid * body_start_span_mask_mlx(
            tgt,
            tgt_out,
            int(body_start_token_id),
            mx,
            span_token_count=int(prefix_token_count or 0),
        )
    elif int(prefix_token_count or 0) > 0:
        positions = mx.arange(tgt_out.shape[1])[None, :]
        valid = valid * (positions < int(prefix_token_count)).astype(mx.float32)
    weights = token_weights[:, 1:]
    weighted_valid = valid * weights
    token_loss = mx.sum(losses * weighted_valid) / mx.maximum(
        mx.sum(weighted_valid), mx.array(1.0, dtype=mx.float32)
    )
    return token_loss + router_aux_loss


def body_start_span_mask_mlx(
    tgt: Any,
    tgt_out: Any,
    body_start_token_id: int,
    mx: Any,
    *,
    span_token_count: int = 0,
) -> Any:
    matches = (tgt == int(body_start_token_id)).astype(mx.int32)
    has_body_start = mx.max(matches, axis=1).astype(mx.float32)
    start_positions = mx.argmax(matches, axis=1)
    positions = mx.arange(tgt_out.shape[1])[None, :]
    valid = positions >= start_positions[:, None]
    if int(span_token_count or 0) > 0:
        valid = valid & (positions < (start_positions[:, None] + int(span_token_count)))
    return valid.astype(mx.float32) * has_body_start[:, None]


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


def body_transition_weight_rows(
    target_rows: list[list[int]],
    *,
    target_vocab: dict[str, int],
    target_mode: str,
    transition_boost: float = 1.0,
) -> tuple[list[list[float]], dict[str, Any]]:
    pad_id = int(target_vocab.get("<pad>", 0))
    bos_id = int(target_vocab.get("<bos>", -1))
    eos_id = int(target_vocab.get("<eos>", -2))
    body_start_id = int(target_vocab.get(PLAN_BODY_START_TOKEN, -999999))
    inverse = {int(idx): str(tok) for tok, idx in target_vocab.items()}
    boost = max(1.0, float(transition_boost if transition_boost is not None else 1.0))
    rows: list[list[float]] = []
    active_positions = 0
    line_boundary_positions = 0
    statement_head_positions = 0
    closure_positions = 0
    body_start_rows = 0
    for row in target_rows:
        weights: list[float] = []
        body_active = int(body_start_id) < 0 or str(target_mode or "") == "body_tokens"
        if int(body_start_id) >= 0 and any(int(value) == int(body_start_id) for value in row):
            body_start_rows += 1
            body_active = False
        previous_token = ""
        for pos, value in enumerate(row):
            token_id = int(value)
            token_text = inverse.get(token_id, "")
            if token_id == pad_id or token_id == bos_id:
                weights.append(0.0)
                previous_token = token_text
                continue
            if token_id == body_start_id:
                weights.append(0.0)
                body_active = True
                previous_token = token_text
                continue
            if not body_active:
                weights.append(0.0)
                previous_token = token_text
                continue
            if token_id == eos_id:
                weights.append(boost)
                active_positions += 1
                closure_positions += 1
                previous_token = token_text
                continue
            weight = 1.0
            if previous_token == "NEWLINE:" or pos <= 1:
                weight = max(weight, boost)
                statement_head_positions += 1
            if token_text in {"NEWLINE:", "DEDENT:", "INDENT:", "NAME:return", "NAME:for", "NAME:if", "NAME:while"}:
                weight = max(weight, boost)
                line_boundary_positions += 1
            if token_text in {"OP:)", "OP:]", "OP:}", "<eos>"}:
                weight = max(weight, boost)
                closure_positions += 1
            weights.append(weight)
            active_positions += 1
            previous_token = token_text
        rows.append(weights)
    return rows, {
        "enabled": bool(active_positions),
        "policy": "private_prefix_conditioned_body_transition_weight_rows_v1",
        "rows": len(target_rows),
        "target_mode": str(target_mode or ""),
        "body_start_token": PLAN_BODY_START_TOKEN,
        "body_start_rows": body_start_rows,
        "active_positions": active_positions,
        "statement_head_positions": statement_head_positions,
        "line_boundary_positions": line_boundary_positions,
        "closure_positions": closure_positions,
        "transition_boost": boost,
        "score_semantics": (
            "Weights admitted private/licensed target-body token positions for the prefix-conditioned "
            "body-transition auxiliary head. The labels are the same target next tokens used for normal "
            "supervised training, masked to body-continuation positions after SLOT:BODY_START when present. "
            "This trains a learned prefix-conditioned projection only; it does not render code, inspect "
            "tests/solutions, use public benchmarks, or grant learned-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def body_action_target_rows(
    target_rows: list[list[int]],
    transition_weight_rows: list[list[float]],
    *,
    target_vocab: dict[str, int],
    target_mode: str,
    class_balance_enabled: bool = False,
    class_balance_min_weight: float = 0.35,
    class_balance_max_weight: float = 3.0,
) -> tuple[list[list[int]], list[list[float]], dict[str, Any]]:
    inverse = {int(idx): str(tok) for tok, idx in target_vocab.items()}
    action_rows: list[list[int]] = []
    action_weight_rows: list[list[float]] = []
    active_positions = 0
    role_counts: dict[str, int] = {}
    for row_index, row in enumerate(target_rows):
        weight_row = transition_weight_rows[row_index] if row_index < len(transition_weight_rows) else []
        action_row: list[int] = []
        action_weight_row: list[float] = []
        for pos, value in enumerate(row):
            weight = float(weight_row[pos]) if pos < len(weight_row) else 0.0
            token_text = inverse.get(int(value), "")
            role = body_action_role_for_token_text(token_text)
            role_id = BODY_ACTION_ROLE_TO_ID.get(role, BODY_ACTION_ROLE_TO_ID["other"])
            if weight <= 0.0:
                role_id = BODY_ACTION_ROLE_TO_ID["ignore"]
                action_weight_row.append(0.0)
            else:
                active_positions += 1
                role_counts[role] = role_counts.get(role, 0) + 1
                action_weight_row.append(weight)
            action_row.append(role_id)
        action_rows.append(action_row)
        action_weight_rows.append(action_weight_row)
    balance_factors: dict[str, float] = {}
    balance_enabled = bool(class_balance_enabled and active_positions and role_counts)
    if balance_enabled:
        mean_count = float(active_positions) / max(1.0, float(len(role_counts)))
        min_weight = max(0.0, float(class_balance_min_weight if class_balance_min_weight is not None else 0.35))
        max_weight = max(min_weight, float(class_balance_max_weight if class_balance_max_weight is not None else 3.0))
        for role, count in role_counts.items():
            raw = (mean_count / max(1.0, float(count))) ** 0.5
            balance_factors[role] = max(min_weight, min(max_weight, raw))
        for row_index, action_row in enumerate(action_rows):
            for pos, role_id in enumerate(action_row):
                if row_index >= len(action_weight_rows) or pos >= len(action_weight_rows[row_index]):
                    continue
                current = float(action_weight_rows[row_index][pos] or 0.0)
                if current <= 0.0:
                    continue
                role = BODY_ACTION_ROLES[int(role_id)] if 0 <= int(role_id) < len(BODY_ACTION_ROLES) else "other"
                action_weight_rows[row_index][pos] = current * float(balance_factors.get(role, 1.0))
    return action_rows, action_weight_rows, {
        "enabled": bool(active_positions),
        "policy": "private_prefix_conditioned_body_action_target_rows_v1",
        "rows": len(target_rows),
        "target_mode": str(target_mode or ""),
        "role_count": len(BODY_ACTION_ROLES),
        "roles": list(BODY_ACTION_ROLES),
        "active_positions": active_positions,
        "role_counts": dict(sorted(role_counts.items())),
        "class_balance": {
            "enabled": balance_enabled,
            "policy": "private_inverse_sqrt_body_action_role_frequency_v1" if balance_enabled else "not_enabled",
            "min_weight": float(class_balance_min_weight if class_balance_min_weight is not None else 0.35),
            "max_weight": float(class_balance_max_weight if class_balance_max_weight is not None else 3.0),
            "role_weight_factors": dict(sorted((role, round(weight, 6)) for role, weight in balance_factors.items())),
            "score_semantics": (
                "Optional train-split-only inverse-sqrt role-frequency weighting over admitted private body "
                "action targets. It changes auxiliary loss weighting only; it does not render code, inspect "
                "tests or solutions, use public data, or grant candidate-generation credit."
            ),
        },
        "score_semantics": (
            "Maps admitted private/licensed target-body tokens onto broad executable action roles "
            "such as return, branch, block exit, update operator, expression closure, and identifier. "
            "The active mask is inherited from the private body-transition target mask. This creates "
            "a trainable structural next-action objective; it does not render code, inspect tests or "
            "solutions, use public benchmark data, or grant learned-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def body_action_role_for_token_text(token_text: str) -> str:
    token = str(token_text or "")
    if token in {"", "<pad>", "<bos>"}:
        return "ignore"
    if token == "<eos>":
        return "eos"
    if token == "NAME:return":
        return "return"
    if token == "DEDENT:":
        return "block_exit"
    if token == "INDENT:":
        return "block_enter"
    if token == "NEWLINE:":
        return "line_boundary"
    if token in {"NAME:for", "NAME:while"}:
        return "loop"
    if token in {"NAME:if", "NAME:elif", "NAME:else", "NAME:try", "NAME:except"}:
        return "branch"
    if token == "OP:=":
        return "assignment"
    if token in {"OP:+=", "OP:-=", "OP:*=", "OP:/=", "OP:%=", "OP://="}:
        return "update_operator"
    if token in {"OP:==", "OP:!=", "OP:<", "OP:<=", "OP:>", "OP:>=", "NAME:in", "NAME:is"}:
        return "comparison"
    if token in {"NAME:and", "NAME:or", "NAME:not"}:
        return "bool_op"
    if token in {"OP:(", "OP:[", "OP:{"}:
        return "open_expr"
    if token in {"OP:)", "OP:]", "OP:}"}:
        return "close_expr"
    if token == "OP:.":
        return "attribute"
    if token.startswith("NAME:"):
        name = token.removeprefix("NAME:")
        if name in {
            "len",
            "range",
            "enumerate",
            "zip",
            "sum",
            "max",
            "min",
            "sorted",
            "abs",
            "int",
            "float",
            "str",
            "list",
            "dict",
            "set",
            "tuple",
            "append",
            "extend",
            "add",
            "discard",
            "get",
            "insert",
            "items",
            "keys",
            "pop",
            "popleft",
            "remove",
            "setdefault",
            "update",
            "values",
            "split",
            "splitlines",
            "join",
            "strip",
            "lower",
            "upper",
        }:
            return "call_or_method"
        return "identifier"
    if token.startswith("NUMBER:") or token.startswith("STRING:"):
        return "literal"
    return "other"


def body_action_role_id_for_token(token_text: str) -> int:
    return BODY_ACTION_ROLE_TO_ID.get(
        body_action_role_for_token_text(token_text),
        BODY_ACTION_ROLE_TO_ID["other"],
    )


def body_operand_target_rows(
    target_rows: list[list[int]],
    transition_weight_rows: list[list[float]],
    *,
    target_vocab: dict[str, int],
    target_mode: str,
    source_texts: list[str] | None = None,
    class_balance_enabled: bool = False,
    class_balance_min_weight: float = 0.35,
    class_balance_max_weight: float = 3.0,
) -> tuple[list[list[int]], list[list[float]], dict[str, Any]]:
    inverse = {int(idx): str(tok) for tok, idx in target_vocab.items()}
    operand_rows: list[list[int]] = []
    operand_weight_rows: list[list[float]] = []
    active_positions = 0
    role_counts: dict[str, int] = {}
    for row_index, row in enumerate(target_rows):
        weight_row = transition_weight_rows[row_index] if row_index < len(transition_weight_rows) else []
        source_text = str(source_texts[row_index] if source_texts and row_index < len(source_texts) else "")
        allowed_names = allowed_parameter_names_from_source_text(source_text)
        line_roles = body_operand_roles_for_target_row(row, inverse, allowed_names=allowed_names)
        operand_row: list[int] = []
        operand_weight_row: list[float] = []
        for pos, value in enumerate(row):
            weight = float(weight_row[pos]) if pos < len(weight_row) else 0.0
            token_text = inverse.get(int(value), "")
            role = line_roles.get(pos) or body_operand_role_for_token_text(
                token_text,
                allowed_names=allowed_names,
                generated_tokens=None,
            )
            role_id = BODY_OPERAND_ROLE_TO_ID.get(role, BODY_OPERAND_ROLE_TO_ID["other"])
            if weight <= 0.0:
                role_id = BODY_OPERAND_ROLE_TO_ID["ignore"]
                operand_weight_row.append(0.0)
            else:
                active_positions += 1
                role_counts[role] = role_counts.get(role, 0) + 1
                operand_weight_row.append(weight)
            operand_row.append(role_id)
        operand_rows.append(operand_row)
        operand_weight_rows.append(operand_weight_row)
    balance_factors: dict[str, float] = {}
    balance_enabled = bool(class_balance_enabled and active_positions and role_counts)
    if balance_enabled:
        mean_count = float(active_positions) / max(1.0, float(len(role_counts)))
        min_weight = max(0.0, float(class_balance_min_weight if class_balance_min_weight is not None else 0.35))
        max_weight = max(min_weight, float(class_balance_max_weight if class_balance_max_weight is not None else 3.0))
        for role, count in role_counts.items():
            raw = (mean_count / max(1.0, float(count))) ** 0.5
            balance_factors[role] = max(min_weight, min(max_weight, raw))
        for row_index, operand_row in enumerate(operand_rows):
            for pos, role_id in enumerate(operand_row):
                if row_index >= len(operand_weight_rows) or pos >= len(operand_weight_rows[row_index]):
                    continue
                current = float(operand_weight_rows[row_index][pos] or 0.0)
                if current <= 0.0:
                    continue
                role = BODY_OPERAND_ROLES[int(role_id)] if 0 <= int(role_id) < len(BODY_OPERAND_ROLES) else "other"
                operand_weight_rows[row_index][pos] = current * float(balance_factors.get(role, 1.0))
    return operand_rows, operand_weight_rows, {
        "enabled": bool(active_positions),
        "policy": "private_prefix_conditioned_body_operand_target_rows_v1",
        "rows": len(target_rows),
        "target_mode": str(target_mode or ""),
        "role_count": len(BODY_OPERAND_ROLES),
        "roles": list(BODY_OPERAND_ROLES),
        "active_positions": active_positions,
        "role_counts": dict(sorted(role_counts.items())),
        "class_balance": {
            "enabled": balance_enabled,
            "policy": "private_inverse_sqrt_body_operand_role_frequency_v1" if balance_enabled else "not_enabled",
            "min_weight": float(class_balance_min_weight if class_balance_min_weight is not None else 0.35),
            "max_weight": float(class_balance_max_weight if class_balance_max_weight is not None else 3.0),
            "role_weight_factors": dict(sorted((role, round(weight, 6)) for role, weight in balance_factors.items())),
            "score_semantics": (
                "Optional train-split-only inverse-sqrt role-frequency weighting over admitted private body "
                "operand targets. It changes auxiliary loss weighting only; it does not render code, inspect "
                "tests or solutions, use public data, or grant candidate-generation credit."
            ),
        },
        "score_semantics": (
            "Maps admitted private/licensed target-body tokens onto operand/value binding roles "
            "such as visible parameter, loop variable, local state, builtin, method, operator, "
            "literal, and statement boundary. Source-visible parameters are taken only from the "
            "prompt/signature source text used by the model. This trains a learned binding head; "
            "it does not render code, inspect tests or solutions, use public benchmark data, or "
            "grant learned-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def body_operand_roles_for_target_row(
    row: list[int],
    inverse: dict[int, str],
    *,
    allowed_names: set[str],
) -> dict[int, str]:
    tokens = [str(inverse.get(int(token_id), "")) for token_id in row]
    lines = token_lines_with_depth(tokens)
    loop_names: set[str] = set()
    local_names: set[str] = set()
    roles: dict[int, str] = {}
    for _depth, positions, values in lines:
        if not values:
            continue
        if values[0] == "for":
            targets = names_before_in(values)
            loop_names.update(targets)
            local_names.update(targets)
            for pos, value in zip(positions, values):
                if value in targets:
                    roles[pos] = "loop_variable"
        assignment_targets = assignment_targets_for_values(values)
        if assignment_targets:
            local_names.update(assignment_targets)
            for pos, value in zip(positions, values):
                if value in assignment_targets:
                    roles[pos] = "local_state"
        for pos, value in zip(positions, values):
            if pos in roles:
                continue
            if value in loop_names:
                roles[pos] = "loop_variable"
            elif value in local_names:
                roles[pos] = "local_state"
            elif value in allowed_names:
                roles[pos] = "visible_parameter"
    return roles


def body_operand_role_for_token_text(
    token_text: str,
    *,
    allowed_names: set[str] | None = None,
    generated_tokens: list[str] | None = None,
    prefix_context: dict[str, set[str]] | None = None,
) -> str:
    token = str(token_text or "")
    allowed = {str(name) for name in (allowed_names or set()) if str(name)}
    if token in {"", "<pad>", "<bos>"}:
        return "ignore"
    if token == "<eos>":
        return "eos"
    if token in {"NEWLINE:", "DEDENT:", "INDENT:"}:
        return "statement_boundary"
    if token == "NAME:return":
        return "return_keyword"
    if token in {"NAME:for", "NAME:while", "NAME:if", "NAME:elif", "NAME:else", "NAME:try", "NAME:except", "NAME:in"}:
        return "control_keyword"
    if token.startswith("NUMBER:") or token.startswith("STRING:"):
        return "literal_value"
    if token in {"OP:="} or token in {"OP:+=", "OP:-=", "OP:*=", "OP:/=", "OP:%=", "OP://="}:
        return "assignment_operator"
    if token in {"OP:+", "OP:-", "OP:*", "OP:/", "OP:%", "OP://", "OP:**"}:
        return "arithmetic_operator"
    if token in {"OP:==", "OP:!=", "OP:<", "OP:<=", "OP:>", "OP:>=", "NAME:is"}:
        return "comparison_operator"
    if token in {"NAME:and", "NAME:or", "NAME:not"}:
        return "boolean_operator"
    if token in {"OP:(", "OP:)"}:
        return "call_delimiter"
    if token in {"OP:[", "OP:]", "OP:{", "OP:}"}:
        return "index_delimiter"
    if token in {"OP:.", "OP:,", "OP::"}:
        return "attribute_name" if token == "OP:." else "punctuation"
    if token.startswith("NAME:"):
        name = token.removeprefix("NAME:")
        context = (
            prefix_context
            if isinstance(prefix_context, dict)
            else body_operand_prefix_context(generated_tokens or [], allowed_names=allowed)
        )
        if (generated_tokens or []) and str((generated_tokens or [""])[-1]) == "OP:.":
            return "method_name" if name in METHOD_OR_ATTRIBUTE_NAMES else "attribute_name"
        if name in allowed:
            return "visible_parameter"
        if name in context.get("loop_names", set()):
            return "loop_variable"
        if name in context.get("local_names", set()):
            return "local_state"
        if name in GRAPH_STATE_NAMES or any(part in name.lower() for part in ("queue", "visit", "frontier", "neighbor", "dist")):
            return "local_state"
        if name in BUILTIN_OPERAND_NAMES:
            return "builtin_function"
        if name in METHOD_OR_ATTRIBUTE_NAMES:
            return "method_name"
        return "local_state" if not allowed else "other"
    return "other"


BUILTIN_OPERAND_NAMES: set[str] = {
    "abs",
    "all",
    "any",
    "bool",
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
METHOD_OR_ATTRIBUTE_NAMES: set[str] = {
    "add",
    "append",
    "clear",
    "discard",
    "extend",
    "get",
    "insert",
    "items",
    "keys",
    "lower",
    "pop",
    "popleft",
    "remove",
    "setdefault",
    "split",
    "splitlines",
    "strip",
    "update",
    "upper",
    "values",
}

GRAPH_STATE_NAMES: set[str] = {
    "adj",
    "dist",
    "distance",
    "distances",
    "edges",
    "frontier",
    "graph",
    "neighbors",
    "next_nodes",
    "queue",
    "seen",
    "stack",
    "visited",
}


def body_operand_prefix_context(
    generated_tokens: list[str],
    *,
    allowed_names: set[str] | None = None,
) -> dict[str, set[str]]:
    allowed = {str(name) for name in (allowed_names or set()) if str(name)}
    lines = token_lines_with_depth([str(token) for token in generated_tokens])
    loop_names: set[str] = set()
    local_names: set[str] = set()
    for _depth, _positions, values in lines:
        if not values:
            continue
        if values[0] == "for":
            targets = names_before_in(values)
            loop_names.update(targets)
            local_names.update(targets)
        targets = assignment_targets_for_values(values)
        local_names.update(targets)
    return {
        "allowed_names": allowed,
        "loop_names": loop_names - allowed,
        "local_names": local_names - allowed,
    }


def body_operand_role_id_for_token(
    token_text: str,
    *,
    allowed_names: set[str] | None = None,
    generated_tokens: list[str] | None = None,
    prefix_context: dict[str, set[str]] | None = None,
) -> int:
    return BODY_OPERAND_ROLE_TO_ID.get(
        body_operand_role_for_token_text(
            token_text,
            allowed_names=allowed_names,
            generated_tokens=generated_tokens,
            prefix_context=prefix_context,
        ),
        BODY_OPERAND_ROLE_TO_ID["other"],
    )


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


def evaluate_specialist_routing_mlx(
    model: Any,
    source_rows: list[list[int]],
    target_rows: list[list[int]],
    *,
    target_vocab: dict[str, int],
    batch_size: int,
    pad_id: int,
    mx: Any,
) -> dict[str, Any]:
    """Attribute sparse expert activation to private target action roles.

    This teacher-forced diagnostic never enters generation, ranking, or loss.
    """

    if not hasattr(model, "specialist_route") or not source_rows or not target_rows:
        return {"active": False, "reason": "specialist_route_unavailable"}
    id_to_token = {int(value): str(key) for key, value in target_vocab.items()}
    assignment_counts: dict[int, int] = {}
    role_counts: dict[int, dict[str, int]] = {}
    routed_token_count = 0
    top_k = 0
    model.eval()
    for start in range(0, len(source_rows), max(1, int(batch_size))):
        src = mx.array(source_rows[start : start + batch_size], dtype=mx.int32)
        tgt = mx.array(target_rows[start : start + batch_size], dtype=mx.int32)
        route = model.specialist_route(src, tgt[:, :-1])
        if not route:
            model.train()
            return {"active": False, "reason": "specialist_route_disabled"}
        indices = route["indices"].tolist()
        target_ids = tgt[:, 1:].tolist()
        for row_indices, row_targets in zip(indices, target_ids):
            for selected, target_id in zip(row_indices, row_targets):
                if int(target_id) == int(pad_id):
                    continue
                role_id = body_action_role_id_for_token(id_to_token.get(int(target_id), ""))
                role = BODY_ACTION_ROLES[role_id] if 0 <= role_id < len(BODY_ACTION_ROLES) else "other"
                top_k = max(top_k, len(selected))
                routed_token_count += 1
                for expert_id in selected:
                    expert_id = int(expert_id)
                    assignment_counts[expert_id] = assignment_counts.get(expert_id, 0) + 1
                    expert_roles = role_counts.setdefault(expert_id, {})
                    expert_roles[role] = expert_roles.get(role, 0) + 1
    model.train()
    total_assignments = sum(assignment_counts.values())
    active_experts = sorted(assignment_counts)
    fractions = [assignment_counts[key] / max(total_assignments, 1) for key in active_experts]
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in fractions)
    normalized_entropy = entropy / math.log(len(active_experts)) if len(active_experts) > 1 else 0.0
    expert_rows = []
    for expert_id in active_experts:
        counts = role_counts.get(expert_id, {})
        dominant_role, dominant_count = max(counts.items(), key=lambda row: row[1]) if counts else ("none", 0)
        expert_rows.append(
            {
                "expert_id": expert_id,
                "assignment_count": assignment_counts[expert_id],
                "assignment_fraction": round(assignment_counts[expert_id] / max(total_assignments, 1), 6),
                "dominant_action_role": dominant_role,
                "dominant_action_role_fraction": round(dominant_count / max(assignment_counts[expert_id], 1), 6),
                "action_role_counts": dict(sorted(counts.items())),
            }
        )
    return {
        "active": True,
        "policy": "private_teacher_forced_sparse_expert_action_attribution_v1",
        "routed_token_count": routed_token_count,
        "total_assignment_count": total_assignments,
        "top_k": top_k,
        "active_expert_count": len(active_experts),
        "assignment_entropy_normalized": round(normalized_entropy, 6),
        "expert_attribution": expert_rows,
        "uses_eval_tests": False,
        "uses_private_target_tokens_for_diagnostic_only": True,
        "candidate_generation_credit": 0,
    }


def evaluate_body_transition_mlx(
    model: Any,
    source_rows: list[list[int]],
    target_rows: list[list[int]],
    transition_weight_rows: list[list[float]],
    *,
    batch_size: int,
    pad_id: int,
    enabled: bool,
    mx: Any,
    nn: Any,
) -> dict[str, Any]:
    if not enabled or not source_rows or not target_rows or not transition_weight_rows:
        return {
            "enabled": bool(enabled),
            "loss": None,
            "active_position_count": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    if not hasattr(model, "body_transition_logits"):
        return {
            "enabled": False,
            "loss": None,
            "reason": "model_has_no_body_transition_head",
            "active_position_count": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    losses: list[float] = []
    active_positions = 0
    model.eval()
    for start in range(0, len(source_rows), batch_size):
        src = mx.array(source_rows[start : start + batch_size], dtype=mx.int32)
        tgt = mx.array(target_rows[start : start + batch_size], dtype=mx.int32)
        weights = mx.array(transition_weight_rows[start : start + batch_size], dtype=mx.float32)
        loss = body_transition_loss_mlx(model, src, tgt, pad_id, weights, mx, nn)
        mx.eval(loss)
        losses.append(float(loss.item()))
        active_positions += sum(
            1
            for row in transition_weight_rows[start : start + batch_size]
            for value in row[1:]
            if float(value or 0.0) > 0.0
        )
    model.train()
    loss = sum(losses) / max(1, len(losses))
    return {
        "enabled": True,
        "policy": "private_prefix_conditioned_body_transition_eval_v1",
        "loss": round(loss, 6),
        "active_position_count": active_positions,
        "score_semantics": (
            "Heldout loss for the prefix-conditioned body-transition head over admitted private/licensed "
            "target-body continuation positions. This is auxiliary training evidence only; it does not "
            "emit candidates or support learned-generation promotion."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def evaluate_body_action_mlx(
    model: Any,
    source_rows: list[list[int]],
    target_rows: list[list[int]],
    action_target_rows: list[list[int]],
    action_weight_rows: list[list[float]],
    *,
    batch_size: int,
    pad_id: int,
    enabled: bool,
    mx: Any,
    nn: Any,
) -> dict[str, Any]:
    if not enabled or not source_rows or not target_rows or not action_target_rows or not action_weight_rows:
        return {
            "enabled": bool(enabled),
            "loss": None,
            "accuracy": None,
            "active_position_count": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    if not hasattr(model, "body_action_logits"):
        return {
            "enabled": False,
            "loss": None,
            "accuracy": None,
            "reason": "model_has_no_body_action_head",
            "active_position_count": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    losses: list[float] = []
    active_positions = 0
    correct = 0
    counted = 0
    role_correct = {role: 0 for role in BODY_ACTION_ROLES}
    role_count = {role: 0 for role in BODY_ACTION_ROLES}
    model.eval()
    for start in range(0, len(source_rows), batch_size):
        src = mx.array(source_rows[start : start + batch_size], dtype=mx.int32)
        tgt = mx.array(target_rows[start : start + batch_size], dtype=mx.int32)
        targets = mx.array(action_target_rows[start : start + batch_size], dtype=mx.int32)
        weights = mx.array(action_weight_rows[start : start + batch_size], dtype=mx.float32)
        loss = body_action_loss_mlx(model, src, tgt, pad_id, targets, weights, mx, nn)
        logits = model.body_action_logits(src, tgt[:, :-1])
        pred = mx.argmax(logits, axis=-1)
        target_out = targets[:, 1:]
        valid = ((tgt[:, 1:] != pad_id).astype(mx.float32) * weights[:, 1:]) > 0.0
        hit = (pred == target_out).astype(mx.float32) * valid.astype(mx.float32)
        batch_correct = mx.sum(hit).astype(mx.float32)
        batch_count = mx.sum(valid.astype(mx.float32))
        mx.eval(loss, pred, target_out, valid, batch_correct, batch_count)
        losses.append(float(loss.item()))
        active_positions += sum(
            1
            for row in action_weight_rows[start : start + batch_size]
            for value in row[1:]
            if float(value or 0.0) > 0.0
        )
        if float(batch_count.item()) > 0.0:
            correct += int(batch_correct.item())
            counted += int(batch_count.item())
            pred_rows = pred.tolist()
            target_rows_local = target_out.tolist()
            valid_rows = valid.tolist()
            for row_index, target_row in enumerate(target_rows_local):
                for col_index, target_id in enumerate(target_row):
                    if not bool(valid_rows[row_index][col_index]):
                        continue
                    role = BODY_ACTION_ROLES[int(target_id)] if 0 <= int(target_id) < len(BODY_ACTION_ROLES) else "other"
                    role_count[role] += 1
                    if int(pred_rows[row_index][col_index]) == int(target_id):
                        role_correct[role] += 1
    model.train()
    loss = sum(losses) / max(1, len(losses))
    return {
        "enabled": True,
        "policy": "private_prefix_conditioned_body_action_eval_v1",
        "loss": round(loss, 6),
        "accuracy": round(correct / counted, 6) if counted else None,
        "correct_count": correct,
        "active_position_count": active_positions,
        "role_accuracy": {
            role: {
                "accuracy": round(role_correct[role] / role_count[role], 6) if role_count[role] else None,
                "correct_count": role_correct[role],
                "active_target_count": role_count[role],
            }
            for role in BODY_ACTION_ROLES
            if role_count[role]
        },
        "score_semantics": (
            "Heldout loss/accuracy for a prefix-conditioned body-action head over broad structural "
            "roles such as return, branch, block exit, update operator, and expression closure. It is "
            "trained only from admitted private/licensed body tokens and emits no candidates."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def evaluate_body_operand_mlx(
    model: Any,
    source_rows: list[list[int]],
    target_rows: list[list[int]],
    operand_target_rows: list[list[int]],
    operand_weight_rows: list[list[float]],
    *,
    batch_size: int,
    pad_id: int,
    enabled: bool,
    mx: Any,
    nn: Any,
) -> dict[str, Any]:
    if not enabled or not source_rows or not target_rows or not operand_target_rows or not operand_weight_rows:
        return {
            "enabled": bool(enabled),
            "loss": None,
            "accuracy": None,
            "active_position_count": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    if not hasattr(model, "body_operand_logits"):
        return {
            "enabled": False,
            "loss": None,
            "accuracy": None,
            "reason": "model_has_no_body_operand_head",
            "active_position_count": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    losses: list[float] = []
    active_positions = 0
    correct = 0
    counted = 0
    role_correct = {role: 0 for role in BODY_OPERAND_ROLES}
    role_count = {role: 0 for role in BODY_OPERAND_ROLES}
    model.eval()
    for start in range(0, len(source_rows), batch_size):
        src = mx.array(source_rows[start : start + batch_size], dtype=mx.int32)
        tgt = mx.array(target_rows[start : start + batch_size], dtype=mx.int32)
        targets = mx.array(operand_target_rows[start : start + batch_size], dtype=mx.int32)
        weights = mx.array(operand_weight_rows[start : start + batch_size], dtype=mx.float32)
        loss = body_operand_loss_mlx(model, src, tgt, pad_id, targets, weights, mx, nn)
        logits = model.body_operand_logits(src, tgt[:, :-1])
        pred = mx.argmax(logits, axis=-1)
        target_out = targets[:, 1:]
        valid = ((tgt[:, 1:] != pad_id).astype(mx.float32) * weights[:, 1:]) > 0.0
        hit = (pred == target_out).astype(mx.float32) * valid.astype(mx.float32)
        batch_correct = mx.sum(hit).astype(mx.float32)
        batch_count = mx.sum(valid.astype(mx.float32))
        mx.eval(loss, pred, target_out, valid, batch_correct, batch_count)
        losses.append(float(loss.item()))
        active_positions += sum(
            1
            for row in operand_weight_rows[start : start + batch_size]
            for value in row[1:]
            if float(value or 0.0) > 0.0
        )
        if float(batch_count.item()) > 0.0:
            correct += int(batch_correct.item())
            counted += int(batch_count.item())
            pred_rows = pred.tolist()
            target_rows_local = target_out.tolist()
            valid_rows = valid.tolist()
            for row_index, target_row in enumerate(target_rows_local):
                for col_index, target_id in enumerate(target_row):
                    if not bool(valid_rows[row_index][col_index]):
                        continue
                    role = BODY_OPERAND_ROLES[int(target_id)] if 0 <= int(target_id) < len(BODY_OPERAND_ROLES) else "other"
                    role_count[role] += 1
                    if int(pred_rows[row_index][col_index]) == int(target_id):
                        role_correct[role] += 1
    model.train()
    loss = sum(losses) / max(1, len(losses))
    return {
        "enabled": True,
        "policy": "private_prefix_conditioned_body_operand_eval_v1",
        "loss": round(loss, 6),
        "accuracy": round(correct / counted, 6) if counted else None,
        "correct_count": correct,
        "active_position_count": active_positions,
        "role_accuracy": {
            role: {
                "accuracy": round(role_correct[role] / role_count[role], 6) if role_count[role] else None,
                "correct_count": role_correct[role],
                "active_target_count": role_count[role],
            }
            for role in BODY_OPERAND_ROLES
            if role_count[role]
        },
        "score_semantics": (
            "Heldout loss/accuracy for a prefix-conditioned body-operand binding head over "
            "visible-parameter, loop-variable, local-state, builtin, method, literal, operator, "
            "and delimiter roles. It is trained only from admitted private/licensed body tokens "
            "and emits no candidates."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def evaluate_body_state_event_mlx(
    model: Any,
    source_rows: list[list[int]],
    target_rows: list[list[int]],
    event_target_rows: list[list[int]],
    event_weight_rows: list[list[float]],
    *,
    batch_size: int,
    pad_id: int,
    enabled: bool,
    mx: Any,
    nn: Any,
) -> dict[str, Any]:
    if not enabled or not source_rows or not target_rows or not event_target_rows or not event_weight_rows:
        return {
            "enabled": bool(enabled),
            "loss": None,
            "accuracy": None,
            "active_position_count": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    if not hasattr(model, "body_state_event_logits"):
        return {
            "enabled": False,
            "loss": None,
            "accuracy": None,
            "reason": "model_has_no_body_state_event_head",
            "active_position_count": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    losses: list[float] = []
    active_positions = 0
    correct = 0
    counted = 0
    role_correct = {role: 0 for role in BODY_STATE_EVENT_ROLES}
    role_count = {role: 0 for role in BODY_STATE_EVENT_ROLES}
    model.eval()
    for start in range(0, len(source_rows), batch_size):
        src = mx.array(source_rows[start : start + batch_size], dtype=mx.int32)
        tgt = mx.array(target_rows[start : start + batch_size], dtype=mx.int32)
        targets = mx.array(event_target_rows[start : start + batch_size], dtype=mx.int32)
        weights = mx.array(event_weight_rows[start : start + batch_size], dtype=mx.float32)
        loss = body_state_event_loss_mlx(model, src, tgt, pad_id, targets, weights, mx, nn)
        logits = model.body_state_event_logits(src, tgt[:, :-1])
        pred = mx.argmax(logits, axis=-1)
        target_out = targets[:, 1:]
        valid = ((tgt[:, 1:] != pad_id).astype(mx.float32) * weights[:, 1:]) > 0.0
        hit = (pred == target_out).astype(mx.float32) * valid.astype(mx.float32)
        batch_correct = mx.sum(hit).astype(mx.float32)
        batch_count = mx.sum(valid.astype(mx.float32))
        mx.eval(loss, pred, target_out, valid, batch_correct, batch_count)
        losses.append(float(loss.item()))
        active_positions += sum(
            1
            for row in event_weight_rows[start : start + batch_size]
            for value in row[1:]
            if float(value or 0.0) > 0.0
        )
        if float(batch_count.item()) > 0.0:
            correct += int(batch_correct.item())
            counted += int(batch_count.item())
            pred_rows = pred.tolist()
            target_rows_local = target_out.tolist()
            valid_rows = valid.tolist()
            for row_index, target_row in enumerate(target_rows_local):
                for col_index, target_id in enumerate(target_row):
                    if not bool(valid_rows[row_index][col_index]):
                        continue
                    role = BODY_STATE_EVENT_ROLES[int(target_id)] if 0 <= int(target_id) < len(BODY_STATE_EVENT_ROLES) else "none"
                    role_count[role] += 1
                    if int(pred_rows[row_index][col_index]) == int(target_id):
                        role_correct[role] += 1
    model.train()
    loss = sum(losses) / max(1, len(losses))
    return {
        "enabled": True,
        "policy": "private_prefix_conditioned_body_state_event_eval_v1",
        "loss": round(loss, 6),
        "accuracy": round(correct / counted, 6) if counted else None,
        "correct_count": correct,
        "active_position_count": active_positions,
        "role_accuracy": {
            role: {
                "accuracy": round(role_correct[role] / role_count[role], 6) if role_count[role] else None,
                "correct_count": role_correct[role],
                "active_target_count": role_count[role],
            }
            for role in BODY_STATE_EVENT_ROLES
            if role_count[role]
        },
        "score_semantics": (
            "Heldout loss/accuracy for a prefix-conditioned body state-machine event head. Event labels "
            "are derived from admitted private action/operand roles and describe traversal/call, state "
            "update, control, finalizer, value-expression, statement-boundary, or none. This head emits "
            "no candidates and grants no learned-generation promotion credit by itself."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def evaluate_body_executable_span_mlx(
    model: Any,
    source_rows: list[list[int]],
    target_rows: list[list[int]],
    span_target_rows: list[list[int]],
    span_weight_rows: list[list[float]],
    *,
    batch_size: int,
    pad_id: int,
    enabled: bool,
    mx: Any,
    nn: Any,
) -> dict[str, Any]:
    if not enabled or not source_rows or not target_rows or not span_target_rows or not span_weight_rows:
        return {
            "enabled": bool(enabled),
            "loss": None,
            "accuracy": None,
            "active_position_count": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    if not hasattr(model, "body_executable_span_logits"):
        return {
            "enabled": False,
            "loss": None,
            "accuracy": None,
            "reason": "model_has_no_body_executable_span_head",
            "active_position_count": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    losses: list[float] = []
    active_positions = 0
    correct = 0
    counted = 0
    role_correct = {role: 0 for role in BODY_EXECUTABLE_SPAN_ROLES}
    role_count = {role: 0 for role in BODY_EXECUTABLE_SPAN_ROLES}
    model.eval()
    for start in range(0, len(source_rows), batch_size):
        src = mx.array(source_rows[start : start + batch_size], dtype=mx.int32)
        tgt = mx.array(target_rows[start : start + batch_size], dtype=mx.int32)
        targets = mx.array(span_target_rows[start : start + batch_size], dtype=mx.int32)
        weights = mx.array(span_weight_rows[start : start + batch_size], dtype=mx.float32)
        loss = body_executable_span_loss_mlx(model, src, tgt, pad_id, targets, weights, mx, nn)
        logits = model.body_executable_span_logits(src, tgt[:, :-1])
        pred = mx.argmax(logits, axis=-1)
        target_out = targets[:, 1:]
        valid = ((tgt[:, 1:] != pad_id).astype(mx.float32) * weights[:, 1:]) > 0.0
        hit = (pred == target_out).astype(mx.float32) * valid.astype(mx.float32)
        batch_correct = mx.sum(hit).astype(mx.float32)
        batch_count = mx.sum(valid.astype(mx.float32))
        mx.eval(loss, pred, target_out, valid, batch_correct, batch_count)
        losses.append(float(loss.item()))
        active_positions += sum(
            1
            for row in span_weight_rows[start : start + batch_size]
            for value in row[1:]
            if float(value or 0.0) > 0.0
        )
        if float(batch_count.item()) > 0.0:
            correct += int(batch_correct.item())
            counted += int(batch_count.item())
            pred_rows = pred.tolist()
            target_rows_local = target_out.tolist()
            valid_rows = valid.tolist()
            for row_index, target_row in enumerate(target_rows_local):
                for col_index, target_id in enumerate(target_row):
                    if not bool(valid_rows[row_index][col_index]):
                        continue
                    role = (
                        BODY_EXECUTABLE_SPAN_ROLES[int(target_id)]
                        if 0 <= int(target_id) < len(BODY_EXECUTABLE_SPAN_ROLES)
                        else "other_span"
                    )
                    role_count[role] += 1
                    if int(pred_rows[row_index][col_index]) == int(target_id):
                        role_correct[role] += 1
    model.train()
    loss = sum(losses) / max(1, len(losses))
    return {
        "enabled": True,
        "policy": "private_prefix_conditioned_body_executable_span_eval_v1",
        "loss": round(loss, 6),
        "accuracy": round(correct / counted, 6) if counted else None,
        "correct_count": correct,
        "active_position_count": active_positions,
        "role_accuracy": {
            role: {
                "accuracy": round(role_correct[role] / role_count[role], 6) if role_count[role] else None,
                "correct_count": role_correct[role],
                "active_target_count": role_count[role],
            }
            for role in BODY_EXECUTABLE_SPAN_ROLES
            if role_count[role]
        },
        "score_semantics": (
            "Heldout loss/accuracy for a prefix-conditioned executable-span head over private "
            "guard/control, traversal, state-update, value-expression, return/finalizer, and "
            "state-reference spans. It emits no candidates and grants no learned-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def evaluate_body_state_event_consistency_mlx(
    model: Any,
    source_rows: list[list[int]],
    target_rows: list[list[int]],
    event_target_rows: list[list[int]],
    event_weight_rows: list[list[float]],
    *,
    batch_size: int,
    pad_id: int,
    enabled: bool,
    mx: Any,
) -> dict[str, Any]:
    if not enabled or not source_rows or not target_rows or not event_target_rows or not event_weight_rows:
        return {
            "enabled": bool(enabled),
            "action_compatible_loss": None,
            "operand_compatible_loss": None,
            "action_compatible_mass": None,
            "operand_compatible_mass": None,
            "candidate_generation_credit": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    if not hasattr(model, "body_action_logits") or not hasattr(model, "body_operand_logits"):
        return {
            "enabled": False,
            "reason": "model_missing_action_or_operand_head",
            "action_compatible_loss": None,
            "operand_compatible_loss": None,
            "action_compatible_mass": None,
            "operand_compatible_mass": None,
            "candidate_generation_credit": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    action_masks = mx.array(BODY_STATE_EVENT_ACTION_COMPATIBILITY_MASK, dtype=mx.float32)
    operand_masks = mx.array(BODY_STATE_EVENT_OPERAND_COMPATIBILITY_MASK, dtype=mx.float32)
    eps = mx.array(1e-9, dtype=mx.float32)
    totals = {
        "action_loss": 0.0,
        "action_mass": 0.0,
        "action_count": 0.0,
        "operand_loss": 0.0,
        "operand_mass": 0.0,
        "operand_count": 0.0,
    }
    model.eval()
    for start in range(0, len(source_rows), batch_size):
        src = mx.array(source_rows[start : start + batch_size], dtype=mx.int32)
        tgt = mx.array(target_rows[start : start + batch_size], dtype=mx.int32)
        targets = mx.array(event_target_rows[start : start + batch_size], dtype=mx.int32)
        weights = mx.array(event_weight_rows[start : start + batch_size], dtype=mx.float32)
        tgt_in = tgt[:, :-1]
        target_out = targets[:, 1:]
        base_valid = (tgt[:, 1:] != pad_id).astype(mx.float32) * weights[:, 1:]

        action_compat = mx.take(action_masks, target_out, axis=0)
        action_valid = base_valid * (mx.sum(action_compat, axis=-1) > 0.0).astype(mx.float32)
        action_probs = mx.softmax(model.body_action_logits(src, tgt_in), axis=-1)
        action_mass = mx.sum(action_probs * action_compat, axis=-1)
        action_losses = -mx.log(mx.maximum(action_mass, eps))
        action_loss_sum = mx.sum(action_losses * action_valid)
        action_mass_sum = mx.sum(action_mass * action_valid)
        action_count = mx.sum(action_valid)

        operand_compat = mx.take(operand_masks, target_out, axis=0)
        operand_valid = base_valid * (mx.sum(operand_compat, axis=-1) > 0.0).astype(mx.float32)
        operand_probs = mx.softmax(model.body_operand_logits(src, tgt_in), axis=-1)
        operand_mass = mx.sum(operand_probs * operand_compat, axis=-1)
        operand_losses = -mx.log(mx.maximum(operand_mass, eps))
        operand_loss_sum = mx.sum(operand_losses * operand_valid)
        operand_mass_sum = mx.sum(operand_mass * operand_valid)
        operand_count = mx.sum(operand_valid)
        mx.eval(
            action_loss_sum,
            action_mass_sum,
            action_count,
            operand_loss_sum,
            operand_mass_sum,
            operand_count,
        )
        totals["action_loss"] += float(action_loss_sum.item())
        totals["action_mass"] += float(action_mass_sum.item())
        totals["action_count"] += float(action_count.item())
        totals["operand_loss"] += float(operand_loss_sum.item())
        totals["operand_mass"] += float(operand_mass_sum.item())
        totals["operand_count"] += float(operand_count.item())
    model.train()
    action_count = max(0.0, totals["action_count"])
    operand_count = max(0.0, totals["operand_count"])
    return {
        "enabled": True,
        "policy": "private_body_state_event_action_operand_consistency_eval_v1",
        "action_compatible_loss": round(totals["action_loss"] / action_count, 6) if action_count > 0.0 else None,
        "operand_compatible_loss": round(totals["operand_loss"] / operand_count, 6) if operand_count > 0.0 else None,
        "action_compatible_mass": round(totals["action_mass"] / action_count, 6) if action_count > 0.0 else None,
        "operand_compatible_mass": round(totals["operand_mass"] / operand_count, 6) if operand_count > 0.0 else None,
        "action_active_position_count": int(round(action_count)),
        "operand_active_position_count": int(round(operand_count)),
        "score_semantics": (
            "Heldout compatibility mass for action/operand heads under the private state-event target. "
            "It rewards probability mass on event-compatible roles, not exact bodies, and emits no "
            "candidate by itself."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


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


def specialist_token_expert_map(
    target_vocab: dict[str, int], specialist_core: dict[str, Any]
) -> tuple[list[list[int]], dict[str, Any]]:
    """Create training-only action/token expert targets for a sparse router."""

    if specialist_core.get("mode") != "sparse_moe":
        return [], {
            "enabled": False,
            "policy": "not_enabled",
            "candidate_generation_credit": 0,
        }
    num_experts = int(specialist_core.get("num_experts") or 1)
    top_k = int(specialist_core.get("top_k") or 1)
    max_token_id = max((int(value) for value in target_vocab.values()), default=-1)
    rows = [[0 for _ in range(top_k)] for _ in range(max_token_id + 1)]
    mapped_experts: set[int] = set()
    for token, token_id_raw in target_vocab.items():
        token_id = int(token_id_raw)
        role_id = body_action_role_id_for_token(str(token))
        primary = role_id % num_experts
        selected = [primary]
        salt = 0
        while len(selected) < top_k:
            digest = stable_hash(f"{token}|{role_id}|{salt}")
            candidate = int(digest[:16], 16) % num_experts
            salt += 1
            if candidate not in selected:
                selected.append(candidate)
        rows[token_id] = selected
        mapped_experts.update(selected)
    return rows, {
        "enabled": float(specialist_core.get("router_supervision_loss_weight") or 0.0) > 0.0,
        "policy": "private_target_action_role_plus_token_shard_router_supervision_v1",
        "loss_weight": float(specialist_core.get("router_supervision_loss_weight") or 0.0),
        "target_vocab_size": len(target_vocab),
        "mapped_expert_count": len(mapped_experts),
        "top_k": top_k,
        "uses_private_train_target_tokens": True,
        "served_at_generation": False,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def parameter_snapshot(model: Any, mlx_utils: Any, mx: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name, value in mlx_utils.tree_flatten(model.trainable_parameters()):
        snapshot[name] = mx.array(value)
    mx.eval(*snapshot.values())
    return snapshot


OPTIONAL_AUXILIARY_PARAMETER_ROOTS = (
    "plan_router",
    "slot_router",
    "slot_role_router",
    "body_transition_router",
    "body_action_router",
    "body_operand_router",
    "body_state_event_router",
    "body_state_event_to_hidden",
    "body_executable_span_router",
    "body_executable_span_to_hidden",
)


def parameter_is_optional_auxiliary(name: str) -> bool:
    """Classify only known opt-in heads as optional.

    Unknown and newly introduced tensors stay in the core denominator so a new
    inactive path cannot silently improve the core-update receipt.
    """

    root = str(name or "").split(".", 1)[0]
    return root in OPTIONAL_AUXILIARY_PARAMETER_ROOTS


def parameter_update_summary(model: Any, before: dict[str, Any], mlx_utils: Any, mx: Any) -> dict[str, Any]:
    total = 0
    changed = 0
    tensor_total = 0
    tensor_changed = 0
    core_total = 0
    core_changed = 0
    core_tensor_total = 0
    core_tensor_changed = 0
    unchanged_core: list[str] = []
    unchanged_optional: list[str] = []
    parameter_count_by_root: Counter[str] = Counter()
    for name, value in mlx_utils.tree_flatten(model.trainable_parameters()):
        root = str(name or "").split(".", 1)[0]
        parameter_count_by_root[root] += int(value.size)
        optional_auxiliary = parameter_is_optional_auxiliary(name)
        tensor_total += 1
        total += int(value.size)
        if not optional_auxiliary:
            core_tensor_total += 1
            core_total += int(value.size)
        old = before.get(name)
        if old is None:
            changed += int(value.size)
            tensor_changed += 1
            if not optional_auxiliary:
                core_changed += int(value.size)
                core_tensor_changed += 1
            continue
        delta = mx.abs(value - old)
        element_changed = int(mx.sum((delta > 1e-10).astype(mx.int32)).item())
        changed += element_changed
        if element_changed:
            tensor_changed += 1
        if not optional_auxiliary:
            core_changed += element_changed
            if element_changed:
                core_tensor_changed += 1
            else:
                unchanged_core.append(str(name))
        elif not element_changed:
            unchanged_optional.append(str(name))
    return {
        "parameter_count": total,
        "updated_parameter_count": changed,
        "parameter_update_fraction": round(changed / total, 6) if total else 0.0,
        "parameter_tensor_count": tensor_total,
        "updated_parameter_tensor_count": tensor_changed,
        "parameter_tensor_update_fraction": round(tensor_changed / tensor_total, 6) if tensor_total else 0.0,
        "core_parameter_count": core_total,
        "updated_core_parameter_count": core_changed,
        "core_parameter_update_fraction": round(core_changed / core_total, 6) if core_total else 0.0,
        "core_parameter_tensor_count": core_tensor_total,
        "updated_core_parameter_tensor_count": core_tensor_changed,
        "core_parameter_tensor_update_fraction": round(core_tensor_changed / core_tensor_total, 6) if core_tensor_total else 0.0,
        "unchanged_core_parameter_tensor_names": sorted(unchanged_core)[:64],
        "unchanged_optional_auxiliary_tensor_names": sorted(unchanged_optional)[:64],
        "parameter_count_by_root": dict(sorted(parameter_count_by_root.items())),
        "parameter_role_policy": "known_optional_auxiliary_roots_else_core_fail_closed_v1",
    }


def active_parameter_accounting(
    update_summary: dict[str, Any],
    specialist_core_estimate: dict[str, Any],
    *,
    active_optional_roots: set[str] | None = None,
) -> dict[str, Any]:
    root_counts = {
        str(key): int(value)
        for key, value in dict_or_empty(update_summary.get("parameter_count_by_root")).items()
    }
    total = int(update_summary.get("parameter_count") or sum(root_counts.values()))
    core_total = int(update_summary.get("core_parameter_count") or 0)
    specialist_total = int(specialist_core_estimate.get("specialist_total_parameter_count") or 0)
    specialist_active = int(
        specialist_core_estimate.get("specialist_active_parameter_count_per_token") or 0
    )
    optional_roots = set(active_optional_roots or set())
    active_optional = sum(root_counts.get(root, 0) for root in optional_roots)
    shared_core = max(0, core_total - specialist_total)
    active = shared_core + specialist_active + active_optional
    return {
        "model_total_parameter_count": total,
        "core_parameter_count": core_total,
        "shared_core_parameter_count_excluding_specialists": shared_core,
        "specialist_total_parameter_count": specialist_total,
        "specialist_active_parameter_count_per_token": specialist_active,
        "active_optional_parameter_roots": sorted(optional_roots),
        "active_optional_parameter_count": active_optional,
        "model_active_parameter_count_per_token": active,
        "model_active_parameter_fraction": round(active / total, 6) if total else 0.0,
        "accounting_policy": "measured_core_plus_selected_experts_plus_active_optional_roots_v2",
    }


def build_gates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    source_contrast = dict_or_empty(payload.get("source_contrastive_loss"))
    semantic_plan = dict_or_empty(payload.get("semantic_plan_auxiliary"))
    semantic_slot = dict_or_empty(payload.get("semantic_slot_auxiliary"))
    specialist_core = dict_or_empty(payload.get("specialist_core"))
    specialist_routing = dict_or_empty(payload.get("specialist_routing"))
    training_plan = dict_or_empty(payload.get("training_plan"))
    data_exposure = dict_or_empty(payload.get("data_exposure"))
    row_summary = dict_or_empty(payload.get("row_summary"))
    target_token_positions = int(training_plan.get("target_token_positions") or 0)
    consumed_token_positions = int(payload.get("optimizer_token_positions_consumed") or 0)
    requested_rungs = [int(value) for value in training_plan.get("rung_token_positions_requested") or []]
    written_rungs = list(payload.get("rung_checkpoints") or [])
    checkpoint_path = resolve(str(payload.get("checkpoint") or ""))
    vocab_path = resolve(str(payload.get("vocab") or ""))
    device_text = str(payload.get("device") or "").lower()
    return [
        gate("active_training_rows_present", bool(payload.get("active")), "hard", payload.get("row_summary", {})),
        gate(
            "family_disjoint_vocab_and_base_lineage_verified",
            not bool(
                dict_or_empty(payload.get("checkpoint_training_lineage")).get(
                    "family_disjoint_claim_required"
                )
            )
            or bool(payload.get("family_disjoint_evidence")),
            "hard",
            payload.get("checkpoint_training_lineage"),
        ),
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
            "open_vocab_encoding_has_zero_unknown_tokens",
            bool(row_summary.get("open_vocab_unknown_free"))
            and int(row_summary.get("source_unknown_token_count") or 0) == 0
            and int(row_summary.get("target_unknown_token_count") or 0) == 0,
            "hard",
            {
                "source_unknown_token_count": row_summary.get("source_unknown_token_count"),
                "target_unknown_token_count": row_summary.get("target_unknown_token_count"),
                "source_byte_fallback_token_count": row_summary.get("source_byte_fallback_token_count"),
                "target_byte_fallback_token_count": row_summary.get("target_byte_fallback_token_count"),
            },
        ),
        gate(
            "unique_data_exposure_reported_without_epoch_inflation",
            int(data_exposure.get("one_pass_total_token_positions") or 0) > 0
            and data_exposure.get("optimizer_repetition_counted_as_unique_data") is False,
            "hard",
            data_exposure,
        ),
        gate(
            "minimum_one_pass_token_per_active_parameter",
            float(data_exposure.get("one_pass_tokens_per_active_parameter") or 0.0) >= 1.0,
            "soft",
            data_exposure,
        ),
        gate(
            "sparse_specialist_executes_less_than_total_when_enabled",
            specialist_core.get("mode") != "sparse_moe"
            or (
                int(specialist_core.get("specialist_active_parameter_count_per_token") or 0) > 0
                and int(specialist_core.get("specialist_active_parameter_count_per_token") or 0)
                < int(specialist_core.get("specialist_total_parameter_count") or 0)
            ),
            "hard",
            specialist_core,
        ),
        gate(
            "sparse_expert_attribution_present_when_enabled",
            specialist_core.get("mode") != "sparse_moe"
            or (
                bool(specialist_routing.get("active"))
                and int(specialist_routing.get("active_expert_count") or 0)
                >= max(2, math.ceil(int(specialist_core.get("num_experts") or 1) * 0.5))
                and int(specialist_routing.get("total_assignment_count") or 0) > 0
                and float(specialist_routing.get("assignment_entropy_normalized") or 0.0) >= 0.75
            ),
            "hard",
            specialist_routing,
        ),
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
