#!/usr/bin/env python3
"""Private-only MLX adaptation for the strict generator.

This continues an admitted MLX strict-generator checkpoint on configured
private prompt/signature -> solution-body rows. It does not run public
calibration, does not call a teacher, does not inspect eval tests/solutions for
generation, and does not credit templates/tools/renderers/fallbacks as learned
generation.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
import time
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neural_seed_code_proposer_comparator import (  # noqa: E402
    deterministic_sample,
    dict_or_empty,
    encode_many,
    get_path,
    load_private_rows,
    rel,
    resolve,
    row_id,
    stable_hash,
)
from neural_seed_token_decoder_support import PLAN_BODY_START_TOKEN, encode_target_rows, semantic_plan_from_body, target_tokens  # noqa: E402
from candidate_integrity import recompute_candidate_integrity  # noqa: E402
from strict_generator_mlx_decode_eval import (  # noqa: E402
    checkpoint_source_text_style,
    load_mlx_checkpoint,
    source_condition_adequacy_for_body,
    source_condition_expectation_from_source_text,
    stable_hash_file,
    strict_generator_decode_source_text,
    strict_generator_source_text_audit,
)
from strict_generator_mlx_replay_selection import select_family_disjoint_rows  # noqa: E402
from strict_generator_mlx_pretraining_probe import (  # noqa: E402
    evaluate_loss_mlx,
    evaluate_semantic_plan_mlx,
    evaluate_source_contrast_mlx,
    allowed_parameter_names_from_source_text,
    parameter_snapshot,
    parameter_update_summary,
    semantic_plan_sample_weights,
    semantic_plan_target_ids,
    semantic_plan_source_contrastive_weighted_loss_fn_mlx,
    apply_primary_dataflow_weight_override,
    selected_budget,
    semantic_plan_weighted_loss_fn_mlx,
    source_contrastive_weighted_loss_fn_mlx,
    target_loss_weight_rows,
    weighted_loss_with_prefix_mlx,
    weighted_loss_fn_mlx,
)
from strict_generator_mlx_adaptation_selection import (  # noqa: E402
    apply_guard_clean_target_weights,
    private_train_balanced_sample_summary,
    private_train_tier_summary,
    private_train_tier_vocab_summary,
    select_private_train_tier_rows,
    strict_target_guard_rows,
    tier_balanced_private_train_sample,
)
from strict_generator_mlx_adaptation_weights import (  # noqa: E402
    apply_return_expression_weight_override,
    apply_default_parameter_return_weights,
    visible_default_argument_candidates,
    apply_truthiness_guard_weights,
    visible_truthiness_argument_candidates,
    apply_source_condition_internalization_weights,
    apply_loop_operation_weights,
    apply_loop_statement_action_weights,
    apply_loop_semantic_operation_weights,
    apply_semantic_slot_prefix_weights,
    apply_loop_expression_synthesis_weights,
    apply_plan_conditioned_body_semantic_weights,
    apply_update_contract_consistency_weights,
    apply_direct_body_emission_path_weights,
    apply_local_return_closure_weights,
    loop_expression_synthesis_spans_for_body,
    expression_body_tokens,
    expression_is_plain_loop_identity,
    expression_tokens_are_trivial,
    semantic_call_expression_nodes,
    expression_kind_for_ast,
    semantic_slot_prefix_role,
    parse_role_filter,
    loop_semantic_operation_spans_for_body,
    semantic_candidate_statements_from_loop,
    classify_loop_semantic_statement,
    classify_top_level_semantic_finalizer,
    value_is_plain_loop_identity,
    value_has_semantic_operation,
    semantic_mutation_call_kind,
    call_is_shallow_loop_identity_accumulation,
    call_attribute_name,
    nested_call_attribute_names,
    loop_statement_action_spans_for_body,
    loop_statement_positive_excluded_tokens,
    body_statement_tokens,
    safe_unparse,
    find_subsequence_positions,
    loop_operation_tokens_for_body,
    ast_load_names,
    ast_store_names,
    compare_token,
    binop_token,
    augassign_token,
)
from strict_generator_pretraining_spine import (  # noqa: E402
    config_with_budget_overrides as strict_spine_config_with_budget_overrides,
    encode_staged_full_state_rows,
    stage_full_state_examples,
)


DEFAULT_CONFIG = ROOT / "configs" / "neural_seed_token_decoder_comparator.json"
DEFAULT_CHECKPOINT_REPORT = ROOT / "reports" / "strict_generator_mlx_pretraining_probe_5m_visible_intent_preloaded_v1.json"
DEFAULT_OUT = ROOT / "reports" / "strict_generator_mlx_private_adaptation_v1.json"
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints" / "strict_generator_mlx_private_adaptation_v1"

SEMANTIC_CONSTRUCTION_REPAIR_PROFILES: dict[str, dict[str, Any]] = {
    "strict_full_body_semantic_construction_v1": {
        "policy": "private_strict_full_body_semantic_construction_repair_profile_v1",
        "target_failure_modes": [
            "prompt_implies_branching_but_no_branch",
            "prompt_implies_structured_output_but_no_collection_construction",
            "prompt_implies_string_processing_but_no_string_ops",
            "loop_update_semantics_collapse_to_identity_or_additive_state",
            "body_token_transition_after_semantic_prefix_collapses_to_generic_accumulator",
        ],
        "required_components": [
            "semantic_plan_auxiliary",
            "semantic_plan_visible_operation_weighting",
            "source_condition_internalization_weighting",
            "semantic_slot_prefix_weighting",
            "loop_semantic_operation_weighting",
            "loop_expression_synthesis_weighting",
            "plan_conditioned_body_weighting",
            "update_contract_consistency_weighting",
            "source_contrastive_loss",
        ],
        "overrides": {
            "train_tier": "any",
            "tier_balanced_sampling": True,
            "private_residual_repair_split": True,
            "source_condition_operation_coverage_min_rows": 2,
            "semantic_plan_loss_weight": 0.08,
            "semantic_plan_visible_operation_loss_boost": 1.5,
            "source_contrastive_loss_weight": 0.05,
            "source_contrastive_prefix_tokens": 48,
            "enable_primary_dataflow_weights": True,
            "primary_dataflow_weight_scale": 1.2,
            "return_expression_loss_boost": 2.0,
            "source_condition_internalization_loss_boost": 2.4,
            "loop_semantic_operation_loss_boost": 3.0,
            "loop_semantic_operation_roles": "loop_semantic_update,top_level_semantic_finalizer",
            "semantic_slot_prefix_loss_boost": 2.2,
            "semantic_slot_prefix_roles": "plan,loop_source,update,guard,finalizer,state,binding,statement",
            "loop_expression_synthesis_loss_boost": 3.0,
            "loop_expression_synthesis_roles": (
                "loop_condition_expression,loop_update_expression,top_level_finalizer_expression"
            ),
            "plan_conditioned_body_loss_boost": 2.8,
            "plan_conditioned_body_roles": (
                "guard_expression,loop_source_expression,loop_condition_expression,"
                "loop_update_statement,plan_key_call_expression,final_return_expression"
            ),
            "update_contract_consistency_loss_boost": 3.0,
        },
        "score_semantics": (
            "Composes existing strict-generator private-only objectives that directly target learned "
            "full-body semantic construction. It does not add templates, renderers, tools, answer-family "
            "labels, public benchmark data, eval tests, eval solutions, teacher output, or candidate credit. "
            "The resulting checkpoint is private repair evidence only until verifier-passing behavior moves."
        ),
    },
    "strict_direct_body_emission_path_v1": {
        "policy": "private_strict_direct_body_emission_path_repair_profile_v1",
        "target_failure_modes": [
            "strict_decode_emits_zero_candidate_rows",
            "body_token_transition_after_plan_prefix_never_reaches_top_level_return",
            "non_loop_task_missing_state_binding_and_return_path",
            "loop_update_semantics_not_connected_to_final_return",
            "prompt_visible_source_condition_operation_not_internalized",
            "policy_gap_improves_without_replay_candidate_emission",
        ],
        "required_components": [
            "source_condition_internalization_weighting",
            "direct_body_emission_path_weighting",
            "local_return_closure_weighting",
            "loop_semantic_operation_weighting",
            "loop_expression_synthesis_weighting",
            "plan_conditioned_body_weighting",
            "update_contract_consistency_weighting",
            "source_contrastive_loss",
        ],
        "overrides": {
            "train_tier": "any",
            "tier_balanced_sampling": True,
            "private_residual_repair_split": True,
            "source_condition_operation_coverage_min_rows": 2,
            "source_contrastive_loss_weight": 0.05,
            "source_contrastive_prefix_tokens": 48,
            "enable_primary_dataflow_weights": True,
            "primary_dataflow_weight_scale": 1.2,
            "return_expression_loss_boost": 3.0,
            "source_condition_internalization_loss_boost": 2.6,
            "direct_body_emission_loss_boost": 4.0,
            "direct_body_emission_roles": (
                "top_level_state_binding,top_level_state_update,top_level_branch_guard,"
                "top_level_loop_statement,loop_source_expression,loop_condition_expression,"
                "loop_body_state_transition,branch_body_state_transition,top_level_local_state_return,"
                "top_level_return_expression"
            ),
            "local_return_closure_loss_boost": 5.0,
            "local_return_closure_roles": (
                "previous_local_state_transition,block_exit_local_return_closure,"
                "final_return_local_name,final_return_local_expression"
            ),
            "loop_semantic_operation_loss_boost": 3.2,
            "loop_semantic_operation_roles": "loop_semantic_update,top_level_semantic_finalizer",
            "loop_expression_synthesis_loss_boost": 3.2,
            "loop_expression_synthesis_roles": (
                "loop_condition_expression,loop_update_expression,top_level_finalizer_expression"
            ),
            "plan_conditioned_body_loss_boost": 3.2,
            "plan_conditioned_body_roles": (
                "guard_expression,loop_source_expression,loop_condition_expression,"
                "loop_update_statement,plan_key_call_expression,final_return_expression"
            ),
            "update_contract_consistency_loss_boost": 3.4,
        },
        "score_semantics": (
            "Composes private-only strict-generator objectives around the current zero-candidate wall: "
            "the direct prompt/signature body-token path must learn top-level state bindings, branch/loop "
            "guards, prompt-visible source-condition operations, loop body updates, and final return "
            "statements from admitted private solution bodies. It does not add templates, renderers, tools, "
            "answer-family labels, public benchmark data, eval tests, eval solutions, teacher output, or "
            "candidate credit. The checkpoint remains private repair evidence only until strict replay emits "
            "non-fallback learned candidates and verifier behavior moves."
        ),
    },
}


def apply_semantic_construction_repair_profile(args: argparse.Namespace) -> dict[str, Any]:
    profile_id = str(getattr(args, "semantic_construction_repair_profile", "none") or "none")
    if profile_id == "none":
        return {
            "enabled": False,
            "profile_id": "none",
            "policy": "not_enabled",
            "candidate_generation_credit": 0,
        }
    profile = dict_or_empty(SEMANTIC_CONSTRUCTION_REPAIR_PROFILES.get(profile_id))
    overrides = dict_or_empty(profile.get("overrides"))
    applied: dict[str, Any] = {}
    preserved: dict[str, Any] = {}

    for key, value in overrides.items():
        current = getattr(args, key)
        should_replace = False
        if isinstance(value, bool):
            should_replace = bool(value) and not bool(current)
        elif isinstance(value, (int, float)):
            should_replace = float(current or 0.0) <= 0.0 or (
                key == "primary_dataflow_weight_scale" and float(current or 0.0) <= 1.0
            )
        elif isinstance(value, str):
            should_replace = not str(current or "").strip() or (key == "train_tier" and str(current or "") == "any")
        else:
            should_replace = current in (None, "", 0, 0.0, False)
        if should_replace:
            setattr(args, key, value)
            applied[key] = value
        else:
            preserved[key] = current

    return {
        "enabled": True,
        "profile_id": profile_id,
        "policy": str(profile.get("policy") or "private_strict_generator_semantic_construction_repair_profile_v1"),
        "target_failure_modes": list(profile.get("target_failure_modes") or []),
        "required_components": list(profile.get("required_components") or []),
        "applied_overrides": applied,
        "preserved_operator_overrides": preserved,
        "public_calibration_eligible": False,
        "candidate_generation_credit": 0,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "external_inference_calls": 0,
        "score_semantics": str(profile.get("score_semantics") or ""),
    }


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def load_adaptation_private_train_rows(
    data_cfg: dict[str, Any],
    *,
    supplemental_private_train_jsonl: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary_path = resolve(str(data_cfg.get("train_jsonl") or ""))
    primary_rows = load_private_rows(primary_path, data_cfg)
    rows: list[dict[str, Any]] = list(primary_rows)
    seen = {row_id(row) for row in rows}
    configured_paths = [resolve(path) for path in string_list(data_cfg.get("adaptation_supplemental_train_jsonl"))]
    requested_paths = [resolve(path) for path in supplemental_private_train_jsonl]
    supplemental_paths: list[Path] = []
    supplemental_seen: set[str] = set()
    for path in [*configured_paths, *requested_paths]:
        key = str(path)
        if key in supplemental_seen:
            continue
        supplemental_seen.add(key)
        supplemental_paths.append(path)

    source_reports: list[dict[str, Any]] = []
    duplicate_row_count = 0
    added_row_count = 0
    for path in supplemental_paths:
        loaded = load_private_rows(path, data_cfg)
        path_added = 0
        path_duplicate = 0
        for row in loaded:
            key = row_id(row)
            if key in seen:
                duplicate_row_count += 1
                path_duplicate += 1
                continue
            seen.add(key)
            rows.append(row)
            added_row_count += 1
            path_added += 1
        source_reports.append(
            {
                "path": rel(path),
                "loaded_rows": len(loaded),
                "added_rows": path_added,
                "duplicate_rows": path_duplicate,
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "candidate_generation_credit": 0,
            }
        )

    audit = {
        "enabled": bool(supplemental_paths),
        "policy": "private_adaptation_supplemental_train_jsonl_v1",
        "primary_path": rel(primary_path),
        "primary_rows": len(primary_rows),
        "supplemental_sources": source_reports,
        "supplemental_source_count": len(supplemental_paths),
        "supplemental_added_rows": added_row_count,
        "supplemental_duplicate_rows": duplicate_row_count,
        "total_rows": len(rows),
        "score_semantics": (
            "Loads additional admitted private-training JSONL files into the existing strict-generator "
            "adaptation pool. The same forbidden public-row flag checks apply to every supplemental row. "
            "This changes private training coverage only; generation remains prompt/signature-only, and "
            "the report is public-calibration-ineligible until heldout replay behavior improves."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }
    return rows, audit


def source_condition_operation_tags_for_row(
    row: dict[str, Any],
    *,
    source_fields: list[Any],
    source_text_style: str,
    source_vocab: dict[str, int],
) -> set[str]:
    source_text = strict_generator_decode_source_text(
        row,
        source_fields,
        source_text_style=source_text_style,
        source_vocab=source_vocab,
    )
    expectation = source_condition_expectation_from_source_text(source_text)
    return {str(tag) for tag in list(expectation.get("operation_tags") or []) if str(tag)}


def apply_source_condition_operation_coverage_sampling(
    selected: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    *,
    requested_rows: int,
    min_rows_per_tag: int,
    seed: int,
    source_fields: list[Any],
    source_text_style: str,
    source_vocab: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = max(0, int(min_rows_per_tag or 0))
    if target <= 0 or not selected:
        return selected, {
            "enabled": False,
            "policy": "not_enabled",
            "min_rows_per_tag": target,
            "candidate_generation_credit": 0,
        }

    selected_rows = list(selected)
    selected_keys = {row_id(row) for row in selected_rows}
    row_tags: dict[str, set[str]] = {}
    pool_by_tag: dict[str, list[dict[str, Any]]] = {}
    for row in pool:
        key = row_id(row)
        tags = source_condition_operation_tags_for_row(
            row,
            source_fields=source_fields,
            source_text_style=source_text_style,
            source_vocab=source_vocab,
        )
        row_tags[key] = tags
        for tag in tags:
            pool_by_tag.setdefault(tag, []).append(row)

    def tag_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            for tag in row_tags.get(row_id(row), set()):
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    inserted_rows: list[dict[str, Any]] = []
    before_counts = tag_counts(selected_rows)
    for tag in sorted(pool_by_tag):
        current_count = before_counts.get(tag, 0) + sum(1 for row in inserted_rows if tag in row_tags.get(row_id(row), set()))
        need = max(0, target - current_count)
        if need <= 0:
            continue
        candidates = [row for row in pool_by_tag[tag] if row_id(row) not in selected_keys]
        for row in deterministic_sample(candidates, need, seed + stable_int(tag)):
            key = row_id(row)
            if key in selected_keys:
                continue
            selected_keys.add(key)
            inserted_rows.append(row)

    if inserted_rows:
        selected_rows.extend(inserted_rows)
    requested = max(0, int(requested_rows or len(selected_rows)))
    dropped_rows: list[dict[str, Any]] = []
    if requested and len(selected_rows) > requested:
        protected = {row_id(row) for row in inserted_rows}
        selected_rows.sort(key=lambda row: stable_hash(f"source-condition-coverage:{seed}:{row_id(row)}"))
        while len(selected_rows) > requested:
            drop_index = next(
                (
                    index
                    for index in range(len(selected_rows) - 1, -1, -1)
                    if row_id(selected_rows[index]) not in protected
                ),
                len(selected_rows) - 1,
            )
            dropped_rows.append(selected_rows.pop(drop_index))

    after_counts = tag_counts(selected_rows)
    selected_rows.sort(key=lambda row: stable_hash(f"source-condition-coverage-final:{seed}:{row_id(row)}"))
    return selected_rows, {
        "enabled": True,
        "policy": "private_prompt_visible_source_condition_operation_coverage_sampling_v1",
        "min_rows_per_tag": target,
        "requested_rows": requested_rows,
        "pool_rows": len(pool),
        "selected_rows_before": len(selected),
        "selected_rows_after": len(selected_rows),
        "inserted_rows": len(inserted_rows),
        "dropped_rows": len(dropped_rows),
        "operation_tag_counts_before": dict(sorted(before_counts.items())),
        "operation_tag_counts_after": dict(sorted(after_counts.items())),
        "pool_operation_tag_counts": {
            tag: len(rows)
            for tag, rows in sorted(pool_by_tag.items())
        },
        "inserted_row_ids": [row_id(row) for row in inserted_rows[:24]],
        "dropped_row_ids": [row_id(row) for row in dropped_rows[:24]],
        "score_semantics": (
            "Deterministically ensures rare prompt-visible operation contracts are represented in the "
            "private adaptation sample. Tags are derived only from strict prompt/signature source text, "
            "not from hidden tests, eval solutions, verifier labels, answer-family labels, public data, "
            "or target-derived decoder metadata. This is data coverage for supervised private training, "
            "not candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def stable_int(text: str) -> int:
    return int(stable_hash(text)[:8], 16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint-report", default=rel(DEFAULT_CHECKPOINT_REPORT))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--vocab", default="")
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--checkpoint-dir", default=rel(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--adaptation-id", default="private_train_adapt_smoke")
    parser.add_argument("--max-train-rows", type=int, default=512)
    parser.add_argument("--max-eval-rows", type=int, default=128)
    parser.add_argument(
        "--supplemental-private-train-jsonl",
        action="append",
        default=[],
        help=(
            "Additional admitted private-training JSONL file to append before adaptation sampling. "
            "May be repeated. Forbidden public-row flags are still enforced for every row."
        ),
    )
    parser.add_argument(
        "--train-tier",
        choices=["any", "simple_return", "loop_accumulate", "algorithmic_small"],
        default="any",
    )
    parser.add_argument(
        "--tier-balanced-sampling",
        action="store_true",
        help=(
            "When --train-tier any is used, sample admitted private rows evenly across "
            "the existing replay tiers before train/eval splitting. This is a sampling "
            "control only; it does not create rows or inspect public/eval tests."
        ),
    )
    parser.add_argument(
        "--private-residual-repair-split",
        action="store_true",
        help=(
            "Disable configured family-disjoint holdout exclusion for this private repair run, "
            "then report the checkpoint as row/variant-heldout repair evidence rather than "
            "family-disjoint evidence. This uses only admitted private rows and remains "
            "public-calibration-ineligible."
        ),
    )
    parser.add_argument(
        "--semantic-construction-repair-profile",
        choices=["none", *sorted(SEMANTIC_CONSTRUCTION_REPAIR_PROFILES)],
        default="none",
        help=(
            "Apply a named private-only semantic construction repair profile. Profiles only compose "
            "existing prompt/signature-to-body training losses and keep the checkpoint public-calibration "
            "ineligible until verifier behavior improves."
        ),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0004)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--semantic-plan-loss-weight", type=float, default=0.0)
    parser.add_argument("--semantic-plan-visible-operation-loss-boost", type=float, default=0.0)
    parser.add_argument("--rehearsal-rows", type=int, default=0)
    parser.add_argument("--source-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--source-contrastive-margin", type=float, default=0.25)
    parser.add_argument("--source-contrastive-prefix-tokens", type=int, default=0)
    parser.add_argument("--guard-clean-target-loss-boost", type=float, default=1.0)
    parser.add_argument("--guard-rejected-target-loss-weight", type=float, default=1.0)
    parser.add_argument("--negative-replay-candidates", default="")
    parser.add_argument("--negative-replay-report", default="")
    parser.add_argument("--negative-starvation-report", default="")
    parser.add_argument("--max-negative-replay-rows", type=int, default=0)
    parser.add_argument("--max-negative-starvation-rows", type=int, default=0)
    parser.add_argument("--negative-unlikelihood-weight", type=float, default=0.0)
    parser.add_argument("--negative-unlikelihood-cap", type=float, default=8.0)
    parser.add_argument("--negative-stage-weighting", choices=["uniform", "reward_inverse"], default="uniform")
    parser.add_argument("--negative-min-stage-weight", type=float, default=0.25)
    parser.add_argument("--negative-max-stage-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-replay-loss-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-replay-objective", choices=["margin", "dpo", "ipo"], default="margin")
    parser.add_argument("--pairwise-dpo-beta", type=float, default=0.1)
    parser.add_argument("--pairwise-ipo-target-margin", type=float, default=1.0)
    parser.add_argument("--pairwise-replay-margin", type=float, default=0.25)
    parser.add_argument("--pairwise-replay-prefix-tokens", type=int, default=0)
    parser.add_argument("--action-trace-replay-loss-boost", type=float, default=0.0)
    parser.add_argument("--enable-primary-dataflow-weights", action="store_true")
    parser.add_argument("--primary-dataflow-weight-scale", type=float, default=1.0)
    parser.add_argument("--return-expression-loss-boost", type=float, default=0.0)
    parser.add_argument("--default-parameter-return-loss-boost", type=float, default=0.0)
    parser.add_argument("--truthiness-guard-loss-boost", type=float, default=0.0)
    parser.add_argument("--source-condition-internalization-loss-boost", type=float, default=0.0)
    parser.add_argument("--source-condition-operation-coverage-min-rows", type=int, default=0)
    parser.add_argument("--loop-operation-loss-boost", type=float, default=0.0)
    parser.add_argument("--loop-statement-action-loss-boost", type=float, default=0.0)
    parser.add_argument(
        "--loop-statement-action-roles",
        default="",
        help=(
            "Optional comma-separated private target span roles to boost, such as "
            "loop_body_update,top_level_finalizer. Empty means all extracted roles."
        ),
    )
    parser.add_argument("--loop-semantic-operation-loss-boost", type=float, default=0.0)
    parser.add_argument(
        "--loop-semantic-operation-roles",
        default="",
        help=(
            "Optional comma-separated private target semantic roles to boost, such as "
            "loop_semantic_update,top_level_semantic_finalizer. Empty means all extracted roles."
        ),
    )
    parser.add_argument("--semantic-slot-prefix-loss-boost", type=float, default=0.0)
    parser.add_argument(
        "--semantic-slot-prefix-roles",
        default="",
        help=(
            "Optional comma-separated semantic slot-prefix roles to boost before SLOT:BODY_START, "
            "such as update,finalizer,guard,init,return_shape,loop_source,state,binding. Empty means all slot roles."
        ),
    )
    parser.add_argument("--loop-expression-synthesis-loss-boost", type=float, default=0.0)
    parser.add_argument(
        "--loop-expression-synthesis-roles",
        default="",
        help=(
            "Optional comma-separated loop expression roles to boost, such as "
            "loop_update_expression,top_level_finalizer_expression. Empty means all extracted roles."
        ),
    )
    parser.add_argument("--plan-conditioned-body-loss-boost", type=float, default=0.0)
    parser.add_argument(
        "--plan-conditioned-body-roles",
        default="",
        help=(
            "Optional comma-separated plan-conditioned body roles to boost, such as "
            "guard_expression,loop_update_statement,plan_key_call_expression,final_return_expression. "
            "Empty means all extracted roles."
        ),
    )
    parser.add_argument("--update-contract-consistency-loss-boost", type=float, default=0.0)
    parser.add_argument("--direct-body-emission-loss-boost", type=float, default=0.0)
    parser.add_argument(
        "--direct-body-emission-roles",
        default="",
        help=(
            "Optional comma-separated direct body-emission roles to boost, such as "
            "top_level_state_binding,top_level_local_state_return,loop_body_state_transition. "
            "Empty means all extracted roles."
        ),
    )
    parser.add_argument("--local-return-closure-loss-boost", type=float, default=0.0)
    parser.add_argument(
        "--local-return-closure-roles",
        default="",
        help=(
            "Optional comma-separated local-return closure roles to boost, such as "
            "previous_local_state_transition,block_exit_local_return_closure,final_return_local_name. "
            "Empty means all extracted roles."
        ),
    )
    parser.add_argument("--seed", type=int, default=23017)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    semantic_construction_repair_profile = apply_semantic_construction_repair_profile(args)
    config = read_json(resolve(args.config))
    checkpoint_report = read_json(resolve(args.checkpoint_report)) if str(args.checkpoint_report or "").strip() else {}
    checkpoint, vocab = resolve_checkpoint_paths(args, checkpoint_report)
    report = run_adaptation(
        config,
        config_path=str(args.config),
        checkpoint_report_path=str(args.checkpoint_report),
        checkpoint_path=checkpoint,
        vocab_path=vocab,
        checkpoint_dir=resolve(args.checkpoint_dir),
        adaptation_id=str(args.adaptation_id or "private_train_adapt_smoke"),
        max_train_rows=max(1, int(args.max_train_rows or 1)),
        max_eval_rows=max(1, int(args.max_eval_rows or 1)),
        supplemental_private_train_jsonl=[
            resolve(path) for path in list(args.supplemental_private_train_jsonl or []) if str(path).strip()
        ],
        train_tier=str(args.train_tier or "any"),
        tier_balanced_sampling=bool(args.tier_balanced_sampling),
        private_residual_repair_split=bool(args.private_residual_repair_split),
        epochs=max(1, int(args.epochs or 1)),
        batch_size=max(1, int(args.batch_size or 1)),
        learning_rate=float(args.learning_rate if args.learning_rate is not None else 0.0004),
        weight_decay=float(args.weight_decay if args.weight_decay is not None else 0.0001),
        semantic_plan_loss_weight=max(0.0, float(args.semantic_plan_loss_weight if args.semantic_plan_loss_weight is not None else 0.0)),
        semantic_plan_visible_operation_loss_boost=max(
            0.0,
            float(
                args.semantic_plan_visible_operation_loss_boost
                if args.semantic_plan_visible_operation_loss_boost is not None
                else 0.0
            ),
        ),
        rehearsal_rows=max(0, int(args.rehearsal_rows or 0)),
        source_contrastive_loss_weight=max(0.0, float(args.source_contrastive_loss_weight if args.source_contrastive_loss_weight is not None else 0.0)),
        source_contrastive_margin=max(0.0, float(args.source_contrastive_margin if args.source_contrastive_margin is not None else 0.0)),
        source_contrastive_prefix_tokens=max(0, int(args.source_contrastive_prefix_tokens or 0)),
        guard_clean_target_loss_boost=max(0.0, float(args.guard_clean_target_loss_boost if args.guard_clean_target_loss_boost is not None else 1.0)),
        guard_rejected_target_loss_weight=max(0.0, float(args.guard_rejected_target_loss_weight if args.guard_rejected_target_loss_weight is not None else 1.0)),
        negative_replay_candidates=resolve(args.negative_replay_candidates) if str(args.negative_replay_candidates or "").strip() else None,
        negative_replay_report=resolve(args.negative_replay_report) if str(args.negative_replay_report or "").strip() else None,
        negative_starvation_report=resolve(args.negative_starvation_report) if str(args.negative_starvation_report or "").strip() else None,
        max_negative_replay_rows=max(0, int(args.max_negative_replay_rows or 0)),
        max_negative_starvation_rows=max(0, int(args.max_negative_starvation_rows or 0)),
        negative_unlikelihood_weight=max(0.0, float(args.negative_unlikelihood_weight if args.negative_unlikelihood_weight is not None else 0.0)),
        negative_unlikelihood_cap=max(0.0, float(args.negative_unlikelihood_cap if args.negative_unlikelihood_cap is not None else 8.0)),
        negative_stage_weighting=str(args.negative_stage_weighting or "uniform"),
        negative_min_stage_weight=max(0.0, float(args.negative_min_stage_weight if args.negative_min_stage_weight is not None else 0.25)),
        negative_max_stage_weight=max(0.0, float(args.negative_max_stage_weight if args.negative_max_stage_weight is not None else 1.0)),
        pairwise_replay_loss_weight=max(0.0, float(args.pairwise_replay_loss_weight if args.pairwise_replay_loss_weight is not None else 0.0)),
        pairwise_replay_objective=str(args.pairwise_replay_objective or "margin"),
        pairwise_dpo_beta=max(1e-6, float(args.pairwise_dpo_beta if args.pairwise_dpo_beta is not None else 0.1)),
        pairwise_ipo_target_margin=float(
            args.pairwise_ipo_target_margin if args.pairwise_ipo_target_margin is not None else 1.0
        ),
        pairwise_replay_margin=max(0.0, float(args.pairwise_replay_margin if args.pairwise_replay_margin is not None else 0.25)),
        pairwise_replay_prefix_tokens=max(0, int(args.pairwise_replay_prefix_tokens or 0)),
        action_trace_replay_loss_boost=max(0.0, float(args.action_trace_replay_loss_boost if args.action_trace_replay_loss_boost is not None else 0.0)),
        enable_primary_dataflow_weights=bool(args.enable_primary_dataflow_weights),
        primary_dataflow_weight_scale=max(0.0, float(args.primary_dataflow_weight_scale if args.primary_dataflow_weight_scale is not None else 1.0)),
        return_expression_loss_boost=max(0.0, float(args.return_expression_loss_boost if args.return_expression_loss_boost is not None else 0.0)),
        default_parameter_return_loss_boost=max(0.0, float(args.default_parameter_return_loss_boost if args.default_parameter_return_loss_boost is not None else 0.0)),
        truthiness_guard_loss_boost=max(0.0, float(args.truthiness_guard_loss_boost if args.truthiness_guard_loss_boost is not None else 0.0)),
        source_condition_internalization_loss_boost=max(
            0.0,
            float(args.source_condition_internalization_loss_boost if args.source_condition_internalization_loss_boost is not None else 0.0),
        ),
        source_condition_operation_coverage_min_rows=max(
            0,
            int(
                args.source_condition_operation_coverage_min_rows
                if args.source_condition_operation_coverage_min_rows is not None
                else 0
            ),
        ),
        loop_operation_loss_boost=max(0.0, float(args.loop_operation_loss_boost if args.loop_operation_loss_boost is not None else 0.0)),
        loop_statement_action_loss_boost=max(
            0.0,
            float(args.loop_statement_action_loss_boost if args.loop_statement_action_loss_boost is not None else 0.0),
        ),
        loop_statement_action_roles=str(args.loop_statement_action_roles or ""),
        loop_semantic_operation_loss_boost=max(
            0.0,
            float(args.loop_semantic_operation_loss_boost if args.loop_semantic_operation_loss_boost is not None else 0.0),
        ),
        loop_semantic_operation_roles=str(args.loop_semantic_operation_roles or ""),
        semantic_slot_prefix_loss_boost=max(
            0.0,
            float(args.semantic_slot_prefix_loss_boost if args.semantic_slot_prefix_loss_boost is not None else 0.0),
        ),
        semantic_slot_prefix_roles=str(args.semantic_slot_prefix_roles or ""),
        loop_expression_synthesis_loss_boost=max(
            0.0,
            float(args.loop_expression_synthesis_loss_boost if args.loop_expression_synthesis_loss_boost is not None else 0.0),
        ),
        loop_expression_synthesis_roles=str(args.loop_expression_synthesis_roles or ""),
        plan_conditioned_body_loss_boost=max(
            0.0,
            float(args.plan_conditioned_body_loss_boost if args.plan_conditioned_body_loss_boost is not None else 0.0),
        ),
        plan_conditioned_body_roles=str(args.plan_conditioned_body_roles or ""),
        update_contract_consistency_loss_boost=max(
            0.0,
            float(args.update_contract_consistency_loss_boost if args.update_contract_consistency_loss_boost is not None else 0.0),
        ),
        direct_body_emission_loss_boost=max(
            0.0,
            float(args.direct_body_emission_loss_boost if args.direct_body_emission_loss_boost is not None else 0.0),
        ),
        direct_body_emission_roles=str(args.direct_body_emission_roles or ""),
        local_return_closure_loss_boost=max(
            0.0,
            float(args.local_return_closure_loss_boost if args.local_return_closure_loss_boost is not None else 0.0),
        ),
        local_return_closure_roles=str(args.local_return_closure_roles or ""),
        semantic_construction_repair_profile=semantic_construction_repair_profile,
        seed=int(args.seed or 23017),
        execute=bool(args.execute),
    )
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW", "PLANNED"} else 2


def run_adaptation(
    config: dict[str, Any],
    *,
    config_path: str,
    checkpoint_report_path: str,
    checkpoint_path: Path,
    vocab_path: Path,
    checkpoint_dir: Path,
    adaptation_id: str,
    max_train_rows: int,
    max_eval_rows: int,
    supplemental_private_train_jsonl: list[Path],
    train_tier: str,
    tier_balanced_sampling: bool,
    private_residual_repair_split: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    semantic_plan_loss_weight: float,
    semantic_plan_visible_operation_loss_boost: float,
    rehearsal_rows: int,
    source_contrastive_loss_weight: float,
    source_contrastive_margin: float,
    source_contrastive_prefix_tokens: int,
    guard_clean_target_loss_boost: float,
    guard_rejected_target_loss_weight: float,
    negative_replay_candidates: Path | None,
    negative_replay_report: Path | None,
    negative_starvation_report: Path | None,
    max_negative_replay_rows: int,
    max_negative_starvation_rows: int,
    negative_unlikelihood_weight: float,
    negative_unlikelihood_cap: float,
    negative_stage_weighting: str,
    negative_min_stage_weight: float,
    negative_max_stage_weight: float,
    pairwise_replay_loss_weight: float,
    pairwise_replay_objective: str,
    pairwise_dpo_beta: float,
    pairwise_ipo_target_margin: float,
    pairwise_replay_margin: float,
    pairwise_replay_prefix_tokens: int,
    action_trace_replay_loss_boost: float,
    enable_primary_dataflow_weights: bool,
    primary_dataflow_weight_scale: float,
    return_expression_loss_boost: float,
    default_parameter_return_loss_boost: float,
    truthiness_guard_loss_boost: float,
    source_condition_internalization_loss_boost: float,
    source_condition_operation_coverage_min_rows: int,
    loop_operation_loss_boost: float,
    loop_statement_action_loss_boost: float,
    loop_statement_action_roles: str,
    loop_semantic_operation_loss_boost: float,
    loop_semantic_operation_roles: str,
    semantic_slot_prefix_loss_boost: float,
    semantic_slot_prefix_roles: str,
    loop_expression_synthesis_loss_boost: float,
    loop_expression_synthesis_roles: str,
    plan_conditioned_body_loss_boost: float,
    plan_conditioned_body_roles: str,
    update_contract_consistency_loss_boost: float,
    direct_body_emission_loss_boost: float,
    direct_body_emission_roles: str,
    local_return_closure_loss_boost: float,
    local_return_closure_roles: str,
    semantic_construction_repair_profile: dict[str, Any],
    seed: int,
    execute: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    data_cfg = dict_or_empty(config.get("data"))
    configured_supplemental_train_jsonl = [
        resolve(path) for path in string_list(data_cfg.get("adaptation_supplemental_train_jsonl"))
    ]
    effective_supplemental_train_jsonl = [
        *configured_supplemental_train_jsonl,
        *list(supplemental_private_train_jsonl or []),
    ]
    if not execute:
        return {
            "policy": "project_theseus_strict_generator_mlx_private_adaptation_v1",
            "created_utc": now(),
            "execute": False,
            "trigger_state": "PLANNED",
            "summary": {
                "adaptation_id": adaptation_id,
                "checkpoint": rel(checkpoint_path),
                "vocab": rel(vocab_path),
                "runtime_overrides": {
                    "primary_dataflow_weights_enabled": bool(enable_primary_dataflow_weights),
                    "primary_dataflow_weight_scale": float(primary_dataflow_weight_scale),
                    "return_expression_loss_boost": float(return_expression_loss_boost or 0.0),
                    "default_parameter_return_loss_boost": float(default_parameter_return_loss_boost or 0.0),
                    "truthiness_guard_loss_boost": float(truthiness_guard_loss_boost or 0.0),
                    "source_condition_internalization_loss_boost": float(source_condition_internalization_loss_boost or 0.0),
                    "source_condition_operation_coverage_min_rows": int(
                        source_condition_operation_coverage_min_rows or 0
                    ),
                    "loop_operation_loss_boost": float(loop_operation_loss_boost or 0.0),
                    "loop_statement_action_loss_boost": float(loop_statement_action_loss_boost or 0.0),
                    "loop_statement_action_roles": str(loop_statement_action_roles or ""),
                    "loop_semantic_operation_loss_boost": float(loop_semantic_operation_loss_boost or 0.0),
                    "loop_semantic_operation_roles": str(loop_semantic_operation_roles or ""),
                    "semantic_slot_prefix_loss_boost": float(semantic_slot_prefix_loss_boost or 0.0),
                    "semantic_slot_prefix_roles": str(semantic_slot_prefix_roles or ""),
                    "loop_expression_synthesis_loss_boost": float(loop_expression_synthesis_loss_boost or 0.0),
                    "loop_expression_synthesis_roles": str(loop_expression_synthesis_roles or ""),
                    "plan_conditioned_body_loss_boost": float(plan_conditioned_body_loss_boost or 0.0),
                    "plan_conditioned_body_roles": str(plan_conditioned_body_roles or ""),
                    "update_contract_consistency_loss_boost": float(update_contract_consistency_loss_boost or 0.0),
                    "direct_body_emission_loss_boost": float(direct_body_emission_loss_boost or 0.0),
                    "direct_body_emission_roles": str(direct_body_emission_roles or ""),
                    "local_return_closure_loss_boost": float(local_return_closure_loss_boost or 0.0),
                    "local_return_closure_roles": str(local_return_closure_roles or ""),
                    "semantic_construction_repair_profile": semantic_construction_repair_profile,
                    "train_tier": train_tier,
                    "tier_balanced_sampling": bool(tier_balanced_sampling),
                    "private_residual_repair_split": bool(private_residual_repair_split),
                    "supplemental_private_train_jsonl": [
                        rel(path) for path in effective_supplemental_train_jsonl
                    ],
                    "negative_replay_requested": bool(negative_replay_candidates),
                    "negative_starvation_replay_requested": bool(negative_starvation_report),
                    "negative_unlikelihood_weight": float(negative_unlikelihood_weight or 0.0),
                    "negative_stage_weighting": negative_stage_weighting,
                    "pairwise_replay_loss_weight": float(pairwise_replay_loss_weight or 0.0),
                    "pairwise_replay_objective": str(pairwise_replay_objective or "margin"),
                    "pairwise_dpo_beta": float(pairwise_dpo_beta or 0.0),
                    "pairwise_ipo_target_margin": float(pairwise_ipo_target_margin or 0.0),
                    "pairwise_replay_margin": float(pairwise_replay_margin or 0.0),
                    "pairwise_replay_prefix_tokens": int(pairwise_replay_prefix_tokens or 0),
                    "action_trace_replay_loss_boost": float(action_trace_replay_loss_boost or 0.0),
                    "semantic_plan_visible_operation_loss_boost": float(semantic_plan_visible_operation_loss_boost or 0.0),
                },
                "semantic_construction_repair_profile": semantic_construction_repair_profile,
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
            "policy": "project_theseus_strict_generator_mlx_private_adaptation_v1",
            "created_utc": now(),
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

    random.seed(seed)
    mx.random.seed(seed)
    loaded = load_mlx_checkpoint(vocab_path, checkpoint_path, mx=mx, nn=nn)
    model = loaded["model"]
    pairwise_objective = str(pairwise_replay_objective or "margin").strip().lower() or "margin"
    if pairwise_objective not in {"margin", "dpo", "ipo"}:
        pairwise_objective = "margin"
    reference_model = None
    if float(pairwise_replay_loss_weight or 0.0) > 0.0 and pairwise_objective in {"dpo", "ipo"}:
        reference_model = load_mlx_checkpoint(vocab_path, checkpoint_path, mx=mx, nn=nn)["model"]
        reference_model.eval()
    vocab_payload = loaded["vocab_payload"]
    source_vocab = dict_or_empty(vocab_payload.get("source_vocab"))
    target_vocab = dict_or_empty(vocab_payload.get("target_vocab"))
    max_source = int(vocab_payload.get("max_source") or 96)
    max_target = int(vocab_payload.get("max_target") or 160)
    target_mode = str(vocab_payload.get("target_mode") or get_path(config, ["body_structure_decoder", "target_mode"], "body_tokens"))
    source_text_style = checkpoint_source_text_style(config, vocab_payload)
    matched_budget = dict_or_empty(config.get("matched_budget"))
    text_views = dict_or_empty(config.get("text_views"))
    source_fields = list(text_views.get("sts_on") or [])
    all_rows, supplemental_train_audit = load_adaptation_private_train_rows(
        data_cfg,
        supplemental_private_train_jsonl=supplemental_private_train_jsonl,
    )
    all_rows, holdout_exclusion = exclude_family_disjoint_holdout_rows(
        config,
        all_rows,
        private_residual_repair_split=private_residual_repair_split,
    )
    tier_selection = select_private_train_tier_rows(all_rows, tier=train_tier)
    all_rows = list(tier_selection.get("rows") or [])
    if tier_balanced_sampling and str(train_tier or "any") == "any":
        balanced_selection = tier_balanced_private_train_sample(
            all_rows,
            limit=max_train_rows + max_eval_rows,
            seed=seed,
        )
        selected = list(balanced_selection.get("rows") or [])
        tier_selection["balanced_sampling"] = private_train_balanced_sample_summary(balanced_selection)
    else:
        selected = deterministic_sample(all_rows, max_train_rows + max_eval_rows, seed)
        tier_selection["balanced_sampling"] = {
            "enabled": False,
            "policy": "not_enabled",
            "reason": "tier_balanced_sampling_flag_not_set_or_specific_train_tier_selected",
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    selected, source_condition_coverage_sampling = apply_source_condition_operation_coverage_sampling(
        selected,
        all_rows,
        requested_rows=max_train_rows + max_eval_rows,
        min_rows_per_tag=source_condition_operation_coverage_min_rows,
        seed=seed,
        source_fields=source_fields,
        source_text_style=source_text_style,
        source_vocab=source_vocab,
    )
    tier_selection["source_condition_operation_coverage_sampling"] = source_condition_coverage_sampling
    eval_rows = selected[: min(max_eval_rows, max(1, len(selected) // 5))]
    train_rows = selected[len(eval_rows) : len(eval_rows) + max_train_rows]
    if not train_rows or not eval_rows:
        return {
            "policy": "project_theseus_strict_generator_mlx_private_adaptation_v1",
            "created_utc": now(),
            "execute": True,
            "trigger_state": "RED",
            "summary": {
                "reason": "insufficient_private_train_rows",
                "selected_rows": len(selected),
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "supplemental_private_train_audit": supplemental_train_audit,
                "source_condition_operation_coverage_sampling": source_condition_coverage_sampling,
                "public_training_rows": 0,
                "external_inference_calls": 0,
            },
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        }

    train_source_texts = [
        strict_generator_decode_source_text(
            row,
            source_fields,
            source_text_style=source_text_style,
            source_vocab=source_vocab,
        )
        for row in train_rows
    ]
    eval_source_texts = [
        strict_generator_decode_source_text(
            row,
            source_fields,
            source_text_style=source_text_style,
            source_vocab=source_vocab,
        )
        for row in eval_rows
    ]
    source_text_audit = {
        "train": strict_generator_source_text_audit(train_rows, train_source_texts),
        "heldout_private_train": strict_generator_source_text_audit(eval_rows, eval_source_texts),
    }
    train_bodies = [str(row.get("solution_body") or "") for row in train_rows]
    eval_bodies = [str(row.get("solution_body") or "") for row in eval_rows]
    rehearsal = select_rehearsal_examples(
        config,
        vocab_payload,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        target_mode=target_mode,
        max_source=max_source,
        max_target=max_target,
        checkpoint_dir=checkpoint_path.parent,
        source_fields=source_fields,
        source_text_style=source_text_style,
        requested_rows=rehearsal_rows,
        seed=seed,
    )
    rehearsal_examples = list(rehearsal.get("examples") or [])
    if rehearsal_examples:
        train_source_texts.extend(str(row.get("source_text") or "") for row in rehearsal_examples)
        train_bodies.extend(str(row.get("body") or "") for row in rehearsal_examples)
    if rehearsal_examples:
        rehearsal_audit_rows = [
            {
                "solution_body": str(row.get("body") or ""),
                "tests": "",
            }
            for row in rehearsal_examples
        ]
        source_text_audit["train_with_rehearsal"] = strict_generator_source_text_audit(
            [*train_rows, *rehearsal_audit_rows],
            train_source_texts,
        )
    train_source_rows = encode_many(train_source_texts, source_vocab, max_source)
    eval_source_rows = encode_many(eval_source_texts, source_vocab, max_source)
    train_target_rows = encode_target_rows(train_bodies, target_vocab, max_target, target_mode=target_mode)
    eval_target_rows = encode_target_rows(eval_bodies, target_vocab, max_target, target_mode=target_mode)
    negative_replay = select_negative_replay_examples(
        config,
        all_rows,
        candidate_path=negative_replay_candidates,
        report_path=negative_replay_report,
        max_rows=max_negative_replay_rows,
        source_fields=source_fields,
        source_text_style=source_text_style,
        source_vocab=source_vocab,
    )
    starvation_replay = select_decode_starvation_replay_examples(
        all_rows,
        report_path=negative_starvation_report,
        max_rows=max_negative_starvation_rows,
        source_fields=source_fields,
        source_text_style=source_text_style,
        source_vocab=source_vocab,
    )
    negative_replay = merge_negative_replay_sources(negative_replay, starvation_replay)
    negative_replay_examples = list(negative_replay.get("examples") or [])
    negative_replay_active = (
        bool(negative_replay_examples)
        and float(negative_unlikelihood_weight or 0.0) > 0.0
        and float(negative_unlikelihood_cap or 0.0) > 0.0
    )
    pairwise_replay_active = (
        bool(negative_replay_examples)
        and float(pairwise_replay_loss_weight or 0.0) > 0.0
    )
    if negative_replay_examples:
        negative_source_texts = [str(row.get("source_text") or "") for row in negative_replay_examples]
        negative_bodies = [str(row.get("body") or "") for row in negative_replay_examples]
        pairwise_positive_bodies = [str(row.get("accepted_body") or "") for row in negative_replay_examples]
        negative_source_rows = encode_many(negative_source_texts, source_vocab, max_source)
        negative_target_rows = encode_target_rows(negative_bodies, target_vocab, max_target, target_mode=target_mode)
        pairwise_positive_target_rows = encode_target_rows(pairwise_positive_bodies, target_vocab, max_target, target_mode=target_mode)
        negative_token_weight_rows, negative_stage_weight_summary = negative_replay_token_weight_rows(
            negative_replay_examples,
            negative_target_rows,
            mode=negative_stage_weighting,
            min_stage_weight=negative_min_stage_weight,
            max_stage_weight=negative_max_stage_weight,
        )
    else:
        negative_source_rows = []
        negative_target_rows = []
        pairwise_positive_target_rows = []
        negative_token_weight_rows = []
        negative_stage_weight_summary = {
            "enabled": False,
            "policy": "not_active",
            "mode": negative_stage_weighting,
            "rows": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    semantic_plan_enabled = float(semantic_plan_loss_weight or 0.0) > 0.0 and hasattr(model, "semantic_plan_logits")
    train_examples = [{"source_text": text} for text in train_source_texts]
    loss_weight_budget: dict[str, Any] = {}
    if enable_primary_dataflow_weights:
        apply_primary_dataflow_weight_override(loss_weight_budget, scale=primary_dataflow_weight_scale)
    return_expression_weighting = apply_return_expression_weight_override(
        loss_weight_budget,
        boost=return_expression_loss_boost,
    )
    token_weight_rows, loss_weight_summary = target_loss_weight_rows(
        train_target_rows,
        train_examples=train_examples,
        target_vocab=target_vocab,
        budget=loss_weight_budget,
        matched_budget=matched_budget,
    )
    target_guard_rows, target_guard_summary = strict_target_guard_rows(
        train_bodies,
        train_source_texts,
        split_name="train",
    )
    eval_target_guard_rows, eval_target_guard_summary = strict_target_guard_rows(
        eval_bodies,
        eval_source_texts,
        split_name="heldout_private_train",
    )
    token_weight_rows, guard_weight_summary = apply_guard_clean_target_weights(
        token_weight_rows,
        target_guard_rows,
        guard_clean_boost=guard_clean_target_loss_boost,
        rejected_weight=guard_rejected_target_loss_weight,
    )
    token_weight_rows, default_parameter_return_weighting = apply_default_parameter_return_weights(
        token_weight_rows,
        train_target_rows,
        train_bodies,
        train_source_texts,
        target_vocab=target_vocab,
        boost=default_parameter_return_loss_boost,
    )
    token_weight_rows, truthiness_guard_weighting = apply_truthiness_guard_weights(
        token_weight_rows,
        train_target_rows,
        train_bodies,
        train_source_texts,
        target_vocab=target_vocab,
        boost=truthiness_guard_loss_boost,
    )
    token_weight_rows, source_condition_internalization_weighting = apply_source_condition_internalization_weights(
        token_weight_rows,
        train_target_rows,
        train_bodies,
        train_source_texts,
        target_vocab=target_vocab,
        boost=source_condition_internalization_loss_boost,
    )
    token_weight_rows, loop_operation_weighting = apply_loop_operation_weights(
        token_weight_rows,
        train_target_rows,
        train_bodies,
        target_vocab=target_vocab,
        boost=loop_operation_loss_boost,
    )
    token_weight_rows, loop_statement_action_weighting = apply_loop_statement_action_weights(
        token_weight_rows,
        train_target_rows,
        train_bodies,
        target_vocab=target_vocab,
        boost=loop_statement_action_loss_boost,
        roles=loop_statement_action_roles,
    )
    token_weight_rows, loop_semantic_operation_weighting = apply_loop_semantic_operation_weights(
        token_weight_rows,
        train_target_rows,
        train_bodies,
        target_vocab=target_vocab,
        boost=loop_semantic_operation_loss_boost,
        roles=loop_semantic_operation_roles,
    )
    token_weight_rows, semantic_slot_prefix_weighting = apply_semantic_slot_prefix_weights(
        token_weight_rows,
        train_target_rows,
        target_vocab=target_vocab,
        boost=semantic_slot_prefix_loss_boost,
        roles=semantic_slot_prefix_roles,
    )
    token_weight_rows, loop_expression_synthesis_weighting = apply_loop_expression_synthesis_weights(
        token_weight_rows,
        train_target_rows,
        train_bodies,
        target_vocab=target_vocab,
        boost=loop_expression_synthesis_loss_boost,
        roles=loop_expression_synthesis_roles,
    )
    token_weight_rows, plan_conditioned_body_weighting = apply_plan_conditioned_body_semantic_weights(
        token_weight_rows,
        train_target_rows,
        train_bodies,
        target_vocab=target_vocab,
        boost=plan_conditioned_body_loss_boost,
        roles=plan_conditioned_body_roles,
    )
    token_weight_rows, update_contract_consistency_weighting = apply_update_contract_consistency_weights(
        token_weight_rows,
        train_target_rows,
        train_bodies,
        target_vocab=target_vocab,
        boost=update_contract_consistency_loss_boost,
    )
    token_weight_rows, direct_body_emission_weighting = apply_direct_body_emission_path_weights(
        token_weight_rows,
        train_target_rows,
        train_bodies,
        target_vocab=target_vocab,
        boost=direct_body_emission_loss_boost,
        roles=direct_body_emission_roles,
    )
    token_weight_rows, local_return_closure_weighting = apply_local_return_closure_weights(
        token_weight_rows,
        train_target_rows,
        train_bodies,
        target_vocab=target_vocab,
        boost=local_return_closure_loss_boost,
        roles=local_return_closure_roles,
    )
    (
        pairwise_positive_weight_rows,
        pairwise_negative_weight_rows,
        pairwise_replay_weight_summary,
    ) = pairwise_replay_token_weight_rows(
        negative_replay_examples,
        pairwise_positive_target_rows,
        negative_target_rows,
        target_vocab=target_vocab,
        active=pairwise_replay_active,
        action_trace_boost=action_trace_replay_loss_boost,
    )
    pad_id = int(target_vocab.get("<pad>", 0))
    semantic_plan_class_ids = sorted(
        int(token_id)
        for token, token_id in target_vocab.items()
        if str(token).startswith("SLOT:PLAN_")
    )
    semantic_plan_class_id_array = (
        mx.array(semantic_plan_class_ids, dtype=mx.int32)
        if semantic_plan_enabled and semantic_plan_class_ids
        else None
    )
    train_plan_targets, train_plan_summary = semantic_plan_target_ids(
        [{"body": body} for body in train_bodies],
        target_vocab=target_vocab,
        pad_id=pad_id,
        enabled=semantic_plan_enabled,
    )
    eval_plan_targets, eval_plan_summary = semantic_plan_target_ids(
        [{"body": body} for body in eval_bodies],
        target_vocab=target_vocab,
        pad_id=pad_id,
        enabled=semantic_plan_enabled,
    )
    train_plan_weights, plan_balance_summary = semantic_plan_sample_weights(
        train_plan_targets,
        target_summary=train_plan_summary,
        pad_id=pad_id,
        config={
            "enabled": semantic_plan_enabled,
            "class_balance_enabled": True,
            "class_balance_policy": "private_adaptation_inverse_sqrt_plan_frequency_v1",
            "class_balance_min_weight": 0.35,
            "class_balance_max_weight": 4.0,
        },
    )
    train_plan_weights, visible_operation_plan_weighting = apply_visible_operation_plan_sample_weights(
        train_plan_weights,
        train_plan_targets,
        train_bodies,
        train_source_texts,
        target_vocab=target_vocab,
        pad_id=pad_id,
        boost=semantic_plan_visible_operation_loss_boost,
    )
    before = parameter_snapshot(model, mlx_utils, mx)
    heldout_before = evaluate_loss_mlx(
        model,
        eval_source_rows,
        eval_target_rows,
        batch_size=batch_size,
        pad_id=pad_id,
        mx=mx,
        nn=nn,
    )
    semantic_plan_before = evaluate_semantic_plan_mlx(
        model,
        eval_source_rows,
        eval_plan_targets,
        batch_size=batch_size,
        pad_id=pad_id,
        mx=mx,
        nn=nn,
        plan_class_ids=semantic_plan_class_id_array,
    ) if semantic_plan_enabled else {"loss": None, "accuracy": None, "active_target_count": 0}
    source_contrastive_active = float(source_contrastive_loss_weight or 0.0) > 0.0
    source_contrastive_span_mode = "prefix"
    source_contrastive_body_start_id = int(target_vocab.get(PLAN_BODY_START_TOKEN, -1))
    source_contrast_before = evaluate_source_contrast_mlx(
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
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    loss_and_grad_plain = nn.value_and_grad(model, weighted_loss_fn_mlx)
    loss_and_grad_plan = nn.value_and_grad(model, semantic_plan_weighted_loss_fn_mlx) if semantic_plan_enabled else None
    loss_and_grad_contrast = (
        nn.value_and_grad(model, source_contrastive_weighted_loss_fn_mlx)
        if source_contrastive_active
        else None
    )
    loss_and_grad_plan_contrast = (
        nn.value_and_grad(model, semantic_plan_source_contrastive_weighted_loss_fn_mlx)
        if source_contrastive_active and semantic_plan_enabled
        else None
    )
    loss_and_grad_negative_plain = (
        nn.value_and_grad(model, negative_replay_weighted_loss_fn_mlx)
        if negative_replay_active
        else None
    )
    loss_and_grad_negative_plan = (
        nn.value_and_grad(model, negative_replay_semantic_plan_weighted_loss_fn_mlx)
        if negative_replay_active and semantic_plan_enabled
        else None
    )
    loss_and_grad_negative_contrast = (
        nn.value_and_grad(model, negative_replay_source_contrastive_weighted_loss_fn_mlx)
        if negative_replay_active and source_contrastive_active
        else None
    )
    loss_and_grad_negative_plan_contrast = (
        nn.value_and_grad(model, negative_replay_semantic_plan_source_contrastive_weighted_loss_fn_mlx)
        if negative_replay_active and source_contrastive_active and semantic_plan_enabled
        else None
    )
    loss_and_grad_pairwise_composite = (
        nn.value_and_grad(model, pairwise_replay_composite_loss_fn_mlx)
        if pairwise_replay_active
        else None
    )
    source_matrix = mx.array(train_source_rows, dtype=mx.int32)
    target_matrix = mx.array(train_target_rows, dtype=mx.int32)
    weight_matrix = mx.array(token_weight_rows, dtype=mx.float32)
    plan_target_matrix = mx.array(train_plan_targets, dtype=mx.int32) if semantic_plan_enabled else None
    plan_weight_matrix = mx.array(train_plan_weights, dtype=mx.float32) if semantic_plan_enabled else None
    negative_source_matrix = mx.array(negative_source_rows, dtype=mx.int32) if negative_replay_active else None
    negative_target_matrix = mx.array(negative_target_rows, dtype=mx.int32) if negative_replay_active else None
    negative_weight_matrix = mx.array(negative_token_weight_rows, dtype=mx.float32) if negative_replay_active else None
    if pairwise_replay_active:
        negative_source_matrix = mx.array(negative_source_rows, dtype=mx.int32)
        negative_target_matrix = mx.array(negative_target_rows, dtype=mx.int32)
    pairwise_positive_matrix = mx.array(pairwise_positive_target_rows, dtype=mx.int32) if pairwise_replay_active else None
    pairwise_positive_weight_matrix = mx.array(pairwise_positive_weight_rows, dtype=mx.float32) if pairwise_replay_active else None
    pairwise_negative_weight_matrix = mx.array(pairwise_negative_weight_rows, dtype=mx.float32) if pairwise_replay_active else None
    mx.eval(source_matrix, target_matrix, weight_matrix)
    if semantic_plan_enabled:
        mx.eval(plan_target_matrix, plan_weight_matrix)
    if negative_replay_active:
        mx.eval(negative_source_matrix, negative_target_matrix, negative_weight_matrix)
    if pairwise_replay_active:
        mx.eval(
            negative_source_matrix,
            negative_target_matrix,
            pairwise_positive_matrix,
            pairwise_positive_weight_matrix,
            pairwise_negative_weight_matrix,
        )
    pairwise_policy_before = (
        evaluate_pairwise_policy_preference_mlx(
            model,
            reference_model,
            negative_source_rows,
            pairwise_positive_target_rows,
            negative_target_rows,
            pairwise_positive_weight_rows,
            pairwise_negative_weight_rows,
            batch_size=batch_size,
            pad_id=pad_id,
            objective=pairwise_objective,
            beta=pairwise_dpo_beta,
            prefix_tokens=pairwise_replay_prefix_tokens,
            mx=mx,
            nn=nn,
        )
        if pairwise_replay_active
        else pairwise_policy_preference_not_active(pairwise_objective)
    )
    order = list(range(len(train_source_rows)))
    losses: list[float] = []
    optimizer_steps = 0
    training_started = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_indices = mx.array(indices, dtype=mx.int32)
            src = source_matrix[batch_indices]
            tgt = target_matrix[batch_indices]
            weights = weight_matrix[batch_indices]
            if semantic_plan_enabled:
                plan_targets = plan_target_matrix[batch_indices]
                plan_weights = plan_weight_matrix[batch_indices]
            if negative_replay_active:
                neg_count = len(negative_source_rows)
                neg_indices = [
                    (optimizer_steps * batch_size + offset) % neg_count
                    for offset in range(len(indices))
                ]
                negative_batch_indices = mx.array(neg_indices, dtype=mx.int32)
                neg_src = negative_source_matrix[negative_batch_indices]
                neg_tgt = negative_target_matrix[negative_batch_indices]
                neg_weights = negative_weight_matrix[negative_batch_indices]
            if pairwise_replay_active and loss_and_grad_pairwise_composite is not None:
                if source_contrastive_active and len(indices) > 1:
                    shifted_indices = indices[1:] + indices[:1]
                    mismatched_src = source_matrix[mx.array(shifted_indices, dtype=mx.int32)]
                    effective_contrastive_weight = source_contrastive_loss_weight
                else:
                    mismatched_src = src
                    effective_contrastive_weight = 0.0
                pair_count = len(negative_source_rows)
                pair_indices = [
                    (optimizer_steps * batch_size + offset) % pair_count
                    for offset in range(len(indices))
                ]
                pair_batch_indices = mx.array(pair_indices, dtype=mx.int32)
                pair_src = negative_source_matrix[pair_batch_indices]
                pair_positive_tgt = pairwise_positive_matrix[pair_batch_indices]
                pair_negative_tgt = negative_target_matrix[pair_batch_indices]
                pair_positive_weights = pairwise_positive_weight_matrix[pair_batch_indices]
                pair_negative_weights = pairwise_negative_weight_matrix[pair_batch_indices]
                if negative_replay_active:
                    neg_src = negative_source_matrix[pair_batch_indices]
                    neg_tgt = negative_target_matrix[pair_batch_indices]
                    neg_weights = negative_weight_matrix[pair_batch_indices]
                    effective_negative_weight = negative_unlikelihood_weight
                else:
                    neg_src = pair_src
                    neg_tgt = pair_negative_tgt
                    neg_weights = pair_negative_weights
                    effective_negative_weight = 0.0
                if semantic_plan_enabled:
                    plan_targets_arg = plan_targets
                    plan_weights_arg = plan_weights
                    effective_plan_weight = semantic_plan_loss_weight
                else:
                    plan_targets_arg = tgt
                    plan_weights_arg = weights
                    effective_plan_weight = 0.0
                loss, grads = loss_and_grad_pairwise_composite(
                    model,
                    src,
                    mismatched_src,
                    tgt,
                    pad_id,
                    weights,
                    float(effective_contrastive_weight),
                    float(source_contrastive_margin),
                    int(source_contrastive_prefix_tokens),
                    source_contrastive_span_mode,
                    source_contrastive_body_start_id,
                    plan_targets_arg,
                    plan_weights_arg,
                    float(effective_plan_weight),
                    neg_src,
                    neg_tgt,
                    neg_weights,
                    float(effective_negative_weight),
                    float(negative_unlikelihood_cap),
                    pair_src,
                    pair_positive_tgt,
                    pair_negative_tgt,
                    pair_positive_weights,
                    pair_negative_weights,
                    float(pairwise_replay_loss_weight),
                    pairwise_objective,
                    reference_model,
                    float(pairwise_dpo_beta),
                    float(pairwise_ipo_target_margin),
                    float(pairwise_replay_margin),
                    int(pairwise_replay_prefix_tokens),
                    mx,
                    nn,
                    semantic_plan_class_id_array,
                )
            elif (
                negative_replay_active
                and source_contrastive_active
                and semantic_plan_enabled
                and loss_and_grad_negative_plan_contrast is not None
            ):
                shifted_indices = indices[1:] + indices[:1]
                mismatched_src = source_matrix[mx.array(shifted_indices, dtype=mx.int32)]
                effective_contrastive_weight = source_contrastive_loss_weight if len(indices) > 1 else 0.0
                loss, grads = loss_and_grad_negative_plan_contrast(
                    model,
                    src,
                    mismatched_src,
                    tgt,
                    pad_id,
                    weights,
                    float(effective_contrastive_weight),
                    float(source_contrastive_margin),
                    int(source_contrastive_prefix_tokens),
                    source_contrastive_span_mode,
                    source_contrastive_body_start_id,
                    plan_targets,
                    plan_weights,
                    float(semantic_plan_loss_weight),
                    neg_src,
                    neg_tgt,
                    neg_weights,
                    float(negative_unlikelihood_weight),
                    float(negative_unlikelihood_cap),
                    mx,
                    nn,
                    semantic_plan_class_id_array,
                )
            elif (
                negative_replay_active
                and source_contrastive_active
                and loss_and_grad_negative_contrast is not None
            ):
                shifted_indices = indices[1:] + indices[:1]
                mismatched_src = source_matrix[mx.array(shifted_indices, dtype=mx.int32)]
                effective_contrastive_weight = source_contrastive_loss_weight if len(indices) > 1 else 0.0
                loss, grads = loss_and_grad_negative_contrast(
                    model,
                    src,
                    mismatched_src,
                    tgt,
                    pad_id,
                    weights,
                    float(effective_contrastive_weight),
                    float(source_contrastive_margin),
                    int(source_contrastive_prefix_tokens),
                    source_contrastive_span_mode,
                    source_contrastive_body_start_id,
                    neg_src,
                    neg_tgt,
                    neg_weights,
                    float(negative_unlikelihood_weight),
                    float(negative_unlikelihood_cap),
                    mx,
                    nn,
                    semantic_plan_class_id_array,
                )
            elif (
                negative_replay_active
                and semantic_plan_enabled
                and loss_and_grad_negative_plan is not None
            ):
                loss, grads = loss_and_grad_negative_plan(
                    model,
                    src,
                    tgt,
                    pad_id,
                    weights,
                    plan_targets,
                    plan_weights,
                    float(semantic_plan_loss_weight),
                    neg_src,
                    neg_tgt,
                    neg_weights,
                    float(negative_unlikelihood_weight),
                    float(negative_unlikelihood_cap),
                    mx,
                    nn,
                    semantic_plan_class_id_array,
                )
            elif negative_replay_active and loss_and_grad_negative_plain is not None:
                loss, grads = loss_and_grad_negative_plain(
                    model,
                    src,
                    tgt,
                    pad_id,
                    weights,
                    neg_src,
                    neg_tgt,
                    neg_weights,
                    float(negative_unlikelihood_weight),
                    float(negative_unlikelihood_cap),
                    mx,
                    nn,
                )
            elif source_contrastive_active and semantic_plan_enabled and loss_and_grad_plan_contrast is not None:
                shifted_indices = indices[1:] + indices[:1]
                mismatched_src = source_matrix[mx.array(shifted_indices, dtype=mx.int32)]
                effective_contrastive_weight = source_contrastive_loss_weight if len(indices) > 1 else 0.0
                loss, grads = loss_and_grad_plan_contrast(
                    model,
                    src,
                    mismatched_src,
                    tgt,
                    pad_id,
                    weights,
                    float(effective_contrastive_weight),
                    float(source_contrastive_margin),
                    int(source_contrastive_prefix_tokens),
                    source_contrastive_span_mode,
                    source_contrastive_body_start_id,
                    plan_targets,
                    plan_weights,
                    float(semantic_plan_loss_weight),
                    mx,
                    nn,
                    semantic_plan_class_id_array,
                )
            elif source_contrastive_active and loss_and_grad_contrast is not None:
                shifted_indices = indices[1:] + indices[:1]
                mismatched_src = source_matrix[mx.array(shifted_indices, dtype=mx.int32)]
                effective_contrastive_weight = source_contrastive_loss_weight if len(indices) > 1 else 0.0
                loss, grads = loss_and_grad_contrast(
                    model,
                    src,
                    mismatched_src,
                    tgt,
                    pad_id,
                    weights,
                    float(effective_contrastive_weight),
                    float(source_contrastive_margin),
                    int(source_contrastive_prefix_tokens),
                    source_contrastive_span_mode,
                    source_contrastive_body_start_id,
                    mx,
                    nn,
                )
            elif semantic_plan_enabled and loss_and_grad_plan is not None:
                loss, grads = loss_and_grad_plan(
                    model,
                    src,
                    tgt,
                    pad_id,
                    weights,
                    plan_targets,
                    plan_weights,
                    float(semantic_plan_loss_weight),
                    mx,
                    nn,
                    semantic_plan_class_id_array,
                )
            else:
                loss, grads = loss_and_grad_plain(model, src, tgt, pad_id, weights, mx, nn)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            losses.append(round(float(loss.item()), 6))
            optimizer_steps += 1
            if optimizer_steps == 1 or optimizer_steps % 50 == 0:
                print(
                    "[strict-generator-mlx-adapt] "
                    f"id={adaptation_id} epoch={epoch + 1}/{epochs} "
                    f"step={optimizer_steps} loss={float(loss.item()):.6f}",
                    flush=True,
                )
    training_wall_ms = int((time.perf_counter() - training_started) * 1000)
    heldout_after = evaluate_loss_mlx(
        model,
        eval_source_rows,
        eval_target_rows,
        batch_size=batch_size,
        pad_id=pad_id,
        mx=mx,
        nn=nn,
    )
    semantic_plan_after = evaluate_semantic_plan_mlx(
        model,
        eval_source_rows,
        eval_plan_targets,
        batch_size=batch_size,
        pad_id=pad_id,
        mx=mx,
        nn=nn,
        plan_class_ids=semantic_plan_class_id_array,
    ) if semantic_plan_enabled else {"loss": None, "accuracy": None, "active_target_count": 0}
    source_contrast_after = evaluate_source_contrast_mlx(
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
    pairwise_policy_after = (
        evaluate_pairwise_policy_preference_mlx(
            model,
            reference_model,
            negative_source_rows,
            pairwise_positive_target_rows,
            negative_target_rows,
            pairwise_positive_weight_rows,
            pairwise_negative_weight_rows,
            batch_size=batch_size,
            pad_id=pad_id,
            objective=pairwise_objective,
            beta=pairwise_dpo_beta,
            prefix_tokens=pairwise_replay_prefix_tokens,
            mx=mx,
            nn=nn,
        )
        if pairwise_replay_active
        else pairwise_policy_preference_not_active(pairwise_objective)
    )
    update_summary = parameter_update_summary(model, before, mlx_utils, mx)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out = checkpoint_dir / f"strict_generator_mlx_{safe_slug(adaptation_id)}.npz"
    vocab_out = checkpoint_dir / f"strict_generator_mlx_{safe_slug(adaptation_id)}_vocab.json"
    model.save_weights(str(checkpoint_out))
    vocab_payload = dict(vocab_payload)
    vocab_payload.update(
        {
            "policy": "project_theseus_strict_generator_mlx_vocab_v1",
            "created_utc": now(),
            "adaptation_id": adaptation_id,
            "source_checkpoint": rel(checkpoint_path),
            "source_vocab": source_vocab,
            "target_vocab": target_vocab,
            "source_vocab_sha256": stable_hash(json.dumps(source_vocab, sort_keys=True)),
            "target_vocab_sha256": stable_hash(json.dumps(target_vocab, sort_keys=True)),
            "source_text_style": source_text_style,
            "target_mode": target_mode,
            "semantic_plan_visible_operation_weighting": visible_operation_plan_weighting,
            "semantic_construction_repair_profile": semantic_construction_repair_profile,
            "family_disjoint_holdout_exclusion": holdout_exclusion,
            "supplemental_private_train_audit": supplemental_train_audit,
            "private_train_tier_selection": private_train_tier_vocab_summary(tier_selection),
            "return_expression_weighting": return_expression_weighting,
            "default_parameter_return_weighting": default_parameter_return_weighting,
            "truthiness_guard_weighting": truthiness_guard_weighting,
            "source_condition_internalization_weighting": source_condition_internalization_weighting,
            "loop_operation_weighting": loop_operation_weighting,
            "loop_statement_action_weighting": loop_statement_action_weighting,
            "loop_semantic_operation_weighting": loop_semantic_operation_weighting,
            "semantic_slot_prefix_weighting": semantic_slot_prefix_weighting,
            "loop_expression_synthesis_weighting": loop_expression_synthesis_weighting,
            "plan_conditioned_body_weighting": plan_conditioned_body_weighting,
            "update_contract_consistency_weighting": update_contract_consistency_weighting,
            "direct_body_emission_weighting": direct_body_emission_weighting,
            "local_return_closure_weighting": local_return_closure_weighting,
            "negative_replay": negative_replay_vocab_summary(negative_replay, active=negative_replay_active),
            "pairwise_replay_preference": pairwise_replay_vocab_summary(
                negative_replay,
                active=pairwise_replay_active,
                weight=pairwise_replay_loss_weight,
                objective=pairwise_objective,
                beta=pairwise_dpo_beta,
                reference_checkpoint_used=reference_model is not None,
            ),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "public_training_rows": 0,
            "external_inference_calls": 0,
        }
    )
    write_json(vocab_out, vocab_payload)
    mx.eval(model.parameters())
    source_nonpad = sum(1 for row in train_source_rows for value in row if int(value) != 0)
    target_nonpad = sum(1 for row in train_target_rows for value in row if int(value) != pad_id)
    optimizer_token_positions = int(source_nonpad + target_nonpad) * epochs
    seconds = max(training_wall_ms / 1000.0, 1e-9)
    payload = {
        "active": True,
        "adaptation_id": adaptation_id,
        "backend": "mlx_high_level_transformer_private_adaptation",
        "device": str(mx.default_device()),
        "source_checkpoint": rel(checkpoint_path),
        "source_checkpoint_report": checkpoint_report_path,
        "checkpoint": rel(checkpoint_out),
        "checkpoint_sha256": stable_hash_file(checkpoint_out),
        "vocab": rel(vocab_out),
        "vocab_sha256": stable_hash_file(vocab_out),
        "train_rows": len(train_rows),
        "rehearsal_rows": len(rehearsal_examples),
        "heldout_private_train_rows": len(eval_rows),
        "source_text_style": source_text_style,
        "target_mode": target_mode,
        "source_text_audit": source_text_audit,
        "train_split_policy": (
            "private_residual_repair_row_holdout_v1"
            if private_residual_repair_split
            else "configured_family_disjoint_holdout_exclusion_v1"
        ),
        "family_disjoint_evidence": not private_residual_repair_split,
        "public_calibration_eligible": False,
        "family_disjoint_holdout_exclusion": holdout_exclusion,
        "supplemental_private_train_audit": supplemental_train_audit,
        "private_train_tier_selection": private_train_tier_summary(tier_selection),
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss_weighting": loss_weight_summary,
        "runtime_overrides": {
            "primary_dataflow_weights_enabled": bool(enable_primary_dataflow_weights),
            "primary_dataflow_weight_scale": float(primary_dataflow_weight_scale),
            "return_expression_loss_boost": float(return_expression_loss_boost or 0.0),
            "default_parameter_return_loss_boost": float(default_parameter_return_loss_boost or 0.0),
            "truthiness_guard_loss_boost": float(truthiness_guard_loss_boost or 0.0),
            "source_condition_internalization_loss_boost": float(source_condition_internalization_loss_boost or 0.0),
            "source_condition_operation_coverage_min_rows": int(source_condition_operation_coverage_min_rows or 0),
            "loop_operation_loss_boost": float(loop_operation_loss_boost or 0.0),
            "loop_statement_action_loss_boost": float(loop_statement_action_loss_boost or 0.0),
            "loop_statement_action_roles": str(loop_statement_action_roles or ""),
            "loop_semantic_operation_loss_boost": float(loop_semantic_operation_loss_boost or 0.0),
            "loop_semantic_operation_roles": str(loop_semantic_operation_roles or ""),
            "semantic_slot_prefix_loss_boost": float(semantic_slot_prefix_loss_boost or 0.0),
            "semantic_slot_prefix_roles": str(semantic_slot_prefix_roles or ""),
            "loop_expression_synthesis_loss_boost": float(loop_expression_synthesis_loss_boost or 0.0),
            "loop_expression_synthesis_roles": str(loop_expression_synthesis_roles or ""),
            "plan_conditioned_body_loss_boost": float(plan_conditioned_body_loss_boost or 0.0),
            "plan_conditioned_body_roles": str(plan_conditioned_body_roles or ""),
            "update_contract_consistency_loss_boost": float(update_contract_consistency_loss_boost or 0.0),
            "direct_body_emission_loss_boost": float(direct_body_emission_loss_boost or 0.0),
            "direct_body_emission_roles": str(direct_body_emission_roles or ""),
            "local_return_closure_loss_boost": float(local_return_closure_loss_boost or 0.0),
            "local_return_closure_roles": str(local_return_closure_roles or ""),
            "semantic_construction_repair_profile": semantic_construction_repair_profile,
            "semantic_plan_visible_operation_loss_boost": float(semantic_plan_visible_operation_loss_boost or 0.0),
            "private_residual_repair_split": bool(private_residual_repair_split),
            "supplemental_private_train_jsonl": [
                rel(path) for path in effective_supplemental_train_jsonl
            ],
            "negative_starvation_report": rel(negative_starvation_report) if negative_starvation_report is not None else "",
            "max_negative_starvation_rows": int(max_negative_starvation_rows or 0),
            "pairwise_replay_loss_weight": float(pairwise_replay_loss_weight or 0.0),
            "pairwise_replay_objective": pairwise_objective,
            "pairwise_dpo_beta": float(pairwise_dpo_beta or 0.0),
            "pairwise_ipo_target_margin": float(pairwise_ipo_target_margin or 0.0),
            "pairwise_replay_margin": float(pairwise_replay_margin or 0.0),
            "pairwise_replay_prefix_tokens": int(pairwise_replay_prefix_tokens or 0),
            "action_trace_replay_loss_boost": float(action_trace_replay_loss_boost or 0.0),
        },
        "return_expression_weighting": return_expression_weighting,
        "default_parameter_return_weighting": default_parameter_return_weighting,
        "truthiness_guard_weighting": truthiness_guard_weighting,
        "source_condition_internalization_weighting": source_condition_internalization_weighting,
        "loop_operation_weighting": loop_operation_weighting,
        "loop_statement_action_weighting": loop_statement_action_weighting,
        "loop_semantic_operation_weighting": loop_semantic_operation_weighting,
        "semantic_slot_prefix_weighting": semantic_slot_prefix_weighting,
        "loop_expression_synthesis_weighting": loop_expression_synthesis_weighting,
        "plan_conditioned_body_weighting": plan_conditioned_body_weighting,
        "update_contract_consistency_weighting": update_contract_consistency_weighting,
        "direct_body_emission_weighting": direct_body_emission_weighting,
        "local_return_closure_weighting": local_return_closure_weighting,
        "semantic_construction_repair_profile": semantic_construction_repair_profile,
        "strict_target_guard": {
            "train": target_guard_summary,
            "heldout_private_train": eval_target_guard_summary,
            "loss_weighting": guard_weight_summary,
        },
        "negative_replay_unlikelihood": negative_replay_summary(
            negative_replay,
            active=negative_replay_active,
            weight=negative_unlikelihood_weight,
            cap=negative_unlikelihood_cap,
            stage_weighting=negative_stage_weight_summary,
        ),
        "pairwise_replay_preference": pairwise_replay_summary(
            negative_replay,
            active=pairwise_replay_active,
            weight=pairwise_replay_loss_weight,
            objective=pairwise_objective,
            beta=pairwise_dpo_beta,
            ipo_target_margin=pairwise_ipo_target_margin,
            margin=pairwise_replay_margin,
            prefix_tokens=pairwise_replay_prefix_tokens,
            token_weighting=pairwise_replay_weight_summary,
            policy_before=pairwise_policy_before,
            policy_after=pairwise_policy_after,
            reference_checkpoint=rel(checkpoint_path) if reference_model is not None else "",
        ),
        "rehearsal": rehearsal_summary(rehearsal, selected_count=len(rehearsal_examples)),
        "source_contrastive_loss": {
            "enabled": source_contrastive_active,
            "policy": "private_adaptation_source_mismatch_margin_loss_v1",
            "weight": float(source_contrastive_loss_weight or 0.0),
            "margin": float(source_contrastive_margin or 0.0),
            "prefix_token_count": int(source_contrastive_prefix_tokens or 0),
            "span_mode": source_contrastive_span_mode,
            "body_start_token_id": source_contrastive_body_start_id,
            "heldout_matched_loss_before": source_contrast_before.get("matched_loss"),
            "heldout_mismatched_loss_before": source_contrast_before.get("mismatched_loss"),
            "heldout_source_loss_gap_before": source_contrast_before.get("loss_gap"),
            "heldout_matched_loss_after": source_contrast_after.get("matched_loss"),
            "heldout_mismatched_loss_after": source_contrast_after.get("mismatched_loss"),
            "heldout_source_loss_gap_after": source_contrast_after.get("loss_gap"),
            "loss_gap_improved": (
                source_contrast_before.get("loss_gap") is not None
                and source_contrast_after.get("loss_gap") is not None
                and float(source_contrast_after["loss_gap"]) > float(source_contrast_before["loss_gap"])
            ),
            "score_semantics": (
                "Uses only admitted private task rows and admitted rehearsal rows. It compares "
                "correct-source body loss against deterministic in-batch mismatched-source body loss "
                "to make private adaptation depend more on visible prompt/signature text. It does not "
                "inspect eval tests, eval solutions, verifier results, public benchmarks, answer "
                "metadata, or teacher output."
            ),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        },
        "semantic_plan_visible_operation_weighting": visible_operation_plan_weighting,
        "semantic_plan_label_space": {
            "policy": "semantic_plan_token_subspace_v1",
            "enabled": bool(semantic_plan_enabled and semantic_plan_class_ids),
            "plan_class_count": len(semantic_plan_class_ids),
            "score_semantics": (
                "Semantic-plan auxiliary loss and eval are restricted to SLOT:PLAN_* target-vocab IDs, "
                "matching the decode-time semantic-plan-prefix choice space. This changes only learned "
                "plan-head supervision; it does not render code, inspect tests/solutions, use public data, "
                "call tools, or grant candidate-generation credit."
            ),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        },
        "semantic_plan_auxiliary": {
            "enabled": semantic_plan_enabled,
            "policy": "private_task_contract_plan_auxiliary_adaptation_v1" if semantic_plan_enabled else "not_enabled",
            "weight": float(semantic_plan_loss_weight or 0.0),
            "train_targets": train_plan_summary,
            "eval_targets": eval_plan_summary,
            "class_balance": plan_balance_summary,
            "heldout_plan_loss_before": semantic_plan_before.get("loss"),
            "heldout_plan_loss_after": semantic_plan_after.get("loss"),
            "heldout_plan_accuracy_before": semantic_plan_before.get("accuracy"),
            "heldout_plan_accuracy_after": semantic_plan_after.get("accuracy"),
            "heldout_plan_improved": (
                semantic_plan_enabled
                and semantic_plan_before.get("loss") is not None
                and semantic_plan_after.get("loss") is not None
                and float(semantic_plan_after["loss"]) < float(semantic_plan_before["loss"])
            ),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "uses_answer_metadata": False,
            "served_at_runtime": False,
            "candidate_generation_credit": 0,
        },
        "parameter_count": update_summary["parameter_count"],
        "parameter_update_fraction": update_summary["parameter_update_fraction"],
        "parameter_tensor_update_fraction": update_summary["parameter_tensor_update_fraction"],
        "optimizer_step_count": optimizer_steps,
        "optimizer_token_positions_consumed": optimizer_token_positions,
        "training_wall_time_ms": training_wall_ms,
        "training_tokens_per_second": round(optimizer_token_positions / seconds, 3),
        "optimizer_steps_per_second": round(optimizer_steps / seconds, 6),
        "heldout_lm_loss_before": heldout_before.get("loss"),
        "heldout_lm_loss_after": heldout_after.get("loss"),
        "heldout_lm_loss_curve": [heldout_before.get("loss"), *losses, heldout_after.get("loss")],
        "heldout_lm_improved": bool(
            heldout_before.get("loss") is not None
            and heldout_after.get("loss") is not None
            and float(heldout_after["loss"]) < float(heldout_before["loss"])
        ),
        "training_batch_materialization": "preloaded_mx_arrays_indexed_by_shuffled_batch_v1",
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "open_or_pretrained_model_weights_used": False,
        "fallback_template_router_tool_credit_count": 0,
    }
    gates = build_gates(payload)
    hard_pass = all(row["passed"] for row in gates if row["severity"] == "hard")
    trigger_state = "GREEN" if hard_pass else "RED"
    if trigger_state == "GREEN" and any(not row["passed"] for row in gates):
        trigger_state = "YELLOW"
    return {
        "policy": "project_theseus_strict_generator_mlx_private_adaptation_v1",
        "created_utc": now(),
        "config": config_path,
        "execute": True,
        "trigger_state": trigger_state,
        "summary": {
            "adaptation_id": adaptation_id,
            "device": payload["device"],
            "checkpoint": payload["checkpoint"],
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "vocab": payload["vocab"],
            "vocab_sha256": payload["vocab_sha256"],
            "train_rows": payload["train_rows"],
            "rehearsal_rows": payload["rehearsal_rows"],
            "heldout_private_train_rows": payload["heldout_private_train_rows"],
            "training_tokens_per_second": payload["training_tokens_per_second"],
            "optimizer_steps_per_second": payload["optimizer_steps_per_second"],
            "optimizer_step_count": payload["optimizer_step_count"],
            "optimizer_token_positions_consumed": payload["optimizer_token_positions_consumed"],
            "heldout_lm_loss_before": payload["heldout_lm_loss_before"],
            "heldout_lm_loss_after": payload["heldout_lm_loss_after"],
            "heldout_lm_improved": payload["heldout_lm_improved"],
            "parameter_update_fraction": payload["parameter_update_fraction"],
            "parameter_tensor_update_fraction": payload["parameter_tensor_update_fraction"],
            "source_text_style": payload["source_text_style"],
            "target_mode": payload["target_mode"],
            "source_text_audit": payload["source_text_audit"],
            "train_split_policy": payload["train_split_policy"],
            "family_disjoint_evidence": payload["family_disjoint_evidence"],
            "public_calibration_eligible": payload["public_calibration_eligible"],
            "rehearsal": payload["rehearsal"],
            "source_contrastive_loss": payload["source_contrastive_loss"],
            "semantic_plan_auxiliary": payload["semantic_plan_auxiliary"],
            "semantic_plan_visible_operation_weighting": payload["semantic_plan_visible_operation_weighting"],
            "semantic_plan_label_space": payload["semantic_plan_label_space"],
            "strict_target_guard": payload["strict_target_guard"],
            "negative_replay_unlikelihood": payload["negative_replay_unlikelihood"],
            "pairwise_replay_preference": payload["pairwise_replay_preference"],
            "family_disjoint_holdout_exclusion": payload["family_disjoint_holdout_exclusion"],
            "private_train_tier_selection": payload["private_train_tier_selection"],
            "return_expression_weighting": payload["return_expression_weighting"],
            "default_parameter_return_weighting": payload["default_parameter_return_weighting"],
            "truthiness_guard_weighting": payload["truthiness_guard_weighting"],
            "source_condition_internalization_weighting": payload["source_condition_internalization_weighting"],
            "loop_operation_weighting": payload["loop_operation_weighting"],
            "loop_statement_action_weighting": payload["loop_statement_action_weighting"],
            "loop_semantic_operation_weighting": payload["loop_semantic_operation_weighting"],
            "semantic_slot_prefix_weighting": payload["semantic_slot_prefix_weighting"],
            "loop_expression_synthesis_weighting": payload["loop_expression_synthesis_weighting"],
            "plan_conditioned_body_weighting": payload["plan_conditioned_body_weighting"],
            "update_contract_consistency_weighting": payload["update_contract_consistency_weighting"],
            "direct_body_emission_weighting": payload["direct_body_emission_weighting"],
            "local_return_closure_weighting": payload["local_return_closure_weighting"],
            "semantic_construction_repair_profile": payload["semantic_construction_repair_profile"],
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "open_or_pretrained_model_weights_used": False,
            "fallback_template_router_tool_credit_count": 0,
        },
        "adaptation": payload,
        "gates": gates,
        "score_semantics": (
            "MLX private adaptation of an admitted strict-generator checkpoint. Training rows come only from "
            "the configured private train JSONL; eval tests/solutions and public benchmark payloads are not "
            "visible to generation or training. This emits no candidate and makes no promotion claim by itself."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def apply_visible_operation_plan_sample_weights(
    plan_weights: list[float],
    plan_targets: list[int],
    bodies: list[str],
    source_texts: list[str],
    *,
    target_vocab: dict[str, int],
    pad_id: int,
    boost: float,
) -> tuple[list[float], dict[str, Any]]:
    value = max(0.0, float(boost if boost is not None else 0.0))
    if value <= 0.0:
        return list(plan_weights), {
            "enabled": False,
            "policy": "not_enabled",
            "semantic_plan_visible_operation_loss_boost": value,
            "boosted_rows": 0,
            "candidate_generation_credit": 0,
        }
    inverse = {int(token_id): str(token) for token, token_id in target_vocab.items()}
    adjusted = [float(item) for item in plan_weights]
    boosted_rows = 0
    visible_operation_hint_rows = 0
    active_target_rows = 0
    non_generic_plan_rows = 0
    boosted_plan_counts: dict[str, int] = {}
    skipped: dict[str, int] = {}
    for index, target_id in enumerate(plan_targets):
        if index >= len(adjusted):
            break
        if int(target_id) == int(pad_id):
            skipped["pad_plan_target"] = skipped.get("pad_plan_target", 0) + 1
            continue
        active_target_rows += 1
        source_text = source_texts[index] if index < len(source_texts) else ""
        body = bodies[index] if index < len(bodies) else ""
        plan_token = inverse.get(int(target_id), "") or f"SLOT:PLAN_{semantic_plan_from_body(body)}"
        if not plan_token.startswith("SLOT:PLAN_"):
            skipped["non_plan_target"] = skipped.get("non_plan_target", 0) + 1
            continue
        plan_name = plan_token.removeprefix("SLOT:PLAN_")
        if plan_name not in {"GENERIC_BODY", "AST_INVALID"}:
            non_generic_plan_rows += 1
        if "\nprompt_operation_hints " not in f"\n{source_text}":
            skipped["missing_visible_operation_hints"] = skipped.get("missing_visible_operation_hints", 0) + 1
            continue
        visible_operation_hint_rows += 1
        if plan_name in {"GENERIC_BODY", "AST_INVALID"}:
            skipped[f"generic_or_invalid_plan_{plan_name}"] = skipped.get(f"generic_or_invalid_plan_{plan_name}", 0) + 1
            continue
        adjusted[index] = max(float(adjusted[index]), float(adjusted[index]) * value)
        boosted_rows += 1
        boosted_plan_counts[plan_token] = boosted_plan_counts.get(plan_token, 0) + 1
    weights_after = [float(adjusted[index]) for index, target_id in enumerate(plan_targets[: len(adjusted)]) if int(target_id) != int(pad_id)]
    return adjusted, {
        "enabled": True,
        "policy": "private_visible_operation_plan_sample_weighting_v1",
        "semantic_plan_visible_operation_loss_boost": value,
        "rows": len(plan_targets),
        "active_target_rows": active_target_rows,
        "visible_operation_hint_rows": visible_operation_hint_rows,
        "non_generic_plan_rows": non_generic_plan_rows,
        "boosted_rows": boosted_rows,
        "boosted_plan_counts": dict(sorted(boosted_plan_counts.items(), key=lambda item: (-item[1], item[0]))[:32]),
        "skipped_counts": dict(sorted(skipped.items())),
        "min_weight_after": round(min(weights_after), 6) if weights_after else None,
        "max_weight_after": round(max(weights_after), 6) if weights_after else None,
        "mean_weight_after": round(sum(weights_after) / max(1, len(weights_after)), 6) if weights_after else None,
        "score_semantics": (
            "Multiplies semantic-plan auxiliary sample weight only for admitted private train rows whose "
            "strict source text contains prompt-visible operation hints and whose private target AST has a "
            "non-generic semantic plan label. It trains source-to-plan selection pressure; it does not "
            "create labels, inspect eval tests/solutions, use public data, call a teacher, render code, "
            "route tools, or grant learned-generation candidate credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def semantic_construction_profile_missing_components(payload: dict[str, Any]) -> list[str]:
    profile = dict_or_empty(payload.get("semantic_construction_repair_profile"))
    if not bool(profile.get("enabled")):
        return []
    required = {str(item) for item in list(profile.get("required_components") or []) if str(item)}
    missing: list[str] = []

    def component(name: str) -> dict[str, Any]:
        return dict_or_empty(payload.get(name))

    if "semantic_plan_auxiliary" in required and not bool(component("semantic_plan_auxiliary").get("enabled")):
        missing.append("semantic_plan_auxiliary")
    if "semantic_plan_visible_operation_weighting" in required:
        item = component("semantic_plan_visible_operation_weighting")
        if not bool(item.get("enabled")) or int(item.get("boosted_rows") or 0) <= 0:
            missing.append("semantic_plan_visible_operation_weighting")
    for name in [
        "semantic_slot_prefix_weighting",
        "loop_semantic_operation_weighting",
        "loop_expression_synthesis_weighting",
        "plan_conditioned_body_weighting",
        "update_contract_consistency_weighting",
        "direct_body_emission_weighting",
        "local_return_closure_weighting",
    ]:
        if name not in required:
            continue
        item = component(name)
        if name == "semantic_slot_prefix_weighting" and semantic_slot_prefix_unavailable_for_target_mode(payload):
            continue
        if not bool(item.get("enabled")) or int(item.get("weighted_token_positions") or 0) <= 0:
            missing.append(name)
    if "source_contrastive_loss" in required:
        item = component("source_contrastive_loss")
        if not bool(item.get("enabled")) or int(item.get("prefix_token_count") or 0) <= 0:
            missing.append("source_contrastive_loss")
    return sorted(set(missing))


def semantic_slot_prefix_unavailable_for_target_mode(payload: dict[str, Any]) -> bool:
    item = dict_or_empty(payload.get("semantic_slot_prefix_weighting"))
    skipped = dict_or_empty(item.get("skipped_counts"))
    rows = int(item.get("rows") or 0)
    return (
        str(payload.get("target_mode") or "") == "body_tokens"
        and bool(item.get("enabled"))
        and rows > 0
        and int(item.get("weighted_token_positions") or 0) == 0
        and int(skipped.get("missing_body_start") or 0) >= rows
    )


def build_gates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    missing_semantic_construction_components = semantic_construction_profile_missing_components(payload)
    semantic_construction_profile = dict_or_empty(payload.get("semantic_construction_repair_profile"))
    semantic_construction_profile_enabled = bool(semantic_construction_profile.get("enabled"))
    checkpoint_path = resolve(str(payload.get("checkpoint") or ""))
    vocab_path = resolve(str(payload.get("vocab") or ""))
    return [
        gate("private_training_rows_present", int(payload.get("train_rows") or 0) > 0, "hard", payload.get("train_rows")),
        gate("heldout_private_train_rows_present", int(payload.get("heldout_private_train_rows") or 0) > 0, "hard", payload.get("heldout_private_train_rows")),
        gate(
            "checkpoint_artifacts_written_with_digests",
            checkpoint_path.exists()
            and vocab_path.exists()
            and is_sha256(payload.get("checkpoint_sha256"))
            and is_sha256(payload.get("vocab_sha256")),
            "hard",
            {
                "checkpoint": payload.get("checkpoint"),
                "checkpoint_sha256": payload.get("checkpoint_sha256"),
                "vocab": payload.get("vocab"),
                "vocab_sha256": payload.get("vocab_sha256"),
            },
        ),
        gate(
            "family_disjoint_holdout_excluded",
            bool(dict_or_empty(payload.get("family_disjoint_holdout_exclusion")).get("clean", True)),
            "hard",
            payload.get("family_disjoint_holdout_exclusion"),
        ),
        gate(
            "private_residual_repair_not_family_disjoint_claim_when_enabled",
            (
                str(payload.get("train_split_policy") or "") != "private_residual_repair_row_holdout_v1"
                or (
                    not bool(payload.get("family_disjoint_evidence"))
                    and not bool(payload.get("public_calibration_eligible"))
                )
            ),
            "soft",
            {
                "train_split_policy": payload.get("train_split_policy"),
                "family_disjoint_evidence": payload.get("family_disjoint_evidence"),
                "public_calibration_eligible": payload.get("public_calibration_eligible"),
            },
        ),
        gate("public_training_rows_zero", int(payload.get("public_training_rows") or 0) == 0, "hard", payload.get("public_training_rows")),
        gate("external_inference_zero", int(payload.get("external_inference_calls") or 0) == 0, "hard", payload.get("external_inference_calls")),
        gate("no_open_or_pretrained_weights", not bool(payload.get("open_or_pretrained_model_weights_used")), "hard", "from_scratch_project_checkpoint_only"),
        gate("no_template_router_tool_credit", int(payload.get("fallback_template_router_tool_credit_count") or 0) == 0, "hard", payload.get("fallback_template_router_tool_credit_count")),
        gate(
            "strict_source_text_audit_clean",
            all(bool(dict_or_empty(row).get("clean")) for row in dict_or_empty(payload.get("source_text_audit")).values()),
            "hard",
            payload.get("source_text_audit"),
        ),
        gate(
            "parameter_tensor_update_recorded",
            float(payload.get("parameter_tensor_update_fraction") or 0.0) >= 0.90,
            "hard",
            {
                "actual": payload.get("parameter_tensor_update_fraction"),
                "minimum": 0.90,
                "reason": (
                    "Policy updates can legitimately skip inactive auxiliary tensors; "
                    "element-wide update size remains a separate soft gate."
                ),
            },
        ),
        gate("parameter_element_update_meaningful", float(payload.get("parameter_update_fraction") or 0.0) >= 0.25, "soft", payload.get("parameter_update_fraction")),
        gate("heldout_lm_improved", bool(payload.get("heldout_lm_improved")), "hard", [payload.get("heldout_lm_loss_before"), payload.get("heldout_lm_loss_after")]),
        gate(
            "semantic_plan_loss_improved_when_enabled",
            (
                not bool(dict_or_empty(payload.get("semantic_plan_auxiliary")).get("enabled"))
                or bool(dict_or_empty(payload.get("semantic_plan_auxiliary")).get("heldout_plan_improved"))
            ),
            "soft",
            dict_or_empty(payload.get("semantic_plan_auxiliary")),
        ),
        gate(
            "semantic_plan_visible_operation_weighting_matched_when_enabled",
            (
                not bool(dict_or_empty(payload.get("semantic_plan_visible_operation_weighting")).get("enabled"))
                or int(dict_or_empty(payload.get("semantic_plan_visible_operation_weighting")).get("boosted_rows") or 0) > 0
            ),
            "soft",
            dict_or_empty(payload.get("semantic_plan_visible_operation_weighting")),
        ),
        gate(
            "semantic_plan_label_space_active_when_enabled",
            (
                not bool(dict_or_empty(payload.get("semantic_plan_auxiliary")).get("enabled"))
                or (
                    bool(dict_or_empty(payload.get("semantic_plan_label_space")).get("enabled"))
                    and int(dict_or_empty(payload.get("semantic_plan_label_space")).get("plan_class_count") or 0) > 0
                )
            ),
            "hard",
            dict_or_empty(payload.get("semantic_plan_label_space")),
        ),
        gate(
            "source_contrastive_gap_improved_when_enabled",
            (
                not bool(dict_or_empty(payload.get("source_contrastive_loss")).get("enabled"))
                or bool(dict_or_empty(payload.get("source_contrastive_loss")).get("loss_gap_improved"))
            ),
            "soft",
            dict_or_empty(payload.get("source_contrastive_loss")),
        ),
        gate("throughput_recorded", float(payload.get("training_tokens_per_second") or 0.0) > 0.0, "hard", payload.get("training_tokens_per_second")),
        gate(
            "strict_target_guard_reported",
            bool(dict_or_empty(payload.get("strict_target_guard")).get("train")),
            "soft",
            dict_or_empty(payload.get("strict_target_guard")).get("train"),
        ),
        gate(
            "private_train_tier_rows_present_when_enabled",
            (
                not bool(dict_or_empty(payload.get("private_train_tier_selection")).get("enabled"))
                or int(dict_or_empty(payload.get("private_train_tier_selection")).get("selected_rows") or 0) > 0
            ),
            "hard",
            dict_or_empty(payload.get("private_train_tier_selection")),
        ),
        gate(
            "return_expression_weighting_reported_when_enabled",
            (
                not bool(dict_or_empty(payload.get("return_expression_weighting")).get("enabled"))
                or str(dict_or_empty(payload.get("return_expression_weighting")).get("policy") or "")
                == "private_return_expression_loss_weight_override_v1"
            ),
            "soft",
            dict_or_empty(payload.get("return_expression_weighting")),
        ),
        gate(
            "default_parameter_return_weighting_matched_when_enabled",
            (
                not bool(dict_or_empty(payload.get("default_parameter_return_weighting")).get("enabled"))
                or int(dict_or_empty(payload.get("default_parameter_return_weighting")).get("weighted_token_positions") or 0) > 0
            ),
            "soft",
            dict_or_empty(payload.get("default_parameter_return_weighting")),
        ),
        gate(
            "truthiness_guard_weighting_matched_when_enabled",
            (
                not bool(dict_or_empty(payload.get("truthiness_guard_weighting")).get("enabled"))
                or int(dict_or_empty(payload.get("truthiness_guard_weighting")).get("weighted_token_positions") or 0) > 0
            ),
            "soft",
            dict_or_empty(payload.get("truthiness_guard_weighting")),
        ),
        gate(
            "source_condition_internalization_weighting_matched_when_enabled",
            (
                not bool(dict_or_empty(payload.get("source_condition_internalization_weighting")).get("enabled"))
                or int(dict_or_empty(payload.get("source_condition_internalization_weighting")).get("weighted_token_positions") or 0) > 0
            ),
            "soft",
            dict_or_empty(payload.get("source_condition_internalization_weighting")),
        ),
        gate(
            "loop_operation_weighting_matched_when_enabled",
            (
                not bool(dict_or_empty(payload.get("loop_operation_weighting")).get("enabled"))
                or int(dict_or_empty(payload.get("loop_operation_weighting")).get("weighted_token_positions") or 0) > 0
            ),
            "soft",
            dict_or_empty(payload.get("loop_operation_weighting")),
        ),
        gate(
            "loop_statement_action_weighting_matched_when_enabled",
            (
                not bool(dict_or_empty(payload.get("loop_statement_action_weighting")).get("enabled"))
                or int(dict_or_empty(payload.get("loop_statement_action_weighting")).get("weighted_token_positions") or 0) > 0
            ),
            "soft",
            dict_or_empty(payload.get("loop_statement_action_weighting")),
        ),
        gate(
            "loop_semantic_operation_weighting_matched_when_enabled",
            (
                not bool(dict_or_empty(payload.get("loop_semantic_operation_weighting")).get("enabled"))
                or int(dict_or_empty(payload.get("loop_semantic_operation_weighting")).get("weighted_token_positions") or 0) > 0
            ),
            "soft",
            dict_or_empty(payload.get("loop_semantic_operation_weighting")),
        ),
        gate(
            "semantic_slot_prefix_weighting_matched_when_enabled",
            (
                not bool(dict_or_empty(payload.get("semantic_slot_prefix_weighting")).get("enabled"))
                or semantic_slot_prefix_unavailable_for_target_mode(payload)
                or int(dict_or_empty(payload.get("semantic_slot_prefix_weighting")).get("weighted_token_positions") or 0) > 0
            ),
            "soft",
            {
                **dict_or_empty(payload.get("semantic_slot_prefix_weighting")),
                "target_mode": payload.get("target_mode"),
                "abi_unavailable_for_target_mode": semantic_slot_prefix_unavailable_for_target_mode(payload),
            },
        ),
        gate(
            "loop_expression_synthesis_weighting_matched_when_enabled",
            (
                not bool(dict_or_empty(payload.get("loop_expression_synthesis_weighting")).get("enabled"))
                or int(dict_or_empty(payload.get("loop_expression_synthesis_weighting")).get("weighted_token_positions") or 0) > 0
            ),
            "soft",
            dict_or_empty(payload.get("loop_expression_synthesis_weighting")),
        ),
        gate(
            "plan_conditioned_body_weighting_matched_when_enabled",
            (
                not bool(dict_or_empty(payload.get("plan_conditioned_body_weighting")).get("enabled"))
                or int(dict_or_empty(payload.get("plan_conditioned_body_weighting")).get("weighted_token_positions") or 0) > 0
            ),
            "soft",
            dict_or_empty(payload.get("plan_conditioned_body_weighting")),
        ),
        gate(
            "update_contract_consistency_weighting_matched_when_enabled",
            (
                not bool(dict_or_empty(payload.get("update_contract_consistency_weighting")).get("enabled"))
                or int(dict_or_empty(payload.get("update_contract_consistency_weighting")).get("weighted_token_positions") or 0) > 0
            ),
            "soft",
            dict_or_empty(payload.get("update_contract_consistency_weighting")),
        ),
        gate(
            "direct_body_emission_weighting_matched_when_enabled",
            (
                not bool(dict_or_empty(payload.get("direct_body_emission_weighting")).get("enabled"))
                or int(dict_or_empty(payload.get("direct_body_emission_weighting")).get("weighted_token_positions") or 0) > 0
            ),
            "soft",
            dict_or_empty(payload.get("direct_body_emission_weighting")),
        ),
        gate(
            "local_return_closure_weighting_matched_when_enabled",
            (
                not bool(dict_or_empty(payload.get("local_return_closure_weighting")).get("enabled"))
                or int(dict_or_empty(payload.get("local_return_closure_weighting")).get("weighted_token_positions") or 0) > 0
            ),
            "soft",
            dict_or_empty(payload.get("local_return_closure_weighting")),
        ),
        gate(
            "semantic_construction_profile_private_only",
            (
                not semantic_construction_profile_enabled
                or (
                    not bool(semantic_construction_profile.get("uses_eval_tests_or_solutions"))
                    and not bool(semantic_construction_profile.get("uses_public_data"))
                    and int(semantic_construction_profile.get("external_inference_calls") or 0) == 0
                    and int(semantic_construction_profile.get("candidate_generation_credit") or 0) == 0
                )
            ),
            "hard",
            semantic_construction_profile,
        ),
        gate(
            "semantic_construction_profile_public_calibration_ineligible",
            (
                not semantic_construction_profile_enabled
                or (
                    not bool(payload.get("public_calibration_eligible"))
                    and bool(payload.get("train_split_policy") == "private_residual_repair_row_holdout_v1")
                )
            ),
            "hard",
            {
                "profile": semantic_construction_profile,
                "public_calibration_eligible": payload.get("public_calibration_eligible"),
                "train_split_policy": payload.get("train_split_policy"),
            },
        ),
        gate(
            "semantic_construction_profile_required_components_active",
            not semantic_construction_profile_enabled or not missing_semantic_construction_components,
            "hard",
            {
                "missing_components": missing_semantic_construction_components,
                "required_components": semantic_construction_profile.get("required_components"),
            },
        ),
        gate(
            "negative_replay_private_train_only_when_enabled",
            (
                not bool(dict_or_empty(payload.get("negative_replay_unlikelihood")).get("enabled"))
                or bool(dict_or_empty(payload.get("negative_replay_unlikelihood")).get("private_train_replay_only"))
            ),
            "hard",
            dict_or_empty(payload.get("negative_replay_unlikelihood")),
        ),
        gate(
            "negative_replay_all_examples_failed_when_enabled",
            (
                not bool(dict_or_empty(payload.get("negative_replay_unlikelihood")).get("enabled"))
                or bool(dict_or_empty(payload.get("negative_replay_unlikelihood")).get("all_selected_examples_failed_intended_behavior"))
            ),
            "hard",
            dict_or_empty(payload.get("negative_replay_unlikelihood")),
        ),
        gate(
            "negative_replay_no_public_or_eval_visibility_when_enabled",
            (
                not bool(dict_or_empty(payload.get("negative_replay_unlikelihood")).get("enabled"))
                or (
                    not bool(dict_or_empty(payload.get("negative_replay_unlikelihood")).get("uses_public_data"))
                    and not bool(dict_or_empty(payload.get("negative_replay_unlikelihood")).get("uses_eval_tests_or_solutions"))
                )
            ),
            "hard",
            dict_or_empty(payload.get("negative_replay_unlikelihood")),
        ),
        gate(
            "pairwise_replay_private_train_only_when_enabled",
            (
                not bool(dict_or_empty(payload.get("pairwise_replay_preference")).get("enabled"))
                or bool(dict_or_empty(payload.get("pairwise_replay_preference")).get("private_train_replay_only"))
            ),
            "hard",
            dict_or_empty(payload.get("pairwise_replay_preference")),
        ),
        gate(
            "pairwise_replay_all_examples_failed_when_enabled",
            (
                not bool(dict_or_empty(payload.get("pairwise_replay_preference")).get("enabled"))
                or bool(dict_or_empty(payload.get("pairwise_replay_preference")).get("all_selected_examples_failed_intended_behavior"))
            ),
            "hard",
            dict_or_empty(payload.get("pairwise_replay_preference")),
        ),
        gate(
            "pairwise_replay_no_public_or_eval_visibility_when_enabled",
            (
                not bool(dict_or_empty(payload.get("pairwise_replay_preference")).get("enabled"))
                or (
                    not bool(dict_or_empty(payload.get("pairwise_replay_preference")).get("uses_public_data"))
                    and not bool(dict_or_empty(payload.get("pairwise_replay_preference")).get("uses_eval_tests_or_solutions"))
                )
            ),
            "hard",
            dict_or_empty(payload.get("pairwise_replay_preference")),
        ),
        gate(
            "pairwise_replay_dpo_ipo_uses_frozen_reference_when_enabled",
            (
                not bool(dict_or_empty(payload.get("pairwise_replay_preference")).get("enabled"))
                or str(dict_or_empty(payload.get("pairwise_replay_preference")).get("preference_update_family") or "").upper()
                not in {"DPO", "IPO"}
                or bool(dict_or_empty(payload.get("pairwise_replay_preference")).get("reference_checkpoint_used"))
            ),
            "hard",
            dict_or_empty(payload.get("pairwise_replay_preference")),
        ),
    ]




def negative_replay_weighted_loss_fn_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    token_weights: Any,
    negative_src: Any,
    negative_tgt: Any,
    negative_token_weights: Any,
    negative_weight: float,
    negative_cap: float,
    mx: Any,
    nn: Any,
    plan_class_ids: Any | None = None,
) -> Any:
    base = weighted_loss_fn_mlx(model, src, tgt, pad_id, token_weights, mx, nn)
    return negative_replay_adjusted_loss_mlx(
        base,
        model,
        negative_src,
        negative_tgt,
        negative_token_weights,
        pad_id,
        negative_weight,
        negative_cap,
        mx,
        nn,
    )


def negative_replay_semantic_plan_weighted_loss_fn_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    token_weights: Any,
    plan_targets: Any,
    plan_sample_weights: Any,
    semantic_plan_weight: float,
    negative_src: Any,
    negative_tgt: Any,
    negative_token_weights: Any,
    negative_weight: float,
    negative_cap: float,
    mx: Any,
    nn: Any,
    plan_class_ids: Any | None = None,
) -> Any:
    base = semantic_plan_weighted_loss_fn_mlx(
        model,
        src,
        tgt,
        pad_id,
        token_weights,
        plan_targets,
        plan_sample_weights,
        semantic_plan_weight,
        mx,
        nn,
        plan_class_ids,
    )
    return negative_replay_adjusted_loss_mlx(
        base,
        model,
        negative_src,
        negative_tgt,
        negative_token_weights,
        pad_id,
        negative_weight,
        negative_cap,
        mx,
        nn,
    )


def negative_replay_source_contrastive_weighted_loss_fn_mlx(
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
    negative_src: Any,
    negative_tgt: Any,
    negative_token_weights: Any,
    negative_weight: float,
    negative_cap: float,
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
    return negative_replay_adjusted_loss_mlx(
        base,
        model,
        negative_src,
        negative_tgt,
        negative_token_weights,
        pad_id,
        negative_weight,
        negative_cap,
        mx,
        nn,
    )


def negative_replay_semantic_plan_source_contrastive_weighted_loss_fn_mlx(
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
    negative_src: Any,
    negative_tgt: Any,
    negative_token_weights: Any,
    negative_weight: float,
    negative_cap: float,
    mx: Any,
    nn: Any,
    plan_class_ids: Any | None = None,
) -> Any:
    base = semantic_plan_source_contrastive_weighted_loss_fn_mlx(
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
        mx,
        nn,
        plan_class_ids,
    )
    return negative_replay_adjusted_loss_mlx(
        base,
        model,
        negative_src,
        negative_tgt,
        negative_token_weights,
        pad_id,
        negative_weight,
        negative_cap,
        mx,
        nn,
    )


def negative_replay_adjusted_loss_mlx(
    base_loss: Any,
    model: Any,
    negative_src: Any,
    negative_tgt: Any,
    negative_token_weights: Any,
    pad_id: int,
    negative_weight: float,
    negative_cap: float,
    mx: Any,
    nn: Any,
) -> Any:
    negative_loss = weighted_loss_fn_mlx(model, negative_src, negative_tgt, pad_id, negative_token_weights, mx, nn)
    capped_negative = mx.minimum(
        negative_loss,
        mx.array(float(negative_cap), dtype=mx.float32),
    )
    return base_loss - (float(negative_weight) * capped_negative)


def pairwise_replay_composite_loss_fn_mlx(
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
    negative_src: Any,
    negative_tgt: Any,
    negative_token_weights: Any,
    negative_weight: float,
    negative_cap: float,
    pair_src: Any,
    pair_positive_tgt: Any,
    pair_negative_tgt: Any,
    pair_positive_weights: Any,
    pair_negative_weights: Any,
    pairwise_weight: float,
    pairwise_objective: str,
    reference_model: Any | None,
    pairwise_beta: float,
    pairwise_ipo_target_margin: float,
    pairwise_margin: float,
    pairwise_prefix_tokens: int,
    mx: Any,
    nn: Any,
    plan_class_ids: Any | None = None,
) -> Any:
    """Composable private replay objective.

    Pairwise replay is an ordering signal over already-generated failed
    private-train replay candidates. It can now compose with semantic-plan,
    source-contrastive, and negative-unlikelihood losses instead of silently
    disabling itself when those better loop-training signals are active.
    """

    if float(semantic_plan_weight or 0.0) > 0.0 and hasattr(model, "semantic_plan_logits"):
        if float(source_contrastive_weight or 0.0) > 0.0:
            base = semantic_plan_source_contrastive_weighted_loss_fn_mlx(
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
                mx,
                nn,
                plan_class_ids,
            )
        else:
            base = semantic_plan_weighted_loss_fn_mlx(
                model,
                src,
                tgt,
                pad_id,
                token_weights,
                plan_targets,
                plan_sample_weights,
                semantic_plan_weight,
                mx,
                nn,
                plan_class_ids,
            )
    elif float(source_contrastive_weight or 0.0) > 0.0:
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
    else:
        base = weighted_loss_fn_mlx(model, src, tgt, pad_id, token_weights, mx, nn)

    if float(negative_weight or 0.0) > 0.0:
        base = negative_replay_adjusted_loss_mlx(
            base,
            model,
            negative_src,
            negative_tgt,
            negative_token_weights,
            pad_id,
            negative_weight,
            negative_cap,
            mx,
            nn,
        )
    return pairwise_replay_adjusted_loss_mlx(
        base,
        model,
        pair_src,
        pair_positive_tgt,
        pair_negative_tgt,
        pair_positive_weights,
        pair_negative_weights,
        pad_id,
        pairwise_weight,
        pairwise_objective,
        reference_model,
        pairwise_beta,
        pairwise_ipo_target_margin,
        pairwise_margin,
        pairwise_prefix_tokens,
        mx,
        nn,
    )


def pairwise_replay_adjusted_loss_mlx(
    base_loss: Any,
    model: Any,
    pair_src: Any,
    pair_positive_tgt: Any,
    pair_negative_tgt: Any,
    pair_positive_weights: Any,
    pair_negative_weights: Any,
    pad_id: int,
    pairwise_weight: float,
    pairwise_objective: str,
    reference_model: Any | None,
    pairwise_beta: float,
    pairwise_ipo_target_margin: float,
    pairwise_margin: float,
    pairwise_prefix_tokens: int,
    mx: Any,
    nn: Any,
) -> Any:
    objective = str(pairwise_objective or "margin").strip().lower()
    if objective in {"dpo", "ipo"}:
        positive_logp = sequence_logprob_mean_mlx(
            model,
            pair_src,
            pair_positive_tgt,
            pad_id,
            pair_positive_weights,
            mx,
            nn,
            prefix_token_count=int(pairwise_prefix_tokens or 0),
        )
        negative_logp = sequence_logprob_mean_mlx(
            model,
            pair_src,
            pair_negative_tgt,
            pad_id,
            pair_negative_weights,
            mx,
            nn,
            prefix_token_count=int(pairwise_prefix_tokens or 0),
        )
        policy_logratio = positive_logp - negative_logp
        if reference_model is not None:
            reference_positive_logp = sequence_logprob_mean_mlx(
                reference_model,
                pair_src,
                pair_positive_tgt,
                pad_id,
                pair_positive_weights,
                mx,
                nn,
                prefix_token_count=int(pairwise_prefix_tokens or 0),
            )
            reference_negative_logp = sequence_logprob_mean_mlx(
                reference_model,
                pair_src,
                pair_negative_tgt,
                pad_id,
                pair_negative_weights,
                mx,
                nn,
                prefix_token_count=int(pairwise_prefix_tokens or 0),
            )
            reference_logratio = reference_positive_logp - reference_negative_logp
        else:
            reference_logratio = mx.zeros_like(policy_logratio)
        delta = policy_logratio - reference_logratio
        beta = mx.array(float(pairwise_beta or 0.1), dtype=mx.float32)
        if objective == "ipo":
            target = mx.array(float(pairwise_ipo_target_margin or 1.0), dtype=mx.float32)
            preference = mx.mean((delta - target) * (delta - target))
        else:
            preference = mx.mean(mx.log(mx.array(1.0, dtype=mx.float32) + mx.exp(-beta * delta)))
    else:
        positive_loss = weighted_loss_with_prefix_mlx(
            model,
            pair_src,
            pair_positive_tgt,
            pad_id,
            pair_positive_weights,
            mx,
            nn,
            prefix_token_count=int(pairwise_prefix_tokens or 0),
        )
        negative_loss = weighted_loss_with_prefix_mlx(
            model,
            pair_src,
            pair_negative_tgt,
            pad_id,
            pair_negative_weights,
            mx,
            nn,
            prefix_token_count=int(pairwise_prefix_tokens or 0),
        )
        preference = mx.maximum(
            mx.array(0.0, dtype=mx.float32),
            mx.array(float(pairwise_margin), dtype=mx.float32) + positive_loss - negative_loss,
        )
    return base_loss + (float(pairwise_weight) * preference)


def sequence_logprob_mean_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    token_weights: Any,
    mx: Any,
    nn: Any,
    *,
    prefix_token_count: int = 0,
) -> Any:
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    logits = model(src, tgt_in)
    losses = nn.losses.cross_entropy(logits, tgt_out, reduction="none")
    valid = (tgt_out != pad_id).astype(mx.float32)
    if int(prefix_token_count or 0) > 0:
        positions = mx.arange(tgt_out.shape[1])[None, :]
        valid = valid * (positions < int(prefix_token_count)).astype(mx.float32)
    weighted_valid = valid * token_weights[:, 1:]
    row_loss = mx.sum(losses * weighted_valid, axis=1) / mx.maximum(
        mx.sum(weighted_valid, axis=1),
        mx.array(1.0, dtype=mx.float32),
    )
    return -row_loss


def evaluate_pairwise_policy_preference_mlx(
    model: Any,
    reference_model: Any | None,
    source_rows: list[list[int]],
    positive_target_rows: list[list[int]],
    negative_target_rows: list[list[int]],
    positive_weight_rows: list[list[float]],
    negative_weight_rows: list[list[float]],
    *,
    batch_size: int,
    pad_id: int,
    objective: str,
    beta: float,
    prefix_tokens: int,
    mx: Any,
    nn: Any,
) -> dict[str, Any]:
    if not source_rows or not positive_target_rows or not negative_target_rows:
        return pairwise_policy_preference_not_active(objective)
    source_matrix = mx.array(source_rows, dtype=mx.int32)
    positive_matrix = mx.array(positive_target_rows, dtype=mx.int32)
    negative_matrix = mx.array(negative_target_rows, dtype=mx.int32)
    positive_weight_matrix = mx.array(positive_weight_rows, dtype=mx.float32)
    negative_weight_matrix = mx.array(negative_weight_rows, dtype=mx.float32)
    mx.eval(source_matrix, positive_matrix, negative_matrix, positive_weight_matrix, negative_weight_matrix)

    count = 0
    accepted_preferred = 0
    gap_sum = 0.0
    ref_gap_sum = 0.0
    delta_sum = 0.0
    dpo_loss_sum = 0.0
    model.eval()
    if reference_model is not None:
        reference_model.eval()
    for start in range(0, len(source_rows), max(1, batch_size)):
        indices = list(range(start, min(len(source_rows), start + max(1, batch_size))))
        idx = mx.array(indices, dtype=mx.int32)
        src = source_matrix[idx]
        pos = positive_matrix[idx]
        neg = negative_matrix[idx]
        pos_w = positive_weight_matrix[idx]
        neg_w = negative_weight_matrix[idx]
        pos_logp = sequence_logprob_mean_mlx(
            model,
            src,
            pos,
            pad_id,
            pos_w,
            mx,
            nn,
            prefix_token_count=int(prefix_tokens or 0),
        )
        neg_logp = sequence_logprob_mean_mlx(
            model,
            src,
            neg,
            pad_id,
            neg_w,
            mx,
            nn,
            prefix_token_count=int(prefix_tokens or 0),
        )
        gap = pos_logp - neg_logp
        if reference_model is not None:
            ref_pos_logp = sequence_logprob_mean_mlx(
                reference_model,
                src,
                pos,
                pad_id,
                pos_w,
                mx,
                nn,
                prefix_token_count=int(prefix_tokens or 0),
            )
            ref_neg_logp = sequence_logprob_mean_mlx(
                reference_model,
                src,
                neg,
                pad_id,
                neg_w,
                mx,
                nn,
                prefix_token_count=int(prefix_tokens or 0),
            )
            ref_gap = ref_pos_logp - ref_neg_logp
        else:
            ref_gap = mx.zeros_like(gap)
        delta = gap - ref_gap
        dpo_loss = mx.log(mx.array(1.0, dtype=mx.float32) + mx.exp(-mx.array(float(beta or 0.1), dtype=mx.float32) * delta))
        preferred = (gap > 0.0).astype(mx.int32)
        mx.eval(gap, ref_gap, delta, dpo_loss, preferred)
        batch_count = len(indices)
        count += batch_count
        accepted_preferred += int(mx.sum(preferred).item())
        gap_sum += float(mx.sum(gap).item())
        ref_gap_sum += float(mx.sum(ref_gap).item())
        delta_sum += float(mx.sum(delta).item())
        dpo_loss_sum += float(mx.sum(dpo_loss).item())
    model.train()
    return {
        "enabled": True,
        "policy": "private_pairwise_policy_preference_eval_v1",
        "objective": str(objective or "margin"),
        "row_count": count,
        "accepted_preferred_count": accepted_preferred,
        "accepted_preferred_rate": round(accepted_preferred / max(1, count), 6),
        "mean_policy_logprob_gap": round(gap_sum / max(1, count), 6),
        "mean_reference_logprob_gap": round(ref_gap_sum / max(1, count), 6),
        "mean_policy_minus_reference_gap": round(delta_sum / max(1, count), 6),
        "mean_dpo_loss": round(dpo_loss_sum / max(1, count), 6),
        "reference_checkpoint_used": reference_model is not None,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
        "score_semantics": (
            "Private pairwise preference diagnostic over accepted private target bodies versus failed "
            "private replay candidates. Positive gap means the model assigns higher mean log-probability "
            "to the accepted private body than to the failed generated body for the same prompt/signature. "
            "This is policy-update evidence only, not verifier behavior or learned-generation promotion."
        ),
    }


def pairwise_policy_preference_not_active(objective: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "policy": "not_active",
        "objective": str(objective or "margin"),
        "row_count": 0,
        "accepted_preferred_rate": None,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def pairwise_replay_weighted_loss_fn_mlx(
    model: Any,
    src: Any,
    tgt: Any,
    pad_id: int,
    token_weights: Any,
    pair_src: Any,
    pair_positive_tgt: Any,
    pair_negative_tgt: Any,
    pair_positive_weights: Any,
    pair_negative_weights: Any,
    pairwise_weight: float,
    pairwise_margin: float,
    pairwise_prefix_tokens: int,
    mx: Any,
    nn: Any,
) -> Any:
    base = weighted_loss_fn_mlx(model, src, tgt, pad_id, token_weights, mx, nn)
    return pairwise_replay_adjusted_loss_mlx(
        base,
        model,
        pair_src,
        pair_positive_tgt,
        pair_negative_tgt,
        pair_positive_weights,
        pair_negative_weights,
        pad_id,
        pairwise_weight,
        "margin",
        None,
        0.1,
        1.0,
        pairwise_margin,
        pairwise_prefix_tokens,
        mx,
        nn,
    )


def select_negative_replay_examples(
    config: dict[str, Any],
    private_rows: list[dict[str, Any]],
    *,
    candidate_path: Path | None,
    report_path: Path | None,
    max_rows: int,
    source_fields: list[Any],
    source_text_style: str,
    source_vocab: dict[str, int],
) -> dict[str, Any]:
    if candidate_path is None or max_rows <= 0:
        return {
            "enabled": False,
            "active": False,
            "reason": "not_requested",
            "examples": [],
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    if not candidate_path.exists():
        return {
            "enabled": True,
            "active": False,
            "reason": "missing_candidate_jsonl",
            "candidate_path": rel(candidate_path),
            "examples": [],
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }

    report_summary: dict[str, Any] = {}
    if report_path is not None and report_path.exists():
        try:
            report = read_json(report_path)
            report_summary = {
                "report_path": rel(report_path),
                "report_trigger_state": report.get("trigger_state"),
                "report_policy": report.get("policy"),
                "report_split": get_path(report, ["summary", "split"], ""),
            }
        except Exception as exc:  # pragma: no cover - defensive reporting path
            report_summary = {
                "report_path": rel(report_path),
                "report_read_error_type": type(exc).__name__,
                "report_read_error": str(exc)[:300],
            }

    private_by_id: dict[str, dict[str, Any]] = {}
    for row in private_rows:
        for key in ("task_id", "source_task_id", "source_id"):
            value = str(row.get(key) or "").strip()
            if value:
                private_by_id[value] = row

    examples: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    stage_counts: dict[str, int] = {}
    residual_counts: dict[str, int] = {}
    action_trace_mismatch_counts: dict[str, int] = {}
    integrity_mismatch_counts: dict[str, int] = {}
    integrity_label_counts: dict[str, int] = {}
    integrity_family_counts: dict[str, int] = {}
    raw_code_negative_body_rows = 0
    integrity_unverified_rows = 0
    integrity_verified_rows = 0
    reward_sum = 0.0
    candidate_rows_read = 0
    private_train_only = True
    no_public_data = True
    no_eval_visibility = True
    all_failed = True

    for candidate in iter_jsonl(candidate_path):
        candidate_rows_read += 1
        label = dict_or_empty(candidate.get("private_verifier_label"))
        provenance = dict_or_empty(candidate.get("provenance"))
        replay_split = str(provenance.get("evaluation_split") or candidate.get("evaluation_split") or "").strip()
        split_counts[replay_split or "missing"] = split_counts.get(replay_split or "missing", 0) + 1
        if replay_split != "private_train_replay":
            private_train_only = False
            increment(skipped, "not_private_train_replay")
            continue
        if not label:
            increment(skipped, "missing_private_verifier_label")
            all_failed = False
            continue
        if bool(label.get("intended_behavior_passed")):
            all_failed = False
            increment(skipped, "intended_behavior_passed")
            continue
        if bool(label.get("uses_public_data")) or bool(candidate.get("public_tests_visible_to_generator")) or bool(candidate.get("public_solutions_visible_to_generator")):
            no_public_data = False
            increment(skipped, "public_visibility_or_use")
            continue
        if bool(label.get("uses_eval_tests_or_solutions_for_generation")) or bool(candidate.get("eval_tests_visible_to_generator")) or bool(candidate.get("eval_solution_visible_to_generator")):
            no_eval_visibility = False
            increment(skipped, "eval_visibility_or_use")
            continue
        if int(candidate.get("external_inference_calls") or 0) != 0:
            increment(skipped, "external_inference_used")
            continue
        source_id = str(candidate.get("source_task_id") or "").strip()
        source_row = private_by_id.get(source_id)
        if source_row is None:
            source_row = private_by_id.get(str(candidate.get("task_id") or "").strip())
        if source_row is None:
            increment(skipped, "missing_private_source_row")
            continue
        candidate_code = str(candidate.get("code") or "")
        integrity_audit = recompute_candidate_integrity(candidate)
        integrity_family = str(integrity_audit.get("recomputed_candidate_family") or "unknown")
        integrity_family_counts[integrity_family] = integrity_family_counts.get(integrity_family, 0) + 1
        if bool(integrity_audit.get("integrity_verified")):
            integrity_verified_rows += 1
        else:
            integrity_unverified_rows += 1
        integrity_labels = integrity_negative_replay_labels(integrity_audit)
        for mismatch in list(integrity_audit.get("integrity_mismatches") or []):
            mismatch_key = str(mismatch)
            integrity_mismatch_counts[mismatch_key] = integrity_mismatch_counts.get(mismatch_key, 0) + 1
        for integrity_label in integrity_labels:
            integrity_label_counts[integrity_label] = integrity_label_counts.get(integrity_label, 0) + 1

        body = candidate_body_from_code(
            candidate_code,
            str(candidate.get("entry_point") or source_row.get("entry_point") or "solve"),
        )
        if not body.strip():
            if not integrity_labels:
                increment(skipped, "missing_candidate_body")
                continue
            body = candidate_code.strip()
            if not body:
                increment(skipped, "missing_candidate_code")
                continue
            raw_code_negative_body_rows += 1
        source_text = strict_generator_decode_source_text(
            source_row,
            source_fields,
            source_text_style=source_text_style,
            source_vocab=source_vocab,
        )
        audit = strict_generator_source_text_audit([source_row], [source_text])
        if not bool(audit.get("clean")):
            increment(skipped, "strict_source_text_audit_failed")
            no_public_data = no_public_data and not bool(audit.get("uses_public_data"))
            no_eval_visibility = no_eval_visibility and not bool(audit.get("uses_eval_tests_or_solutions"))
            continue
        stage = str(label.get("verification_stage") or "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        residual = dict_or_empty(candidate.get("private_task_residual_label"))
        residual_class = str(residual.get("residual_class") or "unknown")
        residual_counts[residual_class] = residual_counts.get(residual_class, 0) + 1
        action_trace = dict_or_empty(candidate.get("body_action_trace"))
        action_trace_mismatches = [str(item) for item in list(action_trace.get("mismatch_labels") or [])]
        replay_mismatch_labels = list(dict.fromkeys(action_trace_mismatches + integrity_labels))
        for mismatch in action_trace_mismatches:
            action_trace_mismatch_counts[mismatch] = action_trace_mismatch_counts.get(mismatch, 0) + 1
        reward_sum += float(label.get("verification_reward") or 0.0)
        examples.append(
            {
                "source_text": source_text,
                "accepted_body": str(source_row.get("solution_body") or ""),
                "body": body,
                "candidate_sha256": str(candidate.get("candidate_sha256") or ""),
                "source_task_id": source_id,
                "candidate_task_id": str(candidate.get("task_id") or ""),
                "verification_stage": stage,
                "verification_reward": float(label.get("verification_reward") or 0.0),
                "residual_class": residual_class,
                "body_action_trace_mismatch_labels": replay_mismatch_labels,
                "body_action_trace_policy": str(action_trace.get("policy") or ""),
                "candidate_integrity": {
                    "policy": str(integrity_audit.get("policy") or ""),
                    "recomputed_candidate_family": integrity_family,
                    "integrity_verified": bool(integrity_audit.get("integrity_verified")),
                    "pure_learned_generation": bool(integrity_audit.get("pure_learned_generation")),
                    "integrity_mismatches": [str(item) for item in list(integrity_audit.get("integrity_mismatches") or [])],
                    "code_shape": dict_or_empty(integrity_audit.get("code_shape")),
                    "replay_labels": integrity_labels,
                    "candidate_generation_credit": 0,
                },
                "policy": "private_train_replay_failed_candidate_integrity_negative_unlikelihood_row_v2",
            }
        )
        if len(examples) >= max_rows:
            break

    return {
        "enabled": True,
        "active": bool(examples),
        "policy": "private_train_replay_failed_candidate_integrity_negative_unlikelihood_v2",
        "candidate_path": rel(candidate_path),
        "report": report_summary,
        "candidate_rows_read": candidate_rows_read,
        "requested_rows": int(max_rows),
        "selected_rows": len(examples),
        "skipped_counts": dict(sorted(skipped.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "verification_stage_counts": dict(sorted(stage_counts.items())),
        "residual_class_counts": dict(sorted(residual_counts.items())),
        "body_action_trace_mismatch_counts": dict(sorted(action_trace_mismatch_counts.items())),
        "candidate_integrity_family_counts": dict(sorted(integrity_family_counts.items())),
        "candidate_integrity_mismatch_counts": dict(sorted(integrity_mismatch_counts.items())),
        "candidate_integrity_replay_label_counts": dict(sorted(integrity_label_counts.items())),
        "candidate_integrity_verified_rows": integrity_verified_rows,
        "candidate_integrity_unverified_rows": integrity_unverified_rows,
        "raw_code_negative_body_rows": raw_code_negative_body_rows,
        "uses_independent_candidate_integrity_audit": True,
        "mean_verification_reward": round(reward_sum / max(1, len(examples)), 6),
        "private_train_replay_only": private_train_only,
        "all_selected_examples_failed_intended_behavior": all_failed and bool(examples),
        "uses_private_train_verifier_labels": True,
        "uses_eval_tests_or_solutions": not no_eval_visibility,
        "uses_public_data": not no_public_data,
        "uses_answer_metadata": False,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "candidate_generation_credit": 0,
        "examples": examples,
        "score_semantics": (
            "Negative replay consumes only failed private-train replay candidates with attached private "
            "verifier labels. Source text is rebuilt from the original private prompt/signature row; "
            "candidate code supplies only the body to be downweighted. Candidate integrity is recomputed "
            "independently so syntax/no-function/full-body mismatches become private replay labels instead "
            "of candidate credit. This is private training pressure, not heldout evidence, not public "
            "calibration, and not a learned-generation promotion claim."
        ),
    }


def integrity_negative_replay_labels(integrity_audit: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    shape = dict_or_empty(integrity_audit.get("code_shape"))
    for mismatch in list(integrity_audit.get("integrity_mismatches") or []):
        clean = str(mismatch or "").strip()
        if clean:
            labels.append(f"candidate_integrity_{clean}")
    if not bool(shape.get("syntax_valid")):
        labels.append("candidate_integrity_syntax_invalid")
    if not bool(shape.get("has_function")):
        labels.append("candidate_integrity_no_function_def")
    if bool(shape.get("inert_stub_like")):
        labels.append("candidate_integrity_inert_stub_like")
    if bool(shape.get("unconditional_trivial_return")):
        labels.append("candidate_integrity_trivial_return")
    if not bool(integrity_audit.get("integrity_verified")):
        labels.append("candidate_integrity_not_verified")
    return list(dict.fromkeys(labels))


def replay_private_row_index(private_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    private_by_id: dict[str, dict[str, Any]] = {}
    for row in private_rows:
        for key in ("task_id", "source_task_id", "source_id"):
            value = str(row.get(key) or "").strip()
            if value:
                private_by_id[value] = row
    return private_by_id


def starvation_replay_labels(raw_labels: list[Any]) -> list[str]:
    labels = [str(label) for label in raw_labels if str(label or "").strip()]
    replay_labels: list[str] = []
    if "inside_loop_without_update" in labels:
        replay_labels.extend(["decode_starvation_inside_loop_without_update", "loop_without_decision_or_state_update"])
    if "missing_local_return" in labels:
        replay_labels.append("decode_starvation_missing_local_return")
    if "inside_loop" in labels:
        replay_labels.append("decode_starvation_inside_loop")
    if "current_line_starts_return" in labels:
        replay_labels.extend(["decode_starvation_return_expression_not_closed", "current_line_starts_return"])
    for label in labels:
        if label not in replay_labels:
            replay_labels.append(label)
    return replay_labels


def select_decode_starvation_replay_examples(
    private_rows: list[dict[str, Any]],
    *,
    report_path: Path | None,
    max_rows: int,
    source_fields: list[Any],
    source_text_style: str,
    source_vocab: dict[str, int],
) -> dict[str, Any]:
    if report_path is None or max_rows <= 0:
        return {
            "enabled": False,
            "active": False,
            "reason": "not_requested",
            "examples": [],
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    if not report_path.exists():
        return {
            "enabled": True,
            "active": False,
            "reason": "missing_decode_starvation_report",
            "report_path": rel(report_path),
            "examples": [],
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }

    try:
        report = read_json(report_path)
    except Exception as exc:  # pragma: no cover - defensive reporting path
        return {
            "enabled": True,
            "active": False,
            "reason": "decode_starvation_report_read_failed",
            "report_path": rel(report_path),
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "examples": [],
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }

    summary = dict_or_empty(report.get("summary"))
    split = str(summary.get("split") or "").strip()
    private_split = dict_or_empty(dict_or_empty(summary.get("split_decode_starvation")).get("private_train_replay"))
    if split and split != "private_train_replay":
        return {
            "enabled": True,
            "active": False,
            "reason": "decode_starvation_report_not_private_train_replay",
            "report_path": rel(report_path),
            "report_split": split,
            "examples": [],
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": bool(private_split.get("uses_public_data")),
            "candidate_generation_credit": 0,
        }
    if bool(private_split.get("uses_public_data")) or bool(private_split.get("uses_eval_tests_or_solutions")):
        return {
            "enabled": True,
            "active": False,
            "reason": "decode_starvation_report_visibility_violation",
            "report_path": rel(report_path),
            "report_split": split or "private_train_replay",
            "examples": [],
            "uses_eval_tests_or_solutions": bool(private_split.get("uses_eval_tests_or_solutions")),
            "uses_public_data": bool(private_split.get("uses_public_data")),
            "candidate_generation_credit": 0,
        }

    private_by_id = replay_private_row_index(private_rows)
    examples: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    state_counts: dict[str, int] = {}
    task_count = 0
    beam_rows_read = 0
    seen: set[tuple[str, str]] = set()
    no_public_data = True
    no_eval_visibility = True

    for task in list(private_split.get("examples") or []):
        task_count += 1
        task_dict = dict_or_empty(task)
        task_id = str(task_dict.get("task_id") or "").strip()
        if not task_id:
            increment(skipped, "missing_task_id")
            continue
        source_row = private_by_id.get(task_id)
        if source_row is None:
            increment(skipped, "missing_private_source_row")
            continue
        source_text = strict_generator_decode_source_text(
            source_row,
            source_fields,
            source_text_style=source_text_style,
            source_vocab=source_vocab,
        )
        audit = strict_generator_source_text_audit([source_row], [source_text])
        if not bool(audit.get("clean")):
            increment(skipped, "strict_source_text_audit_failed")
            no_public_data = no_public_data and not bool(audit.get("uses_public_data"))
            no_eval_visibility = no_eval_visibility and not bool(audit.get("uses_eval_tests_or_solutions"))
            continue
        for beam_index, beam in enumerate(list(task_dict.get("top_beam_examples") or [])):
            beam_rows_read += 1
            beam_dict = dict_or_empty(beam)
            body = str(beam_dict.get("body_preview") or "").strip()
            if not body:
                increment(skipped, "missing_body_preview")
                continue
            body_hash = stable_hash(body)
            dedupe_key = (task_id, body_hash)
            if dedupe_key in seen:
                increment(skipped, "duplicate_task_body")
                continue
            seen.add(dedupe_key)
            loop_state = dict_or_empty(beam_dict.get("loop_prefix_state"))
            raw_state_labels = [str(label) for label in list(loop_state.get("state_labels") or [])]
            replay_labels = starvation_replay_labels(raw_state_labels)
            for label in replay_labels:
                state_counts[label] = state_counts.get(label, 0) + 1
            examples.append(
                {
                    "source_text": source_text,
                    "accepted_body": str(source_row.get("solution_body") or ""),
                    "body": body,
                    "candidate_sha256": body_hash,
                    "source_task_id": task_id,
                    "candidate_task_id": f"{task_id}:decode_starvation_beam_{beam_index}",
                    "beam_index": beam_index,
                    "verification_stage": "decode_starvation",
                    "verification_reward": 0.0,
                    "residual_class": "decode_starvation",
                    "body_action_trace_mismatch_labels": replay_labels,
                    "body_action_trace_policy": "decode_starvation_loop_prefix_state_private_replay_v1",
                    "learned_plan_prefix": dict_or_empty(beam_dict.get("learned_plan_prefix")),
                    "policy": "private_train_replay_decode_starvation_negative_unlikelihood_row_v1",
                    "private_train_replay_only": True,
                    "failed_intended_behavior": True,
                    "uses_public_data": False,
                    "uses_eval_tests_or_solutions": False,
                    "candidate_generation_credit": 0,
                    "score_semantics": (
                        "A private decode-starvation top-beam preview used only as a rejected replay body. "
                        "It is not a generated candidate, not scored as capability, and not public training data."
                    ),
                }
            )
            if len(examples) >= max_rows:
                break
        if len(examples) >= max_rows:
            break

    return {
        "enabled": True,
        "active": bool(examples),
        "policy": "private_train_replay_decode_starvation_negative_unlikelihood_v1",
        "report_path": rel(report_path),
        "report_trigger_state": report.get("trigger_state"),
        "report_policy": report.get("policy"),
        "report_split": split or "private_train_replay",
        "task_rows_read": task_count,
        "beam_rows_read": beam_rows_read,
        "requested_rows": int(max_rows),
        "selected_rows": len(examples),
        "skipped_counts": dict(sorted(skipped.items())),
        "decode_starvation_state_counts": dict(sorted(state_counts.items())),
        "private_train_replay_only": True,
        "all_selected_examples_failed_intended_behavior": bool(examples),
        "uses_private_train_verifier_labels": False,
        "uses_eval_tests_or_solutions": not no_eval_visibility,
        "uses_public_data": not no_public_data,
        "uses_answer_metadata": False,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "candidate_generation_credit": 0,
        "examples": examples,
        "score_semantics": (
            "Decode-starvation replay consumes only private-train top-beam previews from strict generator "
            "diagnostics. Source text is rebuilt from the original private prompt/signature row; body_preview "
            "supplies only a rejected body to downweight or rank below the admitted private target. This is "
            "private repair pressure, not public calibration, not a candidate, and not a learned-generation claim."
        ),
    }


def merge_negative_replay_sources(*sources: dict[str, Any]) -> dict[str, Any]:
    active_sources = [source for source in sources if dict_or_empty(source).get("enabled")]
    if not active_sources:
        return {
            "enabled": False,
            "active": False,
            "reason": "not_requested",
            "examples": [],
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    examples: list[dict[str, Any]] = []
    skipped_counts: dict[str, int] = {}
    for source in active_sources:
        examples.extend(list(source.get("examples") or []))
        for key, value in dict_or_empty(source.get("skipped_counts")).items():
            skipped_counts[key] = skipped_counts.get(key, 0) + int(value or 0)
    if len(active_sources) == 1:
        source = dict(active_sources[0])
        source["combined_sources"] = [source_summary(active_sources[0])]
        return source
    return {
        "enabled": True,
        "active": bool(examples),
        "policy": "private_train_replay_combined_failed_candidate_and_decode_starvation_replay_v1",
        "combined_sources": [source_summary(source) for source in active_sources],
        "requested_rows": sum(int(source.get("requested_rows") or 0) for source in active_sources),
        "selected_rows": len(examples),
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "private_train_replay_only": all(bool(source.get("private_train_replay_only")) for source in active_sources if source.get("active")),
        "all_selected_examples_failed_intended_behavior": bool(examples)
        and all(bool(source.get("all_selected_examples_failed_intended_behavior")) for source in active_sources if source.get("active")),
        "uses_private_train_verifier_labels": any(bool(source.get("uses_private_train_verifier_labels")) for source in active_sources),
        "uses_eval_tests_or_solutions": any(bool(source.get("uses_eval_tests_or_solutions")) for source in active_sources),
        "uses_public_data": any(bool(source.get("uses_public_data")) for source in active_sources),
        "uses_answer_metadata": False,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "candidate_generation_credit": 0,
        "examples": examples,
        "score_semantics": (
            "Combined private negative replay. Each source is independently selected under private-train, "
            "no-public, no-eval-visibility rules and contributes only rejected bodies for negative or "
            "pairwise replay. Combined replay grants no candidate-generation credit."
        ),
    }


def source_summary(source: dict[str, Any]) -> dict[str, Any]:
    summary = {key: value for key, value in source.items() if key != "examples"}
    return summary


def negative_replay_summary(
    negative_replay: dict[str, Any],
    *,
    active: bool,
    weight: float,
    cap: float,
    stage_weighting: dict[str, Any],
) -> dict[str, Any]:
    summary = {key: value for key, value in negative_replay.items() if key != "examples"}
    summary.update(
        {
            "enabled": bool(negative_replay.get("enabled")) and float(weight or 0.0) > 0.0,
            "active": bool(active),
            "objective": "subtract_capped_failed_candidate_ce_from_supervised_private_ce_v1" if active else "not_active",
            "weight": float(weight or 0.0),
            "cap": float(cap or 0.0),
            "stage_weighting": stage_weighting,
            "selected_rows": int(negative_replay.get("selected_rows") or 0),
            "score_semantics": (
                "Bounded unlikelihood pressure against failed private-train replay candidates. It is "
                "used only to reduce recurrence of known bad private candidates while preserving the "
                "supervised private target objective. Optional stage weighting punishes lower-reward "
                "failures more than higher-stage failures, giving an ordering signal without treating "
                "wrong code as a positive target. It grants no candidate-generation credit and does "
                "not use public benchmark payloads."
            ),
            "uses_eval_tests_or_solutions": bool(negative_replay.get("uses_eval_tests_or_solutions")),
            "uses_public_data": bool(negative_replay.get("uses_public_data")),
            "uses_answer_metadata": False,
            "candidate_generation_credit": 0,
        }
    )
    return summary


def negative_replay_token_weight_rows(
    examples: list[dict[str, Any]],
    target_rows: list[list[int]],
    *,
    mode: str,
    min_stage_weight: float,
    max_stage_weight: float,
) -> tuple[list[list[float]], dict[str, Any]]:
    clean_mode = str(mode or "uniform").strip() or "uniform"
    lo = max(0.0, float(min_stage_weight if min_stage_weight is not None else 0.25))
    hi = max(lo, float(max_stage_weight if max_stage_weight is not None else 1.0))
    rows: list[list[float]] = []
    weights: list[float] = []
    by_stage: dict[str, list[float]] = {}
    for index, target in enumerate(target_rows):
        example = examples[index] if index < len(examples) else {}
        reward = max(0.0, min(1.0, float(dict_or_empty(example).get("verification_reward") or 0.0)))
        if clean_mode == "reward_inverse":
            multiplier = lo + ((hi - lo) * (1.0 - reward))
        else:
            multiplier = 1.0
        multiplier = round(float(multiplier), 6)
        stage = str(dict_or_empty(example).get("verification_stage") or "unknown")
        by_stage.setdefault(stage, []).append(multiplier)
        weights.append(multiplier)
        rows.append([multiplier for _ in target])
    stage_summary = {
        stage: {
            "count": len(values),
            "mean_weight": round(sum(values) / max(1, len(values)), 6),
            "min_weight": min(values) if values else 0.0,
            "max_weight": max(values) if values else 0.0,
        }
        for stage, values in sorted(by_stage.items())
    }
    return rows, {
        "enabled": bool(rows),
        "policy": "private_verifier_stage_weighted_negative_replay_v1",
        "mode": clean_mode,
        "rows": len(rows),
        "min_stage_weight": lo,
        "max_stage_weight": hi,
        "mean_weight": round(sum(weights) / max(1, len(weights)), 6),
        "stage_summary": stage_summary,
        "score_semantics": (
            "Weights failed private replay candidates by private verifier reward. In reward_inverse "
            "mode, lint/parse failures get stronger negative pressure than runtime-loaded wrong "
            "answers. This is a preference ordering among failed candidates, not positive training "
            "on wrong code and not candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def pairwise_replay_token_weight_rows(
    examples: list[dict[str, Any]],
    positive_target_rows: list[list[int]],
    negative_target_rows: list[list[int]],
    *,
    target_vocab: dict[str, int],
    active: bool,
    action_trace_boost: float,
) -> tuple[list[list[float]], list[list[float]], dict[str, Any]]:
    if not active:
        return [], [], {
            "enabled": False,
            "policy": "not_active",
            "rows": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    positive_weights = [[1.0 for _ in row] for row in positive_target_rows]
    negative_weights = [[1.0 for _ in row] for row in negative_target_rows]
    missing_accepted = sum(1 for row in examples if not str(dict_or_empty(row).get("accepted_body") or "").strip())
    boost = max(0.0, float(action_trace_boost if action_trace_boost is not None else 0.0))
    if boost > 0.0:
        inverse = {int(token_id): str(token) for token, token_id in target_vocab.items()}
        positive_weighted_positions = 0
        negative_weighted_positions = 0
        mismatch_counts: dict[str, int] = {}
        positive_token_counts: dict[str, int] = {}
        negative_token_counts: dict[str, int] = {}
        for index, example in enumerate(examples):
            labels = [str(item) for item in list(dict_or_empty(example).get("body_action_trace_mismatch_labels") or [])]
            for label in labels:
                mismatch_counts[label] = mismatch_counts.get(label, 0) + 1
            positive_tokens = action_trace_positive_replay_tokens(labels)
            negative_tokens = action_trace_negative_replay_tokens(labels)
            if index < len(positive_target_rows) and index < len(positive_weights):
                for pos, token_id in enumerate(positive_target_rows[index]):
                    token_text = inverse.get(int(token_id), "")
                    if token_text not in positive_tokens:
                        continue
                    positive_weights[index][pos] = max(float(positive_weights[index][pos]), boost)
                    positive_weighted_positions += 1
                    positive_token_counts[token_text] = positive_token_counts.get(token_text, 0) + 1
            if index < len(negative_target_rows) and index < len(negative_weights):
                for pos, token_id in enumerate(negative_target_rows[index]):
                    token_text = inverse.get(int(token_id), "")
                    if token_text not in negative_tokens:
                        continue
                    negative_weights[index][pos] = max(float(negative_weights[index][pos]), boost)
                    negative_weighted_positions += 1
                    negative_token_counts[token_text] = negative_token_counts.get(token_text, 0) + 1
        return positive_weights, negative_weights, {
            "enabled": bool(positive_weights and negative_weights),
            "policy": "private_replay_pairwise_action_trace_body_weighting_v1",
            "rows": min(len(positive_weights), len(negative_weights)),
            "missing_accepted_body_rows": missing_accepted,
            "action_trace_replay_loss_boost": boost,
            "positive_weighted_token_positions": positive_weighted_positions,
            "negative_weighted_token_positions": negative_weighted_positions,
            "body_action_trace_mismatch_counts": dict(sorted(mismatch_counts.items())),
            "positive_weighted_token_counts": dict(sorted(positive_token_counts.items())),
            "negative_weighted_token_counts": dict(sorted(negative_token_counts.items())),
            "score_semantics": (
                "Action-trace-aware CE weighting for same-source private pairwise replay. Failed "
                "candidate body_action_trace labels choose which admitted private target tokens get "
                "extra positive pressure and which failed-candidate tokens get extra negative pressure. "
                "It consumes only private-train replay labels produced after generation, does not render "
                "code, does not inspect public/eval tests or solutions, and grants no candidate-generation credit."
            ),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "uses_answer_metadata": False,
            "candidate_generation_credit": 0,
        }
    return positive_weights, negative_weights, {
        "enabled": bool(positive_weights and negative_weights),
        "policy": "private_replay_pairwise_uniform_body_weighting_v1",
        "rows": min(len(positive_weights), len(negative_weights)),
        "missing_accepted_body_rows": missing_accepted,
        "score_semantics": (
            "Uniform CE weighting for same-source private pairwise replay. The accepted side is the "
            "admitted private solution body for the source row; the rejected side is a failed generated "
            "private-train replay candidate. This weights a preference loss only and grants no "
            "candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def action_trace_positive_replay_tokens(labels: list[str]) -> set[str]:
    tokens: set[str] = set()
    label_set = {str(label) for label in labels}
    if "decode_starvation_missing_local_return" in label_set:
        tokens.update({"DEDENT:", "NAME:return", "NEWLINE:"})
    if "decode_starvation_return_expression_not_closed" in label_set or "current_line_starts_return" in label_set:
        tokens.update({"OP:)", "OP:]", "OP:}", "NEWLINE:", "DEDENT:", "NAME:return"})
    if "decode_starvation_inside_loop" in label_set or "decode_starvation_inside_loop_without_update" in label_set:
        tokens.update({"DEDENT:", "NAME:return", "NEWLINE:"})
    if "loop_without_decision_or_state_update" in label_set:
        tokens.update(
            {
                "NAME:if",
                "NAME:else",
                "NAME:try",
                "NAME:except",
                "OP:==",
                "OP:!=",
                "OP:<",
                "OP:<=",
                "OP:>",
                "OP:>=",
                "OP:+=",
                "OP:-=",
                "NAME:max",
                "NAME:min",
                "NAME:abs",
                "NAME:int",
                "NAME:float",
            }
        )
    if "missing_numeric_accumulation" in label_set:
        tokens.update(
            {
                "OP:+",
                "OP:-",
                "OP:+=",
                "OP:-=",
                "NAME:math",
                "NAME:gcd",
                "NAME:max",
                "NAME:min",
                "NAME:abs",
                "NAME:int",
                "NAME:float",
                "NUMBER:1",
            }
        )
    if "missing_list_construction" in label_set:
        tokens.update({"NAME:append", "NAME:extend", "NAME:range", "NAME:len", "NAME:list", "NAME:sorted", "OP:[", "OP:]"})
    if "missing_windowed_finalizer" in label_set:
        tokens.update({"NAME:range", "NAME:len", "NAME:i", "NAME:values", "OP:[", "OP:]", "OP:+", "OP:-", "NUMBER:1"})
    if "missing_gcd_call" in label_set:
        tokens.update({"NAME:math", "NAME:gcd", "NAME:abs"})
    if "missing_bool_not_local_finalizer" in label_set:
        tokens.update({"NAME:not", "NAME:return", "NAME:stack"})
    if "missing_rle_branch_or_update" in label_set:
        tokens.update({"NAME:if", "NAME:else", "OP:[", "OP:]", "OP:==", "OP:+", "NUMBER:1", "NAME:append"})
    if "early_return_inside_loop" in label_set:
        tokens.update({"DEDENT:", "NAME:return"})
    if "unreachable_loop_update_after_control_flow" in label_set:
        tokens.update({"NAME:if", "NAME:else", "OP:==", "OP:!=", "OP:+=", "OP:+", "OP:-", "NAME:append", "NAME:max", "NAME:min"})
    if "shallow_identity_accumulation" in label_set:
        tokens.update({"NAME:if", "NAME:else", "OP:==", "OP:!=", "OP:+", "OP:-", "OP:+=", "NAME:max", "NAME:min", "NAME:gcd", "NAME:range", "NAME:len"})
    if "candidate_integrity_syntax_invalid" in label_set or "candidate_integrity_no_function_def" in label_set:
        tokens.update({"NAME:return", "NEWLINE:", "DEDENT:", "OP:)", "OP:]", "OP:}"})
    if "candidate_integrity_inert_stub_like" in label_set or "candidate_integrity_trivial_return" in label_set:
        tokens.update({"NAME:if", "NAME:for", "NAME:return", "NAME:len", "NAME:range", "NAME:max", "NAME:min", "NAME:sum", "OP:+", "OP:-", "OP:*", "OP:/"})
    return tokens


def action_trace_negative_replay_tokens(labels: list[str]) -> set[str]:
    tokens: set[str] = set()
    label_set = {str(label) for label in labels}
    if "decode_starvation_return_expression_not_closed" in label_set or "current_line_starts_return" in label_set:
        tokens.update({"OP:[", "OP:.", "NAME:in", "NAME:and", "NAME:or", "NAME:is"})
    if "decode_starvation_missing_local_return" in label_set:
        tokens.update({"OP:[", "OP:.", "NAME:and", "NAME:or", "NAME:in"})
    if "decode_starvation_inside_loop" in label_set or "decode_starvation_inside_loop_without_update" in label_set:
        tokens.update({"NAME:continue", "NAME:break"})
    if "early_return_inside_loop" in label_set:
        tokens.update({"NAME:return"})
    if "unreachable_loop_update_after_control_flow" in label_set:
        tokens.update({"NAME:continue", "NAME:break", "NAME:return"})
    if "shallow_identity_accumulation" in label_set:
        tokens.update({"NAME:append", "OP:.", "NAME:return"})
    if "loop_without_decision_or_state_update" in label_set:
        tokens.update({"NAME:return"})
    if "candidate_integrity_syntax_invalid" in label_set:
        tokens.update({"OP:{", "OP:}", "OP:.", "NAME:else", "NAME:elif"})
    if "candidate_integrity_no_function_def" in label_set:
        tokens.update({"NAME:def", "OP::"})
    if "candidate_integrity_inert_stub_like" in label_set or "candidate_integrity_trivial_return" in label_set:
        tokens.update({"NAME:None", "NUMBER:0", "NUMBER:1", "NAME:return"})
    return tokens


def pairwise_replay_vocab_summary(
    negative_replay: dict[str, Any],
    *,
    active: bool,
    weight: float,
    objective: str = "margin",
    beta: float = 0.1,
    reference_checkpoint_used: bool = False,
) -> dict[str, Any]:
    return {
        "enabled": bool(negative_replay.get("enabled")) and float(weight or 0.0) > 0.0,
        "active": bool(active),
        "policy": str(negative_replay.get("policy") or "not_enabled"),
        "objective": str(objective or "margin"),
        "dpo_beta": float(beta or 0.0),
        "reference_checkpoint_used": bool(reference_checkpoint_used),
        "candidate_path": negative_replay.get("candidate_path"),
        "selected_rows": int(negative_replay.get("selected_rows") or 0),
        "uses_eval_tests_or_solutions": bool(negative_replay.get("uses_eval_tests_or_solutions")),
        "uses_public_data": bool(negative_replay.get("uses_public_data")),
        "candidate_generation_credit": 0,
    }


def pairwise_replay_summary(
    negative_replay: dict[str, Any],
    *,
    active: bool,
    weight: float,
    objective: str,
    beta: float,
    ipo_target_margin: float,
    margin: float,
    prefix_tokens: int,
    token_weighting: dict[str, Any],
    policy_before: dict[str, Any],
    policy_after: dict[str, Any],
    reference_checkpoint: str,
) -> dict[str, Any]:
    summary = {key: value for key, value in negative_replay.items() if key != "examples"}
    clean_objective = str(objective or "margin").strip().lower() or "margin"
    summary.update(
        {
            "enabled": bool(negative_replay.get("enabled")) and float(weight or 0.0) > 0.0,
            "active": bool(active),
            "objective": f"same_source_accepted_over_failed_candidate_{clean_objective}_v1" if active else "not_active",
            "preference_update_family": "DPO" if clean_objective == "dpo" else ("IPO" if clean_objective == "ipo" else "margin"),
            "weight": float(weight or 0.0),
            "dpo_beta": float(beta or 0.0),
            "ipo_target_margin": float(ipo_target_margin or 0.0),
            "margin": float(margin or 0.0),
            "prefix_token_count": int(prefix_tokens or 0),
            "reference_checkpoint": str(reference_checkpoint or ""),
            "reference_checkpoint_used": bool(reference_checkpoint),
            "policy_preference_before": policy_before,
            "policy_preference_after": policy_after,
            "accepted_preferred_rate_delta": accepted_preferred_rate_delta(policy_before, policy_after),
            "mean_policy_minus_reference_gap_delta": mean_policy_gap_delta(policy_before, policy_after),
            "token_weighting": token_weighting,
            "selected_rows": int(negative_replay.get("selected_rows") or 0),
            "score_semantics": (
                "Private-only pairwise replay preference. For each failed private-train replay candidate, "
                "the model is trained to prefer the admitted private solution body over its own failed "
                "generated body for the same prompt/signature. The margin objective uses CE loss ordering; "
                "DPO/IPO use accepted-vs-rejected sequence log-probability gaps against the frozen source "
                "checkpoint as reference. This does not inspect public benchmark payloads, eval tests, eval "
                "solutions, or teacher output; it emits no candidate and grants no learned-generation "
                "promotion credit."
            ),
            "uses_eval_tests_or_solutions": bool(negative_replay.get("uses_eval_tests_or_solutions")),
            "uses_public_data": bool(negative_replay.get("uses_public_data")),
            "uses_answer_metadata": False,
            "candidate_generation_credit": 0,
        }
    )
    return summary


def accepted_preferred_rate_delta(before: dict[str, Any], after: dict[str, Any]) -> float | None:
    before_value = maybe_float(dict_or_empty(before).get("accepted_preferred_rate"))
    after_value = maybe_float(dict_or_empty(after).get("accepted_preferred_rate"))
    if before_value is None or after_value is None:
        return None
    return round(after_value - before_value, 6)


def mean_policy_gap_delta(before: dict[str, Any], after: dict[str, Any]) -> float | None:
    before_value = maybe_float(dict_or_empty(before).get("mean_policy_minus_reference_gap"))
    after_value = maybe_float(dict_or_empty(after).get("mean_policy_minus_reference_gap"))
    if before_value is None or after_value is None:
        return None
    return round(after_value - before_value, 6)


def maybe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def negative_replay_vocab_summary(negative_replay: dict[str, Any], *, active: bool) -> dict[str, Any]:
    return {
        "enabled": bool(negative_replay.get("enabled")),
        "active": bool(active),
        "policy": str(negative_replay.get("policy") or "not_enabled"),
        "candidate_path": negative_replay.get("candidate_path"),
        "selected_rows": int(negative_replay.get("selected_rows") or 0),
        "uses_eval_tests_or_solutions": bool(negative_replay.get("uses_eval_tests_or_solutions")),
        "uses_public_data": bool(negative_replay.get("uses_public_data")),
        "candidate_generation_credit": 0,
    }


def candidate_body_from_code(code: str, entry_point: str) -> str:
    text = str(code or "")
    if not text.strip():
        return ""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return candidate_body_from_code_textual(text, entry_point)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    function = next((node for node in functions if node.name == entry_point), None) or (functions[0] if functions else None)
    if function is None or not function.body:
        return candidate_body_from_code_textual(text, entry_point)
    lines = text.splitlines()
    body_lines: list[str] = []
    for stmt in function.body:
        lineno = int(getattr(stmt, "lineno", 0) or 0)
        end_lineno = int(getattr(stmt, "end_lineno", lineno) or lineno)
        if lineno <= 0:
            continue
        body_lines.extend(lines[lineno - 1 : end_lineno])
    return textwrap.dedent("\n".join(body_lines)).strip("\n")


def candidate_body_from_code_textual(code: str, entry_point: str) -> str:
    lines = str(code or "").splitlines()
    header_index = -1
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def ") and (not entry_point or stripped.startswith(f"def {entry_point}(")):
            header_index = index
            break
    if header_index < 0:
        for index, line in enumerate(lines):
            if line.strip().startswith("def "):
                header_index = index
                break
    if header_index < 0:
        return ""
    body_lines: list[str] = []
    for line in lines[header_index + 1 :]:
        if line and not line.startswith((" ", "\t")):
            break
        body_lines.append(line)
    return textwrap.dedent("\n".join(body_lines)).strip("\n")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def select_rehearsal_examples(
    config: dict[str, Any],
    vocab_payload: dict[str, Any],
    *,
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    target_mode: str,
    max_source: int,
    max_target: int,
    checkpoint_dir: Path,
    source_fields: list[Any],
    source_text_style: str,
    requested_rows: int,
    seed: int,
) -> dict[str, Any]:
    if requested_rows <= 0:
        return {
            "enabled": False,
            "reason": "rehearsal_rows_zero",
            "examples": [],
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    budget_id = str(vocab_payload.get("budget_id") or "").strip()
    if not budget_id:
        return {
            "enabled": True,
            "active": False,
            "reason": "missing_source_checkpoint_budget_id",
            "examples": [],
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    try:
        pretraining_cfg = dict_or_empty(config.get("strict_generator_pretraining"))
        budget = dict(selected_budget(pretraining_cfg, budget_id))
        working_config = strict_spine_config_with_budget_overrides(config, budget)
        staged = stage_full_state_examples(
            working_config,
            budget,
            budget_id=budget_id,
            checkpoint_dir=checkpoint_dir,
            seed=seed,
        )
        encoded = encode_staged_full_state_rows(
            staged,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            max_source=max_source,
            max_target=max_target,
            target_mode=target_mode,
        )
    except Exception as exc:
        return {
            "enabled": True,
            "active": False,
            "reason": "rehearsal_stage_failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "examples": [],
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    staged_summary = dict_or_empty(staged.get("summary"))
    encoded_summary = dict_or_empty(encoded.get("summary"))
    eval_count = int(encoded_summary.get("eval_example_count") or 0)
    staged_examples = [row for row in list(staged.get("examples") or []) if isinstance(row, dict)]
    train_examples = staged_examples[eval_count:] if eval_count < len(staged_examples) else staged_examples
    candidates: list[dict[str, Any]] = []
    for row in train_examples:
        body = str(row.get("body") or "")
        function = str(row.get("function") or "").strip()
        if not function or not body.strip():
            continue
        source_row = strict_rehearsal_source_row(row)
        strict_source = strict_generator_decode_source_text(
            source_row,
            source_fields,
            source_text_style=source_text_style,
            source_vocab=source_vocab,
        )
        audit = strict_generator_source_text_audit(
            [{"solution_body": body, "tests": ""}],
            [strict_source],
        )
        if not bool(audit.get("clean")):
            continue
        candidates.append(
            {
                "source_text": strict_source,
                "body": body,
                "path": str(row.get("path") or ""),
                "function": function,
                "source_policy": "strict_prompt_signature_rebuilt_rehearsal_v1",
            }
        )
    selected = deterministic_sample(candidates, min(len(candidates), requested_rows), seed + 1701)
    return {
        "enabled": True,
        "active": bool(selected),
        "source_policy": "strict_prompt_signature_rebuilt_rehearsal_v1",
        "budget_id": budget_id,
        "checkpoint_stage_cache_dir": rel(checkpoint_dir / "stage_cache"),
        "requested_rows": int(requested_rows),
        "available_train_examples": len(candidates),
        "examples": selected,
        "selected_rows": len(selected),
        "staged_summary": {
            "cache_status": staged_summary.get("cache_status"),
            "cache_path": staged_summary.get("cache_path"),
            "staged_example_count": staged_summary.get("staged_example_count"),
            "row_example_count": staged_summary.get("row_example_count"),
            "public_benchmark_payload_admitted_count": staged_summary.get("public_benchmark_payload_admitted_count"),
        },
        "encoded_summary": {
            "train_example_count": encoded_summary.get("train_example_count"),
            "eval_example_count": encoded_summary.get("eval_example_count"),
            "source_unknown_token_rate": encoded_summary.get("source_unknown_token_rate"),
            "target_unknown_token_rate": encoded_summary.get("target_unknown_token_rate"),
        },
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "candidate_generation_credit": 0,
    }


def strict_rehearsal_source_row(row: dict[str, Any]) -> dict[str, Any]:
    """Build a strict prompt/signature-only rehearsal row from staged code data.

    Staged pretraining rows may carry legacy ``source_text`` with operation tags.
    Rehearsal for strict private adaptation must not trust that cached text.
    This row contains only a generic prompt, visible function name, and coarse
    argument count from the local staged function-quality summary.
    """

    function = str(row.get("function") or "solve").strip() or "solve"
    quality = dict_or_empty(row.get("quality"))
    try:
        argc = int(quality.get("parameter_count") or 1)
    except (TypeError, ValueError):
        argc = 1
    argc = max(1, min(argc, 3))
    return {
        "prompt": f"Implement Python function {function}.",
        "entry_point": function,
        "decoder_contract": {"visible_arg_count_hint": argc},
    }


def rehearsal_summary(rehearsal: dict[str, Any], *, selected_count: int) -> dict[str, Any]:
    return {
        key: value
        for key, value in rehearsal.items()
        if key != "examples"
    } | {
        "selected_rows": int(selected_count),
        "policy": "admitted_source_corpus_rehearsal_mixing_v1" if selected_count else "not_enabled_or_inactive",
        "score_semantics": (
            "Optional rehearsal rows are admitted private/licensed source-corpus function bodies from "
            "the same no-public checkpoint spine, but their source text is rebuilt through the strict "
            "prompt/signature-only builder before mixing. They are mixed only into training to reduce "
            "catastrophic forgetting during private task adaptation; they do not inspect eval tests, "
            "eval solutions, public benchmarks, verifier results, answer metadata, cached operation "
            "tags, or cached body fragments, and they emit no candidates."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def exclude_family_disjoint_holdout_rows(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    private_residual_repair_split: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if private_residual_repair_split:
        return rows, {
            "enabled": False,
            "clean": True,
            "policy": "private_residual_repair_row_holdout_v1",
            "reason": "family_disjoint_holdout_exclusion_disabled_for_private_residual_repair",
            "input_row_count": len(rows),
            "excluded_row_count": 0,
            "remaining_row_count": len(rows),
            "family_disjoint_evidence": False,
            "public_calibration_eligible": False,
            "score_semantics": (
                "This repair mode intentionally keeps configured family-disjoint holdout families inside "
                "the admitted private training pool so residual families can receive supervised private "
                "pressure. Evaluation is only row/variant heldout within the selected private pool; the "
                "result must not be reported as family-disjoint transfer, public calibration evidence, or "
                "learned-generation promotion evidence by itself."
            ),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "candidate_generation_credit": 0,
        }
    try:
        selection = select_family_disjoint_rows(config, max_rows=1)
    except Exception as exc:
        return rows, {
            "enabled": False,
            "clean": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:400],
            "score_semantics": (
                "Private adaptation could not resolve family-disjoint holdout families. "
                "The gate fails rather than risking a contaminated family-disjoint claim."
            ),
        }
    if not bool(selection.get("active")):
        return rows, {
            "enabled": False,
            "clean": True,
            "reason": dict_or_empty(selection.get("summary")).get("reason") or "family_disjoint_selection_inactive",
            "excluded_row_count": 0,
        }
    family_key = str(selection.get("family_key") or "concept_residual_label")
    holdout = {str(value) for value in selection.get("holdout_families") or []}
    if not holdout:
        return rows, {
            "enabled": True,
            "clean": False,
            "reason": "missing_holdout_families",
            "excluded_row_count": 0,
            "family_key": family_key,
        }
    from neural_seed_token_decoder_comparator import strict_disjoint_family_key  # noqa: PLC0415

    kept: list[dict[str, Any]] = []
    excluded = 0
    for row in rows:
        if strict_disjoint_family_key(row, family_key) in holdout:
            excluded += 1
            continue
        kept.append(row)
    return kept, {
        "enabled": True,
        "clean": excluded >= 0 and bool(kept),
        "policy": "exclude_configured_family_disjoint_holdout_families_from_private_adaptation_v1",
        "family_key": family_key,
        "holdout_families": sorted(holdout),
        "input_row_count": len(rows),
        "excluded_row_count": excluded,
        "remaining_row_count": len(kept),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
    }


def gate(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def resolve_checkpoint_paths(args: argparse.Namespace, checkpoint_report: dict[str, Any]) -> tuple[Path, Path]:
    budget = dict_or_empty(checkpoint_report.get("budget"))
    summary = dict_or_empty(checkpoint_report.get("summary"))
    checkpoint_raw = str(args.checkpoint or budget.get("checkpoint") or summary.get("checkpoint") or "").strip()
    vocab_raw = str(args.vocab or budget.get("vocab") or summary.get("vocab") or "").strip()
    if not checkpoint_raw:
        raise SystemExit("missing MLX checkpoint path; pass --checkpoint or --checkpoint-report")
    if not vocab_raw:
        raise SystemExit("missing MLX vocab path; pass --vocab or --checkpoint-report")
    return resolve(checkpoint_raw), resolve(vocab_raw)


def safe_slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "checkpoint"))
    return "_".join(part for part in cleaned.split("_") if part)[:96] or "checkpoint"


def is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
