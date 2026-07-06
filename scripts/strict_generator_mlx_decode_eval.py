#!/usr/bin/env python3
"""Decode/evaluate a strict-generator MLX checkpoint on private instruments.

This connects the MLX strict-generator checkpoint to the existing token-decoder
candidate, static-coherence, and private verifier contracts. It does not train,
does not run public calibration, does not call a teacher, and does not use
templates/renderers/tools/fallbacks as learned-generation evidence.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_verifier import evaluate_private_candidates  # noqa: E402
from neural_seed_code_proposer_comparator import (  # noqa: E402
    dict_or_empty,
    encode_many,
    get_path,
    rel,
    resolve,
    stable_hash,
)
from neural_seed_token_decoder_comparator import (  # noqa: E402
    arm_summary,
    body_structure_summary,
    broad_private_heldout_manifest_rows,
    candidate_schema_summary,
    decode_beam_sort_key,
    family_disjoint_manifest_rows,
    final_decode_beam_sort_key,
    grammar_repair_summary,
    no_cheat_candidate_evidence,
    static_coherence_candidate_pool_size,
    static_coherence_ranker_enabled,
    static_coherence_sort_key,
    static_coherence_summary,
    token_candidate_rows_for_view,
)
from neural_seed_token_decoder_support import (  # noqa: E402
    PLAN_BODY_START_TOKEN,
    baseline_candidate,
    body_like_target_mode,
    body_tokens_for_target_mode,
    call_arg_count,
    current_line_tokens,
    decode_body_tokens,
    forced_lightweight_python_token,
    innermost_open_paren_index,
    invalid_known_builtin_arity,
    invalid_known_method_arity,
    learned_plan_prefix_target_mode,
    learned_plan_semantic_slots_body_target_mode,
    normalize_body_text,
    strict_body_expression_is_likely_noniterable,
    strict_body_prefix_local_static_types,
    strict_body_static_type_from_values,
    split_learned_plan_prefix_tokens,
    token_allowed_by_policy,
    token_values,
)
from neural_seed_decode_static_guard import (  # noqa: E402
    action_trace_call_name,
    ast_expr_is_trivial_constant_like,
    ast_text_for_decode_guard,
    body_has_nontrivial_return,
    body_has_top_level_valued_return,
    body_uses_allowed_parameter,
    completion_ready,
    control_flow_pathology_summary,
    decode_guard_parameter_names,
    decode_guard_return_dependency_summary,
    decode_static_guard,
    definitely_bound_return_summary,
    expression_complexity_for_decode_guard,
    expression_is_direct_parameter_identity,
    expression_load_names,
    expression_store_names,
    parsed_decode_guard_function,
    static_guard_candidate_code,
)
from neural_seed_expression_value_guard import (  # noqa: E402
    EXPRESSION_VALUE_BARE_BUILTINS,
    expression_value_has_bare_builtin_value,
    expression_value_quality_summary,
)
from strict_generator_mlx_replay_selection import (  # noqa: E402
    exclude_configured_holdout_rows_for_replay,
    private_solution_body_features,
    private_train_replay_tier_inventory,
    private_train_replay_tier_match,
    select_broad_private_heldout_rows,
    select_family_disjoint_rows,
    select_private_train_replay_rows,
)
from strict_generator_mlx_decode_reporting import (  # noqa: E402
    build_gates,
    candidate_integrity_summary,
    now,
    read_json,
    resolve_checkpoint_paths,
    resolve_checkpoint_paths_from_report,
    selection_summary,
    stable_hash_file,
    stamp_mlx_rows,
    syntax_summary_from_rows,
    write_json,
    write_jsonl,
)
from strict_generator_mlx_decode_guards import (  # noqa: E402
    allowed_signature_names_for_task,
    condition_prefix_is_only_negation,
    current_body_line_values,
    current_branch_receiver_type,
    current_else_excludes_receiver_type,
    current_prefix_control_depth,
    expression_tail_values,
    first_call_argument_values,
    infer_receiver_tail_kind,
    isinstance_guard_matches,
    isinstance_guard_type,
    isinstance_guard_type_token_redundant_or_contradictory,
    known_builtin_first_arg_type_invalid,
    known_call_prefix_would_be_invalid,
    known_positional_max_args,
    method_receiver_known_invalid,
    normalized_condition_values,
    prefix_lines_with_depth,
    prefix_mentions_allowed_parameter,
    receiver_tail_call_name,
    receiver_tail_is_known_non_dict,
    receiver_tail_is_known_non_string,
    receiver_tail_is_temporary_expression,
    repeated_condition_chain_on_indent,
    token_blocked_by_strict_decode_guard,
)
from strict_generator_mlx_source_text import (  # noqa: E402
    STRICT_GENERATOR_SOURCE_TEXT_POLICY,
    checkpoint_source_text_style,
    strict_generator_decode_source_text,
    strict_generator_source_text_audit,
    visible_parameter_type_hints_from_source_text,
)
from strict_generator_mlx_specialist_routing import (  # noqa: E402
    PRIVATE_TRAIN_REPLAY_SPECIALIST_TIERS,
    load_specialist_heads,
    private_train_replay_specialist_routes,
    specialist_head_routing_summary,
    specialist_route_decode_options,
    specialist_route_record,
)
from strict_generator_mlx_pretraining_probe import (  # noqa: E402
    BODY_ACTION_ROLES,
    BODY_OPERAND_ROLES,
    MlxStrictGenerator,
    body_action_role_id_for_token,
    body_operand_role_id_for_token,
)
from strict_generator_mlx_pretraining_probe import SEMANTIC_SLOT_ROLES  # noqa: E402
from strict_generator_mlx_decode_plans import (  # noqa: E402
    beam_mentions_allowed_parameter,
    visible_parameter_exploration_choices,
    learned_prefix_decision_expectation_from_tokens,
    source_condition_exploration_choices,
    expression_closure_guard_choices,
    direct_local_return_continuation_choices,
    current_line_expected_closer,
    current_line_tail_needs_operand,
    current_line_needs_colon_from_values,
    expression_closure_line_can_end,
    expression_closure_can_dedent,
    source_condition_priority_prefix,
    source_condition_preempts_loop_plan,
    source_condition_prefix_can_add_truthiness,
    source_condition_expectation_from_source_text,
    visible_argument_names_from_source_text,
    source_condition_expectations_summary,
    source_condition_decode_beam_sort_key,
    learned_prefix_decision_final_decode_beam_sort_key,
    learned_prefix_decision_rank_score,
    learned_prefix_loop_plan_adequacy_metadata,
    learned_prefix_loop_expectation_from_tokens,
    loop_plan_adequacy_for_body,
    learned_prefix_body_action_trace_metadata,
    body_action_trace_for_body,
    unreachable_statement_count_after_control_flow,
    expression_has_constant,
    loop_plan_value_shape,
    loop_plan_value_has_stateful_transform,
    loop_iter_mentions_source,
    expression_is_direct_source_return,
    loop_plan_update_arg_is_semantic,
    loop_plan_exploration_choices,
    loop_plan_priority_prefix,
    token_blocked_by_loop_plan,
    filter_loop_plan_blocked_choices,
    filter_expression_value_blocked_choices,
    token_blocked_by_expression_value_guard,
    token_value_for_guard,
    expression_value_allows_empty_initializer,
    expression_value_inside_update_call,
    expression_value_current_call_arg_values,
    expression_value_arg_is_empty_literal,
    loop_plan_initializer_continuation,
    loop_plan_loop_header_continuation,
    loop_plan_update_continuation,
    loop_plan_return_continuation,
    loop_plan_blocks_shallow_identity_append,
    loop_plan_allows_direct_loop_append,
    loop_plan_accumulator_name,
    loop_plan_loop_var_name,
    loop_plan_highest_probability_name,
    loop_plan_source_arg,
    loop_plan_primary_init_shape,
    loop_plan_update_method,
    loop_plan_has_initializer,
    loop_plan_first_assigned_local,
    loop_plan_first_loop_var,
    loop_plan_has_loop_over_source,
    loop_plan_has_update_call,
    loop_plan_has_local_return,
    loop_plan_inside_loop,
    source_condition_final_decode_beam_sort_key,
    source_condition_prefix_progress,
    source_condition_adequacy_for_body,
    source_condition_body_has_truthiness_guard,
    source_condition_body_has_default_return,
    source_condition_body_has_sequence_type_guard,
    isinstance_guard_covers_types,
    source_condition_body_has_guarded_first_item_return,
    source_condition_body_has_guarded_data_return,
    expr_is_first_index_of_name,
    bool_expr_has_and_name,
    bool_expr_is_direct_name_operand,
    bool_expr_contains_name,
    expr_contains_name,
    source_plan_compatibility_for_plan_token,
    source_plan_compatibility_summary,
    source_condition_candidate_summary,
    body_action_trace_candidate_summary,
)


DEFAULT_CONFIG = ROOT / "configs" / "neural_seed_token_decoder_comparator.json"
DEFAULT_CHECKPOINT_REPORT = ROOT / "reports" / "strict_generator_mlx_pretraining_probe_5m_v1.json"
DEFAULT_OUT = ROOT / "reports" / "strict_generator_mlx_decode_eval_v1.json"
DEFAULT_CANDIDATES = ROOT / "reports" / "strict_generator_mlx_decode_eval_v1_candidates.jsonl"


def parse_specialist_checkpoint_report_specs(specs: list[str]) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for raw in specs:
        text = str(raw or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise SystemExit(f"invalid --specialist-checkpoint-report {text!r}; expected TIER=REPORT.json")
        tier, report_path = [part.strip() for part in text.split("=", 1)]
        if tier not in PRIVATE_TRAIN_REPLAY_SPECIALIST_TIERS:
            allowed = ", ".join(PRIVATE_TRAIN_REPLAY_SPECIALIST_TIERS)
            raise SystemExit(f"invalid specialist tier {tier!r}; allowed tiers: {allowed}")
        if tier in parsed:
            raise SystemExit(f"duplicate specialist checkpoint tier {tier!r}")
        resolved = resolve(report_path)
        report = read_json(resolved)
        checkpoint, vocab = resolve_checkpoint_paths_from_report(report)
        parsed[tier] = {
            "tier": tier,
            "report_path": rel(resolved),
            "checkpoint": checkpoint,
            "vocab": vocab,
            "report": report,
        }
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint-report", default=rel(DEFAULT_CHECKPOINT_REPORT))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--vocab", default="")
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--candidates-out", default=rel(DEFAULT_CANDIDATES))
    parser.add_argument("--split", choices=["family_disjoint", "broad_private_heldout", "private_train_replay", "both"], default="both")
    parser.add_argument("--max-family-rows", type=int, default=0)
    parser.add_argument("--max-broad-rows", type=int, default=0)
    parser.add_argument("--max-train-replay-rows", type=int, default=0)
    parser.add_argument("--max-target-tokens-override", type=int, default=0)
    parser.add_argument(
        "--train-replay-tier",
        choices=["any", "simple_return", "loop_accumulate", "algorithmic_small"],
        default="any",
    )
    parser.add_argument("--output-top-k", type=int, default=0)
    parser.add_argument("--require-parameter-use", action="store_true")
    parser.add_argument("--require-nontrivial-return", action="store_true")
    parser.add_argument("--require-top-level-return", action="store_true")
    parser.add_argument("--use-semantic-plan-head-prefix", action="store_true")
    parser.add_argument(
        "--prefer-source-plan-compatibility",
        action="store_true",
        help=(
            "Rerank source-conditioned semantic plan-head choices using broad prompt-visible intent, "
            "operation, and type tags. This is a zero-credit adequacy experiment; it does not render code, "
            "inspect tests/solutions/public payloads, or create learned-generation promotion evidence."
        ),
    )
    parser.add_argument(
        "--use-semantic-slot-head-prefix",
        action="store_true",
        help=(
            "Use the source-conditioned semantic slot auxiliary head, when present, to choose learned "
            "prefix slot tokens before raw body decoding. The head does not render code or grant "
            "learned-generation promotion credit."
        ),
    )
    parser.add_argument(
        "--enable-learned-expression-token-bias",
        action="store_true",
        help=(
            "Use model-generated expression slot prefixes as a soft probability bias for raw body "
            "tokens in expression contexts. This does not force a complete expression, render code, "
            "inspect tests/solutions, or grant learned-generation promotion credit."
        ),
    )
    parser.add_argument(
        "--use-body-transition-head",
        action="store_true",
        help=(
            "Blend the learned prefix-conditioned body-transition projection into raw token logits. "
            "This is a learned checkpoint head only; it does not render code, call tools, inspect tests, "
            "or grant generation credit without candidate-integrity/verifier behavior."
        ),
    )
    parser.add_argument("--body-transition-head-blend", type=float, default=0.35)
    parser.add_argument(
        "--use-body-action-head",
        action="store_true",
        help=(
            "Bias raw token probabilities with the learned prefix-conditioned body-action role head. "
            "The head predicts broad structural roles only; it does not render code, call tools, inspect "
            "tests/solutions, or grant generation credit without candidate-integrity/verifier behavior."
        ),
    )
    parser.add_argument("--body-action-head-blend", type=float, default=0.35)
    parser.add_argument(
        "--use-body-operand-head",
        action="store_true",
        help=(
            "Bias raw token probabilities with the learned prefix-conditioned body-operand binding "
            "head. The head predicts value/operand roles only; it does not render code, call tools, "
            "inspect tests/solutions, or grant generation credit without candidate-integrity/verifier behavior."
        ),
    )
    parser.add_argument("--body-operand-head-blend", type=float, default=0.35)
    parser.add_argument("--prefer-learned-prefix-decision-adequacy", action="store_true")
    parser.add_argument("--prefer-source-condition-adequacy", action="store_true")
    parser.add_argument("--require-source-condition-adequacy", action="store_true")
    parser.add_argument(
        "--specialist-checkpoint-report",
        action="append",
        default=[],
        metavar="TIER=REPORT.json",
        help=(
            "Route an existing private-train replay tier through a specialized learned checkpoint, "
            "for example simple_return=reports/...json. Routing is model/head selection evidence only; "
            "it does not render code, inspect tests/solutions, or grant learned-generation credit."
        ),
    )
    parser.add_argument(
        "--use-specialist-route-profiles",
        action="store_true",
        help=(
            "When specialist heads are configured, use tier-specific decode profiles derived from "
            "existing private evidence. The profile choice is recorded as route metadata and grants "
            "zero candidate-generation credit."
        ),
    )
    parser.add_argument(
        "--block-shallow-loop-identity-update",
        action="store_true",
        help=(
            "Experimentally block direct accumulator.append(loop_var)/add(loop_var) continuations "
            "for non-identity loop plans. This is task-blind decode hygiene, not a renderer."
        ),
    )
    parser.add_argument(
        "--enable-loop-progress-guard",
        action="store_true",
        help=(
            "Task-blind loop-progress decode guard for learned plan-prefix loops. It prioritizes "
            "an update before loop exit/finalizer so beams do not starve inside unfinished blocks."
        ),
    )
    parser.add_argument(
        "--enable-expression-closure-guard",
        action="store_true",
        help=(
            "Task-blind expression/block closure decode guard. It prioritizes delimiter closure, "
            "statement newline, and safe dedent/return closure from generated prefix state only."
        ),
    )
    parser.add_argument(
        "--enable-expression-value-guard",
        action="store_true",
        help=(
            "Task-blind expression value hygiene guard. It rejects generated-prefix pathologies "
            "such as empty update-call arguments or bare builtin objects used as values. This "
            "does not render code and grants no learned-generation credit."
        ),
    )
    parser.add_argument(
        "--require-binding-prefix-groups",
        action="store_true",
        help=(
            "Ablation-only learned-prefix guard requiring loop/branch, update, and result "
            "operand-binding slots before body decoding when the checkpoint vocabulary supports "
            "SLOT:BIND_* tokens. This uses no tests or solutions and grants no generation credit."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = read_json(resolve(args.config))
    checkpoint_report = read_json(resolve(args.checkpoint_report)) if str(args.checkpoint_report or "").strip() else {}
    checkpoint, vocab = resolve_checkpoint_paths(args, checkpoint_report)
    specialist_specs = parse_specialist_checkpoint_report_specs(list(args.specialist_checkpoint_report or []))
    report, candidates = run_decode_eval(
        config,
        config_path=str(args.config),
        checkpoint_report_path=str(args.checkpoint_report),
        checkpoint_path=checkpoint,
        vocab_path=vocab,
        specialist_checkpoint_specs=specialist_specs,
        use_specialist_route_profiles=bool(args.use_specialist_route_profiles),
        split=str(args.split),
        max_family_rows=max(0, int(args.max_family_rows or 0)),
        max_broad_rows=max(0, int(args.max_broad_rows or 0)),
        max_train_replay_rows=max(0, int(args.max_train_replay_rows or 0)),
        max_target_tokens_override=max(0, int(args.max_target_tokens_override or 0)),
        train_replay_tier=str(args.train_replay_tier or "any"),
        output_top_k=max(0, int(args.output_top_k or 0)),
        require_parameter_use=bool(args.require_parameter_use),
        require_nontrivial_return=bool(args.require_nontrivial_return),
        require_top_level_return=bool(args.require_top_level_return),
        use_semantic_plan_head_prefix=bool(args.use_semantic_plan_head_prefix),
        prefer_source_plan_compatibility=bool(args.prefer_source_plan_compatibility),
        use_semantic_slot_head_prefix=bool(args.use_semantic_slot_head_prefix),
        enable_learned_expression_token_bias=bool(args.enable_learned_expression_token_bias),
        use_body_transition_head=bool(args.use_body_transition_head),
        body_transition_head_blend=max(0.0, min(1.0, float(args.body_transition_head_blend if args.body_transition_head_blend is not None else 0.35))),
        use_body_action_head=bool(args.use_body_action_head),
        body_action_head_blend=max(0.0, min(1.0, float(args.body_action_head_blend if args.body_action_head_blend is not None else 0.35))),
        use_body_operand_head=bool(args.use_body_operand_head),
        body_operand_head_blend=max(0.0, min(1.0, float(args.body_operand_head_blend if args.body_operand_head_blend is not None else 0.35))),
        prefer_learned_prefix_decision_adequacy=bool(args.prefer_learned_prefix_decision_adequacy),
        prefer_source_condition_adequacy=bool(args.prefer_source_condition_adequacy),
        require_source_condition_adequacy=bool(args.require_source_condition_adequacy),
        block_shallow_loop_identity_update=bool(args.block_shallow_loop_identity_update),
        enable_loop_progress_guard=bool(args.enable_loop_progress_guard),
        enable_expression_closure_guard=bool(args.enable_expression_closure_guard),
        enable_expression_value_guard=bool(args.enable_expression_value_guard),
        require_binding_prefix_groups=bool(args.require_binding_prefix_groups),
        execute=bool(args.execute),
    )
    write_json(resolve(args.out), report)
    write_jsonl(resolve(args.candidates_out), candidates)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW", "PLANNED"} else 2


def preselected_selection_for_split(
    preselected_splits: dict[str, dict[str, Any]] | None,
    split_name: str,
    *,
    max_rows: int,
) -> dict[str, Any] | None:
    source = dict_or_empty((preselected_splits or {}).get(split_name))
    if not source:
        return None
    selection = json.loads(json.dumps(source))
    eval_rows = list(selection.get("eval_rows") or [])
    if max_rows > 0 and len(eval_rows) > max_rows:
        eval_rows = eval_rows[:max_rows]
        selection["eval_rows"] = eval_rows
    eval_row_ids = [
        str(row.get("task_id") or row.get("source_task_id") or stable_hash(json.dumps(row, sort_keys=True)))
        for row in eval_rows
    ]
    summary = dict_or_empty(selection.get("summary"))
    summary.update(
        {
            "preselected_split_reuse": True,
            "preselected_split_policy": "checkpoint_replay_private_split_reuse_v1",
            "preselected_eval_row_count": len(eval_rows),
            "preselected_eval_row_id_hash": stable_hash(json.dumps(eval_row_ids, sort_keys=True)),
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "fallback_template_router_tool_credit_count": 0,
        }
    )
    selection["summary"] = summary
    return selection


def run_decode_eval(
    config: dict[str, Any],
    *,
    config_path: str,
    checkpoint_report_path: str,
    checkpoint_path: Path,
    vocab_path: Path,
    specialist_checkpoint_specs: dict[str, dict[str, Any]],
    use_specialist_route_profiles: bool,
    split: str,
    max_family_rows: int,
    max_broad_rows: int,
    max_train_replay_rows: int,
    max_target_tokens_override: int,
    train_replay_tier: str,
    output_top_k: int,
    require_parameter_use: bool,
    require_nontrivial_return: bool,
    require_top_level_return: bool,
    use_semantic_plan_head_prefix: bool,
    prefer_source_plan_compatibility: bool,
    use_semantic_slot_head_prefix: bool,
    enable_learned_expression_token_bias: bool,
    use_body_transition_head: bool,
    body_transition_head_blend: float,
    use_body_action_head: bool,
    body_action_head_blend: float,
    use_body_operand_head: bool,
    body_operand_head_blend: float,
    prefer_learned_prefix_decision_adequacy: bool,
    prefer_source_condition_adequacy: bool,
    require_source_condition_adequacy: bool,
    block_shallow_loop_identity_update: bool,
    enable_loop_progress_guard: bool,
    enable_expression_closure_guard: bool,
    enable_expression_value_guard: bool,
    require_binding_prefix_groups: bool,
    execute: bool,
    preselected_splits: dict[str, dict[str, Any]] | None = None,
    checkpoint_loader_cache: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    if not execute:
        return (
            {
                "policy": "project_theseus_strict_generator_mlx_decode_eval_v1",
                "created_utc": now(),
                "trigger_state": "PLANNED",
                "execute": False,
                "summary": {
                    "checkpoint": rel(checkpoint_path),
                    "vocab": rel(vocab_path),
                    "specialist_head_routing": specialist_head_routing_summary(
                        specialist_checkpoint_specs,
                        active=False,
                        routes=[],
                        use_route_profiles=use_specialist_route_profiles,
                    ),
                    "split": split,
                    "train_replay_tier": train_replay_tier,
                    "max_target_tokens_override": int(max_target_tokens_override or 0),
                    "use_semantic_plan_head_prefix": bool(use_semantic_plan_head_prefix),
                    "prefer_source_plan_compatibility": bool(prefer_source_plan_compatibility),
                    "use_semantic_slot_head_prefix": bool(use_semantic_slot_head_prefix),
                    "enable_learned_expression_token_bias": bool(enable_learned_expression_token_bias),
                    "use_body_transition_head": bool(use_body_transition_head),
                    "body_transition_head_blend": float(body_transition_head_blend or 0.0),
                    "use_body_action_head": bool(use_body_action_head),
                    "body_action_head_blend": float(body_action_head_blend or 0.0),
                    "use_body_operand_head": bool(use_body_operand_head),
                    "body_operand_head_blend": float(body_operand_head_blend or 0.0),
                    "prefer_learned_prefix_decision_adequacy": bool(prefer_learned_prefix_decision_adequacy),
                    "prefer_source_condition_adequacy": bool(prefer_source_condition_adequacy),
                    "require_source_condition_adequacy": bool(require_source_condition_adequacy),
                    "block_shallow_loop_identity_update": bool(block_shallow_loop_identity_update),
                    "enable_loop_progress_guard": bool(enable_loop_progress_guard),
                    "enable_expression_closure_guard": bool(enable_expression_closure_guard),
                    "enable_expression_value_guard": bool(enable_expression_value_guard),
                    "require_binding_prefix_groups": bool(require_binding_prefix_groups),
                    "public_training_rows": 0,
                    "external_inference_calls": 0,
                },
                "runtime_ms": int((time.perf_counter() - started) * 1000),
            },
            [],
        )

    try:
        import mlx.core as mx
        import mlx.nn as nn
    except BaseException as exc:
        return (
            {
                "policy": "project_theseus_strict_generator_mlx_decode_eval_v1",
                "created_utc": now(),
                "trigger_state": "RED",
                "execute": True,
                "summary": {
                    "mlx_available": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:600],
                    "public_training_rows": 0,
                    "external_inference_calls": 0,
                },
                "runtime_ms": int((time.perf_counter() - started) * 1000),
            },
            [],
        )

    loaded = load_mlx_checkpoint(vocab_path, checkpoint_path, mx=mx, nn=nn, loader_cache=checkpoint_loader_cache)
    model = loaded["model"]
    vocab_payload = loaded["vocab_payload"]
    specialist_heads = load_specialist_heads(
        specialist_checkpoint_specs,
        loader=lambda vocab, checkpoint, *, mx, nn: load_mlx_checkpoint(vocab, checkpoint, mx=mx, nn=nn),
        mx=mx,
        nn=nn,
    )
    default_head = {
        "head_id": "default",
        "head_type": "default_checkpoint",
        "model": model,
        "vocab_payload": vocab_payload,
        "checkpoint": checkpoint_path,
        "vocab": vocab_path,
        "report_path": checkpoint_report_path,
    }
    specialist_route_records: list[dict[str, Any]] = []
    checkpoint_target_mode = str(vocab_payload.get("target_mode") or get_path(config, ["body_structure_decoder", "target_mode"], "body_tokens"))
    selection_config = json.loads(json.dumps(config))
    selection_config.setdefault("body_structure_decoder", {})
    selection_config["body_structure_decoder"]["target_mode"] = checkpoint_target_mode
    selected: dict[str, Any] = {}
    all_candidates: list[dict[str, Any]] = []
    preselected_reuse_summary = {
        "enabled": bool(preselected_splits),
        "policy": "checkpoint_replay_private_split_reuse_v1",
        "requested_split_keys": sorted(str(key) for key in (preselected_splits or {}).keys()),
        "used_split_keys": [],
        "uses_eval_tests_or_solutions_for_generation": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }

    if split in {"family_disjoint", "both"}:
        family_selection = preselected_selection_for_split(
            preselected_splits,
            "family_disjoint",
            max_rows=max_family_rows,
        )
        if family_selection is not None:
            preselected_reuse_summary["used_split_keys"].append("family_disjoint")
        else:
            family_selection = select_family_disjoint_rows(selection_config, max_rows=max_family_rows)
        family_selection["checkpoint"] = rel(checkpoint_path)
        family_report, family_candidates = evaluate_split(
            selection_config,
            selection=family_selection,
            split_name="family_disjoint",
            model=model,
            vocab_payload=vocab_payload,
            mx=mx,
            max_row_override=max_family_rows,
            max_target_tokens_override=max_target_tokens_override,
            output_top_k=output_top_k,
            require_parameter_use=require_parameter_use,
            require_nontrivial_return=require_nontrivial_return,
            require_top_level_return=require_top_level_return,
            use_semantic_plan_head_prefix=use_semantic_plan_head_prefix,
            prefer_source_plan_compatibility=prefer_source_plan_compatibility,
            use_semantic_slot_head_prefix=use_semantic_slot_head_prefix,
            enable_learned_expression_token_bias=enable_learned_expression_token_bias,
            use_body_transition_head=use_body_transition_head,
            body_transition_head_blend=body_transition_head_blend,
            use_body_action_head=use_body_action_head,
            body_action_head_blend=body_action_head_blend,
            use_body_operand_head=use_body_operand_head,
            body_operand_head_blend=body_operand_head_blend,
            prefer_learned_prefix_decision_adequacy=prefer_learned_prefix_decision_adequacy,
            prefer_source_condition_adequacy=prefer_source_condition_adequacy,
            require_source_condition_adequacy=require_source_condition_adequacy,
            block_shallow_loop_identity_update=block_shallow_loop_identity_update,
            enable_loop_progress_guard=enable_loop_progress_guard,
            enable_expression_closure_guard=enable_expression_closure_guard,
            enable_expression_value_guard=enable_expression_value_guard,
            require_binding_prefix_groups=require_binding_prefix_groups,
        )
        selected["family_disjoint"] = family_report
        all_candidates.extend(family_candidates)

    if split in {"broad_private_heldout", "both"}:
        broad_selection = preselected_selection_for_split(
            preselected_splits,
            "broad_private_heldout",
            max_rows=max_broad_rows,
        )
        if broad_selection is not None:
            preselected_reuse_summary["used_split_keys"].append("broad_private_heldout")
        else:
            broad_selection = select_broad_private_heldout_rows(selection_config, max_rows=max_broad_rows)
        broad_selection["checkpoint"] = rel(checkpoint_path)
        broad_report, broad_candidates = evaluate_split(
            selection_config,
            selection=broad_selection,
            split_name="broad_private_heldout",
            model=model,
            vocab_payload=vocab_payload,
            mx=mx,
            max_row_override=max_broad_rows,
            max_target_tokens_override=max_target_tokens_override,
            output_top_k=output_top_k,
            require_parameter_use=require_parameter_use,
            require_nontrivial_return=require_nontrivial_return,
            require_top_level_return=require_top_level_return,
            use_semantic_plan_head_prefix=use_semantic_plan_head_prefix,
            prefer_source_plan_compatibility=prefer_source_plan_compatibility,
            use_semantic_slot_head_prefix=use_semantic_slot_head_prefix,
            enable_learned_expression_token_bias=enable_learned_expression_token_bias,
            use_body_transition_head=use_body_transition_head,
            body_transition_head_blend=body_transition_head_blend,
            use_body_action_head=use_body_action_head,
            body_action_head_blend=body_action_head_blend,
            use_body_operand_head=use_body_operand_head,
            body_operand_head_blend=body_operand_head_blend,
            prefer_learned_prefix_decision_adequacy=prefer_learned_prefix_decision_adequacy,
            prefer_source_condition_adequacy=prefer_source_condition_adequacy,
            require_source_condition_adequacy=require_source_condition_adequacy,
            block_shallow_loop_identity_update=block_shallow_loop_identity_update,
            enable_loop_progress_guard=enable_loop_progress_guard,
            enable_expression_closure_guard=enable_expression_closure_guard,
            enable_expression_value_guard=enable_expression_value_guard,
            require_binding_prefix_groups=require_binding_prefix_groups,
        )
        selected["broad_private_heldout"] = broad_report
        all_candidates.extend(broad_candidates)

    if split == "private_train_replay":
        routes = private_train_replay_specialist_routes(
            train_replay_tier=train_replay_tier,
            specialist_heads=specialist_heads,
            default_head=default_head,
            max_rows=max_train_replay_rows,
            use_route_profiles=use_specialist_route_profiles,
        )
        for route in routes:
            tier = str(route["tier"])
            split_key = str(route["split_key"])
            head = dict_or_empty(route.get("head"))
            route_max_rows = int(route.get("max_rows") or 0)
            route_options = specialist_route_decode_options(
                route,
                max_target_tokens_override=max_target_tokens_override,
                output_top_k=output_top_k,
                require_parameter_use=require_parameter_use,
                require_nontrivial_return=require_nontrivial_return,
                require_top_level_return=require_top_level_return,
                use_semantic_plan_head_prefix=use_semantic_plan_head_prefix,
                prefer_source_plan_compatibility=prefer_source_plan_compatibility,
                use_semantic_slot_head_prefix=use_semantic_slot_head_prefix,
                enable_learned_expression_token_bias=enable_learned_expression_token_bias,
                use_body_transition_head=use_body_transition_head,
                body_transition_head_blend=body_transition_head_blend,
                use_body_action_head=use_body_action_head,
                body_action_head_blend=body_action_head_blend,
                use_body_operand_head=use_body_operand_head,
                body_operand_head_blend=body_operand_head_blend,
                prefer_learned_prefix_decision_adequacy=prefer_learned_prefix_decision_adequacy,
                prefer_source_condition_adequacy=prefer_source_condition_adequacy,
                require_source_condition_adequacy=require_source_condition_adequacy,
                block_shallow_loop_identity_update=block_shallow_loop_identity_update,
                enable_loop_progress_guard=enable_loop_progress_guard,
                enable_expression_closure_guard=enable_expression_closure_guard,
                enable_expression_value_guard=enable_expression_value_guard,
                require_binding_prefix_groups=require_binding_prefix_groups,
            )
            route["decode_options"] = route_options
            train_replay_selection = select_private_train_replay_rows(
                selection_config,
                max_rows=route_max_rows,
                tier=tier,
            )
            train_replay_selection["checkpoint"] = rel(head.get("checkpoint") or checkpoint_path)
            train_replay_selection["specialist_route"] = specialist_route_record(route)
            train_replay_report, train_replay_candidates = evaluate_split(
                selection_config,
                selection=train_replay_selection,
                split_name="private_train_replay",
                model=head["model"] if "model" in head else model,
                vocab_payload=dict_or_empty(head.get("vocab_payload")) or vocab_payload,
                mx=mx,
                max_row_override=route_max_rows,
                max_target_tokens_override=int(route_options["max_target_tokens_override"]),
                output_top_k=int(route_options["output_top_k"]),
                require_parameter_use=bool(route_options["require_parameter_use"]),
                require_nontrivial_return=bool(route_options["require_nontrivial_return"]),
                require_top_level_return=bool(route_options["require_top_level_return"]),
                use_semantic_plan_head_prefix=bool(route_options["use_semantic_plan_head_prefix"]),
                prefer_source_plan_compatibility=bool(route_options.get("prefer_source_plan_compatibility", prefer_source_plan_compatibility)),
                use_semantic_slot_head_prefix=bool(route_options.get("use_semantic_slot_head_prefix", use_semantic_slot_head_prefix)),
                enable_learned_expression_token_bias=bool(route_options.get("enable_learned_expression_token_bias", enable_learned_expression_token_bias)),
                use_body_transition_head=bool(route_options.get("use_body_transition_head", use_body_transition_head)),
                body_transition_head_blend=float(route_options.get("body_transition_head_blend", body_transition_head_blend)),
                use_body_action_head=bool(route_options.get("use_body_action_head", use_body_action_head)),
                body_action_head_blend=float(route_options.get("body_action_head_blend", body_action_head_blend)),
                use_body_operand_head=bool(route_options.get("use_body_operand_head", use_body_operand_head)),
                body_operand_head_blend=float(route_options.get("body_operand_head_blend", body_operand_head_blend)),
                prefer_learned_prefix_decision_adequacy=bool(route_options["prefer_learned_prefix_decision_adequacy"]),
                prefer_source_condition_adequacy=bool(route_options["prefer_source_condition_adequacy"]),
                require_source_condition_adequacy=bool(route_options["require_source_condition_adequacy"]),
                block_shallow_loop_identity_update=bool(route_options["block_shallow_loop_identity_update"]),
                enable_loop_progress_guard=bool(route_options["enable_loop_progress_guard"]),
                enable_expression_closure_guard=bool(route_options["enable_expression_closure_guard"]),
                enable_expression_value_guard=bool(route_options["enable_expression_value_guard"]),
                require_binding_prefix_groups=bool(route_options["require_binding_prefix_groups"]),
            )
            selected[split_key] = train_replay_report
            all_candidates.extend(train_replay_candidates)
            specialist_route_records.append(specialist_route_record(route))

    integrity_summary = candidate_integrity_summary(all_candidates)
    body_action_trace_summary = body_action_trace_candidate_summary(all_candidates)
    gates = build_gates(selected, all_candidates, integrity_summary=integrity_summary)
    hard_pass = all(row["passed"] for row in gates if row["severity"] == "hard")
    trigger_state = "GREEN" if hard_pass else "RED"
    if trigger_state == "GREEN" and any(not row["passed"] for row in gates):
        trigger_state = "YELLOW"

    summary = {
        "split": split,
        "checkpoint": rel(checkpoint_path),
        "checkpoint_sha256": stable_hash_file(checkpoint_path),
        "checkpoint_report": checkpoint_report_path,
        "vocab": rel(vocab_path),
        "vocab_sha256": stable_hash_file(vocab_path),
        "backend": "mlx_high_level_transformer_direct_decode",
        "decode_scheduler": "mlx_batched_active_beam_scheduler_v1",
        "decode_topk_selection": strict_mlx_decode_topk_selection_receipt(),
        "checkpoint_loader_reuse": dict_or_empty(loaded.get("loader_reuse_receipt")),
        "device": str(mx.default_device()),
        "dims": vocab_payload.get("dims"),
        "max_source": vocab_payload.get("max_source"),
        "max_target": vocab_payload.get("max_target"),
        "target_mode": vocab_payload.get("target_mode") or get_path(config, ["body_structure_decoder", "target_mode"], "body_tokens"),
        "source_vocab_size": len(dict_or_empty(vocab_payload.get("source_vocab"))),
        "target_vocab_size": len(dict_or_empty(vocab_payload.get("target_vocab"))),
        "candidate_rows": int(integrity_summary.get("generated_candidate_count", 0) or 0),
        "generated_candidate_rows": int(integrity_summary.get("generated_candidate_count", 0) or 0),
        "manifest_candidate_rows": len(all_candidates),
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "open_or_pretrained_model_weights_used": False,
        "fallback_template_router_tool_credit_count": 0,
        "specialist_head_routing": specialist_head_routing_summary(
            specialist_checkpoint_specs,
            active=bool(specialist_route_records),
            routes=specialist_route_records,
            use_route_profiles=use_specialist_route_profiles,
        ),
        "split_passes": {
            name: get_path(report, ["summary", "private_verifier", "trained_passed"], 0)
            for name, report in selected.items()
        },
        "split_inert_stub_rates": {
            name: get_path(report, ["summary", "static_coherence", "inert_stub_rate"], 0.0)
            for name, report in selected.items()
        },
        "split_nontrivial_return_rates": {
            name: get_path(report, ["summary", "static_coherence", "nontrivial_return_rate"], 0.0)
            for name, report in selected.items()
        },
        "candidate_integrity": integrity_summary,
        "body_action_trace": body_action_trace_summary,
        "split_body_action_trace": {
            name: get_path(report, ["summary", "body_action_trace"], {})
            for name, report in selected.items()
        },
        "split_source_plan_compatibility": {
            name: get_path(report, ["summary", "source_plan_compatibility"], {})
            for name, report in selected.items()
        },
        "preselected_split_reuse": preselected_reuse_summary,
        "split_decode_starvation": {
            name: get_path(report, ["summary", "decode_starvation"], {})
            for name, report in selected.items()
        },
        "split_private_verifier": {
            name: get_path(report, ["summary", "private_verifier"], {})
            for name, report in selected.items()
        },
        "split_verifier_cache_warmup": {
            name: get_path(report, ["summary", "private_verifier", "verifier_cache_warmup"], {})
            for name, report in selected.items()
        },
        "decode_guard": {
            "require_parameter_use": require_parameter_use,
            "require_nontrivial_return": require_nontrivial_return,
            "require_top_level_return": require_top_level_return,
            "max_target_tokens_override": int(max_target_tokens_override or 0),
            "use_semantic_plan_head_prefix": bool(use_semantic_plan_head_prefix),
            "prefer_source_plan_compatibility": bool(prefer_source_plan_compatibility),
            "use_semantic_slot_head_prefix": bool(use_semantic_slot_head_prefix),
            "enable_learned_expression_token_bias": bool(enable_learned_expression_token_bias),
            "use_body_transition_head": bool(use_body_transition_head),
            "body_transition_head_blend": float(body_transition_head_blend or 0.0),
            "use_body_action_head": bool(use_body_action_head),
            "body_action_head_blend": float(body_action_head_blend or 0.0),
            "use_body_operand_head": bool(use_body_operand_head),
            "body_operand_head_blend": float(body_operand_head_blend or 0.0),
            "prefer_learned_prefix_decision_adequacy": bool(prefer_learned_prefix_decision_adequacy),
            "prefer_source_condition_adequacy": bool(prefer_source_condition_adequacy),
            "require_source_condition_adequacy": bool(require_source_condition_adequacy),
            "block_shallow_loop_identity_update": bool(block_shallow_loop_identity_update),
            "enable_loop_progress_guard": bool(enable_loop_progress_guard),
            "enable_expression_closure_guard": bool(enable_expression_closure_guard),
            "enable_expression_value_guard": bool(enable_expression_value_guard),
            "require_binding_prefix_groups": bool(require_binding_prefix_groups),
            "output_top_k_override": output_top_k,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "fallback_returns_allowed": False,
        },
    }
    return (
        {
            "policy": "project_theseus_strict_generator_mlx_decode_eval_v1",
            "created_utc": now(),
            "trigger_state": trigger_state,
            "execute": True,
            "summary": summary,
            "splits": selected,
            "gates": gates,
            "score_semantics": (
                "Private MLX checkpoint decode/eval bridge. Candidate generation sees prompt/signature text, "
                "the learned MLX checkpoint, grammar legality, and task-blind static hygiene only. It does not "
                "use public benchmarks, teacher inference, hidden tests, solution bodies, templates, semantic "
                "renderers, deterministic tools, or fallback returns as learned-generation evidence."
            ),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        all_candidates,
    )


def load_mlx_checkpoint(
    vocab_path: Path,
    checkpoint_path: Path,
    *,
    mx: Any,
    nn: Any,
    loader_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    load_started = time.perf_counter()
    vocab_sha = stable_hash_file(vocab_path)
    checkpoint_sha = stable_hash_file(checkpoint_path)
    cache = loader_cache if isinstance(loader_cache, dict) else None
    if cache is not None:
        cache.setdefault("policy", "strict_mlx_same_vocab_checkpoint_loader_cache_v1")
        cache.setdefault("enabled", True)
        cache.setdefault("vocab_payloads", {})
        cache.setdefault("models", {})
        cache.setdefault("stats", {})
        stats = dict_or_empty(cache.get("stats"))
        stats.setdefault("vocab_read_count", 0)
        stats.setdefault("vocab_cache_hit_count", 0)
        stats.setdefault("model_construct_count", 0)
        stats.setdefault("model_reuse_count", 0)
        stats.setdefault("checkpoint_weight_load_count", 0)
        stats.setdefault("strict_weight_load_count", 0)
        stats.setdefault("nonstrict_weight_load_count", 0)
        cache["stats"] = stats
    else:
        stats = {}

    vocab_cache = dict_or_empty(cache.get("vocab_payloads")) if cache is not None else {}
    if cache is not None and vocab_sha in vocab_cache:
        vocab_payload = json.loads(json.dumps(vocab_cache[vocab_sha]))
        stats["vocab_cache_hit_count"] = int(stats.get("vocab_cache_hit_count") or 0) + 1
        vocab_cache_hit = True
    else:
        vocab_payload = read_json(vocab_path)
        vocab_cache_hit = False
        if cache is not None:
            stats["vocab_read_count"] = int(stats.get("vocab_read_count") or 0) + 1
            vocab_cache[vocab_sha] = json.loads(json.dumps(vocab_payload))
            cache["vocab_payloads"] = vocab_cache
    source_vocab = dict_or_empty(vocab_payload.get("source_vocab"))
    target_vocab = dict_or_empty(vocab_payload.get("target_vocab"))
    dims = dict_or_empty(vocab_payload.get("dims"))
    model_key_payload = {
        "vocab_sha256": vocab_sha,
        "source_vocab_size": len(source_vocab),
        "target_vocab_size": len(target_vocab),
        "max_source": int(vocab_payload.get("max_source") or 96),
        "max_target": int(vocab_payload.get("max_target") or 160),
        "dims": dims,
    }
    model_key = stable_hash(json.dumps(model_key_payload, sort_keys=True))
    model_cache = cache.setdefault("models", {}) if cache is not None else {}
    if cache is not None and model_key in model_cache:
        model = model_cache[model_key]
        stats["model_reuse_count"] = int(stats.get("model_reuse_count") or 0) + 1
        model_reused = True
    else:
        model = MlxStrictGenerator(
            source_vocab_size=len(source_vocab),
            target_vocab_size=len(target_vocab),
            max_source_len=int(vocab_payload.get("max_source") or 96),
            max_target_len=int(vocab_payload.get("max_target") or 160),
            d_model=int(dims.get("d_model") or 224),
            nhead=int(dims.get("nhead") or 4),
            num_layers=int(dims.get("num_layers") or 2),
            dim_feedforward=int(dims.get("dim_feedforward") or 448),
            mx=mx,
            nn=nn,
        ).model
        model_reused = False
        if cache is not None:
            stats["model_construct_count"] = int(stats.get("model_construct_count") or 0) + 1
            model_cache[model_key] = model
            cache["models"] = model_cache
    strict_load = True
    try:
        model.load_weights(str(checkpoint_path))
    except Exception as exc:
        if not any(name in str(exc) for name in ("plan_router", "slot_router", "body_transition_router", "body_action_router", "body_operand_router")):
            raise
        strict_load = False
        model.load_weights(str(checkpoint_path), strict=False)
    mx.eval(model.parameters())
    model.eval()
    if cache is not None:
        stats["checkpoint_weight_load_count"] = int(stats.get("checkpoint_weight_load_count") or 0) + 1
        if strict_load:
            stats["strict_weight_load_count"] = int(stats.get("strict_weight_load_count") or 0) + 1
        else:
            stats["nonstrict_weight_load_count"] = int(stats.get("nonstrict_weight_load_count") or 0) + 1
        cache["stats"] = stats
        loaded_hashes = list(cache.get("loaded_checkpoint_sha256") or [])
        loaded_hashes.append(checkpoint_sha)
        cache["loaded_checkpoint_sha256"] = loaded_hashes
    return {
        "model": model,
        "vocab_payload": vocab_payload,
        "loader_reuse_receipt": {
            "policy": "strict_mlx_same_vocab_checkpoint_loader_cache_v1",
            "enabled": bool(cache is not None),
            "vocab_sha256": vocab_sha,
            "checkpoint_sha256": checkpoint_sha,
            "vocab_cache_hit": bool(vocab_cache_hit),
            "model_reused": bool(model_reused),
            "model_key": model_key,
            "strict_weight_load": bool(strict_load),
            "checkpoint_weight_reloaded": True,
            "load_runtime_ms": int((time.perf_counter() - load_started) * 1000),
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "fallback_template_router_tool_credit_count": 0,
            "score_semantics": (
                "Runtime-economics receipt only. Same-vocab sweeps may reuse vocab payloads and model "
                "construction, but every checkpoint still reloads its own weights before decoding. This "
                "does not change generation visibility and does not support capability claims."
            ),
        },
    }



def evaluate_split(
    config: dict[str, Any],
    *,
    selection: dict[str, Any],
    split_name: str,
    model: Any,
    vocab_payload: dict[str, Any],
    mx: Any,
    max_row_override: int,
    max_target_tokens_override: int,
    output_top_k: int,
    require_parameter_use: bool,
    require_nontrivial_return: bool,
    require_top_level_return: bool,
    use_semantic_plan_head_prefix: bool,
    prefer_source_plan_compatibility: bool,
    use_semantic_slot_head_prefix: bool,
    enable_learned_expression_token_bias: bool,
    use_body_transition_head: bool,
    body_transition_head_blend: float,
    use_body_action_head: bool,
    body_action_head_blend: float,
    use_body_operand_head: bool,
    body_operand_head_blend: float,
    prefer_learned_prefix_decision_adequacy: bool,
    prefer_source_condition_adequacy: bool,
    require_source_condition_adequacy: bool,
    block_shallow_loop_identity_update: bool,
    enable_loop_progress_guard: bool,
    enable_expression_closure_guard: bool,
    enable_expression_value_guard: bool,
    require_binding_prefix_groups: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    if not selection.get("active"):
        return {
            "enabled": True,
            "active": False,
            "summary": dict_or_empty(selection.get("summary")),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        }, []

    budget = dict_or_empty(config.get("matched_budget"))
    data_cfg = dict_or_empty(config.get("data"))
    text_views = dict_or_empty(config.get("text_views"))
    source_vocab = dict_or_empty(vocab_payload.get("source_vocab"))
    target_vocab = dict_or_empty(vocab_payload.get("target_vocab"))
    target_mode = str(vocab_payload.get("target_mode") or get_path(config, ["body_structure_decoder", "target_mode"], "body_tokens"))
    source_text_style = checkpoint_source_text_style(config, vocab_payload)
    max_source = int(vocab_payload.get("max_source") or budget.get("max_source_tokens") or 96)
    max_target = int(vocab_payload.get("max_target") or budget.get("max_target_tokens") or 160)
    seed = int(selection.get("seed") or (int((budget.get("seeds") or [23])[0]) + 1))
    eval_rows = list(selection.get("eval_rows") or [])
    if max_row_override > 0:
        eval_rows = eval_rows[:max_row_override]

    if split_name == "broad_private_heldout":
        broad_cfg = dict_or_empty(data_cfg.get("broad_private_heldout_eval"))
        max_target_tokens = min(max_target, int(broad_cfg.get("max_target_tokens") or max_target))
        fanout_top_k = int(broad_cfg.get("fanout_top_k") or budget.get("fanout_top_k") or 1)
        pool_size = (
            max(
                fanout_top_k,
                int(dict_or_empty(get_path(config, ["body_structure_decoder", "static_coherence_ranker"], {})).get("candidate_pool_size") or 0),
            )
            if static_coherence_ranker_enabled(config)
            else fanout_top_k
        )
        grammar_top_k = int(broad_cfg.get("grammar_decode_top_k") or budget.get("grammar_decode_top_k") or 64)
        beam_width = int(broad_cfg.get("decode_beam_width") or budget.get("decode_beam_width") or 1)
        branch_factor = int(broad_cfg.get("decode_branching_factor") or budget.get("decode_branching_factor") or 1)
        decode_policy = str(
            broad_cfg.get("body_token_decode_policy")
            or budget.get("body_token_decode_policy")
            or budget.get("body_token_validity_policy")
            or "lightweight_python_v1"
        )
    else:
        max_target_tokens = max_target
        fanout_top_k = int(budget.get("fanout_top_k") or 1)
        pool_size = static_coherence_candidate_pool_size(config, budget)
        grammar_top_k = int(budget.get("grammar_decode_top_k") or 64)
        beam_width = int(budget.get("decode_beam_width") or 4)
        branch_factor = int(budget.get("decode_branching_factor") or 2)
        decode_policy = str(
            budget.get("body_token_decode_policy")
            or budget.get("body_token_validity_policy")
            or "lightweight_python_v1"
        )
    if int(max_target_tokens_override or 0) > 0:
        max_target_tokens = min(max_target_tokens, int(max_target_tokens_override))
    if output_top_k > 1:
        requested = max(1, int(output_top_k))
        fanout_top_k = max(fanout_top_k, requested)
        pool_size = max(pool_size, requested)
        # Candidate diversity is part of the learned-generator search contract.
        # Keep the expansion bounded so verifier cost remains predictable.
        beam_width = max(beam_width, min(24, requested * 2))
        branch_factor = max(branch_factor, min(8, requested))
        grammar_top_k = max(grammar_top_k, 128)

    source_texts = [
        strict_generator_decode_source_text(
            row,
            text_views.get("sts_on", []),
            source_text_style=source_text_style,
            source_vocab=source_vocab,
        )
        for row in eval_rows
    ]
    source_text_audit = strict_generator_source_text_audit(eval_rows, source_texts)
    source_condition_expectations = [source_condition_expectation_from_source_text(text) for text in source_texts]
    source_condition_expectation_summary = source_condition_expectations_summary(source_condition_expectations)
    source_rows = encode_many(source_texts, source_vocab, max_source)
    input_type_hints_by_row = [visible_parameter_type_hints_from_source_text(text) for text in source_texts]
    input_type_hint_counts = Counter(
        str(type_name)
        for hints in input_type_hints_by_row
        for type_name in dict_or_empty(hints).values()
        if str(type_name)
    )
    decode_started = time.perf_counter()
    decoded, decode_guard_diagnostics = generate_candidates_mlx(
        model,
        source_rows,
        target_vocab,
        max_target_tokens=max_target_tokens,
        fanout_top_k=pool_size,
        grammar_top_k=grammar_top_k,
        decode_beam_width=beam_width,
        decode_branching_factor=branch_factor,
        target_mode=target_mode,
        body_token_decode_policy=decode_policy,
        source_texts=source_texts,
        allowed_name_sets=[allowed_signature_names_for_task(row) for row in eval_rows],
        input_type_hints_by_row=input_type_hints_by_row,
        source_condition_expectations=source_condition_expectations,
        require_parameter_use=require_parameter_use,
        require_nontrivial_return=require_nontrivial_return,
        require_top_level_return=require_top_level_return,
        use_semantic_plan_head_prefix=use_semantic_plan_head_prefix,
        prefer_source_plan_compatibility=prefer_source_plan_compatibility,
        use_semantic_slot_head_prefix=use_semantic_slot_head_prefix,
        enable_learned_expression_token_bias=enable_learned_expression_token_bias,
        use_body_transition_head=use_body_transition_head,
        body_transition_head_blend=body_transition_head_blend,
        use_body_action_head=use_body_action_head,
        body_action_head_blend=body_action_head_blend,
        use_body_operand_head=use_body_operand_head,
        body_operand_head_blend=body_operand_head_blend,
        prefer_learned_prefix_decision_adequacy=prefer_learned_prefix_decision_adequacy,
        prefer_source_condition_adequacy=prefer_source_condition_adequacy or require_source_condition_adequacy,
        require_source_condition_adequacy=require_source_condition_adequacy,
        block_shallow_loop_identity_update=block_shallow_loop_identity_update,
        enable_loop_progress_guard=enable_loop_progress_guard,
        enable_expression_closure_guard=enable_expression_closure_guard,
        enable_expression_value_guard=enable_expression_value_guard,
        require_binding_prefix_groups=require_binding_prefix_groups,
        mx=mx,
    )
    for index, diag in enumerate(decode_guard_diagnostics):
        if index < len(eval_rows):
            diag["task_id"] = str(eval_rows[index].get("task_id") or "")
            diag["category"] = str(eval_rows[index].get("category") or "")
    decode_wall_ms = int((time.perf_counter() - decode_started) * 1000)
    decode_guard_rejections = decode_guard_rejection_summary(decode_guard_diagnostics)
    decode_starvation = decode_starvation_summary(decode_guard_diagnostics)
    source_plan_compatibility = source_plan_compatibility_summary(
        [dict_or_empty(item.get("source_plan_compatibility")) for item in decode_guard_diagnostics]
    )
    emitted_top_k = max(0, int(output_top_k or 0)) or fanout_top_k
    rows = token_candidate_rows_for_view(
        eval_rows,
        decoded,
        arm_id="transformer_control",
        substrate="mlx_high_level_transformer_token_decoder",
        phase="private_eval",
        view="sts_on",
        config=config,
        seed=seed,
        target_mode=target_mode,
        residual_context={},
        output_top_k=emitted_top_k,
    )
    rows = attach_decoded_proposal_metadata(rows, decoded)
    rows = stamp_mlx_rows(rows, checkpoint=selection.get("checkpoint"), split_name=split_name)
    source_condition_summary = source_condition_candidate_summary(rows)
    body_action_summary = body_action_trace_candidate_summary(rows)
    baselines = [baseline_candidate(task, arm_id="transformer_control", config=config, seed=seed) for task in eval_rows]
    all_for_verifier = baselines + rows
    verifier_started = time.perf_counter()
    private_eval = evaluate_private_candidates(eval_rows, all_for_verifier)
    verifier_wall_ms = int((time.perf_counter() - verifier_started) * 1000)
    verifier_label_summary = attach_private_verifier_labels(rows, private_eval)
    syntax = syntax_summary_from_rows(rows)
    repair = grammar_repair_summary(rows)
    static = static_coherence_summary(rows)
    body = body_structure_summary(rows)
    no_cheat = no_cheat_candidate_evidence(eval_rows, all_for_verifier)
    manifest_rows = rows
    if split_name == "family_disjoint":
        manifest_rows = family_disjoint_manifest_rows(all_for_verifier)
    elif split_name == "broad_private_heldout":
        manifest_rows = broad_private_heldout_manifest_rows(all_for_verifier)

    view_report = {
        "arm_id": "transformer_control",
        "view": "sts_on",
        "phase": "private_eval",
        "evaluation_split": split_name,
        "substrate": "mlx_high_level_transformer_token_decoder",
        "parameter_count": None,
        "dims": vocab_payload.get("dims"),
        "train": {
            "source": "loaded_mlx_checkpoint_only",
            "checkpoint": selection.get("checkpoint"),
            "public_training_rows": 0,
            "external_inference_calls": 0,
        },
        "specialist_route": dict_or_empty(selection.get("specialist_route")),
        "candidate_rows": len(rows),
        "candidate_tasks": len(eval_rows),
        "fanout_top_k": fanout_top_k,
        "candidate_pool_size": pool_size,
        "emitted_learned_top_k": emitted_top_k,
        "candidate_syntax": syntax,
        "grammar_repair": repair,
        "static_coherence": static,
        "body_structure": body,
        "ranker": "prompt_signature_static_coherence_then_sequence_log_probability",
            "decoder_constraints": {
                "beam_scheduler": "mlx_batched_active_beam_scheduler_v1",
                "top_token_selection_policy": "argpartition_bounded_topk_v1",
                "search_width_policy": "output_top_k_bounded_beam_branch_override_v1",
            "body_transition_head": {
                "enabled": bool(use_body_transition_head and hasattr(model, "body_transition_logits")),
                "requested": bool(use_body_transition_head),
                "blend": float(max(0.0, min(1.0, float(body_transition_head_blend or 0.0)))),
                "policy": "learned_prefix_conditioned_body_transition_logit_blend_v1",
                "score_semantics": (
                    "Blends a checkpoint-trained prefix-conditioned body-transition projection into "
                    "autoregressive token logits. It is model-derived and prompt/signature-visible only, "
                    "but it is not a renderer, fallback, tool call, template, or promotion claim; behavior "
                    "must still be earned through candidate integrity and private verifier replay."
                ),
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "candidate_generation_credit": 0,
            },
            "body_action_head": {
                "enabled": bool(use_body_action_head and hasattr(model, "body_action_logits")),
                "requested": bool(use_body_action_head),
                "blend": float(max(0.0, min(1.0, float(body_action_head_blend or 0.0)))),
                "roles": list(BODY_ACTION_ROLES),
                "policy": "learned_prefix_conditioned_body_action_probability_bias_v1",
                "score_semantics": (
                    "Biases raw token probabilities with a checkpoint-trained broad body-action role "
                    "head. It is model-derived and prompt/signature-visible only, but it is not a "
                    "renderer, fallback, tool call, template, or promotion claim; behavior must still "
                    "be earned through candidate integrity and private verifier replay."
                ),
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "candidate_generation_credit": 0,
            },
            "body_operand_head": {
                "enabled": bool(use_body_operand_head and hasattr(model, "body_operand_logits")),
                "requested": bool(use_body_operand_head),
                "blend": float(max(0.0, min(1.0, float(body_operand_head_blend or 0.0)))),
                "roles": list(BODY_OPERAND_ROLES),
                "policy": "learned_prefix_conditioned_body_operand_probability_bias_v1",
                "score_semantics": (
                    "Biases raw token probabilities with a checkpoint-trained operand/value binding "
                    "head. It uses only model state, generated prefix tokens, and visible signature "
                    "names to classify candidate tokens as visible parameter, local state, builtin, "
                    "method, operator, literal, or delimiter. It is not a renderer, fallback, tool "
                    "call, template, or promotion claim; behavior must still be earned through "
                    "candidate integrity and private verifier replay."
                ),
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "candidate_generation_credit": 0,
            },
            "requested_output_top_k": int(output_top_k or 0),
            "grammar_constrained_top_k": grammar_top_k,
            "decode_beam_width": beam_width,
            "decode_branching_factor": branch_factor,
            "candidate_pool_size": pool_size,
            "emitted_learned_top_k": emitted_top_k,
            "max_target_tokens": int(max_target_tokens),
            "checkpoint_max_target_tokens": int(max_target),
            "max_target_tokens_override": int(max_target_tokens_override or 0),
            "body_token_decode_policy": decode_policy,
            "block_shallow_loop_identity_update": {
                "enabled": bool(block_shallow_loop_identity_update),
                "policy": "task_blind_direct_loop_identity_append_block_v1",
                "score_semantics": (
                    "Optionally blocks accumulator.append(loop_var)/add(loop_var) continuations for "
                    "non-identity learned loop plans. It is an explicit decode experiment that uses only "
                    "generated prefix tokens and prompt-visible plan slots; it does not render a correct "
                    "body, inspect tests/solutions, use public data, or grant candidate-generation credit."
                ),
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "candidate_generation_credit": 0,
            },
            "loop_progress_guard": {
                "enabled": bool(enable_loop_progress_guard),
                "policy": "task_blind_learned_plan_loop_progress_guard_v1",
                "score_semantics": (
                    "When a model-generated plan prefix opens a loop and the beam is at an empty "
                    "loop-body line before any update call, this guard prioritizes an accumulator "
                    "update start before loop exit/finalizer. If the model already emitted a mutation "
                    "call prefix, it only completes that generated statement so the beam can dedent "
                    "and search for a local return. It uses only generated prefix state, visible "
                    "signature names, and task-blind grammar state; it does not inspect tests, "
                    "solutions, public data, or render a full algorithm. Runs with this guard should "
                    "be reported separately from unconstrained learned-generation evidence."
                ),
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "candidate_generation_credit": 0,
            },
            "expression_closure_guard": {
                "enabled": bool(enable_expression_closure_guard),
                "policy": "task_blind_expression_block_closure_guard_v1",
                "score_semantics": (
                    "Prioritizes delimiter closure, colon/newline statement termination, safe loop "
                    "dedent, and top-level local-return start from generated prefix syntax only. It "
                    "does not choose an algorithm, render a body, inspect tests/solutions, use public "
                    "data, or grant candidate-generation credit. Runs with this guard must be reported "
                    "separately from unconstrained learned-generation evidence."
                ),
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "candidate_generation_credit": 0,
            },
            "expression_value_guard": {
                "enabled": bool(enable_expression_value_guard),
                "policy": "task_blind_expression_value_hygiene_guard_v1",
                "score_semantics": (
                    "Rejects generated-prefix-only expression value pathologies such as empty "
                    "update-call arguments, direct set literals in mutation calls, and bare "
                    "builtin function/class objects closed as values, plus comparison/boolean "
                    "expressions used as the object tested by isinstance. It does not choose an "
                    "algorithm, render a body, inspect tests/solutions, use public data, or grant "
                    "candidate-generation credit."
                ),
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "candidate_generation_credit": 0,
            },
            "binding_prefix_groups": {
                "enabled": bool(require_binding_prefix_groups),
                "policy": "task_blind_learned_operand_binding_prefix_groups_v1",
                "score_semantics": (
                    "Ablation-only prefix coverage guard requiring loop/branch, update, and result "
                    "operand-binding slots before body-token decoding when SLOT:BIND_* tokens exist. "
                    "It uses only model-generated prefix tokens and visible signature legality, does not "
                    "inspect tests/solutions/public data, does not render code, and grants zero "
                    "candidate-generation credit."
                ),
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "candidate_generation_credit": 0,
            },
            "signature_name_mask": {
                "enabled": True,
                "policy": "visible_signature_other_extra_mask_v1",
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
            },
            "signature_parameter_exploration": {
                "enabled_when_require_parameter_use": True,
                "policy": "visible_signature_parameter_beam_exploration_v1",
                "score_semantics": (
                    "When the strict decode guard requires parameter use, visible callable signature "
                    "parameter tokens are admitted as additional legal beam branches until one has "
                    "appeared. This does not change model probabilities, render a body, inspect tests "
                    "or solutions, or add fallback returns."
                ),
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
            },
            "source_condition_adequacy": {
                "enabled": bool(prefer_source_condition_adequacy or require_source_condition_adequacy),
                "policy": "prompt_visible_source_condition_adequacy_v2",
                "prefer_source_condition_adequacy": bool(prefer_source_condition_adequacy),
                "require_source_condition_adequacy": bool(require_source_condition_adequacy),
                "expectations": source_condition_expectation_summary,
                "candidate_summary": source_condition_summary,
                "score_semantics": (
                    "Uses only prompt/signature source text to identify visible empty/default-handling "
                    "requirements and broad prompt-operation hints, then prefers or optionally requires "
                    "decoded bodies with corresponding guards or operation evidence. It does not inspect "
                    "tests, solutions, public benchmark payloads, teacher output, or candidate templates."
                ),
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "candidate_generation_credit": 0,
            },
            "source_plan_compatibility": {
                "enabled": bool(prefer_source_plan_compatibility),
                "policy": "prompt_visible_source_plan_compatibility_rerank_v1",
                "summary": source_plan_compatibility,
                "score_semantics": (
                    "Optional first-plan-token reranking using only prompt-visible intent, operation, "
                    "and type tags from the same strict source text that the generator sees. It does "
                    "not render code, inspect tests/solutions/public payloads, call tools, or grant "
                    "learned-generation credit."
                ),
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "candidate_generation_credit": 0,
            },
            "body_action_trace": {
                "enabled": bool(body_action_summary.get("enabled_candidate_rows")),
                "policy": "strict_generator_body_action_trace_summary_v1",
                "candidate_summary": body_action_summary,
                "score_semantics": (
                    "Task-blind AST operation trace over decoded bodies. It uses only prompt/signature "
                    "visible semantic-plan expectations and the candidate body after generation to classify "
                    "statement-order, loop-update, branch/finalizer, and shallow-identity failures. It is "
                    "audit and training-residual evidence only; it does not inspect tests, solutions, public "
                    "benchmark payloads, or teacher output, and it gives zero learned-generation credit."
                ),
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "candidate_generation_credit": 0,
            },
            "target_mode": target_mode,
            "source_text_policy": STRICT_GENERATOR_SOURCE_TEXT_POLICY,
            "source_text_style": source_text_style,
        "source_text_audit": source_text_audit,
        "post_decode_repair_policy": "deterministic_python_body_repair_v1",
        "specialist_route": dict_or_empty(selection.get("specialist_route")),
        "decode_guard": {
                "require_parameter_use": require_parameter_use,
                "require_nontrivial_return": require_nontrivial_return,
                "require_top_level_return": require_top_level_return,
                "use_semantic_plan_head_prefix": bool(use_semantic_plan_head_prefix),
                "use_semantic_slot_head_prefix": bool(use_semantic_slot_head_prefix),
                "enable_learned_expression_token_bias": {
                    "enabled": bool(enable_learned_expression_token_bias),
                    "policy": "model_generated_expression_slot_soft_body_token_bias_v1",
                    "score_semantics": (
                        "Softly rescales raw body-token probabilities in expression contexts using only "
                        "model-generated expression prefix slots, generated prefix syntax, and visible "
                        "signature names. It does not force complete expressions, render code, inspect "
                        "tests/solutions/public payloads, or grant learned-generation promotion credit."
                    ),
                    "candidate_generation_credit": 0,
                    "uses_eval_tests_or_solutions": False,
                    "uses_public_data": False,
                },
                "visible_parameter_type_hints": {
                    "enabled": True,
                    "policy": "prompt_signature_visible_type_shape_to_static_decode_hint_v1",
                    "hinted_task_count": sum(1 for hints in input_type_hints_by_row if hints),
                    "type_counts": dict(sorted(input_type_hint_counts.items())),
                    "uses_only_prompt_signature": True,
                    "uses_eval_tests_or_solutions": False,
                    "uses_public_data": False,
                },
                "rejection_summary": decode_guard_rejections,
                "starvation_summary": decode_starvation,
                "uses_only_prompt_signature": True,
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "fallback_returns_allowed": False,
            },
        },
        "backend": {"framework": "mlx", "device": str(mx.default_device())},
        "wall_time_ms_before_verifier": decode_wall_ms,
    }
    summary = arm_summary({"sts_on": view_report}, private_eval, verifier_wall_ms)
    summary.update(
        {
            "split": split_name,
            "active": True,
            "eval_rows": len(eval_rows),
            "candidate_rows": len(rows),
            "generated_candidate_rows": len(rows),
            "manifest_candidate_rows": len(manifest_rows),
            "decode_guard_rejections": decode_guard_rejections,
            "decode_starvation": decode_starvation,
            "decode_wall_time_ms": decode_wall_ms,
            "decode_rows_per_second": round(len(eval_rows) / max(decode_wall_ms / 1000.0, 1e-9), 6),
            "source_condition_adequacy": source_condition_summary,
            "source_plan_compatibility": source_plan_compatibility,
            "body_action_trace": body_action_summary,
            "private_verifier": {
                "trained_passed": private_eval.get("trained_passed"),
                "trained_rank1_passed": private_eval.get("trained_rank1_passed"),
                "trained_pass_if_any_passed": private_eval.get("trained_pass_if_any_passed"),
                "eval_task_count": private_eval.get("eval_task_count"),
                "residual_count": private_eval.get("residual_count"),
                "correctness_labels": private_eval.get("correctness_labels"),
                "candidate_label_summary": verifier_label_summary,
                "static_cache_hit_count": get_path(private_eval, ["private_verification", "static_cache_hit_count"], 0),
                "sandbox_cache_hit_count": get_path(private_eval, ["private_verification", "sandbox_cache_hit_count"], 0),
                "test_harness_cache_hit_count": get_path(
                    private_eval,
                    ["private_verification", "test_harness_cache_hit_count"],
                    0,
                ),
                "verifier_cache_warmup": private_eval.get("verifier_cache_warmup"),
                "wall_time_ms": verifier_wall_ms,
            },
            "candidate_syntax": syntax,
            "grammar_repair": repair,
            "static_coherence": static,
            "body_structure": body,
            "candidate_schema": candidate_schema_summary(manifest_rows),
            "no_cheat_evidence": dict_or_empty(no_cheat.get("summary")),
            "source_text_audit": source_text_audit,
            "split_overlap_audit": get_path(selection, ["split_audit", "overlap"], {}),
            "zero_train_eval_overlap": all(
                int(get_path(selection, ["split_audit", "overlap", key], 0) or 0) == 0
                for key in dict_or_empty(get_path(selection, ["split_audit", "overlap"], {}))
                if key.endswith("_count")
            ),
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "teacher_used": False,
        }
    )
    return {
        "policy": f"project_theseus_mlx_{split_name}_decode_eval_v1",
        "enabled": True,
        "active": True,
        "created_utc": now(),
        "summary": summary,
        "selection": selection_summary(selection),
        "view": view_report,
        "private_verifier": private_eval,
        "private_verifier_correctness_labels": verifier_label_summary,
        "no_cheat_evidence": no_cheat,
        "score_semantics": (
            f"{split_name} private eval for the MLX strict-generator checkpoint. Split labels and solution bodies "
            "are used only for train/eval separation and overlap audits; generation uses prompt/signature text."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }, manifest_rows


def strict_mlx_decode_topk_selection_receipt() -> dict[str, Any]:
    return {
        "policy": "argpartition_bounded_topk_v1",
        "applies_to": [
            "body_token_legality_window",
            "fallback_legality_window",
            "learned_plan_prefix_selection",
            "learned_slot_prefix_selection",
        ],
        "full_vocabulary_sort_in_hot_loop": False,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
        "score_semantics": (
            "Runtime-economics receipt only. Bounded top-k token selection changes how candidate token "
            "windows are ranked, not what information generation can see and not whether a candidate "
            "counts as learned generation."
        ),
    }



def generate_candidates_mlx(
    model: Any,
    source_rows: list[list[int]],
    target_vocab: dict[str, int],
    *,
    max_target_tokens: int,
    fanout_top_k: int,
    grammar_top_k: int,
    decode_beam_width: int,
    decode_branching_factor: int,
    target_mode: str,
    body_token_decode_policy: str,
    source_texts: list[str] | None,
    allowed_name_sets: list[set[str]] | None,
    input_type_hints_by_row: list[dict[str, str]] | None,
    source_condition_expectations: list[dict[str, Any]] | None,
    require_parameter_use: bool,
    require_nontrivial_return: bool,
    require_top_level_return: bool,
    use_semantic_plan_head_prefix: bool,
    prefer_source_plan_compatibility: bool,
    use_semantic_slot_head_prefix: bool,
    enable_learned_expression_token_bias: bool,
    use_body_transition_head: bool,
    body_transition_head_blend: float,
    use_body_action_head: bool,
    body_action_head_blend: float,
    use_body_operand_head: bool,
    body_operand_head_blend: float,
    prefer_learned_prefix_decision_adequacy: bool,
    prefer_source_condition_adequacy: bool,
    require_source_condition_adequacy: bool,
    block_shallow_loop_identity_update: bool,
    enable_loop_progress_guard: bool,
    enable_expression_closure_guard: bool,
    enable_expression_value_guard: bool,
    require_binding_prefix_groups: bool,
    mx: Any,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    inverse = {idx: tok for tok, idx in target_vocab.items()}
    token_to_id = {str(tok): int(idx) for tok, idx in target_vocab.items()}
    bos = int(target_vocab["<bos>"])
    eos = int(target_vocab["<eos>"])
    model.eval()
    beam_width = max(int(fanout_top_k or 1), int(decode_beam_width or 1))
    branch_factor = max(1, int(decode_branching_factor or 1))
    plan_prefix_choices_by_row, source_plan_compatibility_by_row = semantic_plan_head_prefix_choices(
        model,
        source_rows,
        inverse,
        max_choices=branch_factor,
        target_mode=target_mode,
        enabled=bool(use_semantic_plan_head_prefix),
        source_texts=source_texts,
        prefer_source_plan_compatibility=prefer_source_plan_compatibility,
        mx=mx,
    )
    slot_prefix_probs_by_row = semantic_slot_head_prefix_probs(
        model,
        source_rows,
        inverse,
        target_mode=target_mode,
        enabled=bool(use_semantic_slot_head_prefix),
        mx=mx,
    )
    states: list[dict[str, Any]] = []
    for i, source_row in enumerate(source_rows):
        plan_choices = plan_prefix_choices_by_row[i] if i < len(plan_prefix_choices_by_row) else None
        slot_probs = slot_prefix_probs_by_row[i] if i < len(slot_prefix_probs_by_row) else None
        states.append(
            {
                "source_row": list(source_row),
                "allowed_names": allowed_name_sets[i] if allowed_name_sets and i < len(allowed_name_sets) else None,
                "input_type_hints": input_type_hints_by_row[i] if input_type_hints_by_row and i < len(input_type_hints_by_row) else {},
                "source_condition_expectation": (
                    source_condition_expectations[i]
                    if source_condition_expectations and i < len(source_condition_expectations)
                    else {"enabled": False, "required_features": []}
                ),
                "plan_prefix_choices": plan_choices,
                "plan_prefix_source": (
                    "semantic_plan_head_source_plan_compatibility"
                    if plan_choices and bool((source_plan_compatibility_by_row[i] if i < len(source_plan_compatibility_by_row) else {}).get("enabled"))
                    else "semantic_plan_head"
                    if plan_choices
                    else "decoder_first_token"
                ),
                "source_plan_compatibility": (
                    source_plan_compatibility_by_row[i]
                    if i < len(source_plan_compatibility_by_row)
                    else {"enabled": False, "policy": "prompt_visible_source_plan_compatibility_v1"}
                ),
                "slot_prefix_probs": slot_probs,
                "slot_prefix_source": "semantic_slot_head" if slot_probs else "decoder_autoregressive_slots",
                "beams": [{"generated": [bos], "logprob": 0.0, "done": False}],
                "completed": [],
                "guard_rejection_counts": Counter(),
                "guard_rejection_examples": [],
                "stopped": False,
            }
        )

    for _step in range(max(1, int(max_target_tokens) - 1)):
        refs_by_len: dict[int, list[tuple[int, dict[str, Any], list[int]]]] = {}
        expanded_by_task: list[list[dict[str, Any]]] = [[] for _state in states]
        active_count = 0
        for task_idx, state in enumerate(states):
            if bool(state.get("stopped")):
                continue
            for beam in list(state.get("beams") or []):
                generated = list(beam.get("generated") or [])
                if bool(beam.get("done")):
                    expanded_by_task[task_idx].append(beam)
                    continue
                active_count += 1
                refs_by_len.setdefault(len(generated), []).append((task_idx, beam, generated))
        if active_count == 0:
            break

        for generated_len, refs in sorted(refs_by_len.items()):
            src = mx.array([states[task_idx]["source_row"] for task_idx, _beam, _generated in refs], dtype=mx.int32)
            tgt = mx.array([generated for _task_idx, _beam, generated in refs], dtype=mx.int32)
            logits = model(src, tgt)[:, -1, :]
            transition_blend = max(0.0, min(1.0, float(body_transition_head_blend or 0.0)))
            if bool(use_body_transition_head) and transition_blend > 0.0 and hasattr(model, "body_transition_logits"):
                transition_logits = model.body_transition_logits(src, tgt)[:, -1, :]
                logits = (logits * (1.0 - transition_blend)) + (transition_logits * transition_blend)
            action_prob_rows = None
            action_blend = max(0.0, min(1.0, float(body_action_head_blend or 0.0)))
            if bool(use_body_action_head) and action_blend > 0.0 and hasattr(model, "body_action_logits"):
                action_logits = model.body_action_logits(src, tgt)[:, -1, :]
                action_probs = mx.softmax(action_logits, axis=-1)
                mx.eval(action_probs)
                action_prob_rows = np.asarray(action_probs, dtype=np.float64)
            operand_prob_rows = None
            operand_blend = max(0.0, min(1.0, float(body_operand_head_blend or 0.0)))
            if bool(use_body_operand_head) and operand_blend > 0.0 and hasattr(model, "body_operand_logits"):
                operand_logits = model.body_operand_logits(src, tgt)[:, -1, :]
                operand_probs = mx.softmax(operand_logits, axis=-1)
                mx.eval(operand_probs)
                operand_prob_rows = np.asarray(operand_probs, dtype=np.float64)
            probs = mx.softmax(logits, axis=-1)
            mx.eval(probs)
            prob_rows = np.asarray(probs, dtype=np.float64)
            for row_index, (task_idx, beam, generated) in enumerate(refs):
                state = states[task_idx]
                allowed_names = state.get("allowed_names")
                prob_row = prob_rows[row_index]
                if action_prob_rows is not None:
                    prob_row = body_action_biased_probability_row(
                        prob_row,
                        action_prob_rows[row_index],
                        inverse,
                        blend=action_blend,
                    )
                if operand_prob_rows is not None:
                    prob_row = body_operand_biased_probability_row(
                        prob_row,
                        operand_prob_rows[row_index],
                        inverse,
                        generated,
                        allowed_names=set(allowed_names or []),
                        blend=operand_blend,
                    )
                choices = mlx_token_choices(
                    prob_row,
                    inverse,
                    token_to_id,
                    generated,
                    eos_id=eos,
                    grammar_top_k=grammar_top_k,
                    max_choices=branch_factor,
                    token_policy=body_token_decode_policy,
                    target_mode=target_mode,
                    allowed_names=allowed_names,
                    input_type_hints=dict_or_empty(state.get("input_type_hints")),
                    source_condition_expectation=dict_or_empty(state.get("source_condition_expectation")),
                    plan_prefix_choices=state.get("plan_prefix_choices"),
                    slot_prefix_probs=state.get("slot_prefix_probs"),
                    enable_learned_expression_token_bias=enable_learned_expression_token_bias,
                    require_parameter_use=require_parameter_use,
                    require_nontrivial_return=require_nontrivial_return,
                    require_top_level_return=require_top_level_return,
                    prefer_learned_prefix_decision_adequacy=prefer_learned_prefix_decision_adequacy,
                    prefer_source_condition_adequacy=prefer_source_condition_adequacy or require_source_condition_adequacy,
                    block_shallow_loop_identity_update=block_shallow_loop_identity_update,
                    enable_loop_progress_guard=enable_loop_progress_guard,
                    enable_expression_closure_guard=enable_expression_closure_guard,
                    enable_expression_value_guard=enable_expression_value_guard,
                    require_binding_prefix_groups=require_binding_prefix_groups,
                )
                for next_id, prob in choices:
                    next_generated = generated + [int(next_id)]
                    next_logprob = float(beam.get("logprob") or 0.0) + math.log(max(float(prob), 1e-9))
                    expanded_by_task[task_idx].append(
                        {
                            "generated": next_generated,
                            "logprob": next_logprob,
                            "done": int(next_id) == eos,
                        }
                    )
                    prefix_tokens = [inverse.get(idx, "<unk>") for idx in next_generated[1:]]
                    if (
                        body_like_target_mode(target_mode)
                        and int(next_id) != eos
                        and completion_ready(
                            prefix_tokens,
                            target_mode=target_mode,
                            allowed_names=allowed_names,
                            require_parameter_use=require_parameter_use,
                            require_nontrivial_return=require_nontrivial_return,
                            require_top_level_return=require_top_level_return,
                        )
                    ):
                        state["completed"].append(
                            {
                                "generated": next_generated + [eos],
                                "logprob": next_logprob,
                                "done": True,
                                "completion_source": "mlx_batched_syntax_complete_prefix",
                            }
                        )

        any_active_after_prune = False
        for task_idx, state in enumerate(states):
            if bool(state.get("stopped")):
                continue
            expanded = expanded_by_task[task_idx]
            if not expanded:
                state["stopped"] = True
                continue
            expanded.sort(
                key=lambda row: source_condition_decode_beam_sort_key(
                    row,
                    inverse,
                    target_mode=target_mode,
                    expectation=dict_or_empty(state.get("source_condition_expectation")),
                    prefer=prefer_source_condition_adequacy or require_source_condition_adequacy,
                ),
                reverse=True,
            )
            kept = expanded[:beam_width]
            allowed_names = state.get("allowed_names")
            if require_parameter_use and allowed_names and not any(
                beam_mentions_allowed_parameter(row, inverse, allowed_names=allowed_names, target_mode=target_mode)
                for row in kept
            ):
                parameter_rows = [
                    row
                    for row in expanded
                    if beam_mentions_allowed_parameter(row, inverse, allowed_names=allowed_names, target_mode=target_mode)
                ]
                if parameter_rows:
                    if len(kept) >= beam_width and kept:
                        kept[-1] = parameter_rows[0]
                    else:
                        kept.append(parameter_rows[0])
            state["beams"] = kept
            if any(not bool(row.get("done")) for row in state["beams"]):
                any_active_after_prune = True
        if not any_active_after_prune:
            break

    all_rows: list[list[dict[str, Any]]] = []
    diagnostics: list[dict[str, Any]] = []
    for state in states:
        seen: set[str] = set()
        task_rows: list[dict[str, Any]] = []
        ranked_beams = list(state.get("beams") or []) + list(state.get("completed") or [])
        allowed_names = state.get("allowed_names")
        ranked_beams.sort(
            key=lambda row: learned_prefix_decision_final_decode_beam_sort_key(
                row,
                inverse,
                target_mode=target_mode,
                expectation=dict_or_empty(state.get("source_condition_expectation")),
                prefer_source_condition=prefer_source_condition_adequacy or require_source_condition_adequacy,
                prefer_learned_prefix=prefer_learned_prefix_decision_adequacy,
            ),
            reverse=True,
        )
        for beam in ranked_beams:
            generated = list(beam.get("generated") or [])
            decoded_tokens = [inverse.get(idx, "<unk>") for idx in generated[1:]]
            body_token_stream = body_tokens_for_target_mode(decoded_tokens, target_mode=target_mode)
            body = decode_body_tokens(body_token_stream) if body_like_target_mode(target_mode) else ""
            guard = {
                "passed": True,
                "failures": [],
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
            }
            if body_like_target_mode(target_mode) and (require_parameter_use or require_nontrivial_return or require_top_level_return):
                guard = decode_static_guard(
                    body,
                    allowed_names=allowed_names,
                    require_parameter_use=require_parameter_use,
                    require_nontrivial_return=require_nontrivial_return,
                    require_top_level_return=require_top_level_return,
                )
                if not bool(guard.get("passed")):
                    failures = [str(item) for item in guard.get("failures") or ["unknown_guard_failure"]]
                    state["guard_rejection_counts"].update(failures)
                    examples = state.get("guard_rejection_examples")
                    if isinstance(examples, list) and len(examples) < 3:
                        examples.append(
                            {
                                "failures": failures,
                                "decoded_token_count": len(generated) - 1,
                                "decoded_token_tail": decoded_tokens[-80:],
                                "body_preview": body[:240],
                                "dependency": guard.get("dependency"),
                            }
                    )
                    continue
            condition_adequacy = source_condition_adequacy_for_body(
                body,
                dict_or_empty(state.get("source_condition_expectation")),
                allowed_names=allowed_names,
            )
            if require_source_condition_adequacy and not bool(condition_adequacy.get("adequate")):
                failures = [f"source_condition_{item}" for item in list(condition_adequacy.get("missing_features") or ["inadequate"])]
                state["guard_rejection_counts"].update(failures)
                examples = state.get("guard_rejection_examples")
                if isinstance(examples, list) and len(examples) < 3:
                    examples.append(
                        {
                            "failures": failures,
                            "decoded_token_count": len(generated) - 1,
                            "decoded_token_tail": decoded_tokens[-80:],
                            "body_preview": body[:240],
                            "source_condition_adequacy": condition_adequacy,
                        }
                    )
                continue
            expression_value_quality = expression_value_quality_summary(body, allowed_names=allowed_names)
            if enable_expression_value_guard and int(expression_value_quality.get("invalid_expression_value_count") or 0) > 0:
                state["guard_rejection_counts"].update(["invalid_expression_value"])
                examples = state.get("guard_rejection_examples")
                if isinstance(examples, list) and len(examples) < 3:
                    examples.append(
                        {
                            "failures": ["invalid_expression_value"],
                            "decoded_token_count": len(generated) - 1,
                            "decoded_token_tail": decoded_tokens[-80:],
                            "body_preview": body[:240],
                            "expression_value_quality": expression_value_quality,
                        }
                    )
                continue
            body_sha = (
                stable_hash(normalize_body_text(body))
                if body_like_target_mode(target_mode)
                else stable_hash(" ".join(decoded_tokens))
            )
            if body_sha in seen:
                continue
            seen.add(body_sha)
            task_rows.append(
                {
                    "body": body,
                    "decoded_tokens": decoded_tokens,
                    "learned_plan_prefix": learned_plan_prefix_metadata(decoded_tokens, target_mode=target_mode),
                    "loop_plan_adequacy": learned_prefix_loop_plan_adequacy_metadata(
                        decoded_tokens,
                        body,
                        target_mode=target_mode,
                        allowed_names=allowed_names,
                        enabled=prefer_learned_prefix_decision_adequacy,
                    ),
                    "body_action_trace": learned_prefix_body_action_trace_metadata(
                        decoded_tokens,
                        body,
                        target_mode=target_mode,
                        allowed_names=allowed_names,
                        enabled=prefer_learned_prefix_decision_adequacy,
                    ),
                    "prefix_guided_rank_score": learned_prefix_decision_rank_score(
                        decoded_tokens,
                        body,
                        target_mode=target_mode,
                        allowed_names=allowed_names,
                        enabled=prefer_learned_prefix_decision_adequacy,
                    ),
                    "rank_score": round(float(beam.get("logprob") or 0.0) / max(1, len(generated) - 1), 8),
                    "decoded_token_count": len(generated) - 1,
                    "decoded_token_sha256": stable_hash(" ".join(str(idx) for idx in generated)),
                    "beam_source": str(beam.get("completion_source") or "mlx_batched_grammar_constrained_token_beam"),
                    "plan_prefix_source": str(state.get("plan_prefix_source") or "decoder_first_token"),
                    "source_plan_compatibility": dict_or_empty(state.get("source_plan_compatibility")),
                    "decode_static_guard": guard,
                    "expression_value_quality": expression_value_quality,
                    "source_condition_expectation": dict_or_empty(state.get("source_condition_expectation")),
                    "source_condition_adequacy": condition_adequacy,
                }
            )
            if len(task_rows) >= int(fanout_top_k or 1):
                break
        if not task_rows and (require_parameter_use or require_nontrivial_return or require_top_level_return):
            all_rows.append([])
            rejection_counts = state.get("guard_rejection_counts")
            diagnostics.append(
                {
                    "accepted_candidate_rows": 0,
                    "guard_rejection_counts": dict(rejection_counts) if isinstance(rejection_counts, Counter) else {},
                    "guard_rejection_examples": list(state.get("guard_rejection_examples") or []),
                    "source_condition_expectation": dict_or_empty(state.get("source_condition_expectation")),
                    "source_plan_compatibility": dict_or_empty(state.get("source_plan_compatibility")),
                    "decode_starvation": decode_starvation_for_state(
                        state,
                        inverse,
                        target_mode=target_mode,
                        allowed_names=allowed_names,
                    ),
                }
            )
            continue
        if not task_rows:
            generated = [bos, eos]
            task_rows.append(
                {
                    "body": "",
                    "decoded_tokens": ["<eos>"],
                    "rank_score": -999.0,
                    "decoded_token_count": 1,
                    "decoded_token_sha256": stable_hash(" ".join(str(idx) for idx in generated)),
                    "beam_source": "mlx_direct_grammar_constrained_token_beam_empty",
                }
            )
        all_rows.append(task_rows)
        rejection_counts = state.get("guard_rejection_counts")
        diagnostics.append(
            {
                "accepted_candidate_rows": len(task_rows),
                "guard_rejection_counts": dict(rejection_counts) if isinstance(rejection_counts, Counter) else {},
                "guard_rejection_examples": list(state.get("guard_rejection_examples") or []),
                "source_condition_expectation": dict_or_empty(state.get("source_condition_expectation")),
                "decode_starvation": decode_starvation_for_state(
                    state,
                    inverse,
                    target_mode=target_mode,
                    allowed_names=allowed_names,
                ),
            }
        )
    return all_rows, diagnostics


def decode_starvation_for_state(
    state: dict[str, Any],
    inverse: dict[int, str],
    *,
    target_mode: str,
    allowed_names: set[str] | None,
) -> dict[str, Any]:
    beams = list(state.get("beams") or [])
    completed = list(state.get("completed") or [])
    ranked = sorted(
        [*beams, *completed],
        key=lambda row: float(row.get("logprob") or -1e9),
        reverse=True,
    )
    examples: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    for beam in ranked[:3]:
        generated = list(beam.get("generated") or [])
        decoded_tokens = [inverse.get(int(idx), "<unk>") for idx in generated[1:]]
        body_tokens = body_tokens_for_target_mode(decoded_tokens, target_mode=target_mode)
        body = decode_body_tokens(body_tokens) if body_like_target_mode(target_mode) else ""
        prefix_meta = learned_plan_prefix_metadata(decoded_tokens, target_mode=target_mode)
        prefix_tokens = [str(tok) for tok in list(prefix_meta.get("learned_plan_prefix_tokens") or [])]
        loop_expectation = learned_prefix_loop_expectation_from_tokens(prefix_tokens)
        loop_state = loop_decode_prefix_state(body_tokens, body, loop_expectation, allowed_names=allowed_names)
        for label in list(loop_state.get("state_labels") or []):
            state_counts[str(label)] += 1
        examples.append(
            {
                "decoded_token_count": max(0, len(decoded_tokens)),
                "done": bool(beam.get("done")),
                "completion_source": str(beam.get("completion_source") or ""),
                "avg_logprob": round(float(beam.get("logprob") or 0.0) / max(1, len(decoded_tokens)), 8),
                "decoded_token_tail": decoded_tokens[-60:],
                "body_preview": body[:320],
                "learned_plan_prefix": prefix_meta,
                "source_plan_compatibility": dict_or_empty(state.get("source_plan_compatibility")),
                "loop_prefix_state": loop_state,
            }
        )
    return {
        "policy": "strict_generator_decode_starvation_diagnostic_v1",
        "stopped": bool(state.get("stopped")),
        "beam_count": len(beams),
        "completed_beam_count": len(completed),
        "top_beam_examples": examples,
        "top_beam_state_counts": dict(sorted(state_counts.items())),
        "guard_rejection_counts": dict(state.get("guard_rejection_counts") or {}),
        "score_semantics": (
            "Task-blind diagnostic for zero-candidate strict MLX decode runs. It summarizes surviving "
            "beam prefixes and loop/update/finalizer state so failed private replays are diagnosable. "
            "It does not alter generation, render code, inspect tests/solutions, use public data, or "
            "grant candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def loop_decode_prefix_state(
    body_tokens: list[str],
    body: str,
    expectation: dict[str, Any],
    *,
    allowed_names: set[str] | None,
) -> dict[str, Any]:
    line_values = token_values(current_line_tokens(body_tokens))
    source_arg = loop_plan_source_arg(expectation, allowed_names=allowed_names)
    accumulator = loop_plan_accumulator_name(expectation, body_text=body, inverse={}, arr=[])
    inside_loop = loop_plan_inside_loop(body_tokens)
    has_initializer = loop_plan_has_initializer(body)
    has_loop = loop_plan_has_loop_over_source(body, source_arg=source_arg)
    has_update = loop_plan_has_update_call(body)
    has_return = loop_plan_has_local_return(body, accumulator=accumulator)
    labels: list[str] = []
    if inside_loop:
        labels.append("inside_loop")
    if inside_loop and not has_update:
        labels.append("inside_loop_without_update")
    if inside_loop and not line_values:
        labels.append("inside_loop_at_blank_line")
    if not has_return:
        labels.append("missing_local_return")
    if line_values and line_values[0] in {"continue", "break", "return"}:
        labels.append(f"current_line_starts_{line_values[0]}")
    return {
        "enabled": bool(expectation.get("enabled")),
        "plan": expectation.get("plan"),
        "current_line_values": line_values,
        "inside_loop": inside_loop,
        "has_initializer": has_initializer,
        "has_loop_over_source": has_loop,
        "has_update_call": has_update,
        "has_local_return": has_return,
        "source_arg": source_arg,
        "accumulator": accumulator,
        "state_labels": labels,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def decode_guard_rejection_summary(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    accepted = 0
    rejected_tasks = 0
    for diag in diagnostics:
        accepted += int(diag.get("accepted_candidate_rows") or 0)
        row_counts = Counter({str(key): int(value) for key, value in dict_or_empty(diag.get("guard_rejection_counts")).items()})
        if row_counts:
            rejected_tasks += 1
            counts.update(row_counts)
        for example in list(diag.get("guard_rejection_examples") or []):
            if len(examples) >= 8:
                break
            item = dict_or_empty(example)
            item["task_id"] = str(diag.get("task_id") or "")
            item["category"] = str(diag.get("category") or "")
            examples.append(item)
    return {
        "policy": "strict_generator_decode_static_guard_rejection_summary_v1",
        "task_count": len(diagnostics),
        "task_count_with_guard_rejections": rejected_tasks,
        "accepted_candidate_rows": accepted,
        "guard_rejection_counts": dict(sorted(counts.items())),
        "guard_rejection_examples": examples,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
    }


def decode_starvation_summary(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    zero_candidate_count = 0
    zero_without_guard_count = 0
    stopped_count = 0
    no_beam_count = 0
    state_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for diag in diagnostics:
        accepted = int(diag.get("accepted_candidate_rows") or 0)
        starvation = dict_or_empty(diag.get("decode_starvation"))
        if accepted > 0:
            continue
        zero_candidate_count += 1
        if not dict_or_empty(starvation.get("guard_rejection_counts")):
            zero_without_guard_count += 1
        if bool(starvation.get("stopped")):
            stopped_count += 1
        if int(starvation.get("beam_count") or 0) <= 0 and int(starvation.get("completed_beam_count") or 0) <= 0:
            no_beam_count += 1
        for label, value in dict_or_empty(starvation.get("top_beam_state_counts")).items():
            try:
                state_counts[str(label)] += int(value or 0)
            except (TypeError, ValueError):
                continue
        if len(examples) < 8:
            item = {
                "task_id": str(diag.get("task_id") or ""),
                "category": str(diag.get("category") or ""),
                "stopped": bool(starvation.get("stopped")),
                "beam_count": int(starvation.get("beam_count") or 0),
                "completed_beam_count": int(starvation.get("completed_beam_count") or 0),
                "top_beam_state_counts": dict_or_empty(starvation.get("top_beam_state_counts")),
                "top_beam_examples": list(starvation.get("top_beam_examples") or [])[:2],
            }
            examples.append(item)
    return {
        "policy": "strict_generator_decode_starvation_summary_v1",
        "task_count": len(diagnostics),
        "zero_candidate_task_count": zero_candidate_count,
        "zero_candidate_without_guard_rejection_count": zero_without_guard_count,
        "stopped_zero_candidate_task_count": stopped_count,
        "no_beam_zero_candidate_task_count": no_beam_count,
        "top_beam_state_counts": dict(sorted(state_counts.items())),
        "examples": examples,
        "score_semantics": (
            "Aggregates task-blind decode-starvation diagnostics. This is report evidence for why "
            "strict MLX decode emitted no candidates; it does not alter generation or grant candidate "
            "credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }


def attach_decoded_proposal_metadata(
    rows: list[dict[str, Any]],
    decoded: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Carry strict MLX decode audit metadata through the shared row builder.

    The shared token-decoder row builder is used by several older lanes, so the
    MLX bridge attaches its stricter task-blind guard evidence after row
    construction instead of changing global row semantics.
    """

    by_token_sha: dict[str, dict[str, Any]] = {}
    for task_rows in decoded:
        for proposal in task_rows:
            token_sha = str(proposal.get("decoded_token_sha256") or "")
            if token_sha and token_sha not in by_token_sha:
                by_token_sha[token_sha] = proposal
    for row in rows:
        proposal = by_token_sha.get(str(row.get("decoded_token_sha256") or ""))
        if not proposal:
            continue
        if "decode_static_guard" in proposal:
            row["decode_static_guard"] = proposal.get("decode_static_guard")
        if proposal.get("beam_source"):
            row["beam_source"] = str(proposal.get("beam_source"))
        if proposal.get("plan_prefix_source"):
            row["plan_prefix_source"] = str(proposal.get("plan_prefix_source"))
        if "source_condition_expectation" in proposal:
            row["source_condition_expectation"] = proposal.get("source_condition_expectation")
        if "source_condition_adequacy" in proposal:
            row["source_condition_adequacy"] = proposal.get("source_condition_adequacy")
        if "learned_plan_prefix" in proposal:
            row["learned_plan_prefix"] = proposal.get("learned_plan_prefix")
        if "loop_plan_adequacy" in proposal:
            row["loop_plan_adequacy"] = proposal.get("loop_plan_adequacy")
        if "body_action_trace" in proposal:
            row["body_action_trace"] = proposal.get("body_action_trace")
        row.setdefault("provenance", {})
        row["provenance"]["strict_mlx_decode_guard_attached"] = bool("decode_static_guard" in proposal)
        row["provenance"]["source_condition_adequacy_attached"] = bool("source_condition_adequacy" in proposal)
        if "learned_plan_prefix" in proposal:
            row["provenance"]["learned_plan_prefix_attached"] = True
        if "loop_plan_adequacy" in proposal:
            row["provenance"]["loop_plan_adequacy_attached"] = True
        if "body_action_trace" in proposal:
            row["provenance"]["body_action_trace_attached"] = True
    return rows


def attach_private_verifier_labels(rows: list[dict[str, Any]], private_eval: dict[str, Any]) -> dict[str, Any]:
    """Attach sandbox correctness labels to candidate rows.

    The labels come from the private verifier after candidate generation and
    ranking. They are evidence/training signals only; they do not feed the
    generation path for the evaluated candidates and they contain no private
    test source, expected outputs, solutions, or public benchmark payloads.
    """

    traces = [
        dict_or_empty(row)
        for row in list(private_eval.get("verification_attempt_labels") or [])
        if str(dict_or_empty(row).get("phase") or "") == "private_eval"
    ]
    by_candidate_sha: dict[str, dict[str, Any]] = {}
    for trace in traces:
        candidate_sha = str(trace.get("candidate_sha256") or "")
        if candidate_sha and candidate_sha not in by_candidate_sha:
            by_candidate_sha[candidate_sha] = trace
    residual_by_task = {
        str(row.get("task_id") or ""): dict_or_empty(row)
        for row in list(private_eval.get("residuals") or [])
        if str(dict_or_empty(row).get("task_id") or "")
    }
    label_count = 0
    generated_label_count = 0
    stage_counts: Counter[str] = Counter()
    residual_class_counts: Counter[str] = Counter()
    for row in rows:
        candidate_sha = str(row.get("candidate_sha256") or "")
        trace = by_candidate_sha.get(candidate_sha)
        if trace:
            label = {
                "policy": "private_verifier_candidate_correctness_label_v1",
                "attempt_index": int(trace.get("attempt_index") or 0),
                "verification_stage": str(trace.get("verification_stage") or "unknown"),
                "verification_reward": float(trace.get("verification_reward") or 0.0),
                "lint_passed": bool(trace.get("lint_passed")),
                "compile_passed": bool(trace.get("compile_passed")),
                "runtime_loaded": bool(trace.get("runtime_loaded")),
                "intended_behavior_passed": bool(trace.get("intended_behavior_passed")),
                "reward_breakdown": dict_or_empty(trace.get("reward_breakdown")),
                "verification_cache_key": str(trace.get("verification_cache_key") or ""),
                "uses_eval_tests_or_solutions_for_generation": False,
                "uses_public_data": False,
                "candidate_generation_credit": 0,
            }
            row["private_verifier_label"] = label
            label_count += 1
            if row.get("candidate_generation_mode") == "token_level_code_decoder":
                generated_label_count += 1
            stage_counts[label["verification_stage"]] += 1
        residual = residual_by_task.get(str(row.get("task_id") or ""))
        if residual:
            residual_label = {
                "residual_class": str(residual.get("residual_class") or ""),
                "concept_residual_label": str(residual.get("concept_residual_label") or ""),
                "verification_stage": str(residual.get("verification_stage") or ""),
                "verification_reward": residual.get("verification_reward"),
            }
            row["private_task_residual_label"] = residual_label
            if residual_label["residual_class"]:
                residual_class_counts[residual_label["residual_class"]] += 1
    return {
        "policy": "strict_generator_private_verifier_label_attachment_v1",
        "candidate_rows": len(rows),
        "private_eval_trace_rows": len(traces),
        "attached_label_count": label_count,
        "attached_generated_label_count": generated_label_count,
        "stage_counts": dict(sorted(stage_counts.items())),
        "task_residual_class_counts": dict(sorted(residual_class_counts.items())),
        "uses_eval_tests_or_solutions_for_generation": False,
        "uses_public_data": False,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "candidate_generation_credit": 0,
        "score_semantics": (
            "Attaches private verifier stage/reward labels to already-generated strict MLX candidates. "
            "Labels are post-generation evidence for correctness-in-the-loop training and do not expose "
            "private test code, expected outputs, solutions, public benchmark payloads, or answer templates."
        ),
    }


def mlx_token_choices(
    probs: Any,
    inverse: dict[int, str],
    token_to_id: dict[str, int],
    generated: list[int],
    *,
    eos_id: int,
    grammar_top_k: int,
    max_choices: int,
    token_policy: str,
    target_mode: str,
    allowed_names: set[str] | None,
    input_type_hints: dict[str, str] | None,
    source_condition_expectation: dict[str, Any] | None,
    plan_prefix_choices: list[tuple[int, float]] | None,
    slot_prefix_probs: dict[str, Any] | None,
    enable_learned_expression_token_bias: bool,
    require_parameter_use: bool,
    require_nontrivial_return: bool,
    require_top_level_return: bool,
    prefer_learned_prefix_decision_adequacy: bool,
    prefer_source_condition_adequacy: bool,
    block_shallow_loop_identity_update: bool,
    enable_loop_progress_guard: bool,
    enable_expression_closure_guard: bool,
    enable_expression_value_guard: bool,
    require_binding_prefix_groups: bool,
) -> list[tuple[int, float]]:
    prefix = [inverse.get(idx, "<unk>") for idx in generated[1:]]
    plan_prefix_choices = learned_plan_prefix_choices(
        probs,
        inverse,
        token_to_id,
        prefix,
        target_mode=target_mode,
        max_choices=max_choices,
        override_choices=plan_prefix_choices,
        slot_role_probs=slot_prefix_probs,
        allowed_names=allowed_names,
        require_binding_groups=require_binding_prefix_groups,
    )
    if plan_prefix_choices is not None:
        return plan_prefix_choices
    body_prefix = body_tokens_for_target_mode(prefix, target_mode=target_mode)
    if enable_expression_closure_guard and top_level_return_prefix_ready_for_eos(
        body_prefix,
        target_mode=target_mode,
        allowed_names=allowed_names,
        require_parameter_use=require_parameter_use,
        require_nontrivial_return=require_nontrivial_return,
        require_top_level_return=require_top_level_return,
    ):
        arr = np.asarray(probs, dtype=np.float64)
        return [(eos_id, float(max(arr[eos_id], 1e-9)))]
    ready = completion_ready(
        prefix,
        target_mode=target_mode,
        allowed_names=allowed_names,
        require_parameter_use=require_parameter_use,
        require_nontrivial_return=require_nontrivial_return,
        require_top_level_return=require_top_level_return,
    )
    if ready:
        return [(eos_id, float(max(float(probs[eos_id]), 1e-9)))]
    arr = np.asarray(probs, dtype=np.float64)
    forced = forced_lightweight_python_token(body_prefix, inverse, probs)
    learned_decision_expectation = learned_prefix_decision_expectation_from_tokens(prefix)
    learned_loop_expectation = learned_prefix_loop_expectation_from_tokens(
        list(split_learned_plan_prefix_tokens(prefix)[1].get("learned_plan_prefix_tokens") or [])
    )
    arr = learned_expression_token_biased_array(
        arr,
        inverse,
        prefix,
        body_prefix,
        allowed_names=allowed_names,
        enabled=enable_learned_expression_token_bias,
    )
    if forced is not None and inverse.get(int(forced[0]), "<unk>") == "INDENT:":
        return [forced]
    effective_source_condition_expectation = (
        learned_decision_expectation
        if bool(prefer_learned_prefix_decision_adequacy) and bool(learned_decision_expectation.get("enabled"))
        else dict_or_empty(source_condition_expectation)
    )
    effective_source_condition_enabled = bool(prefer_source_condition_adequacy) or (
        bool(prefer_learned_prefix_decision_adequacy) and bool(learned_decision_expectation.get("enabled"))
    )
    priority_source_choices = source_condition_exploration_choices(
        arr,
        inverse,
        body_prefix,
        expectation=effective_source_condition_expectation,
        seen=set(),
        token_policy=token_policy,
        allowed_names=allowed_names,
        input_type_hints=input_type_hints,
        enabled=effective_source_condition_enabled,
    )
    priority_source_choices = filter_loop_plan_blocked_choices(
        body_prefix,
        priority_source_choices,
        inverse,
        expectation=learned_loop_expectation,
        block_shallow_identity_update=block_shallow_loop_identity_update,
    )
    source_priority_active = bool(priority_source_choices) and source_condition_priority_prefix(
        body_prefix,
        effective_source_condition_expectation,
    )
    direct_return_choices = direct_local_return_continuation_choices(
        arr,
        inverse,
        body_prefix,
        seen=set(),
        token_policy=token_policy,
        allowed_names=allowed_names,
        input_type_hints=input_type_hints,
        require_nontrivial_return=require_nontrivial_return,
        enabled=enable_expression_closure_guard,
    )
    if direct_return_choices:
        return direct_return_choices
    if source_priority_active and source_condition_preempts_loop_plan(effective_source_condition_expectation):
        return priority_source_choices
    loop_plan_choices = loop_plan_exploration_choices(
        arr,
        inverse,
        body_prefix,
        expectation=learned_loop_expectation,
        seen=set(),
        token_policy=token_policy,
        allowed_names=allowed_names,
        input_type_hints=input_type_hints,
        require_nontrivial_return=require_nontrivial_return,
        enabled=bool(prefer_learned_prefix_decision_adequacy),
        enable_loop_progress_guard=enable_loop_progress_guard,
        enable_expression_value_guard=enable_expression_value_guard,
    )
    loop_plan_choices = filter_loop_plan_blocked_choices(
        body_prefix,
        loop_plan_choices,
        inverse,
        expectation=learned_loop_expectation,
        block_shallow_identity_update=block_shallow_loop_identity_update,
    )
    if loop_plan_choices and loop_plan_priority_prefix(body_prefix, learned_loop_expectation):
        return loop_plan_choices
    expression_closure_choices = expression_closure_guard_choices(
        arr,
        inverse,
        body_prefix,
        expectation=learned_loop_expectation,
        seen=set(),
        token_policy=token_policy,
        allowed_names=allowed_names,
        input_type_hints=input_type_hints,
        require_nontrivial_return=require_nontrivial_return,
        block_shallow_identity_update=block_shallow_loop_identity_update,
        enabled=enable_expression_closure_guard,
        enable_expression_value_guard=enable_expression_value_guard,
    )
    if expression_closure_choices:
        return expression_closure_choices
    if source_priority_active:
        return priority_source_choices
    if forced is not None:
        forced_idx = int(forced[0])
        forced_tok = inverse.get(forced_idx, "<unk>")
        if not token_blocked_by_strict_decode_guard(
            body_prefix,
            forced_tok,
            require_nontrivial_return=require_nontrivial_return,
            allowed_names=allowed_names,
            input_type_hints=input_type_hints,
        ) and not token_blocked_by_loop_plan(
            body_prefix,
            forced_tok,
            expectation=learned_loop_expectation,
            block_shallow_identity_update=block_shallow_loop_identity_update,
        ) and not token_blocked_by_expression_value_guard(
            body_prefix,
            forced_tok,
            expectation=learned_loop_expectation,
            enabled=enable_expression_value_guard,
        ):
            seen_forced = {forced_idx}
            source_choices = source_condition_exploration_choices(
                arr,
                inverse,
                body_prefix,
                expectation=effective_source_condition_expectation,
                seen=seen_forced,
                token_policy=token_policy,
                allowed_names=allowed_names,
                input_type_hints=input_type_hints,
                enabled=effective_source_condition_enabled,
            )
            source_choices = filter_loop_plan_blocked_choices(
                body_prefix,
                source_choices,
                inverse,
                expectation=learned_loop_expectation,
                block_shallow_identity_update=block_shallow_loop_identity_update,
            )
            if source_choices:
                return [*source_choices, forced]
            return [forced]
    # Strict legality can reject many malformed high-probability continuations.
    # Search a bounded candidate window so invalid prefixes die cheaply instead
    # of forcing full-vocabulary sorts at every active beam step.
    limit = min(len(arr), max(int(grammar_top_k or 0), int(max_choices or 1) * 16, 128))
    ranked = top_token_indices_desc(arr, limit)
    choices: list[tuple[int, float]] = []
    seen: set[int] = set()
    for next_id in ranked:
        idx = int(next_id)
        if idx in seen:
            continue
        tok = inverse.get(idx, "<unk>")
        if tok == "<eos>" and not ready:
            continue
        if tok.startswith("SLOT:"):
            continue
        if token_blocked_by_strict_decode_guard(
            body_prefix,
            tok,
            require_nontrivial_return=require_nontrivial_return,
            allowed_names=allowed_names,
            input_type_hints=input_type_hints,
        ):
            continue
        if token_blocked_by_loop_plan(
            body_prefix,
            tok,
            expectation=learned_loop_expectation,
            block_shallow_identity_update=block_shallow_loop_identity_update,
        ):
            continue
        if token_blocked_by_expression_value_guard(
            body_prefix,
            tok,
            expectation=learned_loop_expectation,
            enabled=enable_expression_value_guard,
        ):
            continue
        if token_allowed_by_policy(body_prefix, tok, policy=token_policy, allowed_names=allowed_names):
            seen.add(idx)
            choices.append((idx, float(max(arr[idx], 1e-9))))
            if len(choices) >= max(1, int(max_choices or 1)):
                break
    choices.extend(
        visible_parameter_exploration_choices(
            arr,
            inverse,
            body_prefix,
            allowed_names=allowed_names,
            seen=seen,
            token_policy=token_policy,
            require_parameter_use=require_parameter_use,
            input_type_hints=input_type_hints,
        )
    )
    choices.extend(
        source_condition_exploration_choices(
            arr,
            inverse,
            body_prefix,
            expectation=effective_source_condition_expectation,
            seen=seen,
            token_policy=token_policy,
            allowed_names=allowed_names,
            input_type_hints=input_type_hints,
            enabled=effective_source_condition_enabled,
        )
    )
    choices.extend(loop_plan_choices)
    choices.extend(expression_closure_choices)
    choices.extend(direct_return_choices)
    choices = filter_loop_plan_blocked_choices(
        body_prefix,
        choices,
        inverse,
        expectation=learned_loop_expectation,
        block_shallow_identity_update=block_shallow_loop_identity_update,
    )
    choices = filter_expression_value_blocked_choices(
        body_prefix,
        choices,
        inverse,
        expectation=learned_loop_expectation,
        enabled=enable_expression_value_guard,
    )
    if choices:
        return choices
    fallback_limit = min(len(arr), max(limit * 2, 256))
    for next_id in top_token_indices_desc(arr, fallback_limit):
        idx = int(next_id)
        if idx in seen:
            continue
        tok = inverse.get(idx, "<unk>")
        if tok == "<eos>" and not ready:
            continue
        if tok.startswith("SLOT:"):
            continue
        if token_blocked_by_strict_decode_guard(
            body_prefix,
            tok,
            require_nontrivial_return=require_nontrivial_return,
            allowed_names=allowed_names,
            input_type_hints=input_type_hints,
        ):
            continue
        if token_blocked_by_loop_plan(
            body_prefix,
            tok,
            expectation=learned_loop_expectation,
            block_shallow_identity_update=block_shallow_loop_identity_update,
        ):
            continue
        if token_blocked_by_expression_value_guard(
            body_prefix,
            tok,
            expectation=learned_loop_expectation,
            enabled=enable_expression_value_guard,
        ):
            continue
        if token_allowed_by_policy(body_prefix, tok, policy=token_policy, allowed_names=allowed_names):
            return [(idx, float(max(arr[idx], 1e-9)))]
    return [(eos_id, float(max(arr[eos_id], 1e-9)))] if ready else []


def top_token_indices_desc(arr: Any, limit: int) -> list[int]:
    """Return the top token indices without sorting the full vocabulary.

    Strict MLX decode calls this inside the active beam loop. The helper keeps
    the same bounded top-k search semantics as the previous full argsort path,
    but uses ``argpartition`` when only a small legality window is needed.
    """

    values = np.asarray(arr, dtype=np.float64)
    count = len(values)
    requested = max(0, min(count, int(limit or 0)))
    if requested <= 0:
        return []
    if requested >= count:
        candidates = range(count)
    else:
        candidates = np.argpartition(values, count - requested)[count - requested :]
    return [
        int(idx)
        for idx in sorted(
            candidates,
            key=lambda item: (float(values[int(item)]), int(item)),
            reverse=True,
        )
    ]


def top_candidate_indices_desc(arr: Any, candidates: list[int] | set[int], limit: int) -> list[int]:
    values = np.asarray(arr, dtype=np.float64)
    ranked = sorted(
        {int(idx) for idx in candidates if 0 <= int(idx) < len(values)},
        key=lambda idx: (float(values[idx]), idx),
        reverse=True,
    )
    return ranked[: max(0, int(limit or 0))]


def body_action_biased_probability_row(
    token_probs: Any,
    action_probs: Any,
    inverse: dict[int, str],
    *,
    blend: float,
) -> np.ndarray:
    values = np.asarray(token_probs, dtype=np.float64)
    roles = np.asarray(action_probs, dtype=np.float64)
    if values.size <= 0 or roles.size <= 0:
        return values
    weight = max(0.0, min(1.0, float(blend or 0.0)))
    if weight <= 0.0:
        return values
    biased = np.array(values, dtype=np.float64, copy=True)
    for idx in range(len(biased)):
        role_id = body_action_role_id_for_token(str(inverse.get(int(idx), "")))
        role_prob = float(roles[role_id]) if 0 <= int(role_id) < len(roles) else 0.0
        biased[idx] = max(float(biased[idx]), 1e-12) * (max(role_prob, 1e-9) ** weight)
    total = float(np.sum(biased))
    if not math.isfinite(total) or total <= 0.0:
        return values
    return biased / total


def body_operand_biased_probability_row(
    token_probs: Any,
    operand_probs: Any,
    inverse: dict[int, str],
    generated: list[int],
    *,
    allowed_names: set[str],
    blend: float,
) -> np.ndarray:
    values = np.asarray(token_probs, dtype=np.float64)
    roles = np.asarray(operand_probs, dtype=np.float64)
    if values.size <= 0 or roles.size <= 0:
        return values
    weight = max(0.0, min(1.0, float(blend or 0.0)))
    if weight <= 0.0:
        return values
    generated_tokens = [str(inverse.get(int(idx), "")) for idx in list(generated or [])]
    biased = np.array(values, dtype=np.float64, copy=True)
    for idx in range(len(biased)):
        role_id = body_operand_role_id_for_token(
            str(inverse.get(int(idx), "")),
            allowed_names=allowed_names,
            generated_tokens=generated_tokens,
        )
        role_prob = float(roles[role_id]) if 0 <= int(role_id) < len(roles) else 0.0
        biased[idx] = max(float(biased[idx]), 1e-12) * (max(role_prob, 1e-9) ** weight)
    total = float(np.sum(biased))
    if not math.isfinite(total) or total <= 0.0:
        return values
    return biased / total


def learned_expression_token_biased_array(
    arr: Any,
    inverse: dict[int, str],
    prefix: list[str],
    body_prefix: list[str],
    *,
    allowed_names: set[str] | None,
    enabled: bool,
) -> Any:
    """Softly bias body-token probabilities from model-generated expression slots.

    This helper is intentionally not a renderer. It cannot complete an
    expression, cannot force a candidate, and cannot inspect targets, tests,
    solutions, verifier labels, public benchmark payloads, or teacher output.
    It only rescales probabilities for tokens that the model already emitted as
    legal candidates, using generated prefix slots and visible signature names.
    """

    if not enabled or PLAN_BODY_START_TOKEN not in prefix or not body_prefix:
        return arr
    slots = {str(tok) for tok in prefix if str(tok).startswith("SLOT:EXPR_") or str(tok).startswith("SLOT:BIND_")}
    if not slots:
        return arr
    values = list(current_body_line_values(body_prefix))
    if not values:
        return arr
    in_update_call = expression_value_inside_update_call(values)
    wants_operand = expression_value_line_wants_operand(values)
    can_extend_expr = bool(values[-1:] and values[-1] not in {"return", "=", "(", "[", "{", ",", ".", "+", "-", "*", "/", "%", "and", "or", "not"})
    if not (in_update_call or wants_operand or can_extend_expr):
        return arr

    token_to_id = {str(text): int(idx) for idx, text in inverse.items()}
    biased = np.array(arr, dtype=np.float64, copy=True)
    touched = False

    def boost(token: str, multiplier: float) -> None:
        nonlocal touched
        idx = token_to_id.get(token)
        if idx is None or idx < 0 or idx >= len(biased):
            return
        biased[idx] = max(float(biased[idx]), 1e-12) * float(multiplier)
        touched = True

    expression_slots = {slot.removeprefix("SLOT:") for slot in slots if slot.startswith("SLOT:EXPR_")}
    visible_names = sorted({str(name) for name in set(allowed_names or set()) if str(name).isidentifier()})
    generated_names = [
        value
        for value in values
        if value.isidentifier()
        and value not in {"return", "for", "in", "if", "else", "not", "and", "or", "is"}
    ]
    for name in [*visible_names, *generated_names[-3:]]:
        boost(f"NAME:{name}", 1.25 if wants_operand else 1.1)

    if any(slot.startswith("EXPR_CALL_") for slot in expression_slots):
        for slot in expression_slots:
            if not slot.startswith("EXPR_CALL_"):
                continue
            call_name = slot.removeprefix("EXPR_CALL_").lower()
            if call_name == "method_mapping_lookup":
                for name in ("get", "items", "keys", "values"):
                    boost(f"NAME:{name}", 1.25)
            elif call_name == "method_text_transform":
                for name in ("split", "splitlines", "join", "strip", "lower", "upper", "replace"):
                    boost(f"NAME:{name}", 1.18)
            elif call_name == "method_sequence_mutation":
                for name in ("append", "extend", "insert"):
                    boost(f"NAME:{name}", 1.2)
            elif call_name == "method_set_or_map_mutation":
                for name in ("add", "update", "setdefault"):
                    boost(f"NAME:{name}", 1.2)
            elif call_name == "method_ordering_mutation":
                for name in ("sort", "reverse"):
                    boost(f"NAME:{name}", 1.2)
            elif call_name in {"user_or_library", "method_other", "method_unknown"}:
                pass
            elif call_name and call_name.isidentifier():
                boost(f"NAME:{call_name}", 1.8)
        for name in ("len", "sum", "sorted", "str", "int", "float", "list", "tuple", "set"):
            boost(f"NAME:{name}", 1.08)
    if any(slot.startswith("EXPR_BINOP_") for slot in expression_slots):
        operator_slots = {
            "EXPR_BINOP_ADD": "OP:+",
            "EXPR_BINOP_SUB": "OP:-",
            "EXPR_BINOP_MULT": "OP:*",
            "EXPR_BINOP_DIV": "OP:/",
            "EXPR_BINOP_FLOORDIV": "OP://",
            "EXPR_BINOP_MOD": "OP:%",
        }
        for slot, token in operator_slots.items():
            if slot in expression_slots:
                boost(token, 1.7 if can_extend_expr else 1.25)
    if any(slot.startswith("EXPR_BOOLOP_") or slot.startswith("EXPR_COMPARE_") for slot in expression_slots):
        for token in ("NAME:and", "NAME:or", "OP:==", "OP:!=", "OP:<", "OP:>", "NAME:in", "NAME:is"):
            boost(token, 1.15)
    if "EXPR_INDEXING" in expression_slots or any(slot.startswith("EXPR_COMPREHENSION_") for slot in expression_slots):
        boost("OP:[", 1.45 if can_extend_expr or wants_operand else 1.1)
        boost("OP:]", 1.15)
    if any(slot.startswith("EXPR_LOOP_UPDATE_") or slot.startswith("EXPR_TOP_ASSIGN_") for slot in expression_slots):
        for token in ("OP:+", "OP:-", "OP:*", "OP:%", "OP:.", "OP:("):
            boost(token, 1.15)

    if not touched:
        return arr
    total = float(np.sum(biased))
    if not math.isfinite(total) or total <= 0.0:
        return arr
    return biased / total


def expression_value_line_wants_operand(values: list[str]) -> bool:
    if not values:
        return False
    return values[-1] in {"return", "=", "(", "[", "{", ",", ".", "+", "-", "*", "/", "%", "and", "or", "not"}


def top_level_return_prefix_ready_for_eos(
    body_prefix: list[str],
    *,
    target_mode: str,
    allowed_names: set[str] | None,
    require_parameter_use: bool,
    require_nontrivial_return: bool,
    require_top_level_return: bool,
) -> bool:
    """Close a body once a complete top-level return already passes the guard.

    This is a task-blind starvation guard for learned token decoding. It only
    inspects the generated prefix and visible signature names, then applies the
    same static guard used for final candidate acceptance. It does not render a
    return, use tests/solutions, inspect public payloads, or grant generation
    credit.
    """

    if not body_like_target_mode(target_mode):
        return False
    _lines, current_depth, current_values = prefix_lines_with_depth(body_prefix)
    if current_depth != 0 or current_values[:1] != ["return"]:
        return False
    if not expression_closure_line_can_end(list(current_values)):
        return False
    body = decode_body_tokens(body_prefix)
    guard = decode_static_guard(
        body,
        allowed_names=allowed_names,
        require_parameter_use=require_parameter_use,
        require_nontrivial_return=require_nontrivial_return,
        require_top_level_return=require_top_level_return,
    )
    return bool(guard.get("passed"))


def learned_plan_prefix_choices(
    probs: Any,
    inverse: dict[int, str],
    token_to_id: dict[str, int],
    prefix: list[str],
    *,
    target_mode: str,
    max_choices: int,
    override_choices: list[tuple[int, float]] | None = None,
    slot_role_probs: dict[str, Any] | None = None,
    allowed_names: set[str] | None = None,
    require_binding_groups: bool = False,
) -> list[tuple[int, float]] | None:
    if not learned_plan_prefix_target_mode(target_mode):
        return None
    arr = np.asarray(probs, dtype=np.float64)
    prefix_no_eos = [tok for tok in prefix if tok != "<eos>"]
    if not prefix_no_eos:
        if override_choices:
            return list(override_choices)[: max(1, int(max_choices or 1))]
        plan_candidate_ids = [
            int(idx)
            for idx, tok in inverse.items()
            if str(tok).startswith("SLOT:PLAN_")
        ]
        ranked_plan_ids = top_candidate_indices_desc(arr, plan_candidate_ids, max(1, int(max_choices or 1)))
        return [
            (idx, float(max(arr[idx], 1e-9)))
            for idx in ranked_plan_ids
        ]
    if PLAN_BODY_START_TOKEN in prefix_no_eos:
        return None
    if any(tok.startswith("SLOT:PLAN_") for tok in prefix_no_eos) and learned_plan_semantic_slots_body_target_mode(target_mode):
        plan_token = next((tok for tok in prefix_no_eos if tok.startswith("SLOT:PLAN_")), "")
        coverage_choices = learned_prefix_required_category_choices(
            arr,
            inverse,
            prefix_no_eos,
            plan_token=plan_token,
            max_choices=max_choices,
            allowed_names=allowed_names,
            slot_role_probs=slot_role_probs,
        )
        if coverage_choices:
            return coverage_choices
        state_slot_choices = learned_prefix_state_slot_choices(
            arr,
            inverse,
            prefix_no_eos,
            plan_token=plan_token,
            max_choices=max_choices,
            slot_role_probs=slot_role_probs,
        )
        if state_slot_choices:
            return state_slot_choices
        if require_binding_groups:
            binding_choices = learned_prefix_required_binding_choices(
                arr,
                inverse,
                prefix_no_eos,
                plan_token=plan_token,
                max_choices=max_choices,
                allowed_names=allowed_names,
                slot_role_probs=slot_role_probs,
            )
            if binding_choices:
                return binding_choices
        if learned_prefix_ready_for_body_start(prefix_no_eos, plan_token=plan_token):
            idx = token_to_id.get(PLAN_BODY_START_TOKEN)
            return [(idx, float(max(arr[idx], 1e-9)))] if idx is not None else []
        required_slot = learned_prefix_next_required_slot(prefix_no_eos, plan_token=plan_token)
        if required_slot:
            idx = token_to_id.get(required_slot)
            return [(idx, float(max(arr[idx], 1e-9)))] if idx is not None else []
        slot_prefix_count = sum(1 for tok in prefix_no_eos if tok.startswith("SLOT:") and tok != PLAN_BODY_START_TOKEN)
        if slot_prefix_count >= 12:
            idx = token_to_id.get(PLAN_BODY_START_TOKEN)
            return [(idx, float(max(arr[idx], 1e-9)))] if idx is not None else []
        used_prefix_tokens = set(prefix_no_eos)
        slot_candidate_ids = [
            int(idx)
            for idx, tok in inverse.items()
            if str(tok).startswith("SLOT:")
            and str(tok) != "<unk>"
            and not str(tok).startswith("SLOT:PLAN_")
            and str(tok) not in used_prefix_tokens
            and learned_prefix_slot_allowed_for_plan(str(tok), plan_token=plan_token)
            and learned_prefix_slot_allowed_by_signature(str(tok), allowed_names=allowed_names)
        ]
        ranked_slot_ids = top_candidate_indices_desc(arr, slot_candidate_ids, max(1, int(max_choices or 1)))
        return [
            (idx, float(max(arr[idx], 1e-9)))
            for idx in ranked_slot_ids
        ]
    if any(tok.startswith("SLOT:PLAN_") for tok in prefix_no_eos):
        idx = token_to_id.get(PLAN_BODY_START_TOKEN)
        if idx is None:
            return []
        return [(idx, float(max(arr[idx], 1e-9)))]
    return []


def semantic_plan_head_prefix_choices(
    model: Any,
    source_rows: list[list[int]],
    inverse: dict[int, str],
    *,
    max_choices: int,
    target_mode: str,
    enabled: bool,
    source_texts: list[str] | None,
    prefer_source_plan_compatibility: bool,
    mx: Any,
) -> tuple[list[list[tuple[int, float]] | None], list[dict[str, Any]]]:
    if not enabled or not learned_plan_prefix_target_mode(target_mode) or not source_rows:
        return [None for _row in source_rows], [
            {"enabled": False, "policy": "prompt_visible_source_plan_compatibility_v1"}
            for _row in source_rows
        ]
    if not hasattr(model, "semantic_plan_logits"):
        return [None for _row in source_rows], [
            {"enabled": False, "policy": "prompt_visible_source_plan_compatibility_v1", "reason": "model_has_no_semantic_plan_head"}
            for _row in source_rows
        ]
    src = mx.array(source_rows, dtype=mx.int32)
    logits = model.semantic_plan_logits(src)
    probs = mx.softmax(logits, axis=-1)
    mx.eval(probs)
    prob_rows = np.asarray(probs, dtype=np.float64)
    plan_ids = [int(idx) for idx, tok in inverse.items() if str(tok).startswith("SLOT:PLAN_")]
    choices_by_row: list[list[tuple[int, float]] | None] = []
    diagnostics_by_row: list[dict[str, Any]] = []
    for row_index, row in enumerate(prob_rows):
        source_text = source_texts[row_index] if source_texts and row_index < len(source_texts) else ""
        compatibility_rows = {
            int(idx): source_plan_compatibility_for_plan_token(str(inverse.get(int(idx), "")), source_text)
            for idx in plan_ids
        }
        compatibility_enabled = bool(prefer_source_plan_compatibility) and any(
            bool(record.get("enabled")) for record in compatibility_rows.values()
        )
        if compatibility_enabled:
            ranked = sorted(
                plan_ids,
                key=lambda idx: (
                    int(compatibility_rows.get(int(idx), {}).get("score") or 0),
                    float(row[idx]),
                ),
                reverse=True,
            )
        else:
            ranked = sorted(plan_ids, key=lambda idx: float(row[idx]), reverse=True)
        choices = [(idx, float(max(row[idx], 1e-9))) for idx in ranked[: max(1, int(max_choices or 1))]]
        choices_by_row.append(choices or None)
        selected_id = ranked[0] if ranked else None
        selected_record = compatibility_rows.get(int(selected_id), {}) if selected_id is not None else {}
        diagnostics_by_row.append(
            {
                "enabled": bool(compatibility_enabled),
                "requested": bool(prefer_source_plan_compatibility),
                "policy": "prompt_visible_source_plan_compatibility_rerank_v1",
                "selected_plan_token": str(inverse.get(int(selected_id), "")) if selected_id is not None else "",
                "selected_plan": str(selected_record.get("plan") or "").upper(),
                "selected_score": int(selected_record.get("score") or 0),
                "selected_probability": round(float(row[int(selected_id)]), 8) if selected_id is not None else None,
                "top_choices": [
                    {
                        "plan_token": str(inverse.get(int(idx), "")),
                        "probability": round(float(row[int(idx)]), 8),
                        "compatibility": compatibility_rows.get(int(idx), {}),
                    }
                    for idx in ranked[: max(1, int(max_choices or 1))]
                ],
                "score_semantics": (
                    "Reranks semantic plan-head choices with prompt-visible plan compatibility only. "
                    "It does not render code, inspect tests/solutions/public payloads, call tools, or "
                    "grant learned-generation credit."
                ),
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "candidate_generation_credit": 0,
            }
        )
    return choices_by_row, diagnostics_by_row


def semantic_slot_head_prefix_probs(
    model: Any,
    source_rows: list[list[int]],
    inverse: dict[int, str],
    *,
    target_mode: str,
    enabled: bool,
    mx: Any,
) -> list[dict[str, Any] | None]:
    if not enabled or not learned_plan_prefix_target_mode(target_mode) or not source_rows:
        return [None for _row in source_rows]
    if not hasattr(model, "semantic_slot_logits"):
        return [None for _row in source_rows]
    src = mx.array(source_rows, dtype=mx.int32)
    logits = model.semantic_slot_logits(src)
    probs = mx.softmax(logits, axis=-1)
    mx.eval(probs)
    prob_rows = np.asarray(probs, dtype=np.float64)
    role_records: list[dict[str, Any] | None] = []
    for row in prob_rows:
        role_probs: dict[str, Any] = {}
        for role_index, (role, _prefixes) in enumerate(SEMANTIC_SLOT_ROLES):
            if role_index >= row.shape[0]:
                continue
            role_probs[role] = row[role_index]
        role_records.append(role_probs or None)
    return role_records


def learned_prefix_slot_allowed_for_plan(token: str, *, plan_token: str) -> bool:
    if not token.startswith("SLOT:"):
        return False
    if token == PLAN_BODY_START_TOKEN:
        return False
    plan = str(plan_token or "").removeprefix("SLOT:PLAN_")
    if plan == "SAFE_HEAD_DEFAULT":
        return (
            token.startswith("SLOT:RETURN_SHAPE_")
            or token in {
                "SLOT:UPDATE_CALL",
                "SLOT:GUARD_SEQUENCE",
                "SLOT:FINALIZER_RETURN_RESULT",
                "SLOT:FINALIZER_HEAD_DEFAULT",
                "SLOT:COND_SEQUENCE_ARG_DATA",
                "SLOT:COND_TRUTHY_ARG_DATA",
                "SLOT:RETURN_GUARDED_HEAD_ARG_DATA",
                "SLOT:RETURN_DEFAULT_ARG_OTHER",
            }
        )
    if token in learned_prefix_plan_incompatible_slots(plan):
        return False
    return not (
        token.startswith("SLOT:COND_")
        or token.startswith("SLOT:RETURN_GUARDED_")
        or token.startswith("SLOT:RETURN_DEFAULT_")
    )


def learned_prefix_plan_incompatible_slots(plan: str) -> set[str]:
    """Return task-blind slot contradictions for a model-generated plan token.

    These checks do not choose a solution or read verifier/private target data.
    They only keep the learned semantic prefix internally coherent before raw
    body-token decoding starts.
    """
    plan_text = str(plan or "").upper()
    blocked: set[str] = set()
    if "NORMALIZE" in plan_text or "SORT_UNIQUE" in plan_text or "SET_ADD" in plan_text:
        blocked.update(
            {
                "SLOT:INIT_BOOL",
                "SLOT:INIT_CONDITIONAL",
                "SLOT:INIT_DICT",
                "SLOT:INIT_ITEM",
                "SLOT:INIT_LIST",
                "SLOT:INIT_LOOKUP",
                "SLOT:INIT_MAPPING_OR_INPUT",
                "SLOT:INIT_NONE",
                "SLOT:INIT_NUMBER",
                "SLOT:INIT_STR",
                "SLOT:INIT_TUPLE",
                "SLOT:INIT_UNKNOWN",
                "SLOT:UPDATE_ACCUMULATE_NUMERIC",
                "SLOT:UPDATE_APPEND_ITEM",
                "SLOT:UPDATE_APPEND_LABEL",
                "SLOT:UPDATE_APPEND_NAME",
                "SLOT:UPDATE_APPEND_PROJECTED_ROW",
                "SLOT:UPDATE_APPEND_STDIN_PAIR_SUM",
                "SLOT:UPDATE_CALL",
                "SLOT:UPDATE_DICT_SETDEFAULT",
                "SLOT:FINALIZER_SUM",
                "SLOT:FINALIZER_MAX",
                "SLOT:FINALIZER_BOOL_NOT_STACK",
                "SLOT:FINALIZER_JOIN",
                "SLOT:FINALIZER_NONE",
                "SLOT:FINALIZER_RETURN_RESULT",
                "SLOT:FINALIZER_TUPLE",
                "SLOT:RETURN_SHAPE_ATTRIBUTE",
                "SLOT:RETURN_SHAPE_BOOL",
                "SLOT:RETURN_SHAPE_CONDITIONAL",
                "SLOT:RETURN_SHAPE_DICT",
                "SLOT:RETURN_SHAPE_ITEM",
                "SLOT:RETURN_SHAPE_LOOKUP",
                "SLOT:RETURN_SHAPE_MAPPING_OR_INPUT",
                "SLOT:RETURN_SHAPE_NONE",
                "SLOT:RETURN_SHAPE_NUMBER",
                "SLOT:RETURN_SHAPE_STR",
                "SLOT:RETURN_SHAPE_TUPLE",
                "SLOT:RETURN_SHAPE_UNKNOWN",
                "SLOT:LOOP_SOURCE_RANGE",
                "SLOT:LOOP_SOURCE_SORTED",
            }
        )
    if "NUMERIC" in plan_text or "SUM" in plan_text or "GCD" in plan_text or "MAX" in plan_text or "MIN" in plan_text:
        blocked.update(
            {
                "SLOT:INIT_SET",
                "SLOT:UPDATE_SET_ADD",
                "SLOT:FINALIZER_SORTED",
            }
        )
    return blocked


def learned_prefix_slot_allowed_by_signature(token: str, *, allowed_names: set[str] | None) -> bool:
    names = {str(name) for name in set(allowed_names or set()) if str(name)}
    if token.startswith("SLOT:LOOP_SOURCE_"):
        if not names:
            return True
        source = token.removeprefix("SLOT:LOOP_SOURCE_").lower()
        parts = {part for part in source.split("_") if part}
        return bool(source in names or parts.intersection(names))
    return True


def learned_prefix_next_required_slot(prefix_tokens: list[str], *, plan_token: str) -> str:
    plan = str(plan_token or "").removeprefix("SLOT:PLAN_")
    seen = set(str(tok) for tok in prefix_tokens)
    if plan == "SAFE_HEAD_DEFAULT":
        for token in (
            "SLOT:COND_SEQUENCE_ARG_DATA",
            "SLOT:COND_TRUTHY_ARG_DATA",
            "SLOT:RETURN_GUARDED_HEAD_ARG_DATA",
            "SLOT:RETURN_DEFAULT_ARG_OTHER",
        ):
            if token not in seen:
                return token
    return ""


def learned_prefix_ready_for_body_start(prefix_tokens: list[str], *, plan_token: str) -> bool:
    plan = str(plan_token or "").removeprefix("SLOT:PLAN_")
    seen = set(str(tok) for tok in prefix_tokens)
    if plan == "SAFE_HEAD_DEFAULT":
        required = {
            "SLOT:COND_SEQUENCE_ARG_DATA",
            "SLOT:COND_TRUTHY_ARG_DATA",
            "SLOT:RETURN_GUARDED_HEAD_ARG_DATA",
            "SLOT:RETURN_DEFAULT_ARG_OTHER",
        }
        return required.issubset(seen)
    slot_count = sum(1 for tok in prefix_tokens if str(tok).startswith("SLOT:") and tok != PLAN_BODY_START_TOKEN)
    has_loop_source = any(str(tok).startswith("SLOT:LOOP_SOURCE_") for tok in prefix_tokens)
    has_init = any(str(tok).startswith("SLOT:INIT_") for tok in prefix_tokens)
    has_update = any(str(tok).startswith("SLOT:UPDATE_") for tok in prefix_tokens)
    if has_loop_source and has_init and has_update:
        return True
    return slot_count >= 6


def learned_prefix_required_category_choices(
    arr: Any,
    inverse: dict[int, str],
    prefix_tokens: list[str],
    *,
    plan_token: str,
    max_choices: int,
    allowed_names: set[str] | None = None,
    slot_role_probs: dict[str, Any] | None = None,
) -> list[tuple[int, float]]:
    plan = str(plan_token or "").removeprefix("SLOT:PLAN_")
    if not plan or plan == "SAFE_HEAD_DEFAULT":
        return []
    seen = {str(tok) for tok in prefix_tokens}
    categories = [
        ("loop_source", ("SLOT:LOOP_SOURCE_",)),
        ("init", ("SLOT:INIT_",)),
        ("update", ("SLOT:UPDATE_",)),
        ("state_finalizer", ("SLOT:STATE_FINALIZER_",)),
        ("binding_update", ("SLOT:BIND_UPDATE_",)),
        ("binding_finalizer", ("SLOT:BIND_FINALIZER_", "SLOT:BIND_RETURN_")),
        ("statement_sequence", ("SLOT:STMT_SEQ_",)),
        ("statement_loop", ("SLOT:STMT_LOOP_",)),
        ("statement_final", ("SLOT:STMT_FINAL_",)),
        ("expression_call", ("SLOT:EXPR_CALL_",)),
        ("expression_binop", ("SLOT:EXPR_BINOP_",)),
        ("expression_return", ("SLOT:EXPR_RETURN_",)),
        ("expression_update", ("SLOT:EXPR_LOOP_UPDATE_", "SLOT:EXPR_TOP_ASSIGN_")),
        ("expression_branch", ("SLOT:EXPR_LOOP_BRANCH_", "SLOT:EXPR_COMPARE_", "SLOT:EXPR_BOOLOP_")),
        ("expression_structure", ("SLOT:EXPR_INDEXING", "SLOT:EXPR_COMPREHENSION_")),
    ]
    for role, prefixes in categories:
        if any(any(tok.startswith(prefix) for prefix in prefixes) for tok in seen):
            continue
        ids = [
            int(idx)
            for idx, tok in inverse.items()
            if any(str(tok).startswith(prefix) for prefix in prefixes) and str(tok) not in seen
            and learned_prefix_slot_allowed_for_plan(str(tok), plan_token=plan_token)
            and learned_prefix_slot_allowed_by_signature(str(tok), allowed_names=allowed_names)
        ]
        if not ids:
            continue
        role_arr = semantic_slot_role_array(slot_role_probs, role=role)
        if role_arr is None:
            role_arr = arr
        ranked = sorted(ids, key=lambda idx: float(role_arr[idx]), reverse=True)
        return [
            (idx, float(max(role_arr[idx], 1e-9)))
            for idx in ranked[: max(1, int(max_choices or 1))]
        ]
    return []


def learned_prefix_state_slot_choices(
    arr: Any,
    inverse: dict[int, str],
    prefix_tokens: list[str],
    *,
    plan_token: str,
    max_choices: int,
    slot_role_probs: dict[str, Any] | None = None,
) -> list[tuple[int, float]]:
    plan = str(plan_token or "").removeprefix("SLOT:PLAN_")
    if not plan or plan == "SAFE_HEAD_DEFAULT":
        return []
    seen = {str(tok) for tok in prefix_tokens}
    if any(tok.startswith("SLOT:STATE_UPDATE_") for tok in seen):
        return []
    update_state_ids = [
        int(idx)
        for idx, tok in inverse.items()
        if str(tok).startswith("SLOT:STATE_UPDATE_")
        and learned_prefix_update_state_slot_allowed_by_contract(str(tok), prefix_tokens=prefix_tokens)
    ]
    if any(tok.startswith("SLOT:UPDATE_") for tok in seen) and update_state_ids:
        role_arr = semantic_slot_role_array(slot_role_probs, role="state_update")
        if role_arr is None:
            role_arr = arr
        ranked = sorted(update_state_ids, key=lambda idx: float(role_arr[idx]), reverse=True)
        return [
            (idx, float(max(role_arr[idx], 1e-9)))
            for idx in ranked[: max(1, int(max_choices or 1))]
        ]
    if any(tok.startswith("SLOT:STATE_") for tok in seen):
        return []
    # Only require state slots when this checkpoint actually has them. This
    # preserves old vocabularies while letting the state-transition target mode
    # force a learned block/state prefix before raw body-token decoding.
    state_ids = [
        int(idx)
        for idx, tok in inverse.items()
        if str(tok).startswith("SLOT:STATE_")
    ]
    if not state_ids:
        return []
    role_arr = semantic_slot_role_array(slot_role_probs, role="state_finalizer")
    if role_arr is None:
        role_arr = arr
    ranked = sorted(state_ids, key=lambda idx: float(role_arr[idx]), reverse=True)
    return [
        (idx, float(max(role_arr[idx], 1e-9)))
        for idx in ranked[: max(1, int(max_choices or 1))]
    ]


def learned_prefix_update_state_slot_allowed_by_contract(token: str, *, prefix_tokens: list[str]) -> bool:
    """Task-blind compatibility for learned update-state prefix slots.

    The model still chooses among legal state slots by probability. This guard
    only prevents incoherent contracts such as a numeric accumulator requiring a
    mutation-call state before raw body-token decoding. It uses generated prefix
    tokens only; it does not inspect tests, solutions, public payloads, verifier
    outcomes, or target rows at decode time.
    """

    seen = {str(tok) for tok in prefix_tokens}
    init_shapes = {
        tok.removeprefix("SLOT:INIT_").lower()
        for tok in seen
        if tok.startswith("SLOT:INIT_")
    }
    update_ops = {
        tok.removeprefix("SLOT:UPDATE_").lower()
        for tok in seen
        if tok.startswith("SLOT:UPDATE_")
    }
    if "number" in init_shapes and not ({"list", "dict", "set"} & init_shapes):
        if token in {"SLOT:STATE_UPDATE_MUTATION_CALL", "SLOT:STATE_UPDATE_SHALLOW_IDENTITY"}:
            return False
    if "accumulate_numeric" in update_ops:
        return token in {
            "SLOT:STATE_UPDATE_AUGASSIGN",
            "SLOT:STATE_UPDATE_ASSIGN_TRANSFORM",
            "SLOT:STATE_UPDATE_NONIDENTITY",
        }
    if {"append_item", "append_projected_row", "append_label", "append_name", "append_stdin_pair_sum", "set_add", "dict_setdefault"} & update_ops:
        if token in {"SLOT:STATE_UPDATE_AUGASSIGN"}:
            return False
    return True


def learned_prefix_required_binding_choices(
    arr: Any,
    inverse: dict[int, str],
    prefix_tokens: list[str],
    *,
    plan_token: str,
    max_choices: int,
    allowed_names: set[str] | None = None,
    slot_role_probs: dict[str, Any] | None = None,
) -> list[tuple[int, float]]:
    plan = str(plan_token or "").removeprefix("SLOT:PLAN_")
    if not plan or plan == "SAFE_HEAD_DEFAULT":
        return []
    all_binding_ids = [
        int(idx)
        for idx, tok in inverse.items()
        if str(tok).startswith("SLOT:BIND_")
        and learned_prefix_slot_allowed_by_signature(str(tok), allowed_names=allowed_names)
    ]
    if not all_binding_ids:
        return []
    seen = {str(tok) for tok in prefix_tokens}
    required_groups = [
        ("binding_loop", ("SLOT:BIND_LOOP_TARGET_", "SLOT:BIND_BRANCH_")),
        ("binding_update", ("SLOT:BIND_UPDATE_",)),
        ("binding_finalizer", ("SLOT:BIND_FINALIZER_", "SLOT:BIND_RETURN_")),
    ]
    for role, prefixes in required_groups:
        if any(any(tok.startswith(prefix) for prefix in prefixes) for tok in seen):
            continue
        ids = [
            idx
            for idx in all_binding_ids
            if str(inverse.get(idx, "")) not in seen
            and any(str(inverse.get(idx, "")).startswith(prefix) for prefix in prefixes)
        ]
        if not ids:
            continue
        role_arr = semantic_slot_role_array(slot_role_probs, role=role)
        if role_arr is None:
            role_arr = arr
        ranked = sorted(ids, key=lambda idx: float(role_arr[idx]), reverse=True)
        return [
            (idx, float(max(role_arr[idx], 1e-9)))
            for idx in ranked[: max(1, int(max_choices or 1))]
        ]
    return []


def semantic_slot_role_array(slot_role_probs: dict[str, Any] | None, *, role: str) -> Any | None:
    if not slot_role_probs:
        return None
    arr = slot_role_probs.get(role)
    if arr is None:
        return None
    return arr


def learned_plan_prefix_metadata(decoded_tokens: list[str], *, target_mode: str) -> dict[str, Any]:
    if not learned_plan_prefix_target_mode(target_mode):
        return {
            "enabled": False,
            "target_mode": target_mode,
            "candidate_generation_credit": 0,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    _body_tokens, meta = split_learned_plan_prefix_tokens(decoded_tokens)
    prefix_tokens = [str(tok) for tok in list(meta.get("learned_plan_prefix_tokens") or [])]
    return {
        "enabled": True,
        "target_mode": target_mode,
        "learned_plan_prefix_used": bool(meta.get("learned_plan_prefix_used")),
        "learned_plan_prefix_complete": bool(meta.get("learned_plan_prefix_complete")),
        "learned_plan_prefix_tokens": prefix_tokens,
        "predicted_plan_token": str(meta.get("predicted_plan_token") or ""),
        "predicted_plan": str(meta.get("predicted_plan") or ""),
        "policy": "strict_mlx_decoded_learned_plan_prefix_metadata_v1",
        "score_semantics": (
            "Task-blind metadata derived from generated plan-prefix tokens before body decoding. "
            "It is evidence for whether the learned plan head chose the expected coarse route; it "
            "does not render code, inspect tests/solutions, use public data, or grant candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
    }




if __name__ == "__main__":
    raise SystemExit(main())
