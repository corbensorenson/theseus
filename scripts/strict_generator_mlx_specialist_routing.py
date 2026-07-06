#!/usr/bin/env python3
"""Specialist-head routing helpers for strict-generator MLX decode.

Specialist routing chooses among existing private replay checkpoint heads and
task-blind decode profiles. It does not inspect tests, solutions, public data,
verifier labels, or answer templates, and it grants zero generation credit.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from neural_seed_code_proposer_comparator import dict_or_empty, rel


PRIVATE_TRAIN_REPLAY_SPECIALIST_TIERS = ("simple_return", "loop_accumulate", "algorithmic_small")


def load_specialist_heads(
    specs: dict[str, dict[str, Any]],
    *,
    loader: Callable[..., dict[str, Any]],
    mx: Any,
    nn: Any,
) -> dict[str, dict[str, Any]]:
    heads: dict[str, dict[str, Any]] = {}
    for tier, spec in sorted(specs.items()):
        checkpoint = spec["checkpoint"]
        vocab = spec["vocab"]
        loaded = loader(vocab, checkpoint, mx=mx, nn=nn)
        heads[tier] = {
            "head_id": f"specialist:{tier}",
            "head_type": "private_replay_tier_specialist_checkpoint",
            "tier": tier,
            "model": loaded["model"],
            "vocab_payload": loaded["vocab_payload"],
            "checkpoint": checkpoint,
            "vocab": vocab,
            "report_path": spec.get("report_path"),
            "checkpoint_report_trigger_state": dict_or_empty(spec.get("report")).get("trigger_state"),
        }
    return heads


def private_train_replay_specialist_routes(
    *,
    train_replay_tier: str,
    specialist_heads: dict[str, dict[str, Any]],
    default_head: dict[str, Any],
    max_rows: int,
    use_route_profiles: bool,
) -> list[dict[str, Any]]:
    tier = str(train_replay_tier or "any").strip() or "any"
    if tier != "any":
        head = specialist_heads.get(tier, default_head)
        return [
            {
                "tier": tier,
                "split_key": "private_train_replay" if head is default_head else f"private_train_replay:{tier}",
                "head": head,
                "max_rows": int(max_rows or 0),
                "selection_reason": "explicit_train_replay_tier",
                "specialist_selected": head is not default_head,
                "decode_profile": specialist_route_profile_for_tier(tier) if use_route_profiles else "global_cli_profile",
            }
        ]
    if not specialist_heads:
        return [
            {
                "tier": "any",
                "split_key": "private_train_replay",
                "head": default_head,
                "max_rows": int(max_rows or 0),
                "selection_reason": "default_any_private_replay",
                "specialist_selected": False,
                "decode_profile": "global_cli_profile",
            }
        ]
    ordered_tiers = [tier for tier in PRIVATE_TRAIN_REPLAY_SPECIALIST_TIERS if tier in specialist_heads]
    route_count = max(1, len(ordered_tiers))
    per_route_max = int(math.ceil(max_rows / route_count)) if int(max_rows or 0) > 0 else 0
    return [
        {
            "tier": tier,
            "split_key": f"private_train_replay:{tier}",
            "head": specialist_heads[tier],
            "max_rows": per_route_max,
            "selection_reason": "configured_specialist_tier_route",
            "specialist_selected": True,
            "decode_profile": specialist_route_profile_for_tier(tier) if use_route_profiles else "global_cli_profile",
        }
        for tier in ordered_tiers
    ]


def specialist_route_profile_for_tier(tier: str) -> str:
    if tier == "simple_return":
        return "simple_return_safe_head_v1"
    if tier == "loop_accumulate":
        return "loop_operand_binding_v1"
    return "global_cli_profile"


def specialist_route_decode_options(
    route: dict[str, Any],
    *,
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
    prefer_learned_prefix_decision_adequacy: bool,
    prefer_source_condition_adequacy: bool,
    require_source_condition_adequacy: bool,
    block_shallow_loop_identity_update: bool,
    enable_loop_progress_guard: bool,
    enable_expression_closure_guard: bool,
    enable_expression_value_guard: bool,
    require_binding_prefix_groups: bool,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "decode_profile": str(route.get("decode_profile") or "global_cli_profile"),
        "max_target_tokens_override": int(max_target_tokens_override or 0),
        "output_top_k": int(output_top_k or 0),
        "require_parameter_use": bool(require_parameter_use),
        "require_nontrivial_return": bool(require_nontrivial_return),
        "require_top_level_return": bool(require_top_level_return),
        "use_semantic_plan_head_prefix": bool(use_semantic_plan_head_prefix),
        "prefer_source_plan_compatibility": bool(prefer_source_plan_compatibility),
        "use_semantic_slot_head_prefix": bool(use_semantic_slot_head_prefix),
        "enable_learned_expression_token_bias": bool(enable_learned_expression_token_bias),
        "use_body_transition_head": bool(use_body_transition_head),
        "body_transition_head_blend": max(0.0, min(1.0, float(body_transition_head_blend or 0.0))),
        "use_body_action_head": bool(use_body_action_head),
        "body_action_head_blend": max(0.0, min(1.0, float(body_action_head_blend or 0.0))),
        "prefer_learned_prefix_decision_adequacy": bool(prefer_learned_prefix_decision_adequacy),
        "prefer_source_condition_adequacy": bool(prefer_source_condition_adequacy),
        "require_source_condition_adequacy": bool(require_source_condition_adequacy),
        "block_shallow_loop_identity_update": bool(block_shallow_loop_identity_update),
        "enable_loop_progress_guard": bool(enable_loop_progress_guard),
        "enable_expression_closure_guard": bool(enable_expression_closure_guard),
        "enable_expression_value_guard": bool(enable_expression_value_guard),
        "require_binding_prefix_groups": bool(require_binding_prefix_groups),
        "candidate_generation_credit": 0,
        "uses_eval_tests_or_solutions_for_generation": False,
        "uses_public_data": False,
    }
    profile = str(options["decode_profile"])
    if profile == "simple_return_safe_head_v1":
        options.update(
            {
                "max_target_tokens_override": 100,
                "output_top_k": max(4, int(output_top_k or 0)),
                "require_top_level_return": True,
                "use_semantic_plan_head_prefix": True,
                "use_semantic_slot_head_prefix": bool(use_semantic_slot_head_prefix),
                "prefer_learned_prefix_decision_adequacy": True,
                "prefer_source_condition_adequacy": True,
                "require_source_condition_adequacy": False,
                "block_shallow_loop_identity_update": False,
                "enable_expression_closure_guard": True,
                "enable_expression_value_guard": True,
                "require_binding_prefix_groups": False,
            }
        )
    elif profile == "loop_operand_binding_v1":
        options.update(
            {
                "max_target_tokens_override": max(224, int(max_target_tokens_override or 0)),
                "output_top_k": int(output_top_k or 0),
                "require_top_level_return": bool(require_top_level_return),
                "use_semantic_plan_head_prefix": False,
                "use_semantic_slot_head_prefix": bool(use_semantic_slot_head_prefix),
                "prefer_learned_prefix_decision_adequacy": True,
                "prefer_source_condition_adequacy": False,
                "require_source_condition_adequacy": False,
                "block_shallow_loop_identity_update": True,
                "enable_expression_closure_guard": True,
                "enable_expression_value_guard": True,
                "require_binding_prefix_groups": False,
            }
        )
    options["score_semantics"] = (
        "Decode profile for a specialist learned checkpoint route. It changes task-blind grammar/search "
        "settings only; it does not inspect verifier labels, tests, solutions, public data, or answer "
        "templates, does not render code, and grants zero candidate-generation credit."
    )
    return options


def specialist_route_record(route: dict[str, Any]) -> dict[str, Any]:
    head = dict_or_empty(route.get("head"))
    decode_options = dict_or_empty(route.get("decode_options"))
    return {
        "policy": "strict_generator_private_replay_specialist_head_route_v1",
        "enabled": True,
        "tier": str(route.get("tier") or ""),
        "split_key": str(route.get("split_key") or ""),
        "head_id": str(head.get("head_id") or "default"),
        "head_type": str(head.get("head_type") or "default_checkpoint"),
        "checkpoint": rel(head.get("checkpoint")) if head.get("checkpoint") else "",
        "vocab": rel(head.get("vocab")) if head.get("vocab") else "",
        "checkpoint_report": str(head.get("report_path") or ""),
        "checkpoint_report_trigger_state": str(head.get("checkpoint_report_trigger_state") or ""),
        "max_rows": int(route.get("max_rows") or 0),
        "selection_reason": str(route.get("selection_reason") or ""),
        "specialist_selected": bool(route.get("specialist_selected")),
        "decode_profile": str(route.get("decode_profile") or "global_cli_profile"),
        "decode_options": decode_options,
        "score_semantics": (
            "Routes an existing private-train replay tier to a learned checkpoint specialized for that "
            "tier. This is model/head selection only. Generation still sees only strict prompt/signature "
            "source text and the selected learned checkpoint; the route does not inspect tests, solutions, "
            "public data, verifier labels, or answer templates, and it grants zero learned-generation credit."
        ),
        "uses_eval_tests_or_solutions_for_generation": False,
        "uses_public_data": False,
        "external_inference_calls": 0,
        "candidate_generation_credit": 0,
        "fallback_template_router_tool_credit_count": 0,
    }


def specialist_head_routing_summary(
    specs: dict[str, dict[str, Any]],
    *,
    active: bool,
    routes: list[dict[str, Any]],
    use_route_profiles: bool,
) -> dict[str, Any]:
    return {
        "policy": "strict_generator_private_replay_specialist_head_routing_summary_v1",
        "enabled": bool(specs),
        "active": bool(active),
        "use_specialist_route_profiles": bool(use_route_profiles),
        "configured_tiers": sorted(specs),
        "route_count": len(routes),
        "routes": routes,
        "score_semantics": (
            "Specialist heads are existing learned checkpoints routed by private replay tier. This "
            "routing can support engineering evidence about preserving separate skills, but it is not "
            "candidate-generation credit, not a template, not a fallback, and not public benchmark training."
        ),
        "uses_eval_tests_or_solutions_for_generation": False,
        "uses_public_data": False,
        "external_inference_calls": 0,
        "candidate_generation_credit": 0,
        "fallback_template_router_tool_credit_count": 0,
    }
