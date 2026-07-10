#!/usr/bin/env python3
"""Replay MLX strict-generator rung checkpoints through private decode eval.

The pretraining probe can now emit intermediate checkpoints at token-position
milestones. This script keeps checkpoint selection honest by replaying those
rungs through the existing private decode/verifier bridge, without public
benchmarks, teacher inference, templates, tools, or fallback returns.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neural_seed_code_proposer_comparator import dict_or_empty, get_path, rel, resolve, stable_hash  # noqa: E402
from strict_generator_mlx_decode_eval import run_decode_eval, write_json, write_jsonl  # noqa: E402
from strict_generator_mlx_replay_selection import select_broad_private_heldout_rows, select_family_disjoint_rows  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "neural_seed_token_decoder_comparator.json"
DEFAULT_TRAINING_REPORT = ROOT / "reports" / "strict_generator_mlx_pretraining_probe_plan_prefix_plan_aux_30m_step_guard_v1.json"
DEFAULT_OUT = ROOT / "reports" / "strict_generator_mlx_rung_decode_sweep.json"
DEFAULT_ARTIFACTS_DIR = ROOT / "reports" / "strict_generator_mlx_rung_decode_sweep"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--training-report", default=rel(DEFAULT_TRAINING_REPORT))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--artifacts-dir", default=rel(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--split", choices=["family_disjoint", "broad_private_heldout", "both"], default="both")
    parser.add_argument("--max-family-rows", type=int, default=24)
    parser.add_argument("--max-broad-rows", type=int, default=64)
    parser.add_argument("--output-top-k", type=int, default=3)
    parser.add_argument("--include-final", action="store_true")
    parser.add_argument("--checkpoint-id", action="append", default=[], help="Replay only matching checkpoint ids, for example rung_15000000 or final. Repeatable.")
    parser.add_argument("--max-checkpoints", type=int, default=0, help="Replay at most this many selected checkpoints. Zero means no cap.")
    parser.add_argument("--require-parameter-use", action="store_true")
    parser.add_argument("--require-nontrivial-return", action="store_true")
    parser.add_argument("--require-top-level-return", action="store_true")
    parser.add_argument("--use-semantic-plan-head-prefix", action="store_true")
    parser.add_argument("--prefer-source-plan-compatibility", action="store_true")
    parser.add_argument("--use-semantic-slot-head-prefix", action="store_true")
    parser.add_argument("--enable-learned-expression-token-bias", action="store_true")
    parser.add_argument("--use-body-transition-head", action="store_true")
    parser.add_argument("--body-transition-head-blend", type=float, default=0.35)
    parser.add_argument("--use-body-action-head", action="store_true")
    parser.add_argument("--body-action-head-blend", type=float, default=0.35)
    parser.add_argument("--use-body-operand-head", action="store_true")
    parser.add_argument("--body-operand-head-blend", type=float, default=0.35)
    parser.add_argument("--use-body-state-event-head", action="store_true")
    parser.add_argument("--body-state-event-head-blend", type=float, default=0.35)
    parser.add_argument("--prefer-learned-prefix-decision-adequacy", action="store_true")
    parser.add_argument("--prefer-source-condition-adequacy", action="store_true")
    parser.add_argument("--require-source-condition-adequacy", action="store_true")
    parser.add_argument("--block-shallow-loop-identity-update", action="store_true")
    parser.add_argument("--enable-loop-progress-guard", action="store_true")
    parser.add_argument("--enable-expression-closure-guard", action="store_true")
    parser.add_argument("--enable-expression-value-guard", action="store_true")
    parser.add_argument("--enable-semantic-operation-value-construction", action="store_true")
    parser.add_argument("--require-binding-prefix-groups", action="store_true")
    parser.add_argument("--resource-budget-ms", type=int, default=0, help="Optional total decode/eval runtime budget. Zero records observation only.")
    parser.add_argument("--max-child-decode-eval-ms", type=int, default=0, help="Optional per-checkpoint decode/eval runtime budget. Zero records observation only.")
    parser.add_argument("--min-eval-rows-per-second", type=float, default=0.0, help="Optional all-rung replay throughput floor. Zero records observation only.")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = read_json(resolve(args.config))
    training_report_path = resolve(args.training_report)
    training_report = read_json(training_report_path)
    report = run_sweep(
        config,
        config_path=str(args.config),
        training_report_path=training_report_path,
        training_report=training_report,
        artifacts_dir=resolve(args.artifacts_dir),
        split=str(args.split),
        max_family_rows=max(0, int(args.max_family_rows or 0)),
        max_broad_rows=max(0, int(args.max_broad_rows or 0)),
        output_top_k=max(0, int(args.output_top_k or 0)),
        include_final=bool(args.include_final),
        checkpoint_ids=[str(item) for item in list(args.checkpoint_id or []) if str(item).strip()],
        max_checkpoints=max(0, int(args.max_checkpoints or 0)),
        require_parameter_use=bool(args.require_parameter_use),
        require_nontrivial_return=bool(args.require_nontrivial_return),
        require_top_level_return=bool(args.require_top_level_return),
        use_semantic_plan_head_prefix=bool(args.use_semantic_plan_head_prefix),
        prefer_source_plan_compatibility=bool(args.prefer_source_plan_compatibility),
        use_semantic_slot_head_prefix=bool(args.use_semantic_slot_head_prefix),
        enable_learned_expression_token_bias=bool(args.enable_learned_expression_token_bias),
        use_body_transition_head=bool(args.use_body_transition_head),
        body_transition_head_blend=max(0.0, min(1.0, float(args.body_transition_head_blend or 0.0))),
        use_body_action_head=bool(args.use_body_action_head),
        body_action_head_blend=max(0.0, min(1.0, float(args.body_action_head_blend or 0.0))),
        use_body_operand_head=bool(args.use_body_operand_head),
        body_operand_head_blend=max(0.0, min(1.0, float(args.body_operand_head_blend or 0.0))),
        use_body_state_event_head=bool(args.use_body_state_event_head),
        body_state_event_head_blend=max(0.0, min(1.0, float(args.body_state_event_head_blend or 0.0))),
        prefer_learned_prefix_decision_adequacy=bool(args.prefer_learned_prefix_decision_adequacy),
        prefer_source_condition_adequacy=bool(args.prefer_source_condition_adequacy),
        require_source_condition_adequacy=bool(args.require_source_condition_adequacy),
        block_shallow_loop_identity_update=bool(args.block_shallow_loop_identity_update),
        enable_loop_progress_guard=bool(args.enable_loop_progress_guard),
        enable_expression_closure_guard=bool(args.enable_expression_closure_guard),
        enable_expression_value_guard=bool(args.enable_expression_value_guard),
        enable_semantic_operation_value_construction=bool(args.enable_semantic_operation_value_construction),
        require_binding_prefix_groups=bool(args.require_binding_prefix_groups),
        resource_budget_ms=max(0, int(args.resource_budget_ms or 0)),
        max_child_decode_eval_ms=max(0, int(args.max_child_decode_eval_ms or 0)),
        min_eval_rows_per_second=max(0.0, float(args.min_eval_rows_per_second or 0.0)),
        execute=bool(args.execute),
    )
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW", "PLANNED"} else 2


def run_sweep(
    config: dict[str, Any],
    *,
    config_path: str,
    training_report_path: Path,
    training_report: dict[str, Any],
    artifacts_dir: Path,
    split: str,
    max_family_rows: int,
    max_broad_rows: int,
    output_top_k: int,
    include_final: bool,
    checkpoint_ids: list[str],
    max_checkpoints: int,
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
    use_body_state_event_head: bool,
    body_state_event_head_blend: float,
    prefer_learned_prefix_decision_adequacy: bool,
    prefer_source_condition_adequacy: bool,
    require_source_condition_adequacy: bool,
    block_shallow_loop_identity_update: bool,
    enable_loop_progress_guard: bool,
    enable_expression_closure_guard: bool,
    enable_expression_value_guard: bool,
    enable_semantic_operation_value_construction: bool,
    require_binding_prefix_groups: bool,
    resource_budget_ms: int,
    max_child_decode_eval_ms: int,
    min_eval_rows_per_second: float,
    execute: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    available_checkpoints = checkpoint_records_from_training_report(training_report, include_final=include_final)
    checkpoints = select_checkpoint_records(available_checkpoints, checkpoint_ids=checkpoint_ids, max_checkpoints=max_checkpoints)
    selection = checkpoint_selection_summary(available_checkpoints, checkpoints, checkpoint_ids=checkpoint_ids, max_checkpoints=max_checkpoints)
    if not execute:
        return {
            "policy": "project_theseus_strict_generator_mlx_rung_decode_sweep_v1",
            "created_utc": now(),
            "trigger_state": "PLANNED",
            "execute": False,
            "summary": {
                "config": config_path,
                "training_report": rel(training_report_path),
                "artifact_dir": rel(artifacts_dir),
                "available_checkpoint_count": len(available_checkpoints),
                "checkpoint_count": len(checkpoints),
                "checkpoint_selection": selection,
                "checkpoints": checkpoints,
                "split": split,
                "resource_budget_ms": int(resource_budget_ms or 0),
                "max_child_decode_eval_ms": int(max_child_decode_eval_ms or 0),
                "min_eval_rows_per_second": float(min_eval_rows_per_second or 0.0),
                "decode_options": rung_decode_options_receipt(
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
                    use_body_state_event_head=use_body_state_event_head,
                    body_state_event_head_blend=body_state_event_head_blend,
                    prefer_learned_prefix_decision_adequacy=prefer_learned_prefix_decision_adequacy,
                    prefer_source_condition_adequacy=prefer_source_condition_adequacy,
                    require_source_condition_adequacy=require_source_condition_adequacy,
                    block_shallow_loop_identity_update=block_shallow_loop_identity_update,
                    enable_loop_progress_guard=enable_loop_progress_guard,
                    enable_expression_closure_guard=enable_expression_closure_guard,
                    enable_expression_value_guard=enable_expression_value_guard,
                    enable_semantic_operation_value_construction=enable_semantic_operation_value_construction,
                    require_binding_prefix_groups=require_binding_prefix_groups,
                ),
                "public_training_rows": 0,
                "external_inference_calls": 0,
            },
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        }

    preselected_splits, preselection_summary = build_preselected_split_bundle(
        config,
        checkpoints=checkpoints,
        split=split,
        max_family_rows=max_family_rows,
        max_broad_rows=max_broad_rows,
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    child_reports: list[dict[str, Any]] = []
    checkpoint_loader_cache = build_checkpoint_loader_cache(checkpoints)
    for record in checkpoints:
        checkpoint_id = str(record.get("id") or "checkpoint")
        safe_id = safe_slug(checkpoint_id)
        child_out = artifacts_dir / f"{safe_id}_decode_report.json"
        candidates_out = artifacts_dir / f"{safe_id}_candidates.jsonl"
        child_started = time.perf_counter()
        child_report, candidates = run_decode_eval(
            config,
            config_path=config_path,
            checkpoint_report_path=str(training_report_path),
            checkpoint_path=resolve(str(record.get("checkpoint") or "")),
            vocab_path=resolve(str(record.get("vocab") or "")),
            specialist_checkpoint_specs={},
            use_specialist_route_profiles=False,
            split=split,
            max_family_rows=max_family_rows,
            max_broad_rows=max_broad_rows,
            max_train_replay_rows=0,
            max_target_tokens_override=0,
            train_replay_tier="any",
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
            use_body_state_event_head=use_body_state_event_head,
            body_state_event_head_blend=body_state_event_head_blend,
            prefer_learned_prefix_decision_adequacy=prefer_learned_prefix_decision_adequacy,
            prefer_source_condition_adequacy=prefer_source_condition_adequacy,
            require_source_condition_adequacy=require_source_condition_adequacy,
            block_shallow_loop_identity_update=block_shallow_loop_identity_update,
            enable_loop_progress_guard=enable_loop_progress_guard,
            enable_expression_closure_guard=enable_expression_closure_guard,
            enable_expression_value_guard=enable_expression_value_guard,
            enable_semantic_operation_value_construction=enable_semantic_operation_value_construction,
            require_binding_prefix_groups=require_binding_prefix_groups,
            execute=True,
            preselected_splits=preselected_splits,
            checkpoint_loader_cache=checkpoint_loader_cache,
        )
        decode_eval_runtime_ms = int((time.perf_counter() - child_started) * 1000)
        write_started = time.perf_counter()
        write_json(child_out, child_report)
        write_jsonl(candidates_out, candidates)
        artifact_write_runtime_ms = int((time.perf_counter() - write_started) * 1000)
        child_reports.append(child_report)
        rows.append(
            summarize_child(
                record,
                child_report,
                child_out=child_out,
                candidates_out=candidates_out,
                decode_eval_runtime_ms=decode_eval_runtime_ms,
                artifact_write_runtime_ms=artifact_write_runtime_ms,
            )
        )

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            -int(row.get("total_private_passes") or 0),
            -float(row.get("nontrivial_return_rate_mean") or 0.0),
            int(row.get("milestone_token_positions") or 10**18),
        ),
    )
    checkpoint_loader_summary = checkpoint_loader_cache_summary(checkpoint_loader_cache)
    verifier_cache_summary = verifier_cache_warmup_summary(rows, child_reports)
    performance = performance_summary(rows)
    resource_budget = resource_budget_summary(
        rows,
        performance,
        resource_budget_ms=resource_budget_ms,
        max_child_decode_eval_ms=max_child_decode_eval_ms,
        min_eval_rows_per_second=min_eval_rows_per_second,
    )
    route_eligibility = route_eligibility_summary(
        rows,
        child_reports,
        performance=performance,
        resource_budget=resource_budget,
        checkpoint_loader_summary=checkpoint_loader_summary,
        verifier_cache_summary=verifier_cache_summary,
    )
    gates = build_gates(
        checkpoints,
        rows,
        child_reports,
        checkpoint_selection_summary=selection,
        preselection_summary=preselection_summary,
        checkpoint_loader_summary=checkpoint_loader_summary,
        verifier_cache_summary=verifier_cache_summary,
        resource_budget_summary=resource_budget,
        route_eligibility_summary=route_eligibility,
    )
    hard_pass = all(row["passed"] for row in gates if row["severity"] == "hard")
    trigger_state = "GREEN" if hard_pass else "RED"
    if trigger_state == "GREEN" and (
        any(not row["passed"] for row in gates)
        or any(str(row.get("child_trigger_state") or "") == "RED" for row in rows)
    ):
        trigger_state = "YELLOW"
    return {
        "policy": "project_theseus_strict_generator_mlx_rung_decode_sweep_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "execute": True,
        "summary": {
            "config": config_path,
            "training_report": rel(training_report_path),
            "artifact_dir": rel(artifacts_dir),
            "split": split,
            "available_checkpoint_count": len(available_checkpoints),
            "checkpoint_count": len(checkpoints),
            "checkpoint_selection": selection,
            "preselected_split_reuse": preselection_summary,
            "checkpoint_loader_reuse": checkpoint_loader_summary,
            "verifier_cache_warmup": verifier_cache_summary,
            "performance": performance,
            "resource_budget": resource_budget,
            "route_eligibility": route_eligibility,
            "decode_options": rung_decode_options_receipt(
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
                use_body_state_event_head=use_body_state_event_head,
                body_state_event_head_blend=body_state_event_head_blend,
                prefer_learned_prefix_decision_adequacy=prefer_learned_prefix_decision_adequacy,
                prefer_source_condition_adequacy=prefer_source_condition_adequacy,
                require_source_condition_adequacy=require_source_condition_adequacy,
                block_shallow_loop_identity_update=block_shallow_loop_identity_update,
                enable_loop_progress_guard=enable_loop_progress_guard,
                enable_expression_closure_guard=enable_expression_closure_guard,
                enable_expression_value_guard=enable_expression_value_guard,
                enable_semantic_operation_value_construction=enable_semantic_operation_value_construction,
                require_binding_prefix_groups=require_binding_prefix_groups,
            ),
            "best_checkpoint_by_private_passes": rows_sorted[0] if rows_sorted else None,
            "rows": rows,
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "fallback_template_router_tool_credit_count": 0,
        },
        "gates": gates,
        "score_semantics": (
            "Private-only rung comparison for MLX strict-generator checkpoints. It compares replayable "
            "intermediate/final checkpoints through the unchanged private decode/verifier bridge. It may "
            "run a bounded checkpoint subset for Mac/MLX health canaries, but selection is recorded and does "
            "not run public calibration, train on public data, call external inference, use teacher output, "
            "or credit templates, tools, routers, structural adapters, or fallback returns as learned generation."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
}


def rung_decode_options_receipt(
    *,
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
    use_body_state_event_head: bool,
    body_state_event_head_blend: float,
    prefer_learned_prefix_decision_adequacy: bool,
    prefer_source_condition_adequacy: bool,
    require_source_condition_adequacy: bool,
    block_shallow_loop_identity_update: bool,
    enable_loop_progress_guard: bool,
    enable_expression_closure_guard: bool,
    enable_expression_value_guard: bool,
    enable_semantic_operation_value_construction: bool,
    require_binding_prefix_groups: bool,
) -> dict[str, Any]:
    return {
        "policy": "strict_generator_mlx_rung_decode_options_v3",
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
        "use_body_operand_head": bool(use_body_operand_head),
        "body_operand_head_blend": max(0.0, min(1.0, float(body_operand_head_blend or 0.0))),
        "use_body_state_event_head": bool(use_body_state_event_head),
        "body_state_event_head_blend": max(0.0, min(1.0, float(body_state_event_head_blend or 0.0))),
        "prefer_learned_prefix_decision_adequacy": bool(prefer_learned_prefix_decision_adequacy),
        "prefer_source_condition_adequacy": bool(prefer_source_condition_adequacy),
        "require_source_condition_adequacy": bool(require_source_condition_adequacy),
        "block_shallow_loop_identity_update": bool(block_shallow_loop_identity_update),
        "enable_loop_progress_guard": bool(enable_loop_progress_guard),
        "enable_expression_closure_guard": bool(enable_expression_closure_guard),
        "enable_expression_value_guard": bool(enable_expression_value_guard),
        "enable_semantic_operation_value_construction": bool(enable_semantic_operation_value_construction),
        "require_binding_prefix_groups": bool(require_binding_prefix_groups),
        "score_semantics": (
            "Rung sweep decode option receipt. These options are task-blind decoder/search settings "
            "forwarded to strict_generator_mlx_decode_eval.py; they do not inspect tests, solutions, "
            "public benchmark payloads, verifier labels, or answer templates, and they grant zero "
            "candidate-generation credit."
        ),
        "uses_eval_tests_or_solutions_for_generation": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
        "external_inference_calls": 0,
        "public_training_rows": 0,
        "fallback_template_router_tool_credit_count": 0,
    }


def build_checkpoint_loader_cache(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "policy": "strict_mlx_same_vocab_checkpoint_loader_cache_v1",
        "enabled": bool(checkpoints),
        "checkpoint_count": len(checkpoints),
        "vocab_payloads": {},
        "models": {},
        "stats": {
            "vocab_read_count": 0,
            "vocab_cache_hit_count": 0,
            "model_construct_count": 0,
            "model_reuse_count": 0,
            "checkpoint_weight_load_count": 0,
            "strict_weight_load_count": 0,
            "nonstrict_weight_load_count": 0,
        },
        "loaded_checkpoint_sha256": [],
    }


def checkpoint_loader_cache_summary(cache: dict[str, Any]) -> dict[str, Any]:
    stats = dict_or_empty(cache.get("stats"))
    loaded_hashes = [str(item) for item in list(cache.get("loaded_checkpoint_sha256") or [])]
    return {
        "policy": str(cache.get("policy") or "strict_mlx_same_vocab_checkpoint_loader_cache_v1"),
        "enabled": bool(cache.get("enabled")),
        "checkpoint_count": int(cache.get("checkpoint_count") or 0),
        "vocab_cache_entry_count": len(dict_or_empty(cache.get("vocab_payloads"))),
        "model_cache_entry_count": len(dict_or_empty(cache.get("models"))),
        "loaded_checkpoint_count": len(loaded_hashes),
        "loaded_checkpoint_sha256_hash": stable_hash(json.dumps(loaded_hashes, sort_keys=True)),
        "stats": {
            "vocab_read_count": int(stats.get("vocab_read_count") or 0),
            "vocab_cache_hit_count": int(stats.get("vocab_cache_hit_count") or 0),
            "model_construct_count": int(stats.get("model_construct_count") or 0),
            "model_reuse_count": int(stats.get("model_reuse_count") or 0),
            "checkpoint_weight_load_count": int(stats.get("checkpoint_weight_load_count") or 0),
            "strict_weight_load_count": int(stats.get("strict_weight_load_count") or 0),
            "nonstrict_weight_load_count": int(stats.get("nonstrict_weight_load_count") or 0),
        },
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit_count": 0,
        "score_semantics": (
            "Runtime-economics receipt only. Same-vocab checkpoint replay can reuse vocab payloads and "
            "model construction, while every checkpoint must still reload checkpoint weights before decode."
        ),
    }


def build_preselected_split_bundle(
    config: dict[str, Any],
    *,
    checkpoints: list[dict[str, Any]],
    split: str,
    max_family_rows: int,
    max_broad_rows: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    relevant_splits = [
        name
        for name in ["family_disjoint", "broad_private_heldout"]
        if split in {name, "both"}
    ]
    if not checkpoints or not relevant_splits:
        return {}, preselection_receipt(
            enabled=False,
            reason="no_private_selection_reuse_needed",
            runtime_ms=int((time.perf_counter() - started) * 1000),
        )

    target_modes: list[str] = []
    vocab_hashes: list[str] = []
    for record in checkpoints:
        vocab_ref = str(record.get("vocab") or "")
        try:
            vocab_payload = read_json(resolve(vocab_ref))
        except Exception as exc:
            return {}, preselection_receipt(
                enabled=False,
                reason="vocab_read_failed",
                error=f"{type(exc).__name__}: {str(exc)[:240]}",
                runtime_ms=int((time.perf_counter() - started) * 1000),
            )
        target_modes.append(str(vocab_payload.get("target_mode") or get_path(config, ["body_structure_decoder", "target_mode"], "body_tokens")))
        vocab_hashes.append(str(record.get("vocab_sha256") or stable_hash(json.dumps(vocab_payload, sort_keys=True))))

    unique_target_modes = sorted(set(target_modes))
    if len(unique_target_modes) != 1:
        return {}, preselection_receipt(
            enabled=False,
            reason="heterogeneous_checkpoint_target_modes",
            target_modes=unique_target_modes,
            runtime_ms=int((time.perf_counter() - started) * 1000),
        )

    selection_config = json.loads(json.dumps(config))
    selection_config.setdefault("body_structure_decoder", {})
    selection_config["body_structure_decoder"]["target_mode"] = unique_target_modes[0]
    bundle: dict[str, dict[str, Any]] = {}
    split_receipts: dict[str, Any] = {}
    if "family_disjoint" in relevant_splits:
        selection = select_family_disjoint_rows(selection_config, max_rows=max_family_rows)
        bundle["family_disjoint"] = selection
        split_receipts["family_disjoint"] = private_selection_receipt("family_disjoint", selection)
    if "broad_private_heldout" in relevant_splits:
        selection = select_broad_private_heldout_rows(selection_config, max_rows=max_broad_rows)
        bundle["broad_private_heldout"] = selection
        split_receipts["broad_private_heldout"] = private_selection_receipt("broad_private_heldout", selection)
    return bundle, preselection_receipt(
        enabled=bool(bundle),
        reason="private_split_selection_reused_across_checkpoint_replay",
        split=split,
        target_mode=unique_target_modes[0],
        selected_split_count=len(bundle),
        split_receipts=split_receipts,
        checkpoint_count=len(checkpoints),
        checkpoint_vocab_sha256_hash=stable_hash(json.dumps(sorted(vocab_hashes), sort_keys=True)),
        runtime_ms=int((time.perf_counter() - started) * 1000),
    )


def preselection_receipt(*, enabled: bool, reason: str, runtime_ms: int, **extra: Any) -> dict[str, Any]:
    return {
        "policy": "project_theseus_checkpoint_replay_private_split_reuse_v1",
        "enabled": bool(enabled),
        "reason": reason,
        "runtime_ms": int(runtime_ms),
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit_count": 0,
        "uses_eval_tests_or_solutions_for_generation": False,
        "uses_public_data": False,
        "candidate_generation_credit": 0,
        **extra,
    }


def private_selection_receipt(split_name: str, selection: dict[str, Any]) -> dict[str, Any]:
    eval_rows = list(selection.get("eval_rows") or [])
    train_rows = list(selection.get("train_rows") or [])
    eval_row_ids = [
        str(row.get("task_id") or row.get("source_task_id") or stable_hash(json.dumps(row, sort_keys=True)))
        for row in eval_rows
    ]
    train_row_ids = [
        str(row.get("task_id") or row.get("source_task_id") or stable_hash(json.dumps(row, sort_keys=True)))
        for row in train_rows
    ]
    return {
        "split": split_name,
        "active": bool(selection.get("active")),
        "seed": selection.get("seed"),
        "family_key": selection.get("family_key"),
        "train_row_count": len(train_rows),
        "eval_row_count": len(eval_rows),
        "train_row_id_hash": stable_hash(json.dumps(train_row_ids, sort_keys=True)),
        "eval_row_id_hash": stable_hash(json.dumps(eval_row_ids, sort_keys=True)),
        "split_overlap_audit": get_path(selection, ["split_audit", "overlap"], {}),
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit_count": 0,
    }


def checkpoint_records_from_training_report(report: dict[str, Any], *, include_final: bool) -> list[dict[str, Any]]:
    summary = dict_or_empty(report.get("summary"))
    budget = dict_or_empty(report.get("budget"))
    rungs = list(summary.get("rung_checkpoints") or budget.get("rung_checkpoints") or [])
    records: list[dict[str, Any]] = []
    for row in rungs:
        row = dict_or_empty(row)
        milestone = int(row.get("milestone_token_positions") or 0)
        records.append(
            {
                "id": f"rung_{milestone}",
                "kind": "rung",
                "milestone_token_positions": milestone,
                "checkpoint": row.get("checkpoint"),
                "checkpoint_sha256": row.get("checkpoint_sha256"),
                "vocab": row.get("vocab"),
                "vocab_sha256": row.get("vocab_sha256"),
                "heldout_lm_loss": row.get("heldout_lm_loss"),
                "heldout_lm_perplexity": row.get("heldout_lm_perplexity"),
                "heldout_source_loss_gap": row.get("heldout_source_loss_gap"),
                "heldout_plan_loss": row.get("heldout_plan_loss"),
                "heldout_plan_accuracy": row.get("heldout_plan_accuracy"),
                "training_tokens_per_second_so_far": row.get("training_tokens_per_second_so_far"),
            }
        )
    if include_final:
        source = budget or summary
        records.append(
            {
                "id": "final",
                "kind": "final",
                "milestone_token_positions": int(get_path(source, ["training_plan", "target_token_positions"], 0) or 0),
                "checkpoint": source.get("checkpoint"),
                "checkpoint_sha256": source.get("checkpoint_sha256"),
                "vocab": source.get("vocab"),
                "vocab_sha256": source.get("vocab_sha256"),
                "heldout_lm_loss": source.get("heldout_lm_loss_after"),
                "heldout_lm_perplexity": source.get("heldout_lm_perplexity_after"),
                "heldout_source_loss_gap": get_path(source, ["source_contrastive_loss", "heldout_source_loss_gap_after"]),
                "heldout_plan_loss": get_path(source, ["semantic_plan_auxiliary", "heldout_plan_loss_after"]),
                "heldout_plan_accuracy": get_path(source, ["semantic_plan_auxiliary", "heldout_plan_accuracy_after"]),
                "training_tokens_per_second_so_far": source.get("training_tokens_per_second"),
            }
        )
    return [row for row in records if row.get("checkpoint") and row.get("vocab")]


def select_checkpoint_records(
    records: list[dict[str, Any]],
    *,
    checkpoint_ids: list[str],
    max_checkpoints: int,
) -> list[dict[str, Any]]:
    selected = list(records)
    requested = {str(item).strip() for item in checkpoint_ids if str(item).strip()}
    if requested:
        selected = [row for row in selected if str(row.get("id") or "") in requested]
    if max_checkpoints > 0:
        selected = selected[:max_checkpoints]
    return selected


def checkpoint_selection_summary(
    available: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    checkpoint_ids: list[str],
    max_checkpoints: int,
) -> dict[str, Any]:
    requested = [str(item).strip() for item in checkpoint_ids if str(item).strip()]
    available_ids = [str(row.get("id") or "") for row in available]
    selected_ids = [str(row.get("id") or "") for row in selected]
    return {
        "policy": "project_theseus_mlx_rung_checkpoint_selection_v1",
        "available_checkpoint_count": len(available),
        "selected_checkpoint_count": len(selected),
        "available_checkpoint_ids": available_ids,
        "selected_checkpoint_ids": selected_ids,
        "requested_checkpoint_ids": requested,
        "unmatched_requested_checkpoint_ids": sorted(set(requested) - set(available_ids)),
        "max_checkpoints": int(max_checkpoints or 0),
        "bounded_canary": bool(requested or int(max_checkpoints or 0) > 0),
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit_count": 0,
    }


def summarize_child(
    record: dict[str, Any],
    report: dict[str, Any],
    *,
    child_out: Path,
    candidates_out: Path,
    decode_eval_runtime_ms: int,
    artifact_write_runtime_ms: int,
) -> dict[str, Any]:
    summary = dict_or_empty(report.get("summary"))
    split_passes = dict_or_empty(summary.get("split_passes"))
    nontrivial_rates = dict_or_empty(summary.get("split_nontrivial_return_rates"))
    inert_rates = dict_or_empty(summary.get("split_inert_stub_rates"))
    total_passes = sum(int(value or 0) for value in split_passes.values())
    mean_nontrivial = sum(float(value or 0.0) for value in nontrivial_rates.values()) / max(1, len(nontrivial_rates))
    mean_inert = sum(float(value or 0.0) for value in inert_rates.values()) / max(1, len(inert_rates))
    integrity = dict_or_empty(summary.get("candidate_integrity"))
    split_reports = {
        name: dict_or_empty(split_report)
        for name, split_report in dict_or_empty(report.get("splits")).items()
    }
    split_eval_rows = {
        name: int(get_path(split_report, ["summary", "eval_rows"], 0) or 0)
        for name, split_report in split_reports.items()
    }
    split_decode_ms = {
        name: int(get_path(split_report, ["summary", "decode_wall_time_ms"], 0) or 0)
        for name, split_report in split_reports.items()
    }
    split_verifier_ms = {
        name: int(get_path(split_report, ["summary", "private_verifier", "wall_time_ms"], 0) or 0)
        for name, split_report in split_reports.items()
    }
    split_verifier_cache_warmup = {
        name: get_path(split_report, ["summary", "private_verifier", "verifier_cache_warmup"], {})
        for name, split_report in split_reports.items()
    }
    split_test_harness_cache_hits = {
        name: int(get_path(split_report, ["summary", "private_verifier", "test_harness_cache_hit_count"], 0) or 0)
        for name, split_report in split_reports.items()
    }
    return {
        **record,
        "child_trigger_state": report.get("trigger_state"),
        "decode_report": rel(child_out),
        "candidates": rel(candidates_out),
        "decode_eval_runtime_ms": int(decode_eval_runtime_ms),
        "artifact_write_runtime_ms": int(artifact_write_runtime_ms),
        "child_report_runtime_ms": int(report.get("runtime_ms") or 0),
        "checkpoint_loader_reuse": dict_or_empty(summary.get("checkpoint_loader_reuse")),
        "split_eval_rows": split_eval_rows,
        "split_decode_wall_time_ms": split_decode_ms,
        "split_verifier_wall_time_ms": split_verifier_ms,
        "split_verifier_cache_warmup": split_verifier_cache_warmup,
        "split_test_harness_cache_hit_count": split_test_harness_cache_hits,
        "total_eval_rows": sum(split_eval_rows.values()),
        "total_decode_wall_time_ms": sum(split_decode_ms.values()),
        "total_verifier_wall_time_ms": sum(split_verifier_ms.values()),
        "total_test_harness_cache_hit_count": sum(split_test_harness_cache_hits.values()),
        "generated_candidate_rows": summary.get("generated_candidate_rows"),
        "manifest_candidate_rows": summary.get("manifest_candidate_rows"),
        "split_passes": split_passes,
        "total_private_passes": total_passes,
        "split_nontrivial_return_rates": nontrivial_rates,
        "nontrivial_return_rate_mean": round(mean_nontrivial, 6),
        "split_inert_stub_rates": inert_rates,
        "inert_stub_rate_mean": round(mean_inert, 6),
        "integrity_mismatch_count": integrity.get("integrity_mismatch_count"),
        "integrity_verified_candidate_count": integrity.get("integrity_verified_candidate_count"),
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit_count": 0,
    }


def performance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_decode_eval_ms = sum(int(row.get("decode_eval_runtime_ms") or 0) for row in rows)
    total_artifact_write_ms = sum(int(row.get("artifact_write_runtime_ms") or 0) for row in rows)
    total_eval_rows = sum(int(row.get("total_eval_rows") or 0) for row in rows)
    total_generated_candidates = sum(int(row.get("generated_candidate_rows") or 0) for row in rows)
    total_test_harness_cache_hits = sum(int(row.get("total_test_harness_cache_hit_count") or 0) for row in rows)
    return {
        "policy": "project_theseus_mlx_rung_replay_performance_receipt_v1",
        "checkpoint_count": len(rows),
        "total_decode_eval_runtime_ms": total_decode_eval_ms,
        "total_artifact_write_runtime_ms": total_artifact_write_ms,
        "total_eval_rows": total_eval_rows,
        "total_generated_candidate_rows": total_generated_candidates,
        "total_test_harness_cache_hit_count": total_test_harness_cache_hits,
        "eval_rows_per_second": round(total_eval_rows / max(total_decode_eval_ms / 1000.0, 1e-9), 6),
        "generated_candidates_per_second": round(total_generated_candidates / max(total_decode_eval_ms / 1000.0, 1e-9), 6),
    }


def verifier_cache_warmup_summary(rows: list[dict[str, Any]], reports: list[dict[str, Any]]) -> dict[str, Any]:
    receipts: list[dict[str, Any]] = []
    split_receipt_paths: list[str] = []
    for checkpoint_index, report in enumerate(reports):
        for split_name, split_report in dict_or_empty(report.get("splits")).items():
            receipt = dict_or_empty(get_path(split_report, ["summary", "private_verifier", "verifier_cache_warmup"], {}))
            if not receipt:
                continue
            receipts.append(receipt)
            split_receipt_paths.append(f"checkpoint_{checkpoint_index}.{split_name}")

    delta_totals = {
        "static_cache_entry_count": 0,
        "sandbox_cache_entry_count": 0,
        "test_harness_compile_cache_entry_count": 0,
    }
    for receipt in receipts:
        deltas = dict_or_empty(receipt.get("cache_entry_deltas"))
        for key in list(delta_totals):
            delta_totals[key] += int(deltas.get(key) or 0)

    return {
        "policy": "private_verifier_sandbox_warmup_accounting_rollup_v1",
        "receipt_count": len(receipts),
        "split_receipt_paths": split_receipt_paths,
        "all_test_harness_compile_cache_enabled": bool(receipts)
        and all(bool(receipt.get("test_harness_compile_cache_enabled")) for receipt in receipts),
        "all_static_candidate_cache_enabled": bool(receipts)
        and all(bool(receipt.get("static_candidate_cache_enabled")) for receipt in receipts),
        "all_sandbox_result_cache_enabled": bool(receipts)
        and all(bool(receipt.get("sandbox_result_cache_enabled")) for receipt in receipts),
        "worker_count_max": max([int(receipt.get("worker_count") or 0) for receipt in receipts] or [0]),
        "eval_task_count_total": sum(int(receipt.get("eval_task_count") or 0) for receipt in receipts),
        "candidate_count_total": sum(int(receipt.get("candidate_count") or 0) for receipt in receipts),
        "cache_entry_delta_totals": delta_totals,
        "total_test_harness_cache_hit_count": sum(int(row.get("total_test_harness_cache_hit_count") or 0) for row in rows),
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit_count": 0,
        "score_semantics": (
            "Runtime-economics rollup only. It proves verifier cache warmup and hit accounting were emitted "
            "during private checkpoint replay; it does not change pass/fail semantics or learned-generation credit."
        ),
    }


def resource_budget_summary(
    rows: list[dict[str, Any]],
    performance: dict[str, Any],
    *,
    resource_budget_ms: int,
    max_child_decode_eval_ms: int,
    min_eval_rows_per_second: float,
) -> dict[str, Any]:
    total_decode_eval_ms = int(performance.get("total_decode_eval_runtime_ms") or 0)
    eval_rows_per_second = float(performance.get("eval_rows_per_second") or 0.0)
    child_runtimes = {
        str(row.get("id") or f"checkpoint_{idx}"): int(row.get("decode_eval_runtime_ms") or 0)
        for idx, row in enumerate(rows)
    }
    child_decode_wall = {
        str(row.get("id") or f"checkpoint_{idx}"): int(row.get("total_decode_wall_time_ms") or 0)
        for idx, row in enumerate(rows)
    }
    max_child_runtime_ms = max(child_runtimes.values() or [0])
    max_child_decode_wall_ms = max(child_decode_wall.values() or [0])
    slow_child_ids = [
        checkpoint_id
        for checkpoint_id, runtime_ms in child_runtimes.items()
        if int(max_child_decode_eval_ms or 0) > 0 and runtime_ms > int(max_child_decode_eval_ms or 0)
    ]
    slow_decode_wall_ids = [
        checkpoint_id
        for checkpoint_id, runtime_ms in child_decode_wall.items()
        if int(max_child_decode_eval_ms or 0) > 0 and runtime_ms > int(max_child_decode_eval_ms or 0)
    ]
    total_budget_ok = int(resource_budget_ms or 0) <= 0 or total_decode_eval_ms <= int(resource_budget_ms or 0)
    child_budget_ok = int(max_child_decode_eval_ms or 0) <= 0 or max_child_runtime_ms <= int(max_child_decode_eval_ms or 0)
    throughput_ok = float(min_eval_rows_per_second or 0.0) <= 0.0 or eval_rows_per_second >= float(min_eval_rows_per_second or 0.0)
    enforced = any(
        [
            int(resource_budget_ms or 0) > 0,
            int(max_child_decode_eval_ms or 0) > 0,
            float(min_eval_rows_per_second or 0.0) > 0.0,
        ]
    )
    return {
        "policy": "project_theseus_mlx_rung_replay_resource_budget_v1",
        "enforced": bool(enforced),
        "resource_budget_ms": int(resource_budget_ms or 0),
        "max_child_decode_eval_ms": int(max_child_decode_eval_ms or 0),
        "min_eval_rows_per_second": float(min_eval_rows_per_second or 0.0),
        "checkpoint_count": len(rows),
        "total_decode_eval_runtime_ms": total_decode_eval_ms,
        "max_child_decode_eval_runtime_ms": max_child_runtime_ms,
        "max_child_decode_wall_time_ms": max_child_decode_wall_ms,
        "eval_rows_per_second": eval_rows_per_second,
        "child_decode_eval_runtime_ms": child_runtimes,
        "child_decode_wall_time_ms": child_decode_wall,
        "slow_child_decode_eval_ids": slow_child_ids,
        "slow_child_decode_wall_ids": slow_decode_wall_ids,
        "total_budget_ok": bool(total_budget_ok),
        "child_budget_ok": bool(child_budget_ok),
        "throughput_ok": bool(throughput_ok),
        "budget_ok": bool(total_budget_ok and child_budget_ok and throughput_ok),
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit_count": 0,
        "score_semantics": (
            "Runtime routing receipt only. Resource thresholds can fail closed before production MLX routing; "
            "they do not change candidate generation, verifier semantics, or learned-generation credit."
        ),
    }


def route_eligibility_summary(
    rows: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    *,
    performance: dict[str, Any],
    resource_budget: dict[str, Any],
    checkpoint_loader_summary: dict[str, Any],
    verifier_cache_summary: dict[str, Any],
) -> dict[str, Any]:
    total_private_passes = sum(int(row.get("total_private_passes") or 0) for row in rows)
    total_eval_rows = int(performance.get("total_eval_rows") or 0)
    pass_rate = round(total_private_passes / max(1, total_eval_rows), 6)
    integrity_mismatch_count = sum(int(row.get("integrity_mismatch_count") or 0) for row in rows)
    public_training_rows = sum(int(get_path(report, ["summary", "public_training_rows"], 0) or 0) for report in reports)
    external_inference_calls = sum(int(get_path(report, ["summary", "external_inference_calls"], 0) or 0) for report in reports)
    fallback_credit = sum(
        int(get_path(report, ["summary", "fallback_template_router_tool_credit_count"], 0) or 0)
        for report in reports
    )
    resource_enforced = bool(resource_budget.get("enforced"))
    resource_ok = bool(resource_budget.get("budget_ok"))
    no_cheat_ok = public_training_rows == 0 and external_inference_calls == 0 and fallback_credit == 0
    integrity_ok = integrity_mismatch_count == 0
    quality_ok = total_private_passes > 0
    qualification_inputs_met = bool(rows) and resource_enforced and resource_ok and quality_ok and integrity_ok and no_cheat_ok

    if not rows:
        route_state = "fail_closed_no_replay_rows"
        recommended_next_action = "produce_private_replay_rows_before_route_review"
    elif not no_cheat_ok:
        route_state = "fail_closed_no_cheat_violation"
        recommended_next_action = "invalidate_or_repair_noncompliant_child_reports"
    elif not integrity_ok:
        route_state = "fail_closed_candidate_integrity_mismatch"
        recommended_next_action = "repair_candidate_integrity_before_route_review"
    elif not quality_ok:
        route_state = "fail_closed_behavior_quality_zero"
        recommended_next_action = "repair_semantic_candidate_quality_before_production_route"
    elif not resource_enforced:
        route_state = "observation_only_no_resource_thresholds"
        recommended_next_action = "rerun_with_explicit_resource_thresholds_before_production_route"
    elif not resource_ok:
        route_state = "fail_closed_resource_budget"
        recommended_next_action = "optimize_or_route_around_slow_mlx_rung_before_production_route"
    else:
        route_state = "eligible_for_policy_review"
        recommended_next_action = "broaden_private_replay_rows_and_compare_against_cpu_cuda_reference_before_default_route"

    return {
        "policy": "project_theseus_mlx_rung_replay_route_eligibility_v1",
        "production_route_eligible": bool(qualification_inputs_met),
        "route_state": route_state,
        "recommended_next_action": recommended_next_action,
        "qualification_inputs_met": bool(qualification_inputs_met),
        "quality_gate": {
            "total_private_passes": total_private_passes,
            "total_eval_rows": total_eval_rows,
            "pass_rate": pass_rate,
            "quality_ok": bool(quality_ok),
        },
        "resource_gate": {
            "enforced": resource_enforced,
            "budget_ok": resource_ok,
            "total_budget_ok": bool(resource_budget.get("total_budget_ok")),
            "child_budget_ok": bool(resource_budget.get("child_budget_ok")),
            "throughput_ok": bool(resource_budget.get("throughput_ok")),
            "slow_child_decode_eval_ids": list(resource_budget.get("slow_child_decode_eval_ids") or []),
            "slow_child_decode_wall_ids": list(resource_budget.get("slow_child_decode_wall_ids") or []),
            "eval_rows_per_second": float(resource_budget.get("eval_rows_per_second") or 0.0),
            "max_child_decode_eval_runtime_ms": int(resource_budget.get("max_child_decode_eval_runtime_ms") or 0),
        },
        "runtime_reuse_gate": {
            "checkpoint_loader_policy": checkpoint_loader_summary.get("policy"),
            "checkpoint_weight_load_count": int(get_path(checkpoint_loader_summary, ["stats", "checkpoint_weight_load_count"], 0) or 0),
            "model_reuse_count": int(get_path(checkpoint_loader_summary, ["stats", "model_reuse_count"], 0) or 0),
            "verifier_cache_receipt_count": int(verifier_cache_summary.get("receipt_count") or 0),
            "total_test_harness_cache_hit_count": int(verifier_cache_summary.get("total_test_harness_cache_hit_count") or 0),
        },
        "integrity_gate": {
            "integrity_mismatch_count": integrity_mismatch_count,
            "integrity_ok": bool(integrity_ok),
        },
        "no_cheat_gate": {
            "public_training_rows": public_training_rows,
            "external_inference_calls": external_inference_calls,
            "fallback_template_router_tool_credit_count": fallback_credit,
            "no_cheat_ok": bool(no_cheat_ok),
        },
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit_count": 0,
        "score_semantics": (
            "Route-health receipt only. Production MLX routing is eligible only when private behavior, "
            "resource thresholds, integrity, and no-cheat counters all pass. It does not score public "
            "transfer or convert tools/templates/routers into learned-generation credit."
        ),
    }


def build_gates(
    checkpoints: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    *,
    checkpoint_selection_summary: dict[str, Any] | None = None,
    preselection_summary: dict[str, Any] | None = None,
    checkpoint_loader_summary: dict[str, Any] | None = None,
    verifier_cache_summary: dict[str, Any] | None = None,
    resource_budget_summary: dict[str, Any] | None = None,
    route_eligibility_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    selection = dict_or_empty(checkpoint_selection_summary)
    preselection = dict_or_empty(preselection_summary)
    loader = dict_or_empty(checkpoint_loader_summary)
    loader_stats = dict_or_empty(loader.get("stats"))
    verifier_cache = dict_or_empty(verifier_cache_summary)
    resource_budget = dict_or_empty(resource_budget_summary)
    route_eligibility = dict_or_empty(route_eligibility_summary)
    quality_gate = dict_or_empty(route_eligibility.get("quality_gate"))
    resource_gate = dict_or_empty(route_eligibility.get("resource_gate"))
    integrity_gate = dict_or_empty(route_eligibility.get("integrity_gate"))
    no_cheat_gate = dict_or_empty(route_eligibility.get("no_cheat_gate"))
    route_qualified = (
        bool(rows)
        and bool(resource_gate.get("enforced"))
        and bool(resource_gate.get("budget_ok"))
        and int(quality_gate.get("total_private_passes") or 0) > 0
        and bool(integrity_gate.get("integrity_ok"))
        and bool(no_cheat_gate.get("no_cheat_ok"))
    )
    return [
        gate("checkpoint_records_present", bool(checkpoints), "hard", len(checkpoints)),
        gate("all_requested_reports_written", len(rows) == len(checkpoints), "hard", {"rows": len(rows), "checkpoints": len(checkpoints)}),
        gate(
            "all_available_checkpoints_selected_when_unbounded",
            bool(selection.get("bounded_canary"))
            or int(selection.get("selected_checkpoint_count") or 0) == int(selection.get("available_checkpoint_count") or 0),
            "hard",
            selection,
        ),
        gate("public_training_rows_zero", all(int(get_path(row, ["summary", "public_training_rows"], 0) or 0) == 0 for row in reports), "hard", 0),
        gate("external_inference_zero", all(int(get_path(row, ["summary", "external_inference_calls"], 0) or 0) == 0 for row in reports), "hard", 0),
        gate(
            "no_template_router_tool_credit",
            all(int(get_path(row, ["summary", "fallback_template_router_tool_credit_count"], 0) or 0) == 0 for row in reports),
            "hard",
            0,
        ),
        gate(
            "all_integrity_mismatch_counts_zero",
            all(int(row.get("integrity_mismatch_count") or 0) == 0 for row in rows),
            "hard",
            {row.get("id"): row.get("integrity_mismatch_count") for row in rows},
        ),
        gate(
            "at_least_one_child_decode_green_or_yellow",
            any(str(row.get("child_trigger_state") or "") in {"GREEN", "YELLOW"} for row in rows),
            "soft",
            {row.get("id"): row.get("child_trigger_state") for row in rows},
        ),
        gate(
            "private_split_preselection_reuse_receipt_present",
            not bool(preselection.get("enabled")) or int(preselection.get("selected_split_count") or 0) > 0,
            "soft",
            preselection,
        ),
        gate(
            "checkpoint_loader_reloads_each_checkpoint",
            not bool(loader.get("enabled"))
            or int(loader_stats.get("checkpoint_weight_load_count") or 0) == len(checkpoints),
            "hard",
            loader,
        ),
        gate(
            "private_verifier_cache_warmup_receipts_present",
            not rows or int(verifier_cache.get("receipt_count") or 0) >= len(rows),
            "soft",
            verifier_cache,
        ),
        gate(
            "resource_budget_thresholds_respected",
            not bool(resource_budget.get("enforced")) or bool(resource_budget.get("budget_ok")),
            "hard",
            resource_budget,
        ),
        gate(
            "mlx_route_eligibility_fail_closed_when_unqualified",
            bool(route_eligibility)
            and (bool(route_eligibility.get("production_route_eligible")) == bool(route_qualified)),
            "hard",
            route_eligibility,
        ),
        gate(
            "production_route_requires_private_pass_and_budget",
            not bool(route_eligibility.get("production_route_eligible"))
            or (
                bool(resource_gate.get("enforced"))
                and bool(resource_gate.get("budget_ok"))
                and int(quality_gate.get("total_private_passes") or 0) > 0
            ),
            "hard",
            route_eligibility,
        ),
    ]


def gate(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def safe_slug(value: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value or "").strip())
    return slug[:120] or stable_hash(str(value or ""))[:16]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
