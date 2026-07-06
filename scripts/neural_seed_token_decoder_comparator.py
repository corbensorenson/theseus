#!/usr/bin/env python3
"""Matched token-level code decoder comparator for neural seed work.

This script does two bounded private tasks:

1. Re-score the existing body-template candidate-code comparator at task level
   and write a residual gap report.
2. Train matched tiny token decoders for SymLiquid-style and transformer arms,
   emit real token-decoded candidate code rows, and score them through the same
   private verifier path.

It does not run public calibration, live teacher calls, promotion, network
fetches, or runtime external serving. Governed teacher-distillation rows may be
used only when the config enables them and the manifest/gate admits private,
execution-verified training rows.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import sys
import tempfile
import textwrap
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_verifier import run_any, runtime_tmp_dir  # noqa: E402
from neural_seed_code_proposer_comparator import (  # noqa: E402
    SymLiquidStyleClassifier,
    TinyTransformerClassifier,
    build_vocab,
    candidate_counts_by_arm,
    choose_symliquid_dims as choose_symliquid_classifier_dims,
    candidate_rows_do_not_embed_forbidden,
    count_params,
    deterministic_sample,
    dict_or_empty,
    encode_many,
    get_path,
    import_torch,
    load_private_rows,
    maxrss_mb,
    mlx_status,
    ratio,
    rel,
    render_private_function,
    row_id,
    row_text,
    select_torch_device,
    stable_hash,
    syntax_summary,
    train_model as train_classifier_model,
)
from neural_seed_structural_action_decoder_probe import (  # noqa: E402
    action_sequence_id,
    build_action_sequence_library,
    rank_action_sequences,
    structural_candidate_rows,
    structural_source_text,
)
from code_lm_private_verifier import evaluate_private_candidates  # noqa: E402
from neural_seed_token_decoder_support import (  # noqa: E402
    baseline_candidate,
    body_like_target_mode,
    body_tokens_for_target_mode,
    build_target_vocab,
    candidate_row,
    choose_grammar_constrained_token,
    decode_body_tokens,
    decode_candidate_body_tokens,
    encode_target_rows,
    function_body_syntax,
    grammar_constrained_token_choices,
    grammar_repaired_body,
    normalize_body_text,
    render_semantic_slot_body,
    return_shape_for_task,
    semantic_plan_from_body,
    syntax_complete_body_prefix,
    target_tokens,
    token_allowed_by_policy,
    visible_contract_semantic_beams,
    visible_contract_semantic_plan,
)
from neural_seed_teacher_distillation_rows import (  # noqa: E402
    load_governed_teacher_code_lm_training_rows,
    record_verified_self_generated_rows,
    strict_report_holdout_families,
    teacher_code_lm_training_row_decision,
    teacher_code_lm_utility_decision,
    teacher_code_lm_utility_quarantine_summary,
    verified_self_trace_decision,
)
from neural_seed_full_state_pretraining import (  # noqa: E402
    build_full_state_pretraining_rows,
    collect_full_state_python_examples,
    corpus_function_body_quality,
    corpus_pretraining_quality_filter_summary,
    corpus_pretraining_quality_reject_reason,
    full_state_pretraining_config,
    full_state_pretraining_vocab_bodies,
    full_state_pretraining_vocab_source_texts,
    full_state_source_vocab_extension_summary,
    full_state_target_vocab_extension_summary,
    python_function_body_pretraining_examples,
    python_function_body_pretraining_source_text,
    source_summary_tokens,
)
from neural_seed_token_model_backend import (  # noqa: E402
    SymLiquidTokenDecoder,
    TransformerTokenDecoder,
    apply_arm_training_row_cap,
    apply_pretraining_initialization,
    apply_strict_generator_checkpoint,
    build_pretraining_initializers,
    choose_sym_token_dims,
    encode_pretraining_text,
    evaluate_token_model_loss,
    grammar_allowed_token_ids_for_prefix,
    grammar_validity_auxiliary_loss,
    initialize_embedding_module_from_bpe,
    load_pretraining_initializer_for_arm,
    load_strict_generator_checkpoint_vocab_override,
    model_parameter_inventory,
    model_parameter_snapshot,
    model_parameter_update_summary,
    normalize_checkpoint_vocab,
    pretraining_initialization_config,
    pretraining_initialization_report_summary,
    semantic_token_loss_weight_matrix,
    semantic_token_loss_weight_report,
    strict_generator_checkpoint_config,
    strict_generator_vocab_override_report,
    token_to_pretraining_text,
    train_token_model,
    training_budget_for_arm,
)
from neural_seed_candidate_generation import (  # noqa: E402
    decode_beam_sort_key,
    direct_generator_vcm_smoke,
    direct_generator_viea_records,
    final_decode_beam_sort_key,
    generate_candidates,
    merge_decoded_candidates,
    unconstrained_token_choices,
)
from neural_seed_report_io import (  # noqa: E402
    gate,
    is_relative_to,
    now,
    planned_report,
    read_json,
    read_jsonl,
    render_gap_markdown,
    render_markdown,
    resolve,
    stable_hash_file,
    token_score_semantics,
    write_json,
    write_jsonl,
    write_text,
)
DEFAULT_CONFIG = ROOT / "configs" / "neural_seed_token_decoder_comparator.json"
DEFAULT_OUT = ROOT / "reports" / "neural_seed_token_decoder_comparator.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "neural_seed_token_decoder_comparator.md"
DEFAULT_CANDIDATES = ROOT / "reports" / "neural_seed_token_decoder_candidates.jsonl"
TOKEN_DECODER_ARM_IDS = ("symliquid_style", "transformer_control")

from neural_seed_visible_source import (  # noqa: E402
    broad_private_heldout_manifest_rows,
    decoder_source_text,
    decoder_source_text_policy,
    deterministic_family_balanced_sample,
    family_disjoint_manifest_rows,
    family_disjoint_split_audit,
    strict_disjoint_family_key,
    visible_callable_argument_names,
    visible_callable_signature,
    visible_identifier_parts,
    visible_keyword_match,
    visible_prompt_intent_tags,
    visible_prompt_operation_tags,
    visible_prompt_type_shape_tags,
    visible_subword_parts,
)
from neural_seed_static_coherence import (  # noqa: E402
    allowed_signature_names_for_task,
    assignment_has_invalid_multi_target_scalar_unpack,
    assignment_target_names,
    candidate_static_coherence,
    expression_complexity_score,
    expression_contains_bare_builtin_reference,
    expression_direct_name,
    expression_has_bare_builtin_condition,
    expression_has_builtin_type_descriptor_receiver,
    expression_has_invalid_known_builtin_arg_type,
    expression_has_invalid_known_builtin_arity,
    expression_has_invalid_known_local_call,
    expression_has_invalid_known_local_iter,
    expression_has_invalid_known_local_receiver,
    expression_has_invalid_known_method_arity,
    expression_has_invalid_literal_receiver,
    expression_has_invalid_method_receiver,
    expression_hygiene_counts,
    expression_is_ignored_pure_call,
    expression_is_likely_noniterable,
    expression_is_no_effect,
    expression_is_parameter_copy_call,
    expression_is_parameter_free_literal_expression,
    expression_is_parameter_identity_copy,
    expression_is_static_literal_only,
    expression_loaded_names,
    expression_static_type,
    expression_uses_any_name,
    function_arg_names,
    local_static_type_bindings,
    max_repeated_identical_condition_chain,
    mutating_method_return_value_calls,
    return_value_is_trivial,
    self_dependent_assignment_load_count,
    static_coherence_candidate_pool_size,
    static_coherence_ranker_enabled,
    static_coherence_sort_key,
)
from neural_seed_candidate_evidence_summary import (  # noqa: E402
    allowed_generated_candidate_modes,
    body_decode_metadata_recorded,
    body_structure_counts,
    body_structure_summary,
    both_arms_emit_token_code,
    candidate_contains_task_identity,
    candidate_generation_modes_allowed,
    candidate_schema_summary,
    fallback_return_rate_zero,
    grammar_repair_counts,
    grammar_repair_metadata_recorded,
    grammar_repair_summary,
    no_cheat_candidate_evidence,
    no_cheat_exclusion_reasons,
    post_repair_syntax_pass_nonzero,
    raw_syntax_measured,
    raw_syntax_pass_nonzero,
    static_coherence_summary,
    syntax_evidence,
    terminal_null_return_candidate,
)
from neural_seed_route_memory import (  # noqa: E402
    CONTRACT_FEATURE_WEIGHTS,
    CONTRACT_FINGERPRINT_SCOPES,
    CONTRACT_FINGERPRINT_SCOPE_WEIGHTS,
    build_contract_feature_route_memory,
    build_contract_fingerprint_route_memory,
    build_learned_plan_route_memory,
    build_visible_text_plan_route_memory,
    contract_feature_counts,
    contract_feature_route_memory_summary,
    contract_fingerprint_route_keys,
    contract_fingerprint_route_memory_summary,
    deterministic_keep,
    generate_contract_feature_route_candidates,
    generate_contract_fingerprint_route_candidates,
    generate_learned_semantic_route_candidates,
    generate_visible_text_prototype_route_candidates,
    learned_plan_route_memory_summary,
    sparse_cosine,
    text_feature_counts,
    visible_text_plan_route_memory_summary,
)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--candidate-manifest-out", default=str(DEFAULT_CANDIDATES.relative_to(ROOT)))
    parser.add_argument("--gap-out", default="")
    parser.add_argument("--gap-markdown-out", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gap-only", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    config = read_json(resolve(args.config))
    gap_cfg = dict_or_empty(config.get("gap_report"))
    gap_out = args.gap_out or str(gap_cfg.get("out") or "reports/neural_seed_code_proposer_gap_report.json")
    gap_markdown = args.gap_markdown_out or str(gap_cfg.get("markdown_out") or "reports/neural_seed_code_proposer_gap_report.md")
    gap_report = build_gap_report(config, gap_out, gap_markdown)

    if args.gap_only:
        print(json.dumps(gap_report, indent=2))
        return 0 if gap_report.get("trigger_state") in {"GREEN", "YELLOW"} else 2
    if not args.execute:
        report = planned_report(config, args.config, gap_report)
        direct_vcm_smoke = direct_generator_vcm_smoke()
        report["direct_generator_vcm_smoke"] = direct_vcm_smoke
        report["summary"]["direct_generator_vcm_context_ready"] = bool(direct_vcm_smoke.get("ready"))
        report["summary"]["direct_generator_vcm_context_adequacy_state"] = get_path(
            direct_vcm_smoke,
            ["metadata", "vcm_context_adequacy_state"],
            "",
        )
        report["gates"] = [
            gate(
                "direct_generator_vcm_context_ready",
                bool(direct_vcm_smoke.get("ready")),
                {
                    "receipt_id": get_path(direct_vcm_smoke, ["receipt", "receipt_id"], ""),
                    "adequacy_state": get_path(direct_vcm_smoke, ["metadata", "vcm_context_adequacy_state"], ""),
                },
                "hard",
            )
        ]
        report["viea_direct_generator_records"] = direct_generator_viea_records(
            direct_vcm_smoke,
            report_ref=rel(resolve(args.out)),
        )
    else:
        report = run_token_decoder_comparator(config, args.config, args.candidate_manifest_out, gap_report, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def build_gap_report(config: dict[str, Any], out: str, markdown_out: str) -> dict[str, Any]:
    started = time.perf_counter()
    gap_cfg = dict_or_empty(config.get("gap_report"))
    source_report_path = resolve(str(gap_cfg.get("source_report") or "reports/neural_seed_code_proposer_comparator.json"))
    source_report = read_json(source_report_path)
    source_config = read_json(resolve(str(source_report.get("config") or "configs/neural_seed_code_proposer_comparator.json")))
    source_data = dict_or_empty(source_config.get("data"))
    seed = int((get_path(source_config, ["matched_budget", "seeds"], [23]) or [23])[0])
    eval_rows_all = load_private_rows(resolve(str(source_data.get("eval_jsonl") or "")), source_data)
    eval_rows = deterministic_sample(eval_rows_all, int(source_data.get("max_eval_rows") or 96), seed + 1009)
    train_rows = load_private_rows(resolve(str(source_data.get("train_jsonl") or "")), source_data)
    train_template_hashes = {stable_hash(str(row.get("solution_body") or "").strip()) for row in train_rows if row.get("solution_body")}
    candidate_path = resolve(str(gap_cfg.get("source_candidate_manifest") or get_path(source_report, ["summary", "candidate_manifest"], "")))
    candidate_rows = read_jsonl(candidate_path)
    by_arm_task_phase: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_arm_task_phase[
            (
                str(row.get("substrate_arm") or ""),
                str(row.get("task_id") or ""),
                str(row.get("phase") or ""),
            )
        ].append(row)
    for rows in by_arm_task_phase.values():
        rows.sort(key=lambda row: (int(row.get("rank") or 999), -float(row.get("rank_score") or 0.0)))

    task_rows: list[dict[str, Any]] = []
    family_totals: Counter[str] = Counter()
    family_passes: dict[str, Counter[str]] = defaultdict(Counter)
    family_sts_off_passes: dict[str, Counter[str]] = defaultdict(Counter)
    gap_counts: Counter[str] = Counter()
    confusion_counts: Counter[str] = Counter()
    failure_cause_counts: Counter[str] = Counter()
    sts_repair_counts: Counter[str] = Counter()
    sts_regression_counts: Counter[str] = Counter()

    old_timeout = os.environ.get("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS")
    os.environ["THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS"] = str(
        max(1, int(get_path(source_config, ["matched_budget", "private_candidate_timeout_seconds"], 4) or 4))
    )
    try:
        with tempfile.TemporaryDirectory(prefix="theseus_gap_report_", dir=runtime_tmp_dir()) as tmp:
            root = Path(tmp)
            for task in eval_rows:
                task_id = str(task.get("task_id") or "")
                family = gap_family(task)
                family_totals[family] += 1
                row: dict[str, Any] = {
                    "task_id": task_id,
                    "entry_point": task.get("entry_point"),
                    "family": family,
                    "expected_body_in_train_template_library": stable_hash(str(task.get("solution_body") or "").strip()) in train_template_hashes,
                    "arms": {},
                }
                for arm in ["symliquid_style", "transformer_control"]:
                    sts_on_candidates = by_arm_task_phase.get((arm, task_id, "private_eval"), [])
                    sts_off_candidates = by_arm_task_phase.get((arm, task_id, "private_eval_sts_off"), [])
                    baseline_candidates = by_arm_task_phase.get((arm, task_id, "private_baseline"), [])
                    on = run_any(root, task, sts_on_candidates, phase="private_eval")
                    off = run_any(root, task, sts_off_candidates, phase="private_eval_sts_off")
                    baseline = run_any(root, task, baseline_candidates, phase="private_baseline")
                    top = sts_on_candidates[0] if sts_on_candidates else {}
                    family_passes[arm][family] += int(bool(on.get("passed")))
                    family_sts_off_passes[arm][family] += int(bool(off.get("passed")))
                    if on.get("passed") and not off.get("passed"):
                        sts_repair_counts[arm] += 1
                    if off.get("passed") and not on.get("passed"):
                        sts_regression_counts[arm] += 1
                    if not on.get("passed"):
                        confusion_counts[f"{arm}:{top.get('template_id') or 'missing_top_candidate'}"] += 1
                        failure_cause_counts[classify_gap_failure(task, on, top, train_template_hashes)] += 1
                    row["arms"][arm] = {
                        "sts_on_passed": bool(on.get("passed")),
                        "sts_off_passed": bool(off.get("passed")),
                        "baseline_passed": bool(baseline.get("passed")),
                        "sts_on_stage": on.get("verification_stage"),
                        "sts_off_stage": off.get("verification_stage"),
                        "top_template_id": top.get("template_id"),
                        "top_rank_score": top.get("rank_score"),
                        "top_candidate_sha256": top.get("candidate_sha256"),
                        "failure_cause": classify_gap_failure(task, on, top, train_template_hashes) if not on.get("passed") else "passed",
                    }
                sym_pass = bool(get_path(row, ["arms", "symliquid_style", "sts_on_passed"], False))
                tx_pass = bool(get_path(row, ["arms", "transformer_control", "sts_on_passed"], False))
                if sym_pass and tx_pass:
                    status = "both_pass"
                elif (not sym_pass) and (not tx_pass):
                    status = "both_fail"
                elif tx_pass:
                    status = "transformer_only_win"
                else:
                    status = "symliquid_only_win"
                gap_counts[status] += 1
                row["gap_status"] = status
                task_rows.append(row)
    finally:
        if old_timeout is None:
            os.environ.pop("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS", None)
        else:
            os.environ["THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS"] = old_timeout

    family_movement = {}
    for family, total in sorted(family_totals.items()):
        family_movement[family] = {
            "task_count": total,
            "symliquid_sts_on_pass_rate": ratio(family_passes["symliquid_style"][family], total),
            "symliquid_sts_off_pass_rate": ratio(family_sts_off_passes["symliquid_style"][family], total),
            "transformer_sts_on_pass_rate": ratio(family_passes["transformer_control"][family], total),
            "transformer_sts_off_pass_rate": ratio(family_sts_off_passes["transformer_control"][family], total),
        }
    hard_gates = [
        gate("source_report_loaded", bool(source_report), rel(source_report_path), "hard"),
        gate("candidate_manifest_loaded", len(candidate_rows) > 0, {"path": rel(candidate_path), "rows": len(candidate_rows)}, "hard"),
        gate("eval_rows_loaded", len(eval_rows) > 0, len(eval_rows), "hard"),
        gate("task_gap_rows_recorded", len(task_rows) == len(eval_rows), {"tasks": len(task_rows), "eval_rows": len(eval_rows)}, "hard"),
        gate("external_inference_zero", True, 0, "hard"),
    ]
    trigger = "GREEN" if all(row["passed"] for row in hard_gates) else "RED"
    report = {
        "policy": "project_theseus_neural_seed_code_proposer_gap_report_v0",
        "created_utc": now(),
        "trigger_state": trigger,
        "source_report": rel(source_report_path),
        "source_candidate_manifest": rel(candidate_path),
        "summary": {
            "eval_rows": len(eval_rows),
            "candidate_rows": len(candidate_rows),
            "gap_counts": dict(gap_counts),
            "sts_repairs": dict(sts_repair_counts),
            "sts_regressions": dict(sts_regression_counts),
            "top_confusions": dict(confusion_counts.most_common(16)),
            "failure_cause_counts": dict(failure_cause_counts),
            "eval_solution_used_for_coverage_diagnostics_only": bool(gap_cfg.get("score_eval_solutions_for_coverage_diagnostics_only", True)),
            "external_inference_calls": 0,
            "teacher_used": False,
            "public_training_rows": 0,
        },
        "family_movement": family_movement,
        "task_gap_examples": {
            "both_pass": first_examples(task_rows, "both_pass"),
            "both_fail": first_examples(task_rows, "both_fail"),
            "transformer_only_win": first_examples(task_rows, "transformer_only_win"),
            "symliquid_only_win": first_examples(task_rows, "symliquid_only_win"),
        },
        "task_rows": task_rows,
        "gates": hard_gates,
        "score_semantics": (
            "Private diagnostic gap report over existing candidate-code comparator rows. Eval tests and "
            "private eval solution hashes are used only for diagnostics, not generation or training. No "
            "public calibration, teacher call, distillation, promotion, or external inference occurred."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }
    write_json(resolve(out), report)
    write_text(resolve(markdown_out), render_gap_markdown(report))
    return report


def run_token_decoder_comparator(
    config: dict[str, Any],
    config_path: str,
    candidate_manifest_out: str,
    gap_report: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    torch, nn = import_torch()
    data_cfg = dict_or_empty(config.get("data"))
    budget = dict_or_empty(config.get("matched_budget"))
    text_views = dict_or_empty(config.get("text_views"))
    structure_cfg = dict_or_empty(config.get("body_structure_decoder"))
    target_mode = str(structure_cfg.get("target_mode") or "statement_skeleton_v1")
    seed = int((budget.get("seeds") or [23])[0])
    routing_cfg = internal_semantic_routing_summary(config)
    random.seed(seed)
    torch.manual_seed(seed)

    base_train_rows_all = load_private_rows(resolve(str(data_cfg.get("train_jsonl") or "")), data_cfg)
    eval_rows_all = load_private_rows(resolve(str(data_cfg.get("eval_jsonl") or "")), data_cfg)
    teacher_training = load_governed_teacher_code_lm_training_rows(config)
    train_rows_all = [*base_train_rows_all, *teacher_training["rows"]]
    max_train_rows = int(data_cfg.get("max_train_rows") or 512)
    if teacher_training["rows"]:
        base_limit = max(1, max_train_rows - len(teacher_training["rows"]))
        train_rows = [
            *deterministic_sample(base_train_rows_all, base_limit, seed),
            *teacher_training["rows"],
        ]
    else:
        train_rows = deterministic_sample(base_train_rows_all, max_train_rows, seed)
    eval_rows = deterministic_sample(eval_rows_all, int(data_cfg.get("max_eval_rows") or 24), seed + 1009)
    target_vocab_extension_bodies = full_state_pretraining_vocab_bodies(config, seed=seed)
    source_vocab_extension_texts = full_state_pretraining_vocab_source_texts(config, seed=seed)
    source_vocab = build_vocab(
        [
            decoder_source_text(row, text_views.get(view, []))
            for view in ["sts_off", "sts_on"]
            for row in train_rows
        ]
        + source_vocab_extension_texts,
        max_vocab=int(budget.get("max_source_vocab") or 1024),
    )
    source_vocab_extension_summary = full_state_source_vocab_extension_summary(source_vocab_extension_texts)
    target_vocab = build_target_vocab(
        [str(row.get("solution_body") or "") for row in train_rows] + target_vocab_extension_bodies,
        max_vocab=int(budget.get("max_target_vocab") or 512),
        target_mode=target_mode,
    )
    strict_generator_vocab_override = load_strict_generator_checkpoint_vocab_override(config, torch=torch)
    if bool(strict_generator_vocab_override.get("active")):
        source_vocab = dict_or_empty(strict_generator_vocab_override.get("source_vocab"))
        target_vocab = dict_or_empty(strict_generator_vocab_override.get("target_vocab"))
    target_vocab_extension_summary = full_state_target_vocab_extension_summary(target_vocab_extension_bodies)
    max_source = int(budget.get("max_source_tokens") or 96)
    max_target = int(budget.get("max_target_tokens") or 96)
    device, backend_note = select_torch_device(torch)
    mlx = mlx_status()
    pretraining_initializers = build_pretraining_initializers(config, torch=torch, device=device)
    full_state_pretraining_cfg = full_state_pretraining_config(config)
    full_state_pretraining_rows = build_full_state_pretraining_rows(
        config,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        max_source=max_source,
        max_target=max_target,
        target_mode=target_mode,
        seed=seed,
    )

    transformer_cfg = dict_or_empty(dict_or_empty(config.get("arms")).get("transformer_control"))
    transformer_dims = {
        "d_model": int(transformer_cfg.get("d_model") or 48),
        "nhead": int(transformer_cfg.get("nhead") or 2),
        "num_layers": int(transformer_cfg.get("num_layers") or 1),
        "dim_feedforward": int(transformer_cfg.get("dim_feedforward") or 96),
    }
    transformer_param_count = count_params(
        TransformerTokenDecoder(
            len(source_vocab),
            len(target_vocab),
            max_source_len=max_source,
            max_target_len=max_target,
            **transformer_dims,
            torch=torch,
            nn=nn,
        )
    )
    sym_dims, sym_param_count = choose_sym_token_dims(
        config,
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        target_params=transformer_param_count,
        torch=torch,
        nn=nn,
    )
    param_delta = abs(sym_param_count - transformer_param_count) / max(1, transformer_param_count)
    structural_context = build_structural_action_family_context(
        config,
        train_rows=train_rows,
        eval_rows=eval_rows,
        text_views=text_views,
        budget=budget,
        torch=torch,
        nn=nn,
        device=device,
    )

    all_candidates: list[dict[str, Any]] = []
    arm_reports: dict[str, Any] = {}
    residual_context = residual_mining_context(gap_report)
    old_timeout = os.environ.get("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS")
    os.environ["THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS"] = str(
        max(1, int(budget.get("private_candidate_timeout_seconds") or 4))
    )
    try:
        for arm_id in execution_arm_names(config, split="in_family"):
            arm_started = time.perf_counter()
            arm_candidates = [baseline_candidate(task, arm_id=arm_id, config=config, seed=seed) for task in eval_rows]
            view_reports: dict[str, Any] = {}
            for view in arm_view_names(config, arm_id, default=["sts_off", "sts_on"]):
                view_started = time.perf_counter()
                torch.manual_seed(seed)
                random.seed(seed)
                view_routing = internal_semantic_routing_summary(config, view=view)
                plan_router_scale = float(view_routing.get("plan_router_scale") or 0.0)
                plan_auxiliary_loss_weight = float(view_routing.get("auxiliary_plan_loss_weight") or 0.0)
                if arm_id == "symliquid_style":
                    model = SymLiquidTokenDecoder(
                        len(source_vocab),
                        len(target_vocab),
                        hidden_dim=sym_dims["hidden_dim"],
                        reservoir_dim=sym_dims["reservoir_dim"],
                        hv_dim=sym_dims["hv_dim"],
                        plan_router_scale=plan_router_scale,
                        torch=torch,
                        nn=nn,
                    )
                    parameter_count = sym_param_count
                    substrate = "torch_symliquid_style_recurrent_vsa_token_decoder"
                    dims = sym_dims
                else:
                    model = TransformerTokenDecoder(
                        len(source_vocab),
                        len(target_vocab),
                        max_source_len=max_source,
                        max_target_len=max_target,
                        plan_router_scale=plan_router_scale,
                        **transformer_dims,
                        torch=torch,
                        nn=nn,
                    )
                    parameter_count = transformer_param_count
                    substrate = "torch_transformer_encoder_token_decoder"
                    dims = transformer_dims
                model.to(device)
                pretraining_init_summary = apply_pretraining_initialization(
                    model,
                    source_vocab=source_vocab,
                    target_vocab=target_vocab,
                    initializer=dict_or_empty(dict_or_empty(pretraining_initializers.get("by_arm")).get(arm_id)),
                    torch=torch,
                )
                strict_generator_checkpoint_summary = apply_strict_generator_checkpoint(
                    model,
                    config=config,
                    arm_id=arm_id,
                    source_vocab=source_vocab,
                    target_vocab=target_vocab,
                    max_source=max_source,
                    max_target=max_target,
                    dims=dims,
                    torch=torch,
                    device=device,
                )
                full_state_pretraining_summary = run_full_state_pretraining_warmup(
                    model,
                    full_state_pretraining_rows,
                    full_state_pretraining_cfg,
                    arm_id=arm_id,
                    view=view,
                    split="in_family",
                    torch=torch,
                    device=device,
                    pad_id=target_vocab["<pad>"],
                    target_vocab=target_vocab,
                )
                train_x = encode_many([decoder_source_text(row, text_views.get(view, [])) for row in train_rows], source_vocab, max_source)
                text_route_memory = build_visible_text_plan_route_memory(
                    train_rows,
                    text_views.get(view, []),
                    enabled=bool(view_routing.get("enabled") and view_routing.get("visible_text_prototype_route_memory")),
                )
                contract_route_memory = build_contract_fingerprint_route_memory(
                    train_rows,
                    enabled=bool(view_routing.get("enabled") and view_routing.get("contract_fingerprint_route_memory")),
                )
                contract_feature_route_memory = build_contract_feature_route_memory(
                    train_rows,
                    enabled=bool(view_routing.get("enabled") and view_routing.get("contract_feature_route_memory")),
                )
                target_rows = encode_target_rows(
                    [str(row.get("solution_body") or "") for row in train_rows],
                    target_vocab,
                    max_target,
                    target_mode=target_mode,
                )
                arm_budget = training_budget_for_arm(config, arm_id, budget)
                train_x, target_rows = apply_arm_training_row_cap(train_x, target_rows, arm_budget)
                before_mem = maxrss_mb()
                print(
                    f"[strict-decoder] train split=in_family arm={arm_id} view={view} "
                    f"rows={len(train_x)} epochs={int(arm_budget.get('epochs') or 3)} "
                    f"dims={dims}",
                    file=sys.stderr,
                    flush=True,
                )
                train_summary = train_token_model(
                    model,
                    train_x,
                    target_rows,
                    arm_budget,
                    torch=torch,
                    device=device,
                    pad_id=target_vocab["<pad>"],
                    target_vocab=target_vocab,
                    plan_auxiliary_loss_weight=plan_auxiliary_loss_weight,
                    allowed_name_sets=[allowed_signature_names_for_task(row) for row in train_rows],
                )
                train_summary["pretraining_initialization"] = pretraining_init_summary
                train_summary["strict_generator_pretraining_checkpoint"] = strict_generator_checkpoint_summary
                train_summary["full_state_pretraining"] = full_state_pretraining_summary
                route_memory = build_learned_plan_route_memory(
                    model,
                    train_x,
                    target_rows,
                    target_vocab,
                    enabled=bool(view_routing.get("enabled") and view_routing.get("prototype_route_memory")),
                    torch=torch,
                    device=device,
                )
                if route_memory:
                    train_summary["learned_route_memory"] = learned_plan_route_memory_summary(route_memory)
                if contract_route_memory:
                    train_summary["contract_fingerprint_route_memory"] = contract_fingerprint_route_memory_summary(contract_route_memory)
                if contract_feature_route_memory:
                    train_summary["contract_feature_route_memory"] = contract_feature_route_memory_summary(contract_feature_route_memory)
                if text_route_memory:
                    train_summary["visible_text_route_memory"] = visible_text_plan_route_memory_summary(text_route_memory)
                eval_x = encode_many([decoder_source_text(row, text_views.get(view, [])) for row in eval_rows], source_vocab, max_source)
                phase = "private_eval" if view == "sts_on" else "private_eval_sts_off"
                candidate_pool_size = static_coherence_candidate_pool_size(config, budget)
                decoded = generate_candidates(
                    model,
                    eval_x,
                    target_vocab,
                    max_target_tokens=max_target,
                    fanout_top_k=candidate_pool_size,
                    grammar_top_k=int(budget.get("grammar_decode_top_k") or 64),
                    decode_beam_width=arm_decode_setting(config, arm_id, budget, "decode_beam_width"),
                    decode_branching_factor=arm_decode_setting(config, arm_id, budget, "decode_branching_factor"),
                    target_mode=target_mode,
                    body_token_decode_policy=str(
                        budget.get("body_token_decode_policy")
                        or budget.get("body_token_validity_policy")
                        or "lightweight_python_v1"
                    ),
                    allowed_name_sets=[allowed_signature_names_for_task(row) for row in eval_rows],
                    torch=torch,
                    device=device,
                )
                if view_routing.get("enabled"):
                    decoded = merge_decoded_candidates(
                        decoded,
                        generate_learned_semantic_route_candidates(
                            model,
                            eval_x,
                            target_vocab,
                            top_k=int(view_routing.get("learned_route_top_k") or 0),
                            route_memory=route_memory,
                            prototype_route_weight=float(view_routing.get("prototype_route_weight") or 0.0),
                            prototype_route_keep_rate=float(view_routing.get("prototype_route_keep_rate") or 0.0),
                            dropout_salt=f"{arm_id}:{view}:{seed}:context_prototype_route",
                            torch=torch,
                            device=device,
                        ),
                    )
                    decoded = merge_decoded_candidates(
                        decoded,
                        generate_contract_fingerprint_route_candidates(
                            eval_rows,
                            target_vocab,
                            contract_route_memory,
                            top_k=int(view_routing.get("contract_fingerprint_route_top_k") or 0),
                            keep_rate=float(view_routing.get("contract_fingerprint_route_keep_rate") or 0.0),
                            dropout_salt=f"{arm_id}:{view}:{seed}:contract_fingerprint_route",
                        ),
                    )
                    decoded = merge_decoded_candidates(
                        decoded,
                        generate_contract_feature_route_candidates(
                            eval_rows,
                            target_vocab,
                            contract_feature_route_memory,
                            top_k=int(view_routing.get("contract_feature_route_top_k") or 0),
                            keep_rate=float(view_routing.get("contract_feature_route_keep_rate") or 0.0),
                            dropout_salt=f"{arm_id}:{view}:{seed}:contract_feature_route",
                        ),
                    )
                    decoded = merge_decoded_candidates(
                        decoded,
                        generate_visible_text_prototype_route_candidates(
                            eval_rows,
                            text_views.get(view, []),
                            text_route_memory,
                            top_k=int(view_routing.get("visible_text_prototype_route_top_k") or 0),
                            keep_rate=float(view_routing.get("visible_text_prototype_route_keep_rate") or 0.0),
                            dropout_salt=f"{arm_id}:{view}:{seed}:visible_text_route",
                        ),
                    )
                rows = token_candidate_rows_for_view(
                    eval_rows,
                    decoded,
                    arm_id=arm_id,
                    substrate=substrate,
                    phase=phase,
                    view=view,
                    config=config,
                    seed=seed,
                    target_mode=target_mode,
                    residual_context=residual_context,
                    output_top_k=int(budget.get("fanout_top_k") or 2),
                )
                syntax = syntax_summary(rows)
                arm_candidates.extend(rows)
                view_reports[view] = {
                    "arm_id": arm_id,
                    "view": view,
                    "phase": phase,
                    "substrate": substrate,
                    "parameter_count": parameter_count,
                    "dims": dims,
                    "train": train_summary,
                    "candidate_rows": len(rows),
                    "candidate_tasks": len(eval_rows),
                    "fanout_top_k": int(budget.get("fanout_top_k") or 2),
                    "candidate_pool_size": candidate_pool_size,
                    "learned_semantic_route_top_k": int(view_routing.get("learned_route_top_k") or 0),
                    "candidate_syntax": syntax,
                    "grammar_repair": grammar_repair_summary(rows),
                    "static_coherence": static_coherence_summary(rows),
                    "body_structure": body_structure_summary(rows),
                    "ranker": "prompt_signature_static_coherence_then_sequence_log_probability",
                    "decoder_constraints": {
                        "grammar_constrained_top_k": int(budget.get("grammar_decode_top_k") or 64),
                        "decode_beam_width": arm_decode_setting(config, arm_id, budget, "decode_beam_width"),
                        "decode_branching_factor": arm_decode_setting(config, arm_id, budget, "decode_branching_factor"),
                        "candidate_pool_size": candidate_pool_size,
                        "body_token_decode_policy": str(
                            budget.get("body_token_decode_policy")
                            or budget.get("body_token_validity_policy")
                            or "lightweight_python_v1"
                        ),
                        "signature_name_mask": {
                            "enabled": True,
                            "policy": "visible_signature_other_extra_mask_v1",
                            "uses_eval_tests_or_solutions": False,
                            "uses_public_data": False,
                        },
                        "post_decode_repair_policy": "deterministic_python_body_repair_v1",
                        "static_coherence_ranker": dict_or_empty(
                            get_path(config, ["body_structure_decoder", "static_coherence_ranker"], {})
                        ),
                        "target_mode": target_mode,
                        "visible_contract_semantic_beam": visible_contract_semantic_beam_summary(config),
                        "internal_semantic_routing": view_routing,
                        "pretraining_initialization": pretraining_init_summary,
                        "full_state_pretraining": full_state_pretraining_summary,
                        "source_text_policy": decoder_source_text_policy(text_views.get(view, [])),
                    },
                    "backend": {
                        "framework": "torch",
                        "device": str(device),
                        **backend_note,
                        **mlx,
                    },
                    "memory": {
                        "maxrss_mb_before": before_mem,
                        "maxrss_mb_after": maxrss_mb(),
                    },
                    "wall_time_ms_before_verifier": int((time.perf_counter() - view_started) * 1000),
                }
                print(
                    f"[strict-decoder] decoded split=in_family arm={arm_id} view={view} "
                    f"candidate_rows={len(rows)} syntax={syntax.get('syntax_pass_rate')}",
                    file=sys.stderr,
                    flush=True,
                )
            if structural_context:
                structural_report, structural_rows = run_structural_action_family_for_arm(
                    config,
                    structural_context,
                    eval_rows=eval_rows,
                    arm_id=arm_id,
                    seed=seed,
                    torch=torch,
                    device=device,
                )
                arm_candidates.extend(structural_rows)
                view_reports["structural_action"] = structural_report
            verifier_started = time.perf_counter()
            private_eval = evaluate_private_candidates(eval_rows, arm_candidates)
            summary = arm_summary(view_reports, private_eval, int((time.perf_counter() - verifier_started) * 1000))
            arm_reports[arm_id] = {
                "summary": summary,
                "views": view_reports,
                "private_verifier": private_eval,
                "candidate_schema": candidate_schema_summary(arm_candidates),
                "wall_time_ms": int((time.perf_counter() - arm_started) * 1000),
            }
            all_candidates.extend(arm_candidates)
    finally:
        if old_timeout is None:
            os.environ.pop("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS", None)
        else:
            os.environ["THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS"] = old_timeout

    family_disjoint_report, family_disjoint_candidates = run_family_disjoint_eval(
        config,
        train_rows_all=train_rows_all,
        eval_rows_all=eval_rows_all,
        text_views=text_views,
        target_mode=target_mode,
        residual_context=residual_context,
        torch=torch,
        nn=nn,
        device=device,
    )
    all_candidates.extend(family_disjoint_candidates)
    broad_private_heldout_report, broad_private_heldout_candidates = run_broad_private_heldout_eval(
        config,
        train_rows_all=train_rows_all,
        text_views=text_views,
        target_mode=target_mode,
        residual_context=residual_context,
        torch=torch,
        nn=nn,
        device=device,
    )
    all_candidates.extend(broad_private_heldout_candidates)
    write_jsonl(resolve(candidate_manifest_out), all_candidates)
    comparisons = compare_arms(arm_reports)
    no_cheat = no_cheat_candidate_evidence(eval_rows, all_candidates)
    verified_self_generation = record_verified_self_generated_rows(
        config,
        no_cheat,
        candidate_manifest_out=candidate_manifest_out,
    )
    gates = build_token_gates(
        config,
        train_rows,
        eval_rows,
        target_vocab,
        arm_reports,
        all_candidates,
        param_delta,
        gap_report,
        no_cheat,
        family_disjoint_report,
        broad_private_heldout_report,
        teacher_training,
    )
    hard_pass = all(row["passed"] for row in gates if row["severity"] == "hard")
    trigger_state = "GREEN" if hard_pass else "RED"
    if trigger_state == "GREEN" and any(not row["passed"] for row in gates):
        trigger_state = "YELLOW"
    requested_arms = execution_arm_names(config, split="in_family")
    summary = {
        "comparison_level": config.get("comparison_level"),
        "token_decoder_smoke_ready": trigger_state in {"GREEN", "YELLOW"},
        "execution_scope": execution_scope_summary(config, split="in_family"),
        "requested_arms_emit_token_decoded_candidate_code_rows": both_arms_emit_token_code(arm_reports, requested_arms),
        "both_arms_emit_token_decoded_candidate_code_rows": both_arms_emit_token_code(arm_reports),
        "same_private_verifier_for_requested_arms": True,
        "same_private_verifier_for_both_arms": set(requested_arms) == set(TOKEN_DECODER_ARM_IDS),
        "train_rows": len(train_rows),
        "base_train_rows": len(train_rows) - int(teacher_training["summary"].get("accepted_code_lm_training_rows") or 0),
        "teacher_training": teacher_training["summary"],
        "eval_rows": len(eval_rows),
        "source_vocab_size": len(source_vocab),
        "target_vocab_size": len(target_vocab),
        "target_vocab_extension": target_vocab_extension_summary,
        "source_vocab_extension": source_vocab_extension_summary,
        "target_mode": target_mode,
        "max_source_tokens": max_source,
        "max_target_tokens": max_target,
        "candidate_manifest": rel(resolve(candidate_manifest_out)),
        "candidate_rows": len(all_candidates),
        "parameter_match_delta": round(param_delta, 6),
        "parameter_match_within_tolerance": param_delta <= float(config["matched_budget"].get("parameter_match_tolerance") or 0.12),
        "trusted_parameter_match": param_delta <= float(config["matched_budget"].get("trusted_parameter_match_tolerance") or 0.08),
        "symliquid_parameter_count": sym_param_count,
        "transformer_parameter_count": transformer_param_count,
        "best_sts_on_arm_by_verifier_pass_rate": comparisons.get("winner_by_sts_on_verifier_pass_rate"),
        "practical_survival_lane": comparisons.get("practical_survival_lane"),
        "symliquid_discovery_lane_status": comparisons.get("symliquid_discovery_lane_status"),
        "symliquid_minus_transformer_sts_on_verifier_pass_rate": comparisons.get("symliquid_minus_transformer_sts_on_verifier_pass_rate"),
        "symliquid_gap_vs_body_template": compare_against_body_template("symliquid_style", comparisons, gap_report),
        "transformer_gap_vs_body_template": compare_against_body_template("transformer_control", comparisons, gap_report),
        "residual_mining_context": residual_context.get("summary", {}),
        "no_cheat_evidence": no_cheat.get("summary", {}),
        "external_inference_calls": 0,
        "teacher_used": False,
        "teacher_training_used": bool(teacher_training["rows"]),
        "teacher_training_external_inference_calls": teacher_training["summary"].get("external_inference_calls", 0),
        "verified_self_generation": verified_self_generation,
        "public_training_rows": 0,
        "model_promotion_allowed": False,
        "internal_semantic_routing": routing_cfg,
        "pretraining_initialization": pretraining_initialization_report_summary(pretraining_initializers),
        "strict_generator_checkpoint_vocab_override": strict_generator_vocab_override_report(strict_generator_vocab_override),
        "strict_generator_pretraining_checkpoint": {
            "enabled": bool(strict_generator_checkpoint_config(config).get("enabled", False)),
            "configured": strict_generator_checkpoint_config(config),
            "score_semantics": (
                "Full strict generator checkpoints load complete model state for the configured arm. "
                "They are training artifacts only and do not count as templates, tools, routers, or rendered code."
            ),
        },
        "full_state_pretraining": full_state_pretraining_report_summary(full_state_pretraining_cfg, full_state_pretraining_rows),
        "family_disjoint_eval": dict_or_empty(family_disjoint_report.get("summary")),
        "broad_private_heldout_eval": dict_or_empty(broad_private_heldout_report.get("summary")),
    }
    return {
        "policy": "project_theseus_neural_seed_token_decoder_comparator_report_v0",
        "created_utc": now(),
        "config": config_path,
        "trigger_state": trigger_state,
        "execute": True,
        "summary": summary,
        "gap_report": {
            "path": rel(resolve(str(get_path(config, ["gap_report", "out"], "reports/neural_seed_code_proposer_gap_report.json")))),
            "trigger_state": gap_report.get("trigger_state"),
            "summary": gap_report.get("summary", {}),
        },
        "data_contract": data_contract(config, train_rows, eval_rows, teacher_training),
        "body_structure_decoder": {
            "target_mode": target_mode,
            "target_vocab_size": len(target_vocab),
            "target_vocab_extension": target_vocab_extension_summary,
            "body_token_targets_from_private_train": target_mode == "body_tokens",
            "statement_skeleton_targets_from_private_train": target_mode == "statement_skeleton_v1",
            "semantic_slot_targets_from_private_train": target_mode == "semantic_slots_v1",
            "strict_action_targets_from_private_train": target_mode == "strict_action_tokens_v1",
            "strict_action_targets_promotion_evidence": False,
            "residual_mining_context": residual_context,
            "internal_semantic_routing": routing_cfg,
            "visible_contract_semantic_beam": visible_contract_semantic_beam_summary(config),
            "structural_action_family": structural_action_family_summary(config, structural_context),
        },
        "candidate_row_schema": config.get("candidate_row_schema", {}),
        "matched_budget": matched_budget_report(config, sym_dims, transformer_dims),
        "adapter_boundary": config.get("adapter_boundary", {}),
        "arms": arm_reports,
        "family_disjoint_eval": family_disjoint_report,
        "broad_private_heldout_eval": broad_private_heldout_report,
        "comparisons": comparisons,
        "no_cheat_evidence": no_cheat,
        "gates": gates,
        "score_semantics": token_score_semantics(target_mode),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def run_family_disjoint_eval(
    config: dict[str, Any],
    *,
    train_rows_all: list[dict[str, Any]],
    eval_rows_all: list[dict[str, Any]],
    text_views: dict[str, Any],
    target_mode: str,
    residual_context: dict[str, Any],
    torch: Any,
    nn: Any,
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    data_cfg = dict_or_empty(config.get("data"))
    disjoint_cfg = dict_or_empty(data_cfg.get("family_disjoint_eval"))
    if not bool(disjoint_cfg.get("enabled", False)):
        return {
            "enabled": False,
            "summary": {
                "enabled": False,
                "candidate_rows": 0,
                "reason": "family_disjoint_eval_disabled",
            },
        }, []

    budget = dict_or_empty(config.get("matched_budget"))
    seed = int(disjoint_cfg.get("split_seed") or (int((budget.get("seeds") or [23])[0]) + 711))
    min_holdout_families = max(1, int(disjoint_cfg.get("min_holdout_families") or 6))
    family_key_name = str(disjoint_cfg.get("family_key") or "concept_residual_label")
    train_families_all = {strict_disjoint_family_key(row, family_key_name) for row in train_rows_all}
    eval_families_all = {strict_disjoint_family_key(row, family_key_name) for row in eval_rows_all}
    common_families = sorted(family for family in train_families_all & eval_families_all if family and family != "unknown")
    ranked_families = sorted(common_families, key=lambda family: stable_hash(f"{seed}:family_disjoint:{family}"))
    holdout_families = ranked_families[:min_holdout_families]
    if len(holdout_families) < min_holdout_families:
        return {
            "enabled": True,
            "active": False,
            "summary": {
                "enabled": True,
                "active": False,
                "blocked_reason": "insufficient_common_families",
                "common_family_count": len(common_families),
                "required_holdout_family_count": min_holdout_families,
            },
        }, []

    holdout = set(holdout_families)
    teacher_rows_all = [
        row
        for row in train_rows_all
        if bool(row.get("teacher_generated")) or str(row.get("source_kind") or "").startswith("teacher")
    ]
    teacher_holdout_rows = [
        row
        for row in teacher_rows_all
        if strict_disjoint_family_key(row, family_key_name) in holdout
    ]
    disjoint_train_pool = [row for row in train_rows_all if strict_disjoint_family_key(row, family_key_name) not in holdout]
    disjoint_eval_pool = [row for row in eval_rows_all if strict_disjoint_family_key(row, family_key_name) in holdout]
    train_limit = int(disjoint_cfg.get("max_train_rows") or data_cfg.get("max_train_rows") or 512)
    eval_limit = int(disjoint_cfg.get("max_eval_rows") or data_cfg.get("max_eval_rows") or 24)
    train_rows = deterministic_sample(disjoint_train_pool, train_limit, seed)
    eval_rows = deterministic_sample(disjoint_eval_pool, eval_limit, seed + 1009)
    split_audit = family_disjoint_split_audit(
        train_rows,
        eval_rows,
        holdout_families=holdout_families,
        family_key_name=family_key_name,
        target_mode=target_mode,
    )
    if not train_rows or not eval_rows:
        return {
            "enabled": True,
            "active": False,
            "summary": {
                "enabled": True,
                "active": False,
                "blocked_reason": "empty_disjoint_train_or_eval_rows",
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
            },
            "split_audit": split_audit,
        }, []

    target_vocab_extension_bodies = full_state_pretraining_vocab_bodies(config, seed=seed + 9191)
    source_vocab_extension_texts = full_state_pretraining_vocab_source_texts(config, seed=seed + 9191)
    source_vocab = build_vocab(
        [
            decoder_source_text(row, text_views.get(view, []))
            for view in ["sts_off", "sts_on"]
            for row in train_rows
        ]
        + source_vocab_extension_texts,
        max_vocab=int(budget.get("max_source_vocab") or 1024),
    )
    source_vocab_extension_summary = full_state_source_vocab_extension_summary(source_vocab_extension_texts)
    target_vocab = build_target_vocab(
        [str(row.get("solution_body") or "") for row in train_rows] + target_vocab_extension_bodies,
        max_vocab=int(budget.get("max_target_vocab") or 512),
        target_mode=target_mode,
    )
    strict_generator_vocab_override = load_strict_generator_checkpoint_vocab_override(config, torch=torch)
    if bool(strict_generator_vocab_override.get("active")):
        source_vocab = dict_or_empty(strict_generator_vocab_override.get("source_vocab"))
        target_vocab = dict_or_empty(strict_generator_vocab_override.get("target_vocab"))
    target_vocab_extension_summary = full_state_target_vocab_extension_summary(target_vocab_extension_bodies)
    max_source = int(budget.get("max_source_tokens") or 96)
    max_target = int(budget.get("max_target_tokens") or 96)
    full_state_pretraining_cfg = full_state_pretraining_config(config)
    full_state_pretraining_rows = build_full_state_pretraining_rows(
        config,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        max_source=max_source,
        max_target=max_target,
        target_mode=target_mode,
        seed=seed + 9191,
    )
    transformer_cfg = dict_or_empty(dict_or_empty(config.get("arms")).get("transformer_control"))
    transformer_dims = {
        "d_model": int(transformer_cfg.get("d_model") or 48),
        "nhead": int(transformer_cfg.get("nhead") or 2),
        "num_layers": int(transformer_cfg.get("num_layers") or 1),
        "dim_feedforward": int(transformer_cfg.get("dim_feedforward") or 96),
    }
    transformer_param_count = count_params(
        TransformerTokenDecoder(
            len(source_vocab),
            len(target_vocab),
            max_source_len=max_source,
            max_target_len=max_target,
            **transformer_dims,
            torch=torch,
            nn=nn,
        )
    )
    sym_dims, sym_param_count = choose_sym_token_dims(
        config,
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        target_params=transformer_param_count,
        torch=torch,
        nn=nn,
    )
    param_delta = abs(sym_param_count - transformer_param_count) / max(1, transformer_param_count)
    pretraining_initializers = build_pretraining_initializers(config, torch=torch, device=device)
    all_local_candidates: list[dict[str, Any]] = []
    arm_reports: dict[str, Any] = {}
    old_timeout = os.environ.get("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS")
    os.environ["THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS"] = str(
        max(1, int(budget.get("private_candidate_timeout_seconds") or 4))
    )
    try:
        for arm_id in execution_arm_names(config, split="family_disjoint"):
            arm_started = time.perf_counter()
            arm_candidates = [baseline_candidate(task, arm_id=arm_id, config=config, seed=seed) for task in eval_rows]
            view_reports: dict[str, Any] = {}
            requested_views = [str(view) for view in disjoint_cfg.get("views", ["sts_on"]) or ["sts_on"]]
            requested_views = [view for view in requested_views if view in {"sts_off", "sts_on"}] or ["sts_on"]
            arm_views = set(arm_view_names(config, arm_id, default=["sts_on"]))
            requested_views = [view for view in requested_views if view in arm_views] or ["sts_on"]
            for view in requested_views:
                view_started = time.perf_counter()
                torch.manual_seed(seed)
                random.seed(seed)
                if arm_id == "symliquid_style":
                    model = SymLiquidTokenDecoder(
                        len(source_vocab),
                        len(target_vocab),
                        hidden_dim=sym_dims["hidden_dim"],
                        reservoir_dim=sym_dims["reservoir_dim"],
                        hv_dim=sym_dims["hv_dim"],
                        plan_router_scale=0.0,
                        torch=torch,
                        nn=nn,
                    )
                    parameter_count = sym_param_count
                    substrate = "torch_symliquid_style_recurrent_vsa_token_decoder"
                    dims = sym_dims
                else:
                    model = TransformerTokenDecoder(
                        len(source_vocab),
                        len(target_vocab),
                        max_source_len=max_source,
                        max_target_len=max_target,
                        plan_router_scale=0.0,
                        **transformer_dims,
                        torch=torch,
                        nn=nn,
                    )
                    parameter_count = transformer_param_count
                    substrate = "torch_transformer_encoder_token_decoder"
                    dims = transformer_dims
                model.to(device)
                pretraining_init_summary = apply_pretraining_initialization(
                    model,
                    source_vocab=source_vocab,
                    target_vocab=target_vocab,
                    initializer=dict_or_empty(dict_or_empty(pretraining_initializers.get("by_arm")).get(arm_id)),
                    torch=torch,
                )
                strict_generator_checkpoint_summary = apply_strict_generator_checkpoint(
                    model,
                    config=config,
                    arm_id=arm_id,
                    source_vocab=source_vocab,
                    target_vocab=target_vocab,
                    max_source=max_source,
                    max_target=max_target,
                    dims=dims,
                    torch=torch,
                    device=device,
                )
                full_state_pretraining_summary = run_full_state_pretraining_warmup(
                    model,
                    full_state_pretraining_rows,
                    full_state_pretraining_cfg,
                    arm_id=arm_id,
                    view=view,
                    split="family_disjoint",
                    torch=torch,
                    device=device,
                    pad_id=target_vocab["<pad>"],
                    target_vocab=target_vocab,
                )
                train_x = encode_many([decoder_source_text(row, text_views.get(view, [])) for row in train_rows], source_vocab, max_source)
                target_rows = encode_target_rows(
                    [str(row.get("solution_body") or "") for row in train_rows],
                    target_vocab,
                    max_target,
                    target_mode=target_mode,
                )
                arm_budget = training_budget_for_arm(config, arm_id, budget)
                train_x, target_rows = apply_arm_training_row_cap(train_x, target_rows, arm_budget)
                before_mem = maxrss_mb()
                print(
                    f"[strict-decoder] train split=family_disjoint arm={arm_id} view={view} "
                    f"rows={len(train_x)} epochs={int(arm_budget.get('epochs') or 3)} "
                    f"dims={dims}",
                    file=sys.stderr,
                    flush=True,
                )
                train_summary = train_token_model(
                    model,
                    train_x,
                    target_rows,
                    arm_budget,
                    torch=torch,
                    device=device,
                    pad_id=target_vocab["<pad>"],
                    target_vocab=target_vocab,
                    plan_auxiliary_loss_weight=0.0,
                    allowed_name_sets=[allowed_signature_names_for_task(row) for row in train_rows],
                )
                train_summary["pretraining_initialization"] = pretraining_init_summary
                train_summary["strict_generator_pretraining_checkpoint"] = strict_generator_checkpoint_summary
                train_summary["full_state_pretraining"] = full_state_pretraining_summary
                eval_x = encode_many([decoder_source_text(row, text_views.get(view, [])) for row in eval_rows], source_vocab, max_source)
                phase = "private_eval" if view == "sts_on" else "private_eval_sts_off"
                candidate_pool_size = static_coherence_candidate_pool_size(config, budget)
                decoded = generate_candidates(
                    model,
                    eval_x,
                    target_vocab,
                    max_target_tokens=max_target,
                    fanout_top_k=candidate_pool_size,
                    grammar_top_k=int(budget.get("grammar_decode_top_k") or 64),
                    decode_beam_width=arm_decode_setting(config, arm_id, budget, "decode_beam_width"),
                    decode_branching_factor=arm_decode_setting(config, arm_id, budget, "decode_branching_factor"),
                    target_mode=target_mode,
                    body_token_decode_policy=str(
                        budget.get("body_token_decode_policy")
                        or budget.get("body_token_validity_policy")
                        or "lightweight_python_v1"
                    ),
                    allowed_name_sets=[allowed_signature_names_for_task(row) for row in eval_rows],
                    torch=torch,
                    device=device,
                )
                rows = token_candidate_rows_for_view(
                    eval_rows,
                    decoded,
                    arm_id=arm_id,
                    substrate=substrate,
                    phase=phase,
                    view=view,
                    config=config,
                    seed=seed,
                    target_mode=target_mode,
                    residual_context=residual_context,
                    output_top_k=int(budget.get("fanout_top_k") or 2),
                )
                syntax = syntax_summary(rows)
                arm_candidates.extend(rows)
                view_reports[view] = {
                    "arm_id": arm_id,
                    "view": view,
                    "phase": phase,
                    "evaluation_split": "family_disjoint",
                    "substrate": substrate,
                    "parameter_count": parameter_count,
                    "dims": dims,
                    "train": train_summary,
                    "candidate_rows": len(rows),
                    "candidate_tasks": len(eval_rows),
                    "fanout_top_k": int(budget.get("fanout_top_k") or 2),
                    "candidate_pool_size": candidate_pool_size,
                    "candidate_syntax": syntax,
                    "grammar_repair": grammar_repair_summary(rows),
                    "static_coherence": static_coherence_summary(rows),
                    "body_structure": body_structure_summary(rows),
                    "ranker": "prompt_signature_static_coherence_then_sequence_log_probability",
                    "decoder_constraints": {
                        "grammar_constrained_top_k": int(budget.get("grammar_decode_top_k") or 64),
                        "decode_beam_width": arm_decode_setting(config, arm_id, budget, "decode_beam_width"),
                        "decode_branching_factor": arm_decode_setting(config, arm_id, budget, "decode_branching_factor"),
                        "candidate_pool_size": candidate_pool_size,
                        "body_token_decode_policy": str(
                            budget.get("body_token_decode_policy")
                            or budget.get("body_token_validity_policy")
                            or "lightweight_python_v1"
                        ),
                        "signature_name_mask": {
                            "enabled": True,
                            "policy": "visible_signature_other_extra_mask_v1",
                            "uses_eval_tests_or_solutions": False,
                            "uses_public_data": False,
                        },
                        "post_decode_repair_policy": "deterministic_python_body_repair_v1",
                        "static_coherence_ranker": dict_or_empty(
                            get_path(config, ["body_structure_decoder", "static_coherence_ranker"], {})
                        ),
                        "target_mode": target_mode,
                        "visible_contract_semantic_beam": visible_contract_semantic_beam_summary(config),
                        "internal_semantic_routing": {"enabled": False, "reason": "family_disjoint_strict_probe"},
                        "pretraining_initialization": pretraining_init_summary,
                        "full_state_pretraining": full_state_pretraining_summary,
                        "source_text_policy": decoder_source_text_policy(text_views.get(view, [])),
                    },
                    "backend": {"framework": "torch", "device": str(device), **mlx_status()},
                    "memory": {
                        "maxrss_mb_before": before_mem,
                        "maxrss_mb_after": maxrss_mb(),
                    },
                    "wall_time_ms_before_verifier": int((time.perf_counter() - view_started) * 1000),
                }
                print(
                    f"[strict-decoder] decoded split=family_disjoint arm={arm_id} view={view} "
                    f"candidate_rows={len(rows)} syntax={syntax.get('syntax_pass_rate')}",
                    file=sys.stderr,
                    flush=True,
                )
            verifier_started = time.perf_counter()
            private_eval = evaluate_private_candidates(eval_rows, arm_candidates)
            summary = arm_summary(view_reports, private_eval, int((time.perf_counter() - verifier_started) * 1000))
            arm_reports[arm_id] = {
                "summary": summary,
                "views": view_reports,
                "private_verifier": private_eval,
                "candidate_schema": candidate_schema_summary(arm_candidates),
                "wall_time_ms": int((time.perf_counter() - arm_started) * 1000),
            }
            all_local_candidates.extend(arm_candidates)
    finally:
        if old_timeout is None:
            os.environ.pop("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS", None)
        else:
            os.environ["THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS"] = old_timeout

    no_cheat = no_cheat_candidate_evidence(eval_rows, all_local_candidates)
    comparisons = compare_arms(arm_reports)
    manifest_rows = family_disjoint_manifest_rows(all_local_candidates)
    summary = {
        "enabled": True,
        "active": True,
        "execution_scope": execution_scope_summary(config, split="family_disjoint"),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "holdout_family_count": len(holdout_families),
        "holdout_families": holdout_families,
        "family_key": family_key_name,
        "candidate_rows": len(manifest_rows),
        "source_vocab_size": len(source_vocab),
        "source_vocab_extension": source_vocab_extension_summary,
        "target_vocab_size": len(target_vocab),
        "target_vocab_extension": target_vocab_extension_summary,
        "teacher_code_lm_training_rows_available": len(teacher_rows_all),
        "teacher_code_lm_training_rows_in_holdout_families": len(teacher_holdout_rows),
        "teacher_code_lm_training_holdout_exclusion_clean": len(teacher_holdout_rows) == 0,
        "parameter_match_delta": round(param_delta, 6),
        "transformer_sts_on_verifier_pass_rate": get_path(
            arm_reports, ["transformer_control", "summary", "sts_on_verifier_pass_rate"], 0.0
        ),
        "transformer_sts_on_rank1_pass_rate": get_path(
            arm_reports, ["transformer_control", "summary", "sts_on_rank1_pass_rate"], 0.0
        ),
        "transformer_sts_on_pass_if_any_rate": get_path(
            arm_reports, ["transformer_control", "summary", "sts_on_pass_if_any_rate"], 0.0
        ),
        "symliquid_sts_on_verifier_pass_rate": get_path(
            arm_reports, ["symliquid_style", "summary", "sts_on_verifier_pass_rate"], 0.0
        ),
        "symliquid_minus_transformer_sts_on_verifier_pass_rate": comparisons.get(
            "symliquid_minus_transformer_sts_on_verifier_pass_rate"
        ),
        "no_cheat_evidence": dict_or_empty(no_cheat.get("summary")),
        "split_overlap_audit": dict_or_empty(split_audit.get("overlap")),
        "external_inference_calls": 0,
        "teacher_used": False,
        "teacher_training_used": bool(teacher_rows_all),
        "teacher_training_external_inference_calls": sum(int(row.get("external_inference_calls") or 0) for row in teacher_rows_all),
        "public_training_rows": 0,
        "pretraining_initialization": pretraining_initialization_report_summary(pretraining_initializers),
        "strict_generator_checkpoint_vocab_override": strict_generator_vocab_override_report(strict_generator_vocab_override),
        "full_state_pretraining": full_state_pretraining_report_summary(full_state_pretraining_cfg, full_state_pretraining_rows),
    }
    return {
        "policy": "project_theseus_strict_decoder_family_disjoint_eval_v0",
        "enabled": True,
        "active": True,
        "created_utc": now(),
        "summary": summary,
        "split_audit": split_audit,
        "arms": arm_reports,
        "comparisons": comparisons,
        "no_cheat_evidence": no_cheat,
        "score_semantics": (
            "Family-disjoint private eval: selected semantic families are absent from train rows and present in eval rows. "
            "Family labels are used only for splitting/auditing, never as generator text features."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }, manifest_rows


def run_broad_private_heldout_eval(
    config: dict[str, Any],
    *,
    train_rows_all: list[dict[str, Any]],
    text_views: dict[str, Any],
    target_mode: str,
    residual_context: dict[str, Any],
    torch: Any,
    nn: Any,
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    data_cfg = dict_or_empty(config.get("data"))
    broad_cfg = dict_or_empty(data_cfg.get("broad_private_heldout_eval"))
    if not bool(broad_cfg.get("enabled", False)):
        return {
            "enabled": False,
            "summary": {"enabled": False, "candidate_rows": 0, "reason": "broad_private_heldout_eval_disabled"},
        }, []

    eval_paths_cfg = broad_cfg.get("eval_jsonl")
    if isinstance(eval_paths_cfg, list):
        eval_paths = [resolve(str(path)) for path in eval_paths_cfg]
    else:
        eval_paths = [resolve(str(eval_paths_cfg or ""))]
    eval_data_cfg = dict(data_cfg)
    eval_data_cfg["forbidden_row_flags"] = data_cfg.get("forbidden_row_flags", [])
    eval_rows_unfiltered: list[dict[str, Any]] = []
    seen_eval_keys: set[str] = set()
    for eval_path in eval_paths:
        for row in load_private_rows(eval_path, eval_data_cfg):
            key = str(row.get("task_id") or row.get("source_task_id") or stable_hash(json.dumps(row, sort_keys=True)))
            if key in seen_eval_keys:
                continue
            seen_eval_keys.add(key)
            eval_rows_unfiltered.append(row)
    budget = dict_or_empty(config.get("matched_budget"))
    seed = int(broad_cfg.get("split_seed") or (int((budget.get("seeds") or [23])[0]) + 17017))
    family_key_name = str(broad_cfg.get("family_key") or "concept_residual_label")
    train_limit = int(broad_cfg.get("max_train_rows") or data_cfg.get("max_train_rows") or 1024)
    eval_limit = int(broad_cfg.get("max_eval_rows") or 240)
    train_rows = deterministic_sample(train_rows_all, train_limit, seed)
    train_solution_hashes = {
        stable_hash(str(row.get("solution_body") or "").strip())
        for row in train_rows_all
        if str(row.get("solution_body") or "").strip()
    }
    train_families_all = {strict_disjoint_family_key(row, family_key_name) for row in train_rows_all}
    train_token_hashes = {
        stable_hash(" ".join(target_tokens(str(row.get("solution_body") or ""), target_mode=target_mode)))
        for row in train_rows_all
        if str(row.get("solution_body") or "").strip()
    }
    eval_rows_all = []
    overlap_rejected = 0
    family_overlap_rejected = 0
    for row in eval_rows_unfiltered:
        if strict_disjoint_family_key(row, family_key_name) in train_families_all:
            family_overlap_rejected += 1
            continue
        body = str(row.get("solution_body") or "").strip()
        body_hash = stable_hash(body) if body else ""
        token_hash = stable_hash(" ".join(target_tokens(body, target_mode=target_mode))) if body else ""
        if body_hash in train_solution_hashes or token_hash in train_token_hashes:
            overlap_rejected += 1
            continue
        eval_rows_all.append(row)
    eval_rows = deterministic_family_balanced_sample(
        eval_rows_all,
        eval_limit,
        seed + 1009,
        family_key_name=family_key_name,
    )
    split_audit = family_disjoint_split_audit(
        train_rows,
        eval_rows,
        holdout_families=sorted({strict_disjoint_family_key(row, family_key_name) for row in eval_rows}),
        family_key_name=family_key_name,
        target_mode=target_mode,
    )
    overlap = dict_or_empty(split_audit.get("overlap"))
    eval_family_count = int(split_audit.get("eval_family_count") or 0)
    min_eval_rows = int(broad_cfg.get("min_eval_rows") or 200)
    min_eval_families = int(broad_cfg.get("min_eval_families") or 30)
    if not train_rows or not eval_rows:
        return {
            "enabled": True,
            "active": False,
            "summary": {
                "enabled": True,
                "active": False,
                "blocked_reason": "empty_broad_train_or_eval_rows",
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
            },
            "split_audit": split_audit,
        }, []

    target_vocab_extension_bodies = full_state_pretraining_vocab_bodies(config, seed=seed + 17017)
    source_vocab_extension_texts = full_state_pretraining_vocab_source_texts(config, seed=seed + 17017)
    source_vocab = build_vocab(
        [
            decoder_source_text(row, text_views.get(view, []))
            for view in ["sts_off", "sts_on"]
            for row in train_rows
        ]
        + source_vocab_extension_texts,
        max_vocab=int(budget.get("max_source_vocab") or 1024),
    )
    source_vocab_extension_summary = full_state_source_vocab_extension_summary(source_vocab_extension_texts)
    target_vocab = build_target_vocab(
        [str(row.get("solution_body") or "") for row in train_rows] + target_vocab_extension_bodies,
        max_vocab=int(budget.get("max_target_vocab") or 512),
        target_mode=target_mode,
    )
    strict_generator_vocab_override = load_strict_generator_checkpoint_vocab_override(config, torch=torch)
    if bool(strict_generator_vocab_override.get("active")):
        source_vocab = dict_or_empty(strict_generator_vocab_override.get("source_vocab"))
        target_vocab = dict_or_empty(strict_generator_vocab_override.get("target_vocab"))
    target_vocab_extension_summary = full_state_target_vocab_extension_summary(target_vocab_extension_bodies)
    max_source = int(budget.get("max_source_tokens") or 96)
    max_target = int(budget.get("max_target_tokens") or 96)
    full_state_pretraining_cfg = full_state_pretraining_config(config)
    full_state_pretraining_rows = build_full_state_pretraining_rows(
        config,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        max_source=max_source,
        max_target=max_target,
        target_mode=target_mode,
        seed=seed + 17017,
    )
    transformer_cfg = dict_or_empty(dict_or_empty(config.get("arms")).get("transformer_control"))
    transformer_dims = {
        "d_model": int(transformer_cfg.get("d_model") or 48),
        "nhead": int(transformer_cfg.get("nhead") or 2),
        "num_layers": int(transformer_cfg.get("num_layers") or 1),
        "dim_feedforward": int(transformer_cfg.get("dim_feedforward") or 96),
    }
    model = TransformerTokenDecoder(
        len(source_vocab),
        len(target_vocab),
        max_source_len=max_source,
        max_target_len=max_target,
        plan_router_scale=0.0,
        **transformer_dims,
        torch=torch,
        nn=nn,
    )
    parameter_count = count_params(model)
    model.to(device)
    pretraining_initializers = build_pretraining_initializers(config, torch=torch, device=device)
    pretraining_init_summary = apply_pretraining_initialization(
        model,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        initializer=dict_or_empty(dict_or_empty(pretraining_initializers.get("by_arm")).get("transformer_control")),
        torch=torch,
    )
    strict_generator_checkpoint_summary = apply_strict_generator_checkpoint(
        model,
        config=config,
        arm_id="transformer_control",
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        max_source=max_source,
        max_target=max_target,
        dims=transformer_dims,
        torch=torch,
        device=device,
    )
    full_state_pretraining_summary = run_full_state_pretraining_warmup(
        model,
        full_state_pretraining_rows,
        full_state_pretraining_cfg,
        arm_id="transformer_control",
        view="sts_on",
        split="broad_private_heldout",
        torch=torch,
        device=device,
        pad_id=target_vocab["<pad>"],
        target_vocab=target_vocab,
    )
    train_x = encode_many([decoder_source_text(row, text_views.get("sts_on", [])) for row in train_rows], source_vocab, max_source)
    target_rows = encode_target_rows(
        [str(row.get("solution_body") or "") for row in train_rows],
        target_vocab,
        max_target,
        target_mode=target_mode,
    )
    train_summary = train_token_model(
        model,
        train_x,
        target_rows,
        budget,
        torch=torch,
        device=device,
        pad_id=target_vocab["<pad>"],
        target_vocab=target_vocab,
        plan_auxiliary_loss_weight=0.0,
        allowed_name_sets=[allowed_signature_names_for_task(row) for row in train_rows],
    )
    train_summary["pretraining_initialization"] = pretraining_init_summary
    train_summary["strict_generator_pretraining_checkpoint"] = strict_generator_checkpoint_summary
    train_summary["full_state_pretraining"] = full_state_pretraining_summary
    eval_x = encode_many([decoder_source_text(row, text_views.get("sts_on", [])) for row in eval_rows], source_vocab, max_source)
    broad_max_target = int(broad_cfg.get("max_target_tokens") or max_target)
    broad_fanout = int(broad_cfg.get("fanout_top_k") or budget.get("fanout_top_k") or 1)
    broad_pool_size = max(
        broad_fanout,
        int(dict_or_empty(get_path(config, ["body_structure_decoder", "static_coherence_ranker"], {})).get("candidate_pool_size") or 0),
    ) if static_coherence_ranker_enabled(config) else broad_fanout
    decoded = generate_candidates(
        model,
        eval_x,
        target_vocab,
        max_target_tokens=broad_max_target,
        fanout_top_k=broad_pool_size,
        grammar_top_k=int(broad_cfg.get("grammar_decode_top_k") or budget.get("grammar_decode_top_k") or 64),
        decode_beam_width=int(
            broad_cfg.get("decode_beam_width")
            or arm_decode_setting(config, "transformer_control", budget, "decode_beam_width")
        ),
        decode_branching_factor=int(
            broad_cfg.get("decode_branching_factor")
            or arm_decode_setting(config, "transformer_control", budget, "decode_branching_factor")
        ),
        target_mode=target_mode,
        body_token_decode_policy=str(
            broad_cfg.get("body_token_decode_policy")
            or budget.get("body_token_decode_policy")
            or budget.get("body_token_validity_policy")
            or "lightweight_python_v1"
        ),
        allowed_name_sets=[allowed_signature_names_for_task(row) for row in eval_rows],
        torch=torch,
        device=device,
    )
    rows = token_candidate_rows_for_view(
        eval_rows,
        decoded,
        arm_id="transformer_control",
        substrate="torch_transformer_encoder_token_decoder",
        phase="private_eval",
        view="sts_on",
        config=config,
        seed=seed,
        target_mode=target_mode,
        residual_context=residual_context,
        output_top_k=broad_fanout,
    )
    broad_syntax = syntax_summary(rows)
    broad_repair = grammar_repair_summary(rows)
    broad_static = static_coherence_summary(rows)
    verifier_started = time.perf_counter()
    private_eval = evaluate_private_candidates(eval_rows, [baseline_candidate(task, arm_id="transformer_control", config=config, seed=seed) for task in eval_rows] + rows)
    manifest_rows = broad_private_heldout_manifest_rows(rows)
    summary = {
        "enabled": True,
        "active": True,
        "eval_jsonl": [rel(path) for path in eval_paths],
        "train_rows": len(train_rows),
        "eval_rows_before_overlap_filter": len(eval_rows_unfiltered),
        "eval_rows_after_overlap_filter": len(eval_rows_all),
        "eval_rows_rejected_for_train_family_overlap": family_overlap_rejected,
        "eval_rows_rejected_for_train_solution_or_token_overlap": overlap_rejected,
        "eval_rows": len(eval_rows),
        "eval_family_count": eval_family_count,
        "min_eval_rows": min_eval_rows,
        "min_eval_families": min_eval_families,
        "candidate_rows": len(manifest_rows),
        "max_target_tokens": broad_max_target,
        "fanout_top_k": broad_fanout,
        "candidate_pool_size": broad_pool_size,
        "grammar_decode_top_k": int(broad_cfg.get("grammar_decode_top_k") or budget.get("grammar_decode_top_k") or 64),
        "body_token_decode_policy": str(
            broad_cfg.get("body_token_decode_policy")
            or budget.get("body_token_decode_policy")
            or budget.get("body_token_validity_policy")
            or "lightweight_python_v1"
        ),
        "signature_name_mask": {
            "enabled": True,
            "policy": "visible_signature_other_extra_mask_v1",
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        },
        "decode_beam_width": int(
            broad_cfg.get("decode_beam_width")
            or arm_decode_setting(config, "transformer_control", budget, "decode_beam_width")
        ),
        "decode_branching_factor": int(
            broad_cfg.get("decode_branching_factor")
            or arm_decode_setting(config, "transformer_control", budget, "decode_branching_factor")
        ),
        "source_vocab_size": len(source_vocab),
        "source_vocab_extension": source_vocab_extension_summary,
        "target_vocab_size": len(target_vocab),
        "target_vocab_extension": target_vocab_extension_summary,
        "parameter_count": parameter_count,
        "transformer_sts_on_verifier_pass_rate": private_eval.get("trained_pass_rate"),
        "transformer_sts_on_rank1_pass_rate": private_eval.get("trained_rank1_pass_rate"),
        "transformer_sts_on_pass_if_any_rate": private_eval.get("trained_pass_if_any_rate"),
        "candidate_syntax": broad_syntax,
        "grammar_repair": broad_repair,
        "static_coherence": broad_static,
        "private_verifier": {
            "trained_passed": private_eval.get("trained_passed"),
            "eval_task_count": private_eval.get("eval_task_count"),
            "residual_count": private_eval.get("residual_count"),
            "wall_time_ms": int((time.perf_counter() - verifier_started) * 1000),
        },
        "split_overlap_audit": overlap,
        "zero_train_eval_overlap": all(int(overlap.get(key) or 0) == 0 for key in overlap if key.endswith("_count")),
        "pretraining_initialization": pretraining_initialization_report_summary(pretraining_initializers),
        "strict_generator_checkpoint_vocab_override": strict_generator_vocab_override_report(strict_generator_vocab_override),
        "full_state_pretraining": full_state_pretraining_report_summary(full_state_pretraining_cfg, full_state_pretraining_rows),
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "teacher_used": False,
    }
    return {
        "policy": "project_theseus_broad_private_heldout_eval_v1",
        "enabled": True,
        "active": True,
        "created_utc": now(),
        "summary": summary,
        "split_audit": split_audit,
        "train": train_summary,
        "candidate_syntax": broad_syntax,
        "grammar_repair": broad_repair,
        "static_coherence": broad_static,
        "private_verifier": private_eval,
        "score_semantics": (
            "Large existing private heldout eval. Families are disjoint from training rows and family labels are "
            "used only for auditing, never for generation or ranking."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }, manifest_rows


def structural_action_family_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict_or_empty(dict_or_empty(config.get("body_structure_decoder")).get("structural_action_family"))


def build_structural_action_family_context(
    config: dict[str, Any],
    *,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    text_views: dict[str, Any],
    budget: dict[str, Any],
    torch: Any,
    nn: Any,
    device: Any,
) -> dict[str, Any]:
    family_cfg = structural_action_family_config(config)
    if not bool(family_cfg.get("enabled", False)):
        return {}

    library = build_action_sequence_library(train_rows)
    classes = list(library.get("classes") or [])
    class_index = {str(row.get("sequence_id") or ""): idx for idx, row in enumerate(classes)}
    structural_train_rows: list[dict[str, Any]] = []
    train_y: list[int] = []
    for row in train_rows:
        label = action_sequence_id(row)
        if label in class_index:
            structural_train_rows.append(row)
            train_y.append(class_index[label])
    if len(classes) < 2 or not structural_train_rows:
        return {
            "enabled": True,
            "active": False,
            "blocked_reason": "structural_action_class_count_below_minimum",
            "library": library.get("summary", {}),
        }

    source_view = str(family_cfg.get("source_view") or "sts_on")
    source_fields = list(text_views.get(source_view) or text_views.get("sts_on") or [])
    max_source = int(family_cfg.get("max_source_tokens") or budget.get("max_source_tokens") or 96)
    max_vocab = int(family_cfg.get("max_source_vocab") or budget.get("max_source_vocab") or 1024)
    source_vocab = build_vocab(
        [structural_source_text(row, source_fields) for row in structural_train_rows],
        max_vocab=max_vocab,
    )
    train_x = encode_many(
        [structural_source_text(row, source_fields) for row in structural_train_rows],
        source_vocab,
        max_source,
    )
    eval_x = encode_many(
        [structural_source_text(row, source_fields) for row in eval_rows],
        source_vocab,
        max_source,
    )

    transformer_cfg = dict_or_empty(dict_or_empty(config.get("arms")).get("transformer_control"))
    transformer_dims = {
        "d_model": int(family_cfg.get("transformer_d_model") or transformer_cfg.get("d_model") or 48),
        "nhead": int(family_cfg.get("transformer_nhead") or transformer_cfg.get("nhead") or 2),
        "num_layers": int(family_cfg.get("transformer_num_layers") or transformer_cfg.get("num_layers") or 1),
        "dim_feedforward": int(family_cfg.get("transformer_dim_feedforward") or transformer_cfg.get("dim_feedforward") or 96),
        "max_len": max_source,
    }
    transformer_param_count = count_params(
        TinyTransformerClassifier(
            len(source_vocab),
            len(classes),
            **transformer_dims,
            torch=torch,
            nn=nn,
        )
    )
    sym_dims, sym_param_count = choose_symliquid_classifier_dims(
        config,
        vocab_size=len(source_vocab),
        class_count=len(classes),
        target_params=transformer_param_count,
        torch=torch,
        nn=nn,
    )
    train_budget = dict(budget)
    for key in ["epochs", "batch_size", "learning_rate", "weight_decay"]:
        if key in family_cfg:
            train_budget[key] = family_cfg[key]
    train_budget["fanout_top_k"] = int(family_cfg.get("fanout_top_k") or budget.get("fanout_top_k") or 2)
    train_budget["rank_pool_size"] = int(
        family_cfg.get("rank_pool_size")
        or budget.get("rank_pool_size")
        or max(8, int(train_budget["fanout_top_k"]) * 4)
    )
    train_budget["compatibility_rerank"] = str(
        family_cfg.get("compatibility_rerank")
        or budget.get("compatibility_rerank")
        or "on"
    )
    return {
        "enabled": True,
        "active": True,
        "config": family_cfg,
        "library": library,
        "classes": classes,
        "class_count": len(classes),
        "source_view": source_view,
        "source_fields": source_fields,
        "source_vocab": source_vocab,
        "source_vocab_size": len(source_vocab),
        "max_source_tokens": max_source,
        "train_rows": structural_train_rows,
        "train_x": train_x,
        "train_y": train_y,
        "eval_x": eval_x,
        "train_budget": train_budget,
        "transformer_dims": transformer_dims,
        "transformer_param_count": transformer_param_count,
        "sym_dims": sym_dims,
        "sym_param_count": sym_param_count,
        "parameter_match_delta": abs(sym_param_count - transformer_param_count) / max(1, transformer_param_count),
        "device": str(device),
    }


def run_structural_action_family_for_arm(
    config: dict[str, Any],
    context: dict[str, Any],
    *,
    eval_rows: list[dict[str, Any]],
    arm_id: str,
    seed: int,
    torch: Any,
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    if not context.get("active"):
        return {
            "arm_id": arm_id,
            "view": "structural_action",
            "candidate_family": "private_train_structural_action_sequence_decoder",
            "active": False,
            "blocked_reason": context.get("blocked_reason", "not_enabled"),
        }, []

    nn = torch.nn
    torch.manual_seed(seed)
    random.seed(seed)
    if arm_id == "symliquid_style":
        model = SymLiquidStyleClassifier(
            len(context["source_vocab"]),
            len(context["classes"]),
            hidden_dim=int(context["sym_dims"]["hidden_dim"]),
            reservoir_dim=int(context["sym_dims"]["reservoir_dim"]),
            hv_dim=int(context["sym_dims"]["hv_dim"]),
            torch=torch,
            nn=nn,
        )
        substrate = "torch_symliquid_style_structural_action_sequence_classifier"
        dims = context["sym_dims"]
        parameter_count = int(context["sym_param_count"])
    else:
        model = TinyTransformerClassifier(
            len(context["source_vocab"]),
            len(context["classes"]),
            **context["transformer_dims"],
            torch=torch,
            nn=nn,
        )
        substrate = "torch_transformer_structural_action_sequence_classifier"
        dims = context["transformer_dims"]
        parameter_count = int(context["transformer_param_count"])
    model.to(device)
    before_mem = maxrss_mb()
    train_summary = train_classifier_model(
        model,
        context["train_x"],
        context["train_y"],
        context["train_budget"],
        torch=torch,
        device=device,
    )
    proposals = rank_action_sequences(
        model,
        context["eval_x"],
        context["classes"],
        eval_rows=eval_rows,
        fanout_top_k=int(context["train_budget"].get("fanout_top_k") or 2),
        rank_pool_size=int(context["train_budget"].get("rank_pool_size") or 8),
        compatibility_rerank=str(context["train_budget"].get("compatibility_rerank") or "on") == "on",
        torch=torch,
        device=device,
    )
    rows = structural_candidate_rows(
        eval_rows,
        proposals,
        arm_id=arm_id,
        substrate=substrate,
        config=config,
        seed=seed,
    )
    for row in rows:
        row["candidate_source"] = "neural_seed_token_decoder_comparator.structural_action_family"
        row["integrated_candidate_family"] = "structural_action_sequence"
        row["provenance"]["view"] = "sts_on_structural_action_family"
        row["provenance"]["candidate_family"] = "structural_action_sequence"
        row["body_structure_decode"]["integrated_candidate_family"] = True
    syntax = syntax_summary(rows)
    report = {
        "arm_id": arm_id,
        "view": "structural_action",
        "phase": "private_eval",
        "active": True,
        "candidate_family": "private_train_structural_action_sequence_decoder",
        "substrate": substrate,
        "parameter_count": parameter_count,
        "parameter_match_delta_vs_transformer_control": round(float(context["parameter_match_delta"]), 6),
        "dims": dims,
        "train": train_summary,
        "candidate_rows": len(rows),
        "candidate_tasks": len(eval_rows),
        "fanout_top_k": int(context["train_budget"].get("fanout_top_k") or 2),
        "rank_pool_size": int(context["train_budget"].get("rank_pool_size") or 8),
        "compatibility_rerank": str(context["train_budget"].get("compatibility_rerank") or "on"),
        "candidate_syntax": syntax,
        "grammar_repair": grammar_repair_summary(rows),
        "body_structure": body_structure_summary(rows),
        "ranker": "structural_action_sequence_softmax_probability",
        "decoder_constraints": {
            "target_mode": "structural_action_sequence_v0",
            "selection_ablation_axis": "sequence_class_selection",
            "compiler_ablation_axis": "line_action_compilation",
            "finer_ast_synthesis_enabled": False,
            "compiler": "generic_private_train_line_action_compiler_v0",
            "fallback_returns_allowed": False,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        },
        "backend": {
            "framework": "torch",
            "device": str(device),
            **mlx_status(),
        },
        "memory": {
            "maxrss_mb_before": before_mem,
            "maxrss_mb_after": maxrss_mb(),
        },
        "wall_time_ms_before_verifier": int((time.perf_counter() - started) * 1000),
    }
    return report, rows


def structural_action_family_summary(config: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    family_cfg = structural_action_family_config(config)
    context = context or {}
    library = dict_or_empty(context.get("library"))
    library_summary = dict_or_empty(library.get("summary") if library else context.get("library"))
    return {
        "enabled": bool(family_cfg.get("enabled", False)),
        "active": bool(context.get("active", False)),
        "blocked_reason": context.get("blocked_reason", ""),
        "target_mode": str(family_cfg.get("target_mode") or "structural_action_sequence_v0"),
        "target_source": str(family_cfg.get("target_source") or "private_train_solution_body_line_action_sequences"),
        "compiler": str(family_cfg.get("compiler") or "generic_private_train_line_action_compiler_v0"),
        "source_view": str(context.get("source_view") or family_cfg.get("source_view") or "sts_on"),
        "source_vocab_size": int(context.get("source_vocab_size") or 0),
        "structural_action_class_count": int(context.get("class_count") or library_summary.get("class_count") or 0),
        "structural_action_token_count": int(library_summary.get("unique_action_token_count") or 0),
        "fanout_top_k": int(dict_or_empty(context.get("train_budget")).get("fanout_top_k") or family_cfg.get("fanout_top_k") or 0),
        "rank_pool_size": int(dict_or_empty(context.get("train_budget")).get("rank_pool_size") or family_cfg.get("rank_pool_size") or 0),
        "compatibility_rerank": str(dict_or_empty(context.get("train_budget")).get("compatibility_rerank") or family_cfg.get("compatibility_rerank") or "on"),
        "epochs": int(dict_or_empty(context.get("train_budget")).get("epochs") or family_cfg.get("epochs") or 0),
        "sequence_class_selection": bool(family_cfg.get("enabled", False)),
        "line_action_compilation": bool(family_cfg.get("enabled", False)),
        "finer_ast_synthesis": False,
        "fallback_returns_allowed": False,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "teacher_used": False,
    }


def arm_view_names(config: dict[str, Any], arm_id: str, *, default: list[str]) -> list[str]:
    arm_cfg = dict_or_empty(dict_or_empty(config.get("arms")).get(arm_id))
    views = arm_cfg.get("views", default)
    if not isinstance(views, list):
        views = default
    cleaned = [str(view) for view in views if str(view) in {"sts_off", "sts_on"}]
    return cleaned or list(default)


def execution_arm_names(config: dict[str, Any], *, split: str = "") -> list[str]:
    """Return declared arms for a run/split.

    Default behavior remains the matched two-arm verdict. The environment
    override is intentionally scoped to execution, not claims: reports still
    record the requested arms and gates only validate that declared scope.
    """

    raw: Any = None
    source = ""
    env_value = os.environ.get("THESEUS_TOKEN_DECODER_ARMS", "").strip()
    if env_value:
        raw = [part.strip() for part in env_value.split(",")]
        source = "env:THESEUS_TOKEN_DECODER_ARMS"
    execution_cfg = dict_or_empty(config.get("execution"))
    by_split = dict_or_empty(execution_cfg.get("enabled_arms_by_split"))
    if raw is None and split and split in by_split:
        raw = by_split.get(split)
        source = f"config.execution.enabled_arms_by_split.{split}"
    if raw is None:
        raw = execution_cfg.get("enabled_arms")
        source = "config.execution.enabled_arms"
    if raw is None:
        raw = list(TOKEN_DECODER_ARM_IDS)
        source = "default_matched_two_arm"
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    if not isinstance(raw, list):
        raw = list(TOKEN_DECODER_ARM_IDS)
        source = "default_matched_two_arm_invalid_config"
    cleaned: list[str] = []
    for arm_id in raw:
        arm_name = str(arm_id).strip()
        if arm_name in TOKEN_DECODER_ARM_IDS and arm_name not in cleaned:
            cleaned.append(arm_name)
    if not cleaned:
        cleaned = list(TOKEN_DECODER_ARM_IDS)
        source = "default_matched_two_arm_empty_scope"
    return cleaned


def execution_scope_summary(config: dict[str, Any], *, split: str = "") -> dict[str, Any]:
    arms = execution_arm_names(config, split=split)
    env_value = os.environ.get("THESEUS_TOKEN_DECODER_ARMS", "").strip()
    execution_cfg = dict_or_empty(config.get("execution"))
    return {
        "split": split or "all",
        "enabled_arms": arms,
        "matched_two_arm_verdict_scope": set(arms) == set(TOKEN_DECODER_ARM_IDS),
        "env_override": env_value,
        "config_enabled_arms": execution_cfg.get("enabled_arms"),
        "config_enabled_arms_by_split": execution_cfg.get("enabled_arms_by_split"),
        "score_semantics": (
            "Transformer-only scope is a survival-lane repair/validation run, not a SymLiquid-vs-transformer verdict. "
            "Matched substrate claims require both arms to be enabled and executed."
        ),
    }


def arm_decode_setting(config: dict[str, Any], arm_id: str, budget: dict[str, Any], key: str) -> int:
    arm_cfg = dict_or_empty(dict_or_empty(config.get("arms")).get(arm_id))
    value = arm_cfg.get(key, budget.get(key, 0))
    return int(value or 0)


def run_full_state_pretraining_warmup(
    model: Any,
    rows: dict[str, Any],
    cfg: dict[str, Any],
    *,
    arm_id: str,
    view: str,
    split: str,
    torch: Any,
    device: Any,
    pad_id: int,
    target_vocab: dict[str, int] | None = None,
) -> dict[str, Any]:
    summary = dict_or_empty(rows.get("summary"))
    if not bool(cfg.get("enabled", False)):
        return {"enabled": False, "active": False, "reason": "full_state_pretraining_disabled"}
    if not bool(rows.get("active")):
        return {
            "enabled": True,
            "active": False,
            "reason": "no_active_full_state_pretraining_rows",
            "row_summary": summary,
        }
    source_rows = list(rows.get("source_rows") or [])
    target_rows = list(rows.get("target_rows") or [])
    eval_source_rows = list(rows.get("eval_source_rows") or [])
    eval_target_rows = list(rows.get("eval_target_rows") or [])
    examples = list(rows.get("examples") or [])
    eval_examples = list(rows.get("eval_examples") or [])
    arm_example_caps = dict_or_empty(cfg.get("arm_example_caps"))
    arm_eval_caps = dict_or_empty(cfg.get("arm_eval_example_caps"))
    arm_epoch_overrides = dict_or_empty(cfg.get("arm_epoch_overrides"))
    split_example_caps = dict_or_empty(cfg.get("split_example_caps"))
    split_eval_caps = dict_or_empty(cfg.get("split_eval_example_caps"))
    split_epoch_overrides = dict_or_empty(cfg.get("split_epoch_overrides"))
    arm_cap = int(arm_example_caps.get(arm_id) or 0)
    split_cap = int(split_example_caps.get(split) or 0)
    effective_cap = min(cap for cap in [arm_cap, split_cap] if cap > 0) if (arm_cap > 0 or split_cap > 0) else 0
    if effective_cap > 0 and len(source_rows) > effective_cap:
        source_rows = source_rows[:effective_cap]
        target_rows = target_rows[:effective_cap]
        examples = examples[: min(len(examples), 16)]
    arm_eval_cap = int(arm_eval_caps.get(arm_id) or 0)
    split_eval_cap = int(split_eval_caps.get(split) or 0)
    effective_eval_cap = (
        min(cap for cap in [arm_eval_cap, split_eval_cap] if cap > 0) if (arm_eval_cap > 0 or split_eval_cap > 0) else 0
    )
    if effective_eval_cap > 0 and len(eval_source_rows) > effective_eval_cap:
        eval_source_rows = eval_source_rows[:effective_eval_cap]
        eval_target_rows = eval_target_rows[:effective_eval_cap]
        eval_examples = eval_examples[: min(len(eval_examples), 8)]
    epoch_override = split_epoch_overrides.get(split)
    if epoch_override is None:
        epoch_override = arm_epoch_overrides.get(arm_id)
    budget = {
        "epochs": int(epoch_override or cfg.get("epochs") or 1),
        "batch_size": int(cfg.get("batch_size") or 32),
        "learning_rate": float(cfg.get("learning_rate") or 0.001),
        "weight_decay": float(cfg.get("weight_decay") or 0.0001),
        "first_target_token_loss_weight": float(cfg.get("first_target_token_loss_weight") or 1.0),
        "return_token_loss_weight": float(cfg.get("return_token_loss_weight") or 1.0),
        "grammar_validity_auxiliary_loss_weight": float(cfg.get("grammar_validity_auxiliary_loss_weight") or 0.0),
        "grammar_validity_max_positions_per_batch": int(cfg.get("grammar_validity_max_positions_per_batch") or 0),
        "body_token_validity_policy": str(cfg.get("body_token_validity_policy") or "lightweight_python_v1"),
        "progress_label": f"full_state_pretraining split={split} arm={arm_id} view={view}",
    }
    started = time.perf_counter()
    before = model_parameter_snapshot(model, torch=torch)
    print(
        f"[strict-decoder] full-state warmup start split={split} arm={arm_id} view={view} "
        f"rows={len(source_rows)} eval_rows={len(eval_source_rows)} epochs={budget['epochs']} "
        f"batch_size={budget['batch_size']}",
        file=sys.stderr,
        flush=True,
    )
    heldout_before = evaluate_token_model_loss(
        model,
        eval_source_rows,
        eval_target_rows,
        batch_size=int(budget["batch_size"]),
        torch=torch,
        device=device,
        pad_id=pad_id,
    )
    train_summary = train_token_model(
        model,
        source_rows,
        target_rows,
        budget,
        torch=torch,
        device=device,
        pad_id=pad_id,
        target_vocab=target_vocab,
        plan_auxiliary_loss_weight=0.0,
    )
    heldout_after = evaluate_token_model_loss(
        model,
        eval_source_rows,
        eval_target_rows,
        batch_size=int(budget["batch_size"]),
        torch=torch,
        device=device,
        pad_id=pad_id,
    )
    update_summary = model_parameter_update_summary(model, before, torch=torch)
    print(
        f"[strict-decoder] full-state warmup done split={split} arm={arm_id} view={view} "
        f"loss={heldout_before.get('loss')}->{heldout_after.get('loss')} "
        f"update_fraction={update_summary['parameter_update_fraction']} "
        f"non_embedding_update_fraction={update_summary['non_embedding_update_fraction']}",
        file=sys.stderr,
        flush=True,
    )
    train_summary.update(
        {
            "enabled": True,
            "active": True,
            "policy": str(cfg.get("policy") or "comparator_compatible_python_function_full_state_warmup_v1"),
            "arm_id": arm_id,
            "view": view,
            "split": split,
            "row_summary": summary,
            "examples": examples,
            "eval_examples": eval_examples,
            "arm_effective_train_example_count": len(source_rows),
            "arm_effective_eval_example_count": len(eval_source_rows),
            "arm_example_cap": arm_cap,
            "split_example_cap": split_cap,
            "effective_example_cap": effective_cap,
            "arm_eval_example_cap": arm_eval_cap,
            "split_eval_example_cap": split_eval_cap,
            "effective_eval_example_cap": effective_eval_cap,
            "split_epoch_override": epoch_override,
            "updates_entire_model_state": True,
            "full_model_pretraining": True,
            "trainable_parameter_count": update_summary["trainable_parameter_count"],
            "parameter_update_fraction": update_summary["parameter_update_fraction"],
            "non_embedding_update_fraction": update_summary["non_embedding_update_fraction"],
            "updated_parameter_count": update_summary["updated_parameter_count"],
            "updated_non_embedding_parameter_count": update_summary["updated_non_embedding_parameter_count"],
            "parameter_tensor_update_fraction": update_summary["parameter_tensor_update_fraction"],
            "heldout_lm_loss_before": heldout_before.get("loss"),
            "heldout_lm_loss_after": heldout_after.get("loss"),
            "heldout_lm_perplexity_before": heldout_before.get("perplexity"),
            "heldout_lm_perplexity_after": heldout_after.get("perplexity"),
            "heldout_lm_loss_curve": [heldout_before.get("loss"), heldout_after.get("loss")],
            "heldout_lm_improved": bool(
                heldout_before.get("loss") is not None
                and heldout_after.get("loss") is not None
                and float(heldout_after["loss"]) < float(heldout_before["loss"])
            ),
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "teacher_used": False,
            "external_inference_calls": 0,
            "public_training_rows": 0,
            "wall_time_ms": int((time.perf_counter() - started) * 1000),
        }
    )
    return train_summary


def full_state_pretraining_report_summary(cfg: dict[str, Any], rows: dict[str, Any]) -> dict[str, Any]:
    summary = dict_or_empty(rows.get("summary"))
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "active": bool(rows.get("active")),
        "policy": str(cfg.get("policy") or summary.get("policy") or "comparator_compatible_python_function_full_state_warmup_v1"),
        "row_summary": summary,
        "examples": list(rows.get("examples") or []),
        "eval_examples": list(rows.get("eval_examples") or []),
        "updates_entire_model_state": bool(cfg.get("enabled", False) and rows.get("active")),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "teacher_used": False,
        "external_inference_calls": 0,
        "public_training_rows": 0,
    }


def token_candidate_rows_for_view(
    eval_rows: list[dict[str, Any]],
    decoded: list[list[dict[str, Any]]],
    *,
    arm_id: str,
    substrate: str,
    phase: str,
    view: str,
    config: dict[str, Any],
    seed: int,
    target_mode: str,
    residual_context: dict[str, Any],
    output_top_k: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    routing_summary = internal_semantic_routing_summary(config, view=view)
    static_ranker = static_coherence_ranker_enabled(config)
    for task, proposals in zip(eval_rows, decoded):
        task_proposals = list(proposals)
        if view == "sts_on":
            task_proposals.extend(visible_contract_semantic_beams(task, config, len(task_proposals)))
        prepared: list[dict[str, Any]] = []
        for original_rank, proposal in enumerate(task_proposals, start=1):
            decoded_tokens = list(proposal.get("decoded_tokens", []))
            body, structure_decode = decode_candidate_body_tokens(decoded_tokens, task, target_mode=target_mode)
            if proposal.get("beam_source") == "visible_contract_semantic_prior":
                structure_decode["visible_contract_semantic_beam"] = True
                structure_decode["visible_contract_semantic_prior"] = {
                    "plan": proposal.get("contract_prior_plan"),
                    "rule": proposal.get("contract_prior_rule"),
                    "fields_used": proposal.get("contract_prior_fields_used"),
                    "uses_eval_tests_or_solutions": False,
                    "uses_public_data": False,
                }
            if proposal.get("beam_source") == "learned_internal_semantic_route":
                structure_decode["learned_internal_semantic_route"] = True
                structure_decode["learned_internal_semantic_route_rank"] = proposal.get("learned_route_rank")
                structure_decode["learned_internal_semantic_route_probability"] = proposal.get("learned_route_probability")
            repair = grammar_repaired_body(
                task,
                raw_body=body,
                decoded_tokens=decoded_tokens,
            )
            code = render_private_function(task, str(repair["body"]))
            prepared.append(
                {
                    "proposal": proposal,
                    "decoded_tokens": decoded_tokens,
                    "body": body,
                    "structure_decode": structure_decode,
                    "repair": repair,
                    "code": code,
                    "original_rank": original_rank,
                    "static_coherence": candidate_static_coherence(code),
                }
            )
        prefix_guided_ranker = any(
            dict_or_empty(item.get("proposal")).get("prefix_guided_rank_score") is not None
            for item in prepared
        )
        if prefix_guided_ranker:
            prepared.sort(
                key=lambda item: (
                    float(dict_or_empty(item.get("proposal")).get("prefix_guided_rank_score") or 0.0),
                    static_coherence_sort_key(item),
                ),
                reverse=True,
            )
        elif static_ranker:
            prepared.sort(key=static_coherence_sort_key, reverse=True)
        if output_top_k > 0:
            prepared = prepared[:output_top_k]
        for rank, item in enumerate(prepared, start=1):
            proposal = dict_or_empty(item.get("proposal"))
            decoded_tokens = list(item.get("decoded_tokens") or [])
            structure_decode = dict_or_empty(item.get("structure_decode"))
            repair = dict_or_empty(item.get("repair"))
            code = str(item.get("code") or "")
            row = candidate_row(
                task,
                code=code,
                phase=phase,
                arm_id=arm_id,
                substrate=substrate,
                view=view,
                rank=rank,
                rank_score=float(proposal["rank_score"]),
                config=config,
                seed=seed,
                decoded_token_count=int(proposal["decoded_token_count"]),
                decoded_token_sha256=str(proposal["decoded_token_sha256"]),
            )
            row["raw_decoded_body_sha256"] = stable_hash(str(proposal["body"]))
            row["body_structure_decode"] = structure_decode
            row["target_mode"] = target_mode
            row["internal_semantic_routing"] = routing_summary
            row["residual_mining"] = residual_tag_for_task(task, residual_context)
            row["static_coherence"] = item.get("static_coherence")
            if proposal.get("plan_prefix_source"):
                row["plan_prefix_source"] = str(proposal.get("plan_prefix_source"))
                row["provenance"]["plan_prefix_source"] = str(proposal.get("plan_prefix_source"))
            row["provenance"]["target_mode"] = target_mode
            row["provenance"]["training_target"] = f"private_train_{target_mode}"
            if bool(structure_decode.get("rendered_from_strict_actions")):
                row["strict_action_renderer"] = {
                    "enabled": True,
                    "policy": structure_decode.get("strict_action_renderer_policy"),
                    "target_mode": target_mode,
                    "uses_eval_tests_or_solutions": False,
                    "uses_public_data": False,
                    "teacher_used": False,
                    "promotion_evidence": False,
                }
                row["provenance"]["candidate_family"] = "strict_action_structural_target"
                row["provenance"]["candidate_generation_mode_detail"] = (
                    "strict_action_tokens_v1_with_fixed_action_renderer_diagnostic_only"
                )
            row["static_coherence_ranker"] = {
                "enabled": static_ranker or prefix_guided_ranker,
                "policy": (
                    "learned_prefix_decision_rank_score_then_static_coherence_v1"
                    if prefix_guided_ranker
                    else "prompt_signature_static_coherence_v1"
                ),
                "original_rank": int(item.get("original_rank") or rank),
                "rank_changed": int(item.get("original_rank") or rank) != rank,
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "uses_answer_metadata": False,
            }
            if prefix_guided_ranker:
                row["static_coherence_ranker"]["prefix_guided_rank_score"] = dict_or_empty(proposal).get("prefix_guided_rank_score")
                row["static_coherence_ranker"]["candidate_generation_credit"] = 0
            row["provenance"]["ranker"] = (
                "learned_prefix_decision_rank_score_then_static_coherence"
                if prefix_guided_ranker
                else "prompt_signature_static_coherence_then_sequence_log_probability"
                if static_ranker
                else row["provenance"]["ranker"]
            )
            if proposal.get("beam_source") == "visible_contract_semantic_prior":
                row["visible_contract_semantic_beam"] = {
                    "enabled": True,
                    "plan": proposal.get("contract_prior_plan"),
                    "rule": proposal.get("contract_prior_rule"),
                    "fields_used": proposal.get("contract_prior_fields_used"),
                    "uses_eval_tests_or_solutions": False,
                    "uses_public_data": False,
                }
                row["provenance"]["semantic_beam_source"] = "visible_contract_semantic_prior"
            if proposal.get("beam_source") == "learned_internal_semantic_route":
                row["learned_internal_semantic_route"] = {
                    "enabled": True,
                    "plan": proposal.get("learned_route_plan"),
                    "rank": proposal.get("learned_route_rank"),
                    "probability": proposal.get("learned_route_probability"),
                    "strategy": proposal.get("learned_route_strategy"),
                    "fingerprint_scope": proposal.get("learned_route_fingerprint_scope"),
                    "fingerprint_support": proposal.get("learned_route_fingerprint_support"),
                    "fingerprint_total": proposal.get("learned_route_fingerprint_total"),
                    "feature_support": proposal.get("learned_route_feature_support"),
                    "feature_top_score": proposal.get("learned_route_feature_top_score"),
                    "uses_eval_tests_or_solutions": False,
                    "uses_public_data": False,
                    "teacher_used": False,
                }
                row["provenance"]["semantic_route_source"] = "learned_internal_semantic_route"
            row["grammar_repair"] = {
                "policy": "deterministic_python_body_repair_v1",
                "raw_syntax_ok": bool(repair["raw_syntax_ok"]),
                "repaired_syntax_ok": bool(repair["repaired_syntax_ok"]),
                "changed": bool(repair["changed"]),
                "strategy": repair["strategy"],
                "repair_passes": repair["repair_passes"],
                "fallback_return_used": bool(repair["fallback_return_used"]),
                "fallback_return_shape": repair["fallback_return_shape"],
                "contract_return_shape_for_diagnostics": repair["contract_return_shape_for_diagnostics"],
                "raw_failure": repair["raw_failure"],
                "repaired_failure": repair["repaired_failure"],
            }
            rows.append(row)
    return rows



def arm_summary(view_reports: dict[str, Any], private_eval: dict[str, Any], verifier_wall_ms: int) -> dict[str, Any]:
    sts_on = dict_or_empty(view_reports.get("sts_on"))
    sts_off = dict_or_empty(view_reports.get("sts_off"))
    structural = dict_or_empty(view_reports.get("structural_action"))
    private_verification = dict_or_empty(private_eval.get("private_verification"))
    return {
        "sts_on_verifier_pass_rate": private_eval.get("trained_pass_rate"),
        "sts_off_verifier_pass_rate": private_eval.get("sts_off_pass_rate"),
        "sts_on_rank1_pass_rate": private_eval.get("trained_rank1_pass_rate"),
        "sts_off_rank1_pass_rate": private_eval.get("sts_off_rank1_pass_rate"),
        "sts_on_pass_if_any_rate": private_eval.get("trained_pass_if_any_rate"),
        "sts_off_pass_if_any_rate": private_eval.get("sts_off_pass_if_any_rate"),
        "sts_on_rank1_passed": private_eval.get("trained_rank1_passed"),
        "sts_on_pass_if_any_passed": private_eval.get("trained_pass_if_any_passed"),
        "sts_delta": private_eval.get("sts_repair_pass_rate_delta"),
        "accepted_candidate_rate": private_eval.get("trained_pass_rate"),
        "baseline_pass_rate": private_eval.get("baseline_pass_rate"),
        "syntax_pass_rate_sts_on": get_path(sts_on, ["candidate_syntax", "syntax_pass_rate"], 0.0),
        "syntax_pass_rate_sts_off": get_path(sts_off, ["candidate_syntax", "syntax_pass_rate"], 0.0),
        "raw_syntax_pass_rate_sts_on": get_path(sts_on, ["grammar_repair", "raw_syntax_pass_rate"], 0.0),
        "raw_syntax_pass_rate_sts_off": get_path(sts_off, ["grammar_repair", "raw_syntax_pass_rate"], 0.0),
        "grammar_repair_changed_rate_sts_on": get_path(sts_on, ["grammar_repair", "changed_rate"], 0.0),
        "grammar_repair_fallback_rate_sts_on": get_path(sts_on, ["grammar_repair", "fallback_rate"], 0.0),
        "statement_skeleton_render_rate_sts_on": get_path(sts_on, ["body_structure", "statement_skeleton_render_rate"], 0.0),
        "semantic_slot_render_rate_sts_on": get_path(sts_on, ["body_structure", "semantic_slot_render_rate"], 0.0),
        "strict_action_render_rate_sts_on": get_path(sts_on, ["body_structure", "strict_action_render_rate"], 0.0),
        "structural_action_render_rate": get_path(structural, ["body_structure", "structural_action_render_rate"], 0.0),
        "structural_action_candidate_rows": int(structural.get("candidate_rows") or 0),
        "structural_action_syntax_pass_rate": get_path(structural, ["candidate_syntax", "syntax_pass_rate"], 0.0),
        "structural_action_raw_syntax_pass_rate": get_path(structural, ["grammar_repair", "raw_syntax_pass_rate"], 0.0),
        "structural_action_fallback_rate": get_path(structural, ["grammar_repair", "fallback_rate"], 0.0),
        "semantic_plan_supported_rate_sts_on": get_path(sts_on, ["body_structure", "semantic_plan_supported_rate"], 0.0),
        "learned_internal_semantic_route_rate_sts_on": get_path(sts_on, ["body_structure", "learned_internal_semantic_route_rate"], 0.0),
        "predicted_return_shape_rate_sts_on": get_path(sts_on, ["body_structure", "predicted_return_shape_rate"], 0.0),
        "verifier_lint_pass_rate": private_verification.get("lint_pass_rate"),
        "verifier_compile_pass_rate": private_verification.get("compile_pass_rate"),
        "runtime_load_rate": private_verification.get("runtime_load_rate"),
        "verifier_candidate_attempt_count": private_verification.get("candidate_attempt_count"),
        "sts_task_level_improvements": private_eval.get("sts_repair_task_level_improvements"),
        "sts_task_level_regressions": private_eval.get("sts_repair_task_level_regressions"),
        "residual_family_pass_rates": private_eval.get("concept_family_pass_rates", {}),
        "residual_counts": private_eval.get("concept_residual_counts", {}),
        "residual_examples": private_eval.get("residuals", [])[:8],
        "parameter_count": sts_on.get("parameter_count") or sts_off.get("parameter_count"),
        "backend": sts_on.get("backend") or sts_off.get("backend"),
        "memory": {"sts_on": sts_on.get("memory"), "sts_off": sts_off.get("memory")},
        "internal_semantic_routing_sts_on": get_path(sts_on, ["decoder_constraints", "internal_semantic_routing"], {}),
        "internal_plan_loss_curve_sts_on": get_path(sts_on, ["train", "plan_loss_curve"], []),
        "verifier_wall_time_ms": verifier_wall_ms,
    }


def compare_arms(arm_reports: dict[str, Any]) -> dict[str, Any]:
    by_arm = {}
    for arm_id, report in arm_reports.items():
        summary = dict_or_empty(report.get("summary"))
        by_arm[arm_id] = {
            "sts_on_verifier_pass_rate": summary.get("sts_on_verifier_pass_rate"),
            "sts_off_verifier_pass_rate": summary.get("sts_off_verifier_pass_rate"),
            "sts_on_rank1_pass_rate": summary.get("sts_on_rank1_pass_rate"),
            "sts_on_pass_if_any_rate": summary.get("sts_on_pass_if_any_rate"),
            "sts_delta": summary.get("sts_delta"),
            "accepted_candidate_rate": summary.get("accepted_candidate_rate"),
            "syntax_pass_rate_sts_on": summary.get("syntax_pass_rate_sts_on"),
            "raw_syntax_pass_rate_sts_on": summary.get("raw_syntax_pass_rate_sts_on"),
            "grammar_repair_changed_rate_sts_on": summary.get("grammar_repair_changed_rate_sts_on"),
            "grammar_repair_fallback_rate_sts_on": summary.get("grammar_repair_fallback_rate_sts_on"),
            "statement_skeleton_render_rate_sts_on": summary.get("statement_skeleton_render_rate_sts_on"),
            "semantic_slot_render_rate_sts_on": summary.get("semantic_slot_render_rate_sts_on"),
            "strict_action_render_rate_sts_on": summary.get("strict_action_render_rate_sts_on"),
            "structural_action_render_rate": summary.get("structural_action_render_rate"),
            "structural_action_candidate_rows": summary.get("structural_action_candidate_rows"),
            "structural_action_syntax_pass_rate": summary.get("structural_action_syntax_pass_rate"),
            "structural_action_fallback_rate": summary.get("structural_action_fallback_rate"),
            "semantic_plan_supported_rate_sts_on": summary.get("semantic_plan_supported_rate_sts_on"),
            "learned_internal_semantic_route_rate_sts_on": summary.get("learned_internal_semantic_route_rate_sts_on"),
            "predicted_return_shape_rate_sts_on": summary.get("predicted_return_shape_rate_sts_on"),
            "internal_semantic_routing_sts_on": summary.get("internal_semantic_routing_sts_on"),
            "parameter_count": summary.get("parameter_count"),
            "sts_task_level_regressions": summary.get("sts_task_level_regressions"),
        }
    sym_rate = float(get_path(by_arm, ["symliquid_style", "sts_on_verifier_pass_rate"], 0.0) or 0.0)
    tx_rate = float(get_path(by_arm, ["transformer_control", "sts_on_verifier_pass_rate"], 0.0) or 0.0)
    winner = "symliquid_style" if sym_rate >= tx_rate else "transformer_control"
    if tx_rate > sym_rate:
        practical_lane = "transformer_control"
        symliquid_status = "protected_discovery_comparator_until_repeated_strict_win"
    elif sym_rate > tx_rate:
        practical_lane = "symliquid_style"
        symliquid_status = "strict_path_leader_on_this_run_needs_repetition"
    else:
        practical_lane = "undecided_tie"
        symliquid_status = "protected_discovery_comparator"
    return {
        "by_arm": by_arm,
        "symliquid_minus_transformer_sts_on_verifier_pass_rate": round(sym_rate - tx_rate, 6),
        "winner_by_sts_on_verifier_pass_rate": winner,
        "practical_survival_lane": practical_lane,
        "symliquid_discovery_lane_status": symliquid_status,
        "score_semantics": "Private token-decoder smoke only; not public calibration, distillation, or promotion evidence.",
    }


def build_token_gates(
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    target_vocab: dict[str, int],
    arm_reports: dict[str, Any],
    candidates: list[dict[str, Any]],
    param_delta: float,
    gap_report: dict[str, Any],
    no_cheat: dict[str, Any],
    family_disjoint_report: dict[str, Any] | None = None,
    broad_private_heldout_report: dict[str, Any] | None = None,
    teacher_training: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    budget = dict_or_empty(config.get("matched_budget"))
    safety = dict_or_empty(config.get("safety"))
    success = dict_or_empty(config.get("success_criteria"))
    no_cheat_summary = dict_or_empty(no_cheat.get("summary"))
    family_disjoint_report = dict_or_empty(family_disjoint_report)
    family_disjoint_summary = dict_or_empty(family_disjoint_report.get("summary"))
    family_disjoint_overlap = dict_or_empty(family_disjoint_summary.get("split_overlap_audit"))
    broad_private_heldout_report = dict_or_empty(broad_private_heldout_report)
    broad_private_heldout_summary = dict_or_empty(broad_private_heldout_report.get("summary"))
    broad_private_heldout_overlap = dict_or_empty(broad_private_heldout_summary.get("split_overlap_audit"))
    transformer_summary = dict_or_empty(dict_or_empty(arm_reports.get("transformer_control")).get("summary"))
    transformer_raw_floor = float(success.get("transformer_raw_syntax_pass_rate_floor") or 0.0)
    transformer_rank1_floor = int(success.get("transformer_rank1_passed_floor") or 0)
    transformer_pass_any_floor = int(success.get("transformer_pass_if_any_passed_floor") or 0)
    teacher_training = dict_or_empty(teacher_training)
    teacher_summary = dict_or_empty(teacher_training.get("summary"))
    teacher_allowed = bool(safety.get("teacher_distillation_allowed", False))
    teacher_active = bool(teacher_summary.get("enabled")) and int(teacher_summary.get("accepted_code_lm_training_rows") or 0) > 0
    pretraining_cfg = pretraining_initialization_config(config)
    pretraining_enabled = bool(pretraining_cfg.get("enabled", False))
    expected_arms = execution_arm_names(config, split="in_family")
    matched_two_arm_scope = set(expected_arms) == set(TOKEN_DECODER_ARM_IDS)
    pretraining_evidence = {
        arm: get_path(arm_reports, [arm, "views", "sts_on", "train", "pretraining_initialization"], {})
        for arm in expected_arms
    }
    full_state_cfg = full_state_pretraining_config(config)
    full_state_enabled = bool(full_state_cfg.get("enabled", False))
    full_state_evidence = {
        arm: get_path(arm_reports, [arm, "views", "sts_on", "train", "full_state_pretraining"], {})
        for arm in expected_arms
    }
    strict_generator_cfg = strict_generator_checkpoint_config(config)
    strict_generator_checkpoint_enabled = bool(strict_generator_cfg.get("enabled", False))
    strict_generator_checkpoint_evidence = {
        arm: get_path(arm_reports, [arm, "views", "sts_on", "train", "strict_generator_pretraining_checkpoint"], {})
        for arm in expected_arms
        if arm == str(strict_generator_cfg.get("arm_id") or "transformer_control")
    }
    gates = [
        gate(
            "gap_report_recorded_for_residual_context",
            bool(gap_report),
            {
                "trigger_state": gap_report.get("trigger_state"),
                "summary": gap_report.get("summary", {}),
                "score_semantics": "Residual context only; stale body-template/gap reports are not strict learned-generation success evidence.",
            },
            "soft",
        ),
        gate("private_train_rows_loaded", len(train_rows) > 0, f"train_rows={len(train_rows)}", "hard"),
        gate("private_eval_rows_loaded", len(eval_rows) > 0, f"eval_rows={len(eval_rows)}", "hard"),
        gate("target_token_vocab_built", len(target_vocab) > 8, f"target_vocab={len(target_vocab)}", "hard"),
        gate("public_training_forbidden", not safety.get("public_training_allowed", True), safety, "hard"),
        gate("teacher_calls_forbidden", not safety.get("teacher_calls_allowed", True), safety, "hard"),
        gate(
            "teacher_distillation_governed_or_forbidden",
            (not teacher_allowed and not teacher_active)
            or (
                teacher_allowed
                and bool(teacher_summary.get("gate_green"))
                and bool(teacher_summary.get("manifest_safety_clean"))
                and int(teacher_summary.get("public_overlap_hits") or 0) == 0
                and int(teacher_summary.get("holdout_overlap_hits") or 0) == 0
                and int(teacher_summary.get("holdout_family_code_lm_training_rows") or 0) == 0
            ),
            teacher_summary,
            "hard",
        ),
        gate("model_promotion_forbidden", not safety.get("model_promotion_allowed", True), safety, "hard"),
        gate(
            "pretraining_initialization_applied_when_enabled",
            (not pretraining_enabled) or all(bool(dict_or_empty(row).get("active")) for row in pretraining_evidence.values()),
            pretraining_evidence,
            "hard",
        ),
        gate(
            "full_state_pretraining_applied_when_enabled",
            (not full_state_enabled) or all(bool(dict_or_empty(row).get("active")) for row in full_state_evidence.values()),
            full_state_evidence,
            "hard",
        ),
        gate(
            "strict_generator_checkpoint_applied_when_enabled",
            (not strict_generator_checkpoint_enabled)
            or all(bool(dict_or_empty(row).get("active")) for row in strict_generator_checkpoint_evidence.values()),
            strict_generator_checkpoint_evidence,
            "hard",
        ),
        gate("external_inference_zero", all(int(row.get("external_inference_calls") or 0) == 0 for row in candidates), 0, "hard"),
        gate(
            "requested_arms_evaluated",
            set(arm_reports) == set(expected_arms),
            {"expected_arms": expected_arms, "actual_arms": list(arm_reports), "matched_two_arm_scope": matched_two_arm_scope},
            "hard",
        ),
        gate(
            "requested_arms_emit_token_decoded_candidate_code_rows",
            both_arms_emit_token_code(arm_reports, required_arms=expected_arms),
            candidate_counts_by_arm(candidates),
            "hard",
        ),
        gate(
            "both_arms_evaluated_for_substrate_verdict",
            (not matched_two_arm_scope) or set(arm_reports) == set(TOKEN_DECODER_ARM_IDS),
            {"expected_arms": expected_arms, "actual_arms": list(arm_reports), "matched_two_arm_scope": matched_two_arm_scope},
            "hard",
        ),
        gate(
            "both_arms_emit_token_decoded_candidate_code_rows_for_substrate_verdict",
            (not matched_two_arm_scope) or both_arms_emit_token_code(arm_reports),
            candidate_counts_by_arm(candidates),
            "hard",
        ),
        gate("grammar_repair_metadata_recorded", grammar_repair_metadata_recorded(candidates), grammar_repair_counts(candidates), "hard"),
        gate("raw_syntax_measured_before_repair", raw_syntax_measured(candidates), grammar_repair_counts(candidates), "hard"),
        gate("raw_syntax_pass_rate_nonzero", raw_syntax_pass_nonzero(arm_reports, expected_arms), syntax_evidence(arm_reports), "hard"),
        gate("fallback_return_rate_zero", fallback_return_rate_zero(arm_reports, expected_arms), syntax_evidence(arm_reports), "hard"),
        gate(
            "no_cheat_evidence_filter_recorded",
            bool(no_cheat_summary),
            no_cheat_summary,
            "hard",
        ),
        gate(
            "no_fallback_returns_in_no_cheat_audit",
            int(no_cheat_summary.get("fallback_return_rows") or 0) == 0,
            no_cheat_summary,
            "hard",
        ),
        gate(
            "task_identity_branches_absent",
            int(no_cheat_summary.get("task_identity_keyed_rows") or 0) == 0,
            no_cheat_summary,
            "hard",
        ),
        gate(
            "canned_or_null_rows_quarantined",
            bool(no_cheat_summary.get("disallowed_rows_not_promotion_eligible", False)),
            no_cheat_summary,
            "hard",
        ),
        gate(
            "no_cheat_private_evidence_rows_present",
            int(no_cheat_summary.get("eligible_generated_rows") or 0) > 0,
            no_cheat_summary,
            "soft",
        ),
        gate("post_repair_syntax_pass_rate_nonzero", post_repair_syntax_pass_nonzero(arm_reports, expected_arms), syntax_evidence(arm_reports), "hard"),
        gate("body_decode_metadata_recorded", body_decode_metadata_recorded(candidates), body_structure_counts(candidates), "hard"),
        gate("candidate_rows_do_not_embed_tests_or_solutions", candidate_rows_do_not_embed_forbidden(candidates), "no tests/solution fields on candidate rows", "hard"),
        gate(
            "template_selector_mode_disallowed",
            candidate_generation_modes_allowed(config, candidates),
            {"allowed_generated_modes": sorted(allowed_generated_candidate_modes(config))},
            "hard",
        ),
        gate(
            "transformer_control_real",
            get_path(arm_reports, ["transformer_control", "views", "sts_on", "substrate"], "") == "torch_transformer_encoder_token_decoder",
            get_path(arm_reports, ["transformer_control", "views", "sts_on", "substrate"], ""),
            "hard",
        ),
        gate(
            "symliquid_style_adapter_real",
            ("symliquid_style" not in expected_arms)
            or get_path(arm_reports, ["symliquid_style", "views", "sts_on", "substrate"], "") == "torch_symliquid_style_recurrent_vsa_token_decoder",
            get_path(arm_reports, ["symliquid_style", "views", "sts_on", "substrate"], ""),
            "hard",
        ),
        gate(
            "configured_sts_controls_ran",
            {"sts_on", "sts_off"}.issubset(set(dict_or_empty(arm_reports.get("transformer_control")).get("views")))
            and (
                "symliquid_style" not in expected_arms
                or "sts_on" in set(dict_or_empty(dict_or_empty(arm_reports.get("symliquid_style")).get("views")))
            ),
            {arm: list(dict_or_empty(row.get("views"))) for arm, row in arm_reports.items()},
            "hard",
        ),
        gate(
            "same_private_verifier_for_both_arms",
            all(int(get_path(row, ["summary", "verifier_candidate_attempt_count"], 0) or 0) > 0 for row in arm_reports.values()),
            {arm: get_path(row, ["summary", "verifier_candidate_attempt_count"], 0) for arm, row in arm_reports.items()},
            "hard",
        ),
        gate(
            "matched_parameter_budget_within_smoke_tolerance",
            param_delta <= float(budget.get("parameter_match_tolerance") or 0.12),
            {"parameter_match_delta": round(param_delta, 6), "tolerance": budget.get("parameter_match_tolerance")},
            "hard",
        ),
        gate(
            "matched_parameter_budget_within_trusted_tolerance",
            param_delta <= float(budget.get("trusted_parameter_match_tolerance") or 0.08),
            {"parameter_match_delta": round(param_delta, 6), "trusted_tolerance": budget.get("trusted_parameter_match_tolerance")},
            "soft",
        ),
    ]
    if transformer_raw_floor > 0.0:
        gates.append(
            gate(
                "transformer_raw_syntax_pass_rate_floor",
                float(transformer_summary.get("raw_syntax_pass_rate_sts_on") or 0.0) >= transformer_raw_floor,
                {
                    "actual": transformer_summary.get("raw_syntax_pass_rate_sts_on"),
                    "floor": transformer_raw_floor,
                },
                "hard",
            )
        )
    if transformer_rank1_floor > 0:
        gates.append(
            gate(
                "transformer_rank1_passed_floor",
                int(transformer_summary.get("sts_on_rank1_passed") or 0) >= transformer_rank1_floor,
                {
                    "actual": transformer_summary.get("sts_on_rank1_passed"),
                    "floor": transformer_rank1_floor,
                },
                "hard",
            )
        )
    if transformer_pass_any_floor > 0:
        gates.append(
            gate(
                "transformer_pass_if_any_passed_floor",
                int(transformer_summary.get("sts_on_pass_if_any_passed") or 0) >= transformer_pass_any_floor,
                {
                    "actual": transformer_summary.get("sts_on_pass_if_any_passed"),
                    "floor": transformer_pass_any_floor,
                },
                "hard",
            )
        )
    if bool(dict_or_empty(dict_or_empty(config.get("data")).get("family_disjoint_eval")).get("enabled", False)):
        gates.extend(
            [
                gate(
                    "family_disjoint_eval_recorded",
                    bool(family_disjoint_summary.get("active", family_disjoint_report.get("active", False)))
                    and int(family_disjoint_summary.get("eval_rows") or 0) > 0,
                    family_disjoint_summary,
                    "hard",
                ),
                gate(
                    "family_disjoint_min_holdout_families",
                    int(family_disjoint_summary.get("holdout_family_count") or 0)
                    >= int(dict_or_empty(dict_or_empty(config.get("data")).get("family_disjoint_eval")).get("min_holdout_families") or 6),
                    family_disjoint_summary,
                    "hard",
                ),
                gate(
                    "family_disjoint_train_eval_family_overlap_zero",
                    int(family_disjoint_overlap.get("family_overlap_count") or 0) == 0,
                    family_disjoint_overlap,
                    "hard",
                ),
                gate(
                    "family_disjoint_transformer_result_recorded",
                    family_disjoint_summary.get("transformer_sts_on_verifier_pass_rate") is not None
                    and family_disjoint_summary.get("transformer_sts_on_pass_if_any_rate") is not None,
                    family_disjoint_summary,
                    "hard",
                ),
                gate(
                    "family_disjoint_teacher_training_holdout_exclusion_clean",
                    int(family_disjoint_summary.get("teacher_code_lm_training_rows_in_holdout_families") or 0) == 0,
                    family_disjoint_summary,
                    "hard",
                ),
            ]
        )
    broad_cfg = dict_or_empty(dict_or_empty(config.get("data")).get("broad_private_heldout_eval"))
    if bool(broad_cfg.get("enabled", False)):
        gates.extend(
            [
                gate(
                    "broad_private_heldout_eval_recorded",
                    bool(broad_private_heldout_summary.get("active", broad_private_heldout_report.get("active", False)))
                    and int(broad_private_heldout_summary.get("eval_rows") or 0) > 0,
                    broad_private_heldout_summary,
                    "hard",
                ),
                gate(
                    "broad_private_heldout_min_eval_rows",
                    int(broad_private_heldout_summary.get("eval_rows") or 0) >= int(broad_cfg.get("min_eval_rows") or 200),
                    broad_private_heldout_summary,
                    "hard",
                ),
                gate(
                    "broad_private_heldout_min_eval_families",
                    int(broad_private_heldout_summary.get("eval_family_count") or 0)
                    >= int(broad_cfg.get("min_eval_families") or 30),
                    broad_private_heldout_summary,
                    "hard",
                ),
                gate(
                    "broad_private_heldout_train_eval_overlap_zero",
                    all(int(broad_private_heldout_overlap.get(key) or 0) == 0 for key in broad_private_heldout_overlap if key.endswith("_count")),
                    broad_private_heldout_overlap,
                    "hard",
                ),
                gate(
                    "broad_private_heldout_transformer_result_recorded",
                    broad_private_heldout_summary.get("transformer_sts_on_verifier_pass_rate") is not None,
                    broad_private_heldout_summary,
                    "hard",
                ),
            ]
        )
    return gates


def data_contract(
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    teacher_training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data_cfg = dict_or_empty(config.get("data"))
    structure_cfg = dict_or_empty(config.get("body_structure_decoder"))
    teacher_summary = dict_or_empty(dict_or_empty(teacher_training).get("summary"))
    target_mode = str(structure_cfg.get("target_mode") or "body_tokens")
    return {
        "train_jsonl": data_cfg.get("train_jsonl"),
        "eval_jsonl": data_cfg.get("eval_jsonl"),
        "train_rows": len(train_rows),
        "teacher_code_lm_training_rows": int(teacher_summary.get("accepted_code_lm_training_rows") or 0),
        "teacher_training_manifest": teacher_summary.get("manifest"),
        "teacher_training_gate_green": teacher_summary.get("gate_green"),
        "teacher_training_public_overlap_hits": teacher_summary.get("public_overlap_hits"),
        "teacher_training_holdout_overlap_hits": teacher_summary.get("holdout_overlap_hits"),
        "teacher_training_holdout_family_rows": teacher_summary.get("holdout_family_code_lm_training_rows"),
        "teacher_training_external_inference_calls": teacher_summary.get("external_inference_calls", 0),
        "eval_rows": len(eval_rows),
        "private_train_solution_body_tokens_used_as_training_targets": target_mode == "body_tokens",
        "private_train_statement_skeleton_tokens_used_as_training_targets": target_mode == "statement_skeleton_v1",
        "private_train_semantic_slot_tokens_used_as_training_targets": target_mode == "semantic_slots_v1",
        "private_train_strict_action_tokens_used_as_training_targets": target_mode == "strict_action_tokens_v1",
        "strict_action_renderer_rows_are_diagnostic_only": target_mode == "strict_action_tokens_v1",
        "fallback_returns_allowed": bool(structure_cfg.get("fallback_returns_allowed", False)),
        "private_train_tests_used_for_training": False,
        "private_eval_tests_used_by_private_verifier": True,
        "private_eval_tests_loaded_into_features": False,
        "private_eval_solutions_loaded_into_features": False,
        "tests_or_solutions_loaded_into_candidate_generation_features": False,
        "visible_contract_semantic_beam": visible_contract_semantic_beam_summary(config),
        "internal_semantic_routing": internal_semantic_routing_summary(config),
        "public_prompts_tests_solutions_used": False,
        "public_training_rows": 0,
    }


def visible_contract_semantic_beam_summary(config: dict[str, Any]) -> dict[str, Any]:
    beam_cfg = dict_or_empty(dict_or_empty(config.get("body_structure_decoder")).get("visible_contract_semantic_beam"))
    return {
        "enabled": bool(beam_cfg.get("enabled", False)),
        "beam_policy": str(beam_cfg.get("beam_policy") or "shared_low_rank_visible_contract_semantic_prior"),
        "uses_only_allowed_decoder_contract_fields": bool(beam_cfg.get("uses_only_allowed_decoder_contract_fields", True)),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "fields": list(beam_cfg.get("fields") or []),
    }


def internal_semantic_routing_summary(config: dict[str, Any], *, view: str | None = None) -> dict[str, Any]:
    routing_cfg = dict_or_empty(dict_or_empty(config.get("body_structure_decoder")).get("internal_semantic_routing"))
    raw_enabled_views = routing_cfg.get("enabled_views", ["sts_on"])
    enabled_views = [str(item) for item in raw_enabled_views] if isinstance(raw_enabled_views, list) else ["sts_on"]
    configured_enabled = bool(routing_cfg.get("enabled", False))
    view_enabled = configured_enabled and (view is None or str(view) in enabled_views)
    return {
        "enabled": view_enabled,
        "configured_enabled": configured_enabled,
        "current_view": view,
        "enabled_views": enabled_views,
        "routing_policy": str(routing_cfg.get("routing_policy") or "matched_learned_plan_router_from_sts_on_visible_contract_fields"),
        "auxiliary_plan_loss_weight": float(routing_cfg.get("auxiliary_plan_loss_weight") or 0.0) if view_enabled else 0.0,
        "plan_router_scale": float(routing_cfg.get("plan_router_scale") or 0.0) if view_enabled else 0.0,
        "learned_route_top_k": int(routing_cfg.get("learned_route_top_k") or 0) if view_enabled else 0,
        "prototype_route_memory": bool(routing_cfg.get("prototype_route_memory", False)) if view_enabled else False,
        "prototype_route_weight": float(routing_cfg.get("prototype_route_weight") or 0.0) if view_enabled else 0.0,
        "prototype_route_keep_rate": float(routing_cfg.get("prototype_route_keep_rate", 1.0)) if view_enabled else 0.0,
        "contract_fingerprint_route_memory": bool(routing_cfg.get("contract_fingerprint_route_memory", False)) if view_enabled else False,
        "contract_fingerprint_route_top_k": int(routing_cfg.get("contract_fingerprint_route_top_k") or 0) if view_enabled else 0,
        "contract_fingerprint_route_keep_rate": float(routing_cfg.get("contract_fingerprint_route_keep_rate", 1.0)) if view_enabled else 0.0,
        "contract_feature_route_memory": bool(routing_cfg.get("contract_feature_route_memory", False)) if view_enabled else False,
        "contract_feature_route_top_k": int(routing_cfg.get("contract_feature_route_top_k") or 0) if view_enabled else 0,
        "contract_feature_route_keep_rate": float(routing_cfg.get("contract_feature_route_keep_rate", 1.0)) if view_enabled else 0.0,
        "visible_text_prototype_route_memory": bool(routing_cfg.get("visible_text_prototype_route_memory", False)) if view_enabled else False,
        "visible_text_prototype_route_top_k": int(routing_cfg.get("visible_text_prototype_route_top_k") or 0) if view_enabled else 0,
        "visible_text_prototype_route_keep_rate": float(routing_cfg.get("visible_text_prototype_route_keep_rate", 1.0)) if view_enabled else 0.0,
        "route_dropout_policy": str(routing_cfg.get("route_dropout_policy") or "deterministic_task_hash_keep_rate"),
        "target": str(routing_cfg.get("target") or "first_private_train_semantic_slot_plan_token"),
        "source": str(routing_cfg.get("source") or "visible_contract_fields_in_sts_on_text"),
        "uses_only_allowed_decoder_contract_fields": bool(routing_cfg.get("uses_only_allowed_decoder_contract_fields", True)),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "teacher_used": False,
        "matched_for_both_arms": bool(routing_cfg.get("matched_for_both_arms", True)),
    }


def matched_budget_report(config: dict[str, Any], sym_dims: dict[str, int], transformer_dims: dict[str, int]) -> dict[str, Any]:
    budget = dict_or_empty(config.get("matched_budget"))
    return {
        "budget_id": budget.get("budget_id"),
        "seeds": budget.get("seeds"),
        "max_source_tokens": budget.get("max_source_tokens"),
        "max_target_tokens": budget.get("max_target_tokens"),
        "epochs": budget.get("epochs"),
        "batch_size": budget.get("batch_size"),
        "learning_rate": budget.get("learning_rate"),
        "first_target_token_loss_weight": budget.get("first_target_token_loss_weight"),
        "fanout_top_k": budget.get("fanout_top_k"),
        "grammar_decode_top_k": budget.get("grammar_decode_top_k"),
        "private_candidate_timeout_seconds": budget.get("private_candidate_timeout_seconds"),
        "internal_semantic_routing": internal_semantic_routing_summary(config),
        "symliquid_style_dims": sym_dims,
        "transformer_control_dims": transformer_dims,
    }


def compare_against_body_template(arm: str, token_comparisons: dict[str, Any], gap_report: dict[str, Any]) -> dict[str, Any]:
    token_rate = float(get_path(token_comparisons, ["by_arm", arm, "sts_on_verifier_pass_rate"], 0.0) or 0.0)
    # The source gap report does not preserve the original aggregate by-arm rate,
    # so use the source comparator report when available.
    source = read_json(resolve(str(get_path(gap_report, ["source_report"], "reports/neural_seed_code_proposer_comparator.json"))))
    template_rate = float(get_path(source, ["comparisons", "by_arm", arm, "sts_on_verifier_pass_rate"], 0.0) or 0.0)
    return {
        "token_decoder_sts_on_verifier_pass_rate": token_rate,
        "body_template_sts_on_verifier_pass_rate": template_rate,
        "delta": round(token_rate - template_rate, 6),
    }


def classify_gap_failure(task: dict[str, Any], result: dict[str, Any], top: dict[str, Any], train_hashes: set[str]) -> str:
    if result.get("passed"):
        return "passed"
    stage = str(result.get("verification_stage") or "")
    expected_hash = stable_hash(str(task.get("solution_body") or "").strip())
    if expected_hash not in train_hashes:
        return "missing_body_coverage"
    if stage in {"lint_parse_failed", "candidate_compile_failed", "runtime_failed", "runtime_load_failure"}:
        return "signature_or_adapter_or_syntax"
    if not top:
        return "missing_candidate"
    return "ranking_or_behavior_mismatch"


def gap_family(task: dict[str, Any]) -> str:
    return str(
        get_path(task, ["decoder_contract", "type_family"], "")
        or task.get("targeted_private_residual_family_v3")
        or task.get("residual_concept")
        or "unknown"
    )


def residual_mining_context(gap_report: dict[str, Any]) -> dict[str, Any]:
    mining_path = resolve("reports/neural_seed_residual_mining.json")
    mining = read_json(mining_path)
    if not mining:
        mining = {}
    pressure_rows = mining.get("next_private_pressure")
    if not isinstance(pressure_rows, list):
        pressure_rows = []
    pressure_by_family = {}
    for row in pressure_rows:
        if not isinstance(row, dict):
            continue
        family = str(row.get("family") or "")
        if not family:
            continue
        pressure_by_family[family] = {
            "family": family,
            "symliquid_only": int(row.get("symliquid_only") or 0),
            "both_fail": int(row.get("both_fail") or 0),
            "recommended_action": str(row.get("recommended_action") or ""),
        }
    gap_summary = dict_or_empty(gap_report.get("summary"))
    mining_summary = dict_or_empty(mining.get("summary"))
    return {
        "source": rel(mining_path) if mining else "",
        "available": bool(mining),
        "summary": {
            "source_gap_counts": gap_summary.get("gap_counts"),
            "mining_trigger_state": mining.get("trigger_state") if mining else "",
            "symliquid_only_win_count": mining_summary.get("symliquid_only_win_count"),
            "both_fail_count": mining_summary.get("both_fail_count"),
            "pressure_families": len(pressure_by_family),
        },
        "pressure_by_family": pressure_by_family,
        "score_semantics": (
            "Residual mining tags are analysis labels from existing private diagnostic reports. They are not "
            "generation features, not additional training data, and do not expose eval tests or solutions."
        ),
    }


def residual_tag_for_task(task: dict[str, Any], residual_context: dict[str, Any]) -> dict[str, Any]:
    family = gap_family(task)
    pressure_by_family = dict_or_empty(residual_context.get("pressure_by_family"))
    pressure = dict_or_empty(pressure_by_family.get(family))
    return {
        "family": family,
        "source": residual_context.get("source") or "",
        "available": bool(pressure),
        "symliquid_only": int(pressure.get("symliquid_only") or 0),
        "both_fail": int(pressure.get("both_fail") or 0),
        "recommended_action": str(pressure.get("recommended_action") or "none_recorded"),
        "used_for_generation": False,
    }


def first_examples(rows: list[dict[str, Any]], status: str, limit: int = 12) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        if row.get("gap_status") != status:
            continue
        examples.append(
            {
                "task_id": row.get("task_id"),
                "family": row.get("family"),
                "entry_point": row.get("entry_point"),
                "expected_body_in_train_template_library": row.get("expected_body_in_train_template_library"),
                "symliquid": row.get("arms", {}).get("symliquid_style"),
                "transformer": row.get("arms", {}).get("transformer_control"),
            }
        )
        if len(examples) >= limit:
            break
    return examples


if __name__ == "__main__":
    raise SystemExit(main())
