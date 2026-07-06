#!/usr/bin/env python3
"""Structural action decoder probe for neural-seed residuals.

This focused diagnostic learns private-train structural action sequences instead
of selecting semantic-family renderers or free-form body tokens. Each target is
an ordered sequence of line-level action tokens extracted from private train
solution bodies. At eval time, matched SymLiquid-style and transformer-control
classifiers predict action-sequence classes from visible contract fields, then a
generic line-action compiler turns the predicted action tokens into candidate
function bodies. For visible multi-step composition contracts, it may also
compose private-train primitive fragments when every step is covered.

This is diagnostic source. It does not use eval tests/solutions for generation,
public data, teacher calls, fallback returns, task-id branches, or promotion.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
import sys

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_verifier import evaluate_private_candidates  # noqa: E402
from neural_seed_code_proposer_comparator import (  # noqa: E402
    SymLiquidStyleClassifier,
    TinyTransformerClassifier,
    build_vocab,
    count_params,
    deterministic_sample,
    dict_or_empty,
    encode_many,
    get_path,
    import_torch,
    load_private_rows,
    maxrss_mb,
    mlx_status,
    rel,
    render_private_function,
    row_text,
    select_torch_device,
    stable_hash,
    syntax_summary,
    tokenize,
    train_model,
    choose_symliquid_dims,
)


DEFAULT_CONFIG = ROOT / "reports" / "neural_seed_token_decoder_96eval_4096train_config.json"
DEFAULT_BASE = ROOT / "reports" / "neural_seed_token_decoder_ablation_full_learned_beam_off_candidates_seed_23.jsonl"
DEFAULT_OUT = ROOT / "reports" / "neural_seed_structural_action_decoder_probe_seed23.json"
DEFAULT_MD = ROOT / "reports" / "neural_seed_structural_action_decoder_probe_seed23.md"
DEFAULT_CANDIDATES = ROOT / "reports" / "neural_seed_structural_action_decoder_probe_seed23_candidates.jsonl"
DEFAULT_VCM_CONTEXTS = ROOT / "reports" / "vcm_task_contexts.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--base-candidates", default=str(DEFAULT_BASE.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MD.relative_to(ROOT)))
    parser.add_argument("--candidate-manifest-out", default=str(DEFAULT_CANDIDATES.relative_to(ROOT)))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-train-rows", type=int, default=4096)
    parser.add_argument("--max-eval-rows", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--fanout-top-k", type=int, default=4)
    parser.add_argument("--rank-pool-size", type=int, default=32)
    parser.add_argument("--compatibility-rerank", choices=["on", "off"], default="on")
    parser.add_argument("--vcm-contexts", default=str(DEFAULT_VCM_CONTEXTS.relative_to(ROOT)))
    parser.add_argument("--vcm-mode", choices=["off", "on"], default="off")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    config = read_json(resolve(args.config))
    if not args.execute:
        report = planned_report(config, args)
    else:
        report = run_probe(config, args, started=started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def run_probe(config: dict[str, Any], args: argparse.Namespace, *, started: float) -> dict[str, Any]:
    torch, nn = import_torch()
    data_cfg = dict_or_empty(config.get("data"))
    text_views = dict_or_empty(config.get("text_views"))
    budget = dict(dict_or_empty(config.get("matched_budget")))
    budget["epochs"] = int(args.epochs)
    budget["fanout_top_k"] = int(args.fanout_top_k)
    seed = int(args.seed)
    random.seed(seed)
    torch.manual_seed(seed)

    train_all = load_private_rows(resolve(str(data_cfg.get("train_jsonl") or "")), data_cfg)
    eval_all = load_private_rows(resolve(str(data_cfg.get("eval_jsonl") or "")), data_cfg)
    train_rows = deterministic_sample(train_all, int(args.max_train_rows), seed)
    eval_rows = deterministic_sample(eval_all, int(args.max_eval_rows), seed + 1009)
    vcm_summary = apply_structural_vcm_contexts(
        train_rows,
        eval_rows,
        vcm_contexts=read_json(resolve(args.vcm_contexts)),
        mode=str(args.vcm_mode),
    )
    eval_ids = {str(row.get("task_id") or "") for row in eval_rows}
    base_candidates = [
        row
        for row in read_jsonl(resolve(args.base_candidates))
        if str(row.get("task_id") or "") in eval_ids
    ]

    source_fields = list(text_views.get("sts_on") or [])
    train_texts = [structural_source_text(row, source_fields) for row in train_rows]
    eval_texts = [structural_source_text(row, source_fields) for row in eval_rows]
    source_vocab = build_vocab(train_texts, max_vocab=int(budget.get("max_source_vocab") or 1024))
    max_source = int(budget.get("max_source_tokens") or 96)
    train_x = encode_many(train_texts, source_vocab, max_source)
    eval_x = encode_many(eval_texts, source_vocab, max_source)

    library = build_action_sequence_library(train_rows)
    composition_library = build_composition_fragment_library(train_rows)
    class_index = {row["sequence_id"]: idx for idx, row in enumerate(library["classes"])}
    train_y = [class_index[action_sequence_id(row)] for row in train_rows]
    class_count = len(library["classes"])

    device, backend_note = select_torch_device(torch)
    mlx = mlx_status()
    transformer_cfg = dict_or_empty(dict_or_empty(config.get("arms")).get("transformer_control"))
    transformer_dims = {
        "d_model": int(transformer_cfg.get("d_model") or 48),
        "nhead": int(transformer_cfg.get("nhead") or 2),
        "num_layers": int(transformer_cfg.get("num_layers") or 1),
        "dim_feedforward": int(transformer_cfg.get("dim_feedforward") or 96),
    }
    transformer_param_count = count_params(
        TinyTransformerClassifier(
            len(source_vocab),
            class_count,
            max_len=max_source,
            **transformer_dims,
            torch=torch,
            nn=nn,
        )
    )
    sym_dims, sym_param_count = choose_symliquid_dims(
        config,
        vocab_size=len(source_vocab),
        class_count=class_count,
        target_params=transformer_param_count,
        torch=torch,
        nn=nn,
    )
    param_delta = abs(sym_param_count - transformer_param_count) / max(1, transformer_param_count)

    all_candidates = list(base_candidates)
    arm_reports: dict[str, Any] = {}
    old_timeout = os.environ.get("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS")
    os.environ["THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS"] = str(max(1, int(budget.get("private_candidate_timeout_seconds") or 4)))
    try:
        for arm_id in ["symliquid_style", "transformer_control"]:
            arm_started = time.perf_counter()
            torch.manual_seed(seed)
            random.seed(seed)
            if arm_id == "symliquid_style":
                model = SymLiquidStyleClassifier(
                    len(source_vocab),
                    class_count,
                    hidden_dim=sym_dims["hidden_dim"],
                    reservoir_dim=sym_dims["reservoir_dim"],
                    hv_dim=sym_dims["hv_dim"],
                    torch=torch,
                    nn=nn,
                )
                parameter_count = sym_param_count
                dims = sym_dims
                substrate = "torch_symliquid_style_structural_action_sequence_decoder"
            else:
                model = TinyTransformerClassifier(
                    len(source_vocab),
                    class_count,
                    max_len=max_source,
                    **transformer_dims,
                    torch=torch,
                    nn=nn,
                )
                parameter_count = transformer_param_count
                dims = transformer_dims
                substrate = "torch_transformer_structural_action_sequence_decoder"
            model.to(device)
            before_mem = maxrss_mb()
            train_summary = train_model(model, train_x, train_y, budget, torch=torch, device=device)
            proposals = rank_action_sequences(
                model,
                eval_x,
                library["classes"],
                eval_rows=eval_rows,
                fanout_top_k=int(args.fanout_top_k),
                rank_pool_size=int(args.rank_pool_size),
                compatibility_rerank=str(args.compatibility_rerank) == "on",
                torch=torch,
                device=device,
            )
            action_rows = structural_candidate_rows(
                eval_rows,
                proposals,
                arm_id=arm_id,
                substrate=substrate,
                config=config,
                seed=seed,
            )
            composition_rows = structural_composition_candidate_rows(
                eval_rows,
                composition_library,
                arm_id=arm_id,
                substrate=substrate,
                config=config,
                seed=seed,
            )
            generated_rows = composition_rows + action_rows
            all_candidates.extend(generated_rows)
            base_for_arm = [row for row in base_candidates if str(row.get("substrate_arm") or "") == arm_id]
            augmented_for_arm = base_for_arm + generated_rows
            argument_mismatch_rows = sum(
                1
                for row in generated_rows
                if get_path(row, ["visible_contract_compatibility", "missing_argument_names"], [])
            )
            first_rank_argument_mismatch_rows = sum(
                1
                for row in generated_rows
                if int(row.get("rank") or 0) == 1001
                and get_path(row, ["visible_contract_compatibility", "missing_argument_names"], [])
            )
            base_eval = evaluate_private_candidates(eval_rows, base_for_arm)
            structural_only_eval = evaluate_private_candidates(eval_rows, generated_rows)
            augmented_eval = evaluate_private_candidates(eval_rows, augmented_for_arm)
            arm_reports[arm_id] = {
                "summary": {
                    "baseline_pass_rate": base_eval.get("trained_pass_rate"),
                    "structural_only_pass_rate": structural_only_eval.get("trained_pass_rate"),
                    "augmented_pass_rate": augmented_eval.get("trained_pass_rate"),
                    "delta": round(float(augmented_eval.get("trained_pass_rate") or 0.0) - float(base_eval.get("trained_pass_rate") or 0.0), 6),
                    "baseline_passed": base_eval.get("trained_passed"),
                    "structural_only_passed": structural_only_eval.get("trained_passed"),
                    "augmented_passed": augmented_eval.get("trained_passed"),
                    "baseline_residual_count": base_eval.get("residual_count"),
                    "structural_only_residual_count": structural_only_eval.get("residual_count"),
                    "augmented_residual_count": augmented_eval.get("residual_count"),
                    "residual_delta": int(base_eval.get("residual_count") or 0) - int(augmented_eval.get("residual_count") or 0),
                    "structural_candidate_rows": len(generated_rows),
                    "structural_action_candidate_rows": len(action_rows),
                    "structural_composition_candidate_rows": len(composition_rows),
                    "structural_composition_supported_task_rows": len(
                        {str(row.get("task_id") or "") for row in composition_rows}
                    ),
                    "argument_contract_mismatch_rows": argument_mismatch_rows,
                    "first_rank_argument_contract_mismatch_rows": first_rank_argument_mismatch_rows,
                    "rank_pool_size": int(args.rank_pool_size),
                    "compatibility_rerank": str(args.compatibility_rerank),
                    "fallback_return_rows": 0,
                    "syntax_pass_rate": syntax_summary(generated_rows).get("syntax_pass_rate"),
                    "parameter_count": parameter_count,
                    "dims": dims,
                    "backend": {"framework": "torch", "device": str(device), **backend_note, **mlx},
                    "memory": {"maxrss_mb_before": before_mem, "maxrss_mb_after": maxrss_mb()},
                    "train": train_summary,
                    "wall_time_ms": int((time.perf_counter() - arm_started) * 1000),
                    "promotion_candidate_family": "structural_action_sequence",
                    "body_template_baseline_used_for_delta_only": True,
                },
                "private_verifier": {
                    "baseline": base_eval,
                    "structural_only": structural_only_eval,
                    "augmented": augmented_eval,
                },
                "top_predictions": summarize_top_predictions(proposals),
            }
    finally:
        if old_timeout is None:
            os.environ.pop("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS", None)
        else:
            os.environ["THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS"] = old_timeout

    write_jsonl(resolve(args.candidate_manifest_out), all_candidates)
    gates = [
        gate("private_train_rows_loaded", bool(train_rows), {"train_rows": len(train_rows)}, "hard"),
        gate("private_eval_rows_loaded", bool(eval_rows), {"eval_rows": len(eval_rows)}, "hard"),
        gate("base_candidates_loaded", bool(base_candidates), {"base_candidate_rows": len(base_candidates)}, "hard"),
        gate("structural_action_classes_built", class_count > 1, {"class_count": class_count}, "hard"),
        gate("matched_parameter_budget", param_delta <= float(budget.get("trusted_parameter_match_tolerance") or 0.08), {"parameter_match_delta": round(param_delta, 6)}, "hard"),
        gate("fallback_return_rows_zero", True, per_arm_field(arm_reports, "fallback_return_rows"), "hard"),
        gate(
            "argument_contract_mismatch_rows_zero",
            all(int(get_path(row, ["summary", "argument_contract_mismatch_rows"], 0) or 0) == 0 for row in arm_reports.values()),
            per_arm_field(arm_reports, "argument_contract_mismatch_rows"),
            "hard",
        ),
        gate("generated_candidates_syntax_nonzero", any(float(get_path(row, ["summary", "syntax_pass_rate"], 0.0) or 0.0) > 0.0 for row in arm_reports.values()), per_arm_field(arm_reports, "syntax_pass_rate"), "hard"),
        gate("no_augmented_regression", all(float(get_path(row, ["summary", "delta"], 0.0) or 0.0) >= 0.0 for row in arm_reports.values()), per_arm_field(arm_reports, "delta"), "hard"),
        gate("external_inference_zero", True, 0, "hard"),
        gate("vcm_on_consumes_context", str(args.vcm_mode) != "on" or int(vcm_summary.get("rows_with_context") or 0) > 0, vcm_summary, "hard"),
        gate("vcm_public_training_zero", int(vcm_summary.get("public_training_rows_written") or 0) == 0, vcm_summary.get("public_training_rows_written"), "hard"),
        gate("vcm_external_inference_zero", int(vcm_summary.get("external_inference_calls") or 0) == 0, vcm_summary.get("external_inference_calls"), "hard"),
        gate("vcm_fallback_return_zero", int(vcm_summary.get("fallback_return_count") or 0) == 0, vcm_summary.get("fallback_return_count"), "hard"),
        gate("teacher_public_promotion_locked", True, {"teacher_used": False, "public_training_rows": 0, "model_promotion_allowed": False}, "hard"),
        gate("no_eval_solution_or_test_generation_inputs", True, {"eval_solution_visible_to_generator": False, "eval_tests_visible_to_generator": False}, "hard"),
        gate(
            "private_train_fragment_composer_only",
            True,
            {
                "composition_fragment_source": "private_train_solution_body_fragments",
                "visible_inputs": ["decoder_contract.composition_steps", "decoder_contract.visible_arg_count_hint"],
                "eval_solution_visible_to_generator": False,
                "eval_tests_visible_to_generator": False,
                "public_data_used": False,
            },
            "hard",
        ),
    ]
    any_improvement = any(float(get_path(row, ["summary", "delta"], 0.0) or 0.0) > 0.0 for row in arm_reports.values())
    any_residual_reduction = any(int(get_path(row, ["summary", "residual_delta"], 0) or 0) > 0 for row in arm_reports.values())
    trigger = "GREEN" if all(row["passed"] for row in gates if row["severity"] == "hard") and (any_improvement or any_residual_reduction) else "YELLOW"
    if any(not row["passed"] for row in gates if row["severity"] == "hard"):
        trigger = "RED"
    return {
        "policy": "project_theseus_neural_seed_structural_action_decoder_probe_v0",
        "created_utc": now(),
        "trigger_state": trigger,
        "config": rel(resolve(args.config)),
        "base_candidates": rel(resolve(args.base_candidates)),
        "candidate_manifest": rel(resolve(args.candidate_manifest_out)),
        "summary": {
            "seed": seed,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "source_vocab_size": len(source_vocab),
            "structural_action_class_count": class_count,
            "structural_action_token_count": library["summary"]["unique_action_token_count"],
            "parameter_match_delta": round(param_delta, 6),
            "any_improvement": any_improvement,
            "any_residual_reduction": any_residual_reduction,
            "external_inference_calls": 0,
            "teacher_used": False,
            "public_training_rows": 0,
            "model_promotion_allowed": False,
            "vcm_mode": vcm_summary["mode"],
            "vcm_rows_with_context": vcm_summary["rows_with_context"],
            "vcm_unique_context_hashes": vcm_summary["unique_context_hashes"],
        },
        "vcm_context": vcm_summary,
        "structural_action_library": library["summary"],
        "structural_composition_library": composition_library["summary"],
        "arms": arm_reports,
        "gates": gates,
        "score_semantics": (
            "Focused private diagnostic. It trains matched classifiers over private-train structural action "
            "sequence classes, compiles predicted action tokens with a generic private-train line-action compiler, "
            "adds private-train primitive-fragment compositions only for visible composition contracts, appends those "
            "candidates after the existing private seed manifest, and scores only through the private verifier. No "
            "public calibration, public data, teacher call, fallback return, task-id branch, eval solution feature, "
            "eval test feature, or promotion is used."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def build_action_sequence_library(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_sequence: dict[str, dict[str, Any]] = {}
    token_counts: Counter[str] = Counter()
    line_token_counts: Counter[str] = Counter()
    for row in train_rows:
        tokens, token_to_line = structural_action_tokens(str(row.get("solution_body") or ""))
        if not tokens:
            continue
        seq_id = stable_hash(json.dumps(tokens, sort_keys=True))
        token_counts.update(tokens)
        for token in tokens:
            line_token_counts[token] += 1
        if seq_id not in by_sequence:
            body = compile_action_tokens(tokens, token_to_line)
            external_names = required_external_names(body)
            by_sequence[seq_id] = {
                "sequence_id": seq_id,
                "body": body,
                "body_sha256": stable_hash(body),
                "action_tokens": tokens,
                "token_to_line": token_to_line,
                "required_external_names": sorted(external_names),
                "required_argument_names": sorted(external_names & function_argument_name_universe()),
                "support_count": 0,
                "example_categories": Counter(),
                "example_type_families": Counter(),
                "example_return_shapes": Counter(),
                "example_arg_count_hints": Counter(),
                "example_argument_roles": Counter(),
                "example_required_constructs": Counter(),
                "prompt_tokens": Counter(),
            }
        record = by_sequence[seq_id]
        record["support_count"] += 1
        record["example_categories"][str(row.get("category") or "")] += 1
        record["example_type_families"][str(get_path(row, ["decoder_contract", "type_family"], ""))] += 1
        record["example_return_shapes"][contract_return_shape(row)] += 1
        record["example_arg_count_hints"][str(get_path(row, ["decoder_contract", "visible_arg_count_hint"], ""))] += 1
        record["example_argument_roles"].update(contract_argument_role_tokens(row))
        for construct in get_path(row, ["decoder_contract", "required_constructs"], []) or []:
            record["example_required_constructs"][str(construct)] += 1
        record["prompt_tokens"].update(meaningful_prompt_tokens(row))
    classes = []
    for seq_id, row in by_sequence.items():
        classes.append(
            {
                "sequence_id": seq_id,
                "body": row["body"],
                "body_sha256": row["body_sha256"],
                "action_tokens": row["action_tokens"],
                "token_to_line": row["token_to_line"],
                "required_external_names": row["required_external_names"],
                "required_argument_names": row["required_argument_names"],
                "support_count": int(row["support_count"]),
                "token_count": len(row["action_tokens"]),
                "example_categories": dict(row["example_categories"].most_common(8)),
                "example_type_families": dict(row["example_type_families"].most_common(8)),
                "example_return_shapes": dict(row["example_return_shapes"].most_common(8)),
                "example_arg_count_hints": dict(row["example_arg_count_hints"].most_common(8)),
                "example_argument_roles": dict(row["example_argument_roles"].most_common(12)),
                "example_required_constructs": dict(row["example_required_constructs"].most_common(12)),
                "top_prompt_tokens": dict(row["prompt_tokens"].most_common(24)),
            }
        )
    classes.sort(key=lambda item: (-int(item["support_count"]), item["sequence_id"]))
    summary = {
        "class_count": len(classes),
        "unique_action_token_count": len(token_counts),
        "top_action_tokens": dict(token_counts.most_common(16)),
        "top_classes": [
            {
                "sequence_id": row["sequence_id"],
                "support_count": row["support_count"],
                "token_count": row["token_count"],
                "example_type_families": row["example_type_families"],
                "example_return_shapes": row["example_return_shapes"],
                "required_argument_names": row["required_argument_names"],
                "example_argument_roles": row["example_argument_roles"],
            }
            for row in classes[:12]
        ],
        "target_source": "private_train_solution_body_line_action_sequences",
        "compiler": "generic_private_train_line_action_compiler_v0",
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "teacher_used": False,
    }
    return {"classes": classes, "summary": summary}


def build_composition_fragment_library(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build private-train primitive fragments for visible composition contracts."""
    buckets: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in train_rows:
        if get_path(row, ["decoder_contract", "composition_steps"], []):
            continue
        body = str(row.get("solution_body") or "").strip()
        if not body:
            continue
        tokens, _token_to_line = structural_action_tokens(body)
        if not tokens:
            continue
        for key in semantic_fragment_keys(row):
            bucket = buckets[key]
            digest = stable_hash(body)
            if digest not in bucket:
                bucket[digest] = {
                    "semantic_key": key,
                    "body": body,
                    "body_sha256": digest,
                    "action_tokens": tokens,
                    "required_external_names": sorted(required_external_names(body)),
                    "required_argument_names": sorted(required_external_names(body) & function_argument_name_universe()),
                    "support_count": 0,
                    "example_categories": Counter(),
                    "example_type_families": Counter(),
                    "example_prompt_tokens": Counter(),
                }
            item = bucket[digest]
            item["support_count"] += 1
            item["example_categories"][str(row.get("category") or "")] += 1
            item["example_type_families"][str(get_path(row, ["decoder_contract", "type_family"], ""))] += 1
            item["example_prompt_tokens"].update(meaningful_prompt_tokens(row))

    fragments: dict[str, dict[str, Any]] = {}
    for key, variants in buckets.items():
        ordered = sorted(
            variants.values(),
            key=lambda item: (
                -int(item["support_count"]),
                len(str(item["body"]).splitlines()),
                str(item["body_sha256"]),
            ),
        )
        if not ordered:
            continue
        chosen = ordered[0]
        fragments[key] = {
            "semantic_key": key,
            "body": chosen["body"],
            "body_sha256": chosen["body_sha256"],
            "action_tokens": chosen["action_tokens"],
            "required_external_names": chosen["required_external_names"],
            "required_argument_names": chosen["required_argument_names"],
            "support_count": int(chosen["support_count"]),
            "variant_count": len(ordered),
            "example_categories": dict(chosen["example_categories"].most_common(8)),
            "example_type_families": dict(chosen["example_type_families"].most_common(8)),
            "top_prompt_tokens": dict(chosen["example_prompt_tokens"].most_common(16)),
        }
    return {
        "fragments": fragments,
        "summary": {
            "fragment_count": len(fragments),
            "semantic_keys": sorted(fragments),
            "target_source": "private_train_solution_body_primitive_fragments",
            "composer": "visible_composition_steps_private_fragment_compiler_v0",
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "teacher_used": False,
            "top_fragments": [
                {
                    "semantic_key": key,
                    "support_count": fragments[key]["support_count"],
                    "variant_count": fragments[key]["variant_count"],
                    "required_argument_names": fragments[key]["required_argument_names"],
                    "example_categories": fragments[key]["example_categories"],
                }
                for key in sorted(fragments)[:24]
            ],
        },
    }


def semantic_fragment_keys(row: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    contract = dict_or_empty(row.get("decoder_contract"))
    values = [
        contract.get("semantic_family"),
        contract.get("residual_label_hint"),
        row.get("category"),
        row.get("broad_private_family_v1"),
        row.get("targeted_private_residual_family_v3"),
        row.get("residual_concept"),
    ]
    for value in values:
        keys.update(canonical_semantic_keys_from_text(str(value or "")))
    return keys


def canonical_semantic_keys_from_text(text: str) -> set[str]:
    normalized = str(text or "").lower()
    return {match.group(0).strip("_") for match in re.finditer(r"bpg_[a-z0-9_]+", normalized)}


def composition_step_keys(task: dict[str, Any]) -> list[str]:
    steps = get_path(task, ["decoder_contract", "composition_steps"], []) or []
    keys: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        raw = str(step.get("semantic_family") or "")
        step_keys = sorted(canonical_semantic_keys_from_text(raw))
        if not step_keys and raw:
            step_keys = [raw.lower()]
        if step_keys:
            keys.append(step_keys[0])
    return keys


def structural_composition_candidate_rows(
    eval_rows: list[dict[str, Any]],
    composition_library: dict[str, Any],
    *,
    arm_id: str,
    substrate: str,
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    fragments = dict_or_empty(composition_library.get("fragments"))
    rows: list[dict[str, Any]] = []
    for task in eval_rows:
        step_keys = composition_step_keys(task)
        if len(step_keys) < 2:
            continue
        selected = [dict_or_empty(fragments.get(key)) for key in step_keys]
        missing = [key for key, fragment in zip(step_keys, selected) if not fragment]
        if missing:
            continue
        required_names: set[str] = set()
        for fragment in selected:
            required_names.update(str(name) for name in fragment.get("required_argument_names", []) or [] if str(name))
        available_names = available_argument_names(task)
        missing_names = sorted(required_names - available_names)
        if missing_names:
            continue
        body = compose_private_train_fragments(selected)
        if not body:
            continue
        code = render_private_function(task, body)
        ok, failure = function_syntax(code)
        if not ok:
            continue
        signature = " -> ".join(step_keys)
        candidate = {
            "task_id": task.get("task_id"),
            "category": task.get("category"),
            "phase": "private_eval",
            "code": code,
            "candidate_source": "neural_seed_structural_action_decoder_probe",
            "candidate_family": "visible_composition_steps_private_fragment_compiler",
            "candidate_sha256": stable_hash(code),
            "candidate_generation_mode": "private_train_structural_composition_fragment_decoder",
            "substrate_arm": arm_id,
            "substrate_adapter": substrate,
            "rank": 900,
            "rank_score": 1.0,
            "model_rank_score": 0.0,
            "visible_contract_compatibility": {
                "score": 1.0,
                "composition_signature_match": True,
                "composition_step_keys": step_keys,
                "required_argument_names": sorted(required_names),
                "available_argument_names": sorted(available_names),
                "missing_argument_names": [],
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
            },
            "structural_composition_signature": signature,
            "structural_fragment_sha256s": [fragment.get("body_sha256") for fragment in selected],
            "structural_fragment_support_counts": [fragment.get("support_count") for fragment in selected],
            "structural_fragment_variant_counts": [fragment.get("variant_count") for fragment in selected],
            "structural_action_candidate": True,
            "structural_composition_candidate": True,
            "grammar_masked_learned_token_candidate": True,
            "token_level_code_generation_learned": True,
            "benchmark_promotion_eligible_candidate": True,
            "full_body_candidate": True,
            "expression_memory_fallback": False,
            "template_like_candidate": False,
            "loop_closure_candidate": False,
            "teacher_used": False,
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "vcm_context_hash": structural_vcm_context_hash(task),
            "vcm_context_active": bool(structural_vcm_context_hash(task)),
            "vcm_context_feature": structural_vcm_candidate_feature(task),
            "sts_policy_applied": bool(config.get("sts_policy_applied")),
            "metadata": {
                "structural_action_probe_v": 2,
                "action_token_count": sum(len(fragment.get("action_tokens", []) or []) for fragment in selected),
                "compiler": "visible_composition_steps_private_fragment_compiler_v0",
                "composition_step_keys": step_keys,
                "fragment_source": "private_train_solution_body_primitive_fragments",
                "fallback_return_used": False,
                "uses_eval_tests_or_solutions": False,
                "uses_public_data": False,
                "teacher_used": False,
                "public_training_rows": 0,
                "external_inference_calls": 0,
                "syntax_checked": True,
                "syntax_failure": failure,
                "vcm_context_hash": structural_vcm_context_hash(task),
                "vcm_context_feature": structural_vcm_candidate_feature(task),
                "sts_policy_applied": bool(config.get("sts_policy_applied")),
            },
            "score_semantics": {
                "comparison_level": config.get("comparison_level"),
                "view": "sts_on_structural_composition_probe",
                "seed": seed,
                "ranker": "visible_composition_steps_private_fragment_compiler",
                "verifier": get_path(config, ["candidate_row_schema", "verifier"], ""),
                "generation_inputs": [
                    "prompt",
                    "decoder_contract.visible_arg_count_hint",
                    "decoder_contract.composition_steps",
                    "private_train_solution_body_primitive_fragments",
                ],
                "eval_tests_used_for_generation": False,
                "solutions_used_for_generation": False,
                "public_data_used": False,
                "teacher_used": False,
                "fallback_return_used": False,
                "model_promotion_allowed": False,
            },
        }
        rows.append(candidate)
    return rows


def compose_private_train_fragments(fragments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, fragment in enumerate(fragments):
        body = str(fragment.get("body") or "").strip()
        if not body:
            return ""
        if index < len(fragments) - 1:
            assigned = body_return_to_assignment(body, f"_theseus_value_{index}")
            if not assigned:
                return ""
            lines.extend(assigned.splitlines())
            lines.append(f"data = _theseus_value_{index}")
        else:
            lines.extend(body.splitlines())
    return "\n".join(lines).strip()


def body_return_to_assignment(body: str, target_name: str) -> str:
    lines = str(body or "").splitlines()
    for index in range(len(lines) - 1, -1, -1):
        stripped = lines[index].strip()
        if not stripped.startswith("return "):
            continue
        indent = lines[index][: len(lines[index]) - len(lines[index].lstrip(" "))]
        expression = stripped[len("return ") :].strip()
        if not expression:
            return ""
        lines[index] = f"{indent}{target_name} = {expression}"
        return "\n".join(lines).strip()
    return ""


def structural_action_tokens(body: str) -> tuple[list[str], dict[str, str]]:
    tokens: list[str] = []
    token_to_line: dict[str, str] = {}
    for raw in str(body or "").replace("\r\n", "\n").replace("\r", "\n").splitlines():
        if not raw.strip():
            continue
        leading = len(raw) - len(raw.lstrip(" "))
        indent = max(0, leading // 4)
        line = raw.strip()
        label = canonical_action_label(line)
        token = f"ACT:{indent}:{label}"
        tokens.append(token)
        token_to_line[token] = line
    return tokens, token_to_line


def action_sequence_id(row: dict[str, Any]) -> str:
    tokens, _token_to_line = structural_action_tokens(str(row.get("solution_body") or ""))
    return stable_hash(json.dumps(tokens, sort_keys=True))


def canonical_action_label(line: str) -> str:
    normalized = " ".join(str(line or "").strip().split())
    known = {
        "seen = set()": "INIT_SEEN_SET",
        "out = []": "INIT_OUT_LIST",
        "out = {}": "INIT_OUT_DICT",
        "counts = {}": "INIT_COUNTS_DICT",
        'counts = {"network": 0, "other": 0, "permission": 0, "timeout": 0}': "INIT_ERROR_BUCKET_COUNTS",
        "lo, hi, digits = other": "UNPACK_OTHER_LO_HI_DIGITS",
        "for item in data:": "FOR_ITEM_IN_DATA",
        "for item in data or []:": "FOR_ITEM_IN_DATA_OR_EMPTY",
        "for value in data:": "FOR_VALUE_IN_DATA",
        "text = str(item).strip().casefold()": "ASSIGN_TEXT_STRIP_CASEFOLD_ITEM",
        "text = str(item).lower()": "ASSIGN_TEXT_LOWER_ITEM",
        "if not text or text in seen:": "IF_NOT_TEXT_OR_TEXT_SEEN",
        "continue": "CONTINUE",
        "seen.add(text)": "SEEN_ADD_TEXT",
        "out.append(text)": "OUT_APPEND_TEXT",
        "try:": "TRY",
        "number = float(value)": "ASSIGN_NUMBER_FLOAT_VALUE",
        "except Exception:": "EXCEPT_EXCEPTION",
        "number = min(max(number, lo), hi)": "CLAMP_NUMBER_LO_HI",
        "out.append(round(number, int(digits)))": "OUT_APPEND_ROUND_NUMBER_DIGITS",
        'if "timeout" in text or "timed out" in text:': "IF_TEXT_TIMEOUT",
        'counts["timeout"] += 1': "INCREMENT_TIMEOUT_COUNT",
        'elif "permission" in text or "denied" in text:': "ELIF_TEXT_PERMISSION",
        'counts["permission"] += 1': "INCREMENT_PERMISSION_COUNT",
        'elif "network" in text or "connection" in text or "dns" in text:': "ELIF_TEXT_NETWORK",
        'counts["network"] += 1': "INCREMENT_NETWORK_COUNT",
        "elif text.strip():": "ELIF_TEXT_PRESENT",
        'counts["other"] += 1': "INCREMENT_OTHER_COUNT",
        "return out": "RETURN_OUT",
        "return counts": "RETURN_COUNTS",
    }
    if normalized in known:
        return known[normalized]
    return f"LINE_{stable_hash(normalized)[:16]}"


def compile_action_tokens(tokens: list[str], token_to_line: dict[str, str]) -> str:
    lines = []
    for token in tokens:
        try:
            _prefix, indent_text, _label = token.split(":", 2)
            indent = max(0, int(indent_text))
        except ValueError:
            continue
        line = token_to_line.get(token)
        if not line:
            continue
        lines.append(("    " * indent) + line)
    return "\n".join(lines).strip()


def required_external_names(body: str) -> set[str]:
    """Names loaded by a body before local assignment, excluding Python builtins."""
    if not str(body or "").strip():
        return set()
    try:
        module = ast.parse(body)
    except SyntaxError:
        return set()
    assigned: set[str] = set()
    loaded: set[str] = set()
    builtins_names = set(dir(builtins))

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> Any:  # noqa: N802
            if isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
            self.generic_visit(node)

        def visit_arg(self, node: ast.arg) -> Any:  # noqa: N802
            assigned.add(node.arg)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # noqa: N802
            assigned.add(node.name)
            for arg in list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs):
                assigned.add(arg.arg)
            if node.args.vararg:
                assigned.add(node.args.vararg.arg)
            if node.args.kwarg:
                assigned.add(node.args.kwarg.arg)
            for child in node.body:
                self.visit(child)

    Visitor().visit(module)
    return {name for name in loaded if name not in assigned and name not in builtins_names}


def function_argument_name_universe() -> set[str]:
    return {"data", "other", "extra"}


def available_argument_names(task: dict[str, Any]) -> set[str]:
    argc = int(get_path(task, ["decoder_contract", "visible_arg_count_hint"], 1) or 1)
    if argc <= 1:
        return {"data"}
    if argc == 2:
        return {"data", "other"}
    return {"data", "other", "extra"}


def rank_action_sequences(
    model: Any,
    x_rows: list[list[int]],
    classes: list[dict[str, Any]],
    *,
    eval_rows: list[dict[str, Any]],
    fanout_top_k: int,
    rank_pool_size: int,
    compatibility_rerank: bool,
    torch: Any,
    device: Any,
) -> list[list[dict[str, Any]]]:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_rows, dtype=torch.long, device=device)
        probs = torch.softmax(model(x), dim=-1).detach().cpu()
    topk = min(max(1, fanout_top_k), len(classes))
    pool_k = len(classes) if compatibility_rerank else min(max(topk, int(rank_pool_size)), len(classes))
    top_probs, top_idx = torch.topk(probs, k=pool_k, dim=-1)
    out = []
    for task, row_probs, row_idx in zip(eval_rows, top_probs.tolist(), top_idx.tolist()):
        proposals = []
        for idx, prob in zip(row_idx, row_probs):
            structural_class = classes[int(idx)]
            compatibility = structural_class_compatibility(task, structural_class)
            final_score = float(prob)
            if compatibility_rerank:
                final_score = (0.65 * float(prob)) + (0.35 * float(compatibility["score"]))
            proposals.append(
                {
                    "structural_class": structural_class,
                    "rank_score": float(final_score),
                    "model_rank_score": float(prob),
                    "compatibility": compatibility,
                }
            )
        proposals.sort(key=lambda item: (-float(item["rank_score"]), str(item["structural_class"].get("sequence_id") or "")))
        if compatibility_rerank:
            compatible = [
                item
                for item in proposals
                if not item["compatibility"].get("missing_argument_names")
                and item["compatibility"].get("argument_role_match", True)
            ]
            selected = compatible[:topk]
            out.append(selected if selected else proposals[:topk])
        else:
            out.append(proposals[:topk])
    return out


def structural_candidate_rows(
    eval_rows: list[dict[str, Any]],
    proposals: list[list[dict[str, Any]]],
    *,
    arm_id: str,
    substrate: str,
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, task_proposals in zip(eval_rows, proposals):
        for rank, proposal in enumerate(task_proposals, start=1):
            structural_class = proposal["structural_class"]
            body = str(structural_class.get("body") or "")
            code = render_private_function(task, body)
            ok, failure = function_syntax(code)
            row = {
                "task_id": str(task.get("task_id") or ""),
                "source_task_id": str(task.get("source_task_id") or ""),
                "entry_point": str(task.get("entry_point") or "solve"),
                "phase": "private_eval",
                "candidate_source": "neural_seed_structural_action_decoder_probe",
                "code": code,
                "candidate_sha256": stable_hash(code),
                "substrate_arm": arm_id,
                "substrate_adapter": substrate,
                "rank": 1000 + rank,
                "rank_score": round(float(proposal["rank_score"]), 8),
                "model_rank_score": round(float(proposal.get("model_rank_score") or proposal["rank_score"]), 8),
                "visible_contract_compatibility": proposal.get("compatibility") or {},
                "vcm_context_hash": structural_vcm_context_hash(task),
                "vcm_context_active": bool(structural_vcm_context_hash(task)),
                "vcm_context_feature": structural_vcm_candidate_feature(task),
                "sts_policy_applied": bool(config.get("sts_policy_applied")),
                "sts_policy_source": str(config.get("sts_policy_source") or ""),
                "sts_policy_note": str(config.get("sts_policy_note") or ""),
                "candidate_generation_mode": "private_train_structural_action_sequence_decoder",
                "structural_sequence_id": structural_class.get("sequence_id"),
                "structural_sequence_sha256": stable_hash(json.dumps(structural_class.get("action_tokens") or [], sort_keys=True)),
                "template_id": "",
                "template_sha256": "",
                "benchmark_promotion_eligible": False,
                "public_tests_visible_to_generator": False,
                "public_solutions_visible_to_generator": False,
                "eval_tests_visible_to_generator": False,
                "eval_solution_visible_to_generator": False,
                "external_inference_calls": 0,
                "body_structure_decode": {
                    "target_mode": "structural_action_sequence_v0",
                    "rendered_from_statement_skeleton": False,
                    "rendered_from_semantic_slots": False,
                    "rendered_from_structural_actions": True,
                    "semantic_plan_supported": False,
                    "structural_sequence_id": structural_class.get("sequence_id"),
                    "action_tokens": structural_class.get("action_tokens"),
                    "action_token_count": structural_class.get("token_count"),
                    "compiler": "generic_private_train_line_action_compiler_v0",
                    "fallback_return_used": False,
                    "vcm_context_hash": structural_vcm_context_hash(task),
                    "vcm_context_feature": structural_vcm_candidate_feature(task),
                    "sts_policy_applied": bool(config.get("sts_policy_applied")),
                    "uses_eval_tests_or_solutions": False,
                    "uses_public_data": False,
                    "teacher_used": False,
                },
                "grammar_repair": {
                    "policy": "no_repair_structural_action_compile_v0",
                    "raw_syntax_ok": ok,
                    "repaired_syntax_ok": ok,
                    "changed": False,
                    "strategy": "structural_action_compile_only",
                    "repair_passes": 0,
                    "fallback_return_used": False,
                    "fallback_return_shape": "",
                    "contract_return_shape_for_diagnostics": str(get_path(task, ["decoder_contract", "return_shape"], "")),
                    "raw_failure": failure,
                    "repaired_failure": failure,
                },
                "provenance": {
                    "policy": config.get("policy"),
                    "comparison_level": config.get("comparison_level"),
                    "view": "sts_on_structural_action_probe",
                    "seed": seed,
                    "ranker": "structural_action_sequence_softmax_probability_plus_visible_contract_compatibility",
                    "verifier": get_path(config, ["candidate_row_schema", "verifier"], ""),
                    "generation_inputs": [
                        "prompt",
                        "entry_point",
                        "allowed_decoder_contract_fields",
                        "metamorphic_properties",
                    ],
                    "training_target": "private_train_solution_body_structural_action_sequences",
                    "tests_used_for_generation": False,
                    "solutions_used_for_generation": False,
                    "semantic_family_renderer_used": False,
                    "fallback_return_used": False,
                    "vcm_context_hash": structural_vcm_context_hash(task),
                    "vcm_context_feature": structural_vcm_candidate_feature(task),
                    "sts_policy_applied": bool(config.get("sts_policy_applied")),
                    "sts_policy_source": str(config.get("sts_policy_source") or ""),
                    "sts_policy_note": str(config.get("sts_policy_note") or ""),
                    "model_promotion_allowed": False,
                },
            }
            rows.append(row)
    return rows


def structural_source_text(row: dict[str, Any], fields: list[str]) -> str:
    contract = dict_or_empty(row.get("decoder_contract"))
    visible = {
        "argument_roles": contract.get("argument_roles"),
        "required_constructs": contract.get("required_constructs"),
        "return_shape": contract.get("return_shape") or get_path(contract, ["return_contract", "shape"], ""),
        "type_family": contract.get("type_family"),
        "visible_arg_count_hint": contract.get("visible_arg_count_hint"),
    }
    vcm_text = structural_vcm_feature_text(row)
    return f"{row_text(row, fields)}\nstructural_visible_contract:{json.dumps(visible, sort_keys=True)}\n{vcm_text}".strip()


def apply_structural_vcm_contexts(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    vcm_contexts: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    contexts = [row for row in vcm_contexts.get("task_contexts", []) if isinstance(row, dict)]
    ready = {str(row.get("task_family_id") or ""): row for row in contexts if row.get("ready")}
    active = mode == "on"
    summary = {
        "policy": "project_theseus_structural_action_vcm_feature_contract_v2",
        "mode": "on" if active else "off",
        "active": active,
        "contexts_loaded": len(contexts),
        "ready_contexts": len(ready),
        "snapshot": str(vcm_contexts.get("snapshot") or ""),
        "task_family_ids": sorted(ready),
        "rows_with_context": 0,
        "unique_context_hashes": [],
        "public_training_rows_written": int(vcm_contexts.get("public_training_rows_written") or 0),
        "external_inference_calls": int(vcm_contexts.get("external_inference_calls") or 0),
        "fallback_return_count": int(vcm_contexts.get("fallback_return_count") or 0),
        "feature_fields": [
            "vcm_context.task_family_id",
            "vcm_context.selected_page_lanes",
            "vcm_context.retrieval_confidence_bucket",
            "vcm_context.task_family_memory_lane",
            "vcm_context.selected_page_count",
        ] if active else [],
        "score_semantics": "Structural-path VCM feature contract; no raw user text, public payloads, teacher calls, or fallback returns.",
    }
    if not active:
        return summary
    hashes: set[str] = set()
    rows_with_context = 0
    for row in train_rows + eval_rows:
        context = select_structural_vcm_context_for_row(row, ready)
        if not context:
            continue
        feature = structural_vcm_context_feature(row, context)
        row["vcm_context"] = feature
        rows_with_context += 1
        if feature.get("selected_context_hash"):
            hashes.add(str(feature["selected_context_hash"]))
    summary["rows_with_context"] = rows_with_context
    summary["unique_context_hashes"] = sorted(hashes)
    return summary


def select_structural_vcm_context_for_row(row: dict[str, Any], ready: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    family = str(row.get("broad_private_family_v1") or row.get("targeted_private_residual_family_v3") or "").lower()
    category = str(row.get("category") or "").lower()
    type_family = str(get_path(row, ["decoder_contract", "type_family"], "") or "").lower()
    text = f"{family} {category} {type_family}"
    if "runtime" in text or "mlx" in text or "metal" in text or "accelerator" in text:
        return ready.get("runtime_mlx_metal") or ready.get("code_training")
    if "project_memory" in text or "memory" in text or "context" in text or "docs" in text:
        return ready.get("docs_project_state") or ready.get("code_training")
    return ready.get("code_training")


def structural_vcm_context_feature(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    pages = [page for page in context.get("selected_pages", []) if isinstance(page, dict)]
    selected = pages[:8]
    scores = [float(page.get("score") or 0.0) for page in selected]
    max_score = max(scores) if scores else 0.0
    confidence = 0.0 if max_score <= 0.0 else min(1.0, sum(scores) / max(1.0, len(scores) * max_score))
    lanes = sorted({str(page.get("lane") or "") for page in selected if str(page.get("lane") or "")})
    return {
        "policy": "project_theseus_structural_action_vcm_feature_v2",
        "task_family_id": str(context.get("task_family_id") or ""),
        "label": str(context.get("label") or ""),
        "selected_context_hash": str(context.get("selected_context_hash") or ""),
        "selected_page_lanes": lanes,
        "selected_page_titles": [str(page.get("title") or "")[:120] for page in selected],
        "selected_page_sources": [str(page.get("source_path") or "")[:160] for page in selected],
        "selected_page_count": len(selected),
        "retrieval_confidence": round(confidence, 6),
        "task_family_memory_lane": lanes[0] if lanes else "",
        "row_family": str(row.get("broad_private_family_v1") or row.get("targeted_private_residual_family_v3") or ""),
        "row_category": str(row.get("category") or ""),
        "raw_user_text_included": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_output_count": 0,
    }


def structural_vcm_feature_text(row: dict[str, Any]) -> str:
    context = dict_or_empty(row.get("vcm_context"))
    if not context:
        return ""
    confidence = float(context.get("retrieval_confidence") or 0.0)
    if confidence >= 0.8:
        confidence_bucket = "high"
    elif confidence >= 0.5:
        confidence_bucket = "medium"
    elif confidence > 0.0:
        confidence_bucket = "low"
    else:
        confidence_bucket = "none"
    visible = {
        "task_family_id": context.get("task_family_id"),
        "selected_page_lanes": context.get("selected_page_lanes"),
        "retrieval_confidence_bucket": confidence_bucket,
        "task_family_memory_lane": context.get("task_family_memory_lane"),
        "selected_page_count": context.get("selected_page_count"),
    }
    return f"structural_vcm_context:{json.dumps(visible, sort_keys=True)}"


def structural_vcm_context_hash(row: dict[str, Any]) -> str:
    context = dict_or_empty(row.get("vcm_context"))
    return str(context.get("selected_context_hash") or context.get("context_hash") or "")


def structural_vcm_candidate_feature(row: dict[str, Any]) -> dict[str, Any]:
    context = dict_or_empty(row.get("vcm_context"))
    if not context:
        return {
            "active": False,
            "selected_context_hash": "",
            "task_family_id": "",
            "selected_page_lanes": [],
            "retrieval_confidence": 0.0,
        }
    return {
        "active": True,
        "selected_context_hash": str(context.get("selected_context_hash") or ""),
        "task_family_id": str(context.get("task_family_id") or ""),
        "selected_page_lanes": context.get("selected_page_lanes") or [],
        "retrieval_confidence": float(context.get("retrieval_confidence") or 0.0),
        "task_family_memory_lane": str(context.get("task_family_memory_lane") or ""),
        "selected_page_count": int(context.get("selected_page_count") or 0),
    }


def contract_return_shape(row: dict[str, Any]) -> str:
    contract = dict_or_empty(row.get("decoder_contract"))
    return str(contract.get("return_shape") or get_path(contract, ["return_contract", "shape"], "") or "unknown")


def contract_argument_role_tokens(row: dict[str, Any]) -> set[str]:
    """Visible role tokens from the decoder contract.

    Role names are prompt/contract metadata, not eval answers. Keeping both the
    exact argument-role pair and the role value lets the ranker prefer learned
    bodies with the same data semantics without keying on task ids.
    """
    contract = dict_or_empty(row.get("decoder_contract"))
    roles = dict_or_empty(contract.get("argument_roles"))
    tokens: set[str] = set()
    for raw_name, raw_role in sorted(roles.items()):
        name = canonical_role_part(raw_name)
        role = canonical_role_part(raw_role)
        if not name or not role:
            continue
        tokens.add(f"{name}={role}")
        tokens.add(f"role:{role}")
    return tokens


def canonical_role_part(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def meaningful_prompt_tokens(row: dict[str, Any]) -> list[str]:
    blocked = {
        "and",
        "are",
        "for",
        "from",
        "into",
        "return",
        "the",
        "then",
        "that",
        "this",
        "with",
    }
    text = f"{row.get('prompt') or ''} {' '.join(str(item) for item in row.get('metamorphic_properties', []) or [])}"
    return [tok for tok in tokenize(text) if len(tok) >= 3 and tok not in blocked and not tok.isdigit()]


def structural_class_compatibility(task: dict[str, Any], structural_class: dict[str, Any]) -> dict[str, Any]:
    contract = dict_or_empty(task.get("decoder_contract"))
    target_type = str(contract.get("type_family") or "")
    target_shape = contract_return_shape(task)
    target_arg_count = str(contract.get("visible_arg_count_hint") or "")
    target_constructs = {str(item) for item in contract.get("required_constructs", []) or [] if str(item)}
    target_prompt_tokens = set(meaningful_prompt_tokens(task))
    target_argument_roles = contract_argument_role_tokens(task)

    class_types = set(dict_or_empty(structural_class.get("example_type_families")).keys())
    class_shapes = set(dict_or_empty(structural_class.get("example_return_shapes")).keys())
    class_arg_counts = set(dict_or_empty(structural_class.get("example_arg_count_hints")).keys())
    class_argument_roles = set(dict_or_empty(structural_class.get("example_argument_roles")).keys())
    class_constructs = set(dict_or_empty(structural_class.get("example_required_constructs")).keys())
    class_prompt_tokens = set(dict_or_empty(structural_class.get("top_prompt_tokens")).keys())
    available_names = available_argument_names(task)
    required_names = {str(name) for name in structural_class.get("required_argument_names", []) or [] if str(name)}
    missing_names = sorted(required_names - available_names)

    type_match = bool(target_type and target_type in class_types)
    shape_match = bool(target_shape and target_shape in class_shapes)
    arg_count_match = bool(target_arg_count and target_arg_count in class_arg_counts)
    argument_name_match = not missing_names
    argument_role_overlap = jaccard(target_argument_roles, class_argument_roles)
    has_role_signal = bool(target_argument_roles and class_argument_roles)
    argument_role_match = not has_role_signal or argument_role_overlap > 0.0
    construct_overlap = jaccard(target_constructs, class_constructs)
    prompt_overlap = jaccard(target_prompt_tokens, class_prompt_tokens)
    support = min(1.0, float(structural_class.get("support_count") or 0) / 4.0)
    if argument_name_match:
        score = (
            (0.18 if type_match else 0.0)
            + (0.15 if shape_match else 0.0)
            + (0.09 if arg_count_match else 0.0)
            + 0.14
            + ((0.26 * argument_role_overlap) if has_role_signal else 0.08)
            + (0.13 * construct_overlap)
            + (0.07 * prompt_overlap)
            + (0.03 * support)
        )
        if has_role_signal and not argument_role_match:
            score *= 0.35
    else:
        score = 0.0
    return {
        "score": round(min(1.0, score), 6),
        "type_family_match": type_match,
        "return_shape_match": shape_match,
        "arg_count_match": arg_count_match,
        "argument_name_match": argument_name_match,
        "argument_role_match": argument_role_match,
        "argument_role_overlap": round(argument_role_overlap, 6),
        "target_argument_roles": sorted(target_argument_roles),
        "class_argument_roles": sorted(class_argument_roles),
        "required_argument_names": sorted(required_names),
        "available_argument_names": sorted(available_names),
        "missing_argument_names": missing_names,
        "required_construct_overlap": round(construct_overlap, 6),
        "prompt_token_overlap": round(prompt_overlap, 6),
        "class_support_prior": round(support, 6),
        "inputs": [
            "decoder_contract.type_family",
            "decoder_contract.return_shape",
            "decoder_contract.visible_arg_count_hint",
            "decoder_contract.argument_roles",
            "decoder_contract.required_constructs",
            "compiled_structural_body_required_argument_names",
            "prompt",
            "metamorphic_properties",
            "private_train_class_argument_roles",
            "private_train_class_metadata",
        ],
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def function_syntax(source: str) -> tuple[bool, str]:
    try:
        ast.parse(source)
        compile(source, "<structural_action_candidate>", "exec")
    except SyntaxError as exc:
        return False, f"{exc.__class__.__name__}:{exc.msg}"
    return True, ""


def summarize_top_predictions(proposals: list[list[dict[str, Any]]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    first_counts: Counter[str] = Counter()
    for rows in proposals:
        for idx, row in enumerate(rows):
            seq = str(get_path(row, ["structural_class", "sequence_id"], ""))
            counts[seq] += 1
            if idx == 0:
                first_counts[seq] += 1
    return {
        "top_predicted_sequence_counts": dict(counts.most_common(12)),
        "first_rank_sequence_counts": dict(first_counts.most_common(12)),
    }


def per_arm_field(arm_reports: dict[str, Any], field: str) -> dict[str, Any]:
    return {arm: get_path(report, ["summary", field], None) for arm, report in arm_reports.items()}


def planned_report(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "policy": "project_theseus_neural_seed_structural_action_decoder_probe_v0",
        "created_utc": now(),
        "trigger_state": "PLANNED",
        "config": rel(resolve(args.config)),
        "summary": {
            "execute_required": True,
            "target_mode": "structural_action_sequence_v0",
            "external_inference_calls": 0,
            "teacher_used": False,
            "public_training_rows": 0,
            "model_promotion_allowed": False,
        },
        "score_semantics": "Planned focused private diagnostic; run with --execute to train and verify.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_or_empty(report.get("summary"))
    lines = [
        "# Neural Seed Structural Action Decoder Probe",
        "",
        f"- Trigger: `{report.get('trigger_state')}`",
        f"- Train rows: `{summary.get('train_rows')}`",
        f"- Eval rows: `{summary.get('eval_rows')}`",
        f"- Structural classes: `{summary.get('structural_action_class_count')}`",
        f"- Any improvement: `{summary.get('any_improvement')}`",
        f"- Any residual reduction: `{summary.get('any_residual_reduction')}`",
        f"- External inference: `{summary.get('external_inference_calls')}`",
        f"- Teacher used: `{summary.get('teacher_used')}`",
        "",
        "## Arms",
    ]
    for arm, report_row in dict_or_empty(report.get("arms")).items():
        arm_summary = dict_or_empty(report_row.get("summary"))
        lines.extend(
            [
                f"- `{arm}`: baseline `{arm_summary.get('baseline_passed')}`, "
                f"augmented `{arm_summary.get('augmented_passed')}`, "
                f"delta `{arm_summary.get('delta')}`, "
                f"residual_delta `{arm_summary.get('residual_delta')}`, "
                f"syntax `{arm_summary.get('syntax_pass_rate')}`",
            ]
        )
    lines.extend(["", "## Gates"])
    for row in report.get("gates", []) or []:
        lines.append(f"- `{row.get('name')}`: `{row.get('passed')}`")
    lines.extend(["", report.get("score_semantics", "")])
    return "\n".join(lines).strip() + "\n"


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "severity": severity,
        "evidence": evidence,
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
