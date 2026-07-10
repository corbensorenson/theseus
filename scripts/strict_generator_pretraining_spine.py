#!/usr/bin/env python3
"""Checkpointed full-model pretraining for the strict code generator.

This trains the actual prompt/signature -> function-body transformer generator
used by ``neural_seed_token_decoder_comparator.py``. It is intentionally not a
new candidate family, renderer, router, or benchmark lane: it materializes a
reusable strict-generator checkpoint so the comparator can evaluate learned
body-token generation without repeating corpus warmup inside every split.
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

from neural_seed_code_proposer_comparator import (  # noqa: E402
    build_vocab,
    count_params,
    deterministic_sample,
    dict_or_empty,
    encode_many,
    get_path,
    import_torch,
    load_private_rows,
    ratio,
    rel,
    select_torch_device,
    stable_hash,
)
from neural_seed_token_decoder_comparator import (  # noqa: E402
    TransformerTokenDecoder,
    apply_pretraining_initialization,
    build_pretraining_initializers,
    build_target_vocab,
    collect_full_state_python_examples,
    decoder_source_text,
    encode_target_rows,
    evaluate_token_model_loss,
    full_state_pretraining_config,
    full_state_source_vocab_extension_summary,
    full_state_target_vocab_extension_summary,
    model_parameter_snapshot,
    model_parameter_update_summary,
    target_tokens,
    train_token_model,
)


DEFAULT_CONFIG = ROOT / "configs" / "neural_seed_token_decoder_comparator.json"
DEFAULT_OUT = ROOT / "reports" / "strict_generator_pretraining_spine.json"
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints" / "strict_generator_pretraining_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--checkpoint-dir", default=rel(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--budget-id", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    config_path = resolve(args.config)
    config = read_json(config_path)
    report = run_spine(
        config,
        config_path=args.config,
        checkpoint_dir=resolve(args.checkpoint_dir),
        budget_id=str(args.budget_id or ""),
        execute=bool(args.execute),
        started=started,
    )
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW", "PLANNED"} else 2


def run_spine(
    config: dict[str, Any],
    *,
    config_path: str,
    checkpoint_dir: Path,
    budget_id: str,
    execute: bool,
    started: float,
) -> dict[str, Any]:
    cfg = dict_or_empty(config.get("strict_generator_pretraining"))
    budgets = selected_budgets(cfg, budget_id=budget_id)
    if not budgets:
        budgets = default_budgets()
    if not execute:
        return {
            "policy": "project_theseus_strict_generator_pretraining_spine_v1",
            "created_utc": now(),
            "config": config_path,
            "execute": False,
            "trigger_state": "PLANNED",
            "summary": {
                "budget_ids": [str(row.get("id") or row.get("budget_id") or "") for row in budgets],
                "checkpoint_dir": rel(checkpoint_dir),
                "public_training_rows": 0,
                "external_inference_calls": 0,
            },
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        }

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch, nn = import_torch()
    device, backend_note = select_torch_device(torch)
    rung_reports: list[dict[str, Any]] = []
    for budget in budgets:
        rung_reports.append(
            train_budget(
                config,
                budget,
                checkpoint_dir=checkpoint_dir,
                torch=torch,
                nn=nn,
                device=device,
                backend_note=backend_note,
            )
        )

    gates = build_gates(rung_reports)
    hard_pass = all(row["passed"] for row in gates if row["severity"] == "hard")
    trigger_state = "GREEN" if hard_pass else "RED"
    if trigger_state == "GREEN" and any(not row["passed"] for row in gates):
        trigger_state = "YELLOW"
    summary = {
        "budget_count": len(rung_reports),
        "completed_budget_count": sum(1 for row in rung_reports if row.get("active")),
        "max_optimizer_token_positions": max([int(row.get("optimizer_token_positions_consumed") or 0) for row in rung_reports] or [0]),
        "max_optimizer_windows": max([int(row.get("optimizer_windows_consumed") or 0) for row in rung_reports] or [0]),
        "max_parameter_update_fraction": max([float(row.get("parameter_update_fraction") or 0.0) for row in rung_reports] or [0.0]),
        "max_non_embedding_update_fraction": max([float(row.get("non_embedding_update_fraction") or 0.0) for row in rung_reports] or [0.0]),
        "heldout_lm_improved_count": sum(1 for row in rung_reports if row.get("heldout_lm_improved")),
        "device": str(device),
        "backend": backend_note,
        "max_training_tokens_per_second": max([float(row.get("training_tokens_per_second") or 0.0) for row in rung_reports] or [0.0]),
        "max_optimizer_steps_per_second": max([float(row.get("optimizer_steps_per_second") or 0.0) for row in rung_reports] or [0.0]),
        "checkpoint_dir": rel(checkpoint_dir),
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "open_or_pretrained_model_weights_used": False,
        "fallback_template_router_tool_credit_count": 0,
    }
    return {
        "policy": "project_theseus_strict_generator_pretraining_spine_v1",
        "created_utc": now(),
        "config": config_path,
        "execute": True,
        "trigger_state": trigger_state,
        "summary": summary,
        "budgets": rung_reports,
        "gates": gates,
        "score_semantics": (
            "Pretraining only for the strict learned body-token generator. It emits no candidates, "
            "uses no public benchmark payloads, calls no external inference, and credits no templates, "
            "routers, tools, structural adapters, or fallback returns as learned generation."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def train_budget(
    config: dict[str, Any],
    budget: dict[str, Any],
    *,
    checkpoint_dir: Path,
    torch: Any,
    nn: Any,
    device: Any,
    backend_note: dict[str, Any],
) -> dict[str, Any]:
    budget_id = str(budget.get("id") or budget.get("budget_id") or "strict_generator_smoke")
    seed = int(budget.get("seed") or get_path(config, ["matched_budget", "seeds"], [23])[0] or 23)
    working_config = config_with_budget_overrides(config, budget)
    target_mode = str(get_path(working_config, ["body_structure_decoder", "target_mode"], "body_tokens"))
    data_cfg = dict_or_empty(working_config.get("data"))
    text_views = dict_or_empty(working_config.get("text_views"))
    matched_budget = dict_or_empty(working_config.get("matched_budget"))
    train_rows_all = load_private_rows(resolve(str(data_cfg.get("train_jsonl") or "")), data_cfg)
    max_train_rows = int(data_cfg.get("max_train_rows") or 512)
    train_rows = deterministic_sample(train_rows_all, max_train_rows, seed)
    log(f"budget={budget_id} load_private_rows rows={len(train_rows)}")
    staged = stage_full_state_examples(
        working_config,
        budget,
        budget_id=budget_id,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )
    source_vocab_extension_texts = list(staged.get("source_vocab_extension_texts") or [])
    target_vocab_extension_bodies = list(staged.get("target_vocab_extension_bodies") or [])
    log(
        f"budget={budget_id} staged_examples active={staged.get('active')} "
        f"examples={len(staged.get('examples') or [])} cache={staged.get('cache_status')}"
    )
    log(
        f"budget={budget_id} build_source_vocab start train_rows={len(train_rows)} "
        f"source_extension_texts={len(source_vocab_extension_texts)}"
    )
    source_vocab = build_vocab(
        [
            decoder_source_text(row, text_views.get(view, []))
            for view in ["sts_off", "sts_on"]
            for row in train_rows
        ]
        + source_vocab_extension_texts,
        max_vocab=int(matched_budget.get("max_source_vocab") or 4096),
    )
    log(f"budget={budget_id} build_source_vocab done size={len(source_vocab)}")
    log(
        f"budget={budget_id} build_target_vocab start train_rows={len(train_rows)} "
        f"target_extension_bodies={len(target_vocab_extension_bodies)}"
    )
    target_vocab = build_target_vocab(
        [str(row.get("solution_body") or "") for row in train_rows] + target_vocab_extension_bodies,
        max_vocab=int(matched_budget.get("max_target_vocab") or 4096),
        target_mode=target_mode,
    )
    log(f"budget={budget_id} build_target_vocab done size={len(target_vocab)}")
    max_source = int(matched_budget.get("max_source_tokens") or 96)
    max_target = int(matched_budget.get("max_target_tokens") or 160)
    log(f"budget={budget_id} encode_staged_rows start")
    rows = encode_staged_full_state_rows(
        staged,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        max_source=max_source,
        max_target=max_target,
        target_mode=target_mode,
    )
    row_summary = dict_or_empty(rows.get("summary"))
    log(
        f"budget={budget_id} encode_staged_rows done active={rows.get('active')} "
        f"source_rows={len(rows.get('source_rows') or [])} eval_rows={len(rows.get('eval_source_rows') or [])}"
    )
    if not bool(rows.get("active")):
        return {
            "id": budget_id,
            "active": False,
            "reason": "no_active_full_state_rows",
            "row_summary": row_summary,
            "public_training_rows": 0,
            "external_inference_calls": 0,
        }
    dims = transformer_dims_with_budget(working_config, budget)
    log(f"budget={budget_id} init_model start dims={dims} source_vocab={len(source_vocab)} target_vocab={len(target_vocab)}")
    model = TransformerTokenDecoder(
        len(source_vocab),
        len(target_vocab),
        max_source_len=max_source,
        max_target_len=max_target,
        plan_router_scale=0.0,
        **dims,
        torch=torch,
        nn=nn,
    )
    model.to(device)
    log(f"budget={budget_id} init_model done parameters={count_params(model)}")
    pretraining_initializers = build_pretraining_initializers(working_config, torch=torch, device=device)
    embedding_init_summary = apply_pretraining_initialization(
        model,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        initializer=dict_or_empty(dict_or_empty(pretraining_initializers.get("by_arm")).get("transformer_control")),
        torch=torch,
    )
    checkpoint_path = checkpoint_dir / f"strict_generator_transformer_{safe_slug(budget_id)}.pt"
    checkpoint_state = load_checkpoint_if_compatible(
        checkpoint_path,
        model,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        dims=dims,
        max_source=max_source,
        max_target=max_target,
        torch=torch,
        device=device,
    )
    log(
        f"budget={budget_id} checkpoint_resume loaded={checkpoint_state.get('loaded')} "
        f"completed_epochs={checkpoint_state.get('completed_epochs') or 0}"
    )
    completed_epochs = int(checkpoint_state.get("completed_epochs") or 0)
    epochs = max(0, int(budget.get("epochs") or 1))
    train_budget_cfg = {
        "epochs": 1,
        "batch_size": int(budget.get("batch_size") or matched_budget.get("batch_size") or 64),
        "learning_rate": float(budget.get("learning_rate") or matched_budget.get("learning_rate") or 0.0008),
        "weight_decay": float(budget.get("weight_decay") or matched_budget.get("weight_decay") or 0.0001),
        "first_target_token_loss_weight": float(
            budget.get("first_target_token_loss_weight") or matched_budget.get("first_target_token_loss_weight") or 1.0
        ),
        "return_token_loss_weight": float(budget.get("return_token_loss_weight") or matched_budget.get("return_token_loss_weight") or 1.0),
        "grammar_validity_auxiliary_loss_weight": float(
            budget["grammar_validity_auxiliary_loss_weight"]
            if "grammar_validity_auxiliary_loss_weight" in budget
            else matched_budget.get("grammar_validity_auxiliary_loss_weight") or 0.0
        ),
        "grammar_validity_max_positions_per_batch": int(
            budget["grammar_validity_max_positions_per_batch"]
            if "grammar_validity_max_positions_per_batch" in budget
            else matched_budget.get("grammar_validity_max_positions_per_batch") or 0
        ),
        "body_token_validity_policy": str(
            budget.get("body_token_validity_policy") or matched_budget.get("body_token_validity_policy") or "strict_body_token_legality_v1"
        ),
    }
    source_rows = list(rows.get("source_rows") or [])
    target_rows = list(rows.get("target_rows") or [])
    eval_source_rows = list(rows.get("eval_source_rows") or [])
    eval_target_rows = list(rows.get("eval_target_rows") or [])
    pad_id = int(target_vocab.get("<pad>", 0))
    before = model_parameter_snapshot(model, torch=torch)
    log(f"budget={budget_id} heldout_before start eval_rows={len(eval_source_rows)}")
    heldout_before = evaluate_token_model_loss(
        model,
        eval_source_rows,
        eval_target_rows,
        batch_size=int(train_budget_cfg["batch_size"]),
        torch=torch,
        device=device,
        pad_id=pad_id,
    )
    log(f"budget={budget_id} heldout_before done loss={heldout_before.get('loss')}")
    loss_curve: list[float | None] = []
    optimizer_step_count = 0
    started = time.perf_counter()
    training_started = time.perf_counter()
    for epoch in range(completed_epochs, epochs):
        train_budget_cfg["progress_label"] = f"strict_generator_pretraining budget={budget_id}"
        log(f"budget={budget_id} epoch_start {epoch + 1}/{epochs}")
        train_summary = train_token_model(
            model,
            source_rows,
            target_rows,
            train_budget_cfg,
            torch=torch,
            device=device,
            pad_id=pad_id,
            target_vocab=target_vocab,
            plan_auxiliary_loss_weight=0.0,
        )
        optimizer_step_count += int(train_summary.get("optimizer_step_count") or 0)
        curve = list(train_summary.get("loss_curve") or [])
        loss_curve.extend(float(value) for value in curve if value is not None)
        save_checkpoint(
            checkpoint_path,
            model,
            budget_id=budget_id,
            completed_epochs=epoch + 1,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            dims=dims,
            max_source=max_source,
            max_target=max_target,
            target_mode=target_mode,
            row_summary=row_summary,
            torch=torch,
        )
        log(f"budget={budget_id} epoch_done {epoch + 1}/{epochs} checkpoint={rel(checkpoint_path)}")
    training_wall_time_ms = int((time.perf_counter() - training_started) * 1000)
    log(f"budget={budget_id} heldout_after start eval_rows={len(eval_source_rows)}")
    heldout_after = evaluate_token_model_loss(
        model,
        eval_source_rows,
        eval_target_rows,
        batch_size=int(train_budget_cfg["batch_size"]),
        torch=torch,
        device=device,
        pad_id=pad_id,
    )
    log(f"budget={budget_id} heldout_after done loss={heldout_after.get('loss')}")
    update_summary = model_parameter_update_summary(model, before, torch=torch)
    source_nonpad = sum(1 for row in source_rows for value in row if int(value) != 0)
    target_nonpad = sum(1 for row in target_rows for value in row if int(value) != pad_id)
    epochs_trained_this_run = max(0, epochs - completed_epochs)
    optimizer_token_positions = int(source_nonpad + target_nonpad) * int(epochs_trained_this_run)
    optimizer_windows = int(len(source_rows)) * int(epochs_trained_this_run)
    training_seconds = max(training_wall_time_ms / 1000.0, 1e-9)
    final_sha = stable_hash_file(checkpoint_path)
    return {
        "id": budget_id,
        "active": True,
        "device": str(device),
        "backend": backend_note,
        "checkpoint": rel(checkpoint_path),
        "checkpoint_sha256": final_sha,
        "checkpoint_resume": {
            "checkpoint_existed": bool(checkpoint_state.get("loaded")),
            "completed_epochs_before_run": completed_epochs,
            "target_epochs": epochs,
            "epochs_trained_this_run": epochs_trained_this_run,
            "optimizer_state_restored": False,
            "resume_policy": "model_state_epoch_resume_v1",
        },
        "source_vocab_size": len(source_vocab),
        "target_vocab_size": len(target_vocab),
        "source_vocab_sha256": stable_hash(json.dumps(source_vocab, sort_keys=True)),
        "target_vocab_sha256": stable_hash(json.dumps(target_vocab, sort_keys=True)),
        "source_vocab_extension": full_state_source_vocab_extension_summary(source_vocab_extension_texts),
        "target_vocab_extension": full_state_target_vocab_extension_summary(target_vocab_extension_bodies),
        "dims": dims,
        "parameter_count": count_params(model),
        "row_summary": row_summary,
        "embedding_initialization": clean_initializer_summary(embedding_init_summary),
        "trainable_parameter_count": update_summary["trainable_parameter_count"],
        "updated_parameter_count": update_summary["updated_parameter_count"],
        "parameter_update_fraction": update_summary["parameter_update_fraction"],
        "non_embedding_parameter_count": update_summary["non_embedding_parameter_count"],
        "updated_non_embedding_parameter_count": update_summary["updated_non_embedding_parameter_count"],
        "non_embedding_update_fraction": update_summary["non_embedding_update_fraction"],
        "parameter_tensor_update_fraction": update_summary["parameter_tensor_update_fraction"],
        "non_embedding_tensor_update_fraction": update_summary["non_embedding_tensor_update_fraction"],
        "optimizer_step_count": optimizer_step_count,
        "optimizer_token_positions_consumed": optimizer_token_positions,
        "optimizer_windows_consumed": optimizer_windows,
        "training_wall_time_ms": training_wall_time_ms,
        "training_tokens_per_second": round(optimizer_token_positions / training_seconds, 3),
        "optimizer_steps_per_second": round(optimizer_step_count / training_seconds, 6),
        "source_train_token_count": source_nonpad,
        "target_train_token_count": target_nonpad,
        "eval_source_token_count": sum(1 for row in eval_source_rows for value in row if int(value) != 0),
        "eval_target_token_count": sum(1 for row in eval_target_rows for value in row if int(value) != pad_id),
        "heldout_lm_loss_before": heldout_before.get("loss"),
        "heldout_lm_loss_after": heldout_after.get("loss"),
        "heldout_lm_perplexity_before": heldout_before.get("perplexity"),
        "heldout_lm_perplexity_after": heldout_after.get("perplexity"),
        "heldout_lm_loss_curve": [heldout_before.get("loss"), *loss_curve, heldout_after.get("loss")],
        "heldout_lm_improved": bool(
            heldout_before.get("loss") is not None
            and heldout_after.get("loss") is not None
            and float(heldout_after["loss"]) < float(heldout_before["loss"])
        ),
        "transferred_state_fraction_into_generator": 1.0,
        "strict_generator_state_fraction_pretrained": 1.0,
        "updates_entire_model_state": True,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "open_or_pretrained_model_weights_used": False,
        "fallback_template_router_tool_credit_count": 0,
        "wall_time_ms": int((time.perf_counter() - started) * 1000),
    }


def selected_budgets(cfg: dict[str, Any], *, budget_id: str) -> list[dict[str, Any]]:
    budgets = [dict_or_empty(row) for row in cfg.get("budgets", []) if isinstance(row, dict)]
    if not budgets:
        budgets = default_budgets()
    if budget_id:
        budgets = [row for row in budgets if str(row.get("id") or row.get("budget_id") or "") == budget_id]
    return budgets


def default_budgets() -> list[dict[str, Any]]:
    return [
        {
            "id": "strict_generator_smoke",
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
    ]


def stage_full_state_examples(
    config: dict[str, Any],
    budget: dict[str, Any],
    *,
    budget_id: str,
    checkpoint_dir: Path,
    seed: int,
) -> dict[str, Any]:
    cfg = full_state_pretraining_config(config)
    if not bool(cfg.get("enabled", False)):
        return {"active": False, "reason": "full_state_pretraining_disabled", "examples": [], "summary": {"enabled": False}}
    cache_dir = checkpoint_dir / "stage_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_examples = max(
        int(cfg.get("max_examples") or 256),
        int(cfg.get("source_vocab_max_examples") or cfg.get("target_vocab_max_examples") or 0),
        int(cfg.get("target_vocab_max_examples") or 0),
    )
    max_files = max(
        int(cfg.get("max_files") or 64),
        int(cfg.get("source_vocab_max_files") or cfg.get("target_vocab_max_files") or 0),
        int(cfg.get("target_vocab_max_files") or 0),
    )
    cache_key = stable_hash(
        json.dumps(
            {
                "budget_id": budget_id,
                "cfg": cfg,
                "seed": seed,
                "max_examples": max_examples,
                "max_files": max_files,
                "policy": "strict_generator_stage_cache_v3_ast_canonical_prompt_provenance",
            },
            sort_keys=True,
            default=str,
        )
    )[:20]
    cache_path = cache_dir / f"{safe_slug(budget_id)}_{cache_key}.json"
    if cache_path.exists():
        cached = read_json(cache_path)
        examples = list(cached.get("examples") or [])
        summary = dict_or_empty(cached.get("summary"))
        summary["cache_path"] = rel(cache_path)
        summary["cache_status"] = "hit"
        return build_staged_example_views(cfg, examples, summary, budget=budget, seed=seed, cache_status="hit")

    log(
        f"budget={budget_id} collect_full_state_examples start max_files={max_files} "
        f"max_examples={max_examples}"
    )
    examples, summary = collect_full_state_python_examples(
        cfg,
        max_files=max_files,
        max_examples=max_examples,
        min_target_tokens=int(cfg.get("min_target_tokens") or 8),
        max_function_body_chars=int(cfg.get("max_function_body_chars") or 5000),
        seed=seed,
    )
    cache_payload = {
        "policy": "project_theseus_strict_generator_stage_cache_v3_ast_canonical_prompt_provenance",
        "created_utc": now(),
        "budget_id": budget_id,
        "summary": summary,
        "examples": examples,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "public_training_rows": 0,
        "external_inference_calls": 0,
    }
    write_json(cache_path, cache_payload)
    summary = dict(summary)
    summary["cache_path"] = rel(cache_path)
    summary["cache_status"] = "miss_written"
    log(
        f"budget={budget_id} collect_full_state_examples done examples={len(examples)} "
        f"admitted_files={summary.get('admitted_python_files')} cache={rel(cache_path)}"
    )
    return build_staged_example_views(cfg, examples, summary, budget=budget, seed=seed, cache_status="miss_written")


def build_staged_example_views(
    cfg: dict[str, Any],
    examples: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    budget: dict[str, Any],
    seed: int,
    cache_status: str,
) -> dict[str, Any]:
    max_examples = min(len(examples), int(cfg.get("max_examples") or budget.get("max_examples") or 256))
    selection_cfg = dict_or_empty(budget.get("example_selection") or cfg.get("example_selection"))
    row_examples, selection_summary = select_staged_row_examples(
        examples,
        max_examples=max_examples,
        seed=seed,
        selection_cfg=selection_cfg,
    )
    source_vocab_max = min(
        len(examples),
        int(cfg.get("source_vocab_max_examples") or cfg.get("target_vocab_max_examples") or cfg.get("max_examples") or max_examples),
    )
    target_vocab_max = min(
        len(examples),
        int(cfg.get("target_vocab_max_examples") or cfg.get("max_examples") or max_examples),
    )
    source_vocab_examples = deterministic_sample(examples, source_vocab_max, seed + 17)
    target_vocab_examples = deterministic_sample(examples, target_vocab_max, seed + 23)
    staged_summary = dict(summary)
    staged_summary.update(
        {
            "enabled": True,
            "active": bool(row_examples) and int(summary.get("public_benchmark_payload_admitted_count") or 0) == 0,
            "cache_status": cache_status,
            "staged_example_count": len(examples),
            "row_example_count": len(row_examples),
            "source_vocab_example_count": len(source_vocab_examples),
            "target_vocab_example_count": len(target_vocab_examples),
            "example_selection": selection_summary,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
            "public_training_rows": 0,
            "external_inference_calls": 0,
        }
    )
    return {
        "active": bool(staged_summary["active"]),
        "examples": row_examples,
        "source_vocab_extension_texts": [str(row.get("source_text") or "") for row in source_vocab_examples],
        "target_vocab_extension_bodies": [str(row.get("body") or "") for row in target_vocab_examples],
        "summary": staged_summary,
        "cache_status": cache_status,
    }


def select_staged_row_examples(
    examples: list[dict[str, Any]],
    *,
    max_examples: int,
    seed: int,
    selection_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select train rows without adding hidden metadata or target templates.

    The quality-balanced mode uses only AST-local quality fields computed from
    the admitted source-corpus function body itself. It does not inspect eval
    rows, tests, verifier outputs, public benchmark payloads, task families, or
    any answer-identifying metadata.
    """

    policy = str(selection_cfg.get("policy") or "deterministic_uniform_v1")
    limit = min(len(examples), max(0, int(max_examples or 0)))
    if limit <= 0:
        return [], example_selection_summary([], [], policy=policy, selection_cfg=selection_cfg)
    if policy not in {"quality_balanced_ast_local_v1", "quality_balanced_visible_prompt_v1"}:
        selected = deterministic_sample(examples, limit, seed)
        return selected, example_selection_summary(examples, selected, policy=policy, selection_cfg=selection_cfg)

    min_quality_score = float(selection_cfg.get("min_quality_score") or 6.0)
    high_quality_fraction = max(0.0, min(1.0, float(selection_cfg.get("high_quality_fraction") or 0.55)))
    descriptive_prompt_fraction = max(
        0.0,
        min(1.0, float(selection_cfg.get("descriptive_prompt_fraction") or 0.0)),
    )
    min_prompt_characters = max(1, int(selection_cfg.get("min_prompt_characters") or 24))
    high_quota = min(limit, int(round(limit * high_quality_fraction)))
    descriptive_quota = min(limit, int(round(limit * descriptive_prompt_fraction)))
    indexed = list(enumerate(examples))
    descriptive = [
        item
        for item in indexed
        if str(item[1].get("prompt_source") or "") == "docstring"
        and int(item[1].get("prompt_character_count") or 0) >= min_prompt_characters
    ]
    descriptive.sort(
        key=lambda item: (
            -staged_example_quality_score(item[1]),
            stable_hash(f"{seed}:descriptive:{item[0]}:{item[1].get('path')}:{item[1].get('function')}"),
        )
    )
    high_quality = [
        item
        for item in indexed
        if staged_example_quality_score(item[1]) >= min_quality_score
    ]
    high_quality.sort(
        key=lambda item: (
            -staged_example_quality_score(item[1]),
            stable_hash(f"{seed}:quality:{item[0]}:{item[1].get('path')}:{item[1].get('function')}"),
        )
    )
    selected_indexed = descriptive[:descriptive_quota]
    selected_indices = {idx for idx, _row in selected_indexed}
    remaining_high_quota = max(0, high_quota - len(selected_indexed))
    if remaining_high_quota:
        high_quality_remaining = [item for item in high_quality if item[0] not in selected_indices]
        selected_indexed.extend(high_quality_remaining[:remaining_high_quota])
        selected_indices.update(idx for idx, _row in high_quality_remaining[:remaining_high_quota])
    remainder = [row for idx, row in indexed if idx not in selected_indices]
    selected = [row for _idx, row in selected_indexed]
    if len(selected) < limit:
        selected.extend(deterministic_sample(remainder, limit - len(selected), seed + 911))
    return selected[:limit], example_selection_summary(examples, selected[:limit], policy=policy, selection_cfg=selection_cfg)


def example_selection_summary(
    all_examples: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    policy: str,
    selection_cfg: dict[str, Any],
) -> dict[str, Any]:
    before = quality_distribution_summary(all_examples)
    after = quality_distribution_summary(selected)
    return {
        "enabled": policy != "deterministic_uniform_v1",
        "policy": policy,
        "configured_min_quality_score": float(selection_cfg.get("min_quality_score") or 0.0),
        "configured_high_quality_fraction": float(selection_cfg.get("high_quality_fraction") or 0.0),
        "configured_descriptive_prompt_fraction": float(selection_cfg.get("descriptive_prompt_fraction") or 0.0),
        "configured_min_prompt_characters": int(selection_cfg.get("min_prompt_characters") or 0),
        "available_example_count": len(all_examples),
        "selected_example_count": len(selected),
        "quality_before": before,
        "quality_after": after,
        "prompt_alignment_before": prompt_alignment_distribution_summary(all_examples),
        "prompt_alignment_after": prompt_alignment_distribution_summary(selected),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "uses_answer_metadata": False,
        "score_semantics": (
            "Row selection is based only on AST-local quality plus visible licensed-source "
            "docstring presence/length from admitted private/licensed corpus functions. It does not inspect eval rows, "
            "tests, verifier outputs, public benchmark payloads, task labels, solution metadata, "
            "or hidden answer identifiers."
        ),
    }


def prompt_alignment_distribution_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "descriptive_docstring_rows": 0,
            "descriptive_docstring_rate": 0.0,
            "identifier_fallback_rows": 0,
            "mean_prompt_character_count": 0.0,
        }
    descriptive = [
        row
        for row in rows
        if str(row.get("prompt_source") or "") == "docstring"
        and int(row.get("prompt_character_count") or 0) > 0
    ]
    fallback_count = sum(1 for row in rows if str(row.get("prompt_source") or "") == "identifier_fallback")
    return {
        "rows": len(rows),
        "descriptive_docstring_rows": len(descriptive),
        "descriptive_docstring_rate": round(len(descriptive) / len(rows), 6),
        "identifier_fallback_rows": fallback_count,
        "mean_prompt_character_count": round(
            sum(int(row.get("prompt_character_count") or 0) for row in rows) / len(rows),
            4,
        ),
    }


def quality_distribution_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "mean_quality_score": 0.0,
            "high_quality_rate": 0.0,
            "nontrivial_return_rate": 0.0,
            "loop_rate": 0.0,
            "control_flow_rate": 0.0,
            "parameter_load_rate": 0.0,
        }
    scores = [staged_example_quality_score(row) for row in rows]
    qualities = [dict_or_empty(row.get("quality")) for row in rows]
    return {
        "rows": len(rows),
        "mean_quality_score": round(sum(scores) / max(1, len(scores)), 4),
        "high_quality_rate": round(sum(1 for score in scores if score >= 6.0) / len(scores), 6),
        "nontrivial_return_rate": round(
            sum(1 for quality in qualities if int(quality.get("nontrivial_return_count") or 0) > 0) / len(rows),
            6,
        ),
        "loop_rate": round(sum(1 for quality in qualities if int(quality.get("loop_count") or 0) > 0) / len(rows), 6),
        "control_flow_rate": round(
            sum(1 for quality in qualities if int(quality.get("control_flow_count") or 0) > 0) / len(rows),
            6,
        ),
        "parameter_load_rate": round(
            sum(1 for quality in qualities if int(quality.get("parameter_load_count") or 0) > 0) / len(rows),
            6,
        ),
    }


def staged_example_quality_score(row: dict[str, Any]) -> float:
    quality = dict_or_empty(row.get("quality"))
    return float(
        (3.0 * min(3, int(quality.get("nontrivial_return_count") or 0)))
        + (2.0 * min(4, int(quality.get("parameter_load_count") or 0)))
        + (2.0 * min(3, int(quality.get("loop_count") or 0)))
        + (1.5 * min(3, int(quality.get("control_flow_count") or 0)))
        + (1.0 * min(4, int(quality.get("assignment_count") or 0)))
        + (1.0 * min(4, int(quality.get("call_count") or 0)))
        + (1.5 * min(3, int(quality.get("comprehension_count") or 0)))
        + (0.5 * min(6, int(quality.get("local_store_name_count") or 0)))
        - (1.0 * min(3, int(quality.get("literal_only_return_count") or 0)))
        - (1.5 * min(4, int(quality.get("identity_copy_return_count") or 0)))
        - (3.0 if bool(quality.get("identity_copy_shortcut_body")) else 0.0)
        - (2.0 if bool(quality.get("inert_body")) else 0.0)
    )


def encode_staged_full_state_rows(
    staged: dict[str, Any],
    *,
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    max_source: int,
    max_target: int,
    target_mode: str,
) -> dict[str, Any]:
    summary = dict_or_empty(staged.get("summary"))
    if not bool(staged.get("active")):
        return {
            "enabled": True,
            "active": False,
            "reason": "no_active_staged_full_state_examples",
            "source_rows": [],
            "target_rows": [],
            "summary": summary,
        }
    examples = list(staged.get("examples") or [])
    eval_fraction = max(0.0, min(0.5, float(summary.get("eval_fraction") or 0.08)))
    default_eval_examples = max(32, int(len(examples) * eval_fraction)) if examples else 0
    max_eval_examples = max(0, int(summary.get("max_eval_examples") or default_eval_examples))
    eval_count = min(len(examples) // 3, max_eval_examples, default_eval_examples) if examples else 0
    eval_examples = examples[:eval_count]
    train_examples = examples[eval_count:] or examples
    if train_examples is examples:
        eval_examples = []
    source_rows = encode_many([str(row.get("source_text") or "") for row in train_examples], source_vocab, max_source)
    target_rows = encode_target_rows(
        [str(row.get("body") or "") for row in train_examples],
        target_vocab,
        max_target,
        target_mode=target_mode,
    )
    eval_source_rows = encode_many([str(row.get("source_text") or "") for row in eval_examples], source_vocab, max_source)
    eval_target_rows = encode_target_rows(
        [str(row.get("body") or "") for row in eval_examples],
        target_vocab,
        max_target,
        target_mode=target_mode,
    )
    token_stats = full_state_token_stats(source_rows, target_rows, source_vocab, target_vocab)
    encoded_summary = dict(summary)
    encoded_summary.update(
        {
            "train_example_count": len(train_examples),
            "eval_example_count": len(eval_examples),
            "encoded_source_rows": len(source_rows),
            "encoded_target_rows": len(target_rows),
            "encoded_eval_source_rows": len(eval_source_rows),
            "encoded_eval_target_rows": len(eval_target_rows),
            **token_stats,
        }
    )
    return {
        "enabled": True,
        "active": bool(source_rows and target_rows and int(summary.get("public_benchmark_payload_admitted_count") or 0) == 0),
        "source_rows": source_rows,
        "target_rows": target_rows,
        "eval_source_rows": eval_source_rows,
        "eval_target_rows": eval_target_rows,
        "examples": compact_example_refs(train_examples, limit=16, target_mode=target_mode),
        "eval_examples": compact_example_refs(eval_examples, limit=8, target_mode=target_mode),
        "summary": encoded_summary,
    }


def full_state_token_stats(
    source_rows: list[list[int]],
    target_rows: list[list[int]],
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
) -> dict[str, Any]:
    source_unk_id = int(source_vocab.get("<unk>", 1))
    source_pad_id = int(source_vocab.get("<pad>", 0))
    target_unk_id = int(target_vocab.get("<unk>", 1))
    target_pad_id = int(target_vocab.get("<pad>", 0))
    source_total = 0
    source_unknown = 0
    target_total = 0
    target_unknown = 0
    for row in source_rows:
        for token_id in row:
            if int(token_id) == source_pad_id:
                continue
            source_total += 1
            if int(token_id) == source_unk_id:
                source_unknown += 1
    for row in target_rows:
        for token_id in row:
            if int(token_id) == target_pad_id:
                continue
            target_total += 1
            if int(token_id) == target_unk_id:
                target_unknown += 1
    return {
        "source_unknown_token_count": source_unknown,
        "source_total_token_count": source_total,
        "source_unknown_token_rate": ratio(source_unknown, source_total),
        "target_unknown_token_count": target_unknown,
        "target_total_token_count": target_total,
        "target_unknown_token_rate": ratio(target_unknown, target_total),
    }


def compact_example_refs(examples: list[dict[str, Any]], *, limit: int, target_mode: str) -> list[dict[str, Any]]:
    return [
        {
            "path": row.get("path"),
            "function": row.get("function"),
            "source_sha256": stable_hash(str(row.get("source_text") or "")),
            "body_sha256": stable_hash(str(row.get("body") or "")),
            "body_token_count": len(target_tokens(str(row.get("body") or ""), target_mode=target_mode)),
            "quality": dict_or_empty(row.get("quality")),
        }
        for row in examples[:limit]
    ]


def config_with_budget_overrides(config: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(config))
    copied.setdefault("pretraining_initialization", {})
    full_state = dict_or_empty(get_path(copied, ["pretraining_initialization", "full_state_warmup"], {}))
    for key in [
        "max_files",
        "max_examples",
        "max_eval_examples",
        "target_vocab_max_files",
        "target_vocab_max_examples",
        "source_vocab_max_files",
        "source_vocab_max_examples",
        "min_target_tokens",
        "max_function_body_chars",
        "eval_fraction",
        "example_selection",
        "source_text_style",
    ]:
        if key in budget:
            full_state[key] = budget[key]
    full_state["enabled"] = True
    copied["pretraining_initialization"]["full_state_warmup"] = full_state
    copied.setdefault("matched_budget", {})
    matched_budget = dict_or_empty(copied.get("matched_budget"))
    for key in [
        "max_source_tokens",
        "max_target_tokens",
        "max_source_vocab",
        "max_target_vocab",
    ]:
        if key in budget:
            matched_budget[key] = int(budget[key])
    copied["matched_budget"] = matched_budget
    if "target_mode" in budget:
        copied.setdefault("body_structure_decoder", {})
        copied["body_structure_decoder"]["target_mode"] = str(budget["target_mode"])
    return copied


def transformer_dims_with_budget(config: dict[str, Any], budget: dict[str, Any]) -> dict[str, int]:
    """Resolve strict-generator transformer dimensions from arm config plus budget.

    Budget overrides are capacity knobs only. They do not alter data admission,
    prompt visibility, verifier scoring, public-training policy, or candidate
    family semantics.
    """

    transformer_cfg = dict_or_empty(dict_or_empty(config.get("arms")).get("transformer_control"))
    dims = {
        "d_model": int(transformer_cfg.get("d_model") or 224),
        "nhead": int(transformer_cfg.get("nhead") or 4),
        "num_layers": int(transformer_cfg.get("num_layers") or 2),
        "dim_feedforward": int(transformer_cfg.get("dim_feedforward") or 448),
    }
    override_sources = [
        budget,
        dict_or_empty(budget.get("transformer_control")),
        dict_or_empty(budget.get("transformer_dims")),
    ]
    for source in override_sources:
        for key in list(dims):
            if key in source:
                dims[key] = int(source[key])
    if dims["d_model"] <= 0 or dims["nhead"] <= 0 or dims["num_layers"] <= 0 or dims["dim_feedforward"] <= 0:
        raise ValueError(f"invalid_transformer_dims:{dims}")
    if dims["d_model"] % dims["nhead"] != 0:
        raise ValueError(f"d_model_not_divisible_by_nhead:{dims}")
    return dims


def load_checkpoint_if_compatible(
    path: Path,
    model: Any,
    *,
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    dims: dict[str, int],
    max_source: int,
    max_target: int,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    if not path.exists():
        return {"loaded": False, "reason": "missing_checkpoint"}
    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as exc:
        return {"loaded": False, "reason": f"load_failed:{exc.__class__.__name__}"}
    meta = checkpoint.get("meta") if isinstance(checkpoint, dict) else {}
    if not isinstance(meta, dict):
        return {"loaded": False, "reason": "missing_meta"}
    expected = {
        "source_vocab_sha256": stable_hash(json.dumps(source_vocab, sort_keys=True)),
        "target_vocab_sha256": stable_hash(json.dumps(target_vocab, sort_keys=True)),
        "dims": dims,
        "max_source": int(max_source),
        "max_target": int(max_target),
    }
    if any(meta.get(key) != value for key, value in expected.items()):
        return {"loaded": False, "reason": "checkpoint_shape_or_vocab_mismatch", "expected": expected, "actual": meta}
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return {"loaded": True, "completed_epochs": int(meta.get("completed_epochs") or 0), "checkpoint": rel(path)}


def save_checkpoint(
    path: Path,
    model: Any,
    *,
    budget_id: str,
    completed_epochs: int,
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    dims: dict[str, int],
    max_source: int,
    max_target: int,
    target_mode: str,
    row_summary: dict[str, Any],
    torch: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": "project_theseus_strict_generator_pretraining_checkpoint_v1",
            "created_utc": now(),
            "model_state_dict": model.state_dict(),
            "meta": {
                "budget_id": budget_id,
                "completed_epochs": int(completed_epochs),
                "source_vocab": source_vocab,
                "target_vocab": target_vocab,
                "source_vocab_sha256": stable_hash(json.dumps(source_vocab, sort_keys=True)),
                "target_vocab_sha256": stable_hash(json.dumps(target_vocab, sort_keys=True)),
                "dims": dims,
                "max_source": int(max_source),
                "max_target": int(max_target),
                "target_mode": target_mode,
                "row_summary": row_summary,
                "from_scratch": True,
                "public_training_rows": 0,
                "external_inference_calls": 0,
                "open_or_pretrained_model_weights_used": False,
            },
        },
        path,
    )


def clean_initializer_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dict_or_empty(summary).items() if key not in {"embedding", "tokenizer"}}


def build_gates(rungs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        gate("at_least_one_budget_completed", any(row.get("active") for row in rungs), {"budget_count": len(rungs)}, "hard"),
        gate("public_training_rows_zero", all(int(row.get("public_training_rows") or 0) == 0 for row in rungs), 0, "hard"),
        gate("external_inference_zero", all(int(row.get("external_inference_calls") or 0) == 0 for row in rungs), 0, "hard"),
        gate(
            "no_open_or_pretrained_weights",
            all(not bool(row.get("open_or_pretrained_model_weights_used")) for row in rungs),
            "from_scratch_only",
            "hard",
        ),
        gate(
            "full_generator_update_recorded",
            any(float(row.get("parameter_update_fraction") or 0.0) >= 0.98 for row in rungs),
            {row.get("id"): row.get("parameter_update_fraction") for row in rungs},
            "hard",
        ),
        gate(
            "non_embedding_update_recorded",
            any(float(row.get("non_embedding_update_fraction") or 0.0) >= 0.90 for row in rungs),
            {row.get("id"): row.get("non_embedding_update_fraction") for row in rungs},
            "hard",
        ),
        gate(
            "heldout_lm_improved",
            any(bool(row.get("heldout_lm_improved")) for row in rungs),
            {row.get("id"): row.get("heldout_lm_loss_curve") for row in rungs},
            "hard",
        ),
        gate(
            "no_template_router_tool_credit",
            all(int(row.get("fallback_template_router_tool_credit_count") or 0) == 0 for row in rungs),
            0,
            "hard",
        ),
    ]


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_hash_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "budget"


def log(message: str) -> None:
    print(f"[strict-generator-pretraining] {message}", file=sys.stderr, flush=True)


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
