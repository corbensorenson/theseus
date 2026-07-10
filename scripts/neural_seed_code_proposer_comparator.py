#!/usr/bin/env python3
"""Compare SymLiquid-style and transformer candidate-code proposers.

This is the first true code-proposer comparator lane for the neural seed work:
both arms emit real Python candidate rows and those rows go through the existing
private Code LM verifier. It is still intentionally smoke-scale and private-only.
The current adapter selects reusable private-train body templates; it does not
claim free-form token decoding parity yet.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import random
import re
import resource
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neural_seed_open_vocab import SOURCE_BYTE_BEGIN, encode_tokens, populate_open_vocab


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_verifier import evaluate_private_candidates  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "neural_seed_code_proposer_comparator.json"
DEFAULT_CANDIDATES = ROOT / "reports" / "neural_seed_code_proposer_candidates.jsonl"
_TORCH_TRANSFORMER_DEVICE_PROBE_CACHE: dict[str, dict[str, Any]] = {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--out", default="reports/neural_seed_code_proposer_comparator.json")
    parser.add_argument("--markdown-out", default="reports/neural_seed_code_proposer_comparator.md")
    parser.add_argument("--candidate-manifest-out", default=str(DEFAULT_CANDIDATES.relative_to(ROOT)))
    parser.add_argument("--execute", action="store_true", help="Train/evaluate the private code-proposer smoke.")
    args = parser.parse_args()

    started = time.perf_counter()
    config = read_json(resolve(args.config))
    if not args.execute:
        report = planned_report(config, args.config, args.candidate_manifest_out)
    else:
        report = run_comparator(config, args.config, args.candidate_manifest_out, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report["trigger_state"] == "RED" else 0


def run_comparator(
    config: dict[str, Any],
    config_path: str,
    candidate_manifest_out: str,
    started: float,
) -> dict[str, Any]:
    torch, nn = import_torch()
    data_cfg = dict_or_empty(config.get("data"))
    budget = dict_or_empty(config.get("matched_budget"))
    text_views = dict_or_empty(config.get("text_views"))
    seed = int((budget.get("seeds") or [23])[0])
    random.seed(seed)
    torch.manual_seed(seed)

    train_rows_all = load_private_rows(resolve(str(data_cfg.get("train_jsonl") or "")), data_cfg)
    eval_rows_all = load_private_rows(resolve(str(data_cfg.get("eval_jsonl") or "")), data_cfg)
    train_rows = deterministic_sample(train_rows_all, int(data_cfg.get("max_train_rows") or 4096), seed)
    eval_rows = deterministic_sample(eval_rows_all, int(data_cfg.get("max_eval_rows") or 96), seed + 1009)
    if not train_rows or not eval_rows:
        return red_report(config, config_path, "missing_private_rows", {"train": len(train_rows), "eval": len(eval_rows)})

    templates = build_body_templates(train_rows)
    if len(templates) < 2:
        return red_report(config, config_path, "not_enough_private_body_templates", {"templates": len(templates)})
    body_to_id = {row["body"]: idx for idx, row in enumerate(templates)}
    train_y = [body_to_id[str(row.get("solution_body") or "").strip()] for row in train_rows]
    views = ["sts_off", "sts_on"]
    vocab = build_vocab(
        [row_text(row, text_views.get(view, [])) for view in views for row in train_rows],
        max_vocab=int(budget.get("max_vocab") or 1024),
    )
    max_len = int(budget.get("max_sequence_tokens") or 96)
    device, backend_note = select_torch_device(torch)
    mlx = mlx_status()

    transformer_cfg = dict_or_empty(dict_or_empty(config.get("arms")).get("transformer_control"))
    transformer_dims = {
        "d_model": int(transformer_cfg.get("d_model") or 32),
        "nhead": int(transformer_cfg.get("nhead") or 2),
        "num_layers": int(transformer_cfg.get("num_layers") or 1),
        "dim_feedforward": int(transformer_cfg.get("dim_feedforward") or 64),
    }
    transformer_param_count = count_params(
        TinyTransformerClassifier(
            len(vocab),
            len(templates),
            max_len=max_len,
            **transformer_dims,
            torch=torch,
            nn=nn,
        )
    )
    sym_dims, sym_param_count = choose_symliquid_dims(
        config,
        vocab_size=len(vocab),
        class_count=len(templates),
        target_params=transformer_param_count,
        torch=torch,
        nn=nn,
    )
    param_delta = abs(sym_param_count - transformer_param_count) / max(1, transformer_param_count)
    all_candidates: list[dict[str, Any]] = []
    arm_reports: dict[str, Any] = {}
    run_records: list[dict[str, Any]] = []

    old_timeout = os.environ.get("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS")
    os.environ["THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS"] = str(
        max(1, int(budget.get("private_candidate_timeout_seconds") or 4))
    )
    try:
        for arm_id in ["symliquid_style", "transformer_control"]:
            arm_started = time.perf_counter()
            arm_candidates: list[dict[str, Any]] = []
            arm_view_reports: dict[str, Any] = {}
            for task in eval_rows:
                arm_candidates.append(baseline_candidate(task, arm_id=arm_id, config=config, seed=seed))
            for view in views:
                view_started = time.perf_counter()
                torch.manual_seed(seed)
                random.seed(seed)
                if arm_id == "symliquid_style":
                    model = SymLiquidStyleClassifier(
                        len(vocab),
                        len(templates),
                        hidden_dim=sym_dims["hidden_dim"],
                        reservoir_dim=sym_dims["reservoir_dim"],
                        hv_dim=sym_dims["hv_dim"],
                        torch=torch,
                        nn=nn,
                    )
                    parameter_count = sym_param_count
                    substrate = "torch_symliquid_style_recurrent_vsa_code_body_selector"
                    dims = sym_dims
                else:
                    model = TinyTransformerClassifier(
                        len(vocab),
                        len(templates),
                        max_len=max_len,
                        **transformer_dims,
                        torch=torch,
                        nn=nn,
                    )
                    parameter_count = transformer_param_count
                    substrate = "torch_transformer_encoder_code_body_selector"
                    dims = transformer_dims
                model.to(device)
                train_x = encode_many([row_text(row, text_views.get(view, [])) for row in train_rows], vocab, max_len)
                eval_x = encode_many([row_text(row, text_views.get(view, [])) for row in eval_rows], vocab, max_len)
                before_mem = maxrss_mb()
                train_summary = train_model(model, train_x, train_y, budget, torch=torch, device=device)
                proposals = rank_templates(
                    model,
                    eval_x,
                    templates,
                    fanout_top_k=int(budget.get("fanout_top_k") or 4),
                    torch=torch,
                    device=device,
                )
                phase = "private_eval" if view == "sts_on" else "private_eval_sts_off"
                view_candidates = candidate_rows_for_view(
                    eval_rows,
                    proposals,
                    arm_id=arm_id,
                    substrate=substrate,
                    phase=phase,
                    view=view,
                    config=config,
                    seed=seed,
                )
                syntax = syntax_summary(view_candidates)
                arm_candidates.extend(view_candidates)
                arm_view_reports[view] = {
                    "arm_id": arm_id,
                    "view": view,
                    "phase": phase,
                    "substrate": substrate,
                    "parameter_count": parameter_count,
                    "dims": dims,
                    "train": train_summary,
                    "candidate_rows": len(view_candidates),
                    "candidate_tasks": len(eval_rows),
                    "fanout_top_k": int(budget.get("fanout_top_k") or 4),
                    "candidate_syntax": syntax,
                    "ranker": "softmax_probability_descending",
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
            verifier_started = time.perf_counter()
            private_eval = evaluate_private_candidates(eval_rows, arm_candidates)
            verifier_wall_ms = int((time.perf_counter() - verifier_started) * 1000)
            arm_summary = arm_verifier_summary(arm_view_reports, private_eval, verifier_wall_ms)
            arm_reports[arm_id] = {
                "summary": arm_summary,
                "views": arm_view_reports,
                "private_verifier": private_eval,
                "candidate_schema": candidate_schema_summary(arm_candidates),
                "wall_time_ms": int((time.perf_counter() - arm_started) * 1000),
            }
            all_candidates.extend(arm_candidates)
            run_records.append(arm_summary)
    finally:
        if old_timeout is None:
            os.environ.pop("THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS", None)
        else:
            os.environ["THESEUS_PRIVATE_CANDIDATE_TIMEOUT_SECONDS"] = old_timeout

    write_jsonl(resolve(candidate_manifest_out), all_candidates)
    comparisons = compare_arms(arm_reports)
    gates = build_gates(
        config,
        train_rows,
        eval_rows,
        templates,
        arm_reports,
        all_candidates,
        param_delta,
    )
    hard_pass = all(row["passed"] for row in gates if row["severity"] == "hard")
    trigger_state = "GREEN" if hard_pass else "RED"
    if trigger_state == "GREEN" and any(not row["passed"] for row in gates):
        trigger_state = "YELLOW"
    dataset_fingerprint = stable_hash(
        json.dumps(
            {
                "train": [row_id(row) for row in train_rows],
                "eval": [row_id(row) for row in eval_rows],
                "views": text_views,
                "template_count": len(templates),
            },
            sort_keys=True,
        )
    )
    summary = {
        "comparison_level": config.get("comparison_level"),
        "code_proposer_smoke_ready": trigger_state in {"GREEN", "YELLOW"},
        "both_arms_emit_candidate_code_rows": both_arms_emit_code(arm_reports),
        "same_private_verifier_for_both_arms": True,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "body_template_count": len(templates),
        "dataset_fingerprint": dataset_fingerprint,
        "vocab_size": len(vocab),
        "max_sequence_tokens": max_len,
        "candidate_manifest": rel(resolve(candidate_manifest_out)),
        "candidate_rows": len(all_candidates),
        "parameter_match_delta": round(param_delta, 6),
        "parameter_match_within_tolerance": param_delta <= float(budget.get("parameter_match_tolerance") or 0.1),
        "trusted_parameter_match": param_delta <= float(budget.get("trusted_parameter_match_tolerance") or 0.05),
        "symliquid_parameter_count": sym_param_count,
        "transformer_parameter_count": transformer_param_count,
        "best_sts_on_arm_by_verifier_pass_rate": comparisons.get("winner_by_sts_on_verifier_pass_rate"),
        "symliquid_minus_transformer_sts_on_verifier_pass_rate": comparisons.get(
            "symliquid_minus_transformer_sts_on_verifier_pass_rate"
        ),
        "external_inference_calls": 0,
        "teacher_used": False,
        "public_training_rows": 0,
        "model_promotion_allowed": False,
    }
    return {
        "policy": "project_theseus_neural_seed_code_proposer_comparator_report_v0",
        "created_utc": now(),
        "config": config_path,
        "trigger_state": trigger_state,
        "execute": True,
        "summary": summary,
        "data_contract": data_contract(config, train_rows, eval_rows),
        "candidate_row_schema": config.get("candidate_row_schema", {}),
        "matched_budget": matched_budget_report(config, sym_dims, transformer_dims),
        "adapter_boundary": config.get("adapter_boundary", {}),
        "arms": arm_reports,
        "comparisons": comparisons,
        "gates": gates,
        "score_semantics": (
            "Private candidate-code proposer smoke only. The adapters train on private train solution bodies "
            "as reusable body-template targets, emit real Python candidate rows, and score only through the "
            "private verifier. No public calibration, public tests/solutions, teacher call, distillation, "
            "network fetch, long unattended training, or model promotion was performed."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def planned_report(config: dict[str, Any], config_path: str, candidate_manifest_out: str) -> dict[str, Any]:
    return {
        "policy": "project_theseus_neural_seed_code_proposer_comparator_report_v0",
        "created_utc": now(),
        "config": config_path,
        "trigger_state": "PLANNED",
        "execute": False,
        "summary": {
            "comparison_level": config.get("comparison_level"),
            "code_proposer_smoke_ready": False,
            "candidate_manifest": candidate_manifest_out,
        },
        "candidate_row_schema": config.get("candidate_row_schema", {}),
        "adapter_boundary": config.get("adapter_boundary", {}),
        "next_action": "Run with --execute to train and verify the private candidate-code comparator.",
        "external_inference_calls": 0,
    }


def red_report(config: dict[str, Any], config_path: str, reason: str, evidence: Any) -> dict[str, Any]:
    return {
        "policy": "project_theseus_neural_seed_code_proposer_comparator_report_v0",
        "created_utc": now(),
        "config": config_path,
        "trigger_state": "RED",
        "execute": True,
        "summary": {
            "comparison_level": config.get("comparison_level"),
            "code_proposer_smoke_ready": False,
            "reason": reason,
            "evidence": evidence,
        },
        "adapter_boundary": config.get("adapter_boundary", {}),
        "external_inference_calls": 0,
    }


def build_body_templates(train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_body: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        body = str(row.get("solution_body") or "").strip()
        if not body:
            continue
        digest = stable_hash(body)
        item = by_body.setdefault(
            body,
            {
                "template_id": f"private_body_template_{digest[:16]}",
                "template_sha256": digest,
                "body": body,
                "train_count": 0,
                "families": Counter(),
                "constructs": Counter(),
            },
        )
        item["train_count"] += 1
        item["families"][str(get_path(row, ["decoder_contract", "type_family"], "unknown"))] += 1
        for construct in get_path(row, ["decoder_contract", "required_constructs"], []) or []:
            item["constructs"][str(construct)] += 1
    templates = []
    for body, item in by_body.items():
        item = dict(item)
        item["families"] = dict(item["families"].most_common())
        item["constructs"] = dict(item["constructs"].most_common())
        item["body_preview"] = body[:320]
        templates.append(item)
    templates.sort(key=lambda row: (-int(row["train_count"]), str(row["template_sha256"])))
    for idx, row in enumerate(templates):
        row["class_id"] = idx
    return templates


def baseline_candidate(task: dict[str, Any], *, arm_id: str, config: dict[str, Any], seed: int) -> dict[str, Any]:
    code = render_private_function(task, "return None")
    return candidate_row(
        task,
        code=code,
        phase="private_baseline",
        arm_id=arm_id,
        substrate="shared_null_baseline",
        view="baseline",
        rank=1,
        rank_score=1.0,
        template=None,
        config=config,
        seed=seed,
    )


def candidate_rows_for_view(
    eval_rows: list[dict[str, Any]],
    proposals: list[list[dict[str, Any]]],
    *,
    arm_id: str,
    substrate: str,
    phase: str,
    view: str,
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, task_proposals in zip(eval_rows, proposals):
        for rank, proposal in enumerate(task_proposals, start=1):
            template = proposal["template"]
            code = render_private_function(task, str(template["body"]))
            rows.append(
                candidate_row(
                    task,
                    code=code,
                    phase=phase,
                    arm_id=arm_id,
                    substrate=substrate,
                    view=view,
                    rank=rank,
                    rank_score=float(proposal["rank_score"]),
                    template=template,
                    config=config,
                    seed=seed,
                )
            )
    return rows


def candidate_row(
    task: dict[str, Any],
    *,
    code: str,
    phase: str,
    arm_id: str,
    substrate: str,
    view: str,
    rank: int,
    rank_score: float,
    template: dict[str, Any] | None,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    task_id = str(task.get("task_id") or "")
    template_id = template.get("template_id") if isinstance(template, dict) else "shared_null_baseline"
    return {
        "task_id": task_id,
        "source_task_id": str(task.get("source_task_id") or ""),
        "entry_point": str(task.get("entry_point") or "solve"),
        "phase": phase,
        "candidate_source": "neural_seed_code_proposer_comparator",
        "code": code,
        "candidate_sha256": stable_hash(code),
        "substrate_arm": arm_id,
        "substrate_adapter": substrate,
        "rank": rank,
        "rank_score": round(rank_score, 8),
        "template_id": template_id,
        "template_sha256": template.get("template_sha256") if isinstance(template, dict) else "",
        "candidate_generation_mode": "private_train_body_template_selector",
        "benchmark_promotion_eligible": False,
        "public_tests_visible_to_generator": False,
        "public_solutions_visible_to_generator": False,
        "eval_tests_visible_to_generator": False,
        "eval_solution_visible_to_generator": False,
        "external_inference_calls": 0,
        "provenance": {
            "policy": config.get("policy"),
            "comparison_level": config.get("comparison_level"),
            "view": view,
            "seed": seed,
            "ranker": get_path(config, ["candidate_row_schema", "ranker"], ""),
            "verifier": get_path(config, ["candidate_row_schema", "verifier"], ""),
            "generation_inputs": [
                "prompt",
                "entry_point",
                "allowed_decoder_contract_fields",
                "metamorphic_properties",
            ],
            "training_target": get_path(config, ["data", "training_target"], ""),
            "tests_used_for_generation": False,
            "solutions_used_for_generation": False,
            "model_promotion_allowed": False,
        },
    }


def render_private_function(task: dict[str, Any], body: str) -> str:
    entry = str(task.get("entry_point") or "solve")
    argc = int(get_path(task, ["decoder_contract", "visible_arg_count_hint"], 1) or 1)
    if argc <= 1:
        signature = f"def {entry}(data):"
    elif argc == 2:
        signature = f"def {entry}(data, other):"
    else:
        signature = f"def {entry}(data, other=None, *extra):"
    body_lines = body.splitlines() or ["return None"]
    indented = "\n".join(("    " + line) if line else "" for line in body_lines)
    return f"{signature}\n{indented}\n"


def train_model(model: Any, x_rows: list[list[int]], y_rows: list[int], budget: dict[str, Any], *, torch: Any, device: Any) -> dict[str, Any]:
    started = time.perf_counter()
    batch_size = int(budget.get("batch_size") or 64)
    epochs = int(budget.get("epochs") or 6)
    lr = float(budget.get("learning_rate") or 0.003)
    weight_decay = float(budget.get("weight_decay") or 0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    x = torch.tensor(x_rows, dtype=torch.long, device=device)
    y = torch.tensor(y_rows, dtype=torch.long, device=device)
    losses = []
    for _epoch in range(epochs):
        order = torch.randperm(x.shape[0], device=device)
        total_loss = 0.0
        total = 0
        model.train()
        for start in range(0, x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            logits = model(x[idx])
            loss = criterion(logits, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(idx.shape[0])
            total += int(idx.shape[0])
        losses.append(round(total_loss / max(1, total), 6))
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "loss_curve": losses,
        "optimizer": "AdamW",
        "wall_time_ms": int((time.perf_counter() - started) * 1000),
    }


def rank_templates(
    model: Any,
    x_rows: list[list[int]],
    templates: list[dict[str, Any]],
    *,
    fanout_top_k: int,
    torch: Any,
    device: Any,
) -> list[list[dict[str, Any]]]:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_rows, dtype=torch.long, device=device)
        probs = torch.softmax(model(x), dim=-1).detach().cpu()
    topk = min(fanout_top_k, len(templates))
    top_probs, top_idx = torch.topk(probs, k=topk, dim=-1)
    out = []
    for row_probs, row_idx in zip(top_probs.tolist(), top_idx.tolist()):
        out.append(
            [
                {
                    "template": templates[int(idx)],
                    "rank_score": float(prob),
                }
                for idx, prob in zip(row_idx, row_probs)
            ]
        )
    return out


def arm_verifier_summary(
    view_reports: dict[str, Any],
    private_eval: dict[str, Any],
    verifier_wall_ms: int,
) -> dict[str, Any]:
    sts_on = view_reports.get("sts_on", {})
    sts_off = view_reports.get("sts_off", {})
    private_verification = dict_or_empty(private_eval.get("private_verification"))
    return {
        "sts_on_verifier_pass_rate": private_eval.get("trained_pass_rate"),
        "sts_off_verifier_pass_rate": private_eval.get("sts_off_pass_rate"),
        "sts_delta": private_eval.get("sts_repair_pass_rate_delta"),
        "accepted_candidate_rate": private_eval.get("trained_pass_rate"),
        "baseline_pass_rate": private_eval.get("baseline_pass_rate"),
        "syntax_pass_rate_sts_on": get_path(sts_on, ["candidate_syntax", "syntax_pass_rate"], 0.0),
        "syntax_pass_rate_sts_off": get_path(sts_off, ["candidate_syntax", "syntax_pass_rate"], 0.0),
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
        "memory": {
            "sts_on": sts_on.get("memory"),
            "sts_off": sts_off.get("memory"),
        },
        "verifier_wall_time_ms": verifier_wall_ms,
    }


def compare_arms(arm_reports: dict[str, Any]) -> dict[str, Any]:
    by_arm = {}
    for arm_id, report in arm_reports.items():
        summary = dict_or_empty(report.get("summary"))
        by_arm[arm_id] = {
            "sts_on_verifier_pass_rate": summary.get("sts_on_verifier_pass_rate"),
            "sts_off_verifier_pass_rate": summary.get("sts_off_verifier_pass_rate"),
            "sts_delta": summary.get("sts_delta"),
            "accepted_candidate_rate": summary.get("accepted_candidate_rate"),
            "syntax_pass_rate_sts_on": summary.get("syntax_pass_rate_sts_on"),
            "parameter_count": summary.get("parameter_count"),
            "sts_task_level_regressions": summary.get("sts_task_level_regressions"),
        }
    sym = by_arm.get("symliquid_style", {})
    trans = by_arm.get("transformer_control", {})
    sym_rate = float(sym.get("sts_on_verifier_pass_rate") or 0.0)
    trans_rate = float(trans.get("sts_on_verifier_pass_rate") or 0.0)
    return {
        "by_arm": by_arm,
        "symliquid_minus_transformer_sts_on_verifier_pass_rate": round(sym_rate - trans_rate, 6),
        "winner_by_sts_on_verifier_pass_rate": "symliquid_style" if sym_rate >= trans_rate else "transformer_control",
        "score_semantics": "Private candidate-code smoke only; not public calibration, distillation, or promotion evidence.",
    }


def build_gates(
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    templates: list[dict[str, Any]],
    arm_reports: dict[str, Any],
    candidates: list[dict[str, Any]],
    param_delta: float,
) -> list[dict[str, Any]]:
    budget = dict_or_empty(config.get("matched_budget"))
    safety = dict_or_empty(config.get("safety"))
    required = set(get_path(config, ["candidate_row_schema", "required_fields"], []) or [])
    schema_missing = schema_missing_fields(candidates, required)
    return [
        gate("private_train_rows_loaded", len(train_rows) > 0, f"train_rows={len(train_rows)}", "hard"),
        gate("private_eval_rows_loaded", len(eval_rows) > 0, f"eval_rows={len(eval_rows)}", "hard"),
        gate("private_body_templates_built", len(templates) > 1, f"templates={len(templates)}", "hard"),
        gate("public_training_forbidden", not safety.get("public_training_allowed", True), safety, "hard"),
        gate("teacher_calls_forbidden", not safety.get("teacher_calls_allowed", True), safety, "hard"),
        gate("teacher_distillation_forbidden", not safety.get("teacher_distillation_allowed", True), safety, "hard"),
        gate("model_promotion_forbidden", not safety.get("model_promotion_allowed", True), safety, "hard"),
        gate("external_inference_zero", all(int(row.get("external_inference_calls") or 0) == 0 for row in candidates), 0, "hard"),
        gate("both_arms_evaluated", set(arm_reports) == {"symliquid_style", "transformer_control"}, list(arm_reports), "hard"),
        gate("both_arms_emit_candidate_code_rows", both_arms_emit_code(arm_reports), candidate_counts_by_arm(candidates), "hard"),
        gate("candidate_rows_have_required_schema", not schema_missing, schema_missing, "hard"),
        gate("candidate_rows_do_not_embed_tests_or_solutions", candidate_rows_do_not_embed_forbidden(candidates), "no tests/solution fields on candidate rows", "hard"),
        gate(
            "transformer_control_real",
            get_path(arm_reports, ["transformer_control", "views", "sts_on", "substrate"], "") == "torch_transformer_encoder_code_body_selector",
            get_path(arm_reports, ["transformer_control", "views", "sts_on", "substrate"], ""),
            "hard",
        ),
        gate(
            "symliquid_style_adapter_real",
            get_path(arm_reports, ["symliquid_style", "views", "sts_on", "substrate"], "") == "torch_symliquid_style_recurrent_vsa_code_body_selector",
            get_path(arm_reports, ["symliquid_style", "views", "sts_on", "substrate"], ""),
            "hard",
        ),
        gate(
            "both_sts_controls_ran",
            all({"sts_on", "sts_off"}.issubset(set(dict_or_empty(row.get("views")))) for row in arm_reports.values()),
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
            param_delta <= float(budget.get("parameter_match_tolerance") or 0.1),
            {"parameter_match_delta": round(param_delta, 6), "tolerance": budget.get("parameter_match_tolerance")},
            "hard",
        ),
        gate(
            "matched_parameter_budget_within_trusted_tolerance",
            param_delta <= float(budget.get("trusted_parameter_match_tolerance") or 0.05),
            {"parameter_match_delta": round(param_delta, 6), "trusted_tolerance": budget.get("trusted_parameter_match_tolerance")},
            "soft",
        ),
        gate(
            "candidate_syntax_nonzero",
            any(float(get_path(row, ["summary", "syntax_pass_rate_sts_on"], 0.0) or 0.0) > 0.0 for row in arm_reports.values()),
            {arm: get_path(row, ["summary", "syntax_pass_rate_sts_on"], 0.0) for arm, row in arm_reports.items()},
            "hard",
        ),
    ]


def data_contract(config: dict[str, Any], train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    data_cfg = dict_or_empty(config.get("data"))
    forbidden = [str(item) for item in data_cfg.get("forbidden_row_flags", [])]
    return {
        "train_jsonl": data_cfg.get("train_jsonl"),
        "eval_jsonl": data_cfg.get("eval_jsonl"),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "forbidden_flags_checked": forbidden,
        "forbidden_flag_hits": 0,
        "private_train_solution_bodies_used_as_training_targets": True,
        "private_train_tests_used_for_training": False,
        "private_eval_tests_used_by_private_verifier": True,
        "private_eval_tests_loaded_into_features": False,
        "private_eval_solutions_loaded_into_features": False,
        "tests_or_solutions_loaded_into_candidate_generation_features": False,
        "public_prompts_tests_solutions_used": False,
        "public_training_rows": 0,
        "withheld_private_test_or_solution_fields_present": private_test_or_solution_field_count(train_rows + eval_rows),
    }


def matched_budget_report(config: dict[str, Any], sym_dims: dict[str, int], transformer_dims: dict[str, int]) -> dict[str, Any]:
    budget = dict_or_empty(config.get("matched_budget"))
    return {
        "budget_id": budget.get("budget_id"),
        "seeds": budget.get("seeds"),
        "max_sequence_tokens": budget.get("max_sequence_tokens"),
        "epochs": budget.get("epochs"),
        "batch_size": budget.get("batch_size"),
        "learning_rate": budget.get("learning_rate"),
        "fanout_top_k": budget.get("fanout_top_k"),
        "private_candidate_timeout_seconds": budget.get("private_candidate_timeout_seconds"),
        "symliquid_style_dims": sym_dims,
        "transformer_control_dims": transformer_dims,
    }


def candidate_schema_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    phases = Counter(str(row.get("phase") or "") for row in rows)
    return {
        "candidate_rows": len(rows),
        "phase_counts": dict(sorted(phases.items())),
        "all_rows_have_code": all(bool(row.get("code")) for row in rows),
        "candidate_sha256_unique": len({str(row.get("candidate_sha256") or "") for row in rows}),
    }


def syntax_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = 0
    failures = Counter()
    for row in rows:
        try:
            ast.parse(str(row.get("code") or ""))
            passed += 1
        except SyntaxError as exc:
            failures[f"{exc.__class__.__name__}:{exc.msg}"] += 1
    return {
        "candidate_rows": len(rows),
        "syntax_pass_count": passed,
        "syntax_pass_rate": ratio(passed, len(rows)),
        "failure_counts": dict(failures.most_common(8)),
    }


def load_private_rows(path: Path, data_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    forbidden = [str(item) for item in data_cfg.get("forbidden_row_flags", [])]
    for raw in read_jsonl(path):
        if not isinstance(raw, dict):
            continue
        if data_cfg.get("require_private_flags", True):
            bad_flags = [flag for flag in forbidden if bool(raw.get(flag))]
            if bad_flags:
                raise ValueError(f"forbidden public flags in {path}: {bad_flags}")
        rows.append(raw)
    if not rows:
        raise ValueError(f"no private rows loaded from {path}")
    return rows


def deterministic_sample(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return list(rows)
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda item: stable_hash(f"{seed}:{row_id(item[1])}:{item[0]}"))
    return [row for _idx, row in indexed[:limit]]


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("source_task_id") or row.get("entry_point") or stable_hash(json.dumps(row, sort_keys=True))[:16])


def row_text(row: dict[str, Any], fields: list[Any]) -> str:
    chunks = []
    for field in fields:
        value = get_dotted(row, str(field))
        if value is None:
            continue
        chunks.append(flatten_value(value))
    return "\n".join(chunk for chunk in chunks if chunk)


def flatten_value(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key}:{flatten_value(val)}" for key, val in sorted(value.items()))
    if isinstance(value, list):
        return " ".join(flatten_value(item) for item in value)
    return str(value)


def tokenize(text: str) -> list[str]:
    out = []
    token = []
    for ch in text.lower():
        if ch.isalnum() or ch == "_":
            token.append(ch)
        else:
            if token:
                out.append("".join(token))
                token = []
            if not ch.isspace():
                out.append(ch)
    if token:
        out.append("".join(token))
    return out


def build_vocab(texts: list[str], *, max_vocab: int, byte_fallback: bool = False) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    if byte_fallback:
        return populate_open_vocab(vocab, counts, max_vocab=max_vocab, stream="source")
    for token, _count in counts.most_common(max(0, max_vocab - len(vocab))):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_many(texts: list[str], vocab: dict[str, int], max_len: int) -> list[list[int]]:
    rows = []
    for text in texts:
        logical = tokenize(text)
        if SOURCE_BYTE_BEGIN in vocab:
            ids, _receipt = encode_tokens(logical, vocab, stream="source")
        else:
            ids = [vocab.get(tok, 1) for tok in logical]
        ids = ids[:max_len]
        rows.append(ids + [0] * max(0, max_len - len(ids)))
    return rows


def choose_symliquid_dims(
    config: dict[str, Any],
    *,
    vocab_size: int,
    class_count: int,
    target_params: int,
    torch: Any,
    nn: Any,
) -> tuple[dict[str, int], int]:
    arm_cfg = dict_or_empty(dict_or_empty(config.get("arms")).get("symliquid_style"))
    best_dims = {"hidden_dim": 32, "reservoir_dim": 32, "hv_dim": 32}
    best_count = 0
    best_delta = float("inf")
    for hidden in arm_cfg.get("hidden_dim_candidates", [32]):
        for reservoir in arm_cfg.get("reservoir_dim_candidates", [32]):
            for hv in arm_cfg.get("hv_dim_candidates", [32]):
                model = SymLiquidStyleClassifier(
                    vocab_size,
                    class_count,
                    hidden_dim=int(hidden),
                    reservoir_dim=int(reservoir),
                    hv_dim=int(hv),
                    torch=torch,
                    nn=nn,
                )
                count = count_params(model)
                delta = abs(count - target_params) / max(1, target_params)
                if delta < best_delta:
                    best_delta = delta
                    best_count = count
                    best_dims = {"hidden_dim": int(hidden), "reservoir_dim": int(reservoir), "hv_dim": int(hv)}
    return best_dims, best_count


class TinyTransformerClassifier:
    def __new__(cls, *args: Any, torch: Any, nn: Any, **kwargs: Any) -> Any:
        class _Model(nn.Module):
            def __init__(
                self,
                vocab_size: int,
                class_count: int,
                *,
                d_model: int,
                nhead: int,
                num_layers: int,
                dim_feedforward: int,
                max_len: int,
            ) -> None:
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
                self.position = nn.Parameter(torch.zeros(1, max_len, d_model))
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
                self.classifier = nn.Linear(d_model, class_count)

            def forward(self, x: Any) -> Any:
                mask = x.ne(0)
                h = self.embedding(x) + self.position[:, : x.shape[1], :]
                encoded = self.encoder(h, src_key_padding_mask=~mask)
                denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
                pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / denom
                return self.classifier(pooled)

        return _Model(*args, **kwargs)


class SymLiquidStyleClassifier:
    def __new__(cls, *args: Any, torch: Any, nn: Any, **kwargs: Any) -> Any:
        class _Model(nn.Module):
            def __init__(self, vocab_size: int, class_count: int, *, hidden_dim: int, reservoir_dim: int, hv_dim: int) -> None:
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
                self.liquid_in = nn.Linear(hidden_dim, hidden_dim)
                self.liquid_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.tau = nn.Linear(hidden_dim, hidden_dim)
                self.reservoir = nn.Linear(hidden_dim, reservoir_dim)
                self.vsa = nn.Linear(reservoir_dim, hv_dim, bias=False)
                self.classifier = nn.Linear(hv_dim, class_count)

            def forward(self, x: Any) -> Any:
                emb = self.embedding(x)
                mask = x.ne(0).float().unsqueeze(-1)
                h = emb.new_zeros((x.shape[0], self.liquid_h.out_features))
                memory = emb.new_zeros((x.shape[0], self.vsa.out_features))
                for t in range(x.shape[1]):
                    xt = emb[:, t, :]
                    m = mask[:, t, :]
                    candidate = torch.tanh(self.liquid_in(xt) + self.liquid_h(h))
                    alpha = torch.sigmoid(self.tau(xt))
                    h_new = (1.0 - alpha) * h + alpha * candidate
                    h = m * h_new + (1.0 - m) * h
                    r = torch.tanh(self.reservoir(h))
                    hv = torch.tanh(self.vsa(r))
                    memory = m * (0.97 * memory + hv) + (1.0 - m) * memory
                memory = memory / memory.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                return self.classifier(memory)

        return _Model(*args, **kwargs)


def select_torch_device(torch: Any) -> tuple[Any, dict[str, Any]]:
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    requested = os.environ.get("THESEUS_TORCH_DEVICE", "").strip().lower()
    if requested in {"cpu", "cuda", "mps"}:
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda"), {
                "torch_mps_available": mps_available,
                "torch_mps_used": False,
                "torch_device_reason": "env_forced_cuda",
                "torch_device_probe": {"ok": True, "forced": True},
            }
        if requested == "mps" and mps_available:
            probe = torch_transformer_device_probe(torch, torch.device("mps"))
            if probe.get("ok"):
                return torch.device("mps"), {
                    "torch_mps_available": True,
                    "torch_mps_used": True,
                    "torch_device_reason": "env_forced_mps_probe_passed",
                    "torch_device_probe": probe,
                }
            return torch.device("cpu"), {
                "torch_mps_available": True,
                "torch_mps_used": False,
                "torch_device_reason": "env_forced_mps_probe_failed_cpu_fallback",
                "torch_device_probe": probe,
            }
        if requested == "cpu":
            return torch.device("cpu"), {
                "torch_mps_available": mps_available,
                "torch_mps_used": False,
                "torch_device_reason": "env_forced_cpu",
                "torch_device_probe": {"ok": True, "forced": True},
            }
    if torch.cuda.is_available():
        return torch.device("cuda"), {
            "torch_mps_available": mps_available,
            "torch_mps_used": False,
            "torch_device_reason": "cuda_available",
            "torch_device_probe": {"ok": True, "cuda_available": True},
        }
    if mps_available:
        probe = torch_transformer_device_probe(torch, torch.device("mps"))
        if probe.get("ok"):
            return torch.device("mps"), {
                "torch_mps_available": True,
                "torch_mps_used": True,
                "torch_device_reason": "mps_transformer_mask_probe_passed",
                "torch_device_probe": probe,
            }
        return torch.device("cpu"), {
            "torch_mps_available": True,
            "torch_mps_used": False,
            "torch_device_reason": "mps_transformer_mask_probe_failed_cpu_fallback",
            "torch_device_probe": probe,
        }
    return torch.device("cpu"), {
        "torch_mps_available": False,
        "torch_mps_used": False,
        "torch_device_reason": "cpu_default",
        "torch_device_probe": {"ok": True, "mps_available": False},
    }


def torch_transformer_device_probe(torch: Any, device: Any) -> dict[str, Any]:
    cache_key = str(device)
    if cache_key in _TORCH_TRANSFORMER_DEVICE_PROBE_CACHE:
        cached = dict(_TORCH_TRANSFORMER_DEVICE_PROBE_CACHE[cache_key])
        cached["cache_status"] = "hit"
        return cached
    started = time.perf_counter()
    try:
        nn = torch.nn
        d_model = 16
        source_embedding = nn.Embedding(32, d_model, padding_idx=0).to(device)
        target_embedding = nn.Embedding(32, d_model, padding_idx=0).to(device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=32,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        ).to(device)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=32,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        ).to(device)
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False).to(device)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=1).to(device)
        output = nn.Linear(d_model, 32).to(device)
        src = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]], dtype=torch.long, device=device)
        tgt = torch.tensor([[1, 6, 7, 2], [1, 8, 9, 2]], dtype=torch.long, device=device)
        src_mask = src.ne(0)
        tgt_mask = torch.triu(torch.ones((tgt.shape[1], tgt.shape[1]), dtype=torch.bool, device=device), diagonal=1)
        memory = encoder(source_embedding(src), src_key_padding_mask=~src_mask)
        decoded = decoder(target_embedding(tgt), memory, tgt_mask=tgt_mask, memory_key_padding_mask=~src_mask)
        logits = output(decoded)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=0)
        loss.backward()
        if str(device) == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        payload = {
            "ok": True,
            "device": str(device),
            "operation": "transformer_encoder_decoder_bool_masks_forward_backward",
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        _TORCH_TRANSFORMER_DEVICE_PROBE_CACHE[cache_key] = dict(payload)
        return payload
    except BaseException as exc:
        payload = {
            "ok": False,
            "device": str(device),
            "operation": "transformer_encoder_decoder_bool_masks_forward_backward",
            "error_type": type(exc).__name__,
            "error": str(exc)[:600],
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        _TORCH_TRANSFORMER_DEVICE_PROBE_CACHE[cache_key] = dict(payload)
        return payload


def mlx_status() -> dict[str, Any]:
    diagnosed = mlx_status_from_diagnosis()
    if diagnosed:
        return diagnosed

    candidates: list[str] = []
    for raw in [
        os.environ.get("THESEUS_MLX_PYTHON"),
        str(ROOT / ".venv-mlx" / "bin" / "python"),
        sys.executable,
    ]:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.exists():
            continue
        value = str(path)
        if value not in candidates:
            candidates.append(value)
    if not candidates:
        return {"mlx_available": False, "mlx_used": False, "mlx_reason": "no candidate MLX Python found"}

    failures: list[str] = []
    probe_code = (
        "import json, sys\n"
        "import mlx.core as mx\n"
        "x = mx.array([1.0, 2.0])\n"
        "mx.eval(x)\n"
        "print(json.dumps({'ok': True, 'executable': sys.executable, "
        "'default_device': str(mx.default_device()), 'values': x.tolist()}))\n"
    )
    for mlx_python in candidates:
        if mlx_python == sys.executable and importlib.util.find_spec("mlx") is None:
            failures.append(f"{mlx_python}: active Python cannot find mlx")
            continue
        try:
            probe = subprocess.run(
                [mlx_python, "-c", probe_code],
                cwd=str(ROOT),
                text=True,
                capture_output=True,
                timeout=8,
                check=False,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{mlx_python}: mlx.core probe timed out in child process")
            continue
        except Exception as exc:
            failures.append(f"{mlx_python}: child probe failed before import: {type(exc).__name__}:{exc}")
            continue
        if probe.returncode != 0:
            failures.append(f"{mlx_python}: child probe exited {probe.returncode}: {(probe.stderr or '')[-300:]}")
            continue
        try:
            payload = json.loads((probe.stdout or "").splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            payload = {}
        if not payload.get("ok"):
            failures.append(f"{mlx_python}: unusable payload: {(probe.stdout or '')[-300:]}")
            continue
        return {
            "mlx_available": True,
            "mlx_used": False,
            "mlx_default_device": str(payload.get("default_device") or ""),
            "mlx_python": str(payload.get("executable") or mlx_python),
            "mlx_reason": "torch comparator path used for shared CPU-safe smoke; mlx import was isolated in a child process",
        }
    return {
        "mlx_available": False,
        "mlx_used": False,
        "mlx_reason": "no candidate MLX Python passed child tensor-eval probe: " + " | ".join(failures[:3]),
    }


def mlx_status_from_diagnosis() -> dict[str, Any] | None:
    path = ROOT / "reports" / "macos_mlx_environment_diagnosis.json"
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if report.get("trigger_state") != "GREEN":
        return None
    route = report.get("route_decision") if isinstance(report.get("route_decision"), dict) else {}
    recommended = str(route.get("recommended_python") or "")
    if not recommended:
        return None
    default_device = ""
    for row in report.get("python_probes", []) or []:
        if not isinstance(row, dict) or str(row.get("python") or "") != recommended:
            continue
        if not row.get("mlx_core_usable"):
            continue
        core_probe = row.get("core_probe") if isinstance(row.get("core_probe"), dict) else {}
        stdout_tail = str(core_probe.get("stdout_tail") or "").strip()
        try:
            payload = json.loads(stdout_tail.splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            payload = {}
        default_device = str(payload.get("default_device") or "")
        break
    return {
        "mlx_available": True,
        "mlx_used": False,
        "mlx_default_device": default_device,
        "mlx_python": recommended,
        "mlx_reason": "formal macos_mlx_environment_diagnosis report is GREEN; Torch comparator does not consume MLX directly",
    }


def import_torch() -> tuple[Any, Any]:
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency report path
        raise RuntimeError("PyTorch is required for the real transformer control arm") from exc
    return torch, nn


def count_params(model: Any) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def both_arms_emit_code(arm_reports: dict[str, Any]) -> bool:
    for arm_id in ["symliquid_style", "transformer_control"]:
        if int(get_path(arm_reports, [arm_id, "candidate_schema", "candidate_rows"], 0) or 0) <= 0:
            return False
        if not bool(get_path(arm_reports, [arm_id, "candidate_schema", "all_rows_have_code"], False)):
            return False
    return True


def candidate_counts_by_arm(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("substrate_arm") or "") for row in rows)
    return dict(sorted(counts.items()))


def schema_missing_fields(rows: list[dict[str, Any]], required: set[str]) -> dict[str, int]:
    missing: Counter[str] = Counter()
    for row in rows:
        for field in required:
            if field not in row:
                missing[field] += 1
    return dict(missing)


def candidate_rows_do_not_embed_forbidden(rows: list[dict[str, Any]]) -> bool:
    forbidden = {"tests", "solution_body", "canonical_solution", "reference_solution", "private_tests"}
    return all(not (forbidden & set(row)) for row in rows)


def private_test_or_solution_field_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("tests")
        or row.get("solution_body")
        or row.get("canonical_solution")
        or row.get("reference_solution")
    )


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def ratio(num: int, den: int) -> float:
    return round(float(num) / float(den), 6) if den else 0.0


def maxrss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return round(value / (1024 * 1024), 3)
    return round(value / 1024, 3)


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> list[Any]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def get_dotted(value: dict[str, Any], dotted: str) -> Any:
    cur: Any = value
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_or_empty(report.get("summary"))
    lines = [
        "# Neural Seed Code-Proposer Comparator",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- comparison_level: `{summary.get('comparison_level')}`",
        f"- code_proposer_smoke_ready: `{summary.get('code_proposer_smoke_ready')}`",
        f"- train_rows: `{summary.get('train_rows')}`",
        f"- eval_rows: `{summary.get('eval_rows')}`",
        f"- body_template_count: `{summary.get('body_template_count')}`",
        f"- candidate_rows: `{summary.get('candidate_rows')}`",
        f"- parameter_match_delta: `{summary.get('parameter_match_delta')}`",
        f"- best_sts_on_arm_by_verifier_pass_rate: `{summary.get('best_sts_on_arm_by_verifier_pass_rate')}`",
        f"- symliquid_minus_transformer_sts_on_verifier_pass_rate: `{summary.get('symliquid_minus_transformer_sts_on_verifier_pass_rate')}`",
        "",
        "## Score Semantics",
        "",
        str(report.get("score_semantics", "")),
        "",
        "## STS-On Code Results",
        "",
    ]
    arms = dict_or_empty(report.get("arms"))
    for arm_id, arm in arms.items():
        row = dict_or_empty(arm.get("summary"))
        backend = dict_or_empty(row.get("backend"))
        lines.append(
            f"- `{arm_id}`: verifier_pass_rate=`{row.get('sts_on_verifier_pass_rate')}`, "
            f"accepted_candidate_rate=`{row.get('accepted_candidate_rate')}`, "
            f"syntax_pass_rate=`{row.get('syntax_pass_rate_sts_on')}`, "
            f"sts_delta=`{row.get('sts_delta')}`, "
            f"regressions=`{row.get('sts_task_level_regressions')}`, "
            f"params=`{row.get('parameter_count')}`, "
            f"backend=`{backend.get('framework')}:{backend.get('device')}`"
        )
    comparisons = dict_or_empty(report.get("comparisons"))
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"- winner_by_sts_on_verifier_pass_rate: `{comparisons.get('winner_by_sts_on_verifier_pass_rate')}`",
            f"- symliquid_minus_transformer_sts_on_verifier_pass_rate: `{comparisons.get('symliquid_minus_transformer_sts_on_verifier_pass_rate')}`",
            "",
            "## Adapter Boundary",
            "",
            f"- current_status: `{get_path(report, ['adapter_boundary', 'current_status'], '')}`",
            f"- candidate_manifest: `{summary.get('candidate_manifest')}`",
            "",
            "## Gates",
            "",
        ]
    )
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
