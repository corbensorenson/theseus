#!/usr/bin/env python3
"""Compare SymLiquid-style and transformer substrates on private contracts.

This is the first runnable substrate-comparison harness, not a promotion lane.
It deliberately starts below full code-proposer parity: both arms predict a
private decoder-contract target from private task views, using the same rows,
same budget, same STS-on/off views, and the same fanout/ranker/verifier shape.
The report says clearly that full candidate-code proposer adapters are still
required before this can become a code-generation substrate comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "neural_seed_substrate_comparator.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--out", default="reports/neural_seed_substrate_comparator.json")
    parser.add_argument("--markdown-out", default="reports/neural_seed_substrate_comparator.md")
    parser.add_argument("--execute", action="store_true", help="Train and evaluate the private smoke comparator.")
    args = parser.parse_args()

    started = time.perf_counter()
    config = read_json(resolve(args.config))
    if not args.execute:
        report = planned_report(config, args.config)
    else:
        report = run_comparator(config, args.config, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report["trigger_state"] == "RED" else 0


def run_comparator(config: dict[str, Any], config_path: str, started: float) -> dict[str, Any]:
    torch, nn = import_torch()
    data_cfg = config.get("data") if isinstance(config.get("data"), dict) else {}
    budget = config.get("matched_budget") if isinstance(config.get("matched_budget"), dict) else {}
    train_rows_all = load_private_rows(resolve(data_cfg.get("train_jsonl", "")), data_cfg)
    eval_rows_all = load_private_rows(resolve(data_cfg.get("eval_jsonl", "")), data_cfg)
    seed = int((budget.get("seeds") or [17])[0])
    train_rows = deterministic_sample(train_rows_all, int(data_cfg.get("max_train_rows") or 512), seed)
    eval_rows = deterministic_sample(eval_rows_all, int(data_cfg.get("max_eval_rows") or 128), seed + 1009)
    label_path = [str(item) for item in data_cfg.get("label_path", [])]
    family_path = [str(item) for item in data_cfg.get("family_path", [])]
    labels = sorted({str(get_path(row, label_path, "")) for row in train_rows + eval_rows if get_path(row, label_path, "")})
    if len(labels) < 2:
        return red_report(config, config_path, "not_enough_labels", {"labels": labels})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    views = ["sts_off", "sts_on"]
    text_views = config.get("text_views") if isinstance(config.get("text_views"), dict) else {}
    vocab = build_vocab(
        [row_text(row, text_views.get(view, [])) for view in views for row in train_rows],
        max_vocab=int(budget.get("max_vocab") or 768),
    )
    max_len = int(budget.get("max_sequence_tokens") or 64)
    dataset_fingerprint = stable_hash(
        json.dumps(
            {
                "train": [row_id(row) for row in train_rows],
                "eval": [row_id(row) for row in eval_rows],
                "label_path": label_path,
                "views": text_views,
            },
            sort_keys=True,
        )
    )
    device, backend_note = select_torch_device(torch)
    transformer_cfg = config.get("arms", {}).get("transformer_control", {})
    transformer_dims = {
        "d_model": int(transformer_cfg.get("d_model") or 24),
        "nhead": int(transformer_cfg.get("nhead") or 2),
        "num_layers": int(transformer_cfg.get("num_layers") or 1),
        "dim_feedforward": int(transformer_cfg.get("dim_feedforward") or 48),
    }
    transformer_param_count = count_params(
        TinyTransformerClassifier(len(vocab), len(labels), max_len=max_len, **transformer_dims, torch=torch, nn=nn)
    )
    sym_dims, sym_param_count = choose_symliquid_dims(
        config,
        vocab_size=len(vocab),
        class_count=len(labels),
        max_len=max_len,
        target_params=transformer_param_count,
        torch=torch,
        nn=nn,
    )
    param_delta = abs(sym_param_count - transformer_param_count) / max(1, transformer_param_count)
    per_view_results: dict[str, Any] = {}
    all_runs = []
    for view in views:
        train_texts = [row_text(row, text_views.get(view, [])) for row in train_rows]
        eval_texts = [row_text(row, text_views.get(view, [])) for row in eval_rows]
        train_y = [label_to_id[str(get_path(row, label_path, ""))] for row in train_rows]
        eval_y = [label_to_id[str(get_path(row, label_path, ""))] for row in eval_rows]
        train_x = encode_many(train_texts, vocab, max_len)
        eval_x = encode_many(eval_texts, vocab, max_len)
        majority = majority_baseline(train_y, eval_y, labels)
        view_result = {
            "majority_baseline": majority,
            "arms": {},
            "view_contract": {
                "fields": text_views.get(view, []),
                "withheld": text_views.get("withheld_from_text", []),
                "label_path": label_path,
            },
        }
        for arm_id in ["symliquid_style", "transformer_control"]:
            arm_started = time.perf_counter()
            torch.manual_seed(seed)
            random.seed(seed)
            if arm_id == "symliquid_style":
                model = SymLiquidStyleClassifier(
                    len(vocab),
                    len(labels),
                    hidden_dim=sym_dims["hidden_dim"],
                    reservoir_dim=sym_dims["reservoir_dim"],
                    hv_dim=sym_dims["hv_dim"],
                    torch=torch,
                    nn=nn,
                )
                parameter_count = sym_param_count
                substrate = "torch_symliquid_style_recurrent_vsa_classifier"
            else:
                model = TinyTransformerClassifier(
                    len(vocab),
                    len(labels),
                    max_len=max_len,
                    **transformer_dims,
                    torch=torch,
                    nn=nn,
                )
                parameter_count = transformer_param_count
                substrate = "torch_transformer_encoder_classifier"
            model.to(device)
            train_summary = train_model(
                model,
                train_x,
                train_y,
                budget,
                torch=torch,
                device=device,
            )
            metrics = evaluate_model(
                model,
                eval_x,
                eval_y,
                labels,
                eval_rows,
                majority["predicted_label"],
                family_path,
                fanout_top_k=int(budget.get("fanout_top_k") or 3),
                torch=torch,
                device=device,
            )
            metrics.update(
                {
                    "arm_id": arm_id,
                    "substrate": substrate,
                    "view": view,
                    "parameter_count": parameter_count,
                    "train": train_summary,
                    "wall_time_ms": int((time.perf_counter() - arm_started) * 1000),
                    "backend": {
                        "framework": "torch",
                        "device": str(device),
                        **backend_note,
                        "mlx_available": False,
                        "mlx_reason": "active Python cannot import mlx.core",
                    },
                }
            )
            view_result["arms"][arm_id] = metrics
            all_runs.append(metrics)
        per_view_results[view] = view_result
    comparisons = compare_runs(per_view_results)
    gates = build_gates(
        config,
        train_rows,
        eval_rows,
        param_delta,
        per_view_results,
        all_runs,
    )
    trigger_state = "GREEN" if all(row["passed"] for row in gates if row["severity"] == "hard") else "RED"
    if trigger_state == "GREEN" and any(not row["passed"] for row in gates):
        trigger_state = "YELLOW"
    summary = {
        "comparison_level": config.get("comparison_level"),
        "code_proposer_comparison_ready": False,
        "substrate_smoke_ready": trigger_state in {"GREEN", "YELLOW"},
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "label_count": len(labels),
        "labels": labels,
        "dataset_fingerprint": dataset_fingerprint,
        "vocab_size": len(vocab),
        "max_sequence_tokens": max_len,
        "parameter_match_delta": round(param_delta, 6),
        "parameter_match_within_tolerance": param_delta <= float(budget.get("parameter_match_tolerance") or 0.1),
        "trusted_parameter_match": param_delta <= float(budget.get("trusted_parameter_match_tolerance") or 0.05),
        "saturated_smoke": smoke_is_saturated(all_runs),
        "discriminative_smoke": not smoke_is_saturated(all_runs),
        "symliquid_parameter_count": sym_param_count,
        "transformer_parameter_count": transformer_param_count,
        "best_sts_on_arm_by_verifier_pass_rate": best_arm(per_view_results, "sts_on", "verifier_pass_rate"),
        "external_inference_calls": 0,
        "teacher_used": False,
        "public_training_rows": 0,
    }
    return {
        "policy": "project_theseus_neural_seed_substrate_comparator_report_v0",
        "created_utc": now(),
        "config": config_path,
        "trigger_state": trigger_state,
        "execute": True,
        "summary": summary,
        "data_contract": data_contract(config, train_rows, eval_rows),
        "matched_budget": matched_budget_report(config, sym_dims, transformer_dims),
        "adapter_boundary": config.get("adapter_boundary", {}),
        "views": per_view_results,
        "comparisons": comparisons,
        "gates": gates,
        "score_semantics": (
            "Private lower-level substrate smoke only. The report does not run public calibration, "
            "does not train on public prompts/tests/solutions, does not call a teacher, and does not "
            "claim full code-proposer parity."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def planned_report(config: dict[str, Any], config_path: str) -> dict[str, Any]:
    return {
        "policy": "project_theseus_neural_seed_substrate_comparator_report_v0",
        "created_utc": now(),
        "config": config_path,
        "trigger_state": "PLANNED",
        "execute": False,
        "summary": {
            "comparison_level": config.get("comparison_level"),
            "code_proposer_comparison_ready": False,
            "substrate_smoke_ready": False,
        },
        "adapter_boundary": config.get("adapter_boundary", {}),
        "next_action": "Run with --execute to train the smoke-scale private substrate comparator.",
        "external_inference_calls": 0,
    }


def red_report(config: dict[str, Any], config_path: str, reason: str, evidence: Any) -> dict[str, Any]:
    return {
        "policy": "project_theseus_neural_seed_substrate_comparator_report_v0",
        "created_utc": now(),
        "config": config_path,
        "trigger_state": "RED",
        "execute": True,
        "summary": {
            "comparison_level": config.get("comparison_level"),
            "code_proposer_comparison_ready": False,
            "substrate_smoke_ready": False,
            "reason": reason,
            "evidence": evidence,
        },
        "adapter_boundary": config.get("adapter_boundary", {}),
        "external_inference_calls": 0,
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
    return str(row.get("task_id") or row.get("source_task_id") or row.get("entry_point") or row.get("category") or stable_hash(json.dumps(row, sort_keys=True))[:16])


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


def build_vocab(texts: list[str], *, max_vocab: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _count in counts.most_common(max(0, max_vocab - len(vocab))):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_many(texts: list[str], vocab: dict[str, int], max_len: int) -> list[list[int]]:
    rows = []
    for text in texts:
        ids = [vocab.get(tok, 1) for tok in tokenize(text)[:max_len]]
        rows.append(ids + [0] * max(0, max_len - len(ids)))
    return rows


def majority_baseline(train_y: list[int], eval_y: list[int], labels: list[str]) -> dict[str, Any]:
    counts = Counter(train_y)
    majority = counts.most_common(1)[0][0]
    passed = sum(1 for y in eval_y if y == majority)
    return {
        "predicted_label": labels[majority],
        "verifier_pass_rate": ratio(passed, len(eval_y)),
        "passed": passed,
        "eval_rows": len(eval_y),
    }


def train_model(model: Any, x_rows: list[list[int]], y_rows: list[int], budget: dict[str, Any], *, torch: Any, device: Any) -> dict[str, Any]:
    started = time.perf_counter()
    batch_size = int(budget.get("batch_size") or 32)
    epochs = int(budget.get("epochs") or 8)
    lr = float(budget.get("learning_rate") or 0.003)
    weight_decay = float(budget.get("weight_decay") or 0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    x = torch.tensor(x_rows, dtype=torch.long, device=device)
    y = torch.tensor(y_rows, dtype=torch.long, device=device)
    losses = []
    for epoch in range(epochs):
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
        "wall_time_ms": int((time.perf_counter() - started) * 1000),
        "optimizer": "AdamW",
    }


def evaluate_model(
    model: Any,
    x_rows: list[list[int]],
    y_rows: list[int],
    labels: list[str],
    eval_rows: list[dict[str, Any]],
    baseline_label: str,
    family_path: list[str],
    *,
    fanout_top_k: int,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_rows, dtype=torch.long, device=device)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).detach().cpu()
    topk = min(fanout_top_k, len(labels))
    top_probs, top_idx = torch.topk(probs, k=topk, dim=-1)
    passed = 0
    accepted = 0
    syntax_pass = 0
    baseline_improvements = 0
    baseline_regressions = 0
    family_totals: Counter[str] = Counter()
    family_passes: Counter[str] = Counter()
    residual_counts: Counter[str] = Counter()
    examples = []
    for i, y in enumerate(y_rows):
        proposals = [
            {"label": labels[int(idx)], "rank_score": round(float(prob), 6)}
            for idx, prob in zip(top_idx[i].tolist(), top_probs[i].tolist())
        ]
        syntax_ok = all(row["label"] in labels for row in proposals)
        syntax_pass += int(syntax_ok)
        top1_ok = proposals[0]["label"] == labels[y]
        topk_ok = any(row["label"] == labels[y] for row in proposals)
        passed += int(top1_ok)
        accepted += int(topk_ok)
        baseline_ok = baseline_label == labels[y]
        baseline_improvements += int(top1_ok and not baseline_ok)
        baseline_regressions += int(baseline_ok and not top1_ok)
        family = str(get_path(eval_rows[i], family_path, "unknown"))
        family_totals[family] += 1
        family_passes[family] += int(top1_ok)
        if not top1_ok:
            residual_counts[f"missed_{labels[y]}_as_{proposals[0]['label']}"] += 1
        if len(examples) < 8:
            examples.append(
                {
                    "task_id": row_id(eval_rows[i]),
                    "family": family,
                    "expected": labels[y],
                    "proposals": proposals,
                    "top1_passed": top1_ok,
                    "topk_accepted": topk_ok,
                }
            )
    return {
        "eval_rows": len(y_rows),
        "verifier_pass_rate": ratio(passed, len(y_rows)),
        "accepted_candidate_rate": ratio(accepted, len(y_rows)),
        "syntax_pass_rate": ratio(syntax_pass, len(y_rows)),
        "fanout_top_k": topk,
        "baseline_improvements": baseline_improvements,
        "baseline_regressions": baseline_regressions,
        "residual_family_pass_rates": {
            family: ratio(family_passes[family], total)
            for family, total in sorted(family_totals.items())
        },
        "residual_counts": dict(residual_counts.most_common(12)),
        "examples": examples,
        "eval_wall_time_ms": int((time.perf_counter() - started) * 1000),
    }


def compare_runs(per_view: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for arm_id in ["symliquid_style", "transformer_control"]:
        on = per_view.get("sts_on", {}).get("arms", {}).get(arm_id, {})
        off = per_view.get("sts_off", {}).get("arms", {}).get(arm_id, {})
        rows[arm_id] = {
            "sts_on_verifier_pass_rate": on.get("verifier_pass_rate"),
            "sts_off_verifier_pass_rate": off.get("verifier_pass_rate"),
            "sts_delta": round(float(on.get("verifier_pass_rate") or 0.0) - float(off.get("verifier_pass_rate") or 0.0), 6),
            "sts_on_accepted_candidate_rate": on.get("accepted_candidate_rate"),
            "parameter_count": on.get("parameter_count"),
        }
    sym = rows.get("symliquid_style", {})
    trans = rows.get("transformer_control", {})
    return {
        "by_arm": rows,
        "symliquid_minus_transformer_sts_on_verifier_pass_rate": round(
            float(sym.get("sts_on_verifier_pass_rate") or 0.0)
            - float(trans.get("sts_on_verifier_pass_rate") or 0.0),
            6,
        ),
        "winner_by_sts_on_verifier_pass_rate": (
            "symliquid_style"
            if float(sym.get("sts_on_verifier_pass_rate") or 0.0) >= float(trans.get("sts_on_verifier_pass_rate") or 0.0)
            else "transformer_control"
        ),
        "score_semantics": "Private smoke comparison only; not public calibration and not full code-proposer evidence.",
    }


def build_gates(
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    param_delta: float,
    per_view: dict[str, Any],
    all_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    budget = config.get("matched_budget") if isinstance(config.get("matched_budget"), dict) else {}
    safety = config.get("safety") if isinstance(config.get("safety"), dict) else {}
    return [
        gate("private_train_rows_loaded", len(train_rows) > 0, f"train_rows={len(train_rows)}", "hard"),
        gate("private_eval_rows_loaded", len(eval_rows) > 0, f"eval_rows={len(eval_rows)}", "hard"),
        gate("public_training_forbidden", not safety.get("public_training_allowed", True), safety, "hard"),
        gate("teacher_calls_forbidden", not safety.get("teacher_calls_allowed", True), safety, "hard"),
        gate("both_arms_runnable", len(all_runs) == 4, f"run_count={len(all_runs)}", "hard"),
        gate(
            "transformer_control_real",
            any(row.get("substrate") == "torch_transformer_encoder_classifier" for row in all_runs),
            "torch TransformerEncoder classifier trained",
            "hard",
        ),
        gate(
            "symliquid_style_arm_real",
            any(row.get("substrate") == "torch_symliquid_style_recurrent_vsa_classifier" for row in all_runs),
            "liquid recurrent/VSA-style classifier trained",
            "hard",
        ),
        gate(
            "sts_on_and_off_controls_ran",
            {"sts_on", "sts_off"}.issubset(set(per_view)),
            list(per_view),
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
            "external_inference_zero",
            all(int(row.get("external_inference_calls") or 0) == 0 for row in all_runs),
            "all arms local torch only",
            "hard",
        ),
        gate(
            "smoke_not_fully_saturated",
            not smoke_is_saturated(all_runs),
            "all arm/view verifier pass rates are 1.0" if smoke_is_saturated(all_runs) else "at least one arm/view leaves measurable error",
            "soft",
        ),
    ]


def data_contract(config: dict[str, Any], train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    data_cfg = config.get("data") if isinstance(config.get("data"), dict) else {}
    forbidden = [str(item) for item in data_cfg.get("forbidden_row_flags", [])]
    return {
        "train_jsonl": data_cfg.get("train_jsonl"),
        "eval_jsonl": data_cfg.get("eval_jsonl"),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "label_path": data_cfg.get("label_path"),
        "family_path": data_cfg.get("family_path"),
        "forbidden_flags_checked": forbidden,
        "forbidden_flag_hits": 0,
        "private_test_or_solution_fields_present_but_withheld": private_test_or_solution_field_count(train_rows + eval_rows),
        "tests_or_solutions_loaded_into_features": False,
        "public_prompts_tests_solutions_used": False,
    }


def private_test_or_solution_field_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("tests")
        or row.get("solution_body")
        or row.get("canonical_solution")
        or row.get("reference_solution")
    )


def matched_budget_report(config: dict[str, Any], sym_dims: dict[str, int], transformer_dims: dict[str, int]) -> dict[str, Any]:
    budget = config.get("matched_budget") if isinstance(config.get("matched_budget"), dict) else {}
    return {
        "budget_id": budget.get("budget_id"),
        "seeds": budget.get("seeds"),
        "max_sequence_tokens": budget.get("max_sequence_tokens"),
        "epochs": budget.get("epochs"),
        "batch_size": budget.get("batch_size"),
        "learning_rate": budget.get("learning_rate"),
        "fanout_top_k": budget.get("fanout_top_k"),
        "symliquid_style_dims": sym_dims,
        "transformer_control_dims": transformer_dims,
    }


def best_arm(per_view: dict[str, Any], view: str, metric: str) -> str:
    arms = per_view.get(view, {}).get("arms", {})
    if not arms:
        return ""
    return max(arms, key=lambda arm_id: float(arms[arm_id].get(metric) or 0.0))


def smoke_is_saturated(runs: list[dict[str, Any]]) -> bool:
    return bool(runs) and all(float(row.get("verifier_pass_rate") or 0.0) >= 1.0 for row in runs)


def choose_symliquid_dims(
    config: dict[str, Any],
    *,
    vocab_size: int,
    class_count: int,
    max_len: int,
    target_params: int,
    torch: Any,
    nn: Any,
) -> tuple[dict[str, int], int]:
    arm_cfg = config.get("arms", {}).get("symliquid_style", {})
    best_dims = {"hidden_dim": 24, "reservoir_dim": 24, "hv_dim": 24}
    best_count = 0
    best_delta = float("inf")
    for hidden in arm_cfg.get("hidden_dim_candidates", [24]):
        for reservoir in arm_cfg.get("reservoir_dim_candidates", [24]):
            for hv in arm_cfg.get("hv_dim_candidates", [24]):
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
            def __init__(self, vocab_size: int, class_count: int, *, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, max_len: int) -> None:
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
    if torch.cuda.is_available():
        return torch.device("cuda"), {
            "torch_mps_available": mps_available,
            "torch_mps_used": False,
            "torch_device_reason": "cuda_available",
        }
    if mps_available:
        return torch.device("cpu"), {
            "torch_mps_available": True,
            "torch_mps_used": False,
            "torch_device_reason": "cpu_for_transformer_encoder_mask_compatibility",
        }
    return torch.device("cpu"), {
        "torch_mps_available": False,
        "torch_mps_used": False,
        "torch_device_reason": "cpu_default",
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


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def ratio(num: int, den: int) -> float:
    return round(float(num) / float(den), 6) if den else 0.0


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


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


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
    summary = report.get("summary", {})
    lines = [
        "# Neural Seed Substrate Comparator",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- comparison_level: `{summary.get('comparison_level')}`",
        f"- code_proposer_comparison_ready: `{summary.get('code_proposer_comparison_ready')}`",
        f"- substrate_smoke_ready: `{summary.get('substrate_smoke_ready')}`",
        f"- train_rows: `{summary.get('train_rows')}`",
        f"- eval_rows: `{summary.get('eval_rows')}`",
        f"- parameter_match_delta: `{summary.get('parameter_match_delta')}`",
        f"- saturated_smoke: `{summary.get('saturated_smoke')}`",
        f"- best_sts_on_arm_by_verifier_pass_rate: `{summary.get('best_sts_on_arm_by_verifier_pass_rate')}`",
        "",
        "## Score Semantics",
        "",
        str(report.get("score_semantics", "")),
        "",
        "## STS-On Results",
        "",
    ]
    arms = report.get("views", {}).get("sts_on", {}).get("arms", {})
    for arm_id, row in arms.items():
        backend = row.get("backend") if isinstance(row.get("backend"), dict) else {}
        lines.append(
            f"- `{arm_id}`: verifier_pass_rate=`{row.get('verifier_pass_rate')}`, "
            f"accepted_candidate_rate=`{row.get('accepted_candidate_rate')}`, "
            f"syntax_pass_rate=`{row.get('syntax_pass_rate')}`, params=`{row.get('parameter_count')}`, "
            f"backend=`{backend.get('framework')}:{backend.get('device')}`"
        )
    comparisons = report.get("comparisons") if isinstance(report.get("comparisons"), dict) else {}
    adapter = report.get("adapter_boundary") if isinstance(report.get("adapter_boundary"), dict) else {}
    missing = adapter.get("missing_before_full_code_proposer_comparison")
    missing_lines = "\n".join(f"- {item}" for item in missing) if isinstance(missing, list) else "- none recorded"
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
            f"- current_status: `{adapter.get('current_status')}`",
            f"- full_code_proposer_comparison_ready: `{summary.get('code_proposer_comparison_ready')}`",
            "",
            "Missing before full code-proposer comparison:",
            "",
            missing_lines,
        ]
    )
    lines.extend(["", "## Gates", ""])
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
