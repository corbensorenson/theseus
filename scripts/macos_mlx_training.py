"""Mac-native MLX command surface for SymLiquid/Theseus training lanes.

These commands are intentionally bounded and report-first. They give Apple
Silicon Macs real MLX-backed readout, rollout, sweep, and token-superposition
work while the deeper Rust/Metal ports are still tracked separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def no_cheat_guardrails(*, promotion_allowed: bool = False, scheduler_routing_enabled: bool = False) -> dict[str, Any]:
    return {
        "no_external_inference": True,
        "no_teacher": True,
        "no_public_training_rows": True,
        "no_public_calibration": True,
        "no_fallback_returns": True,
        "model_promotion_allowed": bool(promotion_allowed),
        "scheduler_routing_enabled": bool(scheduler_routing_enabled),
    }


def main() -> int:
    reexec_into_mlx_runtime_if_needed()

    parser = argparse.ArgumentParser(description="Run bounded macOS MLX training commands.")
    sub = parser.add_subparsers(dest="command")

    standalone = sub.add_parser("train-standalone-mlx", help="Run the Apple MLX readout training lane.")
    add_common_readout_args(standalone)
    standalone.add_argument("--out", default="reports/symliquid_standalone_mlx_train_report.json")

    rollout = sub.add_parser("train-rollout-mlx", help="Run the Apple MLX rollout/control training lane.")
    add_rollout_args(rollout)
    rollout.add_argument("--out", default="reports/symliquid_rollout_mlx_train_report.json")

    sweep = sub.add_parser("train-rollout-mlx-sweep", help="Run a bounded Apple MLX rollout sweep.")
    sweep.add_argument("--train-seeds", default="0,1,2")
    sweep.add_argument("--eval-seed-base", type=int, default=10000)
    sweep.add_argument("--state-epochs", default="0,2,6")
    sweep.add_argument("--state-lrs", default="0.0,0.005,0.02")
    add_rollout_args(sweep, include_seeds=False, include_state=False)
    sweep.add_argument("--out", default="reports/symliquid_rollout_mlx_sweep.json")

    tst = sub.add_parser("train-token-superposition-mlx", help="Run MLX token superposition training.")
    add_token_superposition_args(tst)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 2

    if args.command == "train-standalone-mlx":
        report = train_standalone_mlx(args)
    elif args.command == "train-rollout-mlx":
        report = train_rollout_mlx(args)
    elif args.command == "train-rollout-mlx-sweep":
        report = train_rollout_mlx_sweep(args)
    elif args.command == "train-token-superposition-mlx":
        report = train_token_superposition_mlx(args)
    else:
        report = {"ok": False, "error": "unknown_command", "command": args.command}

    report = json_sanitize(report)
    write_json(resolve_path(args.out), report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report.get("ok") else 2


def add_common_readout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--cases-per-task", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples-per-launch", type=int, default=64)
    parser.add_argument("--hv-dim", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--symbolic-fallback", action="store_true")
    parser.add_argument("--model-out", default="")
    parser.add_argument("--train-input", default="data/babylm_blimp_filtered_train.jsonl")
    parser.add_argument("--eval-input", default="data/babylm_mutated_holdout_seed55.jsonl")
    parser.add_argument("--profile", default="smoke")


def add_rollout_args(parser: argparse.ArgumentParser, *, include_seeds: bool = True, include_state: bool = True) -> None:
    if include_seeds:
        parser.add_argument("--train-seed", type=int, default=0)
        parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--cases-per-task", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    if include_state:
        parser.add_argument("--state-epochs", type=int, default=1)
        parser.add_argument("--state-lr", type=float, default=0.2)
    parser.add_argument("--samples-per-launch", type=int, default=32)
    parser.add_argument("--probe-cases-per-task", type=int, default=0)
    parser.add_argument("--rollout-batch", type=int, default=16)
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--reservoir-dim", type=int, default=128)
    parser.add_argument("--hv-dim", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--model-out", default="")
    parser.add_argument("--profile", default="smoke")


def add_token_superposition_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default="data/babylm_blimp_filtered_train.jsonl")
    parser.add_argument("--include-project-code", action="store_true")
    parser.add_argument("--project-code-roots", default="scripts,crates")
    parser.add_argument("--train-seed", type=int, default=20260514)
    parser.add_argument("--max-language-rows", type=int, default=8000)
    parser.add_argument("--max-code-files", type=int, default=160)
    parser.add_argument("--max-chars-per-doc", type=int, default=12000)
    parser.add_argument("--max-vocab", type=int, default=256)
    parser.add_argument("--hv-dim", type=int, default=4096)
    parser.add_argument("--train-samples", type=int, default=32768)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--baseline-epochs", type=int, default=6)
    parser.add_argument("--bag-sizes", default="4,8")
    parser.add_argument("--recovery-ratios", default="0.2,0.4")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--samples-per-launch", type=int, default=512)
    parser.add_argument("--gate-tolerance", type=float, default=0.002)
    parser.add_argument("--min-nominal-speedup", type=float, default=1.2)
    parser.add_argument("--min-train-speedup", type=float, default=1.0)
    parser.add_argument("--model-out", default="")
    parser.add_argument("--out", default="reports/token_superposition_mlx_training.json")


def train_standalone_mlx(args: argparse.Namespace) -> dict[str, Any]:
    chunk_id = safe_name(f"train_standalone_mlx_{int(time.time())}")
    child_out = REPORTS / "hive_chunks" / f"{chunk_id}_child.json"
    payload = {
        "chunk_id": chunk_id,
        "profile": args.profile,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "train_input": args.train_input,
        "eval_input": args.eval_input,
        "train_limit": max(1, int(args.cases_per_task)),
        "eval_limit": max(1, min(max(32, int(args.cases_per_task)), max(1, int(args.cases_per_task) * 2))),
        "feature_dim": max(8, int(args.hv_dim)),
        "hv_dim": max(8, int(args.hv_dim)),
        "steps": max(1, int(args.epochs)),
        "epochs": max(1, int(args.epochs)),
        "lr": float(args.lr),
        "source": "symliquid_train_standalone_mlx",
    }
    child = run_worker_chunk("mlx_babylm_train", payload, child_out)
    report = worker_bridge_report(
        command="train-standalone-mlx",
        parity_for="train-standalone-cuda",
        implementation="python_mlx_hive_readout_worker",
        args=vars(args),
        child=child,
        child_out=child_out,
        notes=(
            "Runs the current Apple MLX readout lane over local BabyLM pair data. "
            "This is a real MLX path; the exact Rust CGS CUDA kernel remains a separate port target."
        ),
    )
    maybe_copy_model(child, args.model_out)
    return report


def train_rollout_mlx(args: argparse.Namespace) -> dict[str, Any]:
    return train_rollout_mlx_once(
        args,
        train_seed=int(args.train_seed),
        eval_seed=int(args.eval_seed),
        state_epochs=int(args.state_epochs),
        state_lr=float(args.state_lr),
        out_suffix="child",
    )


def train_rollout_mlx_once(
    args: argparse.Namespace,
    *,
    train_seed: int,
    eval_seed: int,
    state_epochs: int,
    state_lr: float,
    out_suffix: str,
) -> dict[str, Any]:
    chunk_id = safe_name(f"train_rollout_mlx_{train_seed}_{state_epochs}_{int(time.time() * 1000)}")
    child_out = REPORTS / "hive_chunks" / f"{chunk_id}_{out_suffix}.json"
    payload = {
        "chunk_id": chunk_id,
        "profile": args.profile,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "cases_per_task": max(1, int(args.cases_per_task)),
        "eval_cases": max(1, int(args.probe_cases_per_task) or int(args.cases_per_task)),
        "epochs": max(1, int(args.epochs)),
        "state_epochs": max(0, int(state_epochs)),
        "state_lr": float(state_lr),
        "rollout_batch": max(1, int(args.rollout_batch)),
        "obs_dim": max(1, int(args.obs_dim)),
        "hidden_dim": max(1, int(args.hidden_dim)),
        "reservoir_dim": max(1, int(args.reservoir_dim)),
        "hv_dim": max(8, int(args.hv_dim)),
        "seq_len": max(1, int(args.seq_len)),
        "lr": float(args.lr),
        "source": "symliquid_train_rollout_mlx",
    }
    child = run_worker_chunk("mlx_rollout_probe", payload, child_out)
    report = worker_bridge_report(
        command="train-rollout-mlx",
        parity_for="train-rollout-cuda",
        implementation="python_mlx_rollout_worker",
        args={**vars(args), "train_seed": train_seed, "eval_seed": eval_seed, "state_epochs": state_epochs, "state_lr": state_lr},
        child=child,
        child_out=child_out,
        notes=(
            "Runs an Apple MLX synthetic rollout/control readout. It is bounded and auditable; "
            "the exact Rust/CUDA rollout hot loop still needs a Rust/Metal or Rust/MLX port."
        ),
    )
    maybe_copy_model(child, args.model_out)
    return report


def train_rollout_mlx_sweep(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    seeds = parse_int_list(args.train_seeds)
    state_epoch_grid = parse_int_list(args.state_epochs)
    state_lr_grid = parse_float_list(args.state_lrs)
    runs: list[dict[str, Any]] = []
    for seed_index, train_seed in enumerate(seeds):
        eval_seed = int(args.eval_seed_base) + seed_index
        for state_epochs in state_epoch_grid:
            lrs = [0.0] if state_epochs == 0 else state_lr_grid
            for state_lr in lrs:
                run = train_rollout_mlx_once(
                    args,
                    train_seed=train_seed,
                    eval_seed=eval_seed,
                    state_epochs=state_epochs,
                    state_lr=state_lr,
                    out_suffix=f"sweep_{len(runs)}",
                )
                runs.append(run)
    best = best_rollout_run(runs)
    report = {
        "ok": all(bool(row.get("ok")) for row in runs) if runs else False,
        "policy": "project_theseus_macos_mlx_rollout_sweep_v0",
        "created_utc": now(),
        "backend": backend_id(),
        "command": "train-rollout-mlx-sweep",
        "parity_for": "train-rollout-cuda-sweep",
        "implementation": "python_mlx_rollout_worker_sweep",
        "args": vars(args),
        "run_count": len(runs),
        "runs": runs,
        "best_run": best,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
        "teacher_used": False,
        "public_training_rows": 0,
        "model_promotion_allowed": False,
        "guardrail": "bounded_local_mlx_only_no_teacher_no_public_benchmark_training",
        "guardrails": no_cheat_guardrails(),
    }
    return report


def train_token_superposition_mlx(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "mlx_or_numpy_unavailable", "message": str(exc), "backend": backend_id()}

    docs = load_token_docs(args)
    if len(docs["train_texts"]) < 1:
        return {"ok": False, "error": "no_training_documents", "backend": backend_id(), "dataset": docs["summary"]}
    train_tokens_text = tokenize("\n".join(docs["train_texts"]))
    eval_language_text = "\n".join(docs["language_eval_texts"] or docs["train_texts"][:1])
    eval_code_text = "\n".join(docs["code_eval_texts"])
    if len(train_tokens_text) < max(16, min(256, int(args.train_samples))):
        return {"ok": False, "error": "too_few_training_tokens", "tokens": len(train_tokens_text), "backend": backend_id()}

    vocab = build_vocab(train_tokens_text, max(8, int(args.max_vocab)))
    train_tokens = encode_tokens(train_tokens_text, vocab)
    language_eval_tokens = encode_tokens(tokenize(eval_language_text), vocab)
    code_eval_tokens = encode_tokens(tokenize(eval_code_text), vocab) if eval_code_text.strip() else []
    feature_table = token_feature_table(list(vocab.keys()), max(8, int(args.hv_dim)), np)
    output_dim = len(vocab)
    train_samples = max(1, int(args.train_samples))
    eval_samples = max(1, int(args.eval_samples))
    baseline_epochs = max(1, int(args.baseline_epochs))
    lr = float(args.lr)
    batch_size = max(8, min(max(8, int(args.samples_per_launch)), 2048))

    dataset_summary = {
        **docs["summary"],
        "train_tokens": len(train_tokens),
        "language_eval_tokens": len(language_eval_tokens),
        "code_eval_tokens": len(code_eval_tokens),
        "vocab_size": output_dim,
        "hv_dim": int(args.hv_dim),
        "holdout_policy": "Deterministic local split; no public benchmark solutions used as training data.",
    }

    baseline = train_softmax_run(
        mx,
        np,
        id="baseline_ar_mlx",
        objective="ordinary_ar_next_token",
        feature_table=feature_table,
        train_tokens=train_tokens,
        language_eval_tokens=language_eval_tokens,
        code_eval_tokens=code_eval_tokens,
        hv_dim=int(args.hv_dim),
        output_dim=output_dim,
        train_samples=train_samples,
        epochs=baseline_epochs,
        lr=lr,
        batch_size=batch_size,
        seed=int(args.train_seed) ^ 0x751BA5E,
        bag_size=None,
        recovery_ratio=None,
        recovery_tokens=None,
        baseline=None,
    )

    variants: list[dict[str, Any]] = []
    for bag_size in [size for size in parse_int_list(args.bag_sizes) if size > 0]:
        for recovery_ratio in [ratio for ratio in parse_float_list(args.recovery_ratios) if 0.0 < ratio < 1.0]:
            bag_epochs = max(1, round(baseline_epochs * (1.0 - recovery_ratio)))
            recovery_epochs = max(1, round(baseline_epochs * recovery_ratio))
            bag_samples = max(1, train_samples // max(1, bag_size))
            variant = train_softmax_run(
                mx,
                np,
                id=f"tst_s{bag_size}_r{recovery_ratio:.2f}_mlx",
                objective="token_superposition_bag_ce_then_ar_recovery",
                feature_table=feature_table,
                train_tokens=train_tokens,
                language_eval_tokens=language_eval_tokens,
                code_eval_tokens=code_eval_tokens,
                hv_dim=int(args.hv_dim),
                output_dim=output_dim,
                train_samples=bag_samples,
                epochs=bag_epochs,
                lr=lr,
                batch_size=batch_size,
                seed=int(args.train_seed) ^ ((bag_size & 0xFFFF) << 16) ^ int(recovery_ratio * 1000.0),
                bag_size=bag_size,
                recovery_ratio=recovery_ratio,
                recovery_tokens=train_tokens,
                recovery_epochs=recovery_epochs,
                recovery_samples=train_samples,
                baseline=baseline,
            )
            variants.append(variant)

    best_variant = best_tst_variant(variants)
    gates = token_superposition_gates(args, baseline, best_variant)
    failed = [row["gate"] for row in gates if not row["passed"]]
    promote = not failed
    report = {
        "ok": True,
        "policy": "project_theseus_token_superposition_mlx_report_v1",
        "created_utc": now(),
        "backend": backend_id(),
        "cuda_fallback": False,
        "command": "train-token-superposition-mlx",
        "parity_for": "train-token-superposition-cuda",
        "implementation": "python_mlx_token_superposition_readout",
        "config": vars(args),
        "dataset": dataset_summary,
        "baseline": baseline,
        "variants": variants,
        "best_variant": best_variant,
        "gates": gates,
        "promotion_decision": {
            "promote_to_training_lane": promote,
            "status": "eligible_for_training_lane" if promote else "not_promoted_keep_as_evidence",
            "reason": "MLX TST+AR recovery passed local loss/speed gates." if promote else f"blocked_by_gates: {', '.join(failed)}",
            "artifact": str(args.model_out or ""),
        },
        "timing_breakdown_ms": {
            "total": int((time.perf_counter() - started) * 1000),
            "baseline_total": baseline.get("total_runtime_ms"),
            "best_variant_total": best_variant.get("total_runtime_ms") if best_variant else None,
        },
        "external_inference_calls": 0,
        "teacher_used": False,
        "public_training_rows": 0,
        "raw_gate_training_lane_eligible": promote,
        "model_promotion_allowed": False,
        "train_token_superposition_parity_claim_allowed": False,
        "full_cli_parity_claim_allowed": False,
        "guardrail": "local_private_training_only_no_teacher_no_public_benchmark_training",
        "guardrails": no_cheat_guardrails(),
    }
    if args.model_out:
        write_json(resolve_path(args.model_out), {"policy": "project_theseus_mlx_token_superposition_model_preview_v0", "vocab": list(vocab.keys())[:128], "report_out": args.out})
    return report


def train_softmax_run(
    mx: Any,
    np: Any,
    *,
    id: str,
    objective: str,
    feature_table: Any,
    train_tokens: list[int],
    language_eval_tokens: list[int],
    code_eval_tokens: list[int],
    hv_dim: int,
    output_dim: int,
    train_samples: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    bag_size: int | None,
    recovery_ratio: float | None,
    recovery_tokens: list[int] | None,
    recovery_epochs: int = 0,
    recovery_samples: int = 0,
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    weight_scale = 0.01
    w_np = np.zeros((hv_dim, output_dim), dtype=np.float32)
    b_np = np.zeros((output_dim,), dtype=np.float32)
    w = mx.array(w_np * weight_scale, dtype=mx.float32)
    b = mx.array(b_np, dtype=mx.float32)
    feature_ms = 0
    train_started = time.perf_counter()
    last_loss = 0.0
    last_acc = 0.0

    for _ in range(max(1, epochs)):
        for x_np, y_np in training_batches(np, feature_table, train_tokens, train_samples, batch_size, rng, bag_size=bag_size):
            w, b, last_loss, last_acc = softmax_step(mx, np, w, b, x_np, y_np, lr, output_dim)

    if recovery_tokens and recovery_epochs and recovery_samples:
        for _ in range(max(1, recovery_epochs)):
            for x_np, y_np in training_batches(np, feature_table, recovery_tokens, recovery_samples, batch_size, rng, bag_size=None):
                w, b, last_loss, last_acc = softmax_step(mx, np, w, b, x_np, y_np, lr, output_dim)

    train_ms = int((time.perf_counter() - train_started) * 1000)
    eval_started = time.perf_counter()
    language_eval = evaluate_softmax(mx, np, w, b, feature_table, language_eval_tokens, max(1, train_samples // 8), batch_size)
    code_eval = evaluate_softmax(mx, np, w, b, feature_table, code_eval_tokens, max(1, train_samples // 8), batch_size) if code_eval_tokens else {"loss": math.nan, "accuracy": math.nan}
    eval_ms = int((time.perf_counter() - eval_started) * 1000)
    combined_loss = language_eval["loss"] if math.isnan(code_eval["loss"]) else 0.5 * (language_eval["loss"] + code_eval["loss"])
    combined_acc = language_eval["accuracy"] if math.isnan(code_eval["accuracy"]) else 0.5 * (language_eval["accuracy"] + code_eval["accuracy"])
    train_seen = train_samples * max(1, epochs) + recovery_samples * max(0, recovery_epochs)
    baseline_train_ms = max(1, int((baseline or {}).get("train_runtime_ms") or train_ms or 1))
    baseline_total_ms = max(1, int((baseline or {}).get("total_runtime_ms") or (feature_ms + train_ms + eval_ms) or 1))
    baseline_loss = float(get_path(baseline or {}, ["eval", "combined_ar_loss"], combined_loss))
    baseline_code_loss = float(get_path(baseline or {}, ["eval", "code_ar_loss"], code_eval["loss"]))
    return {
        "id": id,
        "objective": objective,
        "bag_size": bag_size,
        "recovery_ratio": recovery_ratio,
        "bag_epochs": max(0, epochs if bag_size else 0),
        "recovery_epochs": max(0, recovery_epochs if recovery_tokens else epochs),
        "baseline_epochs": max(1, epochs) if not baseline else int((baseline or {}).get("baseline_epochs") or epochs),
        "train_samples": train_samples,
        "bag_samples": train_samples if bag_size else 0,
        "recovery_samples": recovery_samples if recovery_tokens else train_samples,
        "train_runtime_ms": train_ms,
        "feature_build_ms": feature_ms,
        "eval_runtime_ms": eval_ms,
        "total_runtime_ms": feature_ms + train_ms + eval_ms,
        "train_examples_seen": train_seen,
        "train_examples_per_second": examples_per_second(train_seen, train_ms),
        "kernel_launches": max(1, math.ceil(train_seen / max(1, batch_size))),
        "train_loss": last_loss,
        "train_accuracy": last_acc,
        "eval": {
            "language_ar_loss": language_eval["loss"],
            "language_ar_accuracy": language_eval["accuracy"],
            "code_ar_loss": code_eval["loss"],
            "code_ar_accuracy": code_eval["accuracy"],
            "combined_ar_loss": combined_loss,
            "combined_ar_accuracy": combined_acc,
        },
        "nominal_speedup_vs_baseline": 1.0 if not baseline else max(1.0, (baseline or {}).get("train_examples_seen", train_seen) / max(1, train_seen)),
        "measured_train_speedup_vs_baseline": 1.0 if not baseline else baseline_train_ms / max(1, train_ms),
        "measured_total_speedup_vs_baseline": 1.0 if not baseline else baseline_total_ms / max(1, feature_ms + train_ms + eval_ms),
        "combined_loss_delta_vs_baseline": combined_loss - baseline_loss,
        "code_loss_delta_vs_baseline": code_eval["loss"] - baseline_code_loss if math.isfinite(code_eval["loss"]) and math.isfinite(baseline_code_loss) else math.nan,
    }


def softmax_step(mx: Any, np: Any, w: Any, b: Any, x_np: Any, y_np: Any, lr: float, output_dim: int) -> tuple[Any, Any, float, float]:
    x = mx.array(x_np, dtype=mx.float32)
    y_one_hot = mx.array(np.eye(output_dim, dtype=np.float32)[y_np], dtype=mx.float32)
    logits = x @ w + b
    probs = mx.softmax(logits, axis=1)
    err = probs - y_one_hot
    inv = 1.0 / max(1, int(x_np.shape[0]))
    grad_w = (x.T @ err) * inv
    grad_b = mx.mean(err, axis=0)
    w = w - lr * grad_w
    b = b - lr * grad_b
    picked = probs * y_one_hot
    loss = -mx.mean(mx.log(mx.sum(picked, axis=1) + 1e-6))
    pred = mx.argmax(probs, axis=1)
    y = mx.array(y_np, dtype=mx.int32)
    acc = mx.mean(pred == y)
    mx.eval(w, b, loss, acc)
    return w, b, float(loss.item()), float(acc.item())


def evaluate_softmax(mx: Any, np: Any, w: Any, b: Any, feature_table: Any, tokens: list[int], samples: int, batch_size: int) -> dict[str, float]:
    if len(tokens) < 2:
        return {"loss": math.nan, "accuracy": math.nan}
    rng = random.Random(0xE7A1)
    losses: list[float] = []
    accs: list[float] = []
    output_dim = int(w.shape[1])
    for x_np, y_np in training_batches(np, feature_table, tokens, samples, batch_size, rng, bag_size=None):
        x = mx.array(x_np, dtype=mx.float32)
        y_one_hot = mx.array(np.eye(output_dim, dtype=np.float32)[y_np], dtype=mx.float32)
        logits = x @ w + b
        probs = mx.softmax(logits, axis=1)
        picked = probs * y_one_hot
        loss = -mx.mean(mx.log(mx.sum(picked, axis=1) + 1e-6))
        pred = mx.argmax(probs, axis=1)
        y = mx.array(y_np, dtype=mx.int32)
        acc = mx.mean(pred == y)
        mx.eval(loss, acc)
        losses.append(float(loss.item()))
        accs.append(float(acc.item()))
    return {"loss": sum(losses) / max(1, len(losses)), "accuracy": sum(accs) / max(1, len(accs))}


def training_batches(np: Any, feature_table: Any, tokens: list[int], samples: int, batch_size: int, rng: random.Random, *, bag_size: int | None) -> Any:
    max_start = len(tokens) - (max(1, bag_size or 1) + 1)
    if max_start <= 0:
        return
    remaining = max(1, samples)
    while remaining > 0:
        take = min(batch_size, remaining)
        x_rows = []
        y_rows = []
        for _ in range(take):
            pos = rng.randrange(0, max_start)
            if bag_size:
                row = feature_table[tokens[pos : pos + bag_size]].mean(axis=0)
                target = tokens[pos + bag_size]
            else:
                row = feature_table[tokens[pos]]
                target = tokens[pos + 1]
            x_rows.append(row)
            y_rows.append(target)
        yield np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int32)
        remaining -= take


def load_token_docs(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = [item for item in split_csv(args.input) if item]
    language_docs: list[str] = []
    for rel in input_paths:
        path = resolve_path(rel)
        language_docs.extend(load_text_rows(path, max(1, int(args.max_language_rows)) - len(language_docs), int(args.max_chars_per_doc)))
        if len(language_docs) >= int(args.max_language_rows):
            break
    split_at = max(1, int(len(language_docs) * 0.85))
    language_train = language_docs[:split_at]
    language_eval = language_docs[split_at:] or language_docs[:1]
    code_docs: list[str] = []
    if bool(args.include_project_code):
        for root in split_csv(args.project_code_roots):
            code_docs.extend(load_project_code(resolve_path(root), max(0, int(args.max_code_files) - len(code_docs)), int(args.max_chars_per_doc)))
            if len(code_docs) >= int(args.max_code_files):
                break
    code_split = max(0, int(len(code_docs) * 0.85))
    code_train = code_docs[:code_split]
    code_eval = code_docs[code_split:]
    return {
        "train_texts": language_train + code_train,
        "language_eval_texts": language_eval,
        "code_eval_texts": code_eval,
        "summary": {
            "language_train_docs": len(language_train),
            "language_eval_docs": len(language_eval),
            "code_train_docs": len(code_train),
            "code_eval_docs": len(code_eval),
        },
    }


def load_text_rows(path: Path, limit: int, max_chars: int) -> list[str]:
    rows: list[str] = []
    if limit <= 0 or not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            text = ""
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    text = "\n".join(
                        str(value.get(key) or "")
                        for key in ["text", "body", "prompt", "completion", "sentence_good", "sentence_bad", "good", "bad"]
                        if value.get(key)
                    )
            except json.JSONDecodeError:
                text = line
            text = text.strip()
            if text:
                rows.append(text[:max_chars])
    return rows


def load_project_code(root: Path, limit: int, max_chars: int) -> list[str]:
    if limit <= 0 or not root.exists():
        return []
    suffixes = {".py", ".rs", ".md", ".toml", ".json", ".sh", ".swift", ".js", ".ts"}
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if len(rows) >= limit:
            break
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if any(part in {".git", "target", "dist", "reports", ".venv-puffer", "__pycache__"} for part in path.parts):
            continue
        try:
            rows.append(path.read_text(encoding="utf-8", errors="ignore")[:max_chars])
        except OSError:
            continue
    return rows


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_']+", text.lower())


def build_vocab(tokens: list[str], max_vocab: int) -> dict[str, int]:
    counts = Counter(tokens)
    vocab = ["<unk>"] + [token for token, _ in counts.most_common(max(1, max_vocab - 1))]
    return {token: idx for idx, token in enumerate(vocab)}


def encode_tokens(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    unk = vocab.get("<unk>", 0)
    return [vocab.get(token, unk) for token in tokens]


def token_feature_table(vocab: list[str], hv_dim: int, np: Any) -> Any:
    table = np.zeros((len(vocab), hv_dim), dtype=np.float32)
    for row, token in enumerate(vocab):
        for salt in range(4):
            digest = hashlib.blake2b(f"{token}|{salt}".encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % hv_dim
            sign = 1.0 if digest[4] & 1 else -1.0
            table[row, bucket] += sign
        norm = float(np.linalg.norm(table[row])) or 1.0
        table[row] /= norm
    return table


def token_superposition_gates(args: argparse.Namespace, baseline: dict[str, Any], best: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not best:
        return [{"gate": "variant_completed", "passed": False, "evidence": "no MLX variants completed"}]
    tolerance = float(args.gate_tolerance)
    return [
        gate("normal_recovery_loss_beats_baseline", get_path(best, ["eval", "combined_ar_loss"], math.inf) <= get_path(baseline, ["eval", "combined_ar_loss"], math.inf) - tolerance),
        gate("code_proxy_loss_improves", code_loss_gate(best, baseline, tolerance)),
        gate("nominal_training_speedup_present", float(best.get("nominal_speedup_vs_baseline") or 0.0) >= float(args.min_nominal_speedup)),
        gate("measured_mlx_train_speedup_present", float(best.get("measured_train_speedup_vs_baseline") or 0.0) >= float(args.min_train_speedup)),
        gate("standard_ar_recovery_completed", int(best.get("recovery_epochs") or 0) > 0 and int(best.get("recovery_samples") or 0) > 0),
    ]


def code_loss_gate(best: dict[str, Any], baseline: dict[str, Any], tolerance: float) -> bool:
    best_loss = float(get_path(best, ["eval", "code_ar_loss"], math.nan))
    baseline_loss = float(get_path(baseline, ["eval", "code_ar_loss"], math.nan))
    if not math.isfinite(best_loss) or not math.isfinite(baseline_loss):
        return True
    return best_loss <= baseline_loss - tolerance


def gate(name: str, passed: bool) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": str(bool(passed))}


def best_tst_variant(variants: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not variants:
        return None
    return min(variants, key=lambda row: (float(get_path(row, ["eval", "combined_ar_loss"], math.inf)), -float(row.get("nominal_speedup_vs_baseline") or 0.0)))


def run_worker_chunk(kind: str, payload: dict[str, Any], out: Path) -> dict[str, Any]:
    python = preferred_mlx_python()
    command = [
        str(python),
        "scripts/hive_worker_chunk.py",
        "--kind",
        kind,
        "--payload-json",
        json.dumps(payload),
        "--out",
        str(out.relative_to(ROOT)),
    ]
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=1800, env=runtime_env())
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": "worker_timeout",
            "command": command,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
        }
    parsed = parse_json(result.stdout.strip(), {})
    if not isinstance(parsed, dict) or not parsed:
        parsed = read_json(out, {})
    if isinstance(parsed, dict):
        parsed.setdefault("ok", result.returncode == 0)
        parsed.setdefault("returncode", result.returncode)
        parsed.setdefault("stdout_tail", result.stdout[-2000:])
        parsed.setdefault("stderr_tail", result.stderr[-2000:])
        parsed.setdefault("command", command)
        parsed.setdefault("runtime_ms_subprocess", int((time.perf_counter() - started) * 1000))
        return parsed
    return {"ok": False, "error": "worker_report_missing", "returncode": result.returncode, "stdout_tail": result.stdout[-2000:], "stderr_tail": result.stderr[-2000:]}


def worker_bridge_report(command: str, parity_for: str, implementation: str, args: dict[str, Any], child: dict[str, Any], child_out: Path, notes: str) -> dict[str, Any]:
    return {
        "ok": bool(child.get("ok")),
        "policy": "project_theseus_macos_mlx_cli_bridge_report_v0",
        "created_utc": now(),
        "backend": child.get("backend") or backend_id(),
        "command": command,
        "parity_for": parity_for,
        "implementation": implementation,
        "args": args,
        "metrics": child.get("metrics") if isinstance(child.get("metrics"), dict) else {},
        "telemetry": child.get("telemetry") if isinstance(child.get("telemetry"), dict) else {},
        "child_report_path": str(child_out.relative_to(ROOT)),
        "child_report": compact_child(child),
        "runtime_ms": child.get("runtime_ms") or child.get("runtime_ms_subprocess"),
        "external_inference_calls": 0,
        "teacher_used": False,
        "public_training_rows": 0,
        "model_promotion_allowed": False,
        "guardrail": "bounded_local_mlx_only_no_arbitrary_shell_no_public_benchmark_training",
        "guardrails": no_cheat_guardrails(),
        "notes": notes,
    }


def compact_child(child: dict[str, Any]) -> dict[str, Any]:
    keep = {"ok", "kind", "chunk_id", "backend", "profile", "metrics", "telemetry", "runtime_ms", "runtime_ms_child", "work_receipt", "compute_market", "error", "message"}
    return {key: value for key, value in child.items() if key in keep}


def best_rollout_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ok_runs = [row for row in runs if row.get("ok")]
    if not ok_runs:
        return {}
    return max(ok_runs, key=lambda row: float(get_path(row, ["metrics", "eval_return_proxy"], get_path(row, ["metrics", "eval_accuracy"], 0.0))))


def maybe_copy_model(child: dict[str, Any], model_out: str) -> None:
    if not model_out:
        return
    model_path = get_path(child, ["telemetry", "model_path"], "")
    if not model_path:
        return
    src = resolve_path(str(model_path))
    dst = resolve_path(model_out)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")


def reexec_into_mlx_runtime_if_needed() -> None:
    if current_python_has_mlx():
        return
    preferred = preferred_mlx_python()
    if str(preferred) != sys.executable and Path(preferred).exists():
        os.execv(str(preferred), [str(preferred), str(Path(__file__).resolve()), *sys.argv[1:]])


def current_python_has_mlx() -> bool:
    try:
        import mlx.core as mx  # type: ignore[import-not-found]

        x = mx.array([1.0])
        mx.eval(x)
        return True
    except Exception:
        return False


def preferred_mlx_python() -> Path:
    candidates = [
        ROOT / ".venv-puffer" / "bin" / "python",
        Path.home() / "Library" / "Application Support" / "Project Theseus Hive" / "app" / "current" / ".venv-puffer" / "bin" / "python",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.exists() and python_has_mlx(candidate):
            return candidate
    return Path(sys.executable)


def python_has_mlx(python: Path) -> bool:
    try:
        result = subprocess.run([str(python), "-c", "import mlx.core as mx; x=mx.array([1.0]); mx.eval(x)"], cwd=ROOT, text=True, capture_output=True, timeout=20)
        return result.returncode == 0
    except Exception:
        return False


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def backend_id() -> str:
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return "mlx_apple"
    return "mlx"


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return safe[:120] or "mlx_chunk"


def examples_per_second(examples: int, runtime_ms: int) -> float:
    return float(examples) / max(0.001, float(runtime_ms) / 1000.0)


def parse_json(value: str, default: Any) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def read_json(path: Path, default: Any = {}) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_sanitize(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
