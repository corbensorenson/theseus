"""Bounded CUDA/MLX worker chunks for the Project Theseus Hive.

This runner is the execution boundary between Hive scheduling and real
training/eval work. It accepts only named chunk kinds, clamps payload values to
policy limits, writes auditable reports, and never shells out to arbitrary
commands from remote payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import compute_market
import theseus_runtime


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "hive_policy.json"
DEFAULT_PROFILES = ROOT / "configs" / "training_profiles_rtx2060super.json"
DEFAULT_LEDGER = ROOT / "reports" / "hive_worker_chunk_ledger.jsonl"
CHUNK_KINDS = {
    "cuda_eval_chunk",
    "cuda_readout_train",
    "cuda_rollout_train",
    "mlx_babylm_eval",
    "mlx_babylm_train",
    "mlx_rollout_probe",
}


def worker_no_cheat_guardrails() -> dict[str, Any]:
    return {
        "no_external_inference": True,
        "no_teacher": True,
        "no_public_training_rows": True,
        "no_public_calibration": True,
        "no_fallback_returns": True,
        "model_promotion_allowed": False,
        "scheduler_routing_enabled": False,
        "arbitrary_shell_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=sorted(CHUNK_KINDS), required=True)
    parser.add_argument("--payload-json", default="{}")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--profiles", default=str(DEFAULT_PROFILES.relative_to(ROOT)))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    started = time.perf_counter()
    policy = read_json(ROOT / args.policy, {})
    profiles = read_json(ROOT / args.profiles, {})
    payload = parse_json(args.payload_json, {})
    chunk_id = safe_name(str(payload.get("chunk_id") or f"{args.kind}_{int(time.time())}"))
    out_path = safe_report_path(
        Path(args.out) if args.out else Path(str(payload.get("out") or f"reports/hive_chunks/{chunk_id}.json"))
    )
    report: dict[str, Any] = {
        "ok": False,
        "policy": "project_theseus_hive_worker_chunk_v0",
        "created_utc": now(),
        "kind": args.kind,
        "chunk_id": chunk_id,
        "profile": str(payload.get("profile") or get_path(policy, ["worker_chunks", "default_profile"], "smoke")),
        "job": job_metadata(args.kind, payload, chunk_id),
        "requester": requester_metadata(payload),
        "orchestration": orchestration_metadata(payload),
        "maintenance_mode": maintenance_mode_from_payload(payload),
        "maintenance_mode_basis": "payload_or_object_only_default",
        "platform": platform_report(),
        "external_inference_calls": 0,
        "teacher_used": False,
        "public_training_rows": 0,
        "model_promotion_allowed": False,
        "guardrails": worker_no_cheat_guardrails(),
    }

    try:
        result = run_chunk(args.kind, payload, policy, profiles, chunk_id)
        report.update(result)
        report["ok"] = bool(result.get("ok", True))
    except Exception as exc:  # noqa: BLE001 - report all worker boundary failures.
        report.update({"ok": False, "error": type(exc).__name__, "message": str(exc)})
    report["runtime_ms"] = int((time.perf_counter() - started) * 1000)
    report["work_receipt"] = build_work_receipt(args.kind, payload, report)
    report["compute_market"] = settle_market_receipt(report["work_receipt"])
    report["review_step_count"] = worker_review_step_count(payload, report)
    report["review_step_basis"] = "profile_validation_chunk_execution_work_receipt_compute_market"
    report["human_edit_minutes"] = None
    report["human_edit_minutes_measured"] = False
    write_json(out_path, report)
    append_jsonl(DEFAULT_LEDGER, {**report, "report_path": project_display_path(out_path)})
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 2


def run_chunk(
    kind: str,
    payload: dict[str, Any],
    policy: dict[str, Any],
    profiles: dict[str, Any],
    chunk_id: str,
) -> dict[str, Any]:
    profile_name = validate_profile(payload, policy)
    if kind in {"cuda_eval_chunk", "cuda_readout_train", "cuda_rollout_train"}:
        return run_cuda_chunk(kind, payload, policy, profiles, profile_name, chunk_id)
    if kind in {"mlx_babylm_eval", "mlx_babylm_train"}:
        return run_mlx_chunk(kind, payload, policy, profiles, profile_name, chunk_id)
    if kind == "mlx_rollout_probe":
        return run_mlx_rollout_chunk(kind, payload, policy, profiles, profile_name, chunk_id)
    return {"ok": False, "error": "unknown_chunk_kind", "kind": kind}


def build_work_receipt(kind: str, payload: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    profile = str(payload.get("profile") or report.get("profile") or "smoke")
    task_kind = hive_task_kind_for_internal(kind)
    hv_dim = safe_int(payload, "hv_dim", int(metrics.get("feature_dim") or 512), 16384)
    epochs = safe_int(payload, "epochs", int(payload.get("steps") or metrics.get("steps") or 1), 100000)
    cases = safe_int(
        payload,
        "cases_per_task",
        int(metrics.get("train_rows") or metrics.get("expanded_train_examples") or 1),
        10_000_000,
    )
    units = max(1, cases) * max(1, epochs) * max(1, hv_dim)
    if kind.startswith("mlx"):
        units = max(1, int(metrics.get("expanded_train_examples") or cases)) * max(1, epochs) * max(1, hv_dim)
    return {
        "version": "theseus_verified_work_receipt_v0",
        "accounting_only": True,
        "accepted": bool(report.get("ok")),
        "task_kind": task_kind,
        "worker_kind": kind,
        "backend": report.get("backend"),
        "profile": profile,
        "difficulty_class": "inner_loop" if profile == "inner_loop" else "smoke",
        "claimed_work_units": units,
        "verifier": "local_bounded_worker_report",
        "anti_cheat_status": "local_private_hive_only",
        "runtime_ms": int(report.get("runtime_ms") or 0),
        "maintenance_mode": maintenance_mode_from_payload(payload),
        "created_utc": now(),
    }


def settle_market_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    try:
        policy = compute_market.read_json(compute_market.POLICY_PATH, {})
        return compute_market.settle_receipt(
            receipt,
            context={"source": "hive_worker_chunk"},
            policy=policy,
            write_report=True,
        )
    except Exception as exc:  # noqa: BLE001 - accounting must never invalidate real worker work.
        return {"ok": False, "error": "compute_market_settlement_failed", "message": str(exc)}


def worker_review_step_count(payload: dict[str, Any], report: dict[str, Any]) -> int:
    steps = 2  # profile/policy validation plus bounded worker execution.
    if report.get("work_receipt"):
        steps += 1
    if report.get("compute_market"):
        steps += 1
    metrics = report.get("metrics")
    if isinstance(metrics, dict) and metrics:
        steps += 1
    if payload.get("orchestration"):
        steps += 1
    return steps


def maintenance_mode_from_payload(payload: dict[str, Any]) -> str:
    explicit = normalize_maintenance_mode(payload.get("maintenance_mode"))
    if explicit:
        return explicit
    orchestration = payload.get("orchestration") if isinstance(payload.get("orchestration"), dict) else {}
    explicit = normalize_maintenance_mode(orchestration.get("maintenance_mode"))
    return explicit or "object_only"


def normalize_maintenance_mode(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ordinary": "object_only",
        "ordinary_current": "object_only",
        "baseline": "object_only",
        "object": "object_only",
        "object_only": "object_only",
        "circle": "circle_seed_rule_rebuild",
        "circle_seed_rule": "circle_seed_rule_rebuild",
        "circle_seed_rule_rebuild": "circle_seed_rule_rebuild",
        "seed_rule_rebuild": "circle_seed_rule_rebuild",
    }
    return aliases.get(text, "")


def run_cuda_chunk(
    kind: str,
    payload: dict[str, Any],
    policy: dict[str, Any],
    profiles: dict[str, Any],
    profile_name: str,
    chunk_id: str,
) -> dict[str, Any]:
    if not query_nvidia().get("available"):
        return {"ok": False, "error": "cuda_unavailable", "nvidia": query_nvidia()}
    profile = get_path(profiles, ["profiles", profile_name], {})
    if not isinstance(profile, dict):
        return {"ok": False, "error": "profile_missing", "profile": profile_name}

    before = query_nvidia()
    rust_report = ROOT / "reports" / "hive_chunks" / f"{chunk_id}_rust.json"
    model_path = ROOT / "reports" / "hive_chunks" / f"{rust_report.stem}.model.json"
    command = build_cuda_command(kind, payload, policy, profile, rust_report, model_path)
    timeout = int(
        payload.get("timeout_seconds")
        or get_path(policy, ["worker_chunks", "max_runtime_seconds_by_kind", hive_task_kind_for_internal(kind)], 900)
    )
    started = time.perf_counter()
    child = run_with_gpu_sampling(command, timeout_seconds=timeout)
    if child.get("timed_out"):
        runtime_ms = int(child.get("runtime_ms") or ((time.perf_counter() - started) * 1000))
        return {
            "ok": False,
            "backend": "rust_cuda",
            "profile": validate_profile(payload, policy),
            "error": "cuda_chunk_timeout",
            "command": command,
            "returncode": 124,
            "timeout_seconds": timeout,
            "runtime_ms_child": runtime_ms,
            "stdout_tail": str(child.get("stdout") or "")[-4000:],
            "stderr_tail": str(child.get("stderr") or "")[-4000:],
            "telemetry": {
                "nvidia_before": before,
                "nvidia_after": query_nvidia(),
                "nvidia_during": summarize_gpu_samples(child.get("gpu_samples") if isinstance(child.get("gpu_samples"), list) else []),
                "child_report_path": str(rust_report.relative_to(ROOT)),
                "model_path": str(model_path.relative_to(ROOT)),
            },
            "metrics": {},
            "child_report": read_json(rust_report, {}),
        }
    runtime_ms = int(child.get("runtime_ms") or ((time.perf_counter() - started) * 1000))
    child_report = read_json(rust_report, {})
    return {
        "ok": int(child.get("returncode") or 0) == 0,
        "backend": "rust_cuda",
        "profile": validate_profile(payload, policy),
        "command": command,
        "returncode": int(child.get("returncode") or 0),
        "timeout_seconds": timeout,
        "runtime_ms_child": runtime_ms,
        "stdout_tail": str(child.get("stdout") or "")[-4000:],
        "stderr_tail": str(child.get("stderr") or "")[-4000:],
        "telemetry": {
            "nvidia_before": before,
            "nvidia_after": query_nvidia(),
            "nvidia_during": summarize_gpu_samples(child.get("gpu_samples") if isinstance(child.get("gpu_samples"), list) else []),
            "child_report_path": str(rust_report.relative_to(ROOT)),
            "model_path": str(model_path.relative_to(ROOT)),
        },
        "metrics": extract_metrics(child_report),
        "child_report": compact_child_report(child_report),
    }


def build_cuda_command(
    kind: str,
    payload: dict[str, Any],
    policy: dict[str, Any],
    profile: dict[str, Any],
    rust_report: Path,
    model_path: Path,
) -> list[str]:
    prefix = symliquid_prefix(needs_cuda=True)
    if kind in {"cuda_eval_chunk", "cuda_readout_train"}:
        readout = profile.get("cgs_cuda_readout") if isinstance(profile.get("cgs_cuda_readout"), dict) else {}
        rollout = profile.get("puffer_ocean_rollout_cuda") if isinstance(profile.get("puffer_ocean_rollout_cuda"), dict) else {}
        cases_default = 4 if kind == "cuda_eval_chunk" else int(readout.get("cases_per_task") or 64)
        epochs_default = 1 if kind == "cuda_eval_chunk" else int(readout.get("epochs") or 2)
        max_cases = int(get_path(policy, ["worker_chunks", "limits", "max_cuda_cases_per_task"], 256))
        max_epochs = int(get_path(policy, ["worker_chunks", "limits", "max_cuda_epochs"], 8))
        samples_default = int(readout.get("samples_per_launch") or rollout.get("samples_per_launch") or 256)
        hv_default = int(readout.get("hv_dim") or rollout.get("hv_dim") or 1024)
        return [
            *prefix,
            "train-standalone-cuda",
            "--train-seed",
            str(safe_int(payload, "train_seed", 0, 1_000_000_000)),
            "--eval-seed",
            str(safe_int(payload, "eval_seed", 10_000, 1_000_000_000)),
            "--cases-per-task",
            str(safe_int(payload, "cases_per_task", cases_default, max_cases)),
            "--epochs",
            str(safe_int(payload, "epochs", epochs_default, max_epochs)),
            "--samples-per-launch",
            str(safe_int(payload, "samples_per_launch", samples_default, 8192)),
            "--hv-dim",
            str(safe_int(payload, "hv_dim", hv_default, 8192)),
            "--lr",
            str(safe_float(payload, "lr", float(readout.get("lr") or 0.08), 1.0)),
            "--model-out",
            str(model_path.relative_to(ROOT)),
            "--out",
            str(rust_report.relative_to(ROOT)),
        ]

    rollout = profile.get("puffer_ocean_rollout_cuda") if isinstance(profile.get("puffer_ocean_rollout_cuda"), dict) else {}
    max_cases = int(get_path(policy, ["worker_chunks", "limits", "max_cuda_cases_per_task"], 256))
    max_epochs = int(get_path(policy, ["worker_chunks", "limits", "max_cuda_epochs"], 8))
    return [
        *prefix,
        "train-rollout-cuda",
        "--train-seed",
        str(safe_int(payload, "train_seed", 0, 1_000_000_000)),
        "--eval-seed",
        str(safe_int(payload, "eval_seed", 20_000, 1_000_000_000)),
        "--cases-per-task",
        str(safe_int(payload, "cases_per_task", int(rollout.get("cases_per_task") or 16), max_cases)),
        "--epochs",
        str(safe_int(payload, "epochs", int(rollout.get("epochs") or 1), max_epochs)),
        "--state-epochs",
        str(safe_int(payload, "state_epochs", int(rollout.get("state_epochs") or 1), max_epochs)),
        "--state-lr",
        str(safe_float(payload, "state_lr", float(rollout.get("state_lr") or 0.02), 1.0)),
        "--samples-per-launch",
        str(safe_int(payload, "samples_per_launch", int(rollout.get("samples_per_launch") or 256), 8192)),
        "--probe-cases-per-task",
        str(safe_int(payload, "probe_cases_per_task", int(rollout.get("probe_cases_per_task") or 4), max_cases)),
        "--rollout-batch",
        str(safe_int(payload, "rollout_batch", int(rollout.get("rollout_batch") or 16), 8192)),
        "--obs-dim",
        str(safe_int(payload, "obs_dim", int(rollout.get("obs_dim") or 16), 4096)),
        "--hidden-dim",
        str(safe_int(payload, "hidden_dim", int(rollout.get("hidden_dim") or 16), 4096)),
        "--reservoir-dim",
        str(safe_int(payload, "reservoir_dim", int(rollout.get("reservoir_dim") or 64), 16384)),
        "--hv-dim",
        str(safe_int(payload, "hv_dim", int(rollout.get("hv_dim") or 1024), 16384)),
        "--seq-len",
        str(safe_int(payload, "seq_len", int(rollout.get("seq_len") or 16), 1024)),
        "--lr",
        str(safe_float(payload, "lr", float(rollout.get("lr") or 0.05), 1.0)),
        "--model-out",
        str(model_path.relative_to(ROOT)),
        "--out",
        str(rust_report.relative_to(ROOT)),
    ]


def run_mlx_chunk(
    kind: str,
    payload: dict[str, Any],
    policy: dict[str, Any],
    profiles: dict[str, Any],
    profile_name: str,
    chunk_id: str,
) -> dict[str, Any]:
    module = str(get_path(policy, ["mac_support", "mlx_python_module"], "mlx.core"))
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - optional Mac backend.
        return {"ok": False, "error": "mlx_unavailable", "module": module, "message": str(exc)}

    profile = get_path(profiles, ["profiles", profile_name], {})
    babylm = profile.get("babylm") if isinstance(profile, dict) and isinstance(profile.get("babylm"), dict) else {}
    train_path = ROOT / str(payload.get("train_input") or "data/babylm_blimp_filtered_train.jsonl")
    eval_path = ROOT / str(payload.get("eval_input") or "data/babylm_mutated_holdout_seed55.jsonl")
    max_train = int(get_path(policy, ["worker_chunks", "limits", "max_mlx_train_rows"], 8192))
    max_eval = int(get_path(policy, ["worker_chunks", "limits", "max_mlx_eval_rows"], 4096))
    train_limit = safe_int(payload, "train_limit", int(babylm.get("train_limit") or 512), max_train)
    eval_limit = safe_int(payload, "eval_limit", int(babylm.get("eval_limit") or 256), max_eval)
    dim = safe_int(payload, "feature_dim", int(payload.get("hv_dim") or babylm.get("hv_dim") or 1024), 8192)
    default_steps = 1 if kind == "mlx_babylm_eval" else 24
    steps = safe_int(payload, "steps", default_steps, int(get_path(policy, ["worker_chunks", "limits", "max_mlx_steps"], 128)))
    lr = safe_float(payload, "lr", float(babylm.get("lr") or 0.05), 1.0)

    train_pairs = read_sentence_pairs(train_path, train_limit)
    eval_pairs = read_sentence_pairs(eval_path, eval_limit)
    if not train_pairs or not eval_pairs:
        return {
            "ok": False,
            "error": "missing_babylm_pairs",
            "train_input": str(train_path.relative_to(ROOT)) if train_path.exists() else str(train_path),
            "eval_input": str(eval_path.relative_to(ROOT)) if eval_path.exists() else str(eval_path),
        }

    started = time.perf_counter()
    feature_started = time.perf_counter()
    x_train, y_train, train_cache = pair_dataset_cached(train_path, train_pairs, train_limit, dim, policy)
    x_eval, y_eval, eval_cache = pair_dataset_cached(eval_path, eval_pairs, eval_limit, dim, policy)
    feature_ms = int((time.perf_counter() - feature_started) * 1000)
    transfer_started = time.perf_counter()
    x = mx.array(x_train, dtype=mx.float32)
    y = mx.array(y_train, dtype=mx.float32)
    xe = mx.array(x_eval, dtype=mx.float32)
    ye = mx.array(y_eval, dtype=mx.float32)
    mx.eval(x, y, xe, ye)
    mlx_transfer_ms = int((time.perf_counter() - transfer_started) * 1000)
    w = mx.zeros((dim,), dtype=mx.float32)
    b = mx.array(0.0, dtype=mx.float32)
    l2 = float(payload.get("l2") or 0.0001)
    losses: list[float] = []
    train_started = time.perf_counter()
    for _ in range(steps):
        logits = x @ w + b
        probs = 1.0 / (1.0 + mx.exp(-logits))
        err = probs - y
        grad_w = (x.T @ err) / max(1, len(y_train)) + l2 * w
        grad_b = mx.mean(err)
        w = w - lr * grad_w
        b = b - lr * grad_b
        loss = -mx.mean(y * mx.log(probs + 1e-6) + (1.0 - y) * mx.log(1.0 - probs + 1e-6))
        mx.eval(w, b, loss)
        losses.append(float(loss.item()))
    mlx_train_ms = int((time.perf_counter() - train_started) * 1000)
    eval_started = time.perf_counter()
    train_acc = binary_accuracy(mx, x, y, w, b)
    eval_acc = binary_accuracy(mx, xe, ye, w, b)
    mlx_eval_ms = int((time.perf_counter() - eval_started) * 1000)
    model_path = ROOT / "reports" / "hive_chunks" / f"{chunk_id}_mlx_linear.json"
    preview = w[: min(32, dim)]
    mx.eval(preview, b)
    write_json(
        model_path,
        {
            "policy": "project_theseus_mlx_linear_probe_v0",
            "created_utc": now(),
            "feature_dim": dim,
            "steps": steps,
            "train_accuracy": train_acc,
            "eval_accuracy": eval_acc,
            "weights_preview": [float(v) for v in preview.tolist()],
            "bias": float(b.item()),
            "feature_cache": {
                "train": train_cache,
                "eval": eval_cache,
            },
        },
    )
    expanded_train_examples = len(y_train)
    expanded_eval_examples = len(y_eval)
    return {
        "ok": True,
        "backend": mlx_backend_id(),
        "profile": profile_name,
        "module": module,
        "runtime_ms_child": int((time.perf_counter() - started) * 1000),
        "train_input": str(train_path.relative_to(ROOT)),
        "eval_input": str(eval_path.relative_to(ROOT)),
        "metrics": {
            "train_rows": len(train_pairs),
            "eval_rows": len(eval_pairs),
            "expanded_train_examples": expanded_train_examples,
            "expanded_eval_examples": expanded_eval_examples,
            "feature_dim": dim,
            "steps": steps,
            "train_accuracy": train_acc,
            "eval_accuracy": eval_acc,
            "loss_initial": losses[0] if losses else None,
            "loss_final": losses[-1] if losses else None,
            "feature_ms": feature_ms,
            "mlx_transfer_ms": mlx_transfer_ms,
            "mlx_train_ms": mlx_train_ms,
            "mlx_eval_ms": mlx_eval_ms,
            "examples_per_second": (expanded_train_examples * max(1, steps))
            / max(0.001, mlx_train_ms / 1000.0),
            "cache_hits": int(bool(train_cache.get("hit"))) + int(bool(eval_cache.get("hit"))),
        },
        "telemetry": {
            "model_path": str(model_path.relative_to(ROOT)),
            "mlx_platform": platform.platform(),
            "feature_cache": {
                "train": train_cache,
                "eval": eval_cache,
            },
        },
    }


def run_mlx_rollout_chunk(
    kind: str,
    payload: dict[str, Any],
    policy: dict[str, Any],
    profiles: dict[str, Any],
    profile_name: str,
    chunk_id: str,
) -> dict[str, Any]:
    module = str(get_path(policy, ["mac_support", "mlx_python_module"], "mlx.core"))
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - optional Mac backend.
        return {"ok": False, "error": "mlx_unavailable", "module": module, "message": str(exc)}

    profile = get_path(profiles, ["profiles", profile_name], {})
    rollout = profile.get("puffer_ocean_rollout_cuda") if isinstance(profile, dict) and isinstance(profile.get("puffer_ocean_rollout_cuda"), dict) else {}
    max_cases = int(get_path(policy, ["worker_chunks", "limits", "max_mlx_rollout_cases"], get_path(policy, ["worker_chunks", "limits", "max_mlx_train_rows"], 8192)))
    max_steps = int(get_path(policy, ["worker_chunks", "limits", "max_mlx_steps"], 128))
    scale_cases = int(rollout.get("cases_per_task") or 64)
    train_cases = max(1, safe_int(payload, "cases_per_task", scale_cases, max_cases))
    eval_cases = max(1, safe_int(payload, "eval_cases", max(32, train_cases), max_cases))
    seq_len = max(1, safe_int(payload, "seq_len", int(rollout.get("seq_len") or 32), 512))
    obs_dim = max(1, safe_int(payload, "obs_dim", int(rollout.get("obs_dim") or 32), 1024))
    dim = max(1, safe_int(payload, "hv_dim", int(rollout.get("hv_dim") or payload.get("feature_dim") or 1024), 8192))
    steps = max(1, safe_int(payload, "epochs", int(rollout.get("epochs") or 6), max_steps))
    lr = safe_float(payload, "lr", float(rollout.get("lr") or 0.04), 1.0)
    l2 = safe_float(payload, "l2", 0.0001, 1.0)
    train_seed = safe_int(payload, "train_seed", 0, 1_000_000_000)
    eval_seed = safe_int(payload, "eval_seed", 20_000, 1_000_000_000)

    started = time.perf_counter()
    dataset_started = time.perf_counter()
    x_train, y_train, train_meta = mlx_rollout_dataset(train_cases, seq_len, obs_dim, dim, train_seed)
    x_eval, y_eval, eval_meta = mlx_rollout_dataset(eval_cases, seq_len, obs_dim, dim, eval_seed)
    dataset_ms = int((time.perf_counter() - dataset_started) * 1000)

    transfer_started = time.perf_counter()
    x = mx.array(x_train, dtype=mx.float32)
    y = mx.array(y_train, dtype=mx.float32)
    xe = mx.array(x_eval, dtype=mx.float32)
    ye = mx.array(y_eval, dtype=mx.float32)
    mx.eval(x, y, xe, ye)
    mlx_transfer_ms = int((time.perf_counter() - transfer_started) * 1000)

    w = mx.zeros((dim,), dtype=mx.float32)
    b = mx.array(0.0, dtype=mx.float32)
    losses: list[float] = []
    train_started = time.perf_counter()
    for _ in range(steps):
        logits = x @ w + b
        probs = 1.0 / (1.0 + mx.exp(-logits))
        err = probs - y
        grad_w = (x.T @ err) / max(1, train_cases) + l2 * w
        grad_b = mx.mean(err)
        w = w - lr * grad_w
        b = b - lr * grad_b
        loss = -mx.mean(y * mx.log(probs + 1e-6) + (1.0 - y) * mx.log(1.0 - probs + 1e-6))
        mx.eval(w, b, loss)
        losses.append(float(loss.item()))
    mlx_train_ms = int((time.perf_counter() - train_started) * 1000)

    eval_started = time.perf_counter()
    train_acc = binary_accuracy(mx, x, y, w, b)
    eval_acc = binary_accuracy(mx, xe, ye, w, b)
    train_return = rollout_return_proxy(train_acc)
    eval_return = rollout_return_proxy(eval_acc)
    mlx_eval_ms = int((time.perf_counter() - eval_started) * 1000)

    model_path = ROOT / "reports" / "hive_chunks" / f"{chunk_id}_mlx_rollout_linear.json"
    preview = w[: min(32, dim)]
    mx.eval(preview, b)
    write_json(
        model_path,
        {
            "policy": "project_theseus_mlx_rollout_probe_v0",
            "created_utc": now(),
            "feature_dim": dim,
            "seq_len": seq_len,
            "obs_dim": obs_dim,
            "steps": steps,
            "train_accuracy": train_acc,
            "eval_accuracy": eval_acc,
            "train_return_proxy": train_return,
            "eval_return_proxy": eval_return,
            "weights_preview": [float(v) for v in preview.tolist()],
            "bias": float(b.item()),
            "dataset": {
                "train": train_meta,
                "eval": eval_meta,
            },
        },
    )

    return {
        "ok": True,
        "backend": mlx_backend_id(),
        "profile": profile_name,
        "module": module,
        "runtime_ms_child": int((time.perf_counter() - started) * 1000),
        "metrics": {
            "train_cases": train_cases,
            "eval_cases": eval_cases,
            "expanded_train_examples": train_cases,
            "expanded_eval_examples": eval_cases,
            "feature_dim": dim,
            "obs_dim": obs_dim,
            "seq_len": seq_len,
            "steps": steps,
            "train_accuracy": train_acc,
            "eval_accuracy": eval_acc,
            "train_return_proxy": train_return,
            "eval_return_proxy": eval_return,
            "loss_initial": losses[0] if losses else None,
            "loss_final": losses[-1] if losses else None,
            "feature_ms": dataset_ms,
            "mlx_transfer_ms": mlx_transfer_ms,
            "mlx_train_ms": mlx_train_ms,
            "mlx_eval_ms": mlx_eval_ms,
            "examples_per_second": (train_cases * max(1, steps)) / max(0.001, mlx_train_ms / 1000.0),
        },
        "telemetry": {
            "model_path": str(model_path.relative_to(ROOT)),
            "mlx_platform": platform.platform(),
            "synthetic_control_task": "binary_action_from_sequence_rollout_v0",
            "dataset": {
                "train": train_meta,
                "eval": eval_meta,
            },
        },
    }


def mlx_rollout_dataset(cases: int, seq_len: int, obs_dim: int, dim: int, seed: int) -> tuple[Any, Any, dict[str, Any]]:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - numpy is a speed path, not a correctness dependency.
        return mlx_rollout_dataset_py(cases, seq_len, obs_dim, dim, seed)

    rng = np.random.default_rng(seed)
    teacher = rng.normal(0.0, 1.0, size=(obs_dim,)).astype(np.float32)
    projection = rng.normal(0.0, 1.0 / max(1.0, math.sqrt(obs_dim)), size=(obs_dim, dim)).astype(np.float32)
    observations = rng.normal(0.0, 1.0, size=(cases, seq_len, obs_dim)).astype(np.float32)
    time_weights = np.linspace(0.35, 1.0, seq_len, dtype=np.float32)
    action_signal = (observations @ teacher * time_weights[None, :]).sum(axis=1)
    y = (action_signal > 0.0).astype(np.float32)
    projected = np.tanh(observations @ projection)
    recency = time_weights / max(float(time_weights.sum()), 1e-6)
    x = (projected * recency[None, :, None]).sum(axis=1).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-6)
    return x, y, {
        "generator": "numpy_seeded_synthetic_rollout_v0",
        "cases": cases,
        "seq_len": seq_len,
        "obs_dim": obs_dim,
        "feature_dim": dim,
        "seed": seed,
        "positive_fraction": float(y.mean()) if cases else 0.0,
    }


def mlx_rollout_dataset_py(cases: int, seq_len: int, obs_dim: int, dim: int, seed: int) -> tuple[list[list[float]], list[float], dict[str, Any]]:
    import random

    rng = random.Random(seed)
    teacher = [rng.gauss(0.0, 1.0) for _ in range(obs_dim)]
    projection = [[rng.gauss(0.0, 1.0 / max(1.0, math.sqrt(obs_dim))) for _ in range(dim)] for _ in range(obs_dim)]
    time_weights = [0.35 + (0.65 * step / max(1, seq_len - 1)) for step in range(seq_len)]
    norm_time = sum(time_weights) or 1.0
    features: list[list[float]] = []
    labels: list[float] = []
    for _ in range(cases):
        row = [0.0] * dim
        signal = 0.0
        for step in range(seq_len):
            obs = [rng.gauss(0.0, 1.0) for _ in range(obs_dim)]
            weight = time_weights[step]
            signal += sum(obs[i] * teacher[i] for i in range(obs_dim)) * weight
            for j in range(dim):
                row[j] += math.tanh(sum(obs[i] * projection[i][j] for i in range(obs_dim))) * (weight / norm_time)
        norm = math.sqrt(sum(value * value for value in row)) or 1.0
        features.append([value / norm for value in row])
        labels.append(1.0 if signal > 0.0 else 0.0)
    positives = sum(labels)
    return features, labels, {
        "generator": "python_seeded_synthetic_rollout_v0",
        "cases": cases,
        "seq_len": seq_len,
        "obs_dim": obs_dim,
        "feature_dim": dim,
        "seed": seed,
        "positive_fraction": positives / max(1, cases),
    }


def rollout_return_proxy(accuracy: float) -> float:
    return max(-1.0, min(1.0, (float(accuracy) - 0.5) * 2.0))


def binary_accuracy(mx: Any, x: Any, y: Any, w: Any, b: Any) -> float:
    logits = x @ w + b
    pred = logits > 0
    acc = mx.mean(pred == (y > 0.5))
    mx.eval(acc)
    return float(acc.item())


def read_sentence_pairs(path: Path, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            good = str(value.get("sentence_good") or value.get("good") or "")
            bad = str(value.get("sentence_bad") or value.get("bad") or "")
            if good and bad:
                rows.append({"good": good, "bad": bad, "rule": str(value.get("rule") or "")})
    return rows


def pair_dataset_cached(
    path: Path,
    rows: list[dict[str, str]],
    limit: int,
    dim: int,
    policy: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    cache_cfg = get_path(policy, ["mac_support", "mlx_feature_cache"], {})
    if cache_cfg == {}:
        cache_cfg = get_path(policy, ["worker_chunks", "mlx_feature_cache"], {})
    enabled = bool(cache_cfg.get("enabled", True))
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - numpy is a speed path, not a correctness dependency.
        features, labels = pair_dataset(rows, dim)
        return features, labels, {"enabled": False, "hit": False, "reason": "numpy_unavailable"}
    cache_path = mlx_feature_cache_path(path, limit, dim, cache_cfg)
    if enabled and cache_path.exists():
        try:
            loaded = np.load(cache_path)
            return loaded["x"], loaded["y"], {
                "enabled": True,
                "hit": True,
                "path": str(cache_path.relative_to(ROOT)),
                "rows": int(loaded["y"].shape[0]),
            }
        except Exception as exc:  # noqa: BLE001 - corrupt cache should rebuild.
            cache_error = str(exc)
    else:
        cache_error = ""

    features, labels = pair_dataset_np(rows, dim, np)
    if enabled:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, x=features, y=labels)
    return features, labels, {
        "enabled": enabled,
        "hit": False,
        "path": str(cache_path.relative_to(ROOT)) if enabled else "",
        "rows": int(labels.shape[0]),
        "rebuild_reason": cache_error or "miss",
    }


def mlx_feature_cache_path(path: Path, limit: int, dim: int, cache_cfg: dict[str, Any]) -> Path:
    root = ROOT / str(cache_cfg.get("root") or "reports/hive_chunks/mlx_feature_cache")
    try:
        stat = path.stat()
        mtime_ns = stat.st_mtime_ns
        size = stat.st_size
    except OSError:
        mtime_ns = 0
        size = 0
    key = hashlib.blake2b(
        f"{path.resolve()}|{mtime_ns}|{size}|{limit}|{dim}|text_pair_v2".encode("utf-8"),
        digest_size=16,
    ).hexdigest()
    return root / f"{key}.npz"


def pair_dataset_np(rows: list[dict[str, str]], dim: int, np: Any) -> tuple[Any, Any]:
    features = np.zeros((len(rows) * 2, dim), dtype=np.float32)
    labels = np.zeros((len(rows) * 2,), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        good = text_features_np(row["good"], dim, np)
        bad = text_features_np(row["bad"], dim, np)
        diff = good - bad
        out = row_idx * 2
        features[out, :] = diff
        labels[out] = 1.0
        features[out + 1, :] = -diff
        labels[out + 1] = 0.0
    return features, labels


def text_features_np(text: str, dim: int, np: Any) -> Any:
    values = np.zeros((dim,), dtype=np.float32)
    normalized = " ".join(text.lower().strip().split())
    tokens = normalized.split()
    grams = tokens + [normalized[i : i + 3] for i in range(max(0, len(normalized) - 2))]
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8", errors="ignore"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        values[bucket] += sign
    norm = float(np.linalg.norm(values)) or 1.0
    values /= norm
    return values


def pair_dataset(rows: list[dict[str, str]], dim: int) -> tuple[list[list[float]], list[float]]:
    features: list[list[float]] = []
    labels: list[float] = []
    for row in rows:
        good = text_features(row["good"], dim)
        bad = text_features(row["bad"], dim)
        diff = [g - b for g, b in zip(good, bad)]
        reverse = [-value for value in diff]
        features.append(diff)
        labels.append(1.0)
        features.append(reverse)
        labels.append(0.0)
    return features, labels


def text_features(text: str, dim: int) -> list[float]:
    values = [0.0] * dim
    normalized = " ".join(text.lower().strip().split())
    tokens = normalized.split()
    grams = tokens + [normalized[i : i + 3] for i in range(max(0, len(normalized) - 2))]
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8", errors="ignore"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        values[bucket] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def symliquid_prefix(*, needs_cuda: bool = False) -> list[str]:
    exe_name = "symliquid-cli.exe" if platform.system() == "Windows" else "symliquid-cli"
    target_dir = os.environ.get("CARGO_TARGET_DIR")
    candidates = []
    if target_dir:
        candidates.append(Path(target_dir) / "release" / exe_name)
    runtime_target = theseus_runtime.runtime_report(create=True).get("paths", {}).get("cargo_target_dir", {}).get("path")
    if runtime_target:
        candidates.append(Path(runtime_target) / "release" / exe_name)
    candidates.append(ROOT / "target" / "release" / exe_name)
    exe = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if exe.exists():
        return [str(exe)]
    if needs_cuda and shutil.which("cargo"):
        return ["cargo", "run", "--release", "-p", "symliquid-cli", "--features", "cuda", "--"]
    command = ["cargo", "run", "--release", "-p", "symliquid-cli"]
    if platform.system() == "Windows":
        command.extend(["--features", "cuda"])
    command.append("--")
    return command


def mlx_backend_id() -> str:
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return "mlx_apple"
    if platform.system() == "Linux" and query_nvidia().get("available"):
        return "mlx_cuda"
    return "mlx"


def validate_profile(payload: dict[str, Any], policy: dict[str, Any]) -> str:
    profile = str(payload.get("profile") or get_path(policy, ["worker_chunks", "default_profile"], "smoke"))
    allowed = get_path(policy, ["worker_chunks", "allowed_profiles"], ["smoke"])
    if profile not in set(allowed):
        raise ValueError(f"profile {profile!r} is not allowed for Hive worker chunks")
    return profile


def hive_task_kind_for_internal(kind: str) -> str:
    return {
        "cuda_eval_chunk": "cuda_eval_chunk",
        "cuda_readout_train": "cuda_training_chunk",
        "cuda_rollout_train": "cuda_rollout_chunk",
        "mlx_babylm_eval": "mlx_eval_chunk",
        "mlx_babylm_train": "mlx_training_chunk",
        "mlx_rollout_probe": "mlx_rollout_chunk",
    }.get(kind, kind)


def job_metadata(kind: str, payload: dict[str, Any], chunk_id: str) -> dict[str, Any]:
    task_kind = hive_task_kind_for_internal(kind)
    return {
        "policy": "project_theseus_hive_worker_job_metadata_v0",
        "job_id": str(payload.get("job_id") or chunk_id),
        "job_family": str(payload.get("job_family") or kind),
        "arm_id": str(payload.get("arm_id") or default_arm_for_internal(kind)),
        "task_kind": task_kind,
        "worker_kind": kind,
        "backend_requirements": payload.get("backend_requirements") if isinstance(payload.get("backend_requirements"), list) else [],
        "merge_policy": str(payload.get("merge_policy") or "append_report_then_gate"),
        "priority": safe_int(payload, "priority", 50, 1000),
        "lease_seconds": safe_int(payload, "lease_seconds", 1800, 86400),
        "output_artifacts": payload.get("output_artifacts") if isinstance(payload.get("output_artifacts"), list) else [],
    }


def default_arm_for_internal(kind: str) -> str:
    if kind.startswith("cuda_"):
        return "rust_cuda_systems_arm"
    if kind.startswith("mlx_rollout"):
        return "apple_mlx_control_arm"
    if kind.startswith("mlx_"):
        return "apple_mlx_worker_arm"
    return "hive_worker_arm"


def requester_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "requester_node_id": str(payload.get("requester_node_id") or ""),
        "requester_node_name": str(payload.get("requester_node_name") or ""),
        "source": str(payload.get("source") or payload.get("reason") or ""),
    }


def orchestration_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("orchestration") if isinstance(payload.get("orchestration"), dict) else {}
    if not value:
        return {}
    allowed = {
        "policy",
        "run_id",
        "round_id",
        "lease_id",
        "lease_expires_utc",
        "arm_id",
        "arm_display_name",
        "owner_node_id",
        "owner_node_name",
        "slot_type",
        "strategy",
        "artifact_flow",
        "input_artifacts",
        "maintenance_mode",
    }
    return {key: value.get(key) for key in allowed if key in value}


def extract_metrics(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    metrics: dict[str, Any] = {}
    for key in [
        "train_accuracy",
        "eval_accuracy",
        "accuracy",
        "examples_per_second",
        "train_examples_per_second",
        "eval_examples_per_second",
        "train_runtime_ms",
        "eval_runtime_ms",
        "runtime_ms",
        "runtime_ms_child",
        "kernel_launches",
        "cuda_fallback",
        "total_ms",
        "kernel_ms",
        "h2d_ms",
        "d2h_ms",
        "feature_ms",
        "mlx_transfer_ms",
        "mlx_train_ms",
        "mlx_eval_ms",
        "cache_hits",
    ]:
        if key in report:
            metrics[key] = report[key]
    nested = report.get("metrics")
    if isinstance(nested, dict):
        metrics.update(nested)
    telemetry = report.get("telemetry")
    if isinstance(telemetry, dict):
        for key in [
            "examples_per_second",
            "train_examples_per_second",
            "kernel_launches",
            "cuda_fallback",
            "feature_ms",
            "mlx_transfer_ms",
            "mlx_train_ms",
            "mlx_eval_ms",
            "cache_hits",
        ]:
            if key in telemetry:
                metrics[key] = telemetry[key]
    return metrics


def compact_child_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    keep = {
        "policy",
        "created_utc",
        "ok",
        "train_accuracy",
        "eval_accuracy",
        "accuracy",
        "examples_per_second",
        "train_examples_per_second",
        "eval_examples_per_second",
        "train_runtime_ms",
        "eval_runtime_ms",
        "runtime_ms",
        "runtime_ms_child",
        "kernel_launches",
        "cuda_fallback",
        "timing_breakdown_ms",
        "runtime_profile",
        "metrics",
        "telemetry",
        "profile",
        "task",
    }
    return {key: value for key, value in report.items() if key in keep}


def query_nvidia() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi_not_found"}
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip() or "nvidia_smi_failed"}
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 6:
            gpus.append(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total_mib": to_float(parts[2]),
                    "memory_used_mib": to_float(parts[3]),
                    "memory_free_mib": to_float(parts[4]),
                    "utilization_gpu_percent": to_float(parts[5]),
                    "utilization_memory_percent": to_float(parts[6]) if len(parts) > 6 else None,
                    "power_draw_w": to_float(parts[7]) if len(parts) > 7 else None,
                    "temperature_c": to_float(parts[8]) if len(parts) > 8 else None,
                }
            )
    return {"available": bool(gpus), "gpus": gpus}


def run_with_gpu_sampling(command: list[str], *, timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    try:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=theseus_runtime.runtime_env(),
        )
    except OSError as exc:
        return {
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "gpu_samples": samples,
            "timed_out": False,
        }

    deadline = started + max(1, timeout_seconds)
    timed_out = False
    while proc.poll() is None:
        samples.append(query_nvidia())
        if time.perf_counter() >= deadline:
            timed_out = True
            proc.kill()
            break
        time.sleep(0.25)
    try:
        stdout, stderr = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        timed_out = True
    return {
        "returncode": proc.returncode if proc.returncode is not None else 124,
        "stdout": stdout or "",
        "stderr": stderr or "",
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "gpu_samples": samples,
        "timed_out": timed_out,
    }


def summarize_gpu_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        gpu
        for sample in samples
        if isinstance(sample, dict)
        for gpu in sample.get("gpus", [])
        if isinstance(gpu, dict)
    ]
    if not rows:
        return {"sample_count": 0}

    def values(key: str) -> list[float]:
        out = []
        for row in rows:
            try:
                if row.get(key) is not None:
                    out.append(float(row[key]))
            except (TypeError, ValueError):
                pass
        return out

    util = values("utilization_gpu_percent")
    mem_util = values("utilization_memory_percent")
    used = values("memory_used_mib")
    power = values("power_draw_w")
    temp = values("temperature_c")
    return {
        "sample_count": len(samples),
        "gpu_sample_count": len(rows),
        "max_gpu_utilization_percent": max(util) if util else None,
        "avg_gpu_utilization_percent": round(sum(util) / len(util), 2) if util else None,
        "max_memory_utilization_percent": max(mem_util) if mem_util else None,
        "max_memory_used_mib": max(used) if used else None,
        "max_power_draw_w": max(power) if power else None,
        "max_temperature_c": max(temp) if temp else None,
    }


def platform_report() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "executable": sys.executable,
    }


def safe_report_path(path: Path) -> Path:
    full = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    reports = (ROOT / "reports").resolve()
    try:
        full.relative_to(reports)
    except ValueError as exc:
        raise ValueError("worker chunk reports must be written under reports/") from exc
    full.parent.mkdir(parents=True, exist_ok=True)
    return full


def project_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        pass
    reports = (ROOT / "reports").resolve()
    try:
        return ("reports/" + str(resolved.relative_to(reports))).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return cleaned[:96] or "chunk"


def safe_int(payload: dict[str, Any], key: str, default: int, max_value: int) -> int:
    try:
        value = int(payload.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(0, min(value, max_value))


def safe_float(payload: dict[str, Any], key: str, default: float, max_value: float) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return max(0.0, min(value, max_value))


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_json(raw: str, default: Any) -> Any:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return default
    return value if isinstance(value, dict) else default


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
