#!/usr/bin/env python3
"""Apple Silicon MLX smoke for the structural-action decoder lane.

The main comparator keeps Torch as the matched cross-platform path. This smoke
proves that the same private structural-action training surface can run through
MLX on Apple Silicon: visible private task fields -> action-sequence labels ->
MLX classifier -> line-action compiler -> private verifier.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import time
from collections import Counter
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
    build_vocab,
    deterministic_sample,
    dict_or_empty,
    encode_many,
    get_path,
    load_private_rows,
    syntax_summary,
)
from neural_seed_structural_action_decoder_probe import (  # noqa: E402
    action_sequence_id,
    build_action_sequence_library,
    structural_class_compatibility,
    structural_candidate_rows,
    structural_source_text,
)


DEFAULT_CONFIG = ROOT / "reports" / "neural_seed_token_decoder_structural_integrated_smoke_config.json"
DEFAULT_OUT = ROOT / "reports" / "macos_mlx_structural_action_smoke.json"
DEFAULT_MD = ROOT / "reports" / "macos_mlx_structural_action_smoke.md"
DEFAULT_CANDIDATES = ROOT / "reports" / "macos_mlx_structural_action_smoke_candidates.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MD.relative_to(ROOT)))
    parser.add_argument("--candidate-manifest-out", default=str(DEFAULT_CANDIDATES.relative_to(ROOT)))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-train-rows", type=int, default=4096)
    parser.add_argument("--max-eval-rows", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--fanout-top-k", type=int, default=4)
    parser.add_argument("--rank-pool-size", type=int, default=32)
    parser.add_argument("--compatibility-rerank", choices=["on", "off"], default="on")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    config = read_json(resolve(args.config))
    if not args.execute:
        report = planned_report(config, args)
    else:
        report = run_smoke(config, args, started=started)
    candidate_rows = report.pop("_candidate_rows", [])
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    write_jsonl(resolve(args.candidate_manifest_out), candidate_rows)
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def run_smoke(config: dict[str, Any], args: argparse.Namespace, *, started: float) -> dict[str, Any]:
    if platform.machine() != "arm64":
        return blocked_report(args, started, "not_apple_silicon_arm64", trigger_state="YELLOW")
    preflight = mlx_import_preflight(timeout_seconds=8)
    if not preflight.get("ok"):
        return blocked_report(args, started, f"mlx_preflight_failed:{preflight}", trigger_state="YELLOW")
    try:
        import mlx  # type: ignore
        import mlx.core as mx  # type: ignore
    except Exception as exc:  # pragma: no cover - local environment gate
        return blocked_report(args, started, f"mlx_import_failed:{type(exc).__name__}:{exc}", trigger_state="YELLOW")

    random.seed(int(args.seed))
    mx.random.seed(int(args.seed))
    data_cfg = dict_or_empty(config.get("data"))
    text_views = dict_or_empty(config.get("text_views"))
    train_all = load_private_rows(resolve(str(data_cfg.get("train_jsonl") or "")), data_cfg)
    eval_all = load_private_rows(resolve(str(data_cfg.get("eval_jsonl") or "")), data_cfg)
    train_rows = deterministic_sample(train_all, int(args.max_train_rows), int(args.seed))
    eval_rows = deterministic_sample(eval_all, int(args.max_eval_rows), int(args.seed) + 1009)

    library = build_action_sequence_library(train_rows)
    classes = list(library.get("classes") or [])
    class_index = {str(row.get("sequence_id") or ""): idx for idx, row in enumerate(classes)}
    labeled_train: list[dict[str, Any]] = []
    labels: list[int] = []
    for row in train_rows:
        label = action_sequence_id(row)
        if label in class_index:
            labeled_train.append(row)
            labels.append(class_index[label])
    if len(classes) < 2 or not labeled_train:
        return blocked_report(args, started, "structural_class_count_below_minimum")

    source_fields = list(text_views.get("sts_on") or [])
    max_source = int(get_path(config, ["matched_budget", "max_source_tokens"], 96) or 96)
    max_vocab = int(get_path(config, ["matched_budget", "max_source_vocab"], 1024) or 1024)
    source_vocab = build_vocab([structural_source_text(row, source_fields) for row in labeled_train], max_vocab=max_vocab)
    train_x_ids = encode_many([structural_source_text(row, source_fields) for row in labeled_train], source_vocab, max_source)
    eval_x_ids = encode_many([structural_source_text(row, source_fields) for row in eval_rows], source_vocab, max_source)
    train_x = mx.array(train_x_ids)
    eval_x = mx.array(eval_x_ids)
    train_y = mx.array(labels)

    train_started = time.perf_counter()
    E, W, b, losses = train_mlx_embedding_classifier(
        mx,
        train_x,
        train_y,
        vocab_size=len(source_vocab),
        class_count=len(classes),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
    )
    logits = mlx_embedding_logits(mx, eval_x, E, W, b)
    probs = softmax(mx, logits)
    mx.eval(probs)
    proposals = proposals_from_probs(
        probs.tolist(),
        classes,
        eval_rows=eval_rows,
        fanout_top_k=int(args.fanout_top_k),
        rank_pool_size=int(args.rank_pool_size),
        compatibility_rerank=str(args.compatibility_rerank) == "on",
    )
    candidate_rows = structural_candidate_rows(
        eval_rows,
        proposals,
        arm_id="symliquid_style",
        substrate="mlx_mean_embedding_structural_action_classifier",
        config=config,
        seed=int(args.seed),
    )
    for row in candidate_rows:
        row["candidate_source"] = "macos_mlx_structural_action_smoke"
        row["integrated_candidate_family"] = "structural_action_sequence"
        row["provenance"]["view"] = "mlx_structural_action_smoke"
        row["provenance"]["candidate_family"] = "structural_action_sequence"
        row["body_structure_decode"]["mlx_smoke"] = True
    private_eval = evaluate_private_candidates(eval_rows, candidate_rows)
    syntax = syntax_summary(candidate_rows)
    fallback_rows = sum(1 for row in candidate_rows if get_path(row, ["grammar_repair", "fallback_return_used"], False))
    verifier_pass_rate = float(private_eval.get("trained_pass_rate") or 0.0)
    loss_decreased = bool(losses and losses[-1] < losses[0])
    gates = [
        gate("apple_silicon_arm64", platform.machine() == "arm64", platform.machine(), "hard"),
        gate("mlx_imported", True, {"mlx_module": getattr(mlx, "__name__", "mlx")}, "hard"),
        gate("mlx_gpu_default_device", "gpu" in str(mx.default_device()).lower(), str(mx.default_device()), "soft"),
        gate("mlx_training_loss_decreased", loss_decreased, {"loss_start": losses[0] if losses else None, "loss_end": losses[-1] if losses else None}, "hard"),
        gate("private_train_rows_loaded", len(labeled_train) > 0, {"train_rows": len(labeled_train)}, "hard"),
        gate("private_eval_rows_loaded", len(eval_rows) > 0, {"eval_rows": len(eval_rows)}, "hard"),
        gate("structural_classes_present", len(classes) > 1, {"class_count": len(classes)}, "hard"),
        gate("candidate_rows_written", len(candidate_rows) > 0, {"candidate_rows": len(candidate_rows)}, "hard"),
        gate("syntax_pass_rate_nonzero", float(syntax.get("syntax_pass_rate") or 0.0) > 0.0, syntax, "hard"),
        gate("fallback_return_zero", fallback_rows == 0, {"fallback_rows": fallback_rows}, "hard"),
        gate("private_verifier_pass_rate_nonzero", verifier_pass_rate > 0.0, {"verifier_pass_rate": verifier_pass_rate}, "hard"),
        gate("external_inference_zero", True, 0, "hard"),
        gate("public_training_rows_zero", True, 0, "hard"),
        gate("teacher_calls_zero", True, 0, "hard"),
    ]
    hard_pass = all(row["passed"] for row in gates if row["severity"] == "hard")
    trigger_state = "GREEN" if hard_pass else "RED"
    if trigger_state == "GREEN" and any(not row["passed"] for row in gates):
        trigger_state = "YELLOW"
    action_counts = Counter(str(row.get("structural_sequence_id") or "") for row in candidate_rows)
    return {
        "policy": "project_theseus_macos_mlx_structural_action_smoke_v0",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "config": args.config,
        "execute": True,
        "summary": {
            "mlx_available": True,
            "mlx_used": True,
            "mlx_default_device": str(mx.default_device()),
            "train_rows": len(labeled_train),
            "eval_rows": len(eval_rows),
            "source_vocab_size": len(source_vocab),
            "structural_action_class_count": len(classes),
            "candidate_rows": len(candidate_rows),
            "syntax_pass_rate": syntax.get("syntax_pass_rate"),
            "fallback_return_rows": fallback_rows,
            "verifier_pass_rate": verifier_pass_rate,
            "verifier_candidate_attempt_count": get_path(private_eval, ["private_verification", "candidate_attempt_count"], 0),
            "train_wall_time_ms": int((time.perf_counter() - train_started) * 1000),
            "rank_pool_size": int(args.rank_pool_size),
            "compatibility_rerank": str(args.compatibility_rerank),
            "external_inference_calls": 0,
            "teacher_used": False,
            "public_training_rows": 0,
            "model_promotion_allowed": False,
        },
        "training": {
            "model": "mlx_mean_embedding_structural_action_classifier",
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "loss_curve": losses,
            "score_semantics": "MLX smoke only; not a replacement for the matched Torch comparator.",
        },
        "structural_action_library": library.get("summary", {}),
        "candidate_sequence_counts": dict(action_counts.most_common(12)),
        "private_verifier": private_eval,
        "candidate_syntax": syntax,
        "gates": gates,
        "score_semantics": (
            "Apple Silicon MLX smoke over the same private structural-action training surface. It uses visible "
            "private fields and private-train structural action labels, compiles predicted line actions, and scores "
            "through the private verifier. No public calibration, public training rows, teacher call, external "
            "inference, fallback return, or model promotion occurred."
        ),
        "_candidate_rows": candidate_rows,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def train_mlx_embedding_classifier(
    mx: Any,
    x: Any,
    y: Any,
    *,
    vocab_size: int,
    class_count: int,
    epochs: int,
    learning_rate: float,
) -> tuple[Any, Any, Any, list[float]]:
    d_model = 64
    E = mx.random.normal((vocab_size, d_model)) * 0.02
    W = mx.random.normal((d_model, class_count)) * 0.02
    b = mx.zeros((class_count,))

    def loss_fn(E: Any, W: Any, b: Any) -> Any:
        logits = mlx_embedding_logits(mx, x, E, W, b)
        log_probs = logits - mx.logsumexp(logits, axis=1, keepdims=True)
        return mx.mean(-log_probs[mx.arange(y.shape[0]), y])

    value_and_grad = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))
    losses: list[float] = []
    for _epoch in range(max(1, int(epochs))):
        loss, grads = value_and_grad(E, W, b)
        E = E - float(learning_rate) * grads[0]
        W = W - float(learning_rate) * grads[1]
        b = b - float(learning_rate) * grads[2]
        mx.eval(E, W, b, loss)
        losses.append(round(float(loss), 6))
    return E, W, b, losses


def mlx_import_preflight(*, timeout_seconds: int) -> dict[str, Any]:
    """Probe MLX in a child process because native Metal failures can abort."""
    code = (
        "import json\n"
        "import mlx.core as mx\n"
        "print(json.dumps({'default_device': str(mx.default_device())}))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_seconds)),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "reason": "timeout",
            "stdout_tail": (exc.stdout or "")[-500:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-500:] if isinstance(exc.stderr, str) else str(exc)[-500:],
        }
    payload: dict[str, Any] = {}
    if result.stdout.strip():
        try:
            payload = json.loads(result.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            payload = {"stdout_tail": result.stdout[-500:]}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "payload": payload,
        "stdout_tail": (result.stdout or "")[-500:],
        "stderr_tail": (result.stderr or "")[-500:],
    }


def mlx_embedding_logits(mx: Any, x: Any, E: Any, W: Any, b: Any) -> Any:
    emb = E[x]
    mask = (x != 0).astype(mx.float32)
    pooled = mx.sum(emb * mask[:, :, None], axis=1) / mx.maximum(mx.sum(mask, axis=1, keepdims=True), 1.0)
    return mx.tanh(pooled) @ W + b


def softmax(mx: Any, logits: Any) -> Any:
    shifted = logits - mx.max(logits, axis=1, keepdims=True)
    exps = mx.exp(shifted)
    return exps / mx.sum(exps, axis=1, keepdims=True)


def proposals_from_probs(
    probs: list[list[float]],
    classes: list[dict[str, Any]],
    *,
    eval_rows: list[dict[str, Any]],
    fanout_top_k: int,
    rank_pool_size: int,
    compatibility_rerank: bool,
) -> list[list[dict[str, Any]]]:
    out = []
    top_k = max(1, min(int(fanout_top_k), len(classes)))
    pool_k = len(classes) if compatibility_rerank else min(max(top_k, int(rank_pool_size)), len(classes))
    for task, row in zip(eval_rows, probs):
        ranked = sorted(enumerate(row), key=lambda item: item[1], reverse=True)[:pool_k]
        proposals = []
        for idx, score in ranked:
            structural_class = classes[idx]
            compatibility = structural_class_compatibility(task, structural_class)
            final_score = float(score)
            if compatibility_rerank:
                final_score = (0.65 * float(score)) + (0.35 * float(compatibility["score"]))
            proposals.append(
                {
                    "structural_class": structural_class,
                    "rank_score": float(final_score),
                    "model_rank_score": float(score),
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
            out.append(compatible[:top_k] if compatible else proposals[:top_k])
        else:
            out.append(proposals[:top_k])
    return out


def planned_report(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "policy": "project_theseus_macos_mlx_structural_action_smoke_v0",
        "created_utc": now(),
        "trigger_state": "PLANNED",
        "config": args.config,
        "execute": False,
        "summary": {
            "comparison_level": config.get("comparison_level"),
            "external_inference_calls": 0,
        },
        "external_inference_calls": 0,
    }


def blocked_report(args: argparse.Namespace, started: float, reason: str, *, trigger_state: str = "RED") -> dict[str, Any]:
    gate_severity = "hard" if trigger_state == "RED" else "soft"
    return {
        "policy": "project_theseus_macos_mlx_structural_action_smoke_v0",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "config": args.config,
        "execute": True,
        "summary": {
            "mlx_available": False,
            "mlx_used": False,
            "blocked_reason": reason,
            "external_inference_calls": 0,
            "teacher_used": False,
            "public_training_rows": 0,
        },
        "gates": [gate("mlx_smoke_available", False, reason, gate_severity)],
        "_candidate_rows": [],
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_or_empty(report.get("summary"))
    lines = [
        "# macOS MLX Structural Action Smoke",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- mlx_available: `{summary.get('mlx_available')}`",
        f"- mlx_used: `{summary.get('mlx_used')}`",
        f"- mlx_default_device: `{summary.get('mlx_default_device')}`",
        f"- train_rows: `{summary.get('train_rows')}`",
        f"- eval_rows: `{summary.get('eval_rows')}`",
        f"- verifier_pass_rate: `{summary.get('verifier_pass_rate')}`",
        f"- syntax_pass_rate: `{summary.get('syntax_pass_rate')}`",
        f"- fallback_return_rows: `{summary.get('fallback_return_rows')}`",
        "",
        "## Gates",
    ]
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    lines.extend(["", str(report.get("score_semantics") or "")])
    return "\n".join(lines).rstrip() + "\n"


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
