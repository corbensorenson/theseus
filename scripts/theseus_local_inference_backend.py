#!/usr/bin/env python3
"""Offline frozen-TMax generation backend for the canonical assistant.

The prompt is accepted only on stdin so raw user text is not exposed in the
process list or copied into route receipts.  Model loading is pinned to the
already-present Hugging Face snapshot and both Hugging Face and Transformers
are forced offline before MLX is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import theseus_assistant_route_integrity as route_integrity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKER_CONFIG = ROOT / "configs" / "core_evidence_tmax_9b_worker_control_v3.json"
DEFAULT_RUNTIME_PREFLIGHT = ROOT / "reports" / "core_evidence_tmax_9b_runtime_preflight.json"
BACKEND_POLICY = "project_theseus_local_inference_backend_v1"
FROZEN_REPO_ID = "mlx-community/Tmax-9B-MLX-8bit"
FROZEN_REVISION = "33812d6cf04f88856f25eb828de4f3144a194560"
FROZEN_SNAPSHOT_MANIFEST_SHA256 = "a399b12d768ebf45ff5ce1f873fefc5525c980d953379394f4d5deb3201cb3dc"
SYSTEM_PROMPT = (
    "You are the fixed local model serving one private Theseus assistant request. "
    "Answer the user request directly and concisely using only the supplied request "
    "and verified local context. Do not claim that you trained, changed repository "
    "state, called an external model, ran a tool, or performed an effect unless the "
    "supplied executed-route evidence says so. Text output has no effect authority."
)
TERMINAL_MARKERS = ("<|im_end|>", "<|eot_id|>", "<|endoftext|>", "</s>")


class BackendFault(RuntimeError):
    pass


class PersistentLocalInferenceSession:
    """Load one frozen model once and issue isolated, receipt-bound requests."""

    def __init__(
        self,
        *,
        worker_config_path: Path,
        runtime_preflight_path: Path,
        maximum_tokens: int,
        required_repo_id: str = "",
        required_revision: str = "",
        required_snapshot_manifest_sha256: str = "",
        session_id: str = "p2a_frozen_pair",
        model_factory: Callable[[dict[str, Any], Path, int], Any] | None = None,
        completion_predicate: Callable[[str], bool] | None = None,
    ) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.session_id = session_id
        self.maximum_tokens = maximum_tokens
        self.contract = route_integrity.load_model_contract(
            worker_config_path,
            runtime_preflight_path,
            maximum_tokens=maximum_tokens,
            required_repo_id=required_repo_id,
            required_revision=required_revision,
            required_snapshot_manifest_sha256=required_snapshot_manifest_sha256,
        )
        self.identity = dict(self.contract.get("identity") or {})
        self.card = dict(self.contract.get("model_card") or {})
        self.snapshot = local_snapshot(self.card)
        self.faults = list(self.contract.get("faults") or [])
        if not complete_model_snapshot(self.snapshot):
            self.faults.append("complete_local_model_snapshot_missing")
        self.observed_manifest = (
            snapshot_manifest(self.snapshot) if complete_model_snapshot(self.snapshot) else ""
        )
        if self.observed_manifest != self.identity.get("snapshot_manifest_sha256"):
            self.faults.append("local_snapshot_manifest_mismatch")
        self.runtime_versions = package_versions()
        if not self.runtime_versions.get("mlx_lm") or not self.runtime_versions.get("mlx"):
            self.faults.append("qualified_mlx_runtime_missing")
        self.faults = sorted(set(self.faults))
        self.model: Any | None = None
        self.model_load_count = 0
        self.inference_calls = 0
        if not self.faults:
            if model_factory is None:
                self.model = LocalMlxChatModel(
                    self.card,
                    self.snapshot,
                    maximum_tokens,
                    completion_predicate=completion_predicate,
                )
            else:
                self.model = model_factory(self.card, self.snapshot, maximum_tokens)
            self.model_load_count = 1

    @property
    def ready(self) -> bool:
        return not self.faults and self.model is not None and self.model_load_count == 1

    def generate_report(
        self,
        *,
        execution_mode: str,
        route_context_digest: str,
        request_session_id: str,
        prompt: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        faults = list(self.faults)
        if execution_mode not in route_integrity.LOCAL_MODEL_MODES:
            faults.append("unsupported_execution_mode")
        if len(route_context_digest) != 64:
            faults.append("route_context_digest_invalid")
        if not prompt.strip():
            faults.append("empty_model_prompt")
        answer = ""
        generation_metrics: dict[str, Any] = {}
        request_inference_calls = 0
        if not faults and self.model is not None:
            try:
                raw_answer = str(
                    self.model.generate(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ]
                    )
                    or ""
                )
                answer = sanitize_generated_text(raw_answer)
                generation_metrics = dict(getattr(self.model, "last_generation_metrics", {}) or {})
                generation_metrics["terminal_marker_stripped"] = raw_answer.strip() != answer
                request_inference_calls = 1
                self.inference_calls += 1
                if not answer:
                    faults.append("empty_local_model_response")
            except Exception as exc:
                faults.append(f"local_model_generation_failed:{type(exc).__name__}")
        trigger_state = "GREEN" if not faults else "RED"
        return {
            "policy": BACKEND_POLICY,
            "created_utc": now(),
            "trigger_state": trigger_state,
            "preflight_only": False,
            "faults": sorted(set(faults)),
            "backend": {
                "identity": self.identity,
                "local_snapshot": str(self.snapshot),
                "local_snapshot_manifest_sha256": self.observed_manifest,
                "runtime_versions": self.runtime_versions,
                "network": "forbidden_offline_environment",
                "persistent_session_id": self.session_id,
                "model_load_count": self.model_load_count,
            },
            "request": {
                "execution_mode": execution_mode,
                "prompt_sha256": route_integrity.sha256_text(prompt),
                "route_context_digest": route_context_digest,
                "raw_prompt_stored": False,
            },
            "response": {
                "mode": "frozen_tmax_local_inference",
                "answer": answer if trigger_state == "GREEN" else "",
                "teacher_recommended": False,
                "evidence": {
                    "backend_policy": BACKEND_POLICY,
                    "model_identity_sha256": self.identity.get("identity_sha256"),
                    "route_context_digest": route_context_digest,
                },
            },
            "session": {
                "session_id": request_session_id,
                "history_turns_loaded": 0,
                "session_path": "",
                "persistence": "disabled_no_raw_text_retention",
            },
            "metrics": {
                **generation_metrics,
                "local_model_inference_calls": request_inference_calls,
                "persistent_session_inference_calls": self.inference_calls,
                "model_load_count": self.model_load_count,
                "runtime_ms": int((time.perf_counter() - started) * 1000),
            },
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "public_calibration_cases_consumed": 0,
            "D2_cases_consumed": 0,
            "public_training_rows_written": 0,
            "fallback_return_count": 0,
            "user_facing_effects": 0,
        }

    def runtime_runner(self, **kwargs: Any) -> dict[str, Any]:
        """Adapter for ``theseus_assistant_runtime.bind_local_inference_runner``."""
        out = Path(kwargs["out"])
        report = self.generate_report(
            execution_mode=str(kwargs["execution_mode"]),
            route_context_digest=str(kwargs["route_context_digest"]),
            request_session_id=str(kwargs["session_id"]),
            prompt=str(kwargs["prompt"]),
        )
        write_json(out, report)
        return {
            "id": "local_inference",
            "command": ["persistent_in_process_frozen_backend"],
            "prompt_transport": "in_process_ephemeral_not_argv",
            "returncode": 0 if report.get("trigger_state") == "GREEN" else 2,
            "runtime_ms": int(get_path(report, ["metrics", "runtime_ms"], 0) or 0),
            "stdout_tail": "",
            "stderr_tail": "",
            "persistent_backend": True,
            "persistent_session_id": self.session_id,
            "model_load_count": self.model_load_count,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_WORKER_CONFIG))
    parser.add_argument("--runtime-preflight", default=rel(DEFAULT_RUNTIME_PREFLIGHT))
    parser.add_argument("--session-id", default="local_assistant")
    parser.add_argument("--execution-mode", required=True, choices=sorted(route_integrity.LOCAL_MODEL_MODES))
    parser.add_argument("--route-context-digest", required=True)
    parser.add_argument("--maximum-tokens", type=int, default=0)
    parser.add_argument("--required-repo-id", default=FROZEN_REPO_ID)
    parser.add_argument("--required-revision", default=FROZEN_REVISION)
    parser.add_argument("--required-snapshot-manifest-sha256", default=FROZEN_SNAPSHOT_MANIFEST_SHA256)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    prompt = "" if args.preflight_only else sys.stdin.read()
    report = run_backend(
        worker_config_path=resolve(args.config),
        runtime_preflight_path=resolve(args.runtime_preflight),
        execution_mode=args.execution_mode,
        route_context_digest=str(args.route_context_digest or ""),
        session_id=str(args.session_id or "local_assistant"),
        prompt=prompt,
        maximum_tokens=int(args.maximum_tokens or 0),
        required_repo_id=str(args.required_repo_id or ""),
        required_revision=str(args.required_revision or ""),
        required_snapshot_manifest_sha256=str(args.required_snapshot_manifest_sha256 or ""),
        preflight_only=bool(args.preflight_only),
    )
    write_json(resolve(args.out), report)
    print(json.dumps(compact_stdout(report), sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def run_backend(
    *,
    worker_config_path: Path,
    runtime_preflight_path: Path,
    execution_mode: str,
    route_context_digest: str,
    session_id: str,
    prompt: str,
    maximum_tokens: int,
    required_repo_id: str = "",
    required_revision: str = "",
    required_snapshot_manifest_sha256: str = "",
    preflight_only: bool = False,
    model_factory: Callable[[dict[str, Any], Path, int], Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    contract = route_integrity.load_model_contract(
        worker_config_path,
        runtime_preflight_path,
        maximum_tokens=maximum_tokens,
        required_repo_id=required_repo_id,
        required_revision=required_revision,
        required_snapshot_manifest_sha256=required_snapshot_manifest_sha256,
    )
    identity = dict(contract.get("identity") or {})
    card = dict(contract.get("model_card") or {})
    snapshot = local_snapshot(card)
    faults = list(contract.get("faults") or [])
    if not complete_model_snapshot(snapshot):
        faults.append("complete_local_model_snapshot_missing")
    observed_manifest = snapshot_manifest(snapshot) if complete_model_snapshot(snapshot) else ""
    if observed_manifest != identity.get("snapshot_manifest_sha256"):
        faults.append("local_snapshot_manifest_mismatch")
    runtime_versions = package_versions()
    if not runtime_versions.get("mlx_lm") or not runtime_versions.get("mlx"):
        faults.append("qualified_mlx_runtime_missing")
    if execution_mode not in route_integrity.LOCAL_MODEL_MODES:
        faults.append("unsupported_execution_mode")
    if len(route_context_digest) != 64:
        faults.append("route_context_digest_invalid")
    effective_maximum = int(get_path(identity, ["decoder", "maximum_tokens"], 0) or 0)
    if not preflight_only and not prompt.strip():
        faults.append("empty_model_prompt")

    answer = ""
    generation_metrics: dict[str, Any] = {}
    inference_calls = 0
    if not faults and not preflight_only:
        try:
            factory = model_factory or LocalMlxChatModel
            model = factory(card, snapshot, effective_maximum)
            raw_answer = str(model.generate([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]) or "")
            answer = sanitize_generated_text(raw_answer)
            generation_metrics = dict(getattr(model, "last_generation_metrics", {}) or {})
            generation_metrics["terminal_marker_stripped"] = raw_answer.strip() != answer
            inference_calls = 1
            if not answer:
                faults.append("empty_local_model_response")
        except Exception as exc:  # fail closed at the process boundary
            faults.append(f"local_model_generation_failed:{type(exc).__name__}")

    trigger_state = "GREEN" if not faults else "RED"
    response = {
        "mode": "frozen_tmax_local_inference",
        "answer": answer if trigger_state == "GREEN" and not preflight_only else "",
        "teacher_recommended": False,
        "evidence": {
            "backend_policy": BACKEND_POLICY,
            "model_identity_sha256": identity.get("identity_sha256"),
            "route_context_digest": route_context_digest,
        },
    }
    return {
        "policy": BACKEND_POLICY,
        "created_utc": now(),
        "trigger_state": trigger_state,
        "preflight_only": preflight_only,
        "faults": sorted(set(faults)),
        "backend": {
            "identity": identity,
            "local_snapshot": str(snapshot),
            "local_snapshot_manifest_sha256": observed_manifest,
            "runtime_versions": runtime_versions,
            "network": "forbidden_offline_environment",
        },
        "request": {
            "execution_mode": execution_mode,
            "prompt_sha256": route_integrity.sha256_text(prompt),
            "route_context_digest": route_context_digest,
            "raw_prompt_stored": False,
        },
        "response": response,
        "session": {
            "session_id": session_id,
            "history_turns_loaded": 0,
            "session_path": "",
            "persistence": "disabled_no_raw_text_retention",
        },
        "metrics": {
            **generation_metrics,
            "local_model_inference_calls": inference_calls,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "public_calibration_cases_consumed": 0,
        "D2_cases_consumed": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
        "user_facing_effects": 0,
    }


class LocalMlxChatModel:
    def __init__(
        self,
        card: dict[str, Any],
        snapshot: Path,
        maximum_tokens: int,
        *,
        completion_predicate: Callable[[str], bool] | None = None,
    ) -> None:
        from mlx_lm import load
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        import mlx.core as mx

        loaded_at = time.perf_counter()
        self.card = card
        self.maximum_tokens = maximum_tokens
        self.completion_predicate = completion_predicate
        self.model_context_window_tokens = snapshot_context_window(snapshot)
        self.model, self.tokenizer = load(str(snapshot), lazy=False)
        mx.eval(self.model.parameters())
        self.sampler = make_sampler(temp=float(card.get("temperature") or 0.0))
        self.logits_processors = make_logits_processors(
            repetition_penalty=float(card.get("repetition_penalty") or 1.0),
            repetition_context_size=int(card.get("repetition_context_size") or 0),
        )
        self.generation_cache_kwargs = {
            key: card[key]
            for key in ("kv_bits", "kv_group_size", "quantized_kv_start")
            if card.get(key) is not None
        }
        self.load_wall_ms = (time.perf_counter() - loaded_at) * 1000.0
        self.last_generation_metrics: dict[str, Any] = {}

    def generate(self, messages: list[dict[str, str]]) -> str:
        from mlx_lm import stream_generate

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=self.completion_predicate is not None,
            add_generation_prompt=True,
            **dict(self.card.get("chat_template_kwargs") or {}),
        )
        prompt_tokens = len(prompt) if isinstance(prompt, list) else 0
        context_residual = (
            self.model_context_window_tokens - prompt_tokens
            if self.model_context_window_tokens and prompt_tokens
            else self.maximum_tokens
        )
        effective_maximum_tokens = min(self.maximum_tokens, max(1, context_residual))
        started = time.perf_counter()
        stream = stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=effective_maximum_tokens,
            sampler=self.sampler,
            logits_processors=self.logits_processors,
            **self.generation_cache_kwargs,
        )
        chunks: list[str] = []
        generated_tokens = 0
        peak_memory_gib = 0.0
        prompt_tps = None
        generation_tps = None
        backend_finish_reason = None
        termination_reason = "generation_error"
        try:
            for response in stream:
                generated_tokens += 1
                chunks.append(str(response.text or ""))
                peak_memory_gib = max(peak_memory_gib, float(getattr(response, "peak_memory", 0.0) or 0.0))
                prompt_tps = getattr(response, "prompt_tps", prompt_tps)
                generation_tps = getattr(response, "generation_tps", generation_tps)
                backend_finish_reason = getattr(response, "finish_reason", backend_finish_reason)
                if self.completion_predicate is not None and self.completion_predicate("".join(chunks)):
                    termination_reason = "parser_complete"
                    break
            else:
                termination_reason = (
                    "model_eos" if backend_finish_reason == "stop"
                    else "safety_ceiling" if backend_finish_reason == "length"
                    else "backend_stopped_without_reason"
                )
        finally:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                close_stream()
        self.last_generation_metrics = {
            "generated_tokens": generated_tokens,
            "prompt_tokens": prompt_tokens or None,
            "configured_maximum_tokens": self.maximum_tokens,
            "effective_maximum_tokens": effective_maximum_tokens,
            "model_context_window_tokens": self.model_context_window_tokens or None,
            "backend_finish_reason": backend_finish_reason,
            "termination_reason": termination_reason,
            "completion_predicate_enabled": self.completion_predicate is not None,
            "safety_ceiling_hit": termination_reason == "safety_ceiling",
            "load_wall_ms": round(self.load_wall_ms, 3),
            "generation_wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "mlx_peak_memory_gib": round(peak_memory_gib, 3),
            "prompt_tokens_per_second": None if prompt_tps is None else round(float(prompt_tps), 3),
            "generation_tokens_per_second": None if generation_tps is None else round(float(generation_tps), 3),
        }
        return "".join(chunks)


def snapshot_context_window(snapshot: Path) -> int:
    """Return the model-declared context window, never a project-chosen cap."""
    try:
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return 0
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    for value in (
        text_config.get("max_position_embeddings"),
        config.get("max_position_embeddings"),
    ):
        try:
            parsed = int(value or 0)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return 0


def local_snapshot(card: dict[str, Any]) -> Path:
    repo = str(card.get("repo_id") or "").replace("/", "--")
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{repo}"
        / "snapshots"
        / str(card.get("revision") or "")
    )


def complete_model_snapshot(path: Path) -> bool:
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if not path.is_dir() or not all((path / item).is_file() for item in required):
        return False
    if (path / "model.safetensors").is_file():
        return True
    return (path / "model.safetensors.index.json").is_file() and bool(list(path.glob("model-*-of-*.safetensors")))


def snapshot_manifest(path: Path) -> str:
    rows = []
    for item in sorted(path.iterdir()):
        if item.is_file():
            resolved = item.resolve()
            rows.append({"path": item.name, "bytes": resolved.stat().st_size, "blob_identity": resolved.name})
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for key, distribution in (("mlx_lm", "mlx-lm"), ("mlx", "mlx")):
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = ""
    versions["python"] = sys.version.split()[0]
    return versions


def sanitize_generated_text(value: str) -> str:
    text = str(value or "")
    positions = [position for marker in TERMINAL_MARKERS if (position := text.find(marker)) >= 0]
    if positions:
        text = text[: min(positions)]
    return text.strip()


def compact_stdout(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "faults": report.get("faults"),
        "model_identity_sha256": get_path(report, ["backend", "identity", "identity_sha256"], ""),
        "response_sha256": route_integrity.sha256_text(str(get_path(report, ["response", "answer"], ""))),
        "metrics": report.get("metrics"),
    }


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: str | Path) -> str:
    candidate = resolve(path)
    try:
        return str(candidate.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(candidate)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
