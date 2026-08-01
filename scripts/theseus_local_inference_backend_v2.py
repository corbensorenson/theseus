#!/usr/bin/env python3
"""Successor offline backend with exact context-residual completion custody."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import theseus_assistant_route_integrity_v2 as route_integrity
import theseus_local_inference_backend as v1


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKER_CONFIG = ROOT / "configs" / "core_evidence_tmax_9b_completion_worker.json"
DEFAULT_RUNTIME_PREFLIGHT = v1.DEFAULT_RUNTIME_PREFLIGHT
BACKEND_POLICY = "project_theseus_local_inference_backend_v2"
FROZEN_REPO_ID = v1.FROZEN_REPO_ID
FROZEN_REVISION = v1.FROZEN_REVISION
FROZEN_SNAPSHOT_MANIFEST_SHA256 = v1.FROZEN_SNAPSHOT_MANIFEST_SHA256
SYSTEM_PROMPT = v1.SYSTEM_PROMPT
BackendFault = v1.BackendFault

# Expose patchable dependencies for focused tests without mutating v1 custody.
local_snapshot = v1.local_snapshot
complete_model_snapshot = v1.complete_model_snapshot
snapshot_manifest = v1.snapshot_manifest
package_versions = v1.package_versions
snapshot_context_window = v1.snapshot_context_window
sanitize_generated_text = v1.sanitize_generated_text


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
    observed_manifest = (
        snapshot_manifest(snapshot) if complete_model_snapshot(snapshot) else ""
    )
    if observed_manifest != identity.get("snapshot_manifest_sha256"):
        faults.append("local_snapshot_manifest_mismatch")
    runtime_versions = package_versions()
    if not runtime_versions.get("mlx_lm") or not runtime_versions.get("mlx"):
        faults.append("qualified_mlx_runtime_missing")
    if execution_mode not in route_integrity.LOCAL_MODEL_MODES:
        faults.append("unsupported_execution_mode")
    if len(route_context_digest) != 64:
        faults.append("route_context_digest_invalid")
    effective_maximum = int(v1.get_path(identity, ["decoder", "maximum_tokens"], 0) or 0)
    if not preflight_only and not prompt.strip():
        faults.append("empty_model_prompt")

    answer = ""
    generation_metrics: dict[str, Any] = {}
    inference_calls = 0
    if not faults and not preflight_only:
        try:
            factory = model_factory or LocalMlxChatModel
            model = factory(card, snapshot, effective_maximum)
            raw_answer = str(
                model.generate([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ])
                or ""
            )
            answer = sanitize_generated_text(raw_answer)
            generation_metrics = dict(
                getattr(model, "last_generation_metrics", {}) or {}
            )
            generation_metrics["terminal_marker_stripped"] = (
                raw_answer.strip() != answer
            )
            inference_calls = 1
            if generation_metrics.get("physical_context_boundary_hit") is True:
                faults.append("instrument_inadequate_generation_boundary_hit")
            if not answer:
                faults.append("empty_local_model_response")
        except Exception as exc:
            faults.append(f"local_model_generation_failed:{type(exc).__name__}")

    trigger_state = "GREEN" if not faults else "RED"
    return {
        "policy": BACKEND_POLICY,
        "created_utc": v1.now(),
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
        "response": {
            "mode": "frozen_tmax_local_inference",
            "answer": answer if trigger_state == "GREEN" and not preflight_only else "",
            "teacher_recommended": False,
            "evidence": {
                "backend_policy": BACKEND_POLICY,
                "model_identity_sha256": identity.get("identity_sha256"),
                "route_context_digest": route_context_digest,
            },
        },
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


class LocalMlxChatModel(v1.LocalMlxChatModel):
    def generate(self, messages: list[dict[str, str]]) -> str:
        from mlx_lm import stream_generate

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **dict(self.card.get("chat_template_kwargs") or {}),
        )
        prompt_tokens = len(prompt) if isinstance(prompt, list) else 0
        if not self.model_context_window_tokens:
            raise BackendFault("model_declared_context_window_missing")
        context_residual = self.model_context_window_tokens - prompt_tokens
        if context_residual <= 0:
            raise BackendFault("prompt_exhausts_model_context_window")
        effective_maximum_tokens = min(self.maximum_tokens, context_residual)
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
                peak_memory_gib = max(
                    peak_memory_gib,
                    float(getattr(response, "peak_memory", 0.0) or 0.0),
                )
                prompt_tps = getattr(response, "prompt_tps", prompt_tps)
                generation_tps = getattr(response, "generation_tps", generation_tps)
                backend_finish_reason = getattr(
                    response, "finish_reason", backend_finish_reason
                )
                if self.completion_predicate is not None and self.completion_predicate(
                    "".join(chunks)
                ):
                    termination_reason = "parser_complete"
                    break
            else:
                termination_reason = (
                    "model_eos"
                    if backend_finish_reason == "stop"
                    else "physical_context_boundary"
                    if backend_finish_reason == "length"
                    else "backend_stopped_without_reason"
                )
        finally:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                close_stream()
        self.last_generation_metrics = {
            "generated_tokens": generated_tokens,
            "prompt_tokens": prompt_tokens,
            "exact_prompt_tokens": prompt_tokens,
            "configured_maximum_tokens": self.maximum_tokens,
            "effective_maximum_tokens": effective_maximum_tokens,
            "effective_context_residual_tokens": context_residual,
            "model_context_window_tokens": self.model_context_window_tokens,
            "project_selected_quality_token_cap": None,
            "backend_finish_reason": backend_finish_reason,
            "termination_reason": termination_reason,
            "completion_predicate_enabled": self.completion_predicate is not None,
            "safety_ceiling_hit": termination_reason == "physical_context_boundary",
            "physical_context_boundary_hit": termination_reason
            == "physical_context_boundary",
            "load_wall_ms": round(self.load_wall_ms, 3),
            "generation_wall_ms": round(
                (time.perf_counter() - started) * 1000.0, 3
            ),
            "mlx_peak_memory_gib": round(peak_memory_gib, 3),
            "prompt_tokens_per_second": (
                None if prompt_tps is None else round(float(prompt_tps), 3)
            ),
            "generation_tokens_per_second": (
                None if generation_tps is None else round(float(generation_tps), 3)
            ),
        }
        return "".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=v1.rel(DEFAULT_WORKER_CONFIG))
    parser.add_argument(
        "--runtime-preflight", default=v1.rel(DEFAULT_RUNTIME_PREFLIGHT)
    )
    parser.add_argument("--session-id", default="local_assistant")
    parser.add_argument(
        "--execution-mode",
        required=True,
        choices=sorted(route_integrity.LOCAL_MODEL_MODES),
    )
    parser.add_argument("--route-context-digest", required=True)
    parser.add_argument("--maximum-tokens", type=int, default=0)
    parser.add_argument("--required-repo-id", default=FROZEN_REPO_ID)
    parser.add_argument("--required-revision", default=FROZEN_REVISION)
    parser.add_argument(
        "--required-snapshot-manifest-sha256",
        default=FROZEN_SNAPSHOT_MANIFEST_SHA256,
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    prompt = "" if args.preflight_only else sys.stdin.read()
    report = run_backend(
        worker_config_path=v1.resolve(args.config),
        runtime_preflight_path=v1.resolve(args.runtime_preflight),
        execution_mode=args.execution_mode,
        route_context_digest=str(args.route_context_digest or ""),
        session_id=str(args.session_id or "local_assistant"),
        prompt=prompt,
        maximum_tokens=int(args.maximum_tokens or 0),
        required_repo_id=str(args.required_repo_id or ""),
        required_revision=str(args.required_revision or ""),
        required_snapshot_manifest_sha256=str(
            args.required_snapshot_manifest_sha256 or ""
        ),
        preflight_only=bool(args.preflight_only),
    )
    v1.write_json(v1.resolve(args.out), report)
    print(json.dumps(v1.compact_stdout(report), sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
