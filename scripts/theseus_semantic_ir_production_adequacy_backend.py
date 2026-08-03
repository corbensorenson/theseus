#!/usr/bin/env python3
"""Adequacy-only local backend with a non-scoring host wall-time interlock."""

from __future__ import annotations

import signal
import time
from typing import Any

import theseus_local_inference_backend_v2 as base


class _HostSafetyWallTimeExpired(TimeoutError):
    """Private control-flow exception raised by the adequacy host interlock."""


class AdequacyLocalMlxChatModel(base.LocalMlxChatModel):
    def __init__(
        self,
        *args: Any,
        maximum_wall_seconds: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if maximum_wall_seconds <= 0:
            raise base.BackendFault("host_safety_wall_time_invalid")
        self.maximum_wall_seconds = float(maximum_wall_seconds)

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
            raise base.BackendFault("model_declared_context_window_missing")
        context_residual = self.model_context_window_tokens - prompt_tokens
        if context_residual <= 0:
            raise base.BackendFault("prompt_exhausts_model_context_window")
        effective_maximum_tokens = min(self.maximum_tokens, context_residual)
        started = time.perf_counter()
        stream = None
        chunks: list[str] = []
        generated_tokens = 0
        peak_memory_gib = 0.0
        prompt_tps = None
        generation_tps = None
        backend_finish_reason = None
        termination_reason = "generation_error"
        previous_alarm_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        if previous_timer != (0.0, 0.0):
            raise base.BackendFault("host_safety_wall_timer_conflict")

        def expire_host_call(_signum: int, _frame: Any) -> None:
            raise _HostSafetyWallTimeExpired

        signal.signal(signal.SIGALRM, expire_host_call)
        signal.setitimer(signal.ITIMER_REAL, self.maximum_wall_seconds)
        try:
            try:
                stream = stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=effective_maximum_tokens,
                    sampler=self.sampler,
                    logits_processors=self.logits_processors,
                    **self.generation_cache_kwargs,
                )
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
                    if time.perf_counter() - started >= self.maximum_wall_seconds:
                        termination_reason = "host_safety_wall_time"
                        break
                else:
                    termination_reason = (
                        "model_eos"
                        if backend_finish_reason == "stop"
                        else "physical_context_boundary"
                        if backend_finish_reason == "length"
                        else "backend_stopped_without_reason"
                    )
            except _HostSafetyWallTimeExpired:
                termination_reason = "host_safety_wall_time"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_alarm_handler)
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
            "host_safety_wall_seconds": self.maximum_wall_seconds,
            "host_safety_wall_time_hit": termination_reason
            == "host_safety_wall_time",
            "safety_ceiling_hit": termination_reason
            in {"physical_context_boundary", "host_safety_wall_time"},
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
