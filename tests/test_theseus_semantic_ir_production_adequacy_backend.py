from __future__ import annotations

import sys
import time
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_semantic_ir_production_adequacy_backend as backend  # noqa: E402


def test_host_wall_is_non_quality_non_physical_infrastructure_boundary(
    monkeypatch,
) -> None:
    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return [1, 2, 3]

    class Response:
        text = "unfinished"
        finish_reason = None
        peak_memory = 0.0
        prompt_tps = 1.0
        generation_tps = 1.0

    times = iter([0.0, 2.0, 2.0])
    monkeypatch.setattr(backend.time, "perf_counter", lambda: next(times))
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        types.SimpleNamespace(stream_generate=lambda *_args, **_kwargs: iter([Response()])),
    )
    model = backend.AdequacyLocalMlxChatModel.__new__(
        backend.AdequacyLocalMlxChatModel
    )
    model.model = object()
    model.tokenizer = Tokenizer()
    model.card = {"chat_template_kwargs": {}}
    model.maximum_tokens = 64
    model.model_context_window_tokens = 64
    model.sampler = None
    model.logits_processors = []
    model.generation_cache_kwargs = {}
    model.load_wall_ms = 0.0
    model.completion_predicate = None
    model.maximum_wall_seconds = 1.0

    model.generate([{"role": "user", "content": "prompt"}])

    metrics = model.last_generation_metrics
    assert metrics["termination_reason"] == "host_safety_wall_time"
    assert metrics["host_safety_wall_time_hit"] is True
    assert metrics["safety_ceiling_hit"] is True
    assert metrics["physical_context_boundary_hit"] is False
    assert metrics["project_selected_quality_token_cap"] is None


def test_host_wall_interrupts_a_blocked_stream_step(monkeypatch) -> None:
    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return [1, 2, 3]

    def blocked_stream():
        time.sleep(1.0)
        yield None

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        types.SimpleNamespace(stream_generate=lambda *_args, **_kwargs: blocked_stream()),
    )
    model = backend.AdequacyLocalMlxChatModel.__new__(
        backend.AdequacyLocalMlxChatModel
    )
    model.model = object()
    model.tokenizer = Tokenizer()
    model.card = {"chat_template_kwargs": {}}
    model.maximum_tokens = 64
    model.model_context_window_tokens = 64
    model.sampler = None
    model.logits_processors = []
    model.generation_cache_kwargs = {}
    model.load_wall_ms = 0.0
    model.completion_predicate = None
    model.maximum_wall_seconds = 0.01

    started = time.perf_counter()
    assert model.generate([{"role": "user", "content": "prompt"}]) == ""
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5
    assert model.last_generation_metrics["termination_reason"] == "host_safety_wall_time"
    assert model.last_generation_metrics["host_safety_wall_time_hit"] is True
