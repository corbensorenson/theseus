from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_semantic_ir_production_adequacy_backend_v2 as backend  # noqa: E402


def test_watchdog_partial_is_retained_only_as_invalid_diagnostic(monkeypatch) -> None:
    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return [1, 2, 3]

    class Response:
        text = "partial artifact"
        finish_reason = None
        peak_memory = 0.0
        prompt_tps = 1.0
        generation_tps = 1.0

    times = iter([0.0, 2.0, 2.0])
    monkeypatch.setattr(backend.base.time, "perf_counter", lambda: next(times))
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        types.SimpleNamespace(
            stream_generate=lambda *_args, **_kwargs: iter([Response()])
        ),
    )
    model = backend.DiagnosticAdequacyLocalMlxChatModel.__new__(
        backend.DiagnosticAdequacyLocalMlxChatModel
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

    assert model.generate([{"role": "user", "content": "prompt"}]) == (
        "partial artifact"
    )
    report = {
        "trigger_state": "GREEN",
        "faults": [],
        "metrics": model.last_generation_metrics,
        "response": {"answer": "partial artifact"},
    }
    backend.attach_invalid_observation_diagnostic(report, model)

    diagnostic = report["invalid_observation_diagnostic"]
    assert report["trigger_state"] == "RED"
    assert report["response"]["answer"] == ""
    assert diagnostic["partial_output_text"] == "partial artifact"
    assert diagnostic["candidate_admission_allowed"] is False
    assert diagnostic["hidden_evaluation_allowed"] is False
