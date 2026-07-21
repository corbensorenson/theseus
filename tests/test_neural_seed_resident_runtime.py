from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neural_seed_resident_runtime import BoundedPromptPrefixCache  # noqa: E402
import openai_compat_server  # noqa: E402


def test_prompt_prefix_cache_is_bounded_lru() -> None:
    cache = BoundedPromptPrefixCache(maximum_entries=2)
    cache.put("a", ("logits-a", "cache-a"))
    cache.put("b", ("logits-b", "cache-b"))
    assert cache.get("a") == ("logits-a", "cache-a")
    cache.put("c", ("logits-c", "cache-c"))
    assert cache.get("b") is None
    assert cache.get("a") == ("logits-a", "cache-a")
    assert cache.get("c") == ("logits-c", "cache-c")
    assert cache.entry_count == 2


def test_openai_compat_routes_enabled_resident_model_without_subprocess() -> None:
    class FakeRuntime:
        def generate(self, prompt: str, **_kwargs: object) -> dict:
            return {
                "text": prompt.upper(),
                "generation": {"state": "GREEN"},
                "runtime_receipt": {"completion_cache_state": "MISS"},
            }

        def status(self) -> dict:
            return {"checkpoint_sha256": "checkpoint-a"}

    cfg = {
        "resident_neural_seed": {
            "enabled": True,
            "model_id": "theseus-neural-seed",
            "max_tokens": 8,
            "beam_width": 2,
            "branching_factor": 2,
            "length_penalty": 0.6,
        }
    }
    result = openai_compat_server.local_answer(
        "hello",
        "theseus-neural-seed",
        cfg,
        {},
        resident_runtime=FakeRuntime(),
    )
    assert result["ok"] is True
    assert result["content"] == "HELLO"
    assert result["mode"] == "resident_neural_seed"
    assert result["external_inference_calls"] == 0


def test_resident_model_never_falls_back_when_enabled_but_unavailable() -> None:
    cfg = {
        "resident_neural_seed": {
            "enabled": True,
            "model_id": "theseus-neural-seed",
        }
    }
    result = openai_compat_server.local_answer(
        "hello", "theseus-neural-seed", cfg, {}, resident_runtime=None
    )
    assert result["ok"] is False
    assert result["content"] == ""
    assert result["mode"] == "resident_neural_seed_unavailable"
    assert result["evidence"]["fault"] == "resident_runtime_not_initialized"


def test_disabled_resident_runtime_does_not_load_model() -> None:
    assert openai_compat_server.initialize_resident_runtime({}) is None
    assert (
        openai_compat_server.initialize_resident_runtime(
            {"resident_neural_seed": {"enabled": False}}
        )
        is None
    )
