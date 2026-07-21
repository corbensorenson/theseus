from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neural_seed_resident_runtime import (  # noqa: E402
    BoundedPromptPrefixCache,
    ContinuousRequestBatcher,
)
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


def test_continuous_request_batcher_coalesces_compatible_concurrent_work() -> None:
    observed_batches: list[list[int]] = []

    def process(payloads: list[dict]) -> list[dict]:
        values = [int(payload["value"]) for payload in payloads]
        observed_batches.append(values)
        return [{"value": value * 2, "runtime_receipt": {}} for value in values]

    batcher = ContinuousRequestBatcher(
        process,
        window_ms=20.0,
        maximum_batch_size=8,
    )
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(
                pool.map(
                    lambda value: batcher.submit("shared-contract", {"value": value}),
                    range(4),
                )
            )
        status = batcher.status()
    finally:
        batcher.close()

    assert [output["value"] for output in outputs] == [0, 2, 4, 6]
    assert len(observed_batches) == 1
    assert sorted(observed_batches[0]) == [0, 1, 2, 3]
    assert status["batch_count"] == 1
    assert status["request_count"] == 4
    assert status["batched_request_count"] == 4
    assert status["maximum_observed_batch_size"] == 4
    assert all(
        output["runtime_receipt"]["continuous_batch_queue_seconds"] >= 0.0
        for output in outputs
    )


def test_continuous_request_batcher_propagates_processor_faults() -> None:
    def fail(_payloads: list[dict]) -> list[dict]:
        raise RuntimeError("native batch fault")

    batcher = ContinuousRequestBatcher(
        fail,
        window_ms=1.0,
        maximum_batch_size=2,
    )
    try:
        try:
            batcher.submit("shared-contract", {"value": 1})
        except RuntimeError as exc:
            assert str(exc) == "native batch fault"
        else:
            raise AssertionError("batch processor fault must reach its caller")
    finally:
        batcher.close()


def test_continuous_request_batcher_never_mixes_incompatible_contracts() -> None:
    observed_batches: list[list[str]] = []

    def process(payloads: list[dict]) -> list[dict]:
        contracts = [str(payload["contract"]) for payload in payloads]
        observed_batches.append(contracts)
        assert len(set(contracts)) == 1
        return [
            {"contract": contract, "runtime_receipt": {}}
            for contract in contracts
        ]

    batcher = ContinuousRequestBatcher(
        process,
        window_ms=20.0,
        maximum_batch_size=8,
    )
    inputs = [("a", "a"), ("b", "b"), ("a", "a"), ("b", "b")]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(
                pool.map(
                    lambda row: batcher.submit(
                        row[0], {"contract": row[1]}
                    ),
                    inputs,
                )
            )
    finally:
        batcher.close()

    assert [output["contract"] for output in outputs] == ["a", "b", "a", "b"]
    assert sorted(sorted(batch) for batch in observed_batches) == [
        ["a", "a"],
        ["b", "b"],
    ]


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


def test_openai_compat_passes_continuous_batch_contract(monkeypatch) -> None:
    observed: dict = {}

    class FakeRuntime:
        def __init__(self, **kwargs: object) -> None:
            observed.update(kwargs)

    monkeypatch.setattr(
        "neural_seed_resident_runtime.NeuralSeedResidentRuntime", FakeRuntime
    )
    runtime = openai_compat_server.initialize_resident_runtime(
        {
            "resident_neural_seed": {
                "enabled": True,
                "continuous_batch_window_ms": 3.5,
                "maximum_request_batch_size": 6,
            }
        }
    )
    assert isinstance(runtime, FakeRuntime)
    assert observed["continuous_batch_window_ms"] == 3.5
    assert observed["maximum_request_batch_size"] == 6
