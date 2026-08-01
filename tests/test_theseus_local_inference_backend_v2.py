from __future__ import annotations

import json
import hashlib
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_local_inference_backend_v2 as backend  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_snapshot(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps({"text_config": {"max_position_embeddings": 64}}),
        encoding="utf-8",
    )
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (path / name).write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    (path / "model-00001-of-00001.safetensors").write_bytes(b"weights")


def write_contract_files(tmp_path: Path, snapshot: Path) -> tuple[Path, Path]:
    worker = tmp_path / "worker.json"
    worker.write_text(
        json.dumps({
            "policy": "test-successor",
            "model": {
                "repo_id": "mlx-community/Tmax-9B-MLX-8bit",
                "revision": "revision-1",
                "chat_template_kwargs": {"enable_thinking": False},
                "temperature": 0.0,
                "maximum_action_tokens": 64,
                "repetition_penalty": 1.05,
                "repetition_context_size": 128,
            },
            "generation_boundary": {
                "policy": "configs/theseus_generation_completion_policy.json",
                "numeric_ceiling_source": "model_declared_context_window",
                "model_declared_context_window_tokens": 64,
                "project_selected_quality_token_cap": None,
                "ceiling_hit_is_instrument_invalid": True,
            },
        }),
        encoding="utf-8",
    )
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps({
            "trigger_state": "GREEN",
            "model_identity": {
                "repo_id": "mlx-community/Tmax-9B-MLX-8bit",
                "revision": "revision-1",
                "snapshot_manifest_sha256": backend.snapshot_manifest(snapshot),
            },
        }),
        encoding="utf-8",
    )
    return worker, preflight


def test_canonical_completion_worker_binds_no_quality_cap_to_route_identity() -> None:
    contract = backend.route_integrity.load_model_contract(
        ROOT / "configs" / "core_evidence_tmax_9b_completion_worker.json",
        ROOT / "reports" / "core_evidence_tmax_9b_runtime_preflight.json",
        maximum_tokens=0,
    )

    assert contract["ready"] is True
    boundary = contract["identity"]["generation_boundary"]
    assert boundary["model_declared_context_window_tokens"] == 262144
    assert boundary["project_selected_quality_token_cap"] is None
    assert boundary["ceiling_hit_is_instrument_invalid"] is True


def test_canonical_runtime_source_binds_successor_backend_route_and_worker() -> None:
    config = json.loads(
        (ROOT / "configs" / "theseus_assistant_runtime.json").read_text(
            encoding="utf-8"
        )
    )["local_inference"]

    for path_field, digest_field in (
        ("backend_entrypoint", "backend_entrypoint_sha256"),
        ("route_integrity_entrypoint", "route_integrity_entrypoint_sha256"),
        (
            "base_route_integrity_entrypoint",
            "base_route_integrity_entrypoint_sha256",
        ),
        ("worker_config", "worker_config_sha256"),
    ):
        assert file_sha256(ROOT / config[path_field]) == config[digest_field]


def test_exact_prompt_residual_and_physical_boundary_are_explicit(monkeypatch) -> None:
    class FakeTokenizer:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> list[int]:
            return [1, 2, 3]

    class Response:
        text = "unfinished"
        finish_reason = "length"
        peak_memory = 0.0
        prompt_tps = 1.0
        generation_tps = 1.0

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        types.SimpleNamespace(stream_generate=lambda *_args, **_kwargs: iter([Response()])),
    )
    model = backend.LocalMlxChatModel.__new__(backend.LocalMlxChatModel)
    model.model = object()
    model.tokenizer = FakeTokenizer()
    model.card = {"chat_template_kwargs": {}}
    model.maximum_tokens = 64
    model.model_context_window_tokens = 64
    model.sampler = None
    model.logits_processors = []
    model.generation_cache_kwargs = {}
    model.load_wall_ms = 0.0
    model.completion_predicate = None

    model.generate([{"role": "user", "content": "prompt"}])

    metrics = model.last_generation_metrics
    assert metrics["termination_reason"] == "physical_context_boundary"
    assert metrics["exact_prompt_tokens"] == 3
    assert metrics["effective_context_residual_tokens"] == 61
    assert metrics["project_selected_quality_token_cap"] is None


def test_physical_boundary_output_fails_closed(tmp_path: Path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    worker, preflight = write_contract_files(tmp_path, snapshot)
    monkeypatch.setattr(backend, "local_snapshot", lambda _card: snapshot)
    monkeypatch.setattr(
        backend,
        "package_versions",
        lambda: {"mlx_lm": "0.31.3", "mlx": "0.32.0", "python": "3.12"},
    )

    class BoundaryModel:
        def __init__(self, *_args: object) -> None:
            self.last_generation_metrics = {
                "termination_reason": "physical_context_boundary",
                "physical_context_boundary_hit": True,
            }

        def generate(self, _messages: list[dict[str, str]]) -> str:
            return "truncated text"

    report = backend.run_backend(
        worker_config_path=worker,
        runtime_preflight_path=preflight,
        execution_mode="direct_local_model",
        route_context_digest="d" * 64,
        session_id="test",
        prompt="private prompt",
        maximum_tokens=0,
        model_factory=BoundaryModel,
    )

    assert report["trigger_state"] == "RED"
    assert "instrument_inadequate_generation_boundary_hit" in report["faults"]
    assert report["response"]["answer"] == ""


def test_prompt_exhaustion_fails_before_generation(monkeypatch) -> None:
    class FakeTokenizer:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> list[int]:
            return [1, 2, 3]

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        types.SimpleNamespace(
            stream_generate=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("generation must not start")
            )
        ),
    )
    model = backend.LocalMlxChatModel.__new__(backend.LocalMlxChatModel)
    model.model = object()
    model.tokenizer = FakeTokenizer()
    model.card = {"chat_template_kwargs": {}}
    model.maximum_tokens = 64
    model.model_context_window_tokens = 3
    model.sampler = None
    model.logits_processors = []
    model.generation_cache_kwargs = {}
    model.load_wall_ms = 0.0
    model.completion_predicate = None

    try:
        model.generate([{"role": "user", "content": "prompt"}])
    except backend.BackendFault as exc:
        assert str(exc) == "prompt_exhausts_model_context_window"
    else:
        raise AssertionError("expected context exhaustion fault")
