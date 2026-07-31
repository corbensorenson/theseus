from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_local_inference_backend as backend  # noqa: E402


def write_snapshot(path: Path) -> None:
    path.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (path / name).write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    (path / "model-00001-of-00001.safetensors").write_bytes(b"weights")


def write_contract_files(tmp_path: Path, snapshot: Path) -> tuple[Path, Path]:
    worker = tmp_path / "worker.json"
    worker.write_text(
        json.dumps(
            {
                "policy": "test",
                "model": {
                    "repo_id": "mlx-community/Tmax-9B-MLX-8bit",
                    "revision": "revision-1",
                    "chat_template_kwargs": {"enable_thinking": False},
                    "temperature": 0.0,
                    "maximum_action_tokens": 64,
                    "repetition_penalty": 1.05,
                    "repetition_context_size": 128,
                },
            }
        ),
        encoding="utf-8",
    )
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "trigger_state": "GREEN",
                "model_identity": {
                    "repo_id": "mlx-community/Tmax-9B-MLX-8bit",
                    "revision": "revision-1",
                    "snapshot_manifest_sha256": backend.snapshot_manifest(snapshot),
                },
            }
        ),
        encoding="utf-8",
    )
    return worker, preflight


class FakeModel:
    messages: list[dict[str, str]] = []

    def __init__(self, card: dict, snapshot: Path, maximum_tokens: int) -> None:
        assert card["repo_id"] == "mlx-community/Tmax-9B-MLX-8bit"
        assert snapshot.is_dir()
        assert maximum_tokens == 32
        self.last_generation_metrics = {"generated_tokens": 5}

    def generate(self, messages: list[dict[str, str]]) -> str:
        FakeModel.messages = messages
        return "offline local answer<|im_end|>ignored"


def test_backend_uses_pinned_snapshot_stdin_prompt_and_no_external_inference(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    worker, preflight = write_contract_files(tmp_path, snapshot)
    monkeypatch.setattr(backend, "local_snapshot", lambda _card: snapshot)
    monkeypatch.setattr(backend, "package_versions", lambda: {"mlx_lm": "0.31.3", "mlx": "0.32.0", "python": "3.12"})

    report = backend.run_backend(
        worker_config_path=worker,
        runtime_preflight_path=preflight,
        execution_mode="direct_local_model",
        route_context_digest="d" * 64,
        session_id="test",
        prompt="private prompt",
        maximum_tokens=32,
        model_factory=FakeModel,
    )

    assert report["trigger_state"] == "GREEN"
    assert report["response"]["answer"] == "offline local answer"
    assert report["request"]["prompt_sha256"] == backend.route_integrity.sha256_text("private prompt")
    assert "private prompt" not in json.dumps(report, sort_keys=True)
    assert FakeModel.messages[-1] == {"role": "user", "content": "private prompt"}
    assert report["metrics"]["local_model_inference_calls"] == 1
    assert report["metrics"]["terminal_marker_stripped"] is True
    assert report["external_inference_calls"] == 0
    assert report["teacher_calls"] == 0


def test_generation_terminal_markers_are_removed_at_the_backend_boundary() -> None:
    assert backend.sanitize_generated_text("answer<|im_end|>") == "answer"
    assert backend.sanitize_generated_text("answer<|eot_id|>ignored") == "answer"
    assert backend.sanitize_generated_text(" plain answer ") == "plain answer"


def test_snapshot_manifest_mismatch_fails_before_model_load(tmp_path: Path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    worker, preflight = write_contract_files(tmp_path, snapshot)
    payload = json.loads(preflight.read_text(encoding="utf-8"))
    payload["model_identity"]["snapshot_manifest_sha256"] = "0" * 64
    preflight.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(backend, "local_snapshot", lambda _card: snapshot)
    monkeypatch.setattr(backend, "package_versions", lambda: {"mlx_lm": "0.31.3", "mlx": "0.32.0", "python": "3.12"})

    report = backend.run_backend(
        worker_config_path=worker,
        runtime_preflight_path=preflight,
        execution_mode="direct_local_model",
        route_context_digest="d" * 64,
        session_id="test",
        prompt="private prompt",
        maximum_tokens=32,
        model_factory=lambda *_args: (_ for _ in ()).throw(AssertionError("must not load")),
    )

    assert report["trigger_state"] == "RED"
    assert "local_snapshot_manifest_mismatch" in report["faults"]
    assert report["metrics"]["local_model_inference_calls"] == 0
    assert report["response"]["answer"] == ""


def test_product_token_budget_cannot_exceed_frozen_worker_budget(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    worker, preflight = write_contract_files(tmp_path, snapshot)

    contract = backend.route_integrity.load_model_contract(worker, preflight, maximum_tokens=65)

    assert contract["ready"] is False
    assert "product_maximum_tokens_out_of_worker_bounds" in contract["faults"]


def test_canonical_frozen_identity_cannot_drift_with_a_self_consistent_preflight(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    worker, preflight = write_contract_files(tmp_path, snapshot)

    contract = backend.route_integrity.load_model_contract(
        worker,
        preflight,
        maximum_tokens=32,
        required_repo_id="mlx-community/Tmax-9B-MLX-8bit",
        required_revision="the-canonical-revision",
        required_snapshot_manifest_sha256=backend.snapshot_manifest(snapshot),
    )

    assert contract["ready"] is False
    assert "frozen_model_revision_mismatch" in contract["faults"]
