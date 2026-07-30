import json
from pathlib import Path

import pytest

from scripts import core_evidence_local_model_preflight as preflight


def test_expected_snapshot_binds_repo_and_revision() -> None:
    path = preflight.expected_snapshot({
        "repo_id": "mlx-community/gpt-oss-20b-MXFP4-Q8",
        "revision": "abc123",
    })

    assert path.parts[-3:] == (
        "models--mlx-community--gpt-oss-20b-MXFP4-Q8",
        "snapshots",
        "abc123",
    )


def test_snapshot_logical_bytes_follows_cached_blob_symlinks(
    tmp_path: Path,
) -> None:
    blob = tmp_path / "blob"
    blob.write_bytes(b"1234567")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").symlink_to(blob)
    (snapshot / "README.md").write_bytes(b"abc")
    (snapshot / "subdir").mkdir()

    assert preflight.snapshot_logical_bytes(snapshot) == 10


@pytest.mark.parametrize(
    ("model_factory", "expected_state", "expected_calls"),
    [
        (
            lambda _card: (_ for _ in ()).throw(RuntimeError("load failed")),
            "RED_LOAD_FAILURE",
            0,
        ),
        (
            lambda _card: FailingGenerationModel(),
            "RED_GENERATION_FAILURE",
            1,
        ),
    ],
)
def test_runtime_failures_still_emit_red_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_factory: object,
    expected_state: str,
    expected_calls: int,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "model": {
            "repo_id": "owner/model",
            "revision": "revision",
            "maximum_action_tokens": 96,
        },
    }))
    monkeypatch.setattr(preflight, "expected_snapshot", lambda _card: snapshot)
    monkeypatch.setattr(preflight, "LocalMlxModel", model_factory)
    monkeypatch.setattr(preflight, "swap_used_mib", lambda: 4.0)
    monkeypatch.setattr(preflight, "sysctl_int", lambda _name: 16)

    report = preflight.run(config)

    assert report["trigger_state"] == expected_state
    assert report["output"]["exact_action_valid"] is False
    assert (
        report["counters"]["local_model_inference_calls"]
        == expected_calls
    )
    assert report["maximum_inference"].endswith(
        "not repository competence evidence."
    )


class FailingGenerationModel:
    snapshot_manifest_sha256 = "manifest"
    last_generation_metrics = {"mlx_peak_memory_gib": 9.5}

    def generate(self, _messages: object) -> str:
        raise RuntimeError("generation failed")
