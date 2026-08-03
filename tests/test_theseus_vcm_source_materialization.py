from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_source_materialization as materialize  # noqa: E402


def test_source_materialization_preflight_is_green_and_execution_free() -> None:
    report = materialize.preflight()
    assert report["trigger_state"] == "GREEN"
    assert report["state"] == "VCM_SOURCE_MATERIALIZATION_PREFLIGHT_GREEN"
    assert report["selected_repository_count"] == 62
    assert report["source_content_retrieval_opened"] is False
    assert report["candidate_packet_materialization_opened"] is False
    assert report["hidden_evaluation_opened"] is False
    assert all(value == 0 for value in report["counters"].values())


def test_title_query_cannot_request_answer_bearing_surfaces() -> None:
    lowered = materialize.TITLE_QUERY.lower()
    assert "title" in lowered
    for forbidden in ("body", "patch", "review", "comment", "files", "commits"):
        assert forbidden not in lowered


def test_decode_content_checks_identity_and_base64() -> None:
    content = b"print('ok')\n"
    payload = {
        "type": "file",
        "path": "src/example.py",
        "content": base64.b64encode(content).decode(),
    }
    assert materialize.decode_content(payload, expected_path="src/example.py") == content


def test_materialize_row_writes_four_deterministic_execution_free_archives(
    tmp_path: Path,
) -> None:
    title = "Repair deterministic behavior"
    selected = {
        "opaque_source_id": "opaque",
        "repository": "owner/repo",
        "pull_request": 7,
        "panel": "claim",
        "query_language": "Python",
        "title_sha256": hashlib.sha256(title.encode()).hexdigest(),
        "base_revision": "a" * 40,
        "head_revision": "b" * 40,
        "license_spdx": "MIT",
        "source_paths": ["src/module.py"],
        "verifier_paths": ["tests/test_module.py"],
    }

    class FakeClient:
        def title(self, repository: str, number: int) -> str:
            assert (repository, number) == ("owner/repo", 7)
            return title

        def license(self, repository: str, revision: str) -> tuple[str, bytes]:
            assert repository == "owner/repo"
            return "LICENSE", b"MIT fixture\n"

        def file(self, repository: str, revision: str, path: str) -> bytes | None:
            values = {
                ("a" * 40, "src/module.py"): b"value = 1\n",
                ("b" * 40, "src/module.py"): b"value = 2\n",
                ("a" * 40, "tests/test_module.py"): b"assert value == 1\n",
                ("b" * 40, "tests/test_module.py"): b"assert value == 2\n",
            }
            return values[(revision, path)]

    first, faults, _bytes = materialize.materialize_row(
        selected, 1, tmp_path, FakeClient(), 2 * 1024 * 1024
    )
    first_hashes = {
        key: value["sha256"] for key, value in first["archives"].items()
    }
    second, faults_again, _bytes_again = materialize.materialize_row(
        selected, 1, tmp_path, FakeClient(), 2 * 1024 * 1024
    )
    assert faults == faults_again == []
    assert set(first["archives"]) == {
        "parent_source", "target_source", "parent_verifier", "target_verifier"
    }
    assert first_hashes == {
        key: value["sha256"] for key, value in second["archives"].items()
    }
    assert first["natural_language_request"] == title


def test_source_checkpoint_is_finalized_before_report_hash(tmp_path: Path) -> None:
    config = p2a.read_json(materialize.DEFAULT_CONFIG)
    policy = config["transport_retry_policy"]
    ledger = materialize.SourceLedger(
        tmp_path / "checkpoint.json", materialize.DEFAULT_CONFIG, policy
    )
    report = {
        "state": "FIXTURE_TERMINAL",
        "selected_repository_count": 62,
        "counters": materialize.zero_counters(),
    }
    finalized = materialize.finalize_receipt(
        report, ledger, SimpleNamespace(title_requests=2, source_requests=3)
    )
    assert finalized["checkpoint_artifact_hash_verified_final"] is True
    assert finalized["checkpoint"]["sha256"] == p2a.sha256_file(ledger.path)
    assert p2a.read_json(ledger.path)["state"] == "FIXTURE_TERMINAL"
    assert finalized["counters"]["public_metadata_title_requests"] == 2
    assert finalized["counters"]["public_source_content_requests"] == 3


def test_source_authority_opens_retrieval_only() -> None:
    authority = p2a.read_json(materialize.DEFAULT_CONFIG)["authority"]
    assert authority["public_source_file_retrieval_authorized"] is True
    assert authority["public_pr_title_metadata_retrieval_authorized"] is True
    assert all(
        value is False
        for key, value in authority.items()
        if key not in {
            "public_source_file_retrieval_authorized",
            "public_pr_title_metadata_retrieval_authorized",
        }
    )
