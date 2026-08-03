from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_source_acquisition_v5 as acquisition_v5  # noqa: E402
import theseus_vcm_source_materialization as materialize  # noqa: E402
import theseus_vcm_source_replacement_28 as replacement  # noqa: E402


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


def test_task_28_replacement_preflight_is_single_slot_and_call_free() -> None:
    report = replacement.preflight()
    config = p2a.read_json(replacement.DEFAULT_CONFIG)
    policy = config["replacement_policy"]
    assert report["trigger_state"] == "GREEN"
    assert report["state"] == "TASK_28_PYTHON_CLAIM_REPLACEMENT_PREFLIGHT_GREEN"
    assert policy["task_index"] == 28
    assert policy["panel"] == "claim"
    assert policy["language"] == "Python"
    assert policy["all_v7_repositories_excluded"] is True
    assert policy["first_ranked_content_qualified_candidate_required"] is True
    assert report["replacement_admitted"] is False
    assert report["source_content_retrieval_opened"] is False
    assert all(value == 0 for value in report["counters"].values())


def test_task_28_replacement_skips_unchanged_verifier_then_takes_next_rank(
    tmp_path: Path, monkeypatch,
) -> None:
    config = p2a.read_json(replacement.DEFAULT_CONFIG)
    config["output_directory"] = str(tmp_path / "archives")
    config_path = tmp_path / "replacement.json"
    p2a.write_json(config_path, config)
    nodes: dict[str, dict[str, object]] = {}

    def fake_rest(
        resource: str, fields: dict[str, object]
    ) -> tuple[dict[str, object], str]:
        assert resource == "search/issues"
        items = []
        for index in range(40):
            repository = f"replacement-fixture/repo-{index:02d}"
            node_id = f"PR_replacement_{index}"
            title = f"Replacement fixture {index}"
            items.append(
                {
                    "repository_url": f"https://api.github.com/repos/{repository}",
                    "number": index + 1,
                    "node_id": node_id,
                    "title": title,
                }
            )
            nodes[node_id] = {
                "__typename": "PullRequest",
                "id": node_id,
                "number": index + 1,
                "url": f"https://github.com/{repository}/pull/{index + 1}",
                "state": "MERGED",
                "isDraft": False,
                "createdAt": "2026-08-01T00:00:00Z",
                "mergedAt": "2026-08-01T02:00:00Z",
                "additions": 5,
                "deletions": 1,
                "changedFiles": 2,
                "baseRefOid": "a" * 40,
                "headRefOid": "b" * 40,
                "mergeCommit": {"oid": "c" * 40},
                "author": {"login": "human"},
                "repository": {
                    "nameWithOwner": repository,
                    "isFork": False,
                    "isArchived": False,
                    "isDisabled": False,
                    "stargazerCount": 5,
                    "primaryLanguage": {"name": "Python"},
                    "licenseInfo": {"spdxId": "MIT"},
                },
                "files": {"nodes": [
                    {"path": "src/module.py", "changeType": "MODIFIED"},
                    {"path": "tests/test_module.py", "changeType": "MODIFIED"},
                ]},
                "commits": {"nodes": [
                    {
                        "commit": {
                            "oid": "b" * 40,
                            "committedDate": "2026-08-01T01:00:00Z",
                        }
                    }
                ]},
            }
        return {"items": items}, "a" * 64

    def fake_graphql(
        resource: str, fields: dict[str, object]
    ) -> tuple[dict[str, object], str]:
        return {
            "data": {"nodes": [nodes[str(node_id)] for node_id in fields["ids"]]}
        }, "b" * 64

    class FakeSourceClient:
        def __init__(self) -> None:
            self.title_requests = 0
            self.source_requests = 0
            self.attempted_repositories: list[str] = []

        def title(self, repository: str, number: int) -> str:
            self.title_requests += 1
            self.attempted_repositories.append(repository)
            index = int(repository.rsplit("-", 1)[1])
            return f"Replacement fixture {index}"

        def license(self, repository: str, revision: str) -> tuple[str, bytes]:
            self.source_requests += 1
            return "LICENSE", b"MIT fixture\n"

        def file(self, repository: str, revision: str, path: str) -> bytes | None:
            self.source_requests += 1
            first = repository == self.attempted_repositories[0]
            if path == "src/module.py":
                return b"value = 1\n" if revision == "a" * 40 else b"value = 2\n"
            if first:
                return b"assert value\n"
            return b"assert value == 1\n" if revision == "a" * 40 else b"assert value == 2\n"

    monkeypatch.setattr(replacement.v1, "api_json", fake_rest)
    monkeypatch.setattr(replacement.v6, "graphql_api", fake_graphql)
    monkeypatch.setattr(replacement.time, "sleep", lambda _seconds: None)
    retry_policy = config["transport_retry_policy"]
    ledger = materialize.SourceLedger(
        tmp_path / "checkpoint.json", config_path, retry_policy
    )
    client = FakeSourceClient()
    report = replacement.acquire(config_path, ledger, client, retry_policy)
    assert report["trigger_state"] == "GREEN"
    assert report["replacement_admitted"] is True
    assert report["replacement_materialization"]["index"] == 28
    assert report["replacement_materialization"]["panel"] == "claim"
    assert report["replacement_materialization"]["query_language"] == "Python"
    assert report["rejection_counts"]["selected_verifier_bytes_unchanged"] == 1
    assert len(client.attempted_repositories) == 2
    assert set(report["replacement_materialization"]["archives"]) == {
        "parent_source", "target_source", "parent_verifier", "target_verifier"
    }
