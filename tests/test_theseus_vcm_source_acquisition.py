from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_source_acquisition as acquisition  # noqa: E402
import theseus_vcm_source_acquisition_v3 as acquisition_v3  # noqa: E402
import theseus_vcm_source_acquisition_v4 as acquisition_v4  # noqa: E402
import theseus_vcm_source_acquisition_v5 as acquisition_v5  # noqa: E402
import theseus_vcm_source_acquisition_v6 as acquisition_v6  # noqa: E402


def test_vcm_source_acquisition_preflight_is_green_and_call_free() -> None:
    report = acquisition.preflight()
    assert report["trigger_state"] == "GREEN"
    assert report["metadata_selection_opened"] is False
    assert report["source_content_retrieval_opened"] is False
    assert report["candidate_packet_materialization_opened"] is False
    assert report["selected_repository_count"] == 0
    assert report["faults"] == []
    assert all(value == 0 for value in report["counters"].values())


def test_vcm_source_acquisition_v2_repairs_only_policy_defects() -> None:
    path = ROOT / "configs" / "theseus_vcm_source_acquisition_v2.json"
    report = acquisition.preflight(path)
    config = p2a.read_json(path)
    assert report["trigger_state"] == "GREEN"
    assert config["selection"]["minimum_repository_stars"] == 1
    assert "GPL-3.0" in config["selection"]["license_spdx_allowlist"]
    assert config["repair"]["predecessor_state"] == "METADATA_SELECTION_INCOMPLETE"
    assert config["repair"]["predecessor_public_metadata_requests"] == 1212
    assert report["metadata_selection_opened"] is False
    assert all(value == 0 for value in report["counters"].values())


def test_vcm_source_acquisition_v3_expands_pool_without_relaxing_filters() -> None:
    report = acquisition_v3.preflight()
    v2 = p2a.read_json(ROOT / "configs" / "theseus_vcm_source_acquisition_v2.json")
    v3 = p2a.read_json(acquisition_v3.DEFAULT_CONFIG)
    assert report["trigger_state"] == "GREEN"
    assert v3["search"]["pages_per_language"] == 10
    assert v3["search"]["maximum_search_metadata_rows"] == 4000
    assert v3["selection"] == v2["selection"]
    assert v3["panels"] == v2["panels"]
    assert v3["chronology"] == v2["chronology"]
    assert v3["authority"] == v2["authority"]
    assert report["metadata_selection_opened"] is False
    assert all(value == 0 for value in report["counters"].values())


def test_vcm_source_acquisition_v4_repairs_transport_only() -> None:
    report = acquisition_v4.preflight()
    v3 = p2a.read_json(acquisition_v3.DEFAULT_CONFIG)
    v4 = p2a.read_json(acquisition_v4.DEFAULT_CONFIG)
    assert report["trigger_state"] == "GREEN"
    assert report["state"] == "METADATA_SELECTION_V4_FORK_SAFE_PREFLIGHT_GREEN"
    assert v4["search"]["pages_per_language"] == v3["search"]["pages_per_language"]
    assert v4["search"]["qualification_workers"] == v3["search"]["qualification_workers"]
    assert v4["selection"] == v3["selection"]
    assert v4["panels"] == v3["panels"]
    assert v4["chronology"] == v3["chronology"]
    assert v4["authority"] == v3["authority"]
    assert report["metadata_selection_opened"] is False
    assert all(value == 0 for value in report["counters"].values())


def test_vcm_source_acquisition_v5_adds_retry_custody_only() -> None:
    report = acquisition_v5.preflight()
    v4 = p2a.read_json(acquisition_v4.DEFAULT_CONFIG)
    v5 = p2a.read_json(acquisition_v5.DEFAULT_CONFIG)
    assert report["trigger_state"] == "GREEN"
    assert report["state"] == "METADATA_SELECTION_V5_RETRY_CHECKPOINT_PREFLIGHT_GREEN"
    assert v5["search"] == v4["search"]
    assert v5["selection"] == v4["selection"]
    assert v5["panels"] == v4["panels"]
    assert v5["chronology"] == v4["chronology"]
    assert v5["authority"] == v4["authority"]
    assert v5["transport_retry_policy"]["maximum_attempts_per_logical_request"] == 4
    assert v5["checkpoint_policy"]["candidate_identities_retained"] is False
    assert report["metadata_selection_opened"] is False
    assert all(value == 0 for value in report["counters"].values())


def test_v5_retry_client_records_transient_retry_without_identity(tmp_path: Path) -> None:
    calls = 0

    def flaky(resource: str, fields: dict[str, object]) -> tuple[dict[str, bool], str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise acquisition_v5.subprocess.CalledProcessError(
                1, ["gh"], stderr="connection reset"
            )
        return {"ok": True}, "a" * 64

    policy = {
        "maximum_attempts_per_logical_request": 4,
        "retry_delays_seconds": [0.0, 0.0, 0.0],
    }
    ledger = acquisition_v5.RequestLedger(
        tmp_path / "checkpoint.json", acquisition_v5.DEFAULT_CONFIG, policy
    )
    client = acquisition_v5.RetryingClient(flaky, ledger, policy)
    payload, digest = client.call("repos/example/project", {})
    assert payload == {"ok": True}
    assert digest == "a" * 64
    summary = ledger.summary()
    assert summary["logical_request_count"] == 1
    assert summary["physical_attempt_count"] == 2
    assert summary["retry_attempt_count"] == 1
    checkpoint = p2a.read_json(tmp_path / "checkpoint.json")
    assert checkpoint["selected_source_identities_retained"] is False
    assert "example/project" not in json.dumps(checkpoint)


def test_v5_retry_client_rejects_permanent_candidate_404_without_retry(tmp_path: Path) -> None:
    def missing(resource: str, fields: dict[str, object]) -> tuple[dict[str, bool], str]:
        raise acquisition_v5.subprocess.CalledProcessError(
            1, ["gh"], stderr="gh: Not Found (HTTP 404)"
        )

    policy = {
        "maximum_attempts_per_logical_request": 4,
        "retry_delays_seconds": [0.0, 0.0, 0.0],
    }
    ledger = acquisition_v5.RequestLedger(
        tmp_path / "checkpoint.json", acquisition_v5.DEFAULT_CONFIG, policy
    )
    client = acquisition_v5.RetryingClient(missing, ledger, policy)
    try:
        client.call("repos/missing/project", {})
    except acquisition_v5.CandidateMetadataUnavailable as exc:
        assert exc.status == 404
    else:
        raise AssertionError("permanent candidate metadata failure was not raised")
    summary = ledger.summary()
    assert summary["physical_attempt_count"] == 1
    assert summary["retry_attempt_count"] == 0
    assert summary["permanent_candidate_failure_count"] == 1


def test_vcm_source_acquisition_v6_batches_transport_only() -> None:
    report = acquisition_v6.preflight()
    v5 = p2a.read_json(acquisition_v5.DEFAULT_CONFIG)
    v6 = p2a.read_json(acquisition_v6.DEFAULT_CONFIG)
    assert report["trigger_state"] == "GREEN"
    assert report["state"] == "METADATA_SELECTION_V6_GRAPHQL_BATCH_LIVE_SCHEMA_QUALIFIED"
    assert v6["selection"] == v5["selection"]
    assert v6["panels"] == v5["panels"]
    assert v6["chronology"] == v5["chronology"]
    assert v6["authority"] == v5["authority"]
    assert v6["graphql_transport"]["node_batch_size"] == 40
    assert v6["graphql_transport"]["maximum_parallel_graphql_requests"] == 1
    lowered = acquisition_v6.GRAPHQL_QUERY.lower()
    assert "body" not in lowered
    assert "patch" not in lowered
    assert "reviews" not in lowered


def test_v6_graphql_fixture_preserves_existing_eligibility() -> None:
    candidate = {
        "repository": "owner/repo",
        "pull_request": 7,
        "query_language": "Python",
        "node_id": "PR_fixture",
        "rank": "f" * 64,
    }
    node = {
        "__typename": "PullRequest",
        "id": "PR_fixture",
        "number": 7,
        "url": "https://github.com/owner/repo/pull/7",
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
            "nameWithOwner": "owner/repo",
            "isFork": False,
            "isArchived": False,
            "isDisabled": False,
            "stargazerCount": 500,
            "primaryLanguage": {"name": "Python"},
            "licenseInfo": {"spdxId": "MIT"},
        },
        "files": {"nodes": [
            {"path": "src/module.py", "changeType": "MODIFIED"},
            {"path": "tests/test_module.py", "changeType": "MODIFIED"},
        ]},
        "commits": {"nodes": [
            {"commit": {"oid": "b" * 40, "committedDate": "2026-08-01T01:00:00Z"}}
        ]},
    }
    config = p2a.read_json(acquisition_v6.DEFAULT_CONFIG)
    row, reasons = acquisition_v6.qualify_node(node, candidate, config)
    assert reasons == []
    assert row["metadata_qualified"] is True
    assert row["head_chronology_source"] == "graphql_pull_request_commit_connection"
    assert row["source_paths"] == ["src/module.py"]
    assert row["verifier_paths"] == ["tests/test_module.py"]
    assert row["candidate_content_retrieved"] is False


def test_v6_graphql_fixture_rejects_head_commit_identity_mismatch() -> None:
    candidate = {
        "repository": "owner/repo",
        "pull_request": 7,
        "query_language": "Python",
        "node_id": "PR_fixture",
        "rank": "f" * 64,
    }
    node = {
        "__typename": "PullRequest",
        "id": "PR_fixture",
        "number": 7,
        "url": "https://github.com/owner/repo/pull/7",
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
            "nameWithOwner": "owner/repo",
            "isFork": False,
            "isArchived": False,
            "isDisabled": False,
            "stargazerCount": 500,
            "primaryLanguage": {"name": "Python"},
            "licenseInfo": {"spdxId": "MIT"},
        },
        "files": {"nodes": [
            {"path": "src/module.py", "changeType": "MODIFIED"},
            {"path": "tests/test_module.py", "changeType": "MODIFIED"},
        ]},
        "commits": {"nodes": [
            {"commit": {"oid": "d" * 40, "committedDate": "2026-08-01T01:00:00Z"}}
        ]},
    }
    config = p2a.read_json(acquisition_v6.DEFAULT_CONFIG)
    row, reasons = acquisition_v6.qualify_node(node, candidate, config)
    assert row["metadata_qualified"] is False
    assert "head_commit_identity_mismatch" in reasons


def test_panel_quotas_are_exact_and_source_disjoint_by_construction() -> None:
    config = p2a.read_json(acquisition.DEFAULT_CONFIG)
    panels = config["panels"]
    assert panels["control_qualification"]["task_count"] == 9
    assert panels["claim"]["task_count"] == 53
    assert panels["total_task_count"] == 62
    assert sum(panels["control_qualification"]["language_quotas"].values()) == 9
    assert sum(panels["claim"]["language_quotas"].values()) == 53
    assert panels["source_disjoint"] is True
    assert panels["reference_outputs_may_select_or_assign_tasks"] is False


def test_metadata_filter_rejects_missing_verifier_and_stale_head() -> None:
    config = p2a.read_json(acquisition.DEFAULT_CONFIG)
    candidate = {"repository": "owner/repo", "pull_request": 7, "query_language": "Python"}
    pr = {
        "state": "closed", "merged_at": "2026-08-01T00:00:00Z",
        "created_at": "2026-08-01T00:00:00Z", "draft": False,
        "changed_files": 2, "additions": 5, "deletions": 1,
        "user": {"login": "human"},
    }
    repo = {
        "fork": False, "archived": False, "disabled": False,
        "language": "Python", "stargazers_count": 500,
        "license": {"spdx_id": "MIT"},
    }
    files = [{"filename": "src/module.py", "status": "modified"}]
    head = {"commit": {"committer": {"date": "2026-07-29T00:00:00Z"}}}
    reasons = acquisition.metadata_rejection_reasons(candidate, pr, repo, files, head, config)
    assert "no_machine_verifier_change" in reasons
    assert "chronology" in reasons


def test_authority_opens_metadata_only_without_model_or_user_gate() -> None:
    authority = p2a.read_json(acquisition.DEFAULT_CONFIG)["authority"]
    assert authority["public_metadata_queries_authorized_after_green_preflight"] is True
    assert authority["public_source_content_retrieval_authorized"] is False
    assert authority["candidate_packet_materialization_authorized"] is False
    assert authority["local_model_calls_authorized"] == 0
    assert authority["external_reference_calls_authorized"] == 0
    assert authority["hidden_evaluation_authorized"] is False
    assert authority["teacher_calls_authorized"] is False
    assert authority["training_rows_authorized"] is False
    assert authority["user_or_operator_gate"] is False
