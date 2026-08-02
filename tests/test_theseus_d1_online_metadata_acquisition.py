from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_d1_online_metadata_acquisition as acquisition  # noqa: E402


CONFIG = ROOT / "configs" / "theseus_d1_source_selection.json"


def survivor() -> dict[str, object]:
    return {
        "created_utc": "2026-08-01T12:00:00Z",
        "trigger_state": "GREEN",
        "policy": "project_theseus_p4v2r2r2_terminal_disposition_v1",
        "scientific_status": "P4V2R2R2_DEVELOPMENT_SURVIVOR_D1_ELIGIBLE",
        "claim_id": "cognitive-compilation-and-semantic-ir.core",
        "consumption": {"eligible_for_D1": True},
        "decision_rule": {
            "survivor_effect_rule_passed": True,
            "effect_decision_authorized": True,
        },
    }


def test_no_survivor_never_authorizes_network() -> None:
    report = acquisition.preflight(
        CONFIG,
        disposition_override={},
        now_override=datetime(2026, 8, 2, 12, tzinfo=timezone.utc),
    )
    assert report["trigger_state"] == "PAUSED"
    assert report["network_acquisition_authorized"] is False
    assert report["network_calls"] == 0
    assert report["archive_fetches"] == 0
    assert report["candidate_or_control_calls"] == 0


def test_survivor_opens_metadata_transport_only() -> None:
    report = acquisition.preflight(
        CONFIG,
        disposition_override=survivor(),
        now_override=datetime(2026, 8, 2, 12, tzinfo=timezone.utc),
    )
    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["network_acquisition_authorized"] is True
    assert report["complete_interval_count"] == 3
    assert report["archive_fetches"] == 0
    assert report["parent_target_oracle_or_evaluator_executions"] == 0
    assert report["candidate_or_control_calls"] == 0


def test_complete_intervals_exclude_current_partial_UTC_day() -> None:
    observed = datetime(2026, 7, 30, 4, 53, 3, tzinfo=timezone.utc)
    current = datetime(2026, 8, 2, 12, tzinfo=timezone.utc)
    intervals = acquisition.complete_utc_intervals(observed, current)
    assert [row["start_date"] for row in intervals] == [
        "2026-07-30",
        "2026-07-31",
        "2026-08-01",
    ]
    assert intervals[0]["start_utc"].startswith("2026-07-30T04:53:03")
    assert intervals[-1]["end_utc"] == "2026-08-02T00:00:00Z"


class SearchClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    def get(self, path: str, parameters: dict[str, object] | None = None) -> object:
        self.calls.append((path, parameters))
        return self.payloads.pop(0)


def test_search_partition_requires_complete_reported_count() -> None:
    issues = [
        {"id": index, "pull_request": {"url": f"https://api.github.com/repos/o/r/pulls/{index}"}}
        for index in range(101)
    ]
    client = SearchClient([
        {"total_count": 101, "items": issues[:100]},
        {"total_count": 101, "items": issues[100:]},
    ])
    partition, rows = acquisition.fetch_search_partition(
        client,
        language="python",
        title_term="fix",
        start_date="2026-08-01",
        end_date="2026-08-01",
    )
    assert partition["complete"] is True
    assert partition["reported_total_count"] == 101
    assert partition["retrieved_issue_count"] == 101
    assert partition["page_count"] == 2
    assert len(rows) == 101


class CandidateClient:
    def __init__(self) -> None:
        self.request_count = 0
        self.response_digests: list[str] = []

    def get(self, path: str, parameters: dict[str, object] | None = None) -> object:
        self.request_count += 1
        if path == "/repos/example/project":
            return {"language": "Python", "license": {"spdx_id": "MIT"}}
        if path.endswith("/commits/" + "b" * 40):
            return {"parents": [{"sha": "a" * 40}]}
        if path.endswith("/pulls/7/files"):
            return [
                {
                    "filename": "package/core.py",
                    "status": "modified",
                    "additions": 3,
                    "deletions": 1,
                    "changes": 4,
                },
                {
                    "filename": "tests/test_core.py",
                    "status": "added",
                    "additions": 9,
                    "deletions": 0,
                    "changes": 9,
                },
            ]
        raise AssertionError(path)


def test_candidate_metadata_uses_merge_parent_and_complete_file_inventory() -> None:
    pull = {
        "number": 7,
        "title": "Fix edge behavior",
        "merged_at": "2026-08-01T13:00:00Z",
        "merge_commit_sha": "b" * 40,
        "changed_files": 2,
        "base": {"repo": {"full_name": "example/project"}},
    }
    row = acquisition.fetch_candidate_row(
        CandidateClient(),
        pull,
        metadata_retrieved_utc="2026-08-02T00:00:00Z",
    )
    assert row is not None
    assert row["repository"] == "example/project"
    assert row["parent_revision"] == "a" * 40
    assert row["target_revision"] == "b" * 40
    assert row["license_spdx"] == "MIT"
    assert row["changed_paths"] == ["package/core.py", "tests/test_core.py"]
    assert row["changed_files"] == [
        {
            "filename": "package/core.py",
            "status": "modified",
            "previous_filename": "",
            "additions": 3,
            "deletions": 1,
            "changes": 4,
        },
        {
            "filename": "tests/test_core.py",
            "status": "added",
            "previous_filename": "",
            "additions": 9,
            "deletions": 0,
            "changes": 9,
        },
    ]
    assert "solution" not in row
    assert "tests" not in row
