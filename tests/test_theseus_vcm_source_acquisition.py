from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_source_acquisition as acquisition  # noqa: E402


def test_vcm_source_acquisition_preflight_is_green_and_call_free() -> None:
    report = acquisition.preflight()
    assert report["trigger_state"] == "GREEN"
    assert report["metadata_selection_opened"] is False
    assert report["source_content_retrieval_opened"] is False
    assert report["candidate_packet_materialization_opened"] is False
    assert report["selected_repository_count"] == 0
    assert report["faults"] == []
    assert all(value == 0 for value in report["counters"].values())


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
