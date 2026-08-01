from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_p4s_source_membership_is_fresh_frozen_and_zero_call() -> None:
    registry = json.loads(
        (ROOT / "configs" / "theseus_p4s_task_sources.json").read_text(encoding="utf-8")
    )
    rows = registry["tasks"]
    boundaries = registry["boundaries"]
    prior = {value.lower() for value in registry["source_disjoint_from_repositories"]}

    assert registry["state"] == (
        "FIXED_BEFORE_ARCHIVE_FETCH_PARENT_TARGET_EXECUTION_OR_CANDIDATE_GENERATION"
    )
    assert registry["instrument_freeze_commit"] == "42abb39b"
    assert registry["task_count"] == registry["distinct_repository_count"] == 10
    assert [row["campaign_index"] for row in rows] == list(range(1, 11))
    assert len({row["repository"].lower() for row in rows}) == 10
    assert not {row["repository"].lower() for row in rows}.intersection(prior)
    assert boundaries["candidate_generation_opened"] is False
    assert boundaries["archive_fetches_after_membership_freeze"] == 0
    assert boundaries["parent_target_oracle_executions"] == 0
    assert boundaries["local_model_calls"] == 0
    assert boundaries["hosted_model_calls"] == 0
    assert boundaries["deterministic_request_compiler_calls"] == 0
    assert boundaries["user_task_label_or_approval_dependency"] is False


def test_p4s_sources_bind_license_information_flow_and_semantic_units() -> None:
    registry = json.loads(
        (ROOT / "configs" / "theseus_p4s_task_sources.json").read_text(encoding="utf-8")
    )

    for row in registry["tasks"]:
        assert row["license_spdx"] in {"Apache-2.0", "BSD-3-Clause", "MIT"}
        assert row["license_paths"]
        assert len(row["parent_revision"]) == len(row["target_revision"]) == 40
        assert row["natural_request"]
        assert [item["id"] for item in row["obligations"]] == ["O1", "O2", "O3"]
        assert row["reads"] and row["searches"] and row["visible_markers"]
        assert row["oracle_units"]
        assert all(unit["path"] in row["allowed_effect_paths"] for unit in row["oracle_units"])
        assert all(unit["operation"] == "REPLACE" for unit in row["oracle_units"])
