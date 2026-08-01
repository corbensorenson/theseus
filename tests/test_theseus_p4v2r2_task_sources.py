from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "theseus_p4v2r2_task_sources.json"
INSTRUMENT = ROOT / "configs" / "theseus_p4v2r2_cognitive_compilation_instrument.json"


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_p4v2r2_source_membership_is_prospectively_fixed_and_zero_call() -> None:
    registry = load_registry()
    rows = registry["tasks"]
    boundaries = registry["boundaries"]
    prior = {value.lower() for value in registry["source_disjoint_from_repositories"]}

    assert registry["policy"] == "project_theseus_p4v2r2_online_source_selection_v1"
    assert registry["state"] == (
        "FIXED_BEFORE_ARCHIVE_FETCH_PARENT_TARGET_EXECUTION_OR_CANDIDATE_GENERATION"
    )
    assert registry["instrument_freeze_commit"] == (
        "10b293eb7972903f11e40f7fcc18be4a437470b7"
    )
    assert registry["instrument_sha256"] == hashlib.sha256(
        INSTRUMENT.read_bytes()
    ).hexdigest()
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
    assert boundaries["teacher_calls"] == 0
    assert boundaries["public_benchmark_cases"] == 0
    assert boundaries["training_rows_written"] == 0
    assert boundaries["D1_cases_consumed"] == boundaries["D2_cases_consumed"] == 0
    assert boundaries["user_task_label_or_approval_dependency"] is False


def test_p4v2r2_sources_bind_licenses_information_flow_and_semantic_units() -> None:
    registry = load_registry()
    allowed_licenses = {"BSD-3-Clause", "MIT", "MIT-CMU", "MPL-2.0"}

    for row in registry["tasks"]:
        assert row["license_spdx"] in allowed_licenses
        assert row["license_paths"]
        assert len(row["parent_revision"]) == len(row["target_revision"]) == 40
        assert len(row["merge_revision"]) == 40
        assert row["natural_request"]
        assert [item["id"] for item in row["obligations"]] == ["O1", "O2", "O3"]
        assert row["obligation_dependencies"]
        assert row["reads"] and row["searches"] and row["visible_markers"]
        assert row["oracle_units"]
        assert len(row["oracle_units"]) <= 8
        assert all(
            unit["path"] in row["allowed_effect_paths"]
            for unit in row["oracle_units"]
        )
        assert all(
            unit["operation"] in {"INSERT_AFTER", "REPLACE"}
            for unit in row["oracle_units"]
        )
        assert all(unit["target_selectors"] for unit in row["oracle_units"])


def test_p4v2r2_disjoint_set_covers_every_previously_consumed_registry() -> None:
    registry = load_registry()
    prior = {value.lower() for value in registry["source_disjoint_from_repositories"]}
    p4s = json.loads(
        (ROOT / "configs" / "theseus_p4s_task_sources.json").read_text(
            encoding="utf-8"
        )
    )
    consumed = {
        value.lower() for value in p4s["source_disjoint_from_repositories"]
    }
    consumed.update(row["repository"].lower() for row in p4s["tasks"])
    consumed.add("python/typing_extensions")

    assert consumed <= prior


def test_p4v2r2_selection_exclusions_are_prospective_not_outcome_conditioned() -> None:
    registry = load_registry()
    excluded = registry["excluded_after_patch_structure_review"]
    assert {(row["repository"], row["pull_request"]) for row in excluded} == {
        ("fastapi/typer", 1843),
        ("pallets/markupsafe", 469),
    }
    assert all(row["reason"] for row in excluded)
    assert registry["contamination_and_use"] == {
        "public_benchmark": False,
        "development_only": True,
        "training_eligible": False,
        "D1_eligible": False,
        "D2_eligible": False,
        "serving_eligible": False,
        "target_patch_tests_or_oracle_candidate_visible": False,
        "automatic_ASI_stack_support_promotion": False,
    }
