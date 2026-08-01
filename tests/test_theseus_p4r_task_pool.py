from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4_cognitive_compilation_evaluator as evaluator  # noqa: E402
import theseus_p4r_task_pool as p4r_pool  # noqa: E402


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_p4r_pool_is_sealed_without_candidate_calls_or_quality_token_cap() -> None:
    pool = read(ROOT / "configs" / "theseus_p4r_task_pool.json")

    assert pool["state"] == "SEALED_BEFORE_CANDIDATE_GENERATION"
    assert pool["task_count"] == pool["green_evaluator_audits"] == 10
    assert pool["distinct_repositories"] == 10
    assert pool["reused_unopened_task_count"] == 7
    assert pool["new_task_count"] == 3
    assert pool["generation_budget"]["project_selected_quality_token_cap"] is None
    assert pool["generation_budget"]["boundary_hit_counts_as_failure"] is False
    assert pool["counters"]["local_model_calls"] == 0
    assert pool["counters"]["hosted_model_calls"] == 0
    assert pool["counters"]["deterministic_request_compiler_calls"] == 0
    assert pool["instrument_freeze_commit"] == "0a8f0bec2cd8883fa6b1176e5f9979554c45539f"
    assert pool["post_selection_mechanics_repair"]["candidate_or_control_calls_before_repair"] == 0
    assert pool["post_selection_mechanics_repair"]["task_membership_changed"] is False
    assert all(row["baseline_parent_failed"] for row in pool["tasks"])
    assert all(row["upstream_target_passed"] for row in pool["tasks"])
    assert all(row["compiler_oracle_passed"] for row in pool["tasks"])
    assert all(row["four_corruptions_rejected"] for row in pool["tasks"])


def test_rebound_tasks_preserve_exact_unopened_source_and_oracle_artifacts() -> None:
    pool = read(ROOT / "configs" / "theseus_p4r_task_pool.json")
    old = read(ROOT / "configs" / "theseus_p4_task_pool.json")
    old_by_index = {row["campaign_index"]: row for row in old["tasks"]}

    for row in pool["tasks"][:7]:
        prior = old_by_index[row["predecessor_campaign_index"]]
        task = read(ROOT / row["task"])
        old_task = read(ROOT / prior["task"])
        assert task["source_archive_sha256"] == old_task["source_archive_sha256"]
        assert row["oracle_ir_sha256"] == prior["oracle_ir_sha256"]
        assert task["opaque_task_id"].startswith("p4r-cognitive-compilation-")


def test_new_repositories_are_disjoint_from_every_prior_development_pool() -> None:
    registry = read(ROOT / "configs" / "theseus_p4r_task_sources.json")
    new_repositories = {row["repository"].lower() for row in registry["new_tasks"]}

    assert not new_repositories.intersection(p4r_pool.prior_repositories())


def test_every_repaired_evaluator_remains_live_green() -> None:
    pool = read(ROOT / "configs" / "theseus_p4r_task_pool.json")
    for row in pool["tasks"]:
        assert p2a.sha256_file(ROOT / row["task"]) == row["task_sha256"]
        assert p2a.sha256_file(ROOT / row["evaluator"]) == row["evaluator_sha256"]
        assert evaluator.audit_evaluator(ROOT / row["evaluator"])["trigger_state"] == "GREEN"
