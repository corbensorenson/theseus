from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4v2r2r2_task_pool as pool  # noqa: E402


def test_fresh_pool_is_sealed_with_all_mechanics_floors() -> None:
    value = p2a.read_json(pool.POOL)

    assert value["state"] == "SEALED_BEFORE_CANDIDATE_GENERATION"
    assert value["faults"] == []
    assert value["task_count"] == value["distinct_repositories"] == 10
    assert value["green_evaluator_audits"] == 10
    assert value["v2r2_oracle_replays_green"] == 10
    assert value["dependency_corruptions_rejected"] == 10
    assert sum(row["revision_correction_applied"] for row in value["tasks"]) == 3


def test_fresh_pool_has_no_candidate_calls_user_gate_or_quality_cap() -> None:
    value = p2a.read_json(pool.POOL)

    assert value["candidate_generation_opened"] is False
    assert value["generation_boundary"]["project_selected_quality_token_cap"] is None
    assert value["generation_boundary"]["boundary_hit_invalidates_observation"] is True
    assert value["counters"]["local_model_calls"] == 0
    assert value["counters"]["hosted_model_calls"] == 0


def test_task_surfaces_are_exactly_bound_and_candidate_blind() -> None:
    value = p2a.read_json(pool.POOL)
    for row in value["tasks"]:
        task_path = ROOT / row["task"]
        evaluator_path = ROOT / row["evaluator"]
        assert p2a.sha256_file(task_path) == row["task_sha256"]
        assert p2a.sha256_file(evaluator_path) == row["evaluator_sha256"]
        task = p2a.read_json(task_path)
        evaluator = p2a.read_json(evaluator_path)
        assert task["visible_verifier"]["candidate_prompt_visibility"] is False
        assert evaluator["blindness"]["hidden_test_candidate_visible"] is False
        assert evaluator["blindness"]["oracle_candidate_visible"] is False
        assert row["baseline_parent_failed"] is True
        assert row["upstream_target_passed"] is True
        assert row["compiler_oracle_v1_passed"] is True
        assert row["four_base_corruptions_rejected"] is True
