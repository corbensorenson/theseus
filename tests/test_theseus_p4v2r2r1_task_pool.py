from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4v2r2r1_task_pool as pool  # noqa: E402


def test_recovery_pool_is_sealed_with_all_mechanics_floors() -> None:
    report = p2a.read_json(pool.POOL)

    assert report["state"] == "SEALED_BEFORE_SUCCESSOR_CANDIDATE_GENERATION"
    assert report["faults"] == []
    assert report["task_count"] == report["distinct_repositories"] == 10
    assert report["green_evaluator_audits"] == 10
    assert report["v2r2_oracle_replays_green"] == 10
    assert report["dependency_corruptions_rejected"] == 10


def test_recovery_pool_retains_exact_custody_and_no_quality_cap() -> None:
    report = p2a.read_json(pool.POOL)
    custody = [row["recovery_custody"] for row in report["tasks"]]

    assert custody.count("candidate_unseen_carried_from_attempt2_interruption") == 9
    assert custody.count("new_prospectively_sealed_source_disjoint_replacement") == 1
    assert report["generation_budget"]["project_selected_quality_token_cap"] is None
    assert report["counters"]["successor_local_model_calls"] == 0
