from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_semantic_ir_production_adequacy_resume_pool_v3 as pool  # noqa: E402


def test_v3_pool_preserves_three_candidates_and_replaces_consumed_task_04() -> None:
    report = pool.assemble()
    assert report["trigger_state"] == "GREEN"
    assert [row["index"] for row in report["rows"]] == list(range(1, 19))
    assert [row["opaque_task_id"] for row in report["rows"][:5]] == [
        "semantic-ir-adequacy-01",
        "semantic-ir-adequacy-02r1",
        "semantic-ir-adequacy-03",
        "semantic-ir-adequacy-04r1",
        "semantic-ir-adequacy-05",
    ]
    assert report["preserved_candidate_count"] == 3
    assert report["resume_generation_indices"] == list(range(4, 19))
    assert report["consumed_task_02_rerun_authorized"] is False
    assert report["consumed_task_04_rerun_authorized"] is False


def test_v3_pool_preserves_balance_and_zero_new_calls() -> None:
    report = pool.assemble()
    assert len(report["stratum_counts"]) == 6
    assert set(report["stratum_counts"].values()) == {3}
    assert report["preserved_denominator_model_calls"] == 6
    assert all(value == 0 for value in report["counters"].values())
