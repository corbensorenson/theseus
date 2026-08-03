from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_semantic_ir_production_adequacy_resume_pool as pool  # noqa: E402


def test_resume_pool_replaces_only_consumed_task_02() -> None:
    report = pool.assemble()
    assert report["trigger_state"] == "GREEN"
    assert [row["index"] for row in report["rows"]] == list(range(1, 19))
    assert report["rows"][0]["opaque_task_id"] == "semantic-ir-adequacy-01"
    assert report["rows"][1]["opaque_task_id"] == "semantic-ir-adequacy-02r1"
    assert report["rows"][2]["opaque_task_id"] == "semantic-ir-adequacy-03"
    assert report["consumed_task_02_rerun_authorized"] is False
    assert report["resume_generation_indices"] == list(range(2, 19))


def test_resume_pool_preserves_balance_and_zero_new_calls() -> None:
    report = pool.assemble()
    assert len(report["stratum_counts"]) == 6
    assert set(report["stratum_counts"].values()) == {3}
    assert report["preserved_candidate_count"] == 1
    assert all(value == 0 for value in report["counters"].values())
