from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4v2r2r1_fetch_source as fetcher  # noqa: E402


def test_sealed_replacement_source_pair_is_exact_and_normalized() -> None:
    report = fetcher.read_json(fetcher.REPORT)

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert len(report["artifacts"]) == 2
    assert {row["label"] for row in report["artifacts"]} == {"parent", "target"}
    assert all(row["missing_required_members"] == [] for row in report["artifacts"])
    assert all(len(row["projection"]["retained_members"]) == 2 for row in report["artifacts"])
    for row in report["artifacts"]:
        assert fetcher.sha256_file(ROOT / row["upstream_path"]) == row["upstream_sha256"]
        assert fetcher.sha256_file(ROOT / row["normalized_path"]) == row["normalized_sha256"]


def test_source_fetch_has_zero_experimental_or_training_calls() -> None:
    report = fetcher.read_json(fetcher.REPORT)

    assert report["candidate_or_control_calls"] == 0
    assert report["parent_target_oracle_or_evaluator_executions"] == 0
    assert report["teacher_calls"] == 0
    assert report["training_rows_written"] == 0
    assert report["project_selected_quality_token_cap"] is None
