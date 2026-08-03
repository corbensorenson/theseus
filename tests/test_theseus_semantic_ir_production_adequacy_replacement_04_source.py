from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_semantic_ir_production_adequacy_replacement_04_source as source  # noqa: E402


def test_preflight_is_green_and_closes_all_non_source_authority() -> None:
    report = source.preflight()
    assert report["trigger_state"] == "GREEN"
    assert report["source_disjoint_from_prior_panel"] is True
    assert report["same_stratum_as_consumed_task"] is True
    assert report["counters"]["local_model_calls"] == 0
    assert report["counters"]["external_inference_calls"] == 0


def test_preflight_rejects_consumed_task_rerun(tmp_path: Path) -> None:
    config = json.loads(source.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["consumed_observation"]["rerun_authorized"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    report = source.preflight(path)
    assert report["trigger_state"] == "RED"
    assert "consumed_task_rerun_not_fail_closed" in report["faults"]


def test_source_change_requires_none_branch_cast_and_parent_negative() -> None:
    parent = b"def pair_align(atol, dtype):\n    gap = 1\n    return gap\n"
    target = (
        b"def pair_align(atol, dtype):\n"
        b"    # Cast tolerance to the same type.\n"
        b"    if atol is None:\n"
        b"        atol = 0.0\n"
        b"    atol = dtype(atol)\n"
        b"    gap = 1\n"
        b"    return gap\n"
    )
    expected = json.loads(source.DEFAULT_CONFIG.read_text(encoding="utf-8"))["expected_change"]
    assert source.source_change_faults(parent, target, expected) == []
    assert source.source_change_faults(target, target, expected)


def test_acquire_metadata_must_match_exact_changed_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    config = json.loads(source.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    replacement = config["replacement"]
    pr = {
        "state": "closed",
        "merged_at": replacement["merged_utc"],
        "merge_commit_sha": replacement["merge_revision"],
        "title": replacement["title"],
        "base": {"sha": replacement["base_revision"]},
        "head": {"sha": replacement["head_revision"]},
    }
    merge = {"sha": replacement["merge_revision"], "parents": [{"sha": replacement["base_revision"]}]}
    files = [{"filename": "skbio/alignment/_pair.py"}]
    assert "changed_file_inventory_mismatch" in source.metadata_faults(config, pr, files, merge)
