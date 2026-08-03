from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_semantic_ir_production_adequacy_replacement_02_source as source  # noqa: E402


def test_preflight_binds_consumed_task_and_source_disjoint_replacement() -> None:
    report = source.preflight()
    assert report["trigger_state"] == "GREEN"
    assert report["source_disjoint_from_prior_panel"] is True
    assert report["same_stratum_as_consumed_task"] is True
    assert report["candidate_packet_materialized"] is False
    assert report["counters"]["local_model_calls"] == 0


def test_source_change_contract_accepts_only_declared_replacement() -> None:
    parent = b"def f(expected_discount):\n  value = (\n    500,\n  )\n"
    target = (
        b"def f(expected_discount):\n  value = (\n    expected_discount,\n"
        b"    \"applied discount amount must match the configured fixed reduction\",\n  )\n"
    )
    expected = json.loads(source.DEFAULT_CONFIG.read_text())["expected_change"]
    assert source.source_change_faults(parent, target, expected) == []
    assert "added_fragment_inventory_mismatch" in source.source_change_faults(
        parent, target.replace(b"configured", b"hard-coded"), expected
    )


def test_metadata_contract_rejects_a_different_changed_file() -> None:
    config = json.loads(source.DEFAULT_CONFIG.read_text())
    replacement = config["replacement"]
    pr = {
        "state": "closed",
        "merged_at": replacement["merged_utc"],
        "title": replacement["title"],
        "base": {"sha": replacement["base_revision"]},
        "head": {"sha": replacement["head_revision"]},
        "merge_commit_sha": replacement["merge_revision"],
    }
    merge = {
        "sha": replacement["merge_revision"],
        "parents": [{"sha": replacement["base_revision"]}],
    }
    faults = source.metadata_faults(config, pr, [{"filename": "other.py"}], merge)
    assert faults == ["changed_file_inventory_mismatch"]
