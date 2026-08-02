from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4v2r2r2_fetch_sources as original  # noqa: E402
import theseus_p4v2r2r2_revision_repair as repair  # noqa: E402


def test_revision_correction_is_bound_before_candidate_generation() -> None:
    audit = repair.audit_corrections()
    value = original.read_json(repair.CORRECTIONS)

    assert audit["trigger_state"] == "GREEN"
    assert audit["faults"] == []
    assert audit["correction_count"] == 3
    assert value["discovery"]["candidate_or_control_calls"] == 0
    assert value["invariants"]["task_membership_changed"] is False
    assert value["invariants"]["revision_identity_repaired"] is True
    assert value["invariants"]["project_selected_quality_token_cap"] is None


def test_corrected_source_pairs_are_green_distinct_and_preserve_originals() -> None:
    report = original.read_json(repair.REPORT)
    old = original.read_json(repair.ORIGINAL_FETCH)

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["corrected_task_count"] == 3
    assert report["corrected_artifact_count"] == 6
    assert all(row["effect_source_differs"] is True for row in report["tasks"])
    assert report["candidate_or_control_calls"] == 0
    assert report["project_selected_quality_token_cap"] is None
    for row in old["tasks"]:
        for artifact in row["artifacts"]:
            path = ROOT / artifact["normalized"]
            assert path.is_file()
            assert original.sha256_file(path) == artifact["normalized_sha256"]
