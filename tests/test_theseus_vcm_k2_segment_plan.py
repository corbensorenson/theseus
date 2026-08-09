from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_instrument_builder as producer  # noqa: E402
import theseus_vcm_k2_segment_plan_audit as auditor  # noqa: E402

CONFIG = ROOT / "configs" / "theseus_vcm_k2_segment_plan.json"


def test_segment_plan_compiles_one_target_free_62_row_manifest() -> None:
    report, manifest = producer.compile_segment_plan(CONFIG)
    assert report["trigger_state"] == "GREEN"
    assert report["task_count"] == 62
    assert report["segment_counts"] == {
        "static_no_project_lock": 8,
        "immutable_resolution_required": 6,
        "locked_closure": 48,
    }
    assert report["panel_admitted"] is False
    assert manifest["broad_parent_effect_root"] == "repository"
    assert len(manifest["rows"]) == 62
    for row in manifest["rows"]:
        assert set(row) == auditor.SAFE_ROW_KEYS
        assert row["parent_archive"].endswith("_parent.tar.gz")
        assert "target" not in row["parent_archive"].lower()


def test_role_audit_rederives_alignment_and_preserves_zero_execution() -> None:
    report, manifest = producer.compile_segment_plan(CONFIG)
    audit = auditor.audit(CONFIG, report=report, manifest=manifest)
    assert audit["trigger_state"] == "GREEN"
    assert audit["audited_task_count"] == 62
    assert audit["target_derived_selector_input_count"] == 0
    assert audit["panel_admitted"] is False
    assert audit["audit_kind"] == "role-separated rederivation"
    assert audit["network_or_dependency_execution_performed"] is False
    assert audit["candidate_or_control_calls"] == 0
    assert audit["external_reference_calls"] == 0
