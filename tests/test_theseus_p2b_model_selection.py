from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p2b_model_selection as selection  # noqa: E402


def test_p2b_selection_uses_retained_evidence_without_claiming_qualification() -> None:
    report = selection.select_model(ROOT / "reports" / "core_evidence_local_model_bakeoff_synthesis.json")

    assert report["trigger_state"] == "GREEN"
    assert report["selected_candidate_id"] == "qwen35_9b_general"
    assert report["selected_model_identity"]["revision"] == "938d8919941c6e7efd3c7150eff7fe9d12afa631"
    assert report["model_qualified"] is False
    assert report["P3_eligible"] is False
    assert report["counters"]["fresh_tasks_consumed"] == 0
    selected = next(row for row in report["candidates"] if row["candidate_id"] == "qwen35_9b_general")
    assert selected["useful"] == 1
    assert selected["infrastructure_failed"] == 2
    assert selected["old_worker_adequacy_passed"] is False
