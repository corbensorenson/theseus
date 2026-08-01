from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p3_local_model_freeze as freeze  # noqa: E402


def test_p3_freezes_qwen35_as_development_denominator_without_capability_promotion() -> None:
    report = freeze.build(
        ROOT / "reports" / "theseus_p2b_local_model_selection.json",
        ROOT / "reports" / "theseus_assistant_p2c_click_3578_disposition.json",
        ROOT / "configs" / "theseus_assistant_p2c_instrument.json",
        ROOT / "configs" / "core_evidence_qwen35_9b_worker.json",
        ROOT / "reports" / "core_evidence_qwen35_9b_preflight.json",
    )

    assert report["trigger_state"] == "GREEN"
    assert report["P3_eligible"] is True
    assert report["model_capability_qualified"] is False
    assert report["selection_state"] == freeze.SELECTION_STATE
    assert report["selected_candidate_id"] == "qwen35_9b_general"
    assert report["selected_model_identity"]["revision"] == (
        "938d8919941c6e7efd3c7150eff7fe9d12afa631"
    )
