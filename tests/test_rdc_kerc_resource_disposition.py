from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import rdc_kerc_resource_disposition as disposition  # noqa: E402


def test_resource_disposition_is_narrow_and_evidence_bound() -> None:
    report = disposition.build_report()
    assert report["trigger_state"] == "GREEN"
    assert report["disposition"] == "RESOURCE_DEFERRED_ON_THIS_HOST"
    assert report["candidate_execution_authorized"] is False
    assert report["scientific_falsification_claimed"] is False
    assert report["capability_claimed"] is False
    assert report["measurements"]["long_panel_peak_mib"] > report["safety_limit_mib"]
    assert report["measurements"]["query_chunk_512_peak_mib"] > report["safety_limit_mib"]
    assert all(report["checks"].values())
