from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_p4_budget_interruption_disposition as disposition  # noqa: E402


def test_interruption_disposition_preserves_consumed_and_unopened_boundaries() -> None:
    report = disposition.build_report()

    assert report["scientific_status"] == "INCONCLUSIVE_EXPERIMENT"
    assert report["execution_custody"]["retained_model_call_reports"] == 14
    assert report["execution_custody"]["consumed_campaign_indices"] == [1, 2, 3]
    assert report["execution_custody"]["unopened_campaign_indices"] == list(range(4, 11))
    assert report["budget_audit"]["observed_ceiling_hits"] == 0
    assert report["budget_audit"]["all_retained_outputs_reached_protocol_terminal_end"] is True
    assert report["budget_audit"]["explicit_finish_reason_present_in_v1_receipts"] is False
    assert report["evidence_disposition"]["claim_support_state_effect"] == "none"
