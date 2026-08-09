from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_k3_route_preflight as owner  # noqa: E402
import theseus_vcm_k3_route_preflight_audit as audit_owner  # noqa: E402

CONFIG = ROOT / "configs" / "theseus_vcm_k3_route_preflight.json"


def fake_counter(system: str, prompt: str) -> int:
    return max(1, (len(system.encode()) + len(prompt.encode())) // 4)


def test_call_free_preflight_materializes_all_six_routes() -> None:
    report, packets = owner.build(CONFIG, token_counter=fake_counter)
    assert report["trigger_state"] == "GREEN"
    assert report["row_count"] == 6
    assert report["route_count"] == 6
    assert report["packet_count"] == 36
    assert report["local_model_calls"] == 0
    assert report["external_reference_calls"] == 0
    assert len(packets["rows"]) == 36


def test_flat_and_vcm_share_parent_information_but_not_route_envelope() -> None:
    report, _ = owner.build(CONFIG, token_counter=fake_counter)
    for row in report["rows"]:
        arms = {arm["route"]: arm for arm in row["arms"]}
        flat = arms["information_matched_flat_direct_context"]
        vcm = arms["governed_vcm"]
        assert flat["context_information_sha256"] == vcm["context_information_sha256"]
        assert flat["prompt_sha256"] != vcm["prompt_sha256"]
        assert sorted(arm["within_row_order"] for arm in row["arms"]) == [1,2,3,4,5,6]


def test_role_audit_rederives_packet_and_analysis_contract() -> None:
    report, packets = owner.build(CONFIG, token_counter=fake_counter)
    audit = audit_owner.audit(CONFIG, actual_report=report, actual_packets=packets, token_counter=fake_counter)
    assert audit["trigger_state"] == "GREEN"
    assert audit["audited_packet_count"] == 36
    assert audit["local_model_calls"] == 0
    assert audit["external_reference_calls"] == 0
