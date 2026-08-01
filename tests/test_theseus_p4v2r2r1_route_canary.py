from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_p4v2r2r1_route_canary as canary  # noqa: E402


def test_real_route_canary_is_green_v2_and_non_task() -> None:
    report = canary.audit()

    assert report["trigger_state"] == "GREEN"
    assert report["faults"] == []
    assert report["route_policy"] == "project_theseus_live_route_integrity_v2"
    assert report["backend_policy"] == "project_theseus_local_inference_backend_v2"
    assert report["route_canary_model_calls"] == 1
    assert report["task_candidate_or_control_calls"] == 0


def test_real_route_canary_has_uncapped_normal_termination() -> None:
    report = canary.audit()

    assert report["termination_reason"] == "model_eos"
    assert report["generated_tokens"] > 0
    assert report["physical_context_boundary_hits"] == 0
    assert report["project_selected_quality_token_cap"] is None
    assert report["external_inference_calls"] == 0
