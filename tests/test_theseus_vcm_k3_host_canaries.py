from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_vcm_k3_host_canaries as owner  # noqa: E402
import theseus_vcm_k3_host_canaries_audit as audit_owner  # noqa: E402

CONFIG = ROOT / "configs" / "theseus_vcm_k3_host_canaries.json"


def fake_runner(**kwargs):
    expected = {
        "no_added_context_floor": (148, 261996),
        "ordinary_direct_retrieval_same_parent_store_query_and_context_opportunity": (123235, 138909),
        "maximal_full_parent_context_when_physically_addressable_and_host_operable": (155487, 106657),
        "information_matched_flat_direct_context": (207580, 54564),
        "governed_vcm": (207653, 54491),
        "hierarchical_summary_or_prompt_compression_same_parent_store_and_context_opportunity": (242182, 19962),
    }
    route = kwargs["prompt"].split("\n\nROUTE\n", 1)[1].split("\n", 1)[0]
    tokens, residual = expected[route]
    return {
        "trigger_state": "GREEN",
        "faults": [],
        "response": {"answer": "CANARY_OK"},
        "metrics": {
            "termination_reason": "parser_complete",
            "exact_prompt_tokens": tokens,
            "effective_context_residual_tokens": residual,
            "generated_tokens": 2,
            "host_safety_wall_time_hit": False,
            "physical_context_boundary_hit": False,
        },
    }


def test_call_free_plan_reconstructs_exact_six_route_prompts() -> None:
    report, selected = owner.build_plan(CONFIG)
    assert report["trigger_state"] == "GREEN"
    assert len(selected) == 6
    assert report["local_model_calls"] == 0
    assert report["hidden_evaluator_calls"] == 0
    assert report["external_reference_calls"] == 0
    assert all(row["prompt_sha256"] for row in selected)


def test_fake_execution_is_non_scoring_and_role_auditable() -> None:
    report = owner.execute(CONFIG, runner=fake_runner, seal_check=lambda _cfg: True, model_factory=lambda *_args: object())
    assert report["trigger_state"] == "GREEN"
    assert report["local_model_calls"] == 6
    assert report["nine_task_screen_authorized"] is True
    assert all(row["raw_response_stored"] is False for row in report["calls"])
    audit = audit_owner.audit(CONFIG, actual=report)
    assert audit["trigger_state"] == "GREEN"
    assert audit["audited_call_count"] == 6


def test_host_interlock_stops_campaign_without_capability_negative() -> None:
    calls = 0
    def wall_runner(**kwargs):
        nonlocal calls
        calls += 1
        result = fake_runner(**kwargs)
        if calls == 2:
            result["metrics"]["termination_reason"] = "host_safety_wall_time"
            result["metrics"]["host_safety_wall_time_hit"] = True
        return result
    report = owner.execute(CONFIG, runner=wall_runner, seal_check=lambda _cfg: True, model_factory=lambda *_args: object())
    assert report["state"] == "INCONCLUSIVE_EXPERIMENT_HOST_OPERABILITY"
    assert report["local_model_calls"] == 2
    assert report["nine_task_screen_authorized"] is False
    assert report["calls"][-1]["capability_or_mechanism_evidence"] is False
