from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "procedural_memory_route_adoption_gate.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("procedural_adoption_hidden", SCRIPT)
assert SPEC and SPEC.loader
adoption = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adoption)


def report(*, procedural_route: str, effect_route: str) -> dict:
    return {
        "trigger_state": "GREEN",
        "summary": {
            "procedural_default_route_id": procedural_route,
            "effect_canary_enabled": True,
            "effect_canary_ready": True,
            "effect_canary_rollback_complete": True,
            "effect_canary_transaction_id": "transaction-1",
            "effect_canary_first_effect_identity": "first",
            "effect_canary_final_effect_identity": "prior",
        },
        "procedural_default_route": {
            "active": True,
            "ready": True,
            "trigger_state": "GREEN",
            "selection": {"matched": True},
            "selected_route": {
                "id": procedural_route,
                "learned_generation_claim_allowed": False,
                "continued_regression_guard": {"armed": True},
            },
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        },
        "effect_canary": {
            "enabled": True,
            "ready": True,
            "policy": "project_theseus_bounded_local_effect_transaction_v1",
            "route_id": effect_route,
            "transaction_id": "transaction-1",
            "proposer_id": "proposer",
            "observer_id": "observer",
            "evaluator_id": "evaluator",
            "dispatch_bound": True,
            "dispatch_trace_id": "trace-1",
            "dispatch_decision_digest": "decision-1",
            "target": "runtime/assistant_effects/test.json",
            "effect_inventory": [{
                "effect_id": "effect-1",
                "path": "runtime/assistant_effects/test.json",
                "operation": "create",
                "before_identity": "prior",
                "intended_content_sha256": "content",
                "rollback_obligation": "remove created path",
            }],
            "observation": {
                "exists": True,
                "matches_intent": True,
                "identity": "first",
                "path": "runtime/assistant_effects/test.json",
                "sha256": "content",
                "expected_content_sha256": "content",
                "expected_route_id": effect_route,
                "expected_dispatch_trace_id": "trace-1",
                "expected_dispatch_decision_digest": "decision-1",
                "observer_id": "observer",
                "parsed_json": {
                    "route_id": effect_route,
                    "capability_id": effect_route,
                    "transaction_id": "transaction-1",
                    "dispatch_trace_id": "trace-1",
                    "dispatch_decision_digest": "decision-1",
                    "public_training_rows_written": 0,
                    "external_inference_calls": 0,
                    "fallback_return_count": 0,
                },
            },
            "rollback": {
                "complete": True,
                "residual_count": 0,
                "before_identity": "prior",
                "first_effect_identity": "first",
                "final_identity": "prior",
                "removed_new_path": True,
                "prior_path_existed": False,
            },
            "residuals": [],
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        },
    }


def test_routes_are_bound_separately() -> None:
    # request_contract:procedural_and_effect_routes_bound_separately
    route = "default.local_planning_assistant_metadata_only_v1"
    result = adoption.audit_adopted_route_effect_reference(
        assistant_report=report(
            procedural_route=route,
            effect_route="assistant.route_authority_effect",
        ),
        adopted_route_ids={route},
    )
    assert result["valid"] is True
    assert result["checks"]["effect_authority_route_is_separate"] is True


def test_unadopted_route_is_rejected() -> None:
    # request_contract:unadopted_procedural_route_rejected
    result = adoption.audit_adopted_route_effect_reference(
        assistant_report=report(
            procedural_route="default.unadopted",
            effect_route="assistant.route_authority_effect",
        ),
        adopted_route_ids={"default.expected"},
    )
    assert result["valid"] is False
    assert "procedural_route_selected" in result["hard_gaps"]


def test_procedural_route_is_not_effect_authority() -> None:
    # request_contract:procedural_route_cannot_grant_effect_authority
    route = "default.local_planning_assistant_metadata_only_v1"
    result = adoption.audit_adopted_route_effect_reference(
        assistant_report=report(procedural_route=route, effect_route=route),
        adopted_route_ids={route},
    )
    assert result["valid"] is False
    assert "effect_transaction_valid" in result["hard_gaps"]
