from __future__ import annotations

import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_assistant_route_integrity as integrity  # noqa: E402


def model_identity() -> dict:
    identity = {
        "repo_id": "mlx-community/Tmax-9B-MLX-8bit",
        "revision": "33812d6cf04f88856f25eb828de4f3144a194560",
        "snapshot_manifest_sha256": "a" * 64,
        "runtime": "mlx_lm_local_metal",
        "worker_config_sha256": "b" * 64,
        "decoder": {"temperature": 0.0, "maximum_tokens": 512},
        "decoder_sha256": "c" * 64,
    }
    identity["identity_sha256"] = integrity.stable_hash(identity)
    return identity


def dispatch() -> dict:
    return {
        "trace_id": "trace:1",
        "decision_digest": "d" * 64,
        "selection": {
            "selected_proposal_ids": ["proposal:1"],
            "terminal_outcome": "prepared",
        },
        "proposals": [
            {
                "proposal_id": "proposal:1",
                "capability_id": "assistant.chat_checkpoint",
                "capability_ids": ["assistant.chat_checkpoint"],
            }
        ],
        "plan_nodes": [
            {
                "node_id": "node:1",
                "capability_id": "assistant.chat_checkpoint",
                "dependencies": [],
                "deadline_ms": 30_000,
            }
        ],
        "effort": {"realized_limits": {"deadline_ms": 30_000, "maximum_parallelism": 1}},
    }


def contexts() -> tuple[dict, dict]:
    address = "vcm://theseus/project-state@v1"
    selected = {
        "ready": True,
        "selected_pages": [{"address": address, "model_visible": True}],
    }
    compiled = {
        "model_visible_pages": [
            {
                "address": address,
                "certificate_id": "cert:1",
                "execution_class": "authorized_task_state",
                "taints": [],
                "title": "Project state",
                "materialized_text": "P1 joins the frozen local model to the canonical runtime.",
            }
        ]
    }
    return selected, compiled


def backend_payload(request: dict, identity: dict, mode: str) -> dict:
    return {
        "policy": "project_theseus_local_inference_backend_v1",
        "trigger_state": "GREEN",
        "backend": {"identity": copy.deepcopy(identity)},
        "request": {
            "execution_mode": mode,
            "prompt_sha256": request["binding"]["model_prompt_sha256"],
            "route_context_digest": request["binding"]["route_packet_sha256"],
        },
        "response": {"answer": "A local answer."},
        "metrics": {"local_model_inference_calls": 1, "generated_tokens": 4},
        "external_inference_calls": 0,
    }


def integrated_request() -> dict:
    selected, compiled = contexts()
    return integrity.build_generation_request(
        execution_mode=integrity.INTEGRATED_MODE,
        prompt="What is next?",
        model_identity=model_identity(),
        selected_context=selected,
        compiled_context=compiled,
        reflexive_dispatch=dispatch(),
        reflexive_verification={"state": "VERIFIED"},
        structured_execution={"active": False, "terminal_outcome": "not_required"},
        tool_evidence={"required": False, "active": False},
        plan_context={"required": False, "active": False},
        procedural_route={"active": False, "required": False},
        effect_canary={"enabled": False, "ready": True},
    )


def test_integrated_request_consumes_content_and_keeps_durable_binding_raw_text_free() -> None:
    request = integrated_request()

    assert request["ready"] is True
    assert "P1 joins the frozen local model" in request["model_prompt"]
    assert "What is next?" in request["model_prompt"]
    assert "P1 joins the frozen local model" not in str(request["binding"])
    assert "What is next?" not in str(request["binding"])
    assert request["binding"]["context_pages"][0]["characters"] > 0
    assert len(request["binding"]["context_pages"][0]["content_sha256"]) == 64


def test_direct_and_integrated_receipts_share_the_exact_model_decoder_and_sandbox() -> None:
    identity = model_identity()
    direct_request = integrity.build_generation_request(
        execution_mode=integrity.DIRECT_MODE,
        prompt="What is next?",
        model_identity=identity,
    )
    integrated = integrated_request()
    direct_receipt = integrity.build_route_integrity_receipt(
        execution_mode=integrity.DIRECT_MODE,
        request_binding=direct_request["binding"],
        expected_model_identity=identity,
        backend_payload=backend_payload(direct_request, identity, integrity.DIRECT_MODE),
    )
    integrated_receipt = integrity.build_route_integrity_receipt(
        execution_mode=integrity.INTEGRATED_MODE,
        request_binding=integrated["binding"],
        expected_model_identity=identity,
        backend_payload=backend_payload(integrated, identity, integrity.INTEGRATED_MODE),
    )

    assert direct_receipt["ready"] is True
    assert integrated_receipt["ready"] is True
    pair = integrity.compare_matched_pair(direct_receipt, integrated_receipt)
    assert pair["ready"] is True
    assert pair["checks"]["model_decoder_snapshot_sandbox_evaluator_equal"] is True


def test_decorative_mode_label_cannot_turn_a_direct_binding_into_integrated_evidence() -> None:
    identity = model_identity()
    direct = integrity.build_generation_request(
        execution_mode=integrity.DIRECT_MODE,
        prompt="What is next?",
        model_identity=identity,
    )
    tampered = copy.deepcopy(direct["binding"])
    tampered["execution_mode"] = integrity.INTEGRATED_MODE
    payload = backend_payload(direct, identity, integrity.INTEGRATED_MODE)

    receipt = integrity.build_route_integrity_receipt(
        execution_mode=integrity.INTEGRATED_MODE,
        request_binding=tampered,
        expected_model_identity=identity,
        backend_payload=payload,
    )

    assert receipt["ready"] is False
    assert "request_binding_digest_valid" in receipt["failed_checks"]
    assert "selected_vcm_content_consumed_by_model" in receipt["failed_checks"]
    assert "decorative_labels_cannot_satisfy_integrated_route" in receipt["failed_checks"]


def test_missing_materialized_vcm_content_fails_before_model_inference() -> None:
    selected, _ = contexts()
    request = integrity.build_generation_request(
        execution_mode=integrity.INTEGRATED_MODE,
        prompt="What is next?",
        model_identity=model_identity(),
        selected_context=selected,
        compiled_context={"model_visible_pages": []},
        reflexive_dispatch=dispatch(),
        reflexive_verification={"state": "VERIFIED"},
    )

    assert request["ready"] is False
    assert "selected_vcm_content_not_consumable" in request["faults"]
    assert any(fault.startswith("selected_vcm_page_not_materialized") for fault in request["faults"])


def test_backend_prompt_or_model_mismatch_holds_the_response() -> None:
    identity = model_identity()
    request = integrated_request()
    payload = backend_payload(request, identity, integrity.INTEGRATED_MODE)
    payload["request"]["prompt_sha256"] = "0" * 64
    payload["backend"]["identity"]["revision"] = "wrong"

    receipt = integrity.build_route_integrity_receipt(
        execution_mode=integrity.INTEGRATED_MODE,
        request_binding=request["binding"],
        expected_model_identity=identity,
        backend_payload=payload,
    )

    assert receipt["release_allowed"] is False
    assert "backend_prompt_bound" in receipt["failed_checks"]
    assert "backend_model_identity_bound" in receipt["failed_checks"]
    assert "verification_controls_release_or_repair" in receipt["failed_checks"]


def test_effect_route_requires_observation_dispatch_binding_and_exact_rollback() -> None:
    selected, compiled = contexts()
    trace = dispatch()
    trace["proposals"][0]["capability_id"] = "assistant.route_authority_effect"
    trace["proposals"][0]["capability_ids"] = ["assistant.route_authority_effect"]
    trace["plan_nodes"][0]["capability_id"] = "assistant.route_authority_effect"
    request = integrity.build_generation_request(
        execution_mode=integrity.INTEGRATED_MODE,
        prompt="Exercise the bounded effect canary.",
        model_identity=model_identity(),
        selected_context=selected,
        compiled_context=compiled,
        reflexive_dispatch=trace,
        reflexive_verification={"state": "VERIFIED"},
        structured_execution={"active": False, "terminal_outcome": "not_required"},
        effect_canary={
            "enabled": True,
            "ready": True,
            "dispatch_bound": True,
            "rollback": {"complete": True, "residual_count": 0},
        },
    )
    receipt = integrity.build_route_integrity_receipt(
        execution_mode=integrity.INTEGRATED_MODE,
        request_binding=request["binding"],
        expected_model_identity=model_identity(),
        backend_payload=backend_payload(request, model_identity(), integrity.INTEGRATED_MODE),
    )

    assert receipt["ready"] is True
    assert receipt["checks"]["effect_observation_and_exact_rollback_bound"] is True

    broken = copy.deepcopy(request["binding"])
    broken["route_packet"]["effect"]["rollback_complete"] = False
    broken["route_packet_sha256"] = integrity.stable_hash(broken["route_packet"])
    broken["request_binding_sha256"] = integrity.binding_digest(broken)
    failed = integrity.build_route_integrity_receipt(
        execution_mode=integrity.INTEGRATED_MODE,
        request_binding=broken,
        expected_model_identity=model_identity(),
        backend_payload=backend_payload({"binding": broken}, model_identity(), integrity.INTEGRATED_MODE),
    )
    assert failed["ready"] is False
    assert "effect_observation_and_exact_rollback_bound" in failed["failed_checks"]


def test_selected_tool_and_planning_routes_must_execute_before_inference() -> None:
    selected, compiled = contexts()
    for capability, expected_fault, evidence in (
        (
            "assistant.deterministic_tool",
            "selected_tool_capability_not_executed",
            {
                "tool_evidence": {
                    "required": True,
                    "active": True,
                    "trigger_state": "GREEN",
                    "summary": {"result_count": 1},
                    "trace": "reports/tool-trace.jsonl",
                }
            },
        ),
        (
            "assistant.plan_dag",
            "selected_planning_capability_not_executed",
            {
                "plan_context": {
                    "required": True,
                    "active": True,
                    "planner_state": "GREEN",
                    "compiled_goal_count": 1,
                }
            },
        ),
    ):
        trace = dispatch()
        trace["proposals"][0]["capability_id"] = capability
        trace["proposals"][0]["capability_ids"] = [capability]
        trace["plan_nodes"][0]["capability_id"] = capability
        missing = integrity.build_generation_request(
            execution_mode=integrity.INTEGRATED_MODE,
            prompt="Use the selected capability.",
            model_identity=model_identity(),
            selected_context=selected,
            compiled_context=compiled,
            reflexive_dispatch=trace,
            reflexive_verification={"state": "VERIFIED"},
        )
        assert missing["ready"] is False
        assert expected_fault in missing["faults"]

        prepared = integrity.build_generation_request(
            execution_mode=integrity.INTEGRATED_MODE,
            prompt="Use the selected capability.",
            model_identity=model_identity(),
            selected_context=selected,
            compiled_context=compiled,
            reflexive_dispatch=trace,
            reflexive_verification={"state": "VERIFIED"},
            **evidence,
        )
        assert prepared["ready"] is True


def test_composite_route_requires_resolved_structured_execution() -> None:
    selected, compiled = contexts()
    trace = dispatch()
    trace["proposals"][0]["capability_id"] = ""
    trace["proposals"][0]["capability_ids"] = ["assistant.deterministic_tool", "assistant.plan_dag"]
    trace["plan_nodes"] = [
        {"node_id": "node:tool", "capability_id": "assistant.deterministic_tool", "dependencies": []},
        {"node_id": "node:plan", "capability_id": "assistant.plan_dag", "dependencies": ["node:tool"]},
    ]
    common = {
        "execution_mode": integrity.INTEGRATED_MODE,
        "prompt": "Execute the composite route.",
        "model_identity": model_identity(),
        "selected_context": selected,
        "compiled_context": compiled,
        "reflexive_dispatch": trace,
        "reflexive_verification": {"state": "VERIFIED"},
        "tool_evidence": {
            "required": True,
            "active": True,
            "trigger_state": "GREEN",
            "summary": {"result_count": 1},
            "trace": "reports/tool-trace.jsonl",
        },
        "plan_context": {
            "required": True,
            "active": True,
            "planner_state": "GREEN",
            "compiled_goal_count": 1,
        },
    }
    held = integrity.build_generation_request(
        **common,
        structured_execution={"active": True, "terminal_outcome": "verification_failed", "failed_count": 1},
    )
    assert held["ready"] is False
    assert "composite_plan_not_executed_or_held" in held["faults"]

    resolved = integrity.build_generation_request(
        **common,
        structured_execution={
            "active": True,
            "terminal_outcome": "resolved",
            "execution_digest": "e" * 64,
            "resolved_count": 2,
            "failed_count": 0,
            "effect_authority_granted": False,
        },
    )
    assert resolved["ready"] is True
