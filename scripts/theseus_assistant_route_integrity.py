#!/usr/bin/env python3
"""Fail-closed route integrity for the canonical local assistant.

This module owns the comparison boundary between the fixed-model direct arm
and the integrated Theseus arm.  It never treats arm labels or backend-emitted
``passed`` flags as evidence.  Instead it recomputes prompt, model, route,
context, verifier, authority, and effect bindings from the observed payloads.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DIRECT_MODE = "direct_local_model"
INTEGRATED_MODE = "integrated_local_model"
LOCAL_MODEL_MODES = {DIRECT_MODE, INTEGRATED_MODE}
ROUTE_POLICY = "project_theseus_live_route_integrity_v1"
BACKEND_RUNTIME = "mlx_lm_local_metal"
STRUCTURAL_VERIFIER = "theseus_local_response_integrity_verifier_v1"
EFFECT_SANDBOX = "theseus_assistant_no_automatic_effects_v1"
DENIED_CONTEXT_TAINTS = {
    "public_benchmark_payload",
    "public_calibration_payload",
    "raw_private_user_text",
    "runtime_external_inference",
    "revoked",
    "deleted",
}
MODEL_FIELDS = (
    "repo_id",
    "revision",
    "snapshot_manifest_sha256",
    "runtime",
    "worker_config_sha256",
    "decoder_sha256",
)


def load_model_contract(
    worker_config_path: str | Path,
    runtime_preflight_path: str | Path,
    *,
    maximum_tokens: int,
    required_repo_id: str = "",
    required_revision: str = "",
    required_snapshot_manifest_sha256: str = "",
) -> dict[str, Any]:
    worker_path = resolve(worker_config_path)
    preflight_path = resolve(runtime_preflight_path)
    worker = read_json(worker_path)
    preflight = read_json(preflight_path)
    card = as_dict(worker.get("model"))
    preflight_identity = as_dict(preflight.get("model_identity"))
    faults: list[str] = []
    for field in ("repo_id", "revision", "temperature", "maximum_action_tokens"):
        if card.get(field) in {None, ""}:
            faults.append(f"model_card_missing_{field}")
    if preflight.get("trigger_state") != "GREEN":
        faults.append("runtime_preflight_not_green")
    for field in ("repo_id", "revision"):
        if str(preflight_identity.get(field) or "") != str(card.get(field) or ""):
            faults.append(f"runtime_preflight_{field}_mismatch")
    snapshot_manifest = str(preflight_identity.get("snapshot_manifest_sha256") or "")
    if len(snapshot_manifest) != 64:
        faults.append("runtime_preflight_snapshot_manifest_missing")
    required_values = {
        "repo_id": required_repo_id,
        "revision": required_revision,
        "snapshot_manifest_sha256": required_snapshot_manifest_sha256,
    }
    observed_values = {
        "repo_id": str(card.get("repo_id") or ""),
        "revision": str(card.get("revision") or ""),
        "snapshot_manifest_sha256": snapshot_manifest,
    }
    for field, required in required_values.items():
        if required and observed_values[field] != required:
            faults.append(f"frozen_model_{field}_mismatch")
    configured_maximum = int(card.get("maximum_action_tokens") or 0)
    effective_maximum = int(maximum_tokens or configured_maximum)
    if effective_maximum <= 0 or effective_maximum > configured_maximum:
        faults.append("product_maximum_tokens_out_of_worker_bounds")
    decoder = {
        "temperature": float(card.get("temperature") or 0.0),
        "repetition_penalty": float(card.get("repetition_penalty") or 1.0),
        "repetition_context_size": int(card.get("repetition_context_size") or 0),
        "chat_template_kwargs": as_dict(card.get("chat_template_kwargs")),
        "maximum_tokens": effective_maximum,
        "worker_maximum_action_tokens": configured_maximum,
        "kv_bits": card.get("kv_bits"),
        "kv_group_size": card.get("kv_group_size"),
        "quantized_kv_start": card.get("quantized_kv_start"),
    }
    identity = {
        "repo_id": str(card.get("repo_id") or ""),
        "revision": str(card.get("revision") or ""),
        "snapshot_manifest_sha256": snapshot_manifest,
        "runtime": BACKEND_RUNTIME,
        "worker_config_sha256": file_sha256(worker_path),
        "decoder": decoder,
        "decoder_sha256": stable_hash(decoder),
    }
    identity["identity_sha256"] = stable_hash(identity)
    return {
        "ready": not faults,
        "identity": identity,
        "model_card": card,
        "worker_config": rel(worker_path),
        "runtime_preflight": rel(preflight_path),
        "faults": sorted(set(faults)),
    }


def build_generation_request(
    *,
    execution_mode: str,
    prompt: str,
    model_identity: dict[str, Any],
    selected_context: dict[str, Any] | None = None,
    compiled_context: dict[str, Any] | None = None,
    reflexive_dispatch: dict[str, Any] | None = None,
    reflexive_verification: dict[str, Any] | None = None,
    structured_execution: dict[str, Any] | None = None,
    tool_evidence: dict[str, Any] | None = None,
    plan_context: dict[str, Any] | None = None,
    procedural_route: dict[str, Any] | None = None,
    effect_canary: dict[str, Any] | None = None,
    maximum_context_pages: int = 8,
    maximum_context_characters: int = 12_000,
) -> dict[str, Any]:
    """Build the transient model prompt and a raw-text-free durable binding."""
    faults: list[str] = []
    if execution_mode not in LOCAL_MODEL_MODES:
        faults.append("unsupported_local_execution_mode")
    if not prompt.strip():
        faults.append("empty_user_prompt")
    if not model_identity.get("identity_sha256"):
        faults.append("model_identity_missing")

    context_rows: list[dict[str, Any]] = []
    route_packet: dict[str, Any]
    if execution_mode == DIRECT_MODE:
        route_packet = {
            "route_kind": "direct_fixed_model",
            "planning": "none",
            "vcm": "none",
            "tool_calls": 0,
            "automatic_effects": 0,
        }
        model_prompt = prompt
    else:
        context_rows, context_faults = materialize_selected_context(
            selected_context or {},
            compiled_context or {},
            maximum_pages=maximum_context_pages,
            maximum_characters=maximum_context_characters,
        )
        faults.extend(context_faults)
        dispatch = reflexive_dispatch or {}
        verification = reflexive_verification or {}
        execution = structured_execution or {}
        tools = tool_evidence or {}
        plan = plan_context or {}
        procedural = procedural_route or {}
        effect = effect_canary or {}
        selected_capabilities = selected_capability_ids(dispatch)
        plan_nodes = compact_plan_nodes(dispatch)
        route_packet = {
            "route_kind": "integrated_theseus",
            "trace_id": dispatch.get("trace_id"),
            "decision_digest": dispatch.get("decision_digest"),
            "terminal_outcome": get_path(dispatch, ["selection", "terminal_outcome"], ""),
            "verification_state": verification.get("state"),
            "selected_capabilities": selected_capabilities,
            "realized_limits": get_path(dispatch, ["effort", "realized_limits"], {}),
            "plan_nodes": plan_nodes,
            "structured_execution": {
                "active": execution.get("active") is True,
                "terminal_outcome": execution.get("terminal_outcome"),
                "execution_digest": execution.get("execution_digest"),
                "resolved_count": int(execution.get("resolved_count") or 0),
                "failed_count": int(execution.get("failed_count") or 0),
                "effect_authority_granted": execution.get("effect_authority_granted") is True,
            },
            "tool": {
                "required": tools.get("required") is True,
                "active": tools.get("active") is True,
                "trigger_state": tools.get("trigger_state"),
                "result_count": int(get_path(tools, ["summary", "result_count"], 0) or 0),
                "evidence_ref": tools.get("trace") or tools.get("report") or "",
            },
            "planner": {
                "required": plan.get("required") is True,
                "active": plan.get("active") is True,
                "state": plan.get("planner_state"),
                "compiled_goal_count": int(plan.get("compiled_goal_count") or 0),
            },
            "procedural_route": {
                "active": procedural.get("active") is True,
                "required": procedural.get("required") is True,
                "ready": procedural.get("ready") is True,
                "route_id": get_path(procedural, ["selected_route", "id"], ""),
                "selection_matched": get_path(procedural, ["selection", "matched"], False) is True,
            },
            "effect": {
                "enabled": effect.get("enabled") is True,
                "ready": effect.get("ready") is True,
                "dispatch_bound": effect.get("dispatch_bound") is True,
                "rollback_complete": get_path(effect, ["rollback", "complete"], False) is True,
                "rollback_residual_count": int(get_path(effect, ["rollback", "residual_count"], 0) or 0),
            },
            "automatic_effects": 0,
        }
        if verification.get("state") != "VERIFIED":
            faults.append("reflexive_dispatch_not_verified")
        if route_packet["terminal_outcome"] != "prepared":
            faults.append("reflexive_dispatch_not_prepared")
        if not selected_capabilities or not plan_nodes:
            faults.append("callable_route_or_plan_missing")
        if len(selected_capabilities) > 1 and not (
            route_packet["structured_execution"]["active"] is True
            and route_packet["structured_execution"]["terminal_outcome"] == "resolved"
            and route_packet["structured_execution"]["failed_count"] == 0
            and bool(route_packet["structured_execution"]["execution_digest"])
        ):
            faults.append("composite_plan_not_executed_or_held")
        if "assistant.deterministic_tool" in selected_capabilities and not (
            route_packet["tool"]["active"] is True
            and route_packet["tool"]["trigger_state"] in {"GREEN", "YELLOW"}
            and route_packet["tool"]["result_count"] > 0
            and bool(route_packet["tool"]["evidence_ref"])
        ):
            faults.append("selected_tool_capability_not_executed")
        if "assistant.plan_dag" in selected_capabilities and not (
            route_packet["planner"]["active"] is True
            and route_packet["planner"]["state"] in {"GREEN", "YELLOW"}
            and route_packet["planner"]["compiled_goal_count"] > 0
        ):
            faults.append("selected_planning_capability_not_executed")
        if "assistant.route_authority_effect" in selected_capabilities and not (
            route_packet["effect"]["enabled"] is True
            and route_packet["effect"]["ready"] is True
            and route_packet["effect"]["dispatch_bound"] is True
            and route_packet["effect"]["rollback_complete"] is True
            and route_packet["effect"]["rollback_residual_count"] == 0
        ):
            faults.append("selected_effect_capability_not_observed_and_rolled_back")
        model_prompt = render_integrated_prompt(prompt, context_rows, route_packet)

    context_binding = [
        {
            "address": row["address"],
            "certificate_id": row.get("certificate_id"),
            "content_sha256": row["content_sha256"],
            "characters": len(row["materialized_text"]),
            "execution_class": row.get("execution_class"),
            "taints": row.get("taints", []),
        }
        for row in context_rows
    ]
    binding = {
        "policy": ROUTE_POLICY,
        "execution_mode": execution_mode,
        "user_prompt_sha256": sha256_text(prompt),
        "model_prompt_sha256": sha256_text(model_prompt),
        "model_identity_sha256": model_identity.get("identity_sha256"),
        "context_pages": context_binding,
        "context_content_sha256": stable_hash(context_binding),
        "route_packet": route_packet,
        "route_packet_sha256": stable_hash(route_packet),
        "structural_verifier_id": STRUCTURAL_VERIFIER,
        "effect_sandbox_id": EFFECT_SANDBOX,
        "automatic_effects": 0,
        "maximum_model_calls": 1,
        "raw_prompt_stored": False,
        "raw_context_text_stored": False,
    }
    binding["request_binding_sha256"] = binding_digest(binding)
    return {
        "ready": not faults,
        "faults": sorted(set(faults)),
        "model_prompt": model_prompt,
        "binding": binding,
    }


def materialize_selected_context(
    selected_context: dict[str, Any],
    compiled_context: dict[str, Any],
    *,
    maximum_pages: int,
    maximum_characters: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    selected = [row for row in as_list(selected_context.get("selected_pages")) if isinstance(row, dict)]
    visible = {
        str(row.get("address") or ""): row
        for row in as_list(compiled_context.get("model_visible_pages"))
        if isinstance(row, dict) and row.get("address")
    }
    faults: list[str] = []
    if selected_context.get("ready") is not True or not selected:
        faults.append("selected_vcm_context_not_ready")
    rows: list[dict[str, Any]] = []
    used = 0
    for selection in selected[: max(0, maximum_pages)]:
        address = str(selection.get("address") or "")
        materialized = visible.get(address)
        if not materialized:
            faults.append(f"selected_vcm_page_not_materialized:{address or 'missing'}")
            continue
        taints = sorted({str(value) for value in as_list(materialized.get("taints")) if value})
        if set(taints).intersection(DENIED_CONTEXT_TAINTS):
            faults.append(f"selected_vcm_page_taint_denied:{address}")
            continue
        text = str(materialized.get("materialized_text") or "").strip()
        if not text:
            faults.append(f"selected_vcm_page_text_missing:{address}")
            continue
        remaining = maximum_characters - used
        if remaining <= 0:
            break
        text = text[:remaining]
        used += len(text)
        rows.append(
            {
                "address": address,
                "certificate_id": materialized.get("certificate_id"),
                "title": materialized.get("title"),
                "execution_class": materialized.get("execution_class"),
                "taints": taints,
                "materialized_text": text,
                "content_sha256": sha256_text(text),
            }
        )
    if not rows:
        faults.append("selected_vcm_content_not_consumable")
    return rows, sorted(set(faults))


def render_integrated_prompt(
    prompt: str,
    context_rows: list[dict[str, Any]],
    route_packet: dict[str, Any],
) -> str:
    context = [
        {
            "address": row["address"],
            "certificate_id": row.get("certificate_id"),
            "execution_class": row.get("execution_class"),
            "content_sha256": row["content_sha256"],
            "content": row["materialized_text"],
        }
        for row in context_rows
    ]
    return (
        f"{prompt}\n\n"
        "[theseus_verified_vcm_context]\n"
        f"{json.dumps(context, sort_keys=True, ensure_ascii=False)}\n\n"
        "[theseus_executed_route]\n"
        f"{json.dumps(route_packet, sort_keys=True, ensure_ascii=False)}"
    )


def build_route_integrity_receipt(
    *,
    execution_mode: str,
    request_binding: dict[str, Any],
    expected_model_identity: dict[str, Any],
    backend_payload: dict[str, Any],
) -> dict[str, Any]:
    """Independently verify one live backend result and decide release."""
    observed_identity = as_dict(get_path(backend_payload, ["backend", "identity"], {}))
    backend_request = as_dict(backend_payload.get("request"))
    response = as_dict(backend_payload.get("response"))
    route_packet = as_dict(request_binding.get("route_packet"))
    checks: dict[str, bool] = {
        "supported_execution_mode": execution_mode in LOCAL_MODEL_MODES,
        "request_binding_digest_valid": request_binding.get("request_binding_sha256") == binding_digest(request_binding),
        "execution_mode_bound": request_binding.get("execution_mode") == execution_mode
        and backend_request.get("execution_mode") == execution_mode,
        "local_model_backend_owned_by_canonical_assistant_runtime": backend_payload.get("policy")
        == "project_theseus_local_inference_backend_v1",
        "backend_completed": backend_payload.get("trigger_state") == "GREEN",
        "backend_prompt_bound": backend_request.get("prompt_sha256") == request_binding.get("model_prompt_sha256"),
        "backend_route_bound": backend_request.get("route_context_digest") == request_binding.get("route_packet_sha256"),
        "backend_model_identity_bound": model_identities_equal(expected_model_identity, observed_identity),
        "response_present": bool(str(response.get("answer") or "").strip()),
        "external_inference_zero": int(backend_payload.get("external_inference_calls") or 0) == 0,
        "teacher_calls_zero": int(backend_payload.get("teacher_calls") or 0) == 0,
        "public_calibration_consumption_zero": int(backend_payload.get("public_calibration_cases_consumed") or 0) == 0,
        "D2_consumption_zero": int(backend_payload.get("D2_cases_consumed") or 0) == 0,
        "public_training_rows_zero": int(backend_payload.get("public_training_rows_written") or 0) == 0,
        "fallback_returns_zero": int(backend_payload.get("fallback_return_count") or 0) == 0,
        "backend_user_facing_effects_zero": int(backend_payload.get("user_facing_effects") or 0) == 0,
        "single_model_call_budget": int(get_path(backend_payload, ["metrics", "local_model_inference_calls"], 0) or 0) == 1,
        "verification_controls_release_or_repair": False,
        "authority_controls_effect_scope": False,
        "effect_observation_and_exact_rollback_bound": False,
        "compiled_plan_executed_or_held": False,
        "selected_vcm_content_consumed_by_model": False,
        "route_changes_callable_capability_budget_tool_or_hold": False,
        "decorative_labels_cannot_satisfy_integrated_route": False,
    }
    structural_verification = {
        "verifier_id": STRUCTURAL_VERIFIER,
        "independent_from_backend": True,
        "checks": {
            key: checks[key]
            for key in (
                "backend_completed",
                "backend_prompt_bound",
                "backend_route_bound",
                "backend_model_identity_bound",
                "response_present",
                "external_inference_zero",
                "teacher_calls_zero",
                "public_calibration_consumption_zero",
                "D2_consumption_zero",
                "public_training_rows_zero",
                "fallback_returns_zero",
                "backend_user_facing_effects_zero",
                "single_model_call_budget",
            )
        },
        "task_correctness_evaluated": False,
    }
    structural_verification["passed"] = all(structural_verification["checks"].values())
    structural_verification["verification_sha256"] = stable_hash(structural_verification)
    checks["verification_controls_release_or_repair"] = structural_verification["passed"] is True

    effect = as_dict(route_packet.get("effect"))
    no_automatic_effects = int(request_binding.get("automatic_effects") or 0) == 0
    if execution_mode == DIRECT_MODE:
        checks["compiled_plan_executed_or_held"] = route_packet.get("planning") == "none"
        checks["selected_vcm_content_consumed_by_model"] = not as_list(request_binding.get("context_pages"))
        checks["route_changes_callable_capability_budget_tool_or_hold"] = route_packet.get("route_kind") == "direct_fixed_model"
        checks["authority_controls_effect_scope"] = no_automatic_effects
        checks["effect_observation_and_exact_rollback_bound"] = no_automatic_effects
        checks["decorative_labels_cannot_satisfy_integrated_route"] = route_packet.get("route_kind") == "direct_fixed_model"
    elif execution_mode == INTEGRATED_MODE:
        capabilities = [str(value) for value in as_list(route_packet.get("selected_capabilities")) if value]
        plan_nodes = [row for row in as_list(route_packet.get("plan_nodes")) if isinstance(row, dict)]
        execution = as_dict(route_packet.get("structured_execution"))
        composite_ok = (
            len(capabilities) <= 1
            or (
                execution.get("active") is True
                and execution.get("terminal_outcome") == "resolved"
                and int(execution.get("failed_count") or 0) == 0
                and bool(execution.get("execution_digest"))
            )
        )
        tool = as_dict(route_packet.get("tool"))
        planner = as_dict(route_packet.get("planner"))
        tool_ok = tool.get("required") is not True or (
            tool.get("active") is True
            and tool.get("trigger_state") in {"GREEN", "YELLOW"}
            and int(tool.get("result_count") or 0) > 0
            and bool(tool.get("evidence_ref"))
        )
        planner_ok = planner.get("required") is not True or (
            planner.get("active") is True
            and planner.get("state") in {"GREEN", "YELLOW"}
            and int(planner.get("compiled_goal_count") or 0) > 0
        )
        effect_ok = no_automatic_effects and (
            effect.get("enabled") is not True
            or (
                effect.get("ready") is True
                and effect.get("dispatch_bound") is True
                and effect.get("rollback_complete") is True
                and int(effect.get("rollback_residual_count") or 0) == 0
            )
        )
        context_pages = [row for row in as_list(request_binding.get("context_pages")) if isinstance(row, dict)]
        checks["compiled_plan_executed_or_held"] = bool(plan_nodes) and composite_ok and structural_verification["passed"]
        checks["selected_vcm_content_consumed_by_model"] = bool(context_pages) and all(
            len(str(row.get("content_sha256") or "")) == 64 and int(row.get("characters") or 0) > 0
            for row in context_pages
        )
        checks["route_changes_callable_capability_budget_tool_or_hold"] = (
            route_packet.get("verification_state") == "VERIFIED"
            and route_packet.get("terminal_outcome") == "prepared"
            and bool(capabilities)
            and bool(as_dict(route_packet.get("realized_limits")))
            and tool_ok
            and planner_ok
        )
        checks["authority_controls_effect_scope"] = effect_ok and execution.get("effect_authority_granted") is not True
        checks["effect_observation_and_exact_rollback_bound"] = effect_ok
        checks["decorative_labels_cannot_satisfy_integrated_route"] = (
            route_packet.get("route_kind") == "integrated_theseus"
            and checks["request_binding_digest_valid"]
            and checks["backend_route_bound"]
            and checks["selected_vcm_content_consumed_by_model"]
            and checks["compiled_plan_executed_or_held"]
        )

    release_allowed = all(checks.values())
    receipt = {
        "policy": ROUTE_POLICY,
        "execution_mode": execution_mode,
        "ready": release_allowed,
        "release_allowed": release_allowed,
        "disposition": "RELEASE_STRUCTURALLY_VERIFIED_LOCAL_RESPONSE" if release_allowed else "HOLD_ROUTE_INTEGRITY_FAILED",
        "checks": checks,
        "failed_checks": sorted(key for key, passed in checks.items() if not passed),
        "request_binding": copy.deepcopy(request_binding),
        "expected_model_identity": compact_identity(expected_model_identity),
        "observed_model_identity": compact_identity(observed_identity),
        "structural_verification": structural_verification,
        "pair_contract": {
            "model_identity_sha256": expected_model_identity.get("identity_sha256"),
            "structural_verifier_id": request_binding.get("structural_verifier_id"),
            "effect_sandbox_id": request_binding.get("effect_sandbox_id"),
            "maximum_model_calls": request_binding.get("maximum_model_calls"),
            "automatic_effects": request_binding.get("automatic_effects"),
        },
        "non_claims": [
            "This receipt verifies route mechanics and response release integrity, not task correctness or subsystem utility.",
            "A passing receipt does not establish Theseus-model credit, public transfer, or causal subsystem value.",
        ],
    }
    receipt["receipt_sha256"] = receipt_digest(receipt)
    return receipt


def compare_matched_pair(direct: dict[str, Any], integrated: dict[str, Any]) -> dict[str, Any]:
    direct_binding = as_dict(direct.get("request_binding"))
    integrated_binding = as_dict(integrated.get("request_binding"))
    checks = {
        "direct_mode_present": direct.get("execution_mode") == DIRECT_MODE,
        "integrated_mode_present": integrated.get("execution_mode") == INTEGRATED_MODE,
        "direct_route_integrity_ready": direct.get("ready") is True,
        "integrated_route_integrity_ready": integrated.get("ready") is True,
        "same_user_request": bool(direct_binding.get("user_prompt_sha256"))
        and direct_binding.get("user_prompt_sha256") == integrated_binding.get("user_prompt_sha256"),
        "model_decoder_snapshot_sandbox_evaluator_equal": direct.get("pair_contract") == integrated.get("pair_contract"),
        "direct_receipt_digest_valid": direct.get("receipt_sha256") == receipt_digest(direct),
        "integrated_receipt_digest_valid": integrated.get("receipt_sha256") == receipt_digest(integrated),
    }
    return {
        "policy": "project_theseus_local_model_matched_pair_integrity_v1",
        "ready": all(checks.values()),
        "checks": checks,
        "failed_checks": sorted(key for key, passed in checks.items() if not passed),
        "direct_receipt_sha256": direct.get("receipt_sha256"),
        "integrated_receipt_sha256": integrated.get("receipt_sha256"),
    }


def selected_capability_ids(dispatch: dict[str, Any]) -> list[str]:
    selected = {str(value) for value in as_list(get_path(dispatch, ["selection", "selected_proposal_ids"], []))}
    result: set[str] = set()
    for proposal in as_list(dispatch.get("proposals")):
        if not isinstance(proposal, dict) or str(proposal.get("proposal_id") or "") not in selected:
            continue
        for key in ("capability_ids", "capabilities"):
            result.update(str(value) for value in as_list(proposal.get(key)) if value)
        capability = proposal.get("capability_id")
        if capability:
            result.add(str(capability))
    return sorted(result)


def compact_plan_nodes(dispatch: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "node_id": row.get("node_id"),
            "capability_id": row.get("capability_id"),
            "depends_on": sorted(
                str(value)
                for value in as_list(row.get("depends_on") or row.get("dependencies"))
                if value
            ),
            "deadline_ms": row.get("deadline_ms"),
        }
        for row in as_list(dispatch.get("plan_nodes"))
        if isinstance(row, dict)
    ]


def model_identities_equal(expected: dict[str, Any], observed: dict[str, Any]) -> bool:
    return all(expected.get(field) == observed.get(field) for field in MODEL_FIELDS) and (
        expected.get("identity_sha256") == observed.get("identity_sha256")
    )


def compact_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return {field: identity.get(field) for field in (*MODEL_FIELDS, "identity_sha256")}


def binding_digest(binding: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in binding.items() if key != "request_binding_sha256"})


def receipt_digest(receipt: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in receipt.items() if key != "receipt_sha256"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit one local route or a matched direct/integrated pair.")
    parser.add_argument("--report", default="")
    parser.add_argument("--direct-report", default="")
    parser.add_argument("--integrated-report", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.report:
        report = read_json(resolve(args.report))
        receipt = as_dict(report.get("route_integrity"))
        digest_valid = receipt.get("receipt_sha256") == receipt_digest(receipt)
        result = {
            "policy": "project_theseus_route_integrity_receipt_audit_v1",
            "ready": receipt.get("ready") is True and digest_valid,
            "receipt_ready": receipt.get("ready") is True,
            "receipt_digest_valid": digest_valid,
            "failed_checks": receipt.get("failed_checks", []),
        }
    elif args.direct_report and args.integrated_report:
        direct = as_dict(read_json(resolve(args.direct_report)).get("route_integrity"))
        integrated = as_dict(read_json(resolve(args.integrated_report)).get("route_integrity"))
        result = compare_matched_pair(direct, integrated)
    else:
        raise SystemExit("provide --report or both --direct-report and --integrated-report")
    if args.out:
        write_json(resolve(args.out), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ready") else 2


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: str | Path) -> str:
    candidate = resolve(path)
    try:
        return str(candidate.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(candidate)


def read_json(path: str | Path) -> dict[str, Any]:
    candidate = resolve(path)
    if not candidate.is_file():
        return {}
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: str | Path, payload: Any) -> None:
    candidate = resolve(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    candidate = resolve(path)
    if not candidate.is_file():
        return ""
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
