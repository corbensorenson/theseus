#!/usr/bin/env python3
"""Sealed OpenAI measurement-only reference adapter for Project Theseus.

The adapter is callable, but execution fails closed unless a prospectively
sealed campaign explicitly authorizes reference calls. Offline qualification
never reads credentials or opens a network connection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "theseus_external_reference_control.json"
DEFAULT_QUALIFICATION = ROOT / "reports" / "theseus_openai_reference_adapter_qualification.json"
DENIED_PACKET_FIELDS = {
    "answer",
    "answer_family",
    "category",
    "expected",
    "hidden_tests",
    "required_constructs",
    "return_shape",
    "solution",
    "solution_body",
    "solution_expr",
    "source_task_id",
    "tests",
    "type_family",
}
REQUIRED_PACKET_FIELDS = {
    "packet_id",
    "campaign_id",
    "claim_id",
    "condition",
    "natural_request",
    "callable_signature",
    "allowed_runtime_context",
    "candidate_visible_protocol",
    "sealed_task_sha256",
    "evaluator_interface_sha256",
    "source_license_approved",
    "public_task",
}
ALLOWED_CONDITIONS = {"direct", "theseus_integrated"}


class ReferenceGateError(RuntimeError):
    """A sealed measurement-only invariant was not satisfied."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=rel(DEFAULT_POLICY))
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--packet", default="")
    parser.add_argument("--seal", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out", default="")
    parser.add_argument("--candidate-out", default="")
    args = parser.parse_args()

    policy_path = resolve(args.policy)
    policy = read_json(policy_path)
    if args.qualify_only:
        report = qualify_adapter(policy, policy_path=policy_path)
        output = resolve(args.out) if args.out else DEFAULT_QUALIFICATION
        write_json(output, report)
        print(json.dumps(qualification_view(report), indent=2, sort_keys=True))
        return 0 if report["trigger_state"] == "GREEN" else 2
    if not args.execute:
        raise SystemExit("use --qualify-only or --execute")
    if not args.packet or not args.seal or not args.out or not args.candidate_out:
        raise SystemExit("--execute requires --packet, --seal, --out, and --candidate-out")
    receipt = execute_reference_call(
        policy=policy,
        packet=read_json(resolve(args.packet)),
        seal=read_json(resolve(args.seal)),
        candidate_out=resolve(args.candidate_out),
    )
    write_json(resolve(args.out), receipt)
    print(json.dumps(receipt_view(receipt), indent=2, sort_keys=True))
    return 0 if receipt.get("trigger_state") == "GREEN" else 2


def build_request(policy: dict[str, Any], packet: dict[str, Any], seal: dict[str, Any]) -> dict[str, Any]:
    faults = validate_execution_authority(policy, packet, seal, require_calls_authorized=False)
    if faults:
        raise ReferenceGateError(",".join(faults))
    reference = mapping(policy.get("reference_model"))
    transport = mapping(policy.get("transport"))
    condition = str(packet.get("condition") or "")
    system_instructions = str(transport.get("system_instructions") or "").strip()
    if not system_instructions:
        raise ReferenceGateError("system_instructions_missing")
    visible_context = packet.get("allowed_runtime_context")
    prompt_packet = {
        "natural_request": packet["natural_request"],
        "callable_signature": packet["callable_signature"],
        "candidate_visible_protocol": packet["candidate_visible_protocol"],
        "allowed_runtime_context": visible_context,
        "system_condition": condition,
    }
    payload = {
        "model": reference.get("requested_model"),
        "reasoning": {
            "effort": reference.get("reasoning_effort"),
            "context": "current_turn",
        },
        "instructions": system_instructions,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": canonical(prompt_packet),
                    }
                ],
            }
        ],
        "store": False,
        "tools": [],
        "parallel_tool_calls": False,
        "metadata": {
            "campaign_id": str(packet["campaign_id"]),
            "packet_id": str(packet["packet_id"]),
            "condition": condition,
            "role": "measurement_only_reference_control",
        },
    }
    if "max_output_tokens" in payload:
        raise ReferenceGateError("project_selected_output_token_cap_present")
    return payload


def validate_execution_authority(
    policy: dict[str, Any],
    packet: dict[str, Any],
    seal: dict[str, Any],
    *,
    require_calls_authorized: bool,
) -> list[str]:
    faults: list[str] = []
    activation = mapping(policy.get("activation_scope"))
    reference = mapping(policy.get("reference_model"))
    transport = mapping(policy.get("transport"))
    admission = mapping(policy.get("admission_gate"))
    counters = mapping(policy.get("counters"))
    missing = sorted(REQUIRED_PACKET_FIELDS - set(packet))
    if missing:
        faults.extend(f"packet_field_missing:{field}" for field in missing)
    faults.extend(f"candidate_hidden_field_present:{field}" for field in sorted(hidden_fields(packet)))
    if packet.get("condition") not in ALLOWED_CONDITIONS:
        faults.append("condition_invalid")
    if packet.get("source_license_approved") is not True or packet.get("public_task") is not True:
        faults.append("task_not_public_license_approved")
    if reference.get("provider") != "OpenAI":
        faults.append("provider_not_openai")
    if reference.get("requested_model") != "gpt-5.6-luna":
        faults.append("model_not_frozen_luna")
    if reference.get("reasoning_effort") != "xhigh":
        faults.append("reasoning_effort_not_xhigh")
    if transport.get("api") != "responses" or transport.get("endpoint") != "https://api.openai.com/v1/responses":
        faults.append("responses_transport_not_exact")
    if transport.get("store") is not False or transport.get("tools") != []:
        faults.append("transport_state_or_tools_not_disabled")
    if transport.get("max_output_tokens") != "OMITTED_NO_PROJECT_QUALITY_CAP":
        faults.append("transport_output_boundary_policy_invalid")
    if require_calls_authorized and activation.get("reference_calls_authorized") is not True:
        faults.append("reference_calls_not_authorized")
    if any(admission.get(field) is not False for field in (
        "raw_private_data_allowed",
        "external_source_effects_allowed",
        "training_row_admission_allowed",
        "runtime_serving_allowed",
        "task_selection_or_tuning_allowed",
        "subsystem_or_prompt_tuning_allowed",
        "local_model_selection_allowed",
        "claim_queue_selection_allowed",
        "local_denominator_membership_allowed",
        "source_effect_credit_allowed",
        "automatic_book_support_promotion_allowed",
    )):
        faults.append("measurement_only_admission_boundary_open")
    if any(int(counters.get(field) or 0) != 0 for field in (
        "reference_calls_made",
        "accepted_training_rows",
        "user_facing_serving_tokens",
        "external_source_effects",
        "tasks_selected_with_reference_outputs",
    )):
        faults.append("preactivation_counter_nonzero")
    faults.extend(validate_seal(policy, packet, seal))
    return sorted(set(faults))


def validate_seal(policy: dict[str, Any], packet: dict[str, Any], seal: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if seal.get("state") != "SEALED_BEFORE_ANY_CANDIDATE_CALL":
        faults.append("campaign_not_prospectively_sealed")
    if seal.get("campaign_id") != packet.get("campaign_id"):
        faults.append("campaign_id_mismatch")
    if seal.get("claim_id") != packet.get("claim_id"):
        faults.append("claim_id_mismatch")
    if seal.get("task_selection_complete") is not True or seal.get("any_candidate_outcome_visible") is not False:
        faults.append("task_selection_not_blindly_closed")
    if seal.get("all_four_cells_sealed") is not True:
        faults.append("factorial_cells_not_sealed")
    if seal.get("reference_outputs_visible_to_local_arms") is not False:
        faults.append("reference_output_information_flow_open")
    if seal.get("reference_outputs_eligible_for_training") is not False:
        faults.append("reference_training_flow_open")
    if seal.get("reference_outputs_eligible_for_task_selection") is not False:
        faults.append("reference_task_selection_flow_open")
    if seal.get("evaluator_interface_sha256") != packet.get("evaluator_interface_sha256"):
        faults.append("evaluator_interface_mismatch")
    packet_hashes = {str(value) for value in sequence(seal.get("sealed_task_sha256s"))}
    if str(packet.get("sealed_task_sha256") or "") not in packet_hashes:
        faults.append("task_not_in_sealed_pool")
    reference = mapping(policy.get("reference_model"))
    if seal.get("reference_model") != reference.get("requested_model"):
        faults.append("sealed_model_mismatch")
    if seal.get("reasoning_effort") != reference.get("reasoning_effort"):
        faults.append("sealed_effort_mismatch")
    spend = mapping(seal.get("spend_authority"))
    maximum = float(spend.get("maximum_total_usd") or 0.0)
    observed = float(spend.get("observed_spend_before_call_usd") or 0.0)
    worst_case = worst_case_call_cost_usd(policy)
    if maximum <= 0 or observed < 0 or maximum - observed + 1e-12 < worst_case:
        faults.append("campaign_spend_authority_insufficient_for_uncapped_physical_call")
    if int(spend.get("maximum_reference_calls") or 0) <= int(spend.get("reference_calls_before_call") or 0):
        faults.append("campaign_call_authority_exhausted")
    return sorted(set(faults))


def execute_reference_call(
    *,
    policy: dict[str, Any],
    packet: dict[str, Any],
    seal: dict[str, Any],
    candidate_out: Path,
) -> dict[str, Any]:
    faults = validate_execution_authority(policy, packet, seal, require_calls_authorized=True)
    if faults:
        raise ReferenceGateError(",".join(faults))
    payload = build_request(policy, packet, seal)
    transport = mapping(policy.get("transport"))
    credential_name = str(transport.get("credential_environment_variable") or "")
    api_key = os.environ.get(credential_name, "") if credential_name else ""
    if not api_key:
        raise ReferenceGateError("openai_credential_missing")
    request_id = stable_hash(
        {
            "campaign_id": packet["campaign_id"],
            "packet_id": packet["packet_id"],
            "condition": packet["condition"],
            "payload": payload,
        }
    )
    body = canonical(payload).encode("utf-8")
    request = urllib.request.Request(
        str(transport["endpoint"]),
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": request_id,
        },
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=int(transport.get("timeout_seconds") or 900)) as response:
            response_body = response.read()
            provider_request_id = str(response.headers.get("x-request-id") or "")
    except urllib.error.HTTPError as exc:
        raise ReferenceGateError(f"provider_http_error:{exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ReferenceGateError(f"provider_transport_error:{type(exc.reason).__name__}") from exc
    try:
        provider = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceGateError("provider_response_not_json") from exc
    candidate = extract_output_text(provider)
    if not candidate:
        raise ReferenceGateError("provider_output_text_missing")
    write_private_text(candidate_out, candidate)
    usage = normalize_usage(mapping(provider.get("usage")))
    receipt = {
        "policy": "project_theseus_openai_measurement_reference_receipt_v1",
        "created_utc": now(),
        "trigger_state": "GREEN",
        "campaign_id": packet["campaign_id"],
        "claim_id": packet["claim_id"],
        "packet_id": packet["packet_id"],
        "condition": packet["condition"],
        "provider": "OpenAI",
        "requested_model": payload["model"],
        "observed_model": provider.get("model"),
        "reasoning_effort": payload["reasoning"]["effort"],
        "service_tier": provider.get("service_tier"),
        "response_id": provider.get("id"),
        "provider_request_id": provider_request_id,
        "response_status": provider.get("status"),
        "incomplete_details": provider.get("incomplete_details"),
        "request_sha256": stable_hash(payload),
        "sealed_task_sha256": packet["sealed_task_sha256"],
        "evaluator_interface_sha256": packet["evaluator_interface_sha256"],
        "candidate": {
            "path": rel(candidate_out),
            "sha256": sha256_text(candidate),
            "bytes": len(candidate.encode("utf-8")),
            "user_facing_serving_allowed": False,
            "training_admission_allowed": False,
            "task_selection_allowed": False,
        },
        "usage": usage,
        "observed_cost_usd": observed_cost_usd(policy, usage),
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "finish_reason": response_finish_reason(provider),
        "physical_boundary_hit": provider.get("status") == "incomplete",
        "physical_boundary_disposition": (
            "INVALID_OBSERVATION_CONTEXT_OR_PROVIDER_BOUNDARY"
            if provider.get("status") == "incomplete"
            else "NOT_HIT"
        ),
        "external_inference_calls": 1,
        "accepted_training_rows": 0,
        "user_facing_serving_tokens": 0,
        "external_source_effects": 0,
        "tasks_selected_with_reference_outputs": 0,
    }
    if receipt["observed_model"] != receipt["requested_model"]:
        receipt["trigger_state"] = "RED"
        receipt["faults"] = ["provider_model_identity_mismatch"]
    else:
        receipt["faults"] = []
    return receipt


def qualify_adapter(policy: dict[str, Any], *, policy_path: Path) -> dict[str, Any]:
    packet = qualification_packet(policy)
    seal = qualification_seal(policy, packet)
    checks: dict[str, bool] = {}
    faults = validate_execution_authority(policy, packet, seal, require_calls_authorized=False)
    checks["valid_sealed_packet_accepted"] = not faults
    try:
        payload = build_request(policy, packet, seal)
    except ReferenceGateError:
        payload = {}
    checks.update(
        {
            "responses_api_bound": payload.get("model") == "gpt-5.6-luna",
            "xhigh_bound": mapping(payload.get("reasoning")).get("effort") == "xhigh",
            "current_turn_reasoning_context": mapping(payload.get("reasoning")).get("context") == "current_turn",
            "store_disabled": payload.get("store") is False,
            "tools_disabled": payload.get("tools") == [],
            "project_output_token_cap_absent": "max_output_tokens" not in payload,
            "measurement_role_bound": get_path(payload, ["metadata", "role"]) == "measurement_only_reference_control",
            "reference_calls_remain_unauthorized": get_path(policy, ["activation_scope", "reference_calls_authorized"]) is False,
            "preactivation_counters_zero": all(int(value or 0) == 0 for value in mapping(policy.get("counters")).values()),
            "adapter_source_hash_bound": mapping(policy.get("transport")).get("adapter_sha256") == file_sha256(Path(__file__)),
            "policy_source_hash_recorded": bool(file_sha256(policy_path)),
            "worst_case_cost_positive": worst_case_call_cost_usd(policy) > 0,
        }
    )
    negative_controls = {}
    negative_controls["hidden_answer_metadata"] = rejects(
        policy,
        {**packet, "hidden_tests": ["forbidden"]},
        seal,
        expected="candidate_hidden_field_present:hidden_tests",
    )
    negative_controls["unsealed_campaign"] = rejects(
        policy,
        packet,
        {**seal, "state": "OPEN"},
        expected="campaign_not_prospectively_sealed",
    )
    negative_controls["insufficient_spend_authority"] = rejects(
        policy,
        packet,
        {**seal, "spend_authority": {**mapping(seal.get("spend_authority")), "maximum_total_usd": 0}},
        expected="campaign_spend_authority_insufficient_for_uncapped_physical_call",
    )
    negative_controls["reference_output_visible_to_local"] = rejects(
        policy,
        packet,
        {**seal, "reference_outputs_visible_to_local_arms": True},
        expected="reference_output_information_flow_open",
    )
    negative_controls["execution_without_activation"] = (
        "reference_calls_not_authorized"
        in validate_execution_authority(policy, packet, seal, require_calls_authorized=True)
    )
    checks["all_negative_controls_rejected"] = all(negative_controls.values())
    hard_gaps = sorted(name for name, passed in checks.items() if not passed)
    return {
        "policy": "project_theseus_openai_reference_adapter_qualification_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if not hard_gaps else "RED",
        "adapter": rel(Path(__file__)),
        "adapter_sha256": file_sha256(Path(__file__)),
        "policy_path": rel(policy_path),
        "policy_sha256": file_sha256(policy_path),
        "request_shape_sha256": stable_hash(payload),
        "checks": checks,
        "negative_controls": negative_controls,
        "hard_gaps": hard_gaps,
        "price_basis": mapping(policy.get("price_basis")),
        "worst_case_physical_call_cost_usd": worst_case_call_cost_usd(policy),
        "counters": {
            "reference_calls_made": 0,
            "accepted_training_rows": 0,
            "user_facing_serving_tokens": 0,
            "external_source_effects": 0,
            "tasks_selected_with_reference_outputs": 0,
        },
        "maximum_inference": "The exact Luna Responses API adapter is offline-qualified and remains unable to call until a future campaign is prospectively sealed and explicitly activated.",
        "non_claims": [
            "No Luna call was made.",
            "This does not establish task quality, subsystem transfer, serving eligibility, or training-data eligibility.",
        ],
    }


def qualification_packet(policy: dict[str, Any]) -> dict[str, Any]:
    campaign_id = str(policy.get("campaign_id") or "")
    claim_id = str(get_path(policy, ["activation_scope", "current_claim_id"]) or "")
    base = {
        "packet_id": "offline-qualification-direct",
        "campaign_id": campaign_id,
        "claim_id": claim_id,
        "condition": "direct",
        "natural_request": "Return a complete artifact satisfying the callable signature.",
        "callable_signature": "solve(value: str) -> str",
        "allowed_runtime_context": [],
        "candidate_visible_protocol": {"tools": [], "effects": []},
        "evaluator_interface_sha256": "e" * 64,
        "source_license_approved": True,
        "public_task": True,
    }
    base["sealed_task_sha256"] = stable_hash(
        {key: value for key, value in base.items() if key != "sealed_task_sha256"}
    )
    return base


def qualification_seal(policy: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": "SEALED_BEFORE_ANY_CANDIDATE_CALL",
        "campaign_id": packet["campaign_id"],
        "claim_id": packet["claim_id"],
        "task_selection_complete": True,
        "any_candidate_outcome_visible": False,
        "all_four_cells_sealed": True,
        "reference_outputs_visible_to_local_arms": False,
        "reference_outputs_eligible_for_training": False,
        "reference_outputs_eligible_for_task_selection": False,
        "evaluator_interface_sha256": packet["evaluator_interface_sha256"],
        "sealed_task_sha256s": [packet["sealed_task_sha256"]],
        "reference_model": get_path(policy, ["reference_model", "requested_model"]),
        "reasoning_effort": get_path(policy, ["reference_model", "reasoning_effort"]),
        "spend_authority": {
            "maximum_total_usd": round(worst_case_call_cost_usd(policy), 6),
            "observed_spend_before_call_usd": 0.0,
            "maximum_reference_calls": 1,
            "reference_calls_before_call": 0,
        },
    }


def worst_case_call_cost_usd(policy: dict[str, Any]) -> float:
    price = mapping(policy.get("price_basis"))
    limits = mapping(price.get("physical_limits"))
    input_tokens = int(limits.get("context_window_tokens") or 0)
    output_tokens = int(limits.get("maximum_output_tokens") or 0)
    threshold = int(price.get("long_context_threshold_tokens") or 0)
    input_multiplier = float(price.get("long_context_input_multiplier") or 1.0) if input_tokens > threshold else 1.0
    output_multiplier = float(price.get("long_context_output_multiplier") or 1.0) if input_tokens > threshold else 1.0
    return round(
        input_tokens / 1_000_000 * float(price.get("input_usd_per_million") or 0.0) * input_multiplier
        + output_tokens / 1_000_000 * float(price.get("output_usd_per_million") or 0.0) * output_multiplier,
        6,
    )


def observed_cost_usd(policy: dict[str, Any], usage: dict[str, int]) -> float:
    price = mapping(policy.get("price_basis"))
    input_tokens = int(usage.get("input_tokens") or 0)
    cached = int(usage.get("cached_input_tokens") or 0)
    cache_write = min(
        int(usage.get("cache_write_input_tokens") or 0),
        max(0, input_tokens - cached),
    )
    uncached = max(0, input_tokens - cached - cache_write)
    output = int(usage.get("output_tokens") or 0)
    threshold = int(price.get("long_context_threshold_tokens") or 0)
    input_multiplier = float(price.get("long_context_input_multiplier") or 1.0) if input_tokens > threshold else 1.0
    output_multiplier = float(price.get("long_context_output_multiplier") or 1.0) if input_tokens > threshold else 1.0
    return round(
        uncached / 1_000_000 * float(price.get("input_usd_per_million") or 0.0) * input_multiplier
        + cached / 1_000_000 * float(price.get("cached_input_usd_per_million") or 0.0) * input_multiplier
        + cache_write
        / 1_000_000
        * float(price.get("input_usd_per_million") or 0.0)
        * float(price.get("cache_write_multiplier") or 1.0)
        * input_multiplier
        + output / 1_000_000 * float(price.get("output_usd_per_million") or 0.0) * output_multiplier,
        8,
    )


def normalize_usage(usage: dict[str, Any]) -> dict[str, int]:
    input_details = mapping(usage.get("input_tokens_details"))
    output_details = mapping(usage.get("output_tokens_details"))
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "cached_input_tokens": int(input_details.get("cached_tokens") or 0),
        "cache_write_input_tokens": int(input_details.get("cache_write_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "reasoning_tokens": int(output_details.get("reasoning_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def extract_output_text(response: dict[str, Any]) -> str:
    parts = []
    for item in sequence(response.get("output")):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in sequence(item.get("content")):
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text") or ""))
    return "".join(parts).strip()


def response_finish_reason(response: dict[str, Any]) -> str:
    if response.get("status") == "completed":
        return "completed"
    incomplete = mapping(response.get("incomplete_details"))
    return str(incomplete.get("reason") or response.get("status") or "unknown")


def hidden_fields(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in DENIED_PACKET_FIELDS:
                found.add(str(key))
            found.update(hidden_fields(item))
    elif isinstance(value, list):
        for item in value:
            found.update(hidden_fields(item))
    return found


def rejects(
    policy: dict[str, Any],
    packet: dict[str, Any],
    seal: dict[str, Any],
    *,
    expected: str,
) -> bool:
    return expected in validate_execution_authority(
        policy,
        packet,
        seal,
        require_calls_authorized=False,
    )


def receipt_view(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: receipt.get(key)
        for key in (
            "policy",
            "trigger_state",
            "campaign_id",
            "packet_id",
            "condition",
            "requested_model",
            "observed_model",
            "reasoning_effort",
            "usage",
            "observed_cost_usd",
            "finish_reason",
            "physical_boundary_disposition",
            "faults",
        )
    }


def qualification_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report.get("policy"),
        "trigger_state": report.get("trigger_state"),
        "hard_gaps": report.get("hard_gaps"),
        "checks": report.get("checks"),
        "negative_controls": report.get("negative_controls"),
        "worst_case_physical_call_cost_usd": report.get("worst_case_physical_call_cost_usd"),
        "counters": report.get("counters"),
    }


def write_private_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def get_path(value: Any, path: list[str]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any) -> str:
    return sha256_text(canonical(value))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: str | Path) -> str:
    candidate = resolve(path)
    try:
        return str(candidate.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(candidate)


if __name__ == "__main__":
    raise SystemExit(main())
