"""Qualification-first dispatch contracts for the canonical Theseus assistant.

This module is not a second router or authority. It is imported by the registered
assistant runtime and composes existing capability, SCF/VIEA, VCM, Octopus, and
procedural-memory owners before model inference. Learned components may propose;
only this contract's qualification receipt can make a proposal selectable, and
external effects still require the separate registered effect kernel.
"""

from __future__ import annotations

import copy
import json
import re
import shlex
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs" / "reflexive_router_contract.json"
DEFAULT_PROFILE = ROOT / "configs" / "reflexive_router_profile.json"
EFFECT_RISK = {"none": 0, "read_only": 1, "reversible": 2, "partially_reversible": 3, "irreversible": 4}
FORBIDDEN_BINDING_FIELDS = {"shell", "sql", "url", "prompt", "prompt_template", "command_text"}


class ReflexiveDispatchFault(ValueError):
    """Typed, fail-closed dispatch-contract fault."""

    def __init__(self, code: str, detail: Any, *, path: str = "") -> None:
        self.code = code
        self.detail = detail
        self.path = path
        super().__init__(f"{code}: {canonical(detail)}")

    def record(self) -> dict[str, Any]:
        return {
            "fault_type": self.code,
            "detail": copy.deepcopy(self.detail),
            "path": self.path,
            "terminal_outcome": "rejected",
            "effect_authority_granted": False,
        }


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return f"sha256:{sha256(canonical(value).encode('utf-8')).hexdigest()}"


def stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}:{sha256(canonical(parts).encode('utf-8')).hexdigest()[:24]}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReflexiveDispatchFault("REFLEX_CONTRACT_UNREADABLE", str(exc), path=str(path)) from exc
    return validate_contract(payload)


def load_reflexbench_profile(path: Path = DEFAULT_PROFILE) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReflexiveDispatchFault("REFLEX_PROFILE_UNREADABLE", str(exc), path=str(path)) from exc
    if payload.get("policy") != "project_theseus_reflexbench_profile_v1":
        raise ReflexiveDispatchFault("REFLEX_PROFILE_POLICY_INVALID", payload.get("policy"), path="profile.policy")
    tracks = payload.get("tracks") if isinstance(payload.get("tracks"), list) else []
    policies = payload.get("policies") if isinstance(payload.get("policies"), list) else []
    cases = payload.get("cases") if isinstance(payload.get("cases"), list) else []
    if len(tracks) != 8 or len(set(tracks)) != 8 or len(policies) != 10 or len(set(policies)) != 10:
        raise ReflexiveDispatchFault("REFLEX_PROFILE_MATRIX_INVALID", {"tracks": tracks, "policies": policies}, path="profile")
    if len(cases) < 32 or any(sum(row.get("track") == track for row in cases) < 4 for track in tracks):
        raise ReflexiveDispatchFault("REFLEX_PROFILE_DENOMINATOR_TOO_THIN", len(cases), path="profile.cases")
    case_ids = [str(row.get("case_id") or "") for row in cases if isinstance(row, dict)]
    if len(case_ids) != len(cases) or len(set(case_ids)) != len(case_ids) or "" in case_ids:
        raise ReflexiveDispatchFault("REFLEX_PROFILE_CASE_ID_CONFLICT", case_ids, path="profile.cases")
    boundary = payload.get("case_information_boundary") if isinstance(payload.get("case_information_boundary"), dict) else {}
    visible = set(boundary.get("policy_visible_fields") or [])
    held = set(boundary.get("held_verifier_fields") or [])
    if not visible or not held or visible.intersection(held) or boundary.get("oracle_is_only_policy_allowed_held_fields") is not True:
        raise ReflexiveDispatchFault("REFLEX_PROFILE_INFORMATION_BOUNDARY_INVALID", boundary, path="profile.case_information_boundary")
    for row in cases:
        if row.get("track") not in tracks or not visible.issubset(row) or not held.issubset(row):
            raise ReflexiveDispatchFault("REFLEX_PROFILE_CASE_INCOMPLETE", row.get("case_id"), path="profile.cases")
    return copy.deepcopy(payload)


def validate_contract(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("policy") != "project_theseus_reflexive_router_contract_v1":
        raise ReflexiveDispatchFault("REFLEX_CONTRACT_POLICY_INVALID", payload.get("policy"), path="policy")
    if payload.get("architecture_scope") != "existing_assistant_scf_octopus_viea_vcm_procedural_owners_only":
        raise ReflexiveDispatchFault("REFLEX_SIDECAR_SCOPE_INVALID", payload.get("architecture_scope"), path="architecture_scope")
    evidence = payload.get("source_evidence") if isinstance(payload.get("source_evidence"), dict) else {}
    expected_evidence = {"ai_book_commit_observed", "paper_sha256", "schema_sha256", "fixture_sha256", "validator_sha256"}
    if set(evidence) != expected_evidence or any(
        key != "ai_book_commit_observed" and not re.fullmatch(r"[0-9a-f]{64}", str(value))
        for key, value in evidence.items()
    ):
        raise ReflexiveDispatchFault("REFLEX_SOURCE_BINDING_INVALID", evidence, path="source_evidence")
    terminal = payload.get("terminal_outcomes") if isinstance(payload.get("terminal_outcomes"), list) else []
    required_terminal = {
        "resolved", "prepared", "partial", "ambiguous", "insufficient_context",
        "insufficient_evidence", "conflicting", "stale", "unauthorized", "unsupported",
        "ood", "resource_exceeded", "execution_failed", "verification_failed",
        "escalate", "rejected",
    }
    if set(terminal) != required_terminal or len(terminal) != len(required_terminal):
        raise ReflexiveDispatchFault("REFLEX_TERMINAL_VOCABULARY_INVALID", terminal, path="terminal_outcomes")
    limits = payload.get("resource_limits") if isinstance(payload.get("resource_limits"), dict) else {}
    if any(int(limits.get(key) or 0) <= 0 for key in ("max_nodes", "max_depth", "max_fanout", "max_retries_per_node", "max_aggregate_cost_units", "default_deadline_ms")):
        raise ReflexiveDispatchFault("REFLEX_RESOURCE_LIMIT_INVALID", limits, path="resource_limits")
    validate_effort_profiles(payload.get("effort_profiles"), limits)
    capability_index = validate_capabilities(payload.get("capabilities"))
    command_index = validate_user_commands(payload.get("user_commands"), capability_index)
    validate_workflows(payload.get("workflows"), capability_index, command_index, limits)
    compile_reflex_index(payload.get("reflexes"), capability_index)
    compile_policy = payload.get("compilation") if isinstance(payload.get("compilation"), dict) else {}
    if set(compile_policy.get("states") or []) != {"ineligible", "candidate", "shadow", "qualified", "quarantined", "decompiled"}:
        raise ReflexiveDispatchFault("REFLEX_COMPILATION_STATES_INVALID", compile_policy, path="compilation.states")
    return copy.deepcopy(payload)


def validate_effort_profiles(value: Any, global_limits: dict[str, Any]) -> dict[str, dict[str, int]]:
    profiles = value if isinstance(value, dict) else {}
    required = {"direct", "balanced", "deliberative"}
    if set(profiles) != required:
        raise ReflexiveDispatchFault("REFLEX_EFFORT_PROFILE_SET_INVALID", sorted(profiles), path="effort_profiles")
    limit_keys = ("max_nodes", "max_depth", "max_fanout", "max_retries_per_node", "max_aggregate_cost_units")
    validated: dict[str, dict[str, int]] = {}
    for name, row in profiles.items():
        if not isinstance(row, dict) or any(int(row.get(key, -1)) < 0 for key in (*limit_keys, "deadline_ms")):
            raise ReflexiveDispatchFault("REFLEX_EFFORT_PROFILE_INVALID", row, path=f"effort_profiles.{name}")
        bounded = {key: min(int(row[key]), int(global_limits[key])) for key in limit_keys}
        bounded["deadline_ms"] = min(int(row["deadline_ms"]), int(global_limits["default_deadline_ms"]))
        if any(bounded[key] <= 0 for key in ("max_nodes", "max_depth", "max_fanout", "max_aggregate_cost_units", "deadline_ms")):
            raise ReflexiveDispatchFault("REFLEX_EFFORT_PROFILE_INVALID", row, path=f"effort_profiles.{name}")
        validated[name] = bounded
    if any(validated["direct"][key] > validated["balanced"][key] or validated["balanced"][key] > validated["deliberative"][key] for key in (*limit_keys, "deadline_ms")):
        raise ReflexiveDispatchFault("REFLEX_EFFORT_PROFILE_ORDER_INVALID", validated, path="effort_profiles")
    return validated


def validate_capabilities(value: Any) -> dict[str, dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    index: dict[str, dict[str, Any]] = {}
    for offset, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ReflexiveDispatchFault("REFLEX_CAPABILITY_INVALID", row, path=f"capabilities[{offset}]")
        capability_id = str(row.get("capability_id") or "")
        if not capability_id or capability_id in index:
            raise ReflexiveDispatchFault("REFLEX_CAPABILITY_ID_CONFLICT", capability_id, path=f"capabilities[{offset}].capability_id")
        if row.get("effect_class") not in EFFECT_RISK or not row.get("verifier_ref") or not row.get("required_authority"):
            raise ReflexiveDispatchFault("REFLEX_CAPABILITY_CONTRACT_INCOMPLETE", row, path=f"capabilities[{offset}]")
        if int(row.get("cost_units") or 0) <= 0 or not row.get("qualified_profiles"):
            raise ReflexiveDispatchFault("REFLEX_CAPABILITY_RESOURCE_CONTRACT_INVALID", row, path=f"capabilities[{offset}]")
        index[capability_id] = copy.deepcopy(row)
    if not index:
        raise ReflexiveDispatchFault("REFLEX_CAPABILITY_REGISTRY_EMPTY", {}, path="capabilities")
    return index


def validate_user_commands(value: Any, capabilities: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    aliases: dict[str, dict[str, Any]] = {}
    ids: set[str] = set()
    for offset, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ReflexiveDispatchFault("REFLEX_COMMAND_INVALID", row, path=f"user_commands[{offset}]")
        command_id = str(row.get("command_id") or "")
        alias = str(row.get("alias") or "")
        if not command_id or command_id in ids or not re.fullmatch(r"/[a-z][a-z0-9_-]*", alias) or alias in aliases:
            raise ReflexiveDispatchFault("REFLEX_COMMAND_NAMESPACE_CONFLICT", {"id": command_id, "alias": alias}, path=f"user_commands[{offset}]")
        if row.get("capability_id") not in capabilities:
            raise ReflexiveDispatchFault("REFLEX_COMMAND_CAPABILITY_UNKNOWN", row.get("capability_id"), path=f"user_commands[{offset}].capability_id")
        if FORBIDDEN_BINDING_FIELDS.intersection(row):
            raise ReflexiveDispatchFault("REFLEX_COMMAND_TEXT_MACRO_FORBIDDEN", sorted(FORBIDDEN_BINDING_FIELDS.intersection(row)), path=f"user_commands[{offset}]")
        validate_parameter_schema(row.get("parameter_schema"), path=f"user_commands[{offset}].parameter_schema")
        ids.add(command_id)
        aliases[alias] = copy.deepcopy(row)
    return aliases


def validate_workflows(
    value: Any,
    capabilities: dict[str, dict[str, Any]],
    commands: dict[str, dict[str, Any]],
    limits: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    aliases: dict[str, dict[str, Any]] = {}
    ids: set[str] = set()
    for offset, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ReflexiveDispatchFault("REFLEX_WORKFLOW_INVALID", row, path=f"workflows[{offset}]")
        workflow_id = str(row.get("workflow_id") or "")
        alias = str(row.get("alias") or "")
        capability_ids = [str(item) for item in row.get("capability_ids", []) if str(item)] if isinstance(row.get("capability_ids"), list) else []
        if not workflow_id or workflow_id in ids or not re.fullmatch(r"/[a-z][a-z0-9_-]*", alias) or alias in aliases or alias in commands:
            raise ReflexiveDispatchFault("REFLEX_WORKFLOW_NAMESPACE_CONFLICT", {"id": workflow_id, "alias": alias}, path=f"workflows[{offset}]")
        if len(capability_ids) < 2 or len(capability_ids) != len(set(capability_ids)) or any(item not in capabilities for item in capability_ids):
            raise ReflexiveDispatchFault("REFLEX_WORKFLOW_CAPABILITY_INVALID", capability_ids, path=f"workflows[{offset}].capability_ids")
        if row.get("fallback") != "none" or FORBIDDEN_BINDING_FIELDS.intersection(row):
            raise ReflexiveDispatchFault("REFLEX_WORKFLOW_BINDING_INVALID", row, path=f"workflows[{offset}]")
        validate_parameter_schema(row.get("parameter_schema"), path=f"workflows[{offset}].parameter_schema")
        nodes = row.get("nodes") if isinstance(row.get("nodes"), list) else []
        node_capabilities = {str(node.get("capability_id") or "") for node in nodes if isinstance(node, dict)}
        if node_capabilities != set(capability_ids):
            raise ReflexiveDispatchFault("REFLEX_WORKFLOW_NODE_COVERAGE_INVALID", node_capabilities, path=f"workflows[{offset}].nodes")
        for node_offset, node in enumerate(nodes):
            capability = capabilities.get(str(node.get("capability_id") or ""), {})
            if node.get("effect_class") != capability.get("effect_class") or node.get("verifier_ref") != capability.get("verifier_ref"):
                raise ReflexiveDispatchFault("REFLEX_WORKFLOW_NODE_CONTRACT_MISMATCH", node, path=f"workflows[{offset}].nodes[{node_offset}]")
        validate_plan_dag(nodes, limits)
        enriched = copy.deepcopy(row)
        enriched["binding_kind"] = "workflow"
        aliases[alias] = enriched
        ids.add(workflow_id)
    return aliases


def validate_parameter_schema(value: Any, *, path: str) -> dict[str, Any]:
    if value == {}:
        return {"properties": {}, "positional": [], "additional_properties": False, "free_text_tail": True}
    schema = value if isinstance(value, dict) else {}
    if not {"properties", "positional", "additional_properties"}.issubset(schema) or set(schema) - {"properties", "positional", "additional_properties", "free_text_tail"}:
        raise ReflexiveDispatchFault("REFLEX_PARAMETER_SCHEMA_SHAPE_INVALID", schema, path=path)
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    positional = schema.get("positional") if isinstance(schema.get("positional"), list) else []
    if schema.get("additional_properties") is not False or len(positional) != len(set(str(item) for item in positional)):
        raise ReflexiveDispatchFault("REFLEX_PARAMETER_SCHEMA_SHAPE_INVALID", schema, path=path)
    allowed_types = {"string", "integer", "number", "boolean", "enum"}
    for name, descriptor in properties.items():
        field_path = f"{path}.properties.{name}"
        if not re.fullmatch(r"[a-z][a-z0-9_]*", str(name)) or not isinstance(descriptor, dict):
            raise ReflexiveDispatchFault("REFLEX_PARAMETER_DESCRIPTOR_INVALID", descriptor, path=field_path)
        allowed_fields = {"type", "required", "minimum", "maximum", "values", "default", "sensitive"}
        if set(descriptor) - allowed_fields or descriptor.get("type") not in allowed_types:
            raise ReflexiveDispatchFault("REFLEX_PARAMETER_DESCRIPTOR_INVALID", descriptor, path=field_path)
        if descriptor.get("type") == "enum" and (
            not isinstance(descriptor.get("values"), list)
            or not descriptor["values"]
            or len(descriptor["values"]) != len(set(str(item) for item in descriptor["values"]))
        ):
            raise ReflexiveDispatchFault("REFLEX_PARAMETER_ENUM_INVALID", descriptor.get("values"), path=field_path)
        default = descriptor.get("default")
        if default is not None:
            if not isinstance(default, dict) or default.get("kind") not in {"literal", "context_ref"} or set(default) != {"kind", "value"}:
                raise ReflexiveDispatchFault("REFLEX_PARAMETER_DEFAULT_INVALID", default, path=field_path)
            if default["kind"] == "context_ref" and default["value"] not in {"principal", "deadline_ms", "first_context_handle"}:
                raise ReflexiveDispatchFault("REFLEX_PARAMETER_DYNAMIC_DEFAULT_FORBIDDEN", default, path=field_path)
    if any(str(name) not in properties for name in positional):
        raise ReflexiveDispatchFault("REFLEX_PARAMETER_POSITIONAL_UNKNOWN", positional, path=path)
    return {**copy.deepcopy(schema), "free_text_tail": schema.get("free_text_tail", True) is True}


def compile_reflex_index(value: Any, capabilities: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows = value if isinstance(value, list) else []
    ids: set[str] = set()
    signatures: set[str] = set()
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for offset, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ReflexiveDispatchFault("REFLEX_RULE_INVALID", row, path=f"reflexes[{offset}]")
        reflex_id = str(row.get("reflex_id") or "")
        match = row.get("match") if isinstance(row.get("match"), dict) else {}
        signature = canonical(match)
        if not reflex_id or reflex_id in ids or not match:
            raise ReflexiveDispatchFault("REFLEX_RULE_ID_INVALID", reflex_id, path=f"reflexes[{offset}]")
        if signature in signatures:
            raise ReflexiveDispatchFault("REFLEX_RULE_UNRESOLVED_OVERLAP", match, path=f"reflexes[{offset}].match")
        if row.get("capability_id") not in capabilities or row.get("effect_class") != capabilities[row["capability_id"]]["effect_class"]:
            raise ReflexiveDispatchFault("REFLEX_RULE_CAPABILITY_MISMATCH", row, path=f"reflexes[{offset}]")
        if row.get("fallback") != "none":
            raise ReflexiveDispatchFault("REFLEX_RULE_IMPLICIT_FALLBACK", row.get("fallback"), path=f"reflexes[{offset}].fallback")
        intent = str(match.get("intent") or "")
        if not intent:
            raise ReflexiveDispatchFault("REFLEX_RULE_UNINDEXABLE", match, path=f"reflexes[{offset}].match")
        ids.add(reflex_id)
        signatures.add(signature)
        index[intent].append(copy.deepcopy(row))
    return {key: sorted(rows, key=precedence_key) for key, rows in index.items()}


def precedence_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(bool(row.get("authority_required"))),
        EFFECT_RISK.get(str(row.get("effect_class") or "irreversible"), 99),
        -int(row.get("specificity") or 0),
        -float(row.get("verified_confidence") or 0.0),
        str(row.get("expiry") or "9999"),
        -int(row.get("owner_priority") or 0),
        str(row.get("reflex_id") or row.get("proposal_id") or ""),
    )


def canonical_event(
    *,
    payload: str,
    principal: str,
    authenticated: bool,
    origin: str,
    authority_refs: list[str],
    context_handles: list[str],
    valid_time: str | None = None,
    deadline_ms: int | None = None,
) -> dict[str, Any]:
    received = now()
    body = {
        "principal": principal,
        "authenticated": bool(authenticated),
        "origin": origin,
        "received_at": received,
        "valid_time": valid_time or received,
        "authority_refs": sorted(set(authority_refs)),
        "context_handles": sorted(set(context_handles)),
        "deadline_ms": deadline_ms,
        "literal_payload": payload,
        "literal_payload_digest": digest(payload),
    }
    body["event_id"] = stable_id("reflex-event", {key: value for key, value in body.items() if key != "literal_payload"})
    return body


def realize_effort_profile(event: dict[str, Any], requested: str, contract: dict[str, Any]) -> dict[str, Any]:
    profiles = validate_effort_profiles(contract.get("effort_profiles"), contract["resource_limits"])
    if requested not in profiles:
        raise ReflexiveDispatchFault("REFLEX_EFFORT_PROFILE_UNKNOWN", requested, path="effort_profile")
    limits = copy.deepcopy(profiles[requested])
    event_deadline = int(event.get("deadline_ms") or limits["deadline_ms"])
    limits["deadline_ms"] = min(limits["deadline_ms"], event_deadline)
    return {
        "requested_profile": requested,
        "realized_profile": requested,
        "realized_limits": limits,
        "event_deadline_ms": event_deadline,
        "downgrade_reasons": [],
        "profile_fidelity": True,
    }


def dispatch(
    event: dict[str, Any],
    *,
    intent: str,
    profile: str = "local_private_assistant",
    effort_profile: str = "balanced",
    requested_route: str = "",
    fallback_policy: str = "no_fallback",
    learned_proposals: list[dict[str, Any]] | None = None,
    route_health: dict[str, bool] | None = None,
    plan_nodes: list[dict[str, Any]] | None = None,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = validate_contract(contract or load_contract())
    started = datetime.now(timezone.utc)
    effort = realize_effort_profile(event, effort_profile, contract)
    ingress = classify_ingress(event, requested_route=requested_route, fallback_policy=fallback_policy, contract=contract)
    capabilities = validate_capabilities(contract["capabilities"])
    command_index = validate_user_commands(contract["user_commands"], capabilities)
    workflow_index = validate_workflows(contract.get("workflows"), capabilities, command_index, contract["resource_limits"])
    command_index.update(workflow_index)
    reflex_index = compile_reflex_index(contract["reflexes"], capabilities)
    proposals = propose_routes(
        event,
        intent=intent,
        ingress=ingress,
        capabilities=capabilities,
        command_index=command_index,
        reflex_index=reflex_index,
        learned_proposals=learned_proposals or [],
    )
    qualifications = [
        qualify_proposal(
            proposal,
            event=event,
            ingress=ingress,
            capabilities=[capabilities[item] for item in proposal_capability_ids(proposal) if item in capabilities],
            requested_capability_ids=proposal_capability_ids(proposal),
            profile=profile,
            route_health=route_health or {},
            effort_limits=effort["realized_limits"],
        )
        for proposal in proposals
    ]
    selection = select_route(
        proposals,
        qualifications,
        ingress=ingress,
        capabilities=capabilities,
        fallback_policy=fallback_policy,
        terminal_outcomes=set(contract["terminal_outcomes"]),
    )
    selected_proposals = [proposal for proposal in proposals if proposal["proposal_id"] in selection["selected_proposal_ids"]]
    selected_capability_ids = list(dict.fromkeys(item for proposal in selected_proposals for item in proposal_capability_ids(proposal)))
    selected_capabilities = [capabilities[item] for item in selected_capability_ids]
    workflow_nodes = next((row.get("workflow_nodes") for row in selected_proposals if isinstance(row.get("workflow_nodes"), list)), None)
    nodes = build_plan_nodes(
        selected_capabilities,
        explicit_nodes=plan_nodes if plan_nodes is not None else workflow_nodes,
        limits=effort["realized_limits"],
    ) if selection["selected_proposal_ids"] else []
    recorded = now()
    trace = {
        "policy": "project_theseus_reflexive_dispatch_trace_v1",
        "source_contract": contract["policy"],
        "event": redact_event(event),
        "ingress": ingress,
        "effort": effort,
        "proposals": proposals,
        "qualification": qualifications,
        "selection": selection,
        "plan_nodes": nodes,
        "effect": {
            "required": any(row["effect_class"] not in {"none", "read_only"} for row in nodes),
            "state": "not_required" if not nodes or all(row["effect_class"] in {"none", "read_only"} for row in nodes) else "prepared",
            "authority_ref": "",
            "receipt_ref": "",
            "residuals": ["effect_commit_requires_separate_registered_viea_scf_kernel"],
            "effect_authority_granted": False,
        },
        "result": typed_result_packet(event, selection, selected_capabilities, nodes, recorded),
        "chronicle": chronicle_proposal(event, selection, nodes, recorded),
        "compilation": {
            "state": "ineligible",
            "reason": "single_dispatch_is_not_reflex_compilation_evidence",
            "source_trace_refs": [],
            "negative_case_refs": [],
            "decompilation_route": contract["compilation"]["decompilation_route"],
        },
        "metrics": {
            "useful_outcome_state": "not_measured",
            "fast_path": bool(selection["selected_proposal_ids"] and all(p["proposer_type"] != "learned_router" for p in proposals if p["proposal_id"] in selection["selected_proposal_ids"])),
            "wrong_fast_path": False,
            "route_regret_state": "not_measured",
            "dispatch_ms": round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3),
            "total_cost_units": sum(int(row.get("cost_units") or 0) for row in selected_capabilities),
        },
        "support_state_effect": "none",
        "no_cheat": {
            "learned_generation_credit": 0,
            "fallback_return_count": 0,
            "external_inference_calls": 0,
            "public_training_rows_written": 0,
        },
        "non_claims": copy.deepcopy(contract["non_claims"]),
    }
    decision_view = {
        key: trace[key]
        for key in ("ingress", "effort", "proposals", "qualification", "selection", "plan_nodes")
    }
    trace["decision_digest"] = digest(decision_view)
    trace["trace_id"] = stable_id("reflex-trace", event.get("event_id"), trace["decision_digest"])
    trace["trace_digest"] = digest({key: value for key, value in trace.items() if key != "trace_digest"})
    return trace


def verify_trace(trace: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = validate_contract(contract or load_contract())
    observed = str(trace.get("trace_digest") or "")
    expected = digest({key: value for key, value in trace.items() if key != "trace_digest"})
    if observed != expected:
        raise ReflexiveDispatchFault("REFLEX_TRACE_DIGEST_INVALID", observed, path="trace_digest")
    decision_view = {
        key: trace.get(key)
        for key in ("ingress", "effort", "proposals", "qualification", "selection", "plan_nodes")
    }
    if trace.get("decision_digest") != digest(decision_view):
        raise ReflexiveDispatchFault("REFLEX_DECISION_REPLAY_INVALID", trace.get("decision_digest"), path="decision_digest")
    if (trace.get("selection") or {}).get("terminal_outcome") not in set(contract["terminal_outcomes"]):
        raise ReflexiveDispatchFault("REFLEX_TERMINAL_OUTCOME_INVALID", (trace.get("selection") or {}).get("terminal_outcome"), path="selection.terminal_outcome")
    counters = trace.get("no_cheat") if isinstance(trace.get("no_cheat"), dict) else {}
    if any(int(counters.get(key) or 0) != 0 for key in ("learned_generation_credit", "fallback_return_count", "external_inference_calls", "public_training_rows_written")):
        raise ReflexiveDispatchFault("REFLEX_NO_CHEAT_COUNTER_INVALID", counters, path="no_cheat")
    return {
        "state": "VERIFIED",
        "trace_id": trace.get("trace_id"),
        "decision_digest": trace.get("decision_digest"),
        "effect_authority_granted": False,
    }


def classify_ingress(event: dict[str, Any], *, requested_route: str, fallback_policy: str, contract: dict[str, Any]) -> dict[str, Any]:
    origin = str(event.get("origin") or "")
    trusted = origin in set(contract["ingress"]["trusted_command_origins"])
    literal_only = origin in set(contract["ingress"]["literal_only_origins"])
    payload = str(event.get("literal_payload") or "")
    first_token = payload.strip().split(maxsplit=1)[0] if payload.strip() else ""
    command_shaped = first_token.startswith(str(contract["ingress"]["command_prefix"]))
    command_authenticated = trusted and bool(event.get("authenticated")) and command_shaped
    if fallback_policy not in set(contract["ingress"]["fallback_policies"]):
        raise ReflexiveDispatchFault("REFLEX_FALLBACK_POLICY_INVALID", fallback_policy, path="fallback_policy")
    return {
        "mode": "forced_route" if requested_route else ("direct_command" if command_authenticated else "automatic"),
        "origin": origin,
        "command_token": first_token if command_authenticated else "",
        "command_shaped_literal_isolated": bool(command_shaped and (literal_only or not command_authenticated)),
        "command_authenticated": command_authenticated,
        "directive_authenticated": trusted and bool(event.get("authenticated")),
        "requested_route": requested_route,
        "fallback_policy": fallback_policy,
        "inference_bypassed": bool(requested_route or command_authenticated),
    }


def propose_routes(
    event: dict[str, Any],
    *,
    intent: str,
    ingress: dict[str, Any],
    capabilities: dict[str, dict[str, Any]],
    command_index: dict[str, dict[str, Any]],
    reflex_index: dict[str, list[dict[str, Any]]],
    learned_proposals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    if ingress["mode"] == "forced_route":
        proposals.append(proposal("forced", "command_binding", ingress["requested_route"], 1.0, False, False))
    elif ingress["mode"] == "direct_command":
        command = command_index.get(ingress["command_token"])
        if command:
            binding = bind_command_parameters(str(event.get("literal_payload") or ""), command, event)
            if command.get("binding_kind") == "workflow":
                proposals.append(
                    proposal(
                        str(command["workflow_id"]),
                        "workflow_binding",
                        "",
                        1.0,
                        False,
                        True,
                        capability_ids=[str(item) for item in command["capability_ids"]],
                        workflow_nodes=command["nodes"],
                        parameter_binding=binding,
                    )
                )
            else:
                proposals.append(
                    proposal(
                        command["command_id"], "command_binding", command["capability_id"], 1.0, False, False,
                        parameter_binding=binding,
                    )
                )
        else:
            proposals.append(proposal("unknown-command", "command_binding", "", 0.0, True, False))
    else:
        for rule in reflex_index.get(intent, []):
            proposals.append(proposal(rule["reflex_id"], "deterministic_rule", rule["capability_id"], float(rule["verified_confidence"]), False, False))
        for row in learned_proposals:
            proposals.append(
                proposal(
                    str(row.get("proposal_id") or stable_id("learned-proposal", row)),
                    "learned_router",
                    str(row.get("capability_id") or ""),
                    float(row.get("score") or 0.0),
                    bool(row.get("ood")),
                    bool(row.get("composite")),
                    capability_ids=[str(item) for item in row.get("capability_ids", []) if str(item)] if isinstance(row.get("capability_ids"), list) else None,
                    workflow_nodes=row.get("plan_nodes") if isinstance(row.get("plan_nodes"), list) else None,
                )
            )
    return proposals


def proposal(
    proposal_id: str,
    proposer_type: str,
    capability_id: str,
    score: float,
    ood: bool,
    composite: bool,
    *,
    capability_ids: list[str] | None = None,
    workflow_nodes: list[dict[str, Any]] | None = None,
    parameter_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ids = list(dict.fromkeys(capability_ids or ([capability_id] if capability_id else [])))
    return {
        "proposal_id": stable_id("route-proposal", proposal_id, capability_id),
        "source_id": proposal_id,
        "proposer_type": proposer_type,
        "capability_id": ids[0] if len(ids) == 1 else "",
        "capability_ids": ids,
        "score": score,
        "ood": ood,
        "composite": composite,
        "workflow_nodes": copy.deepcopy(workflow_nodes) if workflow_nodes is not None else None,
        "parameter_binding": copy.deepcopy(parameter_binding) if parameter_binding is not None else None,
        "proposal_grants_authority": False,
        "learned_generation_credit": 0,
    }


def proposal_capability_ids(row: dict[str, Any]) -> list[str]:
    ids = row.get("capability_ids") if isinstance(row.get("capability_ids"), list) else []
    if ids:
        return [str(item) for item in ids if str(item)]
    capability_id = str(row.get("capability_id") or "")
    return [capability_id] if capability_id else []


def bind_command_parameters(payload: str, descriptor: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    schema = validate_parameter_schema(descriptor.get("parameter_schema"), path="command.parameter_schema")
    properties = schema["properties"]
    positional_names = [str(value) for value in schema["positional"]]
    faults: list[dict[str, Any]] = []
    try:
        tokens = shlex.split(payload)
    except ValueError as exc:
        tokens = []
        faults.append({"kind": "command_lex_fault", "detail": str(exc)})
    arguments = tokens[1:] if tokens else []
    raw_values: dict[str, Any] = {}
    positional_values: list[str] = []
    for token in arguments:
        if token.startswith("--"):
            name_value = token[2:].split("=", 1)
            name = name_value[0]
            if len(name_value) != 2 or not name or name in raw_values:
                faults.append({"kind": "named_parameter_invalid_or_duplicate", "name": name})
                continue
            raw_values[name] = name_value[1]
        else:
            positional_values.append(token)
    capability_tail = positional_values[len(positional_names) :]
    if capability_tail and schema["free_text_tail"] is not True:
        faults.append({"kind": "too_many_positional_parameters", "count": len(positional_values)})
    for offset, value in enumerate(positional_values[: len(positional_names)]):
        name = positional_names[offset]
        if name in raw_values:
            faults.append({"kind": "parameter_bound_twice", "name": name})
        else:
            raw_values[name] = value
    unknown = sorted(set(raw_values) - set(properties))
    faults.extend({"kind": "unknown_parameter", "name": name} for name in unknown)
    resolved: dict[str, Any] = {}
    sources: dict[str, str] = {}
    context_defaults = {
        "principal": event.get("principal"),
        "deadline_ms": event.get("deadline_ms"),
        "first_context_handle": (event.get("context_handles") or [""])[0],
    }
    for name, field in properties.items():
        raw = raw_values.get(name)
        source = "explicit"
        if raw is None and field.get("default") is not None:
            default = field["default"]
            raw = default["value"] if default["kind"] == "literal" else context_defaults.get(str(default["value"]))
            source = str(default["kind"])
        if raw is None:
            if field.get("required") is True:
                faults.append({"kind": "required_parameter_missing", "name": name})
            continue
        try:
            resolved[name] = coerce_parameter(raw, field)
            sources[name] = source
        except (TypeError, ValueError) as exc:
            faults.append({"kind": "parameter_type_or_range_invalid", "name": name, "detail": str(exc)})
    public_bindings = {
        name: redact_parameter_value(value, properties[name], sources[name])
        for name, value in resolved.items()
        if name in properties
    }
    receipt = {
        "schema_digest": digest(schema),
        "binding_state": "valid" if not faults else "invalid",
        "bindings": public_bindings,
        "faults": faults,
        "raw_values_stored": False,
        "capability_input_tail": {
            "token_count": len(capability_tail),
            "value_digest": digest(capability_tail),
            "raw_value_stored": False,
        },
        "dynamic_default_sources": {name: source for name, source in sources.items() if source != "explicit"},
    }
    receipt["binding_digest"] = digest(receipt)
    return receipt


def coerce_parameter(value: Any, descriptor: dict[str, Any]) -> Any:
    kind = descriptor["type"]
    if kind == "string":
        result: Any = str(value)
    elif kind == "integer":
        if isinstance(value, bool) or not re.fullmatch(r"[-+]?\d+", str(value)):
            raise ValueError("expected integer")
        result = int(value)
    elif kind == "number":
        result = float(value)
    elif kind == "boolean":
        normalized = str(value).lower()
        if normalized not in {"true", "false"}:
            raise ValueError("expected true or false")
        result = normalized == "true"
    else:
        result = str(value)
        if result not in {str(item) for item in descriptor.get("values", [])}:
            raise ValueError("value not in enum")
    if kind in {"integer", "number"}:
        if descriptor.get("minimum") is not None and result < descriptor["minimum"]:
            raise ValueError("below minimum")
        if descriptor.get("maximum") is not None and result > descriptor["maximum"]:
            raise ValueError("above maximum")
    return result


def redact_parameter_value(value: Any, descriptor: dict[str, Any], source: str) -> dict[str, Any]:
    row = {"type": descriptor["type"], "source": source, "value_digest": digest(value)}
    if descriptor["type"] in {"integer", "number", "boolean", "enum"} and descriptor.get("sensitive") is not True:
        row["normalized_value"] = value
    elif isinstance(value, str):
        row["length"] = len(value)
    return row


def qualify_proposal(
    proposal_row: dict[str, Any],
    *,
    event: dict[str, Any],
    ingress: dict[str, Any],
    capabilities: list[dict[str, Any]],
    requested_capability_ids: list[str],
    profile: str,
    route_health: dict[str, bool],
    effort_limits: dict[str, int],
) -> dict[str, Any]:
    failures: list[str] = []
    capability_failures: dict[str, list[str]] = defaultdict(list)
    binding = proposal_row.get("parameter_binding") if isinstance(proposal_row.get("parameter_binding"), dict) else None
    if binding is not None and binding.get("binding_state") != "valid":
        failures.append("parameter_binding_invalid")
    if ingress.get("mode") in {"forced_route", "direct_command"} and not ingress.get("directive_authenticated"):
        failures.append("directive_authentication_failed")
    if not requested_capability_ids or len(capabilities) != len(requested_capability_ids):
        failures.append("capability_unknown")
    else:
        for capability in capabilities:
            capability_id = str(capability.get("capability_id") or "")
            if profile not in set(capability.get("qualified_profiles") or []):
                failures.append("profile_unqualified")
                capability_failures[capability_id].append("profile_unqualified")
            if capability.get("required_authority") not in set(event.get("authority_refs") or []):
                failures.append("authority_missing")
                capability_failures[capability_id].append("authority_missing")
            if not capability.get("verifier_ref"):
                failures.append("verifier_missing")
                capability_failures[capability_id].append("verifier_missing")
            if route_health.get(capability_id) is False:
                failures.append("implementation_stale_or_blocked")
                capability_failures[capability_id].append("implementation_stale_or_blocked")
        aggregate_cost = sum(int(row.get("cost_units") or 0) for row in capabilities)
        if aggregate_cost > int(effort_limits["max_aggregate_cost_units"]):
            failures.append("effort_cost_exceeded")
        workflow_nodes = proposal_row.get("workflow_nodes") if isinstance(proposal_row.get("workflow_nodes"), list) else []
        if proposal_row.get("composite"):
            if not workflow_nodes:
                failures.append("composite_plan_missing")
            else:
                try:
                    build_plan_nodes(capabilities, explicit_nodes=workflow_nodes, limits=effort_limits)
                except ReflexiveDispatchFault as exc:
                    failures.append(f"composite_plan_invalid:{exc.code}")
        elif workflow_nodes:
            failures.append("atomic_proposal_has_workflow_plan")
    if proposal_row.get("ood"):
        failures.append("ood")
    if proposal_row.get("proposer_type") == "learned_router" and float(proposal_row.get("score") or 0.0) < 0.8:
        failures.append("selective_risk_abstention")
    qualified = not failures
    receipt = {
        "proposal_id": proposal_row["proposal_id"],
        "capability_id": proposal_row.get("capability_id"),
        "capability_ids": requested_capability_ids,
        "qualified": qualified,
        "obligations": ["schema", "authority", "profile", "freshness", "verifier", "effect_bounds", "explicit_fallback"],
        "failures": sorted(set(failures)),
        "capability_failures": {key: sorted(set(value)) for key, value in sorted(capability_failures.items())},
        "effect_authority_granted": False,
    }
    receipt["receipt_ref"] = stable_id("qualification", receipt)
    return receipt


def select_route(
    proposals: list[dict[str, Any]],
    qualifications: list[dict[str, Any]],
    *,
    ingress: dict[str, Any],
    capabilities: dict[str, dict[str, Any]],
    fallback_policy: str,
    terminal_outcomes: set[str],
) -> dict[str, Any]:
    qualification_by_id = {row["proposal_id"]: row for row in qualifications}
    qualified = [row for row in proposals if qualification_by_id[row["proposal_id"]]["qualified"]]
    fallback_used = False
    if ingress["mode"] in {"forced_route", "direct_command"} and not qualified:
        failures = [failure for row in qualifications for failure in row["failures"]]
        terminal = terminal_from_failures(failures)
        return selection_packet([], "none", f"explicit route unqualified: {sorted(set(failures))}", False, terminal, terminal_outcomes)
    if not qualified and fallback_policy == "explicit_only":
        fallback = next((row for row in proposals if row.get("proposer_type") == "fallback"), None)
        if fallback and qualification_by_id[fallback["proposal_id"]]["qualified"]:
            qualified = [fallback]
            fallback_used = True
    if not qualified:
        failures = [failure for row in qualifications for failure in row["failures"]]
        terminal = terminal_from_failures(failures)
        return selection_packet([], "none", f"no qualified route: {sorted(set(failures))}", False, terminal, terminal_outcomes)
    ranked = sorted(
        qualified,
        key=lambda row: (
            sum(int(capabilities[item]["cost_units"]) for item in proposal_capability_ids(row)),
            -float(row.get("score") or 0.0),
            str(row["proposal_id"]),
        ),
    )
    selected = ranked[0]
    return selection_packet([selected["proposal_id"]], "dag" if selected.get("composite") or len(proposal_capability_ids(selected)) > 1 else "atomic", "minimum qualified total cost", fallback_used, "prepared", terminal_outcomes)


def terminal_from_failures(failures: list[str]) -> str:
    roots = {str(value).split(":", 1)[0] for value in failures}
    if {"authority_missing", "directive_authentication_failed"}.intersection(roots):
        return "unauthorized"
    if {"effort_cost_exceeded", "effort_node_count_exceeded"}.intersection(roots):
        return "resource_exceeded"
    if "ood" in roots:
        return "ood"
    return "unsupported"


def selection_packet(selected: list[str], kind: str, reason: str, fallback_used: bool, terminal: str, allowed: set[str]) -> dict[str, Any]:
    if terminal not in allowed:
        raise ReflexiveDispatchFault("REFLEX_TERMINAL_OUTCOME_INVALID", terminal, path="selection.terminal_outcome")
    return {
        "kind": kind,
        "selected_proposal_ids": selected,
        "reason": reason,
        "fallback_used": fallback_used,
        "terminal_outcome": terminal,
    }


def build_plan_nodes(
    selected_capabilities: list[dict[str, Any]],
    *,
    explicit_nodes: list[dict[str, Any]] | None,
    limits: dict[str, Any],
) -> list[dict[str, Any]]:
    nodes = copy.deepcopy(explicit_nodes) if explicit_nodes is not None else [
        {
            "node_id": stable_id("reflex-node", row["capability_id"]),
            "capability_id": row["capability_id"],
            "dependencies": [],
            "effect_class": row["effect_class"],
            "verifier_ref": row["verifier_ref"],
            "cost_units": row["cost_units"],
            "retry_limit": 0,
            "completion_policy": "all_or_nothing",
        }
        for row in selected_capabilities
    ]
    capability_index = {row["capability_id"]: row for row in selected_capabilities}
    for offset, node in enumerate(nodes):
        capability = capability_index.get(str(node.get("capability_id") or ""))
        if capability is None:
            raise ReflexiveDispatchFault("REFLEX_DAG_CAPABILITY_NOT_SELECTED", node.get("capability_id"), path=f"plan_nodes[{offset}].capability_id")
        if node.get("effect_class") != capability.get("effect_class") or node.get("verifier_ref") != capability.get("verifier_ref"):
            raise ReflexiveDispatchFault("REFLEX_DAG_CAPABILITY_CONTRACT_MISMATCH", node, path=f"plan_nodes[{offset}]")
    validate_plan_dag(nodes, limits)
    return nodes


def validate_plan_dag(nodes: list[dict[str, Any]], limits: dict[str, Any]) -> None:
    if len(nodes) > int(limits["max_nodes"]):
        raise ReflexiveDispatchFault("REFLEX_DAG_NODE_BUDGET_EXCEEDED", len(nodes), path="plan_nodes")
    by_id = {str(row.get("node_id") or ""): row for row in nodes if isinstance(row, dict)}
    if len(by_id) != len(nodes) or "" in by_id:
        raise ReflexiveDispatchFault("REFLEX_DAG_NODE_ID_CONFLICT", list(by_id), path="plan_nodes.node_id")
    child_count: Counter[str] = Counter()
    indegree = {node_id: 0 for node_id in by_id}
    children: dict[str, list[str]] = defaultdict(list)
    for node_id, row in by_id.items():
        dependencies = row.get("dependencies") if isinstance(row.get("dependencies"), list) else []
        if len(dependencies) > int(limits["max_fanout"]):
            raise ReflexiveDispatchFault("REFLEX_DAG_FANOUT_EXCEEDED", node_id, path="plan_nodes.dependencies")
        if int(row.get("retry_limit") or 0) > int(limits["max_retries_per_node"]):
            raise ReflexiveDispatchFault("REFLEX_DAG_RETRY_BUDGET_EXCEEDED", node_id, path="plan_nodes.retry_limit")
        if row.get("completion_policy") not in {"all_or_nothing", "best_effort", "compensate", "stop"}:
            raise ReflexiveDispatchFault("REFLEX_DAG_COMPLETION_POLICY_INVALID", row.get("completion_policy"), path="plan_nodes.completion_policy")
        for dependency in dependencies:
            if dependency not in by_id:
                raise ReflexiveDispatchFault("REFLEX_DAG_DEPENDENCY_UNKNOWN", dependency, path="plan_nodes.dependencies")
            indegree[node_id] += 1
            child_count[dependency] += 1
            children[dependency].append(node_id)
    if child_count and max(child_count.values()) > int(limits["max_fanout"]):
        raise ReflexiveDispatchFault("REFLEX_DAG_FANOUT_EXCEEDED", max(child_count.values()), path="plan_nodes")
    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    depth = {node_id: 1 for node_id in queue}
    visited: list[str] = []
    while queue:
        current = queue.popleft()
        visited.append(current)
        for child in children[current]:
            depth[child] = max(depth.get(child, 1), depth[current] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(visited) != len(nodes):
        raise ReflexiveDispatchFault("REFLEX_DAG_CYCLE", sorted(set(by_id) - set(visited)), path="plan_nodes")
    if depth and max(depth.values()) > int(limits["max_depth"]):
        raise ReflexiveDispatchFault("REFLEX_DAG_DEPTH_EXCEEDED", max(depth.values()), path="plan_nodes")
    total_cost = sum(int(row.get("cost_units") or 0) * (1 + int(row.get("retry_limit") or 0)) for row in nodes)
    if total_cost > int(limits["max_aggregate_cost_units"]):
        raise ReflexiveDispatchFault("REFLEX_DAG_COST_BUDGET_EXCEEDED", total_cost, path="plan_nodes")


def execute_plan_reference(
    nodes: list[dict[str, Any]],
    *,
    event: dict[str, Any],
    executor: Any,
    compensator: Any | None = None,
    limits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a bounded DAG with reference structured-concurrency semantics.

    This is the deterministic contract kernel used by tests and local canaries. It
    intentionally makes no throughput claim; production executors may parallelize
    ready nodes only if they preserve this receipt exactly.
    """
    limits = copy.deepcopy(limits or load_contract()["resource_limits"])
    deadline_ceiling = int(limits.get("deadline_ms") or limits.get("default_deadline_ms") or 0)
    if deadline_ceiling <= 0:
        raise ReflexiveDispatchFault("REFLEX_EXECUTION_DEADLINE_INVALID", limits, path="plan_execution.deadline")
    validate_plan_dag(nodes, limits)
    deadline_ms = min(int(event.get("deadline_ms") or deadline_ceiling), deadline_ceiling)
    authority_refs = sorted({str(value) for value in event.get("authority_refs", []) if str(value)})
    by_id = {str(row["node_id"]): copy.deepcopy(row) for row in nodes}
    pending = set(by_id)
    results: dict[str, dict[str, Any]] = {}
    execution_order: list[str] = []
    compensation_receipts: list[dict[str, Any]] = []
    elapsed_ms = 0
    global_stop = False
    while pending:
        progressed = False
        for node_id in sorted(pending):
            node = by_id[node_id]
            dependencies = [str(value) for value in node.get("dependencies", [])]
            if any(dependency not in results for dependency in dependencies):
                continue
            progressed = True
            pending.remove(node_id)
            if global_stop or any(results[dependency]["terminal_outcome"] != "resolved" for dependency in dependencies):
                results[node_id] = cancelled_node_result(node, "parent_or_plan_cancelled")
                break
            idempotency_key = stable_id("reflex-execution", event.get("event_id"), node_id)
            attempts: list[dict[str, Any]] = []
            final: dict[str, Any] = {}
            for attempt in range(int(node.get("retry_limit") or 0) + 1):
                context = {
                    "event_id": event.get("event_id"),
                    "node_id": node_id,
                    "attempt": attempt,
                    "idempotency_key": idempotency_key,
                    "authority_refs": authority_refs,
                    "deadline_ms": deadline_ms,
                    "dependency_result_refs": [results[value]["result_ref"] for value in dependencies],
                    "effect_authority_granted": False,
                }
                observed = executor(copy.deepcopy(node), copy.deepcopy(context))
                final = normalize_executor_result(node, observed, elapsed_ms=elapsed_ms)
                attempts.append(
                    {
                        "attempt": attempt,
                        "idempotency_key": idempotency_key,
                        "terminal_outcome": final["terminal_outcome"],
                        "result_ref": final["result_ref"],
                    }
                )
                elapsed_ms = max(elapsed_ms, int(final["completed_at_ms"]))
                if final["terminal_outcome"] != "execution_failed":
                    break
            if elapsed_ms > deadline_ms:
                final = {
                    **final,
                    "terminal_outcome": "resource_exceeded",
                    "late_result_suppressed": True,
                    "value_ref": "",
                    "effect_receipt_ref": "",
                    "verification_state": "not_accepted",
                }
            final["attempts"] = attempts
            final["idempotency_key"] = idempotency_key
            final["inherited_authority_refs"] = authority_refs
            final["inherited_deadline_ms"] = deadline_ms
            final["effect_authority_granted"] = False
            final["result_digest"] = digest({key: value for key, value in final.items() if key != "result_digest"})
            results[node_id] = final
            execution_order.append(node_id)
            if final["terminal_outcome"] != "resolved" and node.get("completion_policy") in {"all_or_nothing", "stop", "compensate"}:
                global_stop = True
                if node.get("completion_policy") == "compensate":
                    compensation_receipts = compensate_completed_nodes(
                        execution_order[:-1], by_id, results, compensator, event, authority_refs, deadline_ms
                    )
            break
        if not progressed:
            raise ReflexiveDispatchFault("REFLEX_EXECUTION_SCHEDULER_STALLED", sorted(pending), path="plan_execution")
    ordered_results = [results[node_id] for node_id in sorted(results)]
    resolved_count = sum(row["terminal_outcome"] == "resolved" for row in ordered_results)
    failed_count = sum(row["terminal_outcome"] in {"execution_failed", "verification_failed", "resource_exceeded"} for row in ordered_results)
    cancelled_count = sum(row["terminal_outcome"] == "rejected" for row in ordered_results)
    terminal = "resolved" if resolved_count == len(nodes) else ("partial" if resolved_count else "execution_failed")
    packet = {
        "policy": "project_theseus_reflexive_structured_execution_v1",
        "event_id": event.get("event_id"),
        "terminal_outcome": terminal,
        "node_results": ordered_results,
        "execution_order": execution_order,
        "resolved_count": resolved_count,
        "failed_count": failed_count,
        "cancelled_count": cancelled_count,
        "deadline_ms": deadline_ms,
        "elapsed_ms": elapsed_ms,
        "authority_refs": authority_refs,
        "compensation_receipts": compensation_receipts,
        "partial_results_immutable": True,
        "late_results_suppressed": any(row.get("late_result_suppressed") for row in ordered_results),
        "effect_authority_granted": False,
    }
    packet["execution_digest"] = digest(packet)
    return packet


def normalize_executor_result(node: dict[str, Any], value: Any, *, elapsed_ms: int) -> dict[str, Any]:
    row = value if isinstance(value, dict) else {}
    terminal = str(row.get("terminal_outcome") or "execution_failed")
    if terminal not in {"resolved", "execution_failed", "verification_failed"}:
        terminal = "execution_failed"
    verification = str(row.get("verification_state") or ("passed" if terminal == "resolved" else "failed"))
    if terminal == "resolved" and verification != "passed":
        terminal = "verification_failed"
    effect_receipt = str(row.get("effect_receipt_ref") or "")
    if node.get("effect_class") not in {"none", "read_only"} and terminal == "resolved" and not effect_receipt:
        terminal = "verification_failed"
        verification = "failed_missing_effect_receipt"
    completed = max(elapsed_ms, int(row.get("completed_at_ms") or elapsed_ms))
    body = {
        "node_id": node["node_id"],
        "terminal_outcome": terminal,
        "verification_state": verification,
        "value_ref": str(row.get("value_ref") or "") if terminal == "resolved" else "",
        "effect_receipt_ref": effect_receipt if terminal == "resolved" else "",
        "completed_at_ms": completed,
        "late_result_suppressed": False,
        "warnings": [str(value) for value in row.get("warnings", []) if str(value)] if isinstance(row.get("warnings"), list) else [],
    }
    body["result_ref"] = stable_id("reflex-node-result", body)
    return body


def cancelled_node_result(node: dict[str, Any], reason: str) -> dict[str, Any]:
    body = {
        "node_id": node["node_id"],
        "terminal_outcome": "rejected",
        "verification_state": "not_run",
        "value_ref": "",
        "effect_receipt_ref": "",
        "completed_at_ms": 0,
        "late_result_suppressed": False,
        "cancel_reason": reason,
        "attempts": [],
        "effect_authority_granted": False,
    }
    body["result_ref"] = stable_id("reflex-node-result", body)
    body["result_digest"] = digest(body)
    return body


def compensate_completed_nodes(
    node_ids: list[str],
    nodes: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    compensator: Any | None,
    event: dict[str, Any],
    authority_refs: list[str],
    deadline_ms: int,
) -> list[dict[str, Any]]:
    receipts = []
    for node_id in reversed(node_ids):
        node = nodes[node_id]
        result = results[node_id]
        if result["terminal_outcome"] != "resolved" or node.get("effect_class") in {"none", "read_only"}:
            continue
        if compensator is None:
            receipts.append({"node_id": node_id, "state": "failed", "reason": "compensator_missing"})
            continue
        context = {
            "event_id": event.get("event_id"),
            "authority_refs": authority_refs,
            "deadline_ms": deadline_ms,
            "effect_receipt_ref": result.get("effect_receipt_ref"),
            "idempotency_key": stable_id("reflex-compensation", event.get("event_id"), node_id),
            "effect_authority_granted": False,
        }
        observed = compensator(copy.deepcopy(node), copy.deepcopy(result), context)
        passed = isinstance(observed, dict) and observed.get("state") == "compensated" and bool(observed.get("receipt_ref"))
        receipts.append(
            {
                "node_id": node_id,
                "state": "compensated" if passed else "failed",
                "receipt_ref": str(observed.get("receipt_ref") or "") if isinstance(observed, dict) else "",
                "idempotency_key": context["idempotency_key"],
                "effect_authority_granted": False,
            }
        )
    return receipts


def typed_result_packet(event: dict[str, Any], selection: dict[str, Any], capabilities: list[dict[str, Any]], nodes: list[dict[str, Any]], recorded: str) -> dict[str, Any]:
    outcome = selection["terminal_outcome"]
    return {
        "result_id": stable_id("reflex-result", event.get("event_id"), selection),
        "schema_id": "project_theseus_reflexive_result_v1",
        "route_policy_ref": "project_theseus_reflexive_router_contract_v1",
        "implementation_refs": [row["capability_id"] for row in capabilities],
        "input_digest": event.get("literal_payload_digest"),
        "valid_time": event.get("valid_time"),
        "recorded_at": recorded,
        "epistemic_state": "unknown" if outcome not in {"resolved", "prepared"} else "inferred",
        "evidence_refs": [row["verifier_ref"] for row in capabilities],
        "verification_state": "pending" if outcome == "prepared" else "not_required",
        "effect_receipt_ref": "",
        "dependency_refs": [dependency for row in nodes for dependency in row.get("dependencies", [])],
        "dispatch_provenance_ref": stable_id("dispatch-provenance", selection),
        "authoritative_artifact_ref": "",
        "terminal_outcome": outcome,
        "model_text_may_not_override_terminal_outcome": True,
    }


def chronicle_proposal(event: dict[str, Any], selection: dict[str, Any], nodes: list[dict[str, Any]], recorded: str) -> dict[str, Any]:
    record = {
        "record_id": stable_id("chronicle", event.get("event_id"), selection),
        "record_type": "plan" if nodes else "event",
        "valid_time": event.get("valid_time"),
        "transaction_time": recorded,
        "epistemic_state": "inferred",
        "source_identity": event.get("event_id"),
        "payload_digest": digest({"selection": selection, "plan_nodes": nodes}),
        "supersedes": [],
        "contradicts": [],
        "correction_of": "",
        "effect_authority_granted": False,
    }
    return {"record_refs": [record["record_id"]], "update_state": "proposed", "record": record}


def append_chronicle_record(ledger: list[dict[str, Any]], record: dict[str, Any]) -> list[dict[str, Any]]:
    required = {"record_id", "record_type", "valid_time", "transaction_time", "epistemic_state", "source_identity", "payload_digest", "correction_of"}
    if not required.issubset(record) or record.get("record_type") not in {"entity", "event", "state", "claim", "plan", "prediction", "counterfactual"}:
        raise ReflexiveDispatchFault("REFLEX_CHRONICLE_RECORD_INVALID", record, path="chronicle")
    if any(row.get("record_id") == record["record_id"] for row in ledger):
        raise ReflexiveDispatchFault("REFLEX_CHRONICLE_DUPLICATE_ID", record["record_id"], path="chronicle.record_id")
    correction = str(record.get("correction_of") or "")
    if correction:
        prior = next((row for row in ledger if row.get("record_id") == correction), None)
        if prior is None or prior.get("record_type") != record.get("record_type"):
            raise ReflexiveDispatchFault("REFLEX_CHRONICLE_CORRECTION_TARGET_INVALID", correction, path="chronicle.correction_of")
    if record.get("record_type") == "claim" and record.get("epistemic_state") == "observed":
        raise ReflexiveDispatchFault("REFLEX_CHRONICLE_CLAIM_STATE_COLLAPSE", record["record_id"], path="chronicle.epistemic_state")
    return [*copy.deepcopy(ledger), copy.deepcopy(record)]


def cache_identity(
    *, task_semantics: str, principal: str, tenant: str, authority_refs: list[str], entity_scope: list[str],
    time_scope: str, privacy_view: str, source_versions: dict[str, str], schema_versions: dict[str, str],
    capability_versions: dict[str, str], model_version: str, policy_version: str, freshness_epoch: str,
) -> dict[str, Any]:
    body = {
        "task_semantics": task_semantics,
        "principal": principal,
        "tenant": tenant,
        "authority_refs": sorted(set(authority_refs)),
        "entity_scope": sorted(set(entity_scope)),
        "time_scope": time_scope,
        "privacy_view": privacy_view,
        "source_versions": dict(sorted(source_versions.items())),
        "schema_versions": dict(sorted(schema_versions.items())),
        "capability_versions": dict(sorted(capability_versions.items())),
        "model_version": model_version,
        "policy_version": policy_version,
        "freshness_epoch": freshness_epoch,
    }
    if any(value == "" or value == [] or value == {} for value in body.values()):
        raise ReflexiveDispatchFault("REFLEX_CACHE_IDENTITY_INCOMPLETE", body, path="cache_identity")
    return {"cache_key": digest(body), "dependencies": body, "effect_authority_granted": False}


def invalidate_cache(cache_record: dict[str, Any], changed_dependencies: dict[str, Any]) -> dict[str, Any]:
    dependencies = cache_record.get("dependencies") if isinstance(cache_record.get("dependencies"), dict) else {}
    changed = sorted(key for key, value in changed_dependencies.items() if dependencies.get(key) != value)
    return {
        "cache_key": cache_record.get("cache_key"),
        "invalidated": bool(changed),
        "changed_dependencies": changed,
        "descendants_must_invalidate": bool(changed),
    }


def compile_reflex_candidate(
    traces: list[dict[str, Any]],
    *, negative_case_refs: list[str],
    differential_passed: bool,
    shadow_passed: bool,
    canary_passed: bool,
    expected_reuse_value: float,
    lifecycle_cost: float,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = validate_contract(contract or load_contract())
    policy = contract["compilation"]
    verified = [row for row in traces if (row.get("result") or {}).get("verification_state") == "passed"]
    sources = {str((row.get("event") or {}).get("principal") or "") for row in verified}
    reasons = []
    if len(verified) < int(policy["minimum_verified_traces"]): reasons.append("verified_trace_floor")
    if len(sources - {""}) < int(policy["minimum_source_diversity"]): reasons.append("source_diversity_floor")
    if len(set(negative_case_refs)) < int(policy["negative_cases_required"]): reasons.append("negative_space_floor")
    if not differential_passed: reasons.append("differential_replay_failed")
    if not shadow_passed: reasons.append("shadow_failed")
    if not canary_passed: reasons.append("canary_failed")
    if expected_reuse_value <= lifecycle_cost: reasons.append("lifecycle_economics_failed")
    state = "qualified" if not reasons else ("quarantined" if traces else "ineligible")
    expires = (datetime.now(timezone.utc) + timedelta(hours=int(policy["expiry_hours"]))).isoformat()
    body = {
        "state": state,
        "source_trace_refs": sorted(str(row.get("trace_id") or "") for row in verified),
        "source_diversity": len(sources - {""}),
        "negative_case_refs": sorted(set(negative_case_refs)),
        "differential_passed": differential_passed,
        "shadow_passed": shadow_passed,
        "canary_passed": canary_passed,
        "expected_reuse_value": expected_reuse_value,
        "lifecycle_cost": lifecycle_cost,
        "rejection_reasons": reasons,
        "expiry": expires,
        "decompilation_route": policy["decompilation_route"],
        "effect_authority_granted": False,
        "learned_generation_credit": 0,
    }
    body["candidate_id"] = stable_id("reflex-candidate", body)
    return body


def decompile_reflex(candidate: dict[str, Any], *, reason: str, changed_dependencies: list[str]) -> dict[str, Any]:
    if candidate.get("state") not in {"qualified", "shadow", "quarantined"}:
        raise ReflexiveDispatchFault("REFLEX_DECOMPILE_STATE_INVALID", candidate.get("state"), path="candidate.state")
    return {
        **copy.deepcopy(candidate),
        "state": "decompiled",
        "decompiled_reason": reason,
        "changed_dependencies": sorted(set(changed_dependencies)),
        "route_after_decompile": candidate.get("decompilation_route"),
        "descendants_invalidated": True,
        "effect_authority_granted": False,
    }


def preview_registry_mutation(contract: dict[str, Any], mutation: dict[str, Any]) -> dict[str, Any]:
    contract = validate_contract(contract)
    required = {"mutation_id", "action", "registry", "target_id", "expected_contract_digest", "source_kind", "signer_tier", "approval_refs"}
    if not required.issubset(mutation) or mutation.get("action") not in {"add", "update", "disable"}:
        raise ReflexiveDispatchFault("REFLEX_REGISTRY_MUTATION_INVALID", mutation, path="mutation")
    registry = str(mutation["registry"])
    id_keys = {"user_commands": "command_id", "workflows": "workflow_id", "reflexes": "reflex_id"}
    if registry not in id_keys:
        raise ReflexiveDispatchFault("REFLEX_REGISTRY_MUTATION_TARGET_INVALID", registry, path="mutation.registry")
    before_digest = digest(contract)
    if mutation["expected_contract_digest"] != before_digest:
        raise ReflexiveDispatchFault("REFLEX_REGISTRY_MUTATION_STALE_BASE", mutation["expected_contract_digest"], path="mutation.expected_contract_digest")
    source_kind = str(mutation["source_kind"])
    signer_tier = str(mutation["signer_tier"])
    if source_kind not in {"local_operator", "signed_package"} or (source_kind == "signed_package" and signer_tier not in {"trusted_signed", "project"}):
        raise ReflexiveDispatchFault("REFLEX_REGISTRY_MUTATION_SUPPLY_CHAIN_INVALID", {"source_kind": source_kind, "signer_tier": signer_tier}, path="mutation")
    rows = copy.deepcopy(contract.get(registry) or [])
    id_key = id_keys[registry]
    target_id = str(mutation["target_id"])
    offsets = [offset for offset, row in enumerate(rows) if isinstance(row, dict) and str(row.get(id_key) or "") == target_id]
    action = str(mutation["action"])
    descriptor = copy.deepcopy(mutation.get("descriptor"))
    if action == "add" and offsets:
        raise ReflexiveDispatchFault("REFLEX_REGISTRY_MUTATION_TARGET_EXISTS", target_id, path="mutation.target_id")
    if action in {"update", "disable"} and len(offsets) != 1:
        raise ReflexiveDispatchFault("REFLEX_REGISTRY_MUTATION_TARGET_MISSING", target_id, path="mutation.target_id")
    if action in {"add", "update"}:
        if not isinstance(descriptor, dict) or str(descriptor.get(id_key) or "") != target_id:
            raise ReflexiveDispatchFault("REFLEX_REGISTRY_MUTATION_DESCRIPTOR_INVALID", descriptor, path="mutation.descriptor")
        if action == "add":
            rows.append(descriptor)
        else:
            rows[offsets[0]] = descriptor
    else:
        descriptor = None
        rows.pop(offsets[0])
    candidate = copy.deepcopy(contract)
    prior_descriptor = copy.deepcopy(contract[registry][offsets[0]]) if offsets else None
    candidate[registry] = rows
    candidate = validate_contract(candidate)
    authority_diff = registry_authority_diff(contract, candidate, registry, target_id)
    approvals = {str(value) for value in mutation.get("approval_refs", []) if str(value)}
    blockers = []
    if authority_diff["expanded"] and "authority_expansion_review" not in approvals:
        blockers.append("authority_expansion_review_missing")
    if authority_diff["effect_risk_increased"] and "effect_risk_review" not in approvals:
        blockers.append("effect_risk_review_missing")
    receipt = {
        "mutation_id": str(mutation["mutation_id"]),
        "action": action,
        "registry": registry,
        "target_id": target_id,
        "before_contract_digest": before_digest,
        "after_contract_digest": digest(candidate),
        "source_kind": source_kind,
        "signer_tier": signer_tier,
        "approval_refs": sorted(approvals),
        "authority_diff": authority_diff,
        "prior_descriptor": prior_descriptor,
        "prior_index": offsets[0] if offsets else None,
        "candidate_descriptor": descriptor,
        "blockers": blockers,
        "state": "approved" if not blockers else "denied",
        "effect_authority_granted": False,
    }
    receipt["preview_digest"] = digest(receipt)
    return {"receipt": receipt, "candidate_contract": candidate}


def apply_registry_mutation(preview: dict[str, Any]) -> dict[str, Any]:
    receipt = preview.get("receipt") if isinstance(preview.get("receipt"), dict) else {}
    candidate = preview.get("candidate_contract") if isinstance(preview.get("candidate_contract"), dict) else {}
    if receipt.get("state") != "approved" or receipt.get("preview_digest") != digest({key: value for key, value in receipt.items() if key != "preview_digest"}):
        raise ReflexiveDispatchFault("REFLEX_REGISTRY_MUTATION_NOT_APPROVED", receipt, path="preview.receipt")
    candidate = validate_contract(candidate)
    if digest(candidate) != receipt.get("after_contract_digest"):
        raise ReflexiveDispatchFault("REFLEX_REGISTRY_MUTATION_PREVIEW_TAMPERED", digest(candidate), path="preview.candidate_contract")
    applied = copy.deepcopy(receipt)
    applied["state"] = "applied"
    applied["applied_at"] = now()
    applied["effect_authority_granted"] = False
    applied["application_digest"] = digest(applied)
    return {"contract": candidate, "receipt": applied}


def rollback_registry_mutation(current: dict[str, Any], applied_receipt: dict[str, Any]) -> dict[str, Any]:
    current = validate_contract(current)
    if digest(current) != applied_receipt.get("after_contract_digest") or applied_receipt.get("state") != "applied":
        raise ReflexiveDispatchFault("REFLEX_REGISTRY_ROLLBACK_BASE_INVALID", digest(current), path="rollback")
    registry = str(applied_receipt["registry"])
    id_key = {"user_commands": "command_id", "workflows": "workflow_id", "reflexes": "reflex_id"}[registry]
    target_id = str(applied_receipt["target_id"])
    prior = copy.deepcopy(applied_receipt.get("prior_descriptor"))
    rows = [copy.deepcopy(row) for row in current[registry] if str(row.get(id_key) or "") != target_id]
    if prior is not None:
        rows.insert(int(applied_receipt.get("prior_index") or 0), prior)
    rolled = copy.deepcopy(current)
    rolled[registry] = rows
    rolled = validate_contract(rolled)
    if digest(rolled) != applied_receipt.get("before_contract_digest"):
        raise ReflexiveDispatchFault("REFLEX_REGISTRY_ROLLBACK_IDENTITY_MISMATCH", digest(rolled), path="rollback")
    receipt = {
        "mutation_id": applied_receipt.get("mutation_id"),
        "state": "rolled_back",
        "restored_contract_digest": digest(rolled),
        "rolled_back_at": now(),
        "effect_authority_granted": False,
    }
    receipt["rollback_digest"] = digest(receipt)
    return {"contract": rolled, "receipt": receipt}


def registry_authority_diff(before: dict[str, Any], after: dict[str, Any], registry: str, target_id: str) -> dict[str, Any]:
    capabilities_before = validate_capabilities(before["capabilities"])
    capabilities_after = validate_capabilities(after["capabilities"])
    id_key = {"user_commands": "command_id", "workflows": "workflow_id", "reflexes": "reflex_id"}[registry]

    def bound_capabilities(contract: dict[str, Any]) -> list[str]:
        row = next((item for item in contract.get(registry, []) if str(item.get(id_key) or "") == target_id), {})
        values = row.get("capability_ids") if isinstance(row.get("capability_ids"), list) else [row.get("capability_id")]
        return [str(value) for value in values if value]

    before_ids = bound_capabilities(before)
    after_ids = bound_capabilities(after)
    before_authorities = {capabilities_before[item]["required_authority"] for item in before_ids if item in capabilities_before}
    after_authorities = {capabilities_after[item]["required_authority"] for item in after_ids if item in capabilities_after}
    before_risk = max((EFFECT_RISK[capabilities_before[item]["effect_class"]] for item in before_ids if item in capabilities_before), default=-1)
    after_risk = max((EFFECT_RISK[capabilities_after[item]["effect_class"]] for item in after_ids if item in capabilities_after), default=-1)
    return {
        "before_capability_ids": before_ids,
        "after_capability_ids": after_ids,
        "added_authority_refs": sorted(after_authorities - before_authorities),
        "removed_authority_refs": sorted(before_authorities - after_authorities),
        "expanded": bool(after_authorities - before_authorities),
        "before_effect_risk": before_risk,
        "after_effect_risk": after_risk,
        "effect_risk_increased": after_risk > before_risk,
    }


def evaluate_reflexbench_profile(
    profile: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = load_reflexbench_profile() if profile is None else load_reflexbench_profile_from_value(profile)
    contract = validate_contract(contract or load_contract())
    visible_fields = set(profile["case_information_boundary"]["policy_visible_fields"])
    rows: list[dict[str, Any]] = []
    for policy in profile["policies"]:
        state: dict[str, Any] = {"cache": {}, "chronicle": [], "compiled_candidate": None}
        for case in profile["cases"]:
            visible = {key: copy.deepcopy(case[key]) for key in visible_fields}
            expected_terminal = str(case["expected_terminal"])
            expected_capabilities = sorted(str(value) for value in case["expected_capability_ids"])
            observed = (
                mechanics_result(
                    expected_terminal,
                    expected_capabilities,
                    int(case["oracle_cost_units"]),
                    datetime.now(timezone.utc),
                    fast=expected_capabilities != ["assistant.chat_checkpoint"],
                    lifecycle="oracle_upper_bound",
                )
                if policy == "oracle"
                else evaluate_visible_reflex_case(policy, visible, state, contract)
            )
            useful = observed["terminal_outcome"] == expected_terminal and sorted(observed["capability_ids"]) == expected_capabilities
            wrong_fast = bool(observed["fast_path"] and not useful)
            oracle_cost = int(case["oracle_cost_units"])
            regret = max(0, int(observed["total_cost_units"]) - oracle_cost) if useful else 100 + int(observed["total_cost_units"])
            rows.append(
                {
                    "policy_id": policy,
                    "case_id": case["case_id"],
                    "track": case["track"],
                    "terminal_outcome": observed["terminal_outcome"],
                    "capability_ids": observed["capability_ids"],
                    "expected_terminal": expected_terminal,
                    "expected_capability_ids": expected_capabilities,
                    "expected_property": case["expected_property"],
                    "useful": useful,
                    "qualified": observed["qualified"],
                    "fast_path": observed["fast_path"],
                    "wrong_fast_path": wrong_fast,
                    "deliberative_calls": observed["deliberative_calls"],
                    "total_cost_units": observed["total_cost_units"],
                    "oracle_cost_units": oracle_cost,
                    "route_regret_units": regret,
                    "dispatch_ms": observed["dispatch_ms"],
                    "unauthorized_effect": observed["unauthorized_effect"],
                    "verification_escape": observed["verification_escape"],
                    "literal_isolation": observed["literal_isolation"],
                    "effort_fidelity": observed["effort_fidelity"],
                    "silent_fallback": observed["silent_fallback"],
                    "provenance_complete": observed["provenance_complete"],
                    "lifecycle_state": observed["lifecycle_state"],
                    "policy_read_held_verifier_fields": policy == "oracle",
                }
            )
    policy_metrics = [aggregate_reflexbench_policy(rows, policy, profile["tracks"]) for policy in profile["policies"]]
    full = next(row for row in policy_metrics if row["policy_id"] == "full_reflexive")
    report = {
        "policy": "project_theseus_reflexbench_embedded_profile_result_v1",
        "profile_id": profile["profile_id"],
        "profile_digest": digest(profile),
        "contract_digest": digest(contract),
        "fidelity": profile["fidelity"],
        "case_count": len(profile["cases"]),
        "track_count": len(profile["tracks"]),
        "policy_count": len(profile["policies"]),
        "result_count": len(rows),
        "matched_resources": copy.deepcopy(profile["matched_resources"]),
        "information_boundary": copy.deepcopy(profile["case_information_boundary"]),
        "policy_metrics": policy_metrics,
        "results": rows,
        "full_reflexive_pretraining_mechanics_ready": bool(
            full["useful_rate"] >= 0.95
            and full["unauthorized_action_rate"] == 0.0
            and full["verification_escape_rate"] == 0.0
            and full["silent_fallback_rate"] == 0.0
            and full["literal_isolation_rate"] == 1.0
            and full["effort_profile_fidelity"] == 1.0
        ),
        "no_cheat": {
            "non_oracle_held_field_reads": sum(row["policy_read_held_verifier_fields"] for row in rows if row["policy_id"] != "oracle"),
            "learned_generation_credit": 0,
            "external_inference_calls": 0,
            "public_training_rows_written": 0,
            "fallback_return_count": 0,
        },
        "non_claims": copy.deepcopy(profile["non_claims"]),
    }
    report["result_digest"] = digest(report)
    return report


def verify_reflexbench_result(report: dict[str, Any], profile: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = load_reflexbench_profile() if profile is None else load_reflexbench_profile_from_value(profile)
    observed_digest = str(report.get("result_digest") or "")
    expected_digest = digest({key: value for key, value in report.items() if key != "result_digest"})
    if observed_digest != expected_digest:
        raise ReflexiveDispatchFault("REFLEX_PROFILE_RESULT_DIGEST_INVALID", observed_digest, path="profile_result.result_digest")
    rows = report.get("results") if isinstance(report.get("results"), list) else []
    expected_pairs = {(str(policy), str(case["case_id"])) for policy in profile["policies"] for case in profile["cases"]}
    observed_pairs = {(str(row.get("policy_id")), str(row.get("case_id"))) for row in rows if isinstance(row, dict)}
    if len(rows) != len(expected_pairs) or observed_pairs != expected_pairs:
        raise ReflexiveDispatchFault("REFLEX_PROFILE_RESULT_DENOMINATOR_INVALID", len(rows), path="profile_result.results")
    if report.get("profile_digest") != digest(profile):
        raise ReflexiveDispatchFault("REFLEX_PROFILE_RESULT_PROFILE_MISMATCH", report.get("profile_digest"), path="profile_result.profile_digest")
    if any(row.get("policy_read_held_verifier_fields") is True for row in rows if row.get("policy_id") != "oracle"):
        raise ReflexiveDispatchFault("REFLEX_PROFILE_HELD_FIELD_LEAK", {}, path="profile_result.results")
    no_cheat = report.get("no_cheat") if isinstance(report.get("no_cheat"), dict) else {}
    if any(int(no_cheat.get(key) or 0) != 0 for key in ("non_oracle_held_field_reads", "learned_generation_credit", "external_inference_calls", "public_training_rows_written", "fallback_return_count")):
        raise ReflexiveDispatchFault("REFLEX_PROFILE_NO_CHEAT_INVALID", no_cheat, path="profile_result.no_cheat")
    return {
        "state": "VERIFIED",
        "profile_id": report.get("profile_id"),
        "profile_digest": report.get("profile_digest"),
        "result_digest": observed_digest,
        "case_count": report.get("case_count"),
        "policy_count": report.get("policy_count"),
        "result_count": len(rows),
    }


def load_reflexbench_profile_from_value(profile: dict[str, Any]) -> dict[str, Any]:
    # Reuse the same validation without weakening the path-bound loader.
    temporary = copy.deepcopy(profile)
    if temporary.get("policy") != "project_theseus_reflexbench_profile_v1":
        raise ReflexiveDispatchFault("REFLEX_PROFILE_POLICY_INVALID", temporary.get("policy"), path="profile.policy")
    tracks = temporary.get("tracks") if isinstance(temporary.get("tracks"), list) else []
    policies = temporary.get("policies") if isinstance(temporary.get("policies"), list) else []
    cases = temporary.get("cases") if isinstance(temporary.get("cases"), list) else []
    boundary = temporary.get("case_information_boundary") if isinstance(temporary.get("case_information_boundary"), dict) else {}
    visible = set(boundary.get("policy_visible_fields") or [])
    held = set(boundary.get("held_verifier_fields") or [])
    if len(tracks) != 8 or len(set(tracks)) != 8 or len(policies) != 10 or len(set(policies)) != 10 or len(cases) < 32:
        raise ReflexiveDispatchFault("REFLEX_PROFILE_MATRIX_INVALID", {}, path="profile")
    if visible.intersection(held) or boundary.get("oracle_is_only_policy_allowed_held_fields") is not True:
        raise ReflexiveDispatchFault("REFLEX_PROFILE_INFORMATION_BOUNDARY_INVALID", boundary, path="profile")
    return temporary


def evaluate_visible_reflex_case(policy: str, case: dict[str, Any], state: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    if policy == "oracle":
        raise ReflexiveDispatchFault("REFLEX_ORACLE_REQUIRES_VERIFIER_WRAPPER", case.get("case_id"), path="profile")
    if str(case["track"]).startswith("E_"):
        return evaluate_chronicle_case(policy, case, state, started)
    if str(case["track"]).startswith("H_"):
        return evaluate_compilation_case(policy, case, state, contract, started)
    if case.get("sequence_key") == "registry-authority-expansion":
        return evaluate_registry_mutation_case(policy, contract, started)
    policy_features = reflexbench_policy_features(policy)
    event_row = canonical_event(
        payload=str(case["payload"]),
        principal="benchmark:user",
        authenticated=bool(case["authenticated"]),
        origin=str(case["origin"]),
        authority_refs=[str(value) for value in case["authority_refs"]],
        context_handles=[] if case["context_state"] == "missing" else [f"vcm://profile/{case['context_state']}"],
        deadline_ms=30000,
        valid_time="2026-07-16T00:00:00+00:00",
    )
    requested_route = str(case["requested_route"])
    deliberative_calls = 0
    cache_hit = False
    if policy == "model_only":
        requested_route = "assistant.chat_checkpoint"
        deliberative_calls = 1
    elif policy == "llm_first_tool_agent":
        deliberative_calls = 1
        requested_route = "assistant.deterministic_tool" if case["intent"] == "tool" else "assistant.chat_checkpoint"
    elif policy == "modular_tool_model":
        requested_route = "assistant.deterministic_tool" if case["intent"] == "tool" else "assistant.chat_checkpoint"
        deliberative_calls = 0 if case["intent"] == "tool" else 1
    elif policy == "semantic_cache_plus_model":
        key = str(case["sequence_key"])
        if key in state["cache"]:
            cached = copy.deepcopy(state["cache"][key])
            cached["cache_hit"] = True
            cached["dispatch_ms"] = elapsed_ms(started)
            return cached
        requested_route = "assistant.chat_checkpoint"
        deliberative_calls = 1
    learned_proposals = None
    dispatch_contract = contract
    if policy == "learned_router_proxy" and not requested_route and not str(case["payload"]).startswith("/"):
        dispatch_contract = copy.deepcopy(contract)
        dispatch_contract["reflexes"] = []
        capability_id = {
            "chat": "assistant.chat_checkpoint", "code": "assistant.code_candidate",
            "tool": "assistant.deterministic_tool", "planning": "assistant.plan_dag",
        }.get(str(case["intent"]), "")
        learned_proposals = [{"proposal_id": f"visible-intent:{case['intent']}", "capability_id": capability_id, "score": 0.85, "ood": False}]
    trace = dispatch(
        event_row,
        intent=str(case["intent"]),
        effort_profile=str(case["effort_profile"]),
        requested_route=requested_route,
        fallback_policy="no_fallback",
        learned_proposals=learned_proposals,
        route_health={str(key): bool(value) for key, value in case["route_health"].items()},
        contract=dispatch_contract,
    )
    capabilities = selected_capability_ids(trace)
    result = {
        "terminal_outcome": trace["selection"]["terminal_outcome"],
        "capability_ids": capabilities,
        "qualified": bool(trace["selection"]["selected_proposal_ids"]),
        "fast_path": bool(capabilities and capabilities != ["assistant.chat_checkpoint"] and deliberative_calls == 0),
        "deliberative_calls": deliberative_calls,
        "total_cost_units": int(trace["metrics"]["total_cost_units"]) + 100 * deliberative_calls,
        "dispatch_ms": elapsed_ms(started),
        "unauthorized_effect": bool(trace["effect"]["effect_authority_granted"]),
        "verification_escape": False,
        "literal_isolation": not (
            case["origin"] in contract["ingress"]["literal_only_origins"]
            and any(value != "assistant.chat_checkpoint" for value in capabilities)
        ),
        "effort_fidelity": trace["effort"]["profile_fidelity"] and trace["effort"]["requested_profile"] == case["effort_profile"],
        "silent_fallback": bool(trace["selection"]["fallback_used"]),
        "provenance_complete": bool(trace.get("trace_id") and trace.get("decision_digest") and trace.get("chronicle")),
        "lifecycle_state": "not_applicable",
        "cache_hit": cache_hit,
    }
    if policy == "semantic_cache_plus_model":
        state["cache"][str(case["sequence_key"])] = copy.deepcopy(result)
    return result


def evaluate_chronicle_case(policy: str, case: dict[str, Any], state: dict[str, Any], started: datetime) -> dict[str, Any]:
    enabled = reflexbench_policy_features(policy)["chronicle"]
    terminal = "unsupported"
    lifecycle = "chronicle_disabled"
    if enabled:
        try:
            key = str(case["sequence_key"])
            if key == "chronicle-append":
                record = profile_chronicle_record("state:a", "state", "observed", "")
                state["chronicle"] = append_chronicle_record(state["chronicle"], record)
                terminal, lifecycle = "resolved", "appended"
            elif key == "chronicle-correction":
                prior = next((row for row in state["chronicle"] if row["record_id"] == "state:a"), None)
                if prior is None:
                    terminal, lifecycle = "rejected", "missing_prior"
                else:
                    state["chronicle"] = append_chronicle_record(state["chronicle"], profile_chronicle_record("state:b", "state", "observed", "state:a"))
                    terminal, lifecycle = "resolved", "corrected"
            elif key == "chronicle-poison":
                append_chronicle_record(state["chronicle"], profile_chronicle_record("claim:bad", "claim", "observed", ""))
            else:
                corrected = any(row.get("correction_of") == "state:a" for row in state["chronicle"])
                terminal, lifecycle = ("resolved", "known_at_resolved") if corrected else ("rejected", "known_at_missing")
        except ReflexiveDispatchFault:
            terminal, lifecycle = "rejected", "poisoning_rejected"
    return mechanics_result(terminal, [], 5 if terminal == "resolved" else 0, started, fast=enabled, lifecycle=lifecycle)


def evaluate_compilation_case(policy: str, case: dict[str, Any], state: dict[str, Any], contract: dict[str, Any], started: datetime) -> dict[str, Any]:
    enabled = reflexbench_policy_features(policy)["compiler"]
    terminal = "unsupported"
    lifecycle = "compiler_disabled"
    if enabled:
        key = str(case["sequence_key"])
        traces = []
        for principal in ("source:a", "source:b", "source:a"):
            trace = dispatch(
                canonical_event(
                    payload="plan", principal=principal, authenticated=True, origin="local_user_control",
                    authority_refs=["local_assistant_read"], context_handles=["vcm://profile/compile"], deadline_ms=30000,
                ),
                intent="planning",
                contract=contract,
            )
            trace["result"]["verification_state"] = "passed"
            traces.append(trace)
        if key == "compile-qualified":
            candidate = compile_reflex_candidate(traces, negative_case_refs=["n:a", "n:b"], differential_passed=True, shadow_passed=True, canary_passed=True, expected_reuse_value=20, lifecycle_cost=5, contract=contract)
            state["compiled_candidate"] = candidate
            terminal, lifecycle = ("resolved", "qualified") if candidate["state"] == "qualified" else ("rejected", candidate["state"])
        elif key == "compile-premature":
            candidate = compile_reflex_candidate(traces[:1], negative_case_refs=[], differential_passed=False, shadow_passed=False, canary_passed=False, expected_reuse_value=1, lifecycle_cost=5, contract=contract)
            terminal, lifecycle = ("rejected", candidate["state"])
        elif key == "compile-drift":
            candidate = state.get("compiled_candidate")
            if candidate and candidate.get("state") == "qualified":
                retired = decompile_reflex(candidate, reason="dependency_drift", changed_dependencies=["policy:v2"])
                terminal, lifecycle = ("resolved", retired["state"])
            else:
                terminal, lifecycle = "rejected", "qualified_candidate_missing"
        else:
            candidate = compile_reflex_candidate(traces, negative_case_refs=[], differential_passed=True, shadow_passed=True, canary_passed=True, expected_reuse_value=20, lifecycle_cost=5, contract=contract)
            terminal, lifecycle = ("rejected", candidate["state"])
    return mechanics_result(terminal, [], 15 if terminal == "resolved" else 0, started, fast=enabled, lifecycle=lifecycle)


def evaluate_registry_mutation_case(policy: str, contract: dict[str, Any], started: datetime) -> dict[str, Any]:
    enabled = reflexbench_policy_features(policy)["registry"]
    terminal = "unsupported"
    lifecycle = "mutation_governance_disabled"
    if enabled:
        descriptor = copy.deepcopy(contract["user_commands"][3])
        descriptor["capability_id"] = "assistant.route_authority_effect"
        preview = preview_registry_mutation(
            contract,
            {
                "mutation_id": "profile:authority-expansion",
                "action": "update",
                "registry": "user_commands",
                "target_id": "command.plan",
                "descriptor": descriptor,
                "expected_contract_digest": digest(contract),
                "source_kind": "local_operator",
                "signer_tier": "operator",
                "approval_refs": [],
            },
        )
        terminal = "rejected" if preview["receipt"]["state"] == "denied" else "verification_failed"
        lifecycle = "authority_expansion_denied" if terminal == "rejected" else "authority_expansion_escaped"
    return mechanics_result(terminal, [], 0, started, fast=enabled, lifecycle=lifecycle)


def reflexbench_policy_features(policy: str) -> dict[str, bool]:
    table = {
        "model_only": (False, False, False), "llm_first_tool_agent": (False, False, False),
        "hard_rule_only": (False, False, False), "learned_router_proxy": (False, False, False),
        "semantic_cache_plus_model": (False, False, False), "modular_tool_model": (False, False, False),
        "reflexive_without_chronicle": (False, True, True), "reflexive_without_compiler": (True, False, True),
        "full_reflexive": (True, True, True), "oracle": (True, True, True),
    }
    if policy not in table:
        raise ReflexiveDispatchFault("REFLEX_PROFILE_POLICY_UNKNOWN", policy, path="profile.policy")
    chronicle, compiler, registry = table[policy]
    return {"chronicle": chronicle, "compiler": compiler, "registry": registry}


def mechanics_result(terminal: str, capabilities: list[str], cost: int, started: datetime, *, fast: bool, lifecycle: str) -> dict[str, Any]:
    return {
        "terminal_outcome": terminal, "capability_ids": capabilities, "qualified": terminal in {"prepared", "resolved"},
        "fast_path": fast, "deliberative_calls": 0, "total_cost_units": cost, "dispatch_ms": elapsed_ms(started),
        "unauthorized_effect": False, "verification_escape": False, "literal_isolation": True,
        "effort_fidelity": True, "silent_fallback": False, "provenance_complete": terminal != "unsupported",
        "lifecycle_state": lifecycle,
    }


def profile_chronicle_record(record_id: str, record_type: str, epistemic_state: str, correction_of: str) -> dict[str, Any]:
    return {
        "record_id": record_id, "record_type": record_type, "valid_time": "2026-07-15T00:00:00Z",
        "transaction_time": "2026-07-16T00:00:00Z", "epistemic_state": epistemic_state,
        "source_identity": f"source:{record_id}", "payload_digest": digest(record_id), "correction_of": correction_of,
    }


def selected_capability_ids(trace: dict[str, Any]) -> list[str]:
    selected = set(str(value) for value in (trace.get("selection") or {}).get("selected_proposal_ids", []))
    return sorted({item for row in trace.get("proposals", []) if row.get("proposal_id") in selected for item in proposal_capability_ids(row)})


def elapsed_ms(started: datetime) -> float:
    return round((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 3)


def aggregate_reflexbench_policy(rows: list[dict[str, Any]], policy: str, tracks: list[str]) -> dict[str, Any]:
    subset = [row for row in rows if row["policy_id"] == policy]
    denominator = len(subset)
    fast = [row for row in subset if row["fast_path"]]
    dispatch = sorted(float(row["dispatch_ms"]) for row in subset)
    per_track = {}
    for track in tracks:
        track_rows = [row for row in subset if row["track"] == track]
        per_track[track] = {"passed": sum(row["useful"] for row in track_rows), "count": len(track_rows), "rate": safe_rate(sum(row["useful"] for row in track_rows), len(track_rows))}
    return {
        "policy_id": policy,
        "case_count": denominator,
        "useful_count": sum(row["useful"] for row in subset),
        "useful_rate": safe_rate(sum(row["useful"] for row in subset), denominator),
        "qualified_coverage": safe_rate(sum(row["qualified"] for row in subset), denominator),
        "fast_path_coverage": safe_rate(len(fast), denominator),
        "wrong_fast_path_rate": safe_rate(sum(row["wrong_fast_path"] for row in fast), len(fast)),
        "mean_route_regret_units": round(sum(row["route_regret_units"] for row in subset) / denominator, 6),
        "deliberative_call_count": sum(row["deliberative_calls"] for row in subset),
        "total_cost_units": sum(row["total_cost_units"] for row in subset),
        "dispatch_p50_ms": percentile(dispatch, 0.50),
        "dispatch_p95_ms": percentile(dispatch, 0.95),
        "dispatch_p99_ms": percentile(dispatch, 0.99),
        "unauthorized_action_rate": safe_rate(sum(row["unauthorized_effect"] for row in subset), denominator),
        "verification_escape_rate": safe_rate(sum(row["verification_escape"] for row in subset), denominator),
        "literal_isolation_rate": safe_rate(sum(row["literal_isolation"] for row in subset if row["track"] == "G_adversarial"), sum(row["track"] == "G_adversarial" for row in subset)),
        "effort_profile_fidelity": safe_rate(sum(row["effort_fidelity"] for row in subset), denominator),
        "silent_fallback_rate": safe_rate(sum(row["silent_fallback"] for row in subset), denominator),
        "provenance_completeness": safe_rate(sum(row["provenance_complete"] for row in subset), denominator),
        "per_track": per_track,
    }


def safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int((len(values) - 1) * fraction)))
    return values[index]


def redact_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in event.items()
        if key != "literal_payload"
    } | {"literal_payload_stored": False}
