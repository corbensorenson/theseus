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
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs" / "reflexive_router_contract.json"
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
    capability_index = validate_capabilities(payload.get("capabilities"))
    validate_user_commands(payload.get("user_commands"), capability_index)
    compile_reflex_index(payload.get("reflexes"), capability_index)
    compile_policy = payload.get("compilation") if isinstance(payload.get("compilation"), dict) else {}
    if set(compile_policy.get("states") or []) != {"ineligible", "candidate", "shadow", "qualified", "quarantined", "decompiled"}:
        raise ReflexiveDispatchFault("REFLEX_COMPILATION_STATES_INVALID", compile_policy, path="compilation.states")
    return copy.deepcopy(payload)


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
        schema = row.get("parameter_schema")
        if not isinstance(schema, dict) or any(not re.fullmatch(r"[a-z][a-z0-9_]*", str(key)) for key in schema):
            raise ReflexiveDispatchFault("REFLEX_COMMAND_PARAMETER_SCHEMA_INVALID", schema, path=f"user_commands[{offset}].parameter_schema")
        ids.add(command_id)
        aliases[alias] = copy.deepcopy(row)
    return aliases


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


def dispatch(
    event: dict[str, Any],
    *,
    intent: str,
    profile: str = "local_private_assistant",
    requested_route: str = "",
    fallback_policy: str = "no_fallback",
    learned_proposals: list[dict[str, Any]] | None = None,
    route_health: dict[str, bool] | None = None,
    plan_nodes: list[dict[str, Any]] | None = None,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = validate_contract(contract or load_contract())
    started = datetime.now(timezone.utc)
    ingress = classify_ingress(event, requested_route=requested_route, fallback_policy=fallback_policy, contract=contract)
    capabilities = validate_capabilities(contract["capabilities"])
    command_index = validate_user_commands(contract["user_commands"], capabilities)
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
            capability=capabilities.get(str(proposal.get("capability_id") or "")),
            profile=profile,
            route_health=route_health or {},
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
    selected_capabilities = [
        capabilities[proposal["capability_id"]]
        for proposal in proposals
        if proposal["proposal_id"] in selection["selected_proposal_ids"]
    ]
    nodes = build_plan_nodes(
        selected_capabilities,
        explicit_nodes=plan_nodes,
        limits=contract["resource_limits"],
    ) if selection["selected_proposal_ids"] else []
    recorded = now()
    trace = {
        "policy": "project_theseus_reflexive_dispatch_trace_v1",
        "source_contract": contract["policy"],
        "event": redact_event(event),
        "ingress": ingress,
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
        for key in ("ingress", "proposals", "qualification", "selection", "plan_nodes")
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
        for key in ("ingress", "proposals", "qualification", "selection", "plan_nodes")
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
            proposals.append(proposal(command["command_id"], "command_binding", command["capability_id"], 1.0, False, False))
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
                )
            )
    return proposals


def proposal(proposal_id: str, proposer_type: str, capability_id: str, score: float, ood: bool, composite: bool) -> dict[str, Any]:
    return {
        "proposal_id": stable_id("route-proposal", proposal_id, capability_id),
        "source_id": proposal_id,
        "proposer_type": proposer_type,
        "capability_id": capability_id,
        "score": score,
        "ood": ood,
        "composite": composite,
        "proposal_grants_authority": False,
        "learned_generation_credit": 0,
    }


def qualify_proposal(
    proposal_row: dict[str, Any],
    *,
    event: dict[str, Any],
    ingress: dict[str, Any],
    capability: dict[str, Any] | None,
    profile: str,
    route_health: dict[str, bool],
) -> dict[str, Any]:
    failures: list[str] = []
    if ingress.get("mode") in {"forced_route", "direct_command"} and not ingress.get("directive_authenticated"):
        failures.append("directive_authentication_failed")
    if capability is None:
        failures.append("capability_unknown")
    else:
        if profile not in set(capability.get("qualified_profiles") or []):
            failures.append("profile_unqualified")
        if capability.get("required_authority") not in set(event.get("authority_refs") or []):
            failures.append("authority_missing")
        if not capability.get("verifier_ref"):
            failures.append("verifier_missing")
        if route_health.get(str(capability.get("capability_id"))) is False:
            failures.append("implementation_stale_or_blocked")
    if proposal_row.get("ood"):
        failures.append("ood")
    if proposal_row.get("proposer_type") == "learned_router" and float(proposal_row.get("score") or 0.0) < 0.8:
        failures.append("selective_risk_abstention")
    qualified = not failures
    receipt = {
        "proposal_id": proposal_row["proposal_id"],
        "capability_id": proposal_row.get("capability_id"),
        "qualified": qualified,
        "obligations": ["schema", "authority", "profile", "freshness", "verifier", "effect_bounds", "explicit_fallback"],
        "failures": failures,
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
        terminal = "unauthorized" if {"authority_missing", "directive_authentication_failed"}.intersection(failures) else ("ood" if "ood" in failures else "unsupported")
        return selection_packet([], "none", f"explicit route unqualified: {sorted(set(failures))}", False, terminal, terminal_outcomes)
    if not qualified and fallback_policy == "explicit_only":
        fallback = next((row for row in proposals if row.get("proposer_type") == "fallback"), None)
        if fallback and qualification_by_id[fallback["proposal_id"]]["qualified"]:
            qualified = [fallback]
            fallback_used = True
    if not qualified:
        failures = [failure for row in qualifications for failure in row["failures"]]
        terminal = "ood" if "ood" in failures else "unsupported"
        return selection_packet([], "none", f"no qualified route: {sorted(set(failures))}", False, terminal, terminal_outcomes)
    ranked = sorted(
        qualified,
        key=lambda row: (
            int(capabilities[row["capability_id"]]["cost_units"]),
            -float(row.get("score") or 0.0),
            str(row["proposal_id"]),
        ),
    )
    selected = ranked[0]
    return selection_packet([selected["proposal_id"]], "dag" if selected.get("composite") else "atomic", "minimum qualified total cost", fallback_used, "prepared", terminal_outcomes)


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


def redact_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in event.items()
        if key != "literal_payload"
    } | {"literal_payload_stored": False}
