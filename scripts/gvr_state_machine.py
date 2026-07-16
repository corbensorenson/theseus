#!/usr/bin/env python3
"""Typed generate-verify-repair lifecycle and credit-accounting kernel."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


STATES = {
    "generated",
    "verified_exact",
    "verified_lossy",
    "repaired_exact",
    "literal_fallback_noncredit",
    "quarantined",
}
ALLOWED = {
    "generated": {"verified_exact", "verified_lossy", "literal_fallback_noncredit", "quarantined"},
    "verified_exact": {"quarantined"},
    "verified_lossy": {"repaired_exact", "literal_fallback_noncredit", "quarantined"},
    "repaired_exact": {"quarantined"},
    "literal_fallback_noncredit": {"quarantined"},
    "quarantined": set(),
}


class GVRStateFault(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def create_candidate(*, code_sha256: str, generator_revision: str, checkpoint_id: str, source_context_digest: str) -> dict[str, Any]:
    if not all((code_sha256, generator_revision, checkpoint_id, source_context_digest)):
        raise GVRStateFault("candidate_identity_incomplete")
    identity = {"code_sha256": code_sha256, "generator_revision": generator_revision, "checkpoint_id": checkpoint_id, "source_context_digest": source_context_digest}
    return {
        "policy": "project_theseus_gvr_state_machine_v1",
        "candidate_id": digest(identity),
        "source_candidate_id": digest(identity),
        "state": "generated",
        "identity": identity,
        "current_artifact_sha256": code_sha256,
        "learned_generation_credit": 1,
        "assisted_repair_credit": 0,
        "fallback_count": 0,
        "repair_cost_units": 0,
        "verifier_receipts": [],
        "repair_receipts": [],
        "transitions": [],
        "rollback_snapshot": None,
    }


def transition(candidate: dict[str, Any], target: str, receipt: dict[str, Any]) -> dict[str, Any]:
    source = str(candidate.get("state") or "")
    if source not in STATES or target not in STATES or target not in ALLOWED[source]:
        raise GVRStateFault(f"illegal_transition:{source}->{target}")
    if candidate.get("candidate_id") != candidate.get("source_candidate_id") or candidate.get("candidate_id") != digest(candidate.get("identity")):
        raise GVRStateFault("source_candidate_identity_mutated")
    if target in {"verified_exact", "verified_lossy", "repaired_exact"}:
        validate_verifier_receipt(receipt, candidate, target)
    result = copy.deepcopy(candidate)
    if target == "repaired_exact":
        repair = receipt.get("repair_receipt") or {}
        validate_repair_receipt(repair, result)
        result["rollback_snapshot"] = {"state": source, "current_artifact_sha256": result["current_artifact_sha256"], "verifier_receipt_count": len(result["verifier_receipts"]), "repair_receipt_count": len(result["repair_receipts"])}
        result["current_artifact_sha256"] = repair["repaired_artifact_sha256"]
        result["repair_cost_units"] += int(repair["cost_units"])
        result["repair_receipts"].append(copy.deepcopy(repair))
        result["learned_generation_credit"] = 0
        result["assisted_repair_credit"] = 1
    elif target == "literal_fallback_noncredit":
        if receipt.get("fallback_kind") not in {"literal", "template", "deterministic_renderer"}:
            raise GVRStateFault("fallback_receipt_invalid")
        result["learned_generation_credit"] = 0
        result["fallback_count"] += 1
    elif target == "quarantined":
        if not receipt.get("reason"):
            raise GVRStateFault("quarantine_reason_missing")
        result["learned_generation_credit"] = 0
    if target in {"verified_exact", "verified_lossy", "repaired_exact"}:
        result["verifier_receipts"].append(copy.deepcopy(receipt))
    event = {"sequence": len(result["transitions"]) + 1, "from": source, "to": target, "receipt_digest": digest(receipt), "source_candidate_id": result["source_candidate_id"], "artifact_sha256": result["current_artifact_sha256"]}
    event["transition_digest"] = digest(event)
    result["transitions"].append(event)
    result["state"] = target
    return result


def validate_verifier_receipt(receipt: dict[str, Any], candidate: dict[str, Any], target: str) -> None:
    required = {"verifier_id", "verifier_revision", "candidate_id", "artifact_sha256", "independent", "tests_digest", "verdict"}
    if required.difference(receipt):
        raise GVRStateFault("verifier_receipt_incomplete")
    if receipt["candidate_id"] != candidate["source_candidate_id"] or receipt["artifact_sha256"] != candidate["current_artifact_sha256"]:
        repair = receipt.get("repair_receipt") or {}
        if target != "repaired_exact" or receipt["artifact_sha256"] != repair.get("repaired_artifact_sha256"):
            raise GVRStateFault("verifier_candidate_binding_mismatch")
    if receipt["independent"] is not True:
        raise GVRStateFault("verifier_not_independent")
    expected = {"verified_exact": "exact", "verified_lossy": "lossy", "repaired_exact": "exact"}[target]
    if receipt["verdict"] != expected:
        raise GVRStateFault("verifier_verdict_state_mismatch")


def validate_repair_receipt(repair: dict[str, Any], candidate: dict[str, Any]) -> None:
    required = {"repair_id", "authority", "source_artifact_sha256", "repaired_artifact_sha256", "changed_atom_ids", "cost_units", "candidate_generation_credit"}
    if required.difference(repair):
        raise GVRStateFault("repair_receipt_incomplete")
    if repair["authority"] != "semantic_ir_localized_repair" or repair["source_artifact_sha256"] != candidate["current_artifact_sha256"]:
        raise GVRStateFault("repair_authority_or_source_invalid")
    if not repair["changed_atom_ids"] or int(repair["cost_units"]) < 0 or int(repair["cost_units"]) > 32:
        raise GVRStateFault("repair_scope_or_budget_invalid")
    if int(repair["candidate_generation_credit"]) != 0:
        raise GVRStateFault("assisted_repair_claims_learned_credit")


def rollback_repair(candidate: dict[str, Any], *, reason: str) -> dict[str, Any]:
    snapshot = candidate.get("rollback_snapshot") or {}
    if candidate.get("state") != "repaired_exact" or not snapshot or not reason:
        raise GVRStateFault("repair_rollback_unavailable")
    result = copy.deepcopy(candidate)
    result["state"] = "quarantined"
    result["current_artifact_sha256"] = snapshot["current_artifact_sha256"]
    result["learned_generation_credit"] = 0
    result["assisted_repair_credit"] = 0
    result["rollback_exact"] = result["current_artifact_sha256"] == result["identity"]["code_sha256"]
    result["terminal_reason"] = reason
    return result


def verify_history(candidate: dict[str, Any]) -> bool:
    prior_state = "generated"
    for sequence, row in enumerate(candidate.get("transitions") or [], 1):
        if row.get("sequence") != sequence or row.get("from") != prior_state or row.get("to") not in ALLOWED.get(prior_state, set()):
            return False
        if row.get("source_candidate_id") != candidate.get("source_candidate_id"):
            return False
        expected = digest({key: value for key, value in row.items() if key != "transition_digest"})
        if row.get("transition_digest") != expected:
            return False
        prior_state = row["to"]
    return prior_state == candidate.get("state")


def run_reference_fixture() -> dict[str, Any]:
    base = create_candidate(code_sha256="a" * 64, generator_revision="generator:v1", checkpoint_id="checkpoint:v1", source_context_digest="context:private")
    lossy_receipt = {"verifier_id": "verifier:independent", "verifier_revision": "v1", "candidate_id": base["candidate_id"], "artifact_sha256": base["current_artifact_sha256"], "independent": True, "tests_digest": "tests:private", "verdict": "lossy"}
    lossy = transition(base, "verified_lossy", lossy_receipt)
    repair = {"repair_id": "repair:001", "authority": "semantic_ir_localized_repair", "source_artifact_sha256": lossy["current_artifact_sha256"], "repaired_artifact_sha256": "b" * 64, "changed_atom_ids": ["atom:return-shape"], "cost_units": 3, "candidate_generation_credit": 0}
    exact_receipt = {"verifier_id": "verifier:independent", "verifier_revision": "v1", "candidate_id": base["candidate_id"], "artifact_sha256": repair["repaired_artifact_sha256"], "independent": True, "tests_digest": "tests:private", "verdict": "exact", "repair_receipt": repair}
    repaired = transition(lossy, "repaired_exact", exact_receipt)
    rolled = rollback_repair(repaired, reason="postcommit_verifier_disagreement")
    fallback = transition(base, "literal_fallback_noncredit", {"fallback_kind": "literal"})
    mutations = mutation_controls()
    gates = {"repair_history_valid": verify_history(repaired), "repair_has_zero_learned_credit": repaired["learned_generation_credit"] == 0 and repaired["assisted_repair_credit"] == 1, "fallback_has_zero_learned_credit": fallback["learned_generation_credit"] == 0 and fallback["fallback_count"] == 1, "rollback_exact_and_quarantined": rolled.get("rollback_exact") is True and rolled["state"] == "quarantined", "mutations_rejected": mutations["case_count"] == mutations["passed_count"]}
    return {"policy": "project_theseus_gvr_state_machine_v1", "trigger_state": "GREEN" if all(gates.values()) else "RED", "summary": {"state_count": len(STATES), "transition_count": len(repaired["transitions"]), "repair_cost_units": repaired["repair_cost_units"], "fallback_count": fallback["fallback_count"], "mutation_case_count": mutations["case_count"], "mutation_passed_count": mutations["passed_count"]}, "gates": gates, "repaired_receipt": {"source_candidate_id": repaired["source_candidate_id"], "state": repaired["state"], "learned_generation_credit": repaired["learned_generation_credit"], "assisted_repair_credit": repaired["assisted_repair_credit"], "current_artifact_sha256": repaired["current_artifact_sha256"]}, "rollback_receipt": {"state": rolled["state"], "rollback_exact": rolled["rollback_exact"], "terminal_reason": rolled["terminal_reason"]}, "mutation_controls": mutations, "non_claims": ["Deterministic repaired output is useful-product evidence only, never learned-generation credit.", "State-machine mechanics do not establish repair efficacy or model capability."]}


def mutation_controls() -> dict[str, Any]:
    base = create_candidate(code_sha256="a" * 64, generator_revision="g", checkpoint_id="c", source_context_digest="s")
    cases = []
    def reject(case_id: str, fn: Any, expected: str) -> None:
        observed = "accepted"
        try: fn()
        except GVRStateFault as exc: observed = str(exc)
        cases.append({"case_id": case_id, "passed": observed == expected, "expected": expected, "observed": observed})
    reject("illegal_transition", lambda: transition(base, "repaired_exact", {}), "illegal_transition:generated->repaired_exact")
    mutated = copy.deepcopy(base); mutated["candidate_id"] = "changed"
    reject("identity_mutation", lambda: transition(mutated, "quarantined", {"reason": "fault"}), "source_candidate_identity_mutated")
    bad_verifier = {"verifier_id": "v", "verifier_revision": "1", "candidate_id": base["candidate_id"], "artifact_sha256": base["current_artifact_sha256"], "independent": False, "tests_digest": "t", "verdict": "exact"}
    reject("nonindependent_verifier", lambda: transition(base, "verified_exact", bad_verifier), "verifier_not_independent")
    lossy = transition(base, "verified_lossy", {**bad_verifier, "independent": True, "verdict": "lossy"})
    repair = {"repair_id": "r", "authority": "semantic_ir_localized_repair", "source_artifact_sha256": base["current_artifact_sha256"], "repaired_artifact_sha256": "b" * 64, "changed_atom_ids": ["a"], "cost_units": 1, "candidate_generation_credit": 1}
    exact = {**bad_verifier, "independent": True, "verdict": "exact", "artifact_sha256": "b" * 64, "repair_receipt": repair}
    reject("repair_credit_fraud", lambda: transition(lossy, "repaired_exact", exact), "assisted_repair_claims_learned_credit")
    return {"case_count": len(cases), "passed_count": sum(bool(row["passed"]) for row in cases), "results": cases}

