#!/usr/bin/env python3
"""Governed champion/challenger controller for future improvement campaigns.

The reference campaign is an architecture fixture. It proves bounded campaign
mechanics and intentionally grants no optimizer or runtime authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import full_state_update_causality


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs" / "open_ended_improvement_campaign.json"


class CampaignFault(ValueError):
    """A fail-closed campaign contract violation."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "policy", "schema_version", "owner", "runtime_enabled", "activation",
        "authorities", "allowed_axes", "required_metrics", "fixed_holdout",
        "matched_budget", "debt_ceilings", "promotion", "maximum_generations",
        "required_cost_fields", "stop_reasons", "claim_boundaries",
    }
    missing = required.difference(contract) if isinstance(contract, dict) else required
    if missing:
        raise CampaignFault(f"contract_missing:{','.join(sorted(missing))}")
    authorities = contract["authorities"]
    if len(set(authorities.values())) != len(authorities):
        raise CampaignFault("campaign_authorities_overlap")
    if int(contract["matched_budget"].get("optimizer_steps", 0)) > 8:
        raise CampaignFault("architecture_canary_budget_exceeded")
    if set(contract["required_cost_fields"]) != set(contract["debt_ceilings"]):
        raise CampaignFault("debt_field_coverage_mismatch")
    if not contract["allowed_axes"] or len(contract["allowed_axes"]) != len(set(contract["allowed_axes"])):
        raise CampaignFault("campaign_axis_contract_invalid")
    return contract


def activation_receipt(evidence: dict[str, Any] | None, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    evidence = copy.deepcopy(evidence or {})
    activation = contract["activation"]
    blockers: list[str] = []
    if not contract["runtime_enabled"]:
        blockers.append("runtime_disabled_by_frozen_contract")
    if evidence.get("evidence_kind") != activation["required_evidence_kind"]:
        blockers.append("behavior_positive_evidence_missing")
    if int(evidence.get("verified_pass_count", 0)) < int(activation["minimum_verified_pass_count"]):
        blockers.append("verified_pass_floor_not_met")
    if int(evidence.get("independent_verifier_count", 0)) < int(activation["minimum_independent_verifier_count"]):
        blockers.append("independent_verifier_floor_not_met")
    if evidence.get("evidence_kind") in set(activation["forbidden_evidence_kinds"]):
        blockers.append("forbidden_activation_evidence")
    if evidence.get("public_solution_exposure") or evidence.get("fallback_credit"):
        blockers.append("activation_evidence_integrity_fault")
    return {
        "authorized": not blockers,
        "blockers": blockers,
        "runtime_enabled": bool(contract["runtime_enabled"]),
        "evidence_digest": digest(evidence),
        "authority": contract["owner"],
    }


def assert_runtime_authorized(evidence: dict[str, Any] | None = None, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    receipt = activation_receipt(evidence, contract)
    if not receipt["authorized"]:
        raise CampaignFault("runtime_campaign_not_authorized:" + ",".join(receipt["blockers"]))
    return receipt


def open_campaign(
    manifest: dict[str, Any],
    contract: dict[str, Any] | None = None,
    *,
    architecture_fixture: bool = False,
) -> dict[str, Any]:
    contract = contract or load_contract()
    authorities = contract["authorities"]
    required = {
        "campaign_id", "generator_id", "evaluator_id", "promoter_id", "stop_authority_id",
        "rollback_authority_id", "holdout_contract", "holdout_digest", "budget",
        "champion_state", "best_state", "final_state", "state_inventory_digest",
    }
    missing = required.difference(manifest)
    if missing:
        raise CampaignFault(f"campaign_manifest_incomplete:{','.join(sorted(missing))}")
    if manifest["generator_id"] == manifest["evaluator_id"]:
        raise CampaignFault("generator_evaluator_overlap")
    expected_authority = {
        "generator_id": authorities["generator"],
        "evaluator_id": authorities["evaluator"],
        "promoter_id": authorities["promoter"],
        "stop_authority_id": authorities["stop"],
        "rollback_authority_id": authorities["rollback"],
    }
    for field, expected in expected_authority.items():
        if manifest[field] != expected:
            raise CampaignFault(f"campaign_authority_mismatch:{field}")
    holdout = contract["fixed_holdout"]
    if manifest["holdout_contract"] != holdout["contract_id"] or manifest["holdout_digest"] != holdout["content_digest"]:
        raise CampaignFault("fixed_holdout_mismatch")
    if manifest["budget"] != contract["matched_budget"]:
        raise CampaignFault("campaign_budget_mismatch")
    if not architecture_fixture:
        assert_runtime_authorized(manifest.get("activation_evidence"), contract)
    champion = copy.deepcopy(manifest["champion_state"])
    best = copy.deepcopy(manifest["best_state"])
    final = copy.deepcopy(manifest["final_state"])
    if digest(champion) != manifest.get("champion_digest"):
        raise CampaignFault("champion_identity_invalid")
    if digest(best) != manifest.get("best_digest") or digest(final) != manifest.get("final_digest"):
        raise CampaignFault("best_final_identity_invalid")
    if manifest["best_digest"] == manifest["final_digest"]:
        raise CampaignFault("best_final_authority_collapsed")
    identity = {key: manifest[key] for key in sorted(required)}
    return {
        "policy": contract["policy"],
        "campaign_id": manifest["campaign_id"],
        "campaign_digest": digest(identity),
        "mode": "architecture_fixture" if architecture_fixture else "runtime",
        "state": "active",
        "epoch_token": digest([identity, now()]),
        "authorities": copy.deepcopy(authorities),
        "holdout_contract": manifest["holdout_contract"],
        "holdout_digest": manifest["holdout_digest"],
        "budget": copy.deepcopy(manifest["budget"]),
        "debt": {field: 0.0 for field in contract["required_cost_fields"]},
        "debt_ceilings": copy.deepcopy(contract["debt_ceilings"]),
        "baseline_champion_state": copy.deepcopy(champion),
        "baseline_champion_digest": manifest["champion_digest"],
        "champion_state": copy.deepcopy(champion),
        "champion_digest": manifest["champion_digest"],
        "best_state": copy.deepcopy(best),
        "best_digest": manifest["best_digest"],
        "final_state": copy.deepcopy(final),
        "final_digest": manifest["final_digest"],
        "state_inventory_digest": manifest["state_inventory_digest"],
        "generation": 0,
        "active_challenger": None,
        "history": [],
        "negative_knowledge": [],
        "journal": [],
        "terminal_reason": "",
        "shutdown_handoff": None,
        "optimizer_exposure_steps": 0,
        "runtime_effect_count": 0,
    }


def propose_challenger(campaign: dict[str, Any], proposal: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    assert_active(campaign)
    if campaign["active_challenger"] is not None:
        raise CampaignFault("challenger_already_active")
    if campaign["generation"] >= int(contract["maximum_generations"]):
        raise CampaignFault("generation_limit_reached")
    if proposal.get("generator_id") != campaign["authorities"]["generator"]:
        raise CampaignFault("proposal_generator_unauthorized")
    axes = proposal.get("mutation_axes")
    if not isinstance(axes, list) or len(axes) != 1 or axes[0] not in contract["allowed_axes"]:
        raise CampaignFault("single_axis_mutation_required")
    if proposal.get("budget") != campaign["budget"]:
        raise CampaignFault("challenger_budget_not_matched")
    if proposal.get("holdout_digest") != campaign["holdout_digest"]:
        raise CampaignFault("challenger_holdout_substitution")
    if proposal.get("authority_delta") not in ({}, None):
        raise CampaignFault("challenger_authority_expansion")
    payload = proposal.get("candidate_state")
    if not isinstance(payload, dict):
        raise CampaignFault("challenger_state_invalid")
    fingerprint = digest({"axis": axes[0], "candidate_state": payload})
    if any(row["fingerprint"] == fingerprint for row in campaign["negative_knowledge"]):
        raise CampaignFault("known_failed_candidate_repeated")
    result = copy.deepcopy(campaign)
    result["generation"] += 1
    result["active_challenger"] = {
        "challenger_id": proposal.get("challenger_id") or f"challenger:{result['generation']}",
        "generation": result["generation"],
        "mutation_axis": axes[0],
        "candidate_state": copy.deepcopy(payload),
        "candidate_digest": digest(payload),
        "fingerprint": fingerprint,
        "baseline_champion_digest": result["champion_digest"],
        "budget": copy.deepcopy(proposal["budget"]),
        "holdout_digest": proposal["holdout_digest"],
        "authority_delta": {},
        "state": "proposed",
        "proposal_receipt": str(proposal.get("proposal_receipt") or ""),
    }
    append_journal(result, "challenger_proposed", result["active_challenger"])
    return result


def evaluate_challenger(campaign: dict[str, Any], evaluation: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    assert_active(campaign)
    challenger = campaign.get("active_challenger")
    if not challenger or challenger["state"] != "proposed":
        raise CampaignFault("no_proposed_challenger")
    if evaluation.get("evaluator_id") != campaign["authorities"]["evaluator"]:
        raise CampaignFault("independent_evaluator_required")
    if evaluation.get("evaluator_id") == campaign["authorities"]["generator"]:
        raise CampaignFault("generator_self_evaluation_forbidden")
    if evaluation.get("challenger_digest") != challenger["candidate_digest"]:
        raise CampaignFault("evaluated_challenger_identity_mismatch")
    if evaluation.get("holdout_digest") != campaign["holdout_digest"]:
        raise CampaignFault("evaluation_holdout_substitution")
    if evaluation.get("budget") != campaign["budget"]:
        raise CampaignFault("evaluation_budget_not_matched")
    metrics = evaluation.get("metrics") or {}
    missing_metrics = set(contract["required_metrics"]).difference(metrics)
    if missing_metrics:
        raise CampaignFault(f"evaluation_metrics_incomplete:{','.join(sorted(missing_metrics))}")
    costs = evaluation.get("costs") or {}
    missing_costs = set(contract["required_cost_fields"]).difference(costs)
    if missing_costs or any(float(costs.get(field, -1)) < 0 for field in contract["required_cost_fields"]):
        raise CampaignFault("campaign_costs_incomplete")
    result = copy.deepcopy(campaign)
    for field in contract["required_cost_fields"]:
        result["debt"][field] += float(costs[field])
    result["active_challenger"]["evaluation"] = copy.deepcopy(evaluation)
    result["active_challenger"]["state"] = "evaluated"
    append_journal(result, "challenger_evaluated", {"challenger_digest": challenger["candidate_digest"], "metrics": metrics, "costs": costs})
    exceeded = [field for field, value in result["debt"].items() if value > float(result["debt_ceilings"][field])]
    if exceeded:
        return stop_campaign(result, "debt_ceiling_exceeded", campaign["authorities"]["stop"], contract, details={"exceeded": exceeded})
    return result


def decide_challenger(campaign: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    assert_active(campaign)
    challenger = campaign.get("active_challenger")
    if not challenger or challenger["state"] != "evaluated":
        raise CampaignFault("challenger_not_evaluated")
    evaluation = challenger["evaluation"]
    if evaluation.get("decision_authority") != campaign["authorities"]["promoter"]:
        raise CampaignFault("promotion_authority_required")
    metrics = evaluation["metrics"]
    baseline = evaluation.get("baseline_metrics") or {}
    promotion = contract["promotion"]
    qualifies = (
        float(metrics["primary_utility"]) - float(baseline.get("primary_utility", 0.0)) >= float(promotion["minimum_primary_delta"])
        and float(metrics["novelty"]) >= float(promotion["minimum_novelty"])
        and float(metrics["coverage"]) - float(baseline.get("coverage", 0.0)) >= float(promotion["minimum_coverage_delta"])
        and float(metrics["weak_tail"]) - float(baseline.get("weak_tail", 0.0)) >= float(promotion["minimum_weak_tail_delta"])
        and int(metrics["verification_escape_count"]) <= int(promotion["maximum_verification_escape_count"])
        and int(metrics["authority_violation_count"]) <= int(promotion["maximum_authority_violation_count"])
    )
    result = copy.deepcopy(campaign)
    decided = result["active_challenger"]
    if qualifies:
        result["champion_state"] = copy.deepcopy(decided["candidate_state"])
        result["champion_digest"] = decided["candidate_digest"]
        result["best_state"] = {"role": "best", "candidate": copy.deepcopy(decided["candidate_state"])}
        result["best_digest"] = digest(result["best_state"])
        decided["state"] = "promoted"
        decision = "promoted"
    else:
        decided["state"] = "rejected"
        result["negative_knowledge"].append({
            "fingerprint": decided["fingerprint"],
            "mutation_axis": decided["mutation_axis"],
            "candidate_digest": decided["candidate_digest"],
            "reason": rejection_reason(metrics, baseline, promotion),
            "evaluation_digest": digest(evaluation),
            "generation": decided["generation"],
        })
        decision = "rejected"
    result["final_state"] = {"role": "final", "candidate": copy.deepcopy(decided["candidate_state"])}
    result["final_digest"] = digest(result["final_state"])
    result["history"].append(copy.deepcopy(decided))
    result["active_challenger"] = None
    append_journal(result, "challenger_decided", {"decision": decision, "candidate_digest": decided["candidate_digest"], "champion_digest": result["champion_digest"]})
    return result


def stop_campaign(
    campaign: dict[str, Any],
    reason: str,
    authority: str,
    contract: dict[str, Any] | None = None,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = contract or load_contract()
    if campaign.get("state") != "active":
        raise CampaignFault("campaign_not_active")
    if authority != campaign["authorities"]["stop"]:
        raise CampaignFault("stop_authority_required")
    if reason not in contract["stop_reasons"]:
        raise CampaignFault("stop_reason_invalid")
    result = copy.deepcopy(campaign)
    if result.get("active_challenger"):
        result["history"].append({**copy.deepcopy(result["active_challenger"]), "state": "aborted_by_stop"})
        result["active_challenger"] = None
    result["state"] = "stopped"
    result["terminal_reason"] = reason
    result["epoch_token"] = "sealed:" + digest([result["campaign_id"], reason, result["journal"]])
    result["shutdown_handoff"] = {
        "authority": authority,
        "reason": reason,
        "details": copy.deepcopy(details or {}),
        "champion_digest": result["champion_digest"],
        "best_digest": result["best_digest"],
        "final_digest": result["final_digest"],
        "negative_knowledge_digest": digest(result["negative_knowledge"]),
        "debt": copy.deepcopy(result["debt"]),
        "unresolved_challenger_count": 0,
    }
    append_journal(result, "campaign_stopped", result["shutdown_handoff"])
    return result


def rollback_campaign(campaign: dict[str, Any], authority: str, *, reason: str) -> dict[str, Any]:
    if authority != campaign["authorities"]["rollback"]:
        raise CampaignFault("rollback_authority_required")
    result = copy.deepcopy(campaign)
    result["champion_state"] = copy.deepcopy(result["baseline_champion_state"])
    result["champion_digest"] = result["baseline_champion_digest"]
    result["active_challenger"] = None
    result["state"] = "rolled_back"
    result["terminal_reason"] = reason
    result["epoch_token"] = "sealed:" + digest([result["campaign_id"], "rollback", reason])
    result["rollback_exact"] = digest(result["champion_state"]) == result["baseline_champion_digest"]
    append_journal(result, "campaign_rolled_back", {"reason": reason, "rollback_exact": result["rollback_exact"]})
    return result


def append_journal(campaign: dict[str, Any], event: str, payload: dict[str, Any]) -> None:
    prior = campaign["journal"][-1]["entry_digest"] if campaign["journal"] else digest([campaign["campaign_id"], campaign["campaign_digest"]])
    entry = {"sequence": len(campaign["journal"]) + 1, "event": event, "payload_digest": digest(payload), "prior_entry_digest": prior}
    entry["entry_digest"] = digest(entry)
    campaign["journal"].append(entry)


def verify_journal(campaign: dict[str, Any]) -> bool:
    prior = digest([campaign["campaign_id"], campaign["campaign_digest"]])
    for sequence, entry in enumerate(campaign.get("journal") or [], 1):
        if entry.get("sequence") != sequence or entry.get("prior_entry_digest") != prior:
            return False
        if entry.get("entry_digest") != digest({key: value for key, value in entry.items() if key != "entry_digest"}):
            return False
        prior = entry["entry_digest"]
    return True


def assert_active(campaign: dict[str, Any]) -> None:
    if campaign.get("state") != "active" or str(campaign.get("epoch_token") or "").startswith("sealed:"):
        raise CampaignFault("campaign_not_active")


def rejection_reason(metrics: dict[str, Any], baseline: dict[str, Any], promotion: dict[str, Any]) -> str:
    if int(metrics["verification_escape_count"]) > int(promotion["maximum_verification_escape_count"]):
        return "verification_escape"
    if int(metrics["authority_violation_count"]) > int(promotion["maximum_authority_violation_count"]):
        return "authority_violation"
    if float(metrics["weak_tail"]) < float(baseline.get("weak_tail", 0.0)):
        return "weak_tail_regression"
    if float(metrics["coverage"]) < float(baseline.get("coverage", 0.0)):
        return "coverage_regression"
    return "primary_utility_floor_not_met"


def reference_manifest(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    inventory = full_state_update_causality.build_reference_inventory()
    champion = {"revision": 1, "policy": {"threshold": 0.50}}
    best = {"revision": 1, "role": "best", "score": 0.50}
    final = {"revision": 1, "role": "final", "step": 0}
    return {
        "campaign_id": "campaign:architecture-fixture-v1",
        "generator_id": contract["authorities"]["generator"],
        "evaluator_id": contract["authorities"]["evaluator"],
        "promoter_id": contract["authorities"]["promoter"],
        "stop_authority_id": contract["authorities"]["stop"],
        "rollback_authority_id": contract["authorities"]["rollback"],
        "holdout_contract": contract["fixed_holdout"]["contract_id"],
        "holdout_digest": contract["fixed_holdout"]["content_digest"],
        "budget": copy.deepcopy(contract["matched_budget"]),
        "champion_state": champion,
        "champion_digest": digest(champion),
        "best_state": best,
        "best_digest": digest(best),
        "final_state": final,
        "final_digest": digest(final),
        "state_inventory_digest": inventory["inventory_digest"],
    }


def reference_proposal(campaign: dict[str, Any], challenger_id: str, axis: str, revision: int) -> dict[str, Any]:
    return {
        "challenger_id": challenger_id,
        "generator_id": campaign["authorities"]["generator"],
        "mutation_axes": [axis],
        "candidate_state": {"revision": revision, "policy": {"threshold": 0.50 + revision / 100.0}},
        "budget": copy.deepcopy(campaign["budget"]),
        "holdout_digest": campaign["holdout_digest"],
        "authority_delta": {},
        "proposal_receipt": f"proposal:{challenger_id}",
    }


def reference_evaluation(campaign: dict[str, Any], *, utility: float, coverage: float, weak_tail: float, debt: float = 0.25) -> dict[str, Any]:
    challenger = campaign["active_challenger"]
    return {
        "evaluator_id": campaign["authorities"]["evaluator"],
        "decision_authority": campaign["authorities"]["promoter"],
        "challenger_digest": challenger["candidate_digest"],
        "holdout_digest": campaign["holdout_digest"],
        "budget": copy.deepcopy(campaign["budget"]),
        "baseline_metrics": {"primary_utility": 0.50, "coverage": 0.80, "weak_tail": 0.40},
        "metrics": {"primary_utility": utility, "novelty": 0.10, "coverage": coverage, "weak_tail": weak_tail, "verification_escape_count": 0, "authority_violation_count": 0},
        "costs": {field: debt for field in campaign["debt"]},
        "evaluation_receipt": f"evaluation:{challenger['challenger_id']}",
    }


def run_reference_campaign(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    campaign = open_campaign(reference_manifest(contract), contract, architecture_fixture=True)
    campaign = propose_challenger(campaign, reference_proposal(campaign, "tail-regression", "router_policy", 2), contract)
    rejected_fingerprint = campaign["active_challenger"]["fingerprint"]
    campaign = evaluate_challenger(campaign, reference_evaluation(campaign, utility=0.53, coverage=0.82, weak_tail=0.35), contract)
    campaign = decide_challenger(campaign, contract)
    campaign = propose_challenger(campaign, reference_proposal(campaign, "qualified", "context_policy", 3), contract)
    campaign = evaluate_challenger(campaign, reference_evaluation(campaign, utility=0.54, coverage=0.84, weak_tail=0.42), contract)
    campaign = decide_challenger(campaign, contract)
    champion_before_stop = campaign["champion_digest"]
    campaign = stop_campaign(campaign, "shutdown_handoff", campaign["authorities"]["stop"], contract, details={"next_owner": "operator"})
    rollback = rollback_campaign(campaign, campaign["authorities"]["rollback"], reason="fixture_exact_restore")
    controls = mutation_controls(contract)
    activation = activation_receipt({}, contract)
    full_state = full_state_update_causality.run_reference_fixture()
    gates = {
        "generator_evaluator_separated": contract["authorities"]["generator"] != contract["authorities"]["evaluator"],
        "single_axis_and_matched_budget_enforced": controls["passed_count"] == controls["case_count"],
        "fixed_holdout_preserved": all(row.get("holdout_digest") == contract["fixed_holdout"]["content_digest"] for row in campaign["history"]),
        "rejected_family_retained": any(row["fingerprint"] == rejected_fingerprint for row in campaign["negative_knowledge"]),
        "qualified_challenger_promoted": champion_before_stop != campaign["baseline_champion_digest"],
        "best_final_authority_distinct": campaign["best_digest"] != campaign["final_digest"],
        "full_state_rollback_exact": bool(full_state["gates"]["rollback_exact"]) and rollback["rollback_exact"],
        "debt_visible_and_bounded": all(campaign["debt"][field] <= campaign["debt_ceilings"][field] for field in campaign["debt"]),
        "shutdown_handoff_complete": campaign["shutdown_handoff"]["unresolved_challenger_count"] == 0,
        "journal_valid": verify_journal(campaign) and verify_journal(rollback),
        "runtime_activation_disabled": not activation["authorized"],
        "zero_optimizer_exposure": campaign["optimizer_exposure_steps"] == 0,
        "zero_runtime_effects": campaign["runtime_effect_count"] == 0,
    }
    return {
        "policy": contract["policy"],
        "trigger_state": "GREEN" if all(gates.values()) else "RED",
        "support_state": "synthetic-test-backed",
        "summary": {
            "generation_count": campaign["generation"],
            "promoted_count": sum(row["state"] == "promoted" for row in campaign["history"]),
            "rejected_count": sum(row["state"] == "rejected" for row in campaign["history"]),
            "negative_knowledge_count": len(campaign["negative_knowledge"]),
            "mutation_case_count": controls["case_count"],
            "mutation_passed_count": controls["passed_count"],
            "runtime_authorized": activation["authorized"],
            "rollback_exact": rollback["rollback_exact"],
            "optimizer_exposure_steps": campaign["optimizer_exposure_steps"],
            "runtime_effect_count": campaign["runtime_effect_count"],
        },
        "gates": gates,
        "activation_receipt": activation,
        "campaign_receipt": {
            "campaign_id": campaign["campaign_id"],
            "mode": campaign["mode"],
            "state": campaign["state"],
            "terminal_reason": campaign["terminal_reason"],
            "champion_digest": campaign["champion_digest"],
            "best_digest": campaign["best_digest"],
            "final_digest": campaign["final_digest"],
            "holdout_digest": campaign["holdout_digest"],
            "state_inventory_digest": campaign["state_inventory_digest"],
            "debt": campaign["debt"],
            "negative_knowledge": campaign["negative_knowledge"],
            "shutdown_handoff": campaign["shutdown_handoff"],
            "journal_tail": campaign["journal"][-1],
        },
        "rollback_receipt": {key: rollback.get(key) for key in ("state", "terminal_reason", "rollback_exact", "champion_digest", "baseline_champion_digest", "epoch_token")},
        "mutation_controls": controls,
        "non_claims": copy.deepcopy(contract["claim_boundaries"]),
    }


def mutation_controls(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    manifest = reference_manifest(contract)
    cases: list[dict[str, Any]] = []

    def record(case_id: str, expected: str, action: Any) -> None:
        observed = "accepted"
        try:
            action()
        except CampaignFault as exc:
            observed = str(exc)
        cases.append({"case_id": case_id, "passed": expected in observed, "expected": expected, "observed": observed})

    bad = copy.deepcopy(manifest); bad["evaluator_id"] = bad["generator_id"]
    record("generator_evaluator_overlap", "generator_evaluator_overlap", lambda: open_campaign(bad, contract, architecture_fixture=True))
    bad = copy.deepcopy(manifest); bad["holdout_digest"] = "sha256:substituted"
    record("holdout_substitution", "fixed_holdout_mismatch", lambda: open_campaign(bad, contract, architecture_fixture=True))
    bad = copy.deepcopy(manifest); bad["budget"]["optimizer_steps"] = 7
    record("unmatched_campaign_budget", "campaign_budget_mismatch", lambda: open_campaign(bad, contract, architecture_fixture=True))
    campaign = open_campaign(manifest, contract, architecture_fixture=True)
    proposal = reference_proposal(campaign, "multi-axis", "router_policy", 2); proposal["mutation_axes"] = ["router_policy", "context_policy"]
    record("multi_axis_challenger", "single_axis_mutation_required", lambda: propose_challenger(campaign, proposal, contract))
    proposal = reference_proposal(campaign, "authority", "router_policy", 2); proposal["authority_delta"] = {"network": "expanded"}
    record("authority_expansion", "challenger_authority_expansion", lambda: propose_challenger(campaign, proposal, contract))
    proposal = reference_proposal(campaign, "budget", "router_policy", 2); proposal["budget"]["candidate_count"] += 1
    record("unmatched_challenger_budget", "challenger_budget_not_matched", lambda: propose_challenger(campaign, proposal, contract))
    proposed = propose_challenger(campaign, reference_proposal(campaign, "self-eval", "router_policy", 2), contract)
    evaluation = reference_evaluation(proposed, utility=0.55, coverage=0.9, weak_tail=0.5); evaluation["evaluator_id"] = contract["authorities"]["generator"]
    record("generator_self_evaluation", "independent_evaluator_required", lambda: evaluate_challenger(proposed, evaluation, contract))
    evaluation = reference_evaluation(proposed, utility=0.55, coverage=0.9, weak_tail=0.5); evaluation["holdout_digest"] = "sha256:changed"
    record("evaluation_holdout_substitution", "evaluation_holdout_substitution", lambda: evaluate_challenger(proposed, evaluation, contract))
    evaluation = reference_evaluation(proposed, utility=0.55, coverage=0.9, weak_tail=0.5); evaluation["metrics"].pop("weak_tail")
    record("missing_tail_metric", "evaluation_metrics_incomplete", lambda: evaluate_challenger(proposed, evaluation, contract))
    evaluated = evaluate_challenger(proposed, reference_evaluation(proposed, utility=0.55, coverage=0.9, weak_tail=0.5), contract)
    evaluated["active_challenger"]["evaluation"]["decision_authority"] = contract["authorities"]["generator"]
    record("self_promotion", "promotion_authority_required", lambda: decide_challenger(evaluated, contract))
    record("unauthorized_stop", "stop_authority_required", lambda: stop_campaign(campaign, "operator_stop", "candidate_generator", contract))
    record("unauthorized_rollback", "rollback_authority_required", lambda: rollback_campaign(campaign, "candidate_generator", reason="fault"))
    runtime_manifest = copy.deepcopy(manifest)
    record("silent_runtime_activation", "runtime_campaign_not_authorized", lambda: open_campaign(runtime_manifest, contract, architecture_fixture=False))
    rejected = propose_challenger(campaign, reference_proposal(campaign, "known-failure", "router_policy", 2), contract)
    repeated = copy.deepcopy(campaign); repeated["negative_knowledge"] = [{"fingerprint": rejected["active_challenger"]["fingerprint"]}]
    record("known_failure_repeat", "known_failed_candidate_repeated", lambda: propose_challenger(repeated, reference_proposal(repeated, "known-failure", "router_policy", 2), contract))
    return {"case_count": len(cases), "passed_count": sum(bool(row["passed"]) for row in cases), "results": cases}
