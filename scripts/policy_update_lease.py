#!/usr/bin/env python3
"""Target-scoped behavior-update leases with fail-closed rollback semantics."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs" / "policy_update_lease.json"


class PolicyUpdateLeaseFault(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def utc(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).isoformat()


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"policy", "schema_version", "owner", "issuer", "targets", "invariants"}
    missing = required.difference(value) if isinstance(value, dict) else required
    if missing:
        raise PolicyUpdateLeaseFault(f"contract_missing:{','.join(sorted(missing))}")
    targets = value["targets"]
    if not isinstance(targets, dict) or set(targets) != {"planner", "router", "vcm_selector", "verifier", "executor", "generator", "generation_mode"}:
        raise PolicyUpdateLeaseFault("target_coverage_invalid")
    paths = [row.get("state_path") for row in targets.values()]
    if len(paths) != len(set(paths)) or any(not path for path in paths):
        raise PolicyUpdateLeaseFault("target_state_paths_overlap")
    return value


def issue_lease(request: dict[str, Any], contract: dict[str, Any] | None = None, *, active_leases: list[dict[str, Any]] | None = None, clock: datetime | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    target_id = str(request.get("target_id") or "")
    if target_id not in contract["targets"]:
        raise PolicyUpdateLeaseFault("target_unknown")
    target = contract["targets"][target_id]
    feedback = request.get("feedback_receipt") or {}
    feedback_class = str(feedback.get("class") or "")
    if not feedback.get("receipt_id") or feedback_class not in target["allowed_feedback"]:
        raise PolicyUpdateLeaseFault("feedback_not_admissible")
    if feedback_class in set(contract.get("forbidden_feedback_classes") or []):
        raise PolicyUpdateLeaseFault("feedback_forbidden")
    if request.get("authority_delta") not in ({}, None):
        raise PolicyUpdateLeaseFault("authority_expansion_forbidden")
    baseline = request.get("baseline_state")
    if not isinstance(baseline, dict) or not request.get("baseline_digest") or digest(baseline) != request["baseline_digest"]:
        raise PolicyUpdateLeaseFault("baseline_identity_invalid")
    if request.get("heldout_contract") != target["heldout_contract"]:
        raise PolicyUpdateLeaseFault("heldout_contract_mismatch")
    costs = request.get("cost_budget") or {}
    required_costs = set(contract.get("required_cost_fields") or [])
    if required_costs.difference(costs) or any(float(costs[key]) < 0 for key in required_costs):
        raise PolicyUpdateLeaseFault("cost_budget_incomplete")
    if not request.get("updater_revision") or not request.get("rollback_handler") or not request.get("data_receipt_ids"):
        raise PolicyUpdateLeaseFault("update_provenance_incomplete")
    for active in active_leases or []:
        if active.get("state") == "active" and active.get("target_id") == target_id:
            raise PolicyUpdateLeaseFault("target_lease_conflict")
    started = clock or datetime.now(timezone.utc)
    identity = {
        "target_id": target_id,
        "state_path": target["state_path"],
        "owner": target["owner"],
        "baseline_digest": request["baseline_digest"],
        "feedback_receipt_id": feedback["receipt_id"],
        "updater_revision": request["updater_revision"],
        "heldout_contract": request["heldout_contract"],
        "data_receipt_ids": sorted(request["data_receipt_ids"]),
    }
    return {
        "policy": contract["policy"],
        "lease_id": digest(identity),
        "epoch_token": digest([identity, utc(started)]),
        "target_id": target_id,
        "state_path": target["state_path"],
        "target_owner": target["owner"],
        "state": "active",
        "issued_utc": utc(started),
        "expires_utc": utc(started + timedelta(seconds=int(contract.get("lease_ttl_seconds") or 0))),
        "baseline_state": copy.deepcopy(baseline),
        "baseline_digest": request["baseline_digest"],
        "current_state": copy.deepcopy(baseline),
        "current_digest": request["baseline_digest"],
        "feedback_receipt": copy.deepcopy(feedback),
        "heldout_contract": request["heldout_contract"],
        "updater_revision": request["updater_revision"],
        "data_receipt_ids": sorted(request["data_receipt_ids"]),
        "cost_budget": copy.deepcopy(costs),
        "cost_observed": {key: 0.0 for key in costs},
        "authority_delta": {},
        "rollback_handler": request["rollback_handler"],
        "monitor_window": {"minimum_observations": int(contract.get("minimum_monitor_observations") or 1), "drift_limit": float(target["drift_limit"])},
        "observations": [],
        "journal": [],
        "terminal_reason": "",
    }


def apply_delta(lease: dict[str, Any], delta: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    assert_active(lease)
    if delta.get("target_id") != lease["target_id"] or delta.get("state_path") != lease["state_path"]:
        raise PolicyUpdateLeaseFault("cross_target_write_forbidden")
    if delta.get("before_digest") != lease["current_digest"]:
        raise PolicyUpdateLeaseFault("delta_baseline_stale")
    next_state = delta.get("next_state")
    if not isinstance(next_state, dict):
        raise PolicyUpdateLeaseFault("delta_state_invalid")
    observed_cost = delta.get("cost_observed") or {}
    for key, budget in lease["cost_budget"].items():
        value = float(observed_cost.get(key, 0.0))
        if key not in observed_cost:
            raise PolicyUpdateLeaseFault("cost_observation_incomplete")
        if lease["cost_observed"][key] + value > float(budget):
            raise PolicyUpdateLeaseFault("consequence_budget_exceeded")
    result = copy.deepcopy(lease)
    for key, value in observed_cost.items():
        result["cost_observed"][key] += float(value)
    prior_entry = result["journal"][-1]["entry_digest"] if result["journal"] else digest([result["lease_id"], result["baseline_digest"]])
    entry = {
        "sequence": len(result["journal"]) + 1,
        "target_id": result["target_id"],
        "state_path": result["state_path"],
        "before_digest": result["current_digest"],
        "after_digest": digest(next_state),
        "feedback_receipt_id": result["feedback_receipt"]["receipt_id"],
        "cost_observed": copy.deepcopy(observed_cost),
        "prior_entry_digest": prior_entry,
    }
    entry["entry_digest"] = digest(entry)
    result["journal"].append(entry)
    result["current_state"] = copy.deepcopy(next_state)
    result["current_digest"] = entry["after_digest"]
    return result


def observe(lease: dict[str, Any], observation: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    assert_active(lease)
    if observation.get("heldout_contract") != lease["heldout_contract"]:
        raise PolicyUpdateLeaseFault("monitor_heldout_mismatch")
    required = {"observation_id", "drift", "heldout_delta", "verification_escape_count", "authority_violation_count"}
    if required.difference(observation):
        raise PolicyUpdateLeaseFault("monitor_observation_incomplete")
    result = copy.deepcopy(lease)
    result["observations"].append(copy.deepcopy(observation))
    breached = (
        abs(float(observation["drift"])) > float(result["monitor_window"]["drift_limit"])
        or float(observation["heldout_delta"]) < 0
        or int(observation["verification_escape_count"]) > 0
        or int(observation["authority_violation_count"]) > 0
    )
    if breached:
        return rollback(result, reason="monitor_breach")
    return result


def commit(lease: dict[str, Any]) -> dict[str, Any]:
    assert_active(lease)
    if len(lease["observations"]) < int(lease["monitor_window"]["minimum_observations"]):
        raise PolicyUpdateLeaseFault("monitor_window_incomplete")
    if not lease["journal"] or not verify_journal(lease):
        raise PolicyUpdateLeaseFault("journal_invalid")
    result = copy.deepcopy(lease)
    result["state"] = "committed"
    result["terminal_reason"] = "heldout_and_sentinels_passed"
    result["committed_utc"] = utc()
    return result


def rollback(lease: dict[str, Any], *, reason: str) -> dict[str, Any]:
    result = copy.deepcopy(lease)
    result["state"] = "rolled_back"
    result["current_state"] = copy.deepcopy(result["baseline_state"])
    result["current_digest"] = result["baseline_digest"]
    result["epoch_token"] = "sealed:" + digest([result["lease_id"], reason])
    result["terminal_reason"] = reason
    result["rollback_exact"] = digest(result["current_state"]) == result["baseline_digest"]
    return result


def verify_journal(lease: dict[str, Any]) -> bool:
    prior = digest([lease["lease_id"], lease["baseline_digest"]])
    for sequence, row in enumerate(lease.get("journal") or [], 1):
        if row.get("sequence") != sequence or row.get("prior_entry_digest") != prior:
            return False
        expected = digest({key: value for key, value in row.items() if key != "entry_digest"})
        if row.get("entry_digest") != expected:
            return False
        prior = row["entry_digest"]
    return True


def assert_active(lease: dict[str, Any]) -> None:
    if lease.get("state") != "active" or str(lease.get("epoch_token") or "").startswith("sealed:"):
        raise PolicyUpdateLeaseFault("lease_not_active")


def reference_request(target_id: str, target: dict[str, Any]) -> dict[str, Any]:
    baseline = {"revision": 1, "target_id": target_id, "parameters": {"threshold": 0.5}}
    return {
        "target_id": target_id,
        "baseline_state": baseline,
        "baseline_digest": digest(baseline),
        "feedback_receipt": {"receipt_id": f"feedback:{target_id}:001", "class": target["allowed_feedback"][0]},
        "heldout_contract": target["heldout_contract"],
        "updater_revision": "updater:reference-v1",
        "data_receipt_ids": [f"data:{target_id}:001"],
        "authority_delta": {},
        "rollback_handler": "replacement_transaction_kernel.restore_exact",
        "cost_budget": {"verification": 10, "repair": 10, "human_cleanup": 5, "compute": 20, "energy": 20},
    }


def run_reference_matrix(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    leases = []
    for target_id, target in contract["targets"].items():
        lease = issue_lease(reference_request(target_id, target), contract, active_leases=leases)
        next_state = copy.deepcopy(lease["current_state"])
        next_state["revision"] = 2
        next_state["parameters"]["threshold"] = 0.51
        lease = apply_delta(lease, {
            "target_id": target_id,
            "state_path": target["state_path"],
            "before_digest": lease["current_digest"],
            "next_state": next_state,
            "cost_observed": {"verification": 1, "repair": 0, "human_cleanup": 0, "compute": 2, "energy": 2},
        }, contract)
        for suffix in ("a", "b"):
            lease = observe(lease, {"observation_id": f"{target_id}:{suffix}", "heldout_contract": target["heldout_contract"], "drift": 0.0, "heldout_delta": 0.01, "verification_escape_count": 0, "authority_violation_count": 0}, contract)
        leases.append(commit(lease))
    breach_target = "router"
    breach = issue_lease(reference_request(breach_target, contract["targets"][breach_target]), contract)
    next_state = copy.deepcopy(breach["current_state"])
    next_state["revision"] = 2
    breach = apply_delta(breach, {"target_id": breach_target, "state_path": breach["state_path"], "before_digest": breach["current_digest"], "next_state": next_state, "cost_observed": {"verification": 1, "repair": 0, "human_cleanup": 0, "compute": 1, "energy": 1}}, contract)
    breach = observe(breach, {"observation_id": "router:breach", "heldout_contract": breach["heldout_contract"], "drift": 0.5, "heldout_delta": -0.1, "verification_escape_count": 1, "authority_violation_count": 0}, contract)
    controls = mutation_controls(contract)
    return {
        "policy": contract["policy"],
        "trigger_state": "GREEN" if len(leases) == len(contract["targets"]) and all(row["state"] == "committed" and verify_journal(row) for row in leases) and breach.get("rollback_exact") and controls["passed_count"] == controls["case_count"] else "RED",
        "summary": {"target_count": len(contract["targets"]), "committed_target_count": len(leases), "rollback_canary_exact": bool(breach.get("rollback_exact")), "mutation_case_count": controls["case_count"], "mutation_passed_count": controls["passed_count"]},
        "target_receipts": [{"target_id": row["target_id"], "lease_id": row["lease_id"], "state_path": row["state_path"], "heldout_contract": row["heldout_contract"], "state": row["state"], "journal_digest": row["journal"][-1]["entry_digest"], "cost_observed": row["cost_observed"], "authority_delta": row["authority_delta"]} for row in leases],
        "breach_rollback_receipt": {key: breach.get(key) for key in ("target_id", "state", "terminal_reason", "rollback_exact", "current_digest", "baseline_digest", "epoch_token")},
        "mutation_controls": controls,
        "non_claims": ["Lease mechanics do not prove that any learned update improves behavior.", "Reference thresholds are deterministic fixtures, not deployed target policies.", "A committed lease does not authorize public-data training, external runtime inference, or authority expansion."],
    }


def mutation_controls(contract: dict[str, Any]) -> dict[str, Any]:
    target_id = "planner"
    target = contract["targets"][target_id]
    base = reference_request(target_id, target)
    cases = []

    def rejected(case_id: str, request: dict[str, Any], expected: str, active: list[dict[str, Any]] | None = None) -> None:
        observed = "accepted"
        try:
            issue_lease(request, contract, active_leases=active)
        except PolicyUpdateLeaseFault as exc:
            observed = str(exc)
        cases.append({"case_id": case_id, "passed": expected == observed, "expected": expected, "observed": observed})

    row = copy.deepcopy(base); row["authority_delta"] = {"network": "expanded"}; rejected("authority_expansion", row, "authority_expansion_forbidden")
    row = copy.deepcopy(base); row["feedback_receipt"]["class"] = "public_benchmark"; rejected("public_feedback", row, "feedback_not_admissible")
    row = copy.deepcopy(base); row["heldout_contract"] = "wrong"; rejected("heldout_substitution", row, "heldout_contract_mismatch")
    row = copy.deepcopy(base); row["baseline_digest"] = "sha256:stale"; rejected("stale_baseline", row, "baseline_identity_invalid")
    row = copy.deepcopy(base); row["cost_budget"].pop("human_cleanup"); rejected("hidden_cleanup_cost", row, "cost_budget_incomplete")
    active = issue_lease(base, contract); rejected("overlapping_target", base, "target_lease_conflict", [active])
    cross_write = False
    try:
        apply_delta(active, {"target_id": "router", "state_path": "router.policy", "before_digest": active["current_digest"], "next_state": {}, "cost_observed": active["cost_observed"]}, contract)
    except PolicyUpdateLeaseFault as exc:
        cross_write = str(exc) == "cross_target_write_forbidden"
    cases.append({"case_id": "cross_target_write", "passed": cross_write, "expected": "cross_target_write_forbidden", "observed": "cross_target_write_forbidden" if cross_write else "accepted"})
    return {"case_count": len(cases), "passed_count": sum(bool(row["passed"]) for row in cases), "results": cases}

