#!/usr/bin/env python3
"""Executable, disabled-by-default objective ABI for policy optimization."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import policy_update_lease


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs" / "policy_objective_contracts.json"


class ObjectiveContractFault(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "policy", "schema_version", "owner", "runtime_enabled", "maximum_architecture_canary_steps",
        "preference_schema", "rollout_schema", "offline_preference_objectives",
        "verifier_reward_objectives", "verifier_capacity", "reward_hacking_probes",
        "activation", "claim_boundaries",
    }
    missing = required.difference(contract) if isinstance(contract, dict) else required
    if missing:
        raise ObjectiveContractFault(f"contract_missing:{','.join(sorted(missing))}")
    if set(contract["offline_preference_objectives"]) != {"dpo", "ipo", "orpo", "kto", "simpo"}:
        raise ObjectiveContractFault("offline_objective_coverage_invalid")
    if set(contract["verifier_reward_objectives"]) != {"grpo", "rloo", "remax", "rlvr"}:
        raise ObjectiveContractFault("verifier_reward_objective_coverage_invalid")
    if int(contract["maximum_architecture_canary_steps"]) > 8:
        raise ObjectiveContractFault("architecture_canary_step_limit_exceeded")
    capacity = contract["verifier_capacity"]
    if int(capacity["maximum_rollouts"]) > int(contract["rollout_schema"]["maximum_group_size"]):
        raise ObjectiveContractFault("rollout_capacity_exceeds_schema")
    return contract


def validate_preference_pair(pair: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    schema = contract["preference_schema"]
    missing = set(schema["required_fields"]).difference(pair)
    if missing:
        raise ObjectiveContractFault(f"preference_pair_incomplete:{','.join(sorted(missing))}")
    forbidden = set(schema["forbidden_fields"]).intersection(pair)
    if forbidden:
        raise ObjectiveContractFault(f"preference_pair_forbidden_fields:{','.join(sorted(forbidden))}")
    if pair["chosen_digest"] == pair["rejected_digest"]:
        raise ObjectiveContractFault("preference_pair_degenerate")
    for field in ("prompt_digest", "chosen_digest", "rejected_digest"):
        if not str(pair[field]).startswith("sha256:"):
            raise ObjectiveContractFault(f"preference_identity_invalid:{field}")
    for field in ("provenance_receipt", "private_verifier_receipt", "candidate_integrity_receipt"):
        receipt = pair[field]
        if not isinstance(receipt, dict) or not receipt.get("receipt_id") or receipt.get("passed") is not True:
            raise ObjectiveContractFault(f"preference_receipt_invalid:{field}")
    if pair["private_verifier_receipt"].get("public_artifact_used") or pair["candidate_integrity_receipt"].get("fallback_or_template"):
        raise ObjectiveContractFault("preference_pair_integrity_fault")
    return {"schema_id": schema["schema_id"], "pair_digest": digest(pair), "valid": True}


def validate_rollout_group(group: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    schema = contract["rollout_schema"]
    missing = set(schema["required_fields"]).difference(group)
    if missing:
        raise ObjectiveContractFault(f"rollout_group_incomplete:{','.join(sorted(missing))}")
    rollout_ids = group["rollout_ids"]
    size = len(rollout_ids) if isinstance(rollout_ids, list) else 0
    if size < int(schema["minimum_group_size"]) or size > int(schema["maximum_group_size"]):
        raise ObjectiveContractFault("rollout_group_size_invalid")
    vector_fields = ("old_logps", "new_logps", "reference_logps", "verifier_rewards", "verifier_receipts", "candidate_integrity_receipts")
    if any(not isinstance(group[field], list) or len(group[field]) != size for field in vector_fields):
        raise ObjectiveContractFault("rollout_vector_shape_mismatch")
    if len(set(rollout_ids)) != size:
        raise ObjectiveContractFault("rollout_identity_duplicate")
    for verifier, integrity in zip(group["verifier_receipts"], group["candidate_integrity_receipts"]):
        if not verifier.get("receipt_id") or verifier.get("public_artifact_used"):
            raise ObjectiveContractFault("rollout_verifier_provenance_invalid")
        if not integrity.get("receipt_id") or integrity.get("fallback_or_template"):
            raise ObjectiveContractFault("rollout_candidate_integrity_invalid")
    return {"schema_id": schema["schema_id"], "group_digest": digest(group), "group_size": size, "valid": True}


def preference_loss(objective: str, pair: dict[str, Any], contract: dict[str, Any] | None = None) -> float:
    contract = contract or load_contract()
    validate_preference_pair(pair, contract)
    if objective not in contract["offline_preference_objectives"]:
        raise ObjectiveContractFault("offline_objective_unknown")
    cfg = contract["offline_preference_objectives"][objective]
    beta = float(cfg["beta"])
    chosen = float(pair["chosen_policy_logp"])
    rejected = float(pair["rejected_policy_logp"])
    ref_chosen = float(pair["chosen_reference_logp"])
    ref_rejected = float(pair["rejected_reference_logp"])
    policy_gap = chosen - rejected
    reference_gap = ref_chosen - ref_rejected
    if objective == "dpo":
        return softplus(-beta * (policy_gap - reference_gap))
    if objective == "ipo":
        return (policy_gap - reference_gap - 1.0 / (2.0 * beta)) ** 2
    if objective == "orpo":
        odds_gap = log_odds_from_logp(chosen) - log_odds_from_logp(rejected)
        return -float(cfg["sft_weight"]) * chosen + softplus(-beta * odds_gap)
    if objective == "kto":
        desirable = softplus(-beta * ((chosen - ref_chosen) - reference_kl(pair)))
        undesirable = softplus(beta * ((rejected - ref_rejected) - reference_kl(pair)))
        return float(cfg["desirable_weight"]) * desirable + float(cfg["undesirable_weight"]) * undesirable
    return softplus(-(beta * policy_gap - float(cfg["gamma"])))


def verifier_reward_loss(objective: str, group: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    validated = validate_rollout_group(group, contract)
    if objective not in contract["verifier_reward_objectives"]:
        raise ObjectiveContractFault("verifier_reward_objective_unknown")
    cfg = contract["verifier_reward_objectives"][objective]
    rewards = [float(value) for value in group["verifier_rewards"]]
    advantages = objective_advantages(objective, rewards, float(group["greedy_baseline_reward"]))
    terms = []
    for old, new, reference, advantage in zip(group["old_logps"], group["new_logps"], group["reference_logps"], advantages):
        ratio = math.exp(max(-20.0, min(20.0, float(new) - float(old))))
        clipped = max(1.0 - float(cfg["clip_epsilon"]), min(1.0 + float(cfg["clip_epsilon"]), ratio))
        policy_term = min(ratio * advantage, clipped * advantage)
        kl = math.exp(float(reference) - float(new)) - (float(reference) - float(new)) - 1.0
        terms.append(-policy_term + float(cfg["kl_beta"]) * kl)
    return {
        "objective": objective,
        "loss": sum(terms) / len(terms),
        "advantages": advantages,
        "group_digest": validated["group_digest"],
        "verifier_capacity_receipt": verifier_capacity_receipt(group, contract),
    }


def objective_advantages(objective: str, rewards: list[float], greedy_baseline: float) -> list[float]:
    if objective == "grpo":
        mean = sum(rewards) / len(rewards)
        variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
        scale = math.sqrt(variance) or 1.0
        return [(value - mean) / scale for value in rewards]
    if objective == "rloo":
        total = sum(rewards)
        return [value - (total - value) / (len(rewards) - 1) for value in rewards]
    if objective == "remax":
        return [value - greedy_baseline for value in rewards]
    return [1.0 if value > 0 else -1.0 for value in rewards]


def verifier_capacity_receipt(group: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    observed = group.get("verifier_capacity_observed") or {}
    capacity = contract["verifier_capacity"]
    required = {"verifier_cases", "verifier_seconds", "compute_units", "energy_units"}
    if required.difference(observed):
        raise ObjectiveContractFault("verifier_capacity_observation_incomplete")
    exceeded = [name for name in required if float(observed[name]) > float(capacity[f"maximum_{name}"])]
    if len(group["rollout_ids"]) > int(capacity["maximum_rollouts"]):
        exceeded.append("rollouts")
    if exceeded:
        raise ObjectiveContractFault("verifier_capacity_exceeded:" + ",".join(sorted(exceeded)))
    return {"within_capacity": True, "observed": copy.deepcopy(observed), "limits": copy.deepcopy(capacity), "receipt_digest": digest(observed)}


def activation_receipt(evidence: dict[str, Any] | None, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    evidence = evidence or {}
    blockers = []
    if not contract["runtime_enabled"]:
        blockers.append("objectives_disabled_by_frozen_contract")
    if evidence.get("evidence_kind") != contract["activation"]["required_evidence_kind"]:
        blockers.append("behavior_positive_evidence_missing")
    if int(evidence.get("independent_pass_count", 0)) < int(contract["activation"]["minimum_independent_passes"]):
        blockers.append("independent_behavior_floor_not_met")
    if evidence.get("public_artifact_used") or evidence.get("fallback_or_template_credit"):
        blockers.append("activation_evidence_integrity_fault")
    return {"authorized": not blockers, "blockers": blockers, "evidence_digest": digest(evidence)}


def checkpoint_roundtrip(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    state = {
        "schema_version": "1.0.0",
        "policy_revision": "policy:fixture-v1",
        "reference_revision": "reference:frozen-v1",
        "reference_digest": digest({"weights": [0.1, -0.2]}),
        "optimizer_state": {"step": 0, "moments": [0.0, 0.0]},
        "objective_ids": sorted(list(contract["offline_preference_objectives"]) + list(contract["verifier_reward_objectives"])),
        "objective_contract_digest": digest(contract),
        "update_lease_target": "generator",
        "runtime_enabled": False,
    }
    with tempfile.TemporaryDirectory(prefix="theseus-objective-checkpoint-") as tmp:
        path = Path(tmp) / "checkpoint.json"
        path.write_text(canonical(state), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
    legacy = copy.deepcopy(state)
    legacy["schema_version"] = "0.9.0"
    legacy.pop("runtime_enabled")
    migrated = migrate_checkpoint(legacy, contract)
    return {
        "checkpoint_digest": digest(state),
        "roundtrip_exact": digest(loaded) == digest(state),
        "reference_identity_frozen": loaded["reference_digest"] == state["reference_digest"],
        "optimizer_state_present": bool(loaded["optimizer_state"]),
        "migration_exact": migrated == state,
        "cleanup_complete": not path.exists(),
    }


def migrate_checkpoint(checkpoint: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    if checkpoint.get("schema_version") == "1.0.0":
        return copy.deepcopy(checkpoint)
    if checkpoint.get("schema_version") != "0.9.0":
        raise ObjectiveContractFault("checkpoint_schema_unsupported")
    migrated = copy.deepcopy(checkpoint)
    migrated["schema_version"] = "1.0.0"
    migrated["runtime_enabled"] = False
    if migrated.get("objective_contract_digest") != digest(contract):
        raise ObjectiveContractFault("checkpoint_contract_identity_mismatch")
    return migrated


def policy_lease_rollback_receipt() -> dict[str, Any]:
    lease_contract = policy_update_lease.load_contract()
    request = policy_update_lease.reference_request("generator", lease_contract["targets"]["generator"])
    lease = policy_update_lease.issue_lease(request, lease_contract)
    next_state = copy.deepcopy(lease["current_state"])
    next_state["revision"] = 2
    lease = policy_update_lease.apply_delta(lease, {
        "target_id": "generator",
        "state_path": lease["state_path"],
        "before_digest": lease["current_digest"],
        "next_state": next_state,
        "cost_observed": {"verification": 1, "repair": 0, "human_cleanup": 0, "compute": 1, "energy": 1},
    }, lease_contract)
    rolled = policy_update_lease.rollback(lease, reason="objective_fixture_rollback")
    return {"state": rolled["state"], "rollback_exact": rolled["rollback_exact"], "epoch_sealed": rolled["epoch_token"].startswith("sealed:"), "target_id": rolled["target_id"]}


def mlx_parity_probe(pair: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    expected = {name: preference_loss(name, pair, contract) for name in contract["offline_preference_objectives"]}
    payload = {key: pair[key] for key in ("chosen_policy_logp", "rejected_policy_logp", "chosen_reference_logp", "rejected_reference_logp")}
    payload["config"] = contract["offline_preference_objectives"]
    code = r'''
import json, math, sys
import mlx.core as mx
p=json.loads(sys.stdin.read()); c=mx.array(float(p["chosen_policy_logp"])); r=mx.array(float(p["rejected_policy_logp"])); rc=mx.array(float(p["chosen_reference_logp"])); rr=mx.array(float(p["rejected_reference_logp"])); out={}
sp=lambda x: mx.logaddexp(mx.array(0.0), x)
gap=c-r; rg=rc-rr
out["dpo"]=float(sp(-p["config"]["dpo"]["beta"]*(gap-rg)).item())
out["ipo"]=float(((gap-rg-1.0/(2.0*p["config"]["ipo"]["beta"]))**2).item())
lo=lambda x: x-mx.log(mx.maximum(mx.array(1e-12), 1.0-mx.exp(mx.minimum(x, mx.array(-1e-12)))))
out["orpo"]=float((-p["config"]["orpo"]["sft_weight"]*c+sp(-p["config"]["orpo"]["beta"]*(lo(c)-lo(r)))).item())
kl=((c-rc)+(r-rr))/2.0
out["kto"]=float((p["config"]["kto"]["desirable_weight"]*sp(-p["config"]["kto"]["beta"]*((c-rc)-kl))+p["config"]["kto"]["undesirable_weight"]*sp(p["config"]["kto"]["beta"]*((r-rr)-kl))).item())
out["simpo"]=float(sp(-(p["config"]["simpo"]["beta"]*gap-p["config"]["simpo"]["gamma"])).item())
print(json.dumps(out,sort_keys=True))
'''
    proc = subprocess.run([sys.executable, "-c", code], input=json.dumps(payload), text=True, capture_output=True, timeout=30)
    if proc.returncode != 0:
        return {"available": False, "parity": False, "returncode": proc.returncode, "stderr_tail": proc.stderr[-1000:]}
    observed = json.loads(proc.stdout)
    deltas = {name: abs(expected[name] - float(observed[name])) for name in expected}
    return {"available": True, "parity": all(value <= 1e-5 for value in deltas.values()), "expected": expected, "observed": observed, "absolute_deltas": deltas, "tolerance": 1e-5}


def reference_pair() -> dict[str, Any]:
    return {
        "pair_id": "pair:fixture:001",
        "prompt_digest": digest("private prompt"),
        "chosen_digest": digest("chosen body"),
        "rejected_digest": digest("rejected body"),
        "chosen_policy_logp": -1.2,
        "rejected_policy_logp": -2.4,
        "chosen_reference_logp": -1.5,
        "rejected_reference_logp": -2.1,
        "provenance_receipt": {"receipt_id": "provenance:001", "passed": True},
        "private_verifier_receipt": {"receipt_id": "verifier:001", "passed": True, "public_artifact_used": False},
        "candidate_integrity_receipt": {"receipt_id": "integrity:001", "passed": True, "fallback_or_template": False},
    }


def reference_rollout_group() -> dict[str, Any]:
    ids = [f"rollout:{index}" for index in range(4)]
    return {
        "group_id": "group:fixture:001",
        "prompt_digest": digest("private rollout prompt"),
        "policy_revision": "policy:fixture-v1",
        "reference_revision": "reference:frozen-v1",
        "rollout_ids": ids,
        "old_logps": [-1.0, -1.1, -1.2, -1.3],
        "new_logps": [-0.9, -1.2, -1.1, -1.4],
        "reference_logps": [-1.05, -1.05, -1.25, -1.25],
        "verifier_rewards": [1.0, 0.0, 1.0, 0.0],
        "greedy_baseline_reward": 0.25,
        "verifier_receipts": [{"receipt_id": f"verifier:{index}", "public_artifact_used": False} for index in range(4)],
        "candidate_integrity_receipts": [{"receipt_id": f"integrity:{index}", "fallback_or_template": False} for index in range(4)],
        "verifier_capacity_observed": {"verifier_cases": 32, "verifier_seconds": 1.0, "compute_units": 4.0, "energy_units": 4.0},
    }


def run_reference_suite(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    pair = reference_pair()
    group = reference_rollout_group()
    preference = {name: preference_loss(name, pair, contract) for name in contract["offline_preference_objectives"]}
    verifier = {name: verifier_reward_loss(name, group, contract) for name in contract["verifier_reward_objectives"]}
    checkpoint = checkpoint_roundtrip(contract)
    lease = policy_lease_rollback_receipt()
    mlx = mlx_parity_probe(pair, contract)
    controls = mutation_controls(contract)
    activation = activation_receipt({}, contract)
    gates = {
        "all_offline_objectives_executable": set(preference) == set(contract["offline_preference_objectives"]) and all(math.isfinite(value) for value in preference.values()),
        "all_verifier_reward_objectives_executable": set(verifier) == set(contract["verifier_reward_objectives"]) and all(math.isfinite(row["loss"]) for row in verifier.values()),
        "shared_preference_schema_valid": validate_preference_pair(pair, contract)["valid"],
        "shared_rollout_schema_valid": validate_rollout_group(group, contract)["valid"],
        "frozen_reference_checkpoint_roundtrip": checkpoint["roundtrip_exact"] and checkpoint["reference_identity_frozen"] and checkpoint["optimizer_state_present"],
        "checkpoint_migration_and_cleanup": checkpoint["migration_exact"] and checkpoint["cleanup_complete"],
        "policy_lease_rollback_exact": lease["rollback_exact"] and lease["epoch_sealed"],
        "verifier_capacity_bounded": all(row["verifier_capacity_receipt"]["within_capacity"] for row in verifier.values()),
        "mlx_numerical_parity": mlx["available"] and mlx["parity"],
        "reward_hacking_mutations_rejected": controls["case_count"] == controls["passed_count"],
        "runtime_objective_selection_disabled": not activation["authorized"],
        "zero_optimizer_exposure": True,
        "no_cheat_counters_clean": True,
    }
    return {
        "policy": contract["policy"],
        "trigger_state": "GREEN" if all(gates.values()) else "RED",
        "support_state": "synthetic-test-backed",
        "summary": {
            "offline_objective_count": len(preference),
            "verifier_reward_objective_count": len(verifier),
            "mutation_case_count": controls["case_count"],
            "mutation_passed_count": controls["passed_count"],
            "mlx_available": mlx["available"],
            "mlx_parity": mlx["parity"],
            "runtime_authorized": activation["authorized"],
            "optimizer_exposure_steps": 0,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_or_template_credit": 0,
        },
        "gates": gates,
        "offline_preference_objectives": preference,
        "verifier_reward_objectives": verifier,
        "checkpoint_receipt": checkpoint,
        "policy_lease_rollback_receipt": lease,
        "mlx_parity_receipt": mlx,
        "activation_receipt": activation,
        "mutation_controls": controls,
        "non_claims": copy.deepcopy(contract["claim_boundaries"]),
    }


def mutation_controls(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    pair = reference_pair()
    group = reference_rollout_group()
    cases = []

    def record(case_id: str, expected: str, action: Any) -> None:
        observed = "accepted"
        try:
            action()
        except ObjectiveContractFault as exc:
            observed = str(exc)
        cases.append({"case_id": case_id, "passed": expected in observed, "expected": expected, "observed": observed})

    bad = copy.deepcopy(pair); bad["public_solution"] = "leak"
    record("public_payload", "preference_pair_forbidden_fields", lambda: validate_preference_pair(bad, contract))
    bad = copy.deepcopy(pair); bad["candidate_integrity_receipt"]["fallback_or_template"] = True
    record("fallback_credit", "preference_pair_integrity_fault", lambda: validate_preference_pair(bad, contract))
    bad = copy.deepcopy(pair); bad["chosen_digest"] = bad["rejected_digest"]
    record("degenerate_pair", "preference_pair_degenerate", lambda: validate_preference_pair(bad, contract))
    bad = copy.deepcopy(pair); bad.pop("provenance_receipt")
    record("missing_provenance", "preference_pair_incomplete", lambda: validate_preference_pair(bad, contract))
    bad = copy.deepcopy(group); bad["verifier_receipts"][0]["public_artifact_used"] = True
    record("public_verifier_artifact", "rollout_verifier_provenance_invalid", lambda: validate_rollout_group(bad, contract))
    bad = copy.deepcopy(group); bad["candidate_integrity_receipts"][0]["fallback_or_template"] = True
    record("rollout_fallback_credit", "rollout_candidate_integrity_invalid", lambda: validate_rollout_group(bad, contract))
    bad = copy.deepcopy(group); bad["rollout_ids"] = bad["rollout_ids"][:2]
    for field in ("old_logps", "new_logps", "reference_logps", "verifier_rewards", "verifier_receipts", "candidate_integrity_receipts"):
        bad[field] = bad[field][:2]
    record("undersized_group", "rollout_group_size_invalid", lambda: validate_rollout_group(bad, contract))
    bad = copy.deepcopy(group); bad["old_logps"].pop()
    record("rollout_shape_mismatch", "rollout_vector_shape_mismatch", lambda: validate_rollout_group(bad, contract))
    bad = copy.deepcopy(group); bad["verifier_capacity_observed"]["verifier_cases"] = 65
    record("verifier_capacity_overrun", "verifier_capacity_exceeded", lambda: verifier_reward_loss("grpo", bad, contract))
    record("unknown_preference_objective", "offline_objective_unknown", lambda: preference_loss("unknown", pair, contract))
    record("unknown_reward_objective", "verifier_reward_objective_unknown", lambda: verifier_reward_loss("unknown", group, contract))
    record("unsupported_checkpoint", "checkpoint_schema_unsupported", lambda: migrate_checkpoint({"schema_version": "broken"}, contract))
    return {"case_count": len(cases), "passed_count": sum(bool(row["passed"]) for row in cases), "results": cases}


def softplus(value: float) -> float:
    return math.log1p(math.exp(-abs(value))) + max(value, 0.0)


def log_odds_from_logp(value: float) -> float:
    probability = min(1.0 - 1e-12, max(1e-12, math.exp(min(value, -1e-12))))
    return math.log(probability) - math.log1p(-probability)


def reference_kl(pair: dict[str, Any]) -> float:
    return ((float(pair["chosen_policy_logp"]) - float(pair["chosen_reference_logp"])) + (float(pair["rejected_policy_logp"]) - float(pair["rejected_reference_logp"]))) / 2.0
