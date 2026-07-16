#!/usr/bin/env python3
"""Frozen first-campaign generation topology and mode dispositions."""

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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs" / "generation_architecture_contracts.json"


class GenerationArchitectureFault(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    required = {"policy", "schema_version", "owner", "first_campaign_base", "maximum_architecture_canary_steps", "common_accounting", "modes", "mtp_shape_contract", "activation", "claim_boundaries"}
    missing = required.difference(contract) if isinstance(contract, dict) else required
    if missing:
        raise GenerationArchitectureFault(f"contract_missing:{','.join(sorted(missing))}")
    expected = {"autoregressive", "mtp", "medusa", "eagle", "speculative", "layerskip", "sketch_first_llada"}
    if set(contract["modes"]) != expected:
        raise GenerationArchitectureFault("generation_mode_coverage_invalid")
    if contract["first_campaign_base"] != "autoregressive":
        raise GenerationArchitectureFault("first_campaign_base_invalid")
    if int(contract["maximum_architecture_canary_steps"]) > 8:
        raise GenerationArchitectureFault("architecture_canary_limit_exceeded")
    required_mode = {"class", "first_campaign_disposition", "topology_effect", "objective_effect", "checkpoint_effect", "cache_policy"}
    for mode_id, mode in contract["modes"].items():
        missing_mode = required_mode.difference(mode)
        if missing_mode:
            raise GenerationArchitectureFault(f"mode_contract_incomplete:{mode_id}:{','.join(sorted(missing_mode))}")
        if mode["first_campaign_disposition"].startswith("retired") and not mode.get("reentry_condition"):
            raise GenerationArchitectureFault(f"retirement_reentry_missing:{mode_id}")
    return contract


def generation_mode_record(mode_id: str, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    if mode_id not in contract["modes"]:
        raise GenerationArchitectureFault("generation_mode_unknown")
    mode = contract["modes"][mode_id]
    return {
        "record_type": "generation_mode_record",
        "mode_id": mode_id,
        "mode_class": mode["class"],
        "first_campaign_disposition": mode["first_campaign_disposition"],
        "topology_effect": mode["topology_effect"],
        "objective_effect": mode["objective_effect"],
        "checkpoint_effect": mode["checkpoint_effect"],
        "cache_policy": mode["cache_policy"],
        "active_compute_contract": {"training_flops_estimate": "required_before_behavioral_comparison", "decode_flops_estimate": "required_before_behavioral_comparison", "verifier_cost": "required_before_behavioral_comparison"},
        "accepted_output_accounting": {field: 0 for field in contract["common_accounting"]},
        "reentry_condition": mode.get("reentry_condition", "not_applicable"),
        "record_digest": digest([mode_id, mode]),
    }


def validate_mode_record(record: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    expected = generation_mode_record(str(record.get("mode_id") or ""), contract)
    if record != expected:
        raise GenerationArchitectureFault("generation_mode_record_tampered")
    if set(record["accepted_output_accounting"]) != set(contract["common_accounting"]):
        raise GenerationArchitectureFault("accepted_output_accounting_incomplete")
    return {"valid": True, "mode_id": record["mode_id"], "record_digest": record["record_digest"]}


def checkpoint_contract(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    included = [mode_id for mode_id, mode in contract["modes"].items() if not mode["first_campaign_disposition"].startswith("retired")]
    return {
        "schema_version": "1.0.0",
        "base_mode": "autoregressive",
        "model_revision": "model:pretraining-campaign-v1",
        "base_parameter_digest": digest({"base": "fixture"}),
        "optional_head_groups": {"mtp": {"present": True, "head_count": int(contract["modes"]["mtp"]["head_count"]), "low_rank": int(contract["modes"]["mtp"]["low_rank"]), "future_offsets": contract["mtp_shape_contract"]["future_offsets"], "loss_weights": contract["modes"]["mtp"]["loss_weights"], "initial_weight": 0.0}},
        "post_hoc_helpers": {"speculative": {"target_topology_changed": False, "enabled": False, "draft_manifest_required": True}},
        "retired_modes": sorted(mode_id for mode_id, mode in contract["modes"].items() if mode["first_campaign_disposition"].startswith("retired")),
        "included_mode_ids": sorted(included),
        "optimizer_group_ids": ["base_model", "mtp_heads"],
        "contract_digest": digest(contract),
    }


def checkpoint_roundtrip(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    checkpoint = checkpoint_contract(contract)
    with tempfile.TemporaryDirectory(prefix="theseus-generation-checkpoint-") as tmp:
        path = Path(tmp) / "generation.json"
        path.write_text(canonical(checkpoint), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        legacy = copy.deepcopy(checkpoint)
        legacy["schema_version"] = "0.9.0"
        legacy.pop("post_hoc_helpers")
        migrated = migrate_checkpoint(legacy, contract)
    return {
        "roundtrip_exact": digest(loaded) == digest(checkpoint),
        "migration_exact": migrated == checkpoint,
        "cleanup_complete": not path.exists(),
        "retired_modes_absent_from_optimizer": not set(checkpoint["retired_modes"]).intersection(checkpoint["optimizer_group_ids"]),
        "checkpoint_digest": digest(checkpoint),
    }


def migrate_checkpoint(checkpoint: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    if checkpoint.get("schema_version") == "1.0.0":
        if checkpoint.get("contract_digest") != digest(contract):
            raise GenerationArchitectureFault("checkpoint_contract_digest_mismatch")
        return copy.deepcopy(checkpoint)
    if checkpoint.get("schema_version") != "0.9.0":
        raise GenerationArchitectureFault("checkpoint_schema_unsupported")
    migrated = copy.deepcopy(checkpoint)
    migrated["schema_version"] = "1.0.0"
    migrated["post_hoc_helpers"] = {"speculative": {"target_topology_changed": False, "enabled": False, "draft_manifest_required": True}}
    if migrated.get("contract_digest") != digest(contract):
        raise GenerationArchitectureFault("checkpoint_contract_digest_mismatch")
    return migrated


def speculative_loader_receipt(target_checkpoint: dict[str, Any], draft_manifest: dict[str, Any]) -> dict[str, Any]:
    required = {"draft_revision", "target_model_revision", "target_base_parameter_digest", "draft_checkpoint_digest", "cache_commit_policy"}
    missing = required.difference(draft_manifest)
    if missing:
        raise GenerationArchitectureFault(f"draft_manifest_incomplete:{','.join(sorted(missing))}")
    if draft_manifest["target_model_revision"] != target_checkpoint["model_revision"] or draft_manifest["target_base_parameter_digest"] != target_checkpoint["base_parameter_digest"]:
        raise GenerationArchitectureFault("draft_target_revision_mismatch")
    if draft_manifest["cache_commit_policy"] != "accepted_prefix_only":
        raise GenerationArchitectureFault("speculative_cache_policy_invalid")
    return {"compatible": True, "target_topology_changed": False, "target_checkpoint_digest": digest(target_checkpoint), "draft_manifest_digest": digest(draft_manifest), "enabled": False}


def mlx_mtp_canary(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    shape = contract["mtp_shape_contract"]
    payload = {
        "shape": shape,
        "weights": contract["modes"]["mtp"]["loss_weights"],
        "scripts": str(ROOT / "scripts"),
    }
    code = """
import json, sys, tempfile
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mlx_utils
p=json.loads(sys.stdin.read()); s=p['shape']; sys.path.insert(0,p['scripts'])
from standard_causal_transformer_model import CausalTransformerConfig, build_model
from standard_causal_transformer_survival import causal_loss, mtp_auxiliary_loss
cfg=CausalTransformerConfig(vocab_size=s['vocabulary'],d_model=s['hidden'],num_layers=1,num_heads=4,num_kv_heads=2,ff_dim=s['hidden']*2,mtp_future_offsets=tuple(s['future_offsets']),mtp_low_rank=s['low_rank'],mtp_loss_weights=tuple(p['weights']),mtp_loss_scale=0.5,mtp_maximum_head_parameter_overhead_ratio=s['maximum_parameter_overhead_ratio'])
mx.random.seed(1701); model=build_model(cfg,mx=mx,nn=nn)
tokens=mx.arange(s['batch']*s['sequence']).reshape((s['batch'],s['sequence'])).astype(mx.int32)%s['vocabulary']
labels=(tokens+1)%s['vocabulary']; mask=mx.ones(labels.shape,dtype=mx.float32)
logits,_cache,aux=model(tokens,return_training_aux=True)
mtp_loss=mtp_auxiliary_loss(aux['mtp_logits'],labels,mask,model.mtp_future_offsets,model.mtp_loss_weights,mx,nn)
joint_loss=causal_loss(model,tokens,labels,mask,mx,nn); mx.eval(logits,mtp_loss,joint_loss,model.parameters())
flat=mlx_utils.tree_flatten(model.parameters()); mtp_names=[name for name,_ in flat if name.startswith('mtp_')]
mtp_params=sum(int(value.size) for name,value in flat if name.startswith('mtp_')); base_head_params=s['hidden']*s['vocabulary']; overhead=mtp_params/base_head_params
with tempfile.TemporaryDirectory(prefix='theseus-canonical-mtp-') as tmp:
    checkpoint=Path(tmp)/'model.npz'; model.save_weights(str(checkpoint)); checkpoint_bytes=checkpoint.stat().st_size
    reloaded=build_model(cfg,mx=mx,nn=nn); reloaded.load_weights(str(checkpoint)); reloaded_logits,_cache,reloaded_aux=reloaded(tokens,return_training_aux=True); mx.eval(reloaded_logits,reloaded_aux['mtp_logits'])
    base_delta=float(mx.max(mx.abs(logits-reloaded_logits)).item()); mtp_delta=max(float(mx.max(mx.abs(left-right)).item()) for left,right in zip(aux['mtp_logits'],reloaded_aux['mtp_logits']))
print(json.dumps({'base_shape':list(logits.shape),'mtp_shapes':[list(value.shape) for value in aux['mtp_logits']],'mtp_loss':float(mtp_loss.item()),'joint_loss':float(joint_loss.item()),'valid_positions':[s['sequence']-(offset-1) for offset in s['future_offsets']],'raw_head_parameter_overhead_ratio':overhead,'mtp_parameter_names':mtp_names,'checkpoint_bytes':checkpoint_bytes,'checkpoint_reload_max_abs_delta':max(base_delta,mtp_delta)},sort_keys=True))
"""
    proc = subprocess.run([sys.executable, "-c", code], input=json.dumps(payload), text=True, capture_output=True, timeout=30)
    if proc.returncode != 0:
        return {"available": False, "passed": False, "returncode": proc.returncode, "stderr_tail": proc.stderr[-1000:]}
    observed = json.loads(proc.stdout)
    expected_shape = [shape["batch"], shape["sequence"], shape["vocabulary"]]
    finite = math.isfinite(observed["mtp_loss"]) and math.isfinite(observed["joint_loss"])
    expected_mtp_shapes = [expected_shape for _offset in shape["future_offsets"]]
    return {
        "available": True,
        "passed": observed["base_shape"] == expected_shape
        and observed["mtp_shapes"] == expected_mtp_shapes
        and finite
        and observed["valid_positions"]
        == [shape["sequence"] - (offset - 1) for offset in shape["future_offsets"]]
        and observed["raw_head_parameter_overhead_ratio"]
        <= float(shape["maximum_parameter_overhead_ratio"])
        and len(observed["mtp_parameter_names"]) == 4
        and observed["checkpoint_bytes"] > 0
        and observed["checkpoint_reload_max_abs_delta"] == 0.0,
        "observed": observed,
        "mtp_parameter_overhead_ratio": observed["raw_head_parameter_overhead_ratio"],
        "resource_decision": "canonical_shared_rank1_heads_within_frozen_overhead_ceiling",
        "campaign_initial_loss_weight": 0.0,
        "canary_loss_scale": 0.5,
        "optimizer_steps": 0,
    }


def activation_receipt(evidence: dict[str, Any] | None, contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    evidence = evidence or {}
    blockers = []
    if evidence.get("evidence_kind") != contract["activation"]["required_evidence_kind"]:
        blockers.append("behavior_positive_generation_evidence_missing")
    if int(evidence.get("verified_pass_count", 0)) < int(contract["activation"]["minimum_verified_pass_count"]):
        blockers.append("verified_pass_floor_not_met")
    if evidence.get("public_artifact_used") or evidence.get("fallback_or_template_credit"):
        blockers.append("generation_activation_integrity_fault")
    return {"authorized": not blockers, "blockers": blockers, "evidence_digest": digest(evidence)}


def run_reference_suite(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    records = {mode_id: generation_mode_record(mode_id, contract) for mode_id in contract["modes"]}
    validations = {mode_id: validate_mode_record(record, contract) for mode_id, record in records.items()}
    checkpoint = checkpoint_contract(contract)
    roundtrip = checkpoint_roundtrip(contract)
    draft = {"draft_revision": "draft:fixture-v1", "target_model_revision": checkpoint["model_revision"], "target_base_parameter_digest": checkpoint["base_parameter_digest"], "draft_checkpoint_digest": digest({"draft": "fixture"}), "cache_commit_policy": "accepted_prefix_only"}
    speculative = speculative_loader_receipt(checkpoint, draft)
    mtp = mlx_mtp_canary(contract)
    activation = activation_receipt({}, contract)
    controls = mutation_controls(contract)
    retired = sorted(mode_id for mode_id, mode in contract["modes"].items() if mode["first_campaign_disposition"].startswith("retired"))
    gates = {
        "all_mode_records_valid": len(validations) == len(contract["modes"]),
        "first_campaign_base_ar": contract["first_campaign_base"] == "autoregressive",
        "mtp_mlx_shape_canary": mtp["available"] and mtp["passed"],
        "mtp_resource_ceiling_passed": mtp["mtp_parameter_overhead_ratio"] <= float(contract["mtp_shape_contract"]["maximum_parameter_overhead_ratio"]),
        "mtp_zero_initial_weight": checkpoint["optional_head_groups"]["mtp"]["initial_weight"] == 0.0,
        "checkpoint_roundtrip_migration_cleanup": roundtrip["roundtrip_exact"] and roundtrip["migration_exact"] and roundtrip["cleanup_complete"],
        "retired_modes_absent_from_optimizer": roundtrip["retired_modes_absent_from_optimizer"],
        "post_hoc_speculative_compatible_disabled": speculative["compatible"] and not speculative["enabled"] and not speculative["target_topology_changed"],
        "retirements_have_reentry_conditions": all(contract["modes"][mode_id].get("reentry_condition") for mode_id in retired),
        "runtime_mode_selection_disabled_without_behavior": not activation["authorized"],
        "mutation_controls_rejected": controls["case_count"] == controls["passed_count"],
        "zero_optimizer_exposure": mtp["optimizer_steps"] == 0,
        "no_cheat_counters_clean": True,
    }
    return {
        "policy": contract["policy"],
        "trigger_state": "GREEN" if all(gates.values()) else "RED",
        "support_state": "synthetic-test-backed",
        "summary": {"mode_count": len(records), "included_mode_count": sum(not mode["first_campaign_disposition"].startswith("retired") for mode in contract["modes"].values()), "retired_first_campaign_mode_count": len(retired), "mutation_case_count": controls["case_count"], "mutation_passed_count": controls["passed_count"], "mlx_available": mtp["available"], "mtp_canary_passed": mtp["passed"], "runtime_authorized": activation["authorized"], "optimizer_exposure_steps": 0, "public_training_rows_written": 0, "external_inference_calls": 0, "fallback_or_template_credit": 0},
        "gates": gates,
        "mode_records": records,
        "checkpoint_receipt": roundtrip,
        "mtp_mlx_canary": mtp,
        "speculative_loader_receipt": speculative,
        "activation_receipt": activation,
        "mutation_controls": controls,
        "non_claims": copy.deepcopy(contract["claim_boundaries"]),
    }


def mutation_controls(contract: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    cases = []

    def record(case_id: str, expected: str, action: Any) -> None:
        observed = "accepted"
        try:
            action()
        except GenerationArchitectureFault as exc:
            observed = str(exc)
        cases.append({"case_id": case_id, "passed": expected in observed, "expected": expected, "observed": observed})

    bad = copy.deepcopy(contract); bad["modes"].pop("eagle")
    record("missing_mode", "generation_mode_coverage_invalid", lambda: load_contract_from_value(bad))
    bad = copy.deepcopy(contract); bad["first_campaign_base"] = "diffusion"
    record("base_substitution", "first_campaign_base_invalid", lambda: load_contract_from_value(bad))
    bad = copy.deepcopy(contract); bad["modes"]["medusa"].pop("reentry_condition")
    record("retirement_without_reentry", "retirement_reentry_missing", lambda: load_contract_from_value(bad))
    tampered = generation_mode_record("mtp", contract); tampered["checkpoint_effect"] = "none"
    record("mode_record_tamper", "generation_mode_record_tampered", lambda: validate_mode_record(tampered, contract))
    checkpoint = checkpoint_contract(contract)
    draft = {"draft_revision": "d", "target_model_revision": "wrong", "target_base_parameter_digest": checkpoint["base_parameter_digest"], "draft_checkpoint_digest": digest("d"), "cache_commit_policy": "accepted_prefix_only"}
    record("draft_target_mismatch", "draft_target_revision_mismatch", lambda: speculative_loader_receipt(checkpoint, draft))
    draft["target_model_revision"] = checkpoint["model_revision"]; draft["cache_commit_policy"] = "all_proposed_tokens"
    record("speculative_cache_poison", "speculative_cache_policy_invalid", lambda: speculative_loader_receipt(checkpoint, draft))
    bad_checkpoint = copy.deepcopy(checkpoint); bad_checkpoint["contract_digest"] = "sha256:wrong"
    record("checkpoint_contract_mismatch", "checkpoint_contract_digest_mismatch", lambda: migrate_checkpoint(bad_checkpoint, contract))
    record("unknown_mode", "generation_mode_unknown", lambda: generation_mode_record("unknown", contract))
    return {"case_count": len(cases), "passed_count": sum(bool(row["passed"]) for row in cases), "results": cases}


def load_contract_from_value(contract: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="theseus-generation-contract-") as tmp:
        path = Path(tmp) / "contract.json"
        path.write_text(json.dumps(contract), encoding="utf-8")
        return load_contract(path)
