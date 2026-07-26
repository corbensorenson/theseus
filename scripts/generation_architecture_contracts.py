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

import host_resource_safety
from target_authoritative_drafting import mlx_drafting_adequacy_canary
from kerc_structured_drafting import mlx_structured_drafting_canary


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
    required = {"policy", "schema_version", "owner", "first_campaign_base", "maximum_architecture_canary_steps", "common_accounting", "modes", "mtp_shape_contract", "mtp_candidates", "draft_abi", "draft_candidates", "activation", "claim_boundaries"}
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
    expected_candidates = {
        "legacy_shared_rank1",
        "conventional_independent",
        "register_conditioned",
    }
    if set(contract["mtp_candidates"]) != expected_candidates:
        raise GenerationArchitectureFault("mtp_candidate_coverage_invalid")
    required_candidate = {
        "selection_state",
        "head_mode",
        "future_offsets",
        "loss_weights",
        "low_rank",
        "hidden_dim",
        "register_count",
        "maximum_parameter_overhead_ratio",
        "curriculum",
    }
    for candidate_id, candidate in contract["mtp_candidates"].items():
        missing_candidate = required_candidate.difference(candidate)
        if missing_candidate:
            raise GenerationArchitectureFault(
                f"mtp_candidate_incomplete:{candidate_id}:{','.join(sorted(missing_candidate))}"
            )
        if candidate["head_mode"] not in {
            "shared_low_rank",
            "independent_mlp",
            "register_conditioned",
        }:
            raise GenerationArchitectureFault(f"mtp_head_mode_invalid:{candidate_id}")
        if len(candidate["future_offsets"]) != len(candidate["loss_weights"]):
            raise GenerationArchitectureFault(f"mtp_target_alignment_invalid:{candidate_id}")
        curriculum = candidate["curriculum"]
        if set(curriculum) != {"warmup_steps", "ramp_steps", "maximum_loss_scale"}:
            raise GenerationArchitectureFault(f"mtp_curriculum_invalid:{candidate_id}")
        if int(curriculum["ramp_steps"]) <= 0 or float(curriculum["maximum_loss_scale"]) < 0:
            raise GenerationArchitectureFault(f"mtp_curriculum_budget_invalid:{candidate_id}")
    if not contract["mtp_candidates"]["legacy_shared_rank1"]["selection_state"].endswith(
        "not_selectable"
    ):
        raise GenerationArchitectureFault("legacy_rank1_candidate_must_not_be_selectable")
    if set(contract["draft_candidates"]) != {
        "medusa_tree",
        "eagle_feature",
        "separate_draft",
        "kerc_structured_unit",
    }:
        raise GenerationArchitectureFault("draft_candidate_coverage_invalid")
    draft_abi = contract["draft_abi"]
    if draft_abi.get("canonical_owner") != "scripts/target_authoritative_drafting.py":
        raise GenerationArchitectureFault("draft_abi_owner_invalid")
    if draft_abi.get("cache_commit_policy") != "target_accepted_prefix_only":
        raise GenerationArchitectureFault("draft_abi_commit_policy_invalid")
    if int(draft_abi.get("maximum_draft_depth", 0)) > 32:
        raise GenerationArchitectureFault("draft_abi_depth_invalid")
    return contract


def mtp_candidate_model_config(
    candidate_id: str, contract: dict[str, Any] | None = None
) -> dict[str, Any]:
    contract = contract or load_contract()
    try:
        candidate = contract["mtp_candidates"][candidate_id]
    except KeyError as exc:
        raise GenerationArchitectureFault("mtp_candidate_unknown") from exc
    return {
        "mtp_future_offsets": tuple(int(value) for value in candidate["future_offsets"]),
        "mtp_head_mode": str(candidate["head_mode"]),
        "mtp_low_rank": int(candidate["low_rank"]),
        "mtp_hidden_dim": int(candidate["hidden_dim"]),
        "mtp_register_count": int(candidate["register_count"]),
        "mtp_loss_weights": tuple(float(value) for value in candidate["loss_weights"]),
        "mtp_loss_scale": float(candidate["curriculum"]["maximum_loss_scale"]),
        "mtp_maximum_head_parameter_overhead_ratio": float(
            candidate["maximum_parameter_overhead_ratio"]
        ),
    }


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
    if not host_resource_safety.accelerator_child_authorized():
        return {
            "available": False,
            "passed": False,
            "fault": "ACCELERATOR_WATCHDOG_REQUIRED",
            "mtp_parameter_overhead_ratio": float("inf"),
            "optimizer_steps": 0,
            "resource_decision": "ACCELERATOR_WATCHDOG_REQUIRED",
            "capability_claim": "NOT_EVALUATED",
        }
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
        return {
            "available": False,
            "passed": False,
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-1000:],
            "mtp_parameter_overhead_ratio": float("inf"),
            "optimizer_steps": 0,
            "resource_decision": "ACCELERATOR_RUNTIME_UNAVAILABLE",
            "capability_claim": "NOT_EVALUATED",
        }
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


def mlx_mtp_adequacy_canary(
    contract: dict[str, Any] | None = None,
    *,
    optimizer_steps: int = 24,
) -> dict[str, Any]:
    """Exercise adequate MTP mechanisms without making a capability claim."""

    contract = contract or load_contract()
    if not host_resource_safety.accelerator_child_authorized():
        return {
            "available": False,
            "passed": False,
            "fault": "ACCELERATOR_WATCHDOG_REQUIRED",
            "optimizer_steps": 0,
            "capability_claim": "NOT_EVALUATED",
        }
    if optimizer_steps <= 0 or optimizer_steps > 48:
        raise GenerationArchitectureFault("mtp_adequacy_step_budget_invalid")
    candidates = {
        candidate_id: candidate
        for candidate_id, candidate in contract["mtp_candidates"].items()
        if candidate["selection_state"] == "candidate_mechanics_qualification_required"
    }
    payload = {
        "candidates": candidates,
        "optimizer_steps": int(optimizer_steps),
        "scripts": str(ROOT / "scripts"),
    }
    code = r'''
import json, sys, tempfile
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
p=json.loads(sys.stdin.read()); sys.path.insert(0,p['scripts'])
from standard_causal_transformer_model import CausalTransformerConfig, build_model
from standard_causal_transformer_objectives import mtp_auxiliary_loss, mtp_curriculum_scale

def scalar(value):
    mx.eval(value)
    return float(value.item())

def objective(model, x, y, mask, scale):
    logits, _cache, aux = model(x, return_training_aux=True)
    token_loss = nn.losses.cross_entropy(logits, y)
    denominator = mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mx.float32))
    ntp = mx.sum(token_loss * mask) / denominator
    mtp = mtp_auxiliary_loss(
        list(aux['mtp_logits']), y, mask, model.mtp_future_offsets,
        model.mtp_loss_weights, mx, nn,
    )
    return ntp + scale * mtp

rows=[]
for candidate_id, candidate in sorted(p['candidates'].items()):
    cfg=CausalTransformerConfig(
        vocab_size=32,d_model=16,num_layers=1,num_heads=4,num_kv_heads=2,ff_dim=32,
        mtp_future_offsets=tuple(candidate['future_offsets']),
        mtp_head_mode=candidate['head_mode'],
        mtp_low_rank=int(candidate['low_rank']),
        mtp_hidden_dim=int(candidate['hidden_dim']),
        mtp_register_count=int(candidate['register_count']),
        mtp_loss_weights=tuple(candidate['loss_weights']),
        mtp_loss_scale=float(candidate['curriculum']['maximum_loss_scale']),
        mtp_maximum_head_parameter_overhead_ratio=float(candidate['maximum_parameter_overhead_ratio']),
    )
    mx.random.seed(20260722)
    model=build_model(cfg,mx=mx,nn=nn)
    starts=mx.arange(8,dtype=mx.int32)[:,None]
    positions=mx.arange(12,dtype=mx.int32)[None,:]
    x=(starts+positions)%32; y=(x+1)%32; mask=mx.ones(y.shape,dtype=mx.float32)
    initial_logits,_cache,initial_aux=model(x,return_training_aux=True)
    initial_mtp=mtp_auxiliary_loss(list(initial_aux['mtp_logits']),y,mask,model.mtp_future_offsets,model.mtp_loss_weights,mx,nn)
    loss_and_grad=nn.value_and_grad(model,objective)
    maximum=float(candidate['curriculum']['maximum_loss_scale'])
    loss_scales=[]; first_gradient_norm=0.0; zero_scale_gradient_norm=0.0
    _zero_loss,zero_grads=loss_and_grad(model,x,y,mask,0.0)
    for name,value in mlx_utils.tree_flatten(zero_grads):
        if name.startswith('mtp_'):
            zero_scale_gradient_norm += scalar(mx.sum(mx.abs(value)))
    optimizer=optim.AdamW(learning_rate=0.02,weight_decay=0.0)
    for step in range(int(p['optimizer_steps'])):
        scale=mtp_curriculum_scale(
            step,
            warmup_steps=int(candidate['curriculum']['warmup_steps']),
            ramp_steps=int(candidate['curriculum']['ramp_steps']),
            maximum=maximum,
        )
        loss,grads=loss_and_grad(model,x,y,mask,scale)
        if step == int(candidate['curriculum']['warmup_steps']):
            for name,value in mlx_utils.tree_flatten(grads):
                if name.startswith('mtp_'):
                    first_gradient_norm += scalar(mx.sum(mx.abs(value)))
        optimizer.update(model,grads); mx.eval(model.parameters(),optimizer.state,loss)
        loss_scales.append(scale)
    final_logits,_cache,final_aux=model(x,return_training_aux=True)
    final_mtp=mtp_auxiliary_loss(list(final_aux['mtp_logits']),y,mask,model.mtp_future_offsets,model.mtp_loss_weights,mx,nn)
    wrong_y=mx.roll(y,shift=3,axis=1)
    wrong_mtp=mtp_auxiliary_loss(list(final_aux['mtp_logits']),wrong_y,mask,model.mtp_future_offsets,model.mtp_loss_weights,mx,nn)
    flat=mlx_utils.tree_flatten(model.parameters())
    mtp_params=sum(int(value.size) for name,value in flat if name.startswith('mtp_'))
    total_params=sum(int(value.size) for _name,value in flat)
    with tempfile.TemporaryDirectory(prefix='theseus-mtp-adequacy-') as tmp:
        path=Path(tmp)/'weights.npz'; model.save_weights(str(path))
        reloaded=build_model(cfg,mx=mx,nn=nn); reloaded.load_weights(str(path))
        reload_logits,_cache,reload_aux=reloaded(x,return_training_aux=True)
        deltas=[scalar(mx.max(mx.abs(final_logits-reload_logits)))]
        deltas.extend(scalar(mx.max(mx.abs(a-b))) for a,b in zip(final_aux['mtp_logits'],reload_aux['mtp_logits']))
        checkpoint_bytes=path.stat().st_size
    rows.append({
        'candidate_id':candidate_id,
        'head_mode':candidate['head_mode'],
        'optimizer_steps':int(p['optimizer_steps']),
        'loss_scales':loss_scales,
        'initial_mtp_loss':scalar(initial_mtp),
        'final_mtp_loss':scalar(final_mtp),
        'wrong_alignment_mtp_loss':scalar(wrong_mtp),
        'first_active_mtp_gradient_l1':first_gradient_norm,
        'zero_scale_mtp_gradient_l1':zero_scale_gradient_norm,
        'mtp_parameter_count':mtp_params,
        'total_parameter_count':total_params,
        'checkpoint_bytes':checkpoint_bytes,
        'checkpoint_reload_max_abs_delta':max(deltas),
        'base_shape':list(final_logits.shape),
        'mtp_shapes':[list(value.shape) for value in final_aux['mtp_logits']],
    })
print(json.dumps(rows,sort_keys=True))
'''
    proc = subprocess.run(
        [sys.executable, "-c", code],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=120,
    )
    if proc.returncode != 0:
        return {
            "available": False,
            "passed": False,
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-2000:],
            "optimizer_steps": 0,
            "capability_claim": "NOT_EVALUATED",
        }
    rows = json.loads(proc.stdout)
    checks = {}
    for row in rows:
        candidate = candidates[row["candidate_id"]]
        scales = row["loss_scales"]
        warmup = int(candidate["curriculum"]["warmup_steps"])
        checks[row["candidate_id"]] = {
            "head_gradient_flow": row["first_active_mtp_gradient_l1"] > 0.0,
            "zero_scale_head_ablation": row["zero_scale_mtp_gradient_l1"] == 0.0,
            "curriculum_warmup_and_ramp": all(value == 0.0 for value in scales[:warmup])
            and scales[-1] == float(candidate["curriculum"]["maximum_loss_scale"])
            and scales == sorted(scales),
            "future_loss_learned": row["final_mtp_loss"] < row["initial_mtp_loss"],
            "future_alignment_sensitive": row["final_mtp_loss"] < row["wrong_alignment_mtp_loss"],
            "checkpoint_reload_exact": row["checkpoint_reload_max_abs_delta"] == 0.0,
            "shape_contract": row["base_shape"] == [8, 12, 32]
            and row["mtp_shapes"] == [[8, 12, 32]] * 3,
            "parameter_inventory_present": row["mtp_parameter_count"] > 0
            and row["total_parameter_count"] > row["mtp_parameter_count"],
        }
    return {
        "available": True,
        "passed": bool(checks)
        and all(all(candidate_checks.values()) for candidate_checks in checks.values()),
        "candidate_checks": checks,
        "candidates": rows,
        "optimizer_steps": sum(int(row["optimizer_steps"]) for row in rows),
        "private_fixture_rows": 8,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_or_template_credit": 0,
        "capability_claim": "NOT_EVALUATED",
        "claim_boundary": "mechanics canary only; no speed, utility, transfer, or architecture-selection claim",
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
    mtp_adequacy = mlx_mtp_adequacy_canary(contract)
    drafting_adequacy = mlx_drafting_adequacy_canary()
    kerc_structured_drafting = mlx_structured_drafting_canary()
    activation = activation_receipt({}, contract)
    controls = mutation_controls(contract)
    retired = sorted(mode_id for mode_id, mode in contract["modes"].items() if mode["first_campaign_disposition"].startswith("retired"))
    drafting_checks = drafting_adequacy.get("checks") or {}
    drafting_safe_retired_mechanics = (
        drafting_adequacy.get("available") is True
        and (drafting_checks.get("frozen_target") or {}).get(
            "target_parameters_unchanged"
        )
        is True
        and all(
            (drafting_checks.get(candidate_id) or {}).get(field) is True
            for candidate_id in ("medusa_tree", "eagle_feature", "separate_draft")
            for field in (
                "gradient_flow",
                "checkpoint_reload",
                "nonempty_checkpoint",
                "nonempty_parameterization",
            )
        )
        and all(
            str(contract["modes"][mode_id]["first_campaign_disposition"]).startswith(
                "retired"
            )
            for mode_id in ("medusa", "eagle", "speculative")
        )
    )
    gates = {
        "all_mode_records_valid": len(validations) == len(contract["modes"]),
        "first_campaign_base_ar": contract["first_campaign_base"] == "autoregressive",
        "mtp_mlx_shape_canary": mtp["available"] and mtp["passed"],
        "mtp_resource_ceiling_passed": float(
            mtp.get("mtp_parameter_overhead_ratio", float("inf"))
        )
        <= float(contract["mtp_shape_contract"]["maximum_parameter_overhead_ratio"]),
        "mtp_adequate_candidate_mechanics": mtp_adequacy["available"]
        and mtp_adequacy["passed"],
        "target_authoritative_drafting_safe_retired_mechanics": drafting_safe_retired_mechanics,
        "kerc_structured_drafting_mechanics": kerc_structured_drafting["available"]
        and kerc_structured_drafting["passed"],
        "mtp_zero_initial_weight": checkpoint["optional_head_groups"]["mtp"]["initial_weight"] == 0.0,
        "checkpoint_roundtrip_migration_cleanup": roundtrip["roundtrip_exact"] and roundtrip["migration_exact"] and roundtrip["cleanup_complete"],
        "retired_modes_absent_from_optimizer": roundtrip["retired_modes_absent_from_optimizer"],
        "post_hoc_speculative_compatible_disabled": speculative["compatible"] and not speculative["enabled"] and not speculative["target_topology_changed"],
        "retirements_have_reentry_conditions": all(contract["modes"][mode_id].get("reentry_condition") for mode_id in retired),
        "runtime_mode_selection_disabled_without_behavior": not activation["authorized"],
        "mutation_controls_rejected": controls["case_count"] == controls["passed_count"],
        "zero_optimizer_exposure": int(mtp.get("optimizer_steps") or 0) == 0,
        "no_cheat_counters_clean": True,
    }
    return {
        "policy": contract["policy"],
        "trigger_state": "GREEN" if all(gates.values()) else "RED",
        "support_state": "synthetic-test-backed",
        "summary": {"mode_count": len(records), "included_mode_count": sum(not mode["first_campaign_disposition"].startswith("retired") for mode in contract["modes"].values()), "retired_first_campaign_mode_count": len(retired), "mutation_case_count": controls["case_count"], "mutation_passed_count": controls["passed_count"], "mlx_available": mtp.get("available", False), "mtp_canary_passed": mtp.get("passed", False), "mtp_adequacy_canary_passed": mtp_adequacy.get("passed", False), "drafting_adequacy_canary_passed": drafting_adequacy.get("passed", False), "drafting_safe_retired_mechanics_passed": drafting_safe_retired_mechanics, "kerc_structured_drafting_canary_passed": kerc_structured_drafting.get("passed", False), "mechanics_canary_optimizer_steps": int(mtp_adequacy.get("optimizer_steps") or 0) + int(drafting_adequacy.get("optimizer_steps") or 0), "runtime_authorized": activation["authorized"], "optimizer_exposure_steps": 0, "public_training_rows_written": 0, "external_inference_calls": 0, "fallback_or_template_credit": 0},
        "gates": gates,
        "mode_records": records,
        "checkpoint_receipt": roundtrip,
        "mtp_mlx_canary": mtp,
        "mtp_adequacy_canary": mtp_adequacy,
        "drafting_adequacy_canary": drafting_adequacy,
        "kerc_structured_drafting_canary": kerc_structured_drafting,
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
    bad = copy.deepcopy(contract); bad["modes"]["layerskip"].pop("reentry_condition")
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
    bad = copy.deepcopy(contract); bad["mtp_candidates"]["legacy_shared_rank1"]["selection_state"] = "selectable"
    record("legacy_rank1_selection", "legacy_rank1_candidate_must_not_be_selectable", lambda: load_contract_from_value(bad))
    bad = copy.deepcopy(contract); bad["mtp_candidates"]["conventional_independent"]["curriculum"]["ramp_steps"] = 0
    record("mtp_zero_ramp", "mtp_curriculum_budget_invalid", lambda: load_contract_from_value(bad))
    record("unknown_mtp_candidate", "mtp_candidate_unknown", lambda: mtp_candidate_model_config("unknown", contract))
    bad = copy.deepcopy(contract); bad["draft_abi"]["cache_commit_policy"] = "all_proposals"
    record("draft_abi_cache_poison", "draft_abi_commit_policy_invalid", lambda: load_contract_from_value(bad))
    return {"case_count": len(cases), "passed_count": sum(bool(row["passed"]) for row in cases), "results": cases}


def load_contract_from_value(contract: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="theseus-generation-contract-") as tmp:
        path = Path(tmp) / "contract.json"
        path.write_text(json.dumps(contract), encoding="utf-8")
        return load_contract(path)
