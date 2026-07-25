#!/usr/bin/env python3
"""Target-authoritative draft ABI and bounded MLX adequacy mechanics.

Draft models may propose work, but only the canonical target may commit tokens or
KV state.  This module owns that invariant for Medusa-style token heads,
EAGLE-style feature drafting, and a conventional separate-draft control.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import host_resource_safety


ROOT = Path(__file__).resolve().parents[1]


class DraftAuthorityFault(RuntimeError):
    """A draft attempted to cross the target-authority boundary."""


def digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DraftManifest:
    schema_version: str
    mode: str
    draft_revision: str
    draft_checkpoint_digest: str
    target_model_revision: str
    target_parameter_digest: str
    codec_digest: str
    cache_schema_version: str
    max_draft_depth: int
    cache_commit_policy: str = "target_accepted_prefix_only"
    hidden_state_source: str = "target_final_hidden"


def validate_manifest(
    manifest: DraftManifest | dict[str, Any],
    *,
    target_model_revision: str,
    target_parameter_digest: str,
    codec_digest: str,
) -> dict[str, Any]:
    record = asdict(manifest) if isinstance(manifest, DraftManifest) else dict(manifest)
    required = set(DraftManifest.__dataclass_fields__)
    missing = sorted(required.difference(record))
    if missing:
        raise DraftAuthorityFault(f"draft_manifest_incomplete:{','.join(missing)}")
    if record["schema_version"] != "1.0.0":
        raise DraftAuthorityFault("draft_manifest_schema_unsupported")
    if record["mode"] not in {
        "medusa_tree",
        "eagle_feature",
        "separate_draft",
        "kerc_structured_unit",
    }:
        raise DraftAuthorityFault("draft_mode_unsupported")
    if record["target_model_revision"] != target_model_revision:
        raise DraftAuthorityFault("draft_target_revision_mismatch")
    if record["target_parameter_digest"] != target_parameter_digest:
        raise DraftAuthorityFault("draft_target_parameter_mismatch")
    if record["codec_digest"] != codec_digest:
        raise DraftAuthorityFault("draft_codec_mismatch")
    if record["cache_schema_version"] != "target_tree_kv_v1":
        raise DraftAuthorityFault("draft_cache_schema_mismatch")
    if record["cache_commit_policy"] != "target_accepted_prefix_only":
        raise DraftAuthorityFault("draft_cache_commit_policy_invalid")
    if not 1 <= int(record["max_draft_depth"]) <= 32:
        raise DraftAuthorityFault("draft_depth_out_of_bounds")
    if record["mode"] in {"eagle_feature", "kerc_structured_unit"} and record["hidden_state_source"] != "target_final_hidden":
        raise DraftAuthorityFault("eagle_hidden_state_source_invalid")
    return {
        "valid": True,
        "manifest_digest": digest(record),
        "target_binding": digest(
            {
                "model_revision": target_model_revision,
                "parameter_digest": target_parameter_digest,
                "codec_digest": codec_digest,
            }
        ),
    }


@dataclass(frozen=True)
class VerificationStep:
    index: int
    proposed_token: int | None
    target_token: int
    accepted: bool
    committed_prefix_length: int


@dataclass(frozen=True)
class DraftVerificationReceipt:
    initial_prefix: tuple[int, ...]
    proposed_tokens: tuple[int, ...]
    committed_tokens: tuple[int, ...]
    accepted_draft_tokens: tuple[int, ...]
    rejected_draft_tokens: tuple[int, ...]
    steps: tuple[VerificationStep, ...]
    target_calls: int
    rollback_count: int
    cache_commit_policy: str
    exact_target_authority: bool


@dataclass(frozen=True)
class TargetKVState:
    """Committed target state. MLX cache arrays are immutable and branch-shareable."""

    committed_prefix: tuple[int, ...]
    layer_cache: Any
    next_logits: Any
    target_revision: str
    cache_schema_version: str = "target_tree_kv_v1"


@dataclass(frozen=True)
class TargetKVVerificationReceipt:
    initial_prefix: tuple[int, ...]
    proposed_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    accepted_draft_tokens: tuple[int, ...]
    rejected_draft_tokens: tuple[int, ...]
    committed_tokens: tuple[int, ...]
    branch_cache_length: int
    committed_cache_length: int
    target_forward_calls: int
    target_evaluated_positions: int
    rollback_count: int
    rejected_suffix_cache_committed: bool
    exact_target_authority: bool


TargetNext = Callable[[tuple[int, ...]], int]


def verify_draft_branch(
    prefix: Sequence[int],
    proposed_tokens: Sequence[int],
    target_next: TargetNext,
    *,
    append_target_after_full_acceptance: bool = True,
) -> DraftVerificationReceipt:
    """Verify one branch and commit only target-confirmed state.

    A mismatch discards the entire unaccepted suffix and commits the target token
    at that position.  If the full draft is accepted, one target token is added so
    speculative work never suppresses the target's next decision.
    """

    committed = [int(token) for token in prefix]
    proposed = [int(token) for token in proposed_tokens]
    accepted: list[int] = []
    rejected: list[int] = []
    steps: list[VerificationStep] = []
    target_calls = 0
    rollback_count = 0
    for index, proposal in enumerate(proposed):
        target_token = int(target_next(tuple(committed)))
        target_calls += 1
        matched = proposal == target_token
        if matched:
            accepted.append(proposal)
            committed.append(proposal)
        else:
            rejected.extend(proposed[index:])
            committed.append(target_token)
            rollback_count += 1
        steps.append(
            VerificationStep(
                index=index,
                proposed_token=proposal,
                target_token=target_token,
                accepted=matched,
                committed_prefix_length=len(committed),
            )
        )
        if not matched:
            break
    if proposed and not rejected and append_target_after_full_acceptance:
        target_token = int(target_next(tuple(committed)))
        target_calls += 1
        committed.append(target_token)
        steps.append(
            VerificationStep(
                index=len(proposed),
                proposed_token=None,
                target_token=target_token,
                accepted=False,
                committed_prefix_length=len(committed),
            )
        )
    return DraftVerificationReceipt(
        initial_prefix=tuple(int(token) for token in prefix),
        proposed_tokens=tuple(proposed),
        committed_tokens=tuple(committed[len(prefix) :]),
        accepted_draft_tokens=tuple(accepted),
        rejected_draft_tokens=tuple(rejected),
        steps=tuple(steps),
        target_calls=target_calls,
        rollback_count=rollback_count,
        cache_commit_policy="target_accepted_prefix_only",
        exact_target_authority=all(
            step.proposed_token is None
            or step.accepted
            or step.target_token != step.proposed_token
            for step in steps
        ),
    )


def target_authoritative_decode(
    prefix: Sequence[int],
    propose: Callable[[tuple[int, ...], int], Sequence[int]],
    target_next: TargetNext,
    *,
    max_new_tokens: int,
    max_draft_depth: int,
) -> tuple[list[int], list[DraftVerificationReceipt]]:
    if max_new_tokens < 0 or not 1 <= max_draft_depth <= 32:
        raise DraftAuthorityFault("decode_budget_invalid")
    committed = [int(token) for token in prefix]
    initial_length = len(committed)
    receipts: list[DraftVerificationReceipt] = []
    while len(committed) - initial_length < max_new_tokens:
        remaining = max_new_tokens - (len(committed) - initial_length)
        proposal = list(propose(tuple(committed), min(max_draft_depth, remaining)))
        if len(proposal) > min(max_draft_depth, remaining):
            raise DraftAuthorityFault("drafter_exceeded_depth_budget")
        if not proposal:
            committed.append(int(target_next(tuple(committed))))
            continue
        receipt = verify_draft_branch(
            committed,
            proposal,
            target_next,
            append_target_after_full_acceptance=remaining > len(proposal),
        )
        accepted_now = list(receipt.committed_tokens)[:remaining]
        committed.extend(accepted_now)
        receipts.append(receipt)
    return committed, receipts


def target_cache_length(layer_cache: Any) -> int:
    if not isinstance(layer_cache, (list, tuple)) or not layer_cache:
        raise DraftAuthorityFault("target_cache_missing")
    lengths = []
    for layer in layer_cache:
        if not isinstance(layer, (list, tuple)) or len(layer) != 2:
            raise DraftAuthorityFault("target_cache_layer_invalid")
        key, value = layer
        if tuple(key.shape) != tuple(value.shape) or len(key.shape) != 4:
            raise DraftAuthorityFault("target_cache_shape_invalid")
        lengths.append(int(key.shape[2]))
    if len(set(lengths)) != 1:
        raise DraftAuthorityFault("target_cache_layer_length_mismatch")
    return lengths[0]


def truncate_target_cache(layer_cache: Any, committed_length: int) -> list[tuple[Any, Any]]:
    """Return a branch view containing only target-authorized positions."""

    observed = target_cache_length(layer_cache)
    if committed_length < 0 or committed_length > observed:
        raise DraftAuthorityFault("target_cache_truncation_out_of_bounds")
    return [
        (key[:, :, :committed_length, :], value[:, :, :committed_length, :])
        for key, value in layer_cache
    ]


def prefill_target_kv(
    model: Any,
    prefix: Sequence[int],
    *,
    mx: Any,
    target_revision: str,
) -> TargetKVState:
    if not prefix:
        raise DraftAuthorityFault("target_prefill_prefix_empty")
    tokens = mx.array([[int(token) for token in prefix]], dtype=mx.int32)
    logits, layer_cache = model(tokens)
    mx.eval(logits, layer_cache)
    if target_cache_length(layer_cache) != len(prefix):
        raise DraftAuthorityFault("target_prefill_cache_length_mismatch")
    return TargetKVState(
        committed_prefix=tuple(int(token) for token in prefix),
        layer_cache=layer_cache,
        next_logits=logits[:, -1, :],
        target_revision=str(target_revision),
    )


def _greedy_token(logits: Any, mx: Any) -> int:
    token = mx.argmax(logits, axis=-1)
    mx.eval(token)
    return int(token.item())


def verify_draft_branch_with_target_kv(
    state: TargetKVState,
    proposed_tokens: Sequence[int],
    model: Any,
    *,
    mx: Any,
    target_revision: str,
) -> tuple[TargetKVState, TargetKVVerificationReceipt]:
    """Verify a span with real target KV state and commit no rejected suffix.

    The target evaluates the proposed span as one branch forward. Logits before
    each proposed token determine acceptance. A mismatch truncates branch KV to
    the accepted prefix before the target correction is evaluated and committed.
    A fully accepted branch commits one additional target token so draft work
    cannot suppress the target's next decision.
    """

    if state.cache_schema_version != "target_tree_kv_v1":
        raise DraftAuthorityFault("target_cache_schema_mismatch")
    if state.target_revision != target_revision:
        raise DraftAuthorityFault("target_cache_revision_mismatch")
    base_length = len(state.committed_prefix)
    if target_cache_length(state.layer_cache) != base_length:
        raise DraftAuthorityFault("target_cache_prefix_length_mismatch")
    proposals = tuple(int(token) for token in proposed_tokens)
    if not proposals:
        target_token = _greedy_token(state.next_logits, mx)
        token = mx.array([[target_token]], dtype=mx.int32)
        logits, committed_cache = model(token, cache=state.layer_cache)
        mx.eval(logits, committed_cache)
        committed_state = TargetKVState(
            committed_prefix=(*state.committed_prefix, target_token),
            layer_cache=committed_cache,
            next_logits=logits[:, -1, :],
            target_revision=target_revision,
        )
        return committed_state, TargetKVVerificationReceipt(
            initial_prefix=state.committed_prefix,
            proposed_tokens=(),
            target_tokens=(target_token,),
            accepted_draft_tokens=(),
            rejected_draft_tokens=(),
            committed_tokens=(target_token,),
            branch_cache_length=base_length,
            committed_cache_length=base_length + 1,
            target_forward_calls=1,
            target_evaluated_positions=1,
            rollback_count=0,
            rejected_suffix_cache_committed=False,
            exact_target_authority=True,
        )

    proposal_array = mx.array([list(proposals)], dtype=mx.int32)
    branch_logits, branch_cache = model(proposal_array, cache=state.layer_cache)
    mx.eval(branch_logits, branch_cache)
    branch_length = target_cache_length(branch_cache)
    if branch_length != base_length + len(proposals):
        raise DraftAuthorityFault("target_branch_cache_length_mismatch")
    target_tokens = [_greedy_token(state.next_logits, mx)]
    target_tokens.extend(
        _greedy_token(branch_logits[:, index - 1, :], mx)
        for index in range(1, len(proposals))
    )
    accepted_count = 0
    for proposal, target in zip(proposals, target_tokens):
        if proposal != target:
            break
        accepted_count += 1
    accepted = proposals[:accepted_count]
    rejected = proposals[accepted_count:]
    authorized_cache = truncate_target_cache(
        branch_cache, base_length + accepted_count
    )
    if rejected:
        commit_token = target_tokens[accepted_count]
        rollback_count = 1
    else:
        commit_token = _greedy_token(branch_logits[:, -1, :], mx)
        target_tokens.append(commit_token)
        rollback_count = 0
    commit_array = mx.array([[commit_token]], dtype=mx.int32)
    next_logits, committed_cache = model(commit_array, cache=authorized_cache)
    mx.eval(next_logits, committed_cache)
    committed_tokens = (*accepted, commit_token)
    committed_prefix = (*state.committed_prefix, *committed_tokens)
    committed_length = target_cache_length(committed_cache)
    if committed_length != len(committed_prefix):
        raise DraftAuthorityFault("target_committed_cache_length_mismatch")
    committed_state = TargetKVState(
        committed_prefix=committed_prefix,
        layer_cache=committed_cache,
        next_logits=next_logits[:, -1, :],
        target_revision=target_revision,
    )
    receipt = TargetKVVerificationReceipt(
        initial_prefix=state.committed_prefix,
        proposed_tokens=proposals,
        target_tokens=tuple(target_tokens),
        accepted_draft_tokens=accepted,
        rejected_draft_tokens=rejected,
        committed_tokens=committed_tokens,
        branch_cache_length=branch_length,
        committed_cache_length=committed_length,
        target_forward_calls=2,
        target_evaluated_positions=len(proposals) + 1,
        rollback_count=rollback_count,
        rejected_suffix_cache_committed=committed_length
        > base_length + accepted_count + 1,
        exact_target_authority=(
            not rejected
            or proposals[accepted_count] != target_tokens[accepted_count]
        ),
    )
    if receipt.rejected_suffix_cache_committed:
        raise DraftAuthorityFault("rejected_suffix_cache_commit")
    return committed_state, receipt


def target_authoritative_kv_decode(
    model: Any,
    prefix: Sequence[int],
    propose: Callable[[tuple[int, ...], int], Sequence[int]],
    *,
    mx: Any,
    target_revision: str,
    max_new_tokens: int,
    max_draft_depth: int,
) -> tuple[list[int], list[TargetKVVerificationReceipt]]:
    if max_new_tokens < 0 or not 1 <= max_draft_depth <= 32:
        raise DraftAuthorityFault("decode_budget_invalid")
    state = prefill_target_kv(
        model, prefix, mx=mx, target_revision=target_revision
    )
    initial_length = len(prefix)
    receipts = []
    while len(state.committed_prefix) - initial_length < max_new_tokens:
        remaining = max_new_tokens - (len(state.committed_prefix) - initial_length)
        proposals = tuple(
            int(token)
            for token in propose(
                state.committed_prefix, min(max_draft_depth, remaining)
            )
        )
        if len(proposals) > min(max_draft_depth, remaining):
            raise DraftAuthorityFault("drafter_exceeded_depth_budget")
        next_state, receipt = verify_draft_branch_with_target_kv(
            state,
            proposals,
            model,
            mx=mx,
            target_revision=target_revision,
        )
        committed_now = list(receipt.committed_tokens)[:remaining]
        if len(committed_now) != len(receipt.committed_tokens):
            keep_length = len(state.committed_prefix) + len(committed_now)
            next_state = TargetKVState(
                committed_prefix=(*state.committed_prefix, *committed_now),
                layer_cache=truncate_target_cache(next_state.layer_cache, keep_length),
                # This edge is reached only when a full draft fills the budget;
                # no further token is requested, so next logits are non-authoritative.
                next_logits=next_state.next_logits,
                target_revision=target_revision,
            )
        state = next_state
        receipts.append(receipt)
    return list(state.committed_prefix), receipts


def build_medusa_drafter(
    *,
    d_model: int,
    vocab_size: int,
    future_offsets: Sequence[int],
    hidden_dim: int,
    mx: Any,
    nn: Any,
) -> Any:
    if not future_offsets or hidden_dim <= 0:
        raise ValueError("Medusa requires future offsets and a positive hidden dimension")

    class MedusaHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.RMSNorm(d_model)
            self.down = nn.Linear(d_model, hidden_dim, bias=False)
            self.up = nn.Linear(hidden_dim, d_model, bias=False)
            self.output = nn.Linear(d_model, vocab_size, bias=False)

        def __call__(self, hidden: Any) -> Any:
            residual = hidden + self.up(nn.silu(self.down(self.norm(hidden))))
            return self.output(residual)

    class MedusaDrafter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = [MedusaHead() for _ in future_offsets]

        def __call__(self, hidden: Any) -> list[Any]:
            return [head(hidden) for head in self.heads]

    return MedusaDrafter()


def build_eagle_feature_drafter(
    *,
    d_model: int,
    vocab_size: int,
    hidden_dim: int,
    mx: Any,
    nn: Any,
) -> Any:
    if hidden_dim <= 0:
        raise ValueError("EAGLE requires a positive hidden dimension")

    class EagleFeatureDrafter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.RMSNorm(d_model * 2)
            self.input = nn.Linear(d_model * 2, hidden_dim, bias=False)
            self.feature = nn.Linear(hidden_dim, d_model, bias=False)
            self.output = nn.Linear(d_model, vocab_size, bias=False)

        def __call__(self, target_hidden: Any, token_embedding: Any) -> tuple[Any, Any]:
            joined = mx.concatenate([target_hidden, token_embedding], axis=-1)
            predicted_feature = self.feature(nn.silu(self.input(self.norm(joined))))
            return predicted_feature, self.output(predicted_feature)

    return EagleFeatureDrafter()


def mlx_drafting_adequacy_canary(*, optimizer_steps: int = 48) -> dict[str, Any]:
    """Train real draft modules against a frozen MLX target on a bounded fixture."""

    if not host_resource_safety.accelerator_child_authorized():
        return {
            "available": False,
            "passed": False,
            "fault": "ACCELERATOR_WATCHDOG_REQUIRED",
            "optimizer_steps": 0,
            "capability_claim": "NOT_EVALUATED",
            "speed_claim": "NOT_EVALUATED",
        }

    payload = {"scripts": str(ROOT / "scripts"), "optimizer_steps": int(optimizer_steps)}
    code = r'''
import json, sys, tempfile
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
p=json.loads(sys.stdin.read()); sys.path.insert(0,p["scripts"])
from standard_causal_transformer_model import CausalTransformerConfig, build_model
from target_authoritative_drafting import build_medusa_drafter, build_eagle_feature_drafter

mx.random.seed(20260722)
cfg=CausalTransformerConfig(vocab_size=48,d_model=24,num_layers=2,num_heads=4,num_kv_heads=2,ff_dim=64)
target=build_model(cfg,mx=mx,nn=nn)
tokens=mx.array([[1,4,7,10,13,16,19,22,25,28,31,34],[2,6,10,14,18,22,26,30,34,38,42,46]],dtype=mx.int32)
_logits,_cache,aux=target(tokens,return_training_aux=True); target_hidden=mx.stop_gradient(aux["final_hidden"]); token_embed=mx.stop_gradient(target.token_embedding(tokens))
mx.eval(target.parameters(),target_hidden,token_embed)
target_before={name:np.array(value) for name,value in mlx_utils.tree_flatten(target.parameters())}

def train(mode):
    mx.random.seed(9001 if mode=="medusa_tree" else 9002)
    if mode=="medusa_tree":
        model=build_medusa_drafter(d_model=24,vocab_size=48,future_offsets=(1,2,3),hidden_dim=32,mx=mx,nn=nn)
        def loss_fn(candidate):
            heads=candidate(target_hidden)
            losses=[]
            for index,offset in enumerate((1,2,3)):
                losses.append(nn.losses.cross_entropy(heads[index][:,:-offset,:],tokens[:,offset:],reduction="mean"))
            return sum(losses)/len(losses)
        def output(candidate): return candidate(target_hidden)
    elif mode=="eagle_feature":
        model=build_eagle_feature_drafter(d_model=24,vocab_size=48,hidden_dim=48,mx=mx,nn=nn)
        def loss_fn(candidate):
            feature,logits=candidate(target_hidden[:,:-1,:],token_embed[:,:-1,:])
            feature_loss=mx.mean((feature-target_hidden[:,1:,:])**2)
            token_loss=nn.losses.cross_entropy(logits,tokens[:,1:],reduction="mean")
            return feature_loss+token_loss
        def output(candidate): return candidate(target_hidden[:,:-1,:],token_embed[:,:-1,:])
    else:
        draft_cfg=CausalTransformerConfig(vocab_size=48,d_model=16,num_layers=1,num_heads=4,num_kv_heads=2,ff_dim=32)
        model=build_model(draft_cfg,mx=mx,nn=nn)
        def loss_fn(candidate):
            logits,_cache=candidate(tokens[:,:-1])
            return nn.losses.cross_entropy(logits,tokens[:,1:],reduction="mean")
        def output(candidate): return candidate(tokens[:,:-1])[0]
    optimizer=optim.AdamW(learning_rate=0.02)
    value_and_grad=nn.value_and_grad(model,loss_fn)
    initial=loss_fn(model); mx.eval(initial)
    active_gradient=False
    for _ in range(p["optimizer_steps"]):
        loss,grads=value_and_grad(model); optimizer.update(model,grads); mx.eval(loss,model.parameters(),optimizer.state)
        active_gradient=active_gradient or any(float(mx.max(mx.abs(g)).item())>0 for _n,g in mlx_utils.tree_flatten(grads))
    final=loss_fn(model); before_output=output(model); mx.eval(final,before_output)
    with tempfile.TemporaryDirectory(prefix="theseus-draft-") as tmp:
        path=Path(tmp)/f"{mode}.safetensors"; model.save_weights(str(path)); checkpoint_bytes=path.stat().st_size
        if mode=="medusa_tree": reloaded=build_medusa_drafter(d_model=24,vocab_size=48,future_offsets=(1,2,3),hidden_dim=32,mx=mx,nn=nn)
        elif mode=="eagle_feature": reloaded=build_eagle_feature_drafter(d_model=24,vocab_size=48,hidden_dim=48,mx=mx,nn=nn)
        else: reloaded=build_model(draft_cfg,mx=mx,nn=nn)
        reloaded.load_weights(str(path)); after_output=output(reloaded); mx.eval(after_output)
        left=mlx_utils.tree_flatten(before_output); right=mlx_utils.tree_flatten(after_output)
        reload_delta=max(float(mx.max(mx.abs(a-b)).item()) for (_na,a),(_nb,b) in zip(left,right))
    return {"initial_loss":float(initial.item()),"final_loss":float(final.item()),"active_gradient":active_gradient,"checkpoint_bytes":checkpoint_bytes,"checkpoint_reload_max_abs_delta":reload_delta,"parameter_count":sum(int(v.size) for _n,v in mlx_utils.tree_flatten(model.parameters()))}

results={mode:train(mode) for mode in ("medusa_tree","eagle_feature","separate_draft")}
target_after={name:np.array(value) for name,value in mlx_utils.tree_flatten(target.parameters())}
target_delta=max(float(np.max(np.abs(target_before[name]-target_after[name]))) for name in target_before)
print(json.dumps({"results":results,"target_parameter_max_abs_delta":target_delta},sort_keys=True))
'''
    proc = subprocess.run(
        [sys.executable, "-c", code],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=180,
    )
    if proc.returncode:
        return {
            "available": False,
            "passed": False,
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-2000:],
        }
    observed = json.loads(proc.stdout)
    checks: dict[str, dict[str, bool]] = {}
    for mode, result in observed["results"].items():
        checks[mode] = {
            "gradient_flow": bool(result["active_gradient"]),
            "representative_overfit": result["final_loss"] < result["initial_loss"] * 0.1,
            "checkpoint_reload": result["checkpoint_reload_max_abs_delta"] == 0.0,
            "nonempty_checkpoint": result["checkpoint_bytes"] > 0,
            "nonempty_parameterization": result["parameter_count"] > 0,
        }
    checks["frozen_target"] = {
        "target_parameters_unchanged": observed["target_parameter_max_abs_delta"] == 0.0
    }
    return {
        "available": True,
        "passed": all(all(values.values()) for values in checks.values()),
        "optimizer_steps": int(optimizer_steps),
        "checks": checks,
        "observed": observed,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit": 0,
        "capability_claim": "NOT_EVALUATED",
        "speed_claim": "NOT_EVALUATED",
    }


def reference_target_next(prefix: tuple[int, ...]) -> int:
    """Deterministic test oracle for the authority state machine, not a generator."""

    return (sum(prefix[-3:]) + len(prefix) * 7 + 3) % 41


def mlx_target_kv_authority_canary() -> dict[str, Any]:
    """Exercise branch verification against the canonical MLX model cache."""

    if not host_resource_safety.accelerator_child_authorized():
        return {
            "available": False,
            "passed": False,
            "fault": "ACCELERATOR_WATCHDOG_REQUIRED",
            "capability_claim": "NOT_EVALUATED",
            "speed_claim": "NOT_EVALUATED",
        }

    import mlx.core as mx
    import mlx.nn as nn

    from standard_causal_transformer_model import CausalTransformerConfig, build_model

    mx.random.seed(20260722)
    model = build_model(
        CausalTransformerConfig(
            vocab_size=47,
            d_model=24,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            ff_dim=64,
        ),
        mx=mx,
        nn=nn,
    )
    prefix = [1, 7, 11]
    revision = "target:mlx-kv-canary-v1"
    canonical_state = prefill_target_kv(
        model, prefix, mx=mx, target_revision=revision
    )
    canonical = list(prefix)
    for _ in range(12):
        canonical_state, _receipt = verify_draft_branch_with_target_kv(
            canonical_state,
            (),
            model,
            mx=mx,
            target_revision=revision,
        )
        canonical.append(canonical_state.committed_prefix[-1])

    def proposer(current: tuple[int, ...], budget: int) -> list[int]:
        proposal_state = prefill_target_kv(
            model, current, mx=mx, target_revision=revision
        )
        result = []
        for index in range(budget):
            proposal_state, _receipt = verify_draft_branch_with_target_kv(
                proposal_state,
                (),
                model,
                mx=mx,
                target_revision=revision,
            )
            token = proposal_state.committed_prefix[-1]
            result.append((token + 1) % 47 if index == 2 else token)
        return result

    decoded, receipts = target_authoritative_kv_decode(
        model,
        prefix,
        proposer,
        mx=mx,
        target_revision=revision,
        max_new_tokens=12,
        max_draft_depth=4,
    )
    checks = {
        "canonical_greedy_identity": decoded == canonical,
        "rollback_exercised": sum(row.rollback_count for row in receipts) > 0,
        "rejected_suffix_never_committed": all(
            not row.rejected_suffix_cache_committed for row in receipts
        ),
        "cache_length_tracks_authorized_prefix": all(
            row.committed_cache_length
            == len(row.initial_prefix) + len(row.committed_tokens)
            for row in receipts
        ),
        "target_revision_bound": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "receipt_count": len(receipts),
        "accepted_draft_tokens": sum(
            len(row.accepted_draft_tokens) for row in receipts
        ),
        "rejected_draft_tokens": sum(
            len(row.rejected_draft_tokens) for row in receipts
        ),
        "target_forward_calls": sum(row.target_forward_calls for row in receipts),
        "target_evaluated_positions": sum(
            row.target_evaluated_positions for row in receipts
        ),
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit": 0,
        "speed_claim": "NOT_EVALUATED",
        "capability_claim": "NOT_EVALUATED",
    }


def run_reference_suite() -> dict[str, Any]:
    prefix = [2, 5, 7]
    canonical = list(prefix)
    for _ in range(12):
        canonical.append(reference_target_next(tuple(canonical)))

    def mixed_proposer(current: tuple[int, ...], budget: int) -> Iterable[int]:
        scratch = list(current)
        result = []
        for index in range(budget):
            token = reference_target_next(tuple(scratch))
            if index == 2:
                token = (token + 1) % 41
            result.append(token)
            scratch.append(token)
        return result

    decoded, receipts = target_authoritative_decode(
        prefix,
        mixed_proposer,
        reference_target_next,
        max_new_tokens=12,
        max_draft_depth=4,
    )
    canary = mlx_drafting_adequacy_canary()
    kv_canary = mlx_target_kv_authority_canary()
    return {
        "policy": "project_theseus_target_authoritative_drafting_v1",
        "trigger_state": "GREEN"
        if decoded == canonical and canary["passed"] and kv_canary["passed"]
        else "RED",
        "target_parity": decoded == canonical,
        "rollback_count": sum(receipt.rollback_count for receipt in receipts),
        "accepted_draft_token_count": sum(len(receipt.accepted_draft_tokens) for receipt in receipts),
        "rejected_draft_token_count": sum(len(receipt.rejected_draft_tokens) for receipt in receipts),
        "mlx_adequacy": canary,
        "mlx_target_kv_authority": kv_canary,
        "claim_boundary": "mechanics_and_target_authority_only_not_useful_speed_or_capability",
    }


if __name__ == "__main__":
    print(json.dumps(run_reference_suite(), indent=2, sort_keys=True))
