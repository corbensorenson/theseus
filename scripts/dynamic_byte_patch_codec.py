#!/usr/bin/env python3
"""Causal learned dynamic-byte patch mechanics for the T0A codec slot.

The lossless byte envelope is authoritative for reconstruction. Learned patch
latents are model inputs, not a hidden answer channel, and boundary decisions may
depend only on the visible byte prefix and prefix-derived uncertainty.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import host_resource_safety


ROOT = Path(__file__).resolve().parents[1]


class DynamicPatchFault(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class BytePatch:
    start: int
    stop: int
    payload_hex: str
    payload_sha256: str
    protected: bool


@dataclass(frozen=True)
class BytePatchEnvelope:
    schema_version: str
    codec_revision: str
    original_length: int
    original_sha256: str
    patches: tuple[BytePatch, ...]
    boundary_source: str


def _normalize_protected_spans(
    spans: Iterable[Sequence[int]], length: int
) -> tuple[tuple[int, int], ...]:
    normalized = []
    for span in spans:
        if len(span) != 2:
            raise DynamicPatchFault("protected_span_shape_invalid")
        start, stop = (int(span[0]), int(span[1]))
        if start < 0 or stop <= start or stop > length:
            raise DynamicPatchFault("protected_span_bounds_invalid")
        normalized.append((start, stop))
    normalized.sort()
    for left, right in zip(normalized, normalized[1:]):
        if left[1] > right[0]:
            raise DynamicPatchFault("protected_spans_overlap")
    return tuple(normalized)


def boundaries_from_probabilities(
    probabilities: Sequence[float],
    *,
    threshold: float,
    max_patch_bytes: int,
    protected_spans: Iterable[Sequence[int]] = (),
) -> tuple[int, ...]:
    length = len(probabilities)
    if not 0.0 <= threshold <= 1.0 or max_patch_bytes <= 0:
        raise DynamicPatchFault("boundary_policy_invalid")
    spans = _normalize_protected_spans(protected_spans, length)
    forced = {0, length}
    forbidden = set()
    for start, stop in spans:
        forced.update((start, stop))
        forbidden.update(range(start + 1, stop))
    result = [0]
    for stop in range(1, length + 1):
        must_close = stop - result[-1] >= max_patch_bytes
        learned_close = stop < length and float(probabilities[stop - 1]) >= threshold
        if stop in forced or ((must_close or learned_close) and stop not in forbidden):
            if stop > result[-1]:
                result.append(stop)
    if result[-1] != length:
        result.append(length)
    if any(right - left > max_patch_bytes for left, right in zip(result, result[1:])):
        raise DynamicPatchFault("protected_span_exceeds_patch_budget")
    return tuple(result)


def encode_lossless(
    payload: bytes,
    boundaries: Sequence[int],
    *,
    codec_revision: str,
    boundary_source: str,
    protected_spans: Iterable[Sequence[int]] = (),
) -> BytePatchEnvelope:
    points = tuple(int(value) for value in boundaries)
    if not points or points[0] != 0 or points[-1] != len(payload):
        raise DynamicPatchFault("boundary_extent_invalid")
    if any(left >= right for left, right in zip(points, points[1:])):
        raise DynamicPatchFault("boundary_order_invalid")
    spans = _normalize_protected_spans(protected_spans, len(payload))
    patches = []
    for start, stop in zip(points, points[1:]):
        part = payload[start:stop]
        protected = any(start <= span_start and stop >= span_stop for span_start, span_stop in spans)
        patches.append(
            BytePatch(
                start=start,
                stop=stop,
                payload_hex=part.hex(),
                payload_sha256=sha256_bytes(part),
                protected=protected,
            )
        )
    return BytePatchEnvelope(
        schema_version="1.0.0",
        codec_revision=codec_revision,
        original_length=len(payload),
        original_sha256=sha256_bytes(payload),
        patches=tuple(patches),
        boundary_source=boundary_source,
    )


def decode_lossless(envelope: BytePatchEnvelope | dict[str, Any]) -> bytes:
    record = asdict(envelope) if isinstance(envelope, BytePatchEnvelope) else dict(envelope)
    if record.get("schema_version") != "1.0.0":
        raise DynamicPatchFault("envelope_schema_unsupported")
    cursor = 0
    parts = []
    for raw_patch in record.get("patches", []):
        patch = dict(raw_patch)
        if int(patch.get("start", -1)) != cursor or int(patch.get("stop", -1)) <= cursor:
            raise DynamicPatchFault("patch_extent_invalid")
        try:
            payload = bytes.fromhex(str(patch["payload_hex"]))
        except (KeyError, ValueError) as exc:
            raise DynamicPatchFault("patch_payload_invalid") from exc
        if len(payload) != int(patch["stop"]) - int(patch["start"]):
            raise DynamicPatchFault("patch_length_mismatch")
        if sha256_bytes(payload) != patch.get("payload_sha256"):
            raise DynamicPatchFault("patch_checksum_mismatch")
        parts.append(payload)
        cursor = int(patch["stop"])
    joined = b"".join(parts)
    if cursor != int(record.get("original_length", -1)):
        raise DynamicPatchFault("envelope_length_mismatch")
    if sha256_bytes(joined) != record.get("original_sha256"):
        raise DynamicPatchFault("envelope_checksum_mismatch")
    return joined


def heuristic_boundary_targets(payload: bytes, *, max_patch_bytes: int) -> list[float]:
    """Private canary supervision; not a production or capability oracle."""

    punctuation = set(b" \t\r\n()[]{}:;,.+-=*/<>\"'`_")
    targets = []
    since = 0
    for value in payload:
        since += 1
        boundary = value in punctuation or since >= max_patch_bytes
        targets.append(1.0 if boundary else 0.0)
        if boundary:
            since = 0
    if targets:
        targets[-1] = 1.0
    return targets


def patch_ids_from_boundaries(boundaries: Sequence[int], length: int) -> list[int]:
    ids = [0] * length
    for patch_id, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
        ids[int(start) : int(stop)] = [patch_id] * (int(stop) - int(start))
    return ids


def build_dynamic_patch_model(
    *,
    d_model: int,
    patch_hidden_dim: int,
    max_patches: int,
    max_patch_bytes: int,
    mx: Any,
    nn: Any,
) -> Any:
    if d_model <= 0 or patch_hidden_dim <= 0 or max_patches <= 0 or max_patch_bytes <= 0:
        raise ValueError("dynamic patch dimensions must be positive")

    class DynamicPatchModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.byte_embedding = nn.Embedding(256, d_model)
            self.within_patch_position = nn.Embedding(max_patch_bytes, d_model)
            self.boundary_norm = nn.RMSNorm(d_model * 2 + 1)
            self.boundary_hidden = nn.Linear(d_model * 2 + 1, patch_hidden_dim)
            self.boundary_output = nn.Linear(patch_hidden_dim, 1)
            self.patch_projection = nn.Linear(d_model, d_model, bias=False)
            self.local_previous_projection = nn.Linear(d_model, d_model, bias=False)
            self.local_norm = nn.RMSNorm(d_model)
            self.byte_decoder = nn.Linear(d_model, 256, bias=False)

        def boundary_logits(self, byte_ids: Any, uncertainty: Any) -> Any:
            embedded = self.byte_embedding(byte_ids)
            prefix_sum = mx.cumsum(embedded, axis=1) - embedded
            counts = mx.maximum(
                mx.arange(int(byte_ids.shape[1]), dtype=mx.float32), 1.0
            )[None, :, None]
            prior = prefix_sum / counts
            features = mx.concatenate([embedded, prior, uncertainty[:, :, None]], axis=-1)
            return self.boundary_output(
                nn.silu(self.boundary_hidden(self.boundary_norm(features)))
            )[..., 0]

        def __call__(
            self,
            byte_ids: Any,
            patch_ids: Any,
            within_patch_positions: Any,
            uncertainty: Any,
            byte_mask: Any | None = None,
        ) -> tuple[Any, Any, Any]:
            embedded = self.byte_embedding(byte_ids)
            if byte_mask is None:
                byte_mask = mx.ones(byte_ids.shape, dtype=mx.float32)
            active_patch_count = int(mx.max(patch_ids).item()) + 1
            if active_patch_count <= 0 or active_patch_count > max_patches:
                raise DynamicPatchFault("active_patch_count_invalid")
            one_hot = (
                patch_ids[:, :, None]
                == mx.arange(active_patch_count, dtype=mx.int32)[None, None, :]
            ).astype(mx.float32) * byte_mask[:, :, None]
            counts = mx.maximum(mx.sum(one_hot, axis=1), 1.0)
            patch_latents = mx.matmul(one_hot.transpose(0, 2, 1), embedded)
            patch_latents = self.patch_projection(patch_latents / counts[:, :, None])
            return (
                self.decode_patch_latents(
                    patch_latents,
                    patch_ids,
                    within_patch_positions,
                    byte_ids,
                ),
                self.boundary_logits(byte_ids, uncertainty),
                patch_latents,
            )

        def decode_patch_latents(
            self,
            patch_latents: Any,
            patch_ids: Any,
            within_patch_positions: Any,
            byte_ids: Any,
        ) -> Any:
            """Decode bytes causally from prior-patch state and local byte history.

            The patch latent supplied by the caller is shifted by one patch in the
            causal candidate. The local path sees only the immediately preceding
            byte when it belongs to the same patch; it never embeds the byte being
            predicted or any future byte.
            """

            byte_latents = mx.take_along_axis(
                patch_latents,
                patch_ids[:, :, None],
                axis=1,
            )
            embedded = self.byte_embedding(byte_ids)
            zero = mx.zeros(
                (int(byte_ids.shape[0]), 1, int(embedded.shape[-1])),
                dtype=embedded.dtype,
            )
            previous = mx.concatenate([zero, embedded[:, :-1, :]], axis=1)
            same_patch = mx.concatenate(
                [
                    mx.zeros((int(byte_ids.shape[0]), 1), dtype=mx.float32),
                    (patch_ids[:, 1:] == patch_ids[:, :-1]).astype(mx.float32),
                ],
                axis=1,
            )
            local = self.local_previous_projection(
                previous * same_patch[:, :, None]
            )
            hidden = self.local_norm(
                byte_latents
                + self.within_patch_position(within_patch_positions)
                + local
            )
            return self.byte_decoder(hidden)

        def decode_next_byte(
            self,
            prior_patch_latent: Any,
            within_patch_position: Any,
            previous_byte_id: Any | None,
        ) -> Any:
            """Single-position inference equivalent of the teacher-forced local path."""

            local = mx.zeros_like(prior_patch_latent)
            if previous_byte_id is not None:
                local = self.local_previous_projection(
                    self.byte_embedding(previous_byte_id)
                )
            hidden = self.local_norm(
                prior_patch_latent
                + self.within_patch_position(within_patch_position)
                + local
            )
            return self.byte_decoder(hidden)

    return DynamicPatchModel()


def build_dynamic_patch_causal_candidate(
    *,
    d_model: int,
    patch_hidden_dim: int,
    max_patches: int,
    max_patch_bytes: int,
    vocab_size: int,
    mx: Any,
    nn: Any,
) -> Any:
    """Compose the codec with the canonical causal core without current-patch leakage."""

    from standard_causal_transformer_model import CausalTransformerConfig, build_model

    class DynamicPatchCausalCandidate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.codec = build_dynamic_patch_model(
                d_model=d_model,
                patch_hidden_dim=patch_hidden_dim,
                max_patches=max_patches,
                max_patch_bytes=max_patch_bytes,
                mx=mx,
                nn=nn,
            )
            self.core = build_model(
                CausalTransformerConfig(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    num_layers=1,
                    num_heads=4,
                    num_kv_heads=2,
                    ff_dim=d_model * 2,
                ),
                mx=mx,
                nn=nn,
            )
            self.bos_patch = mx.zeros((1, 1, d_model), dtype=mx.float32)

        def __call__(
            self,
            byte_ids: Any,
            patch_ids: Any,
            within_patch_positions: Any,
            uncertainty: Any,
            byte_mask: Any | None = None,
        ) -> tuple[Any, Any, Any]:
            _local_logits, boundary_logits, patch_latents = self.codec(
                byte_ids,
                patch_ids,
                within_patch_positions,
                uncertainty,
                byte_mask,
            )
            batch = int(byte_ids.shape[0])
            shifted = mx.concatenate(
                [
                    mx.broadcast_to(self.bos_patch, (batch, 1, d_model)),
                    patch_latents[:, :-1, :],
                ],
                axis=1,
            )
            dummy_tokens = mx.zeros(
                (batch, int(shifted.shape[1])), dtype=mx.int32
            )
            _logits, _cache, aux = self.core(
                dummy_tokens,
                input_embeddings=shifted,
                return_training_aux=True,
            )
            byte_logits = self.codec.decode_patch_latents(
                aux["final_hidden"],
                patch_ids,
                within_patch_positions,
                byte_ids,
            )
            return byte_logits, boundary_logits, patch_latents

    return DynamicPatchCausalCandidate()


def mlx_dynamic_patch_adequacy_canary(*, optimizer_steps: int = 64) -> dict[str, Any]:
    if not host_resource_safety.accelerator_child_authorized():
        return {
            "available": False,
            "passed": False,
            "fault": "ACCELERATOR_WATCHDOG_REQUIRED",
            "optimizer_steps": 0,
            "capability_claim": "NOT_EVALUATED",
            "compression_claim": "NOT_EVALUATED",
        }
    payload = {"scripts": str(ROOT / "scripts"), "optimizer_steps": int(optimizer_steps)}
    code = r'''
import json,sys,tempfile
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
p=json.loads(sys.stdin.read());sys.path.insert(0,p["scripts"])
from dynamic_byte_patch_codec import build_dynamic_patch_causal_candidate,heuristic_boundary_targets,boundaries_from_probabilities,patch_ids_from_boundaries,encode_lossless,decode_lossless

rows=[b"Explain why bounded evidence matters.",b"def clamp(x, lo, hi):\n    return max(lo, min(x, hi))",b"const add = (a, b) => a + b;",b"fn twice(x: i32) -> i32 { x * 2 }"]
width=max(len(row) for row in rows);max_patch=12;max_patches=width
ids=[];mask=[];targets=[];patch_ids=[];within_positions=[];uncertainty=[]
for row in rows:
    padded=row+b"\x00"*(width-len(row)); ids.append(list(padded)); mask.append([1.0]*len(row)+[0.0]*(width-len(row)))
    boundary=heuristic_boundary_targets(row,max_patch_bytes=max_patch)+[0.0]*(width-len(row));targets.append(boundary)
    points=boundaries_from_probabilities(boundary[:len(row)],threshold=0.5,max_patch_bytes=max_patch); assignment=patch_ids_from_boundaries(points,len(row)); patch_ids.append(assignment+[assignment[-1] if assignment else 0]*(width-len(row)))
    positions=[]
    for start,stop in zip(points,points[1:]): positions.extend(range(stop-start))
    within_positions.append(positions+[0]*(width-len(row)))
    counts={}; row_uncertainty=[]
    for value in padded:
        counts[value]=counts.get(value,0)+1; row_uncertainty.append(1.0/counts[value])
    uncertainty.append(row_uncertainty)
ids=mx.array(ids,dtype=mx.int32);mask=mx.array(mask,dtype=mx.float32);targets=mx.array(targets,dtype=mx.float32);patch_ids=mx.array(patch_ids,dtype=mx.int32);within_positions=mx.array(within_positions,dtype=mx.int32);uncertainty=mx.array(uncertainty,dtype=mx.float32)
overfit_rows=mx.array([[1.0],[0.0],[0.0],[0.0]],dtype=mx.float32)
mx.random.seed(20260722);model=build_dynamic_patch_causal_candidate(d_model=24,patch_hidden_dim=32,max_patches=max_patches,max_patch_bytes=max_patch,vocab_size=256,mx=mx,nn=nn)
def loss_fn(candidate):
    byte_logits,boundary_logits,_patches=candidate(ids,patch_ids,within_positions,uncertainty)
    predictive_mask=mask*(patch_ids>0).astype(mx.float32)*overfit_rows
    byte_loss=mx.sum(nn.losses.cross_entropy(byte_logits,ids,reduction="none")*predictive_mask)/mx.maximum(mx.sum(predictive_mask),1.0)
    boundary_loss=mx.sum(nn.losses.binary_cross_entropy(boundary_logits,targets,with_logits=True,reduction="none")*mask)/mx.sum(mask)
    return byte_loss+boundary_loss
optimizer=optim.AdamW(learning_rate=0.02);value_and_grad=nn.value_and_grad(model,loss_fn)
initial=loss_fn(model);mx.eval(initial);active_gradient=False
for _ in range(p["optimizer_steps"]):
    loss,grads=value_and_grad(model);optimizer.update(model,grads);mx.eval(loss,model.parameters(),optimizer.state)
    active_gradient=active_gradient or any(float(mx.max(mx.abs(g)).item())>0 for _n,g in mlx_utils.tree_flatten(grads))
final=loss_fn(model);byte_logits,boundary_logits,patch_latents=model(ids,patch_ids,within_positions,uncertainty);mx.eval(final,byte_logits,boundary_logits,patch_latents)

prefix=rows[0][:12];left=prefix+b" first future";right=prefix+b" second future"
def logits_for(row):
    array=mx.array([list(row)],dtype=mx.int32);u=mx.ones(array.shape,dtype=mx.float32);value=model.codec.boundary_logits(array,u);mx.eval(value);return np.array(value)
left_logits=logits_for(left);right_logits=logits_for(right);causal_delta=float(np.max(np.abs(left_logits[:,:len(prefix)]-right_logits[:,:len(prefix)])))
before=np.array(boundary_logits);saved=[(n,np.array(v)) for n,v in mlx_utils.tree_flatten(model.parameters()) if ".boundary_" in n]
base_byte_logits=np.array(byte_logits)
altered_ids=np.array(ids); changed_position=5; altered_ids[0,changed_position]=(altered_ids[0,changed_position]+17)%256
altered_output=model(mx.array(altered_ids,dtype=mx.int32),patch_ids,within_positions,uncertainty);mx.eval(altered_output)
local_causal_delta=float(np.max(np.abs(base_byte_logits[0,:changed_position+1]-np.array(altered_output[0])[0,:changed_position+1])))
for name,value in mlx_utils.tree_flatten(model.parameters()):
    if name.endswith("boundary_output.weight"): model.codec.boundary_output.weight=mx.zeros_like(value)
    if name.endswith("boundary_output.bias"): model.codec.boundary_output.bias=mx.zeros_like(value)
intervened=model.codec.boundary_logits(ids,uncertainty);mx.eval(intervened);intervention_delta=float(np.max(np.abs(before-np.array(intervened))))
for name,value in saved:
    if name.endswith("boundary_output.weight"): model.codec.boundary_output.weight=mx.array(value)
    elif name.endswith("boundary_output.bias"): model.codec.boundary_output.bias=mx.array(value)
restored=model(ids,patch_ids,within_positions,uncertainty);mx.eval(restored)
with tempfile.TemporaryDirectory(prefix="theseus-dynamic-patch-") as tmp:
    path=Path(tmp)/"codec.safetensors";model.save_weights(str(path));checkpoint_bytes=path.stat().st_size
    reloaded=build_dynamic_patch_causal_candidate(d_model=24,patch_hidden_dim=32,max_patches=max_patches,max_patch_bytes=max_patch,vocab_size=256,mx=mx,nn=nn);reloaded.load_weights(str(path));again=reloaded(ids,patch_ids,within_positions,uncertainty);mx.eval(again)
    reload_delta=max(float(mx.max(mx.abs(a-b)).item()) for a,b in zip(restored,again))
probabilities=1/(1+np.exp(-np.array(restored[1])))
roundtrip=True;patch_counts=[]
for index,row in enumerate(rows):
    points=boundaries_from_probabilities(probabilities[index,:len(row)].tolist(),threshold=0.5,max_patch_bytes=max_patch)
    envelope=encode_lossless(row,points,codec_revision="dynamic-byte-patch-canary-v1",boundary_source="causal_learned_prefix")
    roundtrip=roundtrip and decode_lossless(envelope)==row;patch_counts.append(len(points)-1)
print(json.dumps({"initial_loss":float(initial.item()),"final_loss":float(final.item()),"active_gradient":active_gradient,"causal_prefix_max_abs_delta":causal_delta,"local_byte_causal_max_abs_delta":local_causal_delta,"boundary_intervention_max_abs_delta":intervention_delta,"checkpoint_reload_max_abs_delta":reload_delta,"checkpoint_bytes":checkpoint_bytes,"lossless_roundtrip":roundtrip,"raw_bytes":sum(len(row) for row in rows),"patch_positions":sum(patch_counts),"parameter_count":sum(int(v.size) for _n,v in mlx_utils.tree_flatten(model.parameters()))},sort_keys=True))
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
    checks = {
        "gradient_flow": observed["active_gradient"],
        "representative_overfit": observed["final_loss"] < observed["initial_loss"] * 0.1,
        "causal_prefix_invariance": observed["causal_prefix_max_abs_delta"] == 0.0,
        "within_patch_local_causality": observed["local_byte_causal_max_abs_delta"] == 0.0,
        "boundary_intervention": observed["boundary_intervention_max_abs_delta"] > 0.0,
        "checkpoint_reload": observed["checkpoint_reload_max_abs_delta"] == 0.0,
        "lossless_byte_roundtrip": observed["lossless_roundtrip"],
        "sequence_contraction": observed["patch_positions"] < observed["raw_bytes"],
    }
    return {
        "available": True,
        "passed": all(checks.values()),
        "optimizer_steps": int(optimizer_steps),
        "checks": checks,
        "observed": observed,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "fallback_template_router_tool_credit": 0,
        "capability_claim": "NOT_EVALUATED",
        "compression_claim": "NOT_EVALUATED",
    }


def run_reference_suite() -> dict[str, Any]:
    payload = b"name=Jos\xc3\xa9\x00\xff\nvalue={x + 1}"
    spans = ((5, 10),)
    probabilities = [0.1] * len(payload)
    probabilities[3] = probabilities[10] = probabilities[18] = 0.95
    boundaries = boundaries_from_probabilities(
        probabilities,
        threshold=0.5,
        max_patch_bytes=12,
        protected_spans=spans,
    )
    envelope = encode_lossless(
        payload,
        boundaries,
        codec_revision="dynamic-byte-patch-v1",
        boundary_source="causal_learned_prefix",
        protected_spans=spans,
    )
    canary = mlx_dynamic_patch_adequacy_canary()
    return {
        "policy": "project_theseus_dynamic_byte_patch_codec_v1",
        "trigger_state": "GREEN" if decode_lossless(envelope) == payload and canary["passed"] else "RED",
        "lossless_roundtrip": decode_lossless(envelope) == payload,
        "patch_count": len(envelope.patches),
        "protected_patch_count": sum(patch.protected for patch in envelope.patches),
        "mlx_adequacy": canary,
        "claim_boundary": "codec_mechanics_only_not_dynamic_representation_superiority_or_capability",
    }


if __name__ == "__main__":
    print(json.dumps(run_reference_suite(), indent=2, sort_keys=True))
