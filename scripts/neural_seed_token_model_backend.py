#!/usr/bin/env python3
"""Token-model backend and training helpers for the neural seed comparator.

This module owns the matched transformer/SymLiquid token decoder classes,
checkpoint/vocab initialization, private supervised training loops, grammar
auxiliary loss, and parameter update accounting. It does not orchestrate public
calibration, emit candidate claims, or serve external inference.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_artifact_admission  # noqa: E402
from neural_seed_code_proposer_comparator import (  # noqa: E402
    count_params,
    dict_or_empty,
    get_path,
    ratio,
    rel,
    stable_hash,
)
from neural_seed_token_decoder_support import (  # noqa: E402
    syntax_complete_body_prefix,
    token_allowed_by_policy,
)
from narrow_corpus_pretraining_spine import (  # noqa: E402
    apply_merges as apply_bpe_merges,
    basic_tokens as bpe_basic_tokens,
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def stable_hash_file(path: Path) -> str:
    if not path.exists():
        return ""
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class TransformerTokenDecoder:
    def __new__(cls, *args: Any, torch: Any, nn: Any, **kwargs: Any) -> Any:
        class _Model(nn.Module):
            def __init__(
                self,
                source_vocab_size: int,
                target_vocab_size: int,
                *,
                d_model: int,
                nhead: int,
                num_layers: int,
                dim_feedforward: int,
                max_source_len: int,
                max_target_len: int,
                plan_router_scale: float = 1.0,
            ) -> None:
                super().__init__()
                self.plan_router_scale = float(plan_router_scale)
                self.source_embedding = nn.Embedding(source_vocab_size, d_model, padding_idx=0)
                self.target_embedding = nn.Embedding(target_vocab_size, d_model, padding_idx=0)
                self.source_position = nn.Parameter(torch.zeros(1, max_source_len, d_model))
                self.target_position = nn.Parameter(torch.zeros(1, max_target_len, d_model))
                source_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                )
                target_layer = nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                )
                self.source_encoder = nn.TransformerEncoder(source_layer, num_layers=num_layers, enable_nested_tensor=False)
                self.target_decoder = nn.TransformerDecoder(target_layer, num_layers=num_layers)
                self.output = nn.Linear(d_model, target_vocab_size)
                self.plan_router = nn.Linear(d_model, target_vocab_size)
                if self.plan_router_scale == 0.0:
                    for param in self.plan_router.parameters():
                        param.requires_grad_(False)

            def source_memory(self, src: Any) -> tuple[Any, Any]:
                mask = src.ne(0)
                h = self.source_embedding(src) + self.source_position[:, : src.shape[1], :]
                encoded = self.source_encoder(h, src_key_padding_mask=~mask)
                return encoded, mask

            def encode_source(self, src: Any) -> Any:
                encoded, mask = self.source_memory(src)
                denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
                return (encoded * mask.unsqueeze(-1)).sum(dim=1) / denom

            def semantic_plan_logits(self, src: Any) -> Any:
                return self.plan_router(self.encode_source(src))

            def init_decode_state(self, src: Any) -> dict[str, Any]:
                memory, src_mask = self.source_memory(src)
                encoded_context = (memory * src_mask.unsqueeze(-1)).sum(dim=1) / src_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
                return {
                    "memory": memory,
                    "src_mask": src_mask,
                    "encoded_context": encoded_context,
                    "generated": [],
                    "step": 0,
                }

            def decode_next_logits(self, token_id: int, state: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
                memory = state["memory"]
                src_mask = state["src_mask"]
                generated = list(state.get("generated") or [])
                generated.append(int(token_id))
                tgt_in = torch.tensor([generated], dtype=torch.long, device=memory.device)
                h = self.target_embedding(tgt_in) + self.target_position[:, : tgt_in.shape[1], :]
                causal = torch.triu(
                    torch.ones((tgt_in.shape[1], tgt_in.shape[1]), dtype=torch.bool, device=tgt_in.device),
                    diagonal=1,
                )
                decoded = self.target_decoder(
                    h,
                    memory,
                    tgt_mask=causal,
                    memory_key_padding_mask=~src_mask,
                )
                logits = self.output(decoded[:, -1, :])
                step = int(state.get("step") or 0)
                if step == 0 and self.plan_router_scale != 0.0:
                    logits = logits + self.plan_router(state["encoded_context"]) * self.plan_router_scale
                return logits, {
                    "memory": memory,
                    "src_mask": src_mask,
                    "encoded_context": state["encoded_context"],
                    "generated": generated,
                    "step": step + 1,
                }

            def forward(self, src: Any, tgt_in: Any) -> Any:
                memory, src_mask = self.source_memory(src)
                encoded_context = (memory * src_mask.unsqueeze(-1)).sum(dim=1) / src_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
                h = self.target_embedding(tgt_in) + self.target_position[:, : tgt_in.shape[1], :]
                causal = torch.triu(
                    torch.ones((tgt_in.shape[1], tgt_in.shape[1]), dtype=torch.bool, device=tgt_in.device),
                    diagonal=1,
                )
                decoded = self.target_decoder(
                    h,
                    memory,
                    tgt_mask=causal,
                    memory_key_padding_mask=~src_mask,
                )
                logits = self.output(decoded)
                if self.plan_router_scale != 0.0 and logits.shape[1] > 0:
                    logits = logits.clone()
                    logits[:, 0, :] = logits[:, 0, :] + self.plan_router(encoded_context) * self.plan_router_scale
                return logits

        return _Model(*args, **kwargs)


class SymLiquidTokenDecoder:
    def __new__(cls, *args: Any, torch: Any, nn: Any, **kwargs: Any) -> Any:
        class _Model(nn.Module):
            def __init__(
                self,
                source_vocab_size: int,
                target_vocab_size: int,
                *,
                hidden_dim: int,
                reservoir_dim: int,
                hv_dim: int,
                plan_router_scale: float = 1.0,
            ) -> None:
                super().__init__()
                self.plan_router_scale = float(plan_router_scale)
                self.source_embedding = nn.Embedding(source_vocab_size, hidden_dim, padding_idx=0)
                self.target_embedding = nn.Embedding(target_vocab_size, hidden_dim, padding_idx=0)
                self.liquid_in = nn.Linear(hidden_dim, hidden_dim)
                self.liquid_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.tau = nn.Linear(hidden_dim, hidden_dim)
                self.reservoir = nn.Linear(hidden_dim, reservoir_dim)
                self.vsa = nn.Linear(reservoir_dim, hv_dim, bias=False)
                self.source_mean_to_vsa = nn.Linear(hidden_dim, hv_dim)
                self.source_max_to_vsa = nn.Linear(hidden_dim, hv_dim, bias=False)
                self.source_context_norm = nn.LayerNorm(hv_dim)
                self.context_to_hidden = nn.Linear(hv_dim, hidden_dim)
                self.decode_in = nn.Linear(hidden_dim + hv_dim, hidden_dim)
                self.decode_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.decode_tau = nn.Linear(hidden_dim + hv_dim, hidden_dim)
                self.output = nn.Linear(hidden_dim, target_vocab_size)
                self.plan_router = nn.Linear(hv_dim, target_vocab_size)

            def encode_source(self, src: Any) -> Any:
                emb = self.source_embedding(src)
                mask = src.ne(0).float().unsqueeze(-1)
                h = emb.new_zeros((src.shape[0], self.liquid_h.out_features))
                memory = emb.new_zeros((src.shape[0], self.vsa.out_features))
                mean_memory = emb.new_zeros((src.shape[0], self.vsa.out_features))
                mean_embedding = emb.new_zeros((src.shape[0], emb.shape[-1]))
                max_embedding = emb.new_full((src.shape[0], emb.shape[-1]), -1e4)
                token_count = emb.new_zeros((src.shape[0], 1))
                for t in range(src.shape[1]):
                    xt = emb[:, t, :]
                    m = mask[:, t, :]
                    candidate = torch.tanh(self.liquid_in(xt) + self.liquid_h(h))
                    alpha = torch.sigmoid(self.tau(xt))
                    h_new = (1.0 - alpha) * h + alpha * candidate
                    h = m * h_new + (1.0 - m) * h
                    hv = torch.tanh(self.vsa(torch.tanh(self.reservoir(h))))
                    memory = m * (0.97 * memory + hv) + (1.0 - m) * memory
                    mean_memory = mean_memory + m * hv
                    mean_embedding = mean_embedding + m * xt
                    max_embedding = torch.where(m.bool(), torch.maximum(max_embedding, xt), max_embedding)
                    token_count = token_count + m
                decayed = memory / memory.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                mean = mean_memory / token_count.clamp(min=1.0)
                mean = mean / mean.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                mean_embedding = mean_embedding / token_count.clamp(min=1.0)
                max_embedding = torch.where(token_count.gt(0), max_embedding, emb.new_zeros(max_embedding.shape))
                source_bag = torch.tanh(self.source_mean_to_vsa(mean_embedding) + self.source_max_to_vsa(max_embedding))
                source_bag = source_bag / source_bag.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                context = self.source_context_norm((0.45 * decayed) + (0.25 * mean) + (0.30 * source_bag))
                return context / context.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            def semantic_plan_logits(self, src: Any) -> Any:
                return self.plan_router(self.encode_source(src))

            def init_decode_state(self, src: Any) -> dict[str, Any]:
                context = self.encode_source(src)
                return {
                    "context": context,
                    "h": torch.tanh(self.context_to_hidden(context)),
                    "step": 0,
                }

            def decode_next_logits(self, token_id: int, state: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
                context = state["context"]
                h = state["h"]
                token = torch.tensor([int(token_id)], dtype=torch.long, device=context.device)
                emb = self.target_embedding(token)
                xt = torch.cat([emb, context], dim=-1)
                candidate = torch.tanh(self.decode_in(xt) + self.decode_h(h))
                alpha = torch.sigmoid(self.decode_tau(xt))
                next_h = (1.0 - alpha) * h + alpha * candidate
                logits = self.output(next_h)
                step = int(state.get("step") or 0)
                if step == 0 and self.plan_router_scale != 0.0:
                    logits = logits + self.plan_router(context) * self.plan_router_scale
                return logits, {"context": context, "h": next_h, "step": step + 1}

            def forward(self, src: Any, tgt_in: Any) -> Any:
                context = self.encode_source(src)
                emb = self.target_embedding(tgt_in)
                h = torch.tanh(self.context_to_hidden(context))
                outs = []
                for t in range(tgt_in.shape[1]):
                    xt = torch.cat([emb[:, t, :], context], dim=-1)
                    candidate = torch.tanh(self.decode_in(xt) + self.decode_h(h))
                    alpha = torch.sigmoid(self.decode_tau(xt))
                    h = (1.0 - alpha) * h + alpha * candidate
                    logits = self.output(h)
                    if t == 0 and self.plan_router_scale != 0.0:
                        logits = logits + self.plan_router(context) * self.plan_router_scale
                    outs.append(logits.unsqueeze(1))
                return torch.cat(outs, dim=1)

        return _Model(*args, **kwargs)


def choose_sym_token_dims(
    config: dict[str, Any],
    *,
    source_vocab_size: int,
    target_vocab_size: int,
    target_params: int,
    torch: Any,
    nn: Any,
) -> tuple[dict[str, int], int]:
    arm_cfg = dict_or_empty(dict_or_empty(config.get("arms")).get("symliquid_style"))
    best_dims = {"hidden_dim": 40, "reservoir_dim": 40, "hv_dim": 40}
    best_count = 0
    best_delta = float("inf")
    for hidden in arm_cfg.get("hidden_dim_candidates", [40]):
        for reservoir in arm_cfg.get("reservoir_dim_candidates", [40]):
            for hv in arm_cfg.get("hv_dim_candidates", [40]):
                model = SymLiquidTokenDecoder(
                    source_vocab_size,
                    target_vocab_size,
                    hidden_dim=int(hidden),
                    reservoir_dim=int(reservoir),
                    hv_dim=int(hv),
                    torch=torch,
                    nn=nn,
                )
                count = count_params(model)
                delta = abs(count - target_params) / max(1, target_params)
                if delta < best_delta:
                    best_delta = delta
                    best_count = count
                    best_dims = {"hidden_dim": int(hidden), "reservoir_dim": int(reservoir), "hv_dim": int(hv)}
    return best_dims, best_count


def training_budget_for_arm(config: dict[str, Any], arm_id: str, base_budget: dict[str, Any]) -> dict[str, Any]:
    budget = dict(base_budget)
    overrides = dict_or_empty(get_path(config, ["arms", arm_id, "training_budget_overrides"], {}))
    if overrides:
        budget.update(overrides)
        budget["arm_training_budget_override_applied"] = True
        budget["arm_training_budget_override_arm"] = arm_id
    return budget


def apply_arm_training_row_cap(
    source_rows: list[list[int]],
    target_rows: list[list[int]],
    budget: dict[str, Any],
) -> tuple[list[list[int]], list[list[int]]]:
    cap = int(budget.get("max_train_rows") or 0)
    if cap > 0 and len(source_rows) > cap:
        return source_rows[:cap], target_rows[:cap]
    return source_rows, target_rows


def pretraining_initialization_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict_or_empty(config.get("pretraining_initialization"))


def build_pretraining_initializers(config: dict[str, Any], *, torch: Any, device: Any) -> dict[str, Any]:
    cfg = pretraining_initialization_config(config)
    if not bool(cfg.get("enabled", False)):
        return {
            "enabled": False,
            "active": False,
            "reason": "pretraining_initialization_disabled",
            "by_arm": {},
        }
    tokenizer_path = resolve(str(cfg.get("tokenizer") or ""))
    tokenizer = read_json(tokenizer_path)
    by_arm: dict[str, Any] = {}
    for arm_id, checkpoint_rel in dict_or_empty(cfg.get("checkpoints")).items():
        checkpoint_path = resolve(str(checkpoint_rel))
        by_arm[str(arm_id)] = load_pretraining_initializer_for_arm(
            arm_id=str(arm_id),
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_path,
            checkpoint_path=checkpoint_path,
            torch=torch,
            device=device,
            admission_config=cfg,
        )
    return {
        "enabled": True,
        "active": any(bool(row.get("active")) for row in by_arm.values()),
        "policy": str(cfg.get("policy") or "from_scratch_narrow_corpus_pretraining_embedding_init_v1"),
        "tokenizer": rel(tokenizer_path),
        "tokenizer_vocab_size": len(dict_or_empty(tokenizer.get("vocab"))),
        "tokenizer_merge_count": len(tokenizer.get("merges") if isinstance(tokenizer.get("merges"), list) else []),
        "by_arm": by_arm,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "open_or_pretrained_model_weights_used": False,
    }


def load_pretraining_initializer_for_arm(
    *,
    arm_id: str,
    tokenizer: dict[str, Any],
    tokenizer_path: Path,
    checkpoint_path: Path,
    torch: Any,
    device: Any,
    admission_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not tokenizer or not dict_or_empty(tokenizer.get("vocab")):
        return {"active": False, "arm_id": arm_id, "reason": "missing_or_invalid_tokenizer", "tokenizer": rel(tokenizer_path)}
    if not checkpoint_path.exists():
        return {"active": False, "arm_id": arm_id, "reason": "missing_checkpoint", "checkpoint": rel(checkpoint_path)}
    admission = theseus_artifact_admission.admit_from_config(checkpoint_path, admission_config or {})
    if admission.get("required") and not admission.get("admitted"):
        return {
            "active": False,
            "arm_id": arm_id,
            "reason": f"artifact_admission_rejected:{admission.get('reason')}",
            "checkpoint": rel(checkpoint_path),
            "artifact_admission": admission,
        }
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:  # pragma: no cover - depends on local torch serialization.
        return {
            "active": False,
            "arm_id": arm_id,
            "reason": f"checkpoint_load_failed:{exc.__class__.__name__}",
            "checkpoint": rel(checkpoint_path),
        }
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else {}
    embedding_key = str(checkpoint.get("embedding_key") or "embedding.weight") if isinstance(checkpoint, dict) else "embedding.weight"
    embedding = state.get(embedding_key) if isinstance(state, dict) else None
    if embedding is None:
        return {
            "active": False,
            "arm_id": arm_id,
            "reason": "missing_embedding_weight",
            "checkpoint": rel(checkpoint_path),
            "embedding_key": embedding_key,
        }
    return {
        "active": True,
        "arm_id": arm_id,
        "checkpoint": rel(checkpoint_path),
        "checkpoint_sha256": stable_hash_file(checkpoint_path),
        "embedding_key": embedding_key,
        "embedding": embedding.detach().to(device).float(),
        "embedding_shape": list(embedding.shape),
        "tokenizer": tokenizer,
        "tokenizer_path": rel(tokenizer_path),
        "tokenizer_sha256": stable_hash_file(tokenizer_path),
        "from_scratch": bool(checkpoint.get("open_or_pretrained_model_weights_used") is False) if isinstance(checkpoint, dict) else True,
        "public_training_rows": int(checkpoint.get("public_training_rows") or 0) if isinstance(checkpoint, dict) else 0,
        "external_inference_calls": int(checkpoint.get("external_inference_calls") or 0) if isinstance(checkpoint, dict) else 0,
        "artifact_admission": admission,
    }


def apply_pretraining_initialization(
    model: Any,
    *,
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    initializer: dict[str, Any] | None,
    torch: Any,
) -> dict[str, Any]:
    initializer = dict_or_empty(initializer)
    if not bool(initializer.get("active")):
        return {
            "enabled": False,
            "active": False,
            "reason": initializer.get("reason") or "no_active_pretraining_initializer_for_arm",
        }
    tokenizer = dict_or_empty(initializer.get("tokenizer"))
    embedding = initializer.get("embedding")
    if embedding is None:
        return {"enabled": True, "active": False, "reason": "initializer_embedding_missing"}
    source_summary = initialize_embedding_module_from_bpe(
        model.source_embedding,
        source_vocab,
        tokenizer,
        embedding,
        token_kind="source",
        torch=torch,
    )
    target_summary = initialize_embedding_module_from_bpe(
        model.target_embedding,
        target_vocab,
        tokenizer,
        embedding,
        token_kind="target",
        torch=torch,
    )
    inventory = model_parameter_inventory(model)
    transferred_parameter_count = int(source_summary.get("copied_parameter_count") or 0) + int(
        target_summary.get("copied_parameter_count") or 0
    )
    return {
        "enabled": True,
        "active": True,
        "policy": "from_scratch_narrow_corpus_bpe_embedding_initialization_v1",
        "checkpoint": initializer.get("checkpoint"),
        "checkpoint_sha256": initializer.get("checkpoint_sha256"),
        "tokenizer": initializer.get("tokenizer_path"),
        "tokenizer_sha256": initializer.get("tokenizer_sha256"),
        "pretraining_embedding_shape": initializer.get("embedding_shape"),
        "source_embedding": source_summary,
        "target_embedding": target_summary,
        "trainable_parameter_count": inventory["trainable_parameter_count"],
        "transferred_parameter_count": transferred_parameter_count,
        "transferred_state_fraction_into_generator": ratio(
            transferred_parameter_count,
            int(inventory["trainable_parameter_count"] or 0),
        ),
        "non_embedding_transferred_parameter_count": 0,
        "non_embedding_transferred_fraction": 0.0,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "synthesizes_code": False,
        "open_or_pretrained_model_weights_used": False,
        "public_training_rows": int(initializer.get("public_training_rows") or 0),
        "external_inference_calls": int(initializer.get("external_inference_calls") or 0),
    }


def strict_generator_checkpoint_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict_or_empty(get_path(config, ["pretraining_initialization", "strict_generator_checkpoint"], {}))


def load_strict_generator_checkpoint_vocab_override(config: dict[str, Any], *, torch: Any) -> dict[str, Any]:
    cfg = strict_generator_checkpoint_config(config)
    if not bool(cfg.get("enabled", False)):
        return {"enabled": False, "active": False, "reason": "strict_generator_checkpoint_disabled"}
    if not bool(cfg.get("load_vocab", False)):
        return {"enabled": True, "active": False, "reason": "checkpoint_vocab_override_not_requested"}
    checkpoint_path = resolve(str(cfg.get("checkpoint") or ""))
    if not checkpoint_path.exists():
        return {"enabled": True, "active": False, "reason": "missing_strict_generator_checkpoint", "checkpoint": rel(checkpoint_path)}
    admission = theseus_artifact_admission.admit_from_config(checkpoint_path, cfg)
    if admission.get("required") and not admission.get("admitted"):
        return {
            "enabled": True,
            "active": False,
            "reason": f"artifact_admission_rejected:{admission.get('reason')}",
            "checkpoint": rel(checkpoint_path),
            "artifact_admission": admission,
        }
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - local torch serialization surface.
        return {
            "enabled": True,
            "active": False,
            "reason": f"checkpoint_vocab_load_failed:{exc.__class__.__name__}",
            "checkpoint": rel(checkpoint_path),
        }
    meta = checkpoint.get("meta") if isinstance(checkpoint, dict) else {}
    if not isinstance(meta, dict):
        return {"enabled": True, "active": False, "reason": "missing_checkpoint_meta", "checkpoint": rel(checkpoint_path)}
    if int(meta.get("public_training_rows") or 0) != 0 or int(meta.get("external_inference_calls") or 0) != 0:
        return {
            "enabled": True,
            "active": False,
            "reason": "checkpoint_vocab_rejected_for_public_or_external_training",
            "checkpoint": rel(checkpoint_path),
            "public_training_rows": int(meta.get("public_training_rows") or 0),
            "external_inference_calls": int(meta.get("external_inference_calls") or 0),
        }
    if bool(meta.get("open_or_pretrained_model_weights_used", False)):
        return {
            "enabled": True,
            "active": False,
            "reason": "checkpoint_vocab_rejected_for_open_or_pretrained_weights",
            "checkpoint": rel(checkpoint_path),
        }
    source_vocab = normalize_checkpoint_vocab(meta.get("source_vocab"))
    target_vocab = normalize_checkpoint_vocab(meta.get("target_vocab"))
    if not source_vocab or not target_vocab:
        return {"enabled": True, "active": False, "reason": "checkpoint_vocab_missing_or_invalid", "checkpoint": rel(checkpoint_path)}
    source_hash = stable_hash(json.dumps(source_vocab, sort_keys=True))
    target_hash = stable_hash(json.dumps(target_vocab, sort_keys=True))
    if source_hash != meta.get("source_vocab_sha256") or target_hash != meta.get("target_vocab_sha256"):
        return {
            "enabled": True,
            "active": False,
            "reason": "checkpoint_vocab_hash_mismatch",
            "checkpoint": rel(checkpoint_path),
            "source_vocab_sha256": source_hash,
            "target_vocab_sha256": target_hash,
            "checkpoint_source_vocab_sha256": meta.get("source_vocab_sha256"),
            "checkpoint_target_vocab_sha256": meta.get("target_vocab_sha256"),
        }
    return {
        "enabled": True,
        "active": True,
        "policy": "strict_generator_checkpoint_vocab_override_v1",
        "checkpoint": rel(checkpoint_path),
        "checkpoint_sha256": stable_hash_file(checkpoint_path),
        "source_vocab": source_vocab,
        "target_vocab": target_vocab,
        "source_vocab_size": len(source_vocab),
        "target_vocab_size": len(target_vocab),
        "source_vocab_sha256": source_hash,
        "target_vocab_sha256": target_hash,
        "public_training_rows": 0,
        "external_inference_calls": 0,
        "open_or_pretrained_model_weights_used": False,
        "artifact_admission": admission,
        "score_semantics": (
            "Optional checkpoint-vocab replay for strict generator checkpoint evaluation. "
            "It imports only the saved tokenizer dictionaries from a clean local checkpoint "
            "so the full model state can be loaded exactly; it emits no candidates and "
            "does not credit templates, routers, tools, or fallback returns."
        ),
    }


def normalize_checkpoint_vocab(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in payload.items():
        try:
            normalized[str(key)] = int(value)
        except (TypeError, ValueError):
            return {}
    for required in ["<pad>", "<unk>"]:
        if required not in normalized:
            return {}
    return normalized


def strict_generator_vocab_override_report(summary: dict[str, Any]) -> dict[str, Any]:
    clean = {key: value for key, value in dict_or_empty(summary).items() if key not in {"source_vocab", "target_vocab"}}
    return clean


def apply_strict_generator_checkpoint(
    model: Any,
    *,
    config: dict[str, Any],
    arm_id: str,
    source_vocab: dict[str, int],
    target_vocab: dict[str, int],
    max_source: int,
    max_target: int,
    dims: dict[str, int],
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    cfg = strict_generator_checkpoint_config(config)
    if not bool(cfg.get("enabled", False)):
        return {"enabled": False, "active": False, "reason": "strict_generator_checkpoint_disabled"}
    if str(cfg.get("arm_id") or "transformer_control") != arm_id:
        return {
            "enabled": True,
            "active": False,
            "reason": "checkpoint_arm_not_applicable",
            "configured_arm": str(cfg.get("arm_id") or "transformer_control"),
            "arm_id": arm_id,
        }
    checkpoint_path = resolve(str(cfg.get("checkpoint") or ""))
    if not checkpoint_path.exists():
        return {"enabled": True, "active": False, "reason": "missing_strict_generator_checkpoint", "checkpoint": rel(checkpoint_path)}
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:  # pragma: no cover - local torch serialization surface.
        return {
            "enabled": True,
            "active": False,
            "reason": f"checkpoint_load_failed:{exc.__class__.__name__}",
            "checkpoint": rel(checkpoint_path),
        }
    meta = checkpoint.get("meta") if isinstance(checkpoint, dict) else {}
    if not isinstance(meta, dict):
        return {"enabled": True, "active": False, "reason": "missing_checkpoint_meta", "checkpoint": rel(checkpoint_path)}
    source_hash = stable_hash(json.dumps(source_vocab, sort_keys=True))
    target_hash = stable_hash(json.dumps(target_vocab, sort_keys=True))
    expected = {
        "source_vocab_sha256": source_hash,
        "target_vocab_sha256": target_hash,
        "dims": dims,
        "max_source": int(max_source),
        "max_target": int(max_target),
    }
    mismatches = {
        key: {"expected": value, "actual": meta.get(key)}
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatches:
        return {
            "enabled": True,
            "active": False,
            "reason": "strict_generator_checkpoint_shape_or_vocab_mismatch",
            "checkpoint": rel(checkpoint_path),
            "mismatches": mismatches,
        }
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        return {"enabled": True, "active": False, "reason": "missing_model_state_dict", "checkpoint": rel(checkpoint_path)}
    before = model_parameter_snapshot(model, torch=torch)
    model.load_state_dict(state, strict=True)
    update = model_parameter_update_summary(model, before, torch=torch)
    inventory = model_parameter_inventory(model)
    return {
        "enabled": True,
        "active": True,
        "policy": str(checkpoint.get("policy") or "project_theseus_strict_generator_pretraining_checkpoint_v1"),
        "checkpoint": rel(checkpoint_path),
        "checkpoint_sha256": stable_hash_file(checkpoint_path),
        "arm_id": arm_id,
        "budget_id": meta.get("budget_id"),
        "completed_epochs": int(meta.get("completed_epochs") or 0),
        "trainable_parameter_count": inventory["trainable_parameter_count"],
        "transferred_parameter_count": inventory["trainable_parameter_count"],
        "transferred_state_fraction_into_generator": 1.0,
        "non_embedding_transferred_parameter_count": inventory["non_embedding_parameter_count"],
        "non_embedding_transferred_fraction": 1.0,
        "load_changed_parameter_fraction": update["parameter_update_fraction"],
        "load_changed_non_embedding_fraction": update["non_embedding_update_fraction"],
        "source_vocab_sha256": source_hash,
        "target_vocab_sha256": target_hash,
        "from_scratch": bool(meta.get("from_scratch", True)),
        "open_or_pretrained_model_weights_used": bool(meta.get("open_or_pretrained_model_weights_used", False)),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "public_training_rows": int(meta.get("public_training_rows") or 0),
        "external_inference_calls": int(meta.get("external_inference_calls") or 0),
    }


def initialize_embedding_module_from_bpe(
    embedding_module: Any,
    vocab: dict[str, int],
    tokenizer: dict[str, Any],
    pretrain_embedding: Any,
    *,
    token_kind: str,
    torch: Any,
) -> dict[str, Any]:
    bpe_vocab = dict_or_empty(tokenizer.get("vocab"))
    merges = [
        (str(pair[0]), str(pair[1]))
        for pair in tokenizer.get("merges", [])
        if isinstance(pair, list) and len(pair) == 2
    ]
    ranks = {pair: idx for idx, pair in enumerate(merges)}
    mapped = 0
    unknown = 0
    copied_dims = min(int(embedding_module.weight.shape[1]), int(pretrain_embedding.shape[1]))
    with torch.no_grad():
        for token, idx in vocab.items():
            text = token_to_pretraining_text(str(token), token_kind=token_kind)
            ids = encode_pretraining_text(text, bpe_vocab, ranks)
            ids = [item for item in ids if 0 <= item < int(pretrain_embedding.shape[0])]
            if not ids:
                unknown += 1
                continue
            vector = pretrain_embedding[torch.tensor(ids, dtype=torch.long, device=pretrain_embedding.device)].mean(dim=0)
            embedding_module.weight[int(idx), :copied_dims] = vector[:copied_dims]
            if str(token) == "<pad>":
                embedding_module.weight[int(idx), :] = 0.0
            mapped += 1
    return {
        "token_kind": token_kind,
        "vocab_size": len(vocab),
        "mapped_token_count": mapped,
        "unknown_token_count": unknown,
        "mapped_rate": ratio(mapped, len(vocab)),
        "copied_dims": copied_dims,
        "copied_parameter_count": int(mapped) * int(copied_dims),
        "target_embedding_dim": int(embedding_module.weight.shape[1]),
        "pretraining_embedding_dim": int(pretrain_embedding.shape[1]),
    }


def token_to_pretraining_text(token: str, *, token_kind: str) -> str:
    if token in {"<pad>", "<unk>", "<bos>", "<eos>"}:
        return token
    if token_kind == "target":
        kind, sep, value = token.partition(":")
        if kind == "NEWLINE":
            return "\n"
        if kind == "INDENT":
            return "    "
        if kind == "DEDENT":
            return "\n"
        if sep:
            return value or kind
    return token.replace("_", " ").replace(":", " ")


def encode_pretraining_text(text: str, bpe_vocab: dict[str, Any], ranks: dict[tuple[str, str], int]) -> list[int]:
    unk = int(bpe_vocab.get("<unk>", 1))
    ids: list[int] = []
    for token in bpe_basic_tokens(text):
        pieces = apply_bpe_merges(tuple(token), ranks)
        ids.extend(int(bpe_vocab.get(piece, unk)) for piece in pieces)
    return ids


def pretraining_initialization_report_summary(initializers: dict[str, Any]) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm_id, row in dict_or_empty(initializers.get("by_arm")).items():
        clean = {key: value for key, value in dict_or_empty(row).items() if key not in {"embedding", "tokenizer"}}
        by_arm[str(arm_id)] = clean
    return {
        "enabled": bool(initializers.get("enabled", False)),
        "active": bool(initializers.get("active", False)),
        "policy": initializers.get("policy"),
        "tokenizer": initializers.get("tokenizer"),
        "tokenizer_vocab_size": initializers.get("tokenizer_vocab_size"),
        "tokenizer_merge_count": initializers.get("tokenizer_merge_count"),
        "by_arm": by_arm,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "open_or_pretrained_model_weights_used": False,
    }


def train_token_model(
    model: Any,
    source_rows: list[list[int]],
    target_rows: list[list[int]],
    budget: dict[str, Any],
    *,
    torch: Any,
    device: Any,
    pad_id: int,
    target_vocab: dict[str, int] | None = None,
    plan_auxiliary_loss_weight: float = 0.0,
    allowed_name_sets: list[set[str]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    batch_size = int(budget.get("batch_size") or 64)
    epochs = int(budget.get("epochs") or 3)
    lr = float(budget.get("learning_rate") or 0.003)
    weight_decay = float(budget.get("weight_decay") or 0.0001)
    first_token_loss_weight = float(budget.get("first_target_token_loss_weight") or 1.0)
    return_token_loss_weight = max(0.0, float(budget.get("return_token_loss_weight") or 1.0))
    grammar_auxiliary_loss_weight = max(0.0, float(budget.get("grammar_validity_auxiliary_loss_weight") or 0.0))
    grammar_auxiliary_max_positions = max(0, int(budget.get("grammar_validity_max_positions_per_batch") or 0))
    body_token_validity_policy = str(budget.get("body_token_validity_policy") or "lightweight_python_v1")
    progress_label = str(budget.get("progress_label") or "")
    semantic_weight_cfg = dict_or_empty(budget.get("semantic_token_loss_weights"))
    semantic_weight_enabled = bool(semantic_weight_cfg.get("enabled", False) and target_vocab)
    grammar_inverse = {idx: tok for tok, idx in (target_vocab or {}).items()} if target_vocab else {}
    grammar_eos_id = int((target_vocab or {}).get("<eos>", 2))
    return_token_id = int((target_vocab or {}).get("NAME:return", -1)) if target_vocab else -1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction="none")
    src = torch.tensor(source_rows, dtype=torch.long, device=device)
    target = torch.tensor(target_rows, dtype=torch.long, device=device)
    tgt_in = target[:, :-1]
    tgt_out = target[:, 1:]
    losses = []
    plan_losses = []
    grammar_losses = []
    optimizer_step_count = 0
    grammar_auxiliary_position_count = 0
    grammar_auxiliary_active_steps = 0
    semantic_weighted_token_count = 0
    semantic_parameter_token_count = 0
    semantic_update_token_count = 0
    semantic_operator_token_count = 0
    semantic_control_token_count = 0
    allowed_cache: dict[tuple[int, ...], list[int]] = {}
    target_nonpad_token_count = int(target.ne(pad_id).sum().detach().cpu()) if target.numel() else 0
    source_nonpad_token_count = int(src.ne(0).sum().detach().cpu()) if src.numel() else 0
    for epoch_index in range(epochs):
        order = torch.randperm(src.shape[0], device=device)
        total_loss = 0.0
        total_plan_loss = 0.0
        total_grammar_loss = 0.0
        total = 0
        plan_total = 0
        grammar_total = 0
        model.train()
        for start in range(0, src.shape[0], batch_size):
            idx = order[start : start + batch_size]
            logits = model(src[idx], tgt_in[idx])
            token_loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out[idx].reshape(-1))
            token_loss = token_loss.reshape(tgt_out[idx].shape)
            valid = tgt_out[idx].ne(pad_id).float()
            weights = torch.ones_like(valid)
            if first_token_loss_weight != 1.0 and weights.shape[1] > 0:
                weights[:, 0] = first_token_loss_weight
            if return_token_loss_weight != 1.0 and return_token_id >= 0:
                return_mask = tgt_out[idx].eq(return_token_id)
                weights = torch.where(return_mask, torch.full_like(weights, return_token_loss_weight), weights)
            if semantic_weight_enabled:
                semantic_weights, semantic_summary = semantic_token_loss_weight_matrix(
                    tgt_out[idx],
                    target_vocab=target_vocab or {},
                    allowed_name_sets=allowed_name_sets,
                    row_indices=[int(value) for value in idx.detach().cpu().tolist()],
                    cfg=semantic_weight_cfg,
                    torch=torch,
                    device=device,
                )
                weights = weights * semantic_weights
                semantic_weighted_token_count += int(semantic_summary.get("weighted_token_count") or 0)
                semantic_parameter_token_count += int(semantic_summary.get("parameter_token_count") or 0)
                semantic_update_token_count += int(semantic_summary.get("update_token_count") or 0)
                semantic_operator_token_count += int(semantic_summary.get("operator_token_count") or 0)
                semantic_control_token_count += int(semantic_summary.get("control_token_count") or 0)
            loss = (token_loss * valid * weights).sum() / (valid * weights).sum().clamp(min=1.0)
            plan_loss = None
            if plan_auxiliary_loss_weight > 0.0 and hasattr(model, "semantic_plan_logits") and tgt_out[idx].shape[1] > 0:
                plan_logits = model.semantic_plan_logits(src[idx])
                plan_target = tgt_out[idx][:, 0]
                plan_loss = torch.nn.functional.cross_entropy(plan_logits, plan_target, ignore_index=pad_id)
                loss = loss + (plan_auxiliary_loss_weight * plan_loss)
            grammar_loss = None
            if grammar_auxiliary_loss_weight > 0.0 and grammar_inverse and grammar_auxiliary_max_positions > 0:
                grammar_loss, grammar_summary = grammar_validity_auxiliary_loss(
                    logits,
                    target[idx],
                    target_vocab=target_vocab or {},
                    inverse=grammar_inverse,
                    eos_id=grammar_eos_id,
                    pad_id=pad_id,
                    max_positions=grammar_auxiliary_max_positions,
                    body_token_policy=body_token_validity_policy,
                    allowed_cache=allowed_cache,
                    allowed_name_sets=[
                        allowed_name_sets[int(value)]
                        if allowed_name_sets and int(value) < len(allowed_name_sets)
                        else None
                        for value in idx.detach().cpu().tolist()
                    ],
                    torch=torch,
                )
                grammar_auxiliary_position_count += int(grammar_summary.get("position_count") or 0)
                if grammar_loss is not None:
                    loss = loss + (grammar_auxiliary_loss_weight * grammar_loss)
                    grammar_auxiliary_active_steps += 1
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_step_count += 1
            total_loss += float(loss.detach().cpu()) * int(idx.shape[0])
            if plan_loss is not None:
                total_plan_loss += float(plan_loss.detach().cpu()) * int(idx.shape[0])
                plan_total += int(idx.shape[0])
            if grammar_loss is not None:
                total_grammar_loss += float(grammar_loss.detach().cpu()) * int(idx.shape[0])
                grammar_total += int(idx.shape[0])
            total += int(idx.shape[0])
        epoch_loss = round(total_loss / max(1, total), 6)
        epoch_plan_loss = round(total_plan_loss / max(1, plan_total), 6) if plan_total else None
        epoch_grammar_loss = round(total_grammar_loss / max(1, grammar_total), 6) if grammar_total else None
        losses.append(epoch_loss)
        plan_losses.append(epoch_plan_loss)
        grammar_losses.append(epoch_grammar_loss)
        if progress_label:
            print(
                f"[strict-decoder] progress {progress_label} epoch={epoch_index + 1}/{epochs} "
                f"loss={epoch_loss} steps={optimizer_step_count}",
                file=sys.stderr,
                flush=True,
            )
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "first_target_token_loss_weight": first_token_loss_weight,
        "return_token_loss_weight": return_token_loss_weight,
        "return_token_weight_active": bool(return_token_loss_weight != 1.0 and return_token_id >= 0),
        "auxiliary_plan_loss_weight": float(plan_auxiliary_loss_weight),
        "plan_router_supervision_enabled": bool(plan_auxiliary_loss_weight > 0.0 and hasattr(model, "semantic_plan_logits")),
        "grammar_validity_auxiliary_loss_weight": grammar_auxiliary_loss_weight,
        "grammar_validity_auxiliary_enabled": bool(
            grammar_auxiliary_loss_weight > 0.0 and bool(grammar_inverse) and grammar_auxiliary_max_positions > 0
        ),
        "grammar_validity_max_positions_per_batch": grammar_auxiliary_max_positions,
        "grammar_validity_auxiliary_position_count": grammar_auxiliary_position_count,
        "grammar_validity_auxiliary_active_steps": grammar_auxiliary_active_steps,
        "grammar_validity_auxiliary_policy": "training_target_prefix_allowed_token_mass_v1",
        "body_token_validity_policy": body_token_validity_policy,
        "grammar_validity_uses_eval_tests_or_solutions": False,
        "grammar_validity_uses_public_data": False,
        "semantic_token_loss_weights": semantic_token_loss_weight_report(
            semantic_weight_cfg,
            enabled=semantic_weight_enabled,
            weighted_token_count=semantic_weighted_token_count,
            parameter_token_count=semantic_parameter_token_count,
            update_token_count=semantic_update_token_count,
            operator_token_count=semantic_operator_token_count,
            control_token_count=semantic_control_token_count,
        ),
        "loss_curve": losses,
        "plan_loss_curve": plan_losses,
        "grammar_validity_loss_curve": grammar_losses,
        "training_example_count": int(src.shape[0]),
        "target_nonpad_token_count": target_nonpad_token_count,
        "source_nonpad_token_count": source_nonpad_token_count,
        "optimizer_step_count": optimizer_step_count,
        "pretraining_target_tokens_seen": int(target_nonpad_token_count) * int(epochs),
        "pretraining_source_tokens_seen": int(source_nonpad_token_count) * int(epochs),
        "pretraining_total_tokens_seen": (int(target_nonpad_token_count) + int(source_nonpad_token_count)) * int(epochs),
        "pretraining_windows_seen": int(src.shape[0]) * int(epochs),
        "optimizer": "AdamW",
        "wall_time_ms": int((time.perf_counter() - started) * 1000),
    }


def semantic_token_loss_weight_report(
    cfg: dict[str, Any],
    *,
    enabled: bool,
    weighted_token_count: int,
    parameter_token_count: int,
    update_token_count: int,
    operator_token_count: int,
    control_token_count: int,
) -> dict[str, Any]:
    if not bool(cfg.get("enabled", False)):
        return {
            "enabled": False,
            "uses_eval_tests_or_solutions": False,
            "uses_public_data": False,
        }
    return {
        "enabled": enabled,
        "policy": str(cfg.get("policy") or "visible_parameter_and_expression_token_weighting_v1"),
        "parameter_identifier_weight": float(cfg.get("parameter_identifier_weight") or 1.0),
        "update_call_name_weight": float(cfg.get("update_call_name_weight") or 1.0),
        "operator_token_weight": float(cfg.get("operator_token_weight") or 1.0),
        "control_token_weight": float(cfg.get("control_token_weight") or 1.0),
        "max_weight": float(cfg.get("max_weight") or 4.0),
        "weighted_token_count": int(weighted_token_count),
        "parameter_token_count": int(parameter_token_count),
        "update_token_count": int(update_token_count),
        "operator_token_count": int(operator_token_count),
        "control_token_count": int(control_token_count),
        "score_semantics": (
            "Supervised loss weighting over admitted training target tokens only. It favors visible signature "
            "parameters and expression/update syntax, but does not synthesize code, read eval tests/solutions, "
            "use public benchmark payloads, or change verifier scoring."
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
    }


def semantic_token_loss_weight_matrix(
    target_batch: Any,
    *,
    target_vocab: dict[str, int],
    allowed_name_sets: list[set[str]] | None,
    row_indices: list[int],
    cfg: dict[str, Any],
    torch: Any,
    device: Any,
) -> tuple[Any, dict[str, int]]:
    weights = torch.ones_like(target_batch, dtype=torch.float, device=device)
    inverse = {idx: tok for tok, idx in target_vocab.items()}
    parameter_weight = float(cfg.get("parameter_identifier_weight") or 1.0)
    update_weight = float(cfg.get("update_call_name_weight") or 1.0)
    operator_weight = float(cfg.get("operator_token_weight") or 1.0)
    control_weight = float(cfg.get("control_token_weight") or 1.0)
    max_weight = max(1.0, float(cfg.get("max_weight") or 4.0))
    update_names = set(cfg.get("update_call_names") or [
        "append",
        "add",
        "extend",
        "get",
        "items",
        "join",
        "setdefault",
        "split",
        "splitlines",
        "update",
    ])
    operator_values = set(cfg.get("operator_tokens") or ["=", ".", "[", "]", "+", "-", "%", "==", "!=", "<", ">"])
    control_names = set(cfg.get("control_names") or ["for", "in", "while"])
    cpu_rows = target_batch.detach().cpu().tolist()
    weighted = 0
    parameter_count = 0
    update_count = 0
    operator_count = 0
    control_count = 0
    for local_row, row in enumerate(cpu_rows):
        row_index = row_indices[local_row] if local_row < len(row_indices) else local_row
        allowed = allowed_name_sets[row_index] if allowed_name_sets and row_index < len(allowed_name_sets) else set()
        for pos, token_id_raw in enumerate(row):
            token_id = int(token_id_raw)
            tok = str(inverse.get(token_id, ""))
            if not tok or tok in {"<pad>", "<bos>", "<eos>", "<unk>"}:
                continue
            kind, _, value = tok.partition(":")
            token_weight = 1.0
            token_class = ""
            if kind == "NAME" and value in allowed and parameter_weight > token_weight:
                token_weight = parameter_weight
                token_class = "parameter"
            elif kind == "NAME" and value in update_names and update_weight > token_weight:
                token_weight = update_weight
                token_class = "update"
            elif kind == "OP" and value in operator_values and operator_weight > token_weight:
                token_weight = operator_weight
                token_class = "operator"
            elif kind == "NAME" and value in control_names and control_weight > token_weight:
                token_weight = control_weight
                token_class = "control"
            if token_weight <= 1.0:
                continue
            token_weight = min(max_weight, token_weight)
            weights[local_row, pos] = float(token_weight)
            weighted += 1
            if token_class == "parameter":
                parameter_count += 1
            elif token_class == "update":
                update_count += 1
            elif token_class == "operator":
                operator_count += 1
            elif token_class == "control":
                control_count += 1
    return weights, {
        "weighted_token_count": weighted,
        "parameter_token_count": parameter_count,
        "update_token_count": update_count,
        "operator_token_count": operator_count,
        "control_token_count": control_count,
    }


def grammar_validity_auxiliary_loss(
    logits: Any,
    target_batch: Any,
    *,
    target_vocab: dict[str, int],
    inverse: dict[int, str],
    eos_id: int,
    pad_id: int,
    max_positions: int,
    body_token_policy: str,
    allowed_cache: dict[tuple[int, ...], list[int]],
    allowed_name_sets: list[set[str] | None] | None = None,
    torch: Any,
) -> tuple[Any | None, dict[str, Any]]:
    """Penalize invalid next-token probability mass on training prefixes only.

    This is objective-side syntax pressure, not a renderer. It never looks at
    eval rows, tests, solutions outside the admitted training target, public
    benchmark payloads, or answer metadata. The gold next token is always
    admitted to the allowed set so a conservative grammar bug cannot contradict
    the supervised token target.
    """

    if logits.numel() == 0 or target_batch.numel() == 0 or max_positions <= 0:
        return None, {"position_count": 0}
    cpu_rows = target_batch.detach().cpu().tolist()
    valid_positions: list[tuple[int, int, int, tuple[int, ...]]] = []
    for row_index, row in enumerate(cpu_rows):
        if not row:
            continue
        for step in range(max(0, min(logits.shape[1], len(row) - 1))):
            next_id = int(row[step + 1])
            if next_id == pad_id:
                continue
            prefix = tuple(int(tok) for tok in row[: step + 1] if int(tok) != pad_id)
            valid_positions.append((row_index, step, next_id, prefix))
    if not valid_positions:
        return None, {"position_count": 0}
    if len(valid_positions) > max_positions:
        stride = max(1, len(valid_positions) // max_positions)
        selected = valid_positions[::stride][:max_positions]
    else:
        selected = valid_positions
    probs = torch.softmax(logits, dim=-1)
    terms = []
    device = logits.device
    for row_index, step, next_id, prefix in selected:
        allowed_names = (
            allowed_name_sets[row_index]
            if allowed_name_sets and row_index < len(allowed_name_sets)
            else None
        )
        allowed = grammar_allowed_token_ids_for_prefix(
            prefix,
            next_id,
            inverse=inverse,
            eos_id=eos_id,
            target_vocab=target_vocab,
            body_token_policy=body_token_policy,
            allowed_cache=allowed_cache,
            allowed_names=allowed_names,
        )
        if not allowed:
            allowed = [next_id]
        allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=device)
        mass = probs[row_index, step, allowed_tensor].sum().clamp(min=1e-9)
        terms.append(-torch.log(mass))
    if not terms:
        return None, {"position_count": 0}
    return torch.stack(terms).mean(), {"position_count": len(terms)}


def grammar_allowed_token_ids_for_prefix(
    prefix_ids: tuple[int, ...],
    gold_next_id: int,
    *,
    inverse: dict[int, str],
    eos_id: int,
    target_vocab: dict[str, int],
    body_token_policy: str,
    allowed_cache: dict[tuple[int, ...], list[int]],
    allowed_names: set[str] | None = None,
) -> list[int]:
    allowed_name_key = tuple(sorted(str(value) for value in (allowed_names or set())))
    cache_key = (hash(str(body_token_policy)), hash(allowed_name_key), *tuple(prefix_ids))
    cached = allowed_cache.get(cache_key)
    if cached is not None:
        if gold_next_id in cached:
            return cached
        return [*cached, gold_next_id]
    prefix_tokens = [inverse.get(idx, "<unk>") for idx in prefix_ids[1:]]
    allowed: list[int] = []
    for idx, tok in inverse.items():
        token_id = int(idx)
        if token_id == int(target_vocab.get("<pad>", 0)) or token_id == int(target_vocab.get("<bos>", 1)):
            continue
        if token_id == eos_id:
            if syntax_complete_body_prefix(prefix_tokens):
                allowed.append(token_id)
            continue
        if token_allowed_by_policy(prefix_tokens, str(tok), policy=body_token_policy, allowed_names=allowed_names):
            allowed.append(token_id)
    if gold_next_id not in allowed:
        allowed.append(gold_next_id)
    allowed_cache[cache_key] = allowed
    return allowed


def evaluate_token_model_loss(
    model: Any,
    source_rows: list[list[int]],
    target_rows: list[list[int]],
    *,
    batch_size: int,
    torch: Any,
    device: Any,
    pad_id: int,
) -> dict[str, Any]:
    if not source_rows or not target_rows:
        return {"active": False, "loss": None, "perplexity": None, "example_count": 0}
    src = torch.tensor(source_rows, dtype=torch.long, device=device)
    target = torch.tensor(target_rows, dtype=torch.long, device=device)
    tgt_in = target[:, :-1]
    tgt_out = target[:, 1:]
    model.eval()
    losses: list[float] = []
    total_weight = 0
    with torch.no_grad():
        for start in range(0, src.shape[0], max(1, int(batch_size))):
            idx = slice(start, min(start + max(1, int(batch_size)), src.shape[0]))
            logits = model(src[idx], tgt_in[idx])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tgt_out[idx].reshape(-1),
                ignore_index=pad_id,
                reduction="sum",
            )
            weight = int(tgt_out[idx].ne(pad_id).sum().detach().cpu())
            losses.append(float(loss.detach().cpu()))
            total_weight += weight
    model.train()
    if total_weight <= 0:
        return {"active": False, "loss": None, "perplexity": None, "example_count": int(src.shape[0])}
    loss_value = round(sum(losses) / total_weight, 6)
    return {
        "active": True,
        "loss": loss_value,
        "perplexity": round(math.exp(min(20.0, loss_value)), 6),
        "example_count": int(src.shape[0]),
        "target_token_count": total_weight,
    }


def model_parameter_inventory(model: Any) -> dict[str, int]:
    trainable = 0
    non_embedding = 0
    tensors = 0
    non_embedding_tensors = 0
    for name, param in model.named_parameters():
        if not bool(getattr(param, "requires_grad", False)):
            continue
        count = int(param.numel())
        trainable += count
        tensors += 1
        if "embedding" not in str(name):
            non_embedding += count
            non_embedding_tensors += 1
    return {
        "trainable_parameter_count": trainable,
        "non_embedding_parameter_count": non_embedding,
        "trainable_tensor_count": tensors,
        "non_embedding_tensor_count": non_embedding_tensors,
    }


def model_parameter_snapshot(model: Any, *, torch: Any) -> dict[str, Any]:
    return {
        name: param.detach().clone().cpu()
        for name, param in model.named_parameters()
        if bool(getattr(param, "requires_grad", False))
    }


def model_parameter_update_summary(model: Any, before: dict[str, Any], *, torch: Any) -> dict[str, Any]:
    changed = 0
    total = 0
    changed_non_embedding = 0
    total_non_embedding = 0
    changed_tensors = 0
    tensor_total = 0
    changed_non_embedding_tensors = 0
    non_embedding_tensor_total = 0
    eps = 1e-12
    for name, param in model.named_parameters():
        if not bool(getattr(param, "requires_grad", False)) or name not in before:
            continue
        current = param.detach().cpu()
        delta = (current - before[name]).abs()
        changed_elements = int(delta.gt(eps).sum().item())
        count = int(param.numel())
        total += count
        changed += changed_elements
        tensor_total += 1
        tensor_changed = changed_elements > 0
        if tensor_changed:
            changed_tensors += 1
        if "embedding" not in str(name):
            total_non_embedding += count
            changed_non_embedding += changed_elements
            non_embedding_tensor_total += 1
            if tensor_changed:
                changed_non_embedding_tensors += 1
    return {
        "trainable_parameter_count": total,
        "updated_parameter_count": changed,
        "parameter_update_fraction": ratio(changed, total),
        "non_embedding_parameter_count": total_non_embedding,
        "updated_non_embedding_parameter_count": changed_non_embedding,
        "non_embedding_update_fraction": ratio(changed_non_embedding, total_non_embedding),
        "trainable_tensor_count": tensor_total,
        "updated_tensor_count": changed_tensors,
        "parameter_tensor_update_fraction": ratio(changed_tensors, tensor_total),
        "non_embedding_tensor_count": non_embedding_tensor_total,
        "updated_non_embedding_tensor_count": changed_non_embedding_tensors,
        "non_embedding_tensor_update_fraction": ratio(changed_non_embedding_tensors, non_embedding_tensor_total),
    }
