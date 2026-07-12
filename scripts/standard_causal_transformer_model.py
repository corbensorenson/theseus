#!/usr/bin/env python3
"""Modern MLX decoder-only transformer used by the practical survival lane."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CausalTransformerConfig:
    vocab_size: int
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 2
    ff_dim: int = 768
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-5
    attention_policy: str = "causal"
    source_target_separator_token_id: int = 2
    source_encoder_layers: int = 0
    source_copy_mode: str = "none"
    source_copy_auxiliary_loss_weight: float = 0.0
    expert_adapter_dim: int = 0
    source_expert_adapter_dim: int = 0
    state_memory_slots: int = 0
    state_memory_chunk_size: int = 32
    state_memory_local_window: int = 96
    state_memory_mode: str = "none"
    state_memory_ablation: str = "none"
    state_memory_read_policy: str = "unrestricted"
    semantic_plan_feature_count: int = 0
    semantic_plan_separator_token_id: int = 2
    semantic_plan_bottleneck_dim: int = 0
    semantic_plan_slot_count: int = 0
    semantic_plan_conditioning_mode: str = "global_additive"
    semantic_plan_probability_mode: str = "independent_sigmoid"
    semantic_plan_factor_group_sizes: tuple[int, ...] = ()

    def validate(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must divide evenly across query heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("query heads must divide evenly across KV heads")
        if self.num_layers <= 0 or self.vocab_size <= 0 or self.ff_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.attention_policy not in {"causal", "prefix_lm", "encoder_decoder"}:
            raise ValueError(
                "attention policy must be causal, prefix_lm, or encoder_decoder"
            )
        if self.source_encoder_layers < 0:
            raise ValueError("source encoder layers cannot be negative")
        if self.attention_policy == "encoder_decoder" and self.source_encoder_layers <= 0:
            raise ValueError("encoder-decoder attention requires source encoder layers")
        if self.attention_policy != "encoder_decoder" and self.source_encoder_layers:
            raise ValueError("source encoder layers require encoder-decoder attention")
        if self.source_copy_mode not in {"none", "pointer_generator"}:
            raise ValueError("source copy mode must be none or pointer_generator")
        if self.source_copy_mode != "none" and self.attention_policy != "encoder_decoder":
            raise ValueError("source copying requires encoder-decoder attention")
        if not 0.0 <= self.source_copy_auxiliary_loss_weight <= 2.0:
            raise ValueError("source copy auxiliary loss weight must be between zero and two")
        if self.source_copy_auxiliary_loss_weight and self.source_copy_mode == "none":
            raise ValueError("source copy auxiliary loss requires source copying")
        if self.expert_adapter_dim < 0:
            raise ValueError("expert adapter dimension cannot be negative")
        if self.source_expert_adapter_dim < 0:
            raise ValueError("source expert adapter dimension cannot be negative")
        if self.source_expert_adapter_dim and self.attention_policy != "encoder_decoder":
            raise ValueError("source expert adapters require encoder-decoder attention")
        if not 0 <= self.source_target_separator_token_id < self.vocab_size:
            raise ValueError("source-target separator token must be in vocabulary")
        if self.state_memory_mode not in {"none", "semantic_roles", "hash_control"}:
            raise ValueError("state memory mode must be none, semantic_roles, or hash_control")
        if self.state_memory_ablation not in {"none", "zero", "shuffle"}:
            raise ValueError("state memory ablation must be none, zero, or shuffle")
        if self.state_memory_read_policy not in {"unrestricted", "role_dependency"}:
            raise ValueError("state memory read policy must be unrestricted or role_dependency")
        if self.state_memory_mode == "none" and self.state_memory_slots != 0:
            raise ValueError("state memory slots must be zero when state memory is disabled")
        if self.attention_policy == "prefix_lm" and self.state_memory_mode != "none":
            raise ValueError(
                "prefix-LM attention is not yet compatible with chunked executable state memory"
            )
        if self.attention_policy == "encoder_decoder" and self.state_memory_mode != "none":
            raise ValueError(
                "encoder-decoder attention is not yet compatible with executable state memory"
            )
        if self.state_memory_mode != "none" and self.state_memory_slots <= 1:
            raise ValueError("enabled state memory requires at least two slots")
        if self.state_memory_chunk_size <= 0 or self.state_memory_local_window <= 0:
            raise ValueError("state memory chunk and local-window sizes must be positive")
        if self.state_memory_chunk_size > self.state_memory_local_window:
            raise ValueError("state memory chunk size cannot exceed its local attention window")
        if self.semantic_plan_feature_count < 0:
            raise ValueError("semantic plan feature count cannot be negative")
        if self.semantic_plan_bottleneck_dim < 0:
            raise ValueError("semantic plan bottleneck dimension cannot be negative")
        if self.semantic_plan_feature_count == 0 and self.semantic_plan_bottleneck_dim:
            raise ValueError("semantic plan bottleneck requires semantic plan features")
        if self.semantic_plan_conditioning_mode not in {"global_additive", "slot_attention"}:
            raise ValueError("semantic plan conditioning must be global_additive or slot_attention")
        if self.semantic_plan_probability_mode not in {
            "independent_sigmoid",
            "slot_categorical",
            "factorized_step",
        }:
            raise ValueError(
                "semantic plan probability mode must be independent_sigmoid, slot_categorical, or factorized_step"
            )
        if (
            self.semantic_plan_probability_mode in {"slot_categorical", "factorized_step"}
            and self.semantic_plan_conditioning_mode != "slot_attention"
        ):
            raise ValueError("structured slot probabilities require slot attention")
        if self.semantic_plan_conditioning_mode == "slot_attention":
            if self.semantic_plan_slot_count <= 0:
                raise ValueError("slot attention requires positive semantic plan slots")
            if self.semantic_plan_feature_count % self.semantic_plan_slot_count:
                raise ValueError("semantic plan features must divide evenly across slots")
            if self.semantic_plan_bottleneck_dim <= 0:
                raise ValueError("slot attention requires a low-rank semantic plan bottleneck")
        if self.semantic_plan_probability_mode == "factorized_step":
            groups = tuple(int(value) for value in self.semantic_plan_factor_group_sizes)
            slot_width = self.semantic_plan_feature_count // self.semantic_plan_slot_count
            if len(groups) < 2 or groups[0] != 1 or sum(groups) != slot_width:
                raise ValueError(
                    "factorized plan groups must begin with presence and cover one slot"
                )
        if self.semantic_plan_feature_count > 0 and not (
            0 <= self.semantic_plan_separator_token_id < self.vocab_size
        ):
            raise ValueError("semantic plan separator token must be in vocabulary")


def build_model(
    config: CausalTransformerConfig,
    *,
    mx: Any,
    nn: Any,
    state_role_lookup: Any | None = None,
    source_to_target_lookup: Any | None = None,
) -> Any:
    """Build a pre-norm RoPE/GQA/SwiGLU causal LM with tied embeddings."""

    config.validate()
    head_dim = config.d_model // config.num_heads
    half_head_dim = head_dim // 2
    rope_inverse_frequency = mx.array(
        [config.rope_base ** (-(2.0 * index) / head_dim) for index in range(half_head_dim)],
        dtype=mx.float32,
    )
    state_enabled = config.state_memory_mode != "none"
    source_encoder_enabled = config.attention_policy == "encoder_decoder"
    pointer_generator_enabled = config.source_copy_mode == "pointer_generator"
    expert_adapter_enabled = config.expert_adapter_dim > 0
    source_expert_adapter_enabled = config.source_expert_adapter_dim > 0
    if pointer_generator_enabled:
        if source_to_target_lookup is None:
            raise ValueError("pointer-generator mode requires a source-to-target lookup")
        if tuple(source_to_target_lookup.shape) != (config.vocab_size,):
            raise ValueError("source-to-target lookup must match the model vocabulary")
        source_to_target_lookup = mx.array(source_to_target_lookup, dtype=mx.int32)
    plan_enabled = config.semantic_plan_feature_count > 0
    plan_slot_attention_enabled = (
        plan_enabled and config.semantic_plan_conditioning_mode == "slot_attention"
    )
    if state_enabled:
        if state_role_lookup is None:
            raise ValueError("enabled state memory requires a causal token-role lookup")
        if tuple(state_role_lookup.shape) != (config.vocab_size, config.state_memory_slots):
            raise ValueError("state-role lookup shape must match vocabulary and slot counts")
        state_role_lookup = mx.array(state_role_lookup, dtype=mx.float32)
        state_role_interaction = mx.eye(config.state_memory_slots, dtype=mx.float32)
        if config.state_memory_slots == 8:
            dependencies = {
                0: (5, 7),
                1: (0, 5),
                2: (0, 1, 5),
                3: (1, 2, 5),
                4: (2, 3, 5),
                5: (1, 2, 3, 4),
                6: (2, 3, 4, 5),
                7: (0, 1, 5),
            }
            rows = []
            for query_role in range(config.state_memory_slots):
                row = [1.0 if memory_role == query_role else 0.0 for memory_role in range(config.state_memory_slots)]
                for memory_role in dependencies[query_role]:
                    row[memory_role] = max(row[memory_role], 0.5)
                rows.append(row)
            state_role_interaction = mx.array(rows, dtype=mx.float32)

    def apply_rope(value: Any, *, offset: int) -> Any:
        length = int(value.shape[2])
        positions = mx.arange(offset, offset + length, dtype=mx.float32)
        angles = positions[:, None] * rope_inverse_frequency[None, :]
        cosine = mx.cos(angles)[None, None, :, :]
        sine = mx.sin(angles)[None, None, :, :]
        first = value[..., :half_head_dim]
        second = value[..., half_head_dim:]
        return mx.concatenate(
            [first * cosine - second * sine, first * sine + second * cosine],
            axis=-1,
        )

    class CausalAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(config.d_model, config.num_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(config.d_model, config.num_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(config.d_model, config.num_kv_heads * head_dim, bias=False)
            self.out_proj = nn.Linear(config.num_heads * head_dim, config.d_model, bias=False)

        def __call__(
            self,
            hidden: Any,
            cache: tuple[Any, Any] | None = None,
            memory: Any | None = None,
            role_weights: Any | None = None,
            attention_mask: Any | None = None,
        ) -> tuple[Any, tuple[Any, Any]]:
            batch, length, _dims = hidden.shape
            offset = int(cache[0].shape[2]) if cache is not None else 0
            query = self.q_proj(hidden).reshape(batch, length, config.num_heads, head_dim).transpose(0, 2, 1, 3)
            key = self.k_proj(hidden).reshape(batch, length, config.num_kv_heads, head_dim).transpose(0, 2, 1, 3)
            value = self.v_proj(hidden).reshape(batch, length, config.num_kv_heads, head_dim).transpose(0, 2, 1, 3)
            query = apply_rope(query, offset=offset)
            key = apply_rope(key, offset=offset)
            if cache is not None:
                key = mx.concatenate([cache[0], key], axis=2)
                value = mx.concatenate([cache[1], value], axis=2)
            mask = (
                attention_mask
                if attention_mask is not None and cache is None and memory is None
                else "causal"
                if cache is None and length > 1 and memory is None
                else None
            )
            if cache is not None and length > 1 and memory is None:
                key_positions = mx.arange(offset + length, dtype=mx.int32)
                allowed_through = offset + mx.arange(length, dtype=mx.int32) + 1
                mask = mx.where(
                    key_positions[None, :] < allowed_through[:, None],
                    mx.array(0.0, dtype=mx.float32),
                    mx.array(-1e9, dtype=mx.float32),
                )
            attention_key = key
            attention_value = value
            memory_width = 0
            if memory is not None:
                memory_key = self.k_proj(memory).reshape(
                    batch, config.state_memory_slots, config.num_kv_heads, head_dim
                ).transpose(0, 2, 1, 3)
                memory_value = self.v_proj(memory).reshape(
                    batch, config.state_memory_slots, config.num_kv_heads, head_dim
                ).transpose(0, 2, 1, 3)
                memory_width = config.state_memory_slots
                local_start = max(0, int(key.shape[2]) - config.state_memory_local_window)
                attention_key = mx.concatenate([memory_key, key[:, :, local_start:, :]], axis=2)
                attention_value = mx.concatenate([memory_value, value[:, :, local_start:, :]], axis=2)
                local_width = int(attention_key.shape[2]) - memory_width
                prior_local = local_width - length
                if length > 1:
                    key_positions = mx.arange(memory_width + local_width, dtype=mx.int32)
                    allowed_through = (
                        memory_width
                        + prior_local
                        + mx.arange(length, dtype=mx.int32)
                        + 1
                    )
                    mask = mx.where(
                        key_positions[None, :] < allowed_through[:, None],
                        mx.array(0.0, dtype=mx.float32),
                        mx.array(-1e9, dtype=mx.float32),
                    )
                if role_weights is not None and config.state_memory_read_policy == "role_dependency":
                    read_access = mx.minimum(
                        mx.matmul(role_weights, state_role_interaction),
                        mx.array(1.0, dtype=mx.float32),
                    )
                    memory_bias = mx.log(mx.maximum(read_access, 0.05))
                    local_bias = mx.zeros(
                        (batch, length, local_width), dtype=mx.float32
                    )
                    role_bias = mx.concatenate([memory_bias, local_bias], axis=-1)[:, None, :, :]
                    if mask is None:
                        mask = role_bias
                    else:
                        mask = mask[None, None, :, :] + role_bias
            if config.num_kv_heads != config.num_heads:
                repeats = config.num_heads // config.num_kv_heads
                attention_key = mx.repeat(attention_key, repeats=repeats, axis=1)
                attention_value = mx.repeat(attention_value, repeats=repeats, axis=1)
            attended = mx.fast.scaled_dot_product_attention(
                query,
                attention_key,
                attention_value,
                scale=head_dim ** -0.5,
                mask=mask,
            )
            attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, config.num_heads * head_dim)
            return self.out_proj(attended), (key, value)

    class CrossAttention(nn.Module):
        def __init__(self, *, zero_output: bool = False) -> None:
            super().__init__()
            self.q_proj = nn.Linear(config.d_model, config.num_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(config.d_model, config.num_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(config.d_model, config.num_kv_heads * head_dim, bias=False)
            self.out_proj = nn.Linear(config.num_heads * head_dim, config.d_model, bias=False)
            if zero_output:
                self.out_proj.weight = mx.zeros_like(self.out_proj.weight)

        def __call__(
            self,
            hidden: Any,
            memory: Any,
            key_mask: Any | None = None,
        ) -> Any:
            batch, length, _dims = hidden.shape
            slots = int(memory.shape[1])
            query = self.q_proj(hidden).reshape(
                batch, length, config.num_heads, head_dim
            ).transpose(0, 2, 1, 3)
            key = self.k_proj(memory).reshape(
                batch, slots, config.num_kv_heads, head_dim
            ).transpose(0, 2, 1, 3)
            value = self.v_proj(memory).reshape(
                batch, slots, config.num_kv_heads, head_dim
            ).transpose(0, 2, 1, 3)
            if config.num_kv_heads != config.num_heads:
                repeats = config.num_heads // config.num_kv_heads
                key = mx.repeat(key, repeats=repeats, axis=1)
                value = mx.repeat(value, repeats=repeats, axis=1)
            attended = mx.fast.scaled_dot_product_attention(
                query,
                key,
                value,
                scale=head_dim ** -0.5,
                mask=(
                    mx.where(
                        key_mask[:, None, None, :] > 0,
                        mx.array(0.0, dtype=mx.float32),
                        mx.array(-1e9, dtype=mx.float32),
                    )
                    if key_mask is not None
                    else None
                ),
            )
            attended = attended.transpose(0, 2, 1, 3).reshape(
                batch, length, config.num_heads * head_dim
            )
            return self.out_proj(attended)

    class SourceEncoderBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention_norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
            self.attention = CausalAttention()
            self.ffn_norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
            self.feed_forward = SwiGLU()
            if source_expert_adapter_enabled:
                self.expert_adapter = ExpertAdapter(config.source_expert_adapter_dim)

        def __call__(self, hidden: Any, source_mask: Any) -> Any:
            key_mask = mx.where(
                source_mask[:, None, None, :] > 0,
                mx.array(0.0, dtype=mx.float32),
                mx.array(-1e9, dtype=mx.float32),
            )
            attended, _cache = self.attention(
                self.attention_norm(hidden), attention_mask=key_mask
            )
            hidden = hidden + attended
            hidden = hidden + self.feed_forward(self.ffn_norm(hidden))
            if source_expert_adapter_enabled:
                hidden = hidden + self.expert_adapter(hidden)
            return hidden

    class SwiGLU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Linear(config.d_model, config.ff_dim, bias=False)
            self.up = nn.Linear(config.d_model, config.ff_dim, bias=False)
            self.down = nn.Linear(config.ff_dim, config.d_model, bias=False)

        def __call__(self, hidden: Any) -> Any:
            return self.down(nn.silu(self.gate(hidden)) * self.up(hidden))

    class ExpertAdapter(nn.Module):
        def __init__(self, dimension: int) -> None:
            super().__init__()
            self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
            self.down = nn.Linear(config.d_model, dimension, bias=False)
            self.up = nn.Linear(dimension, config.d_model, bias=False)
            self.up.weight = mx.zeros_like(self.up.weight)

        def __call__(self, hidden: Any) -> Any:
            return self.up(nn.silu(self.down(self.norm(hidden))))

    class DecoderBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention_norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
            self.attention = CausalAttention()
            self.ffn_norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
            self.feed_forward = SwiGLU()
            if expert_adapter_enabled:
                self.expert_adapter = ExpertAdapter(config.expert_adapter_dim)
            if source_expert_adapter_enabled:
                self.source_expert_adapter = ExpertAdapter(
                    config.source_expert_adapter_dim
                )
            if state_enabled:
                self.state_embedding = nn.Embedding(config.state_memory_slots, config.d_model)
                self.state_candidate = nn.Linear(config.d_model * 2, config.d_model, bias=False)
                self.state_gate = nn.Linear(config.d_model * 2, config.d_model, bias=True)

        def initial_memory(self, batch: int) -> Any:
            roles = self.state_embedding(mx.arange(config.state_memory_slots, dtype=mx.int32))
            return mx.broadcast_to(roles[None, :, :], (batch, config.state_memory_slots, config.d_model))

        def update_memory(
            self,
            memory: Any,
            hidden: Any,
            role_weights: Any,
            pending_sum: Any,
            pending_count: Any,
            *,
            commit: bool,
        ) -> tuple[Any, Any, Any]:
            weights = role_weights.transpose(0, 2, 1)
            pending_sum = pending_sum + mx.matmul(weights, hidden)
            pending_count = pending_count + mx.sum(weights, axis=-1, keepdims=True)
            if not commit:
                return memory, pending_sum, pending_count
            pooled = pending_sum / mx.maximum(pending_count, 1.0)
            joined = mx.concatenate([memory, pooled], axis=-1)
            candidate = mx.tanh(self.state_candidate(joined))
            gate = mx.sigmoid(self.state_gate(joined))
            updated = gate * memory + (1.0 - gate) * candidate
            present = pending_count > 0
            next_memory = mx.where(present, updated, memory)
            return next_memory, mx.zeros_like(pending_sum), mx.zeros_like(pending_count)

        def __call__(
            self,
            hidden: Any,
            cache: tuple[Any, ...] | None = None,
            role_weights: Any | None = None,
            commit_state: bool = False,
            plan_memory: Any | None = None,
            plan_access: Any | None = None,
            source_memory: Any | None = None,
            source_mask: Any | None = None,
            source_access: Any | None = None,
            attention_mask: Any | None = None,
        ) -> tuple[Any, tuple[Any, ...]]:
            token_cache = (cache[0], cache[1]) if cache is not None else None
            memory = cache[2] if cache is not None and len(cache) >= 3 else None
            pending_sum = cache[3] if cache is not None and len(cache) == 5 else None
            pending_count = cache[4] if cache is not None and len(cache) == 5 else None
            if state_enabled and memory is None:
                memory = self.initial_memory(int(hidden.shape[0]))
            if state_enabled and pending_sum is None:
                pending_sum = mx.zeros_like(memory)
                pending_count = mx.zeros((*memory.shape[:2], 1), dtype=mx.float32)
            attended, next_cache = self.attention(
                self.attention_norm(hidden),
                token_cache,
                mx.zeros_like(memory) if config.state_memory_ablation == "zero" else memory,
                role_weights,
                attention_mask,
            )
            hidden = hidden + attended
            if source_encoder_enabled and source_memory is not None:
                source_attended = self.source_attention(
                    self.source_attention_norm(hidden), source_memory, source_mask
                )
                access = (
                    source_access
                    if source_access is not None
                    else mx.ones(hidden.shape[:2], dtype=mx.float32)
                )
                hidden = hidden + source_attended * access[:, :, None]
                if source_expert_adapter_enabled:
                    hidden = hidden + self.source_expert_adapter(hidden) * access[:, :, None]
            if plan_slot_attention_enabled and plan_memory is not None:
                plan_attended = self.plan_attention(
                    self.plan_attention_norm(hidden), plan_memory
                )
                access = (
                    plan_access
                    if plan_access is not None
                    else mx.ones(hidden.shape[:2], dtype=mx.float32)
                )
                hidden = hidden + plan_attended * access[:, :, None]
            hidden = hidden + self.feed_forward(self.ffn_norm(hidden))
            if expert_adapter_enabled:
                hidden = hidden + self.expert_adapter(hidden)
            if not state_enabled:
                return hidden, next_cache
            next_memory = memory
            if config.state_memory_ablation != "zero" and role_weights is not None:
                next_memory, pending_sum, pending_count = self.update_memory(
                    memory,
                    hidden,
                    role_weights,
                    pending_sum,
                    pending_count,
                    commit=commit_state,
                )
            return hidden, (
                next_cache[0],
                next_cache[1],
                next_memory,
                pending_sum,
                pending_count,
            )

    class StandardCausalTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.layers = [DecoderBlock() for _ in range(config.num_layers)]
            self.final_norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
            self.scale = math.sqrt(config.d_model)
            self.copy_auxiliary_loss_weight = float(
                config.source_copy_auxiliary_loss_weight
            )
            if source_encoder_enabled:
                self.source_layers = [
                    SourceEncoderBlock() for _ in range(config.source_encoder_layers)
                ]
                self.source_final_norm = nn.RMSNorm(
                    config.d_model, eps=config.rms_norm_eps
                )
                for layer in self.layers:
                    layer.source_attention_norm = nn.RMSNorm(
                        config.d_model, eps=config.rms_norm_eps
                    )
                    layer.source_attention = CrossAttention()
                if pointer_generator_enabled:
                    self.copy_query = nn.Linear(config.d_model, config.d_model, bias=False)
                    self.copy_key = nn.Linear(config.d_model, config.d_model, bias=False)
                    self.copy_gate = nn.Linear(config.d_model, 1, bias=True)
            if plan_slot_attention_enabled:
                for layer in self.layers:
                    layer.plan_attention_norm = nn.RMSNorm(
                        config.d_model, eps=config.rms_norm_eps
                    )
                    layer.plan_attention = CrossAttention(zero_output=True)
            if plan_enabled:
                plan_dim = config.semantic_plan_bottleneck_dim or config.d_model
                self.semantic_plan_encoder = (
                    nn.Linear(config.d_model, plan_dim, bias=False)
                    if plan_dim != config.d_model
                    else None
                )
                self.semantic_plan_classifier = nn.Linear(
                    plan_dim, config.semantic_plan_feature_count, bias=True
                )
                self.semantic_plan_features = nn.Embedding(
                    config.semantic_plan_feature_count, plan_dim
                )
                self.semantic_plan_projection = nn.Linear(
                    plan_dim, config.d_model, bias=False
                )
                if not plan_slot_attention_enabled:
                    self.semantic_plan_projection.weight = mx.zeros_like(
                        self.semantic_plan_projection.weight
                    )

        def role_weights(self, tokens: Any) -> Any | None:
            if not state_enabled:
                return None
            weights = state_role_lookup[tokens]
            if config.state_memory_ablation == "shuffle":
                permutation = mx.arange(config.state_memory_slots - 1, -1, -1, dtype=mx.int32)
                weights = weights[:, :, permutation]
            return weights

        def freeze_to_expert_adapter(self) -> None:
            self.freeze_to_language_expert("adapter_only")

        def freeze_to_language_expert(self, scope: str) -> None:
            if not expert_adapter_enabled:
                raise ValueError("model has no expert adapter to train")
            if scope not in {
                "adapter_only",
                "source_conditioned_delta",
                "low_rank_source_adapters",
            }:
                raise ValueError(f"unsupported language expert scope: {scope}")
            self.freeze()
            for layer in self.layers:
                layer.expert_adapter.unfreeze()
            if scope == "source_conditioned_delta":
                if not source_encoder_enabled or not pointer_generator_enabled:
                    raise ValueError(
                        "source-conditioned expert scope requires encoder-decoder pointer mode"
                    )
                for layer in self.layers:
                    layer.source_attention_norm.unfreeze()
                    layer.source_attention.unfreeze()
                for layer in self.source_layers:
                    layer.unfreeze()
                self.source_final_norm.unfreeze()
                self.copy_query.unfreeze()
                self.copy_key.unfreeze()
                self.copy_gate.unfreeze()
            elif scope == "low_rank_source_adapters":
                if not source_encoder_enabled or not source_expert_adapter_enabled:
                    raise ValueError(
                        "low-rank source expert scope requires source expert adapters"
                    )
                for layer in self.layers:
                    layer.source_expert_adapter.unfreeze()
                for layer in self.source_layers:
                    layer.expert_adapter.unfreeze()
                if pointer_generator_enabled:
                    self.copy_gate.unfreeze()

        def sequence_attention_mask(self, tokens: Any, cache: Any | None) -> Any | None:
            if config.attention_policy != "prefix_lm" or cache is not None:
                return None
            batch, length = int(tokens.shape[0]), int(tokens.shape[1])
            if length <= 1:
                return None
            separator = tokens == config.source_target_separator_token_id
            has_separator = mx.sum(separator.astype(mx.int32), axis=1) > 0
            separator_position = mx.argmax(separator.astype(mx.int32), axis=1)
            query_positions = mx.arange(length, dtype=mx.int32)[None, :, None]
            key_positions = mx.arange(length, dtype=mx.int32)[None, None, :]
            causal = key_positions <= query_positions
            source_query = query_positions <= separator_position[:, None, None]
            source_key = key_positions <= separator_position[:, None, None]
            prefix_bidirectional = (
                source_query & source_key & has_separator[:, None, None]
            )
            allowed = causal | prefix_bidirectional
            additive = mx.where(
                allowed,
                mx.array(0.0, dtype=mx.float32),
                mx.array(-1e9, dtype=mx.float32),
            )
            return additive[:, None, :, :].reshape(batch, 1, length, length)

        def source_partition(self, tokens: Any) -> tuple[Any, Any, Any]:
            """Return source keys, target access, and separator presence per row."""

            separator = tokens == config.source_target_separator_token_id
            seen_separator = mx.cumsum(separator.astype(mx.int32), axis=1)
            has_separator = (mx.sum(separator.astype(mx.int32), axis=1) > 0).astype(
                mx.float32
            )
            source_mask = (seen_separator == 0).astype(mx.float32) * has_separator[:, None]
            target_access = ((seen_separator > 0) & ~separator).astype(mx.float32)
            return source_mask, target_access, has_separator

        def encode_source(
            self, tokens: Any
        ) -> tuple[Any | None, Any | None, Any | None, Any | None]:
            """Encode only the prompt partition; target values cannot affect this memory."""

            if not source_encoder_enabled:
                return None, None, None, None
            source_mask, target_access, has_separator = self.source_partition(tokens)
            if not bool(mx.any(has_separator > 0)):
                return None, None, None, None
            hidden = self.token_embedding(tokens) * self.scale
            hidden = hidden * source_mask[:, :, None]
            for layer in self.source_layers:
                hidden = layer(hidden, source_mask)
                hidden = hidden * source_mask[:, :, None]
            memory = self.source_final_norm(hidden) * source_mask[:, :, None]
            copy_ids = source_to_target_lookup[tokens] if pointer_generator_enabled else None
            return memory, source_mask, target_access, copy_ids

        def output_logits(
            self,
            hidden: Any,
            source_memory: Any | None,
            source_mask: Any | None,
            source_copy_ids: Any | None,
        ) -> tuple[Any, dict[str, Any] | None]:
            generator_logits = self.token_embedding.as_linear(hidden)
            if (
                not pointer_generator_enabled
                or source_memory is None
                or source_mask is None
                or source_copy_ids is None
            ):
                return generator_logits, None
            query = self.copy_query(hidden)
            key = self.copy_key(source_memory)
            pointer_scores = mx.matmul(query, key.transpose(0, 2, 1)) / math.sqrt(
                config.d_model
            )
            valid = (source_mask > 0) & (source_copy_ids >= 0)
            source_length = int(source_copy_ids.shape[1])
            positions = mx.arange(source_length, dtype=mx.int32)
            same_id = source_copy_ids[:, :, None] == source_copy_ids[:, None, :]
            later = positions[None, None, :] > positions[None, :, None]
            has_later_copy = mx.any(
                same_id & later & valid[:, None, :], axis=-1
            )
            unique_valid = valid & ~has_later_copy
            pointer_scores = mx.where(
                unique_valid[:, None, :],
                pointer_scores,
                mx.array(-1e9, dtype=mx.float32),
            )
            indices = mx.broadcast_to(
                mx.maximum(source_copy_ids, 0)[:, None, :], pointer_scores.shape
            )
            pointer_logits = mx.full(generator_logits.shape, -1e9, dtype=mx.float32)
            pointer_logits = mx.put_along_axis(
                pointer_logits, indices, pointer_scores, axis=2
            )
            generator_log_probs = generator_logits - mx.logsumexp(
                generator_logits, axis=-1, keepdims=True
            )
            pointer_log_probs = pointer_logits - mx.logsumexp(
                pointer_logits, axis=-1, keepdims=True
            )
            gate = mx.sigmoid(self.copy_gate(hidden))
            logits = mx.logaddexp(
                generator_log_probs + mx.log(mx.maximum(gate, 1e-6)),
                pointer_log_probs + mx.log(mx.maximum(1.0 - gate, 1e-6)),
            )
            return logits, {
                "pointer_scores": pointer_scores,
                "source_copy_ids": source_copy_ids,
                "source_copy_valid": unique_valid,
                "generator_gate": gate[:, :, 0],
            }

        def conditioned_embeddings(
            self,
            tokens: Any,
            cached_plan_context: Any | None,
            source_mask: Any | None = None,
        ) -> tuple[Any, Any | None, Any | None, Any | None]:
            hidden = self.token_embedding(tokens) * self.scale
            if source_encoder_enabled and source_mask is not None:
                neutral = self.token_embedding(
                    mx.zeros(tokens.shape, dtype=mx.int32)
                ) * self.scale
                hidden = mx.where(source_mask[:, :, None] > 0, neutral, hidden)
            if not plan_enabled:
                return hidden, None, None, None
            if cached_plan_context is not None:
                context = cached_plan_context
                plan_logits = None
                target_mask = mx.ones(tokens.shape, dtype=mx.float32)
            else:
                separator = tokens == config.semantic_plan_separator_token_id
                seen_separator = mx.cumsum(separator.astype(mx.int32), axis=1)
                has_separator = (mx.sum(separator.astype(mx.int32), axis=1) > 0).astype(
                    mx.float32
                )
                source_mask = (seen_separator == 0).astype(mx.float32) * has_separator[:, None]
                denominator = mx.maximum(mx.sum(source_mask, axis=1, keepdims=True), 1.0)
                source_summary = mx.sum(hidden * source_mask[:, :, None], axis=1) / denominator
                plan_summary = (
                    self.semantic_plan_encoder(source_summary)
                    if self.semantic_plan_encoder is not None
                    else source_summary
                )
                plan_logits = self.semantic_plan_classifier(plan_summary)
                feature_matrix = self.semantic_plan_features(
                    mx.arange(config.semantic_plan_feature_count, dtype=mx.int32)
                )
                if plan_slot_attention_enabled:
                    slot_count = config.semantic_plan_slot_count
                    slot_width = config.semantic_plan_feature_count // slot_count
                    slot_logits = plan_logits.reshape(
                        int(tokens.shape[0]), slot_count, slot_width
                    )
                    slot_features = feature_matrix.reshape(
                        slot_count, slot_width, int(feature_matrix.shape[-1])
                    )
                    if config.semantic_plan_probability_mode == "factorized_step":
                        groups = tuple(
                            int(value) for value in config.semantic_plan_factor_group_sizes
                        )
                        presence = mx.sigmoid(slot_logits[:, :, :1])
                        presence_state = presence[:, :, :, None] * slot_features[
                            None, :, :1, :
                        ]
                        factor_states = [presence_state[:, :, 0, :]]
                        offset = 1
                        for width in groups[1:]:
                            probabilities = mx.softmax(
                                slot_logits[:, :, offset : offset + width], axis=-1
                            )
                            factor_states.append(
                                mx.sum(
                                    probabilities[:, :, :, None]
                                    * slot_features[
                                        None, :, offset : offset + width, :
                                    ],
                                    axis=2,
                                )
                            )
                            offset += width
                        slot_state = presence * mx.mean(
                            mx.stack(factor_states, axis=2), axis=2
                        )
                    elif config.semantic_plan_probability_mode == "slot_categorical":
                        empty_logits = mx.zeros(
                            (int(tokens.shape[0]), slot_count, 1), dtype=slot_logits.dtype
                        )
                        slot_probabilities = mx.softmax(
                            mx.concatenate([empty_logits, slot_logits], axis=-1),
                            axis=-1,
                        )[:, :, 1:]
                    else:
                        slot_probabilities = mx.sigmoid(slot_logits)
                    if config.semantic_plan_probability_mode != "factorized_step":
                        slot_mass = mx.sum(slot_probabilities, axis=-1, keepdims=True)
                        slot_state = mx.sum(
                            slot_probabilities[:, :, :, None]
                            * slot_features[None, :, :, :],
                            axis=2,
                        ) / mx.maximum(slot_mass, 1.0)
                    context = self.semantic_plan_projection(slot_state) * has_separator[
                        :, None, None
                    ]
                else:
                    probabilities = mx.sigmoid(plan_logits)
                    context = mx.matmul(probabilities, feature_matrix) / mx.maximum(
                        mx.sum(probabilities, axis=-1, keepdims=True), 1.0
                    )
                    context = self.semantic_plan_projection(context) * has_separator[:, None]
                target_mask = ((seen_separator > 0) & ~separator).astype(mx.float32)
            if not plan_slot_attention_enabled:
                hidden = hidden + context[:, None, :] * target_mask[:, :, None]
            return hidden, context, plan_logits, target_mask

        def __call__(
            self,
            tokens: Any,
            cache: list[tuple[Any, ...]] | None = None,
            *,
            return_plan_logits: bool = False,
            return_copy_aux: bool = False,
        ) -> Any:
            cached_plan_context = None
            cached_source_memory = None
            cached_source_mask = None
            cached_source_copy_ids = None
            layer_cache_input = cache
            trailing = 0
            if (
                plan_enabled
                and cache is not None
                and len(cache) > config.num_layers
            ):
                cached_plan_context = cache[-1][0]
                trailing += 1
            if (
                source_encoder_enabled
                and cache is not None
                and len(cache) > config.num_layers + trailing
            ):
                source_entry = cache[-(trailing + 1)]
                cached_source_memory, cached_source_mask, cached_source_copy_ids = source_entry
                trailing += 1
            if trailing:
                layer_cache_input = cache[:-trailing]
            source_memory = cached_source_memory
            source_mask = cached_source_mask
            source_copy_ids = cached_source_copy_ids
            source_access = None
            if source_encoder_enabled:
                if source_memory is None:
                    (
                        source_memory,
                        source_mask,
                        source_access,
                        source_copy_ids,
                    ) = self.encode_source(tokens)
                elif cache is not None:
                    source_access = mx.ones(tokens.shape, dtype=mx.float32)
            conditioned_hidden, plan_context, plan_logits, plan_access = self.conditioned_embeddings(
                tokens,
                cached_plan_context,
                source_mask if cached_source_memory is None else None,
            )
            attention_mask = self.sequence_attention_mask(tokens, cache)
            if not state_enabled:
                hidden = conditioned_hidden
                next_cache: list[tuple[Any, ...]] = []
                for index, layer in enumerate(self.layers):
                    layer_cache = layer_cache_input[index] if layer_cache_input is not None else None
                    hidden, layer_next = layer(
                        hidden,
                        layer_cache,
                        plan_memory=plan_context if plan_slot_attention_enabled else None,
                        plan_access=plan_access,
                        source_memory=source_memory,
                        source_mask=source_mask,
                        source_access=source_access,
                        attention_mask=attention_mask,
                    )
                    next_cache.append(layer_next)
                if source_encoder_enabled and source_memory is not None:
                    next_cache.append((source_memory, source_mask, source_copy_ids))
                if plan_enabled:
                    next_cache.append((plan_context,))
                logits, copy_aux = self.output_logits(
                    self.final_norm(hidden),
                    source_memory,
                    source_mask,
                    source_copy_ids,
                )
                if return_plan_logits and return_copy_aux:
                    return logits, next_cache, plan_logits, copy_aux
                if return_plan_logits:
                    return logits, next_cache, plan_logits
                if return_copy_aux:
                    return logits, next_cache, copy_aux
                return logits, next_cache

            role_weights = self.role_weights(tokens)
            current_cache = layer_cache_input
            outputs = []
            offset = int(layer_cache_input[0][0].shape[2]) if layer_cache_input is not None else 0
            start = 0
            while start < int(tokens.shape[1]):
                position_in_chunk = (offset + start) % config.state_memory_chunk_size
                remaining_in_chunk = config.state_memory_chunk_size - position_in_chunk
                stop = min(int(tokens.shape[1]), start + remaining_in_chunk)
                hidden = conditioned_hidden[:, start:stop, :]
                chunk_roles = role_weights[:, start:stop, :]
                commit_state = (offset + stop) % config.state_memory_chunk_size == 0
                next_cache = []
                for index, layer in enumerate(self.layers):
                    layer_cache = current_cache[index] if current_cache is not None else None
                    hidden, layer_next = layer(
                        hidden,
                        layer_cache,
                        chunk_roles,
                        commit_state=commit_state,
                        plan_memory=plan_context if plan_slot_attention_enabled else None,
                        plan_access=plan_access[:, start:stop] if plan_access is not None else None,
                    )
                    next_cache.append(layer_next)
                outputs.append(self.final_norm(hidden))
                current_cache = next_cache
                start = stop
            logits = self.token_embedding.as_linear(mx.concatenate(outputs, axis=1))
            final_cache = current_cache or []
            if plan_enabled:
                final_cache.append((plan_context,))
            return (logits, final_cache, plan_logits) if return_plan_logits else (logits, final_cache)

    return StandardCausalTransformer()


def parameter_count(model: Any, mlx_utils: Any) -> int:
    return int(sum(value.size for _name, value in mlx_utils.tree_flatten(model.parameters())))
