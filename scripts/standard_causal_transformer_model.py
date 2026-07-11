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
    state_memory_slots: int = 0
    state_memory_chunk_size: int = 32
    state_memory_local_window: int = 96
    state_memory_mode: str = "none"
    state_memory_ablation: str = "none"
    state_memory_read_policy: str = "unrestricted"

    def validate(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must divide evenly across query heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("query heads must divide evenly across KV heads")
        if self.num_layers <= 0 or self.vocab_size <= 0 or self.ff_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.state_memory_mode not in {"none", "semantic_roles", "hash_control"}:
            raise ValueError("state memory mode must be none, semantic_roles, or hash_control")
        if self.state_memory_ablation not in {"none", "zero", "shuffle"}:
            raise ValueError("state memory ablation must be none, zero, or shuffle")
        if self.state_memory_read_policy not in {"unrestricted", "role_dependency"}:
            raise ValueError("state memory read policy must be unrestricted or role_dependency")
        if self.state_memory_mode == "none" and self.state_memory_slots != 0:
            raise ValueError("state memory slots must be zero when state memory is disabled")
        if self.state_memory_mode != "none" and self.state_memory_slots <= 1:
            raise ValueError("enabled state memory requires at least two slots")
        if self.state_memory_chunk_size <= 0 or self.state_memory_local_window <= 0:
            raise ValueError("state memory chunk and local-window sizes must be positive")
        if self.state_memory_chunk_size > self.state_memory_local_window:
            raise ValueError("state memory chunk size cannot exceed its local attention window")


def build_model(
    config: CausalTransformerConfig,
    *,
    mx: Any,
    nn: Any,
    state_role_lookup: Any | None = None,
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
            mask = "causal" if cache is None and length > 1 and memory is None else None
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

    class SwiGLU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Linear(config.d_model, config.ff_dim, bias=False)
            self.up = nn.Linear(config.d_model, config.ff_dim, bias=False)
            self.down = nn.Linear(config.ff_dim, config.d_model, bias=False)

        def __call__(self, hidden: Any) -> Any:
            return self.down(nn.silu(self.gate(hidden)) * self.up(hidden))

    class DecoderBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention_norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
            self.attention = CausalAttention()
            self.ffn_norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
            self.feed_forward = SwiGLU()
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
            )
            hidden = hidden + attended
            hidden = hidden + self.feed_forward(self.ffn_norm(hidden))
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

        def role_weights(self, tokens: Any) -> Any | None:
            if not state_enabled:
                return None
            weights = state_role_lookup[tokens]
            if config.state_memory_ablation == "shuffle":
                permutation = mx.arange(config.state_memory_slots - 1, -1, -1, dtype=mx.int32)
                weights = weights[:, :, permutation]
            return weights

        def __call__(
            self,
            tokens: Any,
            cache: list[tuple[Any, ...]] | None = None,
        ) -> tuple[Any, list[tuple[Any, ...]]]:
            if not state_enabled:
                hidden = self.token_embedding(tokens) * self.scale
                next_cache: list[tuple[Any, ...]] = []
                for index, layer in enumerate(self.layers):
                    layer_cache = cache[index] if cache is not None else None
                    hidden, layer_next = layer(hidden, layer_cache)
                    next_cache.append(layer_next)
                logits = self.token_embedding.as_linear(self.final_norm(hidden))
                return logits, next_cache

            role_weights = self.role_weights(tokens)
            current_cache = cache
            outputs = []
            offset = int(cache[0][0].shape[2]) if cache is not None else 0
            start = 0
            while start < int(tokens.shape[1]):
                position_in_chunk = (offset + start) % config.state_memory_chunk_size
                remaining_in_chunk = config.state_memory_chunk_size - position_in_chunk
                stop = min(int(tokens.shape[1]), start + remaining_in_chunk)
                hidden = self.token_embedding(tokens[:, start:stop]) * self.scale
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
                    )
                    next_cache.append(layer_next)
                outputs.append(self.final_norm(hidden))
                current_cache = next_cache
                start = stop
            logits = self.token_embedding.as_linear(mx.concatenate(outputs, axis=1))
            return logits, current_cache or []

    return StandardCausalTransformer()


def parameter_count(model: Any, mlx_utils: Any) -> int:
    return int(sum(value.size for _name, value in mlx_utils.tree_flatten(model.parameters())))
