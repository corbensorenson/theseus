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

    def validate(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must divide evenly across query heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("query heads must divide evenly across KV heads")
        if self.num_layers <= 0 or self.vocab_size <= 0 or self.ff_dim <= 0:
            raise ValueError("model dimensions must be positive")


def build_model(config: CausalTransformerConfig, *, mx: Any, nn: Any) -> Any:
    """Build a pre-norm RoPE/GQA/SwiGLU causal LM with tied embeddings."""

    config.validate()
    head_dim = config.d_model // config.num_heads
    half_head_dim = head_dim // 2
    rope_inverse_frequency = mx.array(
        [config.rope_base ** (-(2.0 * index) / head_dim) for index in range(half_head_dim)],
        dtype=mx.float32,
    )

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
            mask = "causal" if cache is None and length > 1 else None
            attention_key = key
            attention_value = value
            if config.num_kv_heads != config.num_heads:
                repeats = config.num_heads // config.num_kv_heads
                attention_key = mx.repeat(key, repeats=repeats, axis=1)
                attention_value = mx.repeat(value, repeats=repeats, axis=1)
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

        def __call__(
            self,
            hidden: Any,
            cache: tuple[Any, Any] | None = None,
        ) -> tuple[Any, tuple[Any, Any]]:
            attended, next_cache = self.attention(self.attention_norm(hidden), cache)
            hidden = hidden + attended
            hidden = hidden + self.feed_forward(self.ffn_norm(hidden))
            return hidden, next_cache

    class StandardCausalTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.layers = [DecoderBlock() for _ in range(config.num_layers)]
            self.final_norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
            self.scale = math.sqrt(config.d_model)

        def __call__(
            self,
            tokens: Any,
            cache: list[tuple[Any, Any]] | None = None,
        ) -> tuple[Any, list[tuple[Any, Any]]]:
            hidden = self.token_embedding(tokens) * self.scale
            next_cache: list[tuple[Any, Any]] = []
            for index, layer in enumerate(self.layers):
                layer_cache = cache[index] if cache is not None else None
                hidden, layer_next = layer(hidden, layer_cache)
                next_cache.append(layer_next)
            logits = self.token_embedding.as_linear(self.final_norm(hidden))
            return logits, next_cache

    return StandardCausalTransformer()


def parameter_count(model: Any, mlx_utils: Any) -> int:
    return int(sum(value.size for _name, value in mlx_utils.tree_flatten(model.parameters())))
