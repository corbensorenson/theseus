#!/usr/bin/env python3
"""Shared MLX strict-generator model and sparse-specialist architecture.

The sparse path executes only the selected experts for each token.  It does not
render code, inspect verifier data, or change the candidate-integrity contract.
"""

from __future__ import annotations

import math
from typing import Any


SPECIALIST_CORE_MODES = {"disabled", "sparse_moe", "dense_active_control"}


def normalize_specialist_core_config(value: Any) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, dict) else {}
    enabled = bool(raw.get("enabled"))
    mode = str(raw.get("mode") or ("sparse_moe" if enabled else "disabled"))
    if not enabled:
        mode = "disabled"
    if mode not in SPECIALIST_CORE_MODES:
        raise ValueError(f"unsupported specialist core mode: {mode}")
    num_experts = max(1, int(raw.get("num_experts") or 1))
    top_k = max(1, int(raw.get("top_k") or 1))
    if top_k > num_experts:
        raise ValueError("specialist core top_k cannot exceed num_experts")
    expert_hidden_dim = max(1, int(raw.get("expert_hidden_dim") or 1))
    if mode == "dense_active_control":
        num_experts = 1
        top_k = 1
    return {
        "enabled": mode != "disabled",
        "mode": mode,
        "num_experts": num_experts,
        "top_k": top_k,
        "expert_hidden_dim": expert_hidden_dim,
        "router_aux_loss_weight": max(0.0, float(raw.get("router_aux_loss_weight") or 0.0)),
        "router_z_loss_weight": max(0.0, float(raw.get("router_z_loss_weight") or 0.0)),
        "router_supervision_loss_weight": max(
            0.0, float(raw.get("router_supervision_loss_weight") or 0.0)
        ),
        "routing_policy": "token_top_k_no_capacity_drop_v1" if mode == "sparse_moe" else mode,
        "candidate_generation_credit": 0,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
    }


def specialist_core_parameter_estimate(d_model: int, value: Any) -> dict[str, Any]:
    cfg = normalize_specialist_core_config(value)
    d_model = max(1, int(d_model))
    hidden = int(cfg["expert_hidden_dim"])
    expert_parameters = 3 * d_model * hidden
    if cfg["mode"] == "sparse_moe":
        total = int(cfg["num_experts"]) * expert_parameters + d_model * int(cfg["num_experts"])
        active = int(cfg["top_k"]) * expert_parameters + d_model * int(cfg["num_experts"])
    elif cfg["mode"] == "dense_active_control":
        total = expert_parameters
        active = expert_parameters
    else:
        total = 0
        active = 0
    return {
        **cfg,
        "expert_parameter_count": expert_parameters,
        "specialist_total_parameter_count": total,
        "specialist_active_parameter_count_per_token": active,
        "specialist_active_parameter_fraction": round(active / total, 6) if total else 0.0,
    }


def matched_dense_control_config(value: Any) -> dict[str, Any]:
    sparse = normalize_specialist_core_config(value)
    if sparse["mode"] != "sparse_moe":
        raise ValueError("matched dense control requires a sparse_moe source config")
    return normalize_specialist_core_config(
        {
            "enabled": True,
            "mode": "dense_active_control",
            "expert_hidden_dim": int(sparse["expert_hidden_dim"]) * int(sparse["top_k"]),
        }
    )


class MlxStrictGenerator:
    """Encoder-decoder generator with an optional real sparse specialist block."""

    def __init__(
        self,
        *,
        source_vocab_size: int,
        target_vocab_size: int,
        max_source_len: int,
        max_target_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        semantic_slot_role_count: int,
        semantic_slot_head: bool = True,
        body_action_role_count: int,
        body_operand_role_count: int,
        body_state_event_role_count: int,
        body_executable_span_role_count: int,
        auxiliary_head_policy: str = "legacy_materialized_v1",
        output_projection_policy: str = "independent_output_v1",
        coupled_state_body_constructor: bool = False,
        coupled_state_body_constructor_scale: float = 0.35,
        body_executable_span_head: bool = False,
        executable_span_body_constructor: bool = False,
        executable_span_body_constructor_scale: float = 0.25,
        specialist_core: Any = None,
        specialist_token_expert_ids: list[list[int]] | None = None,
        mx: Any,
        nn: Any,
    ) -> None:
        specialist_cfg = normalize_specialist_core_config(specialist_core)
        if auxiliary_head_policy not in {
            "legacy_materialized_v1",
            "shared_factorized_on_demand_v1",
        }:
            raise ValueError(f"unsupported auxiliary_head_policy={auxiliary_head_policy!r}")
        if output_projection_policy not in {
            "independent_output_v1",
            "tied_target_embedding_v1",
        }:
            raise ValueError(f"unsupported output_projection_policy={output_projection_policy!r}")
        token_expert_ids = tuple(
            tuple(int(expert_id) for expert_id in row[: int(specialist_cfg["top_k"])])
            for row in (specialist_token_expert_ids or [])
        )

        class _Expert(nn.Module):
            def __init__(self, hidden_dim: int) -> None:
                super().__init__()
                self.gate = nn.Linear(d_model, hidden_dim, bias=False)
                self.up = nn.Linear(d_model, hidden_dim, bias=False)
                self.down = nn.Linear(hidden_dim, d_model, bias=False)

            def __call__(self, hidden: Any) -> Any:
                return self.down(nn.silu(self.gate(hidden)) * self.up(hidden))

        class _SparseMoE(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = int(specialist_cfg["num_experts"])
                self.top_k = int(specialist_cfg["top_k"])
                self.router = nn.Linear(d_model, self.num_experts, bias=False)
                self.experts = [
                    _Expert(int(specialist_cfg["expert_hidden_dim"]))
                    for _ in range(self.num_experts)
                ]

            def route(self, hidden: Any) -> tuple[Any, Any, Any]:
                logits = self.router(hidden)
                indices = mx.argsort(logits, axis=-1)[..., -self.top_k :]
                selected_logits = mx.take_along_axis(logits, indices, axis=-1)
                weights = mx.softmax(selected_logits, axis=-1)
                return logits, indices, weights

            def auxiliary_loss(
                self, logits: Any, indices: Any, valid_mask: Any | None = None
            ) -> tuple[Any, Any, Any]:
                probabilities = mx.softmax(logits, axis=-1)
                flat_probabilities = probabilities.reshape((-1, self.num_experts))
                flat_indices = indices.reshape((-1, self.top_k))
                valid = (
                    valid_mask.reshape((-1,)).astype(mx.float32)
                    if valid_mask is not None
                    else mx.ones((flat_indices.shape[0],), dtype=mx.float32)
                )
                valid_count = mx.maximum(mx.sum(valid), mx.array(1.0, dtype=mx.float32))
                probability_mass = mx.sum(flat_probabilities * valid[:, None], axis=0) / valid_count
                soft_importance = float(self.num_experts) * mx.sum(mx.square(probability_mass))
                token_entropy = mx.sum(
                    -mx.sum(
                        flat_probabilities * mx.log(mx.maximum(flat_probabilities, 1e-9)),
                        axis=-1,
                    )
                    * valid
                ) / valid_count
                mass_entropy = -mx.sum(
                    probability_mass * mx.log(mx.maximum(probability_mass, 1e-9))
                )
                # Minimize batch imbalance while maximizing mutual information:
                # routes should be decisive per token but diverse over the batch.
                balance = soft_importance + token_entropy - mass_entropy
                flat_z = mx.square(mx.logsumexp(logits, axis=-1)).reshape((-1,))
                z_loss = mx.sum(flat_z * valid) / valid_count
                weighted = (
                    float(specialist_cfg["router_aux_loss_weight"]) * balance
                    + float(specialist_cfg["router_z_loss_weight"]) * z_loss
                )
                return weighted, balance, z_loss

            def __call__(self, hidden: Any, valid_mask: Any | None = None) -> tuple[Any, dict[str, Any]]:
                logits, indices, weights = self.route(hidden)
                flat_hidden = hidden.reshape((-1, d_model))
                repeated_hidden = mx.repeat(flat_hidden[:, None, :], self.top_k, axis=1).reshape(
                    (-1, d_model)
                )
                assignments = indices.reshape((-1,))
                sort_order = mx.argsort(assignments)
                sorted_hidden = repeated_hidden[sort_order]
                sorted_assignments = assignments[sort_order]
                expert_outputs: list[Any] = []
                cursor = 0
                for expert_id, expert in enumerate(self.experts):
                    count = int(mx.sum(sorted_assignments == expert_id).item())
                    if count > 0:
                        segment = sorted_hidden[cursor : cursor + count]
                        expert_outputs.append(expert(segment))
                    cursor += count
                sorted_output = mx.concatenate(expert_outputs, axis=0)
                inverse_order = mx.argsort(sort_order)
                routed = sorted_output[inverse_order].reshape(
                    (*hidden.shape[:-1], self.top_k, d_model)
                )
                combined = mx.sum(routed * weights[..., None], axis=-2)
                weighted_aux, balance, z_loss = self.auxiliary_loss(logits, indices, valid_mask)
                return combined, {
                    "weighted_aux_loss": weighted_aux,
                    "balance_loss": balance,
                    "router_z_loss": z_loss,
                    "indices": indices,
                    "weights": weights,
                    "logits": logits,
                }

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.source_embedding = nn.Embedding(source_vocab_size, d_model)
                self.target_embedding = nn.Embedding(target_vocab_size, d_model)
                self.source_position = mx.zeros((1, max_source_len, d_model), dtype=mx.float32)
                self.target_position = mx.zeros((1, max_target_len, d_model), dtype=mx.float32)
                self.encoder = nn.TransformerEncoder(
                    num_layers, d_model, nhead, dim_feedforward, 0.0, nn.gelu, True
                )
                self.decoder = nn.TransformerDecoder(
                    num_layers, d_model, nhead, dim_feedforward, 0.0, nn.gelu, True
                )
                self.output_projection_policy = output_projection_policy
                if output_projection_policy == "independent_output_v1":
                    self.output = nn.Linear(d_model, target_vocab_size)
                else:
                    self.output_bias = mx.zeros((target_vocab_size,), dtype=mx.float32)
                self.auxiliary_head_policy = auxiliary_head_policy
                if auxiliary_head_policy == "legacy_materialized_v1":
                    self.plan_router = nn.Linear(d_model, target_vocab_size)
                    self.slot_router = nn.Linear(d_model, target_vocab_size * semantic_slot_role_count)
                    self.body_transition_router = nn.Linear(d_model, target_vocab_size)
                else:
                    # Share the trained token projection for vocab-sized auxiliary
                    # logits and learn only compact role-specific query transforms.
                    self.semantic_slot_head_enabled = bool(semantic_slot_head)
                    if self.semantic_slot_head_enabled:
                        self.slot_role_router = nn.Linear(
                            d_model, d_model * semantic_slot_role_count, bias=False
                        )
                self.body_action_router = nn.Linear(d_model, body_action_role_count)
                self.body_operand_router = nn.Linear(d_model, body_operand_role_count)
                self.body_state_event_router = nn.Linear(d_model, body_state_event_role_count)
                self.coupled_state_body_constructor_enabled = bool(coupled_state_body_constructor)
                self.coupled_state_body_constructor_scale = float(coupled_state_body_constructor_scale or 0.0)
                if self.coupled_state_body_constructor_enabled:
                    self.body_state_event_to_hidden = nn.Linear(body_state_event_role_count, d_model)
                self.body_executable_span_head_enabled = bool(
                    body_executable_span_head or executable_span_body_constructor
                )
                self.executable_span_body_constructor_enabled = bool(executable_span_body_constructor)
                self.executable_span_body_constructor_scale = float(executable_span_body_constructor_scale or 0.0)
                if self.body_executable_span_head_enabled:
                    self.body_executable_span_router = nn.Linear(d_model, body_executable_span_role_count)
                if self.executable_span_body_constructor_enabled:
                    self.body_executable_span_to_hidden = nn.Linear(body_executable_span_role_count, d_model)
                if specialist_cfg["mode"] == "sparse_moe":
                    self.specialist_core = _SparseMoE()
                elif specialist_cfg["mode"] == "dense_active_control":
                    self.specialist_core = _Expert(int(specialist_cfg["expert_hidden_dim"]))

            def encode_source(self, src: Any) -> tuple[Any, Any]:
                src_mask = additive_padding_mask(src, mx)
                valid = (src != 0).astype(mx.float32)[:, :, None]
                source = self.source_embedding(src) + self.source_position[:, : src.shape[1], :]
                memory = self.encoder(source, src_mask)
                pooled = mx.sum(memory * valid, axis=1) / mx.maximum(
                    mx.sum(valid, axis=1), mx.array(1.0, dtype=mx.float32)
                )
                return memory, pooled

            def semantic_plan_logits(self, src: Any) -> Any:
                _memory, pooled = self.encode_source(src)
                if self.auxiliary_head_policy == "legacy_materialized_v1":
                    return self.plan_router(pooled)
                return self.project_output(pooled)

            def semantic_slot_logits(self, src: Any) -> Any:
                _memory, pooled = self.encode_source(src)
                if self.auxiliary_head_policy == "legacy_materialized_v1":
                    flat = self.slot_router(pooled)
                    return flat.reshape((flat.shape[0], semantic_slot_role_count, target_vocab_size))
                if not self.semantic_slot_head_enabled:
                    raise RuntimeError("semantic slot head was not materialized for this checkpoint")
                role_hidden = self.slot_role_router(pooled).reshape(
                    (pooled.shape[0], semantic_slot_role_count, d_model)
                )
                output_weight = (
                    self.output.weight
                    if self.output_projection_policy == "independent_output_v1"
                    else self.target_embedding.weight
                )
                logits = role_hidden @ output_weight.T
                output_bias = (
                    self.output.bias
                    if self.output_projection_policy == "independent_output_v1"
                    else self.output_bias
                )
                if output_bias is not None:
                    logits = logits + output_bias
                return logits

            def project_output(self, hidden: Any) -> Any:
                if self.output_projection_policy == "independent_output_v1":
                    return self.output(hidden)
                return hidden @ self.target_embedding.weight.T + self.output_bias

            def decode_hidden(self, src: Any, tgt_in: Any) -> Any:
                src_mask = additive_padding_mask(src, mx)
                tgt_mask = nn.MultiHeadAttention.create_additive_causal_mask(tgt_in.shape[1], mx.float32)
                target = self.target_embedding(tgt_in) + self.target_position[:, : tgt_in.shape[1], :]
                memory, _pooled = self.encode_source(src)
                return self.decoder(target, memory, tgt_mask, src_mask)

            @staticmethod
            def _project_attention_keys_values(attention: Any, values: Any) -> tuple[Any, Any]:
                keys = attention.key_proj(values)
                projected_values = attention.value_proj(values)
                keys = mx.unflatten(keys, -1, (attention.num_heads, -1)).transpose(0, 2, 1, 3)
                projected_values = mx.unflatten(
                    projected_values, -1, (attention.num_heads, -1)
                ).transpose(0, 2, 1, 3)
                return keys, projected_values

            @staticmethod
            def _attention_from_projected(
                attention: Any,
                queries: Any,
                keys: Any,
                values: Any,
                mask: Any | None = None,
            ) -> Any:
                projected_queries = attention.query_proj(queries)
                projected_queries = mx.unflatten(
                    projected_queries, -1, (attention.num_heads, -1)
                ).transpose(0, 2, 1, 3)
                output = mx.fast.scaled_dot_product_attention(
                    projected_queries,
                    keys,
                    values,
                    scale=math.sqrt(1 / projected_queries.shape[-1]),
                    mask=mask,
                )
                output = output.transpose(0, 2, 1, 3).flatten(-2, -1)
                return attention.out_proj(output)

            def prepare_incremental_decode(self, src: Any) -> dict[str, Any]:
                """Encode source and cross-attention K/V once for autoregressive decode."""

                memory_mask = additive_padding_mask(src, mx)
                memory, _pooled = self.encode_source(src)
                cross_keys: list[Any] = []
                cross_values: list[Any] = []
                for layer in self.decoder.layers:
                    keys, values = self._project_attention_keys_values(
                        layer.cross_attention, memory
                    )
                    cross_keys.append(keys)
                    cross_values.append(values)
                return {
                    "memory_mask": memory_mask,
                    "cross_keys": cross_keys,
                    "cross_values": cross_values,
                    "source_length": int(src.shape[1]),
                    "layer_count": len(self.decoder.layers),
                }

            def incremental_decode_hidden(
                self,
                token: Any,
                *,
                position: int,
                decode_context: dict[str, Any],
                self_cache: list[dict[str, Any]] | None = None,
            ) -> tuple[Any, list[dict[str, Any]]]:
                """Decode one token using immutable per-layer self-attention K/V state."""

                if token.shape[1] != 1:
                    raise ValueError("incremental decode requires exactly one target token")
                if int(position) < 0 or int(position) >= int(self.target_position.shape[1]):
                    raise ValueError(f"incremental decode position out of range: {position}")
                layer_count = len(self.decoder.layers)
                if int(decode_context.get("layer_count") or 0) != layer_count:
                    raise ValueError("incremental decode context layer count mismatch")
                if self_cache is not None and len(self_cache) != layer_count:
                    raise ValueError("incremental self-cache layer count mismatch")

                hidden = self.target_embedding(token) + self.target_position[
                    :, int(position) : int(position) + 1, :
                ]
                next_cache: list[dict[str, Any]] = []
                for layer_index, layer in enumerate(self.decoder.layers):
                    if not bool(layer.norm_first):
                        raise ValueError("incremental decode supports norm-first checkpoints only")
                    normalized = layer.ln1(hidden)
                    new_keys, new_values = self._project_attention_keys_values(
                        layer.self_attention, normalized
                    )
                    prior = self_cache[layer_index] if self_cache is not None else {}
                    prior_keys = prior.get("keys") if isinstance(prior, dict) else None
                    prior_values = prior.get("values") if isinstance(prior, dict) else None
                    keys = (
                        mx.concatenate([prior_keys, new_keys], axis=2)
                        if prior_keys is not None
                        else new_keys
                    )
                    values = (
                        mx.concatenate([prior_values, new_values], axis=2)
                        if prior_values is not None
                        else new_values
                    )
                    attended = self._attention_from_projected(
                        layer.self_attention, normalized, keys, values
                    )
                    hidden = hidden + layer.dropout1(attended)

                    normalized = layer.ln2(hidden)
                    attended = self._attention_from_projected(
                        layer.cross_attention,
                        normalized,
                        decode_context["cross_keys"][layer_index],
                        decode_context["cross_values"][layer_index],
                        decode_context["memory_mask"],
                    )
                    hidden = hidden + layer.dropout2(attended)

                    normalized = layer.ln3(hidden)
                    feedforward = layer.linear2(
                        layer.dropout3(layer.activation(layer.linear1(normalized)))
                    )
                    hidden = hidden + feedforward
                    next_cache.append({"keys": keys, "values": values})
                return self.decoder.ln(hidden), next_cache

            def _specialize(self, hidden: Any, valid_mask: Any | None = None) -> tuple[Any, dict[str, Any]]:
                if specialist_cfg["mode"] == "sparse_moe":
                    delta, route = self.specialist_core(hidden, valid_mask)
                    return hidden + delta, route
                if specialist_cfg["mode"] == "dense_active_control":
                    return hidden + self.specialist_core(hidden), {
                        "weighted_aux_loss": mx.array(0.0, dtype=mx.float32)
                    }
                return hidden, {"weighted_aux_loss": mx.array(0.0, dtype=mx.float32)}

            def body_constructor_hidden_from_decoded(
                self, hidden: Any, valid_mask: Any
            ) -> tuple[Any, dict[str, Any]]:
                if self.coupled_state_body_constructor_enabled:
                    event_probs = mx.softmax(self.body_state_event_router(hidden), axis=-1)
                    hidden = hidden + self.coupled_state_body_constructor_scale * mx.tanh(
                        self.body_state_event_to_hidden(event_probs)
                    )
                if self.executable_span_body_constructor_enabled:
                    span_probs = mx.softmax(self.body_executable_span_router(hidden), axis=-1)
                    hidden = hidden + self.executable_span_body_constructor_scale * mx.tanh(
                        self.body_executable_span_to_hidden(span_probs)
                    )
                return self._specialize(hidden, valid_mask)

            def body_constructor_hidden_with_route(self, src: Any, tgt_in: Any) -> tuple[Any, dict[str, Any]]:
                hidden = self.decode_hidden(src, tgt_in)
                return self.body_constructor_hidden_from_decoded(hidden, tgt_in != 0)

            def body_constructor_hidden(self, src: Any, tgt_in: Any) -> Any:
                hidden, _route = self.body_constructor_hidden_with_route(src, tgt_in)
                return hidden

            def forward_with_router_loss(
                self, src: Any, tgt_in: Any, tgt_out: Any | None = None
            ) -> tuple[Any, Any]:
                hidden, route = self.body_constructor_hidden_with_route(src, tgt_in)
                router_loss = route["weighted_aux_loss"]
                if (
                    specialist_cfg["mode"] == "sparse_moe"
                    and token_expert_ids
                    and tgt_out is not None
                    and float(specialist_cfg["router_supervision_loss_weight"]) > 0.0
                ):
                    target_map = mx.array(token_expert_ids, dtype=mx.int32)
                    bounded_targets = mx.minimum(
                        mx.maximum(tgt_out, 0), int(target_map.shape[0]) - 1
                    )
                    expected_experts = target_map[bounded_targets]
                    log_probabilities = nn.log_softmax(route["logits"], axis=-1)
                    selected_log_probabilities = mx.take_along_axis(
                        log_probabilities, expected_experts, axis=-1
                    )
                    valid = (tgt_out != 0).astype(mx.float32)
                    supervision_loss = -mx.sum(
                        mx.mean(selected_log_probabilities, axis=-1) * valid
                    ) / mx.maximum(mx.sum(valid), mx.array(1.0, dtype=mx.float32))
                    router_loss = router_loss + (
                        float(specialist_cfg["router_supervision_loss_weight"]) * supervision_loss
                    )
                return self.project_output(hidden), router_loss

            def specialist_route(self, src: Any, tgt_in: Any) -> dict[str, Any]:
                if specialist_cfg["mode"] != "sparse_moe":
                    return {}
                hidden = self.decode_hidden(src, tgt_in)
                logits, indices, weights = self.specialist_core.route(hidden)
                weighted, balance, z_loss = self.specialist_core.auxiliary_loss(
                    logits, indices, tgt_in != 0
                )
                return {
                    "indices": indices,
                    "weights": weights,
                    "weighted_aux_loss": weighted,
                    "balance_loss": balance,
                    "router_z_loss": z_loss,
                }

            def body_transition_logits(self, src: Any, tgt_in: Any) -> Any:
                hidden = self.body_constructor_hidden(src, tgt_in)
                if self.auxiliary_head_policy == "legacy_materialized_v1":
                    return self.body_transition_router(hidden)
                return self.project_output(hidden)

            def body_action_logits(self, src: Any, tgt_in: Any) -> Any:
                return self.body_action_router(self.body_constructor_hidden(src, tgt_in))

            def body_operand_logits(self, src: Any, tgt_in: Any) -> Any:
                return self.body_operand_router(self.body_constructor_hidden(src, tgt_in))

            def body_state_event_logits(self, src: Any, tgt_in: Any) -> Any:
                return self.body_state_event_router(self.decode_hidden(src, tgt_in))

            def body_executable_span_logits(self, src: Any, tgt_in: Any) -> Any:
                return self.body_executable_span_router(self.decode_hidden(src, tgt_in))

            def logits_bundle_from_decoded(
                self, decoded_hidden: Any, valid_mask: Any
            ) -> dict[str, Any]:
                body_hidden, _route = self.body_constructor_hidden_from_decoded(
                    decoded_hidden, valid_mask
                )
                bundle = {
                    "token": self.project_output(body_hidden),
                    "body_transition": (
                        self.body_transition_router(body_hidden)
                        if self.auxiliary_head_policy == "legacy_materialized_v1"
                        else self.project_output(body_hidden)
                    ),
                    "body_action": self.body_action_router(body_hidden),
                    "body_operand": self.body_operand_router(body_hidden),
                    "body_state_event": self.body_state_event_router(decoded_hidden),
                }
                if self.body_executable_span_head_enabled:
                    bundle["body_executable_span"] = self.body_executable_span_router(
                        decoded_hidden
                    )
                return bundle

            def logits_bundle(self, src: Any, tgt_in: Any) -> dict[str, Any]:
                return self.logits_bundle_from_decoded(
                    self.decode_hidden(src, tgt_in), tgt_in != 0
                )

            def incremental_logits_bundle(
                self,
                token: Any,
                *,
                position: int,
                decode_context: dict[str, Any],
                self_cache: list[dict[str, Any]] | None = None,
            ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
                decoded_hidden, next_cache = self.incremental_decode_hidden(
                    token,
                    position=position,
                    decode_context=decode_context,
                    self_cache=self_cache,
                )
                return self.logits_bundle_from_decoded(
                    decoded_hidden, token != 0
                ), next_cache

            def __call__(self, src: Any, tgt_in: Any) -> Any:
                logits, _router_loss = self.forward_with_router_loss(src, tgt_in)
                return logits

        self.model = _Model()


def additive_padding_mask(tokens: Any, mx: Any) -> Any:
    valid = tokens != 0
    return mx.where(valid[:, None, None, :], 0.0, -1e9).astype(mx.float32)
