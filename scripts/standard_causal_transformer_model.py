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
    attention_residual_mode: str = "none"
    attention_residual_block_size: int = 0
    feed_forward_activation: str = "swiglu"
    situ_glu_gate_beta: float = 4.0
    situ_glu_up_beta: float = 25.0
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
    mtp_future_offsets: tuple[int, ...] = ()
    mtp_head_mode: str = "shared_low_rank"
    mtp_low_rank: int = 0
    mtp_hidden_dim: int = 0
    mtp_register_count: int = 0
    mtp_loss_weights: tuple[float, ...] = ()
    mtp_loss_scale: float = 0.0
    mtp_maximum_head_parameter_overhead_ratio: float = 0.25
    kerc_task_token_ids: tuple[int, ...] = ()
    kerc_stage_adapter_dim: int = 0
    kerc_decoder_stage_adapter_dim: int = 0
    kerc_reasoner_output_delta_dim: int = 0
    kerc_residual_choice_count: int = 0
    kerc_residual_bottleneck_dim: int = 0
    kerc_residual_unit_kind_count: int = 0
    kerc_residual_unit_feature_dim: int = 0
    kerc_residual_unit_byte_vocab_size: int = 0
    kerc_verifier_dim: int = 0
    kerc_verifier_output_dim: int = 4
    kerc_decision_bottleneck_dim: int = 0
    kerc_decision_output_dim: int = 0
    kerc_surface_token_start: int = 0
    kerc_surface_token_end: int = 0
    kerc_kernel_token_start: int = 0
    kerc_kernel_token_end: int = 0
    kerc_pointer_token_start: int = 0
    kerc_pointer_token_end: int = 0
    kerc_end_token_id: int = 0
    kerc_stage_routing_ablation: str = "none"
    kerc_residual_ablation: str = "none"
    kerc_interaction_residual_ablation: str = "none"
    kerc_verifier_ablation: str = "none"
    kerc_decision_ablation: str = "none"

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
        if self.attention_residual_mode not in {"none", "block"}:
            raise ValueError("attention residual mode must be none or block")
        if self.attention_residual_mode == "none":
            if self.attention_residual_block_size != 0:
                raise ValueError(
                    "disabled attention residuals require zero block size"
                )
        elif (
            self.attention_residual_block_size <= 0
            or self.attention_residual_block_size > self.num_layers
        ):
            raise ValueError(
                "block attention residuals require a valid block size"
            )
        if self.feed_forward_activation not in {"swiglu", "situ_glu"}:
            raise ValueError(
                "feed-forward activation must be swiglu or situ_glu"
            )
        if (
            not math.isfinite(self.situ_glu_gate_beta)
            or not math.isfinite(self.situ_glu_up_beta)
            or self.situ_glu_gate_beta <= 0.0
            or self.situ_glu_up_beta <= 0.0
        ):
            raise ValueError("SiTU-GLU beta values must be finite and positive")
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
        if (
            self.attention_residual_mode != "none"
            and self.state_memory_mode != "none"
        ):
            raise ValueError(
                "attention residuals are not qualified with recurrent state memory"
            )
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
        mtp_offsets = tuple(int(value) for value in self.mtp_future_offsets)
        mtp_weights = tuple(float(value) for value in self.mtp_loss_weights)
        if self.mtp_head_mode not in {
            "shared_low_rank",
            "independent_mlp",
            "register_conditioned",
        }:
            raise ValueError("unsupported MTP head mode")
        mtp_enabled = bool(
            mtp_offsets
            or self.mtp_low_rank
            or self.mtp_hidden_dim
            or self.mtp_register_count
            or mtp_weights
        )
        if mtp_enabled:
            if not mtp_offsets or len(mtp_offsets) != len(mtp_weights):
                raise ValueError("MTP offsets and loss weights must be nonempty and aligned")
            if tuple(sorted(set(mtp_offsets))) != mtp_offsets or mtp_offsets[0] < 2:
                raise ValueError("MTP future offsets must be unique, increasing, and at least two")
            if any(weight < 0.0 for weight in mtp_weights) or not any(mtp_weights):
                raise ValueError("MTP loss weights must be nonnegative with positive mass")
            if self.mtp_head_mode == "shared_low_rank":
                if self.mtp_low_rank <= 0 or self.mtp_hidden_dim or self.mtp_register_count:
                    raise ValueError(
                        "shared-low-rank MTP requires only a positive low-rank projection"
                    )
                head_parameters = (
                    self.d_model * self.mtp_low_rank
                    + len(mtp_offsets) * self.mtp_low_rank * self.vocab_size
                )
            elif self.mtp_head_mode == "independent_mlp":
                if self.mtp_hidden_dim <= 0 or self.mtp_low_rank or self.mtp_register_count:
                    raise ValueError(
                        "independent MTP requires only a positive hidden dimension"
                    )
                head_parameters = len(mtp_offsets) * (
                    self.d_model * self.mtp_hidden_dim
                    + self.mtp_hidden_dim * self.vocab_size
                )
            else:
                if (
                    self.mtp_hidden_dim <= 0
                    or self.mtp_low_rank
                    or self.mtp_register_count != len(mtp_offsets)
                ):
                    raise ValueError(
                        "register-conditioned MTP requires one register per offset and a positive hidden dimension"
                    )
                head_parameters = (
                    self.mtp_register_count * self.d_model
                    + self.d_model * self.mtp_hidden_dim
                    + len(mtp_offsets) * self.mtp_hidden_dim * self.vocab_size
                )
            base_head_parameters = self.d_model * self.vocab_size
            if head_parameters / base_head_parameters > float(
                self.mtp_maximum_head_parameter_overhead_ratio
            ):
                raise ValueError("MTP optional heads exceed the parameter-overhead ceiling")
        elif self.mtp_loss_scale:
            raise ValueError("MTP loss scale requires enabled MTP heads")
        if self.mtp_loss_scale < 0.0:
            raise ValueError("MTP loss scale cannot be negative")
        kerc_tokens = tuple(int(value) for value in self.kerc_task_token_ids)
        kerc_enabled = bool(kerc_tokens)
        for name, value in (
            ("stage routing", self.kerc_stage_routing_ablation),
            ("residual", self.kerc_residual_ablation),
            ("interaction residual", self.kerc_interaction_residual_ablation),
            ("verifier", self.kerc_verifier_ablation),
            ("decision", self.kerc_decision_ablation),
        ):
            if value not in {"none", "zero"}:
                raise ValueError(f"KERC {name} ablation must be none or zero")
            if not kerc_enabled and value != "none":
                raise ValueError(f"KERC {name} ablation requires KERC")
        if kerc_enabled:
            if self.attention_policy != "encoder_decoder":
                raise ValueError("KERC requires the encoder-decoder architecture")
            if self.source_copy_mode != "pointer_generator":
                raise ValueError("KERC requires the copy-aware pointer generator")
            if len(kerc_tokens) != 4 or len(set(kerc_tokens)) != 4:
                raise ValueError("KERC requires four distinct trusted task tokens")
            if any(token_id < 0 or token_id >= self.vocab_size for token_id in kerc_tokens):
                raise ValueError("KERC task tokens must be inside the model vocabulary")
            if self.kerc_stage_adapter_dim <= 0:
                raise ValueError("KERC requires positive stage adapters")
            if self.kerc_decoder_stage_adapter_dim < 0:
                raise ValueError("KERC decoder stage adapter dimension cannot be negative")
            if self.kerc_reasoner_output_delta_dim < 0:
                raise ValueError("KERC reasoner output delta dimension cannot be negative")
            if self.kerc_residual_choice_count < 4:
                raise ValueError("KERC requires at least four residual fidelity choices")
            if self.kerc_residual_bottleneck_dim <= 0:
                raise ValueError("KERC requires a learned residual bottleneck")
            unit_allocator_enabled = any(
                (
                    self.kerc_residual_unit_kind_count,
                    self.kerc_residual_unit_feature_dim,
                    self.kerc_residual_unit_byte_vocab_size,
                )
            )
            if unit_allocator_enabled and (
                self.kerc_residual_unit_kind_count < 5
                or self.kerc_residual_unit_feature_dim <= 0
                or self.kerc_residual_unit_byte_vocab_size != 257
            ):
                raise ValueError(
                    "KERC per-unit allocation requires five kinds, positive candidate features, and 257 byte symbols"
                )
            if self.kerc_verifier_dim <= 0:
                raise ValueError("KERC requires an independent verifier dimension")
            if self.kerc_verifier_output_dim <= 0:
                raise ValueError("KERC requires positive verifier output dimensions")
            if self.kerc_decision_bottleneck_dim <= 0:
                raise ValueError("KERC requires a learned decision bottleneck")
            if self.kerc_decision_output_dim < 3:
                raise ValueError("KERC requires answer, clarification, and abstention decisions")
            ranges = (
                (self.kerc_surface_token_start, self.kerc_surface_token_end, "surface"),
                (self.kerc_kernel_token_start, self.kerc_kernel_token_end, "kernel"),
                (self.kerc_pointer_token_start, self.kerc_pointer_token_end, "pointer"),
            )
            for start, end, name in ranges:
                if not 0 <= start < end <= self.vocab_size:
                    raise ValueError(f"KERC {name} token range is invalid")
            ordered = sorted((start, end, name) for start, end, name in ranges)
            if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
                raise ValueError("KERC token ranges must be disjoint")
            if not 0 <= self.kerc_end_token_id < self.vocab_size:
                raise ValueError("KERC end token must be inside the model vocabulary")
        elif any(
            value
            for value in (
                self.kerc_stage_adapter_dim,
                self.kerc_decoder_stage_adapter_dim,
                self.kerc_reasoner_output_delta_dim,
                self.kerc_residual_choice_count,
                self.kerc_residual_bottleneck_dim,
                self.kerc_residual_unit_kind_count,
                self.kerc_residual_unit_feature_dim,
                self.kerc_residual_unit_byte_vocab_size,
                self.kerc_verifier_dim,
                self.kerc_decision_bottleneck_dim,
                self.kerc_decision_output_dim,
                self.kerc_surface_token_start,
                self.kerc_surface_token_end,
                self.kerc_kernel_token_start,
                self.kerc_kernel_token_end,
                self.kerc_pointer_token_start,
                self.kerc_pointer_token_end,
                self.kerc_end_token_id,
            )
        ):
            raise ValueError("KERC dimensions require trusted task tokens")


def analytical_parameter_breakdown(
    config: CausalTransformerConfig,
) -> dict[str, int]:
    """Count the declared model graph without importing or allocating MLX tensors.

    This is the canonical planning-time accounting path.  Keep every term tied to
    an owning module in ``build_model`` so dry-run planning cannot initialize the
    Metal runtime merely to inspect architecture size.
    """

    config.validate()
    d_model = int(config.d_model)
    head_dim = d_model // int(config.num_heads)
    kv_width = int(config.num_kv_heads) * head_dim
    attention = 2 * d_model * d_model + 2 * d_model * kv_width
    feed_forward = 3 * d_model * int(config.ff_dim)
    block_norms = 2 * d_model
    base_block = attention + feed_forward + block_norms

    def adapter(dimension: int) -> int:
        return d_model + 2 * d_model * int(dimension) if dimension else 0

    decoder_layer = base_block
    decoder_layer += adapter(int(config.expert_adapter_dim))
    decoder_layer += adapter(int(config.source_expert_adapter_dim))
    if config.state_memory_mode != "none":
        decoder_layer += (
            int(config.state_memory_slots) * d_model
            + 4 * d_model * d_model
            + d_model
        )

    result = {
        "token_embedding": int(config.vocab_size) * d_model,
        "decoder_layers": int(config.num_layers) * decoder_layer,
        "final_norm": d_model,
    }
    if config.attention_residual_mode == "block":
        result["attention_residual_queries"] = (
            int(config.num_layers) + 1
        ) * d_model

    if config.attention_policy == "encoder_decoder":
        source_block = base_block + adapter(int(config.source_expert_adapter_dim))
        result["source_encoder_layers"] = int(config.source_encoder_layers) * source_block
        result["source_final_norm"] = d_model
        result["decoder_cross_attention"] = int(config.num_layers) * (
            d_model + attention
        )
        if config.source_copy_mode == "pointer_generator":
            result["pointer_generator"] = 2 * d_model * d_model + d_model + 1

    if (
        config.semantic_plan_feature_count > 0
        and config.semantic_plan_conditioning_mode == "slot_attention"
    ):
        result["semantic_plan_cross_attention"] = int(config.num_layers) * (
            d_model + attention
        )
    if config.semantic_plan_feature_count > 0:
        feature_count = int(config.semantic_plan_feature_count)
        plan_dim = int(config.semantic_plan_bottleneck_dim) or d_model
        result["semantic_plan"] = (
            (d_model * plan_dim if plan_dim != d_model else 0)
            + plan_dim * feature_count
            + feature_count
            + feature_count * plan_dim
            + plan_dim * d_model
        )

    future_count = len(tuple(config.mtp_future_offsets))
    if future_count:
        if config.mtp_head_mode == "shared_low_rank":
            result["mtp_heads"] = (
                d_model * int(config.mtp_low_rank)
                + future_count * int(config.mtp_low_rank) * int(config.vocab_size)
            )
        elif config.mtp_head_mode == "independent_mlp":
            result["mtp_heads"] = future_count * (
                d_model * int(config.mtp_hidden_dim)
                + int(config.mtp_hidden_dim) * int(config.vocab_size)
            )
        else:
            result["mtp_heads"] = (
                int(config.mtp_register_count) * d_model
                + d_model * int(config.mtp_hidden_dim)
                + future_count * int(config.mtp_hidden_dim) * int(config.vocab_size)
            )

    if config.kerc_task_token_ids:
        stage_count = 4
        choice_count = int(config.kerc_residual_choice_count)
        bottleneck = int(config.kerc_residual_bottleneck_dim)
        verifier_dim = int(config.kerc_verifier_dim)
        verifier_outputs = int(config.kerc_verifier_output_dim)
        decision_dim = int(config.kerc_decision_bottleneck_dim)
        decision_outputs = int(config.kerc_decision_output_dim)
        kerc = (
            stage_count * d_model
            + stage_count * adapter(int(config.kerc_stage_adapter_dim))
            + int(config.num_layers)
            * stage_count
            * adapter(int(config.kerc_decoder_stage_adapter_dim))
            + int(config.kerc_reasoner_output_delta_dim)
            * (d_model + int(config.vocab_size))
            + 2 * d_model * int(config.vocab_size)
            + d_model * bottleneck
            + bottleneck * stage_count * choice_count
            + stage_count * choice_count
            + stage_count * choice_count * d_model
            + int(config.vocab_size) * verifier_dim
            + 2 * verifier_dim * verifier_dim
            + 4 * verifier_dim * verifier_outputs
            + verifier_outputs
            + d_model * decision_dim
            + decision_dim * decision_outputs
            + decision_outputs
        )
        if int(config.kerc_residual_unit_feature_dim) > 0:
            unit_kinds = int(config.kerc_residual_unit_kind_count)
            unit_features = int(config.kerc_residual_unit_feature_dim)
            byte_vocab = int(config.kerc_residual_unit_byte_vocab_size)
            kerc += (
                byte_vocab * d_model
                + unit_kinds * d_model
                + 2 * d_model * bottleneck
                + 4 * bottleneck * bottleneck
                + unit_features * bottleneck
                + choice_count * bottleneck
                + 2 * (bottleneck + 1)
                + choice_count * d_model
            )
        result["kerc_heads"] = kerc
    return result


def analytical_parameter_count(config: CausalTransformerConfig) -> int:
    """Return the exact parameter count for the declared model graph."""

    return sum(analytical_parameter_breakdown(config).values())


def analytical_trainable_parameter_count(
    config: CausalTransformerConfig, scope: str
) -> int:
    """Count parameters exposed by ``freeze_to_language_expert(scope)``."""

    config.validate()
    d_model = int(config.d_model)

    def adapter(dimension: int) -> int:
        return d_model + 2 * d_model * int(dimension) if dimension else 0

    if int(config.expert_adapter_dim) <= 0:
        raise ValueError("model has no expert adapter to train")
    if scope not in {
        "adapter_only",
        "source_conditioned_delta",
        "low_rank_source_adapters",
    }:
        raise ValueError(f"unsupported language expert scope: {scope}")
    total = int(config.num_layers) * adapter(int(config.expert_adapter_dim))
    if scope == "adapter_only":
        return total
    head_dim = d_model // int(config.num_heads)
    kv_width = int(config.num_kv_heads) * head_dim
    attention = 2 * d_model * d_model + 2 * d_model * kv_width
    feed_forward = 3 * d_model * int(config.ff_dim)
    base_block = attention + feed_forward + 2 * d_model
    if scope == "source_conditioned_delta":
        if (
            config.attention_policy != "encoder_decoder"
            or config.source_copy_mode != "pointer_generator"
        ):
            raise ValueError(
                "source-conditioned expert scope requires encoder-decoder pointer mode"
            )
        total += int(config.num_layers) * (d_model + attention)
        total += int(config.source_encoder_layers) * (
            base_block + adapter(int(config.source_expert_adapter_dim))
        )
        total += d_model + 2 * d_model * d_model + d_model + 1
        return total
    if (
        config.attention_policy != "encoder_decoder"
        or int(config.source_expert_adapter_dim) <= 0
    ):
        raise ValueError(
            "low-rank source expert scope requires source expert adapters"
        )
    total += int(config.num_layers) * adapter(
        int(config.source_expert_adapter_dim)
    )
    total += int(config.source_encoder_layers) * adapter(
        int(config.source_expert_adapter_dim)
    )
    if config.source_copy_mode == "pointer_generator":
        total += d_model + 1
    return total


def build_model(
    config: CausalTransformerConfig,
    *,
    mx: Any,
    nn: Any,
    state_role_lookup: Any | None = None,
    source_to_target_lookup: Any | None = None,
    rope_kernel: str = "manual_reference",
    gradient_checkpointing: bool = False,
    attention_query_chunk_size: int = 0,
    attention_key_chunk_size: int = 0,
    compact_encoder_decoder_partitions: bool = False,
    compact_output_projection: bool = False,
    compact_partition_width_quantum: int = 0,
    parameter_initialization_dtype: str = "float32",
    exact_checkpoint_placeholder_initialization: bool = False,
    self_attention_projection: str = "separate",
) -> Any:
    """Build a pre-norm RoPE/GQA/gated-FFN causal LM with tied embeddings."""

    config.validate()
    if parameter_initialization_dtype not in {"float32", "bfloat16"}:
        raise ValueError(
            "parameter initialization dtype must be float32 or bfloat16"
        )
    if (
        exact_checkpoint_placeholder_initialization
        and parameter_initialization_dtype != "bfloat16"
    ):
        raise ValueError(
            "exact-checkpoint placeholder initialization requires bfloat16"
        )
    if compact_partition_width_quantum < 0 or (
        compact_partition_width_quantum
        and not compact_encoder_decoder_partitions
    ):
        raise ValueError(
            "compact partition width quantum requires compact encoder-decoder partitions"
        )
    if rope_kernel not in {"manual_reference", "mlx_fast"}:
        raise ValueError(f"unsupported RoPE kernel: {rope_kernel}")
    if self_attention_projection not in {"separate", "fused_qkv"}:
        raise ValueError(
            "self-attention projection must be separate or fused_qkv"
        )
    if attention_query_chunk_size < 0:
        raise ValueError("attention query chunk size cannot be negative")
    if attention_key_chunk_size < 0:
        raise ValueError("attention key chunk size cannot be negative")

    if parameter_initialization_dtype == "bfloat16":
        raw_nn = nn

        class InitializationDtypeNN:
            """Materialize each parameterized leaf directly into its retained dtype."""

            Module = raw_nn.Module

            def __getattr__(self, name: str) -> Any:
                return getattr(raw_nn, name)

            @staticmethod
            def _compact(module: Any) -> Any:
                if exact_checkpoint_placeholder_initialization:
                    for name, value in module.parameters().items():
                        setattr(
                            module,
                            name,
                            mx.zeros(value.shape, dtype=mx.bfloat16),
                        )
                else:
                    module.set_dtype(mx.bfloat16)
                # Materialize large leaves immediately. Smaller leaves are
                # evaluated once at their enclosing block boundary below,
                # avoiding hundreds of Metal synchronizations while still
                # preventing a whole FP32 model of lazy cast sources.
                if (
                    not exact_checkpoint_placeholder_initialization
                    and sum(
                    int(value.size)
                    for value in module.parameters().values()
                    )
                    >= 1_000_000
                ):
                    mx.eval(module.parameters())
                return module

            def Linear(self, *args: Any, **kwargs: Any) -> Any:
                return self._compact(raw_nn.Linear(*args, **kwargs))

            def Embedding(self, *args: Any, **kwargs: Any) -> Any:
                return self._compact(raw_nn.Embedding(*args, **kwargs))

            def RMSNorm(self, *args: Any, **kwargs: Any) -> Any:
                return self._compact(raw_nn.RMSNorm(*args, **kwargs))

        nn = InitializationDtypeNN()

    def materialize_initialization(module: Any) -> None:
        if (
            parameter_initialization_dtype == "bfloat16"
            and not exact_checkpoint_placeholder_initialization
        ):
            mx.eval(module.parameters())

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
    kerc_enabled = bool(config.kerc_task_token_ids)
    kerc_decoder_stage_adapter_enabled = bool(
        kerc_enabled and config.kerc_decoder_stage_adapter_dim > 0
    )
    kerc_reasoner_output_delta_enabled = bool(
        kerc_enabled and config.kerc_reasoner_output_delta_dim > 0
    )
    kerc_unit_allocator_enabled = bool(
        kerc_enabled and config.kerc_residual_unit_feature_dim > 0
    )
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
    mtp_enabled = bool(config.mtp_future_offsets)
    mtp_head_mode = str(config.mtp_head_mode)

    def uniform_kerc_stage_index(stage_weights: Any | None) -> int | None:
        """Return an exact shared one-hot KERC stage, otherwise fail to fallback.

        Stage-only KERC training uses one trusted control token per row.  The
        generic router used to evaluate every stage adapter and projection and
        multiply three results by zero.  Detecting a uniform one-hot batch lets
        those mathematically inactive branches remain unmaterialized while
        preserving the mixed-stage and malformed-control fallback.
        """

        if stage_weights is None or int(stage_weights.shape[0]) == 0:
            return None
        stage_indices = mx.argmax(stage_weights, axis=-1)
        first = int(stage_indices[0].item())
        exact_one_hot = mx.sum(stage_weights, axis=-1) == 1.0
        shared_stage = stage_indices == first
        if bool(mx.all(exact_one_hot & shared_stage).item()):
            return first
        return None

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
        if rope_kernel == "mlx_fast":
            return mx.fast.rope(
                value,
                head_dim,
                traditional=False,
                base=float(config.rope_base),
                scale=1.0,
                offset=offset,
            )
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

    def bounded_scaled_dot_product_attention(
        query: Any,
        key: Any,
        value: Any,
        *,
        scale: float,
        mask: Any | None,
    ) -> Any:
        """Evaluate attention with optional query and online-softmax KV chunks."""

        query_length = int(query.shape[2])
        key_length = int(key.shape[2])
        query_position_offset = max(0, key_length - query_length)
        query_chunk_size = int(attention_query_chunk_size)
        key_chunk_size = int(attention_key_chunk_size)
        if query_chunk_size <= 0:
            query_chunk_size = query_length

        def native_sdpa_mask(value: Any | None) -> Any | None:
            if value is None or isinstance(value, str):
                return value
            if value.dtype == mx.bool_:
                return value
            return value.astype(query.dtype)

        if key_chunk_size <= 0 or key_length <= key_chunk_size:
            if query_length <= query_chunk_size:
                return mx.fast.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    scale=scale,
                    mask=native_sdpa_mask(mask),
                )
            outputs = []
            key_positions = mx.arange(key_length, dtype=mx.int32)
            for start in range(0, query_length, query_chunk_size):
                stop = min(query_length, start + query_chunk_size)
                chunk_mask = mask
                if isinstance(mask, str) and mask == "causal":
                    query_positions = mx.arange(
                        query_position_offset + start,
                        query_position_offset + stop,
                        dtype=mx.int32,
                    )
                    chunk_mask = mx.where(
                        key_positions[None, :] <= query_positions[:, None],
                        mx.array(0.0, dtype=mx.float32),
                        mx.array(-1e9, dtype=mx.float32),
                    )
                elif mask is not None and not isinstance(mask, str):
                    mask_query_width = (
                        int(mask.shape[-2]) if len(mask.shape) >= 2 else 1
                    )
                    if mask_query_width == query_length:
                        chunk_mask = mask[..., start:stop, :]
                    elif mask_query_width != 1:
                        raise ValueError(
                            "attention mask query width is incompatible with chunking"
                        )
                outputs.append(
                    mx.fast.scaled_dot_product_attention(
                        query[:, :, start:stop, :],
                        key,
                        value,
                        scale=scale,
                        mask=native_sdpa_mask(chunk_mask),
                    )
                )
            return mx.concatenate(outputs, axis=2)

        if gradient_checkpointing:
            def native_chunk_mask(start: int, stop: int) -> Any:
                chunk_mask = mask
                if isinstance(mask, str):
                    if mask != "causal":
                        raise ValueError("unsupported string attention mask")
                    query_positions = mx.arange(
                        query_position_offset + start,
                        query_position_offset + stop,
                        dtype=mx.int32,
                    )
                    key_positions = mx.arange(key_length, dtype=mx.int32)
                    chunk_mask = mx.where(
                        key_positions[None, :] <= query_positions[:, None],
                        mx.array(0.0, dtype=mx.float32),
                        mx.array(-1e9, dtype=mx.float32),
                    )
                elif mask is not None:
                    mask_query_width = (
                        int(mask.shape[-2]) if len(mask.shape) >= 2 else 1
                    )
                    if mask_query_width == query_length:
                        chunk_mask = mask[..., start:stop, :]
                    elif mask_query_width != 1:
                        raise ValueError(
                            "attention mask query width is incompatible with chunking"
                        )
                return native_sdpa_mask(chunk_mask)

            @mx.custom_function
            def checkpointed_query_stream(
                stream_query: Any,
                stream_key: Any,
                stream_value: Any,
            ) -> Any:
                chunks = []
                for start in range(0, query_length, query_chunk_size):
                    stop = min(query_length, start + query_chunk_size)
                    chunk = mx.fast.scaled_dot_product_attention(
                        stream_query[:, :, start:stop, :],
                        stream_key,
                        stream_value,
                        scale=scale,
                        mask=native_chunk_mask(start, stop),
                    )
                    # MLX is lazy: without this boundary every query tile's
                    # score workspace can be scheduled together, defeating the
                    # bounded-memory contract. The custom VJP below restores
                    # exact end-to-end gradients through these materialized
                    # forward chunks.
                    mx.eval(chunk)
                    chunks.append(chunk)
                return mx.concatenate(chunks, axis=2)

            @checkpointed_query_stream.vjp
            def checkpointed_query_stream_vjp(
                primals: tuple[Any, Any, Any],
                cotangent: Any,
                _output: Any,
            ) -> tuple[Any, Any, Any]:
                stream_query, stream_key, stream_value = primals
                batch = int(stream_query.shape[0])
                query_heads = int(stream_query.shape[1])
                kv_heads = int(stream_key.shape[1])
                if query_heads % kv_heads:
                    raise ValueError(
                        "query heads must divide evenly across KV heads"
                    )
                query_groups = query_heads // kv_heads

                def grouped_chunk_mask(start: int, stop: int) -> Any | None:
                    selected = native_chunk_mask(start, stop)
                    if selected is None:
                        return None
                    if len(selected.shape) == 2:
                        selected = selected[None, None, None, :, :]
                    elif len(selected.shape) == 3:
                        selected = selected[:, None, None, :, :]
                    elif len(selected.shape) == 4:
                        mask_heads = int(selected.shape[1])
                        if mask_heads == query_heads:
                            selected = selected.reshape(
                                int(selected.shape[0]),
                                kv_heads,
                                query_groups,
                                int(selected.shape[2]),
                                int(selected.shape[3]),
                            )
                        elif mask_heads in {1, kv_heads}:
                            selected = selected[:, :, None, :, :]
                        else:
                            raise ValueError(
                                "attention mask head width is incompatible with grouped-query attention"
                            )
                    elif len(selected.shape) != 5:
                        raise ValueError("unsupported attention mask rank")
                    return selected.astype(mx.float32)

                query_gradients = []
                key_gradient = mx.zeros_like(stream_key)
                value_gradient = mx.zeros_like(stream_value)
                for start in range(0, query_length, query_chunk_size):
                    stop = min(query_length, start + query_chunk_size)
                    query_width = stop - start
                    grouped_query = stream_query[
                        :, :, start:stop, :
                    ].reshape(
                        batch,
                        kv_heads,
                        query_groups,
                        query_width,
                        head_dim,
                    )
                    grouped_cotangent = cotangent[
                        :, :, start:stop, :
                    ].reshape(
                        batch,
                        kv_heads,
                        query_groups,
                        query_width,
                        head_dim,
                    )
                    grouped_key = stream_key[:, :, None, :, :]
                    grouped_value = stream_value[:, :, None, :, :]
                    scores = mx.matmul(
                        grouped_query.astype(mx.float32),
                        mx.swapaxes(grouped_key.astype(mx.float32), -1, -2),
                    ) * float(scale)
                    chunk_mask = grouped_chunk_mask(start, stop)
                    if chunk_mask is not None:
                        scores = scores + chunk_mask
                    probabilities = mx.softmax(scores, axis=-1)
                    cotangent_fp32 = grouped_cotangent.astype(mx.float32)
                    value_fp32 = grouped_value.astype(mx.float32)
                    probability_gradient = mx.matmul(
                        cotangent_fp32,
                        mx.swapaxes(value_fp32, -1, -2),
                    )
                    score_gradient = probabilities * (
                        probability_gradient
                        - mx.sum(
                            probability_gradient * probabilities,
                            axis=-1,
                            keepdims=True,
                        )
                    )
                    chunk_query_gradient = mx.matmul(
                        score_gradient,
                        grouped_key.astype(mx.float32),
                    ) * float(scale)
                    chunk_key_gradient = mx.sum(
                        mx.matmul(
                            mx.swapaxes(score_gradient, -1, -2),
                            grouped_query.astype(mx.float32),
                        ),
                        axis=2,
                    ) * float(scale)
                    chunk_value_gradient = mx.sum(
                        mx.matmul(
                            mx.swapaxes(probabilities, -1, -2),
                            cotangent_fp32,
                        ),
                        axis=2,
                    )
                    chunk_query_gradient = chunk_query_gradient.reshape(
                        batch, query_heads, query_width, head_dim
                    ).astype(stream_query.dtype)
                    chunk_key_gradient = chunk_key_gradient.astype(
                        stream_key.dtype
                    )
                    chunk_value_gradient = chunk_value_gradient.astype(
                        stream_value.dtype
                    )
                    mx.eval(
                        chunk_query_gradient,
                        chunk_key_gradient,
                        chunk_value_gradient,
                    )
                    query_gradients.append(chunk_query_gradient)
                    key_gradient = key_gradient + chunk_key_gradient
                    value_gradient = value_gradient + chunk_value_gradient
                    # Fold the shared K/V contribution now so the score,
                    # probability, and derivative workspaces for this tile can
                    # be released before the next query tile is constructed.
                    mx.eval(key_gradient, value_gradient)
                return (
                    mx.concatenate(query_gradients, axis=2),
                    key_gradient,
                    value_gradient,
                )

            return checkpointed_query_stream(query, key, value)

        batch = int(query.shape[0])
        query_heads = int(query.shape[1])
        kv_heads = int(key.shape[1])
        if query_heads % kv_heads:
            raise ValueError("query heads must divide evenly across KV heads")
        query_groups = query_heads // kv_heads

        def sliced_mask(
            query_start: int,
            query_stop: int,
            key_start: int,
            key_stop: int,
        ) -> Any:
            if isinstance(mask, str):
                if mask != "causal":
                    raise ValueError("unsupported string attention mask")
                query_positions = mx.arange(
                    query_position_offset + query_start,
                    query_position_offset + query_stop,
                    dtype=mx.int32,
                )
                key_positions = mx.arange(key_start, key_stop, dtype=mx.int32)
                selected = mx.where(
                    key_positions[None, :] <= query_positions[:, None],
                    mx.array(0.0, dtype=mx.float32),
                    mx.array(-1e9, dtype=mx.float32),
                )
            elif mask is None:
                return mx.zeros((1, 1, 1, 1, 1), dtype=mx.float32)
            else:
                selected = mask
                mask_query_width = (
                    int(selected.shape[-2]) if len(selected.shape) >= 2 else 1
                )
                mask_key_width = int(selected.shape[-1])
                if mask_query_width == query_length:
                    selected = selected[..., query_start:query_stop, :]
                elif mask_query_width != 1:
                    raise ValueError(
                        "attention mask query width is incompatible with chunking"
                    )
                if mask_key_width == key_length:
                    selected = selected[..., key_start:key_stop]
                elif mask_key_width != 1:
                    raise ValueError(
                        "attention mask key width is incompatible with chunking"
                    )
            if len(selected.shape) == 2:
                selected = selected[None, None, None, :, :]
            elif len(selected.shape) == 3:
                selected = selected[:, None, None, :, :]
            elif len(selected.shape) == 4:
                mask_heads = int(selected.shape[1])
                if mask_heads == query_heads:
                    selected = selected.reshape(
                        int(selected.shape[0]),
                        kv_heads,
                        query_groups,
                        int(selected.shape[2]),
                        int(selected.shape[3]),
                    )
                elif mask_heads in {1, kv_heads}:
                    selected = selected[:, :, None, :, :]
                else:
                    raise ValueError(
                        "attention mask head width is incompatible with grouped-query attention"
                    )
            elif len(selected.shape) != 5:
                raise ValueError("unsupported attention mask rank")
            return selected.astype(mx.float32)

        has_mask = mask is not None

        def online_block(
            running_maximum: Any,
            running_denominator: Any,
            running_numerator: Any,
            grouped_query: Any,
            block_key: Any,
            block_value: Any,
            block_mask: Any,
        ) -> tuple[Any, Any, Any]:
            scores = mx.matmul(
                grouped_query,
                mx.swapaxes(block_key, -1, -2),
            ) * float(scale)
            if has_mask:
                scores = scores + block_mask
            block_maximum = mx.max(scores, axis=-1, keepdims=True)
            next_maximum = mx.maximum(running_maximum, block_maximum)
            finite_next = mx.isfinite(next_maximum)
            safe_next_maximum = mx.where(
                finite_next, next_maximum, mx.zeros_like(next_maximum)
            )
            prior_scale = mx.where(
                mx.isfinite(running_maximum) & finite_next,
                mx.exp(running_maximum - safe_next_maximum),
                mx.zeros_like(next_maximum),
            )
            block_weights = mx.where(
                finite_next,
                mx.exp(scores - safe_next_maximum),
                mx.zeros_like(scores),
            )
            next_denominator = (
                running_denominator * prior_scale
                + mx.sum(block_weights, axis=-1, keepdims=True)
            )
            next_numerator = (
                running_numerator * prior_scale
                + mx.matmul(block_weights, block_value)
            )
            return next_maximum, next_denominator, next_numerator

        # The enclosing transformer layer is already rematerialized when
        # gradient checkpointing is enabled. Checkpointing every online-softmax
        # tile again creates a query-chunk x key-chunk graph of nested replay
        # boundaries and can exhaust Metal's graph resource table on long rows.
        # Keep one layer-level replay boundary and let the exact online
        # recurrence live inside it.
        update_block = online_block
        outputs = []
        for query_start in range(0, query_length, query_chunk_size):
            query_stop = min(query_length, query_start + query_chunk_size)
            query_width = query_stop - query_start
            grouped_query = query[:, :, query_start:query_stop, :].reshape(
                batch,
                kv_heads,
                query_groups,
                query_width,
                head_dim,
            ).astype(mx.float32)
            running_maximum = mx.full(
                (batch, kv_heads, query_groups, query_width, 1),
                -float("inf"),
                dtype=mx.float32,
            )
            running_denominator = mx.zeros_like(running_maximum)
            running_numerator = mx.zeros(
                (batch, kv_heads, query_groups, query_width, head_dim),
                dtype=mx.float32,
            )
            for key_start in range(0, key_length, key_chunk_size):
                key_stop = min(key_length, key_start + key_chunk_size)
                block_key = key[:, :, None, key_start:key_stop, :].astype(
                    mx.float32
                )
                block_value = value[:, :, None, key_start:key_stop, :].astype(
                    mx.float32
                )
                running_maximum, running_denominator, running_numerator = (
                    update_block(
                        running_maximum,
                        running_denominator,
                        running_numerator,
                        grouped_query,
                        block_key,
                        block_value,
                        sliced_mask(
                            query_start, query_stop, key_start, key_stop
                        ),
                    )
                )
            normalized = running_numerator / mx.maximum(
                running_denominator,
                mx.array(1e-30, dtype=mx.float32),
            )
            outputs.append(
                normalized.reshape(
                    batch, query_heads, query_width, head_dim
                ).astype(query.dtype)
            )
        return mx.concatenate(outputs, axis=2)

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
            if self_attention_projection == "fused_qkv":
                # Preserve the three authoritative parameter leaves and their
                # checkpoint/optimizer ABI. Concatenation happens only inside
                # the execution graph, reducing three same-input projections
                # to one matmul while gradients still flow to each leaf.
                projected = mx.matmul(
                    hidden,
                    mx.concatenate(
                        [
                            self.q_proj.weight,
                            self.k_proj.weight,
                            self.v_proj.weight,
                        ],
                        axis=0,
                    ).T,
                )
                query_width = config.num_heads * head_dim
                kv_width = config.num_kv_heads * head_dim
                query, key, value = mx.split(
                    projected,
                    (query_width, query_width + kv_width),
                    axis=-1,
                )
            else:
                query = self.q_proj(hidden)
                key = self.k_proj(hidden)
                value = self.v_proj(hidden)
            query = query.reshape(
                batch, length, config.num_heads, head_dim
            ).transpose(0, 2, 1, 3)
            key = key.reshape(
                batch, length, config.num_kv_heads, head_dim
            ).transpose(0, 2, 1, 3)
            value = value.reshape(
                batch, length, config.num_kv_heads, head_dim
            ).transpose(0, 2, 1, 3)
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
            # MLX executes grouped-query attention directly. Pre-tiling KV heads
            # multiplies memory traffic and defeats the native GQA kernel.
            attended = bounded_scaled_dot_product_attention(
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
            materialize_initialization(self)

        def __call__(
            self,
            hidden: Any,
            memory: Any,
            key_mask: Any | None = None,
            query_access: Any | None = None,
        ) -> Any:
            batch, original_length, _dims = hidden.shape
            query_start = 0
            if compact_encoder_decoder_partitions and query_access is not None:
                active_positions = mx.any(query_access > 0, axis=0)
                if not bool(mx.any(active_positions)):
                    return mx.zeros_like(hidden)
                query_start = int(mx.argmax(active_positions.astype(mx.int32)).item())
                hidden = hidden[:, query_start:, :]
            length = int(hidden.shape[1])
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
            attended = bounded_scaled_dot_product_attention(
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
            projected = self.out_proj(attended)
            if query_start:
                projected = mx.concatenate(
                    [
                        mx.zeros(
                            (batch, query_start, config.d_model),
                            dtype=projected.dtype,
                        ),
                        projected,
                    ],
                    axis=1,
                )
                if int(projected.shape[1]) != original_length:
                    raise ValueError("compacted cross-attention output lost alignment")
            return projected

    class SourceEncoderBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention_norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
            self.attention = CausalAttention()
            self.ffn_norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
            self.feed_forward = SwiGLU()
            if source_expert_adapter_enabled:
                self.expert_adapter = ExpertAdapter(config.source_expert_adapter_dim)
            materialize_initialization(self)

        def __call__(self, hidden: Any, source_mask: Any) -> Any:
            key_mask = mx.where(
                source_mask[:, None, None, :] > 0,
                mx.array(0.0, dtype=mx.float32),
                mx.array(-1e9, dtype=mx.float32),
            )
            attended, _cache = self.attention(
                self.attention_norm(hidden),
                attention_mask=key_mask,
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
            gate = self.gate(hidden)
            up = self.up(hidden)
            if config.feed_forward_activation == "situ_glu":
                gate_beta = float(config.situ_glu_gate_beta)
                up_beta = float(config.situ_glu_up_beta)
                bounded_gate = (
                    gate_beta
                    * mx.tanh(gate / gate_beta)
                    * mx.sigmoid(gate)
                )
                bounded_up = up_beta * mx.tanh(up / up_beta)
                return self.down(bounded_gate * bounded_up)
            return self.down(nn.silu(gate) * up)

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
            # A Python scalar is deliberate: structural-growth masks are
            # schedule state rather than learned parameters.  A zero mask is
            # an exact identity at a growth boundary; intermediate values
            # introduce the newly materialized block continuously.
            self.structural_growth_mask = 1.0
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
            if kerc_decoder_stage_adapter_enabled:
                self.kerc_decoder_stage_adapters = [
                    ExpertAdapter(config.kerc_decoder_stage_adapter_dim)
                    for _stage in range(4)
                ]
            if state_enabled:
                self.state_embedding = nn.Embedding(config.state_memory_slots, config.d_model)
                self.state_candidate = nn.Linear(config.d_model * 2, config.d_model, bias=False)
                self.state_gate = nn.Linear(config.d_model * 2, config.d_model, bias=True)
            materialize_initialization(self)

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
            kerc_stage_weights: Any | None = None,
            attention_mask: Any | None = None,
            structural_growth_mask: Any | None = None,
        ) -> tuple[Any, tuple[Any, ...]]:
            growth_input = hidden
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
                    self.source_attention_norm(hidden),
                    source_memory,
                    source_mask,
                    source_access if compact_encoder_decoder_partitions else None,
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
            if (
                kerc_decoder_stage_adapter_enabled
                and kerc_stage_weights is not None
            ):
                uniform_stage = uniform_kerc_stage_index(kerc_stage_weights)
                if uniform_stage is None:
                    stage_delta = mx.sum(
                        mx.stack(
                            [
                                adapter(hidden)
                                for adapter in self.kerc_decoder_stage_adapters
                            ],
                            axis=1,
                        )
                        * kerc_stage_weights[:, :, None, None],
                        axis=1,
                    )
                else:
                    stage_delta = self.kerc_decoder_stage_adapters[
                        uniform_stage
                    ](hidden)
                access = (
                    source_access
                    if source_access is not None
                    else mx.ones(hidden.shape[:2], dtype=mx.float32)
                )
                hidden = hidden + stage_delta * access[:, :, None]
            growth_mask = structural_growth_mask
            if growth_mask is not None:
                if cache is not None or state_enabled:
                    raise ValueError(
                        "masked structural growth is qualified only for "
                        "cache-free non-recurrent training"
                    )
                hidden = growth_input + growth_mask * (hidden - growth_input)
            elif float(self.structural_growth_mask) != 1.0:
                if cache is not None or state_enabled:
                    raise ValueError(
                        "masked structural growth is qualified only for "
                        "cache-free non-recurrent training"
                    )
                eager_growth_mask = float(self.structural_growth_mask)
                hidden = (
                    growth_input
                    if eager_growth_mask == 0.0
                    else growth_input
                    + eager_growth_mask * (hidden - growth_input)
                )
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
            if config.attention_residual_mode == "block":
                self.attention_residual_queries = nn.Embedding(
                    config.num_layers + 1, config.d_model
                )
            # A mechanics-only scalar permits an exact reduction-to-control
            # check. The qualified candidate itself always uses scale 1.
            self.attention_residual_scale = 1.0
            self.scale = math.sqrt(config.d_model)
            self.gradient_checkpointing = bool(gradient_checkpointing)
            self.kerc_stage_output_isolation = False
            self.compact_output_projection = bool(
                compact_encoder_decoder_partitions
                or compact_output_projection
            )
            self.copy_auxiliary_loss_weight = float(
                config.source_copy_auxiliary_loss_weight
            )
            self.mtp_future_offsets = tuple(int(value) for value in config.mtp_future_offsets)
            self.mtp_loss_weights = tuple(float(value) for value in config.mtp_loss_weights)
            self.mtp_loss_scale = float(config.mtp_loss_scale)
            self.kerc_task_token_ids = tuple(
                int(value) for value in config.kerc_task_token_ids
            )
            self.kerc_residual_choice_count = int(
                config.kerc_residual_choice_count
            )
            if mtp_enabled:
                if mtp_head_mode == "shared_low_rank":
                    self.mtp_shared_projection = nn.Linear(
                        config.d_model, config.mtp_low_rank, bias=False
                    )
                    self.mtp_output_heads = [
                        nn.Linear(config.mtp_low_rank, config.vocab_size, bias=False)
                        for _offset in self.mtp_future_offsets
                    ]
                elif mtp_head_mode == "independent_mlp":
                    self.mtp_input_heads = [
                        nn.Linear(config.d_model, config.mtp_hidden_dim, bias=False)
                        for _offset in self.mtp_future_offsets
                    ]
                    self.mtp_output_heads = [
                        nn.Linear(config.mtp_hidden_dim, config.vocab_size, bias=False)
                        for _offset in self.mtp_future_offsets
                    ]
                else:
                    self.mtp_registers = nn.Embedding(
                        config.mtp_register_count, config.d_model
                    )
                    self.mtp_shared_projection = nn.Linear(
                        config.d_model, config.mtp_hidden_dim, bias=False
                    )
                    self.mtp_output_heads = [
                        nn.Linear(config.mtp_hidden_dim, config.vocab_size, bias=False)
                        for _offset in self.mtp_future_offsets
                    ]
            if kerc_enabled:
                self.kerc_stage_embedding = nn.Embedding(4, config.d_model)
                self.kerc_stage_adapters = [
                    ExpertAdapter(config.kerc_stage_adapter_dim) for _stage in range(4)
                ]
                self.kerc_kernel_output = nn.Linear(
                    config.d_model, config.vocab_size, bias=False
                )
                self.kerc_surface_output = nn.Linear(
                    config.d_model, config.vocab_size, bias=False
                )
                if kerc_reasoner_output_delta_enabled:
                    self.kerc_reasoner_output_delta_down = nn.Linear(
                        config.d_model,
                        config.kerc_reasoner_output_delta_dim,
                        bias=False,
                    )
                    self.kerc_reasoner_output_delta_up = nn.Linear(
                        config.kerc_reasoner_output_delta_dim,
                        config.vocab_size,
                        bias=False,
                    )
                    self.kerc_reasoner_output_delta_up.weight = mx.zeros_like(
                        self.kerc_reasoner_output_delta_up.weight
                    )
                self.kerc_residual_encoder = nn.Linear(
                    config.d_model,
                    config.kerc_residual_bottleneck_dim,
                    bias=False,
                )
                self.kerc_residual_allocator = nn.Linear(
                    config.kerc_residual_bottleneck_dim,
                    4 * config.kerc_residual_choice_count,
                    bias=True,
                )
                self.kerc_residual_values = nn.Embedding(
                    4 * config.kerc_residual_choice_count, config.d_model
                )
                if kerc_unit_allocator_enabled:
                    self.kerc_unit_byte_embedding = nn.Embedding(
                        config.kerc_residual_unit_byte_vocab_size, config.d_model
                    )
                    self.kerc_unit_kind_embedding = nn.Embedding(
                        config.kerc_residual_unit_kind_count, config.d_model
                    )
                    self.kerc_unit_content_projection = nn.Linear(
                        config.d_model,
                        config.kerc_residual_bottleneck_dim,
                        bias=False,
                    )
                    self.kerc_unit_source_projection = nn.Linear(
                        config.d_model,
                        config.kerc_residual_bottleneck_dim,
                        bias=False,
                    )
                    self.kerc_unit_query = nn.Linear(
                        config.kerc_residual_bottleneck_dim,
                        config.kerc_residual_bottleneck_dim,
                        bias=False,
                    )
                    self.kerc_unit_key = nn.Linear(
                        config.kerc_residual_bottleneck_dim,
                        config.kerc_residual_bottleneck_dim,
                        bias=False,
                    )
                    self.kerc_unit_value = nn.Linear(
                        config.kerc_residual_bottleneck_dim,
                        config.kerc_residual_bottleneck_dim,
                        bias=False,
                    )
                    self.kerc_unit_attention_output = nn.Linear(
                        config.kerc_residual_bottleneck_dim,
                        config.kerc_residual_bottleneck_dim,
                        bias=False,
                    )
                    self.kerc_unit_candidate_projection = nn.Linear(
                        config.kerc_residual_unit_feature_dim,
                        config.kerc_residual_bottleneck_dim,
                        bias=False,
                    )
                    self.kerc_unit_fidelity_feature_embedding = nn.Embedding(
                        config.kerc_residual_choice_count,
                        config.kerc_residual_bottleneck_dim,
                    )
                    self.kerc_unit_candidate_scorer = nn.Linear(
                        config.kerc_residual_bottleneck_dim, 1, bias=True
                    )
                    self.kerc_unit_confidence = nn.Linear(
                        config.kerc_residual_bottleneck_dim, 1, bias=True
                    )
                    self.kerc_unit_fidelity_values = nn.Embedding(
                        config.kerc_residual_choice_count, config.d_model
                    )
                # The verifier intentionally has its own token embedding and projections;
                # it does not reuse answer-producing hidden states.
                self.kerc_verifier_embedding = nn.Embedding(
                    config.vocab_size, config.kerc_verifier_dim
                )
                self.kerc_verifier_source = nn.Linear(
                    config.kerc_verifier_dim, config.kerc_verifier_dim, bias=False
                )
                self.kerc_verifier_target = nn.Linear(
                    config.kerc_verifier_dim, config.kerc_verifier_dim, bias=False
                )
                self.kerc_verifier_classifier = nn.Linear(
                    4 * config.kerc_verifier_dim,
                    config.kerc_verifier_output_dim,
                    bias=True,
                )
                self.kerc_decision_encoder = nn.Linear(
                    config.d_model,
                    config.kerc_decision_bottleneck_dim,
                    bias=False,
                )
                self.kerc_decision_classifier = nn.Linear(
                    config.kerc_decision_bottleneck_dim,
                    config.kerc_decision_output_dim,
                    bias=True,
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
            materialize_initialization(self)

        def attention_residual_mix(
            self,
            *,
            query_index: int,
            sources: list[Any],
            sequential_hidden: Any,
        ) -> Any:
            """Retrieve over depth exactly as K3 Block AttnRes equations 8-10."""

            if config.attention_residual_mode != "block":
                return sequential_hidden
            scale = float(self.attention_residual_scale)
            if scale == 0.0:
                return sequential_hidden
            values = mx.stack(sources, axis=2)
            denominator = mx.sqrt(
                mx.mean(mx.square(values), axis=-1, keepdims=True)
                + float(config.rms_norm_eps)
            )
            keys = values / denominator
            query = self.attention_residual_queries.weight[query_index]
            logits = mx.sum(keys * query[None, None, None, :], axis=-1)
            weights = mx.softmax(logits, axis=-1)
            mixed = mx.sum(weights[:, :, :, None] * values, axis=2)
            if scale == 1.0:
                return mixed
            return sequential_hidden + scale * (
                mixed - sequential_hidden
            )

        def set_structural_growth_masks(self, masks: tuple[float, ...]) -> None:
            """Set cache-free decoder residual masks for staged depth growth."""

            if len(masks) != len(self.layers):
                raise ValueError(
                    "one structural-growth mask is required per decoder layer"
                )
            normalized = tuple(float(value) for value in masks)
            if any(not 0.0 <= value <= 1.0 for value in normalized):
                raise ValueError("structural-growth masks must be in [0, 1]")
            if state_enabled and any(value != 1.0 for value in normalized):
                raise ValueError(
                    "masked structural growth is not qualified with recurrent state"
                )
            for layer, value in zip(self.layers, normalized):
                layer.structural_growth_mask = value

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

        def freeze_to_kerc_delta(
            self, *, include_source_conditioned_bridge: bool = False
        ) -> None:
            """Preserve the warm trunk and train only faithful KERC modules."""

            if not kerc_enabled:
                raise ValueError("model has no KERC delta to train")
            self.freeze()
            module_names = [
                "kerc_stage_embedding",
                "kerc_kernel_output",
                "kerc_surface_output",
                "kerc_reasoner_output_delta_down",
                "kerc_reasoner_output_delta_up",
                "kerc_residual_encoder",
                "kerc_residual_allocator",
                "kerc_residual_values",
                "kerc_unit_byte_embedding",
                "kerc_unit_kind_embedding",
                "kerc_unit_content_projection",
                "kerc_unit_source_projection",
                "kerc_unit_query",
                "kerc_unit_key",
                "kerc_unit_value",
                "kerc_unit_attention_output",
                "kerc_unit_candidate_projection",
                "kerc_unit_fidelity_feature_embedding",
                "kerc_unit_candidate_scorer",
                "kerc_unit_confidence",
                "kerc_unit_fidelity_values",
                "kerc_verifier_embedding",
                "kerc_verifier_source",
                "kerc_verifier_target",
                "kerc_verifier_classifier",
                "kerc_decision_encoder",
                "kerc_decision_classifier",
            ]
            for adapter in self.kerc_stage_adapters:
                adapter.unfreeze()
            if kerc_decoder_stage_adapter_enabled:
                for layer in self.layers:
                    for adapter in layer.kerc_decoder_stage_adapters:
                        adapter.unfreeze()
            for name in module_names:
                module = getattr(self, name, None)
                if module is not None:
                    module.unfreeze()
            if include_source_conditioned_bridge:
                if not source_encoder_enabled:
                    raise ValueError(
                        "KERC source-conditioned bridge requires the source encoder"
                    )
                for layer in self.layers:
                    layer.source_attention_norm.unfreeze()
                    layer.source_attention.unfreeze()
                for layer in self.source_layers:
                    layer.unfreeze()
                self.source_final_norm.unfreeze()

        def freeze_to_kerc_stage(
            self,
            stage_index: int,
            *,
            include_stage_embedding: bool = True,
            detach_frozen_trunk: bool = False,
        ) -> None:
            """Train one KERC generation stage without changing sibling stages."""

            if not kerc_enabled or stage_index not in range(4):
                raise ValueError("KERC stage-only scope requires a stage index in [0, 3]")
            if detach_frozen_trunk and (
                stage_index != 1 or include_stage_embedding
            ):
                raise ValueError(
                    "detached KERC trunk isolation requires compiler stage 1 "
                    "with a frozen stage embedding"
                )
            self.freeze()
            self.kerc_stage_output_isolation = bool(detach_frozen_trunk)
            if include_stage_embedding:
                self.kerc_stage_embedding.unfreeze()
            self.kerc_stage_adapters[stage_index].unfreeze()
            if kerc_decoder_stage_adapter_enabled:
                for layer in self.layers:
                    layer.kerc_decoder_stage_adapters[stage_index].unfreeze()
            if stage_index == 1:
                self.kerc_kernel_output.unfreeze()
            elif stage_index == 2 and kerc_reasoner_output_delta_enabled:
                self.kerc_reasoner_output_delta_down.unfreeze()
                self.kerc_reasoner_output_delta_up.unfreeze()
            elif stage_index == 3:
                self.kerc_surface_output.unfreeze()

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
            # Token zero is the canonical padding id.  Excluding right-padding
            # here lets compact decoder execution recover the semantic target
            # extent even when the outer training tensor is shape-bucketed.
            target_access = (
                (seen_separator > 0) & ~separator & (tokens != 0)
            ).astype(mx.float32)
            return source_mask, target_access, has_separator

        def encode_source(
            self, tokens: Any, *, assume_separator: bool = False
        ) -> tuple[Any | None, Any | None, Any | None, Any | None]:
            """Encode only the prompt partition; target values cannot affect this memory."""

            if not source_encoder_enabled:
                return None, None, None, None
            source_mask, target_access, has_separator = self.source_partition(tokens)
            if not assume_separator and not bool(mx.any(has_separator > 0)):
                return None, None, None, None
            source_tokens = tokens
            if compact_encoder_decoder_partitions:
                source_width = int(mx.max(mx.sum(source_mask, axis=1)).item())
                if source_width <= 0:
                    return None, None, target_access, None
                source_tokens = tokens[:, :source_width]
                source_mask = source_mask[:, :source_width]
                if compact_partition_width_quantum:
                    bucket_width = int(
                        math.ceil(
                            source_width / compact_partition_width_quantum
                        )
                        * compact_partition_width_quantum
                    )
                    padding = bucket_width - source_width
                    if padding:
                        source_tokens = mx.pad(
                            source_tokens, ((0, 0), (0, padding))
                        )
                        source_mask = mx.pad(
                            source_mask, ((0, 0), (0, padding))
                        )
            hidden = self.token_embedding(source_tokens) * self.scale
            hidden = hidden * source_mask[:, :, None]
            for layer in self.source_layers:
                if self.gradient_checkpointing and self.training:
                    def checkpointed_source(
                        parameters: Any,
                        layer_hidden: Any,
                        layer_mask: Any,
                        layer_module: Any = layer,
                    ) -> Any:
                        layer_module.update(parameters)
                        return layer_module(layer_hidden, layer_mask)

                    hidden = mx.checkpoint(checkpointed_source)(
                        layer.trainable_parameters(), hidden, source_mask
                    )
                else:
                    hidden = layer(hidden, source_mask)
                hidden = hidden * source_mask[:, :, None]
            memory = self.source_final_norm(hidden) * source_mask[:, :, None]
            copy_ids = (
                source_to_target_lookup[source_tokens]
                if pointer_generator_enabled
                else None
            )
            return memory, source_mask, target_access, copy_ids

        def output_logits(
            self,
            hidden: Any,
            source_memory: Any | None,
            source_mask: Any | None,
            source_copy_ids: Any | None,
            generator_logits: Any | None = None,
            pointer_access: Any | None = None,
        ) -> tuple[Any, dict[str, Any] | None]:
            if generator_logits is None:
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
            vocabulary_size = int(generator_logits.shape[-1])
            valid = (
                (source_mask > 0)
                & (source_copy_ids >= 0)
                & (source_copy_ids < vocabulary_size)
            )
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
            copy_indices = mx.broadcast_to(
                source_copy_ids[:, None, :], pointer_scores.shape
            )
            # Every non-copy source position gets a distinct scratch column. Reusing
            # vocabulary slot zero (or one shared sentinel) creates duplicate scatter
            # indices whose Metal write order is undefined and made identical forwards
            # disagree. Valid copy IDs are already unique per row above.
            scratch_indices = mx.broadcast_to(
                (
                    vocabulary_size
                    + mx.arange(source_length, dtype=mx.int32)
                )[None, None, :],
                pointer_scores.shape,
            )
            indices = mx.where(
                unique_valid[:, None, :], copy_indices, scratch_indices
            )
            pointer_logits_with_scratch = mx.full(
                (*generator_logits.shape[:-1], vocabulary_size + source_length),
                -1e9,
                dtype=mx.float32,
            )
            pointer_logits_with_scratch = mx.put_along_axis(
                pointer_logits_with_scratch, indices, pointer_scores, axis=2
            )
            pointer_logits = pointer_logits_with_scratch[:, :, :vocabulary_size]
            generator_log_probs = generator_logits - mx.logsumexp(
                generator_logits, axis=-1, keepdims=True
            )
            pointer_log_probs = pointer_logits - mx.logsumexp(
                pointer_logits, axis=-1, keepdims=True
            )
            gate = mx.sigmoid(self.copy_gate(hidden))
            if pointer_access is not None:
                access = pointer_access[:, None, None]
                gate = gate * access + (1.0 - access)
            logits = mx.logaddexp(
                generator_log_probs + mx.log(mx.maximum(gate, 1e-6)),
                pointer_log_probs + mx.log(mx.maximum(1.0 - gate, 1e-6)),
            )
            if pointer_access is not None:
                # A zero-authority stage must be generator-only exactly. The
                # numerical 1e-6 mixture floor above is appropriate for an
                # active learned gate, but otherwise lets copied source IDs
                # bypass stage-specific output vocabulary masks.
                logits = mx.where(
                    pointer_access[:, None, None] > 0,
                    logits,
                    generator_log_probs,
                )
            return logits, {
                "pointer_scores": pointer_scores,
                "source_copy_ids": source_copy_ids,
                "source_copy_valid": unique_valid,
                "generator_gate": gate[:, :, 0],
            }

        def kerc_stage_weights(self, tokens: Any) -> Any | None:
            if not kerc_enabled:
                return None
            matches = mx.stack(
                [
                    mx.any(tokens == token_id, axis=1).astype(mx.float32)
                    for token_id in self.kerc_task_token_ids
                ],
                axis=-1,
            )
            count = mx.sum(matches, axis=-1, keepdims=True)
            # Multiple control tokens fail closed to the ordinary surface route.
            valid = count == 1.0
            weights = mx.where(valid, matches, mx.zeros_like(matches))
            if config.kerc_stage_routing_ablation == "zero":
                return mx.zeros_like(weights)
            return weights

        def kerc_allocate_units(
            self,
            *,
            unit_byte_ids: Any,
            unit_byte_mask: Any | None,
            unit_kind_ids: Any,
            unit_candidate_features: Any,
            unit_mask: Any,
            unit_hard_block_mask: Any,
            unit_byte_offsets: Any | None = None,
            source_summary: Any | None = None,
        ) -> dict[str, Any]:
            """Score concrete residual candidates after compiler unit creation."""

            if not kerc_unit_allocator_enabled:
                raise ValueError("KERC per-unit allocator is not configured")
            batch, unit_count = (int(value) for value in unit_kind_ids.shape)
            choices = int(config.kerc_residual_choice_count)
            if tuple(unit_candidate_features.shape) != (
                batch,
                unit_count,
                choices,
                int(config.kerc_residual_unit_feature_dim),
            ):
                raise ValueError("KERC unit candidate feature shape mismatch")
            if tuple(unit_mask.shape) != (batch, unit_count):
                raise ValueError("KERC unit mask shape mismatch")
            if tuple(unit_hard_block_mask.shape) != (batch, unit_count, choices):
                raise ValueError("KERC unit hard-block mask shape mismatch")
            if unit_byte_offsets is not None:
                if unit_byte_mask is not None or len(unit_byte_ids.shape) != 1:
                    raise ValueError("KERC ragged bytes require flat IDs without a mask")
                if tuple(unit_byte_offsets.shape) != (batch, unit_count, 2):
                    raise ValueError("KERC unit byte offsets shape mismatch")
                embedded = self.kerc_unit_byte_embedding(
                    unit_byte_ids.astype(mx.int32)
                )
                prefix = mx.concatenate(
                    [
                        mx.zeros((1, config.d_model), dtype=embedded.dtype),
                        mx.cumsum(embedded, axis=0),
                    ],
                    axis=0,
                )
                starts = unit_byte_offsets[:, :, 0].astype(mx.int32)
                ends = unit_byte_offsets[:, :, 1].astype(mx.int32)
                byte_denominator = mx.maximum(
                    (ends - starts).astype(mx.float32)[:, :, None], 1.0
                )
                content = (
                    mx.take(prefix, ends, axis=0)
                    - mx.take(prefix, starts, axis=0)
                ) / byte_denominator
            else:
                if len(unit_byte_ids.shape) != 3:
                    raise ValueError("KERC dense bytes require rank-three IDs")
                byte_width = int(unit_byte_ids.shape[2])
                if unit_byte_mask is None or tuple(unit_byte_mask.shape) != (
                    batch,
                    unit_count,
                    byte_width,
                ):
                    raise ValueError("KERC unit byte mask shape mismatch")
                embedded = self.kerc_unit_byte_embedding(
                    unit_byte_ids.astype(mx.int32)
                )
                byte_authority = unit_byte_mask.astype(mx.float32)
                byte_denominator = mx.maximum(
                    mx.sum(byte_authority, axis=2, keepdims=True), 1.0
                )
                content = mx.sum(
                    embedded * byte_authority[:, :, :, None], axis=2
                ) / byte_denominator
            content = content + self.kerc_unit_kind_embedding(
                unit_kind_ids.astype(mx.int32)
            )
            hidden = self.kerc_unit_content_projection(content)
            if source_summary is not None:
                if tuple(source_summary.shape) != (batch, config.d_model):
                    raise ValueError("KERC unit source summary shape mismatch")
                hidden = hidden + self.kerc_unit_source_projection(source_summary)[:, None, :]
            hidden = nn.silu(hidden)
            query = self.kerc_unit_query(hidden)
            key = self.kerc_unit_key(hidden)
            value = self.kerc_unit_value(hidden)
            scores = mx.matmul(query, mx.swapaxes(key, 1, 2)) / math.sqrt(
                config.kerc_residual_bottleneck_dim
            )
            key_mask = unit_mask.astype(mx.float32)[:, None, :]
            scores = mx.where(
                key_mask > 0.0,
                scores,
                mx.full(scores.shape, -1e9, dtype=mx.float32),
            )
            attended = mx.matmul(mx.softmax(scores, axis=-1), value)
            hidden = nn.silu(
                hidden + self.kerc_unit_attention_output(attended)
            ) * unit_mask[:, :, None].astype(mx.float32)
            candidate_state = (
                self.kerc_unit_candidate_projection(unit_candidate_features)
                + self.kerc_unit_fidelity_feature_embedding(
                    mx.arange(choices, dtype=mx.int32)
                )[None, None, :, :]
            )
            unit_state = hidden[:, :, None, :]
            # Allocation is a source-candidate relation. The multiplicative term
            # lets source semantics reorder an otherwise identical rate schedule.
            candidate_hidden = nn.silu(
                candidate_state
                + unit_state
                + candidate_state
                * unit_state
                / math.sqrt(config.kerc_residual_bottleneck_dim)
            )
            logits = self.kerc_unit_candidate_scorer(candidate_hidden)[..., 0]
            logits = mx.where(
                unit_hard_block_mask.astype(mx.bool_),
                mx.full(logits.shape, -1e9, dtype=mx.float32),
                logits,
            )
            logits = mx.where(
                unit_mask[:, :, None] > 0,
                logits,
                mx.zeros_like(logits),
            )
            confidence_logits = self.kerc_unit_confidence(hidden)[..., 0]
            probabilities = mx.softmax(logits, axis=-1)
            fidelity_values = self.kerc_unit_fidelity_values(
                mx.arange(choices, dtype=mx.int32)
            )
            selected_context = mx.sum(
                probabilities[:, :, :, None] * fidelity_values[None, None, :, :],
                axis=2,
            )
            denominator = mx.maximum(
                mx.sum(unit_mask.astype(mx.float32), axis=1, keepdims=True), 1.0
            )
            residual_context = mx.sum(
                selected_context * unit_mask[:, :, None].astype(mx.float32), axis=1
            ) / denominator
            if config.kerc_residual_ablation == "zero":
                logits = mx.zeros_like(logits)
                confidence_logits = mx.zeros_like(confidence_logits)
                residual_context = mx.zeros_like(residual_context)
            return {
                "logits": logits,
                "confidence_logits": confidence_logits,
                "residual_context": residual_context,
                "unit_hidden": hidden,
            }

        def kerc_context(
            self,
            tokens: Any,
            source_memory: Any | None,
            source_mask: Any | None,
            *,
            unit_byte_ids: Any | None = None,
            unit_byte_mask: Any | None = None,
            unit_byte_offsets: Any | None = None,
            unit_kind_ids: Any | None = None,
            unit_candidate_features: Any | None = None,
            unit_mask: Any | None = None,
            unit_hard_block_mask: Any | None = None,
        ) -> tuple[Any, Any, Any, Any | None, Any | None]:
            stage_weights = self.kerc_stage_weights(tokens)
            if stage_weights is None or source_memory is None or source_mask is None:
                batch = int(tokens.shape[0])
                return (
                    mx.zeros((batch, config.d_model), dtype=mx.float32),
                    mx.zeros((batch, 4), dtype=mx.float32),
                    mx.zeros(
                        (batch, 4, config.kerc_residual_choice_count),
                        dtype=mx.float32,
                    ),
                    None,
                    None,
                )
            denominator = mx.maximum(mx.sum(source_mask, axis=1, keepdims=True), 1.0)
            summary = mx.sum(source_memory * source_mask[:, :, None], axis=1) / denominator
            stage_context = mx.matmul(stage_weights, self.kerc_stage_embedding.weight)
            # Residual fidelity is conditional on both source content and the trusted
            # compiler/core/renderer stage. Supplying stage context explicitly avoids
            # washing a single trusted token out of long-source mean pooling.
            residual_hidden = nn.silu(
                self.kerc_residual_encoder(summary + stage_context)
            )
            residual_logits = self.kerc_residual_allocator(residual_hidden).reshape(
                int(tokens.shape[0]), 4, config.kerc_residual_choice_count
            )
            residual_probabilities = mx.softmax(residual_logits, axis=-1)
            values = self.kerc_residual_values(
                mx.arange(4 * config.kerc_residual_choice_count, dtype=mx.int32)
            ).reshape(4, config.kerc_residual_choice_count, config.d_model)
            residual_levels = mx.sum(
                residual_probabilities[:, :, :, None] * values[None, :, :, :],
                axis=2,
            )
            if config.kerc_interaction_residual_ablation == "zero":
                residual_logits = mx.concatenate(
                    [
                        mx.zeros_like(residual_logits[:, :1, :]),
                        residual_logits[:, 1:, :],
                    ],
                    axis=1,
                )
                residual_levels = mx.concatenate(
                    [
                        mx.zeros_like(residual_levels[:, :1, :]),
                        residual_levels[:, 1:, :],
                    ],
                    axis=1,
                )
            residual_context = mx.mean(residual_levels, axis=1)
            unit_logits = None
            unit_confidence_logits = None
            unit_arguments = (
                unit_byte_ids,
                unit_kind_ids,
                unit_candidate_features,
                unit_mask,
                unit_hard_block_mask,
            )
            if any(value is not None for value in unit_arguments):
                if not all(value is not None for value in unit_arguments):
                    raise ValueError("KERC per-unit allocator inputs must be supplied together")
                if (unit_byte_mask is None) == (unit_byte_offsets is None):
                    raise ValueError(
                        "KERC per-unit allocator requires exactly one byte layout"
                    )
                unit_aux = self.kerc_allocate_units(
                    unit_byte_ids=unit_byte_ids,
                    unit_byte_mask=unit_byte_mask,
                    unit_byte_offsets=unit_byte_offsets,
                    unit_kind_ids=unit_kind_ids,
                    unit_candidate_features=unit_candidate_features,
                    unit_mask=unit_mask,
                    unit_hard_block_mask=unit_hard_block_mask,
                    source_summary=summary,
                )
                residual_context = unit_aux["residual_context"]
                unit_logits = unit_aux["logits"]
                unit_confidence_logits = unit_aux["confidence_logits"]
            if config.kerc_residual_ablation == "zero":
                residual_logits = mx.zeros_like(residual_logits)
                residual_context = mx.zeros_like(residual_context)
            # Residual state belongs on compiler and renderer paths, not the core-only
            # reasoner or the conventional surface control.
            residual_access = (stage_weights[:, 1] + stage_weights[:, 3])[:, None]
            return (
                stage_context + residual_context * residual_access,
                stage_weights,
                residual_logits,
                unit_logits,
                unit_confidence_logits,
            )

        def kerc_generator_logits(
            self, hidden: Any, stage_weights: Any | None
        ) -> Any:
            if not kerc_enabled or stage_weights is None:
                return self.token_embedding.as_linear(hidden)
            token_ids = mx.arange(config.vocab_size, dtype=mx.int32)
            surface_allowed = (
                (
                    (token_ids >= config.kerc_surface_token_start)
                    & (token_ids < config.kerc_surface_token_end)
                )
                | (token_ids == config.kerc_end_token_id)
            )
            kernel_allowed = (
                (
                    (token_ids >= config.kerc_kernel_token_start)
                    & (token_ids < config.kerc_kernel_token_end)
                )
                | (
                    (token_ids >= config.kerc_pointer_token_start)
                    & (token_ids < config.kerc_pointer_token_end)
                )
                | (token_ids == config.kerc_end_token_id)
            )
            surface_mask = mx.where(
                surface_allowed,
                mx.array(0.0, dtype=mx.float32),
                mx.array(-1e9, dtype=mx.float32),
            )
            kernel_mask = mx.where(
                kernel_allowed,
                mx.array(0.0, dtype=mx.float32),
                mx.array(-1e9, dtype=mx.float32),
            )
            uniform_stage = uniform_kerc_stage_index(stage_weights)
            if uniform_stage is not None:
                stage_hidden = hidden + self.kerc_stage_adapters[
                    uniform_stage
                ](hidden)
                if uniform_stage == 0:
                    return self.token_embedding.as_linear(
                        stage_hidden
                    ) + surface_mask
                if uniform_stage == 1:
                    return self.kerc_kernel_output(stage_hidden) + kernel_mask
                if uniform_stage == 2:
                    reasoner_delta = (
                        self.kerc_reasoner_output_delta_up(
                            nn.silu(
                                self.kerc_reasoner_output_delta_down(
                                    stage_hidden
                                )
                            )
                        )
                        if kerc_reasoner_output_delta_enabled
                        else 0.0
                    )
                    return (
                        self.token_embedding.as_linear(stage_hidden)
                        + reasoner_delta
                        + kernel_mask
                    )
                return self.kerc_surface_output(stage_hidden) + surface_mask
            base = self.token_embedding.as_linear(hidden)
            stage_hidden = [
                hidden + adapter(hidden) for adapter in self.kerc_stage_adapters
            ]
            stage_logits = mx.stack(
                [
                    self.token_embedding.as_linear(stage_hidden[0])
                    + surface_mask,
                    self.kerc_kernel_output(stage_hidden[1])
                    + kernel_mask,
                    self.token_embedding.as_linear(stage_hidden[2])
                    + (
                        self.kerc_reasoner_output_delta_up(
                            nn.silu(
                                self.kerc_reasoner_output_delta_down(
                                    stage_hidden[2]
                                )
                            )
                        )
                        if kerc_reasoner_output_delta_enabled
                        else 0.0
                    )
                    + kernel_mask,
                    self.kerc_surface_output(stage_hidden[3])
                    + surface_mask,
                ],
                axis=1,
            )
            selected = mx.sum(
                stage_logits * stage_weights[:, :, None, None], axis=1
            )
            active = mx.sum(stage_weights, axis=-1)[:, None, None]
            return base * (1.0 - active) + selected

        def kerc_verifier_logits(self, tokens: Any) -> Any | None:
            if not kerc_enabled:
                return None
            if config.kerc_verifier_ablation == "zero":
                return mx.zeros(
                    (int(tokens.shape[0]), config.kerc_verifier_output_dim),
                    dtype=mx.float32,
                )
            separator = tokens == config.source_target_separator_token_id
            seen = mx.cumsum(separator.astype(mx.int32), axis=1)
            has_separator = (mx.sum(separator.astype(mx.int32), axis=1) > 0).astype(
                mx.float32
            )
            source_mask = (seen == 0).astype(mx.float32) * has_separator[:, None]
            target_mask = ((seen > 0) & ~separator).astype(mx.float32)
            embedded = self.kerc_verifier_embedding(tokens)
            source = mx.sum(embedded * source_mask[:, :, None], axis=1) / mx.maximum(
                mx.sum(source_mask, axis=1, keepdims=True), 1.0
            )
            target = mx.sum(embedded * target_mask[:, :, None], axis=1) / mx.maximum(
                mx.sum(target_mask, axis=1, keepdims=True), 1.0
            )
            source = self.kerc_verifier_source(source)
            target = self.kerc_verifier_target(target)
            features = mx.concatenate(
                [source, target, mx.abs(source - target), source * target], axis=-1
            )
            return self.kerc_verifier_classifier(features) * has_separator[:, None]

        def kerc_decision_logits(
            self,
            tokens: Any,
            source_memory: Any | None,
            source_mask: Any | None,
        ) -> Any | None:
            """Predict answer disposition from source and trusted objective only."""

            if not kerc_enabled or source_memory is None or source_mask is None:
                return None
            batch = int(tokens.shape[0])
            if config.kerc_decision_ablation == "zero":
                return mx.zeros(
                    (batch, config.kerc_decision_output_dim), dtype=mx.float32
                )
            stage_weights = self.kerc_stage_weights(tokens)
            if stage_weights is None:
                return None
            denominator = mx.maximum(
                mx.sum(source_mask, axis=1, keepdims=True), 1.0
            )
            source_summary = (
                mx.sum(source_memory * source_mask[:, :, None], axis=1)
                / denominator
            )
            stage_context = mx.matmul(
                stage_weights, self.kerc_stage_embedding.weight
            )
            active = mx.sum(stage_weights, axis=-1, keepdims=True)
            hidden = nn.silu(
                self.kerc_decision_encoder(source_summary + stage_context)
            )
            return self.kerc_decision_classifier(hidden) * active

        def mtp_logits(self, hidden: Any) -> list[Any]:
            if not mtp_enabled:
                return []
            if mtp_head_mode == "shared_low_rank":
                shared = self.mtp_shared_projection(hidden)
                return [head(shared) for head in self.mtp_output_heads]
            if mtp_head_mode == "independent_mlp":
                return [
                    output_head(nn.silu(input_head(hidden)))
                    for input_head, output_head in zip(
                        self.mtp_input_heads, self.mtp_output_heads
                    )
                ]
            return [
                output_head(
                    nn.silu(
                        self.mtp_shared_projection(
                            hidden + self.mtp_registers.weight[index]
                        )
                    )
                )
                for index, output_head in enumerate(self.mtp_output_heads)
            ]

        def conditioned_embeddings(
            self,
            tokens: Any,
            cached_plan_context: Any | None,
            source_mask: Any | None = None,
            input_embeddings: Any | None = None,
        ) -> tuple[Any, Any | None, Any | None, Any | None]:
            hidden = (
                input_embeddings
                if input_embeddings is not None
                else self.token_embedding(tokens) * self.scale
            )
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
            source_conditioning: bool | None = None,
            return_plan_logits: bool = False,
            return_copy_aux: bool = False,
            return_training_aux: bool = False,
            input_embeddings: Any | None = None,
            kerc_unit_byte_ids: Any | None = None,
            kerc_unit_byte_mask: Any | None = None,
            kerc_unit_byte_offsets: Any | None = None,
            kerc_unit_kind_ids: Any | None = None,
            kerc_unit_candidate_features: Any | None = None,
            kerc_unit_mask: Any | None = None,
            kerc_unit_hard_block_mask: Any | None = None,
            output_position_mask: Any | None = None,
            auxiliary_only: bool = False,
            structural_growth_masks: Any | None = None,
        ) -> Any:
            if return_training_aux and (return_plan_logits or return_copy_aux):
                raise ValueError(
                    "return_training_aux cannot be combined with legacy auxiliary returns"
                )
            if output_position_mask is not None and tuple(output_position_mask.shape) != tuple(
                tokens.shape
            ):
                raise ValueError("output position mask must match token batch and length")
            if structural_growth_masks is not None:
                if tuple(structural_growth_masks.shape) != (config.num_layers,):
                    raise ValueError(
                        "structural-growth masks must have one value per decoder layer"
                    )
                if cache is not None or state_enabled:
                    raise ValueError(
                        "masked structural growth is qualified only for "
                        "cache-free non-recurrent training"
                    )
                if config.attention_residual_mode != "none":
                    raise ValueError(
                        "masked structural growth and attention residuals "
                        "require separate qualification"
                    )
            if input_embeddings is not None:
                if tuple(input_embeddings.shape) != (
                    int(tokens.shape[0]),
                    int(tokens.shape[1]),
                    config.d_model,
                ):
                    raise ValueError("input embeddings must match token batch, length, and model width")
                if source_encoder_enabled or plan_enabled or kerc_enabled or state_enabled:
                    raise ValueError(
                        "external input embeddings currently require the plain causal core"
                    )
            if auxiliary_only and (
                not return_training_aux
                or not kerc_enabled
                or cache is not None
                or input_embeddings is not None
            ):
                raise ValueError(
                    "auxiliary-only execution requires cache-free KERC training auxiliaries"
                )
            cached_plan_context = None
            cached_source_memory = None
            cached_source_mask = None
            cached_source_copy_ids = None
            cached_kerc_context = None
            cached_kerc_stage_weights = None
            cached_kerc_residual_logits = None
            cached_kerc_unit_logits = None
            cached_kerc_unit_confidence_logits = None
            layer_cache_input = cache
            trailing = 0
            if kerc_enabled and cache is not None and len(cache) > config.num_layers:
                kerc_cache = cache[-1]
                if len(kerc_cache) == 3:
                    (
                        cached_kerc_context,
                        cached_kerc_stage_weights,
                        cached_kerc_residual_logits,
                    ) = kerc_cache
                elif len(kerc_cache) == 5:
                    (
                        cached_kerc_context,
                        cached_kerc_stage_weights,
                        cached_kerc_residual_logits,
                        cached_kerc_unit_logits,
                        cached_kerc_unit_confidence_logits,
                    ) = kerc_cache
                else:
                    raise ValueError("KERC cache entry has an unsupported shape")
                trailing += 1
            if (
                plan_enabled
                and cache is not None
                and len(cache) > config.num_layers + trailing
            ):
                cached_plan_context = cache[-(trailing + 1)][0]
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
            if source_encoder_enabled and source_conditioning is not False:
                if source_memory is None:
                    (
                        source_memory,
                        source_mask,
                        source_access,
                        source_copy_ids,
                    ) = self.encode_source(
                        tokens, assume_separator=source_conditioning is True
                    )
                elif cache is not None:
                    source_access = mx.ones(tokens.shape, dtype=mx.float32)
            if auxiliary_only:
                (
                    _kerc_context,
                    kerc_stage_weights,
                    kerc_residual_logits,
                    kerc_unit_logits,
                    kerc_unit_confidence_logits,
                ) = self.kerc_context(
                    tokens,
                    source_memory,
                    source_mask,
                    unit_byte_ids=kerc_unit_byte_ids,
                    unit_byte_mask=kerc_unit_byte_mask,
                    unit_byte_offsets=kerc_unit_byte_offsets,
                    unit_kind_ids=kerc_unit_kind_ids,
                    unit_candidate_features=kerc_unit_candidate_features,
                    unit_mask=kerc_unit_mask,
                    unit_hard_block_mask=kerc_unit_hard_block_mask,
                )
                batch = int(tokens.shape[0])
                # No supervised generator position exists on corruption-only
                # verifier/decision rows.  Decoder, pointer, and vocabulary
                # projection terms therefore have exactly zero loss and zero
                # gradient; omit them rather than materializing a large graph.
                logits = mx.zeros((batch, 1, config.vocab_size), dtype=mx.float32)
                return logits, [], {
                    "final_hidden": mx.zeros(
                        (batch, 1, config.d_model), dtype=mx.float32
                    ),
                    "output_position_start": int(tokens.shape[1]) - 1,
                    "plan_logits": None,
                    "copy_aux": None,
                    "mtp_logits": [],
                    "decoder_execution": "pruned_zero_authority",
                    "kerc": {
                        "stage_weights": kerc_stage_weights,
                        "residual_logits": kerc_residual_logits,
                        "unit_residual_logits": kerc_unit_logits,
                        "unit_confidence_logits": kerc_unit_confidence_logits,
                        "verifier_logits": self.kerc_verifier_logits(tokens),
                        "decision_logits": self.kerc_decision_logits(
                            tokens, source_memory, source_mask
                        ),
                    },
                }
            conditioned_hidden, plan_context, plan_logits, plan_access = self.conditioned_embeddings(
                tokens,
                cached_plan_context,
                (
                    self.source_partition(tokens)[0]
                    if compact_encoder_decoder_partitions
                    and cached_source_memory is None
                    and source_mask is not None
                    else source_mask if cached_source_memory is None else None
                ),
                input_embeddings,
            )
            decoder_tokens = tokens
            decoder_position_start = 0
            decoder_unpadded_width = 0
            if (
                compact_encoder_decoder_partitions
                and source_memory is not None
                and source_access is not None
                and cache is None
            ):
                active_decoder = mx.any(source_access > 0, axis=0)
                if not bool(mx.any(active_decoder)):
                    raise ValueError(
                        "compact encoder-decoder requires a nonempty target partition"
                    )
                decoder_position_start = int(
                    mx.argmax(active_decoder.astype(mx.int32)).item()
                )
                decoder_position_stop = int(
                    int(active_decoder.shape[0])
                    - mx.argmax(active_decoder[::-1].astype(mx.int32)).item()
                )
                conditioned_hidden = conditioned_hidden[
                    :, decoder_position_start:decoder_position_stop, :
                ]
                decoder_tokens = tokens[
                    :, decoder_position_start:decoder_position_stop
                ]
                source_access = source_access[
                    :, decoder_position_start:decoder_position_stop
                ]
                if (
                    plan_access is not None
                    and len(plan_access.shape) == 2
                    and int(plan_access.shape[1]) == int(tokens.shape[1])
                ):
                    plan_access = plan_access[
                        :, decoder_position_start:decoder_position_stop
                    ]
                if compact_partition_width_quantum:
                    decoder_width = int(conditioned_hidden.shape[1])
                    decoder_unpadded_width = decoder_width
                    bucket_width = int(
                        math.ceil(
                            decoder_width / compact_partition_width_quantum
                        )
                        * compact_partition_width_quantum
                    )
                    padding = bucket_width - decoder_width
                    if padding:
                        conditioned_hidden = mx.pad(
                            conditioned_hidden,
                            ((0, 0), (0, padding), (0, 0)),
                        )
                        decoder_tokens = mx.pad(
                            decoder_tokens, ((0, 0), (0, padding))
                        )
                        source_access = mx.pad(
                            source_access, ((0, 0), (0, padding))
                        )
                        if plan_access is not None:
                            plan_access = mx.pad(
                                plan_access, ((0, 0), (0, padding))
                            )
            if kerc_enabled:
                if cached_kerc_context is None:
                    (
                        kerc_context,
                        kerc_stage_weights,
                        kerc_residual_logits,
                        kerc_unit_logits,
                        kerc_unit_confidence_logits,
                    ) = self.kerc_context(
                        tokens,
                        source_memory,
                        source_mask,
                        unit_byte_ids=kerc_unit_byte_ids,
                        unit_byte_mask=kerc_unit_byte_mask,
                        unit_byte_offsets=kerc_unit_byte_offsets,
                        unit_kind_ids=kerc_unit_kind_ids,
                        unit_candidate_features=kerc_unit_candidate_features,
                        unit_mask=kerc_unit_mask,
                        unit_hard_block_mask=kerc_unit_hard_block_mask,
                    )
                else:
                    kerc_context = cached_kerc_context
                    kerc_stage_weights = cached_kerc_stage_weights
                    kerc_residual_logits = cached_kerc_residual_logits
                    kerc_unit_logits = cached_kerc_unit_logits
                    kerc_unit_confidence_logits = cached_kerc_unit_confidence_logits
                kerc_access = (
                    source_access
                    if source_access is not None
                    else mx.ones(tokens.shape, dtype=mx.float32)
                )
                conditioned_hidden = (
                    conditioned_hidden
                    + kerc_context[:, None, :] * kerc_access[:, :, None]
                )
            else:
                kerc_context = None
                kerc_stage_weights = None
                kerc_residual_logits = None
                kerc_unit_logits = None
                kerc_unit_confidence_logits = None
            attention_mask = self.sequence_attention_mask(decoder_tokens, cache)
            if not state_enabled:
                hidden = conditioned_hidden
                completed_attention_residual_blocks = (
                    [conditioned_hidden]
                    if config.attention_residual_mode == "block"
                    else []
                )
                partial_attention_residual_block = None
                attention_residual_block_size = int(
                    config.attention_residual_block_size
                )
                next_cache: list[tuple[Any, ...]] = []
                for index, layer in enumerate(self.layers):
                    if config.attention_residual_mode == "block":
                        sources = list(
                            completed_attention_residual_blocks
                        )
                        if partial_attention_residual_block is not None:
                            sources.append(
                                partial_attention_residual_block
                            )
                        hidden = self.attention_residual_mix(
                            query_index=index,
                            sources=sources,
                            sequential_hidden=hidden,
                        )
                    layer_cache = layer_cache_input[index] if layer_cache_input is not None else None
                    if self.gradient_checkpointing and self.training:
                        if layer_cache is not None or plan_slot_attention_enabled:
                            raise ValueError(
                                "gradient checkpointing requires cache-free non-plan training"
                            )
                        if source_memory is None:
                            def checkpointed_decoder(
                                parameters: Any,
                                layer_hidden: Any,
                                layer_module: Any = layer,
                            ) -> Any:
                                layer_module.update(parameters)
                                layer_output, _unused_training_cache = layer_module(
                                    layer_hidden,
                                    kerc_stage_weights=kerc_stage_weights,
                                    attention_mask=attention_mask,
                                    structural_growth_mask=(
                                        structural_growth_masks[index]
                                        if structural_growth_masks is not None
                                        else None
                                    ),
                                )
                                # This branch is explicitly cache-free training.
                                # Returning K/V makes each layer cache a checkpoint
                                # output, retaining it across backward even though
                                # the scalar objective discards the model cache.
                                return layer_output

                            hidden = mx.checkpoint(
                                checkpointed_decoder
                            )(layer.trainable_parameters(), hidden)
                            layer_next = ()
                        else:
                            def checkpointed_source_decoder(
                                parameters: Any,
                                layer_hidden: Any,
                                layer_source_memory: Any,
                                layer_source_mask: Any,
                                layer_source_access: Any,
                                layer_module: Any = layer,
                            ) -> Any:
                                layer_module.update(parameters)
                                layer_output, _unused_training_cache = layer_module(
                                    layer_hidden,
                                    source_memory=layer_source_memory,
                                    source_mask=layer_source_mask,
                                    source_access=layer_source_access,
                                    kerc_stage_weights=kerc_stage_weights,
                                    attention_mask=attention_mask,
                                    structural_growth_mask=(
                                        structural_growth_masks[index]
                                        if structural_growth_masks is not None
                                        else None
                                    ),
                                )
                                return layer_output

                            hidden = mx.checkpoint(
                                checkpointed_source_decoder
                            )(
                                layer.trainable_parameters(),
                                hidden,
                                source_memory,
                                source_mask,
                                source_access,
                            )
                            layer_next = ()
                    else:
                        hidden, layer_next = layer(
                            hidden,
                            layer_cache,
                            plan_memory=plan_context if plan_slot_attention_enabled else None,
                            plan_access=plan_access,
                            source_memory=source_memory,
                            source_mask=source_mask,
                            source_access=source_access,
                            kerc_stage_weights=kerc_stage_weights,
                            attention_mask=attention_mask,
                            structural_growth_mask=(
                                structural_growth_masks[index]
                                if structural_growth_masks is not None
                                else None
                            ),
                        )
                    if config.attention_residual_mode == "block":
                        partial_attention_residual_block = (
                            hidden
                            if partial_attention_residual_block is None
                            else partial_attention_residual_block + hidden
                        )
                        if (
                            (index + 1) % attention_residual_block_size == 0
                            or index + 1 == config.num_layers
                        ):
                            completed_attention_residual_blocks.append(
                                partial_attention_residual_block
                            )
                            partial_attention_residual_block = None
                    next_cache.append(layer_next)
                if source_encoder_enabled and source_memory is not None:
                    next_cache.append((source_memory, source_mask, source_copy_ids))
                if plan_enabled:
                    next_cache.append((plan_context,))
                if kerc_enabled:
                    next_cache.append(
                        (
                            kerc_context,
                            kerc_stage_weights,
                            kerc_residual_logits,
                            kerc_unit_logits,
                            kerc_unit_confidence_logits,
                        )
                    )
                if config.attention_residual_mode == "block":
                    hidden = self.attention_residual_mix(
                        query_index=config.num_layers,
                        sources=completed_attention_residual_blocks,
                        sequential_hidden=hidden,
                    )
                final_hidden = self.final_norm(hidden)
                if decoder_unpadded_width and output_position_mask is None:
                    final_hidden = final_hidden[:, :decoder_unpadded_width, :]
                output_position_start = 0
                output_position_stop = int(final_hidden.shape[1])
                projected_hidden = final_hidden
                if output_position_mask is not None:
                    if not self.compact_output_projection or cache is not None:
                        raise ValueError(
                            "output compaction requires the cache-free compact "
                            "output policy"
                        )
                    active_output = output_position_mask > 0
                    active_any = mx.any(active_output, axis=0)
                    if bool(mx.any(active_any)):
                        output_position_start = int(
                            mx.argmax(active_any.astype(mx.int32)).item()
                        )
                        for row_index in range(int(tokens.shape[0])):
                            row_active = active_output[row_index]
                            if not bool(mx.any(row_active)):
                                continue
                            first = int(
                                mx.argmax(row_active.astype(mx.int32)).item()
                            )
                            last = int(
                                tokens.shape[1]
                                - 1
                                - mx.argmax(
                                    row_active[::-1].astype(mx.int32)
                                ).item()
                            )
                            if not bool(mx.all(row_active[first : last + 1])):
                                raise ValueError(
                                    "output compaction requires one contiguous supervised window per row"
                                )
                            output_position_stop = max(
                                output_position_stop
                                if row_index
                                else 0,
                                last + 1,
                            )
                    else:
                        # Verifier/decision corruption rows can intentionally
                        # carry no generator target.  Preserve one zero-masked
                        # position so the scalar token objective remains shape
                        # valid without projecting the full sequence vocabulary.
                        output_position_start = int(tokens.shape[1]) - 1
                        output_position_stop = output_position_start + 1
                    if decoder_position_start:
                        if output_position_start < decoder_position_start:
                            raise ValueError(
                                "supervised output begins inside the compacted source partition"
                            )
                        projected_hidden = final_hidden[
                            :,
                            output_position_start - decoder_position_start :
                            output_position_stop - decoder_position_start,
                            :,
                        ]
                    else:
                        projected_hidden = final_hidden[
                            :, output_position_start:output_position_stop, :
                        ]
                generator_logits = self.kerc_generator_logits(
                    (
                        mx.stop_gradient(projected_hidden)
                        if self.kerc_stage_output_isolation
                        else projected_hidden
                    ),
                    kerc_stage_weights,
                )
                pointer_access = None
                if kerc_enabled and kerc_stage_weights is not None:
                    active = mx.sum(kerc_stage_weights, axis=-1)
                    pointer_access = (
                        1.0
                        - active
                        + kerc_stage_weights[:, 0]
                    )
                logits, copy_aux = self.output_logits(
                    projected_hidden,
                    source_memory,
                    source_mask,
                    source_copy_ids,
                    generator_logits,
                    pointer_access,
                )
                if return_training_aux:
                    return logits, next_cache, {
                        "final_hidden": projected_hidden,
                        "output_position_start": output_position_start,
                        "output_position_stop": output_position_stop,
                        "plan_logits": plan_logits,
                        "copy_aux": copy_aux,
                        "mtp_logits": self.mtp_logits(final_hidden),
                        "kerc": {
                            "stage_weights": kerc_stage_weights,
                            "residual_logits": kerc_residual_logits,
                            "unit_residual_logits": kerc_unit_logits,
                            "unit_confidence_logits": kerc_unit_confidence_logits,
                            "verifier_logits": (
                                self.kerc_verifier_logits(tokens)
                                if cache is None
                                else None
                            ),
                            "decision_logits": (
                                self.kerc_decision_logits(
                                    tokens, source_memory, source_mask
                                )
                                if cache is None
                                else None
                            ),
                        }
                        if kerc_enabled
                        else None,
                    }
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
                    if self.gradient_checkpointing and self.training:
                        raise ValueError(
                            "gradient checkpointing is not qualified for recurrent state memory"
                        )
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
            final_hidden = mx.concatenate(outputs, axis=1)
            logits = self.token_embedding.as_linear(final_hidden)
            final_cache = current_cache or []
            if plan_enabled:
                final_cache.append((plan_context,))
            if return_training_aux:
                return logits, final_cache, {
                    "final_hidden": final_hidden,
                    "plan_logits": plan_logits,
                    "copy_aux": None,
                    "mtp_logits": self.mtp_logits(final_hidden),
                }
            return (logits, final_cache, plan_logits) if return_plan_logits else (logits, final_cache)

    return StandardCausalTransformer()


def parameter_count(model: Any, mlx_utils: Any) -> int:
    return int(sum(value.size for _name, value in mlx_utils.tree_flatten(model.parameters())))
